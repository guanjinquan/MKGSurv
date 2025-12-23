import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from modules.base_modules.aggregation_utils import masked_mean_pool
import json
import random
import math



# --- 基础组件 ---
class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
    

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GELU(),  # ReLU之后要跟LayerNorm，但是GeLU之后本身就是高斯分布，不需要再归一化
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# --- SafeCrossAttnEncoder ---
class SafeCrossAttnEncoder(nn.Module):
    """
    升级版交叉注意力模块。
    结构: CrossAttention -> Add & Norm -> FeedForward -> Add & Norm
    包含了防 NaN 的安全机制。
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 1. Attention 部分
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 2. FFN 部分  
        self.ffn = FeedForward(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False) -> torch.Tensor:
        """
        query: (B, Lq, D)
        key:   (B, Lk, D)
        value: (B, Lk, D)
        key_padding_mask: (B, Lk), True 为 padding
        """

        query = self.norm_q(query)
        
        # --- 核心修复逻辑 (Safe Logic) ---
        if key_padding_mask is not None:
            # 检测哪些样本的所有 Key 都是 Padding
            all_masked_rows = key_padding_mask.all(dim=1) # (B,) bool

            if all_masked_rows.any():
                # 只有当存在全 Mask 的情况时，才进行克隆和修改
                key_padding_mask = key_padding_mask.clone()
                # 将全 Mask 行的第一个位置设为 False (有效)，防止 Softmax NaN
                key_padding_mask[all_masked_rows, 0] = False
        else:
            all_masked_rows = None  

        # --- 1. Attention Block ---
        # 正常计算 MHA
        attn_out, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights)
            
        # 清理垃圾值：将那些原本全无效的行的输出置为 0
        if all_masked_rows is not None and all_masked_rows.any():
            attn_out[all_masked_rows] = 0.0

        # Residual + Norm (Post-Norm 风格)
        x = query + self.dropout(attn_out)
        
        # --- 2. FFN Block (新增逻辑) ---
        ffn_out = self.ffn(self.norm_ffn(x))

        x = x + self.dropout(ffn_out)

        if need_weights:
            return x, attn_weights
        return x



class EdgeContextualizer(nn.Module):
    """
    使用Edge作为Query，连接的节点特征作为Key/Value。
    让知识(Edge)根据具体的病人数据(Node)进行动态调整。
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = SafeCrossAttnEncoder(embed_dim, num_heads)

    def forward(self, edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                node_i: torch.Tensor, node_i_mask: torch.Tensor,
                node_j: torch.Tensor, node_j_mask: torch.Tensor) -> torch.Tensor:
        
        # 1. 拼接两个模态的特征作为上下文 (B, Ni+Nj, D)
        context_feat = torch.cat([node_i, node_j], dim=1)
        
        # 2. 拼接Mask (B, Ni+Nj)
        # 注意：输入的mask是1有效0无效，MHA通常需要True为无效(padding)
        # 这里先拼接原始mask (1有效)
        context_mask_raw = torch.cat([node_i_mask, node_j_mask], dim=1)
        
        # 转换为MHA需要的格式: True为Padding(无效), False为有效
        key_padding_mask = (context_mask_raw == 0)

        # 3. Edge更新: Edge query Context
        # Edge mask自身不需要传入attn mask，因为它是query，长度不变，padding位置的输出后续会被mask掉或忽略
        updated_edge = self.cross_attn(query=edge_feat, key=context_feat, value=context_feat, 
                                     key_padding_mask=key_padding_mask)
        
        # 4. Apply Edge Mask: 确保无效的 Edge Token 输出保持为 0
        # updated_edge: (B, Le, D), edge_mask: (B, Le)
        if edge_mask is not None:
            updated_edge = updated_edge * edge_mask.unsqueeze(-1).type_as(updated_edge)
        
        return updated_edge
    




class MedKGATFusion(nn.Module):
    def __init__(self, args, embed_dim: int, 
            max_modalities: int = 10, 
            max_groups: int = 10, 
            ff_dropout_rate: float = 0.1, 
            attn_dropout_rate: float = 0.1, 
            num_intra_layers: int = 1, num_inter_layers: int = 1):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.drop_edge_ratio = 0.1

        # 1. Knowledge Projection (768 -> embed_dim)
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(ff_dropout_rate)
        )

        # 2. Intra-group Interaction
        self.num_intra_layers = num_intra_layers
        self.intra_group_transformer = nn.ModuleList([
            SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate)
            for _ in range(num_intra_layers)
        ])

        # 3. GAT Interaction Components (Inter-Group)
        self.num_inter_layers = num_inter_layers
        self.shared_inter_layer = nn.ModuleDict({
            'edge_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate),
            'node_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate),
            'edge_updater': EdgeContextualizer(embed_dim, num_heads=8)
        })

        # 4. Global Aggregation
        self.global_transformer = SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate)

        # 5. Post Fusion Norm
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def _intra_group_step(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], groups: List[List[int]]) -> List[torch.Tensor]: 
        updated_embeddings = list(embeddings)
        
        for group_indices in groups:
            if not group_indices:
                continue
                
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            
            lengths = [f.shape[1] for f in group_feats]
            
            concat_feat = torch.cat(group_feats, dim=1)
            concat_mask = torch.cat(group_masks, dim=1)
            
            padding_mask = (concat_mask == 0) # True is invalid
            
            # Safe Transformer Check
            all_masked_rows = padding_mask.all(dim=1)
            if all_masked_rows.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_masked_rows, 0] = False

            # The TransformerEncoder handles num_layers internally
            for i in range(self.num_intra_layers):
                concat_feat = self.intra_group_transformer[i](
                    query=concat_feat, 
                    key=concat_feat, 
                    value=concat_feat, 
                    key_padding_mask=padding_mask
                )

                if all_masked_rows.any():
                    concat_feat[all_masked_rows] = 0.0

            split_feats = torch.split(concat_feat, lengths, dim=1)
            
            for i, idx in enumerate(group_indices):
                updated_embeddings[idx] = split_feats[i]
                
        return updated_embeddings

    def _inter_group_step(self, target_node: torch.Tensor, target_mask: torch.Tensor,
                          source_node: torch.Tensor, source_mask: torch.Tensor,
                          edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                          layer_modules: nn.ModuleDict) -> torch.Tensor:
        """
        One-way interaction: Source -> Edge -> Target
        Updated to take layer_modules dict
        """
        source_padding_mask = (source_mask == 0)
        edge_padding_mask = (edge_mask == 0)
        
        # Step 1: Edge queries Source to get relevant info (Gating)
        gated_source = layer_modules['edge_to_node_attn'](
            query=edge_feat, 
            key=source_node, 
            value=source_node, 
            key_padding_mask=source_padding_mask
        )
        
        # Step 2: Target queries Gated Source to update itself
        # Note: Key/Value mask depends on Edge because gated_source has shape of Edge
        updated_target = layer_modules['node_to_node_attn'](
            query=target_node,
            key=gated_source,
            value=gated_source,
            key_padding_mask=edge_padding_mask
        )
        
        if target_mask is not None:
            updated_target = updated_target * target_mask.unsqueeze(-1).type_as(updated_target)

        return updated_target
    
    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor],
        analysis_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # 0. Ensure symmetric keys removal
        for (i, j), v in list(fusion_knowledge.items()): 
            if (j, i) in fusion_knowledge and i < j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        edge_keys = list(fusion_knowledge.keys())
        if self.training:
            random.shuffle(edge_keys)

        # 1. Project Knowledge Edges
        # This will be our initial edge state
        current_proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            current_proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction (Multi-layer handled inside TransformerEncoder)
        info_level_embeddings = self._intra_group_step(embeddings, masks, embeddings_groups)

        # 3. Create Group-Level Embeddings
        group_embeddings = []
        group_masks = []
    
        for group_indices in embeddings_groups:
            if not group_indices:
                raise ValueError("Empty group found in embeddings_groups")

            curr_feats = [info_level_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]

            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)

            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # Pre-calculate validity masks for Weights (based on INPUT embeddings/masks)
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # 4. Inter-Group Interaction (Multi-Layer GNN / GAT)
        # We loop self.num_inter_layers times
        
        current_group_embeddings = group_embeddings # Points to current node features

        for layer_idx in range(self.num_inter_layers):
            layer_modules = self.shared_inter_layer
            num_groups = len(current_group_embeddings)

            for (idx_a, idx_b) in edge_keys:
                edge_feat = current_proj_knowledge.get((idx_a, idx_b))

                if self.training and getattr(self, 'drop_edge_ratio', 0.0) > 0.0:
                    if random.random() < self.drop_edge_ratio:
                        continue
                
                edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                if edge_mask is None:
                    edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

                # Get Group Data
                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a] # Masks don't change
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # --- GNN Update Logic for this Layer ---
                # Update Edge Features
                updated_edge_feat = layer_modules['edge_updater'](
                    edge_feat, edge_mask, 
                    feat_a, mask_a, 
                    feat_b, mask_b
                )
                
                # Store updated edge for the next layer
                current_proj_knowledge[(idx_a, idx_b)] = updated_edge_feat

                # Update Node B using Node A and Edge
                update_for_b = self._inter_group_step(
                    target_node=feat_b, target_mask=mask_b,
                    source_node=feat_a, source_mask=mask_a,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_b] = update_for_b

                # Update Node A using Node B and Edge
                update_for_a = self._inter_group_step(
                    target_node=feat_a, target_mask=mask_a,
                    source_node=feat_b, source_mask=mask_b,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_a] = update_for_a

        # Final embeddings after all GAT layers
        final_group_embeddings = current_group_embeddings

        # ------------------------------------------------------------------
        # 4b. Data Collection for KL Loss (Post-GAT)
        # ------------------------------------------------------------------
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_edge_pairs_list = []
        edge_score_valid_flag = False

        # Iterate edge keys to align Ground Truth with Predictions
        for (idx_a, idx_b) in edge_keys:
            # Retrieve Ground Truth Edge Score
            edge_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            
            # Handle missing edge scores
            if edge_score is None:
                edge_score = torch.zeros(embeddings[0].shape[0], device=embeddings[0].device)
            
            if edge_score.dim() > 1:
                edge_score = edge_score.view(-1)
            if edge_score.dim() == 0:
                edge_score = edge_score.expand(embeddings[0].shape[0])

            # Check if both groups in the pair are valid (have data)
            has_a = group_validity_masks[idx_a].float()
            has_b = group_validity_masks[idx_b].float()
            pair_validity = has_a * has_b

            edge_score_valid_flag |= edge_score.sum().item() > 0
            
            all_edge_scores_list.append(edge_score)
            all_valid_masks_list.append(pair_validity)
            all_edge_pairs_list.append((idx_a, idx_b))

        # ------------------------------------------------------------------
        # 6. Random Token Selection & Dot Products (Updated Logic)
        # ------------------------------------------------------------------
        all_dot_prods_list = [] 
        
        if len(all_edge_pairs_list) > 0 and edge_score_valid_flag:
            
            # --- Random Sampling Step ---
            # Instead of pooling, we select one valid token per group per batch
            sampled_group_embeddings = []
            
            for g_feat, g_mask in zip(final_group_embeddings, group_masks):
                # g_feat: [Batch, SeqLen, Dim]
                # g_mask: [Batch, SeqLen]
                
                # Create probabilities based on mask (Uniform over valid tokens)
                probs = g_mask.float()
                row_sums = probs.sum(dim=1, keepdim=True)
                
                # Handle cases where row_sums is 0 (empty group) to avoid div by zero
                # We will just sample index 0 if empty, the validity mask in KL loss will ignore it anyway
                probs = torch.where(row_sums > 0, probs / row_sums, torch.zeros_like(probs))
                
                # If a row is all zeros, set prob of index 0 to 1.0 to satisfy multinomial
                probs[row_sums.squeeze() == 0, 0] = 1.0
                
                # Sample one index per batch item
                # sampled_indices shape: [Batch]
                sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # Gather the selected tokens
                # We use advanced indexing: [0..B-1, sampled_indices]
                batch_indices = torch.arange(g_feat.size(0), device=g_feat.device)
                
                # [Batch, Dim]
                selected_token_feat = g_feat[batch_indices, sampled_indices]
                sampled_group_embeddings.append(selected_token_feat)

            # --- Dot Product Step ---
            scale_factor = 1.0 / (self.embed_dim ** 0.5)

            for idx_a, idx_b in all_edge_pairs_list:
                # Shape: (Batch_Size, Dim)
                vec_a = sampled_group_embeddings[idx_a]
                vec_b = sampled_group_embeddings[idx_b]
                
                # Dot Product + Scaling
                # Shape: (Batch_Size, )
                dot_prod = torch.sum(vec_a * vec_b, dim=1) * scale_factor
                all_dot_prods_list.append(dot_prod)

        # Save Points (Visualization/Debugging)
        if hasattr(self, 'save_points'):
            self.save_points(final_group_embeddings, group_masks, groups_relationships)

        # 7. Global Aggregation
        # Note: Global aggregation still uses ALL tokens for the downstream task, 
        # the sampling above was specifically for the KL constraint.
        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        global_transformed, attn_weights = self.global_transformer(
            query=global_concat, key=global_concat, value=global_concat, key_padding_mask=global_padding_mask, need_weights=True)
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        fused_embedding, _ = masked_mean_pool(global_transformed, global_mask)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        if not self.training and hasattr(self, 'view_groups_contribution'):
            self.view_groups_contribution(attn_weights, global_concat, group_masks)

        # 8. Compute KL Divergence Loss (Updated)
        fusion_loss = torch.tensor(0.0, device=fused_embedding.device)
        
        if len(all_edge_scores_list) > 0 and edge_score_valid_flag:
            # (Batch_Size, Num_Edges)
            all_scores_tensor = torch.stack(all_edge_scores_list, dim=1)  # Ground Truth
            all_dots_tensor = torch.stack(all_dot_prods_list, dim=1)      # Predictions (Already Scaled)
            all_masks_tensor = torch.stack(all_valid_masks_list, dim=1)   # Validity

            # --- Target (Ground Truth) Processing ---
            # Use Softmax to create probability distribution P(x)
            scores_masked = all_scores_tensor.clone().float()
            scores_masked[all_masks_tensor == 0] = -1e9 # Mask invalid edges
            target_probs = F.softmax(scores_masked, dim=1) 

            # --- Prediction Processing ---
            # Use Log-Softmax to create log-probability distribution log(Q(x))
            # KLDiv Loss expects input in log-space!
            dots_masked = all_dots_tensor.clone()
            dots_masked[all_masks_tensor == 0] = -1e9
            
            # Apply log_softmax (Note: dots were already scaled in Step 6)
            pred_log_probs = F.log_softmax(dots_masked, dim=1) 
            
            # --- KL Divergence ---
            # KL(P || Q) = sum(P(x) * (log P(x) - log Q(x)))
            # PyTorch F.kl_div(input, target) computes this where input=log_probs, target=probs
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none')
            
            # Ensure masked values don't contribute 
            kl_loss = kl_loss * all_masks_tensor
            
            # Sum over edges (dim=1) to get loss per patient
            kl_loss_per_patient = kl_loss.sum(dim=1) 
            
            # Filter out patients who had NO valid edges (all_masks_tensor row sum is 0)
            valid_patients_mask = (all_masks_tensor.sum(dim=1) > 0).float() 
            num_valid_patients = valid_patients_mask.sum()
            
            if num_valid_patients > 0:
                fusion_loss = (kl_loss_per_patient * valid_patients_mask).sum() / num_valid_patients
    
        if not self.training:
            if hasattr(self.args, 'save_umap_path') and self.args.save_umap_path:
                self.save_features_for_umap(final_group_embeddings, group_masks, fused_embedding)

        # ==================================================================
        # [修复] 逻辑分离：捕获归捕获，触发归触发
        # ==================================================================

        # 1. 捕获逻辑 (Capture Logic)
        # 只有在 analysis_mode=True 时，或者你强制想看梯度时运行
        if analysis_mode:
            # 清空旧数据
            self.captured_group_feats = []
            self.captured_group_masks = []
            
            # print("\n[Debug Forward] Start capturing (Inner Loop)...")
            for feat in group_embeddings:
                # 只有带梯度的才保留，避免报错
                if feat.requires_grad:
                    feat.retain_grad()
                # else:
                    # print("[Warning] Feature has no gradient requirements!")
            
            # 保存当前计算图中的 Tensor
            self.captured_group_feats = group_embeddings
            self.captured_group_masks = group_masks


        # 2. 触发逻辑 (Trigger Logic)
        # 只有在测试模式、配置了路径、且不在递归中时运行
        should_run_gradcam = (
            not self.training 
            and hasattr(self.args, 'gradcam_save_path') 
            and self.args.gradcam_save_path is not None
            and not analysis_mode  # <--- 防止递归的关键
        )

        if should_run_gradcam:
            # print("[Debug Forward] Triggering GradCAM Analysis...")
            self.gradcam_analyse(
                embeddings=embeddings,
                masks=masks,
                embeddings_groups=embeddings_groups,
                groups_relationships=groups_relationships,
                fusion_knowledge=fusion_knowledge,
                fusion_knowledge_mask=fusion_knowledge_mask
            )

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": 2 * fusion_loss,
            }
        }
    
    def save_features_for_umap(self, group_embeddings, group_masks, fused_embedding):
        """
        保存特征用于 UMAP 可视化。
        修改：保存所有 Valid Tokens，而不是 Pooling 后的向量。
        
        JSONL 格式变更:
        {
            "groups": [[dim1...], [dim2...] ...],   # 所有 Valid Tokens 的列表 (Flattened)
            "group_ids": [0, 0, 1, 2...],           # 每个 Token 对应的 Group Index
            "fused": [dim1, dim2...]                # 融合后的特征
        }
        """
        import os
        import json
        
        # 确保路径存在
        save_path = self.args.save_umap_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        batch_size = fused_embedding.shape[0]
        
        # 1. 准备数据 (移到 CPU)
        # fused: (B, D)
        fused_emb = fused_embedding.detach().cpu()
        # groups: List of (B, L, D)
        cpu_groups = [g.detach().cpu() for g in group_embeddings]
        # masks: List of (B, L)
        cpu_masks = [m.detach().cpu() for m in group_masks]

        # 3. 写入文件
        with open(save_path, 'a', encoding='utf-8') as f:
            for b in range(batch_size):
                all_tokens = []
                all_token_ids = []
                
                # 遍历每个 Group
                for g_idx, (g_feat, g_mask) in enumerate(zip(cpu_groups, cpu_masks)):
                    # g_feat[b]: (Seq, Dim)
                    # g_mask[b]: (Seq)
                    
                    curr_feat = g_feat[b]
                    curr_mask = g_mask[b]
                    
                    # 获取 Valid Indices (Mask != 0)
                    # nonzero 返回 (Num_Valid, 1), squeeze 后变成 (Num_Valid)
                    # 假设 mask 中 0 是 padding
                    valid_indices = torch.nonzero(curr_mask).squeeze(-1)
                    
                    if valid_indices.numel() > 0:
                        # 提取 Valid Tokens
                        valid_tokens = curr_feat[valid_indices] # (Num_Valid, Dim)
                        
                        # 添加到列表
                        for tok in valid_tokens:
                            all_tokens.append(tok.tolist())
                            all_token_ids.append(g_idx)

                record = {
                    "groups": all_tokens,       # Flattened valid tokens from all groups
                    "group_ids": all_token_ids, # Corresponding group index for each token
                    "fused": fused_emb[b].tolist()
                }
                f.write(json.dumps(record) + "\n")
                
    def view_groups_contribution(self, attn_weights: torch.Tensor, values: torch.Tensor, group_masks: List[torch.Tensor]):
        """
        方案1实现：基于范数(Energy)的贡献度分析。
        保存格式：与之前一致，JSONL 每行一个列表 [g0_ratio, g1_ratio, ...]
        
        Args:
            attn_weights: (B, L, L) or (B, H, L, L) - 注意力权重
            values: (B, L, D) - Transformer 的输入 (即 Global Concat)
            group_masks: List[(B, L_g)]
        """
        if not hasattr(self.args, 'view_groups_attention_path') or self.args.view_groups_attention_path is None:
            return
        
        save_path = self.args.view_groups_attention_path
        # 为了区分，建议修改一下文件名，或者保持原样覆盖
        # save_path = save_path.replace('.jsonl', '_contribution.jsonl') 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if attn_weights is None or values is None:
            return
        
        print("attn_weights shape: ", attn_weights.shape)

        # 1. 维度与数据检查
        # 如果是多头 (B, H, L, L)，先平均成 (B, L, L)
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)

        # 确保 attn 是 (B, L, L)
        if attn_weights.shape[0] != group_masks[0].shape[0]:
            attn_weights = attn_weights.permute(1, 0, 2)
            
        # 确保 values 是 (B, L, D)
        if values.shape[0] != group_masks[0].shape[0]:
            values = values.transpose(0, 1)

        # 2. 强制 Softmax 检查 (Contribution 分析必须基于概率)
        check_sum = attn_weights[0, 0, :].sum().item()
        if check_sum > 1.1 or check_sum < 0.9:
            # print("[Info] Applying Softmax for contribution analysis...")
            attn_weights = torch.softmax(attn_weights, dim=-1)

        # 3. 准备 Mask 和 Offsets
        global_mask = torch.cat(group_masks, dim=1).float() # (B, L_total)
        num_valid_queries = global_mask.sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1)

        group_lengths = [gm.shape[1] for gm in group_masks]
        offsets = [0]
        for l in group_lengths:
            offsets.append(offsets[-1] + l)

        # 4. 核心计算循环：计算每个组的 Energy
        group_energy_list = []

        for i in range(len(group_masks)):
            start, end = offsets[i], offsets[i+1]
            
            # A. 取出该组对应的 Attention 概率 (B, L_total, L_group)
            # 代表：每个 Token 对该组分配了多少关注
            attn_slice = attn_weights[:, :, start:end]
            
            # # B. 取出该组对应的 Feature Values (B, L_group, D)
            # value_slice = values[:, start:end, :]
            
            # # C. 矩阵乘法：加权求和
            # # (B, L_total, L_group) @ (B, L_group, D) -> (B, L_total, D)
            # # 含义：该组特征实际上向 Residual Stream 注入了多少更新向量
            # weighted_update = torch.bmm(attn_slice, value_slice)
            
            # D. 计算能量 (L2 Norm)
            # (B, L_total) -> 每个位置收到的来自该组的更新强度
            update_norm = torch.norm(attn_slice, p=2, dim=-1)
            
            # E. Mask 掉 Padding 位置 (我们只关心有效 Token 收到的贡献)
            update_norm = update_norm * global_mask
            
            # F. 平均化：得到该样本中，该组的平均贡献强度
            avg_energy = update_norm.mean(dim=1) #  / num_valid_queries.squeeze(-1) # (B,)
            
            group_energy_list.append(avg_energy)

        # 5. 堆叠与归一化 (转为比例)
        # 结果 shape: (B, Num_Groups)
        group_energies = torch.stack(group_energy_list, dim=1)
        
        # 计算总能量，归一化成 0~1 的比例，方便和之前的 Attention Score 对比
        total_energy = group_energies.sum(dim=1, keepdim=True)
        contribution_ratios = group_energies / torch.clamp(total_energy, min=1)

        # 6. 保存到 JSONL
        batch_ratios = contribution_ratios.detach().cpu().tolist()
        
        try:
            with open(save_path, 'a', encoding='utf-8') as f:
                for sample_ratios in batch_ratios:
                    # 格式: [0.85, 0.10, 0.05]
                    f.write(json.dumps(sample_ratios) + "\n")
        except Exception as e:
            print(f"Warning: Failed to save contribution scores: {e}")
            
    def save_points(self, final_group_embeddings, final_group_masks, groups_relationships):
        if self.args.points_save_path is None:
            return 

        group_mean_embeddings = []
        for i in range(len(final_group_embeddings)):
            res = masked_mean_pool(final_group_embeddings[i], final_group_masks[i])
            if isinstance(res, tuple):
                mean_emb = res[0]
            else:
                mean_emb = res
            group_mean_embeddings.append(mean_emb)

        batch_size = final_group_embeddings[0].shape[0]
        device = final_group_embeddings[0].device
        
        sum_edge_scores = torch.zeros((batch_size, 1), device=device)
        sum_cos_sims = torch.zeros((batch_size, 1), device=device)
        
        raw_data_cache = {} 
        valid_pairs = []

        for (idx_a, idx_b), _ in groups_relationships.items():
            raw_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            
            if raw_score is not None:
                if raw_score.dim() == 1:
                    raw_score = raw_score.view(-1, 1)
                
                embed_a = group_mean_embeddings[idx_a]
                embed_b = group_mean_embeddings[idx_b]
                
                raw_cos = torch.cosine_similarity(embed_a, embed_b, dim=1).view(-1, 1)
                raw_cos_positive = torch.clamp(raw_cos, min=1e-9) 

                sum_edge_scores += raw_score
                sum_cos_sims += raw_cos_positive
                
                raw_data_cache[(idx_a, idx_b)] = (raw_cos_positive, raw_score)
                valid_pairs.append((idx_a, idx_b))

        sum_edge_scores = torch.clamp(sum_edge_scores, min=1e-9)
        sum_cos_sims = torch.clamp(sum_cos_sims, min=1e-9)

        if len(valid_pairs) > 0:
            save_points_path = self.args.points_save_path
            os.makedirs(os.path.dirname(save_points_path), exist_ok=True)
            
            current_batch_points = []
            
            for (idx_a, idx_b) in valid_pairs:
                raw_cos, raw_score = raw_data_cache[(idx_a, idx_b)]
                
                norm_cos = raw_cos / sum_cos_sims
                norm_score = raw_score / sum_edge_scores
                
                norm_cos_list = norm_cos.view(-1).detach().cpu().tolist()
                norm_score_list = norm_score.view(-1).detach().cpu().tolist()
                
                for pat_idx in range(len(norm_cos_list)):
                    current_batch_points.append([norm_cos_list[pat_idx], norm_score_list[pat_idx]])

            if current_batch_points:
                try:
                    with open(save_points_path, 'a') as f:
                        for point in current_batch_points:
                            f.write(json.dumps(point) + "\n")
                except Exception as e:
                    print(f"Warning: Failed to save points data: {e}")
    
    def gradcam_analyse(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor]
    ):
        # 路径检查省略...
        if hasattr(self.args, 'gradcam_save_path') and self.args.gradcam_save_path:
            save_path = self.args.gradcam_save_path
        elif hasattr(self.args, 'view_groups_attention_path') and self.args.view_groups_attention_path:
            save_path = self.args.view_groups_attention_path.replace('.jsonl', '_gradcam.jsonl')
        else:
            return 
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 强制开启梯度上下文
        with torch.enable_grad():
            self.zero_grad()
            
            # [调试] 打印原始输入状态
            # print(f"[Debug GradCAM] Original embedding 0 grad: {embeddings[0].requires_grad}")

            # -------------------------------------------------
            # 关键：构建带梯度的新输入
            # -------------------------------------------------
            inputs_with_grad = []
            for emb in embeddings:
                # 必须 detach 出来建立新图
                new_emb = emb.detach().clone().requires_grad_(True)
                inputs_with_grad.append(new_emb)

            knowledge_with_grad = {}
            for k, v in fusion_knowledge.items():
                new_v = v.detach().clone().requires_grad_(True)
                knowledge_with_grad[k] = new_v

            # [调试] 确认输入确实开启了梯度
            # print(f"[Debug GradCAM] New Input 0 requires_grad: {inputs_with_grad[0].requires_grad} (Should be True)")

            # -------------------------------------------------
            # 运行 Forward
            # -------------------------------------------------
            try:
                outputs = self.forward(
                    inputs_with_grad,      # 必须传入新的列表
                    masks, 
                    embeddings_groups, 
                    groups_relationships, 
                    knowledge_with_grad,   # 必须传入新的字典
                    fusion_knowledge_mask,
                    analysis_mode=True
                )
            except RuntimeError as e:
                print(f"[Critical Error in Forward]: {e}")
                return

            fused_emb = outputs['fused_embedding']
            
            # [调试] 检查输出是否有梯度
            if not fused_emb.requires_grad:
                print("[Fatal Error] fused_embedding lost gradients! Check modules (e.g. frozen weights?).")
                return

            # 定义目标：L2 Norm
            target_score = torch.norm(fused_emb, p=2, dim=1).sum()
            
            # 反向传播
            target_score.backward()

            # -------------------------------------------------
            # 计算重要性
            # -------------------------------------------------
            batch_group_scores = [] 
            
            for i, feat in enumerate(self.captured_group_feats):
                grad = feat.grad 
                mask = self.captured_group_masks[i]

                if grad is None:
                    # 如果打印了这个，说明 retain_grad 成功了，但是 backward 没传回来
                    # 这通常意味着 feat 没有参与 target_score 的计算
                    print(f"[Warning] Group {i} grad is None. (Did not participate in fusion?)")
                    grad = torch.zeros_like(feat)
                
                # HiResCAM Logic
                weighted_map = (feat * grad).sum(dim=-1)
                importance_map = F.relu(weighted_map)
                
                # Masking & Pooling
                mask_float = mask.float()
                importance_map = importance_map * mask_float
                valid_token_counts = mask_float.sum(dim=1).clamp(min=1.0)
                avg_group_importance = importance_map.sum(dim=1) / valid_token_counts
                
                batch_group_scores.append(avg_group_importance.detach().cpu())

            # 保存逻辑 (同之前)...
            if batch_group_scores:
                all_scores_tensor = torch.stack(batch_group_scores, dim=1)
                row_sums = all_scores_tensor.sum(dim=1, keepdim=True)
                contribution_ratios = all_scores_tensor / torch.clamp(row_sums, min=1e-9)
                
                # [调试] 打印第一个样本的比例，看看是不是还是全0
                # print(f"[Debug Result] Sample 0 Ratios: {contribution_ratios[0].tolist()}")

                batch_ratios_list = contribution_ratios.tolist()
                import json
                import math
                try:
                    with open(save_path, 'a', encoding='utf-8') as f:
                        for row in batch_ratios_list:
                            clean_row = [0.0 if (math.isnan(x) or math.isinf(x)) else x for x in row]
                            f.write(json.dumps(clean_row) + "\n")
                except Exception as e:
                    print(f"Save Error: {e}")

        self.captured_group_feats = []
        self.captured_group_masks = []
        self.zero_grad()