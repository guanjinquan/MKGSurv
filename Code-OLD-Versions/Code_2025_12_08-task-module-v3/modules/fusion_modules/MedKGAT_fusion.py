"""

LUAD:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6278 ± 0.0595
 - List = [0.6914241960183767, 0.5793714746172441, 0.6935866983372921, 0.6304176516942475, 0.5443833464257659]
C-Index-IPCW_Validation Set: 0.6114 ± 0.0745
 - List = [0.6956163600283107, 0.5174109370040318, 0.6819038379586689, 0.6311054414259185, 0.531146816399386]
Test Summary:
C-Index_Test Set: 0.6299 ± 0.0634
 - List = [0.5383423702556158, 0.7238993710691823, 0.5868055555555556, 0.6401617250673854, 0.6601671309192201]
C-Index-IPCW_Test Set: 0.5979 ± 0.0638
 - List = [0.5157434146668128, 0.6695260553772572, 0.5410494311683957, 0.5931222869899677, 0.670245588973929]
Training run tcga_luad_run001 finished.


"""

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
    
    def _compute_group_pairwise_kl_loss(self, 
                                        attn_weights: torch.Tensor, 
                                        group_masks: List[torch.Tensor], 
                                        groups_relationships: Dict[Tuple[int, int], torch.Tensor]) -> torch.Tensor:
        """
        计算 Group-Level 的 Pairwise KL Loss。
        
        逻辑：
        1. 将 Token-level Attention 聚合为 Group-level Attention Matrix (B, N_groups, N_groups)。
        2. 提取所有 Group Pairs (i, j) 的 Attention Mean。
        3. 提取所有 Group Pairs (i, j) 的 Ground Truth Score。
        4. 将两者分别归一化为概率分布，计算 KL 散度。
        """
        # 1. 维度检查与修正
        if attn_weights.dim() == 4: attn_weights = attn_weights.mean(dim=1) # (B, H, L, L) -> (B, L, L)
        if attn_weights.shape[0] != group_masks[0].shape[0]: attn_weights = attn_weights.permute(1, 0, 2)
        
        # 确保是概率 (Safety check)
        if attn_weights[0, 0, :].sum() > 1.1: attn_weights = torch.softmax(attn_weights, dim=-1)

        batch_size = attn_weights.shape[0]
        num_groups = len(group_masks)
        
        # 2. 准备 Offsets 和 Global Mask
        # global_mask = torch.cat(group_masks, dim=1).float() # (B, L_total)
        group_lengths = [gm.shape[1] for gm in group_masks]
        offsets = [0]
        for l in group_lengths: offsets.append(offsets[-1] + l)

        # 3. 收集成对的 Attention Mean 和 Target Score
        pred_pairs_list = []   # 存储预测的 Attention 强度
        target_pairs_list = [] # 存储真实的 Edge Score
        
        # 遍历所有可能的 Pair (i, j)
        # 这里的策略：我们关注 Upper Triangle (无向图关系) 或者 所有非对角线关系
        # 根据你的 groups_relationships 结构，通常是无向的，所以我们取 i < j
        valid_pair_found = False

        for i in range(num_groups):
            for j in range(num_groups):
                if i == j: continue # 跳过自环，或者根据需求保留
                if i > j: continue  # 只取上三角，避免重复 (i,j) 和 (j,i)

                # --- A. 计算 Predicted Attention Mean (i -> j) ---
                start_i, end_i = offsets[i], offsets[i+1]
                start_j, end_j = offsets[j], offsets[j+1]
                
                # Slice: Queries from Group i, Keys from Group j
                # Shape: (B, Li, Lj)
                block_attn = attn_weights[:, start_i:end_i, start_j:end_j]
                
                # Mask: 只有当 Query 和 Key 都有效时，Attention 才有效
                # mask_i: (B, Li), mask_j: (B, Lj)
                mask_i = group_masks[i].float()
                mask_j = group_masks[j].float()
                
                # pairwise_mask: (B, Li, Lj)
                block_mask = mask_i.unsqueeze(2) * mask_j.unsqueeze(1)
                
                # 计算 Mean Attention (Sum / Count)
                # 加上 1e-9 防止除零
                valid_counts = block_mask.sum(dim=(1, 2)) # (B,)
                sum_attn = (block_attn * block_mask).sum(dim=(1, 2)) # (B,)
                
                mean_attn_score = sum_attn / torch.clamp(valid_counts, min=1.0)
                
                # 如果这个 pair 完全无效(valid_counts=0)，mean_attn_score 应该是 0 (或者被mask掉)
                # 这里简单处理：如果 valid_counts 是 0，mean 也是 0
                mean_attn_score = mean_attn_score * (valid_counts > 0).float()

                # --- B. 获取 Target Score ---
                # 尝试获取 (i, j) 或 (j, i)
                edge_score = groups_relationships.get((i, j), groups_relationships.get((j, i), None))
                
                if edge_score is None:
                    # 如果没有定义关系，设为 0 (弱关系)
                    edge_score = torch.zeros(batch_size, device=attn_weights.device)
                else:
                    if edge_score.dim() > 1: edge_score = edge_score.view(-1)
                    if edge_score.dim() == 0: edge_score = edge_score.expand(batch_size)
                
                # 只有当该 pair 在当前 batch 确实存在有效 token 时，才计入 Loss
                # 我们使用 block_mask 的 valid性来判断
                # (B,)
                pair_is_valid = (valid_counts > 0).float()
                
                # 为了批量处理，我们先不管 invalid 的 batch sample，全部存下来，最后 mask
                pred_pairs_list.append(mean_attn_score)   # List of (B,)
                target_pairs_list.append(edge_score)      # List of (B,)
                
                valid_pair_found = True

        if not valid_pair_found or len(pred_pairs_list) == 0:
            return torch.tensor(0.0, device=attn_weights.device)

        # 4. 堆叠构建分布向量
        # Shape: (B, Num_Pairs)
        preds = torch.stack(pred_pairs_list, dim=1) 
        targets = torch.stack(target_pairs_list, dim=1)

        # 5. Mask 掉那些全为 0 的样本 (比如 Modality Dropout 导致的)
        # 如果一个样本的所有 Pair 预测都是 0 (意味着没有有效 Token)，这行数据不应计算 Loss
        valid_samples_mask = (preds.sum(dim=1) > 0).float() # (B,)

        # 6. 归一化为概率分布 (Softmax)
        # 注意：Attention Mean 本身很小，直接 Softmax 会导致分布过于平滑(High Entropy)。
        # 可以除以一个 temperature 让差异更明显
        temperature = 0.05 
        
        # Log Softmax for Prediction (Input to KLDiv)
        # 加上 1e-9 避免 log(0)
        preds_log_probs = F.log_softmax(preds / temperature, dim=1)
        
        # Softmax for Targets
        # Targets 也是 score，需要转为概率
        targets_probs = F.softmax(targets / temperature, dim=1)
        
        # 7. 计算 KL Divergence
        # reduction='none' 以便我们可以应用 valid_samples_mask
        # Shape: (B, Num_Pairs) -> Sum over pairs -> (B,)
        kl_loss = F.kl_div(preds_log_probs, targets_probs, reduction='none').sum(dim=1)
        
        # 8. 平均 Loss
        if valid_samples_mask.sum() > 0:
            final_loss = (kl_loss * valid_samples_mask).sum() / valid_samples_mask.sum()
        else:
            final_loss = torch.tensor(0.0, device=attn_weights.device)
            
        return final_loss

    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor],
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

        # 7. Global Aggregation
        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        global_padding_mask = (global_mask == 0)
        
        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        global_transformed, attn_weights = self.global_transformer(
            query=global_concat, key=global_concat, value=global_concat, 
            key_padding_mask=global_padding_mask, need_weights=True
        )
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        fused_embedding, _ = masked_mean_pool(global_transformed, global_mask)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # [Analysis] 保存 Contribution 分析 (只在 Eval 时)
        if not self.training:
             self.view_groups_contribution(attn_weights, global_concat, group_masks)

        # =========================================================================================
        # 8. Compute New Group-Level Pairwise KL Loss
        attn_kl_loss = self._compute_group_pairwise_kl_loss(attn_weights, group_masks, groups_relationships)

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": attn_kl_loss, 
            }
        }

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
            
            # B. 取出该组对应的 Feature Values (B, L_group, D)
            value_slice = values[:, start:end, :]
            
            # C. 矩阵乘法：加权求和
            # (B, L_total, L_group) @ (B, L_group, D) -> (B, L_total, D)
            # 含义：该组特征实际上向 Residual Stream 注入了多少更新向量
            weighted_update = torch.bmm(attn_slice, value_slice)
            
            # D. 计算能量 (L2 Norm)
            # (B, L_total) -> 每个位置收到的来自该组的更新强度
            update_norm = torch.norm(weighted_update, p=2, dim=-1)
            
            # E. Mask 掉 Padding 位置 (我们只关心有效 Token 收到的贡献)
            update_norm = update_norm * global_mask
            
            # F. 平均化：得到该样本中，该组的平均贡献强度
            avg_energy = update_norm.sum(dim=1) / num_valid_queries.squeeze(-1) # (B,)
            
            group_energy_list.append(avg_energy)

        # 5. 堆叠与归一化 (转为比例)
        # 结果 shape: (B, Num_Groups)
        group_energies = torch.stack(group_energy_list, dim=1)
        
        # 计算总能量，归一化成 0~1 的比例，方便和之前的 Attention Score 对比
        total_energy = group_energies.sum(dim=1, keepdim=True)
        contribution_ratios = group_energies / torch.clamp(total_energy, min=1e-9)

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