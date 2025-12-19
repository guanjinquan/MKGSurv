import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import json
import random

# --- 辅助函数 ---
def masked_mean_pool(prob: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    prob: (B, L, D)
    mask: (B, L)
    return: (B, D), (B,)
    """
    mask = mask.unsqueeze(-1).float()  # (B, L, 1)
    sum_prob = (prob * mask).sum(dim=1)  # (B, D)
    sum_mask = mask.sum(dim=1)  # (B, 1)
    
    # Avoid division by zero
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_prob = sum_prob / sum_mask
    
    # Validity mask for the batch items (1 if at least one token was valid)
    valid_mask = (mask.sum(dim=1) > 0).float().squeeze(-1)
    
    return mean_prob, valid_mask

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
            GELU(),
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
        all_masked_rows = None
        if key_padding_mask is not None:
            # 检测哪些样本的所有 Key 都是 Padding
            all_masked_rows = key_padding_mask.all(dim=1) # (B,) bool

            if all_masked_rows.any():
                # 只有当存在全 Mask 的情况时，才进行克隆和修改
                key_padding_mask = key_padding_mask.clone()
                # 将全 Mask 行的第一个位置设为 False (有效)，防止 Softmax NaN
                key_padding_mask[all_masked_rows, 0] = False
        
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
        context_mask_raw = torch.cat([node_i_mask, node_j_mask], dim=1)
        
        # 转换为MHA需要的格式: True为Padding(无效), False为有效
        key_padding_mask = (context_mask_raw == 0)

        # 3. Edge更新: Edge query Context
        updated_edge = self.cross_attn(query=edge_feat, key=context_feat, value=context_feat, 
                                     key_padding_mask=key_padding_mask)
        
        # 4. Apply Edge Mask
        if edge_mask is not None:
            updated_edge = updated_edge * edge_mask.unsqueeze(-1).type_as(updated_edge)
        
        return updated_edge
    


class MedKGATFusion_healnet(nn.Module):
    def __init__(self, args, embed_dim: int, 
             max_modalities: int = 10, 
             max_groups: int = 10, 
             ff_dropout_rate: float = 0.25, 
             attn_dropout_rate: float = 0.1, 
             num_intra_layers: int = 1, num_inter_layers: int = 1):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.drop_edge_ratio = 0.1

        # 1. Knowledge Projection (768 -> embed_dim)
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(ff_dropout_rate)
        )

        n_latent = 32
        # Latent Query tokens for each group
        self.group_latent = nn.ParameterList([
            nn.Parameter(torch.randn((n_latent, self.embed_dim)))
            for _ in range(max_groups)
        ])

        # 2. Intra-group Interaction (Now Cross-Attention + Self-Attention)
        self.num_intra_layers = num_intra_layers
        
        # Cross-Attention: Latent Q -> Modalities KV
        self.intra_group_cross_attn = nn.ModuleList([
            SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate)
            for _ in range(max_groups)
        ])
        
        # Self-Attention: Latent Q -> Latent KV
        self.intra_group_self_attn = nn.ModuleList([
            SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate)
            for _ in range(max_groups)
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

    def _intra_group_step(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], groups: List[List[int]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Modified Logic:
        1. Iterate over each group.
        2. Expand the Group Latent to batch size -> Query.
        3. Concat all modalities in the group -> Key/Value.
        4. Cross Attention: Latent queries Modalities.
        5. Self Attention: Latent queries Latent.
        Returns:
            group_embeddings: List of (B, n_latent, D)
            group_masks: List of (B, n_latent)
        """
        batch_size = embeddings[0].shape[0]
        device = embeddings[0].device
        
        group_embeddings = []
        group_masks = []
        
        for group_idx, group_indices in enumerate(groups):
            if not group_indices:
                # Handle empty group case (if any) -> Create zero placeholder
                latent_dim = self.group_latent[group_idx].shape[0]
                zero_embed = torch.zeros((batch_size, latent_dim, self.embed_dim), device=device)
                zero_mask = torch.zeros((batch_size, latent_dim), device=device)
                group_embeddings.append(zero_embed)
                group_masks.append(zero_mask)
                continue

            # 1. Prepare Query (Latent)
            # (n_latent, D) -> (B, n_latent, D)
            latent_query = self.group_latent[group_idx].unsqueeze(0).expand(batch_size, -1, -1)
            latent_dim = latent_query.shape[1]
            
            # 2. Prepare Key/Value (Concatenated Modalities)
            group_feats_list = [embeddings[i] for i in group_indices]
            group_masks_list = [masks[i] for i in group_indices]
            
            # (B, Sum_L, D)
            kv_feat = torch.cat(group_feats_list, dim=1)
            # (B, Sum_L) - 1 is valid, 0 is padding
            kv_mask = torch.cat(group_masks_list, dim=1)
            
            padding_mask = (kv_mask == 0) # True is invalid for MHA
            
            # Check for completely empty patients (all modalities padding) to handle safety
            all_masked_rows = padding_mask.all(dim=1)
            
            # 3. Cross Attention: Latent queries Modalities
            updated_latent = self.intra_group_cross_attn[group_idx](
                query=latent_query,
                key=kv_feat,
                value=kv_feat,
                key_padding_mask=padding_mask
            )
            
            # If a patient had NO valid info in any modality of this group, 
            # the latent output should be zeroed out (safe check).
            if all_masked_rows.any():
                updated_latent[all_masked_rows] = 0.0

            # 4. Self Attention: Latent queries Latent
            # Create mask for self-attention.
            # If the patient has valid data, latent is fully valid. 
            # If the patient has NO data (all_masked_rows), the latent is invalid (masked).
            latent_padding_mask = None
            if all_masked_rows.any():
                latent_padding_mask = torch.zeros((batch_size, latent_dim), dtype=torch.bool, device=device)
                latent_padding_mask[all_masked_rows] = True

            updated_latent = self.intra_group_self_attn[group_idx](
                query=updated_latent,
                key=updated_latent,
                value=updated_latent,
                key_padding_mask=latent_padding_mask
            )

            # Re-zero out just to be safe (layer norm/bias might introduce non-zero values)
            if all_masked_rows.any():
                updated_latent[all_masked_rows] = 0.0
                
            group_embeddings.append(updated_latent)
            
            # 5. Generate Mask for the Group Latent
            # The latent itself is always "present" (length 32), so mask is 1s.
            # UNLESS the patient had absolutely no input for this group, then mask is 0s.
            latent_mask = torch.ones((batch_size, latent_dim), device=device)
            if all_masked_rows.any():
                latent_mask[all_masked_rows] = 0.0
            
            group_masks.append(latent_mask)
            
        return group_embeddings, group_masks

    def _inter_group_step(self, target_node: torch.Tensor, target_mask: torch.Tensor,
                          source_node: torch.Tensor, source_mask: torch.Tensor,
                          edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                          layer_modules: nn.ModuleDict) -> torch.Tensor:
        """
        One-way interaction: Source -> Edge -> Target
        Note: Nodes here are now the Group Latents.
        """
        source_padding_mask = (source_mask == 0)
        edge_padding_mask = (edge_mask == 0)
        
        # Step 1: Edge queries Source (Latent) to get relevant info
        gated_source = layer_modules['edge_to_node_attn'](
            query=edge_feat, 
            key=source_node, 
            value=source_node, 
            key_padding_mask=source_padding_mask
        )
        
        # Step 2: Target (Latent) queries Gated Source to update itself
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
        current_proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            current_proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction & Latent Extraction
        # Now returns the Group Latents directly as the node representations
        # group_embeddings: List of (B, n_latent, D)
        # group_masks: List of (B, n_latent)
        group_embeddings, group_masks = self._intra_group_step(embeddings, masks, embeddings_groups)

        # Pre-calculate validity masks for Loss weights later
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # 3. Inter-Group Interaction (Multi-Layer GNN / GAT)
        # The nodes in the graph are now the Group Latents
        current_group_embeddings = group_embeddings 

        for layer_idx in range(self.num_inter_layers):
            layer_modules = self.shared_inter_layer
            
            for (idx_a, idx_b) in edge_keys:
                edge_feat = current_proj_knowledge.get((idx_a, idx_b))

                if self.training and getattr(self, 'drop_edge_ratio', 0.0) > 0.0:
                    if random.random() < self.drop_edge_ratio:
                        continue
                
                edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                if edge_mask is None:
                    edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

                # Get Group Data (Latents)
                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a]
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # --- GNN Update Logic ---
                # Update Edge Features
                updated_edge_feat = layer_modules['edge_updater'](
                    edge_feat, edge_mask, 
                    feat_a, mask_a, 
                    feat_b, mask_b
                )
                
                current_proj_knowledge[(idx_a, idx_b)] = updated_edge_feat

                # Update Node B (Latent) using Node A (Latent) and Edge
                update_for_b = self._inter_group_step(
                    target_node=feat_b, target_mask=mask_b,
                    source_node=feat_a, source_mask=mask_a,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_b] = update_for_b

                # Update Node A (Latent) using Node B (Latent) and Edge
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
        # 4. Data Collection for KL Loss (Post-GAT)
        # ------------------------------------------------------------------
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_edge_pairs_list = []
        edge_score_valid_flag = False

        for (idx_a, idx_b) in edge_keys:
            # Retrieve Ground Truth Edge Score
            edge_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            if edge_score is None:
                edge_score = torch.zeros(embeddings[0].shape[0], device=embeddings[0].device)
            
            if edge_score.dim() > 1:
                edge_score = edge_score.view(-1)
            if edge_score.dim() == 0:
                edge_score = edge_score.expand(embeddings[0].shape[0])

            has_a = group_validity_masks[idx_a].float()
            has_b = group_validity_masks[idx_b].float()
            pair_validity = has_a * has_b

            edge_score_valid_flag |= edge_score.sum().item() > 0
            all_edge_scores_list.append(edge_score)
            all_valid_masks_list.append(pair_validity)
            all_edge_pairs_list.append((idx_a, idx_b))

        # 5. Compute Similarities on FINAL Embeddings (Latents)
        all_cos_sims_list = []
        
        if len(all_edge_pairs_list) > 0 and edge_score_valid_flag:
            # Pool the Latents to get a single vector per group for cosine similarity
            final_pooled_results = [masked_mean_pool(g, m) for g, m in zip(final_group_embeddings, group_masks)]
            final_pooled_group_embeddings = [res[0] for res in final_pooled_results]
            final_pooled_group_embeddings = [F.normalize(g, p=2, dim=1) for g in final_pooled_group_embeddings]
            
            for idx_a, idx_b in all_edge_pairs_list:
                sim = torch.sum(final_pooled_group_embeddings[idx_a] * final_pooled_group_embeddings[idx_b], dim=1)
                sim = torch.clamp(sim, -1.0, 1.0)
                all_cos_sims_list.append(sim)

        # 6. Global Aggregation
        # Concatenate all Group Latents to form the global sequence
        global_concat = torch.cat(final_group_embeddings, dim=1) # (B, Num_Groups * n_latent, D)
        global_mask = torch.cat(group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        global_transformed = self.global_transformer(
            query=global_concat, key=global_concat, value=global_concat, key_padding_mask=global_padding_mask, need_weights=False)
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        fused_embedding, _ = masked_mean_pool(global_transformed, global_mask)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # 7. Compute KL Divergence Loss
        fusion_loss = torch.tensor(0.0, device=fused_embedding.device)
        
        if len(all_edge_scores_list) > 0 and edge_score_valid_flag:
            all_scores_tensor = torch.stack(all_edge_scores_list, dim=1)  
            all_sims_tensor = torch.stack(all_cos_sims_list, dim=1)       
            all_masks_tensor = torch.stack(all_valid_masks_list, dim=1)   

            scores_masked = all_scores_tensor.clone().float()
            scores_masked[all_masks_tensor == 0] = -1e9
            target_probs = F.softmax(scores_masked, dim=1)

            temperature = 0.1
            sims_masked = all_sims_tensor.clone() / temperature
            sims_masked[all_masks_tensor == 0] = -1e9
            
            pred_log_probs = F.log_softmax(sims_masked, dim=1)
            
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none')
            kl_loss_per_patient = kl_loss.sum(dim=1)
            
            valid_patients = (all_masks_tensor.sum(dim=1) > 1).float()
            
            if valid_patients.sum() > 0:
                fusion_loss = (kl_loss_per_patient * valid_patients).sum() / valid_patients.sum()
    
        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": 2 * fusion_loss,
            }
        }