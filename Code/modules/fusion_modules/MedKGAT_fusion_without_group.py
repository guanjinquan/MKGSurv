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



class SafeCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # batch_first=True 输出 (B, L, D)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query: (B, Lq, D)
        key:   (B, Lk, D)
        value: (B, Lk, D)
        key_padding_mask: (B, Lk), True 为 padding
        """
        
        # --- 核心修复逻辑 Start ---
        if key_padding_mask is not None:
            # 1. 检测哪些样本的所有 Key 都是 Padding
            # key_padding_mask: True 表示无效
            all_masked_rows = key_padding_mask.all(dim=1) # (B,) bool

            if all_masked_rows.any():
                # 2. 只有当存在全 Mask 的情况时，才进行克隆和修改，避免影响原始数据
                key_padding_mask = key_padding_mask.clone()
                
                # 3. 将全 Mask 行的第一个位置设为 False (有效)
                # 这样 Softmax 分母就不会是 0，避免 NaN
                key_padding_mask[all_masked_rows, 0] = False
        # --- 核心修复逻辑 End ---

        # 正常计算 MHA，此时绝对不会报 NaN
        attn_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        
        # --- 可选：对垃圾值进行清理 ---
        # 如果你非常介意那些原本无效的样本输出不为 0，可以在这里再次 mask 掉。
        # 但通常因为后续还有 global pooling 或者 target mask，这步不是必须的。
        if key_padding_mask is not None and all_masked_rows.any():
             # 将那些原本全无效的行的输出置为 0
             attn_out[all_masked_rows] = 0.0

        return self.norm(query + self.dropout(attn_out))
    

class EdgeContextualizer(nn.Module):
    """
    使用Edge作为Query，连接的节点特征作为Key/Value。
    让知识(Edge)根据具体的病人数据(Node)进行动态调整。
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = SafeCrossAttentionBlock(embed_dim, num_heads)

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
    


class MedKGATFusion_without_Group(nn.Module):
    def __init__(self, embed_dim: int, max_modalities: int = 10, dropout_rate: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        # 1. Knowledge Projection (768 -> embed_dim)
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(self.dropout_rate)
        )

        # 2. Intra-group Interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.intra_group_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 3. GAT Interaction Components
        # 注意：这里假设 SafeCrossAttentionBlock 和 EdgeContextualizer 已经在外部定义或引入
        # 如果没有定义，运行时会报错。这里为了代码完整性保留原样。
        # M_0_1 -> Feat_m1 (Gating)
        self.edge_to_node_attn = SafeCrossAttentionBlock(embed_dim, num_heads=8)
        # Feat_m1 -> Gated_m0 (Update)
        self.node_to_node_attn = SafeCrossAttentionBlock(embed_dim, num_heads=8)
        # Edge Updater
        self.edge_updater = EdgeContextualizer(embed_dim, num_heads=8)

        # 4. Global Aggregation
        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(global_encoder_layer, num_layers=1)

        # 5. Post Fusion Norm
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def _intra_group_interaction(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], 
                               groups: List[List[int]]) -> List[torch.Tensor]: 
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

            transformed = self.intra_group_transformer(concat_feat, src_key_padding_mask=padding_mask)
            
            if all_masked_rows.any():
                transformed[all_masked_rows] = 0.0

            split_feats = torch.split(transformed, lengths, dim=1)
            
            for i, idx in enumerate(group_indices):
                updated_embeddings[idx] = split_feats[i]
                
        return updated_embeddings

    def _inter_group_step(self, target_node: torch.Tensor, target_mask: torch.Tensor,
                          source_node: torch.Tensor, source_mask: torch.Tensor,
                          edge_feat: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        One-way interaction: Source -> Edge -> Target
        """
        source_padding_mask = (source_mask == 0)
        edge_padding_mask = (edge_mask == 0)
        
        # Step 1: Edge queries Source to get relevant info (Gating)
        gated_source = self.edge_to_node_attn(
            query=edge_feat, 
            key=source_node, 
            value=source_node, 
            key_padding_mask=source_padding_mask
        )
        
        # Step 2: Target queries Gated Source to update itself
        # Note: Key/Value mask depends on Edge because gated_source has shape of Edge
        updated_target = self.node_to_node_attn(
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
            if (j, i) in fusion_knowledge and i > j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        # 1. Project Knowledge Edges
        proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction
        # info_level_embeddings = self._intra_group_interaction(embeddings, masks, embeddings_groups)
        info_level_embeddings = embeddings  

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

        # 4. Inter-Group Interaction (GNN / GAT)
        num_groups = len(group_embeddings)
        group_updates_buffer = [[] for _ in range(num_groups)]

        # --- Data Collection for KL Loss Calculation ---
        # We collect (batch_size, ) tensors for every edge defined in the graph
        # But we will calculate Similarity AFTER the GNN updates.
        all_edge_scores_list = []
        all_valid_masks_list = [] # To handle patients missing specific groups
        all_edge_pairs_list = []  # Store indices (idx_a, idx_b) to compute Sim later
        edge_score_valid_flag = False

        # Pre-calculate validity masks for Weights (based on INPUT embeddings/masks)
        # We need this to determine if a patient has a specific modality group.
        # Note: GNN updates don't change padding, so initial mask is valid throughout.
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # Iterate over projected edges
        for (idx_a, idx_b), edge_feat in proj_knowledge.items():

            edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
            if edge_mask is None:
                edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

            # Retrieve Ground Truth Edge Score
            edge_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            if edge_score is None:
                edge_score = torch.zeros(edge_feat.shape[0], device=edge_feat.device) # Assume batch size
            
            # Ensure edge_score is (B,)
            if edge_score.dim() > 1:
                edge_score = edge_score.view(-1)
            if edge_score.dim() == 0:
                edge_score = edge_score.expand(edge_feat.shape[0])

            # Get Group Data (Inputs for GNN)
            feat_a = group_embeddings[idx_a]
            mask_a = group_masks[idx_a]
            feat_b = group_embeddings[idx_b]
            mask_b = group_masks[idx_b]

            # --- Validity Checks ---
            # weight_for_b: Does this patient have Group A? (B, 1, 1)
            has_a = group_validity_masks[idx_a].float()
            has_b = group_validity_masks[idx_b].float()
            
            # Pair validity for this patient
            pair_validity = has_a * has_b # (B,)

            weight_for_b = has_a.view(-1, 1, 1) # * edge_score.view(-1, 1, 1) 不用edge score输入，好像会掉点
            weight_for_a = has_b.view(-1, 1, 1) # * edge_score.view(-1, 1, 1)

            # --- GNN Update Logic ---
            updated_edge_feat = self.edge_updater(
                edge_feat, edge_mask, 
                feat_a, mask_a, 
                feat_b, mask_b
            )

            update_for_b = self._inter_group_step(
                target_node=feat_b, target_mask=mask_b,
                source_node=feat_a, source_mask=mask_a,
                edge_feat=updated_edge_feat, edge_mask=edge_mask
            )
            group_updates_buffer[idx_b].append((update_for_b, weight_for_b))

            update_for_a = self._inter_group_step(
                target_node=feat_a, target_mask=mask_a,
                source_node=feat_b, source_mask=mask_b,
                edge_feat=updated_edge_feat, edge_mask=edge_mask
            )
            group_updates_buffer[idx_a].append((update_for_a, weight_for_a))

            # --- Collect Data for Loss (Deferred Sim Calculation) ---
            edge_score_valid_flag |= edge_score.sum().item() > 0
            all_edge_scores_list.append(edge_score)
            all_valid_masks_list.append(pair_validity)
            all_edge_pairs_list.append((idx_a, idx_b))

        # 5. Apply Updates to Groups
        final_group_embeddings = []
        for i in range(num_groups):
            original_group_feat = group_embeddings[i]
            updates_and_weights = group_updates_buffer[i]

            if len(updates_and_weights) > 0:
                updates = [u for u, w in updates_and_weights]   
                weights = [w for u, w in updates_and_weights]   

                stacked_updates = torch.stack(updates, dim=0)
                stacked_weights = torch.stack(weights, dim=0)
                
                sum_updates = torch.sum(stacked_updates * stacked_weights, dim=0)
                sum_counts = torch.sum(stacked_weights, dim=0)
                
                aggregated_feat = torch.where(
                    sum_counts > 0,
                    sum_updates / sum_counts.clamp(min=1e-9),
                    original_group_feat
                )
                final_group_embeddings.append(aggregated_feat)
            else:
                final_group_embeddings.append(original_group_feat)

        # 6. Compute Similarities on FINAL Embeddings 
        all_cos_sims_list = []
        
        # Only compute if we have edges to compare
        if len(all_edge_pairs_list) > 0 and edge_score_valid_flag:
            # 1. Pool Final Group Embeddings
            # We use the original masks because padding doesn't change
            final_pooled_results = [masked_mean_pool(g, m) for g, m in zip(final_group_embeddings, group_masks)]
            
            # Extract just the embeddings (we already have validity from before)
            final_pooled_group_embeddings = [res[0] for res in final_pooled_results]
            
            # 2. Normalize
            final_pooled_group_embeddings = [F.normalize(g, p=2, dim=1) for g in final_pooled_group_embeddings]
            
            # 3. Compute Similarity for each recorded edge pair
            for idx_a, idx_b in all_edge_pairs_list:
                # Compute Cosine Similarity (B,)
                sim = torch.sum(final_pooled_group_embeddings[idx_a] * final_pooled_group_embeddings[idx_b], dim=1)
                
                # Clamp similarity to [-1, 1] for numerical stability
                sim = torch.clamp(sim, -1.0, 1.0)
                
                all_cos_sims_list.append(sim)

        # 7. Global Aggregation
        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        global_transformed = self.global_transformer(global_concat, src_key_padding_mask=global_padding_mask)
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        fused_embedding, _ = masked_mean_pool(global_transformed, global_mask)
             
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # 8. Compute KL Divergence Loss (Patient-wise) 
        fusion_loss = torch.tensor(0.0, device=fused_embedding.device)
        
        if len(all_edge_scores_list) > 0 and edge_score_valid_flag:
            # Stack into (B, Num_Edges)
            # Row `i` represents patient `i`, Columns represent all possible edges in the graph
            all_scores_tensor = torch.stack(all_edge_scores_list, dim=1) # (B, E)
            all_sims_tensor = torch.stack(all_cos_sims_list, dim=1)      # (B, E)
            all_masks_tensor = torch.stack(all_valid_masks_list, dim=1)  # (B, E) -> 1 if patient has both nodes, 0 else

            # Prepare for Softmax:
            # We want to compare the distribution of Scores vs Distribution of Sims *per patient*.
            # If a pair is invalid for a patient, its score/sim should not contribute to the distribution.
            # We set them to -1e9 so Softmax makes them ~0.
            
            # 1. Target Distribution (Edge Scores)
            # [FIX] Convert scores to float before softmax because weights might be Long integers
            scores_masked = all_scores_tensor.clone().float()
            scores_masked[all_masks_tensor == 0] = -1e9
            target_probs = F.softmax(scores_masked, dim=1) # (B, E) Sums to 1 per patient

            # 2. Predicted Distribution (Cosine Sims)
            # Map Cosine [-1, 1] to something softmax-friendly. 
            # Often useful to use a temperature T to sharpen/smooth
            temperature = 0.1 
            sims_masked = all_sims_tensor.clone() / temperature
            sims_masked[all_masks_tensor == 0] = -1e9
            
            # KLDivLoss expects: Input = Log_Softmax, Target = Probabilities
            pred_log_probs = F.log_softmax(sims_masked, dim=1) # (B, E)
            
            # 3. Compute KL Divergence
            # reduction='none' so we can mask out patients who have NO valid edges
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none') # (B, E)
            
            # Sum over edges dimension to get loss per patient
            kl_loss_per_patient = kl_loss.sum(dim=1) # (B,)
            
            # 4. Filter valid patients
            # If a patient has 0 or 1 valid edge, distribution matching might be trivial or undefined
            # We only average loss over patients who have at least 2 valid edges to form a distribution
            valid_patients = (all_masks_tensor.sum(dim=1) > 1).float()
            
            if valid_patients.sum() > 0:
                fusion_loss = (kl_loss_per_patient * valid_patients).sum() / valid_patients.sum()
            
        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": fusion_loss
            }
        }