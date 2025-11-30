import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from modules.base_modules.aggregation_utils import masked_mean_pool


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
    


class MedKGATFusion(nn.Module):
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
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        # 0. Ensure symmetric keys removal (GNN uses bidirectional edges implicitly if designed so)
        for (i, j), v in list(fusion_knowledge.items()): 
            if (j, i) in fusion_knowledge and i > j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        # 1. Project Knowledge Edges
        proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction
        # (Modality level mixing within groups)
        info_level_embeddings = self._intra_group_interaction(embeddings, masks, embeddings_groups)

        # 3. Create Group-Level Embeddings
        # Concatenate tokens from all modalities within a group to form a "Group Node"
        group_embeddings = []
        group_masks = []
    
        for group_indices in embeddings_groups:
            if not group_indices:
                # Handle empty groups if necessary, creating dummy zero tensor
                # For now assume groups are populated
                raise ValueError("Empty group found in embeddings_groups")

            # Collect features and masks for this group
            # info_level_embeddings has shape (B, Li, D)
            curr_feats = [info_level_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]

            # Concatenate along sequence length dimension
            # Result: (B, Sum(Li), D)
            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)

            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # 4. Inter-Group Interaction (GNN / GAT)
        # Iterate over edges defined in fusion_knowledge (which are Group-to-Group edges)
        num_groups = len(group_embeddings)
        # Buffer now stores tuples: (update_tensor, validity_mask)
        group_updates_buffer = [[] for _ in range(num_groups)]
        
        # Iterate over projected edges
        # key (idx_a, idx_b) refers to indices in `embeddings_groups` (i.e., Group A and Group B)
        for (idx_a, idx_b), edge_feat in proj_knowledge.items():
            edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
            if edge_mask is None:
                # Fallback if mask missing
                edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

            # Get Group Data
            feat_a = group_embeddings[idx_a]
            mask_a = group_masks[idx_a]
            feat_b = group_embeddings[idx_b]
            mask_b = group_masks[idx_b]

            # --- Validity Checks for Aggregation ---
            # Determine if Source Group and Edge are valid for the current patient in batch
            # Shape: (B, 1, 1) for broadcasting
            weight_for_b = (mask_a.sum(dim=1) > 0).float().view(-1, 1, 1)    # B receives from A
            weight_for_a = (mask_b.sum(dim=1) > 0).float().view(-1, 1, 1)    # A receives from B


            # 4.1 Update Edge Context based on connected Groups
            updated_edge_feat = self.edge_updater(
                edge_feat, edge_mask, 
                feat_a, mask_a, 
                feat_b, mask_b
            )

            # 4.2 Message Passing A -> B (Update B using A info)
            update_for_b = self._inter_group_step(
                target_node=feat_b, target_mask=mask_b,
                source_node=feat_a, source_mask=mask_a,
                edge_feat=updated_edge_feat, edge_mask=edge_mask
            )
            # Store update along with its validity weight
            group_updates_buffer[idx_b].append((update_for_b, weight_for_b))

            # 4.3 Message Passing B -> A (Update A using B info)
            update_for_a = self._inter_group_step(
                target_node=feat_a, target_mask=mask_a,
                source_node=feat_b, source_mask=mask_b,
                edge_feat=updated_edge_feat, edge_mask=edge_mask
            )
            group_updates_buffer[idx_a].append((update_for_a, weight_for_a))

        # 5. Apply Updates to Groups
        # Masked Mean Aggregation: Sum(Updates * Weights) / Sum(Weights)
        final_group_embeddings = []
        
        for i in range(num_groups):
            original_group_feat = group_embeddings[i]
            updates_and_weights = group_updates_buffer[i]

            if len(updates_and_weights) > 0:
                # Unzip updates and weights
                updates = [u for u, w in updates_and_weights] + [original_group_feat]
                weights = [w for u, w in updates_and_weights] + [(group_masks[i].sum(dim=1) > 0).float().view(-1, 1, 1)]

                # Stack: (Num_Neighbors, B, L_group, D) and (Num_Neighbors, B, 1, 1)
                stacked_updates = torch.stack(updates, dim=0)
                stacked_weights = torch.stack(weights, dim=0)
                
                # Weighted Sum of Updates (Invalid updates contribute 0)
                sum_updates = torch.sum(stacked_updates * stacked_weights, dim=0) # (B, L, D)        
                # Sum of Weights (Count of valid neighbors)
                sum_counts = torch.sum(stacked_weights, dim=0) # (B, 1, 1)
                
                # Compute Mean only where sum_counts > 0
                # If a patient has NO valid neighbors (sum_counts == 0), keep original features
                aggregated_feat = torch.where(
                    sum_counts > 0,
                    sum_updates / sum_counts.clamp(min=1e-9), # Safe division
                    original_group_feat
                )
                
                final_group_embeddings.append(aggregated_feat)
            else:
                # No neighbors in knowledge graph
                final_group_embeddings.append(original_group_feat)

        # 6. Global Aggregation
        # Flatten all groups into one sequence for the final transformer
        # (B, Sum(All_L), D)
        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1) # Re-concat masks similarly


        global_padding_mask = (global_mask == 0)

        # Safety check
        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            # Handle strictly or clone/unmask
            # Here we follow the Safe logic used before or raise error as per user code
            # raise ValueError("There is a patient without any modalities...")
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False


        # save final_group_embeddings into pkl
        # PID: 



        global_transformed = self.global_transformer(global_concat, src_key_padding_mask=global_padding_mask)
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        fused_embedding, pooled_mask = masked_mean_pool(global_transformed, global_mask)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        return {"fused_embedding": fused_embedding}