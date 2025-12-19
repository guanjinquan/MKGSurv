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
    



# --- Main Fusion Class ---
class MedKGATFusion(nn.Module):
    def __init__(self, args, embed_dim: int, 
             max_modalities: int = 10, 
             max_groups: int = 10, 
             ff_dropout_rate: float = 0.25, 
             attn_dropout_rate: float = 0.1, 
             num_layers: int = 1): 
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.drop_edge_ratio = 0.1
        self.dropout = nn.Dropout(attn_dropout_rate)

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(ff_dropout_rate)
        )

        # 2. Stacked Layers (Intra + Inter)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_modules = nn.ModuleDict({
                # Intra-Group: Self-attention within a group
                'intra_group_transformer': SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate),
                
                # Intra-Group Enhancement: Context Gating
                'intra_group_gate': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.Sigmoid()
                ),
                
                # Inter-Group: Edge Update logic
                'edge_updater': EdgeContextualizer(embed_dim, num_heads=8),
                
                # Inter-Group Enhancement: Joint Neighbor-Edge Attention
                # Target Node attends to [Neighbor Node, Edge Feature]
                'inter_node_updater': SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate),
            })
            self.layers.append(layer_modules)

        # --- Revert to Global Transformer + Mean Pool ---
        # This is the stable, proven approach.
        self.global_transformer = SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate)
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def _intra_group_step(self, 
                          group_embeddings: List[torch.Tensor], 
                          group_masks: List[torch.Tensor], 
                          layer_module: nn.ModuleDict) -> List[torch.Tensor]: 
        """
        Improvement: Added Context Gating after Self-Attention.
        """
        updated_groups = []
        for g_feat, g_mask in zip(group_embeddings, group_masks):
            padding_mask = (g_mask == 0) 
            
            # Safety for fully padded groups
            all_masked_rows = padding_mask.all(dim=1)
            if all_masked_rows.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_masked_rows, 0] = False

            # 1. Self Attention
            x = layer_module['intra_group_transformer'](
                query=g_feat, key=g_feat, value=g_feat, key_padding_mask=padding_mask
            )

            if all_masked_rows.any():
                x[all_masked_rows] = 0.0

            # 2. Context Gating (New Improvement)
            # Calculate mean context of the group to determine what's important
            # (B, L, D) -> (B, D)
            context, _ = masked_mean_pool(x, g_mask) 
            # (B, D) -> (B, 1, D)
            gate = layer_module['intra_group_gate'](context).unsqueeze(1)
            
            # Apply gate: Suppress noise if the group context suggests so
            x = x * gate
            
            updated_groups.append(x)
        return updated_groups

    def _inter_group_interaction(self, 
                                 target_node: torch.Tensor, target_mask: torch.Tensor,
                                 source_node: torch.Tensor, source_mask: torch.Tensor,
                                 edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                                 layer_module: nn.ModuleDict) -> torch.Tensor:
        """
        Improvement: Joint Neighbor-Edge Attention.
        Target attends to [Source, Edge] simultaneously.
        """
        # 1. Concatenate Source and Edge to form the Key/Value context
        # Source: (B, Ls, D), Edge: (B, Le, D) -> Context: (B, Ls+Le, D)
        kv_context = torch.cat([source_node, edge_feat], dim=1)
        
        # 2. Concatenate Masks
        # Source Mask: (B, Ls), Edge Mask: (B, Le) -> (B, Ls+Le)
        kv_mask = torch.cat([source_mask, edge_mask], dim=1)
        
        # Convert to MHA format (True = Padding)
        key_padding_mask = (kv_mask == 0)

        # 3. Target queries the joint context
        # This allows Target to see the Neighbor *through the lens of* the Edge
        updated_target = layer_module['inter_node_updater'](
            query=target_node, 
            key=kv_context, 
            value=kv_context, 
            key_padding_mask=key_padding_mask
        )
        
        # Apply mask to keep original padding zero
        if target_mask is not None:
            updated_target = updated_target * target_mask.unsqueeze(-1).type_as(updated_target)

        return updated_target

    def _inter_group_step_full(self, 
                               group_embeddings: List[torch.Tensor], 
                               group_masks: List[torch.Tensor],
                               current_edges: Dict[Tuple[int, int], torch.Tensor],
                               edge_masks: Dict[Tuple[int, int], torch.Tensor],
                               edge_keys: List[Tuple[int, int]],
                               layer_module: nn.ModuleDict) -> Tuple[List[torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]:
        
        next_edges = {k: v.clone() for k, v in current_edges.items()}

        for (idx_a, idx_b) in edge_keys:
            edge_feat = current_edges.get((idx_a, idx_b))
            
            if self.training and self.drop_edge_ratio > 0.0:
                if random.random() < self.drop_edge_ratio:
                    continue
            
            edge_mask = edge_masks.get((idx_a, idx_b))
            if edge_mask is None:
                edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

            feat_a = group_embeddings[idx_a]
            mask_a = group_masks[idx_a]
            feat_b = group_embeddings[idx_b]
            mask_b = group_masks[idx_b]

            # 1. Update Edge (Unchanged, uses EdgeContextualizer)
            updated_edge_feat = layer_module['edge_updater'](
                edge_feat, edge_mask, feat_a, mask_a, feat_b, mask_b
            )
            next_edges[(idx_a, idx_b)] = updated_edge_feat

            # 2. Update Node B (Jointly attends A and Edge)
            update_for_b = self._inter_group_interaction(
                target_node=feat_b, target_mask=mask_b,
                source_node=feat_a, source_mask=mask_a,
                edge_feat=updated_edge_feat, edge_mask=edge_mask,
                layer_module=layer_module
            )
            group_embeddings[idx_b] = update_for_b

            # 3. Update Node A (Jointly attends B and Edge)
            update_for_a = self._inter_group_interaction(
                target_node=feat_a, target_mask=mask_a,
                source_node=feat_b, source_mask=mask_b,
                edge_feat=updated_edge_feat, edge_mask=edge_mask,
                layer_module=layer_module
            )
            group_embeddings[idx_a] = update_for_a

        return group_embeddings, next_edges
    
    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        # 0. Cleanup
        for (i, j), v in list(fusion_knowledge.items()): 
            if (j, i) in fusion_knowledge and i < j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))
        edge_keys = list(fusion_knowledge.keys())
        if self.training: random.shuffle(edge_keys)

        # 1. Projection
        current_edges = {}
        for k, v in fusion_knowledge.items():
            current_edges[k] = self.know_proj(v)

        # 2. Form Initial Groups
        group_embeddings = []
        group_masks = []
        for group_indices in embeddings_groups:
            if not group_indices: raise ValueError("Empty group")
            curr_feats = [embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            group_embeddings.append(torch.cat(curr_feats, dim=1))
            group_masks.append(torch.cat(curr_masks, dim=1))

        # Validity Check
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # 3. Graph Layers
        for layer_module in self.layers:
            group_embeddings = self._intra_group_step(group_embeddings, group_masks, layer_module)
            group_embeddings, current_edges = self._inter_group_step_full(
                group_embeddings, group_masks, current_edges, fusion_knowledge_mask, edge_keys, layer_module
            )

        # --- 4. Global Fusion (Reverted to Transformer + Mean Pool) ---
        
        # Concatenate all groups
        global_concat = torch.cat(group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        # Safety Check for Transformer
        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        # Global Cross-Modal Interaction
        global_transformed = self.global_transformer(
            query=global_concat, 
            key=global_concat, 
            value=global_concat, 
            key_padding_mask=global_padding_mask
        )
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        # Mean Pooling for final embedding
        fused_embedding, _ = masked_mean_pool(global_transformed, global_mask)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # --- 5. KL Loss Calculation ---
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_cos_sims_list = []
        edge_score_valid_flag = False

        final_pooled_results = [masked_mean_pool(g, m) for g, m in zip(group_embeddings, group_masks)]
        final_pooled_vecs = [F.normalize(res[0], p=2, dim=1) for res in final_pooled_results]

        for (idx_a, idx_b) in edge_keys:
            edge_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            if edge_score is None: edge_score = torch.zeros(fused_embedding.shape[0], device=fused_embedding.device)
            if edge_score.dim() > 1: edge_score = edge_score.view(-1)
            if edge_score.dim() == 0: edge_score = edge_score.expand(fused_embedding.shape[0])

            has_a = group_validity_masks[idx_a].float()
            has_b = group_validity_masks[idx_b].float()
            pair_validity = has_a * has_b
            edge_score_valid_flag |= (edge_score.sum().item() > 0)
            
            all_edge_scores_list.append(edge_score)
            all_valid_masks_list.append(pair_validity)

            sim = torch.sum(final_pooled_vecs[idx_a] * final_pooled_vecs[idx_b], dim=1)
            sim = torch.clamp(sim, -1.0, 1.0)
            all_cos_sims_list.append(sim)

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
            
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none').sum(dim=1)
            valid_patients = (all_masks_tensor.sum(dim=1) > 1).float()
            
            if valid_patients.sum() > 0:
                fusion_loss = (kl_loss * valid_patients).sum() / valid_patients.sum()
    
        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": 2 * fusion_loss,
            }
        }