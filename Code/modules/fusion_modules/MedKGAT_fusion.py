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

        # 1. 知识投影层 (768 -> embed_dim)
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(self.dropout_rate)
        )

        # 2. 组内交互 (Intra-group) - 所有group使用同一个，方便把各模态信息对齐到同一个空间
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.intra_group_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 3. GAT 交互组件
        # 3.1 你的 Cross-Attention 逻辑组件
        # M_0_1 -> Feat_m1 (Gating)
        self.edge_to_node_attn = SafeCrossAttentionBlock(embed_dim, num_heads=8)
        # Feat_m1 -> Gated_m0 (Update)
        self.node_to_node_attn = SafeCrossAttentionBlock(embed_dim, num_heads=8)
        
        # 3.2 Edge更新组件
        self.edge_updater = EdgeContextualizer(embed_dim, num_heads=8)


        # 4. 全局聚合 (Global Aggregation)
        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(global_encoder_layer, num_layers=1)

        # 5. 聚合后的归一化层，用于恢复 L2 Norm 分布
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def _intra_group_interaction(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], 
                               groups: List[List[int]]) -> List[torch.Tensor]: 
        # 创建副本以免原地修改影响后续
        updated_embeddings = list(embeddings)
        
        for group_indices in groups:
            if not group_indices:
                continue
                
            # 收集该组的所有特征和mask
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            
            # 记录切分长度
            lengths = [f.shape[1] for f in group_feats]
            
            # 拼接 (B, Sum(Ni), D)
            concat_feat = torch.cat(group_feats, dim=1)
            concat_mask = torch.cat(group_masks, dim=1)
            
            # Transformer Encoder 需要 src_key_padding_mask (True为mask掉)
            # 输入mask: 1有效, 0无效 -> 转换: True无效, False有效
            padding_mask = (concat_mask == 0)
            
            # --- 1. 安全修复: 防止 Transformer 报 NaN ---
            # 检测哪些样本在这个组内完全没有数据 (整行为 True)
            all_masked_rows = padding_mask.all(dim=1) # (B,)
            
            if all_masked_rows.any():
                # 只有存在全 Mask 情况时才 clone，节省显存
                padding_mask = padding_mask.clone()
                padding_mask[all_masked_rows, 0] = False

            # 交互
            transformed = self.intra_group_transformer(concat_feat, src_key_padding_mask=padding_mask)
            
            # --- 2. 逻辑修复: 将原本全 Mask 的数据强制置为 0 ---
            if all_masked_rows.any():
                transformed[all_masked_rows] = 0.0

            # 切分回原来的形状
            split_feats = torch.split(transformed, lengths, dim=1)
            
            # 更新列表
            for i, idx in enumerate(group_indices):
                updated_embeddings[idx] = split_feats[i]
                
        return updated_embeddings

    def _interaction_step(self, target_node: torch.Tensor, target_mask: torch.Tensor,
                          source_node: torch.Tensor, source_mask: torch.Tensor,
                          edge_feat: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        单边交互:
        1. gated_source = CrossAttn(Q=Edge, K=Source, V=Source)
        2. updated_target = CrossAttn(Q=Target, K=gated_source, V=gated_source) + Target (残差在Block里做了)
        """
        # 准备padding masks (True为无效)
        source_padding_mask = (source_mask == 0)
        edge_padding_mask = (edge_mask == 0)
        
        # Step 1: Knowledge filters Source Modality
        # Q=Edge, K=Source, V=Source
        # 输出形状: (B, Edge_Len, D)
        gated_source = self.edge_to_node_attn(
            query=edge_feat, 
            key=source_node, 
            value=source_node, 
            key_padding_mask=source_padding_mask
        )
        
        # Step 2: Update Target using Gated Source
        # Q=Target, K=Gated_Source, V=Gated_Source
        # 注意：这里Key Mask应该基于Edge的Mask，因为gated_source的长度等于Edge长度
        updated_target = self.node_to_node_attn(
            query=target_node,
            key=gated_source,
            value=gated_source,
            key_padding_mask=edge_padding_mask
        )
        
        # Apply Target Mask: 确保无效的 Target Token 输出保持为 0
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
        
        batch_size = embeddings[0].shape[0]
        num_modalities = len(embeddings)

        # 0. 保证知识对称性, GNN已经利用了双向边
        for (i, j), v in list(fusion_knowledge.items()): 
            if (j, i) in fusion_knowledge:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        # 1. 知识投影
        proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            proj_knowledge[k] = self.know_proj(v)

        # 2. 组内交互
        current_embeddings = self._intra_group_interaction(embeddings, masks, embeddings_groups)

        # 3. GAT / GNN 交互
        updates_buffer = [[] for _ in range(num_modalities)]
        
        for (idx_a, idx_b), edge_feat in proj_knowledge.items():
            edge_mask = fusion_knowledge_mask[(idx_a, idx_b)]
            
            feat_a = current_embeddings[idx_a]
            mask_a = masks[idx_a]
            feat_b = current_embeddings[idx_b]
            mask_b = masks[idx_b]
            
            updated_edge_feat = self.edge_updater(
                edge_feat, edge_mask, 
                feat_a, mask_a, 
                feat_b, mask_b
            )
            
            update_for_a = self._interaction_step(
                target_node=feat_a, target_mask=mask_a,
                source_node=feat_b, source_mask=mask_b,
                edge_feat=updated_edge_feat, edge_mask=edge_mask
            )
            updates_buffer[idx_a].append(update_for_a)
            
            update_for_b = self._interaction_step(
                target_node=feat_b, target_mask=mask_b,
                source_node=feat_a, source_mask=mask_a,
                edge_feat=updated_edge_feat, edge_mask=edge_mask
            )
            updates_buffer[idx_b].append(update_for_b)

        next_layer_embeddings = []
        for i in range(num_modalities):
            original_feat = current_embeddings[i]
            updates = updates_buffer[i]
            
            if len(updates) > 0:
                stacked_updates = torch.stack(updates, dim=0)
                mean_updates = torch.mean(stacked_updates, dim=0)
                next_feat = mean_updates  # InterationStep的CrossAttention已经使用残差连接，这里直接平均所有更新
            else:
                next_feat = original_feat
            
            next_layer_embeddings.append(next_feat)

        # 4. 全局聚合
        global_concat = torch.cat(next_layer_embeddings, dim=1)
        global_mask = torch.cat(masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            raise ValueError("There is a patient without any modalities, please check the data.")

        global_padding_mask = (global_mask == 0)
        global_transformed = self.global_transformer(global_concat, src_key_padding_mask=global_padding_mask)
        
        fused_embedding, pooled_mask = masked_mean_pool(global_transformed, global_mask)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        return {"fused_embedding": fused_embedding}




