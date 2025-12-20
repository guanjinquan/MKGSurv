"""
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.8220 ± 0.0389
 - List = [0.7997237569060773, 0.8683127572016461, 0.7795414462081128, 0.792803970223325, 0.8697394789579158]
C-Index-IPCW_Validation Set: 0.7802 ± 0.0465
 - List = [0.8047089406049975, 0.7973293421509138, 0.6895800484509258, 0.8209030653767571, 0.7885440718453258]
Test Summary:
C-Index_Test Set: 0.7517 ± 0.0341
 - List = [0.8093699515347335, 0.7225433526011561, 0.7712230215827338, 0.7350877192982456, 0.7201225740551583]
C-Index-IPCW_Test Set: 0.7231 ± 0.0793
 - List = [0.7970649742535062, 0.7245307599917301, 0.5745152080348945, 0.7840243585843866, 0.7352023151659869]
Training run tcga_kirc_run014 finished.
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
    





class MedKGATFusionMaxSim(nn.Module):
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

    @staticmethod
    def _compute_max_sim(feat_a: torch.Tensor, mask_a: torch.Tensor, 
                         feat_b: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
        """
        Late Interaction (MaxSim) 计算的核心逻辑。
        无需参数，计算 A 中每个 token 在 B 中找到的最佳匹配分数的均值。
        
        Args:
            feat_a: (B, La, D)
            mask_a: (B, La) - 1 valid, 0 invalid
            feat_b: (B, Lb, D)
            mask_b: (B, Lb)
        Returns:
            score: (B,)
        """
        # 1. L2 归一化，保证点积即为余弦相似度
        feat_a = F.normalize(feat_a, p=2, dim=-1) # (B, La, D)
        feat_b = F.normalize(feat_b, p=2, dim=-1) # (B, Lb, D)

        # 2. 计算相似度矩阵 (B, La, Lb)
        # sim_matrix[b, i, j] = feat_a[b, i] dot feat_b[b, j]
        sim_matrix = torch.bmm(feat_a, feat_b.transpose(1, 2))

        # 3. Mask B (Columns): 这里的 Mask 逻辑是不让 A 匹配到 B 的 padding
        # mask_b (B, Lb) -> (B, 1, Lb)
        mask_b_expanded = mask_b.unsqueeze(1).bool() 
        # 将无效列填充为负无穷，这样 max 操作就不会选中它们
        sim_matrix = sim_matrix.masked_fill(~mask_b_expanded, -1e9)

        # 4. Max over B: 为 A 中的每个 token 找到 B 中最相关的 token
        max_sim_values, _ = sim_matrix.max(dim=-1) # (B, La)

        # 5. Average over A: 只对 A 中有效的 token 求平均
        mask_a_bool = mask_a.bool()
        
        # 将 A 中 padding 位置的 max 分数置 0 (防止它们影响 sum)
        # 注意：这里置 0 而不是负无穷，因为我们要算 sum
        max_sim_values = max_sim_values.masked_fill(~mask_a_bool, 0.0)
        
        sum_sim = max_sim_values.sum(dim=-1) # (B,)
        valid_len_a = mask_a.sum(dim=-1)     # (B,)
        
        # 避免除以 0
        score = sum_sim / (valid_len_a + 1e-6)
        
        return score

    def _intra_group_step(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], groups: List[List[int]]) -> List[torch.Tensor]: 
        updated_embeddings = list(embeddings)
        
        for group_idx, group_indices in enumerate(groups):
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

        # 2. Intra-Group Interaction
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

        # Pre-calculate validity masks for Weights
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # 4. Inter-Group Interaction (Multi-Layer GAT)
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

                # Get Group Data
                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a]
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # Update Edge Features
                updated_edge_feat = layer_modules['edge_updater'](
                    edge_feat, edge_mask, 
                    feat_a, mask_a, 
                    feat_b, mask_b
                )
                current_proj_knowledge[(idx_a, idx_b)] = updated_edge_feat

                # Update Node B
                update_for_b = self._inter_group_step(
                    target_node=feat_b, target_mask=mask_b,
                    source_node=feat_a, source_mask=mask_a,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_b] = update_for_b

                # Update Node A
                update_for_a = self._inter_group_step(
                    target_node=feat_a, target_mask=mask_a,
                    source_node=feat_b, source_mask=mask_b,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_a] = update_for_a

        # Final embeddings
        final_group_embeddings = current_group_embeddings

        # ------------------------------------------------------------------
        # 4b. Data Collection for KL Loss
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

        # ------------------------------------------------------------------
        # 6. Compute Similarities using MaxSim (Late Interaction)
        # ------------------------------------------------------------------
        all_max_sims_list = []
        
        if len(all_edge_pairs_list) > 0 and edge_score_valid_flag:
            
            for idx_a, idx_b in all_edge_pairs_list:
                # 获取最新的 token-level 特征
                feat_a = final_group_embeddings[idx_a]
                mask_a = group_masks[idx_a]
                feat_b = final_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # MaxSim 是非对称的，通常计算双向均值
                score_ab = self._compute_max_sim(feat_a, mask_a, feat_b, mask_b)
                score_ba = self._compute_max_sim(feat_b, mask_b, feat_a, mask_a)
                
                # 双向均值 (范围约 -1 ~ 1)
                sim = (score_ab + score_ba) / 2.0
                sim = torch.clamp(sim, -1.0, 1.0) # 稍微截断一下保证数值稳定
                
                all_max_sims_list.append(sim)

        # 7. Global Aggregation
        global_concat = torch.cat(final_group_embeddings, dim=1)
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

        # 8. Compute KL Divergence Loss
        fusion_loss = torch.tensor(0.0, device=fused_embedding.device)
        
        if len(all_edge_scores_list) > 0 and edge_score_valid_flag:
            all_scores_tensor = torch.stack(all_edge_scores_list, dim=1)  # (Batch, Num_Edges) - LLM GT
            all_sims_tensor = torch.stack(all_max_sims_list, dim=1)       # (Batch, Num_Edges) - MaxSim Pred
            all_masks_tensor = torch.stack(all_valid_masks_list, dim=1)   # (Batch, Num_Edges)

            # 8.1 处理 LLM GT -> Softmax 概率分布
            scores_masked = all_scores_tensor.clone().float()
            scores_masked[all_masks_tensor == 0] = -1e9
            target_probs = F.softmax(scores_masked, dim=1)

            # 8.2 处理 MaxSim -> LogSoftmax
            # 这里的 temperature 非常关键！
            # 因为 MaxSim 的值域通常在 0.2~0.8 之间，直接 softmax 会很平坦
            temperature = 0.1 
            sims_masked = all_sims_tensor.clone() / temperature
            sims_masked[all_masks_tensor == 0] = -1e9
            
            pred_log_probs = F.log_softmax(sims_masked, dim=1)
            
            # 8.3 KL Divergence
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none')
            kl_loss_per_patient = kl_loss.sum(dim=1)
            
            valid_patients = (all_masks_tensor.sum(dim=1) > 1).float() 
            
            if valid_patients.sum() > 0:
                fusion_loss = (kl_loss_per_patient * valid_patients).sum() / (valid_patients.sum() + 1e-6)
    
        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": 2 * fusion_loss,
            }
        }