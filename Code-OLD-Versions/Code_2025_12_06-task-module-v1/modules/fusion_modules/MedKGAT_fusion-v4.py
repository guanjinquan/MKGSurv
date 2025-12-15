"""
LUAD:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6240 ± 0.0420
 - List = [0.6019339242546333, 0.5545954438334643, 0.6351457840819543, 0.6730007917656373, 0.655436447166922]
C-Index-IPCW_Validation Set: 0.5841 ± 0.0487
 - List = [0.5493144805132043, 0.514160666541348, 0.587795671556513, 0.6167431235832589, 0.6525874516542554]
Test Summary:
C-Index_Test Set: 0.6319 ± 0.0436
 - List = [0.6792452830188679, 0.6810584958217271, 0.623989218328841, 0.6083333333333333, 0.5670023237800155]
C-Index-IPCW_Test Set: 0.5964 ± 0.0703
 - List = [0.5925055015932195, 0.7294198027044958, 0.5812463549725575, 0.5282941463612507, 0.5502892011076638]
Training run tcga_luad_run001 finished.


LUSC:

--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6326 ± 0.0331
 - List = [0.6656908665105387, 0.655561829474873, 0.6494002181025081, 0.5745734727572922, 0.6179073399271213]
C-Index-IPCW_Validation Set: 0.6113 ± 0.0506
 - List = [0.681733110526359, 0.6499388772572924, 0.6129992497559165, 0.5668792640310187, 0.5451123524654183]
Test Summary:
C-Index_Test Set: 0.6224 ± 0.0367
 - List = [0.5502793296089385, 0.6280814576634512, 0.6439123790117167, 0.6478021978021978, 0.6421180766889836]
C-Index-IPCW_Test Set: 0.6478 ± 0.0361
 - List = [0.5932648485657376, 0.647860737870939, 0.6881047607904826, 0.6850809463827991, 0.6248887353869791]
Training run tcga_lusc_run001 finished.
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


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class RelationAwareCrossAttention(nn.Module):
    """
    [核心设计] 关系感知交叉注意力
    不同于简单的门控，我们将 Edge 信息直接注入到 Key 和 Value 中。
    这使得 Target 节点在查询 Source 节点时，会受到 Edge (关系) 的显式引导。
    
    Formula: Attention(Q=Target, K=Source+Edge, V=Source+Edge)
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Edge 适配器，将 Edge 特征转换到与 Node 相同的语义空间
        self.edge_proj = nn.Linear(embed_dim, embed_dim)
        
        # FFN 部分
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout=dropout)

    def forward(self, target, source, edge, source_mask=None):
        """
        target: (B, Lt, D)
        source: (B, Ls, D)
        edge: (B, D) -> 将被广播到 (B, Ls, D)
        source_mask: (B, Ls) True for valid
        """
        # Pre-Norm 结构，利于深层训练
        target_norm = self.norm(target)
        source_norm = self.norm(source) # 共享 Norm 或独立 Norm 均可，这里用独立实例但在外部共享参数逻辑
        
        # 1. Relation Injection (关键步骤)
        # 将 Edge 信息广播并注入到 Source 的特征中
        edge_feat = self.edge_proj(edge).unsqueeze(1) # (B, 1, D)
        
        # Key 和 Value 携带了关系的语义
        # 例如：如果关系是"抑制"，Source特征向量的方向会发生特定的偏移
        key_value_input = source_norm + edge_feat 
        
        # 2. Mask 处理 (MHA 需要 True 为 Padding/Invalid)
        key_padding_mask = None
        if source_mask is not None:
            key_padding_mask = (source_mask == 0)
            # 防 NaN 保护
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
        
        # 3. Attention Interaction
        # Target 查询 (Source + Edge)
        attn_out, _ = self.mha(
            query=target_norm, 
            key=key_value_input, 
            value=key_value_input, 
            key_padding_mask=key_padding_mask
        )
        
        if source_mask is not None:
             # 再次清理无效行的输出（虽然 MHA 内部处理了 softmax，但 output 可能仍有残留）
             all_masked = (source_mask == 0).all(dim=1)
             if all_masked.any():
                 attn_out[all_masked] = 0.0

        # Residual Connection
        x = target + self.dropout(attn_out)
        
        # FFN
        x = x + self.ffn(self.norm_ffn(x))
        return x

class SelfAttnEncoder(nn.Module):
    """
    标准的 Transformer Encoder Layer，用于模态内部的特征精炼
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-Norm 更稳定
        )
        self.norm = nn.LayerNorm(embed_dim) # Post-layer norm for output stability

    def forward(self, x, mask=None):
        # x: (B, L, D)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        
        # 防 NaN
        if src_key_padding_mask is not None:
             all_masked = src_key_padding_mask.all(dim=1)
             if all_masked.any():
                 src_key_padding_mask = src_key_padding_mask.clone()
                 src_key_padding_mask[all_masked, 0] = False
        
        out = self.layer(x, src_key_padding_mask=src_key_padding_mask)
        
        if src_key_padding_mask is not None and src_key_padding_mask.all(dim=1).any():
            all_masked = src_key_padding_mask.all(dim=1)
            out[all_masked] = 0.0
            
        return self.norm(out)


class MedKGATFusion(nn.Module):
    def __init__(self, args, embed_dim: int, 
             max_modalities: int = 10, 
             max_groups: int = 10, 
             ff_dropout_rate: float = 0.1, 
             attn_dropout_rate: float = 0.1, 
             num_intra_layers: int = 1, 
             num_inter_layers: int = 2): 
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.drop_edge_ratio = 0.0 # 关闭 Drop Edge，防止欠拟合

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # 2. Intra-Group Refinement (Self-Attention)
        # 每个组先进行内部特征整合，理解自己的上下文
        self.intra_group_layers = nn.ModuleList([
            SelfAttnEncoder(embed_dim, num_heads=4, dropout=attn_dropout_rate)
            for _ in range(num_intra_layers)
        ])

        # 3. Inter-Group Interaction (Relation-Aware Cross Attention)
        # 这是新设计的核心
        self.num_inter_layers = num_inter_layers
        self.inter_layer = nn.ModuleDict({
            'rel_cross_attn': RelationAwareCrossAttention(embed_dim, num_heads=4, dropout=attn_dropout_rate)
        })
        
        # 4. Global Fusion
        # 将所有组的特征聚合
        self.global_transformer = SelfAttnEncoder(embed_dim, num_heads=4, dropout=attn_dropout_rate)
        self.post_fusion_norm = nn.LayerNorm(embed_dim)
        
        # Learnable Temp
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.3025)

    def _intra_group_process(self, embeddings, masks, groups):
        # 对每个组内的特征进行 Self-Attention 增强
        updated_embeddings = list(embeddings)
        
        for group_indices in groups:
            if not group_indices: continue
            
            # Gather features
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            lengths = [f.shape[1] for f in group_feats]
            
            # Concat -> (B, Total_L, D)
            concat_feat = torch.cat(group_feats, dim=1)
            concat_mask = torch.cat(group_masks, dim=1)
            
            # Apply Self Attention
            for layer in self.intra_group_layers:
                concat_feat = layer(concat_feat, mask=concat_mask)
            
            # Scatter back
            split_feats = torch.split(concat_feat, lengths, dim=1)
            for i, idx in enumerate(group_indices):
                updated_embeddings[idx] = split_feats[i]
                
        return updated_embeddings

    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        # 0. Data Prep
        for (i, j), v in list(fusion_knowledge.items()): 
            if (j, i) in fusion_knowledge and i < j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))
        edge_keys = list(fusion_knowledge.keys())
        if self.training: random.shuffle(edge_keys)

        # 1. Project Edge Knowledge (Static)
        projected_edges = {k: self.know_proj(v) for k, v in fusion_knowledge.items()}

        # 2. Intra-Group Processing (Self-Attention)
        # 这一步非常重要，确保每个模态在交互前已经充分理解了自身内容
        processed_embeddings = self._intra_group_process(embeddings, masks, embeddings_groups)

        # 3. Construct Group Embeddings
        # 我们这里保留序列维度 (B, L_group, D)，而不是过早 pooling
        group_embeddings = [] 
        group_masks = []
        
        for group_indices in embeddings_groups:
            curr_feats = [processed_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            # Concat along sequence dim
            g_feat = torch.cat(curr_feats, dim=1) # (B, L_total, D)
            g_mask = torch.cat(curr_masks, dim=1) # (B, L_total)
            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # 4. Inter-Group Interaction (Relation-Aware GAT)
        # 迭代更新多次
        current_group_embeddings = group_embeddings
        
        for layer_idx in range(self.num_inter_layers):
            next_step_embeddings = [e.clone() for e in current_group_embeddings]
            
            for (idx_a, idx_b) in edge_keys:
                edge_feat = projected_edges[(idx_a, idx_b)] # (B, D)
                if edge_feat.dim() == 3:
                    # 如果 Edge 也是序列，先 Pool 成向量，或者扩展 Attention 机制
                    # 这里简化为向量注入
                    e_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                    edge_feat, _ = masked_mean_pool(edge_feat, e_mask)
                
                feat_a = current_group_embeddings[idx_a] # (B, La, D)
                mask_a = group_masks[idx_a]
                feat_b = current_group_embeddings[idx_b] # (B, Lb, D)
                mask_b = group_masks[idx_b]
                
                # A 吸收 B 的信息 (Guided by Edge)
                # update_a = RelationCrossAttn(Q=A, K=B+E, V=B+E)
                # 注意：这会返回与 A 形状相同的 Tensor，包含了 B 的上下文信息
                update_a = self.inter_layer['rel_cross_attn'](
                    target=feat_a, source=feat_b, edge=edge_feat, source_mask=mask_b
                )
                
                # B 吸收 A 的信息
                update_b = self.inter_layer['rel_cross_attn'](
                    target=feat_b, source=feat_a, edge=edge_feat, source_mask=mask_a
                )
                
                # 累加更新 (Residual accumulation across edges)
                # 这一步模拟了 GCN 的 sum aggregation，但发生在特征空间
                next_step_embeddings[idx_a] = next_step_embeddings[idx_a] + update_a
                next_step_embeddings[idx_b] = next_step_embeddings[idx_b] + update_b
            
            # Normalize after aggregation to prevent value explosion
            current_group_embeddings = [F.layer_norm(e, e.shape[-1:]) for e in next_step_embeddings]

        # 5. Global Aggregation
        # 此时 current_group_embeddings 包含了经过多轮关系感知交互的特征序列
        
        # 先做一次 Group 内的 Pooling，减少计算量，得到每个 Group 的 [CLS] 效果
        group_pooled_list = []
        group_validity_list = []
        
        for i, g_emb in enumerate(current_group_embeddings):
             # (B, L, D) -> (B, D)
             pooled, valid = masked_mean_pool(g_emb, group_masks[i])
             group_pooled_list.append(pooled)
             group_validity_list.append(valid)
             
        # Stack -> (B, Num_Groups, D)
        global_seq = torch.stack(group_pooled_list, dim=1)
        global_mask_tensor = torch.stack(group_validity_list, dim=1) # (B, Num_Groups)
        
        # Global Self Attention across Modalities
        # 让所有模态最后再进行一次全局互通
        global_out = self.global_transformer(global_seq, mask=global_mask_tensor)
        
        # Final Embedding (Average over modalities)
        fused_embedding, _ = masked_mean_pool(global_out, global_mask_tensor)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # --- Loss Calculation ---
        # 使用更新后的特征计算对比损失
        loss_dict = self._compute_kl_loss(
            group_pooled_list, # 使用 Pool 过的特征计算相似度
            group_validity_list, 
            edge_keys, 
            groups_relationships
        )
        
        # Save points
        self.save_points(group_pooled_list, group_validity_list, groups_relationships)

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": loss_dict
        }

    def _compute_kl_loss(self, embeddings_list, masks_list, edge_keys, relationships):
        # 保持原有的稳健逻辑
        all_edge_scores = []
        all_sims = []
        all_valid_masks = []
        
        logit_scale = torch.clamp(self.logit_scale, max=4.6052)
        temperature = torch.exp(-logit_scale)

        has_valid_data = False
        
        for (idx_a, idx_b) in edge_keys:
            score = relationships.get((idx_a, idx_b), relationships.get((idx_b, idx_a), None))
            if score is None: continue
            if score.dim() > 1: score = score.view(-1)
            if score.dim() == 0: score = score.expand(embeddings_list[0].shape[0])
            if score.sum() > 0: has_valid_data = True

            ea = F.normalize(embeddings_list[idx_a], p=2, dim=1)
            eb = F.normalize(embeddings_list[idx_b], p=2, dim=1)
            sim = torch.sum(ea * eb, dim=1)
            
            valid = masks_list[idx_a] * masks_list[idx_b]
            
            all_edge_scores.append(score)
            all_sims.append(sim)
            all_valid_masks.append(valid)
            
        if not has_valid_data or len(all_edge_scores) == 0:
            return {"total_loss": torch.tensor(0.0, device=embeddings_list[0].device)}

        scores_stack = torch.stack(all_edge_scores, dim=1).float()
        sims_stack = torch.stack(all_sims, dim=1)
        masks_stack = torch.stack(all_valid_masks, dim=1)

        scores_masked = scores_stack.clone()
        scores_masked[masks_stack == 0] = -1e9
        target_probs = F.softmax(scores_masked, dim=1)

        sims_masked = sims_stack / temperature
        sims_masked[masks_stack == 0] = -1e9
        pred_log_probs = F.log_softmax(sims_masked, dim=1)

        kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none')
        kl_loss_sum = kl_loss.sum(dim=1)
        valid_patients = (masks_stack.sum(dim=1) > 0).float()
        final_loss = (kl_loss_sum * valid_patients).sum() / (valid_patients.sum() + 1e-9)
        
        return {"total_loss": 2 * final_loss}

    def save_points(self, final_group_embeddings, final_group_masks, groups_relationships):
        # 逻辑保持不变，用于可视化或调试
        if self.args.points_save_path is None: return 
        batch_size = final_group_embeddings[0].shape[0]
        device = final_group_embeddings[0].device
        sum_edge_scores = torch.zeros((batch_size, 1), device=device)
        sum_cos_sims = torch.zeros((batch_size, 1), device=device)
        raw_data_cache = {} 
        valid_pairs = []

        for (idx_a, idx_b), _ in groups_relationships.items():
            raw_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            if raw_score is not None:
                if raw_score.dim() == 1: raw_score = raw_score.view(-1, 1)
                
                embed_a = final_group_embeddings[idx_a]
                embed_b = final_group_embeddings[idx_b]
                
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