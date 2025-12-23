"""
LUAD:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6510 ± 0.0643
 - List = [0.5987107171635777, 0.5930871956009427, 0.6122931442080378, 0.6935866983372921, 0.7572741194486983]
C-Index-IPCW_Validation Set: 0.6390 ± 0.0770
 - List = [0.5572435259147067, 0.5589893960606276, 0.6228271718651672, 0.7121765362119361, 0.74355211352138]
Test Summary:
C-Index_Test Set: 0.6034 ± 0.0524
 - List = [0.6773584905660377, 0.6344011142061281, 0.5363881401617251, 0.6166666666666667, 0.552285050348567]
C-Index-IPCW_Test Set: 0.5864 ± 0.0445
 - List = [0.5987090737584267, 0.6583201647843587, 0.5555888819935181, 0.5926500129048089, 0.5264994203178877]
Training run tcga_luad_run001 finished.

LUSC: 
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6625 ± 0.0326
 - List = [0.6545667447306791, 0.6837944664031621, 0.7022900763358778, 0.6059438635112824, 0.665799062988027]
C-Index-IPCW_Validation Set: 0.6318 ± 0.0335
 - List = [0.6454700775625689, 0.6846124420503881, 0.5930995119801192, 0.598461457941499, 0.637227345741866]
Test Summary:
C-Index_Test Set: 0.6193 ± 0.0375
 - List = [0.5575418994413408, 0.6141479099678456, 0.60825267447784, 0.6587912087912088, 0.6579427875836884]
C-Index-IPCW_Test Set: 0.6385 ± 0.0305
 - List = [0.5917518790640188, 0.6322272843471183, 0.6601033625904246, 0.681277712668965, 0.6272346754461051]
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

class KnowledgeAwareCoAttention(nn.Module):
    """
    [细粒度融合核心] 知识引导的协同注意力层
    
    1. 保留序列长度 (B, L, D)，不进行 Early Pooling。
    2. Edge-Guided Query: 将 Edge 信息注入 Query，引导 Target 去关注 Source 中
       符合特定关系 (Edge) 的 Token。
       Q = Target + Proj(Edge)
    3. 允许 Token-to-Token 的细粒度交互。
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 将 Edge 映射到 Query 空间，作为"搜索意图"的偏置
        self.edge_proj = nn.Linear(embed_dim, embed_dim)
        
        # FFN 用于特征整合
        self.ffn = FeedForward(embed_dim, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, target, source, edge_feat, source_mask=None):
        """
        target: (B, Lt, D) - 序列
        source: (B, Ls, D) - 序列
        edge_feat: (B, D)  - 向量
        """
        # 1. Edge-Guided Query Injection
        # "基于这个关系，我应该找什么？"
        # Edge 广播: (B, D) -> (B, 1, D)
        edge_bias = self.edge_proj(edge_feat).unsqueeze(1)
        
        # 这种加法注入比简单的 Concat 更能保持特征空间的语义一致性
        # 并强迫 Attention 考虑 Edge 的影响
        query = target + edge_bias
        
        # 2. Mask Handling
        key_padding_mask = None
        if source_mask is not None:
            key_padding_mask = (source_mask == 0) # True for padding
            # 防 NaN: 如果某行全 Mask，解开第一个
            if key_padding_mask.all(dim=1).any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[key_padding_mask.all(dim=1), 0] = False
        
        # 3. Cross Attention (Fine-grained Interaction)
        # attn_out: (B, Lt, D)
        attn_out, _ = self.mha(query, source, source, key_padding_mask=key_padding_mask)
        
        # 4. Residual & Norm
        # 注意: Residual 连接的是原始 target，而不是被修改过 query 的 target
        x = self.norm(target + self.dropout(attn_out))
        
        # 5. FFN
        x = self.norm_ffn(x + self.ffn(x))
        
        return x

class SelfAttnEncoder(nn.Module):
    """
    标准的 Self-Attention 用于组内特征提取 (保持不变，处理序列信息)
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        src_key_padding_mask = (mask == 0) if mask is not None else None
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
        self.drop_edge_ratio = 0.1 

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # 2. Intra-Group Refinement (Seq -> Seq)
        self.intra_group_layers = nn.ModuleList([
            SelfAttnEncoder(embed_dim, num_heads=4, dropout=attn_dropout_rate)
            for _ in range(num_intra_layers)
        ])

        # 3. Inter-Group Interaction (Seq -> Seq)
        # [核心改变] 恢复序列级交互，使用知识引导的 Co-Attention
        self.num_inter_layers = num_inter_layers
        self.inter_layers = nn.ModuleList([
            KnowledgeAwareCoAttention(embed_dim, num_heads=4, dropout=attn_dropout_rate)
            for _ in range(num_inter_layers)
        ])
        
        # 4. Global Fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Learnable Temp
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.3025)

    def _intra_group_process(self, embeddings, masks, groups):
        updated_embeddings = list(embeddings)
        for group_indices in groups:
            if not group_indices: continue
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            lengths = [f.shape[1] for f in group_feats]
            
            concat_feat = torch.cat(group_feats, dim=1)
            concat_mask = torch.cat(group_masks, dim=1)
            
            for layer in self.intra_group_layers:
                concat_feat = layer(concat_feat, mask=concat_mask)
            
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

        # 1. Project Edge Knowledge
        projected_edges = {k: self.know_proj(v) for k, v in fusion_knowledge.items()}

        # 2. Intra-Group Processing (处理序列)
        processed_embeddings = self._intra_group_process(embeddings, masks, embeddings_groups)

        # 3. Construct Group Sequences (保留序列长度!)
        # 不再做 Early Pooling，保留 (B, L, D) 供后续细粒度交互
        group_seqs = [] 
        group_masks = []
        
        for group_indices in embeddings_groups:
            curr_feats = [processed_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            
            # Concat in length dimension
            g_feat = torch.cat(curr_feats, dim=1) # (B, L_total, D)
            g_mask = torch.cat(curr_masks, dim=1)
            
            group_seqs.append(g_feat)
            group_masks.append(g_mask)

        # 4. Inter-Group Interaction (Fine-grained Co-Attention)
        current_group_seqs = group_seqs
        
        for layer_idx in range(self.num_inter_layers):
            co_attn_layer = self.inter_layers[layer_idx]
            
            next_step_seqs = [v.clone() for v in current_group_seqs]
            node_updates = {i: [] for i in range(len(group_seqs))}
            
            for (idx_a, idx_b) in edge_keys:
                if self.training and self.drop_edge_ratio > 0.0 and random.random() < self.drop_edge_ratio:
                    continue

                edge_feat = projected_edges[(idx_a, idx_b)]
                # Edge 通常较短或是单一语义，先 Pool 成向量作为 Guide 比较合适
                if edge_feat.dim() == 3:
                    e_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                    edge_feat, _ = masked_mean_pool(edge_feat, e_mask)
                
                seq_a = current_group_seqs[idx_a]
                mask_a = group_masks[idx_a]
                seq_b = current_group_seqs[idx_b]
                mask_b = group_masks[idx_b]
                
                # A 更新 B (Token-to-Token 交互)
                # target=B, source=A, edge=Guide
                update_b = co_attn_layer(target=seq_b, source=seq_a, edge_feat=edge_feat, source_mask=mask_a)
                node_updates[idx_b].append(update_b)
                
                # B 更新 A
                update_a = co_attn_layer(target=seq_a, source=seq_b, edge_feat=edge_feat, source_mask=mask_b)
                node_updates[idx_a].append(update_a)
            
            # Aggregate Updates (Mean at Sequence Level)
            for i in range(len(group_seqs)):
                updates = node_updates[i]
                if updates:
                    # (Num_Updates, B, L, D) -> Mean -> (B, L, D)
                    aggregated_update = torch.stack(updates, dim=0).mean(dim=0)
                    next_step_seqs[i] = aggregated_update
            
            current_group_seqs = next_step_seqs

        # 5. Late Pooling & Global Fusion
        # 交互全部完成后，再进行 Pooling
        group_vecs = []
        group_validity_list = []

        for i, g_seq in enumerate(current_group_seqs):
            pooled, valid = masked_mean_pool(g_seq, group_masks[i])
            group_vecs.append(pooled)
            group_validity_list.append(valid)

        # (B, Num_Groups, D)
        global_stack = torch.stack(group_vecs, dim=1)
        global_mask_tensor = torch.stack(group_validity_list, dim=1)
        
        fused_embedding, _ = masked_mean_pool(global_stack, global_mask_tensor)
        fused_embedding = self.fusion_mlp(fused_embedding)

        # --- Loss Calculation ---
        loss_dict = self._compute_kl_loss(
            group_vecs, 
            group_validity_list, 
            edge_keys, 
            groups_relationships
        )
        
        self.save_points(group_vecs, group_validity_list, groups_relationships)

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": loss_dict
        }

    def _compute_kl_loss(self, embeddings_list, masks_list, edge_keys, relationships):
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

            # embeddings_list 已经是 (B, D)，直接计算
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