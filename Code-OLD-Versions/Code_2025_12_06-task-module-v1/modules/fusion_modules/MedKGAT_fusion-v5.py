"""
LUAD
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6235 ± 0.0450
 - List = [0.6140209508460919, 0.5435978004713276, 0.6288416075650118, 0.6579572446555819, 0.6730474732006125]
C-Index-IPCW_Validation Set: 0.5958 ± 0.0601
 - List = [0.5608904209164863, 0.5030687210438581, 0.6013749584898144, 0.6370885956040062, 0.676541868929666]
Test Summary:
C-Index_Test Set: 0.6321 ± 0.0611
 - List = [0.6987421383647798, 0.6984679665738162, 0.6132075471698113, 0.6131944444444445, 0.5367931835786213]
C-Index-IPCW_Test Set: 0.5977 ± 0.0600
 - List = [0.6080725839309298, 0.7020200248971343, 0.6011238264345496, 0.5428115766821504, 0.5346568835005038]
Training run tcga_luad_run001 finished.


LUSC:

--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6501 ± 0.0177
 - List = [0.6610070257611241, 0.6499153020892151, 0.6543075245365322, 0.6169510181618052, 0.6684018740239459]
C-Index-IPCW_Validation Set: 0.6044 ± 0.0334
 - List = [0.6687771626427087, 0.5975319937980823, 0.5987307828766714, 0.5801195932312729, 0.5769932476533483]
Test Summary:
C-Index_Test Set: 0.6317 ± 0.0481
 - List = [0.5513966480446927, 0.6629153269024651, 0.6265919510952623, 0.6225274725274725, 0.6950699939135727]
C-Index-IPCW_Test Set: 0.6452 ± 0.0290
 - List = [0.5901230336216143, 0.6606084325564991, 0.673116529358521, 0.6448555061882476, 0.6571541633599449]
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

class GraphBiasedCoAttention(nn.Module):
    """
    [SOTA 关键设计] 图偏置协同注意力 (Graph-Biased Co-Attention)
    
    核心创新点：
    1. Attention Bias: Edge 不仅作为特征输入，更直接投射为 Attention Score 的偏置项。
       Score_ij = (Q_i * K_j) / sqrt(d) + Proj(Edge_ij)
       这强制模型关注知识图谱中连接紧密的部分。
    
    2. Co-Attention: 同时计算 A->B 和 B->A 的影响，共享部分参数，减少过拟合。
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Edge Bias Generator: 将 Edge 特征映射为 Head 数量的偏置标量
        # 输入 (B, D) -> 输出 (B, Num_Heads) -> 广播到 Attention Matrix
        self.edge_bias_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_heads) 
        )
        
        # Output projections
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Gating Mechanism (LSTM-style update gate)
        # 用来决定新信息与旧信息的融合比例，比简单的残差相加更精细
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x_target, x_source, edge_feat, source_mask=None):
        """
        x_target: (B, Lt, D) - 待更新的节点
        x_source: (B, Ls, D) - 提供信息的邻居
        edge_feat: (B, D)    - 它们之间的关系
        source_mask: (B, Ls)
        """
        B, Lt, D = x_target.shape
        Ls = x_source.shape[1]
        
        # 1. Linear Projections
        q = self.q_proj(x_target).view(B, Lt, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, Lt, d)
        k = self.k_proj(x_source).view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, Ls, d)
        v = self.v_proj(x_source).view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, Ls, d)
        
        # 2. Compute Raw Attention Scores
        # (B, H, Lt, Ls)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # 3. Apply Graph Bias (关键步骤)
        # edge_feat: (B, D) -> bias: (B, H) -> (B, H, 1, 1) -> Broadcast to (B, H, Lt, Ls)
        # 我们假设 Edge 是对于这一对 Group 整体的关系，所以对所有 token 施加相同的强偏置
        edge_bias = self.edge_bias_proj(edge_feat).view(B, self.num_heads, 1, 1)
        attn_scores = attn_scores + edge_bias
        
        # 4. Masking
        if source_mask is not None:
            # source_mask: (B, Ls) -> (B, 1, 1, Ls)
            mask = source_mask.view(B, 1, 1, Ls)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        # 5. Softmax & Aggregation
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # (B, H, Lt, d) -> (B, Lt, H, d) -> (B, Lt, D)
        context = (attn_probs @ v).transpose(1, 2).contiguous().view(B, Lt, D)
        context = self.out_proj(context)
        
        # 6. Gated Update (比简单的 Residual 更强)
        # Gate = Sigmoid(Linear(Target || Context))
        gate = torch.sigmoid(self.gate_proj(torch.cat([x_target, context], dim=-1)))
        
        # Out = Gate * Context + (1 - Gate) * Target
        # 允许模型完全保留原始信息，或者完全接受新信息
        out = gate * context + (1 - gate) * x_target
        
        return self.norm(out)


class SelfAttnEncoder(nn.Module):
    """
    标准的 Self-Attention 用于组内特征提取
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
        # 适当增加 Dropout 防止过拟合，因为我们现在的 Attention 结构更强了
        self.drop_edge_ratio = 0.2 if num_inter_layers > 1 else 0.0

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # 2. Intra-Group Refinement (Self-Attention)
        self.intra_group_layers = nn.ModuleList([
            SelfAttnEncoder(embed_dim, num_heads=4, dropout=attn_dropout_rate)
            for _ in range(num_intra_layers)
        ])

        # 3. Inter-Group Interaction (Graph-Biased Co-Attention)
        # 核心：使用强 Graph Bias
        self.num_inter_layers = num_inter_layers
        self.inter_layers = nn.ModuleList([
            GraphBiasedCoAttention(embed_dim, num_heads=4, dropout=attn_dropout_rate)
            for _ in range(num_inter_layers)
        ])
        
        # FFN 用于层间过渡，增加非线性
        self.inter_ffns = nn.ModuleList([
            FeedForward(embed_dim, dropout=attn_dropout_rate)
            for _ in range(num_inter_layers)
        ])
        self.inter_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_inter_layers)
        ])
        
        # 4. Global Fusion
        self.global_transformer = SelfAttnEncoder(embed_dim, num_heads=4, dropout=attn_dropout_rate)
        self.post_fusion_norm = nn.LayerNorm(embed_dim)
        
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

        # 2. Intra-Group Processing
        processed_embeddings = self._intra_group_process(embeddings, masks, embeddings_groups)

        # 3. Construct Group Embeddings (Sequence Mode)
        group_embeddings = [] 
        group_masks = []
        for group_indices in embeddings_groups:
            curr_feats = [processed_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)
            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # 4. Inter-Group Interaction (Graph-Biased Co-Attention)
        current_group_embeddings = group_embeddings
        
        for layer_idx in range(self.num_inter_layers):
            co_attn_layer = self.inter_layers[layer_idx]
            ffn_layer = self.inter_ffns[layer_idx]
            norm_layer = self.inter_norms[layer_idx]
            
            # 存储本轮更新，避免顺序依赖
            next_step_embeddings = [e.clone() for e in current_group_embeddings]
            
            # 用字典记录每个节点的累积更新量和更新次数，用于取平均
            # 这种 Aggregation 方式比单纯累加更稳定
            node_updates = {i: [] for i in range(len(group_embeddings))}
            
            for (idx_a, idx_b) in edge_keys:
                if self.training and self.drop_edge_ratio > 0.0 and random.random() < self.drop_edge_ratio:
                    continue

                edge_feat = projected_edges[(idx_a, idx_b)]
                if edge_feat.dim() == 3:
                    e_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                    edge_feat, _ = masked_mean_pool(edge_feat, e_mask)
                
                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a]
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]
                
                # A 更新 B (使用 A 作为 Source)
                # update_b = Attention(Q=B, K=A, Edge)
                update_b = co_attn_layer(
                    x_target=feat_b, x_source=feat_a, edge_feat=edge_feat, source_mask=mask_a
                )
                node_updates[idx_b].append(update_b)
                
                # B 更新 A (使用 B 作为 Source)
                update_a = co_attn_layer(
                    x_target=feat_a, x_source=feat_b, edge_feat=edge_feat, source_mask=mask_b
                )
                node_updates[idx_a].append(update_a)
            
            # Aggregate Updates & Apply FFN
            for i in range(len(group_embeddings)):
                updates = node_updates[i]
                if updates:
                    # 如果有邻居更新，取平均 (Mean Aggregation)
                    # stack: (N, B, L, D) -> mean -> (B, L, D)
                    aggregated_update = torch.stack(updates, dim=0).mean(dim=0)
                    # 更新状态
                    # Current = Norm(Current + FFN(Update))
                    # 注意：co_attn_layer 内部已经包含了 Residual + Gate，所以这里 aggregated_update 已经是融合后的结果
                    # 我们只需要在这里做一次 FFN 增强
                    feat = norm_layer(aggregated_update + ffn_layer(aggregated_update))
                    next_step_embeddings[i] = feat
                else:
                    # 孤立节点，保持原样或仅过 FFN
                    feat = current_group_embeddings[i]
                    next_step_embeddings[i] = norm_layer(feat + ffn_layer(feat))
            
            current_group_embeddings = next_step_embeddings

        # 5. Global Aggregation
        group_pooled_list = []
        group_validity_list = []
        
        for i, g_emb in enumerate(current_group_embeddings):
             pooled, valid = masked_mean_pool(g_emb, group_masks[i])
             group_pooled_list.append(pooled)
             group_validity_list.append(valid)
             
        global_seq = torch.stack(group_pooled_list, dim=1)
        global_mask_tensor = torch.stack(group_validity_list, dim=1)
        
        global_out = self.global_transformer(global_seq, mask=global_mask_tensor)
        fused_embedding, _ = masked_mean_pool(global_out, global_mask_tensor)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # --- Loss Calculation ---
        loss_dict = self._compute_kl_loss(
            group_pooled_list, 
            group_validity_list, 
            edge_keys, 
            groups_relationships
        )
        
        self.save_points(group_pooled_list, group_validity_list, groups_relationships)

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