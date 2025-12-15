"""
LUAD:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6241 ± 0.0806
 - List = [0.5898468976631749, 0.5341712490180676, 0.5594956658786446, 0.6935866983372921, 0.7434915773353752]
C-Index-IPCW_Validation Set: 0.6178 ± 0.0701
 - List = [0.5771971865711031, 0.5576896838728155, 0.5520788586681143, 0.6776452303337663, 0.7244508295564358]
Test Summary:
C-Index_Test Set: 0.5905 ± 0.0374
 - List = [0.6226415094339622, 0.637883008356546, 0.5828840970350404, 0.5777777777777777, 0.5313710302091402]
C-Index-IPCW_Test Set: 0.5774 ± 0.0587
 - List = [0.5503201467983571, 0.6817870186606281, 0.5871895163334118, 0.5632582606309826, 0.5045132886688753]
Training run tcga_luad_run001 finished.

LUSC:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6221 ± 0.0504
 - List = [0.6241217798594848, 0.6792772444946358, 0.6717557251908397, 0.5448541552008805, 0.5903175429463821]
C-Index-IPCW_Validation Set: 0.6165 ± 0.0624
 - List = [0.6364751876395208, 0.7225036539964808, 0.614238327009675, 0.5635395973693931, 0.5457149644768097]
Test Summary:
C-Index_Test Set: 0.5890 ± 0.0472
 - List = [0.524022346368715, 0.5637727759914255, 0.5751400916963831, 0.6230769230769231, 0.6591600730371272]
C-Index-IPCW_Test Set: 0.6093 ± 0.0459
 - List = [0.5550452472556658, 0.5711722475564887, 0.6485166774272876, 0.5961177227973663, 0.6757507312924795]
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

class EdgeBottleneckFusion(nn.Module):
    """
    [防过拟合终极方案] 边缘瓶颈融合层 (Edge-Centric Bottleneck Fusion) - Strict Pre-Norm
    
    结构:
    1. Read: Edge_Updated = Edge + Attn(Norm(Edge), Norm(Source), Norm(Source))
    2. Write: Target_Updated = Target + Attn(Norm(Target), Norm(Edge_Updated), Norm(Edge_Updated))
    3. FFN: Output = Target_Updated + FFN(Norm(Target_Updated))
    
    优势: 
    Pre-Norm 结构保证了在深层网络中的梯度流动，防止梯度消失或爆炸。
    对 Source 进行 Norm 保证了 Attention Key/Value 的分布稳定性。
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. Read Phase: Edge (Query) 查 Source (Key/Value)
        self.read_query_norm = nn.LayerNorm(embed_dim)
        self.read_source_norm = nn.LayerNorm(embed_dim) # 新增: 对 Source 进行 Norm
        self.read_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 2. Write Phase: Target (Query) 查 Edge (Key/Value)
        self.write_query_norm = nn.LayerNorm(embed_dim)
        self.write_edge_norm = nn.LayerNorm(embed_dim) # 新增: 对 Updated Edge 进行 Norm
        self.write_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Edge 映射
        self.edge_proj = nn.Linear(embed_dim, embed_dim)
        
        # 3. FFN Phase
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout=dropout)

    def forward(self, target, source, edge_feat, source_mask=None):
        """
        target: (B, Lt, D) - Residual Stream
        source: (B, Ls, D) - Residual Stream
        edge_feat: (B, D)
        """
        # --- Step 1: Prepare Edge Query ---
        edge_q = self.edge_proj(edge_feat).unsqueeze(1) # (B, 1, D)
        
        # --- Step 2: READ Phase (Filtering) ---
        # Q=Norm(Edge), K=Norm(Source), V=Norm(Source)
        
        # Mask handling
        key_padding_mask = None
        if source_mask is not None:
            key_padding_mask = (source_mask == 0)
            if key_padding_mask.all(dim=1).any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[key_padding_mask.all(dim=1), 0] = False
        
        # Strict Pre-Norm inputs
        edge_q_norm = self.read_query_norm(edge_q)
        source_norm = self.read_source_norm(source) # Normalize Key/Value
        
        edge_context, _ = self.read_attn(edge_q_norm, source_norm, source_norm, key_padding_mask=key_padding_mask)
        
        # Residual Update for Edge
        updated_edge = edge_q + self.dropout(edge_context)
        
        # --- Step 3: WRITE Phase (Broadcasting) ---
        # Q=Norm(Target), K=Norm(Updated_Edge), V=Norm(Updated_Edge)
        
        target_norm = self.write_query_norm(target)
        edge_kv_norm = self.write_edge_norm(updated_edge) # Normalize Key/Value
        
        target_context, _ = self.write_attn(target_norm, edge_kv_norm, edge_kv_norm)
        
        # Residual Update for Target
        x = target + self.dropout(target_context)
        
        # --- Step 4: FFN Phase ---
        # Output = x + FFN(Norm(x))
        x_norm = self.ffn_norm(x)
        x = x + self.ffn(x_norm)
        
        return x

class SelfAttnEncoder(nn.Module):
    """
    标准的 Self-Attention 用于组内特征提取
    [修改]: 移除最后的 self.norm，使其成为纯粹的 Pre-Norm 残差块。
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
            norm_first=True # Pre-Norm Setting: x + Attn(Norm(x))
        )
        # 移除了 self.norm，因为在 Pre-Norm 架构中，Layer 的输出应该是未 Norm 的残差流

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
            
        return out # 返回 Raw Residual Stream

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

        # 3. Inter-Group Interaction (Seq -> Edge -> Seq)
        # 使用 Strict Pre-Norm 的 EdgeBottleneckFusion
        self.num_inter_layers = num_inter_layers
        self.inter_layers = nn.ModuleList([
            EdgeBottleneckFusion(embed_dim, num_heads=4, dropout=attn_dropout_rate)
            for _ in range(num_inter_layers)
        ])
        
        # 4. Global Fusion
        # [关键]: Pre-Norm 架构需要在最后的 Pooling/Prediction 之前加一个 Final Norm
        self.final_norm = nn.LayerNorm(embed_dim) 
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
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

        # 2. Intra-Group Processing
        processed_embeddings = self._intra_group_process(embeddings, masks, embeddings_groups)

        # 3. Construct Group Sequences
        group_seqs = [] 
        group_masks = []
        
        for group_indices in embeddings_groups:
            curr_feats = [processed_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            
            g_feat = torch.cat(curr_feats, dim=1) # (B, L_total, D)
            g_mask = torch.cat(curr_masks, dim=1)
            
            group_seqs.append(g_feat)
            group_masks.append(g_mask)

        # 4. Inter-Group Interaction (Edge Bottleneck)
        current_group_seqs = group_seqs
        
        for layer_idx in range(self.num_inter_layers):
            bottleneck_layer = self.inter_layers[layer_idx]
            
            next_step_seqs = [v.clone() for v in current_group_seqs]
            node_updates = {i: [] for i in range(len(group_seqs))}
            
            for (idx_a, idx_b) in edge_keys:
                if self.training and self.drop_edge_ratio > 0.0 and random.random() < self.drop_edge_ratio:
                    continue

                edge_feat = projected_edges[(idx_a, idx_b)]
                if edge_feat.dim() == 3:
                    e_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                    edge_feat, _ = masked_mean_pool(edge_feat, e_mask)
                
                seq_a = current_group_seqs[idx_a]
                mask_a = group_masks[idx_a]
                seq_b = current_group_seqs[idx_b]
                mask_b = group_masks[idx_b]
                
                # A 更新 B (必须经过 Edge 瓶颈)
                update_b = bottleneck_layer(target=seq_b, source=seq_a, edge_feat=edge_feat, source_mask=mask_a)
                node_updates[idx_b].append(update_b)
                
                # B 更新 A
                update_a = bottleneck_layer(target=seq_a, source=seq_b, edge_feat=edge_feat, source_mask=mask_b)
                node_updates[idx_a].append(update_a)
            
            # Aggregate Updates
            for i in range(len(group_seqs)):
                updates = node_updates[i]
                if updates:
                    aggregated_update = torch.stack(updates, dim=0).mean(dim=0)
                    next_step_seqs[i] = aggregated_update
            
            current_group_seqs = next_step_seqs

        # 5. Late Pooling & Global Fusion
        group_vecs = []
        group_validity_list = []

        # Apply Final Norm BEFORE Pooling
        # 这是 Pre-Norm 架构的标准做法 (e.g., ViT 的 ln_f)
        # 确保 Pooling 层接收到的是经过缩放的、数值稳定的特征
        for i, g_seq in enumerate(current_group_seqs):
            # (B, L, D) -> Norm -> (B, L, D)
            g_seq_norm = self.final_norm(g_seq)
            
            # Pooling
            pooled, valid = masked_mean_pool(g_seq_norm, group_masks[i])
            group_vecs.append(pooled)
            group_validity_list.append(valid)

        # (B, Num_Groups, D)
        global_stack = torch.stack(group_vecs, dim=1)
        global_mask_tensor = torch.stack(group_validity_list, dim=1)
        
        # Final Embedding
        fused_embedding, _ = masked_mean_pool(global_stack, global_mask_tensor)
        
        # Output MLP (Pre-Norm style input from fusion_norm, but here we just MLP)
        # Note: fused_embedding is derived from Normalized Inputs, so it is relatively stable.
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