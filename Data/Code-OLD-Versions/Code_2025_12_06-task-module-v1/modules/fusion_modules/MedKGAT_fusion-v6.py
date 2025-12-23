"""
LUAD
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6198 ± 0.0367
 - List = [0.6019339242546333, 0.5561665357423409, 0.6375098502758078, 0.6547901821060966, 0.6485451761102603]
C-Index-IPCW_Validation Set: 0.5802 ± 0.0576
 - List = [0.5436195727334526, 0.5047528192854487, 0.5713997685188819, 0.6076121039744352, 0.6737084965388886]
Test Summary:
C-Index_Test Set: 0.6153 ± 0.0761
 - List = [0.6880503144654088, 0.685933147632312, 0.6024258760107817, 0.6208333333333333, 0.47947327652982186]
C-Index-IPCW_Test Set: 0.5953 ± 0.0772
 - List = [0.6207035641367509, 0.7217895727183274, 0.5895262124727316, 0.5569987611932221, 0.48741510375820024]
Training run tcga_luad_run001 finished.

LUSC
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6410 ± 0.0236
 - List = [0.6738875878220141, 0.6403162055335968, 0.6537622682660851, 0.6020913593835994, 0.6350858927641854]
C-Index-IPCW_Validation Set: 0.5956 ± 0.0404
 - List = [0.6726942552559986, 0.5901733052845296, 0.5846511748714789, 0.57547508535319, 0.5547973266494248]
Test Summary:
C-Index_Test Set: 0.6226 ± 0.0410
 - List = [0.5592178770949721, 0.6634512325830654, 0.6230259806418746, 0.5983516483516483, 0.6688983566646378]
C-Index-IPCW_Test Set: 0.6329 ± 0.0234
 - List = [0.5934332134476954, 0.642507398093775, 0.6656512970460909, 0.6289758627144602, 0.6336887077450327]
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

class EdgeAdaptiveLayerNorm(nn.Module):
    """
    [SOTA 核心 Trick] 基于 Edge 的自适应归一化 (AdaLN)
    
    原理：
    Edge 不仅仅作为输入，而是作为"条件 (Condition)" 来动态预测 LayerNorm 的仿射参数 (Scale & Shift)。
    这使得 Node 特征在进入 Attention 之前，已经根据关系类型进行了"语境重构"。
    这是 Diffusion Transformer (DiT) 等 SOTA 模型的核心机制。
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False) # 关闭自带参数
        # 从 Edge 生成 gamma (scale) 和 beta (shift)
        self.edge_modulator = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 2) 
        )

    def forward(self, x, edge_feat):
        """
        x: (B, L, D)
        edge_feat: (B, D)
        """
        # 1. Standard Norm
        x_norm = self.norm(x)
        
        # 2. Predict adaptive parameters
        # edge_feat (B, D) -> (B, 2*D) -> split
        style = self.edge_modulator(edge_feat).unsqueeze(1) # (B, 1, 2*D)
        gamma, beta = style.chunk(2, dim=-1) # (B, 1, D)
        
        # 3. Modulate
        # x_mod = x * (1 + gamma) + beta
        return x_norm * (1 + gamma) + beta

class EdgeAdaptiveInteractionLayer(nn.Module):
    """
    [Knowledge-Guided Adaptive Transformer Layer]
    结合了 AdaLN (特征级融合) 和 Bias-Attention (结构级融合)。
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 1. Edge-Conditioned Norms (替代普通 LayerNorm)
        self.ada_norm_target = EdgeAdaptiveLayerNorm(embed_dim)
        self.ada_norm_source = EdgeAdaptiveLayerNorm(embed_dim)
        
        # 2. Attention Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 3. Structural Bias (Attention Bias)
        self.edge_bias_proj = nn.Sequential(
            nn.Linear(embed_dim, num_heads), # 直接映射到 heads
        )
        
        # 4. Output & FFN
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 5. Gating
        self.gate_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x_target, x_source, edge_feat, source_mask=None):
        """
        x_target: (B, Lt, D)
        x_source: (B, Ls, D)
        edge_feat: (B, D)
        """
        B, Lt, D = x_target.shape
        Ls = x_source.shape[1]
        
        # 1. Adaptive Normalization (AdaLN)
        # 关键点：根据 Edge 关系，动态调整 Target 和 Source 的特征分布
        x_target_mod = self.ada_norm_target(x_target, edge_feat)
        x_source_mod = self.ada_norm_source(x_source, edge_feat)
        
        # 2. Linear Projections
        q = self.q_proj(x_target_mod).view(B, Lt, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_source_mod).view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_source_mod).view(B, Ls, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Attention Score with Graph Bias
        # (B, H, Lt, Ls)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        
        # Inject Structural Bias
        # edge: (B, D) -> (B, H) -> (B, H, 1, 1)
        bias = self.edge_bias_proj(edge_feat).view(B, self.num_heads, 1, 1)
        attn_scores = attn_scores + bias
        
        # Masking
        if source_mask is not None:
            mask = source_mask.view(B, 1, 1, Ls)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Context
        context = (attn_probs @ v).transpose(1, 2).contiguous().view(B, Lt, D)
        context = self.out_proj(context)
        
        # 4. Gated Residual Update
        # Gate = Sigmoid(Linear(Original_Target || New_Context))
        # 使用原始未 Norm 的 x_target 来保持梯度流动的直接性
        gate = torch.sigmoid(self.gate_proj(torch.cat([x_target, context], dim=-1)))
        out = gate * context + (1 - gate) * x_target
        
        return out


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
        # SOTA 策略: 稍微加大 Dropout 配合强 Attention
        self.drop_edge_ratio = 0.15

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

        # 3. Inter-Group Interaction (AdaLN-Based Transformer)
        self.num_inter_layers = num_inter_layers
        self.inter_layers = nn.ModuleList([
            EdgeAdaptiveInteractionLayer(embed_dim, num_heads=4, dropout=attn_dropout_rate)
            for _ in range(num_inter_layers)
        ])
        
        # FFN (Adaptive Interaction 后通常接 FFN)
        self.inter_ffns = nn.ModuleList([
            FeedForward(embed_dim, dropout=attn_dropout_rate)
            for _ in range(num_inter_layers)
        ])
        # 使用普通 Norm，因为 AdaNorm 已经在 Attention 内部做了
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

        # 3. Construct Group Embeddings
        group_embeddings = [] 
        group_masks = []
        for group_indices in embeddings_groups:
            curr_feats = [processed_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)
            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # 4. Inter-Group Interaction (AdaLN Transformer)
        current_group_embeddings = group_embeddings
        
        for layer_idx in range(self.num_inter_layers):
            inter_layer = self.inter_layers[layer_idx]
            ffn_layer = self.inter_ffns[layer_idx]
            norm_layer = self.inter_norms[layer_idx]
            
            next_step_embeddings = [e.clone() for e in current_group_embeddings]
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
                
                # A 更新 B (Contextualized by Edge via AdaLN)
                update_b = inter_layer(
                    x_target=feat_b, x_source=feat_a, edge_feat=edge_feat, source_mask=mask_a
                )
                node_updates[idx_b].append(update_b)
                
                # B 更新 A
                update_a = inter_layer(
                    x_target=feat_a, x_source=feat_b, edge_feat=edge_feat, source_mask=mask_b
                )
                node_updates[idx_a].append(update_a)
            
            # Aggregate & FeedForward
            for i in range(len(group_embeddings)):
                updates = node_updates[i]
                if updates:
                    # Mean Aggregation of contextualized updates
                    aggregated_update = torch.stack(updates, dim=0).mean(dim=0)
                    
                    # FFN Block: Norm(Residual + FFN(Residual))
                    # 注意: inter_layer 已经包含了一次 Skip-Connection (Gate机制)
                    # 所以这里的 Input 已经是混合了 Neighbor info 的状态
                    feat = norm_layer(aggregated_update + ffn_layer(aggregated_update))
                    next_step_embeddings[i] = feat
                else:
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