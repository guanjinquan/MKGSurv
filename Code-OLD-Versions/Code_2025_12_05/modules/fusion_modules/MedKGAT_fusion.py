# 这个代码应该是v11了
"""
LUAD:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6188 ± 0.0524
 - List = [0.6325543916196615, 0.5765907305577376, 0.570527974783294, 0.714172604908947, 0.6003062787136294]
C-Index-IPCW_Validation Set: 0.5850 ± 0.0734
 - List = [0.5724608431713001, 0.5231046137229556, 0.5293511444646414, 0.7254133406451077, 0.5744687199861874]
Test Summary:
C-Index_Test Set: 0.6252 ± 0.0389
 - List = [0.6534591194968553, 0.6371866295264624, 0.6556603773584906, 0.6298611111111111, 0.5499612703330752]
C-Index-IPCW_Test Set: 0.5688 ± 0.0675
 - List = [0.514138756272405, 0.6401458460108551, 0.641716295162905, 0.5756256598664075, 0.4721479699647622]
Training run tcga_luad_run001 finished.

LUSC:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6238 ± 0.0293
 - List = [0.6492974238875878, 0.6386222473178995, 0.6074154852780806, 0.5740231150247661, 0.6496616345653305]
C-Index-IPCW_Validation Set: 0.5983 ± 0.0383
 - List = [0.6665163823891566, 0.5738427379020292, 0.5701495721726063, 0.5662836982653962, 0.6147324029080908]
Test Summary:
C-Index_Test Set: 0.6099 ± 0.0509
 - List = [0.6022346368715084, 0.665058949624866, 0.5231788079470199, 0.6016483516483516, 0.657334144856969]
C-Index-IPCW_Test Set: 0.6186 ± 0.0281
 - List = [0.6372405539180422, 0.6311785691245235, 0.5923372153226066, 0.5791103883105335, 0.6530010270478994]
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


# --- Building Blocks (Using PyTorch Native Components) ---

class StandardCrossAttention(nn.Module):
    """
    Wrapper around nn.MultiheadAttention to handle Pre-Norm and Residuals cleanly.
    """
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        # Simple FFN part
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, x, context, key_padding_mask=None):
        """
        x: Query (B, Lq, D)
        context: Key/Value (B, Lk, D)
        key_padding_mask: (B, Lk) - True where value is Padding
        """
        # 1. Attention (Pre-Norm)
        q = self.norm_q(x)
        k = self.norm_kv(context)
        v = k 
        
        # Safety: prevent NaN if a row is all padding
        if key_padding_mask is not None:
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False
        
        attn_out, _ = self.mha(q, k, v, key_padding_mask=key_padding_mask)
        
        # Restore zero for all-pad rows
        if key_padding_mask is not None and 'all_pad' in locals() and all_pad.any():
            attn_out[all_pad] = 0.0

        # Residual 1
        x = x + self.dropout(attn_out)
        
        # 2. FFN
        x = x + self.ffn(self.norm_ffn(x))
        return x

class AttentionPooling(nn.Module):
    """
    Gated Attention Pooling for the final global aggregation.
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (B, L, D)
        scores = self.attn_proj(x).squeeze(-1) # (B, L)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        return out

# --- Main Fusion Model ---

class MedKGATFusion(nn.Module):
    def __init__(self, args, embed_dim: int, 
            max_modalities: int = 10, 
            max_groups: int = 10, 
            ff_dropout_rate: float = 0.1, 
            attn_dropout_rate: float = 0.1, 
            num_intra_layers: int = 1, num_inter_layers: int = 1):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.drop_edge_ratio = 0.1 # Simple regularization
        
        # Learnable Temperature for KL Loss
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.65) 

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(ff_dropout_rate)
        )

        # 2. Intra-group: Native TransformerEncoder
        self.num_intra_layers = num_intra_layers
        self.intra_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4, 
            dropout=attn_dropout_rate, batch_first=True, norm_first=True
        )

        # 3. Inter-Group: GAT Components (Shared Weights)
        self.num_inter_layers = num_inter_layers
        # Shared Attention Modules to prevent overfitting
        self.edge_updater = StandardCrossAttention(embed_dim, heads=4, dropout=attn_dropout_rate)
        self.node_updater = StandardCrossAttention(embed_dim, heads=4, dropout=attn_dropout_rate)
        
        self.resid_dropout = nn.Dropout(attn_dropout_rate)

        # 4. Global Aggregation
        self.global_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*4,
            dropout=attn_dropout_rate, batch_first=True, norm_first=True
        )
        
        # 5. Final Pooling (Attention Gated)
        self.final_pool = AttentionPooling(embed_dim, dropout=attn_dropout_rate)
        
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def _intra_group_step(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], groups: List[List[int]]) -> List[torch.Tensor]: 
        """
        Encapsulates the Intra-group interactions using TransformerEncoder.
        """
        updated_embeddings = list(embeddings)
        
        for group_indices in groups:
            if not group_indices: continue
                
            # Gather
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            lengths = [f.shape[1] for f in group_feats]
            
            # Concat
            concat_feat = torch.cat(group_feats, dim=1) # (B, Sum_L, D)
            concat_mask = torch.cat(group_masks, dim=1) # (B, Sum_L)
            
            # Create padding mask for PyTorch (True = Padding)
            key_padding_mask = (concat_mask == 0)
            
            # Safety check for all-pad rows
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False
            
            # Process through layers
            curr = concat_feat
            for _ in range(self.num_intra_layers):
                curr = self.intra_layer(curr, src_key_padding_mask=key_padding_mask)
            
            if all_pad.any():
                curr[all_pad] = 0.0
            
            # Split back
            split_feats = torch.split(curr, lengths, dim=1)
            for i, idx in enumerate(group_indices):
                updated_embeddings[idx] = split_feats[i]
                
        return updated_embeddings

    def _inter_group_step(self, 
                          idx_a: int, idx_b: int, 
                          group_embeddings: List[torch.Tensor], 
                          group_masks: List[torch.Tensor], 
                          edge_feat: torch.Tensor, 
                          edge_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encapsulates One Step of GAT interaction (Update Edge -> Update A -> Update B).
        Returns updated tensors: (new_edge, new_node_a, new_node_b)
        """
        feat_a = group_embeddings[idx_a]
        mask_a = group_masks[idx_a]
        feat_b = group_embeddings[idx_b]
        mask_b = group_masks[idx_b]

        # 1. Update Edge: Edge queries [Node A, Node B]
        context_feat = torch.cat([feat_a, feat_b], dim=1)
        context_mask = torch.cat([mask_a, mask_b], dim=1)
        key_pad_mask = (context_mask == 0) # MHA needs True=Pad

        updated_edge = self.edge_updater(edge_feat, context_feat, key_padding_mask=key_pad_mask)
        
        # 2. Update Nodes: Node queries Updated Edge
        # Since Edge is the Context now, we use edge_mask for padding mask
        edge_pad_mask = (edge_mask == 0) if edge_mask is not None else None

        # Update B
        delta_b = self.node_updater(feat_b, updated_edge, key_padding_mask=edge_pad_mask)
        new_feat_b = feat_b + self.resid_dropout(delta_b) # Residual Connection

        # Update A
        delta_a = self.node_updater(feat_a, updated_edge, key_padding_mask=edge_pad_mask)
        new_feat_a = feat_a + self.resid_dropout(delta_a) # Residual Connection

        return updated_edge, new_feat_a, new_feat_b

    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        device = embeddings[0].device
        
        # 0. Clean symmetric keys
        for (i, j) in list(fusion_knowledge.keys()): 
            if i > j and (j, i) in fusion_knowledge:
                fusion_knowledge.pop((i, j), None)
                fusion_knowledge_mask.pop((i, j), None)

        edge_keys = list(fusion_knowledge.keys())
        if self.training:
            random.shuffle(edge_keys)

        # 1. Project Knowledge
        current_proj_knowledge = {k: self.know_proj(v) for k, v in fusion_knowledge.items()}

        # 2. Intra-Group Step
        info_level_embeddings = self._intra_group_step(embeddings, masks, embeddings_groups)

        # 3. Create Group-Level Embeddings
        group_embeddings = []
        group_masks = []
        for group_indices in embeddings_groups:
            if not group_indices: raise ValueError("Empty group")
            g_feat = torch.cat([info_level_embeddings[i] for i in group_indices], dim=1)
            g_mask = torch.cat([masks[i] for i in group_indices], dim=1)
            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # 4. Inter-Group Step (GAT)
        # Iterate layers
        for layer_idx in range(self.num_inter_layers):
            for (idx_a, idx_b) in edge_keys:
                
                # Edge Dropout
                if self.training and self.drop_edge_ratio > 0 and random.random() < self.drop_edge_ratio:
                    continue

                edge_feat = current_proj_knowledge[(idx_a, idx_b)]
                edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                
                # Safety for edge mask
                if edge_mask is None: 
                    edge_mask = torch.ones(edge_feat.shape[:2], device=device)

                # Call encapsulated step
                new_edge, new_a, new_b = self._inter_group_step(
                    idx_a, idx_b, group_embeddings, group_masks, edge_feat, edge_mask
                )
                
                # Apply updates
                current_proj_knowledge[(idx_a, idx_b)] = new_edge
                group_embeddings[idx_a] = new_a
                group_embeddings[idx_b] = new_b

        # 5. Global Aggregation
        global_concat = torch.cat(group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        global_pad_mask = (global_mask == 0)
        
        all_pad = global_pad_mask.all(dim=1)
        if all_pad.any():
            global_pad_mask = global_pad_mask.clone()
            global_pad_mask[all_pad, 0] = False

        global_out = self.global_transformer(global_concat, src_key_padding_mask=global_pad_mask)
        
        if all_pad.any():
            global_out[all_pad] = 0.0

        # 6. Final Pooling (Attention Gated)
        fused_embedding = self.final_pool(global_out, global_mask)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # 7. Compute KL Loss and Save Points
        # NOTE: Using mask inside save_points and KL loss to be precise
        self.save_points(group_embeddings, group_masks, groups_relationships)

        loss_dict = self._compute_kl_loss(
            group_embeddings, group_masks, 
            edge_keys, groups_relationships, device
        )

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": loss_dict
        }

    def _compute_kl_loss(self, group_embeddings, group_masks, edge_keys, groups_relationships, device):
        """
        Helper to compute KL loss with proper masking.
        """
        kl_loss = torch.tensor(0.0, device=device)
        
        if not edge_keys:
            return {"total_loss": kl_loss}

        # Pre-pool for similarity: Using Masked Mean Pool for robust representation of the group
        pooled_groups = []
        group_validity = []
        for g, m in zip(group_embeddings, group_masks):
            p, v_mask = masked_mean_pool(g, m)
            pooled_groups.append(F.normalize(p, p=2, dim=1))
            group_validity.append(v_mask)

        scores_list = []
        sims_list = []
        pair_masks_list = []

        for (idx_a, idx_b) in edge_keys:
            # GT Score
            score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            if score is None: continue
            
            if score.dim() == 0: score = score.expand(group_embeddings[0].shape[0])
            if score.dim() > 1: score = score.view(-1)
            
            # Pred Sim
            sim = torch.sum(pooled_groups[idx_a] * pooled_groups[idx_b], dim=1)
            
            # Mask (Both groups must be valid)
            pair_mask = group_validity[idx_a] * group_validity[idx_b]
            
            scores_list.append(score)
            sims_list.append(sim)
            pair_masks_list.append(pair_mask)

        if not scores_list:
            return {"total_loss": kl_loss}

        # Stack
        scores_tensor = torch.stack(scores_list, dim=1) 
        sims_tensor = torch.stack(sims_list, dim=1)    
        masks_tensor = torch.stack(pair_masks_list, dim=1)

        # Masking for Softmax
        scores_masked = scores_tensor.clone()
        scores_masked[masks_tensor == 0] = -1e9
        target_probs = F.softmax(scores_masked.float(), dim=1)

        # Masking for Log Softmax
        logit_scale = self.logit_scale.exp()
        sims_scaled = sims_tensor * logit_scale
        sims_scaled[masks_tensor == 0] = -1e9
        pred_log_probs = F.log_softmax(sims_scaled, dim=1)

        # KL Div
        kl_raw = F.kl_div(pred_log_probs, target_probs, reduction='none')
        
        # Valid patients check
        kl_raw = kl_raw * masks_tensor 
        loss_per_patient = kl_raw.sum(dim=1)
        
        valid_patients = (masks_tensor.sum(dim=1) > 1).float()
        
        if valid_patients.sum() > 0:
            kl_loss = (loss_per_patient * valid_patients).sum() / valid_patients.sum()

        return {"total_loss": 2.0 * kl_loss}

    def save_points(self, final_group_embeddings, final_group_masks, groups_relationships):
        if self.args.points_save_path is None:
            return 

        group_mean_embeddings = []
        for i in range(len(final_group_embeddings)):
            res = masked_mean_pool(final_group_embeddings[i], final_group_masks[i])
            if isinstance(res, tuple):
                mean_emb = res[0]
            else:
                mean_emb = res
            group_mean_embeddings.append(mean_emb)

        batch_size = final_group_embeddings[0].shape[0]
        device = final_group_embeddings[0].device
        
        sum_edge_scores = torch.zeros((batch_size, 1), device=device)
        sum_cos_sims = torch.zeros((batch_size, 1), device=device)
        
        raw_data_cache = {} 
        valid_pairs = []

        for (idx_a, idx_b), _ in groups_relationships.items():
            raw_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            
            if raw_score is not None:
                if raw_score.dim() == 1:
                    raw_score = raw_score.view(-1, 1)
                
                embed_a = group_mean_embeddings[idx_a]
                embed_b = group_mean_embeddings[idx_b]
                
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