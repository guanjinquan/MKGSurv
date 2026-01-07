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
from collections import defaultdict

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
            nn.LayerNorm(dim * mult * 2),
            GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# --- SafeCrossAttnEncoder ---
class SafeCrossAttnEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False) -> torch.Tensor:
        
        query = self.norm_q(query)
        
        if key_padding_mask is not None:
            all_masked_rows = key_padding_mask.all(dim=1)
            if all_masked_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked_rows, 0] = False
        else:
            all_masked_rows = None  

        # Pass need_weights to mha
        attn_out, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights)
            
        if all_masked_rows is not None and all_masked_rows.any():
            attn_out[all_masked_rows] = 0.0

        x = query + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm_ffn(x))
        x = x + self.dropout(ffn_out)

        if need_weights:
            return x, attn_weights
        return x


class MedKGATFusion_group_msa(nn.Module):
    def __init__(self, args, embed_dim: int, 
             max_modalities: int = 10, 
             max_groups: int = 10, 
             ff_dropout_rate: float = 0.1, 
             attn_dropout_rate: float = 0.1, 
             num_intra_layers: int = 1, num_inter_layers: int = 1):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim

        # Placeholder for intra-group projection or interaction
        # In a real scenario, this might be a TransformerEncoder or GAT
        self.know_proj = nn.Sequential(
            nn.Linear(1, embed_dim), # Assuming edge weights are scalar
            nn.ReLU()
        )

        # 4. Global Aggregation
        self.global_transformer = SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate)

        # 5. Post Fusion Norm
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def _intra_group_step(self, embeddings, masks, embeddings_groups):
        # Simplified placeholder for intra-group logic
        # Just passing embeddings through for this example
        return embeddings
    
    # --- New Analysis Method 1: Contribution View ---
    def view_groups_contribution(self, attn_weights: torch.Tensor, values: torch.Tensor, group_masks: List[torch.Tensor]):
        """
        Implementation of Contribution Analysis based on Energy (Norm).
        Saves to JSONL if path is configured.
        
        Args:
            attn_weights: (B, L, L) or (B, H, L, L)
            values: (B, L, D) - Transformer Input (Global Concat)
            group_masks: List[(B, L_g)]
        """
        # Check configuration
        if not hasattr(self.args, 'view_groups_attention_path') or self.args.view_groups_attention_path is None:
            return
        
        save_path = self.args.view_groups_attention_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if attn_weights is None or values is None:
            return

        # 1. Dimensions Check
        # If Multi-head (B, H, L, L), average heads -> (B, L, L)
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)

        # Ensure (B, L, L)
        if attn_weights.shape[0] != group_masks[0].shape[0]:
            attn_weights = attn_weights.permute(1, 0, 2)
            
        # Ensure values (B, L, D)
        if values.shape[0] != group_masks[0].shape[0]:
            values = values.transpose(0, 1)

        # 2. Softmax Check
        check_sum = attn_weights[0, 0, :].sum().item()
        if check_sum > 1.1 or check_sum < 0.9:
            attn_weights = torch.softmax(attn_weights, dim=-1)

        # 3. Prepare Mask and Offsets
        global_mask = torch.cat(group_masks, dim=1).float() # (B, L_total)
        num_valid_queries = global_mask.sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1)

        group_lengths = [gm.shape[1] for gm in group_masks]
        offsets = [0]
        for l in group_lengths:
            offsets.append(offsets[-1] + l)

        # 4. Core Calculation: Group Energy
        group_energy_list = []

        for i in range(len(group_masks)):
            start, end = offsets[i], offsets[i+1]
            
            # A. Attention slice (B, L_total, L_group)
            attn_slice = attn_weights[:, :, start:end]
            
            # B. Value slice (B, L_group, D)
            value_slice = values[:, start:end, :]
            
            # C. Weighted Sum: (B, L_total, D)
            # How much update vector does this group inject into the stream
            weighted_update = torch.bmm(attn_slice, value_slice)
            
            # D. Energy (L2 Norm)
            # (B, L_total)
            update_norm = torch.norm(weighted_update, p=2, dim=-1)
            
            # E. Mask Padding
            update_norm = update_norm * global_mask
            
            # F. Average over valid queries
            avg_energy = update_norm.sum(dim=1) / num_valid_queries.squeeze(-1) # (B,)
            
            group_energy_list.append(avg_energy)

        # 5. Stack and Normalize
        # (B, Num_Groups)
        group_energies = torch.stack(group_energy_list, dim=1)
        
        total_energy = group_energies.sum(dim=1, keepdim=True)
        contribution_ratios = group_energies / torch.clamp(total_energy, min=1e-9)

        # 6. Save
        batch_ratios = contribution_ratios.detach().cpu().tolist()
        
        try:
            with open(save_path, 'a', encoding='utf-8') as f:
                for sample_ratios in batch_ratios:
                    f.write(json.dumps(sample_ratios) + "\n")
        except Exception as e:
            print(f"Warning: Failed to save contribution scores: {e}")

    # --- New Analysis Method 2: Save Scatter Points ---
    def save_points(self, final_group_embeddings, final_group_masks, groups_relationships):
        """
        Saves (Normalized Cosine Sim, Normalized Ground Truth) pairs for scatter plots.
        """
        if not hasattr(self.args, 'points_save_path') or self.args.points_save_path is None:
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

        # Iterate edges to collect raw values and sums for normalization
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

        # Create Group-Level Embeddings
        group_embeddings = []
        group_masks = []
    
        for group_indices in embeddings_groups:
            if not group_indices:
                raise ValueError("Empty group found in embeddings_groups")

            curr_feats = [embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]

            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)

            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # Final embeddings after all GAT layers (Simplified here)
        final_group_embeddings = group_embeddings
        final_group_masks = group_masks

        # 7. Global Aggregation
        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        # --- MODIFIED: Check if we need weights for analysis ---
        need_vis_weights = (not self.training) and \
                           hasattr(self.args, 'view_groups_attention_path') and \
                           self.args.view_groups_attention_path is not None

        if need_vis_weights:
            global_transformed, attn_weights = self.global_transformer(
                query=global_concat, key=global_concat, value=global_concat, 
                key_padding_mask=global_padding_mask, need_weights=True)
            self.view_groups_contribution(attn_weights, global_concat, group_masks)
        else:
            global_transformed = self.global_transformer(
                query=global_concat, key=global_concat, value=global_concat, 
                key_padding_mask=global_padding_mask, need_weights=False)
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        fused_embedding = masked_mean_pool(global_transformed, global_mask)
        if isinstance(fused_embedding, tuple):
             fused_embedding = fused_embedding[0]
             
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # --- MODIFIED: Call Save Points Analysis at end of inference ---
        if not self.training:
            self.save_points(final_group_embeddings, final_group_masks, groups_relationships)

        return {
            "fused_embedding": fused_embedding,
        }