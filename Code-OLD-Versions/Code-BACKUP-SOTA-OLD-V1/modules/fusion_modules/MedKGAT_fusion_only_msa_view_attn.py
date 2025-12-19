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
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        query = self.norm_q(query)
        
        if key_padding_mask is not None:
            all_masked_rows = key_padding_mask.all(dim=1)
            if all_masked_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked_rows, 0] = False
        else:
            all_masked_rows = None  

        attn_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
            
        if all_masked_rows is not None and all_masked_rows.any():
            attn_out[all_masked_rows] = 0.0

        x = query + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm_ffn(x))
        x = x + self.dropout(ffn_out)

        return x


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
            # Ensure v is at least 2D (B, 1) for Linear layer
            if v.dim() == 1: v = v.unsqueeze(-1)
            current_proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction
        info_level_embeddings = self._intra_group_step(embeddings, masks, embeddings_groups)

        # 3. Create Group-Level Embeddings
        group_embeddings = []
        group_masks = []
    
        for group_indices in embeddings_groups:
            if not group_indices:
                # Handle empty groups if necessary, or raise error
                # For safety here, creating dummy if needed or raising error
                raise ValueError("Empty group found in embeddings_groups")

            curr_feats = [info_level_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]

            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)

            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # Final embeddings after all GAT layers (Simplified here)
        final_group_embeddings = group_embeddings
        final_group_masks = group_masks

        # --- Call the Analysis Function Here ---
        self.view_groups_attention(final_group_embeddings, final_group_masks)
        
        # 7. Global Aggregation
        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        global_transformed = self.global_transformer(
            query=global_concat, key=global_concat, value=global_concat, key_padding_mask=global_padding_mask)
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        fused_embedding = masked_mean_pool(global_transformed, global_mask)
        # Handle tuple return if masked_mean_pool returns (emb, weights)
        if isinstance(fused_embedding, tuple):
             fused_embedding = fused_embedding[0]
             
        fused_embedding = self.post_fusion_norm(fused_embedding)

        return {
            "fused_embedding": fused_embedding,
        }

    def view_groups_attention(self, group_embeddings: List[torch.Tensor], group_masks: List[torch.Tensor]):
        """
        Calculates and saves the pairwise cosine similarity between groups for distribution analysis.
        Saves as a JSONL file (one JSON object per line).
        """
        # 1. Check path validity
        if not hasattr(self.args, 'view_groups_attention_path') or self.args.view_groups_attention_path is None:
            return
        
        save_path = self.args.view_groups_attention_path
        
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except OSError:
            pass # Handle root path cases

        # 2. Pool Group Embeddings into Vectors
        # Input: List of (B, L_g, D) -> Process to List of (B, D)
        pooled_group_vecs = []
        for feat, mask in zip(group_embeddings, group_masks):
            # Use the robust masked_mean_pool
            res = masked_mean_pool(feat, mask)
            if isinstance(res, tuple):
                mean_emb = res[0]
            else:
                mean_emb = res
            pooled_group_vecs.append(mean_emb) # (B, D)

        if not pooled_group_vecs:
            return

        # 3. Stack and Normalize for efficient computation
        # Stack shape: (B, Num_Groups, D)
        group_stack = torch.stack(pooled_group_vecs, dim=1)
        
        # Normalize to unit length so Dot Product == Cosine Similarity
        # (B, Num_Groups, D)
        group_stack_norm = F.normalize(group_stack, p=2, dim=2)

        # 4. Compute Pairwise Similarity Matrix
        # (B, Num_Groups, D) @ (B, D, Num_Groups) -> (B, Num_Groups, Num_Groups)
        similarity_matrices = torch.bmm(group_stack_norm, group_stack_norm.transpose(1, 2))

        # 5. Serialization (CPU move)
        # Detach and move to CPU
        similarity_matrices = similarity_matrices.detach().cpu()
        batch_size = similarity_matrices.shape[0]
        num_groups = similarity_matrices.shape[1]

        # 6. Write to File (JSON Lines format)
        try:
            with open(save_path, 'a', encoding='utf-8') as f:
                for b in range(batch_size):
                    # Extract single matrix (Num_Groups, Num_Groups)
                    mat = similarity_matrices[b].tolist()
                    
                    # Create a record structure
                    record = {
                        "batch_sample_idx": b, # Relative index in this batch
                        "num_groups": num_groups,
                        "similarity_matrix": mat,
                        # Optional: flatten lower triangle for distribution histogram later
                        "flattened_sims": [
                            mat[i][j] 
                            for i in range(num_groups) 
                            for j in range(i + 1, num_groups) # Only upper triangle, exclude diagonal (always 1)
                        ]
                    }
                    
                    f.write(json.dumps(record) + "\n")
                    
        except Exception as e:
            print(f"Warning: Failed to save groups attention/similarity: {e}")

# --- Example Usage for Testing ---
if __name__ == "__main__":
    # Mock Args
    class Args:
        view_groups_attention_path = "./output/group_attention.jsonl"
        points_save_path = "./output/points.jsonl"
    
    args = Args()
    
    # Initialize Model
    model = MedKGATFusion(args, embed_dim=32)
    model.eval() # Set to eval mode
    
    # Mock Data
    B, D = 2, 32
    # Create 3 groups of embeddings
    emb1 = torch.randn(B, 5, D)
    mask1 = torch.ones(B, 5)
    emb2 = torch.randn(B, 4, D)
    mask2 = torch.ones(B, 4)
    emb3 = torch.randn(B, 6, D)
    mask3 = torch.ones(B, 6)
    
    embeddings = [emb1, emb2, emb3] # Flattened input list (simplified logic)
    masks = [mask1, mask2, mask3]
    
    # Groups definition: Group 0 uses index 0, Group 1 uses index 1, Group 2 uses index 2
    embeddings_groups = [[0], [1], [2]]
    
    # Relationships (Empty for test)
    rels = {}
    know = {}
    know_mask = {}
    
    print(f"Running Forward Pass... Saving to {args.view_groups_attention_path}")
    out = model(embeddings, masks, embeddings_groups, rels, know, know_mask)
    
    print("Done. Check the output file.")
    
    # Validate Output
    if os.path.exists(args.view_groups_attention_path):
        with open(args.view_groups_attention_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                print("Loaded JSON record keys:", data.keys())
                print("Similarity Matrix Shape:", len(data['similarity_matrix']), "x", len(data['similarity_matrix'][0]))
                break