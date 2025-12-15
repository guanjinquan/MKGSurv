import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, reduce

# Import the new masked mean pool function
from modules.base_modules.aggregation_utils import masked_mean_pool
from nystrom_attention import NystromAttention

from typing import List, Optional, Dict, Tuple

def exists(val):
    """Checks if a value is not None."""
    return val is not None

class FeedForward(nn.Module):
    """
    A simple feed-forward network, as is common in Transformer blocks.
    Matches the implied structure from the original repository.
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # GELU is common in transformers, vs ReLU in the old to_logits
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SurvPath(nn.Module):
    """
    Multimodal Fusion Module inspired by SurvPath (using Nystrom Attention).

    This module implements the core fusion logic seen in the provided GitHub
    code, but adapted to the requested generic interface. It fuses two
    modalities (assumed to be pathway tokens and patch tokens) using
    Nystrom self-attention, followed by modality-specific mean pooling
    and concatenation.

    Args:
        embed_dim: The embedding dimension (d) of the input tokens.
                   All input tensors in the `embeddings` list must
                   have this dimension.
        dropout: Dropout rate for the feed-forward network.
        nystrom_heads: Number of attention heads for Nystrom Attention.
        nystrom_landmarks: Number of landmarks for Nystrom Attention.
    """
    def __init__(
        self, 
        args, 
        embed_dim: int = 512, 
        max_modalities: int = 2,
        dropout: float = 0.1, 
        nystrom_heads: int = 2, 
        nystrom_landmarks: int = 256
    ) -> None:
        super().__init__()
        
        self.args = args
        self.embed_dim = embed_dim
        self.num_modalities = max_modalities # Hard-coded based on paper (Pathways + Patches)
        
        # Identity layer (e.g., for Captum attributions)
        self.identity = nn.Identity()

        # Nystrom Attention, as used in the provided GitHub code
        # This approximates full self-attention to remain computationally feasible
        self.cross_attender = NystromAttention(
            dim = embed_dim,
            dim_head = embed_dim // nystrom_heads,
            heads = nystrom_heads,
            num_landmarks = nystrom_landmarks,
            pinv_iterations = 6,    # From original code
            residual = False,       # From original code
            dropout = dropout
        )

        # Post-attention Feed-Forward network
        self.ffn = FeedForward(dim=embed_dim, hidden_dim=embed_dim * 4, dropout=dropout)
        
        # Post-attention Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # New linear layer to project from 2*D down to D
        self.fusion_projection = nn.Linear(embed_dim * 2, embed_dim)

    def _masked_mean_pooling(
        self, 
        tokens: torch.Tensor, 
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs mean pooling on tokens, respecting a boolean mask.
        
        Args:
            tokens: Tensor of shape (B, N, D)
            mask: Boolean tensor of shape (B, N)
        
        Returns:
            Tensor of shape (B, D)
        """
        if mask is None:
            # No mask, just do simple mean pooling
            return torch.mean(tokens, dim=1)
        
        # Expand mask to (B, N, 1) for broadcasting
        mask_expanded = mask.unsqueeze(-1).float()
        
        # Sum tokens where mask is True
        summed_tokens = (tokens * mask_expanded).sum(dim=1)
        
        # Count tokens where mask is True
        token_counts = mask_expanded.sum(dim=1)
        
        # Calculate mean, adding epsilon to avoid division by zero
        # if a sample has no valid tokens
        return summed_tokens / (token_counts + 1e-8)

    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[Optional[torch.Tensor]]
    ) -> Dict:
        """
        Forward pass for SurvPath fusion.
        
        Args:
            embeddings: List of tensors [pathway_tokens, patch_tokens].
                        pathway_tokens: (B, N_P, D)
                        patch_tokens: (B, N_H, D)
                        Both must have D = self.embed_dim.
            masks: List of masks [pathway_mask, patch_mask] or [None, None].
                   pathway_mask: (B, N_P)
                   patch_mask: (B, N_H)
                   Masks are boolean, where True indicates a valid token.
                   
        Returns:
            Dictionary with fused embedding.
            {
                "fused_embedding": (B, D * 2),
                "loss_dict": {}
            }
        """
        
        if len(embeddings) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, but got {len(embeddings)}")
            
        pathway_tokens = embeddings[0]
        patch_tokens = embeddings[1]
        
        b, n_p, d_p = pathway_tokens.shape
        b, n_h, d_h = patch_tokens.shape
        
        if d_p != self.embed_dim or d_h != self.embed_dim:
            raise ValueError(f"All input embeddings must have dimension {self.embed_dim}")
            
        # --- 1. Concatenate all tokens ---
        # (B, N_P + N_H, D)
        tokens = torch.cat([pathway_tokens, patch_tokens], dim=1)
        tokens = self.identity(tokens) # For attribution
        
        # --- 2. Prepare combined mask ---
        combined_mask = None
        if masks is not None and all(m is not None for m in masks):
            # (B, N_P + N_H)
            combined_mask = torch.cat(masks, dim=1)
        elif masks is not None:
            # Handle partially None masks by creating default True masks
            filled_masks = []
            for i, m in enumerate(masks):
                if m is None:
                    filled_masks.append(
                        torch.ones(
                            embeddings[i].shape[:2], 
                            dtype=torch.bool, 
                            device=embeddings[i].device
                        )
                    )
                else:
                    filled_masks.append(m)
            combined_mask = torch.cat(filled_masks, dim=1)

        # --- 3. Multimodal Attention (Nystrom) ---
        # (B, N_P + N_H, D)
        # NystromAttention expects mask where True means *keep*
        mm_embed = self.cross_attender(x=tokens, mask=combined_mask)
        
        # --- 4. Post-Attention Processing ---
        # (B, N_P + N_H, D)
        mm_embed = self.ffn(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        # --- 5. Modality-Specific Aggregation ---
        
        # Split tokens back into pathways and patches
        # (B, N_P, D)
        paths_post_sa = mm_embed[:, :n_p, :]
        # (B, N_H, D)
        wsi_post_sa = mm_embed[:, n_p:, :]
        
        # Get individual masks
        pathway_mask = masks[0] if masks else None
        patch_mask = masks[1] if masks else None

        # Perform masked mean pooling using the imported function
        # This function returns (embedding, mask_bool), so we unpack
        paths_embed_mean, _ = masked_mean_pool(paths_post_sa, pathway_mask)
        wsi_embed_mean, _ = masked_mean_pool(wsi_post_sa, patch_mask)
        
        # --- 6. Final Fused Representation ---
        # Concatenate first (B, 2*D)
        fused_cat = torch.cat([paths_embed_mean, wsi_embed_mean], dim=1)
        
        # Project down to (B, D)
        fused_embedding = self.fusion_projection(fused_cat)
        
        # No specific fusion losses are mentioned in the paper or code
        fusion_losses_dict = {}

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": fusion_losses_dict
        }
    
    

if __name__ == '__main__':
    # --- Example Usage ---
    
    BATCH_SIZE = 2
    EMBED_DIM = 512
    
    NUM_PATHWAYS = 50
    NUM_PATCHES = 1000
    
    # 1. Create the fusion model
    fusion_model = SurvPath(
        embed_dim=EMBED_DIM,
        dropout=0.1,
        nystrom_heads=8,
        nystrom_landmarks=128
    )
    
    # 2. Create dummy input tensors (already projected to EMBED_DIM)
    
    # (B, N_P, D)
    pathway_tokens = torch.randn(BATCH_SIZE, NUM_PATHWAYS, EMBED_DIM)
    # (B, N_H, D)
    patch_tokens = torch.randn(BATCH_SIZE, NUM_PATCHES, EMBED_DIM)
    
    embeddings_list = [pathway_tokens, patch_tokens]
    
    # 3. Create dummy masks (optional)
    # Here, we'll mask out the last 10 pathways and last 100 patches
    
    # (B, N_P)
    pathway_mask = torch.ones(BATCH_SIZE, NUM_PATHWAYS, dtype=torch.bool)
    pathway_mask[:, -10:] = False
    
    # (B, N_H)
    patch_mask = torch.ones(BATCH_SIZE, NUM_PATCHES, dtype=torch.bool)
    patch_mask[:, -100:] = False
    
    masks_list = [pathway_mask, patch_mask]
    
    # --- 4. Forward pass with masks ---
    output_with_mask = fusion_model(embeddings_list, masks_list)
    fused_vec_masked = output_with_mask["fused_embedding"]
    
    print(f"--- With Masking ---")
    print(f"Input pathways shape:   {pathway_tokens.shape}")
    print(f"Input patches shape:    {patch_tokens.shape}")
    print(f"Input pathway mask:     {pathway_mask.shape} ({pathway_mask.sum()} True)")
    print(f"Input patch mask:       {patch_mask.shape} ({patch_mask.sum()} True)")
    print(f"Fused embedding shape:  {fused_vec_masked.shape}") # This will now show (B, D)
    
    # --- 5. Forward pass without masks ---
    output_no_mask = fusion_model(embeddings_list, [None, None])
    fused_vec_no_mask = output_no_mask["fused_embedding"]

    print(f"\n--- Without Masking ---")
    print(f"Fused embedding shape:  {fused_vec_no_mask.shape}") # This will also show (B, D)

    # Check that the outputs are different
    assert not torch.allclose(fused_vec_masked, fused_vec_no_mask)
    print("\nMasked and unmasked outputs are different (as expected).")