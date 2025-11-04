import torch
from torch import nn
from typing import List, Optional, Dict

#==============================================================================
# HELPERS
#==============================================================================

def masked_mean_pool(embedding: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Performs masked mean pooling on a sequence.
    This is used to aggregate variable-length sequences into a single feature vector.

    Args:
        embedding (torch.Tensor): Input sequence of shape (B, N, D).
        mask (Optional[torch.Tensor]): Boolean or binary mask of shape (B, N), with 1s for valid tokens.

    Returns:
        torch.Tensor: Pooled embedding of shape (B, D).
    """
    if mask is None:
        return embedding.mean(dim=1)
    
    # Ensure mask is float for multiplication and summation
    mask = mask.float().unsqueeze(-1)
    
    # Masked pooling
    summed = (embedding * mask).sum(dim=1)
    # Count non-zero elements in the mask for each batch item to get the sequence length
    count = mask.sum(dim=1)
    # Avoid division by zero for empty sequences
    count = torch.max(count, torch.ones_like(count))
    
    return summed / count

#==============================================================================
# FUSION BLOCKS (REFACTORED)
#==============================================================================

class ConcatBlock(nn.Module):
    """
    Fuses modalities by concatenating their pooled feature vectors.
    """
    def __init__(self, num_modalities: int, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # The neck expects a flattened vector of all modalities
        self.neck = nn.Sequential(
            nn.Linear(self.in_dim * self.num_modalities, self.out_dim),
            nn.LayerNorm(self.out_dim), 
            nn.ReLU(),
        )
        print(f"ConcatBlock Params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor of stacked, pooled embeddings of shape (B, num_modalities, D).
        """
        assert x.dim() == 3, f"Expected input shape (B, num_modalities, D), but got {x.shape}"
        batch_size = x.shape[0]
        
        # Flatten the features from all modalities
        x = x.view(batch_size, -1)
        return self.neck(x)

class LowRankFusionBlock(nn.Module):
    """
    Fuses modalities using Low-Rank Tensor Fusion.
    Ref: https://arxiv.org/abs/1707.07250
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.rank = 16
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Learnable factor parameter for tensor fusion
        self.factor = nn.Parameter(
            torch.randn(self.rank, self.out_dim + 1, self.out_dim + 1),
            requires_grad=True
        ) 
        
        self.transition = nn.Linear(self.in_dim, self.out_dim)
        
        self.neck = nn.Sequential(
            nn.Linear(in_features=self.rank * (self.out_dim + 1), out_features=self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),  
        )
        print(f"LowRankFusionBlock Params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor of stacked, pooled embeddings of shape (B, num_modalities, D).
        """
        assert x.dim() == 3, f"Expected input shape (B, num_modalities, D), but got {x.shape}"
        device = x.device
        b, n = x.shape[0], x.shape[1]  # batch_size, num_modalities
        
        # [B, N, D_in] -> [B, N, D_out]
        x = self.transition(x)
        
        # Permute for fusion logic: [B, N, D_out] -> [N, B, D_out]
        x = x.permute(1, 0, 2)
        
        # Add a bias dimension: [N, B, D_out] -> [N, B, D_out+1]
        x = torch.cat([torch.ones(n, b, 1, device=device), x], dim=-1)
        
        # Tensor fusion calculation
        # [1, 1, R, D+1, D+1] @ [N, B, 1, D+1, 1] -> [N, B, R, D+1]
        fused = (self.factor.unsqueeze(0).unsqueeze(0) @ x.unsqueeze(2).unsqueeze(4)).squeeze(4)
        
        # Product across modalities: [N, B, R, D+1] -> [B, R, D+1]
        fused = fused.prod(dim=0)
        
        # Flatten and pass through neck
        fused = fused.view(b, -1)
        return self.neck(fused)

class GatedFusionBlock(nn.Module):
    """
    Fuses modalities using a gating mechanism.
    Ref: https://arxiv.org/pdf/1702.01992
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transition = nn.Linear(self.in_dim, self.out_dim)
        print(f"GatedFusionBlock Params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor of stacked, pooled embeddings of shape (B, num_modalities, D).
        """
        assert x.dim() == 3, f"Expected input shape (B, num_modalities, D), but got {x.shape}"
        
        # [B, N, D_in] -> [B, N, D_out]
        x = self.transition(x)
        
        # Permute for gating: [B, N, D_out] -> [N, B, D_out]
        x = x.permute(1, 0, 2)
        
        # Gating mechanism
        gate = torch.sigmoid(x)
        content = torch.tanh(x)
        
        # Gated sum over modalities: [N, B, D] -> [B, D]
        fused = (gate * content).sum(dim=0)
        return fused

class MSAFusionBlock(nn.Module):
    """
    Fuses modalities by applying self-attention over their concatenated sequences.
    This version correctly handles padding masks using nn.TransformerEncoder.
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.layers_num = 1
        self.attn_heads = 8
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.transition = nn.Linear(self.in_dim, self.out_dim)
        
        assert self.out_dim % self.attn_heads == 0, f"out_dim must be a multiple of attn_heads"
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.out_dim,
            nhead=self.attn_heads,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.layers_num)
        
        print(f"MSAFusionBlock Params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): A tensor of concatenated sequences of shape (B, N_total, D).
            mask (torch.Tensor): The key_padding_mask of shape (B, N_total), where True indicates a padded token should be ignored.
        """
        assert x.dim() == 3, f"Expected input shape (B, N, D), but got {x.shape}"

        # Project to the attention dimension
        x = self.transition(x)
        
        # Apply self-attention layers with the mask
        fused_sequence = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Masked mean pool the fused sequence to get a single feature vector
        if mask is not None:
            # Invert mask for pooling (1s for valid tokens)
            pool_mask = ~mask
        else:
            pool_mask = None
            
        return masked_mean_pool(fused_sequence, pool_mask)

#==============================================================================
# MAIN FUSION WRAPPER
#==============================================================================

class SimpleFusion(nn.Module):
    """
    A generic wrapper for multimodal fusion. It handles two scenarios:
    1.  'msa': Concatenates sequences and applies self-attention.
    2.  'concat', 'low_rank', 'gated': Pools each sequence into a single vector
        and then applies the chosen fusion logic.
    """


    def __init__(self, embed_dim: int, fusion_type: str = 'msa', max_modalities: int = 3):
        super().__init__()
        
        if fusion_type not in ['msa', 'concat', 'lmf', 'gated']:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.max_modalities = max_modalities
        
        self.fusion_block = self._get_fusion_block(embed_dim, max_modalities)

    def _get_fusion_block(self, dim: int, num_modalities: int) -> nn.Module:
        if self.fusion_type == 'concat':
            return ConcatBlock(num_modalities, dim, dim)
        elif self.fusion_type == 'msa':
            return MSAFusionBlock(dim, dim)
        elif self.fusion_type == 'lmf':
            return LowRankFusionBlock(dim, dim)
        elif self.fusion_type == 'gated':
            return GatedFusionBlock(dim, dim)
        else:
            # This case is already handled in __init__, but included for safety
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def forward(
        self, 
        embeddings: List[Optional[torch.Tensor]], 
        masks: Optional[List[Optional[torch.Tensor]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Fuses a list of embeddings from different modalities.

        Args:
            embeddings (List[Optional[torch.Tensor]]): A list of embedding tensors.
                Each tensor should have shape (B, N, D). The list can contain `None`
                for missing modalities.
            masks (Optional[List[Optional[torch.Tensor]]]): An optional list of masks.
                Each mask corresponds to an embedding and should have shape (B, N).
                A mask contains 1s for valid tokens and 0s for padding.
        """

        present_indices = [i for i, e in enumerate(embeddings) if e is not None]
        present_embeddings = [embeddings[i] for i in present_indices]
        device = present_embeddings[0].device
        batch_size = present_embeddings[0].shape[0]

        if len(present_embeddings) > self.max_modalities:
             raise ValueError(f"Number of embeddings ({len(present_indices)}) exceeds max_modalities ({self.max_modalities})")
        
        if not present_indices:
            raise ValueError("No embeddings provided to Fusion module")
            
        # --- MSA Fusion: Concatenate sequences ---
        if self.fusion_type == 'msa':
            # for emb in present_embeddings:
                # print("Shape = ", emb.shape)
            concatenated_embeddings = torch.cat(present_embeddings, dim=1)
            
            # Prepare the attention mask
            key_padding_mask = None
            present_masks_for_msa = []
            if masks is not None:
                # Handle cases where some modalities might not have a mask
                for i in present_indices:
                    if masks[i] is not None:
                        present_masks_for_msa.append(masks[i])
                    else:
                        # If no mask is provided for a present modality, assume no padding
                        emb = embeddings[i]
                        present_masks_for_msa.append(torch.ones(emb.shape[0], emb.shape[1], device=emb.device))
                
                if present_masks_for_msa:
                    concatenated_masks = torch.cat(present_masks_for_msa, dim=1)
                    # The mask for TransformerEncoder needs to be boolean, with True for padded tokens
                    key_padding_mask = (concatenated_masks == 0)

            patient_wise_mask = torch.any(~key_padding_mask, dim=1)
            assert patient_wise_mask.shape == (batch_size,), f"Expected patient-wise mask shape (B,), got {patient_wise_mask.shape}"
            assert patient_wise_mask.all().item(), "All patients must have at least one valid token"
            fused_embedding = self.fusion_block(concatenated_embeddings, mask=key_padding_mask)
        
        # --- Pooling-based Fusion: Pool then fuse ---
        else:
            if masks is None:
                masks = [None] * len(embeddings)
            
            present_masks = [masks[i] for i in present_indices]
            
            # 1. Pool each present modality from (B, N, D) to (B, D)
            pooled_embeddings = [
                masked_mean_pool(emb, m) for emb, m in zip(present_embeddings, present_masks)
            ]
            
            # 2. Create a full list, padding with zero tensors for missing modalities
            padded_pooled_list = []
            pooled_iter = iter(pooled_embeddings)
            for emb in embeddings:
                if emb is not None:
                    padded_pooled_list.append(next(pooled_iter))
                else:
                    padded_pooled_list.append(torch.zeros(batch_size, self.embed_dim, device=device))

            # 3. Stack to create the final input tensor for the fusion block
            stacked_for_fusion = torch.stack(padded_pooled_list, dim=1)
            fused_embedding = self.fusion_block(stacked_for_fusion)
            
        return {"fused_embedding": fused_embedding}

