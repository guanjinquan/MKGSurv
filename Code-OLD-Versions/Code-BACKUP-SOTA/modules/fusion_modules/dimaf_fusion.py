import torch
from torch import nn
from typing import List, Optional, Dict
from einops import rearrange

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
# DIMAF FUSION MODULE
#==============================================================================

class CrossAttentionLayer(nn.Module):
    """ Single attention layer in the attention module. """

    def __init__(
            self,
            dim=512,
            dim_head=64,
            heads=1
    ):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)

    
    def forward(self, x, y, return_attention=False):
        x_norm = self.norm_x(x)
        y_norm = self.norm_y(y)

        # derive query, keys, values 
        q = self.to_q(x_norm)
        k = self.to_k(y_norm)
        v = self.to_v(y_norm)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # regular transformer scaling
        q = q * self.scale

        einops_eq = '... i d, ... j d -> ... i j'
        pre_soft_attn_matrix = torch.einsum(einops_eq, q, k)

        attn_matrix = pre_soft_attn_matrix.softmax(dim=-1)

        out  = attn_matrix @ v

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)

        if return_attention:
            # Also return the attention weights
            return out, attn_matrix.squeeze().detach().cpu()
    
        return out


class DistanceCorrelationLoss(nn.Module):
    """ Distance correlation loss to enforce linear and nonlinear disentanglement. Implemented from Liu, Xiao, et al. "Measuring the biases and effectiveness of content-style disentanglement." arXiv preprint arXiv:2008.12378 (2020)."""
    def __init__(self, weight_D1=0.5, weight_D2=0.5, epsilon=1e-8):
        super().__init__()
        self.weight_D1 = weight_D1
        self.weight_D2 = weight_D2
        self.epsilon = epsilon
    
    def compute_dist_corr(self, x, y):
        # Compute Euclidean pairwise distance matrices
        a = torch.cdist(x, x, p=2)
        b = torch.cdist(y, y, p=2)

        # Double-centering
        A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()

        # Compute distance covariance
        dcov = (A * B).mean().sqrt()

        # Compute distance variances
        dvar_x = (A * A).mean().sqrt()
        dvar_y = (B * B).mean().sqrt()

        # Compute distance correlation
        dcor = dcov / torch.sqrt(dvar_x * dvar_y + self.epsilon) 
        return dcor

    def __call__(self, uni_repr_rna, uni_repr_wsi, uni_repr_rna_wsi, uni_repr_wsi_rna):
        """ Compute the distance correlation loss. """ 
        shared_repr = torch.concat([uni_repr_wsi_rna, uni_repr_rna_wsi], dim=1)
        single_repr = torch.concat([uni_repr_rna, uni_repr_wsi], dim=1)

        # Disentanglement between the modality-specific and modality-shared representations (D2 disentanglement)
        dcor_D2 = self.compute_dist_corr(shared_repr, single_repr)

        # Disentanglement between the two modality-specific representations (D1 disentanglement)
        dcor_D1 = self.compute_dist_corr(uni_repr_rna, uni_repr_wsi)

        dcor_total = self.weight_D2*dcor_D2 + self.weight_D1*dcor_D1

        return dcor_total, {'disentanglement_loss': dcor_total, 'disentanglement_D1_loss': dcor_D1, 'disentanglement_D2_loss': dcor_D2}


class DIMAFFusionModule(nn.Module):
    """
    Disentangled and Interpretable Multimodal Attention Fusion module.
    Based on the DIMAF model from https://github.com/mahmoodlab/DIMAF
    """

    def __init__(self, args, embed_dim: int = 512, max_modalities: int = 2) -> None:
        super().__init__()
        
        self.args = args
        self.embed_dim = embed_dim
        self.max_modalities = max_modalities
        
        # For simplicity, we assume 2 modalities (like in the original DIMAF: WSI and RNA)
        self.num_modalities = 2
        
        # Prototype embedding dimensions
        self.append_dim = 32
        self.single_out_dim = embed_dim
        self.path_proj_dim_new = self.single_out_dim + self.append_dim # 512+32
        
        # Create prototype embeddings for each modality
        # In the original DIMAF, these are learned per prototype type (WSI and RNA pathways)
        # Here we simplify to per-modality embeddings
        self.modality_embeddings = nn.Parameter(
            torch.randn(1, self.num_modalities, self.append_dim), 
            requires_grad=True
        )
        
        multi_out_dim = self.single_out_dim // 2

        # 4 separate attention blocks (as in original DIMAF)
        self.self_attention_mod1 = CrossAttentionLayer(
                dim=self.path_proj_dim_new,
                dim_head=multi_out_dim,
                heads=1)
        
        self.self_attention_mod2 = CrossAttentionLayer(
                dim=self.path_proj_dim_new,
                dim_head=multi_out_dim,
                heads=1)

        self.cross_attention_1to2 = CrossAttentionLayer(
                dim=self.path_proj_dim_new,
                dim_head=multi_out_dim,
                heads=1)

        self.cross_attention_2to1 = CrossAttentionLayer(
                dim=self.path_proj_dim_new,
                dim_head=multi_out_dim,
                heads=1)
    
        # Layer normalization
        self.layer_norm = nn.LayerNorm(multi_out_dim)

        # Final fusion and projection
        out_classifier_dim = 4 * multi_out_dim
        self.fusion_projection = nn.Linear(out_classifier_dim, self.embed_dim)
        
        # Initialize the distance correlation loss module
        self.dist_corr_loss = DistanceCorrelationLoss(weight_D1=0.5, weight_D2=0.5)
        
        print(f"DIMAFFusionModule Params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def append_modality_embeddings(self, embeddings):
        """ Append the learnable modality embeddings to each feature. """
        batch_size = embeddings[0].size(0)
        
        # Expand modality embeddings for the batch
        modality_embeddings_exp = self.modality_embeddings.expand(batch_size, -1, -1)
        
        # Concatenate original embeddings with modality embeddings
        enhanced_embeddings = []
        for i, emb in enumerate(embeddings):
            enhanced_emb = torch.cat([emb, modality_embeddings_exp[:, i:i+1, :].expand(-1, emb.size(1), -1)], dim=-1)
            enhanced_embeddings.append(enhanced_emb)
        
        return enhanced_embeddings

    def disentangled_attention_fusion(self, emb1, emb2):
        """ Pass through disentangled fusion. """
        # Self-attention within each modality
        zp_11 = self.self_attention_mod1(emb1, emb1)
        zp_22 = self.self_attention_mod2(emb2, emb2)

        # Cross-attention between modalities
        zp_12 = self.cross_attention_1to2(emb1, emb2)
        zp_21 = self.cross_attention_2to1(emb2, emb1)

        token_nums = [zp_11.size(1), zp_21.size(1), zp_12.size(1), zp_22.size(1)]  # [N1, N2, N1, N2]
        return zp_11, zp_21, zp_12, zp_22, token_nums

    def compute_fusion_losses(self, z_11, z_21, z_12, z_22):
        """计算融合过程中的解耦损失，使用距离相关损失替代正交性损失"""
        # 使用距离相关损失计算解耦损失
        dist_corr_loss, loss_components = self.dist_corr_loss(z_11, z_22, z_12, z_21)
        
        losses = {}
        losses['disentanglement_loss'] = dist_corr_loss
        
        return losses

    def forward(self, embeddings: List[torch.Tensor], masks: List[Optional[torch.Tensor]]) -> Dict:
        """
        Forward pass for DIMAF fusion.
        
        Args:
            embeddings: List of tensors, each of shape (B, N_i, D) where N_i is the number of tokens for modality i
            masks: List of masks corresponding to each modality, or None if modality is not present
            
        Returns:
            Dictionary with fused embedding and optional loss components
        """
        device = embeddings[0].device if embeddings[0] is not None else 'cpu'
        
        # Filter out None embeddings and corresponding masks
        present_embeddings = [e for e in embeddings if e is not None]
        present_masks = [m for e, m in zip(embeddings, masks) if e is not None]
        
        fusion_losses_dict = {}
        
        if len(present_embeddings) < 2:
            # If we don't have at least 2 modalities, we can't do cross-attention
            # Just pool and concatenate the available modalities
            pooled_embeddings = []
            for emb, mask in zip(present_embeddings, present_masks):
                pooled_emb = masked_mean_pool(emb, mask)  # (B, D)
                pooled_embeddings.append(pooled_emb)
            
            if len(pooled_embeddings) == 1:
                # Only one modality available, expand it
                fused_embedding = pooled_embeddings[0]
            else:
                # Concatenate available embeddings
                fused_embedding = torch.cat(pooled_embeddings, dim=-1)
                # Project to embed_dim if needed
                if fused_embedding.size(-1) != self.embed_dim:
                    fused_embedding = self.fusion_projection(fused_embedding)
        else:
            # We have at least 2 modalities, use full DIMAF fusion
            # For simplicity, we'll use the first two present modalities

            assert len(present_embeddings) == 2, f"DIMAF Fusion Module Only Support two modalities input." 

            emb1, emb2 = present_embeddings[0], present_embeddings[1]
            mask1, mask2 = present_masks[0], present_masks[1]
            
            # Append modality embeddings
            enhanced_embeddings = self.append_modality_embeddings([emb1, emb2])
            enhanced_emb1, enhanced_emb2 = enhanced_embeddings[0], enhanced_embeddings[1]

            zp_11, zp_21, zp_12, zp_22, token_nums = self.disentangled_attention_fusion(enhanced_emb1, enhanced_emb2)
            # 直接对每个zp单独归一化，无需拼接再分割
            zp_11 = self.layer_norm(zp_11)
            zp_21 = self.layer_norm(zp_21)
            zp_12 = self.layer_norm(zp_12)
            zp_22 = self.layer_norm(zp_22)

            # Pool each attention type using masks if available
            # For simplicity in this implementation, we're using mean pooling
            z_11 = masked_mean_pool(zp_11, mask1)  # zp_11对应mod1，用mask1
            z_21 = masked_mean_pool(zp_21, mask2)  # zp_21对应mod2，用mask2
            z_12 = masked_mean_pool(zp_12, mask1)  # zp_12对应mod1，用mask1
            z_22 = masked_mean_pool(zp_22, mask2)  # zp_22对应mod2，用mask2
            
            # 计算解耦损失
            fusion_losses_dict = self.compute_fusion_losses(z_11, z_21, z_12, z_22)
            
            # Concatenate disentangled embeddings
            disentangled_embedding = torch.cat([z_11, z_21, z_12, z_22], dim=1)
            
            # Project to final embedding dimension
            fused_embedding = self.fusion_projection(disentangled_embedding)
        
        # 计算总融合损失
        total_fusion_loss = fusion_losses_dict.get('disentanglement_loss', torch.tensor(0.0, device=device))
        fusion_losses_dict['total'] = total_fusion_loss
        
        return {
            "fused_embedding": fused_embedding,
            "loss_dict": fusion_losses_dict
        }