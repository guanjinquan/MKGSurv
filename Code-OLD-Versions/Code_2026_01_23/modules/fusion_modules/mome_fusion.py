import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

# ==============================================================================
# 1. Helper Layers & Utilities (Adapted from MoME Reference)
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SNN_Block(nn.Module):
    """
    Self-Normalizing Neural Network Block.
    Based on description: Linear -> ELU -> AlphaDropout
    """
    def __init__(self, dim1, dim2, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim1, dim2)
        self.act = nn.ELU()
        self.dropout = nn.AlphaDropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.act(self.linear(x)))

class StandardTransLayer(nn.Module):
    """
    Replaces the external dependency 'admin_torch' and 'nystrom_attention'.
    Uses standard PyTorch MultiheadAttention for stability and portability.
    """
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, dropout=0.1):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # Feed Forward part commonly found in Transformer Layers
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        # Self Attention with Residual
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Feed Forward with Residual
        x_norm2 = self.norm2(x)
        x = x + self.ff(x_norm2)
        return x

# ==============================================================================
# 2. Expert Modules
# ==============================================================================

class TransFusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.translayer = StandardTransLayer(norm_layer, dim)

    def forward(self, x1, x2):
        # Concatenate along sequence dimension
        x = torch.cat([x1, x2], dim=1)
        x = self.translayer(x)
        # Return only the portion corresponding to x1
        return x[:, :x1.shape[1], :]

class BottleneckTransFusion(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.n_bottlenecks = n_bottlenecks
        self.attn1 = StandardTransLayer(nn.LayerNorm, dim=dim)
        self.attn2 = StandardTransLayer(nn.LayerNorm, dim=dim)
        # Changed from .cuda() to nn.Parameter for device compatibility
        self.bottleneck = nn.Parameter(torch.randn(1, n_bottlenecks, dim))

    def forward(self, x1, x2):
        b_size = x1.shape[0]
        # Expand bottleneck to batch size
        bottleneck_tokens = self.bottleneck.expand(b_size, -1, -1)
        
        # Interaction between Bottleneck and x2
        bottleneck_combined = torch.cat([bottleneck_tokens, x2], dim=1)
        bottleneck_out = self.attn2(bottleneck_combined)[:, :self.n_bottlenecks, :]
        
        # Interaction between x1 and Updated Bottleneck
        x = torch.cat([x1, bottleneck_out], dim=1)
        x = self.attn1(x)
        return x[:, :x1.shape[1], :]

class AddFusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.snn1 = SNN_Block(dim1=dim, dim2=dim)
        self.snn2 = SNN_Block(dim1=dim, dim2=dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        # x2 is averaged to interact as a global context context
        x2_context = self.snn2(self.norm2(x2)).mean(dim=1, keepdim=True)
        return self.snn1(self.norm1(x1)) + x2_context

class DropX2Fusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()

    def forward(self, x1, x2):
        return x1

# ==============================================================================
# 3. Gating & Routing Logic
# ==============================================================================

def diff_softmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through estimator
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class RoutingNetwork(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=256):
        super(RoutingNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim, dim),
            norm_layer(dim),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim, dim),
            norm_layer(dim),
            nn.GELU(),
        )
        self.clsfer = nn.Linear(dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        x1_mapped = self.fc1(x1).mean(dim=1)
        x2_mapped = self.fc2(x2).mean(dim=1)
        x_sum = x1_mapped + x2_mapped
        logits = diff_softmax(self.clsfer(x_sum), tau=temp, hard=hard, dim=1)
        return logits

class MoME(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256):
        super().__init__()
        self.TransFusion = TransFusion(norm_layer, dim)
        self.BottleneckTransFusion = BottleneckTransFusion(n_bottlenecks, norm_layer, dim)
        self.AddFusion = AddFusion(norm_layer, dim)
        self.DropX2Fusion = DropX2Fusion(norm_layer, dim)
        
        self.routing_network = RoutingNetwork(4, norm_layer=norm_layer, dim=dim)
        self.routing_dict = nn.ModuleDict({
            '0': self.TransFusion,
            '1': self.BottleneckTransFusion,
            '2': self.AddFusion,
            '3': self.DropX2Fusion,
        })

    def forward(self, x1, x2, hard=True):
        """
        x1: Modality to be encoded
        x2: Reference modality
        """
        # Get routing weights. 
        # Shape: (Batch, 4)
        # If hard=True, these are one-hot vectors (via straight-through estimator).
        logits = self.routing_network(x1, x2, hard=hard)
        
        # Calculate weighted average of experts
        # We process all experts and apply the routing weights.
        # This supports batches where different samples route to different experts.
        output = torch.zeros_like(x1)
        for i in range(4):
            # weight shape: (Batch, 1, 1) to broadcast over sequence and dim
            weight = logits[:, i].view(-1, 1, 1)
            expert_out = self.routing_dict[str(i)](x1, x2)
            output = output + (weight * expert_out)
            
        return output

# ==============================================================================
# 4. Main Class Interface
# ==============================================================================

class MOME_fusion(nn.Module):
    def __init__(self, args, embed_dim: int = 512, max_modalities: int = 2) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        # Default bottleneck size if not in args
        self.n_bottlenecks = getattr(args, 'n_bottlenecks', 4)
        
        # -----------------------------------------------------------
        # Biased Progressive Encoding Structure
        # -----------------------------------------------------------
        
        # Round 1:
        self.mome_r1_m1 = MoME(n_bottlenecks=self.n_bottlenecks, dim=embed_dim) # Updates M1 using M2
        self.mome_r1_m2 = MoME(n_bottlenecks=self.n_bottlenecks, dim=embed_dim) # Updates M2 using M1
        
        # Round 2:
        self.mome_r2_m1 = MoME(n_bottlenecks=self.n_bottlenecks, dim=embed_dim)
        self.mome_r2_m2 = MoME(n_bottlenecks=self.n_bottlenecks, dim=embed_dim)

        # -----------------------------------------------------------
        # Final Aggregation
        # -----------------------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.final_attn = StandardTransLayer(dim=embed_dim)
        
    def forward(self, embeddings: List[torch.Tensor], masks: List[Optional[torch.Tensor]] = None) -> Dict:
        """
        Args:
            embeddings: List of 2 tensors. Each tensor shape: (B, Seq_Len, Dim) or (B, Dim)
            masks: List of masks (not strictly used in basic MoME logic, but kept for interface)
        """
        assert len(embeddings) == 2, f"MOMEFusion Module Only Support two modalities input. Got {len(embeddings)}"

        # 1. Standardization: Ensure inputs are (Batch, Seq, Dim)
        m1 = embeddings[0]
        m2 = embeddings[1]

        if m1.dim() == 2: m1 = m1.unsqueeze(1)
        if m2.dim() == 2: m2 = m2.unsqueeze(1)

        # 2. Biased Progressive Encoding (Round 1)
        # Update M1 utilizing M2 as reference
        h1_r1 = self.mome_r1_m1(m1, m2, hard=True)
        # Update M2 utilizing M1 as reference
        h2_r1 = self.mome_r1_m2(m2, m1, hard=True)

        # 3. Biased Progressive Encoding (Round 2)
        # Deep fusion: Use updated features from Round 1
        h1_final = self.mome_r2_m1(h1_r1, h2_r1, hard=True)
        h2_final = self.mome_r2_m2(h2_r1, h1_r1, hard=True)

        # 4. Final Aggregation
        # Combine [CLS_Token, H1, H2]
        b_size = m1.shape[0]
        cls_tokens = self.cls_token.expand(b_size, -1, -1)
        
        # Concatenate all tokens
        combined = torch.cat([cls_tokens, h1_final, h2_final], dim=1)
        
        # Pass through final attention layer
        fused_sequence = self.final_attn(combined)
        
        # Extract the CLS token as the global fused representation
        # Shape: (Batch, Dim)
        fused_embedding = fused_sequence[:, 0, :]

        # MoME relies on routing decisions, which are discrete in 'hard' mode or soft otherwise.
        # Often auxiliary losses are added for load balancing in MoE, 
        # but the provided reference didn't explicitly return loss terms for the gating network.
        # We return an empty dict or placeholders if needed.
        fusion_losses_dict = {}

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": fusion_losses_dict
        }

# ==============================================================================
# 5. Unit Testing
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    print("Initializing MOME_fusion Test...")
    
    # Mock Args
    class Args:
        n_bottlenecks = 4
    args = Args()
    
    # Settings
    batch_size = 2
    dim = 64 # Reduced dim for quick testing
    
    # Initialize Model
    model = MOME_fusion(args, embed_dim=dim)
    model.eval() # Test in eval mode usually, though MoME has routing logic
    
    # Create Dummy Data
    # Simulating WSI features (Batch, 100 patches, dim)
    input_wsi = torch.randn(batch_size, 100, dim)
    # Simulating Genomic features (Batch, 6 signatures, dim)
    input_omic = torch.randn(batch_size, 6, dim)
    
    embeddings = [input_wsi, input_omic]
    masks = [None, None]
    
    # Forward Pass
    try:
        output = model(embeddings, masks)
        fused = output["fused_embedding"]
        
        print("\nTest Successful!")
        print(f"Input 1 Shape: {input_wsi.shape}")
        print(f"Input 2 Shape: {input_omic.shape}")
        print(f"Fused Output Shape: {fused.shape}") # Should be (Batch, dim)
        
        assert fused.shape == (batch_size, dim), "Output shape mismatch"
        assert not torch.isnan(fused).any(), "Output contains NaNs"
        
    except Exception as e:
        print(f"\nTest Failed with error: {e}")
        import traceback
        traceback.print_exc()