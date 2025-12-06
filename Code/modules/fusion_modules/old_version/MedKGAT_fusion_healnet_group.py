import torch
from torch import nn, einsum
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Union
from functools import wraps
from math import pi

from einops import rearrange, repeat
from einops.layers.torch import Reduce

# ==========================================
# --- Helper Functions & Classes (As Provided) ---
# ==========================================

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., snn: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2)
        )
        self.attn_weights = None

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            # Rearrange mask to match attention map shape
            # Expecting mask shape: (b, j) where j is context length
            # If mask is boolean: True = keep, False = mask out
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        self.attn_weights = attn
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


# ==========================================
# --- HealNet Group Implementation ---
# ==========================================

class HealNet_Group(nn.Module): # V3版本
    """
    A modified HealNet implementation that fuses grouped embeddings.
    
    This replaces the 'SafeCrossAttnEncoder' from the snippet with 
    PreNorm(Attention(...)) blocks consistent with the provided helper classes.
    """
    def __init__(
        self, 
        args,
        embed_dim: int, 
        max_modalities: int = 10, 
        max_groups: int = 10, 
        num_latents: int = 32,
        latent_heads: int = 8,
        latent_dim_head: int = 64,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        depth: int = 1 # Number of times to iterate self-attention/ff per group step
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.max_groups = max_groups
        self.max_modalities = max_modalities

        # 1. Initialize Latent Queries (B, num_latents, embed_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))
        
        # 2. Cross-Attention Layers (One per potential group)
        # We also include a FeedForward block for each cross-attn step to ensure capacity
        self.cross_attn_layers = nn.ModuleList([])
        self.cross_ff_layers = nn.ModuleList([])

        for _ in range(max_modalities):
            self.cross_attn_layers.append(
                PreNorm(embed_dim, Attention(
                    query_dim=embed_dim, 
                    context_dim=embed_dim, 
                    heads=latent_heads, 
                    dim_head=latent_dim_head, 
                    dropout=attn_dropout
                ), context_dim=embed_dim)
            )
            self.cross_ff_layers.append(
                PreNorm(embed_dim, FeedForward(embed_dim, dropout=ff_dropout))
            )

        # 3. Latent Self-Attention Layer (Post-Fusion)
        # This allows latents to reason about the fused information
        self.self_attn_layers = nn.ModuleList([])
        self.self_ff_layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.self_attn_layers.append(
                PreNorm(embed_dim, Attention(
                    query_dim=embed_dim, 
                    heads=latent_heads, 
                    dim_head=latent_dim_head, 
                    dropout=attn_dropout
                ))
            )
            self.self_ff_layers.append(
                PreNorm(embed_dim, FeedForward(embed_dim, dropout=ff_dropout))
            )

        # 4. Final Normalization
        self.post_fusion_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),  # 还有可能是最后一层的问题
        )

    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: List of tensors, usually shape (B, Seq_Len, D) or (B, 1, D)
            masks: List of mask tensors, shape (B, Seq_Len) or (B, 1). 1=Valid, 0=Padding.
            embeddings_groups: List of lists, e.g. [[0, 1], [2], [3, 4]] indicating which
                               embeddings belong to which modality group.
        """
        
        # --- 1. Create Group-Level Embeddings ---
        # group_embeddings = []
        # group_masks = []
        
        # # We iterate through the provided indices to concatenate relevant embeddings
        # for group_indices in embeddings_groups:
        #     if not group_indices:
        #         continue

        #     # Gather features for this group
        #     curr_feats = [embeddings[i] for i in group_indices]
        #     curr_masks = [masks[i] for i in group_indices]

        #     # Concatenate along the sequence dimension (dim=1)
        #     # Assuming inputs are (B, N, D), result is (B, Sum_N, D)
        #     g_feat = torch.cat(curr_feats, dim=1)
        #     g_mask = torch.cat(curr_masks, dim=1)

        #     group_embeddings.append(g_feat)
        #     group_masks.append(g_mask)
            
        # --- 2. Initialize Latents ---
        batch_size = embeddings[0].shape[0] if embeddings else 1
        x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        # --- 3. Iterative Fusion (HealNet Style) ---
        # Latents attend to each group sequentially
        
        for i, group_emb in enumerate(embeddings):
            if i >= len(self.cross_attn_layers):
                break # Safety check if groups exceed max_groups

            if group_emb is None:
                # print(f"Skipping group {i} because it's None")
                raise ValueError(f"Group {i} is None")
            #     continue

            # Helper classes handle mask logic (True = keep, False = mask)
            # Input masks are usually 1/0, convert to Bool if needed
            # The Attention class above uses `~mask` to fill with neg infinity,
            # so we want mask to be True for valid tokens.
            context_mask = masks[i] > 0
            
            # Cross Attention: Latents (Q) attend to Group (K, V)
            attn_out = self.cross_attn_layers[i](x, context=group_emb, mask=context_mask)
            x = x + attn_out
            
            # Feed Forward
            ff_out = self.cross_ff_layers[i](x)
            x = x + ff_out

        # --- 4. Latent Self-Attention & Processing ---
        # Process the latents among themselves after gathering info from all groups
        for self_attn, self_ff in zip(self.self_attn_layers, self.self_ff_layers):
            x = x + self_attn(x)
            x = x + self_ff(x)

        # --- 5. Pooling & Output ---
        # Mean Pool Latents to get final embedding (B, D)
        # Note: Original HealNet often uses the latents directly for classification,
        # but the snippet requested a pooled output.
        fused_embedding = x.mean(dim=1)
        fused_embedding = self.post_fusion_layer(fused_embedding)

        return {
            "fused_embedding": fused_embedding,
        }
