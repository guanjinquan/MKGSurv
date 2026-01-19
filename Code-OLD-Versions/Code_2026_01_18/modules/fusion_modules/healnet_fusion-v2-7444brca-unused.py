from math import pi
from functools import wraps
from typing import *

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce
import random


# --- Helper utility functions and classes from the original HealNet implementation ---

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

        self.to_q = nn.Linear(query_dim, inner_dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.attn_weights = None

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
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


# --- Modified HealNet Implementation ---
class HealNet(nn.Module):
    def __init__(
        self,
        *,
        n_modalities: int,
        channel_dims: List,
        num_spatial_axes: List,
        out_dims: int,
        depth: int = 3,
        num_freq_bands: int = 2,
        max_freq: float=10.,
        l_c: int = 128,
        l_d: int = 128,
        x_heads: int = 8,
        l_heads: int = 8,
        cross_dim_head: int = 64,
        latent_dim_head: int = 64,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        weight_tie_layers: bool = True,
        fourier_encode_data: bool = True,
        self_per_cross_attn: int = 1,
        final_classifier_head: bool = True,
        snn: bool = True,
    ):
        super().__init__()
        assert len(channel_dims) == len(num_spatial_axes), 'input channels and input axis must be of the same length'
        assert len(num_spatial_axes) == n_modalities, 'input axis must be of the same length as the number of modalities'

        self.input_axes = num_spatial_axes
        self.input_channels=channel_dims
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.modalities = n_modalities
        self.self_per_cross_attn = self_per_cross_attn
        self.fourier_encode_data = fourier_encode_data

        fourier_channels = [(axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0 for axis in num_spatial_axes]
        input_dims = [f + i for f, i in zip(fourier_channels, channel_dims)]

        self.latents = nn.Parameter(torch.randn(l_c, l_d))

        funcs = [
            lambda m=m: PreNorm(l_d, Attention(l_d, input_dims[m], heads = x_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dims[m])
            for m in range(n_modalities)
        ]
        cross_attn_funcs = tuple(map(cache_fn, tuple(funcs)))

        get_latent_attn = lambda: PreNorm(l_d, Attention(l_d, heads = l_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_cross_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))
        get_latent_ff = lambda: PreNorm(l_d, FeedForward(l_d, dropout = ff_dropout, snn = snn))

        get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])
            for block_ind in range(self_per_cross_attn):
                self_attns.append(get_latent_attn(**cache_args, key = block_ind))
                self_attns.append(get_latent_ff(**cache_args, key = block_ind))

            cross_attn_layers = []
            for j in range(n_modalities):
                cross_attn_layers.append(cross_attn_funcs[j](**cache_args))
                cross_attn_layers.append(get_cross_ff(**cache_args))
            
            self.layers.append(nn.ModuleList([*cross_attn_layers, self_attns]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(l_d), 
            nn.Linear(l_d, out_dims)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        tensors: List[Union[torch.Tensor, None]],
        masks: Optional[List[Optional[torch.Tensor]]] = None,
        return_embeddings: bool = False,
        verbose: bool = False
    ):
        b, device, dtype = -1, None, None
        processed_tensors = [None] * len(tensors)
        missing_idx = [i for i, t in enumerate(tensors) if t is None]
        
        # Find batch size from the first available tensor
        for t in tensors:
            if t is not None:
                b, device, dtype = t.shape[0], t.device, t.dtype
                break
        
        if b == -1: # all modalities are missing
            return None

        for i, data in enumerate(tensors):
            if i in missing_idx:
                continue

            b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
            assert len(axis) == self.input_axes[i], (
                f'Input data for modality {i+1} must have the same number of axes as the input_axes parameter')

            if self.fourier_encode_data:
                axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
                pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
                enc_pos = repeat(enc_pos, '... -> b ...', b=b)
                data = torch.cat((data, enc_pos), dim=-1)
            
            processed_tensors[i] = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        for layer_idx, layer in enumerate(self.layers):

            for i in range(self.modalities):
                if i in missing_idx:
                    if verbose:
                        print(f"Skipping update in fusion layer {layer_idx + 1} for missing modality {i+1}")
                    continue
                
                cross_attn = layer[i * 2]
                cross_ff = layer[(i * 2) + 1]
                current_mask = masks[i] if exists(masks) and i < len(masks) and exists(masks[i]) else None
                
                context_data = processed_tensors[i]
                if exists(context_data):
                    x = cross_attn(x, context=context_data, mask=current_mask) + x
                    x = cross_ff(x) + x

            if self.self_per_cross_attn > 0:
                for self_attn, self_ff in zip(layer[-1][::2], layer[-1][1::2]):
                    x = self_attn(x) + x
                    x = self_ff(x) + x

        if return_embeddings:
            return x
        
        return self.to_logits(x)


# --- Main HealNet Fusion Module for Project Integration ---
class HealNetFusionModule(nn.Module):
    """
    HealNet fusion module adapted for dynamic multimodal fusion.
    This module wraps the HealNet architecture to provide a consistent
    interface for fusing a list of embeddings, each with an associated mask.
    """
    def __init__(
        self,
        args,
        embed_dim: int,
        max_modalities: int,
        num_latents: int = 128, # Corresponds to l_c, the number of queries (k)
        depth: int = 3,
        num_heads: int = 8,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0
    ):
        super(HealNetFusionModule, self).__init__()
        self.args = args
        self.max_modalities = max_modalities
        self.embed_dim = embed_dim

        self.healnet = HealNet(
            n_modalities=max_modalities,
            channel_dims=[embed_dim] * max_modalities,
            num_spatial_axes=[1] * max_modalities,
            out_dims=embed_dim, 
            l_c=num_latents,    
            l_d=embed_dim,      
            depth=depth,
            x_heads=num_heads,
            l_heads=num_heads,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            fourier_encode_data=False 
        )

    def forward(self, embeddings: List[Optional[torch.Tensor]], masks: List[Optional[torch.Tensor]], **kargs) -> Dict:
        """
        Forward pass for the HealNet fusion module.

        Args:
            embeddings (List[Optional[torch.Tensor]]): A list of embedding tensors for each modality.
                                                       Each tensor should be of shape (b, n, d).
                                                       A value of None indicates a missing modality.
            masks (List[Optional[torch.Tensor]]): A list of boolean masks for each modality.
                                                  Each tensor should be of shape (b, n).
                                                  A value of None indicates a missing mask.

        Returns:
            Dict: A dictionary containing the fused embedding and a None loss_dict.
                  {'fused_embedding': torch.Tensor, 'loss_dict': None}
        """
        assert len(embeddings) <= self.max_modalities, \
            f"Number of embeddings ({len(embeddings)}) exceeds max_modalities ({self.max_modalities})"
        
        # Check Data type 
        masks = [m.to(torch.bool) if m is not None else None for m in masks]
        embeddings = [e.to(torch.float) if e is not None else None for e in embeddings]

        # Pad the lists with None up to max_modalities if needed
        num_provided = len(embeddings)
        if num_provided < self.max_modalities:
            embeddings.extend([None] * (self.max_modalities - num_provided))
            masks.extend([None] * (self.max_modalities - num_provided))

        # The input embeddings are already in (b, n, d) format.
        # HealNet's internal rearrange('b ... d -> b (...) d') will handle this shape correctly.
        # The masks are (b, n) and will also be handled correctly.
        fused_embedding = self.healnet(tensors=embeddings, masks=masks)

        return {"fused_embedding": fused_embedding}
