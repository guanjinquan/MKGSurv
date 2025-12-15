import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# 1. 基础组件 (Helper Classes & Transformer Blocks)
#    保持不变，用于构建 Transformer 结构
# ==============================================================================

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
    def __init__(self, dim, mult = 4, dropout = 0.):
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
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            # Mask shape handling: (b, j) -> (b*h, i, j)
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask.bool(), max_neg_value)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)


# ==============================================================================
# 2. HealNet Layer
#    逻辑：Latents 是共用的，依次与每个模态进行 Cross-Attention
# ==============================================================================

class HealNetLayer(nn.Module):
    """
    Standard HealNet Layer where latents are shared across modalities.
    """
    def __init__(self, latent_dim, max_modalities, 
                 x_heads=8, l_heads=8, cross_dim_head=64, latent_dim_head=64, 
                 attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.max_modalities = max_modalities
        
        # 1. Cross Attention Blocks: 为每个模态分配一个独立的 Cross Attention
        # 即使 Latent 是共用的，不同模态的特征空间可能不同，因此保留独立的 Cross Attn 参数
        self.cross_attn_blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': PreNorm(latent_dim, Attention(latent_dim, latent_dim, heads=x_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=latent_dim),
                'ff': PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
            }) for _ in range(max_modalities)
        ])

        # 2. Global Self Attention
        # 融合所有信息后的 Latents 进行自交互
        self.self_attn = PreNorm(latent_dim, Attention(latent_dim, heads=l_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        self.self_ff = PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

    def forward(self, latents, modalities, masks):
        """
        Args:
            latents: (B, num_latents, D) - SHARED latents
            modalities: List[Tensor | None], length = max_modalities
            masks: List[Tensor | None], length = max_modalities
        """
        x = latents

        # A. Sequential Cross Attention (Latents attend to Modality 1, then Modality 2, etc.)
        for i, (feat, mask) in enumerate(zip(modalities, masks)):
            block = self.cross_attn_blocks[i]
            
            if feat is not None:
                # 只有当模态存在时，才进行 Cross Attention
                # 共用的 x 获取当前模态的信息并更新
                out = block['attn'](x, context=feat, mask=mask) + x
                out = block['ff'](out) + out
                x = out
            # else: 模态缺失，x 保持不变，携带之前的信息继续传递

        # B. Global Self Attention
        x = self.self_attn(x) + x
        x = self.self_ff(x) + x

        return x


# ==============================================================================
# 3. HealNet Fusion Model (Shared Latents)
# ==============================================================================

class HealNetFusion(nn.Module):
    """
    Pure HealNet Fusion with Shared Latents.
    Latents 是一组共用的查询向量，依次查询所有可用的模态。
    """
    def __init__(
        self,
        args,
        embed_dim: int,
        max_modalities: int,
        num_latents: int = 256,     # Latent 总数 (共用)
        depth: int = 2,
        num_heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        **kwargs # 吸收多余参数
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.max_modalities = max_modalities
        self.num_latents = num_latents
        
        # 1. Latent Initialization (Shared)
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim))
        
        # 2. Layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(HealNetLayer(
                latent_dim=embed_dim,
                max_modalities=max_modalities,
                x_heads=num_heads,
                l_heads=num_heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            ))

        # 3. Final Output Head
        self.to_logits = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        # --- 0. 基础信息获取 ---
        batch_size = -1
        device = None
        for e in embeddings:
            if e is not None:
                batch_size = e.shape[0]
                device = e.device
                break
        
        if batch_size == -1: return None 

        # --- 1. 整理 Input List ---
        # 确保输入列表长度为 max_modalities，不足的补 None
        
        proc_embeddings = list(embeddings)
        proc_masks = list(masks)

        # 填充或截断列表至 max_modalities
        if len(proc_embeddings) < self.max_modalities:
            diff = self.max_modalities - len(proc_embeddings)
            proc_embeddings.extend([None] * diff)
            proc_masks.extend([None] * diff)
        
        proc_embeddings = proc_embeddings[:self.max_modalities]
        proc_masks = proc_masks[:self.max_modalities]

        # --- 2. Latent Forward ---
        # 广播 Shared Latents 到当前 Batch
        x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        for layer in self.layers:
            x = layer(x, proc_embeddings, proc_masks)

        # --- 3. Output Generation (Global Pooling) ---
        
        # 策略：由于 Latents 是共用的，它们包含了所有存在模态的信息。
        # 直接对 Latent 序列进行 Mean Pooling 得到最终特征。
        
        fused_embedding = x.mean(dim=1) # (B, D)
        fused_embedding = self.to_logits(fused_embedding)

        return {
            "fused_embedding": fused_embedding,
        }