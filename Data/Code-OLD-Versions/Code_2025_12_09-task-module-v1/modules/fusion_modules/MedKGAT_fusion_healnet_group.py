import torch
from torch import nn, einsum
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Union
from functools import wraps
from math import pi

from einops import rearrange, repeat
from einops.layers.torch import Reduce
import os
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

        self.args = args
        
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
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        analysis_mode = False,
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
        group_embeddings = []
        group_masks = []
        
        # We iterate through the provided indices to concatenate relevant embeddings
        for group_indices in embeddings_groups:
            if not group_indices:
                continue

            # Gather features for this group
            curr_feats = [embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]

            # Concatenate along the sequence dimension (dim=1)
            # Assuming inputs are (B, N, D), result is (B, Sum_N, D)
            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)

            group_embeddings.append(g_feat)
            group_masks.append(g_mask)
            
        # --- 2. Initialize Latents ---
        batch_size = embeddings[0].shape[0] if embeddings else 1
        x = repeat(self.latents, 'n d -> b n d', b=batch_size)

        # --- 3. Iterative Fusion (HealNet Style) ---
        # Latents attend to each group sequentially
        
        for i, group_emb in enumerate(group_embeddings):
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
            context_mask = group_masks[i] > 0
            
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


                # --- analysis --- 
        # Save Features
        if not self.training:
            if hasattr(self.args, 'save_umap_path') and self.args.save_umap_path:
                self.save_features_for_umap(group_embeddings, group_masks, fused_embedding)

        # Save Points (Visualization/Debugging)
        if hasattr(self, 'save_points'):
            self.save_points(group_embeddings, group_masks, groups_relationships)

        
        # 1. 捕获逻辑 (Capture Logic)
        # 只有在 analysis_mode=True 时，或者你强制想看梯度时运行
        if analysis_mode:
            # 清空旧数据
            self.captured_group_feats = []
            self.captured_group_masks = []
            
            # print("\n[Debug Forward] Start capturing (Inner Loop)...")
            for feat in group_embeddings:
                # 只有带梯度的才保留，避免报错
                if feat.requires_grad:
                    feat.retain_grad()
                # else:
                    # print("[Warning] Feature has no gradient requirements!")
            
            # 保存当前计算图中的 Tensor
            self.captured_group_feats = group_embeddings
            self.captured_group_masks = group_masks


        # GradCAM Analysis
        should_run_gradcam = (
            not self.training 
            and hasattr(self.args, 'gradcam_save_path') 
            and self.args.gradcam_save_path is not None
            and not analysis_mode  # <--- 防止递归的关键
        )

        if should_run_gradcam:
            # print("[Debug Forward] Triggering GradCAM Analysis...")
            self.gradcam_analyse(
                embeddings=embeddings,
                masks=masks,
                embeddings_groups=embeddings_groups,
                groups_relationships=groups_relationships
            )



        return {
            "fused_embedding": fused_embedding,
        }


    def save_features_for_umap(self, group_embeddings, group_masks, fused_embedding):
        """
        保存特征用于 UMAP 可视化。
        修改：保存所有 Valid Tokens，而不是 Pooling 后的向量。
        
        JSONL 格式变更:
        {
            "groups": [[dim1...], [dim2...] ...],   # 所有 Valid Tokens 的列表 (Flattened)
            "group_ids": [0, 0, 1, 2...],           # 每个 Token 对应的 Group Index
            "fused": [dim1, dim2...]                # 融合后的特征
        }
        """
        import os
        import json
        
        # 确保路径存在
        save_path = self.args.save_umap_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        batch_size = fused_embedding.shape[0]
        
        # 1. 准备数据 (移到 CPU)
        # fused: (B, D)
        fused_emb = fused_embedding.detach().cpu()
        # groups: List of (B, L, D)
        cpu_groups = [g.detach().cpu() for g in group_embeddings]
        # masks: List of (B, L)
        cpu_masks = [m.detach().cpu() for m in group_masks]

        # 3. 写入文件
        with open(save_path, 'a', encoding='utf-8') as f:
            for b in range(batch_size):
                all_tokens = []
                all_token_ids = []
                
                # 遍历每个 Group
                for g_idx, (g_feat, g_mask) in enumerate(zip(cpu_groups, cpu_masks)):
                    # g_feat[b]: (Seq, Dim)
                    # g_mask[b]: (Seq)
                    
                    curr_feat = g_feat[b]
                    curr_mask = g_mask[b]
                    
                    # 获取 Valid Indices (Mask != 0)
                    # nonzero 返回 (Num_Valid, 1), squeeze 后变成 (Num_Valid)
                    # 假设 mask 中 0 是 padding
                    valid_indices = torch.nonzero(curr_mask).squeeze(-1)
                    
                    if valid_indices.numel() > 0:
                        # 提取 Valid Tokens
                        valid_tokens = curr_feat[valid_indices] # (Num_Valid, Dim)
                        
                        # 添加到列表
                        for tok in valid_tokens:
                            all_tokens.append(tok.tolist())
                            all_token_ids.append(g_idx)

                record = {
                    "groups": all_tokens,       # Flattened valid tokens from all groups
                    "group_ids": all_token_ids, # Corresponding group index for each token
                    "fused": fused_emb[b].tolist()
                }
                f.write(json.dumps(record) + "\n")
                
    def view_groups_contribution(self, attn_weights: torch.Tensor, values: torch.Tensor, group_masks: List[torch.Tensor]):
        """
        方案1实现：基于范数(Energy)的贡献度分析。
        保存格式：与之前一致，JSONL 每行一个列表 [g0_ratio, g1_ratio, ...]
        
        Args:
            attn_weights: (B, L, L) or (B, H, L, L) - 注意力权重
            values: (B, L, D) - Transformer 的输入 (即 Global Concat)
            group_masks: List[(B, L_g)]
        """
        if not hasattr(self.args, 'view_groups_attention_path') or self.args.view_groups_attention_path is None:
            return
        
        save_path = self.args.view_groups_attention_path
        # 为了区分，建议修改一下文件名，或者保持原样覆盖
        # save_path = save_path.replace('.jsonl', '_contribution.jsonl') 
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if attn_weights is None or values is None:
            return
        
        print("attn_weights shape: ", attn_weights.shape)

        # 1. 维度与数据检查
        # 如果是多头 (B, H, L, L)，先平均成 (B, L, L)
        if attn_weights.dim() == 4:
            attn_weights = attn_weights.mean(dim=1)

        # 确保 attn 是 (B, L, L)
        if attn_weights.shape[0] != group_masks[0].shape[0]:
            attn_weights = attn_weights.permute(1, 0, 2)
            
        # 确保 values 是 (B, L, D)
        if values.shape[0] != group_masks[0].shape[0]:
            values = values.transpose(0, 1)

        # 2. 强制 Softmax 检查 (Contribution 分析必须基于概率)
        check_sum = attn_weights[0, 0, :].sum().item()
        if check_sum > 1.1 or check_sum < 0.9:
            # print("[Info] Applying Softmax for contribution analysis...")
            attn_weights = torch.softmax(attn_weights, dim=-1)

        # 3. 准备 Mask 和 Offsets
        global_mask = torch.cat(group_masks, dim=1).float() # (B, L_total)
        num_valid_queries = global_mask.sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1)

        group_lengths = [gm.shape[1] for gm in group_masks]
        offsets = [0]
        for l in group_lengths:
            offsets.append(offsets[-1] + l)

        # 4. 核心计算循环：计算每个组的 Energy
        group_energy_list = []

        for i in range(len(group_masks)):
            start, end = offsets[i], offsets[i+1]
            
            # A. 取出该组对应的 Attention 概率 (B, L_total, L_group)
            # 代表：每个 Token 对该组分配了多少关注
            attn_slice = attn_weights[:, :, start:end]
            
            # # B. 取出该组对应的 Feature Values (B, L_group, D)
            # value_slice = values[:, start:end, :]
            
            # # C. 矩阵乘法：加权求和
            # # (B, L_total, L_group) @ (B, L_group, D) -> (B, L_total, D)
            # # 含义：该组特征实际上向 Residual Stream 注入了多少更新向量
            # weighted_update = torch.bmm(attn_slice, value_slice)
            
            # D. 计算能量 (L2 Norm)
            # (B, L_total) -> 每个位置收到的来自该组的更新强度
            update_norm = torch.norm(attn_slice, p=2, dim=-1)
            
            # E. Mask 掉 Padding 位置 (我们只关心有效 Token 收到的贡献)
            update_norm = update_norm * global_mask
            
            # F. 平均化：得到该样本中，该组的平均贡献强度
            avg_energy = update_norm.mean(dim=1) #  / num_valid_queries.squeeze(-1) # (B,)
            
            group_energy_list.append(avg_energy)

        # 5. 堆叠与归一化 (转为比例)
        # 结果 shape: (B, Num_Groups)
        group_energies = torch.stack(group_energy_list, dim=1)
        
        # 计算总能量，归一化成 0~1 的比例，方便和之前的 Attention Score 对比
        total_energy = group_energies.sum(dim=1, keepdim=True)
        contribution_ratios = group_energies / torch.clamp(total_energy, min=1)

        # 6. 保存到 JSONL
        batch_ratios = contribution_ratios.detach().cpu().tolist()
        
        try:
            with open(save_path, 'a', encoding='utf-8') as f:
                for sample_ratios in batch_ratios:
                    # 格式: [0.85, 0.10, 0.05]
                    f.write(json.dumps(sample_ratios) + "\n")
        except Exception as e:
            print(f"Warning: Failed to save contribution scores: {e}")
            
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
    
    def gradcam_analyse(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
    ):
        # 路径检查省略...
        if hasattr(self.args, 'gradcam_save_path') and self.args.gradcam_save_path:
            save_path = self.args.gradcam_save_path
        elif hasattr(self.args, 'view_groups_attention_path') and self.args.view_groups_attention_path:
            save_path = self.args.view_groups_attention_path.replace('.jsonl', '_gradcam.jsonl')
        else:
            return 
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 强制开启梯度上下文
        with torch.enable_grad():
            self.zero_grad()
            
            # [调试] 打印原始输入状态
            # print(f"[Debug GradCAM] Original embedding 0 grad: {embeddings[0].requires_grad}")

            # -------------------------------------------------
            # 关键：构建带梯度的新输入
            # -------------------------------------------------
            inputs_with_grad = []
            for emb in embeddings:
                # 必须 detach 出来建立新图
                new_emb = emb.detach().clone().requires_grad_(True)
                inputs_with_grad.append(new_emb)


            # [调试] 确认输入确实开启了梯度
            # print(f"[Debug GradCAM] New Input 0 requires_grad: {inputs_with_grad[0].requires_grad} (Should be True)")

            # -------------------------------------------------
            # 运行 Forward
            # -------------------------------------------------
            try:
                outputs = self.forward(
                    inputs_with_grad,      # 必须传入新的列表
                    masks, 
                    embeddings_groups, 
                    groups_relationships, 
                    analysis_mode=True
                )
            except RuntimeError as e:
                print(f"[Critical Error in Forward]: {e}")
                return

            fused_emb = outputs['fused_embedding']
            
            # [调试] 检查输出是否有梯度
            if not fused_emb.requires_grad:
                print("[Fatal Error] fused_embedding lost gradients! Check modules (e.g. frozen weights?).")
                return

            # 定义目标：L2 Norm
            target_score = torch.norm(fused_emb, p=2, dim=1).sum()
            
            # 反向传播
            target_score.backward()

            # -------------------------------------------------
            # 计算重要性
            # -------------------------------------------------
            batch_group_scores = [] 
            
            for i, feat in enumerate(self.captured_group_feats):
                grad = feat.grad 
                mask = self.captured_group_masks[i]

                if grad is None:
                    # 如果打印了这个，说明 retain_grad 成功了，但是 backward 没传回来
                    # 这通常意味着 feat 没有参与 target_score 的计算
                    print(f"[Warning] Group {i} grad is None. (Did not participate in fusion?)")
                    grad = torch.zeros_like(feat)
                
                # HiResCAM Logic
                weighted_map = (feat * grad).sum(dim=-1)
                importance_map = F.relu(weighted_map)
                
                # Masking & Pooling
                mask_float = mask.float()
                importance_map = importance_map * mask_float
                valid_token_counts = mask_float.sum(dim=1).clamp(min=1.0)
                avg_group_importance = importance_map.sum(dim=1) / valid_token_counts
                
                batch_group_scores.append(avg_group_importance.detach().cpu())

            # 保存逻辑 (同之前)...
            if batch_group_scores:
                all_scores_tensor = torch.stack(batch_group_scores, dim=1)
                row_sums = all_scores_tensor.sum(dim=1, keepdim=True)
                contribution_ratios = all_scores_tensor / torch.clamp(row_sums, min=1e-9)
                
                # [调试] 打印第一个样本的比例，看看是不是还是全0
                # print(f"[Debug Result] Sample 0 Ratios: {contribution_ratios[0].tolist()}")

                batch_ratios_list = contribution_ratios.tolist()
                import json
                import math
                try:
                    with open(save_path, 'a', encoding='utf-8') as f:
                        for row in batch_ratios_list:
                            clean_row = [0.0 if (math.isnan(x) or math.isinf(x)) else x for x in row]
                            f.write(json.dumps(clean_row) + "\n")
                except Exception as e:
                    print(f"Save Error: {e}")

        self.captured_group_feats = []
        self.captured_group_masks = []
        self.zero_grad()