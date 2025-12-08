"""

LUAD:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6307 ± 0.0652
 - List = [0.6470588235294118, 0.5074626865671642, 0.6312056737588653, 0.6745843230403801, 0.6929555895865237]
C-Index-IPCW_Validation Set: 0.6229 ± 0.0604
 - List = [0.6021030734880771, 0.518100187731403, 0.6321993306989561, 0.6837300260181681, 0.6782786214835385]
Test Summary:
C-Index_Test Set: 0.6218 ± 0.0407
 - List = [0.6119496855345912, 0.6831476323119777, 0.6145552560646901, 0.6409722222222223, 0.5584817970565453]
C-Index-IPCW_Test Set: 0.6089 ± 0.0618
 - List = [0.5320618740943621, 0.7170017075489766, 0.5913601301537448, 0.6251545108441059, 0.5787189729633913]
Training run tcga_luad_run001 finished.


LUSC:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6678 ± 0.0453
 - List = [0.6715456674473068, 0.6408808582721626, 0.722464558342421, 0.597138139790864, 0.706923477355544]
C-Index-IPCW_Validation Set: 0.6362 ± 0.0426
 - List = [0.6803442956291568, 0.6242147296891702, 0.621099973597898, 0.5703087559394675, 0.6852344066969189]
Test Summary:
C-Index_Test Set: 0.6347 ± 0.0309
 - List = [0.6162011173184357, 0.6709539121114684, 0.6062149770759042, 0.6065934065934065, 0.6737674984783932]
C-Index-IPCW_Test Set: 0.6609 ± 0.0218
 - List = [0.6276121395565804, 0.6782813940147256, 0.6795898876608397, 0.6769565403620537, 0.6419328811124999]
Training run tcga_lusc_run001 finished.

"""
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

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import json
import random

# 假设 aggregation_utils 依然可用，如果不可用请替换为标准实现
from modules.base_modules.aggregation_utils import masked_mean_pool

# --- 基础组件 ---

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample.
    From timm library.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


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
            GELU(),  # ReLU之后要跟LayerNorm，但是GeLU之后本身就是高斯分布，不需要再归一化
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# --- 改进版 Attention 组件 ---
class PreNormSafeCrossAttn(nn.Module):
    """
    Pre-Norm 结构的 Cross Attention。
    相比 Post-Norm，在深层网络中训练更稳定，更不易过拟合。
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mult=2, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Pre-Norm: 先 Norm 再 Attention
        q_norm = self.norm_q(query)
        k_norm = self.norm_kv(key)
        v_norm = self.norm_kv(value)
        
        # --- Safe Mask Logic ---
        if key_padding_mask is not None:
            all_masked_rows = key_padding_mask.all(dim=1)
            if all_masked_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked_rows, 0] = False
        else:
            all_masked_rows = None

        attn_out, _ = self.mha(q_norm, k_norm, v_norm, key_padding_mask=key_padding_mask)
        
        if all_masked_rows is not None and all_masked_rows.any():
            attn_out[all_masked_rows] = 0.0

        # Residual 1 with DropPath
        x = query + self.drop_path(attn_out)
        
        # Residual 2 (FFN) with Pre-Norm
        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        
        return x

class GatedInteractionLayer(nn.Module):
    """
    用于 GAT 交互的门控层。
    不像 Transformer 那样直接相加，而是学习一个 Gate 来融合信息。
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, original, update):
        # original: (B, L, D), update: (B, L, D)
        gate = self.gate_net(torch.cat([original, update], dim=-1))
        return self.norm(original * (1 - gate) + update * gate)

class AttentionPooling(nn.Module):
    """
    替代 Global Mean Pooling。
    使用一个可学习的 Query 来聚合所有 Group 的信息。
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # Learnable Summary Token
        self.attn = PreNormSafeCrossAttn(embed_dim, num_heads, dropout=dropout)
        
    def forward(self, context, context_mask):
        # context: (B, Total_L, D)
        B = context.shape[0]
        # 扩展 query 到 batch size
        query = self.query_token.expand(B, -1, -1)
        
        padding_mask = (context_mask == 0)
        
        # Query 关注 Context
        out = self.attn(query, context, context, key_padding_mask=padding_mask)
        return out.squeeze(1) # (B, D)

# --- 主模型 ---

class MedKGATFusion(nn.Module):
    def __init__(self, args, embed_dim: int, 
             max_modalities: int = 10, 
             max_groups: int = 10, 
             ff_dropout_rate: float = 0.2,     # 稍微增加 dropout
             attn_dropout_rate: float = 0.1, 
             num_intra_layers: int = 1, 
             num_inter_layers: int = 1):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        # 增加 Edge Dropout 概率，防止过拟合
        self.drop_edge_ratio = 0.2 if num_inter_layers > 1 else 0.1

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),

            nn.Linear(self.embed_dim, self.embed_dim * 2),
            GELU(),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(ff_dropout_rate)
        )

        # 2. Intra-group Interaction
        # 使用 ModuleList 允许不同层参数不同，或者共享参数（此处独立）
        self.num_intra_layers = num_intra_layers
        self.intra_group_transformer = nn.ModuleList([
            PreNormSafeCrossAttn(embed_dim, num_heads=8, dropout=attn_dropout_rate, drop_path_rate=0.1)
            for _ in range(num_intra_layers)
        ])

        # 3. GAT Interaction Components (Inter-Group)
        self.num_inter_layers = num_inter_layers
        
        # 共享权重以减少参数
        self.edge_updater_attn = PreNormSafeCrossAttn(embed_dim, num_heads=4, dropout=attn_dropout_rate)
        self.node_updater_attn = PreNormSafeCrossAttn(embed_dim, num_heads=4, dropout=attn_dropout_rate)
        
        # 门控融合，比直接残差相加更适合图传播
        self.edge_gate = GatedInteractionLayer(embed_dim)
        self.node_gate = GatedInteractionLayer(embed_dim)

        # 4. Global Aggregation (Attention Pooling)
        self.global_pool = AttentionPooling(embed_dim, num_heads=8, dropout=attn_dropout_rate)

        # 5. Final Norm
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def _intra_group_step(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], groups: List[List[int]]) -> List[torch.Tensor]: 
        updated_embeddings = list(embeddings)
        
        for group_indices in groups:
            if not group_indices:
                continue
                
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            
            lengths = [f.shape[1] for f in group_feats]
            
            concat_feat = torch.cat(group_feats, dim=1)
            concat_mask = torch.cat(group_masks, dim=1)
            padding_mask = (concat_mask == 0) 
            
            # Safe Check
            all_masked_rows = padding_mask.all(dim=1)
            if all_masked_rows.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_masked_rows, 0] = False

            # Intra Transformer
            for layer in self.intra_group_transformer:
                # Self-Attention: Q=K=V=concat_feat
                concat_feat = layer(concat_feat, concat_feat, concat_feat, key_padding_mask=padding_mask)

            if all_masked_rows.any():
                concat_feat[all_masked_rows] = 0.0

            split_feats = torch.split(concat_feat, lengths, dim=1)
            
            for i, idx in enumerate(group_indices):
                updated_embeddings[idx] = split_feats[i]
                
        return updated_embeddings

    def forward(
        self, 
        embeddings: List[torch.Tensor], 
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        # 0. Clean symmetric keys
        for (i, j), v in list(fusion_knowledge.items()): 
            if (j, i) in fusion_knowledge and i < j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        edge_keys = list(fusion_knowledge.keys())
        # Shuffle edges during training for robustness
        if self.training:
            random.shuffle(edge_keys)

        # 1. Project Knowledge Edges
        current_proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            current_proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction
        # (B, L_i, D) -> Intra -> (B, L_i, D)
        embeddings = self._intra_group_step(embeddings, masks, embeddings_groups)

        # 3. Prepare Group Embeddings
        # Flatten intra-group feats to one tensor per group
        group_embeddings = []
        group_masks = []
    
        for group_indices in embeddings_groups:
            curr_feats = [embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            group_embeddings.append(torch.cat(curr_feats, dim=1))
            group_masks.append(torch.cat(curr_masks, dim=1))

        # Pre-calc validity for loss
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # 4. Inter-Group Interaction (GAT with Global Residual)
        
        # 保存进入 GAT 之前的状态作为全局残差
        initial_group_embeddings = [g.clone() for g in group_embeddings]
        current_group_embeddings = group_embeddings

        for layer_idx in range(self.num_inter_layers):
            
            # Temporary storage for updates in this layer (Synchronous update)
            next_group_embeddings = [g.clone() for g in current_group_embeddings]
            # Accumulate updates for nodes involved in multiple edges
            node_updates_buffer = {i: [] for i in range(len(current_group_embeddings))}

            for (idx_a, idx_b) in edge_keys:
                edge_feat = current_proj_knowledge.get((idx_a, idx_b))
                edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                
                # Drop Edge during training
                if self.training and self.drop_edge_ratio > 0.0:
                    if random.random() < self.drop_edge_ratio:
                        continue
                
                if edge_mask is None:
                    edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)
                
                edge_padding_mask = (edge_mask == 0)

                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a]
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # --- Interaction Logic ---
                
                # A. Update Edge: Edge queries [Node A, Node B]
                context_feat = torch.cat([feat_a, feat_b], dim=1)
                context_mask = torch.cat([mask_a, mask_b], dim=1)
                context_padding_mask = (context_mask == 0)
                
                # Attn
                edge_update = self.edge_updater_attn(edge_feat, context_feat, context_feat, key_padding_mask=context_padding_mask)
                # Gated Fuse
                edge_feat_new = self.edge_gate(edge_feat, edge_update)
                
                # Apply Mask
                edge_feat_new = edge_feat_new * edge_mask.unsqueeze(-1).type_as(edge_feat_new)
                current_proj_knowledge[(idx_a, idx_b)] = edge_feat_new # Update edge immediately for next layer

                # B. Update Nodes: Node queries [Updated Edge]
                # 这里做简化：Node 只去 attend Edge，因为 Edge 已经聚合了另一个 Node 的信息
                
                # Update A from Edge
                update_a = self.node_updater_attn(feat_a, edge_feat_new, edge_feat_new, key_padding_mask=edge_padding_mask)
                node_updates_buffer[idx_a].append(update_a)
                
                # Update B from Edge
                update_b = self.node_updater_attn(feat_b, edge_feat_new, edge_feat_new, key_padding_mask=edge_padding_mask)
                node_updates_buffer[idx_b].append(update_b)

            # Aggregate updates and apply Gating for Nodes
            for idx, updates in node_updates_buffer.items():
                if updates:
                    # Mean pool of all updates from different edges
                    aggregated_update = torch.stack(updates).mean(dim=0)
                    # Gated Fuse with previous state
                    next_group_embeddings[idx] = self.node_gate(current_group_embeddings[idx], aggregated_update)
            
            current_group_embeddings = next_group_embeddings

        # Apply Global Residual (Skip Connection from Init to Final)
        # 这对于防止深层 GNN 的 Oversmoothing 非常重要
        final_group_embeddings = []
        for init, curr in zip(initial_group_embeddings, current_group_embeddings):
            final_group_embeddings.append(init + curr) # Simple Additive Residual

        # --- Loss Calculation Prep ---
        # (This logic is kept largely same to match requirements, but ensures safety)
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_cos_sims_list = []
        edge_score_valid_flag = False

        if len(edge_keys) > 0:
            # Pooling for Cosine Sim
            final_pooled_groups = []
            for g, m in zip(final_group_embeddings, group_masks):
                pooled, _ = masked_mean_pool(g, m)
                final_pooled_groups.append(F.normalize(pooled, p=2, dim=1))

            for (idx_a, idx_b) in edge_keys:
                edge_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
                if edge_score is None: continue

                if edge_score.dim() > 1: edge_score = edge_score.view(-1)
                if edge_score.dim() == 0: edge_score = edge_score.expand(embeddings[0].shape[0])

                has_a = group_validity_masks[idx_a].float()
                has_b = group_validity_masks[idx_b].float()
                pair_validity = has_a * has_b

                if pair_validity.sum() > 0:
                    edge_score_valid_flag = True

                all_edge_scores_list.append(edge_score)
                all_valid_masks_list.append(pair_validity)

                # Cos Sim
                sim = torch.sum(final_pooled_groups[idx_a] * final_pooled_groups[idx_b], dim=1)
                all_cos_sims_list.append(torch.clamp(sim, -1.0, 1.0))

        # Save Points Logic (保持原样接口)
        self.save_points(final_group_embeddings, group_masks, groups_relationships)

        # 5. Global Aggregation (Replaced with Attention Pooling)
        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        
        # 使用 Attention Pooling 而不是 Mean Pooling
        fused_embedding = self.global_pool(global_concat, global_mask)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # 6. Loss Calculation
        fusion_loss = torch.tensor(0.0, device=fused_embedding.device)
        
        if len(all_edge_scores_list) > 0 and edge_score_valid_flag:
            all_scores_tensor = torch.stack(all_edge_scores_list, dim=1)
            all_sims_tensor = torch.stack(all_cos_sims_list, dim=1)
            all_masks_tensor = torch.stack(all_valid_masks_list, dim=1)

            scores_masked = all_scores_tensor.clone().float()
            scores_masked[all_masks_tensor == 0] = -1e9
            target_probs = F.softmax(scores_masked, dim=1)

            # Add temperature scaling for better gradient flow
            temperature = 0.1
            sims_masked = all_sims_tensor.clone() / temperature
            sims_masked[all_masks_tensor == 0] = -1e9
            
            pred_log_probs = F.log_softmax(sims_masked, dim=1)
            
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none')
            kl_loss_per_patient = kl_loss.sum(dim=1)
            
            valid_patients = (all_masks_tensor.sum(dim=1) > 0).float() # Fixed logic: >0 is enough for softmax
            
            if valid_patients.sum() > 0:
                fusion_loss = (kl_loss_per_patient * valid_patients).sum() / (valid_patients.sum() + 1e-6)

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": 2.0 * fusion_loss, 
            }
        }    
    
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