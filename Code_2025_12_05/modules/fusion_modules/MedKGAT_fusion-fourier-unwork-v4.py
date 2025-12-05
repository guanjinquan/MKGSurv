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
import math
from einops import rearrange, repeat



# --- Helper Function: Fourier Encoding (From HealNet) ---
def fourier_encode(x, max_freq, num_bands=4):
    """
    Applies Fourier positional encoding to the input tensor.
    x: Input tensor of shape (..., input_dim) or sequence positions.
       We assume x represents positions in [-1, 1].
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    # Create frequency bands
    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    # Calculate Fourier features: [sin(x*w), cos(x*w), ...]
    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    
    # Concatenate original input/position with fourier features
    # Shape becomes: ..., (num_bands * 2 + 1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


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
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# --- SafeCrossAttnEncoder ---
class SafeCrossAttnEncoder(nn.Module):
    """
    升级版交叉注意力模块。
    结构: CrossAttention -> Add & Norm -> FeedForward -> Add & Norm
    包含了防 NaN 的安全机制。
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 1. Attention 部分
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 2. FFN 部分 (新增逻辑)
        self.ffn = FeedForward(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        query: (B, Lq, D)
        key:   (B, Lk, D)
        value: (B, Lk, D)
        key_padding_mask: (B, Lk), True 为 padding
        """
        
        # --- 核心修复逻辑 (Safe Logic) ---
        if key_padding_mask is not None:
            # 检测哪些样本的所有 Key 都是 Padding
            all_masked_rows = key_padding_mask.all(dim=1) # (B,) bool

            if all_masked_rows.any():
                # 只有当存在全 Mask 的情况时，才进行克隆和修改
                key_padding_mask = key_padding_mask.clone()
                # 将全 Mask 行的第一个位置设为 False (有效)，防止 Softmax NaN
                key_padding_mask[all_masked_rows, 0] = False
        else:
            all_masked_rows = torch.zeros(query.size(0), dtype=torch.bool, device=query.device)

        # --- 1. Attention Block ---
        # 正常计算 MHA
        attn_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        
        # 清理垃圾值：将那些原本全无效的行的输出置为 0
        if all_masked_rows.any():
             attn_out[all_masked_rows] = 0.0

        # Residual + Norm (Post-Norm 风格)
        x = self.norm(query + self.dropout(attn_out))
        
        # --- 2. FFN Block (新增逻辑) ---
        ffn_out = self.ffn(x)
        
        # 如果 Query 本身有无效行（例如全是 padding），FFN 可能会产生非零偏差
        # 但通常 Query Mask 由外部控制，或者在下一步会被 mask 掉，这里暂不做额外 mask 处理
        
        # Residual + Norm
        x = self.norm(x + self.dropout(ffn_out))

        return x


class EdgeContextualizer(nn.Module):
    """
    使用Edge作为Query，连接的节点特征作为Key/Value。
    让知识(Edge)根据具体的病人数据(Node)进行动态调整。
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.cross_attn = SafeCrossAttnEncoder(embed_dim, num_heads)

    def forward(self, edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                node_i: torch.Tensor, node_i_mask: torch.Tensor,
                node_j: torch.Tensor, node_j_mask: torch.Tensor) -> torch.Tensor:
        
        # 1. 拼接两个模态的特征作为上下文 (B, Ni+Nj, D)
        context_feat = torch.cat([node_i, node_j], dim=1)
        
        # 2. 拼接Mask (B, Ni+Nj)
        # 注意：输入的mask是1有效0无效，MHA通常需要True为无效(padding)
        # 这里先拼接原始mask (1有效)
        context_mask_raw = torch.cat([node_i_mask, node_j_mask], dim=1)
        
        # 转换为MHA需要的格式: True为Padding(无效), False为有效
        key_padding_mask = (context_mask_raw == 0)

        # 3. Edge更新: Edge query Context
        # Edge mask自身不需要传入attn mask，因为它是query，长度不变，padding位置的输出后续会被mask掉或忽略
        updated_edge = self.cross_attn(query=edge_feat, key=context_feat, value=context_feat, 
                                     key_padding_mask=key_padding_mask)
        
        # 4. Apply Edge Mask: 确保无效的 Edge Token 输出保持为 0
        # updated_edge: (B, Le, D), edge_mask: (B, Le)
        if edge_mask is not None:
            updated_edge = updated_edge * edge_mask.unsqueeze(-1).type_as(updated_edge)
        
        return updated_edge
    



# --- Main Class: MedKGATFusion with Fourier Encoding ---

class MedKGATFusion(nn.Module):
    def __init__(self, 
            args, 
            embed_dim: int, 
            max_modalities: int = 10, 
            dropout_rate: float = 0.1, 
            num_intra_layers: int = 1, 
            num_inter_layers: int = 1,
            # Fourier Args
            fourier_encode_data: bool = True,
            num_freq_bands: int = 6,
            max_freq: float = 10.):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.num_inter_layers = num_inter_layers
        
        # Fourier Encoding Parameters
        self.fourier_encode_data = fourier_encode_data
        self.num_freq_bands = num_freq_bands
        self.max_freq = max_freq
        
        # Calculate Fourier Output Dimension: (num_freq_bands * 2) + 1 (original)
        self.fourier_dim = (num_freq_bands * 2) + 1 if fourier_encode_data else 0

        # 1. Knowledge Projection (768 -> embed_dim)
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(self.dropout_rate)
        )

        # 1.5 Fourier Adapters
        # Since Fourier features are concatenated, the dimension increases. 
        # We need to project it back to embed_dim for the Transformer.
        # We create a list of adapters, one for each potential modality index.
        if self.fourier_encode_data:
            self.fourier_adapters = nn.Sequential(
                nn.Linear(self.embed_dim + self.fourier_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.GELU()
            )

        # 2. Intra-group Interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            activation=F.gelu,
            batch_first=True,
            dropout=dropout_rate
        )
        self.intra_group_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_intra_layers)

        # 3. GAT Interaction Components (Inter-Group)
        self.inter_layers = nn.ModuleList()
        for _ in range(num_inter_layers):
            self.inter_layers.append(nn.ModuleDict({
                'edge_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8),
                'node_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8),
                'edge_updater': EdgeContextualizer(embed_dim, num_heads=8)
            }))

        # 4. Global Aggregation
        global_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            activation=F.gelu,
            batch_first=True,
            dropout=dropout_rate
        )
        self.global_transformer = nn.TransformerEncoder(global_encoder_layer, num_layers=1)

        # 5. Post Fusion Norm
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
            
            padding_mask = (concat_mask == 0) # True is invalid
            
            # Safe Transformer Check
            all_masked_rows = padding_mask.all(dim=1)
            if all_masked_rows.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_masked_rows, 0] = False

            # The TransformerEncoder handles num_layers internally
            transformed = self.intra_group_transformer(concat_feat, src_key_padding_mask=padding_mask)
            
            if all_masked_rows.any():
                transformed[all_masked_rows] = 0.0

            split_feats = torch.split(transformed, lengths, dim=1)
            
            for i, idx in enumerate(group_indices):
                updated_embeddings[idx] = split_feats[i]
                
        return updated_embeddings

    def _inter_group_step(self, target_node: torch.Tensor, target_mask: torch.Tensor,
                          source_node: torch.Tensor, source_mask: torch.Tensor,
                          edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                          layer_modules: nn.ModuleDict) -> torch.Tensor:
        """
        One-way interaction: Source -> Edge -> Target
        Updated to take layer_modules dict
        """
        source_padding_mask = (source_mask == 0)
        edge_padding_mask = (edge_mask == 0)
        
        # Step 1: Edge queries Source to get relevant info (Gating)
        gated_source = layer_modules['edge_to_node_attn'](
            query=edge_feat, 
            key=source_node, 
            value=source_node, 
            key_padding_mask=source_padding_mask
        )
        
        # Step 2: Target queries Gated Source to update itself
        # Note: Key/Value mask depends on Edge because gated_source has shape of Edge
        updated_target = layer_modules['node_to_node_attn'](
            query=target_node,
            key=gated_source,
            value=gated_source,
            key_padding_mask=edge_padding_mask
        )
        
        if target_mask is not None:
            updated_target = updated_target * target_mask.unsqueeze(-1).type_as(updated_target)

        return updated_target
    
    def _apply_fourier_encoding(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor]) -> List[torch.Tensor]:
        encoded_embeddings = list(embeddings)
        
        if self.fourier_encode_data:
            for i, emb in enumerate(encoded_embeddings):
                b, seq_len, _ = emb.shape
                device, dtype = emb.device, emb.dtype
                
                # 1. Create Linear Positions [-1, 1]
                pos = torch.linspace(-1., 1., steps=seq_len, device=device, dtype=dtype)
                
                # 2. Encode Positions
                # enc_pos shape: (seq_len, fourier_dim)
                enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
                
                # 3. Expand to Batch Size
                # enc_pos: (b, seq_len, fourier_dim)
                enc_pos = repeat(enc_pos, 's d -> b s d', b=b)
                
                # 4. Concatenate with original embeddings
                # emb shape: (b, s, embed_dim + fourier_dim)
                emb_with_pos = torch.cat((emb, enc_pos), dim=-1)
                
                # 5. Project back to embed_dim using modality-specific adapter
                projected_emb = self.fourier_adapters(emb_with_pos)

                # 6. Apply Mask to zero out padding positions
                # If mask is 0, the position is meaningless, so we zero it out to remove 
                # any noise from fourier encoding/projection.
                if i < len(masks) and masks[i] is not None:
                    # masks[i] shape is usually (Batch, SeqLen)
                    # We need to unsqueeze to broadcast: (Batch, SeqLen, 1)
                    curr_mask = masks[i].unsqueeze(-1).type_as(projected_emb)
                    projected_emb = projected_emb * curr_mask
                
                encoded_embeddings[i] = projected_emb

        return encoded_embeddings
    
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
            if (j, i) in fusion_knowledge and i > j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        # 1. Project Knowledge Edges
        current_proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            current_proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction 
        info_level_embeddings = self._intra_group_step(embeddings, masks, embeddings_groups)

        # 3. Create Group-Level Embeddings
        group_embeddings = []
        group_masks = []
    
        for group_indices in embeddings_groups:
            if not group_indices:
                # raise ValueError("Empty group found in embeddings_groups")
                # Handle empty groups gracefully if necessary or skip
                continue

            curr_feats = [info_level_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]

            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)

            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # Apply Fourier Encoding
        group_embeddings = self._apply_fourier_encoding(group_embeddings, group_masks)

        # Pre-calculate validity masks for Weights (based on INPUT embeddings/masks)
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # 4. Inter-Group Interaction (Multi-Layer GNN / GAT)
        current_group_embeddings = group_embeddings 

        for layer_idx in range(self.num_inter_layers):
            layer_modules = self.inter_layers[layer_idx]
            num_groups = len(current_group_embeddings)
            group_updates_buffer = [[] for _ in range(num_groups)]
            
            next_proj_knowledge = {} 
            
            for (idx_a, idx_b), edge_feat in current_proj_knowledge.items():
                
                edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                if edge_mask is None:
                    edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

                # Get Group Data
                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a] 
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # Validity Checks
                has_a = group_validity_masks[idx_a].float()
                has_b = group_validity_masks[idx_b].float()
                weight_for_b = has_a.view(-1, 1, 1) 
                weight_for_a = has_b.view(-1, 1, 1) 

                # --- GNN Update Logic ---
                updated_edge_feat = layer_modules['edge_updater'](
                    edge_feat, edge_mask, 
                    feat_a, mask_a, 
                    feat_b, mask_b
                )
                
                next_proj_knowledge[(idx_a, idx_b)] = updated_edge_feat

                # Update Node B 
                update_for_b = self._inter_group_step(
                    target_node=feat_b, target_mask=mask_b,
                    source_node=feat_a, source_mask=mask_a,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                group_updates_buffer[idx_b].append((update_for_b, weight_for_b))

                # Update Node A
                update_for_a = self._inter_group_step(
                    target_node=feat_a, target_mask=mask_a,
                    source_node=feat_b, source_mask=mask_b,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                group_updates_buffer[idx_a].append((update_for_a, weight_for_a))

            # Apply Updates
            next_group_embeddings = []
            for i in range(num_groups):
                original_group_feat = current_group_embeddings[i]
                updates_and_weights = group_updates_buffer[i]

                if len(updates_and_weights) > 0:
                    updates = [u for u, w in updates_and_weights]   
                    weights = [w for u, w in updates_and_weights]   

                    stacked_updates = torch.stack(updates, dim=0)
                    stacked_weights = torch.stack(weights, dim=0)
                    
                    sum_updates = torch.sum(stacked_updates * stacked_weights, dim=0)
                    sum_counts = torch.sum(stacked_weights, dim=0)
                    
                    aggregated_feat = torch.where(
                        sum_counts > 0,
                        sum_updates / sum_counts.clamp(min=1e-9),
                        original_group_feat
                    )
                    next_group_embeddings.append(aggregated_feat)
                else:
                    next_group_embeddings.append(original_group_feat)
            
            current_group_embeddings = next_group_embeddings
            current_proj_knowledge = next_proj_knowledge

        final_group_embeddings = current_group_embeddings

        # ------------------------------------------------------------------
        # 4b. Data Collection for KL Loss
        # ------------------------------------------------------------------
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_edge_pairs_list = []
        edge_score_valid_flag = False

        for (idx_a, idx_b) in current_proj_knowledge.keys():
            
            edge_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            if edge_score is None:
                edge_score = torch.zeros(embeddings[0].shape[0], device=embeddings[0].device)
            
            if edge_score.dim() > 1:
                edge_score = edge_score.view(-1)
            if edge_score.dim() == 0:
                edge_score = edge_score.expand(embeddings[0].shape[0])

            has_a = group_validity_masks[idx_a].float()
            has_b = group_validity_masks[idx_b].float()
            pair_validity = has_a * has_b

            edge_score_valid_flag |= edge_score.sum().item() > 0
            all_edge_scores_list.append(edge_score)
            all_valid_masks_list.append(pair_validity)
            all_edge_pairs_list.append((idx_a, idx_b))

        # 6. Compute Similarities on FINAL Embeddings 
        all_cos_sims_list = []
        
        if len(all_edge_pairs_list) > 0 and edge_score_valid_flag:
            final_pooled_results = [masked_mean_pool(g, m) for g, m in zip(final_group_embeddings, group_masks)]
            final_pooled_group_embeddings = [res[0] for res in final_pooled_results]
            final_pooled_group_embeddings = [F.normalize(g, p=2, dim=1) for g in final_pooled_group_embeddings]
            
            for idx_a, idx_b in all_edge_pairs_list:
                sim = torch.sum(final_pooled_group_embeddings[idx_a] * final_pooled_group_embeddings[idx_b], dim=1)
                sim = torch.clamp(sim, -1.0, 1.0)
                all_cos_sims_list.append(sim)

        # Save Points
        self.save_points(final_group_embeddings, group_masks, groups_relationships)

        # 7. Global Aggregation
        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        global_transformed = self.global_transformer(global_concat, src_key_padding_mask=global_padding_mask)
        
        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        fused_embedding, _ = masked_mean_pool(global_transformed, global_mask)
             
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # 8. Compute KL Divergence Loss
        fusion_loss = torch.tensor(0.0, device=fused_embedding.device)
        
        if len(all_edge_scores_list) > 0 and edge_score_valid_flag:
            all_scores_tensor = torch.stack(all_edge_scores_list, dim=1)
            all_sims_tensor = torch.stack(all_cos_sims_list, dim=1)
            all_masks_tensor = torch.stack(all_valid_masks_list, dim=1) 

            scores_masked = all_scores_tensor.clone().float()
            scores_masked[all_masks_tensor == 0] = -1e9
            target_probs = F.softmax(scores_masked, dim=1)

            temperature = 0.1 
            sims_masked = all_sims_tensor.clone() / temperature
            sims_masked[all_masks_tensor == 0] = -1e9
            
            pred_log_probs = F.log_softmax(sims_masked, dim=1)
            
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none')
            kl_loss_per_patient = kl_loss.sum(dim=1)
            
            valid_patients = (all_masks_tensor.sum(dim=1) > 1).float()
            
            if valid_patients.sum() > 0:
                fusion_loss = (kl_loss_per_patient * valid_patients).sum() / valid_patients.sum()
            
        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": fusion_loss
            }
        }

    def save_points(self, final_group_embeddings, final_group_masks, groups_relationships):
        if not hasattr(self.args, 'points_save_path') or self.args.points_save_path is None:
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