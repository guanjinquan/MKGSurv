"""
LUAD:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6197 ± 0.0407
 - List = [0.5898468976631749, 0.5569520816967792, 0.6304176516942475, 0.66270783847981, 0.6584992343032159]
C-Index-IPCW_Validation Set: 0.5955 ± 0.0691
 - List = [0.5322252869279261, 0.5087898663111493, 0.5996876835537229, 0.6389012293398817, 0.6976634427969028]
Test Summary:
C-Index_Test Set: 0.6211 ± 0.0508
 - List = [0.660377358490566, 0.685933147632312, 0.621967654986523, 0.5972222222222222, 0.5398915569326104]
C-Index-IPCW_Test Set: 0.6035 ± 0.0567
 - List = [0.6123382359444144, 0.697998264109131, 0.6159077576853468, 0.535587093514896, 0.5556119797657599]
Training run tcga_luad_run001 finished.


LUSC:
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6406 ± 0.0242
 - List = [0.6633489461358314, 0.6640316205533597, 0.6363140676117776, 0.5976884975233901, 0.6418532014575742]
C-Index-IPCW_Validation Set: 0.6159 ± 0.0482
 - List = [0.6699032717470398, 0.6673456595293706, 0.6163888483822326, 0.5784379696957523, 0.5475636786609986]
Test Summary:
C-Index_Test Set: 0.6194 ± 0.0455
 - List = [0.5452513966480447, 0.597534833869239, 0.6459500764136525, 0.6291208791208791, 0.6792452830188679]
C-Index-IPCW_Test Set: 0.6325 ± 0.0413
 - List = [0.5761448894658118, 0.5894558004374028, 0.6740963611056202, 0.6677889137011686, 0.655020431176266]
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

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SimpleGatedGCNLayer(nn.Module):
    """
    [极简主义设计] 门控图卷积层 (Vector-Level Gated GCN)
    
    跳出 Transformer 的 Attention 机制。
    直接在 Vector 层面进行基于 Edge 的特征门控。
    
    公式:
    Update = Linear(Source) * Sigmoid(Linear(Edge))
    Target = Target + Update
    
    参数量极少，归纳偏置极强（Edge 显式控制信息流），非常适合防止过拟合。
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        # Source 变换
        self.src_proj = nn.Linear(embed_dim, embed_dim)
        
        # Edge 门控生成器
        self.edge_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid() 
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout=dropout)
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, target, source, edge_feat):
        """
        输入都是已经 Pooling 过的向量: (B, D)
        """
        # 1. Message Passing
        # 源节点特征变换
        src_feat = self.src_proj(source) # (B, D)
        
        # Edge 生成门控系数 (0~1)
        # 含义：Edge 认为 Source 的哪些特征对 Target 是有用的？
        gate = self.edge_gate(edge_feat) # (B, D)
        
        # 加权消息
        message = src_feat * gate
        
        # 2. Residual Update
        # 类似于 ResNet Block
        out = self.norm(target + self.dropout(message))
        
        # 3. FFN (Point-wise MLP)
        out = self.norm_ffn(out + self.ffn(out))
        
        return out

class SelfAttnEncoder(nn.Module):
    """
    标准的 Self-Attention 用于组内特征提取 (保持不变，处理序列信息)
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4, 
            dropout=dropout, 
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        src_key_padding_mask = (mask == 0) if mask is not None else None
        if src_key_padding_mask is not None:
             all_masked = src_key_padding_mask.all(dim=1)
             if all_masked.any():
                 src_key_padding_mask = src_key_padding_mask.clone()
                 src_key_padding_mask[all_masked, 0] = False
        
        out = self.layer(x, src_key_padding_mask=src_key_padding_mask)
        
        if src_key_padding_mask is not None and src_key_padding_mask.all(dim=1).any():
            all_masked = src_key_padding_mask.all(dim=1)
            out[all_masked] = 0.0
            
        return self.norm(out)


class MedKGATFusion(nn.Module):
    def __init__(self, args, embed_dim: int, 
             max_modalities: int = 10, 
             max_groups: int = 10, 
             ff_dropout_rate: float = 0.1, 
             attn_dropout_rate: float = 0.1, 
             num_intra_layers: int = 1, 
             num_inter_layers: int = 1): 
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        # 极简模式下，Dropout 可以稍微低一点，或者保持标准
        self.drop_edge_ratio = 0.1 

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # 2. Intra-Group Refinement (Seq -> Seq)
        # 组内还是需要处理序列信息的，这里保留 Transformer 是合理的
        self.intra_group_layers = nn.ModuleList([
            SelfAttnEncoder(embed_dim, num_heads=4, dropout=attn_dropout_rate)
            for _ in range(num_intra_layers)
        ])

        # 3. Inter-Group Interaction (Vector -> Vector)
        # [核心改变] 放弃 Transformer，使用 Gated GCN
        self.num_inter_layers = num_inter_layers
        self.inter_layers = nn.ModuleList([
            SimpleGatedGCNLayer(embed_dim, dropout=attn_dropout_rate)
            for _ in range(num_inter_layers)
        ])
        
        # 4. Global Fusion
        # 最后简单的聚合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Learnable Temp
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.3025)

    def _intra_group_process(self, embeddings, masks, groups):
        updated_embeddings = list(embeddings)
        for group_indices in groups:
            if not group_indices: continue
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            lengths = [f.shape[1] for f in group_feats]
            
            concat_feat = torch.cat(group_feats, dim=1)
            concat_mask = torch.cat(group_masks, dim=1)
            
            for layer in self.intra_group_layers:
                concat_feat = layer(concat_feat, mask=concat_mask)
            
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
        
        # 0. Data Prep
        for (i, j), v in list(fusion_knowledge.items()): 
            if (j, i) in fusion_knowledge and i < j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))
        edge_keys = list(fusion_knowledge.keys())
        if self.training: random.shuffle(edge_keys)

        # 1. Project Edge Knowledge
        projected_edges = {k: self.know_proj(v) for k, v in fusion_knowledge.items()}

        # 2. Intra-Group Processing (处理序列)
        processed_embeddings = self._intra_group_process(embeddings, masks, embeddings_groups)

        # 3. [关键步骤] Early Pooling (序列 -> 向量)
        # 在进入复杂的组间交互之前，先把每个组变成一个向量。
        # 这极大地减少了噪声，强制模型关注全局语义。
        group_vecs = [] 
        group_validity_list = []
        
        for group_indices in embeddings_groups:
            curr_feats = [processed_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            
            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)
            
            # Pool immediately!
            pooled, valid = masked_mean_pool(g_feat, g_mask)
            group_vecs.append(pooled) # (B, D)
            group_validity_list.append(valid) # (B,)

        # 4. Inter-Group Interaction (Gated GCN)
        # 现在的操作对象是向量列表，而非序列
        current_group_vecs = group_vecs
        
        for layer_idx in range(self.num_inter_layers):
            gcn_layer = self.inter_layers[layer_idx]
            
            next_step_vecs = [v.clone() for v in current_group_vecs]
            node_updates = {i: [] for i in range(len(group_vecs))}
            
            for (idx_a, idx_b) in edge_keys:
                if self.training and self.drop_edge_ratio > 0.0 and random.random() < self.drop_edge_ratio:
                    continue

                edge_feat = projected_edges[(idx_a, idx_b)]
                # 如果 Edge 是序列，也 Pool 成向量
                if edge_feat.dim() == 3:
                    e_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                    edge_feat, _ = masked_mean_pool(edge_feat, e_mask)
                
                vec_a = current_group_vecs[idx_a]
                vec_b = current_group_vecs[idx_b]
                
                # A 更新 B (使用 Gated GCN)
                # B = B + Linear(A) * Sigmoid(Edge)
                update_b = gcn_layer(target=vec_b, source=vec_a, edge_feat=edge_feat)
                node_updates[idx_b].append(update_b)
                
                # B 更新 A
                update_a = gcn_layer(target=vec_a, source=vec_b, edge_feat=edge_feat)
                node_updates[idx_a].append(update_a)
            
            # Aggregate Updates (Mean)
            for i in range(len(group_vecs)):
                updates = node_updates[i]
                if updates:
                    # 所有邻居发来的更新取平均
                    aggregated_update = torch.stack(updates, dim=0).mean(dim=0)
                    # 因为 GCN Layer 内部已经加了 Residual 和 Norm，这里直接赋值即可
                    # 但为了多邻居聚合的稳定性，我们通常在这里做替换
                    next_step_vecs[i] = aggregated_update
            
            current_group_vecs = next_step_vecs

        # 5. Global Fusion
        # current_group_vecs 已经是交互好的向量列表了
        
        # (B, Num_Groups, D)
        global_stack = torch.stack(current_group_vecs, dim=1)
        global_mask_tensor = torch.stack(group_validity_list, dim=1)
        
        # 简单的加权平均融合，不再使用 Transformer
        fused_embedding, _ = masked_mean_pool(global_stack, global_mask_tensor)
        fused_embedding = self.fusion_mlp(fused_embedding)

        # --- Loss Calculation ---
        # 使用 GCN 更新后的向量计算 Loss
        loss_dict = self._compute_kl_loss(
            current_group_vecs, 
            group_validity_list, 
            edge_keys, 
            groups_relationships
        )
        
        self.save_points(current_group_vecs, group_validity_list, groups_relationships)

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": loss_dict
        }

    def _compute_kl_loss(self, embeddings_list, masks_list, edge_keys, relationships):
        all_edge_scores = []
        all_sims = []
        all_valid_masks = []
        
        logit_scale = torch.clamp(self.logit_scale, max=4.6052)
        temperature = torch.exp(-logit_scale)

        has_valid_data = False
        
        for (idx_a, idx_b) in edge_keys:
            score = relationships.get((idx_a, idx_b), relationships.get((idx_b, idx_a), None))
            if score is None: continue
            if score.dim() > 1: score = score.view(-1)
            if score.dim() == 0: score = score.expand(embeddings_list[0].shape[0])
            if score.sum() > 0: has_valid_data = True

            # embeddings_list 已经是 (B, D)，直接计算
            ea = F.normalize(embeddings_list[idx_a], p=2, dim=1)
            eb = F.normalize(embeddings_list[idx_b], p=2, dim=1)
            sim = torch.sum(ea * eb, dim=1)
            
            valid = masks_list[idx_a] * masks_list[idx_b]
            
            all_edge_scores.append(score)
            all_sims.append(sim)
            all_valid_masks.append(valid)
            
        if not has_valid_data or len(all_edge_scores) == 0:
            return {"total_loss": torch.tensor(0.0, device=embeddings_list[0].device)}

        scores_stack = torch.stack(all_edge_scores, dim=1).float()
        sims_stack = torch.stack(all_sims, dim=1)
        masks_stack = torch.stack(all_valid_masks, dim=1)

        scores_masked = scores_stack.clone()
        scores_masked[masks_stack == 0] = -1e9
        target_probs = F.softmax(scores_masked, dim=1)

        sims_masked = sims_stack / temperature
        sims_masked[masks_stack == 0] = -1e9
        pred_log_probs = F.log_softmax(sims_masked, dim=1)

        kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none')
        kl_loss_sum = kl_loss.sum(dim=1)
        valid_patients = (masks_stack.sum(dim=1) > 0).float()
        final_loss = (kl_loss_sum * valid_patients).sum() / (valid_patients.sum() + 1e-9)
        
        return {"total_loss": 2 * final_loss}

    def save_points(self, final_group_embeddings, final_group_masks, groups_relationships):
        if self.args.points_save_path is None: return 
        batch_size = final_group_embeddings[0].shape[0]
        device = final_group_embeddings[0].device
        sum_edge_scores = torch.zeros((batch_size, 1), device=device)
        sum_cos_sims = torch.zeros((batch_size, 1), device=device)
        raw_data_cache = {} 
        valid_pairs = []

        for (idx_a, idx_b), _ in groups_relationships.items():
            raw_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            if raw_score is not None:
                if raw_score.dim() == 1: raw_score = raw_score.view(-1, 1)
                
                embed_a = final_group_embeddings[idx_a]
                embed_b = final_group_embeddings[idx_b]
                
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