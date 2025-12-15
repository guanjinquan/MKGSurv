"""
LUAD
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.5873 ± 0.0481
 - List = [0.6277195809830781, 0.49410840534171246, 0.5918045705279747, 0.6057007125890737, 0.6171516079632465]
C-Index-IPCW_Validation Set: 0.5937 ± 0.0471
 - List = [0.566275189637892, 0.5195030244346615, 0.6067995843732945, 0.619124373260572, 0.6570393849128838]
Test Summary:
C-Index_Test Set: 0.5879 ± 0.0525
 - List = [0.6572327044025157, 0.5967966573816156, 0.6071428571428571, 0.5826388888888889, 0.4957397366382649]
C-Index-IPCW_Test Set: 0.5667 ± 0.0473
 - List = [0.581508163981889, 0.6314717251745374, 0.5954533515529293, 0.5187402918386006, 0.5063810379642368]
Training run tcga_luad_run001 finished.

LUSC
--- Testing Complete ---
Validation Summary:
C-Index_Validation Set: 0.6498 ± 0.0228
 - List = [0.6282201405152225, 0.6922642574816488, 0.6477644492911668, 0.631810676940011, 0.6491410723581468]
C-Index-IPCW_Validation Set: 0.6246 ± 0.0344
 - List = [0.6367343469545722, 0.6867298576162301, 0.5928714476773442, 0.6031543874547485, 0.6032664324791667]
Test Summary:
C-Index_Test Set: 0.6096 ± 0.0327
 - List = [0.5469273743016759, 0.609860664523044, 0.6225165562913907, 0.6335164835164835, 0.63542300669507]
C-Index-IPCW_Test Set: 0.6254 ± 0.0255
 - List = [0.5899388043537807, 0.6124077727068789, 0.6568145425885024, 0.6526091997255505, 0.6152532763973515]
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


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class AttentionPooling(nn.Module):
    """
    替换 MeanPool，使用 Attention 机制聚合组内特征。
    能够自动识别组内哪些 token 更重要。
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # x: (B, L, D)
        # mask: (B, L)
        scores = self.attn(x).squeeze(-1) # (B, L)
        
        # 处理 Mask: 将无效位置设为极小值
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=1) # (B, L)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted Sum
        # (B, 1, L) @ (B, L, D) -> (B, 1, D) -> (B, D)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        
        # 生成 validity mask
        if mask is not None:
            valid_mask = (mask.sum(dim=1) > 0).float()
            # 如果某行全是 padding，pooling 结果置零
            pooled = pooled * valid_mask.unsqueeze(-1)
        else:
            valid_mask = torch.ones(x.shape[0], device=x.device)
            
        return pooled, valid_mask

class SafeCrossAttnEncoder(nn.Module):
    """
    保留原本的安全 Attention Encoder，用于 Intra-Group 和 Global 交互。
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        if key_padding_mask is not None:
            all_masked_rows = key_padding_mask.all(dim=1)
            if all_masked_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked_rows, 0] = False
        else:
            all_masked_rows = None

        attn_out, _ = self.mha(query, key, value, key_padding_mask=key_padding_mask)
        
        if all_masked_rows is not None and all_masked_rows.any():
            attn_out[all_masked_rows] = 0.0

        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class GatedKnowledgeInteraction(nn.Module):
    """
    [核心修改] 轻量级门控交互层。
    替代了原本复杂的 3 个 Transformer 循环。
    原理：Target = Target + Gate * Projection(Source)
    其中 Gate 是由 (Source, Target, Edge) 共同决定的。
    """
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 门控网络：输入 Source+Target+Edge，输出 0-1 的系数
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid() 
        )
        
        # 信息变换网络
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, target_node, source_node, edge_feat):
        """
        target_node: (B, D)
        source_node: (B, D)
        edge_feat: (B, D)
        """
        # 1. 计算门控系数 (B, D)
        # 融合三者信息来决定有多少知识需要传递
        concat_feat = torch.cat([target_node, source_node, edge_feat], dim=-1)
        gate = self.gate_net(concat_feat)
        
        # 2. 变换源节点信息
        source_transformed = self.proj(source_node)
        
        # 3. 加权更新 (B, D)
        # Message = Gate * Source_Transformed
        message = gate * source_transformed
        
        # 4. 残差连接 + Norm (原 Target + Message)
        out = self.norm(target_node + self.dropout(message))
        
        return out


class MedKGATFusion(nn.Module):
    def __init__(self, args, embed_dim: int, 
             max_modalities: int = 10, 
             max_groups: int = 10, 
             ff_dropout_rate: float = 0.1, 
             attn_dropout_rate: float = 0.1, 
             num_intra_layers: int = 1, 
             num_inter_layers: int = 2): # 建议 layers=2
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.drop_edge_ratio = 0.1 # 训练时随机丢弃边以防止过拟合

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(ff_dropout_rate)
        )

        # 2. Intra-group Interaction (保持 Transformer，适合序列建模)
        self.intra_group_transformer = nn.ModuleList([
            SafeCrossAttnEncoder(embed_dim, num_heads=4, dropout=attn_dropout_rate, ffn_mult=2) # 减小 heads 和 ffn_mult
            for _ in range(num_intra_layers)
        ])

        # 3. Pooling (升级为 Attention Pooling)
        self.group_pooling = AttentionPooling(embed_dim, dropout=attn_dropout_rate)

        # 4. Inter-Group Interaction (升级为 Gated Mechanism)
        self.num_inter_layers = num_inter_layers
        self.inter_group_layers = nn.ModuleList([
            GatedKnowledgeInteraction(embed_dim, dropout=attn_dropout_rate)
            for _ in range(num_inter_layers)
        ])
        
        # 5. Global Aggregation
        self.global_transformer = SafeCrossAttnEncoder(embed_dim, num_heads=4, dropout=attn_dropout_rate)
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

        # 6. Learnable Temperature for Loss (让模型自适应调整分布的尖锐度)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.3025) # 初始化为 ln(10) ~= 2.3

    def _intra_group_step(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], groups: List[List[int]]) -> List[torch.Tensor]: 
        # 简单的浅拷贝
        updated_embeddings = list(embeddings)
        
        for group_indices in groups:
            if not group_indices: continue
            
            # Gather
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            lengths = [f.shape[1] for f in group_feats]
            
            concat_feat = torch.cat(group_feats, dim=1)
            concat_mask = torch.cat(group_masks, dim=1)
            padding_mask = (concat_mask == 0)
            
            # Transformer Process
            for layer in self.intra_group_transformer:
                concat_feat = layer(concat_feat, concat_feat, concat_feat, key_padding_mask=padding_mask)
            
            # Scatter back
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
        
        # 0. 清理对称键
        for (i, j), v in list(fusion_knowledge.items()): 
            if (j, i) in fusion_knowledge and i < j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        edge_keys = list(fusion_knowledge.keys())
        if self.training:
            random.shuffle(edge_keys) # 随机化处理顺序，增强鲁棒性

        # 1. 投影 Knowledge (Static Edges)
        # 我们这里不再动态更新 Edge，因为 Edge 本身是先验知识，
        # 保持 Edge 稳定有助于防止 Overfitting，让模型专注于更新 Node。
        projected_edges = {}
        for k, v in fusion_knowledge.items():
            projected_edges[k] = self.know_proj(v)

        # 2. 组内交互 (Intra-Group)
        # 让模态内部先充分融合 (例如: 文字和图片在同一组)
        info_level_embeddings = self._intra_group_step(embeddings, masks, embeddings_groups)

        # 3. 生成组级别 Embedding (Attention Pooling)
        group_embeddings = [] # List of (B, D)
        group_validity_masks = [] # List of (B,)
        
        for group_indices in embeddings_groups:
            curr_feats = [info_level_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            
            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)
            
            # 使用 Attention Pooling 替代 Mean Pool
            pooled_emb, valid_mask = self.group_pooling(g_feat, g_mask)
            
            group_embeddings.append(pooled_emb)
            group_validity_masks.append(valid_mask)

        # 将 List 转为 Tensor 以便批量处理 (Num_Groups, B, D) -> (B, Num_Groups, D)
        # 假设所有 batch 的 group 数量一致（由 max_groups 控制或 padding）
        # 这里为了通用性保持 List 操作，或者手动 stack
        
        # 4. 组间交互 (Inter-Group GAT)
        # [Residual Connection Strategy]: Node_new = Node_old + GAT_Update
        
        current_group_embeddings = group_embeddings # 引用
        
        for layer_idx in range(self.num_inter_layers):
            interaction_layer = self.inter_group_layers[layer_idx]
            
            # 使用 temp buffer 存储这一层的更新，避免顺序依赖（模拟同步更新）
            next_step_embeddings = [e.clone() for e in current_group_embeddings]
            
            for (idx_a, idx_b) in edge_keys:
                # Drop Edge Augmentation
                if self.training and self.drop_edge_ratio > 0.0 and random.random() < self.drop_edge_ratio:
                    continue

                edge_feat = projected_edges[(idx_a, idx_b)] # (B, D)
                # 处理 edge mask: 这里简化，假设 edge feat 已经是 pooled 或 [CLS]
                # 如果 edge_feat 是 (B, L, D)，需要先 pool 成 (B, D)
                if edge_feat.dim() == 3:
                    edge_m = fusion_knowledge_mask.get((idx_a, idx_b))
                    edge_feat, _ = masked_mean_pool(edge_feat, edge_m)

                feat_a = current_group_embeddings[idx_a]
                feat_b = current_group_embeddings[idx_b]
                
                # 双向更新: A <-> B (基于 Edge)
                # Update B based on A
                # 注意：这里我们累加更新量，类似于 GCN 的聚合
                update_b = interaction_layer(target_node=feat_b, source_node=feat_a, edge_feat=edge_feat)
                # 因为加入了 Residual，interaction_layer 输出的是完整的 emb，我们取差值累加或者直接替换
                # 更好的方式是：Aggregation -> Update。这里简化为直接迭代更新。
                # 采用移动平均或者直接替换: 这里选择替换，因为 GatedInteraction 内部已经含有了残差
                
                next_step_embeddings[idx_b] = update_b
                
                # Update A based on B
                update_a = interaction_layer(target_node=feat_a, source_node=feat_b, edge_feat=edge_feat)
                next_step_embeddings[idx_a] = update_a

            current_group_embeddings = next_step_embeddings

        final_group_embeddings = current_group_embeddings

        # 5. Global Aggregation
        # 将更新后的组 Embedding 拼起来做最后一次交互
        # Stack: (B, Num_Groups, D)
        global_seq = torch.stack(final_group_embeddings, dim=1) 
        
        # 构建 Global Mask
        # (B, Num_Groups)
        global_mask_tensor = torch.stack(group_validity_masks, dim=1)
        global_padding_mask = (global_mask_tensor == 0)

        # Transformer
        # (B, Num_Groups, D)
        global_out = self.global_transformer(
            query=global_seq, key=global_seq, value=global_seq, 
            key_padding_mask=global_padding_mask
        )
        
        # Final Pool (Mean pool on Transformer output is standard)
        fused_embedding, _ = masked_mean_pool(global_out, global_mask_tensor)
        fused_embedding = self.post_fusion_norm(fused_embedding)

        # --- Loss Calculation & Data Collection ---
        # 重新计算相似度用于 Loss，逻辑与原版一致，但增加 Temperature 缩放
        
        loss_dict = self._compute_kl_loss(
            final_group_embeddings, 
            group_validity_masks, 
            edge_keys, 
            groups_relationships
        )
        
        # Save Points Logic (保持原样逻辑，仅调用)
        self.save_points(final_group_embeddings, group_validity_masks, groups_relationships)

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": loss_dict
        }

    def _compute_kl_loss(self, embeddings, masks, edge_keys, relationships):
        all_edge_scores = []
        all_sims = []
        all_valid_masks = []
        
        # Clamp logit scale to max 4.6 (temp min ~0.01) to prevent instability
        logit_scale = torch.clamp(self.logit_scale, max=4.6052)
        temperature = torch.exp(-logit_scale) # 学习到的温度系数

        has_valid_data = False
        
        for (idx_a, idx_b) in edge_keys:
            # GT Score
            score = relationships.get((idx_a, idx_b), relationships.get((idx_b, idx_a), None))
            if score is None: continue
            
            if score.dim() > 1: score = score.view(-1)
            if score.dim() == 0: score = score.expand(embeddings[0].shape[0])
            
            if score.sum() > 0: has_valid_data = True

            # Cosine Sim
            ea = F.normalize(embeddings[idx_a], p=2, dim=1)
            eb = F.normalize(embeddings[idx_b], p=2, dim=1)
            sim = torch.sum(ea * eb, dim=1) # (B,)
            
            # Valid Mask
            valid = masks[idx_a] * masks[idx_b]
            
            all_edge_scores.append(score)
            all_sims.append(sim)
            all_valid_masks.append(valid)
            
        if not has_valid_data or len(all_edge_scores) == 0:
            return {"total_loss": torch.tensor(0.0, device=embeddings[0].device)}

        # Stack
        scores_stack = torch.stack(all_edge_scores, dim=1) # (B, E)
        sims_stack = torch.stack(all_sims, dim=1) # (B, E)
        masks_stack = torch.stack(all_valid_masks, dim=1) # (B, E)

        # Masking
        # GT Distribution
        scores_masked = scores_stack.clone().float() # Ensure float for Softmax
        scores_masked[masks_stack == 0] = -1e9
        target_probs = F.softmax(scores_masked, dim=1)

        # Pred Distribution (Scaled by Learnable Temp)
        sims_masked = sims_stack / temperature
        sims_masked[masks_stack == 0] = -1e9
        pred_log_probs = F.log_softmax(sims_masked, dim=1)

        # KL Div
        kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none') # (B, E)
        kl_loss_sum = kl_loss.sum(dim=1) # (B,)
        
        valid_patients = (masks_stack.sum(dim=1) > 0).float()
        final_loss = (kl_loss_sum * valid_patients).sum() / (valid_patients.sum() + 1e-9)
        
        return {"total_loss": 2 * final_loss} # 保持原权重的 scale

    def save_points(self, final_group_embeddings, final_group_masks, groups_relationships):
        # ... (保持原本的 save_points 代码逻辑，完全不需要变动) ...
        # 注意：这里 final_group_embeddings 已经是 pooled 过的 (B, D)，
        # 所以原代码里的 masked_mean_pool 这一步可以省去，直接用。
        if self.args.points_save_path is None: return 

        batch_size = final_group_embeddings[0].shape[0]
        device = final_group_embeddings[0].device
        sum_edge_scores = torch.zeros((batch_size, 1), device=device)
        sum_cos_sims = torch.zeros((batch_size, 1), device=device)
        raw_data_cache = {} 
        valid_pairs = []

        # ... (逻辑相同，只需注意 embeddings 已经是向量了) ...
        for (idx_a, idx_b), _ in groups_relationships.items():
            raw_score = groups_relationships.get((idx_a, idx_b), groups_relationships.get((idx_b, idx_a), None))
            if raw_score is not None:
                if raw_score.dim() == 1: raw_score = raw_score.view(-1, 1)
                
                # 直接使用，不需要再次 pool
                embed_a = final_group_embeddings[idx_a]
                embed_b = final_group_embeddings[idx_b]
                
                raw_cos = torch.cosine_similarity(embed_a, embed_b, dim=1).view(-1, 1)
                raw_cos_positive = torch.clamp(raw_cos, min=1e-9) 

                sum_edge_scores += raw_score
                sum_cos_sims += raw_cos_positive
                
                raw_data_cache[(idx_a, idx_b)] = (raw_cos_positive, raw_score)
                valid_pairs.append((idx_a, idx_b))
        
        # ... (后续写入文件的代码保持不变) ...
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