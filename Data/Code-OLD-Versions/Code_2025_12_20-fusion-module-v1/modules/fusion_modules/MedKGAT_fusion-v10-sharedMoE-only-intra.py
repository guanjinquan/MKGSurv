# still 0.66
import sys
import os
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from modules.base_modules.aggregation_utils import masked_mean_pool
import json
import random


# --- 基础组件 ---
class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
    

class ExpertFeedForward(nn.Module):
    """
    Standard FFN used as a single Expert.
    """
    def __init__(self, dim, mult = 4, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MoEFeedForward(nn.Module):
    """
    Shared Mixture-of-Experts (MoE) Layer.
    
    Structure:
    - Shared Expert: Always active for all tokens. Serves as a backbone.
    - Sparse Experts: Top-K routing.
    
    Output = Shared_Expert(x) + Weighted_Sum(Routed_Experts(x))
    """
    def __init__(self, dim, num_experts=4, num_selected=2, mult=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        
        # Gating Network
        self.gate = nn.Linear(dim, num_experts)
        
        # --- Shared Expert ---
        # Acts as a backbone. Always active for all tokens.
        # This reduces overfitting by ensuring a shared representation across all inputs.
        self.shared_expert = ExpertFeedForward(dim, mult=mult, dropout=dropout)
        
        # --- Routed Experts ---
        # Reduce 'mult' for routed experts to half size to control parameter count
        # while keeping Shared Expert at full size.
        expert_mult = max(2, mult // 2) 
        
        self.experts = nn.ModuleList([
            ExpertFeedForward(dim, mult=expert_mult, dropout=dropout) 
            for _ in range(num_experts)
        ])
        
        self.register_buffer('aux_loss', torch.tensor(0.0))
        
    def forward(self, x):
        """
        x: (Batch, Seq_Len, Dim)
        """
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim) # (B*S, D)
        
        # --- 1. Shared Expert Execution (Always Active) ---
        # This provides a stable gradient and baseline performance
        shared_out = self.shared_expert(x_flat)
        final_output = shared_out # Start with shared output as base
        
        # --- 2. Gating ---
        gate_logits = self.gate(x_flat)
        
        if self.training:
            # Noisy gating for regularization
            noise = torch.randn_like(gate_logits) * (1.0 / self.num_experts)
            gate_logits = gate_logits + noise
            
        probs = F.softmax(gate_logits, dim=-1)
        
        # selected_probs: (N, K), selected_indices: (N, K)
        selected_probs, selected_indices = torch.topk(probs, self.num_selected, dim=-1)
        
        # Normalize weights so they sum to 1 for the selected experts
        selected_probs = selected_probs / selected_probs.sum(dim=-1, keepdim=True)
        
        # --- 3. Aux Loss (Load Balancing) ---
        # We use a simple mean-square auxiliary loss
        mean_probs = probs.mean(dim=0)
        aux_loss = (self.num_experts * (mean_probs ** 2).sum())
        self.aux_loss = aux_loss
        
        # --- 4. Vectorized Dispatch Loop ---
        # Loop over experts to batch computations
        for i in range(self.num_experts):
            # Create a mask for tokens that selected this expert
            # (N, K) == scalar -> (N, K) -> any(dim=1) -> (N,)
            selection_mask_k = (selected_indices == i)
            token_mask = selection_mask_k.any(dim=1) 
            
            if token_mask.any():
                # 1. Gather inputs: Extract tokens assigned to expert i
                expert_input = x_flat[token_mask]
                
                # 2. Run Expert
                expert_out = self.experts[i](expert_input)
                
                # 3. Apply Routing Weights
                # We need the weight corresponding to this expert for each selected token.
                weight_chunk = selected_probs[token_mask]       # (M, K)
                mask_chunk = selection_mask_k[token_mask].float() # (M, K)
                
                # Sum over K to get the single weight scalar for this expert per token
                # (M, 1)
                specific_weights = (weight_chunk * mask_chunk).sum(dim=1, keepdim=True)
                
                # 4. Scatter Add (Accumulate)
                final_output[token_mask] += expert_out * specific_weights

        return final_output.view(batch_size, seq_len, dim)


# --- SafeCrossAttnEncoder ---
class SafeCrossAttnEncoder(nn.Module):
    """
    升级版交叉注意力模块。
    结构: CrossAttention -> Add & Norm -> FFN/MoE -> Add & Norm
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4, use_moe: bool = False, use_ffn: bool = True):
        super().__init__() 
        self.norm_q = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 1. Attention 部分
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 2. FFN 部分 (Conditional MoE)
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm_ffn = nn.LayerNorm(embed_dim)
            if use_moe:
                # Intra-step usually benefits from higher capacity to digest information within group
                self.ffn = MoEFeedForward(
                    embed_dim, 
                    num_experts=4,     
                    num_selected=2,    
                    mult=ffn_mult, 
                    dropout=dropout
                )
            else:
                # Standard FFN for Inter-step and Global aggregation to keep stability
                self.ffn = ExpertFeedForward(dim=embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False) -> torch.Tensor:
        """
        query: (B, Lq, D)
        key:   (B, Lk, D)
        value: (B, Lk, D)
        key_padding_mask: (B, Lk), True 为 padding
        """

        query = self.norm_q(query)
        
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
            all_masked_rows = None  

        # --- 1. Attention Block ---
        attn_out, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights)
            
        # 清理垃圾值
        if all_masked_rows is not None and all_masked_rows.any():
            attn_out[all_masked_rows] = 0.0

        # Residual + Norm
        x = query + self.dropout(attn_out)
        
        # --- 2. FFN Block (Standard or MoE) ---
        if self.use_ffn:
            ffn_out = self.ffn(self.norm_ffn(x))
            x = x + self.dropout(ffn_out)

        if need_weights:
            return x, attn_weights
        return x


class EdgeContextualizer(nn.Module):
    """
    使用Edge作为Query，连接的节点特征作为Key/Value。
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        # Edge update DOES NOT use MoE
        self.cross_attn = SafeCrossAttnEncoder(embed_dim, num_heads, use_moe=False)

    def forward(self, edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                node_i: torch.Tensor, node_i_mask: torch.Tensor,
                node_j: torch.Tensor, node_j_mask: torch.Tensor) -> torch.Tensor:
        
        context_feat = torch.cat([node_i, node_j], dim=1)
        context_mask_raw = torch.cat([node_i_mask, node_j_mask], dim=1)
        key_padding_mask = (context_mask_raw == 0)

        updated_edge = self.cross_attn(query=edge_feat, key=context_feat, value=context_feat, 
                                      key_padding_mask=key_padding_mask)
        
        if edge_mask is not None:
            updated_edge = updated_edge * edge_mask.unsqueeze(-1).type_as(updated_edge)
        
        return updated_edge
    

class MedKGATFusion(nn.Module):
    def __init__(self, args, embed_dim: int, 
             max_modalities: int = 10, 
             max_groups: int = 10, 
             ff_dropout_rate: float = 0.25, 
             attn_dropout_rate: float = 0.1, 
             num_intra_layers: int = 1, num_inter_layers: int = 1):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.drop_edge_ratio = 0.1
        self.group_drop_ratio = 0.25
        self.fusion_loss_weight = nn.Parameter(torch.tensor(1.0))
        self.moe_loss_weight = 0.01 

        # 1. Knowledge Projection
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(ff_dropout_rate)
        )

        # 2. Intra-group Interaction -> USE MOE HERE
        self.num_intra_layers = num_intra_layers
        self.intra_group_transformer = nn.ModuleList([
            SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate, use_moe=True)
            for _ in range(num_intra_layers)
        ])

        # 3. GAT Interaction Components (Inter-Group) -> NO MOE HERE
        self.num_inter_layers = num_inter_layers
        self.shared_inter_layer = nn.ModuleDict({
            'edge_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate, use_moe=False, use_ffn=False),
            'node_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate, use_moe=False, use_ffn=False),
            'edge_updater': EdgeContextualizer(embed_dim, num_heads=8) # internally False
        })

        # 4. Global Aggregation -> NO MOE HERE
        self.global_transformer = SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate, use_moe=False)

        # 5. Post Fusion Norm
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def compute_moe_aux_loss(self) -> torch.Tensor:
        """
        Recursively find all MoEFeedForward layers and sum their aux_losses.
        """
        total_aux_loss = []
        # Iterate over all modules to find MoE layers
        for module in self.modules():
            if isinstance(module, MoEFeedForward):
                total_aux_loss.append(module.aux_loss)
        
        if len(total_aux_loss) == 0:
            return torch.tensor(0.0, device=self.fusion_loss_weight.device)
            
        # mean of aux loss
        total_aux_loss = torch.stack(total_aux_loss, dim=0)
        return torch.mean(total_aux_loss)

    def _intra_group_step(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], groups: List[List[int]]) -> List[torch.Tensor]: 
        updated_embeddings = list(embeddings)
        
        for group_idx, group_indices in enumerate(groups):
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
            for i in range(self.num_intra_layers):
                concat_feat = self.intra_group_transformer[i](
                    query=concat_feat, 
                    key=concat_feat, 
                    value=concat_feat, 
                    key_padding_mask=padding_mask
                )

                if all_masked_rows.any():
                    concat_feat[all_masked_rows] = 0.0

            split_feats = torch.split(concat_feat, lengths, dim=1)
            
            for i, idx in enumerate(group_indices):
                updated_embeddings[idx] = split_feats[i]
                
        return updated_embeddings

    def _inter_group_step(self, target_node: torch.Tensor, target_mask: torch.Tensor,
                          source_node: torch.Tensor, source_mask: torch.Tensor,
                          edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                          layer_modules: nn.ModuleDict) -> torch.Tensor:
        """
        One-way interaction: Source -> Edge -> Target
        """
        source_padding_mask = (source_mask == 0)
        edge_padding_mask = (edge_mask == 0)
        
        # Step 1: Edge queries Source
        gated_source = layer_modules['edge_to_node_attn'](
            query=edge_feat, 
            key=source_node, 
            value=source_node, 
            key_padding_mask=source_padding_mask
        )
        
        # Step 2: Target queries Gated Source
        updated_target = layer_modules['node_to_node_attn'](
            query=target_node,
            key=gated_source,
            value=gated_source,
            key_padding_mask=edge_padding_mask
        )
        
        if target_mask is not None:
            updated_target = updated_target * target_mask.unsqueeze(-1).type_as(updated_target)

        return updated_target
    
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
            if (j, i) in fusion_knowledge and i < j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        edge_keys = list(fusion_knowledge.keys())
        if self.training:
            random.shuffle(edge_keys)

        # 1. Project Knowledge Edges
        current_proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            current_proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction (Uses MOE)
        info_level_embeddings = self._intra_group_step(embeddings, masks, embeddings_groups)

        # 3. Create Group-Level Embeddings
        group_embeddings = []
        group_masks = []
    
        for group_indices in embeddings_groups:
            if not group_indices:
                raise ValueError("Empty group found in embeddings_groups")

            curr_feats = [info_level_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]

            g_feat = torch.cat(curr_feats, dim=1)
            g_mask = torch.cat(curr_masks, dim=1)

            group_embeddings.append(g_feat)
            group_masks.append(g_mask)

        # Pre-calculate validity masks
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # 4. Inter-Group Interaction (Standard Attention, NO MOE)
        current_group_embeddings = group_embeddings

        for layer_idx in range(self.num_inter_layers):
            layer_modules = self.shared_inter_layer
            
            for (idx_a, idx_b) in edge_keys:
                edge_feat = current_proj_knowledge.get((idx_a, idx_b))

                if self.training and getattr(self, 'drop_edge_ratio', 0.0) > 0.0:
                    if random.random() < self.drop_edge_ratio:
                        continue
                
                edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                if edge_mask is None:
                    edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a]
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # Update Edge Features
                updated_edge_feat = layer_modules['edge_updater'](
                    edge_feat, edge_mask, 
                    feat_a, mask_a, 
                    feat_b, mask_b
                )
                
                current_proj_knowledge[(idx_a, idx_b)] = updated_edge_feat

                # Update Node B
                update_for_b = self._inter_group_step(
                    target_node=feat_b, target_mask=mask_b,
                    source_node=feat_a, source_mask=mask_a,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_b] = update_for_b

                # Update Node A
                update_for_a = self._inter_group_step(
                    target_node=feat_a, target_mask=mask_a,
                    source_node=feat_b, source_mask=mask_b,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_a] = update_for_a

        final_group_embeddings = current_group_embeddings

        # ------------------------------------------------------------------
        # 4b. Data Collection for KL Loss
        # ------------------------------------------------------------------
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_edge_pairs_list = []
        edge_score_valid_flag = False

        for (idx_a, idx_b) in edge_keys:
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

        # 7. Global Aggregation (Standard Attention, NO MOE)
        current_group_masks = list(group_masks) 

        if self.training and self.group_drop_ratio > 0.0:
            batch_size = embeddings[0].shape[0]
            num_groups = len(final_group_embeddings)
            drop_decision = torch.rand((batch_size, num_groups), device=embeddings[0].device) < self.group_drop_ratio
            all_dropped = drop_decision.all(dim=1)
            if all_dropped.any():
                indices_to_keep = torch.randint(0, num_groups, (all_dropped.sum(),), device=embeddings[0].device)
                dropped_rows = torch.where(all_dropped)[0]
                drop_decision[dropped_rows, indices_to_keep] = False
            for g_idx in range(num_groups):
                should_drop = drop_decision[:, g_idx].unsqueeze(1) 
                keep_factor = (~should_drop).float()
                current_group_masks[g_idx] = current_group_masks[g_idx] * keep_factor

        global_concat = torch.cat(final_group_embeddings, dim=1)
        global_mask = torch.cat(current_group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        global_transformed = self.global_transformer(
            query=global_concat, key=global_concat, value=global_concat, key_padding_mask=global_padding_mask, need_weights=False)
        
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
    
        # 9. Compute MoE Auxiliary Loss (Load Balancing)
        moe_loss = self.compute_moe_aux_loss()

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": (self.fusion_loss_weight * fusion_loss - F.tanh(self.fusion_loss_weight)) + (self.moe_loss_weight * moe_loss),
                "KL_loss": fusion_loss,
                "MoE_loss": moe_loss
            }
        }