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
    Sparse Mixture-of-Experts (MoE) Layer.
    To prevent overfitting (as requested):
    1. Uses Noisy Top-K Gating.
    2. Computes Load Balancing Loss to force diverse expert usage.
    3. Retains Dropout within experts.
    """
    def __init__(self, dim, num_experts=4, num_selected=2, mult=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        
        # Gating Network: Projects input to expert probabilities
        self.gate = nn.Linear(dim, num_experts)
        
        # Experts: A list of FeedForward networks
        self.experts = nn.ModuleList([
            ExpertFeedForward(dim, mult=mult, dropout=dropout) 
            for _ in range(num_experts)
        ])
        
        # Storage for auxiliary loss (Load Balancing)
        self.register_buffer('aux_loss', torch.tensor(0.0))
        
    def forward(self, x):
        """
        x: (Batch, Seq_Len, Dim)
        """
        # Save original shape
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim) # (B*S, D)
        
        # 1. Gating Score
        # Add noise during training for regularization (Noisy Gating)
        gate_logits = self.gate(x_flat) # (B*S, Num_Experts)
        
        if self.training:
            noise = torch.randn_like(gate_logits) * (1.0 / self.num_experts)
            gate_logits = gate_logits + noise
            
        probs = F.softmax(gate_logits, dim=-1) # (B*S, Num_Experts)
        
        # 2. Top-K Routing
        # selected_probs: (B*S, K), selected_indices: (B*S, K)
        selected_probs, selected_indices = torch.topk(probs, self.num_selected, dim=-1)
        
        # Normalize probabilities so they sum to 1 for the selected experts
        selected_probs = selected_probs / selected_probs.sum(dim=-1, keepdim=True)
        
        # 3. Compute Load Balancing Loss (Auxiliary Loss)
        # Importance: Sum of probabilities assigned to each expert
        importance = probs.sum(0) # (Num_Experts,)
        # Load: Count of how many times each expert was selected as Top-K (approximated by prob sum for differentiability or hard count)
        # Here we use the "switch transformer" style load balancing loss based on softmax probs
        # Loss = Num_Experts * sum(importance * load)
        
        # For simplicity and standard stability:
        # We want the distribution of routed tokens to be uniform.
        # Mean probability per expert across batch
        mean_probs = probs.mean(dim=0)
        # Mean routing frequency (this is hard selection)
        # We approximate load using the probs to keep it differentiable or just use mean_probs^2
        aux_loss = (self.num_experts * (mean_probs ** 2).sum())
        self.aux_loss = aux_loss
        
        # 4. Dispatch and Calculate
        # We will loop over experts. While less efficient than optimized CUDA kernels, 
        # it is robust and works on all devices for this scale.
        
        output = torch.zeros_like(x_flat)
        
        # Create a mask for each expert
        # selected_indices is (Total_Tokens, K)
        for i in range(self.num_experts):
            # Find tokens that selected expert 'i' as one of their Top-K
            # batch_mask: (Total_Tokens, K) -> Boolean
            expert_mask = (selected_indices == i)
            
            # If this expert is used by any token
            if expert_mask.any():
                # We need to process tokens that selected this expert.
                # Since a token might select multiple experts, we aggregate weighted outputs.
                
                # Get indices of tokens that selected expert i
                # We flatten the mask logic to simple indexing
                # But simple loop: iterate K
                pass

        # Optimized Dispatch Loop
        # Re-initialize output
        output = torch.zeros_like(x_flat)
        
        for k in range(self.num_selected):
            # Get the k-th selected expert for each token
            idx_k = selected_indices[:, k] # (Total_Tokens,)
            prob_k = selected_probs[:, k]  # (Total_Tokens,)
            
            for expert_idx in range(self.num_experts):
                # Mask for tokens routed to this specific expert at rank k
                mask = (idx_k == expert_idx)
                
                if mask.any():
                    # Extract inputs
                    inp = x_flat[mask]
                    # Process
                    expert_out = self.experts[expert_idx](inp)
                    # Weighted addition to output
                    # We utilize index_add_ or masked assignment
                    # Here masked assignment is safer for gradients in pure pytorch
                    output[mask] += expert_out * prob_k[mask].unsqueeze(-1)

        return output.view(batch_size, seq_len, dim)


# --- SafeCrossAttnEncoder ---
class SafeCrossAttnEncoder(nn.Module):
    """
    升级版交叉注意力模块。
    结构: CrossAttention -> Add & Norm -> MoE-FFN -> Add & Norm
    包含了防 NaN 的安全机制。
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 1. Attention 部分
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 2. FFN 部分 - Replaced with MoE
        # Using 4 experts, selecting top 2.
        self.ffn = MoEFeedForward(
            embed_dim, 
            num_experts=4,     # Can be increased to 8 if data allows
            num_selected=2,    # Standard Top-2 Routing
            mult=ffn_mult, 
            dropout=dropout
        )

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
        # 正常计算 MHA
        attn_out, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights)
            
        # 清理垃圾值：将那些原本全无效的行的输出置为 0
        if all_masked_rows is not None and all_masked_rows.any():
            attn_out[all_masked_rows] = 0.0

        # Residual + Norm (Post-Norm 风格)
        x = query + self.dropout(attn_out)
        
        # --- 2. FFN Block (MoE) ---
        # The MoE layer internally handles the aux_loss calculation and storage
        ffn_out = self.ffn(self.norm_ffn(x))

        x = x + self.dropout(ffn_out)

        if need_weights:
            return x, attn_weights
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
        # Weight for the MoE Load Balancing Loss
        self.moe_loss_weight = 0.01 

        # 1. Knowledge Projection (768 -> embed_dim)
        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(ff_dropout_rate)
        )

        # 2. Intra-group Interaction
        self.num_intra_layers = num_intra_layers
        self.intra_group_transformer = nn.ModuleList([
            SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate)
            for _ in range(num_intra_layers)
        ])

        # 3. GAT Interaction Components (Inter-Group)
        self.num_inter_layers = num_inter_layers
        self.shared_inter_layer = nn.ModuleDict({
            'edge_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate),
            'node_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate),
            'edge_updater': EdgeContextualizer(embed_dim, num_heads=8)
        })

        # 4. Global Aggregation
        self.global_transformer = SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate)

        # 5. Post Fusion Norm
        self.post_fusion_norm = nn.LayerNorm(embed_dim)

    def compute_moe_aux_loss(self) -> torch.Tensor:
        """
        Recursively find all MoEFeedForward layers and sum their aux_losses.
        """
        total_aux_loss = 0.0
        # Iterate over all modules to find MoE layers
        for module in self.modules():
            if isinstance(module, MoEFeedForward):
                total_aux_loss += module.aux_loss
        return total_aux_loss

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
        # This will be our initial edge state
        current_proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            current_proj_knowledge[k] = self.know_proj(v)

        # 2. Intra-Group Interaction (Multi-layer handled inside TransformerEncoder)
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

        # Pre-calculate validity masks for Weights (based on INPUT embeddings/masks)
        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            _, valid_mask = masked_mean_pool(g, m)
            group_validity_masks.append(valid_mask)

        # 4. Inter-Group Interaction (Multi-Layer GNN / GAT)
        # We loop self.num_inter_layers times
        
        current_group_embeddings = group_embeddings # Points to current node features

        for layer_idx in range(self.num_inter_layers):
            layer_modules = self.shared_inter_layer
            num_groups = len(current_group_embeddings)

            for (idx_a, idx_b) in edge_keys:
                edge_feat = current_proj_knowledge.get((idx_a, idx_b))

                if self.training and getattr(self, 'drop_edge_ratio', 0.0) > 0.0:
                    if random.random() < self.drop_edge_ratio:
                        continue
                
                edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                if edge_mask is None:
                    edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

                # Get Group Data
                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a] # Masks don't change
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # --- GNN Update Logic for this Layer ---
                # Update Edge Features
                updated_edge_feat = layer_modules['edge_updater'](
                    edge_feat, edge_mask, 
                    feat_a, mask_a, 
                    feat_b, mask_b
                )
                
                # Store updated edge for the next layer
                current_proj_knowledge[(idx_a, idx_b)] = updated_edge_feat

                # Update Node B using Node A and Edge
                update_for_b = self._inter_group_step(
                    target_node=feat_b, target_mask=mask_b,
                    source_node=feat_a, source_mask=mask_a,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_b] = update_for_b

                # Update Node A using Node B and Edge
                update_for_a = self._inter_group_step(
                    target_node=feat_a, target_mask=mask_a,
                    source_node=feat_b, source_mask=mask_b,
                    edge_feat=updated_edge_feat, edge_mask=edge_mask,
                    layer_modules=layer_modules
                )
                current_group_embeddings[idx_a] = update_for_a

        # Final embeddings after all GAT layers
        final_group_embeddings = current_group_embeddings

        # ------------------------------------------------------------------
        # 4b. Data Collection for KL Loss (Post-GAT)
        # We iterate the edges one last time (using the FINAL edge/node states) 
        # just to gather the lists needed for loss calculation.
        # ------------------------------------------------------------------
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_edge_pairs_list = []
        edge_score_valid_flag = False

        # Note: We can use current_proj_knowledge (the latest edge feats) or original. 
        # Usually, edge structure doesn't change, just features. We iterate keys.    
        for (idx_a, idx_b) in edge_keys:
            # Retrieve Ground Truth Edge Score
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

        # 7. Global Aggregation
        # global_concat = self.group_dropout(global_concat)  # 没必要使用dropout，因为attn后面自带dropout的
        current_group_masks = list(group_masks) # Shallow copy to preserve original masks for other calculations if needed

        if self.training and self.group_drop_ratio > 0.0:
            batch_size = embeddings[0].shape[0]
            num_groups = len(final_group_embeddings)

            # Generate random dropout mask: True means DROP, False means KEEP
            # shape: (Batch, Num_Groups)
            drop_decision = torch.rand((batch_size, num_groups), device=embeddings[0].device) < self.group_drop_ratio
            
            # SAFETY CHECK: Ensure at least one group is kept per sample to avoid NaN
            all_dropped = drop_decision.all(dim=1) # (Batch,)
            if all_dropped.any():
                # For indices where everything was dropped, randomly select one group to keep
                indices_to_keep = torch.randint(0, num_groups, (all_dropped.sum(),), device=embeddings[0].device)
                
                # Get the row indices that need fixing
                dropped_rows = torch.where(all_dropped)[0]
                
                # Force decision to False (Keep) for the selected group
                drop_decision[dropped_rows, indices_to_keep] = False

            # Apply the decision to the masks
            for g_idx in range(num_groups):
                # If drop_decision[b, g] is True, we want mask to be 0
                # If drop_decision[b, g] is False, we keep mask as is (multiply by 1)
                # Expand to match sequence length dim: (Batch, 1)
                should_drop = drop_decision[:, g_idx].unsqueeze(1) 
                keep_factor = (~should_drop).float()
                
                # Update the mask for this group
                current_group_masks[g_idx] = current_group_masks[g_idx] * keep_factor  # Drop Tokens -> 0

        # Concatenate embeddings and modified masks
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
            all_scores_tensor = torch.stack(all_edge_scores_list, dim=1)  # (Batch_Size, Num_Edges) stack all edge of one patient
            all_sims_tensor = torch.stack(all_cos_sims_list, dim=1)       # (Batch_Size, Num_Edges) stack all sim of one patient
            all_masks_tensor = torch.stack(all_valid_masks_list, dim=1)   # (Batch_Size, Num_Edges)

            scores_masked = all_scores_tensor.clone().float()
            scores_masked[all_masks_tensor == 0] = -1e9
            target_probs = F.softmax(scores_masked, dim=1)

            temperature = 0.1
            sims_masked = all_sims_tensor.clone() / temperature
            sims_masked[all_masks_tensor == 0] = -1e9
            
            pred_log_probs = F.log_softmax(sims_masked, dim=1)
            
            kl_loss = F.kl_div(pred_log_probs, target_probs, reduction='none')
            kl_loss_per_patient = kl_loss.sum(dim=1)
            
            valid_patients = (all_masks_tensor.sum(dim=1) > 1).float()   # (Batch_Size, )
            
            if valid_patients.sum() > 0:
                fusion_loss = (kl_loss_per_patient * valid_patients).sum() / valid_patients.sum()
    
        # 9. Compute MoE Auxiliary Loss (Load Balancing)
        # Prevent expert collapse or overfitting to single experts
        moe_loss = self.compute_moe_aux_loss()

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": {
                "total_loss": (self.fusion_loss_weight * fusion_loss - F.tanh(self.fusion_loss_weight)) + (self.moe_loss_weight * moe_loss),
                "KL_loss": fusion_loss,
                "MoE_loss": moe_loss
            }
        }