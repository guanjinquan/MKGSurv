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
from collections import defaultdict


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
    Sparse Mixture-of-Experts (MoE) Layer with Shared Expert.
    
    Optimizations for Speed & Overfitting:
    1. Shared Expert: Always active, captures common knowledge, stabilizes training.
    2. Vectorized Dispatch: Loops only over experts (not top-k), batching inputs.
    3. Reduced Expert Capacity: routing experts use smaller `mult` to prevent overfitting.
    """
    def __init__(self, dim, num_experts=4, num_selected=2, mult=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        
        # Gating Network
        self.gate = nn.Linear(dim, num_experts)
        
        # --- Optimization 1: Shared Expert ---
        # Acts as a backbone. Always active for all tokens.
        # This significantly reduces overfitting by ensuring a shared representation.
        self.shared_expert = ExpertFeedForward(dim, mult=mult, dropout=dropout)
        
        # --- Optimization 2: Smaller Routed Experts ---
        # Reduce 'mult' for routed experts to 2 (half size) to control parameter count
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
        final_output = shared_out # Start with shared output
        
        # --- 2. Gating ---
        gate_logits = self.gate(x_flat)
        
        if self.training:
            # Noisy gating for regularization
            noise = torch.randn_like(gate_logits) * (1.0 / self.num_experts)
            gate_logits = gate_logits + noise
            
        probs = F.softmax(gate_logits, dim=-1)
        
        # selected_probs: (N, K), selected_indices: (N, K)
        selected_probs, selected_indices = torch.topk(probs, self.num_selected, dim=-1)
        
        # Normalize weights
        selected_probs = selected_probs / selected_probs.sum(dim=-1, keepdim=True)
        
        # --- 3. Aux Loss (Load Balancing) ---
        # We use a simple mean-square auxiliary loss
        mean_probs = probs.mean(dim=0)
        aux_loss = (self.num_experts * (mean_probs ** 2).sum())
        self.aux_loss = aux_loss
        
        # --- 4. Optimized Dispatch Loop (Speed Up) ---
        # Instead of looping K times then Expert times, we just loop Experts.
        # We gather all tokens destined for Expert i (whether 1st or 2nd choice).
        
        for i in range(self.num_experts):
            # Create a mask for tokens that selected this expert
            # (N, K) == scalar -> (N, K) -> any(dim=1) -> (N,)
            selection_mask_k = (selected_indices == i)
            token_mask = selection_mask_k.any(dim=1) 
            
            if token_mask.any():
                # 1. Gather inputs
                # Extract tokens assigned to expert i
                expert_input = x_flat[token_mask]
                
                # 2. Run Expert
                expert_out = self.experts[i](expert_input)
                
                # 3. Apply Routing Weights
                # We need the weight corresponding to this expert for each selected token.
                # selected_probs[token_mask] gives (M, K)
                # selection_mask_k[token_mask] gives (M, K) boolean indicating position
                
                # Extract the specific weight for expert i from the top-k weights
                # Sum is safe because expert i only appears once per row in top-k
                weight_chunk = selected_probs[token_mask]
                mask_chunk = selection_mask_k[token_mask].float()
                
                # (M, 1)
                specific_weights = (weight_chunk * mask_chunk).sum(dim=1, keepdim=True)
                
                # 4. Scatter Add (Accumulate)
                # final_output[token_mask] += expert_out * specific_weights
                # Using index_add is often faster/cleaner for gradients, but masked set is fine here
                final_output[token_mask] += expert_out * specific_weights

        return final_output.view(batch_size, seq_len, dim)


# --- SafeCrossAttnEncoder ---
class SafeCrossAttnEncoder(nn.Module):
    """
    升级版交叉注意力模块。
    结构: CrossAttention -> Add & Norm -> FFN/MoE -> Add & Norm
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 4, use_moe: bool = False):
        super().__init__() 
        self.norm_q = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # 1. Attention 部分
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 2. FFN 部分 (Conditional MoE)
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
    


class MedKGATFusion_v10(nn.Module):
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
        self.log_temperature = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

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
            SafeCrossAttnEncoder(embed_dim, num_heads=8, dropout=attn_dropout_rate, use_moe=True)
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

    def _compute_edge_guided_context(self, 
                                   source_node: torch.Tensor, source_mask: torch.Tensor,
                                   edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                                   layer_modules: nn.ModuleDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper for Step 2 of GAT: 
        Use Updated Edge as Query, Source Node as Key/Value.
        Returns the "Source-Context-via-Edge" and the corresponding mask.
        """
        source_padding_mask = (source_mask == 0)
        
        # Edge (Query) queries Source (Key/Value)
        # Output shape will be same as Edge (B, Edge_Len, D)
        context_feat = layer_modules['edge_to_node_attn'](
            query=edge_feat,
            key=source_node,
            value=source_node,
            key_padding_mask=source_padding_mask
        )
        
        # Context mask is essentially the Edge mask, because the output aligns with Edge tokens
        context_mask = edge_mask
        
        # Apply mask to zero out invalid positions
        if context_mask is not None:
            context_feat = context_feat * context_mask.unsqueeze(-1).type_as(context_feat)
            
        return context_feat, context_mask

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

        # 4. Inter-Group Interaction (Multi-Layer GAT)
        # Re-implemented logic: Update ALL edges -> Enum Node -> Collect Contexts -> Update Node
        
        current_group_embeddings = group_embeddings # Points to current node features

        for layer_idx in range(self.num_inter_layers):
            layer_modules = self.shared_inter_layer
            num_groups = len(current_group_embeddings)
            
            # --- Sub-step 4.1: Global Edge Update ---
            # Update all edges based on their current connected nodes
            next_step_edge_feats = {} # Store updated edges for this layer
            
            # Pre-build Adjacency Map for Step 4.2
            # Structure: target_node_idx -> list of (source_node_idx, edge_key)
            adjacency_map = defaultdict(list)

            for (idx_a, idx_b) in edge_keys:
                edge_feat = current_proj_knowledge.get((idx_a, idx_b))

                # Edge Drop Logic
                if self.training and getattr(self, 'drop_edge_ratio', 0.0) > 0.0:
                    if random.random() < self.drop_edge_ratio:
                        continue
                
                edge_mask = fusion_knowledge_mask.get((idx_a, idx_b))
                if edge_mask is None:
                    edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

                # Get Group Data
                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a]
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                # Update Edge Features (Edge queries context of Node A and Node B)
                updated_edge_feat = layer_modules['edge_updater'](
                    edge_feat, edge_mask, 
                    feat_a, mask_a, 
                    feat_b, mask_b
                )
                
                # Store for usage in Node Update and next layer
                next_step_edge_feats[(idx_a, idx_b)] = updated_edge_feat
                
                # Record connections for Node Update Step
                adjacency_map[idx_a].append((idx_b, (idx_a, idx_b))) # b is source for a
                adjacency_map[idx_b].append((idx_a, (idx_a, idx_b))) # a is source for b

            # Update the knowledge dict for the *next* layer (and for usage in Node Update)
            # Note: We use the updated edges immediately for the node update in the current layer logic
            current_proj_knowledge.update(next_step_edge_feats)

            # --- Sub-step 4.2: Node Update ---
            # Enumerate every node, gather updated edges, update node
            next_group_embeddings = []

            for target_idx in range(num_groups):
                target_node = current_group_embeddings[target_idx]
                target_mask = group_masks[target_idx]
                
                neighbors = adjacency_map.get(target_idx, [])
                
                if not neighbors:
                    # Isolated node, keep features as is (or maybe apply simple self-attention/FFN if needed)
                    next_group_embeddings.append(target_node)
                    continue
                
                # Gather Contexts: Enum Edges of this node
                gathered_contexts = []
                gathered_masks = []
                
                for source_idx, edge_key in neighbors:
                    # 1. Get Source Node and Updated Edge
                    source_node = current_group_embeddings[source_idx]
                    source_mask = group_masks[source_idx]
                    
                    updated_edge = current_proj_knowledge[edge_key]
                    edge_mask = fusion_knowledge_mask.get(edge_key)
                    if edge_mask is None:
                        edge_mask = torch.ones(updated_edge.shape[:2], device=updated_edge.device)

                    # 2. Get "Source Updated Edge" (Edge as Query, Source as Key/Value)
                    ctx_feat, ctx_mask = self._compute_edge_guided_context(
                        source_node, source_mask,
                        updated_edge, edge_mask,
                        layer_modules
                    )
                    
                    gathered_contexts.append(ctx_feat)
                    gathered_masks.append(ctx_mask)
                
                # 3. Concatenate all source_updated_edges
                if gathered_contexts:
                    concat_context = torch.cat(gathered_contexts, dim=1)
                    concat_context_mask = torch.cat(gathered_masks, dim=1)
                    context_padding_mask = (concat_context_mask == 0)
                    
                    # 4. Update Target Node: Target queries Concatenated Context
                    updated_target = layer_modules['node_to_node_attn'](
                        query=target_node,
                        key=concat_context,
                        value=concat_context,
                        key_padding_mask=context_padding_mask
                    )
                    
                    if target_mask is not None:
                        updated_target = updated_target * target_mask.unsqueeze(-1).type_as(updated_target)
                    
                    next_group_embeddings.append(updated_target)
                else:
                    next_group_embeddings.append(target_node)
            
            # Update current embeddings for the next layer
            current_group_embeddings = next_group_embeddings

        # Final embeddings after all GAT layers
        final_group_embeddings = current_group_embeddings

        # ------------------------------------------------------------------
        # 4b. Data Collection for KL Loss (Post-GAT)
        # We iterate the edges one last time to gather stats
        # ------------------------------------------------------------------
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_edge_pairs_list = []
        edge_score_valid_flag = False

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
        current_group_masks = list(group_masks) # Shallow copy

        if self.training and self.group_drop_ratio > 0.0:
            batch_size = embeddings[0].shape[0]
            num_groups = len(final_group_embeddings)

            # Generate random dropout mask
            drop_decision = torch.rand((batch_size, num_groups), device=embeddings[0].device) < self.group_drop_ratio
            
            # SAFETY CHECK: Ensure at least one group is kept per sample
            all_dropped = drop_decision.all(dim=1) 
            if all_dropped.any():
                indices_to_keep = torch.randint(0, num_groups, (all_dropped.sum(),), device=embeddings[0].device)
                dropped_rows = torch.where(all_dropped)[0]
                drop_decision[dropped_rows, indices_to_keep] = False

            # Apply the decision to the masks
            for g_idx in range(num_groups):
                should_drop = drop_decision[:, g_idx].unsqueeze(1) 
                keep_factor = (~should_drop).float()
                current_group_masks[g_idx] = current_group_masks[g_idx] * keep_factor 

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
            all_scores_tensor = torch.stack(all_edge_scores_list, dim=1) 
            all_sims_tensor = torch.stack(all_cos_sims_list, dim=1) 
            all_masks_tensor = torch.stack(all_valid_masks_list, dim=1) 

            scores_masked = all_scores_tensor.clone().float()
            scores_masked[all_masks_tensor == 0] = -1e9
            target_probs = F.softmax(scores_masked, dim=1)

            temperature = self.log_temperature.exp().clamp(min=0.01, max=100)
            sims_masked = all_sims_tensor.clone() * temperature
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
                "total_loss": 5 * fusion_loss,
                "temperature": self.log_temperature.exp(),
            }
        }