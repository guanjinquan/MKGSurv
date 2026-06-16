import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from modules.base_modules.aggregation_utils import masked_mean_pool
import random
from collections import defaultdict


class GELU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
    

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 2, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            nn.LayerNorm(dim * mult * 2),
            GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SafeCrossAttnEncoder(nn.Module):
    """Cross-attention block with an all-masked guard."""
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, ffn_mult: int = 2):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForward(embed_dim, mult=ffn_mult, dropout=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None, need_weights: bool = False) -> torch.Tensor:
        """Inputs use shape (B, L, D); padding mask uses True for invalid tokens."""

        query = self.norm_q(query)
        
        all_masked_rows = None
        if key_padding_mask is not None:
            all_masked_rows = key_padding_mask.all(dim=1)

            if all_masked_rows.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked_rows, 0] = False
        
        attn_out, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights)
            
        if all_masked_rows is not None and all_masked_rows.any():
            attn_out[all_masked_rows] = 0.0

        x = query + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm_ffn(x))
        x = x + self.dropout(ffn_out)

        if need_weights:
            return x, attn_weights
        return x


class EdgeContextualizer(nn.Module):
    """Updates edge embeddings using their endpoint node embeddings."""
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.cross_attn = SafeCrossAttnEncoder(embed_dim, num_heads)

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


class IntraGroupStep(nn.Module):
    """Applies within-group token interaction."""
    def __init__(self, embed_dim: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.intra_group_transformer = nn.ModuleList([
            SafeCrossAttnEncoder(embed_dim, num_heads=8)
            for _ in range(num_layers)
        ])
        
    def forward(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], 
                groups: List[List[int]]) -> List[torch.Tensor]:
        updated_embeddings = list(embeddings)
        
        for group_idx, group_indices in enumerate(groups):
            if not group_indices:
                continue
                
            group_feats = [updated_embeddings[i] for i in group_indices]
            group_masks = [masks[i] for i in group_indices]
            
            lengths = [f.shape[1] for f in group_feats]
            
            concat_feat = torch.cat(group_feats, dim=1)
            concat_mask = torch.cat(group_masks, dim=1)
            
            padding_mask = (concat_mask == 0)
            
            all_masked_rows = padding_mask.all(dim=1)
            if all_masked_rows.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_masked_rows, 0] = False

            for i in range(self.num_layers):
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


class InterGroupStep(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.drop_path_ratio = 0.1
        
        self.KG_GAT = nn.ModuleDict({
            'edge_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8),
            'node_to_node_attn': SafeCrossAttnEncoder(embed_dim, num_heads=8),
            'edge_updater': EdgeContextualizer(embed_dim, num_heads=8)
        })
        
    def _compute_edge_guided_context(self, 
                                   source_node: torch.Tensor, source_mask: torch.Tensor,
                                   edge_feat: torch.Tensor, edge_mask: torch.Tensor,
                                   layer_modules: nn.ModuleDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Builds an edge-conditioned source context."""
        source_padding_mask = (source_mask == 0)
        
        context_feat = layer_modules['edge_to_node_attn'](
            query=edge_feat,
            key=source_node,
            value=source_node,
            key_padding_mask=source_padding_mask
        )
        
        context_mask = edge_mask
        
        if context_mask is not None:
            context_feat = context_feat * context_mask.unsqueeze(-1).type_as(context_feat)
            
        return context_feat, context_mask
        
    def forward(self, 
                group_embeddings: List[torch.Tensor], 
                group_masks: List[torch.Tensor],
                edge_keys: List[Tuple[int, int]],
                edge_feats: Dict[Tuple[int, int], torch.Tensor],
                edge_masks: Dict[Tuple[int, int], torch.Tensor]) -> Tuple[List[torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]:
        current_group_embeddings = group_embeddings
        current_edge_feats = edge_feats.copy()
        
        final_layer_attns = {}
        
        for layer_idx in range(self.num_layers):
            layer_modules = self.KG_GAT
            num_groups = len(current_group_embeddings)
            
            next_step_edge_feats = {}
            adjacency_map = defaultdict(list)

            for (idx_a, idx_b) in edge_keys:

                edge_feat = current_edge_feats.get((idx_a, idx_b))
                if edge_feat is None:
                    continue
                    
                if self.training and getattr(self, 'drop_path_ratio', 0.0) > 0.0:
                    if random.random() < self.drop_path_ratio:
                        continue

                edge_mask = edge_masks.get((idx_a, idx_b))
                if edge_mask is None:
                    edge_mask = torch.ones(edge_feat.shape[:2], device=edge_feat.device)

                feat_a = current_group_embeddings[idx_a]
                mask_a = group_masks[idx_a]
                feat_b = current_group_embeddings[idx_b]
                mask_b = group_masks[idx_b]

                updated_edge_feat = self.KG_GAT['edge_updater'](
                    edge_feat, edge_mask, 
                    feat_a, mask_a, 
                    feat_b, mask_b
                )

                next_step_edge_feats[(idx_a, idx_b)] = updated_edge_feat
                adjacency_map[idx_a].append((idx_b, (idx_a, idx_b))) 
                adjacency_map[idx_b].append((idx_a, (idx_a, idx_b))) 

            current_edge_feats.update(next_step_edge_feats)

            next_group_embeddings = []
            current_layer_attns = {}

            for target_idx in range(num_groups):
                target_node = current_group_embeddings[target_idx]
                target_mask = group_masks[target_idx]
                
                neighbors = adjacency_map.get(target_idx, [])
                
                if not neighbors:
                    next_group_embeddings.append(target_node)
                    continue
                
                gathered_contexts = []
                gathered_masks = []
                source_idx_map = []
                
                for source_idx, edge_key in neighbors:
                    source_node = current_group_embeddings[source_idx]
                    source_mask = group_masks[source_idx]
                    
                    updated_edge = current_edge_feats[edge_key]
                    edge_mask = edge_masks.get(edge_key)
                    if edge_mask is None:
                        edge_mask = torch.ones(updated_edge.shape[:2], device=updated_edge.device)

                    ctx_feat, ctx_mask = self._compute_edge_guided_context(
                        source_node, source_mask,
                        updated_edge, edge_mask,
                        self.KG_GAT
                    )

                    gathered_contexts.append(ctx_feat)
                    gathered_masks.append(ctx_mask)
                    source_idx_map.append((source_idx, ctx_feat.shape[1]))
                
                if gathered_contexts:
                    concat_context = torch.cat(gathered_contexts, dim=1)
                    concat_context_mask = torch.cat(gathered_masks, dim=1)
                    context_padding_mask = (concat_context_mask == 0)
                    
                    updated_target, attn_weights = layer_modules['node_to_node_attn'](
                        query=target_node,
                        key=concat_context,
                        value=concat_context,
                        key_padding_mask=context_padding_mask,
                        need_weights=True
                    )
                    
                    if target_mask is not None:
                        updated_target = updated_target * target_mask.unsqueeze(-1).type_as(updated_target)
                    
                    next_group_embeddings.append(updated_target)

                    mean_attn_over_queries = attn_weights.mean(dim=1) 
                    
                    start_idx = 0
                    for source_idx, length in source_idx_map:
                        source_attn_chunk = mean_attn_over_queries[:, start_idx : start_idx + length]
                        group_attn_score = source_attn_chunk.sum(dim=1) 
                        
                        edge_pair = tuple(sorted((target_idx, source_idx)))
                        if edge_pair not in current_layer_attns:
                            current_layer_attns[edge_pair] = group_attn_score
                        else:
                            current_layer_attns[edge_pair] = current_layer_attns[edge_pair] + group_attn_score
                            
                        start_idx += length
                else:
                    next_group_embeddings.append(target_node)
            
            current_group_embeddings = next_group_embeddings
            final_layer_attns = current_layer_attns

        return current_group_embeddings, final_layer_attns


class GlobalAggregator(nn.Module):
    def __init__(self, embed_dim: int): 
        super().__init__() 
        self.global_transformer = SafeCrossAttnEncoder(embed_dim, num_heads=8) 
        self.post_fusion_norm = nn.LayerNorm(embed_dim) 
        self.drop_path_ratio = 0.25
        
    def forward(self, group_embeddings: List[torch.Tensor], 
                group_masks: List[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        current_group_masks = list(group_masks)
        if self.training and self.drop_path_ratio > 0.0:
            batch_size = group_embeddings[0].shape[0]
            num_groups = len(group_embeddings)

            drop_decision = torch.rand((batch_size, num_groups), device=group_embeddings[0].device) < self.drop_path_ratio
            all_dropped = drop_decision.all(dim=1) 
            if all_dropped.any():
                indices_to_keep = torch.randint(0, num_groups, (all_dropped.sum(),), device=group_embeddings[0].device)
                dropped_rows = torch.where(all_dropped)[0]
                drop_decision[dropped_rows, indices_to_keep] = False

            for g_idx in range(num_groups):
                should_drop = drop_decision[:, g_idx].unsqueeze(1) 
                keep_factor = (~should_drop).float()
                current_group_masks[g_idx] = current_group_masks[g_idx] * keep_factor 

        global_concat = torch.cat(group_embeddings, dim=1)
        global_mask = torch.cat(current_group_masks, dim=1)
        global_padding_mask = (global_mask == 0)

        all_masked_rows = global_padding_mask.all(dim=1)
        if all_masked_rows.any():
            global_padding_mask = global_padding_mask.clone()
            global_padding_mask[all_masked_rows, 0] = False

        global_transformed = self.global_transformer(
            query=global_concat, key=global_concat, value=global_concat, 
            key_padding_mask=global_padding_mask, need_weights=False)

        if all_masked_rows.any():
            global_transformed[all_masked_rows] = 0.0
        
        res_pool = masked_mean_pool(global_transformed, global_mask)
        fused_embedding = res_pool[0] if isinstance(res_pool, tuple) else res_pool
        fused_embedding = self.post_fusion_norm(fused_embedding)
        
        return fused_embedding


class MKGSurvFusion(nn.Module):
    def __init__(self, args, embed_dim: int,
                 max_modalities: int = 10,
                 max_groups: int = 10,
                 num_intra_layers: int = 1,
                 num_inter_layers: int = 1):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.log_temperature = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.03)))

        num_inter_layers = getattr(args, "num_layers", None) or 1
        self.loss_weight = getattr(args, "kl_loss_weight", None) or 1

        self.know_proj = nn.Sequential(
            nn.Linear(768, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.25)
        )

        self.intra_group_step = IntraGroupStep(embed_dim, num_intra_layers)
        self.inter_group_step = InterGroupStep(embed_dim, num_inter_layers)
        self.global_aggregator = GlobalAggregator(embed_dim)

    def forward(
        self,
        embeddings: List[torch.Tensor],
        masks: List[torch.Tensor],
        embeddings_groups: List[List[int]],
        groups_relationships: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge: Dict[Tuple[int, int], torch.Tensor],
        fusion_knowledge_mask: Dict[Tuple[int, int], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        batch_size = embeddings[0].shape[0]
        device = embeddings[0].device

        for (i, j), v in list(fusion_knowledge.items()):
            if (j, i) in fusion_knowledge and i < j:
                fusion_knowledge.pop((j, i))
                fusion_knowledge_mask.pop((j, i))

        edge_keys = list(fusion_knowledge.keys())
        if self.training:
            random.shuffle(edge_keys)

        current_proj_knowledge = {}
        for k, v in fusion_knowledge.items():
            current_proj_knowledge[k] = self.know_proj(v)

        info_level_embeddings = self.intra_group_step(embeddings, masks, embeddings_groups)

        group_embeddings = []
        group_masks = []
        for group_indices in embeddings_groups:
            if not group_indices:
                raise ValueError("Empty group found in embeddings_groups")
            curr_feats = [info_level_embeddings[i] for i in group_indices]
            curr_masks = [masks[i] for i in group_indices]
            group_embeddings.append(torch.cat(curr_feats, dim=1))
            group_masks.append(torch.cat(curr_masks, dim=1))

        group_validity_masks = []
        for g, m in zip(group_embeddings, group_masks):
            res = masked_mean_pool(g, m)
            valid_mask = res[1] if isinstance(res, tuple) else (m.sum(1) > 0)
            group_validity_masks.append(valid_mask)

        final_group_embeddings, inter_group_attns = self.inter_group_step(
            group_embeddings, group_masks,
            edge_keys, current_proj_knowledge, fusion_knowledge_mask
        )

        validity_dict = {}
        score_dict_llm = {}
        all_edge_scores_list = []
        all_valid_masks_list = []
        all_edge_pairs_list = []
        edge_score_valid_flag = False

        for (idx_a, idx_b) in edge_keys:
            edge_score = groups_relationships.get((idx_a, idx_b),
                          groups_relationships.get((idx_b, idx_a), None))
            if edge_score is None:
                edge_score = torch.zeros(batch_size, device=device)
            if edge_score.dim() > 1:
                edge_score = edge_score.view(-1)
            if edge_score.dim() == 0:
                edge_score = edge_score.expand(batch_size)

            has_a = group_validity_masks[idx_a].float()
            has_b = group_validity_masks[idx_b].float()
            pair_validity = has_a * has_b

            validity_dict[(idx_a, idx_b)] = pair_validity
            score_dict_llm[(idx_a, idx_b)] = edge_score

            edge_score_valid_flag |= edge_score.sum().item() > 0
            all_edge_scores_list.append(edge_score)
            all_valid_masks_list.append(pair_validity)
            all_edge_pairs_list.append((idx_a, idx_b))

        all_cos_sims_list = []
        if len(all_edge_pairs_list) > 0 and edge_score_valid_flag:
            final_pooled_results = [masked_mean_pool(g, m) for g, m in zip(final_group_embeddings, group_masks)]
            final_pooled_group_embeddings = [res[0] if isinstance(res, tuple) else res for res in final_pooled_results]
            final_pooled_group_embeddings = [F.normalize(g, p=2, dim=1) for g in final_pooled_group_embeddings]

            for idx_a, idx_b in all_edge_pairs_list:
                sim = torch.sum(final_pooled_group_embeddings[idx_a] * final_pooled_group_embeddings[idx_b], dim=1)
                sim = torch.clamp(sim, -1.0, 1.0)
                all_cos_sims_list.append(sim)

        final_sim_dict = {}
        for (idx_a, idx_b), sim in zip(all_edge_pairs_list, all_cos_sims_list):
            final_sim_dict[(idx_a, idx_b)] = sim

        fused_embedding = self.global_aggregator(final_group_embeddings, group_masks)

        fusion_loss = torch.tensor(0.0, device=fused_embedding.device)
        target_probs = None

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

        loss_dict = {
            "total_loss": self.loss_weight * fusion_loss,
            "temperature": self.log_temperature.exp(),
        }
        if target_probs is not None:
            loss_dict['target_probs'] = target_probs.mean(dim=0)

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": loss_dict,
        }
    
