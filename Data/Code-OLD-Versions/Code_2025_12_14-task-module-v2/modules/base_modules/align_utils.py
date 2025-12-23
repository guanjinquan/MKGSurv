import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from itertools import combinations



class AlignmentModule(nn.Module):
    """
    A generalized module to align multi-modal features. It applies contrastive loss for
    pre-defined strong pairs and optimal transport (Sinkhorn distance) for all other (weak) pairs.
    It now handles cases where individual patients are missing a modality.
    """
    def __init__(self, embed_dim: int, temp: float = 0.07, sinkhorn_reg: float = 0.05, sinkhorn_iterations: int = 7):
        super().__init__()
        self.embed_dim = embed_dim
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iterations = sinkhorn_iterations
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))  # temp = 0.07 is the default temperature in SimCLR

    def _contrastive_loss(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """Calculates the symmetric contrastive loss (InfoNCE)."""
        if features_a.shape[0] < 2:
            return torch.tensor(0.0, device=features_a.device)
            
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)

        logits_per_a = self.logit_scale.exp() * features_a @ features_b.t()
        logits_per_b = logits_per_a.t()
        
        batch_size = features_a.shape[0]
        labels = torch.arange(batch_size, device=features_a.device)
        
        loss_a = F.cross_entropy(logits_per_a, labels)
        loss_b = F.cross_entropy(logits_per_b, labels)
        return (loss_a + loss_b) / 2

    def _sinkhorn_distance(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """Calculates the Sinkhorn distance between two distributions of features."""
        if features_a.shape[0] < 2:
            return torch.tensor(0.0, device=features_a.device)

        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        cost_matrix = 1 - torch.matmul(features_a, features_b.t())
        
        batch_size = features_a.shape[0]
        mu = torch.ones(batch_size, device=features_a.device) / batch_size
        nu = torch.ones(batch_size, device=features_a.device) / batch_size
        
        K = torch.exp(-cost_matrix / self.sinkhorn_reg)
        v = torch.ones_like(nu)
        
        for _ in range(self.sinkhorn_iterations):
            u = mu / (K @ v)
            v = nu / (K.t() @ u)
            
        T = torch.diag(u) @ K @ torch.diag(v)
        distance = torch.sum(T * cost_matrix)
        return distance

    def forward(self, pooled_embeddings: List[torch.Tensor], pooled_masks: List[torch.Tensor], strong_related_pairs: List[Tuple[int, int]] = None) -> Dict:
        """
        The forward pass for the generalized alignment module.
        Args:
            pooled_embeddings (List[torch.Tensor]): List of sample-level embeddings (Batch, Dim).
            pooled_masks (List[torch.Tensor]): List of boolean masks (Batch,) indicating valid samples.
            strong_related_pairs (List[Tuple[int, int]]): List of indices for strongly-paired embeddings.
        Returns:
            dict: A dictionary containing the total loss and all individual loss components.
        """
        
        assert None not in pooled_embeddings, f"Cannot process None embeddings."
        assert None not in pooled_masks, f"Cannot process None masks."

        num_modalities = len(pooled_embeddings)
        if num_modalities < 2:
            return {'total_loss': torch.tensor(0.0, device=pooled_embeddings[0].device)}

        device = pooled_embeddings[0].device
        
        all_possible_pairs = set(combinations(range(num_modalities), 2))
        strong_pairs_set = set()
        if strong_related_pairs:
            for i, j in strong_related_pairs:
                strong_pairs_set.add(tuple(sorted((i, j))))

        weak_pairs_set = all_possible_pairs - strong_pairs_set

        total_contrastive_loss = torch.tensor(0.0, device=device)
        total_sinkhorn_loss = torch.tensor(0.0, device=device)
        loss_details = {}

        def calculate_loss_for_pair(pair, loss_fn, loss_type):
            i, j = pair
            mask_a, mask_b = pooled_masks[i], pooled_masks[j]
            
            # Find common valid patients for this pair
            common_mask = mask_a & mask_b
            
            # If less than 2 patients have both modalities, we can't compute the loss.
            if common_mask.sum() < 2:
                return torch.tensor(0.0, device=device)
            
            emb_a = pooled_embeddings[i][common_mask]
            emb_b = pooled_embeddings[j][common_mask]
            
            return loss_fn(emb_a, emb_b)

        for pair in strong_pairs_set:
            i, j = pair
            loss = calculate_loss_for_pair(pair, self._contrastive_loss, 'contrastive')
            total_contrastive_loss += loss
            loss_details[f'contrastive_loss_({i},{j})'] = loss

        # 先不用 sinkhorn loss
        # for pair in weak_pairs_set:
        #     i, j = pair
        #     loss = calculate_loss_for_pair(pair, self._sinkhorn_distance, 'sinkhorn')
        #     total_sinkhorn_loss += loss
        #     loss_details[f'sinkhorn_loss_({i},{j})'] = loss
        
        total_loss = total_contrastive_loss + total_sinkhorn_loss
        
        loss_details['total_loss'] = total_loss
        loss_details['total_contrastive_loss'] = total_contrastive_loss
        loss_details['total_sinkhorn_loss'] = total_sinkhorn_loss

        return loss_details

