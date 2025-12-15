import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Dict


# --- Helper class for the re-weighting MLP ---
class MLP(nn.Module):
    """A simple multi-layer perceptron."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLPReWeighting(nn.Module):
    """Re-weights the outputs of all interaction experts using an MLP, with support for masks."""
    def __init__(self, num_modalities, num_branches, hidden_dim, hidden_dim_rw, num_layers, temperature):
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim_rw,
            num_branches,
            num_layers,
            activation=nn.ReLU(),
            dropout=0.5,
        )

    def temperature_scaled_softmax(self, scores):
        scores = scores / self.temperature
        return torch.softmax(scores, dim=1)

    def forward(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor]):
        """
        Performs masked average pooling on inputs before feeding them to the MLP.
        """
        pooled_inputs = []
        for emb, msk in zip(embeddings, masks):
            # emb: (B, N, D), msk: (B, N)
            # expand mask for element-wise multiplication
            mask_expanded = msk.unsqueeze(-1).expand_as(emb)
            sum_embeddings = torch.sum(emb * mask_expanded, dim=1)
            # count valid tokens, add epsilon for safety
            sum_mask = msk.sum(dim=1).unsqueeze(-1) + 1e-9
            pooled_inputs.append(sum_embeddings / sum_mask)
            
        x = torch.cat(pooled_inputs, dim=1)
        x = self.mlp(x)
        return self.temperature_scaled_softmax(x)


# --- Base Fusion Model and Expert Wrapper ---
class BaseFusionModel(nn.Module):
    """
    A base fusion model that concatenates input token-level features and uses a transformer
    encoder for fusion, properly handling padding via an attention mask.
    """
    def __init__(self, embed_dim: int, num_layers: int = 1, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = embed_dim

    def forward(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenates embeddings and uses concatenated masks to create a key padding mask
        for the transformer encoder.
        """
        # Concatenate token embeddings and their corresponding masks
        concat_embeddings = torch.cat(embeddings, dim=1)  # Shape: (B, sum_of_tokens, D)
        concat_masks = torch.cat(masks, dim=1)            # Shape: (B, sum_of_tokens)

        # TransformerEncoder expects a key padding mask where True indicates a value to be IGNORED.
        # Our mask has 1 for valid tokens, so we invert it.
        src_key_padding_mask = (concat_masks == 0)

        # Pass concatenated tokens and the padding mask to the transformer
        fused_sequence = self.transformer_encoder(
            concat_embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        # Perform masked average pooling on the fused sequence for the final output
        output_mask_expanded = concat_masks.unsqueeze(-1).expand_as(fused_sequence)
        sum_embeddings = torch.sum(fused_sequence * output_mask_expanded, dim=1)
        sum_mask = concat_masks.sum(dim=1).unsqueeze(-1) + 1e-9

        final_embedding = sum_embeddings / sum_mask

        assert final_embedding.shape == (concat_embeddings.shape[0], self.embed_dim), f"Expected shape: (B, D), got: {final_embedding.shape}"
        return final_embedding


class InteractionExpert(nn.Module):
    """Interaction expert, wrapping a base fusion model. Handles embeddings and masks."""
    def __init__(self, fusion_model):
        super(InteractionExpert, self).__init__()
        self.fusion_model = fusion_model

    def _forward_with_replacement(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor], replace_index: Optional[int] = None):
        if replace_index is not None:
            device = embeddings[0].device
            # Create random embedding and a corresponding mask of all ones
            random_embedding = torch.randn_like(embeddings[replace_index], device=device)
            random_mask = torch.ones_like(masks[replace_index], device=device)

            modified_embeddings = embeddings[:replace_index] + [random_embedding] + embeddings[replace_index + 1:]
            modified_masks = masks[:replace_index] + [random_mask] + masks[replace_index + 1:]
            
            return self.fusion_model(modified_embeddings, modified_masks)
        
        return self.fusion_model(embeddings, masks)

    def forward_multiple(self, embeddings: List[torch.Tensor], masks: List[torch.Tensor]):
        outputs = []
        # Fusion with all original modalities
        outputs.append(self._forward_with_replacement(embeddings, masks, replace_index=None))
        # Fusion with one modality replaced by noise
        for i in range(len(embeddings)):
            outputs.append(self._forward_with_replacement(embeddings, masks, replace_index=i))
        return outputs




# --- Main I2MoE Fusion Module ---
class I2MoEFusionModule(nn.Module):
    """
    I²MoE fusion module, adapted to handle a dynamic number of modalities with token-level masks.
    """

    def __init__(self, args, embed_dim: int, max_modalities: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super(I2MoEFusionModule, self).__init__()

        self.args = args
        self.max_modalities = max_modalities
        self.embed_dim = embed_dim
        
        # Total number of experts = N (uniqueness) + 1 (synergy) + 1 (redundancy)
        num_branches = self.max_modalities + 2

        base_fusion_model = BaseFusionModel(embed_dim, num_layers, num_heads, dropout)
        self.interaction_experts = nn.ModuleList(
            [InteractionExpert(deepcopy(base_fusion_model)) for _ in range(num_branches)]
        )
        self.reweight = MLPReWeighting(
            num_modalities=self.max_modalities,
            num_branches=num_branches,
            hidden_dim=embed_dim,
            hidden_dim_rw=embed_dim,
            num_layers=2,
            temperature=1.0,
        )

    def _uniqueness_loss(self, anchor, positives, neg):
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        total_loss = 0
        if not positives: return torch.tensor(0.0, device=anchor.device)
        for pos in positives:
            total_loss += triplet_loss(anchor, pos, neg)
        return total_loss / len(positives)

    def _synergy_loss(self, anchor, negatives):
        if not negatives: return torch.tensor(0.0, device=anchor.device)
        total_syn_loss = 0
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        for neg in negatives:
            neg_norm = F.normalize(neg, p=2, dim=1)
            cosine_sim = torch.sum(anchor_norm * neg_norm, dim=1)
            total_syn_loss += torch.mean(cosine_sim)
        return total_syn_loss / len(negatives)

    def _redundancy_loss(self, anchor, positives):
        if not positives: return torch.tensor(0.0, device=anchor.device)
        total_red_loss = 0
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        for pos in positives:
            pos_norm = F.normalize(pos, p=2, dim=1)
            cosine_sim = torch.sum(anchor_norm * pos_norm, dim=1)
            total_red_loss += torch.mean(1 - cosine_sim)
        return total_red_loss / len(positives)

    def forward(self, embeddings: List[Optional[torch.Tensor]], masks: List[Optional[torch.Tensor]], **kargs) -> Dict:
        """
        Forward pass that handles missing modalities by creating dummy tensors and calculates
        losses only on the present modalities.
        """
        assert len(embeddings) == self.max_modalities, f"Expected {self.max_modalities} embeddings, got {len(embeddings)}"
        assert len(masks) == self.max_modalities, f"Expected {self.max_modalities} masks, got {len(masks)}"

        present_indices = [i for i, e in enumerate(embeddings) if e is not None]
        present_embeddings = [embeddings[i] for i in present_indices]
        num_present = len(present_embeddings)

        if num_present == 0:
            return {"fused_embedding": None, "loss_dict": {}}

        # Create dummy tensors for missing modalities to maintain a fixed input structure
        device = present_embeddings[0].device
        batch_size = present_embeddings[0].size(0)
        dummy_emb = torch.zeros(batch_size, 1, self.embed_dim, device=device)
        dummy_mask = torch.zeros(batch_size, 1, device=device)

        full_embeddings = [embeddings[i] if embeddings[i] is not None else dummy_emb for i in range(self.max_modalities)]
        full_masks = [masks[i] if masks[i] is not None else dummy_mask for i in range(self.max_modalities)]
        
        presence_mask = [e is not None for e in embeddings]

        # Pass data through all experts
        expert_outputs = [expert.forward_multiple(full_embeddings, full_masks) for expert in self.interaction_experts]
        
        # --- Loss Calculation ---
        interaction_losses = {}
        for i in range(self.max_modalities):
            if presence_mask[i]:
                outputs = expert_outputs[i]
                anchor = outputs[0]    # fused all modalities  
                positives = [outputs[j + 1] for j in range(self.max_modalities) if i != j and presence_mask[j]]  # 把屏蔽掉其他模态的作为正例
                neg = outputs[i + 1]  # 把屏蔽掉自己模态的作为负例
                loss = self._uniqueness_loss(anchor, positives, neg)
                interaction_losses[f"uniqueness_{i}"] = loss
        
        synergy_expert_outputs = expert_outputs[self.max_modalities]
        synergy_anchor = synergy_expert_outputs[0]
        synergy_negatives = [synergy_expert_outputs[i + 1] for i in range(self.max_modalities) if presence_mask[i]]
        interaction_losses["synergy"] = self._synergy_loss(synergy_anchor, synergy_negatives)
        
        redundancy_expert_outputs = expert_outputs[self.max_modalities + 1]
        redundancy_anchor = redundancy_expert_outputs[0]
        redundancy_positives = [redundancy_expert_outputs[i + 1] for i in range(self.max_modalities) if presence_mask[i]]
        interaction_losses["redundancy"] = self._redundancy_loss(redundancy_anchor, redundancy_positives)
        
        interaction_losses['total_loss'] = sum(l for l in interaction_losses.values() if isinstance(l, torch.Tensor))

        # --- Re-weighting and Fusion ---
        all_primary_embeddings = torch.stack([output[0] for output in expert_outputs], dim=1)
        interaction_weights = self.reweight(full_embeddings, full_masks)
        fused_embedding = (all_primary_embeddings * interaction_weights.unsqueeze(2)).sum(dim=1)

        return {"fused_embedding": fused_embedding, "loss_dict": interaction_losses}


