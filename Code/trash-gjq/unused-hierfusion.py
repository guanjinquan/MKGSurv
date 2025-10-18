import os
import sys
from typing import List, Dict
from itertools import permutations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import torch.nn as nn
import torch.nn.functional as F
import torch

# from modules.task_modules.multi_oscc_rec_pred import MultiOSCCRecPred
from modules.task_modules.multi_oscc_rec_pred_image import MultiOSCCRecPredImage
from modules.align_utils import AlignmentModule
from NewWork.MedAlignFusion.Code.modules.fusion_modules.i2moe_fusion import I2MoEFusionModule
from modules.fusion_modules.simple_fusion import SimpleFusion
from modules.aggregation_utils import AggregationHead




class HiAF(nn.Module):
    """
    Main orchestrator model, corrected to handle missing modalities per patient.
    """
    def __init__(self, device: str = 'cuda', modalities: str = 'all', fusion_type: str = 'moe', loss_weights: dict = None):
        super(HiAF, self).__init__()
        self.device = device
        self.modalities = modalities
        self.fusion_type = fusion_type

        # --- 1. Instantiate all sub-modules ---
        self.task_head = MultiOSCCRecPredImage(device=device, modalities=modalities)
        self.agg_head = AggregationHead(embed_dim=self.task_head.embed_dim)
        self.align_module = AlignmentModule(embed_dim=self.task_head.embed_dim)
        
        if self.fusion_type == 'moe':
            self.fusion_module = I2MoEFusionModule(embed_dim=self.task_head.embed_dim, max_modalities=self.task_head.max_modalities_num)
        elif self.fusion_type in ['concat', 'msa']:
            self.fusion_module = SimpleFusion(embed_dim=self.task_head.embed_dim, fusion_type=self.fusion_type, max_modalities=self.task_head.max_modalities_num)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        self.consistency_loss_fn = nn.KLDivLoss(reduction='none') 

        self.loss_weights = loss_weights or {
            "align": 1.0, "unimodal_task": 0.5, "consistency": 0.5,
            "fusion_interaction": 1.0, "multimodal_task": 1.0,
        }
            
    def _kl_loss(self, p_logits, p_mask, q_logits, q_mask):
        """Helper to compute KL divergence loss only on co-occurring samples."""
        # Ensure masks are boolean
        p_mask = p_mask.bool()
        q_mask = q_mask.bool()
        
        common_mask = p_mask & q_mask
        if common_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        p_softmax = F.softmax(p_logits[common_mask], dim=-1)
        q_log_softmax = F.log_softmax(q_logits[common_mask], dim=-1)
        
        kl_div = self.consistency_loss_fn(q_log_softmax, p_softmax.detach())
        # The loss is averaged over the samples that are present in both modalities
        return kl_div.sum(dim=1).mean()

    def forward(self, data_dicts: List[Dict]):
        labels = data_dicts['labels']
        batch_size = len(labels)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=self.device)

        encodings = self.task_head.encode(data_dicts)
        all_embeddings, all_masks, strong_related_pairs = encodings['embeddings'], encodings['masks'], encodings['strong_related_pairs']
        
        present_indices = [i for i, e in enumerate(all_embeddings) if e is not None]
        present_embeddings = [all_embeddings[i] for i in present_indices]
        present_masks = [all_masks[i] for i in present_indices]
        num_present = len(present_embeddings)
        assert num_present > 0, "No embeddings present for fusion and final prediction"
        
        # --- Initialize losses ---
        align_loss = torch.tensor(0.0, device=self.device)
        unimodal_task_loss = torch.tensor(0.0, device=self.device)
        consistency_loss = torch.tensor(0.0, device=self.device)
        fusion_interaction_loss = torch.tensor(0.0, device=self.device)
        multimodal_task_loss = torch.tensor(0.0, device=self.device)
        # Use the out_dim from the task_head for correct shape
        final_logits = torch.zeros(batch_size, self.task_head.out_dim, device=self.device)
        align_losses_dict = {}
        fusion_losses_dict = {}

        # --- Pool embeddings for Alignment and Unimodal steps ---
        pooled_embeddings, pooled_masks = [], []
        pooled_outputs = [self.agg_head(emb, mask) for emb, mask in zip(present_embeddings, present_masks)]
        pooled_embeddings = [out[0] for out in pooled_outputs]
        pooled_masks = [out[1] for out in pooled_outputs]

        # --- Step 2: Alignment Loss ---
        if num_present >= 2 and batch_size > 1:
            strong_pairs_remapped = [
                (present_indices.index(i), present_indices.index(j))
                for i, j in strong_related_pairs
                if i in present_indices and j in present_indices
            ]
            align_losses_dict = self.align_module(pooled_embeddings, pooled_masks, strong_pairs_remapped)
            align_loss = align_losses_dict.get('total_loss', torch.tensor(0.0, device=self.device))

        # --- Step 3 & 4: Uni-modal Prediction and Consistency ---
        # Use the POOLED embeddings and masks for unimodal decoding
        if num_present > 1:
            # Decode each modality
            unimodal_outputs = [self.task_head.decode(p_emb, p_mask, labels) for p_emb, p_mask in zip(pooled_embeddings, pooled_masks)]
            
            # Average the loss across present modalities
            unimodal_task_loss = sum(out['loss'] for out in unimodal_outputs) / num_present
        
            unimodal_logits = [out['logits'] for out in unimodal_outputs]
            # The pooled_masks are already the patient-level masks needed for KL loss
            kl_losses = [self._kl_loss(p_logits, p_mask, q_logits, q_mask) for (p_logits, p_mask), (q_logits, q_mask) in permutations(zip(unimodal_logits, pooled_masks), 2)]
            consistency_loss = torch.stack(kl_losses).mean()

        # --- Step 5 & 6: Fusion and Final Prediction ---
        # Pre-aggregate embeddings and masks if the fusion module requires it

        assert num_present > 1 or self.fusion_type == 'msa', "At least two modalities are required for fusion or use MSA fusion which can handle single modality."
        if hasattr(self.fusion_module, 'NeedAggregation') and self.fusion_module.NeedAggregation:
            aggregated_embeddings = []
            aggregated_masks = []
            for emb, mask in zip(all_embeddings, all_masks):
                if emb is not None:
                    pooled_emb, pooled_mask = self.agg_head(emb, mask)
                    aggregated_embeddings.append(pooled_emb)
                    aggregated_masks.append(pooled_mask)
                else:
                    aggregated_embeddings.append(None)
                    aggregated_masks.append(None)
            embeddings_for_fusion = aggregated_embeddings
            masks_for_fusion = aggregated_masks
        else:
            # Pass token-level embeddings and masks for modules that handle aggregation internally (e.g., MoE)
            embeddings_for_fusion = all_embeddings
            masks_for_fusion = all_masks

        fusion_output = self.fusion_module(embeddings_for_fusion, masks_for_fusion)
        fused_embedding = fusion_output["fused_embedding"]

        fusion_losses_dict = fusion_output.get("loss_dict") or {}
        fusion_interaction_loss = fusion_losses_dict.get('total', torch.tensor(0.0, device=self.device))
        
        # For the final fused embedding, all patients are considered valid, so mask is None.
        assert fused_embedding.dim() == 2, f"Fused embedding must be a 2D tensor, shaping in (batch_size, embed_dim), but got {fused_embedding.shape}"
        final_output = self.task_head.decode(fused_embedding, None, labels)
        multimodal_task_loss = final_output['loss']
        final_logits = final_output['logits']

        # --- Step 7: Combine All Losses ---
        total_loss = (self.loss_weights['align'] * align_loss +
                      self.loss_weights['unimodal_task'] * unimodal_task_loss +
                      self.loss_weights['consistency'] * consistency_loss +
                      self.loss_weights['fusion_interaction'] * fusion_interaction_loss +
                      self.loss_weights['multimodal_task'] * multimodal_task_loss)

        # --- Step 8: Return logits and all loss components for logging ---
        all_losses = {
            'total_loss': total_loss, 'align_loss': align_loss, 'unimodal_task_loss': unimodal_task_loss,
            'consistency_loss': consistency_loss, 'fusion_interaction_loss': fusion_interaction_loss,
            'multimodal_task_loss': multimodal_task_loss,
        }
        all_losses.update({f'align_{k}': v for k, v in align_losses_dict.items() if 'total' not in k})
        all_losses.update({f'fusion_{k}': v for k, v in fusion_losses_dict.items() if k != 'total'})
        
        return {"logits": final_logits, "losses": all_losses}

    def get_backbone_params(self):
        return self.task_head.get_backbone_params()
    
    def get_others_params(self):
        all_params = list(self.parameters())
        backbone_params = set(self.get_backbone_params())
        return [p for p in all_params if p not in backbone_params]

    