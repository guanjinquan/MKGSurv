import os
import sys
from typing import List, Dict
from itertools import permutations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import torch.nn as nn
import torch.nn.functional as F
import torch

# --- Task Modules ---
from modules.task_modules.multi_oscc_rec_pred import MultiOSCCRecPred
from modules.task_modules.hancock_survival_pred import HANCOCKSurvivalPred
from modules.task_modules.multi_oscc_rec_pred_it import MultiOSCCRecPred_IT
from modules.task_modules.multi_oscc_rec_pred_split import MultiOSCCRecPred_Split
from modules.task_modules.oscc_inhouse_survival_pred import OSCCSurvivalPred


# --- Fusion Modules
from modules.fusion_modules.i2moe_fusion import I2MoEFusionModule
from modules.fusion_modules.hier_align_fusion import HierAlignFusionModule
from modules.fusion_modules.simple_fusion import SimpleFusion
from modules.fusion_modules.healnet_fusion import HealNetFusionModule
from modules.fusion_modules.KL_gated_fusion import KLGatedFusion
from modules.fusion_modules.MIBF_fusion import MIBF_fusion

# --- Common Modules ---
from modules.common_modules.align_utils import AlignmentModule
from modules.common_modules.aggregation_utils import masked_mean_pool



def GetModel(args):

    if args.fusion_type == 'kl_gated':
        return ModelInterfaceWithDeepSupervision(
            model_task=args.model_task, 
            modalities=args.modalities, 
            fusion_type=args.fusion_type
        )
    
    if args.fusion_type == "MIBF_fusion":
        return ModelInterfaceWithDeepSupervision(
            model_task=args.model_task, 
            modalities=args.modalities, 
            fusion_type=args.fusion_type
        )


    # with_multimodal_align
    if args.with_multimodal_align:
        return ModelInterfaceWithAlign(
            model_task=args.model_task, 
            modalities=args.modalities, 
            fusion_type=args.fusion_type
        )

    # fusion_type in ['concat', 'msa', 'lmf', 'gated', 'moe', 'i2moe', 'hier_align', 'healnet']
    else:
        return ModelInterface(
            model_task=args.model_task, 
            modalities=args.modalities, 
            fusion_type=args.fusion_type
        )




class ModelInterface(nn.Module):

    def __init__(self, model_task: str = "multi_oscc", modalities: str = 'all', fusion_type: str = 'moe'):
        super(ModelInterface, self).__init__()

        assert model_task in [  # Tasks that the model can handle
            "multi_oscc",
            "hancock",
            "multi_oscc_it",
            "multi_oscc_split",
            "oscc_inhouse",
        ], f"Unknown model task: {model_task}"

        self.modalities = modalities
        self.fusion_type = fusion_type

        # --- 1. Instantiate taks sub-modules ---
        # self.task_head should define: 
        #   1. self.task_head.embed_dim, 
        #   2. self.task_head.max_modalities_num
        if model_task == "multi_oscc":
            self.task_head = MultiOSCCRecPred(modalities=modalities)
        elif model_task == "hancock":
            self.task_head = HANCOCKSurvivalPred(modalities=modalities)
        elif model_task == 'multi_oscc_it':
            self.task_head = MultiOSCCRecPred_IT(modalities=modalities)
        elif model_task == 'multi_oscc_split':
            self.task_head = MultiOSCCRecPred_Split(modalities=modalities)
        elif model_task == "oscc_inhouse":
            self.task_head = OSCCSurvivalPred(modalities=modalities)
        else:
            raise ValueError(f"Unknown model task: {model_task}")

        self.embed_dim = self.task_head.embed_dim
        self.max_modalities = self.task_head.max_modalities_num

        
        # --- 2. Instantiate fusion sub-modules ---
        if self.fusion_type == 'hier_align':
            self.fusion_module = HierAlignFusionModule(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == 'i2moe':
            self.fusion_module = I2MoEFusionModule(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == 'healnet':
            self.fusion_module = HealNetFusionModule(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type in ['concat', 'msa', 'lmf', 'gated']:
            self.fusion_module = SimpleFusion(embed_dim=self.task_head.embed_dim, fusion_type=self.fusion_type, max_modalities=self.max_modalities)
        elif self.fusion_type == "kl_gated":
            self.fusion_module = KLGatedFusion(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == "MIBF_fusion":
            self.fusion_module = MIBF_fusion(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def forward(self, batch_size, data_dicts: List[Dict]):
        
        device = next(self.parameters()).device

        # --- Step 1: Unimodal Encoding ---
        encodings = self.task_head.encode(data_dicts)
        all_embeddings, all_masks = encodings['embeddings'], encodings['masks']

        # Check the data type of tensor
        all_embeddings = [e.to(torch.float) if e is not None else None for e in all_embeddings]
        all_masks = [m.to(torch.bool) if m is not None else None for m in all_masks]
        
        # Filter out None embeddings and corresponding masks for fusion
        present_embeddings = [e for e in all_embeddings if e is not None]
        present_masks = [m for e, m in zip(all_embeddings, all_masks) if e is not None]
        num_present = len(present_embeddings)
        assert num_present > 0, "No embeddings present for fusion and final prediction"
    
        # --- Step 2: Multimodal Fusion ---
        assert num_present > 1 or self.fusion_type == 'msa', "At least two modalities are required for fusion or use MSA fusion which can handle single modality."
        fusion_output = self.fusion_module(embeddings=all_embeddings, masks=all_masks)
        fused_embedding = fusion_output["fused_embedding"]
        fusion_losses_dict = fusion_output.get("loss_dict") or {}
        total_fusion_loss = fusion_losses_dict.get('total', torch.tensor(0.0, device=device))
        
        # --- Step 3: Multimodal Task Prediction ---
        assert fused_embedding.dim() == 2, f"Fused embedding must be a 2D tensor, shaping in (batch_size, embed_dim), but got {fused_embedding.shape}"

        # Create patient-wise mask: if any modality is present for a patient, mark as True
        present_masks = [m.to(device).bool() for m in present_masks if m is not None]
        if len(present_masks) == 0:
            patient_wise_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            #  mask.any(dim=1) -> (batch,), stack -> (num_mods, batch), any(dim=0) -> (batch,)
            patient_wise_mask = torch.stack([m.any(dim=1) for m in present_masks], dim=0).any(dim=0).bool()

        final_output = self.task_head.decode(fused_embedding, patient_wise_mask, data_dicts)
        multimodal_task_loss = final_output['loss']
        final_logits = final_output['logits']

        # --- Step 4: Combine All Losses, then return logits and all loss components for logging ---
        total_loss = multimodal_task_loss + 0.1 * total_fusion_loss
        # all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss}
        all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss, 'task_loss': multimodal_task_loss}  # For detailed logging
        all_losses.update({f'fusion_{k}': v for k, v in fusion_losses_dict.items() if 'total' not in k})
        
        return {"logits": final_logits, "losses": all_losses}

    def get_backbone_params(self):
        return self.task_head.get_backbone_params()
    
    def get_others_params(self):
        all_params = list(self.parameters())
        backbone_params = set(self.get_backbone_params())
        return [p for p in all_params if p not in backbone_params]



class ModelInterfaceWithAlign(ModelInterface):
    def __init__(self, model_task: str = "multi_oscc", modalities: str = 'all', fusion_type: str = 'moe'):
        super(ModelInterfaceWithAlign, self).__init__(model_task, modalities, fusion_type)
        self.align_module = AlignmentModule(embed_dim=self.embed_dim)
    
    def forward(self, batch_size, data_dicts: List[Dict]):

        device = next(self.parameters()).device

        # --- Step 1: Unimodal Encoding ---
        encodings = self.task_head.encode(data_dicts)
        all_embeddings, all_masks, align_pairs = encodings['embeddings'], encodings['masks'], encodings['align_pairs']

        # Check the data type of tensor
        all_embeddings = [e.to(torch.float) if e is not None else None for e in all_embeddings]
        all_masks = [m.to(torch.bool) if m is not None else None for m in all_masks]
        
        # Filter out None embeddings and corresponding masks for fusion
        present_indices = [i for i, e in enumerate(all_embeddings) if e is not None]
        present_embeddings = [e for e in all_embeddings if e is not None]
        present_masks = [m for e, m in zip(all_embeddings, all_masks) if e is not None]
        present_align_pairs = [(present_indices.index(p), present_indices.index(q)) for p, q in align_pairs if p in present_indices and q in present_indices]
        num_present = len(present_embeddings)
        assert num_present > 0, "No embeddings present for fusion and final prediction"

        # --- Extra Step: Align pairs adjustment ---
        align_losses = {}
        if num_present > 1 and len(align_pairs) > 0:
            pooled_output = [masked_mean_pool(embedding, mask) for embedding, mask in zip(present_embeddings, present_masks)]
            pooled_embeddings = [pooled[0] for pooled in pooled_output]
            pooled_masks = [pooled[1] for pooled in pooled_output]
            align_losses = self.align_module(pooled_embeddings, pooled_masks, present_align_pairs)
            total_align_loss = align_losses.get('total_loss', torch.tensor(0.0, device=device))

        # --- Step 2: Multimodal Fusion ---
        assert num_present > 1 or self.fusion_type == 'msa', "At least two modalities are required for fusion or use MSA fusion which can handle single modality."
        fusion_output = self.fusion_module(embeddings=all_embeddings, masks=all_masks)
        fused_embedding = fusion_output["fused_embedding"]
        fusion_losses_dict = fusion_output.get("loss_dict") or {}
        total_fusion_loss = fusion_losses_dict.get('total_loss', torch.tensor(0.0, device=device))
        
        # --- Step 3: Multimodal Task Prediction ---
        assert fused_embedding.dim() == 2, f"Fused embedding must be a 2D tensor, shaping in (batch_size, embed_dim), but got {fused_embedding.shape}"

        # Create patient-wise mask: if any modality is present for a patient, mark as True
        present_masks = [m.to(device).bool() for m in present_masks if m is not None]
        if len(present_masks) == 0:
            patient_wise_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            #  mask.any(dim=1) -> (batch,), stack -> (num_mods, batch), any(dim=0) -> (batch,)
            patient_wise_mask = torch.stack([m.any(dim=1) for m in present_masks], dim=0).any(dim=0)

        final_output = self.task_head.decode(fused_embedding, patient_wise_mask, data_dicts)
        multimodal_task_loss = final_output['loss']
        final_logits = final_output['logits']

        # --- Step 4: Combine All Losses, then return logits and all loss components for logging ---
        total_loss = multimodal_task_loss + 0.5 * total_align_loss + 0.1 * total_fusion_loss
        # all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss, 'align_loss': total_align_loss}
        all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss, 'align_loss': total_align_loss, 'task_loss': multimodal_task_loss}  # For detailed logging
        all_losses.update({f'fusion_{k}': v for k, v in fusion_losses_dict.items() if 'total' not in k})
        all_losses.update({f"align_{k}": v for k, v in align_losses.items() if 'total' not in k})
        
        return {"logits": final_logits, "losses": all_losses}
    


class ModelInterfaceWithDeepSupervision(ModelInterface):
    def __init__(self, model_task: str = "multi_oscc", modalities: str = 'all', fusion_type: str = 'moe'):
        super(ModelInterfaceWithDeepSupervision, self).__init__(model_task, modalities, fusion_type)
        

    def forward(self, batch_size, data_dicts: List[Dict]):
        device = next(self.parameters()).device

        # --- Step 1: Unimodal Encoding ---
        encodings = self.task_head.encode(data_dicts)
        all_embeddings, all_masks, _ = encodings['embeddings'], encodings['masks'], encodings['align_pairs']
        assert len(all_embeddings) == 2, "KLfusion requires two modalities"

        # Check the data type of tensor
        all_embeddings = [e.to(torch.float) if e is not None else None for e in all_embeddings]
        all_masks = [m.to(torch.bool) if m is not None else None for m in all_masks]
        
        # Filter out None embeddings and corresponding masks for fusion
        present_embeddings = [e for e in all_embeddings if e is not None]
        present_masks = [m for e, m in zip(all_embeddings, all_masks) if e is not None]
        num_present = len(present_embeddings)
        assert num_present > 0, "No embeddings present for fusion and final prediction"

        # --- Step 2: Multimodal Fusion ---
        assert num_present > 1 or self.fusion_type == 'msa', "At least two modalities are required for fusion or use MSA fusion which can handle single modality."
        fusion_output = self.fusion_module(embeddings=all_embeddings, masks=all_masks, task_head=self.task_head, batch=data_dicts)
        fused_embedding = fusion_output["fused_embedding"]
        fusion_losses_dict = fusion_output.get("loss_dict") or {}
        total_fusion_loss = fusion_losses_dict.get('total_loss', torch.tensor(0.0, device=device))
        
        # --- Step 3: Multimodal Task Prediction ---
        assert fused_embedding.dim() == 2, f"Fused embedding must be a 2D tensor, shaping in (batch_size, embed_dim), but got {fused_embedding.shape}"

        # Create patient-wise mask: if any modality is present for a patient, mark as True
        present_masks = [m.to(device).bool() for m in present_masks if m is not None]
        if len(present_masks) == 0:
            patient_wise_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        else:
            #  mask.any(dim=1) -> (batch,), stack -> (num_mods, batch), any(dim=0) -> (batch,)
            patient_wise_mask = torch.stack([m.any(dim=1) for m in present_masks], dim=0).any(dim=0)

        final_output = self.task_head.decode(fused_embedding, patient_wise_mask, data_dicts)
        multimodal_task_loss = final_output['loss']
        final_logits = final_output['logits']

        # --- Step 4: Combine All Losses, then return logits and all loss components for logging ---
        total_loss = multimodal_task_loss + 0.5 * total_fusion_loss
        all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss,  'task_loss': multimodal_task_loss}  # For detailed logging
        all_losses.update({f'fusion_{k}': v for k, v in fusion_losses_dict.items() if 'total' not in k})
        
        return {"logits": final_logits, "losses": all_losses}