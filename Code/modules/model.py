import os
import sys
from typing import List, Dict
from itertools import permutations

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset


# --- Task Modules ---
from modules.task_modules.oscc_inhouse_survival_pred import OSCCSurvivalPred
from modules.task_modules.hancock_survival_pred import HANCOCKSurvivalPred
from modules.task_modules.tcga_luad_survival_pred import TCGA_LUAD_SurvivalPred
from modules.task_modules.tcga_lusc_survival_pred import TCGA_LUSC_SurvivalPred

# --- Fusion Modules
from modules.fusion_modules.i2moe_fusion import I2MoEFusionModule
from modules.fusion_modules.hier_align_fusion import HierAlignFusionModule
from modules.fusion_modules.simple_fusion import SimpleFusion
from modules.fusion_modules.healnet_fusion import HealNetFusionModule
from modules.fusion_modules.js_gated_mas_fusion import KLGatedFusion
from modules.fusion_modules.MIBF_fusion import MIBF_fusion
from modules.fusion_modules.hgcn_fusion import HGCNFusionModule
from modules.fusion_modules.dimaf_fusion import DIMAFFusionModule
from modules.fusion_modules.surv_path import SurvPath
from modules.fusion_modules.MedKGAT_fusion import MedKGATFusion

# --- Common Modules ---
from modules.base_modules.align_utils import AlignmentModule
from modules.base_modules.aggregation_utils import masked_mean_pool
from modules.base_modules.multimodal_vib import TokenWiseMultiModalVIB


def GetModel(args, dataset):

    if args.fusion_type == 'kl_gated':
        return ModelInterfaceWithDeepSupervision(
            args, 
            dataset,
            decode_task=args.decode_task,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )
    
    if args.fusion_type == "mibf_fusion":
        return ModelInterfaceWithDeepSupervisionWeightedLoss(
            args, 
            dataset,
            decode_task=args.decode_task,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )
    
    if args.fusion_type == "dimaf_fusion":
        return ModelInterface(
            args, 
            dataset,
            decode_task=args.decode_task,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )
    
    if args.fusion_type == "hgcn_fusion":
        return ModelInterfaceWithDeepSupervisionWeightedLoss(
            args, 
            dataset,
            decode_task=args.decode_task,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )


    # with_multimodal_align
    if args.fusion_type == 'medkgat_fusion':
        return ModelInterfaceWithMedicalKnowledge(
            args, 
            dataset,
            decode_task=args.decode_task,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )

    # fusion_type in ['concat', 'msa', 'lmf', 'gated', 'moe', 'i2moe', 'hier_align', 'healnet']
    else:
        return ModelInterface(
            args, 
            dataset,
            decode_task=args.decode_task,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )




class ModelInterface(nn.Module):

    def __init__(self, args, dataset: Dataset, decode_task: str = "surv_pred", model_task: str = "multi_oscc", fusion_type: str = 'moe'):
        super(ModelInterface, self).__init__()

        assert model_task in [  # Tasks that the model can handle
            "hancock",
            "oscc_inhouse",
            "tcga_luad",
            "tcga_lusc",
        ], f"Unknown model task: {model_task}"

        modalities = dataset.get_active_modalities()
        assert len(modalities) > 0, f"Want at least one modality, but got no modalities passed in."
        self.modalities = modalities
        self.fusion_type = fusion_type

        # --- 1. Instantiate taks sub-modules ---
        # self.task_head should define: 
        #   1. self.task_head.embed_dim, 
        #   2. self.task_head.max_modalities_num
        if model_task == "hancock":
            self.task_head = HANCOCKSurvivalPred(args, decode_task, dataset=dataset)
        elif model_task == "oscc_inhouse":
            self.task_head = OSCCSurvivalPred(args, decode_task, dataset=dataset)
        elif model_task == "tcga_luad":
            self.task_head = TCGA_LUAD_SurvivalPred(args, decode_task, dataset=dataset)
        elif model_task == "tcga_lusc":
            self.task_head = TCGA_LUSC_SurvivalPred(args, decode_task, dataset=dataset)
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
        elif self.fusion_type == "mibf_fusion":
            self.fusion_module = MIBF_fusion(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == 'hgcn_fusion':
            self.fusion_module = HGCNFusionModule(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == "dimaf_fusion":
            self.fusion_module = DIMAFFusionModule(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == "surv_path":
            self.fusion_module = SurvPath(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == 'medkgat_fusion':
            self.fusion_module = MedKGATFusion(embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
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
        total_loss = multimodal_task_loss + total_fusion_loss
        # all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss}
        all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss, 'task_loss': multimodal_task_loss}  # For detailed logging
        all_losses.update({f'fusion_{k}': v for k, v in fusion_losses_dict.items() if 'total' not in k})
        
        return {"logits": final_logits, "losses": all_losses}


    


class ModelInterfaceWithDeepSupervision(ModelInterface):
    def __init__(self, args, dataset: Dataset, decode_task: str = "surv_pred", model_task: str = "multi_oscc", fusion_type: str = 'moe'):
        super(ModelInterfaceWithDeepSupervision, self).__init__(args, dataset, decode_task, model_task, fusion_type)

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
        total_loss = multimodal_task_loss + total_fusion_loss
        all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss,  'task_loss': multimodal_task_loss}  # For detailed logging
        all_losses.update({f'fusion_{k}': v for k, v in fusion_losses_dict.items() if 'total' not in k})
        
        return {"logits": final_logits, "losses": all_losses}
    


class ModelInterfaceWithDeepSupervisionWeightedLoss(ModelInterface):
    def __init__(self, args, dataset: Dataset, decode_task: str = "surv_pred", model_task: str = "multi_oscc", fusion_type: str = 'moe'):
        super(ModelInterfaceWithDeepSupervisionWeightedLoss, self).__init__(args, dataset, decode_task, model_task, fusion_type)

    def forward(self, batch_size, data_dicts: List[Dict]):
        device = next(self.parameters()).device

        # --- Step 1: Unimodal Encoding ---
        encodings = self.task_head.encode(data_dicts)
        all_embeddings, all_masks = encodings['embeddings'], encodings['masks']
        # assert len(all_embeddings) == 2, "KLfusion requires two modalities"

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
        task_loss_weights = fusion_output.get('weights', torch.ones(batch_size, device=device))
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
        assert final_output['loss_tensor'].shape == task_loss_weights.shape, f"Expect final_output['loss_tensor'].shape == task_loss_weights.shape, but got final_output['loss_tensor'] = {final_output['loss_tensor'].shape}, and  task_loss_weights = {task_loss_weights.shape}."
        multimodal_task_loss = torch.mul(final_output['loss_tensor'], task_loss_weights).mean()
        final_logits = final_output['logits']

        # --- Step 4: Combine All Losses, then return logits and all loss components for logging ---
        total_loss = multimodal_task_loss + total_fusion_loss
        all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss,  'task_loss': multimodal_task_loss}  # For detailed logging
        all_losses.update({f'fusion_{k}': v for k, v in fusion_losses_dict.items() if 'total' not in k})
        
        return {"logits": final_logits, "losses": all_losses}



class ModelInterfaceWithMedicalKnowledge(ModelInterface):
    def __init__(self, args, dataset: Dataset, decode_task: str = "surv_pred", model_task: str = "multi_oscc", fusion_type: str = 'moe'):
        super(ModelInterfaceWithMedicalKnowledge, self).__init__(args, dataset, decode_task, model_task, fusion_type)
    
    def forward(self, batch_size, data_dicts: List[Dict]):

        device = next(self.parameters()).device

        # --- Step 1: Unimodal Encoding ---
        encodings = self.task_head.encode(data_dicts)
        all_embeddings, all_masks, medical_knowledge, medical_knowledge_mask, modalities_groups, groups_relationships = \
            encodings['embeddings'], encodings['masks'], encodings['medical_knowledge'], \
                encodings['medical_knowledge_mask'], encodings['modalities_groups'], encodings['groups_relationships']

        # Check the data type of tensor
        all_embeddings = [e.to(torch.float) if e is not None else None for e in all_embeddings]
        all_masks = [m.to(torch.bool) if m is not None else None for m in all_masks]
        
        # Filter out None embeddings and corresponding masks for fusion
        present_indices = [i for i, e in enumerate(all_embeddings) if e is not None]
        present_embeddings = [e for e in all_embeddings if e is not None]
        present_masks = [m for e, m in zip(all_embeddings, all_masks) if e is not None]
        num_present = len(present_embeddings)
        assert num_present > 0, "No embeddings present for fusion and final prediction"

        for e in present_embeddings:
            assert not torch.any(torch.isnan(e)).item(), "Embedding contains NaN values"

        # --- Step 2: Multimodal Fusion ---
        assert num_present > 1 or self.fusion_type == 'msa', "At least two modalities are required for fusion or use MSA fusion which can handle single modality."
        fusion_output = self.fusion_module(
            embeddings=all_embeddings, 
            masks=all_masks, 
            embeddings_groups=modalities_groups, 
            groups_relationships=groups_relationships,
            fusion_knowledge=medical_knowledge, 
            fusion_knowledge_mask=medical_knowledge_mask
        )  
        fused_embedding = fusion_output["fused_embedding"]
        fusion_losses_dict = fusion_output.get("loss_dict") or {}
        total_fusion_loss = fusion_losses_dict.get('total_loss', torch.tensor(0.0, device=device))

        assert not torch.any(torch.isnan(fused_embedding)).item(), "Fused embedding contains NaN values"
        
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
        total_loss = multimodal_task_loss + total_fusion_loss

        all_losses = {'total_loss': total_loss, 'fusion_loss': total_fusion_loss, 'task_loss': multimodal_task_loss}  # For detailed logging
        all_losses.update({f'fusion_{k}': v for k, v in fusion_losses_dict.items() if 'total' not in k})
        
        return {"logits": final_logits, "losses": all_losses}