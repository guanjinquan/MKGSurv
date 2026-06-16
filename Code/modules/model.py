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
try:
    from modules.task_modules.oscc_inhouse_survival_pred import OSCCSurvivalPred
except ModuleNotFoundError:
    OSCCSurvivalPred = None

try:
    from modules.task_modules.oscc_inhouse_survival_pred_gelu import OSCCSurvivalPred_GELU
except ModuleNotFoundError:
    OSCCSurvivalPred_GELU = None

from modules.task_modules.tcga_luad_survival_pred import TCGA_LUAD_SurvivalPred
from modules.task_modules.tcga_lusc_survival_pred import TCGA_LUSC_SurvivalPred
from modules.task_modules.tcga_brca_survival_pred import TCGA_BRCA_SurvivalPred
from modules.task_modules.tcga_kirc_survival_pred import TCGA_KIRC_SurvivalPred

# --- Fusion Modules
try:
    from modules.fusion_modules.i2moe_fusion import I2MoEFusionModule
except ModuleNotFoundError:
    I2MoEFusionModule = None

try:
    from modules.fusion_modules.simple_fusion import SimpleFusion
except ModuleNotFoundError:
    SimpleFusion = None

try:
    from modules.fusion_modules.healnet_fusion import HealNetFusionModule
except ModuleNotFoundError:
    HealNetFusionModule = None

try:
    from modules.fusion_modules.hgcn_fusion import HGCNFusionModule
except ModuleNotFoundError:
    HGCNFusionModule = None

try:
    from modules.fusion_modules.dimaf_fusion import DIMAFFusionModule
except ModuleNotFoundError:
    DIMAFFusionModule = None

try:
    from modules.fusion_modules.surv_path import SurvPath
except ModuleNotFoundError:
    SurvPath = None

try:
    from modules.fusion_modules.mome_fusion import MOME_fusion
except ModuleNotFoundError:
    MOME_fusion = None

from modules.fusion_modules.mkgsurv_fusion import MKGSurvFusion 


def GetModel(args, dataset):
    
    if args.fusion_type == "hgcn_fusion":
        return ModelInterfaceWithDeepSupervisionWeightedLoss(
            args, 
            dataset,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )


    # with_multimodal_align
    if 'mkgsurv_fusion' in args.fusion_type:
        assert args.use_medical_knowledge == True, f"If you want to run random, else you must use medical knowledge"

        return ModelInterfaceWithMedicalKnowledge(
            args, 
            dataset,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )

    # randomly initialize medical knowledge
    elif 'mkgsurv_random_fusion' in args.fusion_type:
        args.use_medical_knowledge == False  # Asign False !!

        temp_fusion_type = args.fusion_type.replace("mkgsurv_random_fusion", "mkgsurv_fusion")
        print("Using Random Knowledge, Model is:", temp_fusion_type)

        return ModelInterfaceWithMedicalKnowledge(
            args, 
            dataset,
            model_task=args.model_task, 
            fusion_type=temp_fusion_type
        )

    elif "llm_baseline" in args.fusion_type:
        return ModelInterfaceWithMedicalKnowledge(
            args, 
            dataset,
            model_task=args.model_task, 
            fusion_type=args.fusion_type
        )

    return ModelInterface(
        args, 
        dataset,
        model_task=args.model_task, 
        fusion_type=args.fusion_type
    )



class ModelInterface(nn.Module):

    def __init__(self, args, dataset: Dataset, model_task: str = "multi_oscc", fusion_type: str = 'moe'):
        super(ModelInterface, self).__init__()

        assert model_task in [  # Tasks that the model can handle
            "hancock",
            "oscc_inhouse",
            "oscc_inhouse_gelu",
            "tcga_luad",
            "tcga_lusc",
            "tcga_brca",
            "tcga_kirc",
        ], f"Unknown model task: {model_task}"

        modalities = dataset.get_active_modalities()
        assert len(modalities) > 0, f"Want at least one modality, but got no modalities passed in."
        self.modalities = modalities
        self.fusion_type = fusion_type

        # --- 1. Instantiate taks sub-modules ---
        # self.task_head should define: 
        #   1. self.task_head.embed_dim, 
        #   2. self.task_head.max_modalities_num
        if model_task == "oscc_inhouse":
            if OSCCSurvivalPred is None:
                raise ImportError("OSCCSurvivalPred is unavailable: modules.task_modules.oscc_inhouse_survival_pred is missing.")
            self.task_head = OSCCSurvivalPred(args, dataset=dataset)
        elif model_task == "oscc_inhouse_gelu":
            if OSCCSurvivalPred_GELU is None:
                raise ImportError("OSCCSurvivalPred_GELU is unavailable: modules.task_modules.oscc_inhouse_survival_pred_gelu is missing.")
            self.task_head = OSCCSurvivalPred_GELU(args, dataset=dataset)
        elif model_task == "tcga_luad":
            self.task_head = TCGA_LUAD_SurvivalPred(args, dataset=dataset)
        elif model_task == "tcga_lusc":
            self.task_head = TCGA_LUSC_SurvivalPred(args, dataset=dataset)
        elif model_task == "tcga_brca":
            self.task_head = TCGA_BRCA_SurvivalPred(args, dataset=dataset)
        elif model_task == "tcga_kirc":
            self.task_head = TCGA_KIRC_SurvivalPred(args, dataset=dataset)
        else:
            raise ValueError(f"Unknown model task: {model_task}")

        self.embed_dim = self.task_head.embed_dim
        self.max_modalities = self.task_head.max_modalities_num
        self.max_groups = self.task_head.max_groups_num

        
        # --- 2. Instantiate fusion sub-modules ---
        
        # --- 2. Instantiate fusion sub-modules ---
        if self.fusion_type == 'i2moe':
            if I2MoEFusionModule is None:
                raise ImportError("I2MoEFusionModule is unavailable because an optional dependency is missing.")
            self.fusion_module = I2MoEFusionModule(args, embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == 'healnet':
            if HealNetFusionModule is None:
                raise ImportError("HealNetFusionModule is unavailable because an optional dependency is missing.")
            self.fusion_module = HealNetFusionModule(args, embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type in ['concat', 'msa', 'lmf', 'gated']:
            if SimpleFusion is None:
                raise ImportError("SimpleFusion is unavailable because an optional dependency is missing.")
            self.fusion_module = SimpleFusion(args, embed_dim=self.task_head.embed_dim, fusion_type=self.fusion_type, max_modalities=self.max_modalities)
        elif self.fusion_type == 'hgcn_fusion':
            if HGCNFusionModule is None:
                raise ImportError("HGCNFusionModule is unavailable because an optional dependency is missing.")
            self.fusion_module = HGCNFusionModule(args, embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == "dimaf_fusion":
            if DIMAFFusionModule is None:
                raise ImportError("DIMAFFusionModule is unavailable because an optional dependency is missing.")
            self.fusion_module = DIMAFFusionModule(args, embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == "surv_path":
            if SurvPath is None:
                raise ImportError("SurvPath is unavailable because an optional dependency is missing.")
            self.fusion_module = SurvPath(args, embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == 'mome_fusion':
            if MOME_fusion is None:
                raise ImportError("MOME_fusion is unavailable because an optional dependency is missing.")
            self.fusion_module = MOME_fusion(args, embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities)
        elif self.fusion_type == 'mkgsurv_fusion':
            self.fusion_module = MKGSurvFusion(args, embed_dim=self.task_head.embed_dim, max_modalities=self.max_modalities, max_groups=self.max_groups)
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
        assert len(all_embeddings) == self.max_modalities, f"Expected {self.max_modalities} embeddings, got {len(all_embeddings)}"
        
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
    def __init__(self, args, dataset: Dataset, model_task: str = "multi_oscc", fusion_type: str = 'moe'):
        super(ModelInterfaceWithDeepSupervision, self).__init__(args, dataset, model_task, fusion_type)

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
    def __init__(self, args, dataset: Dataset, model_task: str = "multi_oscc", fusion_type: str = 'moe'):
        super(ModelInterfaceWithDeepSupervisionWeightedLoss, self).__init__(args, dataset, model_task, fusion_type)

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
    def __init__(self, args, dataset: Dataset, model_task: str = "multi_oscc", fusion_type: str = 'moe'):
        super(ModelInterfaceWithMedicalKnowledge, self).__init__(args, dataset, model_task, fusion_type)
    
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

        # keep other infomation in fusion_output dict
        other_info = {k: v for k, v in fusion_output.items() if k not in ['fused_embedding', 'loss_dict']}

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
        
        return {"logits": final_logits, "losses": all_losses, "other_info": other_info}
