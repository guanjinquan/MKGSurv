import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from nystrom_attention import NystromAttention
from modules.base_modules.surv_loss import CustomCoxPHLoss
from modules.general_utils.metrics import survival_metrics
from modules.base_modules.init_weights import init_kaiming_norm

# ==========================================================================================
# Main Encoder-Decoder Model for HANCOCK Dataset
# ==========================================================================================
class HANCOCKSurvivalPred(nn.Module):
    
    METRICS_FN = staticmethod(survival_metrics)

    def __init__(
        self,
        args,
        decode_task: str,
        dataset: torch.utils.data.Dataset
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.embed_dim = 512
        self.dropout_rate = 0.25

        # --- Modality Setup ---
        self.active_modalities = dataset.get_active_modalities()
        self.max_modalities_num = len(self.active_modalities)
        print(f"Model initialized for modalities: {self.active_modalities}")

        # ======================================================================
        # 1. Encoders / Projection Layers
        # ======================================================================

        # ----- Image Branch (image-pathology) -----
        if 'image-pathology' in self.active_modalities:
            print("Initializing Image Projection")
            image_input_dim = 1024 * 2 + 1  
            self.image_proj = nn.Sequential(
                nn.Linear(image_input_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(self.dropout_rate)
            )
            init_kaiming_norm(self.image_proj)

        # ----- Text Branch (text-clinical / text-treatment) -----
        # Assuming inputs are pre-extracted BERT features (768 dim)
        if any('text' in modal for modal in self.active_modalities):
            print("Initializing Text Encoder (Linear Projector)")
            self.text_proj = nn.Sequential(
                nn.Linear(768, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.LayerNorm(self.embed_dim),
            )
            init_kaiming_norm(self.text_proj)

        # ----- Tabular Branch (from CSVs) -----
        self.tabular_encoders = nn.ModuleDict()
        for mod_name in self.active_modalities:
            if "tabular" in mod_name:
                try:
                    # Parse dimension from name "tabular-clinical-52" -> 52
                    in_dim = int(mod_name.split('-')[-1])
                    print(f"Initializing Tabular Encoder for '{mod_name}' (In: {in_dim}, Out: {self.embed_dim})")
                    
                    self.tabular_encoders[mod_name] = nn.Sequential(
                        nn.Linear(in_dim, self.embed_dim),
                        nn.LayerNorm(self.embed_dim),
                        nn.Linear(self.embed_dim, self.embed_dim),
                        nn.ReLU(),
                        nn.LayerNorm(self.embed_dim),
                        nn.Dropout(self.dropout_rate)
                    )
                    init_kaiming_norm(self.tabular_encoders[mod_name])
                except (ValueError, IndexError):
                    print(f"ERROR: Could not parse dimension from tabular modality name: '{mod_name}'")

        # ======================================================================
        # 2. Prediction Head
        # ======================================================================
        self.out_dim = 1
        self.loss_fn = CustomCoxPHLoss(reduction='none')

        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim // 2),
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim // 2, self.out_dim)
        )
        init_kaiming_norm(self.prediction_head)
        self.to(self.device)

    def _pad_and_mask_modality(self, data_list: List[Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Handles padding for a list of variable length tensors.
        
        Args:
            data_list: List of length B. Elements are either Tensor(N, D) or None.
        
        Returns:
            padded_batch: (B, Max_N, D) on self.device
            mask_batch: (B, Max_N) on self.device (1 for data, 0 for padding/None)
        """
        device = self.device
        batch_size = len(data_list)
        
        # 1. Identify valid tensors and determine dimensions
        valid_tensors = [t for t in data_list if t is not None]
        
        if not valid_tensors:
            print("WARNING: No valid tensors found in batch. Returning empty tensors.")
            return None, None
            
        # Determine max sequence length in this batch
        # Note: data_list elements can be (N, D) or just (D). If (D,), treat as (1, D)
        max_seq_len = 0
        input_dim = 0
        
        processed_list = []
        for t in data_list:
            if t is None:
                processed_list.append(None)
                continue
            
            # Ensure tensor is on device and float
            t = t.to(device).float()
            if t.dim() == 1:
                t = t.unsqueeze(0) # (D,) -> (1, D)
            
            current_len = t.shape[0]
            if current_len > max_seq_len:
                max_seq_len = current_len
            
            input_dim = t.shape[1]
            processed_list.append(t)

        # 2. Create Padded Tensor and Mask
        padded_batch = torch.zeros(batch_size, max_seq_len, input_dim, device=device)
        mask_batch = torch.zeros(batch_size, max_seq_len, device=device) # Float or Bool

        for i, t in enumerate(processed_list):
            if t is not None:
                length = t.shape[0]
                # Fill data
                padded_batch[i, :length, :] = t
                # Fill mask
                mask_batch[i, :length] = 1.0
            # else: Leave as zeros (masked out)

        return padded_batch, mask_batch

    def encode(self, batch: Dict[str, Any]) -> Dict:
        """
        Encodes all modalities in the batch into aligned embedding spaces.
        
        Returns:
            Dict containing:
            - "embeddings": List[Tensor(B, N_mod, Embed_Dim)]
            - "masks": List[Tensor(B, N_mod)]
            - "modalities_groups": List[List[int]] (indices of modalities in the returned list)
        """
        # Determine batch size from the first active modality present
        present_modalities = [m for m in self.active_modalities if m in batch]
        if not present_modalities:
             # Should generally not happen in training
            return {"embeddings": [], "masks": [], "modalities_groups": []}
        
        # Safe extraction of batch size
        first_mod_data = batch[present_modalities[0]]
        batch_size = len(first_mod_data)
        
        device = self.device

        all_embeddings = []
        all_masks = []
        
        # Mapping for groups
        modality_group_map = {} # 'pathology' -> group_index
        modalities_groups = []  # List[List[int]]
        
        # Mapping for tracking indices
        current_list_index = 0

        # =========================================================
        # 1. Main Modality Encoding Loop
        # =========================================================
        for mod_name in self.active_modalities:
            if mod_name not in batch:
                continue
                
            raw_data = batch[mod_name] # List[Tensor | None]
            
            # A. Pad and Mask
            padded_features, mask = self._pad_and_mask_modality(raw_data)
            
            if padded_features is None:
                # If a modality is completely missing for the whole batch, skip
                continue 
            
            # B. Project to unified embedding space
            encoded_feat = None
            
            if mod_name == 'image-pathology':
                encoded_feat = self.image_proj(padded_features)
                
            elif 'text' in mod_name:
                encoded_feat = self.text_proj(padded_features)
                
            elif 'tabular' in mod_name:
                if mod_name in self.tabular_encoders:
                    encoded_feat = self.tabular_encoders[mod_name](padded_features)
            
            if encoded_feat is None:
                continue

            # C. Collect
            all_embeddings.append(encoded_feat)
            all_masks.append(mask)

            # D. Track Indices and Groups
            group_name = mod_name.split('-')[1]  # e.g., "pathology" from "image-pathology"

            # Create group if not exists
            if group_name not in modality_group_map:
                modality_group_map[group_name] = len(modalities_groups)
                modalities_groups.append([])
            
            # Add current index to group
            modalities_groups[modality_group_map[group_name]].append(current_list_index)

            current_list_index += 1

        return {
            "embeddings": all_embeddings,           # List[Tensor]
            "masks": all_masks,                     # List[Tensor]
            "modalities_groups": modalities_groups  # List[List[int]]
        }

    def decode(self, pooled_embeddings: torch.Tensor, pooled_mask: Optional[torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Args:
            pooled_embeddings: (B, embed_dim)
            pooled_mask: (B,)
            batch: 包含 batch['labels'] = {'label_time': [y1, y2,...], 'label_event': [c1, c2,...]}
        """
        batch_size = pooled_embeddings.shape[0]
        device = pooled_embeddings.device

        logits = torch.zeros(batch_size, self.out_dim, device=device)
        loss = torch.tensor(0.0, device=device)
        
        # 1. 创建患者掩码
        patient_mask = pooled_mask.bool().to(device) if pooled_mask is not None else torch.ones(batch_size, dtype=torch.bool, device=device)

        # 2. 如果没有有效患者，立即返回
        if not patient_mask.any():
            return {"logits": logits, "loss": loss}

        # 3. 过滤有效的嵌入
        valid_embeddings = pooled_embeddings[patient_mask]
        
        if valid_embeddings.shape[0] == 0:
             print("Warning: decode valid_embeddings is empty.")
             return {"logits": logits, "loss": loss}

        # 4. batch['labels'] 是 {'label_Y': [...], 'label_c': [...]}
        do_mixup_list = [batch['labels'][i]['do_mixup'] for i in range(batch_size)]
        label_time_list = [batch['labels'][i]['label_time'] for i in range(batch_size)]
        label_event_list = [batch['labels'][i]['label_event'] for i in range(batch_size)]

        Y_full = torch.tensor(label_time_list, device=device, dtype=torch.float32)
        c_full = torch.tensor(label_event_list, device=device, dtype=torch.float32)
        m_full = torch.tensor(do_mixup_list, device=device, dtype=torch.bool)

        valid_Y = Y_full[patient_mask]
        valid_c = c_full[patient_mask]
        valid_m = m_full[patient_mask]

        # 5. 仅在有效子集上进行预测和损失计算
        valid_logits = self.prediction_head(valid_embeddings)
        loss_tensor_unreduced = self.loss_fn(valid_logits, valid_Y, valid_c, valid_m)
        loss = loss_tensor_unreduced.mean()

        # 6. 将 logits 映射回原始 (B, out_dim) 张量
        logits[patient_mask] = valid_logits

        # 创建一个 (B,) 的 loss_tensor 以保持一致性 (可选)
        full_loss_tensor = torch.zeros(batch_size, device=device)
        full_loss_tensor[patient_mask] = loss_tensor_unreduced.squeeze(1) if loss_tensor_unreduced.dim() == 2 else loss_tensor_unreduced

        return {"logits": logits, "loss": loss, "loss_tensor": full_loss_tensor}

    def get_backbone_params(self) -> List[nn.Parameter]:
        # Since we're not using the BERT models in this version, return empty list
        return []
    
    
    def get_others_params(self) -> List[nn.Parameter]:
        # Return all parameters since we're not using separate backbone models
        return list(self.parameters())