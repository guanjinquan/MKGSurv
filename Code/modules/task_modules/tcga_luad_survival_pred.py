import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from modules.base_modules.surv_loss import CustomCoxPHLoss, mean_by_event
from general_utils.metrics import survival_metrics, multiple_classification_metrics
from modules.base_modules.init_weights import init_kaiming_norm

class TCGA_LUAD_SurvivalPred(nn.Module):

    # Required Class Atributes
    METRICS_FN = None
    embed_dim = None
    max_modalities_num = None
    max_groups_num = None

    def __init__(
        self,
        args,
        dataset: torch.utils.data.Dataset
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.embed_dim = 512
        self.dropout_rate = 0.25

        # --- Modality Setup ---
        self.active_modalities = dataset.get_active_modalities()
        self.max_modalities_num = len(self.active_modalities)
        self.max_groups_num = len(dataset.get_active_groups())
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
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.embed_dim, self.embed_dim) 
            )
            init_kaiming_norm(self.image_proj)

        # ----- Genomics Branch (genomics-genomics) -----
        if 'genomics-genomics' in self.active_modalities:
            print("Initializing Genomics Encoder")
            genomic_input_dim = 512
            self.genomics_encoder = nn.Sequential(
                nn.Linear(genomic_input_dim, self.embed_dim),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.embed_dim, self.embed_dim) 
            )
            init_kaiming_norm(self.genomics_encoder)

        # ----- Text Branch (text-pathology / text-treatment) -----
        # Assuming inputs are pre-extracted BERT features (768 dim)
        if any('text' in modal for modal in self.active_modalities):
            print("Initializing Text Encoder (Linear Projector)")
            text_input_dim = 768
            self.text_proj = nn.Sequential(
                nn.Linear(text_input_dim, self.embed_dim),
                nn.GELU(),
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.embed_dim, self.embed_dim) 
            )
            init_kaiming_norm(self.text_proj)

        # ----- Tabular Branch (from CSVs) -----
        self.tabular_encoders = nn.ModuleDict()
        for mod_name in self.active_modalities:
            if "tabular" in mod_name:
                try:
                    # Parse dimension from name "tabular-clinical-9" -> 9
                    in_dim = int(mod_name.split('-')[-1])
                    print(f"Initializing Tabular Encoder for '{mod_name}' (In: {in_dim}, Out: {self.embed_dim})")
                    self.tabular_encoders[mod_name] = nn.Sequential(
                        nn.Linear(in_dim, self.embed_dim),
                        nn.GELU(),
                        nn.LayerNorm(self.embed_dim),
                        nn.Dropout(self.dropout_rate),
                        nn.Linear(self.embed_dim, self.embed_dim) 
                    )
                    init_kaiming_norm(self.tabular_encoders[mod_name])
                except (ValueError, IndexError):
                    print(f"ERROR: Could not parse dimension from tabular modality name: '{mod_name}'")

        # ======================================================================
        # 2. Prediction Head
        # ======================================================================
        self.out_dim = 1
        self.loss_fn = CustomCoxPHLoss(reduction='none')
        self.METRICS_FN = survival_metrics

        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim // 2),
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim // 2, self.out_dim)
        )
        init_kaiming_norm(self.prediction_head)

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
        # Note: data_list elements can be (N, D) or just (D,). If (D,), treat as (1, D)
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
            - "medical_knowledge": Dict {(i,j): Tensor}
            - "modalities_groups": List[List[int]] (indices of modalities in the returned list)
        """
        # Determine batch size from the first active modality present
        present_modalities = [m for m in self.active_modalities if m in batch]
        if not present_modalities:
            raise ValueError("No active modalities found in batch.")
        
        # Safe extraction of batch size
        # Check if it's a list or tensor
        first_mod_data = batch[present_modalities[0]]
        batch_size = len(first_mod_data)
        
        device = self.device

        all_embeddings = []
        all_masks = []
        
        # Mapping for groups
        modality_group_map = {}   # 'pathology' -> group_index
        modalities_groups = []    # List[List[int]]
        
        # Mapping for Medical Knowledge retrieval: "mod_name" -> index in all_embeddings list
        modality_name_to_index = {} 
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
                # If a modality is completely missing for the whole batch, we skip it
                # to avoid dimension errors in fusion.
                continue 
            
            # B. Project
            if mod_name == 'image-pathology':
                encoded_feat = self.image_proj(padded_features)
                
            elif mod_name == 'genomics-genomics':
                encoded_feat = self.genomics_encoder(padded_features)
            
            elif 'text' in mod_name:
                encoded_feat = self.text_proj(padded_features)
                
            elif 'tabular' in mod_name:
                if mod_name in self.tabular_encoders:
                    encoded_feat = self.tabular_encoders[mod_name](padded_features)

            else:
                continue

            # C. Collect
            all_embeddings.append(encoded_feat)
            all_masks.append(mask)

            # D. Track Indices and Groups
            modality_name_to_index[mod_name] = current_list_index
            group_name = mod_name.split('-')[1]

            # Create group if not exists
            if group_name not in modality_group_map:
                modality_group_map[group_name] = len(modalities_groups)
                modalities_groups.append([])
            
            # Add current index to group
            modalities_groups[modality_group_map[group_name]].append(current_list_index)

            current_list_index += 1

        # =========================================================
        # 2. Medical Knowledge (Interaction Terms) Processing
        # =========================================================
        medical_knowledge = {}
        medical_knowledge_mask = {}
        groups_relationships = {}

        # Iterate over all pairs of *successfully encoded* modalities
        valid_groups = sorted(list(modality_group_map.keys()))

        for i in range(len(valid_groups)):
            for j in range(i + 1, len(valid_groups)):

                name_i = valid_groups[i]
                name_j = valid_groups[j]
                idx_i = modality_group_map[name_i]
                idx_j = modality_group_map[name_j]

                # If medical knowledge is available
                if "medical-knowledge" in batch:  
                    mk_batch = batch["medical-knowledge"]
                    pair_data_list = []     
                    score_list = []

                    # Iterate through batch samples to collect the specific pair
                    for sample_mk in mk_batch:
                        val = sample_mk.get((name_i, name_j), sample_mk.get((name_j, name_i), None))
                        pair_data_list.append(val['knowledge'])
                        score_list.append(val['score'])

                    # Pad and mask the MK data
                    mk_feat, mk_mask = self._pad_and_mask_modality(pair_data_list)
                    score_tensor = torch.tensor(score_list, device=device, dtype=torch.float32)

                else:
                    # Using 768 as default BERT dim
                    score_tensor = torch.ones(batch_size, 1, device=device, dtype=torch.float32)
                    mk_feat = torch.randn(batch_size, 1, 768, device=device)
                    mk_mask = torch.ones(batch_size, 1, device=device)

                # Save into dict
                groups_relationships[(idx_i, idx_j)] = score_tensor
                medical_knowledge[(idx_i, idx_j)] = mk_feat
                medical_knowledge_mask[(idx_i, idx_j)] = mk_mask

        return {
            "embeddings": all_embeddings,                        # List[Tensor]
            "masks": all_masks,                                  # List[Tensor]
            "medical_knowledge": medical_knowledge,              # Dict{(i,j): Tensor}
            "medical_knowledge_mask": medical_knowledge_mask,    # Dict{(i,j): Tensor}
            "groups_relationships": groups_relationships,        # Dict{(i,j): Tensor}
            "modalities_groups": modalities_groups,              # List[List[int]]
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
        weights_list = [batch['labels'][i]['sample_weight'] for i in range(batch_size)]
        label_time_list = [batch['labels'][i]['label_time'] for i in range(batch_size)]
        label_event_list = [batch['labels'][i]['label_event'] for i in range(batch_size)]

        Y_full = torch.tensor(label_time_list, device=device, dtype=torch.float32)
        c_full = torch.tensor(label_event_list, device=device, dtype=torch.float32)
        w_full = torch.tensor(weights_list, device=device, dtype=torch.float32)

        valid_Y = Y_full[patient_mask]
        valid_c = c_full[patient_mask]
        valid_w = w_full[patient_mask]

        # 5. 仅在有效子集上进行预测和损失计算
        valid_logits = self.prediction_head(valid_embeddings)
        loss_tensor_unreduced = self.loss_fn(valid_logits, valid_Y, valid_c, valid_w)
        loss = mean_by_event(loss_tensor_unreduced, valid_c)

        # 6. 将 logits 映射回原始 (B, out_dim) 张量
        logits[patient_mask] = valid_logits

        # 创建一个 (B,) 的 loss_tensor 以保持一致性 (可选)
        full_loss_tensor = torch.zeros(batch_size, device=device)
        full_loss_tensor[patient_mask] = loss_tensor_unreduced.squeeze(1) if loss_tensor_unreduced.dim() == 2 else loss_tensor_unreduced

        return {"logits": logits, "loss": loss, "loss_tensor": full_loss_tensor}
        

