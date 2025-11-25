import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from modules.base_modules.surv_loss import CustomCoxPHLoss, mean_by_event
from modules.general_utils.metrics import survival_metrics, multiple_classification_metrics
from modules.base_modules.init_weights import init_kaiming_norm




class OSCCSurvivalPred(nn.Module):

    # Required Class Atributes
    METRICS_FN = None
    embed_dim = None
    max_modalities_num = None
    
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
        # Get active modalities directly from the dataset instance
        self.active_modalities = dataset.get_active_modalities()
        self.max_modalities_num = len(self.active_modalities)
        print(f"OSCC Model initialized for modalities: {self.active_modalities}")

        # ======================================================================
        # 1. Encoders / Projection Layers
        # ======================================================================

        # ----- Image Branch (image-pathology) -----
        if 'image-pathology' in self.active_modalities:
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

        # ----- Text Branch (text-clinical / text-pathology / text-treatment) -----
        # Assuming inputs are pre-extracted BERT features (768 dim)
        if any('text' in modal for modal in self.active_modalities):
            self.text_proj = nn.Sequential(
                nn.Linear(768, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.LayerNorm(self.embed_dim),
                nn.Dropout(self.dropout_rate)
            )
            init_kaiming_norm(self.text_proj)

        # ----- Tabular Branch (from CSVs) -----
        # Handles: tabular-metadata-4, tabular-history-9, tabular-blood-5, etc.
        self.tabular_encoders = nn.ModuleDict()
        for mod_name in self.active_modalities:
            if "tabular" in mod_name:
                try:
                    # Parse dimension from name "tabular-metadata-4" -> 4
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
        self.decode_task = decode_task
        
        if decode_task == 'surv_pred':
            self.out_dim = 1
            self.loss_fn = CustomCoxPHLoss(reduction='none')
            self.METRICS_FN = survival_metrics
            print("Task: Survival Prediction (CoxPH)")
        elif decode_task == 'treatment_pred':
            self.out_dim = 12 # OSCC dataset has 12 treatment types
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            self.METRICS_FN = multiple_classification_metrics
            print("Task: Treatment Prediction (12-class Multi-label)")
        else:
            raise ValueError(f"Unsupported task = {decode_task}")

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
        Handles padding for a list of variable length tensors (Bags).
        """
        device = self.device
        batch_size = len(data_list)
        
        processed_list = []
        valid_input_dim = None
        max_seq_len = 0

        # 1. 预处理所有数据，过滤无效数据，并确定维度
        for t in data_list:
            # 情况A: 此时就是 None
            if t is None:
                processed_list.append(None)
                continue
            
            t = t.to(device).float()
            
            # 情况B: Tensor 是空的 (例如 shape是 [0] 或 [])
            if t.numel() == 0:
                processed_list.append(None)
                continue

            # 统一维度: 如果是 (D,) 转为 (1, D)
            if t.dim() == 1:
                t = t.unsqueeze(0) 
            
            # 记录最大长度
            current_len = t.shape[0]
            if current_len > max_seq_len:
                max_seq_len = current_len
            
            # 锁定 input_dim (只取第一个有效数据的维度)
            if valid_input_dim is None:
                valid_input_dim = t.shape[1]
            elif t.shape[1] != valid_input_dim:
                # 可选：如果有数据维度不一致（比如有的768有的1024），这里可以报警
                # print(f"Warning: Inconsistent dim {t.shape[1]} vs {valid_input_dim}")
                pass

            processed_list.append(t)

        # 2. 如果整个 batch 都没有有效数据
        if valid_input_dim is None or max_seq_len == 0:
            return None, None

        # 3. 创建 Padded Tensor 和 Mask
        # 使用确定好的 valid_input_dim，而不是循环中最后那个 t 的维度
        padded_batch = torch.zeros(batch_size, max_seq_len, valid_input_dim, device=device)
        mask_batch = torch.zeros(batch_size, max_seq_len, device=device)

        for i, t in enumerate(processed_list):
            if t is not None:
                length = t.shape[0]
                # Fill data
                padded_batch[i, :length, :] = t
                # Fill mask
                mask_batch[i, :length] = 1.0

        return padded_batch, mask_batch

    def encode(self, batch: Dict[str, Any]) -> Dict:
        """
        Encodes all modalities in the batch into aligned embedding spaces.
        
        Returns:
            Dict containing:
            - "embeddings": List[Tensor(B, N_mod, Embed_Dim)]
            - "masks": List[Tensor(B, N_mod)]
        """
        all_embeddings = []
        all_masks = []
        
        # Iterate through active modalities defined in dataset
        for mod_name in self.active_modalities:
            if mod_name not in batch:
                continue
                
            raw_data = batch[mod_name]  # List[Tensor | None]
            
            # 1. Pad and Mask (Handle missing patients or variable bag sizes)
            padded_features, mask = self._pad_and_mask_modality(raw_data)
            
            if padded_features is None:
                # Typically implies entire batch is missing this modality
                # Create dummy zero embedding to maintain list alignment if needed, 
                # OR skip. Skipping is safer for downstream fusion that expects valid inputs.
                # Here we skip, fusion modules should handle variable length lists.
                continue 
            
            # 2. Project to Embed Dim based on modality type
            if mod_name == 'image-pathology':
                # padded_features: (B, N_patches, D_img)
                encoded_feat = self.image_proj(padded_features)
                
            elif 'text' in mod_name:
                # padded_features: (B, N_tokens, D_bert)
                encoded_feat = self.text_proj(padded_features)
                
            elif 'tabular' in mod_name:
                # padded_features: (B, 1, D_tab)
                if mod_name in self.tabular_encoders:
                    encoded_feat = self.tabular_encoders[mod_name](padded_features)
                else:
                    print(f"Warning: Tabular encoder missing for {mod_name}")
                    continue
            else:
                print(f"Warning: Modality {mod_name} has no encoder defined.")
                continue

            # 3. Append to lists
            # encoded_feat is (B, N, Embed_Dim), mask is (B, N)
            all_embeddings.append(encoded_feat)
            all_masks.append(mask)

        return {
            "embeddings": all_embeddings, # List of (B, N_i, D)
            "masks": all_masks,           # List of (B, N_i)
            "align_pairs": []             # Reserved for alignment losses
        }

    def _surv_decode(self, pooled_embeddings: torch.Tensor, pooled_mask: Optional[torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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

        Y_full = torch.tensor(label_time_list, device=device, dtype=torch.long)
        c_full = torch.tensor(label_event_list, device=device, dtype=torch.long)
        m_full = torch.tensor(do_mixup_list, device=device, dtype=torch.bool)

        valid_Y = Y_full[patient_mask]
        valid_c = c_full[patient_mask]
        valid_m = m_full[patient_mask]

        # 5. 仅在有效子集上进行预测和损失计算
        valid_logits = self.prediction_head(valid_embeddings)
        loss_tensor_unreduced = self.loss_fn(valid_logits, valid_Y, valid_c, valid_m)
        loss = mean_by_event(loss_tensor_unreduced, valid_c)

        # 6. 将 logits 映射回原始 (B, out_dim) 张量
        logits[patient_mask] = valid_logits

        # 创建一个 (B,) 的 loss_tensor 以保持一致性 (可选)
        full_loss_tensor = torch.zeros(batch_size, device=device)
        full_loss_tensor[patient_mask] = loss_tensor_unreduced.squeeze(1) if loss_tensor_unreduced.dim() == 2 else loss_tensor_unreduced

        return {"logits": logits, "loss": loss, "loss_tensor": full_loss_tensor}

    def _treatment_decode(self, pooled_embeddings: torch.Tensor, pooled_mask: Optional[torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
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

        # 4. batch['labels'] 是 {'treatment_type_onehot': [..]}
        label_onehot_list = [batch['labels'][i]['treatment_type_onehot'] for i in range(batch_size)]
        Y_full = torch.tensor(label_onehot_list, device=device, dtype=torch.float32)
        valid_Y = Y_full[patient_mask]

        # 5. 仅在有效子集上进行预测和损失计算
        valid_logits = self.prediction_head(valid_embeddings)
        loss_tensor_unreduced = self.loss_fn(valid_logits, valid_Y)
        loss = loss_tensor_unreduced.mean()

        # 6. 将 logits 映射回原始 (B, out_dim) 张量
        logits[patient_mask] = valid_logits

        # 创建一个 (B,) 的 loss_tensor 以保持一致性 (可选)
        full_loss_tensor = torch.zeros(batch_size, device=device)
        full_loss_tensor[patient_mask] = loss_tensor_unreduced.mean(1) if loss_tensor_unreduced.dim() == 2 else loss_tensor_unreduced

        return {"logits": logits, "loss": loss, "loss_tensor": full_loss_tensor}

    def decode(self, pooled_embeddings: torch.Tensor, pooled_mask: Optional[torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        if self.decode_task == 'surv_pred':
            return self._surv_decode(pooled_embeddings, pooled_mask, batch)
        elif self.decode_task == 'treatment_pred':
            return self._treatment_decode(pooled_embeddings, pooled_mask, batch)
        else:
            raise ValueError("Unsupport decode")