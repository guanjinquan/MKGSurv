import os
import sys

import torch.utils
# 假设这个文件在 'modules/models' 目录下，调整路径以导入同级 'common_modules'
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from nystrom_attention import NystromAttention
from modules.common_modules.surv_loss import CustomCoxPHLoss, mean_by_event
from training_utils.metrics import survival_metrics, multiple_classification_metrics
from modules.common_modules import GetImageAggregater
from modules.common_modules.init_weights import init_kaiming_norm



# ==========================================================================================
# Main Encoder-Decoder Model for TCGA-LUAD Dataset
# ==========================================================================================
class TCGA_LUSC_SurvivalPred(nn.Module):

    def __init__(
        self,
        args,
        decode_task: str,
        dataset: torch.utils.data.Dataset
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.embed_dim = 512

        # --- Modality Setup ---
        self.active_modalities = dataset.get_active_modalities()
        self.max_modalities_num = len(self.active_modalities)
        print(f"Model initialized for modalities: {self.active_modalities}")

        # ----- 1. (Graph) Image Branch (image-pathology) -----
        if 'image-pathology' in self.active_modalities:
            print("Initializing Image MIL (AggregatingTransMIL)")
            self.num_image_tokens = 16

            self.image_mil = GetImageAggregater(
                args.image_aggregater,
                InputDim=1024,
                OutputDim=self.embed_dim,
                OutputTokenNum=self.num_image_tokens,
                PrototypesData=dataset.get_training_image_embeddings_prototypes(self.num_image_tokens)
            )
            init_kaiming_norm(self.image_mil)

        # ----- 2. (Graph) Genomics Branch (genomics-genomics) -----
        # --- MODIFIED: Per user request, use simple Linear for N=5 tokens, not MIL ---
        if 'genomics-genomics' in self.active_modalities:
            print("Initializing Genomics Encoder (Linear layer for N=5 tokens)")
            self.genomics_encoder = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, self.embed_dim)
            )
            init_kaiming_norm(self.genomics_encoder)

        # ----- 3. Text Branch (text-pathology) -----
        if any('text' in modal for modal in self.active_modalities):
            print("Initializing Text Encoder (ClinicalBERT)")
            self.text_model_name = "medicalai/ClinicalBERT"
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, use_fast=True)
            self.bert = AutoModel.from_pretrained(self.text_model_name)
            bert_hidden_size = self.bert.config.hidden_size
            self.bert_proj = nn.Linear(bert_hidden_size, self.embed_dim) if bert_hidden_size != self.embed_dim else nn.Identity()
            init_kaiming_norm(self.bert_proj)

        # ----- 4. Tabular Branch (from CSVs) -----
        self.tabular_encoder = nn.ModuleDict()
        for mod_name in self.active_modalities:
            if "tabular" in mod_name:
                try:
                    # 从 "tabular-clinical-56" 中提取 "56"
                    in_dim = int(mod_name.split('-')[-1])
                    print(f"Initializing Tabular Encoder for '{mod_name}' (In: {in_dim}, Out: {self.embed_dim})")
                    self.tabular_encoder[mod_name] = nn.Sequential(
                        nn.LayerNorm(in_dim),
                        nn.Linear(in_dim, self.embed_dim)
                    )
                    init_kaiming_norm(self.tabular_encoder[mod_name])
                except (ValueError, IndexError):
                    print(f"ERROR: Could not parse dimension from tabular modality name: '{mod_name}'")
                    print("Expected format: 'tabular-type-DIMENSION' (e.g., 'tabular-clinical-56')")

        # ----- Prediction Head (for Decode step) -----
        self.decode_task = decode_task
        if decode_task == 'surv_pred':
            self.out_dim = 1
            self.loss_fn = CustomCoxPHLoss(reduction='none')
            self.METRICS_FN = survival_metrics
        elif decode_task == 'treatment_pred':
            self.out_dim = 5
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            self.METRICS_FN = multiple_classification_metrics
        else:
            raise ValueError(f"Unsupport task = {decode_task}")

        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dim // 4),
            nn.Dropout(0.3),

            nn.Linear(self.embed_dim // 4, self.out_dim)
        )
        init_kaiming_norm(self.prediction_head)



    def _chunk_token_ids(self, ids: List[int], chunk_size: int) -> List[List[int]]:
        """Splits a list of token ids into chunks."""
        return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    def _encode_text(self, texts_list: List[Optional[str | List[str]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes a batch of texts, handling List[str] as separate items and chunking long inputs."""
        batch_size = len(texts_list) # <--- 修正了拼写错误 (was text_list)
        chunk_payload = 510  # 512 - 2 for [CLS] and [SEP]
        
        all_chunks = []
        mapping_info = [] # (original_batch_index, num_chunks)
        
        for i, item in enumerate(texts_list):
            
            # 存放这个批次项 (item) 最终对应的所有 token 块
            item_specific_chunks = [] 
            
            # --- 这是新的核心逻辑 ---
            texts_to_process = []
            if isinstance(item, str) and item.strip():
                # 1. 如果是单个字符串，将其放入待处理列表
                texts_to_process.append(item)
            elif isinstance(item, list):
                # 2. 如果是列表，过滤掉无效字符串后，全部放入待处理列表
                texts_to_process.extend([t for t in item if isinstance(t, str) and t.strip()])
            
            # 3. 如果 item 是 None, [], 或 ["", " "]，texts_to_process 将为空
            if not texts_to_process:
                mapping_info.append({'index': i, 'n': 0})
                continue
            
            # 4. 统一处理所有待处理的文本
            # 无论是单个 str 还是 List[str] 中的每个 str，
            # 它们现在都被同等对待：tokenize -> chunk
            for text in texts_to_process:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                # _chunk_token_ids 会处理长文本和短文本
                chunks = self._chunk_token_ids(token_ids, chunk_payload) 
                if chunks:
                    item_specific_chunks.extend(chunks)

            # 记录这个批次项 (item) 总共产生了多少个 chunk
            if not item_specific_chunks:
                mapping_info.append({'index': i, 'n': 0})
                continue
                
            mapping_info.append({'index': i, 'n': len(item_specific_chunks)})
            all_chunks.extend(item_specific_chunks)


        if not all_chunks:
            # 返回一个 (B, 1, D) 和 (B, 1) 的空张量，与你原始逻辑保持一致
            return torch.zeros(batch_size, 1, self.embed_dim, device=self.device), torch.zeros(batch_size, 1, device=self.device).bool()

        # 将所有 chunk 转换成 token 字符串（注意：这里有潜在的效率问题，但忠于你的原始代码）
        inputs = self.tokenizer(
            [' '.join(self.tokenizer.convert_ids_to_tokens(c)) for c in all_chunks],
            return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        # BERT 批量推理
        with torch.no_grad():
            bert_outputs = self.bert(**inputs)
        pooled = self.bert_proj(bert_outputs.last_hidden_state[:, 0, :]) # Use [CLS] token

        # 重组：将 (TotalChunks, D) 恢复成 (B, max_chunks, D)
        max_chunks = max((m['n'] for m in mapping_info), default=1)
        final_embeddings = torch.zeros(batch_size, max_chunks, self.embed_dim, device=self.device)
        final_mask = torch.zeros(batch_size, max_chunks, device=self.device).bool()

        chunk_idx = 0
        for i in range(batch_size):
            # 查找索引为 i 的批次项有多少个 chunk
            num_chunks = next((m['n'] for m in mapping_info if m['index'] == i), 0)
            
            if num_chunks > 0:
                # 从 'pooled' 中提取这些 chunk 的 embedding
                patient_chunks = pooled[chunk_idx : chunk_idx + num_chunks]
                # 放入最终的张量
                final_embeddings[i, :num_chunks] = patient_chunks
                final_mask[i, :num_chunks] = True
                # 移动指针
                chunk_idx += num_chunks
        
        return final_embeddings, final_mask.bool()
    
    
    def encode(self, batch: Dict[str, Any]) -> Dict:
        """Dynamically encodes modalities based on what's present in the batch."""
        
        device = next(self.parameters()).device
        all_embeddings, all_masks = [], []
        present_modalities = []
        batch_size = len(batch.get('labels', []))

        if 'image-pathology' in self.active_modalities:
            wsi_tensors = batch['image-pathology'] # List[Optional[Tensor(N, D)]]
            valid_tensors = [t for t in wsi_tensors if t is not None]
            
            if valid_tensors:
                # Find max patches
                max_patches = max(t.shape[0] for t in valid_tensors)
                # Get the embedding dimension from the first valid tensor
                input_dim = valid_tensors[0].shape[1]
                
                padded_wsi, is_valid_wsi, patch_masks = [], [], []
                device = self.device # Assuming 'device' is accessible via 'self'
                
                for tensor in wsi_tensors:
                    if tensor is not None:
                        num_patches = tensor.shape[0]
                        pad_len = max_patches - num_patches
                        
                        # Pad the tensor data
                        padded = F.pad(tensor, (0, 0, 0, pad_len), 'constant', 0)
                        padded_wsi.append(padded.to(device))
                        is_valid_wsi.append(True)
                        
                        # Create a mask of 1s for real patches
                        mask = torch.ones(num_patches, device=device)
                        # Pad the mask with 0s for the padding
                        mask_padded = F.pad(mask, (0, pad_len), 'constant', 0)
                        patch_masks.append(mask_padded)    
                    else:
                        # Append a zero tensor for the WSI
                        padded_wsi.append(torch.zeros(max_patches, input_dim, device=device))
                        is_valid_wsi.append(False)
            
                        # Append a mask of all 0s for the absent WSI
                        patch_masks.append(torch.zeros(max_patches, device=device))

                wsi_batch = torch.stack(padded_wsi).to(device)
                
                # Stack the individual patch masks into a batch mask
                # This 'mask' will have shape (B, max_patches)
                mask = torch.stack(patch_masks).to(device)
                
                # Pass the calculated padding mask to the image_mil module
                wsi_token_embeds = self.image_mil(wsi_batch, mask) # (B, p (TransMIL) or 2 * p + 1 (PANTHER), D)
                
                # This mask (wsi_mask) is for *after* MIL, to mask out entire
                # absent slides, based on your original logic.
                wsi_mask = torch.tensor(is_valid_wsi, device=device).unsqueeze(1).expand(-1, wsi_token_embeds.shape[1])
                
                all_embeddings.append(wsi_token_embeds)
                all_masks.append(wsi_mask)
                present_modalities.append("image-pathology")

        # --- 2.  Genomics Branch ---
        if 'genomics-genomics' in self.active_modalities:
            rna_tensors = batch['genomics-genomics'] # List[Optional[Tensor(N, D)]]
            
            max_nodes = max(tensor.shape[0] for tensor in rna_tensors if tensor is not None) if any(t is not None for t in rna_tensors) else 0
            
            # [修改] 修复了当 rna_tensors 为空或全为 None 时的 max_nodes=0 错误
            if max_nodes == 0:
                raise ValueError("Error genomics")

            padded_rna = []
            rna_mask_list = []
            
            assert len(rna_tensors) > 0, f"Must have rna_tensors"
            for tensor in rna_tensors:
                if tensor is not None:

                    n_nodes = tensor.shape[0]
                    # 我们将截断/填充到 max_nodes
                    if n_nodes > max_nodes:
                        padded = tensor[:max_nodes].to(device)
                        mask = torch.ones(max_nodes, device=device).bool()
                    else:
                        pad_len = max_nodes - n_nodes
                        padded = F.pad(tensor, (0, 0, 0, pad_len), 'constant', 0).to(device)
                        mask = torch.zeros(max_nodes, device=device).bool()
                        mask[:n_nodes] = True
                    
                    padded_rna.append(padded)
                    rna_mask_list.append(mask)
                else:
                    # 缺少此模态的患者
                    padded_rna.append(torch.zeros(max_nodes, 1024, device=device)) # 1024 = input_dim
                    rna_mask_list.append(torch.zeros(max_nodes, device=device).bool())

            rna_batch = torch.stack(padded_rna).to(device) # (B, N, 1024)
            rna_mask = torch.stack(rna_mask_list).to(device) # (B, N)

            # 应用 Linear Encoder: (B, n, 1024) -> (B, n, 512)
            rna_token_embeds = self.genomics_encoder(rna_batch) 

            all_embeddings.append(rna_token_embeds)
            all_masks.append(rna_mask)
            present_modalities.append("genomics-genomics")

        # --- 3. Text Branch ---
        if 'text-pathology' in self.active_modalities and batch['text-pathology']:
            # _encode_text 期望一个 (B,) 的列表
            embeds, mask = self._encode_text(batch['text-pathology'])

            all_embeddings.append(embeds)
            all_masks.append(mask)
            present_modalities.append("text-pathology")

        if 'text-treatment' in self.active_modalities and batch['text-treatment']:
            # _encode_text 期望一个 (B,) 的列表
            embeds, mask = self._encode_text(batch['text-treatment'])

            all_embeddings.append(embeds)
            all_masks.append(mask)
            present_modalities.append("text-treatment")

        # --- 4. Tabular Branch ---
        for mod_name in self.active_modalities:
            if "tabular" in mod_name and mod_name in self.tabular_encoder and batch[mod_name]: 
                
                table_features = []
                table_masks = []
                
                # batch[mod_name] 是一个 List[Optional[Tensor(L,)]]
                for table_tensor in batch[mod_name]:
                    if table_tensor is not None:

                        # 确保它是正确类型和设备
                        modality_stack_tensor = table_tensor.to(device).float()
                        
                        # 编码
                        tabular_feature = self.tabular_encoder[mod_name](modality_stack_tensor)
                        tabular_feature = tabular_feature.reshape(1, 1, -1)  # (1, 1, D)
                        tabular_mask = torch.ones(1, 1, device=device).bool()
                    else: 
                        # 某些患者缺少此模态
                        tabular_feature = torch.zeros((1, 1, self.embed_dim), device=device).float()
                        tabular_mask = torch.zeros(1, 1, device=device).bool()
                    
                    table_features.append(tabular_feature)
                    table_masks.append(tabular_mask)

                # 堆叠 (B, 1, D)
                table_features_batch = torch.cat(table_features, dim=0)

                table_masks_batch = torch.cat(table_masks, dim=0)
                
                all_embeddings.append(table_features_batch)
                all_masks.append(table_masks_batch)
                present_modalities.append(mod_name)

        # --- Define Alignment Pairs ---
        align_pairs = []

        return {
            "embeddings": all_embeddings,
            "masks": all_masks,
            "align_pairs": align_pairs
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
        label_time_list = [batch['labels'][i]['label_time'] for i in range(batch_size)]
        label_event_list = [batch['labels'][i]['label_event'] for i in range(batch_size)]

        Y_full = torch.tensor(label_time_list, device=device, dtype=torch.long)
        c_full = torch.tensor(label_event_list, device=device, dtype=torch.long)

        valid_Y = Y_full[patient_mask]
        valid_c = c_full[patient_mask]

        # 5. 仅在有效子集上进行预测和损失计算
        valid_logits = self.prediction_head(valid_embeddings)
        loss_tensor_unreduced = self.loss_fn(valid_logits, valid_Y, valid_c)
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
        
    def get_backbone_params(self) -> List[nn.Parameter]:
        try:
            parms_in_clinical_bert = [p for p in self.bert.parameters()]
            return parms_in_clinical_bert
        except AttributeError:
            # text-pathology (bert) 未被激活
            return []
    
    def get_params(self) -> List[nn.Parameter]:
        backbone_params_ids = {id(p) for p in self.get_backbone_params()}
        parms_in_others = [p for p in self.parameters() if id(p) not in backbone_params_ids]
        return parms_in_others