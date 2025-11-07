import os
import sys
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
from modules.common_modules.surv_loss import CustomCoxPHLoss
from training_utils.metrics import survival_metrics


# ==========================================================================================
# TransMIL Components (Copied from HANCOCK/OSCC Model)
# ==========================================================================================
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x

class PPEG(nn.Module):
    def __init__(self, num_aggregated_tokens=128, dim=512):
        super(PPEG, self).__init__()
        self.num_aggregated_tokens = num_aggregated_tokens
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, :self.num_aggregated_tokens], x[:, self.num_aggregated_tokens:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token, x), dim=1)
        return x


class AggregatingTransMIL(nn.Module):
    """
    一个通用的 MIL 聚合器，将 (B, N, input_dim) 聚合成 (B, num_aggregated_tokens, embed_dim)
    """
    def __init__(self, input_dim=1024, embed_dim=512, num_aggregated_tokens: int = 16):
        super(AggregatingTransMIL, self).__init__()
        self.num_aggregated_tokens = num_aggregated_tokens
        self.pos_layer = PPEG(num_aggregated_tokens=num_aggregated_tokens, dim=embed_dim) # 确保 PPEG 知道有多少 token
        self._fc1 = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, self.num_aggregated_tokens, embed_dim)) # K 个 tokens
        self.layer1 = TransLayer(dim=embed_dim)
        self.layer2 = TransLayer(dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h):
        # 检查输入
        # TCGA_LUAD_SurvivalPred.check_nan_inf(h, "AggregatingTransMIL Input (h)")

        h = self._fc1(h)  # [B, n, embed_dim]
        # TCGA_LUAD_SurvivalPred.check_nan_inf(h, "AggregatingTransMIL After _fc1")
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h) #---->Translayer x1
        # TCGA_LUAD_SurvivalPred.check_nan_inf(h, "AggregatingTransMIL After layer1")

        h = self.pos_layer(h, _H, _W)  #---->PPEG
        # TCGA_LUAD_SurvivalPred.check_nan_inf(h, "AggregatingTransMIL After pos_layer")
        
        h = self.layer2(h)  #---->Translayer x2
        # TCGA_LUAD_SurvivalPred.check_nan_inf(h, "AggregatingTransMIL After layer2")

        h = self.norm(h)  #---->Return K aggregated token embeddings
        # TCGA_LUAD_SurvivalPred.check_nan_inf(h, "AggregatingTransMIL After norm (Output)")
        
        return h[:, 0:self.num_aggregated_tokens, :]




# ==========================================================================================
# Main Encoder-Decoder Model for TCGA-LUAD Dataset
# ==========================================================================================
class TCGA_LUAD_SurvivalPred(nn.Module):
    
    METRICS_FN = staticmethod(survival_metrics)

    def __init__(
        self,
        modalities: List[str],  # 来自 TCGA_LUAD_Dataset 的模态列表
        # tabular_dims: Dict[str, int] = None # <-- 不再需要
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.embed_dim = 512

        # --- Modality Setup ---
        self.active_modalities = modalities
        self.max_modalities_num = len(self.active_modalities)
        print(f"Model initialized for modalities: {self.active_modalities}")

        # ----- 1. (Graph) Image Branch (image-pathology) -----
        if 'image-pathology' in self.active_modalities:
            print("Initializing Image MIL (AggregatingTransMIL)")
            self.image_mil = AggregatingTransMIL(
                input_dim=1024,
                embed_dim=self.embed_dim,
                num_aggregated_tokens=16 # 16 tokens for image
            )
            self.num_image_tokens = self.image_mil.num_aggregated_tokens

        # ----- 2. (Graph) Genomics Branch (genomics-genomics) -----
        # --- MODIFIED: Per user request, use simple Linear for N=5 tokens, not MIL ---
        if 'genomics-genomics' in self.active_modalities:
            print("Initializing Genomics Encoder (Linear layer for N=5 tokens)")
            self.genomics_encoder = nn.Sequential(
                        nn.LayerNorm(1024),
                        nn.Linear(1024, self.embed_dim)
                    )

        # ----- 3. Text Branch (text-pathology) -----
        if 'text-pathology' in self.active_modalities:
            print("Initializing Text Encoder (ClinicalBERT)")
            self.text_model_name = "medicalai/ClinicalBERT"
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, use_fast=True)
            self.bert = AutoModel.from_pretrained(self.text_model_name)
            bert_hidden_size = self.bert.config.hidden_size
            self.bert_proj = nn.Linear(bert_hidden_size, self.embed_dim) if bert_hidden_size != self.embed_dim else nn.Identity()

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
                except (ValueError, IndexError):
                    print(f"ERROR: Could not parse dimension from tabular modality name: '{mod_name}'")
                    print("Expected format: 'tabular-type-DIMENSION' (e.g., 'tabular-clinical-56')")


        # ----- Prediction Head (for Decode step) -----
        self.prediction_head = nn.Linear(self.embed_dim, 1)   # Predicts risk [0 means low risk to death/recurrence, 1 means high risk]
        self.loss_fn = CustomCoxPHLoss(reduction='none')

    def _chunk_token_ids(self, ids: List[int], chunk_size: int) -> List[List[int]]:
        """Splits a list of token ids into chunks."""
        return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    def _encode_text(self, texts_list: List[Optional[str | List[str]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes a batch of texts, handling List[str] as separate items and chunking long inputs."""
        batch_size = len(texts_list)
        chunk_payload = 510  # 512 - 2 for [CLS] and [SEP]
        
        all_chunks = []
        mapping_info = [] # (original_batch_index, num_chunks)
        
        for i, item in enumerate(texts_list):
            
            item_specific_chunks = [] 
            texts_to_process = []
            
            if isinstance(item, str) and item.strip():
                texts_to_process.append(item)
            elif isinstance(item, list):
                texts_to_process.extend([t for t in item if isinstance(t, str) and t.strip()])
            
            if not texts_to_process:
                mapping_info.append({'index': i, 'n': 0})
                continue
            
            for text in texts_to_process:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                chunks = self._chunk_token_ids(token_ids, chunk_payload) 
                if chunks:
                    item_specific_chunks.extend(chunks)

            if not item_specific_chunks:
                mapping_info.append({'index': i, 'n': 0})
                continue
                
            mapping_info.append({'index': i, 'n': len(item_specific_chunks)})
            all_chunks.extend(item_specific_chunks)

        if not all_chunks:
            return torch.zeros(batch_size, 1, self.embed_dim, device=self.device), torch.zeros(batch_size, 1, device=self.device).bool()

        # 将所有 chunk 转换成 token 字符串
        # 注意：这里直接传递 token IDs (int) 列表给 tokenizer 会更高效
        # 但为了保持与 HANCOCK 代码一致，我们使用 convert_ids_to_tokens
        inputs = self.tokenizer(
            [' '.join(self.tokenizer.convert_ids_to_tokens(c)) for c in all_chunks],
            return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        bert_outputs = self.bert(**inputs)
        
        # [新增] 检查 BERT 输出
        # self.check_nan_inf(bert_outputs.last_hidden_state, "_encode_text BERT last_hidden_state")

        pooled = self.bert_proj(bert_outputs.last_hidden_state[:, 0, :]) # Use [CLS] token
        
        # [新增] 检查 BERT 投影后的输出
        # self.check_nan_inf(pooled, "_encode_text BERT projected [CLS] (pooled)")

        max_chunks = max((m['n'] for m in mapping_info), default=1)
        final_embeddings = torch.zeros(batch_size, max_chunks, self.embed_dim, device=self.device)
        final_mask = torch.zeros(batch_size, max_chunks, device=self.device).bool()

        chunk_idx = 0
        for i in range(batch_size):
            num_chunks = next((m['n'] for m in mapping_info if m['index'] == i), 0)
            
            if num_chunks > 0:
                patient_chunks = pooled[chunk_idx : chunk_idx + num_chunks]
                final_embeddings[i, :num_chunks] = patient_chunks
                final_mask[i, :num_chunks] = True
                chunk_idx += num_chunks
        
        # [新增] 检查最终的文本 embedding
        # self.check_nan_inf(final_embeddings, "_encode_text Final Embeddings")
        
        return final_embeddings, final_mask

    def encode(self, batch: Dict[str, Any]) -> Dict:
        """Dynamically encodes modalities based on what's present in the batch."""
        
        device = next(self.parameters()).device
        all_embeddings, all_masks = [], []
        present_modalities = []
        batch_size = len(batch.get('labels', []))

        # --- 1. (Graph) Image Branch ---
        if 'image-pathology' in self.active_modalities:
            wsi_tensors = batch['image-pathology'] # List[Optional[Tensor(N, D)]]
            valid_tensors = [t for t in wsi_tensors if t is not None]
            
            if valid_tensors:
                max_patches = max(t.shape[0] for t in valid_tensors)
                padded_wsi, is_valid_wsi = [], []
                
                for tensor in wsi_tensors:
                    if tensor is not None:
                        pad_len = max_patches - tensor.shape[0]
                        padded = F.pad(tensor, (0, 0, 0, pad_len), 'constant', 0)
                        padded_wsi.append(padded.to(device))
                        is_valid_wsi.append(True)
                    else:
                        padded_wsi.append(torch.zeros(max_patches, 1024, device=device)) # 1024 = input_dim
                        is_valid_wsi.append(False)

                wsi_batch = torch.stack(padded_wsi).to(device)
                wsi_token_embeds = self.image_mil(wsi_batch) # (B, num_image_tokens, D)
                wsi_mask = torch.tensor(is_valid_wsi, device=device).unsqueeze(1).expand(-1, self.num_image_tokens)
                
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
            # [新增] 检查填充后的 Genomics batch
            # self.check_nan_inf(rna_batch, "encode Genomics Padded Batch")

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

        # --- 4. Tabular Branch ---
        for mod_name in self.active_modalities:
            if "tabular" in mod_name and mod_name in self.tabular_encoder:
                
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
                # [新增] 检查 Tabular encoder 的输出
                # self.check_nan_inf(table_features_batch, f"encode {mod_name} Encoder Output")
                
                table_masks_batch = torch.cat(table_masks, dim=0)
                
                all_embeddings.append(table_features_batch)
                all_masks.append(table_masks_batch)
                present_modalities.append(mod_name)

        # --- Define Alignment Pairs ---
        align_pairs = []
        # (在此处定义您的模态对齐逻辑)
        # e.g.,
        # if "image-pathology" in present_modalities and "text-pathology" in present_modalities:
        #     img_idx = present_modalities.index("image-pathology")
        #     txt_idx = present_modalities.index("text-pathology")
        #     align_pairs.append((img_idx, txt_idx))

        # [新增] 最终检查所有 embedding
        # for i, emb in enumerate(all_embeddings):
        #     self.check_nan_inf(emb, f"encode Final all_embeddings[{i}] ({present_modalities[i]})")

        return {
            "embeddings": all_embeddings,
            "masks": all_masks,
            "align_pairs": align_pairs
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

        out_dim = self.prediction_head.out_features
        logits = torch.zeros(batch_size, out_dim, device=device)
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
        loss = loss_tensor_unreduced.mean()

        # 6. 将 logits 映射回原始 (B, out_dim) 张量
        logits[patient_mask] = valid_logits

        # 创建一个 (B,) 的 loss_tensor 以保持一致性 (可选)
        full_loss_tensor = torch.zeros(batch_size, device=device)
        full_loss_tensor[patient_mask] = loss_tensor_unreduced.squeeze(1) if loss_tensor_unreduced.dim() == 2 else loss_tensor_unreduced

        return {"logits": logits, "loss": loss, "loss_tensor": full_loss_tensor}

    def get_backbone_params(self) -> List[nn.Parameter]:
        try:
            parms_in_clinical_bert = [p for p in self.bert.parameters()]
            return parms_in_clinical_bert
        except AttributeError:
            # text-pathology (bert) 未被激活
            return []
    
    def get_others_params(self) -> List[nn.Parameter]:
        backbone_params = set(self.get_backbone_params())
        parms_in_others = [p for p in self.parameters() if p not in backbone_params]
        return parms_in_others