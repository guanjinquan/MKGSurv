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
from modules.common_modules.surv_loss import NLLSurvLoss
from modules.training_utils.metrics import survival_metrics



# ==========================================================================================
# TransMIL Components for WSI Processing
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
    A modified TransMIL that aggregates information into K tokens.
    """
    def __init__(self, input_dim=1024, embed_dim=512, num_aggregated_tokens: int = 16):
        super(AggregatingTransMIL, self).__init__()
        self.num_aggregated_tokens = num_aggregated_tokens
        self.pos_layer = PPEG(num_aggregated_tokens=1, dim=embed_dim)
        self._fc1 = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layer1 = TransLayer(dim=embed_dim)
        self.layer2 = TransLayer(dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h):
        h = self._fc1(h)  # [B, n, embed_dim]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            # Pad with a subset of existing patches
            h = torch.cat([h, h[:, :add_length, :]], dim=1)

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h) #---->Translayer x1
        h = self.pos_layer(h, _H, _W)  #---->PPEG
        h = self.layer2(h)  #---->Translayer x2
        h = self.norm(h)  #---->Return K aggregated token embeddings
        return h[:, 0:self.num_aggregated_tokens, :]




# ==========================================================================================
# Main Encoder-Decoder Model for HANCOCK Dataset
# ==========================================================================================
class HANCOCKSurvivalPred(nn.Module):
    
    METRICS_FN = staticmethod(survival_metrics)

    def __init__(
        self,
        modalities: List[str]  # list of modalities to use according to the dataset class
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.embed_dim = 512

        # --- Modality Setup ---
        self.active_modalities = modalities
        self.max_modalities_num = len(self.active_modalities)  # For reference in fusion module
        print(f"Model initialized for modalities: {self.active_modalities}")

        # ----- WSI Branch (AggregatingTransMIL) -----
        if 'image-pathology' in self.active_modalities:
            self.wsi_mil = AggregatingTransMIL(
                input_dim=1024,
                embed_dim=self.embed_dim,
            )
            self.num_wsi_tokens = self.wsi_mil.num_aggregated_tokens

        # ----- Text Branch (ClinicalBERT) -----
        if 'text-clinical' in self.active_modalities:
            self.text_model_name = "medicalai/ClinicalBERT"
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, use_fast=True)
            self.bert = AutoModel.from_pretrained(self.text_model_name)
            bert_hidden_size = self.bert.config.hidden_size
            self.bert_proj = nn.Linear(bert_hidden_size, self.embed_dim) if bert_hidden_size != self.embed_dim else nn.Identity()

        # ----- Tabular Branch  -----
        self.tabular_encoder = nn.ModuleDict()
        for i, modality in enumerate(modalities):
            if "tabular" in modality:
                tabular_dim = int(modality.split("-")[-1])
                self.tabular_encoder[modality] = nn.Linear(tabular_dim, self.embed_dim)

        # ----- Prediction Head (for Decode step) -----
        self.prediction_head = nn.Linear(self.embed_dim, 10)  # Predicts risk for 10 time intervals
        self.loss_fn = NLLSurvLoss(reduction='none')

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
            # --- 新逻辑结束 ---

            # 记录这个批次项 (item) 总共产生了多少个 chunk
            if not item_specific_chunks:
                mapping_info.append({'index': i, 'n': 0})
                continue
                
            mapping_info.append({'index': i, 'n': len(item_specific_chunks)})
            all_chunks.extend(item_specific_chunks)

        # --- 从这里开始，你原有的代码逻辑完全不变 ---

        if not all_chunks:
            # 返回一个 (B, 1, D) 和 (B, 1) 的空张量，与你原始逻辑保持一致
            return torch.zeros(batch_size, 1, self.embed_dim, device=self.device), torch.zeros(batch_size, 1, device=self.device).bool()

        # 将所有 chunk 转换成 token 字符串（注意：这里有潜在的效率问题，但忠于你的原始代码）
        inputs = self.tokenizer(
            [' '.join(self.tokenizer.convert_ids_to_tokens(c)) for c in all_chunks],
            return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        # BERT 批量推理
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
        
        return final_embeddings, final_mask

    def encode(self, batch: Dict[str, Any]) -> Dict:
        """Dynamically encodes modalities based on what's present in the batch."""
        
        device = next(self.parameters()).device
        all_embeddings, all_masks = [], []
        present_modalities = []

        # --- 1. WSI Branch ---
        if 'image-pathology' in batch and batch['image-pathology']:
            wsi_tensors = batch['image-pathology']
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
                        padded_wsi.append(torch.zeros(max_patches, 1024, device=device))
                        is_valid_wsi.append(False)

                wsi_batch = torch.stack(padded_wsi).to(device)
                wsi_token_embeds = self.wsi_mil(wsi_batch)
                wsi_mask = torch.tensor(is_valid_wsi, device=device).unsqueeze(1).expand(-1, self.num_wsi_tokens)
                
                all_embeddings.append(wsi_token_embeds)
                all_masks.append(wsi_mask)
                present_modalities.append("image-pathology")

        # --- 2. Text Branch ---
        if 'text-clinical' in batch and batch['text-clinical']:
            embeds, mask = self._encode_text(batch['text-clinical'])
            all_embeddings.append(embeds)
            all_masks.append(mask)
            present_modalities.append("text-clinical")

        # --- 3. Tabuler Branch ---
        # ----- Tabular branch -----
        for i, modality in enumerate(self.active_modalities):
            if "tabular" in modality and modality in batch and batch[modality]:
                table_features = []
                table_masks = []
                for table in batch[modality]:
                    if table:
                        modality_stack_tensor = torch.tensor(table).to(device).float()
                        tabular_feature = self.tabular_encoder[modality](modality_stack_tensor).reshape(1, 1, -1)  # (1, 1, D)  B, N, D
                        tabular_mask = torch.ones(1, 1, device=self.device).bool()
                    else:  # Some patient missing modality
                        tabular_feature = torch.zeros((1, 1, self.embed_dim)).to(device).float()
                        tabular_mask = torch.zeros(1, 1, device=self.device).bool()
                    table_features.append(tabular_feature)
                    table_masks.append(tabular_mask)

                table_features = torch.cat(table_features, dim=0)
                table_masks = torch.cat(table_masks, dim=0)
                all_embeddings.append(table_features)
                all_masks.append(table_masks)
                present_modalities.append(modality)

        # --- Define Alignment Pairs ---
        align_pairs = []
        # if "images" in present_modalities and "strong_related_text" in present_modalities:
        #     image_idx = present_modalities.index("images")
        #     strong_text_idx = present_modalities.index("strong_related_text")
        #     align_pairs.append((image_idx, strong_text_idx))

        return {
            "embeddings": all_embeddings,
            "masks": all_masks,
            "align_pairs": align_pairs
        }

    def decode(self, pooled_embeddings: torch.Tensor, pooled_mask: Optional[torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Applies masking to the decoding process, calculating logits and loss only for valid (unmasked) data.

        Args:
            pooled_embeddings: Tensor of shape (B, embed_dim) containing patient embeddings.
            pooled_mask: Optional boolean tensor of shape (B,) where True indicates a valid patient.
                         If None, all patients are considered valid.
            batch: A list of dictionary containing labels, including 'label_Y' and 'label_c'.

        Returns:
            A dictionary containing:
            - 'logits': Tensor of shape (B, out_dim) with predictions. Logits for masked-out
                        patients will be zero.
            - 'loss': A scalar tensor representing the loss, calculated only on the valid data.
        """
        batch_size = pooled_embeddings.shape[0]
        device = pooled_embeddings.device

        # Assuming the prediction head outputs a single score (out_dim = 1)
        out_dim = self.prediction_head.out_features
        logits = torch.zeros(batch_size, out_dim, device=device)
        loss_tensor = torch.zeros((batch_size, 1), device=device)
        loss = torch.tensor(0.0, device=device)

        # 1. Create a boolean mask for valid (present) patients.
        # If pooled_mask is None, we assume all data in the batch is valid.
        patient_mask = pooled_mask.bool().to(device) if pooled_mask is not None else torch.ones(batch_size, dtype=torch.bool, device=device)

        # 2. If no patients are valid in this batch, return zeros immediately.
        if not patient_mask.any():
            return {"logits": logits, "loss": loss, 'loss_tensor': loss_tensor}

        # 3. Filter the embeddings and labels to only include the valid data.
        valid_embeddings = pooled_embeddings[patient_mask]

        label_Y_list = [batch['labels'][i]['label_Y'] for i in range(batch_size)]
        label_c_list = [batch['labels'][i]['label_c'] for i in range(batch_size)]

        Y_full = torch.tensor(label_Y_list).to(device).to(torch.long)
        c_full = torch.tensor(label_c_list).to(device).to(torch.long)

        valid_Y = Y_full[patient_mask]
        valid_c = c_full[patient_mask]

        # 4. Perform prediction and loss calculation only on the valid subset.
        valid_logits = self.prediction_head(valid_embeddings)
        loss_tensor = self.loss_fn(valid_logits, None, valid_Y, valid_c)
        loss = loss_tensor.mean()

        # 5. Place the calculated logits for the valid data back into the original tensor.
        # The positions for masked-out data remain zero.
        logits[patient_mask] = valid_logits

        return {"logits": logits, "loss": loss, "loss_tensor": loss_tensor}

    def get_backbone_params(self) -> List[nn.Parameter]:
        parms_in_clinical_bert = [p for p in self.bert.parameters()]
        return parms_in_clinical_bert
    
    def get_others_params(self) -> List[nn.Parameter]:
        backbone_params = set(self.get_backbone_params())
        parms_in_others = [p for p in self.parameters() if p not in backbone_params]
        return parms_in_others



if __name__ == '__main__':
    import os
    os.chdir("/home/Guanjq/NewWork/MedAlignFusion/Code")
    # ==========================================================================================
    # Debugging and Testing Block
    # ==========================================================================================
    
    # 确保可以找到 hancock_dataset 模块。
    # 假设项目结构为 MedAlignFusion/Code/datasets/ 和 MedAlignFusion/Code/modules/
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
    from datasets.hancock_dataset import HANCOCKDataset, hancock_custom_collate_fn
    from torch.utils.data import DataLoader

    # 切换到 Code 目录以正确解析相对数据路径 ../Data/HANCOCK
    try:
        os.chdir(os.path.join(os.path.dirname(__file__), '../../'))
        print(f"Current working directory: {os.getcwd()}")
    except FileNotFoundError:
        print("Error: Could not change directory. Please run this script from its original location.")
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f"调试将在设备上运行: {device}")

    # 1. 实例化数据集和数据加载器
    print("\n[1/4] 初始化数据集和数据加载器...")
    try:
        # 使用 "all" 模态进行测试以检查完整逻辑
        dataset = HANCOCKDataset(mode='train', modalities='all')
        if len(dataset) == 0:
            raise ValueError("数据集为空。请检查数据路径和拆分配置。")
        
        data_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False, # 为了可复现的调试，设置为 False
            collate_fn=hancock_custom_collate_fn
        )
        print("数据集和数据加载器初始化成功。")
    except Exception as e:
        print(f"初始化数据加载器时出错: {e}")
        print("请确保 'hancock_dataset.py' 位于正确的路径并且数据文件存在。")
        exit()

    # 2. 实例化模型
    print("\n[2/4] 初始化 HANCOCKSurvivalPred 模型...")
    try:
        model = HANCOCKSurvivalPred(modalities='all').to(device)
        model.eval() # 设置为评估模式进行测试
        print("模型初始化成功。")
    except Exception as e:
        print(f"初始化模型时出错: {e}")
        exit()

    # 3. 获取一个批次并运行 encode 函数
    print("\n[3/4] 获取一个批次并运行 encode() 方法...")
    try:
        with torch.no_grad(): # 禁用梯度计算以进行推理
            batch = next(iter(data_loader))
            
            encoded_output = model.encode(batch)

            print("encode() 方法执行成功。")
            print("--- 编码输出 ---")
            print(f"嵌入张量数量: {len(encoded_output['embeddings'])}")
            print(f"掩码张量数量: {len(encoded_output['masks'])}")
            
            for i, (emb, mask) in enumerate(zip(encoded_output['embeddings'], encoded_output['masks'])):
                print(f"  - 模态 {i}:")
                if emb is not None:
                    print(f"    - 嵌入形状: {emb.shape}")
                    print(f"    - 掩码形状: {mask.shape}")
                    print(f"    - 掩码数据类型: {mask.dtype}")
                else:
                    print("    - 此批次中不存在此模态。")
            
            print(f"对齐对: {encoded_output['align_pairs']}")

    except Exception as e:
        import traceback
        print(f"在 encode() 过程中出错: {e}")
        traceback.print_exc()
        exit()
        
    # 4. 模拟融合并运行 decode 函数
    print("\n[4/4] 模拟融合并运行 decode() 方法...")
    try:
        with torch.no_grad():
            # 这是一个用于测试 decode 函数接口的伪融合过程。
            # 实际的融合逻辑会更复杂。
            batch_size = batch['label_Y'].shape[0]
            dummy_fused_embedding = torch.randn(batch_size, model.embed_dim).to(device)
            print("为测试 decode() 创建了一个虚拟的融合嵌入。")
            
            decoded_output = model.decode(dummy_fused_embedding, batch)
            
            print("decode() 方法执行成功。")
            print("--- 解码输出 ---")
            print(f"风险分数形状: {decoded_output['risk'].shape}")
            print(f"计算出的损失: {decoded_output['loss'].item():.4f}")
            print("\n调试测试成功完成！")

    except Exception as e:
        import traceback
        print(f"在 decode() 过程中出错: {e}")
        traceback.print_exc()
