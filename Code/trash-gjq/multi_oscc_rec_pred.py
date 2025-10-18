import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Add parent directories to sys.path to handle relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import json
import torch
import torch.nn as nn
from PIL import Image
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

# Import the new custom model wrapper
import open_clip
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)

from collections import deque
import math


class MultiOSCCRecPred(nn.Module):

    def __init__(self, device: str = "cuda", modalities: str = "all"):
        super(MultiOSCCRecPred, self).__init__()
        self.device = torch.device(device)

        valid_modalities = [
            "all", "image", "strong_related_text", "weak_related_text",
            "image,strong_related_text", "image,weak_related_text", "strong_related_text,weak_related_text"
        ]
        cleaned_modalities = ",".join(modalities.split('-'))
        assert cleaned_modalities in valid_modalities, f"Invalid modalities specified: {modalities}"
        self.modalities = cleaned_modalities
        self.max_modalities_num = 3 if self.modalities == 'all' else len(modalities.split(','))
        
        # --- 1. Initialize CustomBiomedCLIP model ---
        checkpoint_dir = "../PretrainedWeights/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

        with open(f"{checkpoint_dir}/open_clip_config.json", "r") as f:
            config = json.load(f)
            model_cfg = config["model_cfg"]
            preprocess_cfg = config["preprocess_cfg"]

        model_name = "biomedclip_local"
        if (not model_name.startswith(HF_HUB_PREFIX)
            and model_name not in _MODEL_CONFIGS
            and config is not None):
            _MODEL_CONFIGS[model_name] = model_cfg

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=f"{checkpoint_dir}/open_clip_pytorch_model.bin",
            **{f"image_{k}": v for k, v in preprocess_cfg.items()},
        )

        # --- 2. Define classifier and loss ---
        # token-level embedding dimension (ViT-Base / PubMedBERT typical)
        self.embed_dim = 512
        self.out_dim = 2

        self.classifier =  nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(self.embed_dim // 2, self.out_dim)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()


        # ---------- weight check config ----------
        # 阈值（默认 1e-6），比较的步数（i 与 i-steps 比较）
        self.weight_check_threshold = 1e-6
        self.weight_check_steps = 3

        # 历史记录队列（保存每次 encode 时的 norm）
        # 不设 maxlen（或可设为较大数）以便保留长期历史
        self._weight_norm_history = deque()

        # 变动记录列表（当检测到变化时 append 一个 dict）
        self.weight_change_records = []

    def get_backbone_params(self):
        return self.model.parameters()
    
    def _encode_text(self, text_list: List[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a list of texts, handling empty or None strings by returning zero vectors.
        """
        batch_size = len(text_list)
        context_length = self.tokenizer.context_length - 3  # Account for special tokens
        stride = context_length // 2
        pad_token_id = self.tokenizer.tokenizer.pad_token_id if hasattr(self.tokenizer.tokenizer, 'pad_token_id') and self.tokenizer.tokenizer.pad_token_id is not None else 0

        # --- Step 1: Identify valid texts and their original indices ---
        valid_texts_with_indices = []
        for i, text in enumerate(text_list):
            if text and text.strip():  # Check for non-empty, non-whitespace strings
                valid_texts_with_indices.append({'index': i, 'text': text})

        # --- Edge Case: If the entire batch is empty ---
        if not valid_texts_with_indices:
            # Return zero tensors with a sequence length of 1
            seq_len = 1
            final_embeddings = torch.zeros(batch_size, seq_len, self.embed_dim, device=self.device)
            final_mask = torch.zeros(batch_size, seq_len, device=self.device)
            return final_embeddings, final_mask

        # --- Step 2: Process only the valid texts ---
        valid_texts = [item['text'] for item in valid_texts_with_indices]
        
        num_of_embeddings_valid = []
        final_tokens_list_valid = []

        for text in valid_texts:
            text_tokens = self.tokenizer.tokenizer.encode(text, truncation=False)

            if len(text_tokens) <= context_length:
                chunk = text_tokens + [pad_token_id] * (context_length - len(text_tokens))
                final_tokens_list_valid.append(torch.tensor(chunk, device=self.device).unsqueeze(0))
                num_of_embeddings_valid.append(1)
            else:
                chunks_for_this_text = []
                for i in range(0, len(text_tokens), stride):
                    if i + context_length > len(text_tokens):
                        chunk = text_tokens[-context_length:]
                        chunks_for_this_text.append(torch.tensor(chunk, device=self.device).unsqueeze(0))
                        break
                    else:
                        chunk = text_tokens[i:i + context_length]
                        chunks_for_this_text.append(torch.tensor(chunk, device=self.device).unsqueeze(0))
                
                final_tokens_list_valid.extend(chunks_for_this_text)
                num_of_embeddings_valid.append(len(chunks_for_this_text))
        
        final_tokens_tensor = torch.cat(final_tokens_list_valid, dim=0)
        # Encode all valid chunks in one go
        valid_text_features = self.model.text(final_tokens_tensor)

        # --- Step 3: Reconstruct the full batch, inserting zero vectors for empty texts ---
        final_embeddings_list = []
        final_mask_list = []
        
        max_num = max(num_of_embeddings_valid) if num_of_embeddings_valid else 1
        
        current_valid_pos = 0
        valid_item_counter = 0
        original_indices_of_valid = {item['index'] for item in valid_texts_with_indices}

        for i in range(batch_size):
            if i in original_indices_of_valid:
                # This was a valid text; retrieve its features
                num_chunks = num_of_embeddings_valid[valid_item_counter]
                features_for_text = valid_text_features[current_valid_pos : current_valid_pos + num_chunks]
                current_valid_pos += num_chunks
                valid_item_counter += 1

                mask = torch.ones(features_for_text.shape[0], device=self.device)
                
                # Pad the sequence of embeddings and mask to the max length in the batch
                if features_for_text.shape[0] < max_num:
                    pad_len = max_num - features_for_text.shape[0]
                    padding_features = torch.zeros(pad_len, self.embed_dim, device=self.device)
                    features_for_text = torch.cat([features_for_text, padding_features], dim=0)
                    padding_mask = torch.zeros(pad_len, device=self.device)
                    mask = torch.cat([mask, padding_mask], dim=0)

                final_embeddings_list.append(features_for_text)
                final_mask_list.append(mask)
            else:
                # This was an empty text; append zero tensors
                final_embeddings_list.append(torch.zeros(max_num, self.embed_dim, device=self.device))
                final_mask_list.append(torch.zeros(max_num, device=self.device))

        final_embeddings = torch.stack(final_embeddings_list)
        final_mask = torch.stack(final_mask_list)

        # Final check for safety
        if torch.isnan(final_embeddings).any():
            print("NaN detected in embeddings even after fix:")
            print(text_list)
            raise ValueError("NaN detected in embeddings post-fix")

        return final_embeddings, final_mask

    def encode(self, batch: Dict) -> Dict:
        """
        Encodes a collated batch of data, returning a dictionary containing TOKEN-LEVEL
        embeddings (with Nones for missing modalities) and a list of strongly-related pairs.

        Args:
            batch (Dict): A dictionary from the custom collate function with keys:
                          'images': List[List[PIL.Image]],
                          'labels': torch.Tensor,
                          'strong_related_text': List[Optional[str]],
                          'weak_related_text': List[Optional[str]]
        """
        # --- 自动 weight check: 每次 encode 调用都会记录一次 norm 并和 i-5 做对比 ---
        try:
            check_res = self._auto_weight_check()
            # 只在 "checked" 且检测到变化时额外打印详情（上面函数已经做了基本打印）
            if check_res.get("status") == "checked" and check_res.get("changed", False):
                # 这里可以做额外处理（例如：保存 snapshot, 发出告警, 写文件等）
                # 示例：保存到磁盘（可选）
                # import json; open('weight_change_records.json','w').write(json.dumps(self.weight_change_records[-1]))
                pass
        except Exception as e:
            # 防护：若计算 norm 出错，不影响后续 encode 正常工作
            print(f"[WEIGHT-CHECK][ERROR] _auto_weight_check failed: {e}")



        batch_size = len(batch.get('labels', []))
        if batch_size == 0:
            return {"embeddings": [None, None, None], "masks": [None, None, None], "strong_related_pairs": []}
        
        # --- Initialize all features to None ---
        image_features, strong_text_features, weak_text_features = None, None, None
        image_mask, strong_text_mask, weak_text_mask = None, None, None

        # --- Conditionally encode based on self.modalities ---
        # Tune the backbone params of biomedclip  # don't use with torch.no_grad():
        # with torch.no_grad():
        if 'image' in self.modalities or self.modalities == 'all':
            # batch['images'] is a list of lists: [[p1_img1, p1_img2..], [p2_img1, p2_img2..]]
            list_of_image_lists = batch.get('images', [])
            
            # Check if there are patients and if the first patient has images
            if list_of_image_lists and isinstance(list_of_image_lists[0], list) and list_of_image_lists[0]:
                num_images_per_patient = len(list_of_image_lists[0])
                # Flatten the list of lists into a single list of images
                all_images = [img for patient_images in list_of_image_lists for img in patient_images]
                
                if all_images:
                    images_tensor = torch.stack([self.preprocess(img) for img in all_images]).to(self.device)
                    # The following reshape assumes self.model.visual returns a single feature vector per image.
                    image_features_raw = self.model.visual(images_tensor) 
                    image_features = image_features_raw.reshape(batch_size, num_images_per_patient, self.embed_dim)
                    image_mask = torch.ones(batch_size, num_images_per_patient).to(self.device)

        if 'strong_related_text' in self.modalities or self.modalities == 'all':
            # Get list of texts, which may contain None for missing clinical data
            all_strong_texts = batch.get('strong_related_text', [])
            # The _encode_text function now handles None or empty strings internally
            if all_strong_texts:
                strong_text_features, strong_text_mask = self._encode_text(all_strong_texts)
        
        if 'weak_related_text' in self.modalities or self.modalities == 'all':
            all_weak_texts = batch.get('weak_related_text', [])
            if all_weak_texts:
                weak_text_features, weak_text_mask = self._encode_text(all_weak_texts)

        # --- Consolidate embeddings and identify strong pairs ---
        all_embeddings = [image_features, strong_text_features, weak_text_features]
        all_masks = [image_mask, strong_text_mask, weak_text_mask]
        strong_related_pairs = []
        
        if image_features is not None and strong_text_features is not None:
            strong_related_pairs.append((0, 1))


        # Debug check there is no nan in embeddings
        for i, emb in enumerate(all_embeddings):
            if emb is not None:
                if torch.isnan(emb).any():
                    print(f"[ENCODE][ERROR] NaN detected in embedding {i}")

        return {
            "embeddings": all_embeddings, 
            "masks": all_masks,
            "strong_related_pairs": strong_related_pairs
        }


    def decode(self, pooled_embeddings: torch.Tensor, pooled_mask: Optional[torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decodes POOLED embeddings into logits and computes loss, handling missing modalities.
        Args:
            pooled_embeddings (torch.Tensor): A tensor of pooled embeddings for each patient in the batch.
            pooled_mask (Optional[torch.Tensor]): A boolean tensor indicating which patients are valid.
                                                  If None, all patients are assumed to be valid.
            labels (torch.Tensor): The ground truth labels for the batch.
        """
        batch_size = pooled_embeddings.shape[0]
        device = pooled_embeddings.device
        
        labels = labels.to(device)

        # Initialize outputs
        logits = torch.zeros(batch_size, self.out_dim, device=device)
        loss = torch.tensor(0.0, device=device)

        # If pooled_mask is None, all patients are considered valid.
        if pooled_mask is None:
            patient_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            patient_mask = pooled_mask.bool().to(device)
        
        # If no patients have valid data, return zeros
        if not patient_mask.any():
            return {'logits': logits, 'loss': loss}

        # Select data only for valid patients
        valid_embeddings = pooled_embeddings[patient_mask]
        valid_labels = labels[patient_mask]

        # Get logits for valid patients (input is already pooled)
        valid_logits = self.classifier(valid_embeddings)
        
        # Compute loss only on valid patients
        # The target labels are likely already one-hot encoded. Applying F.one_hot again
        # was causing a shape mismatch error.
        if valid_labels.ndim == 1:
            target_labels = F.one_hot(valid_labels, num_classes=self.out_dim).float()
        else:
            target_labels = valid_labels.float()
        loss = self.loss_fn(valid_logits, target_labels)

        # Place the computed logits back into the full-batch tensor
        logits[patient_mask] = valid_logits
        
        return {'logits': logits, 'loss': loss}
    
    def get_metrics(self, logits, labels):

        """
        计算多分类/二分类任务的各项评估指标。
        此函数兼容两种标签格式:
        1. 单类别索引 (e.g., [0, 1, 0, ...])
        2. One-hot 编码 (e.g., [[1, 0], [0, 1], [1, 0], ...])

        Args:
            logits (list or np.ndarray): 模型的原始输出，形状为 (样本数, 类别数)。
            labels (list or np.ndarray): 真实的标签。

        Returns:
            dict: 包含 Accuracy, MacroF1, MacroRecall, MacroPrecision, 和 AUC 的字典。
        """
        # 步骤 1: 将输入转换为 NumPy 数组
        logits_np = np.array(logits)
        labels_np = np.array(labels)

        # --- 核心修改：处理 one-hot 编码的标签 ---
        # 检查标签是否是 one-hot 格式 (二维数组)
        if labels_np.ndim == 2 and labels_np.shape[1] > 1:
            # 如果是，通过 argmax 将其转换回单类别索引格式，以便与 y_pred 进行比较
            labels_for_metrics = np.argmax(labels_np, axis=1)
        else:
            # 如果已经是单类别索引，直接使用
            labels_for_metrics = labels_np
        
        # 步骤 2: 从 logits 中获取预测类别 (这部分逻辑保持不变)
        # np.argmax 返回最可能类别的索引
        y_pred = np.argmax(logits_np, axis=1)

        # 步骤 3: 使用 Softmax 将 logits 转换为概率，用于计算 AUC
        # Softmax 适用于评估 "选择一个最可能" 的场景
        logits_tensor = torch.tensor(logits_np, dtype=torch.float32)
        y_prob = F.softmax(logits_tensor, dim=1).numpy()
        
        # 步骤 4: 使用转换后的单类别标签计算各项指标
        # 设置 zero_division=0 可以在某个类别没有被预测到时，避免警告，并将其 F1 等指标记为0
        acc = accuracy_score(labels_for_metrics, y_pred)
        macro_f1 = f1_score(labels_for_metrics, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(labels_for_metrics, y_pred, average='macro', zero_division=0)
        macro_precision = precision_score(labels_for_metrics, y_pred, average='macro', zero_division=0)

        # 步骤 5: 计算 AUC
        num_classes = logits_np.shape[1]
        if num_classes > 2:
            try:
                # One-vs-Rest 宏平均 AUC
                auc = roc_auc_score(labels_for_metrics, y_prob, multi_class='ovr', average='macro')
            except ValueError:
                # 如果某个数据分割中只包含一个类别，AUC无法计算，这里将其设为0
                auc = 0.0
        elif num_classes == 2:
            # 二分类问题，直接使用正类(类别1)的概率计算
            auc = roc_auc_score(labels_for_metrics, y_prob[:, 1])
        else:
            # 类别数小于2，无法计算AUC
            auc = 0.0

        # 步骤 6: 整理成字典返回
        metrics = {
            "Acc": acc,
            "F1": macro_f1,
            "Recall": macro_recall,
            "Precision": macro_precision,
            "AUC": auc
        }
        return metrics
    
    def _compute_weight_norm(self, backbone_only: bool = True) -> float:
        """
        计算当前（backbone 或 全模型）权重 L2-norm（返回 Python float）。
        使用 detach().cpu() 安全地搬到 CPU 计算，不会改变梯度/设备状态。
        """
        raw_model = getattr(self.model, "module", self.model)
        if backbone_only and hasattr(raw_model, "get_backbone_params"):
            params = list(raw_model.get_backbone_params())
        else:
            params = list(raw_model.parameters())

        total_sq = 0.0
        for p in params:
            if p is None:
                continue
            # 安全取值（detach -> cpu -> float tensor）
            tensor = p.detach().cpu().float()
            if tensor.numel() == 0:
                continue
            total_sq += float(torch.sum(tensor * tensor).item())
        return math.sqrt(total_sq)

    def _auto_weight_check(self, threshold: Optional[float] = 1e-6, steps: Optional[int] = 3, backbone_only: bool = True) -> Dict:
        """
        自动检测：在每次 encode 被调用时调用此函数。
        - 记录当前 norm 到 self._weight_norm_history
        - 若历史长度 >= steps + 1，则比较当前（i）与 i-steps 的差值
        - 若超过 threshold，会记录一条变动记录到 self.weight_change_records 并打印告警

        返回格式示例：
          {'status':'counting', 'idx': i, 'norm': cur_norm}
          {'status':'checked', 'changed': True/False, 'pre_norm':.., 'post_norm':.., 'delta':.., ...}
        """
        if threshold is None:
            threshold = self.weight_check_threshold
        if steps is None:
            steps = self.weight_check_steps

        cur_norm = float(self._compute_weight_norm(backbone_only=backbone_only))
        # append current
        self._weight_norm_history.append(cur_norm)
        idx = len(self._weight_norm_history) - 1

        # 如果历史长度不够，返回计数状态
        if len(self._weight_norm_history) <= steps:
            return {"status": "counting", "idx": idx, "norm": cur_norm}

        # 对比 i 与 i-steps
        pre_idx = -(steps + 1)
        pre_norm = float(self._weight_norm_history[pre_idx])
        delta = abs(cur_norm - pre_norm)
        changed = delta > float(threshold)

        record = {
            "idx_now": idx,
            "idx_prev": idx - steps,
            "pre_norm": pre_norm,
            "post_norm": cur_norm,
            "delta": delta,
            "threshold": float(threshold),
            "steps": int(steps),
            "changed": bool(changed)
        }

        # 若检测到显著变化，记录并打印一条警告（便于训练时观察）
        if changed:
            # 记录时间戳可选：import time; record['ts']=time.time()
            self.weight_change_records.append(record)
            print(f"[WEIGHT-CHECK][WARN] weight norm changed > {threshold:.1e}: Δ={delta:.4e} (idx {record['idx_prev']} -> {record['idx_now']})")

        return {"status": "checked", **record}





if __name__ == '__main__':
    os.chdir("/home/Guanjq/NewWork/MedAlignFusion/Code")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running Module Test on device: {device} ---")
    
    batch_size = 2
    num_images = 6
    
    dummy_data = [{
        'images': [Image.new('RGB', (224, 224), color='red') for _ in range(num_images)],
        'strong_related_text': 'Pathology: Squamous cell carcinoma, moderately differentiated.' * 2000,
        'weak_related_text': 'Clinical: Male, 65 years old, history of smoking.'
    }] * batch_size
    dummy_labels = torch.randint(0, 2, (batch_size,), device=device)

    test_configs = ['all', 'image,strong_related_text', 'image', 'weak_related_text']

    for config in test_configs:
        print(f"\n--- Testing with modalities = '{config}' ---")
        try:
            task_module = MultiOSCCRecPred(device=device, modalities=config).to(device)

            print("Testing encode()...")
            encoded_output = task_module.encode(dummy_data)
            all_embeddings = encoded_output["embeddings"]
            strong_pairs = encoded_output["strong_related_pairs"]
            
            print(f"  > Returned {len(all_embeddings)} embedding slots and {len(strong_pairs)} strong pair(s).")
            
            present_count = 0
            if all_embeddings[0] is not None:
                print(f"  - Image features shape: {all_embeddings[0].shape}")
                assert all_embeddings[0].dim() == 3 # Check for token-level
                present_count += 1
            else:
                print("  - Image features: None")
            
            if all_embeddings[1] is not None:
                print(f"  - Strong text features shape: {all_embeddings[1].shape}")
                assert all_embeddings[1].dim() == 3 # Check for token-level
                present_count += 1
            else:
                print("  - Strong text features: None")
            
            if all_embeddings[2] is not None:
                print(f"  - Weak text features shape: {all_embeddings[2].shape}")
                assert all_embeddings[2].dim() == 3 # Check for token-level
                present_count += 1
            else:
                print("  - Weak text features: None")
            
            print(f"  - Strong pairs detected: {strong_pairs}")

            if present_count > 0:
                print("Testing decode() on the first present embedding...")
                first_present_embedding = next(e for e in all_embeddings if e is not None)
                decode_output = task_module.decode(first_present_embedding, dummy_labels)
                print(f"  - Logits shape: {decode_output['logits'].shape}")
                print(f"  - Loss value: {decode_output['loss'].item():.4f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"An error occurred during test for config '{config}': {e}")

  