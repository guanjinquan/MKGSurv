import os
import math
from typing import List, Dict, Optional, Tuple
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CONCH dependencies
# You may need to install with: pip install git+https://github.com/Mahmoodlab/CONCH.git
# and potentially: pip install open_clip_torch
try:
    from conch.open_clip_custom import create_model_from_pretrained, tokenize as conch_tokenize
except ImportError:
    print("Please install CONCH: pip install git+https://github.com/Mahmoodlab/CONCH.git")
    create_model_from_pretrained = None
    conch_tokenize = None


class MultiOSCCRecPred(nn.Module):
    """
    Multimodal model using CONCH (ViT-B-16) for image and text encoding.
    - All modality embeddings are handled by the CONCH model.
    - encode() returns token-level embeddings for each modality as:
        embeddings = [image_embs, strong_text_embs, weak_text_embs]
        masks = [image_mask, strong_text_mask, weak_text_mask]
    - decode() accepts pooled embeddings (B, embed_dim) and labels to compute logits & loss.
    """

    def __init__(
        self,
        modalities: str = "all"
    ):
        super().__init__()
        
        if create_model_from_pretrained is None or conch_tokenize is None:
            raise RuntimeError("CONCH library not found. Please install it to use this model.")

        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        # ----- CONCH Backbone -----
        # Note: You may need to provide your huggingface user access token via
        # hf_auth_token=<your_token> to create_model_from_pretrained for authentification.
        # See the HF documentation for more details.
        # Alternatively, you can download the checkpoint manually and load from a local path.
        try:
            self.conch_model, self.image_preprocess = create_model_from_pretrained(
                'conch_ViT-B-16', 
                "../PretrainedWeights/MahmoodLab/CONCH/pytorch_model.bin"
            )
        except Exception as e:
            print(f"Failed to load CONCH model from Hugging Face Hub: {e}")
            print("Please ensure you have requested access and are using an auth token if required.")
            raise

        self.conch_model = self.conch_model.to(self.device)
        self.tokenizer = conch_tokenize
        self.embed_dim = 512
        self.out_dim = 2

        # Projection
        # CONCH ViT-B/16 embedding dimension is text:768, visual:512
        self.visual_embed_dim = 512
        if self.visual_embed_dim != self.embed_dim:
            self.visual_proj = nn.Linear(self.visual_embed_dim, self.embed_dim)
        else:
            self.visual_proj = nn.Identity()

        self.text_embed_dim = 768
        if self.text_embed_dim != self.embed_dim:
            self.text_proj = nn.Linear(self.text_embed_dim, self.embed_dim)
        else:
            self.text_proj = nn.Identity()

        # Retain original inputs for parsing
        self._raw_modalities = modalities
        valid_modalities = [
            "all", "image", "strong_related_text", "weak_related_text",
            "image,strong_related_text", "image,weak_related_text", "strong_related_text,weak_related_text"
        ]
        cleaned_modalities = ",".join(modalities.split('-'))
        assert cleaned_modalities in valid_modalities, f"Invalid modalities specified: {modalities}"
        self.modalities = cleaned_modalities
        self.max_modalities_num = 3 if self.modalities == 'all' else len(cleaned_modalities.split(','))

        # ----- Classifier head on pooled embedding -----
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim // 2, self.out_dim),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Optional: small history for weight norms (if desired)
        self._weight_norm_history = deque()
        self.weight_change_records = []
        self.weight_check_threshold = 1e-6
        self.weight_check_steps = 3

    # -------------------------
    # Text encoding
    # -------------------------
    def _encode_text(self, text_list: List[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a list of texts (batch) using CONCH's text encoder.
        Returns:
            final_embeddings: Tensor (B, 1, embed_dim)
            final_mask: Tensor (B, 1) (1 where text exists)
        Behavior:
            - empty/None/blank string => all-zero embeddings and zero mask
            - Text is tokenized, truncated to model's context length, and encoded.
        """
        batch_size = len(text_list)
        
        # Filter out empty texts while keeping track of original indices
        valid_texts_with_idx = [
            {"index": i, "text": t} 
            for i, t in enumerate(text_list) 
            if t and isinstance(t, str) and t.strip()
        ]

        if not valid_texts_with_idx:
            return torch.zeros(batch_size, 1, self.embed_dim, device=self.device), torch.zeros(batch_size, 1, device=self.device).bool()

        # Encode valid texts in a single batch
        texts_to_encode = [item["text"] for item in valid_texts_with_idx]
        tokenized_texts = self.tokenizer(texts_to_encode).to(self.device)
        
        text_embeddings = self.conch_model.encode_text(tokenized_texts) # (num_valid_texts, embed_dim)
        text_embeddings = self.text_proj(text_embeddings)

        # Create final output tensors, placing embeddings at their original batch positions
        final_embeddings = torch.zeros(batch_size, self.embed_dim, device=self.device)
        final_mask = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        
        for i, item in enumerate(valid_texts_with_idx):
            original_index = item["index"]
            final_embeddings[original_index] = text_embeddings[i]
            final_mask[original_index] = True
            
        # Reshape to (B, 1, D) and (B, 1) to match expected downstream format
        return final_embeddings.unsqueeze(1), final_mask.unsqueeze(1)

    # -------------------------
    # Public encode: multimodal
    # -------------------------
    def encode(self, batch: Dict) -> Dict:
        """
        Encodes a collated batch of data, returning a dictionary containing token-level
        embeddings (with Nones for missing modalities).

        Args:
            batch (Dict): A dictionary from a collate function with keys like
                          'images', 'labels', 'strong_related_text', 'weak_related_text'.
        """
        self.conch_model.eval()
        
        try:
            check_res = self._auto_weight_check()
            if check_res.get("status") == "checked" and check_res.get("changed", False):
                pass
        except Exception as e:
            print(f"[WEIGHT-CHECK][ERROR] _auto_weight_check failed: {e}")

        batch_size = len(batch.get('labels', []))
        if batch_size == 0:
            return {"embeddings": [None, None, None], "masks": [None, None, None], "strong_related_pairs": []}

        image_features, strong_text_features, weak_text_features = None, None, None
        image_mask, strong_text_mask, weak_text_mask = None, None, None

        # ----- Image branch -----
        if 'image' in self.modalities or self.modalities == 'all':
            list_of_image_lists = batch.get('images', [])
            # Check if there is image data to process
            if list_of_image_lists and isinstance(list_of_image_lists[0], list) and any(list_of_image_lists):
                num_images_per_patient = len(list_of_image_lists[0])
                all_images = [img for patient_images in list_of_image_lists for img in patient_images]

                if all_images:
                    # Preprocess PIL images and stack them into a tensor
                    processed_images = torch.stack([self.image_preprocess(img) for img in all_images]).to(self.device)
                    
                    # Get image embeddings from CONCH
                    image_features_raw = self.conch_model.encode_image(
                        processed_images, 
                        proj_contrast=False, 
                        normalize=False
                    )
                    image_features_projed = self.visual_proj(image_features_raw)
                    
                    # Reshape to (B, num_images_per_patient, embed_dim)
                    image_features = image_features_projed.reshape(batch_size, num_images_per_patient, self.embed_dim)
                    image_mask = torch.ones(batch_size, num_images_per_patient, device=self.device).bool()

        # ----- Strong text branch -----
        if 'strong_related_text' in self.modalities or self.modalities == 'all':
            all_strong_texts = batch.get('strong_related_text', [])
            if all_strong_texts:
                strong_text_features, strong_text_mask = self._encode_text(all_strong_texts)

        # ----- Weak text branch -----
        if 'weak_related_text' in self.modalities or self.modalities == 'all':
            all_weak_texts = batch.get('weak_related_text', [])
            if all_weak_texts:
                weak_text_features, weak_text_mask = self._encode_text(all_weak_texts)

        all_embeddings = [image_features, strong_text_features, weak_text_features]
        all_masks = [image_mask, strong_text_mask, weak_text_mask]
        strong_related_pairs = []
        if image_features is not None and strong_text_features is not None:
            strong_related_pairs.append((0, 1))

        # Debug check NaN
        for i, emb in enumerate(all_embeddings):
            if emb is not None and torch.isnan(emb).any():
                print(f"[ENCODE][ERROR] NaN detected in embedding {i}")

        return {
            "embeddings": all_embeddings,
            "masks": all_masks,
            "strong_related_pairs": strong_related_pairs
        }

    # -------------------------
    # decode & metrics (compatible)
    # -------------------------
    def decode(self, pooled_embeddings: torch.Tensor, pooled_mask: Optional[torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        pooled_embeddings: (B, embed_dim)
        pooled_mask: (B,) boolean or None
        labels: torch.Tensor (B,) or (B, out_dim)
        returns {'logits': (B, out_dim), 'loss': scalar_tensor}
        """
        batch_size = pooled_embeddings.shape[0]
        device = pooled_embeddings.device
        labels = labels.to(device)

        logits = torch.zeros(batch_size, self.out_dim, device=device)
        loss = torch.tensor(0.0, device=device)

        patient_mask = pooled_mask.bool().to(device) if pooled_mask is not None else torch.ones(batch_size, dtype=torch.bool, device=device)
        if not patient_mask.any():
            return {"logits": logits, "loss": loss}

        valid_embeddings = pooled_embeddings[patient_mask]
        valid_labels = labels[patient_mask]

        valid_logits = self.classifier(valid_embeddings)

        if valid_labels.ndim == 1:
            target = F.one_hot(valid_labels, num_classes=self.out_dim).float()
        else:
            target = valid_labels.float()
        loss = self.loss_fn(valid_logits, target)
        logits[patient_mask] = valid_logits
        return {"logits": logits, "loss": loss}

    def get_metrics(self, logits, labels):
        import numpy as _np
        from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

        logits_np = _np.array(logits)
        labels_np = _np.array(labels)

        if labels_np.ndim == 2 and labels_np.shape[1] > 1:
            labels_for_metrics = _np.argmax(labels_np, axis=1)
        else:
            labels_for_metrics = labels_np

        y_pred = _np.argmax(logits_np, axis=1)

        logits_tensor = torch.tensor(logits_np, dtype=torch.float32)
        y_prob = F.softmax(logits_tensor, dim=1).numpy()

        acc = accuracy_score(labels_for_metrics, y_pred)
        macro_f1 = f1_score(labels_for_metrics, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(labels_for_metrics, y_pred, average='macro', zero_division=0)
        macro_precision = precision_score(labels_for_metrics, y_pred, average='macro', zero_division=0)

        num_classes = logits_np.shape[1] if logits_np.ndim == 2 else 2
        auc = 0.0
        try:
            if num_classes == 2:
                auc = roc_auc_score(labels_for_metrics, y_prob[:, 1])
            elif num_classes > 2:
                auc = roc_auc_score(labels_for_metrics, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0

        return {"Acc": acc, "F1": macro_f1, "Recall": macro_recall, "Precision": macro_precision, "AUC": auc}

    # -------------------------
    # utility: compute L2 norm of backbone weights (optional)
    # -------------------------
    def _compute_weight_norm(self, backbone_only: bool = True) -> float:
        params = []
        if backbone_only:
            # CONCH model attributes for vision and text towers
            if hasattr(self.conch_model, 'visual'):
                params.extend(list(self.conch_model.visual.parameters()))
            if hasattr(self.conch_model, 'transformer'):
                 params.extend(list(self.conch_model.transformer.parameters()))
        else:
            params = list(self.parameters())
        
        total_sq = 0.0
        for p in params:
            if p is not None and p.numel() > 0:
                t = p.detach().cpu().float()
                total_sq += float(torch.sum(t * t).item())
        return math.sqrt(total_sq)

    def _auto_weight_check(self, threshold: Optional[float] = None, steps: Optional[int] = None, backbone_only: bool = True) -> Dict:
        if threshold is None: threshold = self.weight_check_threshold
        if steps is None: steps = self.weight_check_steps
        
        cur_norm = float(self._compute_weight_norm(backbone_only=backbone_only))
        self._weight_norm_history.append(cur_norm)
        idx = len(self._weight_norm_history) - 1
        
        if len(self._weight_norm_history) <= steps:
            return {"status": "counting", "idx": idx, "norm": cur_norm}
        
        # Pop the oldest entry to keep the deque size manageable
        if len(self._weight_norm_history) > (steps + 1):
             self._weight_norm_history.popleft()
             
        pre_norm = float(self._weight_norm_history[0])
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
        
        if changed:
            self.weight_change_records.append(record)
            print(f"[WEIGHT-CHECK][WARN] Δ={delta:.4e} > {threshold:.1e}")
            
        return {"status": "checked", **record}

