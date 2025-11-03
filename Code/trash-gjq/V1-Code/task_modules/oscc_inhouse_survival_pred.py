# multimodal_clinical_vit.py
import os
import math
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# vision dependencies
from scipy import ndimage
from timm.models.vision_transformer import VisionTransformer

# text / transformers
from transformers import AutoTokenizer, AutoModel

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from modules.common_modules.surv_loss import NLLSurvLoss
from modules.training_utils.metrics import survival_metrics

import math
from collections import deque
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from transformers import AutoTokenizer, AutoModel



# ---------------------------
# small vit loader (from your code, adapted)
# ---------------------------
def vit_get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained: bool = True, progress: bool = True, key: str = "DINO_p16"):
    """
    Return a VisionTransformer (small) with head removed.
    The returned model's head is Identity and returns per-image feature (embed_dim=384).
    """
    patch_size = 16
    img_size = 512
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=384, num_heads=6)
    model.head = nn.Identity()
    if pretrained:
        try:
            pretrained_url = vit_get_pretrained_url(key)
            state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
            net_dict = {k: v for k, v in state_dict.items() if k in model.state_dict().keys() and k != "pos_embed"}

            posemb = state_dict["pos_embed"]
            posemb_new = model.state_dict()["pos_embed"]
            ntok_new = posemb_new.size(1)
            posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
            posemb_zoom = np.expand_dims(posemb_zoom, 0)
            net_dict.update({"pos_embed": torch.from_numpy(posemb_zoom)})

            verbose = model.load_state_dict(net_dict, strict=False)
            print("[vit_small] load_state_dict:", verbose)
        except Exception as e:
            print("[vit_small] warning: failed to load pretrained weights:", e)
    return model


def get_vit_small_pathology(pretrained: bool = True, progress: bool = True, key: str = "DINO_p16"):
    model = vit_small(pretrained=pretrained, progress=progress, key=key)
    return model, 384  # original embedding dim




class OSCCSurvivalPred(nn.Module):
    """
    Multimodal model using ClinicalBERT (text) + vit_small_patho (image).
    - All modality embeddings are projected to embed_dim (default 512).
    - encode() returns token-level embeddings for each modality as:
        embeddings = [image_embs, strong_text_embs, weak_text_embs]
        masks = [image_mask, strong_text_mask, weak_text_mask]
    - decode() accepts pooled embeddings (B, embed_dim) and labels to compute logits & loss.
    """

    METRICS_FN = staticmethod(survival_metrics)

    def __init__(
        self,
        modalities: str = "all"
    ):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        self.embed_dim = 512

        # 保留原始输入，用于解析
        self._raw_modalities = modalities
        valid_modalities = [
            "all", "images", "strong_related_text", "weak_related_text",
            "images,strong_related_text", "images,weak_related_text", "strong_related_text,weak_related_text"
        ]
        cleaned_modalities = ",".join(modalities.split('-'))
        assert cleaned_modalities in valid_modalities, f"Invalid modalities specified: {modalities}"
        self.modalities = cleaned_modalities
        self.max_modalities_num = 3 if self.modalities == 'all' else len(cleaned_modalities.split(','))

        # ----- Vision backbone (vit small) -----
        vit_model, vit_emb = get_vit_small_pathology(pretrained=True, progress=True, key="DINO_p16")
        vit_model = vit_model.to(self.device)
        self.vit = vit_model
        self.vit_orig_dim = vit_emb
        if self.vit_orig_dim != self.embed_dim:
            self.vit_proj = nn.Linear(self.vit_orig_dim, self.embed_dim)
        else:
            self.vit_proj = nn.Identity()

        # ----- Text backbone (ClinicalBERT) -----
        self.text_model_name = "medicalai/ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, use_fast=True)
        self.bert = AutoModel.from_pretrained(self.text_model_name)
        bert_hidden = getattr(self.bert.config, "hidden_size", 768)
        self.bert_orig_dim = bert_hidden
        if bert_hidden != self.embed_dim:
            self.bert_proj = nn.Linear(bert_hidden, self.embed_dim)
        else:
            self.bert_proj = nn.Identity()
        self.bert.to(self.device)


        # ----- Prediction Head (for Decode step) -----
        self.prediction_head = nn.Linear(self.embed_dim, 10) # Predicts risk for 10 time intervals
        self.loss_fn = NLLSurvLoss(reduction='none')

        # optional: small history for weight norms (if desired)
        self._weight_norm_history = deque()
        self.weight_change_records = []
        self.weight_check_threshold = 1e-6
        self.weight_check_steps = 3

    def get_backbone_params(self):
        backbone_params = []
        backbone_params.extend(list(self.vit.parameters()))
        backbone_params.extend(list(self.bert.parameters()))
        return backbone_params
    
    def get_others_params(self):
        backbone_ids = {id(p) for p in self.get_backbone_params()}
        all_params = list(self.parameters())
        others_params = [p for p in all_params if id(p) not in backbone_ids]
        return others_params
    

    # -------------------------
    # Text chunking & encoding
    # -------------------------
    def _chunk_token_ids(self, ids: List[int], chunk_size: int) -> List[List[int]]:
        """Split list of token ids (without special tokens) into chunks of chunk_size."""
        chunks = []
        for i in range(0, len(ids), chunk_size):
            chunks.append(ids[i:i + chunk_size])
        return chunks

    def _encode_text(self, text_list: List[Optional[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a list of texts (batch) and return:
            final_embeddings: Tensor (B, max_chunks, embed_dim)
            final_mask: Tensor (B, max_chunks)  (1 places where chunk exists)
        Behavior:
            - empty/None/blank string => all-zero embeddings and zero mask
            - long text is split into chunks (model_max_length - 2 for [CLS]/[SEP])
            - each chunk is fed to bert, we take pooler_output if available, else mean-pool
        """
        batch_size = len(text_list)
        model_max = 512
        chunk_payload = max(1, model_max - 2)

        valid_texts_with_idx = []
        for i, t in enumerate(text_list):
            if t and isinstance(t, str) and t.strip():
                valid_texts_with_idx.append({"index": i, "text": t})

        if not valid_texts_with_idx:
            return torch.zeros(batch_size, 1, self.embed_dim, device=self.device), torch.zeros(batch_size, 1, device=self.device).bool()

        chunk_input_ids = []
        chunk_attention_masks = []
        mapping_info = []  # per valid text: (orig_index, num_chunks)
        for item in valid_texts_with_idx:
            txt = item["text"]
            idx = item["index"]
            ids = self.tokenizer.encode(txt, add_special_tokens=False)
            chunks = self._chunk_token_ids(ids, chunk_payload)
            mapping_info.append({"index": idx, "n": len(chunks)})
            for ch in chunks:
                input_ids = [self.tokenizer.cls_token_id] + ch + [self.tokenizer.sep_token_id]
                att_mask = [1] * len(input_ids)
                pad_len = model_max - len(input_ids)
                if pad_len > 0:
                    pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    input_ids = input_ids + [pad_token] * pad_len
                    att_mask = att_mask + [0] * pad_len
                chunk_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                chunk_attention_masks.append(torch.tensor(att_mask, dtype=torch.long))

        chunk_input_ids = torch.stack(chunk_input_ids, dim=0).to(self.device)
        chunk_attention_masks = torch.stack(chunk_attention_masks, dim=0).to(self.device)

        # Extract pooled embeddings
        outputs = self.bert(input_ids=chunk_input_ids, attention_mask=chunk_attention_masks)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            last_hidden = outputs.last_hidden_state
            att = chunk_attention_masks.unsqueeze(-1).float()
            summed = (last_hidden * att).sum(dim=1)
            lengths = att.sum(dim=1).clamp(min=1e-6)
            pooled = summed / lengths

        pooled_proj = self.bert_proj(pooled)  # (total_chunks, embed_dim)

        num_chunks_list = [m["n"] for m in mapping_info]
        max_chunks = max(num_chunks_list) if num_chunks_list else 1

        final_embeddings_list = []
        final_mask_list = []
        cur = 0
        for m in mapping_info:
            n = m["n"]
            emb = pooled_proj[cur: cur + n]  # (n, embed_dim)
            cur += n
            if n < max_chunks:
                pad_n = max_chunks - n
                emb = torch.cat([emb, torch.zeros(pad_n, self.embed_dim, device=self.device)], dim=0)
                mask = torch.cat([torch.ones(n, device=self.device), torch.zeros(pad_n, device=self.device)], dim=0)
            else:
                mask = torch.ones(n, device=self.device)
            final_embeddings_list.append(emb)
            final_mask_list.append(mask)

        final_embeddings = []
        final_masks = []
        valid_indices = [m["index"] for m in mapping_info]
        valid_iter = iter(final_embeddings_list)
        mask_iter = iter(final_mask_list)
        for i in range(batch_size):
            if i in valid_indices:
                final_embeddings.append(next(valid_iter))
                final_masks.append(next(mask_iter))
            else:
                final_embeddings.append(torch.zeros(max_chunks, self.embed_dim, device=self.device))
                final_masks.append(torch.zeros(max_chunks, device=self.device))

        final_embeddings = torch.stack(final_embeddings, dim=0)
        final_masks = torch.stack(final_masks, dim=0)

        return final_embeddings, final_masks.bool()

    # -------------------------
    # Public encode: multimodal
    # -------------------------
    def encode(self, batch: Dict) -> Dict:
        """
        Dynamically encodes modalities based on the keys present in the collated batch.
        Returns a dictionary with token-level embeddings and masks.
        """
        batch_size = len(batch.get('labels', []))
        if batch_size == 0:
            return {"embeddings": [None, None, None], "masks": [None, None, None], "align_pairs": []}

        image_features, strong_text_features, weak_text_features = None, None, None
        image_mask, strong_text_mask, weak_text_mask = None, None, None

        # --- Process modalities only if their key exists in the batch ---
        all_embeddings = []  #[image_features, strong_text_features, weak_text_features]
        all_masks = [] # [image_mask, strong_text_mask, weak_text_mask]

        # ----- Image branch -----
        if 'images' in batch and batch['images']:
            list_of_image_lists = batch.get('images', [])
            if list_of_image_lists and isinstance(list_of_image_lists[0], list) and list_of_image_lists[0]:
                num_images_per_patient = len(list_of_image_lists[0])
                all_images = [img for patient_images in list_of_image_lists for img in patient_images]

                # Handle tensor inputs
                images_tensor = torch.stack([img.to(self.device) for img in all_images]).to(self.device)
                # Pass through vision backbone (assume vit returns (N, vit_orig_dim))
                image_features_raw = self.vit(images_tensor)
                # project to embed_dim if necessary
                image_features_proj = self.vit_proj(image_features_raw)
                # reshape to (B, num_images_per_patient, embed_dim)
                image_features = image_features_proj.reshape(batch_size, num_images_per_patient, self.embed_dim)
                image_mask = torch.ones(batch_size, num_images_per_patient, device=self.device).bool()

                all_embeddings.append(image_features)
                all_masks.append(image_mask)

        # ----- Strong text branch -----
        if 'strong_related_text' in batch and batch['strong_related_text']:
            strong_text_features, strong_text_mask = self._encode_text(batch['strong_related_text'])
            all_embeddings.append(strong_text_features)
            all_masks.append(strong_text_mask)

        # ----- Weak text branch -----
        if 'weak_related_text' in batch and batch['weak_related_text']:
            weak_text_features, weak_text_mask = self._encode_text(batch['weak_related_text'])
            all_embeddings.append(weak_text_features)
            all_masks.append(weak_text_mask)

        # Define which modalities are strongly related (e.g., for cross-attention)
        # Index 0: image, Index 1: strong_related_text
        strong_related_pairs = []
        if image_features is not None and strong_text_features is not None:
            strong_related_pairs.append((0, 1))

        # Check number of present modalities
        assert len(all_embeddings) <= self.max_modalities_num, f"Number of present modalities exceeds the maximum allowed: {self.max_modalities_num}"

        # print("Modalities present:", len(all_embeddings))

        return {
            "embeddings": all_embeddings,
            "masks": all_masks,
            "align_pairs": strong_related_pairs,
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

