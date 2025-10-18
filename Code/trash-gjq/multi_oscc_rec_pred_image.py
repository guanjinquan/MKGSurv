import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Dict, Optional
import numpy as np
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)
from collections import deque
import math


import torch.nn as nn
import numpy as np
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from scipy import ndimage
from timm.models.vision_transformer import VisionTransformer
import torch


def vit_get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def vit_small(pretrained, progress, key):
    patch_size = 16
    img_size = 512
    model = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=384, num_heads=6)
    
    model.head = nn.Identity()

    if pretrained:
        pretrained_url = vit_get_pretrained_url(key)
        state_dict = torch.hub.load_state_dict_from_url(pretrained_url, progress=progress)
        
        # 获取除了pos_embeds和head的参数
        net_dict = {k:v for k, v in state_dict.items() if k in model.state_dict().keys() and k != "pos_embed"}
        
        # 获取pos_embeds
        posemb = state_dict["pos_embed"]
        posemb_new = model.state_dict()["pos_embed"]
        ntok_new = posemb_new.size(1)
        posemb_zoom = ndimage.zoom(posemb[0], (ntok_new / posemb.size(1), 1), order=1)
        posemb_zoom = np.expand_dims(posemb_zoom, 0)
        net_dict.update({"pos_embed": torch.from_numpy(posemb_zoom)})
        
        # 更新参数
        verbose = model.load_state_dict(net_dict, strict=False)
        print(verbose)  # _IncompatibleKeys(missing_keys=['head.weight', 'head.bias'], unexpected_keys=[])
        
    return model


class VitPathology(nn.Module):
    def __init__(self):
        super(VitPathology, self).__init__()
        self.extractor = vit_small(pretrained=True, progress=True, key="DINO_p16")
        
    def forward(self, x):
        x = self.extractor(x)
        return x


def get_vit_small_pathology():
    model = VitPathology()
    return model, 384




class MultiOSCCRecPredImage(nn.Module):
    """
    A modified version of MultiOSCCRecPred that exclusively uses an image modality,
    powered by the vit_small pathology backbone.
    """

    def __init__(self, device: str = "cuda", modalities: str = "image"):
        """
        Initializes the image-only prediction model.

        Args:
            device (str): The device to run the model on ('cuda' or 'cpu').
            modalities (str): Kept for interface compatibility, but is ignored as this is an image-only model.
            img_size (int): The input size for the images (e.g., 512 for 512x512).
            freezed_backbone (bool): Whether to freeze the weights of the ViT backbone.
        """
        super(MultiOSCCRecPredImage, self).__init__()
        self.device = torch.device(device)

        # The 'modalities' parameter is ignored.
        # We can add a check to inform the user if they provide something other than 'image' or 'all'.
        if modalities != "image":
            print(f"Warning: 'modalities' is set to '{modalities}' but this is an image-only model. The parameter will be ignored.")

        self.model, self.embed_dim = get_vit_small_pathology()
        self.max_modalities_num = 1
        self.modalities = modalities
    

        # --- 3. Define classifier and loss ---
        # The output dimension is 2 for binary classification (recurrence vs. no recurrence)
        self.out_dim = 2
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim // 2, self.out_dim)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def get_backbone_params(self):
        """Returns the parameters of the ViT backbone for the optimizer."""
        return self.model.extractor.parameters()

    def encode(self, batch: Dict) -> Dict:
        """
        Encodes a collated batch of image data.

        Args:
            batch (Dict): A dictionary containing 'images': List[List[PIL.Image]].
        
        Returns:
            Dict: A dictionary containing token-level image embeddings and masks.
        """

        batch_size = len(batch.get('labels', []))
        if batch_size == 0:
            return {"embeddings": [None, None, None], "masks": [None, None, None], "strong_related_pairs": []}
        
        image_features, image_mask = None, None

        list_of_image_lists = batch.get('images', [])
        
        if list_of_image_lists and isinstance(list_of_image_lists[0], list) and list_of_image_lists[0]:
            num_images_per_patient = len(list_of_image_lists[0])
            all_images = [img for patient_images in list_of_image_lists for img in patient_images]
            
            if all_images:
                images_tensor = torch.stack([img for img in all_images]).to(self.device)
                
                # The VitPathology model returns features of shape (B * N_images, embed_dim)
                image_features_raw = self.model(images_tensor) 
                
                # Reshape to (B, N_images, embed_dim) to represent sequences of images
                image_features = image_features_raw.reshape(batch_size, num_images_per_patient, self.embed_dim)
                image_mask = torch.ones(batch_size, num_images_per_patient).to(self.device)

        # The output structure is kept similar to the original for compatibility,
        # but only the image slot (index 0) is populated.
        return {
            "embeddings": [image_features, None, None], 
            "masks": [image_mask, None, None],
            "strong_related_pairs": [] # No text modality, so no pairs
        }

    def decode(self, pooled_embeddings: torch.Tensor, pooled_mask: Optional[torch.Tensor], labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decodes POOLED embeddings into logits and computes loss.
        (This function is unchanged from the original as it's modality-agnostic).
        """
        batch_size = pooled_embeddings.shape[0]
        device = pooled_embeddings.device
        labels = labels.to(device)

        logits = torch.zeros(batch_size, self.out_dim, device=device)
        loss = torch.tensor(0.0, device=device)

        patient_mask = pooled_mask.bool().to(device) if pooled_mask is not None else torch.ones(batch_size, dtype=torch.bool, device=device)
        
        if not patient_mask.any():
            return {'logits': logits, 'loss': loss}

        valid_embeddings = pooled_embeddings[patient_mask]
        valid_labels = labels[patient_mask]
        valid_logits = self.classifier(valid_embeddings)
        
        target_labels = F.one_hot(valid_labels, num_classes=self.out_dim).float() if valid_labels.ndim == 1 else valid_labels.float()
        loss = self.loss_fn(valid_logits, target_labels)

        logits[patient_mask] = valid_logits
        return {'logits': logits, 'loss': loss}

    def get_metrics(self, logits, labels):
        """
        Calculates classification metrics from logits and labels.
        (This function is unchanged from the original).
        """
        logits_np = np.array(logits)
        labels_np = np.array(labels)

        labels_for_metrics = np.argmax(labels_np, axis=1) if labels_np.ndim == 2 and labels_np.shape[1] > 1 else labels_np
        y_pred = np.argmax(logits_np, axis=1)
        y_prob = F.softmax(torch.tensor(logits_np, dtype=torch.float32), dim=1).numpy()
        
        acc = accuracy_score(labels_for_metrics, y_pred)
        macro_f1 = f1_score(labels_for_metrics, y_pred, average='macro', zero_division=0)
        macro_recall = recall_score(labels_for_metrics, y_pred, average='macro', zero_division=0)
        macro_precision = precision_score(labels_for_metrics, y_pred, average='macro', zero_division=0)

        num_classes = logits_np.shape[1]
        auc = 0.0
        if num_classes == 2:
            auc = roc_auc_score(labels_for_metrics, y_prob[:, 1])
        elif num_classes > 2:
            try:
                auc = roc_auc_score(labels_for_metrics, y_prob, multi_class='ovr', average='macro')
            except ValueError:
                auc = 0.0
        
        return {"Acc": acc, "F1": macro_f1, "Recall": macro_recall, "Precision": macro_precision, "AUC": auc}

