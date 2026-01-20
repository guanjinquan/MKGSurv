import os

# 1. Set HF Mirror before importing transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import torch.nn as nn
import pickle
import numpy as np
import random
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================= Core Encoder Class =================

class ClinicalBertEncoder(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.embed_dim = 768 
        
        print(f"Initializing Text Encoder (ClinicalBERT) on {self.device}...")
        self.text_model_name = "medicalai/ClinicalBERT"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, use_fast=True)
            self.bert = AutoModel.from_pretrained(self.text_model_name)
        except Exception as e:
            print(f"Error loading model from HF Mirror: {e}")
            print("Please check your internet connection or HF_ENDPOINT settings.")
            raise e
            
        self.to(self.device)
        self.eval() 

    def _chunk_token_ids(self, ids: List[int], chunk_size: int) -> List[List[int]]:
        """Splits token IDs into chunks of specific size."""
        return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    def _encode_text(self, texts_list: List[str]) -> List[torch.Tensor]:
        """
        Input: List of text strings.
        Output: List of Tensors. Each Tensor is (N_chunks, 768), on CPU.
        """
        batch_size = len(texts_list)
        chunk_payload = 510 # BERT max 512 - 2 (CLS/SEP)
        
        all_chunks = []
        mapping_info = [] 
        
        # 1. Preprocessing & Chunking
        for i, text in enumerate(texts_list):
            item_specific_chunks = [] 
            
            if isinstance(text, str) and text.strip():
                clean_text = text.strip()
                token_ids = self.tokenizer.encode(clean_text, add_special_tokens=False)
                chunks = self._chunk_token_ids(token_ids, chunk_payload) 
                if chunks:
                    item_specific_chunks.extend(chunks)
            
            # Record mapping info
            if not item_specific_chunks:
                mapping_info.append({'index': i, 'n': 0})
            else:
                mapping_info.append({'index': i, 'n': len(item_specific_chunks)})
                all_chunks.extend(item_specific_chunks)

        # Handle case with no valid chunks
        if not all_chunks:
            # 保持 float32
            return [torch.zeros(0, self.embed_dim, dtype=torch.float32) for _ in range(batch_size)]

        # 2. Batch Encoding
        chunk_texts = [self.tokenizer.decode(c, clean_up_tokenization_spaces=True) for c in all_chunks]
        
        bert_batch_size = 32
        pooled_outputs = []
        
        for i in range(0, len(chunk_texts), bert_batch_size):
            batch_texts = chunk_texts[i : i + bert_batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)

            with torch.no_grad():
                bert_outputs = self.bert(**inputs)
            
            # Extract CLS token (index 0)
            batch_pooled = bert_outputs.last_hidden_state[:, 0, :]
            pooled_outputs.append(batch_pooled.cpu())
            
        if pooled_outputs:
            pooled = torch.cat(pooled_outputs, dim=0)
        else:
            pooled = torch.zeros(0, self.embed_dim)

        # 3. Reconstruct / Gather results
        output_list = []
        chunk_cursor = 0
        
        for i in range(batch_size):
            info = next((m for m in mapping_info if m['index'] == i), None)
            n_chunks = info['n'] if info else 0
            
            if n_chunks > 0:
                # ================= 核心修改 =================
                # 只加 .clone() 解决内存共享导致的文件膨胀问题
                # 不加 .half()，保持 float32
                valid_features = pooled[chunk_cursor : chunk_cursor + n_chunks].clone()
                output_list.append(valid_features)
                chunk_cursor += n_chunks
            else:
                # 保持 float32
                output_list.append(torch.zeros((0, self.embed_dim), dtype=torch.float32))

        return output_list
    

# ================= Augmentation Helper =================
def shuffle_text_sentences(text: str) -> str:
    """
    Splits text by '.', shuffles the sentences, and rejoins them.
    """
    if not text:
        return ""
    # Split by period and remove empty strings
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if not sentences:
        return text
        
    random.shuffle(sentences)
    # Rejoin with period and space, ensure it ends with period
    return ". ".join(sentences) + "."

# ================= Main Processing Function =================

def process_analysis_file(json_path: str, output_path: str, max_augmentations: int = 10):
    """
    Reads the Qwen analysis JSON, encodes features using ClinicalBERT,
    performs sentence-shuffling augmentation, and saves to a pickle file.
    """
    
    # 1. Load Data
    print(f"Loading analysis file: {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input file not found: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. Initialize Encoder
    encoder = ClinicalBertEncoder()
    
    # 3. Process each patient
    final_dataset = {}
    
    print(f"Processing {len(data)} patients with max {max_augmentations} augmentations...")
    
    # Iterate with progress bar
    for patient_id, analysis_list in tqdm(data.items(), desc="Encoding Features"):
        patient_features = {}
        
        for entry in analysis_list:
            # key: tuple of modalities
            modal_key = tuple(entry.get("modalPairs", []))
            
            if not modal_key:
                continue
            
            score = entry.get("score", 0)
            raw_relationship = entry.get("relationship", "")
            raw_survival = entry.get("survival", "")
            
            assert "relationship" in entry, "Missing 'relationship' field in analysis entry."
            assert "survival" in entry, "Missing 'survival' field in analysis entry."
            assert "score" in entry, "Missing 'score' field in analysis entry."
            # --- Prepare Text Batch (Original + Augmentations) ---
            # We will encode all variations in one go for efficiency
            
            texts_part_1_batch = []
            texts_part_2_batch = []
            
            # 1. Add Original (Unshuffled)
            texts_part_1_batch.append(f"Relationship Analysis: {raw_relationship}")
            texts_part_2_batch.append(f"Survival Risk Analysis: {raw_survival}")
            
            # 2. Add Augmentations
            for _ in range(max_augmentations):
                # Augment content text separately
                aug_rel = shuffle_text_sentences(raw_relationship)
                aug_surv = shuffle_text_sentences(raw_survival)
                
                texts_part_1_batch.append(f"Relationship Analysis: {aug_rel}")
                texts_part_2_batch.append(f"Survival Risk Analysis: {aug_surv}")
            
            # --- Encoding ---
            # _encode_text handles list inputs efficiently
            feats_1_list = encoder._encode_text(texts_part_1_batch)
            feats_2_list = encoder._encode_text(texts_part_2_batch)
            
            # --- Combine Results ---
            knowledge_list = []
            
            # Zip original + augmented results together
            for f1, f2 in zip(feats_1_list, feats_2_list):
                # Concatenate features: (N1+N2, 768)
                combined_tensor = torch.cat([f1, f2], dim=0)
                knowledge_list.append(combined_tensor)
            
            # Store in the structure requested: List[combined_tensor, ...]
            # Index 0 is original, Index 1-10 are augmented
            patient_features[modal_key] = {
                "score": score,
                "knowledge_list": knowledge_list,
            }
            
        final_dataset[patient_id] = patient_features

    # 4. Save to Pickle
    print(f"Saving features to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
        
    print("Done.")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # File Paths
    INPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_kimi.json"
    OUTPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/features_medical_knowledge_kimi.pkl"

    # INPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_qwen.json"
    # OUTPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/features_medical_knowledge_qwen.pkl"
    
    # INPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_deepseek.json"
    # OUTPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/features_medical_knowledge_deepseek.pkl"
    
    # Run
    process_analysis_file(INPUT_FILE, OUTPUT_FILE, max_augmentations=20)