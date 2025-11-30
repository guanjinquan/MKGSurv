import os

# 1. Set HF Mirror before importing transformers
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import torch.nn as nn
import pickle
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # Recommended for progress tracking

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
                # Encode text to IDs without special tokens first
                token_ids = self.tokenizer.encode(clean_text, add_special_tokens=False)
                # Split into chunks
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
            return [torch.zeros(0, self.embed_dim) for _ in range(batch_size)]

        # 2. Batch Encoding
        # Convert IDs back to string for batch encoding (robustness)
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
                valid_features = pooled[chunk_cursor : chunk_cursor + n_chunks]
                output_list.append(valid_features)
                chunk_cursor += n_chunks
            else:
                output_list.append(torch.zeros((0, self.embed_dim)))

        return output_list

# ================= Main Processing Function =================

def process_analysis_file(json_path: str, output_path: str):
    """
    Reads the Qwen analysis JSON, encodes features using ClinicalBERT,
    and saves to a pickle file.
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
    
    print(f"Processing {len(data)} patients...")
    
    # Iterate with progress bar
    for patient_id, analysis_list in tqdm(data.items(), desc="Encoding Features"):
        patient_features = {}
        
        for entry in analysis_list:
            # key: tuple of modalities, e.g., ('clinical', 'pathology')
            modal_key = tuple(entry.get("modalPairs", []))
            
            if not modal_key:
                continue
            
            # --- Text Construction Logic ---
            # Part 1: Score + Relationship
            # "score=xxx,拼接上relationship"
            score = entry.get("score", 0)
            relationship_text = entry.get("relationship", "")
            text_part_1 = f"Score: {score}. Relationship Analysis: {relationship_text}"
            
            # Part 2: Survival
            # "survival is another separate string to encode"
            survival_text = entry.get("survival", "")
            text_part_2 = f"Survival Risk Analysis: {survival_text}"
            
            # --- Encoding ---
            # Encode both parts. _encode_text accepts a list, so we pass both at once for efficiency if desired,
            # or separately. Here we pass separately to keep logic clear.
            
            # feat_1 shape: (N_chunks_1, 768)
            feat_1 = encoder._encode_text([text_part_1])[0]
            
            # feat_2 shape: (N_chunks_2, 768)
            feat_2 = encoder._encode_text([text_part_2])[0]
            
            # --- Concatenation ---
            # "Concatenate into (n, 768)"
            # If feat_1 is (2, 768) and feat_2 is (1, 768), result is (3, 768)
            combined_tensor = torch.cat([feat_1, feat_2], dim=0)
            
            patient_features[modal_key] = {
                "score": score,
                "knowledge": combined_tensor,
            }
            
        final_dataset[patient_id] = patient_features

    # 4. Save to Pickle
    print(f"Saving features to {output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
        
    print("Done.")


if __name__ == "__main__":
    # File Paths
    INPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/medical_analysis_deepseek.json"
    OUTPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/features_medical_knowledge.pkl"
    
    process_analysis_file(INPUT_FILE, OUTPUT_FILE)