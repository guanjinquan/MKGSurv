import os

# 1. Set HF Mirror before importing transformers (Must be at the very top)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import torch.nn as nn
import pickle
import numpy as np
import random
from typing import List, Dict, Tuple, Any
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ================= Configuration =================
# You can modify these paths as needed
# ANALYSIS_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_deepseek.json"
# ANALYSIS_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_qwen.json"
ANALYSIS_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_kimi.json"
KNOWLEDGE_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/pairs_knowledge_qwen.json"
OUTPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/features_medical_knowledge_kimi.pkl"
MAX_AUGMENTATIONS = 20
BATCH_SIZE = 32 # Batch size for BERT inference

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
        
        pooled_outputs = []
        
        for i in range(0, len(chunk_texts), BATCH_SIZE):
            batch_texts = chunk_texts[i : i + BATCH_SIZE]
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

def normalize_modal_key(pair_list: List[str]) -> Tuple[str, ...]:
    """Sorts modal pair list to ensure consistent keys (e.g. ('clinical', 'pathology'))"""
    return tuple(sorted(pair_list))

# ================= Main Processing Logic =================

def process_data_fusion(analysis_path: str, knowledge_path: str, output_path: str, max_augmentations: int = 10):
    
    # 1. Load Data
    print(f"Loading Analysis file: {analysis_path}")
    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
        
    print(f"Loading Knowledge file: {knowledge_path}")
    with open(knowledge_path, 'r', encoding='utf-8') as f:
        knowledge_data = json.load(f)

    # 2. Pre-process Knowledge Data into a Lookup Map
    # Structure: { patient_id: { (modal, pair): "knowledge text" } }
    print("Indexing knowledge data...")
    knowledge_map = {}
    for pid, entries in knowledge_data.items():
        knowledge_map[pid] = {}
        for entry in entries:
            key = normalize_modal_key(entry.get("modalPairs", []))
            if key: # Store only valid keys
                knowledge_map[pid][key] = entry.get("knowledge", "")

    # 3. Pre-process Analysis Data into a Lookup Map (New Step for Union)
    # Structure: { patient_id: { (modal, pair): {analysis_object} } }
    print("Indexing analysis data...")
    analysis_map = {}
    for pid, entries in analysis_data.items():
        analysis_map[pid] = {}
        for entry in entries:
            key = normalize_modal_key(entry.get("modalPairs", []))
            if key: # Store only valid keys
                analysis_map[pid][key] = entry

    # 4. Initialize Encoder
    encoder = ClinicalBertEncoder()
    
    final_dataset = {}
    
    # Get Union of all Patient IDs
    all_patient_ids = sorted(list(set(analysis_data.keys()) | set(knowledge_data.keys())))
    print(f"Processing {len(all_patient_ids)} patients (Union of both files). Max augmentations: {max_augmentations}")
    
    # Iterate with progress bar
    for patient_id in tqdm(all_patient_ids, desc="Encoding Patients"):
        
        patient_features = {}
        
        # Get Union of keys for this specific patient
        p_analysis_entries = analysis_map.get(patient_id, {})
        p_knowledge_entries = knowledge_map.get(patient_id, {})
        
        all_modal_keys = sorted(list(set(p_analysis_entries.keys()) | set(p_knowledge_entries.keys())))
        
        for modal_key in all_modal_keys:
            if not modal_key:
                continue

            # === Retrieve Data with Union Logic ===
            # Try to get analysis data, default to empty/zero if missing
            analysis_entry = p_analysis_entries.get(modal_key, {})
            raw_relationship = analysis_entry.get("relationship", "")
            raw_survival = analysis_entry.get("survival", "")
            score = analysis_entry.get("score", 0)

            # Try to get knowledge data, default to empty string if missing
            raw_knowledge = p_knowledge_entries.get(modal_key, "")
            
            # === Prepare Batch for Encoding ===
            # We need to encode: [Original_Rel, Original_Surv, Original_Know]
            # And then: [Aug_Rel, Aug_Surv, Aug_Know] * max_augmentations
            
            # To be efficient, we put everything in one flat list
            # Order: [Rel_0, Surv_0, Know_0, Rel_1, Surv_1, Know_1, ...]
            
            flat_text_batch = []
            
            # --- 1. Original (Index 0) ---
            flat_text_batch.append(f"Relationship Analysis: {raw_relationship}" if raw_relationship else "")
            flat_text_batch.append(f"Survival Risk Analysis: {raw_survival}" if raw_survival else "")
            flat_text_batch.append(f"Medical Knowledge: {raw_knowledge}" if raw_knowledge else "")
            
            # --- 2. Augmentations (Indices 1 to N) ---
            for _ in range(max_augmentations):
                # Augment only if text exists, otherwise empty string remains empty
                aug_rel = shuffle_text_sentences(raw_relationship) if raw_relationship else ""
                aug_surv = shuffle_text_sentences(raw_survival) if raw_survival else ""
                aug_know = shuffle_text_sentences(raw_knowledge) if raw_knowledge else ""
                
                flat_text_batch.append(f"Score: {score}. Relationship Analysis: {aug_rel}" if aug_rel else "")
                flat_text_batch.append(f"Survival Risk Analysis: {aug_surv}" if aug_surv else "")
                flat_text_batch.append(f"Medical Knowledge: {aug_know}" if aug_know else "")
            
            # === Batch Encode ===
            # resulting list has size: (1 + max_augmentations) * 3
            encoded_tensors = encoder._encode_text(flat_text_batch)
            
            # === Reassemble ===
            knowledge_list = []
            
            # Step is 3 because we have (Rel, Surv, Know) per version
            num_versions = 1 + max_augmentations
            
            for i in range(num_versions):
                idx_base = i * 3
                t_rel = encoded_tensors[idx_base]
                t_surv = encoded_tensors[idx_base + 1]
                t_know = encoded_tensors[idx_base + 2]
                
                # Concatenate features along dimension 0 (sequence length)
                # Result shape: (N_rel + N_surv + N_know, 768)
                combined_tensor = torch.cat([t_rel, t_surv, t_know], dim=0)
                
                knowledge_list.append(combined_tensor)
            
            # Store in result dictionary
            patient_features[modal_key] = {
                "score": score,
                "knowledge_list": knowledge_list, # List of tensors [Original, Aug1, Aug2...]
            }
        
        # Only add patient if we found valid pairs (which should be true if they were in the keys union)
        if patient_features:
            final_dataset[patient_id] = patient_features

    # 4. Save to Pickle
    print(f"Saving combined features to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
        
    print(f"Done. Processed {len(final_dataset)} patients.")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    process_data_fusion(ANALYSIS_FILE, KNOWLEDGE_FILE, OUTPUT_FILE, MAX_AUGMENTATIONS)