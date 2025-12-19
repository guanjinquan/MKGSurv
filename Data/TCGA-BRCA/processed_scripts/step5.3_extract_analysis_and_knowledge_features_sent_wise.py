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
ANALYSIS_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/medical_analysis_deepseek.json"
# ANALYSIS_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/medical_analysis_qwen.json"
KNOWLEDGE_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/pairs_knowledge_qwen.json"
OUTPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA/processed/features_medical_knowledge.pkl"
MAX_AUGMENTATIONS = 10
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

    def _encode_text_list(self, texts_list: List[str]) -> List[torch.Tensor]:
        """
        Input: List of sentences.
        Output: List of Tensors. Each Tensor is (1, 768) typically.
        If a single sentence is > 512 tokens, it will be (N_chunks, 768).
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
                # Split into chunks (handles rare cases where one sentence > 512 tokens)
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
                # If a sentence was split into multiple chunks, we can average them or take the first.
                # Here we keep them all (stacking logic handled downstream)
                valid_features = pooled[chunk_cursor : chunk_cursor + n_chunks]
                output_list.append(valid_features)
                chunk_cursor += n_chunks
            else:
                output_list.append(torch.zeros((0, self.embed_dim)))

        return output_list

# ================= Helper Functions =================

def split_text_by_sentence(text: str) -> List[str]:
    """
    Splits text strictly by ". " (period + space).
    Returns a list of non-empty strings.
    """
    if not text:
        return []
    
    # Split by delimiter
    parts = text.split(". ")
    
    # Clean and filter
    sentences = []
    for p in parts:
        s = p.strip()
        if s:
            sentences.append(s)
            
    return sentences

def normalize_modal_key(pair_list: List[str]) -> Tuple[str, ...]:
    """Sorts modal pair list to ensure consistent keys"""
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
    print("Indexing knowledge data...")
    knowledge_map = {}
    for pid, entries in knowledge_data.items():
        knowledge_map[pid] = {}
        for entry in entries:
            key = normalize_modal_key(entry.get("modalPairs", []))
            if key:
                knowledge_map[pid][key] = entry.get("knowledge", "")

    # 3. Pre-process Analysis Data into a Lookup Map
    print("Indexing analysis data...")
    analysis_map = {}
    for pid, entries in analysis_data.items():
        analysis_map[pid] = {}
        for entry in entries:
            key = normalize_modal_key(entry.get("modalPairs", []))
            if key: 
                analysis_map[pid][key] = entry

    # 4. Initialize Encoder
    encoder = ClinicalBertEncoder()
    
    final_dataset = {}
    
    # Get Union of all Patient IDs
    all_patient_ids = sorted(list(set(analysis_data.keys()) | set(knowledge_data.keys())))
    print(f"Processing {len(all_patient_ids)} patients. Max augmentations: {max_augmentations}")
    
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

            # === Retrieve Data ===
            analysis_entry = p_analysis_entries.get(modal_key, {})
            raw_relationship = analysis_entry.get("relationship", "")
            raw_survival = analysis_entry.get("survival", "")
            score = analysis_entry.get("score", 0)
            raw_knowledge = p_knowledge_entries.get(modal_key, "")
            
            # === Split into Sentences (Core Logic Change) ===
            # We split strictly by ". "
            sentences_rel = split_text_by_sentence(raw_relationship)
            sentences_surv = split_text_by_sentence(raw_survival)
            sentences_know = split_text_by_sentence(raw_knowledge)
            
            # Add prefix to the first sentence of specific sections if needed, 
            # or keep them pure. Here we keep them pure but you could prepend "Analysis:" etc.
            # To preserve context, let's prepend the score to the very first sentence of relationship if available
            if sentences_rel:
                sentences_rel[0] = f"Score {score}. " + sentences_rel[0]
            elif sentences_surv:
                 sentences_surv[0] = f"Score {score}. " + sentences_surv[0]

            knowledge_list_tensors = []
            
            # Loop for Original + Augmentations
            # i=0 is Original, i>0 are shuffled versions
            for i in range(1 + max_augmentations):
                
                # Make copies of lists to avoid modifying original
                curr_rel = sentences_rel[:]
                curr_surv = sentences_surv[:]
                curr_know = sentences_know[:]
                
                # Apply Augmentation (Shuffling) only if i > 0
                if i > 0:
                    random.shuffle(curr_rel)
                    random.shuffle(curr_surv)
                    random.shuffle(curr_know)
                
                # Combine all sentences into one flat list for this augmentation round
                # Order: Relationship Sentences -> Survival Sentences -> Knowledge Sentences
                combined_sentences = curr_rel + curr_surv + curr_know
                
                if not combined_sentences:
                    # Handle empty case
                    knowledge_list_tensors.append(torch.zeros((0, 768)))
                    continue
                
                # === Encode ===
                # This returns a list of tensors, where each tensor corresponds to a sentence
                encoded_list = encoder._encode_text_list(combined_sentences)
                
                # Stack them: Result shape (Total_Num_Sentences, 768)
                # Note: if a single sentence was split into chunks by BERT limit, encoded_list[j] might have shape (n_chunks, 768)
                # torch.cat on dim=0 handles both (1, 768) and (n, 768) seamlessly
                if encoded_list:
                    stacked_tensor = torch.cat(encoded_list, dim=0)
                else:
                    stacked_tensor = torch.zeros((0, 768))
                    
                knowledge_list_tensors.append(stacked_tensor)
            
            # Store in result dictionary
            patient_features[modal_key] = {
                "score": score,
                "knowledge_list": knowledge_list_tensors, # List of tensors [Original_Stacked, Aug1_Stacked...]
            }
        
        if patient_features:
            final_dataset[patient_id] = patient_features

    # 4. Save to Pickle
    print(f"Saving combined features to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(final_dataset, f)
        
    print(f"Done. Processed {len(final_dataset)} patients.")

if __name__ == "__main__":
    # Seed for reproducibility of shuffling
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    process_data_fusion(ANALYSIS_FILE, KNOWLEDGE_FILE, OUTPUT_FILE, MAX_AUGMENTATIONS)