import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np
import random
from typing import List, Optional, Tuple, Dict
from transformers import AutoTokenizer, AutoModel

# ================= Configuration & Paths =================
BASE_DIR = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset"
INPUT_CSV = os.path.join(BASE_DIR, "clinical_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed")

# Ensure output dir exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output Paths
OUTPUT_CLINICAL_PKL = os.path.join(OUTPUT_DIR, "features_text_clinical.pkl")
OUTPUT_TREATMENT_PKL = os.path.join(OUTPUT_DIR, "features_text_treatment.pkl")
OUTPUT_PATHOLOGY_PKL = os.path.join(OUTPUT_DIR, "features_text_pathology.pkl")
OUTPUT_ALL_OPTIONS_PKL = os.path.join(OUTPUT_DIR, "features_all_treatment_options.pkl")


# ================= ClinicalBERT Encoder =================
class ClinicalBertEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.embed_dim = 768 
        
        print("Initializing Text Encoder (ClinicalBERT)")
        self.text_model_name = "medicalai/ClinicalBERT"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name, use_fast=True)
        self.bert = AutoModel.from_pretrained(self.text_model_name)
            
        self.to(self.device)
        self.eval() 

    def _chunk_token_ids(self, ids: List[int], chunk_size: int) -> List[List[int]]:
        return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    def _encode_text(self, texts_list: List[str]) -> List[torch.Tensor]:
        """
        Encodes a batch of texts.
        Returns: List[torch.Tensor], where each tensor is (N_chunks, 768).
        Tensors are moved to CPU.
        """
        batch_size = len(texts_list)
        chunk_payload = 510 
        
        all_chunks = []
        mapping_info = [] 
        
        # 1. Tokenization & Chunking
        for i, text in enumerate(texts_list):
            item_specific_chunks = [] 
            
            if isinstance(text, str) and text.strip():
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                chunks = self._chunk_token_ids(token_ids, chunk_payload) 
                if chunks:
                    item_specific_chunks.extend(chunks)
            
            if not item_specific_chunks:
                mapping_info.append({'index': i, 'n': 0})
            else:
                mapping_info.append({'index': i, 'n': len(item_specific_chunks)})
                all_chunks.extend(item_specific_chunks)

        # Handle case where all inputs are empty
        if not all_chunks:
            return [torch.zeros(0, self.embed_dim) for _ in range(batch_size)]

        # 2. Batch Encoding
        # Flatten all chunks into one large batch
        inputs = self.tokenizer(
            [' '.join(self.tokenizer.convert_ids_to_tokens(c)) for c in all_chunks],
            return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            bert_outputs = self.bert(**inputs)
        
        # Use CLS token (idx 0) from last hidden state: (Total_Chunks, 768)
        pooled = bert_outputs.last_hidden_state[:, 0, :] 

        # 3. Reconstruction
        output_list = []
        chunk_cursor = 0
        
        for i in range(batch_size):
            info = next((m for m in mapping_info if m['index'] == i), None)
            n_chunks = info['n'] if info else 0
            
            if n_chunks > 0:
                # Extract chunks belonging to this text
                valid_features = pooled[chunk_cursor : chunk_cursor + n_chunks] # (n_chunks, 768)
                output_list.append(valid_features.cpu())
                chunk_cursor += n_chunks
            else:
                # Return empty tensor for empty text
                output_list.append(torch.zeros((0, self.embed_dim)))

        return output_list

# ================= Augmentation Logic =================
def augment_text(text: str) -> List[str]:
    """
    Splits text by '+' or '.', shuffles segments, and rejoins.
    Max 10 variations. First element is always original text.
    """
    if not text or not isinstance(text, str):
        return []
    
    augmented_versions = []
    
    # Check delimiters
    delimiters = []
    if '+' in text: 
        delimiters.append('+')
    if '.' in text: 
        delimiters.append('.')
    
    # If no delimiters, we can't augment structurally, return original
    if not delimiters:
        return [text]

    # Use the first valid delimiter found for splitting logic
    for tag in delimiters:
        parts = [p.strip() for p in text.split(tag) if p.strip()]
    
        for _ in range(10): # Try 10 times
            
            shuffled_parts = parts.copy()
            random.shuffle(shuffled_parts)
            
            # Rejoin
            if tag == '.':
                new_text = ". ".join(shuffled_parts) + "."
            else:
                new_text = "+".join(shuffled_parts)
            
            if new_text != text and new_text not in augmented_versions:
                augmented_versions.append(new_text)

    return [text] + augmented_versions

# ================= Text Generation Logic =================
def generate_patient_texts(patient_series: pd.Series) -> Dict[str, str]:
    """
    Generates 'clinical', 'treatment', and 'pathology' strings from a row.
    """
    
    # Define source columns
    sources_with_columns = {
        "clinical": [
            "TumorLocation",
            "PreoperativeHistoryDetails",
            "Age(Y)", 
            "Gender(0male/1female)",
            "TumorT", "TumorN", "TumorM"
        ],
        "treatment": [
            "SurgicalMethod",
            "Radiotherapy(0no/1yes)", 
            "Chemotherapy(0no/1yes)"
        ],
        "pathology": [
            "Pathology",
            "Flap",
            "PD_L1",
            "TumorDifferentiation(1high/2med/3low)"
        ]
    }
    
    result_texts = {}

    def get_sentence(column_name, value):
        if pd.isna(value) or str(value).strip() in ['/', '', 'nan']: return None
        if isinstance(value, float) and value.is_integer(): value = int(value)
        
        sentence = ""


        # Staging
        if column_name == "TumorT": sentence = f"The primary tumor stage (T stage) is {value}."
        elif column_name == "TumorN": sentence = f"The regional lymph node stage (N stage) is {value}."
        elif column_name == "TumorM": sentence = f"The distant metastasis stage (M stage) is {value}."
        
        # Differentiation
        elif column_name == "TumorDifferentiation(1high/2med/3low)":
            diff_map = {1: "well-differentiated", 2: "moderately-differentiated", 3: "poorly-differentiated"}
            diff = diff_map.get(value, None)
            if diff: sentence = f"The tumor differentiation is {diff}."
        
        # Binary / Status
        elif "(0/1)" in column_name or "(+)" in column_name:
            status = "present" if value == 1 else "absent"
            feature_name = column_name.replace("(0/1)", "").replace("(+)", "").replace("_", " ")
            sentence = f"{feature_name} is {status}."
        elif "(0no/1yes)" in column_name:
            status = "yes" if value == 1 else "no"
            feature_name = column_name.replace("(0no/1yes)", "").replace("History", " history")
            sentence = f"The patient has a record of {feature_name}: {status}."
            
        # Demographics
        elif column_name == "Age(Y)": sentence = f"The patient's age is {value} years."
        elif column_name == "Gender(0male/1female)": sentence = f"The patient is {'female' if value == 1 else 'male'}."
        
        # Descriptive columns
        elif column_name in ["Pathology", "SurgicalMethod", "TumorLocation", "Ki-67", "PD_L1", "PreoperativeHistoryDetails", "Flap"]:
            if len(str(value)) > 1:
                sentence = f"The {column_name.lower()} is recorded as: {value}."
        
        return sentence

    for category, cols in sources_with_columns.items():
        cat_sentences = []
        for col in cols:
            if col in patient_series:
                val = patient_series[col]
                sent = get_sentence(col, val)
                if sent:
                    cat_sentences.append(sent)
        
        # Join based on category type
        if category == 'treatment' and "12_treatment_type" in cols:
            final_str = "".join(cat_sentences) 
        else:
            final_str = " ".join(cat_sentences)
            
        result_texts[category] = final_str

    return result_texts

# ================= Helper: Validation =================
def filter_valid_tensors(tensor_list: List[torch.Tensor], dim: int = 768) -> List[torch.Tensor]:
    """
    Filters list of tensors.
    Keeps only tensors with shape (n, dim) where n > 0.
    """
    valid_list = []
    for t in tensor_list:
        # Check dimensions: (n, 768)
        if t.ndim == 2 and t.shape[1] == dim and t.shape[0] > 0:
            valid_list.append(t)
    return valid_list

# ================= Main Execution =================
def main():
    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory {BASE_DIR} does not exist.")
        return

    # Initialize Model
    encoder = ClinicalBertEncoder(device=device)

    # Load Data
    print(f"Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {INPUT_CSV}")
        return
    
    # Ensure PID is string
    df['PID'] = df['PID'].astype(str).str.strip()
    
    # ---------------------------------------------------------
    # Task 1: Generate All Treatment Options (Labels)
    # ---------------------------------------------------------
    print("\n--- Processing All Treatment Options ---")
    if '12_treatment_type' in df.columns:
        unique_treatments = sorted(df['12_treatment_type'].dropna().unique().tolist())
        unique_treatments = [str(x).strip() for x in unique_treatments if str(x).strip()]
        
        print(f"Found {len(unique_treatments)} unique treatment combinations.")
        
        # Encode without augmentation
        # Returns List[Tensor]
        all_opt_feats_raw = encoder._encode_text(unique_treatments)
        
        # Validate (must be n > 0, d = 768)
        # For options list, we expect 1-to-1 mapping, so we handle empty differently if needed.
        # But assuming inputs are stripped and valid, usually result is valid.
        valid_options_feat_list = []
        valid_options_str_list = []
        
        for txt, feat in zip(unique_treatments, all_opt_feats_raw):
            if feat.shape[0] > 0 and feat.shape[1] == 768:
                valid_options_feat_list.append(feat)
                valid_options_str_list.append(txt)
            else:
                print(f"Warning: Treatment option '{txt}' resulted in empty features. Skipping.")

        with open(OUTPUT_ALL_OPTIONS_PKL, 'wb') as f:
            pickle.dump({
                "ALL_TREATMENT_OPTIONS_STR": valid_options_str_list,
                "ALL_TREATMENT_OPTIONS_FEAT": valid_options_feat_list
            }, f)
        print(f"Saved {OUTPUT_ALL_OPTIONS_PKL}")
    else:
        print("Warning: '12_treatment_type' column missing. Skipping options generation.")

    # ---------------------------------------------------------
    # Task 2: Process Patient Data (Clinical, Treatment, Pathology)
    # ---------------------------------------------------------
    print("\n--- Processing Patient Data (Clinical, Treatment, Pathology) ---")
    
    clinical_dict = {}
    treatment_dict = {}
    pathology_dict = {}
    
    total_patients = len(df)
    
    for idx, row in df.iterrows():
        pid = row['PID']
        
        # 1. Generate Base Texts
        generated_texts = generate_patient_texts(row)
        
        # 2. Process each modality
        tasks = [
            ('clinical', clinical_dict),
            ('treatment', treatment_dict),
            ('pathology', pathology_dict)
        ]
        
        for modal_name, modal_dict in tasks:
            base_text = generated_texts.get(modal_name, "")
            
            # Augment (Returns list: [Original, Aug1, Aug2...])
            text_versions = augment_text(base_text)
            
            # Encode
            # Returns List[Tensor(n, 768)]
            raw_feats = encoder._encode_text(text_versions)
            
            # Validate & Filter
            # Requirement: Check (n, d) where d=768 and n>0
            valid_feats = filter_valid_tensors(raw_feats, dim=768)
            
            # Only save if we have valid features
            if valid_feats:
                modal_dict[pid] = valid_feats

        if idx % 50 == 0:
            print(f"Processed {idx}/{total_patients} patients...")

    # ---------------------------------------------------------
    # Task 3: Save Output Files
    # ---------------------------------------------------------
    print(f"\nSaving outputs to {OUTPUT_DIR}...")
    
    with open(OUTPUT_CLINICAL_PKL, 'wb') as f:
        pickle.dump(clinical_dict, f)
    print(f"Saved Clinical features: {len(clinical_dict)} patients")

    with open(OUTPUT_TREATMENT_PKL, 'wb') as f:
        pickle.dump(treatment_dict, f)
    print(f"Saved Treatment features: {len(treatment_dict)} patients")

    with open(OUTPUT_PATHOLOGY_PKL, 'wb') as f:
        pickle.dump(pathology_dict, f)
    print(f"Saved Pathology features: {len(pathology_dict)} patients")

    # Verification Print
    if len(treatment_dict) > 0:
        first_pid = list(treatment_dict.keys())[0]
        first_feat = treatment_dict[first_pid][0]
        print(f"\n[Verification] Sample PID: {first_pid}")
        print(f"Feature List Length: {len(treatment_dict[first_pid])}")
        print(f"First Tensor Shape: {first_feat.shape} (Expected: n>0, 768)")

    print("\nDone!")

if __name__ == "__main__":
    main()