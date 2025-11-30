import os

# 1. Set HF Mirror (Must be before importing transformers)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.manifold import TSNE
import re

# ================= Configuration =================
INPUT_FILE = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/medical_analysis_deepseek.json"
# This will serve as the base path/directory for outputs
OUTPUT_BASE_PATH = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed_scripts/vis.png"
MODEL_NAME = "medicalai/ClinicalBERT"

# ================= Core Encoder Class =================
class ClinicalBertEncoder(nn.Module):
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        self.embed_dim = 768 
        
        print(f"Initializing Text Encoder ({MODEL_NAME}) on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
            self.bert = AutoModel.from_pretrained(MODEL_NAME)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
            
        self.to(self.device)
        self.eval() 

    def _chunk_token_ids(self, ids: List[int], chunk_size: int) -> List[List[int]]:
        return [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]

    def encode_and_pool(self, texts_list: List[str]) -> np.ndarray:
        """
        Encodes list of texts, concatenates them, and then performs MEAN POOLING
        to return a single (768,) numpy vector.
        """
        batch_size = len(texts_list)
        chunk_payload = 510 
        
        all_chunks = []
        mapping_info = [] 
        
        # --- Preprocessing & Chunking ---
        for i, text in enumerate(texts_list):
            item_specific_chunks = [] 
            if isinstance(text, str) and text.strip():
                clean_text = text.strip()
                token_ids = self.tokenizer.encode(clean_text, add_special_tokens=False)
                chunks = self._chunk_token_ids(token_ids, chunk_payload) 
                if chunks:
                    item_specific_chunks.extend(chunks)
            
            if not item_specific_chunks:
                mapping_info.append({'index': i, 'n': 0})
            else:
                mapping_info.append({'index': i, 'n': len(item_specific_chunks)})
                all_chunks.extend(item_specific_chunks)

        if not all_chunks:
            return np.zeros(self.embed_dim)

        # --- Batch Inference ---
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
            
            # Extract CLS token
            batch_pooled = bert_outputs.last_hidden_state[:, 0, :]
            pooled_outputs.append(batch_pooled)
            
        if pooled_outputs:
            pooled = torch.cat(pooled_outputs, dim=0) # (Total_Chunks, 768)
        else:
            return np.zeros(self.embed_dim)

        # --- Reconstruction per text input (Concatenation logic) ---
        # Note: Your logic requires concatenating Part1 and Part2 features first
        output_list = []
        chunk_cursor = 0
        
        for i in range(batch_size):
            info = next((m for m in mapping_info if m['index'] == i), None)
            n_chunks = info['n'] if info else 0
            
            if n_chunks > 0:
                valid_features = pooled[chunk_cursor : chunk_cursor + n_chunks] # (N_chunks, 768)
                output_list.append(valid_features)
                chunk_cursor += n_chunks
            else:
                output_list.append(torch.zeros((0, self.embed_dim)).to(self.device))

        # --- Final Aggregation: Concat Part1 & Part2 then Mean Pool ---
        # output_list[0] is feat_1, output_list[1] is feat_2
        if len(output_list) >= 2:
            combined = torch.cat(output_list, dim=0) # (N_total_chunks, 768)
        else:
            combined = output_list[0] if output_list else torch.zeros((1, self.embed_dim)).to(self.device)
            
        # *** Crucial Step: Mean Pooling to get (1, 768) ***
        # Avoid mean on empty tensor
        if combined.size(0) == 0:
             final_vector = torch.zeros(self.embed_dim).cpu().numpy()
        else:
             final_vector = torch.mean(combined, dim=0).cpu().numpy() # Shape (768,)
             
        return final_vector

# ================= Main Pipeline =================

def save_plot(df, output_path, title_suffix=""):
    """Helper function to generate and save a scatter plot"""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Scatter Plot
    # X: t-SNE feature, Y: Normalized Score, Hue: Pair Type
    scatter = sns.scatterplot(
        data=df,
        x='tsne_1d',
        y='norm_score',  # Use Normalized Score
        hue='pair',
        style='pair', 
        palette='viridis',
        s=100,
        alpha=0.8,
        edgecolor='w'
    )
    
    plt.title(f'Distribution of Medical Analysis Features (t-SNE 1D vs Norm Score)\n{title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1 (Semantic Feature Space)', fontsize=14)
    plt.ylabel('Normalized Score (Min-Max per Patient)', fontsize=14)
    plt.legend(title='Modality Pairs', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set Y-axis limit strictly to [0, 1] with some padding
    plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Close memory
    print(f"Visualization saved to: {output_path}")

def run_pipeline():
    # 1. Load Data
    print(f"Loading analysis file: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. Initialize Encoder
    encoder = ClinicalBertEncoder()
    
    # List to store processed data for visualization
    # Format: [{'pair': str, 'norm_score': float, 'feature': numpy_array}, ...]
    viz_data = []
    
    print(f"Processing {len(data)} patients...")
    
    # 3. Feature Extraction Loop
    total_entries = sum(len(v) for v in data.values())
    pbar = tqdm(total=total_entries, desc="Encoding & Pooling")
    
    for patient_id, analysis_list in data.items():
        if not analysis_list:
            continue
            
        # --- Pre-calculate stats for Normalization (Per Patient) ---
        scores = [e.get("score", 0) for e in analysis_list]
        if not scores:
            continue
            
        min_s = min(scores)
        max_s = max(scores)
        score_range = max_s - min_s
        
        for entry in analysis_list:
            # key: stringify tuple for visualization labels, e.g., "Clinical+Pathology"
            modal_pairs = entry.get("modalPairs", [])
            if not modal_pairs:
                pbar.update(1)
                continue
                
            pair_label = " + ".join(sorted(modal_pairs)) # Sort to ensure "A+B" == "B+A"
            
            # Construct Text
            raw_score = entry.get("score", 0)
            relationship_text = entry.get("relationship", "")
            survival_text = entry.get("survival", "")
            
            text_part_1 = f"Score: {raw_score}. Relationship Analysis: {relationship_text}"
            text_part_2 = f"Survival Risk Analysis: {survival_text}"
            
            # Encode -> Concat -> Mean Pool -> Numpy
            feature_vector = encoder.encode_and_pool([text_part_1, text_part_2])
            
            # --- Normalize Score ---
            if score_range == 0:
                # If max == min (e.g., patient has only one pair, or all pairs score same),
                # we map it to 0.0 (baseline) or 1.0. 
                # Standard MinMax scaler maps constant values to 0. 
                # Or if you prefer to show them as "Top" (1.0), change this to 1.0.
                norm_score = 0.0 
            else:
                norm_score = (raw_score - min_s) / score_range
            
            viz_data.append({
                'pair': pair_label,
                'norm_score': float(norm_score),
                'feature': feature_vector
            })
            pbar.update(1)
            
    pbar.close()
    
    if not viz_data:
        print("No valid data found to visualize.")
        return

    # 4. Prepare Data for t-SNE
    print("Preparing data for t-SNE...")
    df_viz = pd.DataFrame(viz_data)
    
    # Stack features into matrix (N_samples, 768)
    X = np.stack(df_viz['feature'].values)
    
    # Check if we have enough samples
    n_samples = X.shape[0]
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    
    print(f"Running t-SNE (1 component) on matrix shape {X.shape} with perplexity={perplexity}...")
    
    # 5. t-SNE Dimensionality Reduction (d -> 1)
    # FIX: Removed n_iter=1000 to avoid TypeError on some sklearn versions
    tsne = TSNE(n_components=1, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X) # Shape (N, 1)
    
    # Add embedded coordinate to DataFrame
    df_viz['tsne_1d'] = X_embedded.flatten()
    
    # 6. Visualization
    output_dir = os.path.dirname(OUTPUT_BASE_PATH)
    base_name = os.path.basename(OUTPUT_BASE_PATH)
    name_no_ext, ext = os.path.splitext(base_name)
    
    print("Generating plots...")
    
    # 6a. Global Plot (All pairs)
    global_path = os.path.join(output_dir, f"{name_no_ext}_ALL{ext}")
    save_plot(df_viz, global_path, title_suffix="All Pairs")
    
    # 6b. Individual Plots per Pair
    unique_pairs = df_viz['pair'].unique()
    print(f"Found {len(unique_pairs)} unique pairs. Generating individual plots...")
    
    for pair in unique_pairs:
        # Create a safe filename (replace spaces and + with underscores)
        safe_name = re.sub(r'[^\w\-_]', '_', pair)
        safe_name = re.sub(r'_{2,}', '_', safe_name) # Remove duplicate underscores
        
        pair_path = os.path.join(output_dir, f"{name_no_ext}_{safe_name}{ext}")
        
        # Subset data
        df_subset = df_viz[df_viz['pair'] == pair]
        
        # Save
        save_plot(df_subset, pair_path, title_suffix=f"Pair: {pair}")

if __name__ == "__main__":
    run_pipeline()