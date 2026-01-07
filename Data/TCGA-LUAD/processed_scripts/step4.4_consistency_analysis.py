import json
import pandas as pd
import numpy as np
import os
from scipy.stats import rankdata
from collections import Counter, defaultdict

# --- Configuration: File Paths ---
# Please ensure these paths are correct in your environment
FILE_PATHS = {
    'kimi': '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/medical_analysis_kimi.json',
    'qwen': '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/medical_analysis_qwen.json',
    'deepseek': '/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/medical_analysis_deepseek.json'
}

def load_json_data(filepath):
    """Loads JSON data with error handling."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def normalize_modal_pair_key(pair_list):
    """
    Sorts the modal pair list to ensure 'clinical-pathology' 
    is treated the same as 'pathology-clinical'.
    """
    return "-".join(sorted(pair_list))

def calculate_kendall_w_corrected(ratings_matrix):
    """
    Calculates Kendall's Coefficient of Concordance (W) with correction for ties.
    
    Args:
        ratings_matrix (np.array): shape (m_raters, n_objects) containing raw scores.
        
    Returns:
        float: Kendall's W value (0 to 1).
    """
    m, n = ratings_matrix.shape
    
    if n <= 1:
        return 0.0 # Cannot rank single items
        
    # 1. Convert raw scores to ranks (handling ties by averaging)
    # axis=1 ranks across the columns (objects) for each row (rater)
    ranked_matrix = np.apply_along_axis(lambda x: rankdata(x, method='average'), 1, ratings_matrix)
    
    # 2. Sum of ranks for each object (column)
    R_j = np.sum(ranked_matrix, axis=0)
    
    # 3. Mean of sums of ranks
    R_bar = np.mean(R_j)
    
    # 4. Sum of squared deviations (S)
    S = np.sum((R_j - R_bar) ** 2)
    
    # 5. Calculate Tie Correction Factor (T)
    # T_i = sum(t^3 - t) for each rater, where t is count of each tied rank
    T_correction = 0
    for rater_ranks in ranked_matrix:
        # Count occurrences of each rank
        counts = Counter(rater_ranks).values()
        # Sum (t^3 - t) for ranks that appear more than once
        t_sum = sum(t**3 - t for t in counts if t > 1)
        T_correction += t_sum
        
    # 6. Calculate W
    # Denominator with tie correction
    denominator = (m**2 * (n**3 - n)) - (m * T_correction)
    
    if denominator == 0:
        return 0.0
        
    W = (12 * S) / denominator
    return W

def analyze_data():
    # 1. Load Data
    data_sources = {}
    for model_name, path in FILE_PATHS.items():
        data = load_json_data(path)
        if data is None:
            return
        data_sources[model_name] = data
        print(f"Loaded {model_name}: {len(data)} patients")

    # 2. Find common patients
    patient_sets = [set(d.keys()) for d in data_sources.values()]
    common_patients = sorted(list(set.intersection(*patient_sets)))
    print(f"Number of common patients to analyze: {len(common_patients)}")

    w_scores_list = []
    # Structure to hold raw scores: scores_accumulator[pair_key][model_name] = [score1, score2, ...]
    scores_accumulator = defaultdict(lambda: defaultdict(list))
    
    # 3. Iterate through patients
    for pid in common_patients:
        # Dictionary to store scores: { 'clinical-pathology': {'kimi': 9, 'qwen': 8...} }
        pair_scores = {}
        
        valid_patient = True
        
        # Extract and Align Scores
        for model_name in data_sources.keys():
            patient_data = data_sources[model_name].get(pid, [])
            
            # Check if data is missing or malformed
            if not patient_data or len(patient_data) != 6:
                valid_patient = False
                break
                
            for entry in patient_data:
                pair_key = normalize_modal_pair_key(entry['modalPairs'])
                if pair_key not in pair_scores:
                    pair_scores[pair_key] = {}
                pair_scores[pair_key][model_name] = entry['score']
        
        if not valid_patient or len(pair_scores) != 6:
            continue

        # Create the matrix for this patient: Rows = Models, Cols = Modal Pairs
        # Ensure consistent order of pairs
        sorted_pairs = sorted(pair_scores.keys())
        models = list(data_sources.keys())
        
        # Matrix shape: (3 models, 6 pairs)
        score_matrix = np.zeros((len(models), len(sorted_pairs)))
        
        for r_idx, model in enumerate(models):
            for c_idx, pair in enumerate(sorted_pairs):
                val = pair_scores[pair][model]
                score_matrix[r_idx, c_idx] = val
                # Accumulate raw score for stats
                scores_accumulator[pair][model].append(val)

        # --- Analysis 1: Kendall's W (Overall Consistency) ---
        w_score = calculate_kendall_w_corrected(score_matrix)
        w_scores_list.append(w_score)

    # 4. Output Results
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS RESULTS (Console Only)")
    print("="*60)
    
    # --- Kendall's W Stats ---
    w_mean = np.mean(w_scores_list)
    w_std = np.std(w_scores_list)
    print(f"\n[Kendall's Coefficient of Concordance (W)]")
    print(f"  Mean W across {len(w_scores_list)} patients: {w_mean:.4f}")
    print(f"  Std Dev of W: {w_std:.4f}")

    # --- Score Stats per Modal Pair ---
    print(f"\n[Score Statistics per Modal Pair (Mean ± Std)]")
    # Header
    print(f"{'Modal Pair':<30} | {'Kimi':<20} | {'Qwen':<20} | {'DeepSeek':<20}")
    print("-" * 100)
    
    # We want consistent model order for printing
    model_order = ['kimi', 'qwen', 'deepseek'] 
    
    for pair in sorted(scores_accumulator.keys()):
        row_str = f"{pair:<30} | "
        for model in model_order:
            scores = scores_accumulator[pair][model]
            if scores:
                m_mean = np.mean(scores)
                m_std = np.std(scores)
                stats_str = f"{m_mean:.2f} ± {m_std:.2f}"
            else:
                stats_str = "N/A"
            row_str += f"{stats_str:<20} | "
        print(row_str)

if __name__ == "__main__":
    analyze_data()