import json
import pandas as pd
import numpy as np
import os
from scipy.stats import rankdata
from collections import Counter, defaultdict

# --- Configuration: File Paths ---
# Please ensure these paths are correct in your environment
FILE_PATHS = {
    'kimi': '/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_kimi.json',
    'qwen': '/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_qwen.json',
    'deepseek': '/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/processed/medical_analysis_deepseek.json'
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
    
    # 至少需要2个对象才能进行排序比较
    if n < 2:
        return 0.0 
        
    # 1. Convert raw scores to ranks (handling ties by averaging)
    ranked_matrix = np.apply_along_axis(lambda x: rankdata(x, method='average'), 1, ratings_matrix)
    
    # 2. Sum of ranks for each object (column)
    R_j = np.sum(ranked_matrix, axis=0)
    
    # 3. Mean of sums of ranks
    R_bar = np.mean(R_j)
    
    # 4. Sum of squared deviations (S)
    S = np.sum((R_j - R_bar) ** 2)
    
    # 5. Calculate Tie Correction Factor (T)
    T_correction = 0
    for rater_ranks in ranked_matrix:
        counts = Counter(rater_ranks).values()
        t_sum = sum(t**3 - t for t in counts if t > 1)
        T_correction += t_sum
        
    # 6. Calculate W
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
    scores_accumulator = defaultdict(lambda: defaultdict(list))
    
    valid_patient_count = 0
    
    # 3. Iterate through patients
    for pid in common_patients:
        pair_scores = {}
        
        # Extract scores
        for model_name in data_sources.keys():
            patient_data = data_sources[model_name].get(pid, [])
            
            if not patient_data:
                continue
                
            for entry in patient_data:
                pair_key = normalize_modal_pair_key(entry['modalPairs'])
                if pair_key not in pair_scores:
                    pair_scores[pair_key] = {}
                pair_scores[pair_key][model_name] = entry['score']
        
        # 找出三个模型都打分了的 pair
        # 只有当一个 pair 在所有模型中都有分才算有效 pair
        models = list(data_sources.keys())
        valid_pairs = []
        for pair, scores in pair_scores.items():
            if len(scores) == len(models):
                valid_pairs.append(pair)
        
        # 排序以保证矩阵列顺序一致
        valid_pairs.sort()
        
        # 【关键修改】动态检查 pair 数量
        # 只要有效 pair 数量 >= 2，就可以计算一致性（不强制要求6个）
        if len(valid_pairs) < 2:
            continue
            
        valid_patient_count += 1

        # Create the matrix: Rows = Models, Cols = Valid Modal Pairs
        score_matrix = np.zeros((len(models), len(valid_pairs)))
        
        for r_idx, model in enumerate(models):
            for c_idx, pair in enumerate(valid_pairs):
                val = pair_scores[pair][model]
                score_matrix[r_idx, c_idx] = val
                scores_accumulator[pair][model].append(val)

        # --- Analysis 1: Kendall's W ---
        w_score = calculate_kendall_w_corrected(score_matrix)
        w_scores_list.append(w_score)

    # 4. Output Results
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS RESULTS (Console Only)")
    print("="*60)
    
    if len(w_scores_list) == 0:
        print("Error: No valid patients found with common modal pairs across all models.")
        print("Please check if the modal pair names match exactly across JSON files.")
        return

    # --- Kendall's W Stats ---
    w_mean = np.mean(w_scores_list)
    w_std = np.std(w_scores_list)
    print(f"\n[Kendall's Coefficient of Concordance (W)]")
    print(f"  Patients with valid data: {valid_patient_count}")
    print(f"  Number of pairs analyzed per patient: {len(scores_accumulator.keys())} (variable)")
    print(f"  Mean W: {w_mean:.4f}")
    print(f"  Std Dev of W: {w_std:.4f}")

    # --- Score Stats per Modal Pair ---
    print(f"\n[Score Statistics per Modal Pair (Mean ± Std)]")
    print(f"{'Modal Pair':<30} | {'Kimi':<20} | {'Qwen':<20} | {'DeepSeek':<20}")
    print("-" * 100)
    
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