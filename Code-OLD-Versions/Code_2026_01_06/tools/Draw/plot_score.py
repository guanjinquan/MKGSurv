import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
from collections import Counter, defaultdict

# --- Configuration ---
DATASETS = [
    ("TCGA-LUAD", "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD"),
    ("TCGA-LUSC", "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUSC"),
    ("TCGA-BRCA", "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-BRCA"),
    ("TCGA-KIRC", "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-KIRC"),
    ("Multi-OSCC", "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset"),
]

MODELS = ['deepseek', 'kimi', 'qwen']
MODEL_COLORS = {'deepseek': '#D62728', 'kimi': '#1F77B4', 'qwen': '#2CA02C'} # Nature-like palette

# Mapping for abbreviations
MODALITY_MAP = {
    "clinical": "C",
    "pathology": "P",
    "treatment": "T",
    "genomics": "G"
}

def normalize_pair(pair_list):
    """Sorts and abbreviates modal pairs."""
    abbr = sorted([MODALITY_MAP.get(m.lower(), m) for m in pair_list])
    return "&".join(abbr)

def calculate_kendall_w(ratings_matrix):
    """Calculates Kendall's Coefficient of Concordance (W) with tie correction."""
    m, n = ratings_matrix.shape
    if n < 2: return 0.0
    
    # Convert raw scores to ranks (averaging ties)
    ranked_matrix = np.apply_along_axis(lambda x: rankdata(x, method='average'), 1, ratings_matrix)
    
    R_j = np.sum(ranked_matrix, axis=0)
    R_bar = np.mean(R_j)
    S = np.sum((R_j - R_bar) ** 2)
    
    T_correction = 0
    for rater_ranks in ranked_matrix:
        counts = Counter(rater_ranks).values()
        T_correction += sum(t**3 - t for t in counts if t > 1)
        
    denominator = (m**2 * (n**3 - n)) - (m * T_correction)
    return (12 * S) / denominator if denominator != 0 else 0.0

def process_dataset(name, base_path):
    print(f"Processing {name}...")
    dataset_data = {model: {} for model in MODELS}
    
    for model in MODELS:
        file_path = os.path.join(base_path, "processed", f"medical_analysis_{model}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset_data[model] = json.load(f)
        else:
            print(f"  Warning: {file_path} not found.")

    # Find common patients
    common_pids = set.intersection(*(set(d.keys()) for d in dataset_data.values() if d))
    
    all_w = []
    pair_stats = defaultdict(lambda: {m: [] for m in MODELS})

    for pid in common_pids:
        # Extract scores for this patient across all models
        p_scores = {} # pair -> {model: score}
        for model in MODELS:
            for entry in dataset_data[model][pid]:
                pair_key = normalize_pair(entry['modalPairs'])
                if pair_key not in p_scores: p_scores[pair_key] = {}
                p_scores[pair_key][model] = entry['score']
        
        # Keep only pairs present in all 3 models
        valid_pairs = sorted([p for p, s in p_scores.items() if len(s) == len(MODELS)])
        if len(valid_pairs) < 2: continue
        
        matrix = np.zeros((len(MODELS), len(valid_pairs)))
        for r, model in enumerate(MODELS):
            for c, pair in enumerate(valid_pairs):
                val = p_scores[pair][model]
                matrix[r, c] = val
                pair_stats[pair][model].append(val)
        
        all_w.append(calculate_kendall_w(matrix))

    # Calculate final stats for plotting
    plot_df = []
    for pair, model_data in pair_stats.items():
        for model, scores in model_data.items():
            if scores:
                plot_df.append({
                    'Pair': pair,
                    'Model': model.capitalize(),
                    'Mean': np.mean(scores),
                    'Std': np.std(scores)
                })
    
    return pd.DataFrame(plot_df), np.mean(all_w) if all_w else 0.0

def main():
    # Nature-style styling
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 1.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'font.size': 11
    })

    # Restored to a more balanced aspect ratio to avoid overly tall bars
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    
    for i, (name, path) in enumerate(DATASETS):
        df, mean_w = process_dataset(name, path)
        
        if df.empty:
            axes[i].text(0.5, 0.5, "No Common Data", ha='center')
            continue

        # Sort pairs for consistent X axis
        df = df.sort_values('Pair')
        pairs = df['Pair'].unique()
        models = df['Model'].unique()
        
        x = np.arange(len(pairs))
        # Bar width from your provided version
        width = 0.3
        
        for j, model in enumerate(models):
            sub = df[df['Model'] == model]
            axes[i].bar(x + (j - 1) * width, sub['Mean'], width, 
                        yerr=sub['Std'], label=model if i == 0 else "",
                        color=MODEL_COLORS[model.lower()], capsize=2,
                        error_kw={'elinewidth': 1.0, 'markeredgewidth': 1.0},
                        alpha=0.9, edgecolor='black', linewidth=0.7)

        # Removed xlabel and moved Name and Kendall's W into a single top text box
        axes[i].text(0.5, 0.98, f"{name}\nKendall's $W$ = {mean_w:.3f}", 
                    transform=axes[i].transAxes, ha='center', va='top',
                    fontsize=9.5, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(pairs, rotation=45, ha='right', fontsize=9.5)
        axes[i].set_ylim(0, 11)
        axes[i].grid(axis='y', linestyle=':', alpha=0.4)
        
        # Style cleanup
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        
        if i == 0:
            axes[i].set_ylabel("Relevance Score (0-10)", fontweight='bold', fontsize=12)
            axes[i].legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.15), ncol=3, fontsize=10.5)

    # Tighten subplots gap
    plt.subplots_adjust(wspace=0.15, bottom=0.2)
    
    output_path = "/home/Guanjq/NewWork/MedAlignFusion/Code/tools/Draw/Results/modal_reliability_analysis.svg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='svg')
    # plt.show()

if __name__ == "__main__":
    main()