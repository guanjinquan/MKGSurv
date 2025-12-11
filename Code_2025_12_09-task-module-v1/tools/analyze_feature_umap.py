import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import random

# ==========================================
# 1. Data Loading
# ==========================================
def load_data(jsonl_path, max_samples=2000):
    """
    读取数据。
    返回:
        all_vectors: numpy array (N_total, Dim) 包含所有 Token 和 Fused 向量
        labels: numpy array (N_total,) 对应的标签
        sample_map: List[Tuple(start, end, fused_idx)] 记录每个样本在 all_vectors 中的位置范围
    """
    all_vectors = []
    labels = []   
    sample_map = [] # (groups_start_idx, groups_end_idx, fused_idx)
    
    print(f"Loading data from {jsonl_path}...")
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            current_global_idx = 0
            
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                
                # 1. 处理 Groups (Tokens)
                group_tokens = data['groups']     # List of lists
                group_ids = data['group_ids']     # List of ints
                fused_vec = data['fused']         # List
                
                if not group_tokens:
                    continue

                # 记录该样本 Group Tokens 的起始位置
                groups_start = current_global_idx
                
                # 添加 Token 向量和标签
                for vec, g_id in zip(group_tokens, group_ids):
                    all_vectors.append(vec)
                    labels.append(f"Modality {g_id+1}")
                    current_global_idx += 1
                
                groups_end = current_global_idx
                
                # 2. 处理 Fused Vector
                all_vectors.append(fused_vec)
                labels.append("Fused (Ours)")
                fused_idx = current_global_idx
                current_global_idx += 1
                
                # 记录索引映射: [start, end) 是 tokens, fused_idx 是 fused vector
                sample_map.append((groups_start, groups_end, fused_idx))

    except FileNotFoundError:
        print(f"Error: File {jsonl_path} not found.")
        return np.array([]), np.array([]), []

    print(f"Loaded {len(sample_map)} samples. Total vectors (tokens+fused): {len(all_vectors)}")
    return np.array(all_vectors, dtype=np.float32), np.array(labels), sample_map

# ==========================================
# 2. Alignment (Linear Projection)
# ==========================================
def align_vectors_with_linear(vectors, sample_map, input_dim, train_steps=200):
    """
    训练一个线性层将 Fused 向量映射到该样本所有 Modality Tokens 的几何中心。
    """
    print(f"\n[Alignment] Training Linear Layer on Fused Vectors (Steps={train_steps})...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class AlignmentProjector(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim) 
        def forward(self, x):
            return self.norm(self.linear(x))
    
    model = AlignmentProjector(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    vectors_tensor = torch.tensor(vectors).to(device)
    
    # 构建训练对：(Input: Fused, Target: Specific Token)
    # 我们将 Fused 向量重复多次，分别对应它的每一个 Token
    input_indices = []
    target_indices = []
    unique_fused_indices = []
    
    for g_start, g_end, f_idx in sample_map:
        unique_fused_indices.append(f_idx)
        # 该样本所有的 Token 索引
        token_idxs = list(range(g_start, g_end))
        # Fused 索引重复 N 次
        input_indices.extend([f_idx] * len(token_idxs))
        target_indices.extend(token_idxs)
            
    input_idx_tensor = torch.tensor(input_indices, dtype=torch.long).to(device)
    target_idx_tensor = torch.tensor(target_indices, dtype=torch.long).to(device)
    unique_fused_tensor = torch.tensor(unique_fused_indices, dtype=torch.long).to(device)
    
    # Training Loop
    model.train()
    batch_size = 4096 # 如果 token 太多，分批处理
    num_pairs = len(input_indices)
    
    for step in range(train_steps):
        permutation = torch.randperm(num_pairs).to(device)
        
        total_loss = 0
        for i in range(0, num_pairs, batch_size):
            indices = permutation[i:i+batch_size]
            batch_in = input_idx_tensor[indices]
            batch_tgt = target_idx_tensor[indices]
            
            optimizer.zero_grad()
            
            # Input: Fused vectors
            inputs = vectors_tensor[batch_in]
            # Target: Modality Tokens
            targets = vectors_tensor[batch_tgt].detach() # Stop gradient on targets
            
            projected = model(inputs)
            loss = criterion(projected, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if step % 50 == 0:
            print(f"  Step {step}/{train_steps}, Loss: {total_loss / (num_pairs/batch_size):.6f}")

    # Apply Transformation
    model.eval()
    with torch.no_grad():
        aligned_vectors = vectors.copy()
        # 只取唯一的 fused 向量进行变换
        original_fused = vectors_tensor[unique_fused_tensor]
        transformed_fused = model(original_fused).cpu().numpy()
        aligned_vectors[unique_fused_indices] = transformed_fused
        
    print("[Alignment] Done.")
    return aligned_vectors

# ==========================================
# 3. PCA & Centroid Logic
# ==========================================
def plot_pca_centroid_inference(vectors, labels, save_plot_path="pca_inference.png", max_scatter_points=5000):
    """
    可视化逻辑：
    1. 计算所有 Modality Tokens 的真实平均值 (True Centroids)。
    2. 计算 Fused 向量离哪个 Centroid 最近。
    3. PCA 降维并绘图。为了避免图太乱，会对背景的 Token 点进行下采样。
    """
    if vectors.size == 0:
        return

    # --- 1. 分离数据 ---
    is_fused = np.array([l == "Fused (Ours)" for l in labels])
    
    X_tokens = vectors[~is_fused]
    y_tokens = labels[~is_fused]
    X_fused = vectors[is_fused]
    
    # --- 2. 计算真实质心 (使用所有 Token) ---
    unique_modalities = sorted([l for l in np.unique(y_tokens) if "Modality" in l])
    centroids = []
    
    print(f"\n[Centroids] Computing means over {len(X_tokens)} tokens...")
    for mod in unique_modalities:
        indices = np.where(y_tokens == mod)[0]
        centroid = np.mean(X_tokens[indices], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # --- 3. 推理统计 (Nearest Centroid) ---
    closest_indices, _ = pairwise_distances_argmin_min(X_fused, centroids)
    fused_counts = Counter(closest_indices)
    total_fused = len(X_fused)
    
    print("\n" + "="*40)
    print(" FUSED EMBEDDINGS ASSIGNMENT (Nearest Centroid) ")
    print("="*40)
    for i, mod_name in enumerate(unique_modalities):
        count = fused_counts.get(i, 0)
        pct = (count / total_fused) * 100
        print(f"  -> Closest to {mod_name}: {count} samples ({pct:.2f}%)")
    print("="*40 + "\n")

    # --- 4. PCA 降维 ---
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    
    # Fit on a subset to save memory/time if data is huge, or fit on all
    # 这里我们 fit 所有数据以获得准确的全局视图
    X_pca = pca.fit_transform(vectors)
    
    # 获取转换后的坐标
    coords_tokens = X_pca[~is_fused]
    coords_fused = X_pca[is_fused]
    coords_centroids = pca.transform(centroids)

    # --- 5. 绘图 (下采样 Token 以保持清晰度) ---
    plt.figure(figsize=(12, 10), dpi=300)
    palette = sns.color_palette("bright", len(unique_modalities) + 1)
    
    # A. 绘制 Modality Tokens (背景)
    # 如果点太多，随机采样一部分来画，否则图会画不动且看不清
    print(f"Plotting tokens (downsampling to max {max_scatter_points} points per modality for clarity)...")
    
    for i, mod_name in enumerate(unique_modalities):
        # 找到该 modality 的所有点
        idxs = np.where(y_tokens == mod_name)[0]
        current_coords = coords_tokens[idxs]
        
        # 下采样
        if len(current_coords) > max_scatter_points:
            choice = np.random.choice(len(current_coords), max_scatter_points, replace=False)
            current_coords = current_coords[choice]
            
        plt.scatter(
            current_coords[:, 0], current_coords[:, 1],
            c=[palette[i]], 
            label=mod_name,
            alpha=0.15, # 很高的透明度，形成"云"的效果
            s=10,
            edgecolors='none'
        )

    # B. 绘制 Fused Vectors
    plt.scatter(
        coords_fused[:, 0], coords_fused[:, 1],
        c='black',
        label='Fused (Ours)',
        alpha=0.6,
        s=20,
        marker='^' # 三角形表示 Fused
    )

    # C. 绘制 Centroids (大红叉/大标记)
    for i, mod_name in enumerate(unique_modalities):
        cx, cy = coords_centroids[i]
        plt.scatter(
            cx, cy, 
            c=[palette[i]], 
            s=400, 
            marker='X', 
            edgecolors='black', 
            linewidth=2,
            zorder=10 # 保证在最上层
        )
        plt.text(
            cx, cy + 0.5, f"{mod_name}\nCenter", 
            ha='center', fontsize=11, fontweight='bold', color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

    plt.title('PCA: Fused Vectors vs Modality Token Clouds', fontsize=16)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} Var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} Var)')
    
    # Legend deduplication handle
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc=2)
    
    plt.tight_layout()
    plt.savefig(save_plot_path)
    print(f"Plot saved to {save_plot_path}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 指定文件路径
    features_path = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run001+medkgat_fusion/umap_features.jsonl" 
    
    save_plot_path = features_path.replace('.jsonl', '_token_inference_plot.png')
    
    # 1. 加载 (支持新的 List of List 格式)
    X, y, sample_map = load_data(features_path)
    
    if len(X) > 0:
        input_dim = X.shape[1]
        
        # 2. 对齐 (1 Fused -> N Tokens)
        X_aligned = align_vectors_with_linear(X, sample_map, input_dim, train_steps=200)
        
        # 3. 可视化
        plot_pca_centroid_inference(X_aligned, y, save_plot_path)