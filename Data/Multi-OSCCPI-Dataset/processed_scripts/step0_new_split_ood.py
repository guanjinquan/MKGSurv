import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from tqdm import tqdm
from itertools import combinations
import warnings
import os
import json
import matplotlib.pyplot as plt
import random



# --- 1. 数据加载与特征处理 ---

def load_and_reduce_features_from_npy(
    npy_file_paths, 
    n_components=128
):
    """从.npy文件列表加载数据，进行聚合、扁平化和PCA降维。"""
    num_patients = len(npy_file_paths)
    print(f"将从 {num_patients} 个.npy文件中加载数据并进行处理...")

    high_dim_features = []
    for file_path in tqdm(npy_file_paths, desc="加载并聚合数据"):
        patient_data = np.load(file_path)
        mean_image = np.mean(patient_data, axis=0)
        high_dim_features.append(mean_image.flatten())
    
    high_dim_features = np.array(high_dim_features)
    print(f"数据扁平化完成，高维特征矩阵形状: {high_dim_features.shape}")

    print(f"正在使用PCA将特征从 {high_dim_features.shape[1]} 维降至 {n_components} 维...")
    pca = PCA(n_components=n_components, random_state=42)
    low_dim_features = pca.fit_transform(high_dim_features)
    print(f"PCA降维完成，解释的总方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return low_dim_features

# --- 2. 核心划分逻辑 ---

def _find_best_ood_cluster_split(
    patient_features, 
    patient_labels, 
    n_clusters,
    split_size
):
    """
    内部辅助函数：执行聚类并寻找最优簇组合以形成一个划分。
    返回两个列表的索引：(group1_indices, group2_indices)
    """
    num_patients = len(patient_labels)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    patient_cluster_ids = kmeans.fit_predict(patient_features)
    
    target_count = int(num_patients * split_size)
    overall_label_counts = np.bincount(patient_labels, minlength=2)
    target_ratio = overall_label_counts[0] / (overall_label_counts[1] + 1e-6)
    
    clusters_info = []
    for i in range(n_clusters):
        members_indices = np.where(patient_cluster_ids == i)[0]
        if len(members_indices) == 0: continue
        
        labels_in_cluster = np.array(patient_labels)[members_indices]
        label_counts = np.bincount(labels_in_cluster, minlength=2)
        
        clusters_info.append({
            'id': i,
            'indices': members_indices,
            'size': len(members_indices),
            'label_counts': label_counts
        })

    best_combination = None
    lowest_cost = float('inf')

    for i in range(1, n_clusters + 1):
        for combo in combinations(clusters_info, i):
            combo_size = sum(c['size'] for c in combo)
            if abs(combo_size - target_count) > (target_count * 0.5): continue

            combo_label_0 = sum(c['label_counts'][0] for c in combo)
            combo_label_1 = sum(c['label_counts'][1] for c in combo)
            current_ratio = combo_label_0 / (combo_label_1 + 1e-6)
            
            cost_size = ((combo_size - target_count) / target_count) ** 2
            cost_ratio = ((current_ratio - target_ratio) / (target_ratio + 1e-6)) ** 2
            total_cost = cost_size + 5 * cost_ratio

            if total_cost < lowest_cost:
                lowest_cost = total_cost
                best_combination = combo
    
    if best_combination is None:
        raise RuntimeError("未能找到合适的簇组合。请尝试调整 n_clusters 或 split_size。")

    group2_indices = np.concatenate([c['indices'] for c in best_combination]).tolist()
    all_patient_indices = set(range(num_patients))
    group1_indices = list(all_patient_indices - set(group2_indices))
    
    return group1_indices, group2_indices

def create_train_val_test_split(
    patient_features,
    patient_labels,
    n_clusters=12,
    test_size=0.2,
    val_size=0.1
):
    """主函数：将患者划分为训练集、验证集和测试集。"""
    print("\n--- 开始第一步：创建分布外(OOD)的测试集 ---")
    if n_clusters > 16:
        warnings.warn(f"簇数量({n_clusters})过高，可能会导致组合搜索非常耗时。建议 n_clusters <= 16。")

    train_val_indices, test_indices = _find_best_ood_cluster_split(
        patient_features,
        patient_labels,
        n_clusters=n_clusters,
        split_size=test_size
    )

    print("\n--- 开始第二步：从剩余数据中分层抽样出验证集 ---")
    val_split_ratio = val_size / (1.0 - test_size)
    train_val_labels = np.array(patient_labels)[train_val_indices]
    
    train_indices, val_indices, _, _ = train_test_split(
        train_val_indices,
        train_val_labels,
        test_size=val_split_ratio,
        random_state=42,
        stratify=train_val_labels
    )

    return sorted(train_indices), sorted(val_indices), sorted(test_indices)

# --- 3. 新增的可视化函数 ---

def visualize_split(
    features_2d, 
    train_indices, 
    val_indices, 
    test_indices, 
    save_path
):
    """将划分结果绘制成2D散点图并保存。"""
    print(f"\n正在生成2D可视化图并保存至: {save_path}")
    
    plt.figure(figsize=(12, 10))
    
    # 绘制训练集
    plt.scatter(
        features_2d[train_indices, 0], 
        features_2d[train_indices, 1], 
        c='blue', 
        label=f'Train ({len(train_indices)})', 
        alpha=0.6,
        s=15 # 点的大小
    )
    # 绘制验证集
    plt.scatter(
        features_2d[val_indices, 0], 
        features_2d[val_indices, 1], 
        c='orange', 
        label=f'Validation ({len(val_indices)})', 
        alpha=0.8,
        s=20
    )
    # 绘制测试集
    plt.scatter(
        features_2d[test_indices, 0], 
        features_2d[test_indices, 1], 
        c='red', 
        label=f'Test (OOD) ({len(test_indices)})', 
        alpha=0.9,
        s=25,
        marker='*' # 用星号标记测试集
    )
    
    plt.title('Patient Data Split Visualization (t-SNE)', fontsize=16)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # --- 固定随机种子以保证结果可复现 ---
    np.random.seed(42)
    random.seed(42)

    # --- 文件与路径设置 ---
    BASE_DIR = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset"
    NPY_ROOT_DIR = os.path.join(BASE_DIR, "Multi-OSCCPI-Npy-224")
    META_FILE = os.path.join(BASE_DIR, "all_metadata.json")
    SAVE_SPLIT_FILE = os.path.join(BASE_DIR, "split_OOD.json")
    SAVE_VIS_FILE = os.path.join(BASE_DIR, "split_OOD_visualization.png")
    
    # --- 加载元数据 ---
    with open(META_FILE, 'r') as f:
        meta_data = json.load(f)['datainfo']
    pids = [int(item['pid']) for item in meta_data]
    patient_labels = [int(item['REC']) for item in meta_data]
    
    # --- 准备NPY文件路径 ---
    all_npy_paths = []
    for pid in pids:
        npy_path = os.path.join(NPY_ROOT_DIR, f"{pid}.npy")
        assert os.path.exists(npy_path), f"文件不存在: {npy_path}"
        all_npy_paths.append(npy_path)

    # --- 步骤 1: 加载并降维 ---
    patient_features_reduced = load_and_reduce_features_from_npy(
        all_npy_paths, 
        n_components=128
    )

    # --- 步骤 2: 进行训练/验证/测试划分 ---
    train_pids_idx, val_pids_idx, test_pids_idx = create_train_val_test_split(
        patient_features_reduced,
        patient_labels,
        n_clusters=12,
        test_size=0.2,
        val_size=0.1
    )

    # --- 步骤 3: 新增 - 为可视化再次降维至2D ---
    print("\n--- 准备可视化：使用t-SNE降维至2D ---")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(patient_features_reduced)

    # --- 步骤 4: 调用可视化函数 ---
    visualize_split(
        features_2d,
        train_pids_idx,
        val_pids_idx,
        test_pids_idx,
        SAVE_VIS_FILE
    )

    # --- 步骤 5: 保存划分结果的PID ---
    train_pat_ids = [pids[i] for i in train_pids_idx]
    val_pat_ids = [pids[i] for i in val_pids_idx]
    test_pat_ids = [pids[i] for i in test_pids_idx]

    with open(SAVE_SPLIT_FILE, 'w') as f:
        json.dump({
            'train': train_pat_ids,
            'val': val_pat_ids,
            'test': test_pat_ids
        }, f, indent=4)
    print(f"\n划分结果已保存至: {SAVE_SPLIT_FILE}")

    # --- 最终输出统计信息 ---
    print("\n\n" + "="*50)
    print("--- 患者级别的 Train/Validation/Test 划分完成 ---")
    print(f"训练集患者数量: {len(train_pids_idx)}")
    print(f"验证集患者数量: {len(val_pids_idx)}")
    print(f"测试集患者数量: {len(test_pids_idx)}")
    print(f"总计: {len(train_pids_idx) + len(val_pids_idx) + len(test_pids_idx)}")

    y_train = np.array(patient_labels)[train_pids_idx]
    y_val = np.array(patient_labels)[val_pids_idx]
    y_test = np.array(patient_labels)[test_pids_idx]
    
    train_counts = np.bincount(y_train, minlength=2)
    val_counts = np.bincount(y_val, minlength=2)
    test_counts = np.bincount(y_test, minlength=2)
    
    def get_ratio(counts):
        return counts[0] / (counts[1] + 1e-6)

    print("\n--- 最终类别分布统计 ---")
    print(f"训练集: {train_counts[0]}/{train_counts[1]} (比例 ≈ {get_ratio(train_counts):.2f})")
    print(f"验证集: {val_counts[0]}/{val_counts[1]} (比例 ≈ {get_ratio(val_counts):.2f})")
    print(f"测试集: {test_counts[0]}/{test_counts[1]} (比例 ≈ {get_ratio(test_counts):.2f})")


   