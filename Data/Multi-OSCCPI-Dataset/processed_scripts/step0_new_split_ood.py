import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.manifold import TSNE
from tqdm import tqdm
import warnings
import os
import json
import matplotlib.pyplot as plt
import h5py
import random

# --- 1. 数据加载与特征处理 (H5 Version) ---

def load_and_reduce_features_from_h5(
    pid_list, 
    h5_root_dir,
    n_components=128
):
    """
    从.h5文件列表加载数据。
    每个h5文件包含 (n, 1024) 的特征。
    计算均值得到 (1024,)，然后进行PCA降维。
    """
    print(f"准备从 {len(pid_list)} 个 .h5 文件中加载数据...")

    high_dim_features = []
    valid_pids = []
    missing_pids = []

    for pid in tqdm(pid_list, desc="加载H5数据"):
        file_path = os.path.join(h5_root_dir, f"{pid}.h5")
        
        if not os.path.exists(file_path):
            missing_pids.append(pid)
            continue
            
        try:
            with h5py.File(file_path, 'r') as f:
                # 自动查找包含数据的key (假设文件里有一个主要的数据集)
                data_key = None
                for key in f.keys():
                    # 简单判断：取第一个是Dataset类型的key
                    if isinstance(f[key], h5py.Dataset):
                        data_key = key
                        break
                
                if data_key is None:
                    print(f"Warning: {pid}.h5 中未找到数据集，跳过。")
                    continue

                patient_data = f[data_key][:] # Shape: (n, 1024)
                
                # 聚合：(n, 1024) -> (1024,)
                mean_feature = np.mean(patient_data, axis=0)
                high_dim_features.append(mean_feature)
                valid_pids.append(pid)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if len(missing_pids) > 0:
        print(f"警告: {len(missing_pids)} 个文件未找到 (例如 PID: {missing_pids[:3]}...)")

    high_dim_features = np.array(high_dim_features)
    print(f"数据加载完成，高维特征矩阵形状: {high_dim_features.shape}")

    # PCA 降维 (1024 -> n_components)
    # 这一步是为了让KMeans聚类更稳定，减少噪声影响
    print(f"正在使用PCA将特征从 {high_dim_features.shape[1]} 维降至 {n_components} 维...")
    pca = PCA(n_components=n_components, random_state=42)
    low_dim_features = pca.fit_transform(high_dim_features)
    print(f"PCA降维完成，解释的总方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return low_dim_features, valid_pids

# --- 2. 核心划分逻辑 (5-Fold OOD) ---

def create_5fold_ood_split(
    patient_features, 
    patient_labels, 
    patient_ids,
    n_clusters=20,  # 簇的数量要大于Fold数量，建议设大一点以便组合
    n_splits=5,
    val_size=0.2
):
    """
    使用基于聚类的 StratifiedGroupKFold 实现 5折交叉验证。
    逻辑：
    1. 先对所有病人聚类（例如聚成20类）。
    2. 使用 StratifiedGroupKFold，其中 'groups' 是聚类ID。
       这保证了同一个簇（代表一种特定的数据分布/OOD）完全在训练集或完全在测试集，
       绝不会被切分。
    3. 在每折的训练数据中，再随机切分出验证集。
    """
    
    print(f"\n--- 开始 OOD 5折交叉验证划分 ---")
    print(f"第一步：执行 KMeans 聚类 (n_clusters={n_clusters}) ...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_ids = kmeans.fit_predict(patient_features)
    
    # StratifiedGroupKFold: 
    # - Stratified: 尽量保持每折中 label (REC) 的比例一致
    # - Group: 保证同一个 Group (这里是 Cluster ID) 的数据不跨折泄漏
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    
    folds_result = {}
    
    # 需要将 list 转换为 np.array 以便索引
    y_arr = np.array(patient_labels)
    pids_arr = np.array(patient_ids)
    
    # sgkf.split 返回的是索引
    for fold_idx, (train_val_indices, test_indices) in enumerate(sgkf.split(patient_features, y_arr, groups=cluster_ids)):
        
        # 获取当前的训练+验证 ID 和 标签
        X_train_val = patient_features[train_val_indices]
        y_train_val = y_arr[train_val_indices]
        pids_train_val = pids_arr[train_val_indices]
        
        # 从 Train+Val 中根据 val_size 划分出 Validation Set
        # 这里使用普通的 StratifiedSplit，因为仅仅是为了监控训练过程
        train_indices_local, val_indices_local = train_test_split(
            np.arange(len(train_val_indices)),
            test_size=val_size,
            stratify=y_train_val,
            random_state=42 + fold_idx # 每一折变个种子增加随机性
        )
        
        # 映射回原始索引 (虽然我们在JSON里存PID，但为了逻辑严谨)
        global_train_indices = train_val_indices[train_indices_local]
        global_val_indices = train_val_indices[val_indices_local]
        
        # 获取PID列表
        fold_train_pids = pids_arr[global_train_indices].tolist()
        fold_val_pids = pids_arr[global_val_indices].tolist()
        fold_test_pids = pids_arr[test_indices].tolist()
        
        # 记录统计信息
        test_clusters = np.unique(cluster_ids[test_indices])
        print(f"\n[Fold {fold_idx}]")
        print(f"  Train: {len(fold_train_pids)} | Val: {len(fold_val_pids)} | Test: {len(fold_test_pids)}")
        print(f"  Test集包含的簇 ID: {test_clusters} (体现了OOD特性)")
        
        folds_result[f"fold_{fold_idx+1}"] = {
            "train": sorted(fold_train_pids),
            "valid": sorted(fold_val_pids),
            "test": sorted(fold_test_pids),
            "indices": {
                "train": global_train_indices,
                "valid": global_val_indices,
                "test": test_indices
            }
        }
        
    return folds_result, cluster_ids

# --- 3. 可视化函数 ---

def visualize_5fold(
    features_2d, 
    folds_result,
    save_dir
):
    """
    为每一个Fold生成一张可视化图。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\n正在生成每折的可视化图...")
    
    for fold_name, fold_data in folds_result.items():
        indices = fold_data['indices']
        train_idx = indices['train']
        val_idx = indices['valid']
        test_idx = indices['test']
        
        plt.figure(figsize=(10, 8))
        
        # 绘制所有点作为灰色背景，表现整体分布
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c='lightgray', alpha=0.2, s=10)

        # 绘制训练集
        plt.scatter(features_2d[train_idx, 0], features_2d[train_idx, 1], 
                    c='blue', label='Train', alpha=0.5, s=15)
        # 绘制验证集
        plt.scatter(features_2d[val_idx, 0], features_2d[val_idx, 1], 
                    c='orange', label='Valid', alpha=0.7, s=20)
        # 绘制测试集 (OOD)
        plt.scatter(features_2d[test_idx, 0], features_2d[test_idx, 1], 
                    c='red', label='Test (OOD)', alpha=0.9, s=30, marker='*')
        
        plt.title(f'5-Fold OOD Split - {fold_name}\n(Test set should be distinct clusters)', fontsize=14)
        plt.legend()
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"{fold_name}_vis.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  已保存: {save_path}")

# --- 主程序 ---

if __name__ == "__main__":
    # --- 配置 ---
    np.random.seed(42)
    random.seed(42)

    BASE_DIR = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset"
    # H5文件路径
    H5_ROOT_DIR = os.path.join(BASE_DIR, "backup/h5_files") 
    META_FILE = os.path.join(BASE_DIR, "all_metadata.json")
    
    # 输出文件路径
    SAVE_JSON_FILE = os.path.join(BASE_DIR, "split_OOD_5fold.json")
    VIS_SAVE_DIR = os.path.join(BASE_DIR, "vis_5fold_ood")

    # --- 1. 加载元数据 ---
    with open(META_FILE, 'r') as f:
        meta_data = json.load(f)['datainfo']
    
    # 原始PID和Label列表
    all_pids_raw = [int(item['pid']) for item in meta_data]
    all_labels_map = {int(item['pid']): int(item['REC']) for item in meta_data}

    # --- 2. 加载数据 (H5 -> Mean -> PCA) ---
    # 注意：load函数会过滤掉不存在文件的PID，所以返回valid_pids
    features_pca, valid_pids = load_and_reduce_features_from_h5(
        all_pids_raw, 
        H5_ROOT_DIR, 
        n_components=128
    )
    
    # 对齐 Label
    valid_labels = [all_labels_map[pid] for pid in valid_pids]
    
    # --- 3. 执行 5-Fold OOD 划分 ---
    # 使用 n_clusters=20，这样5折每折大约分到4个簇，保证多样性
    folds_data, cluster_ids = create_5fold_ood_split(
        features_pca, 
        valid_labels, 
        valid_pids,
        n_clusters=20, 
        n_splits=5,
        val_size=0.2
    )

    # --- 4. 生成 T-SNE 用于可视化 ---
    print("\n计算 t-SNE (2D) 用于可视化报告...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_pca)

    visualize_5fold(features_2d, folds_data, VIS_SAVE_DIR)

    # --- 5. 保存 JSON ---
    # 清理掉 numpy array 等非序列化对象，只保留 PID list
    json_output = {}
    for fold_name, data in folds_data.items():
        json_output[fold_name] = {
            "train": data['train'],
            "valid": data['valid'],
            "test": data['test']
        }
        
    with open(SAVE_JSON_FILE, 'w') as f:
        json.dump(json_output, f, indent=4)
        
    print("\n" + "="*50)
    print(f"完成！5折 OOD 划分文件已保存至: {SAVE_JSON_FILE}")
    print(f"可视化图片已保存至目录: {VIS_SAVE_DIR}")
    print("="*50)