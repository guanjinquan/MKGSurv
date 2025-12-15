import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

# 定义输入和输出路径
splits_dir = '/home/Zhengzx/DIMAF/src/data/data_files/tcga_kirc/splits'
output_path = '/home/Zhengzx/MedAlignFusion/Data/TCGA-KIRC/processed/kirc_patients_5fold.json'

# 收集所有患者ID和生存信息
all_patients = {}
clinical_data_path = '/home/Zhengzx/DIMAF/src/data/data_files/tcga_kirc/clinical_data_all.csv'
clinical_df = pd.read_csv(clinical_data_path)

# 提取患者信息：case_id, dss_survival_days, dss_censorship
clinical_df_selected = clinical_df[['case_id', 'dss_survival_days', 'dss_censorship']].copy()
clinical_df_selected = clinical_df_selected.dropna(subset=['dss_survival_days', 'dss_censorship'])

# 移除重复的case_id，只保留第一个
clinical_df_selected = clinical_df_selected.drop_duplicates(subset=['case_id'], keep='first')
clinical_df_selected.set_index('case_id', inplace=True)

# 初始化folds_data
folds_data = []

# 处理每个fold
for fold in range(5):
    fold_path = os.path.join(splits_dir, str(fold))
    train_csv = os.path.join(fold_path, 'train.csv')
    test_csv = os.path.join(fold_path, 'test.csv')
    
    # 读取当前fold的训练集和测试集
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # 获取当前fold的患者ID
    original_train_patients = train_df['case_id'].unique().tolist()
    test_patients = test_df['case_id'].unique().tolist()
    
    # 从训练集中按照3:1的比例划分出新的训练集和验证集
    # 获取训练集患者的生存信息用于分层
    train_patients_info = clinical_df_selected.loc[clinical_df_selected.index.isin(original_train_patients)]
    
    # 计算分层标签（基于生存时间和事件）
    time_mean = np.median(train_patients_info['dss_survival_days'])
    time_classes = (train_patients_info['dss_survival_days'] > time_mean).astype(int)
    stratify_labels = time_classes * 2 + train_patients_info['dss_censorship']
    
    try:
        # 使用分层抽样划分训练集和验证集 (3:1)
        new_train_patients, valid_patients = train_test_split(
            original_train_patients,
            test_size=0.25,  # 1/(3+1) = 0.25
            random_state=2026,
            stratify=stratify_labels
        )
    except ValueError:
        # 如果分层失败，则进行随机划分
        new_train_patients, valid_patients = train_test_split(
            original_train_patients,
            test_size=0.25,
            random_state=2026
        )
    
    # 添加到folds_data
    folds_data.append({
        "fold": fold + 1,  # 保持与原来一致的编号方式
        "train": new_train_patients,
        "valid": valid_patients,
        "test": test_patients
    })
    
    print(f"Fold {fold}:")
    print(f"  原始训练集患者数: {len(original_train_patients)}")
    print(f"  新训练集患者数: {len(new_train_patients)}")
    print(f"  验证集患者数: {len(valid_patients)}")
    print(f"  测试集患者数: {len(test_patients)}")

# 构建最终结果
result = {
    "split_ratio": "6:2:2 (Train:Valid:Test)",
    "strategy": "Based on existing DIMAF splits with further train/valid split (6:2) using stratification by (dss_survival_days > Median) AND dss_censorship",
    "random_seed": 2026,
    "n_folds": 5,
    "folds": folds_data
}

# 保存到JSON文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"\n新的五折交叉验证划分已完成并保存到: {output_path}")
print("划分策略:")
print("  1. 使用DIMAF现有的五折划分作为基础")
print("  2. 将每折的训练集按照3:1的比例重新划分为新的训练集和验证集")
print("  3. 使用分层抽样确保训练集和验证集在生存时间和事件分布上保持一致")