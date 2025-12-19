import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# --- 1. 定义文件路径 ---
patient_list_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-KIRC/source/kirc_patients.json"
labels_csv_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-KIRC/processed/kirc_patient_labels.csv"
output_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-KIRC/processed/kirc_patients_5fold.json"

RANDOM_SEED = 2026

# --- 2. 加载数据 ---
print("正在加载数据...")
try:
    with open(patient_list_path, 'rb') as f:
        original_patient_list = json.load(f)
    print(f"从 .pkl 加载了 {len(original_patient_list)} 个患者 ID。")
except FileNotFoundError:
    print(f"错误: 找不到患者列表文件 {patient_list_path}")
    exit()

try:
    labels_df = pd.read_csv(labels_csv_path)
    print(f"从 .csv 加载了 {labels_df.shape[0]} 行标签数据。")
except FileNotFoundError:
    print(f"错误: 找不到标签文件 {labels_csv_path}")
    exit()

# --- 3. 数据预处理与对齐 ---

# 3.1 清洗 ID 并确保列存在
try:
    labels_df['cases.submitter_id'] = labels_df['cases.submitter_id'].astype(str).str.strip()
    # 检查必要的列
    if 'DFS_time' not in labels_df.columns or 'DFS_event' not in labels_df.columns:
        raise KeyError("CSV 中缺少 'DFS_time' 或 'DFS_event' 列")
except KeyError as e:
    print(f"错误: {e}")
    exit()

# 3.2 创建查找字典: PID -> {'event': ..., 'time': ...}
# 注意：这里需要处理可能的空值或非数值类型的 DFS_time
# 先删除 DFS_time 为空的行，以免影响均值计算
labels_df = labels_df.dropna(subset=['DFS_time', 'DFS_event'])
# 确保 time 是数值型
labels_df['DFS_time'] = pd.to_numeric(labels_df['DFS_time'], errors='coerce')
labels_df = labels_df.dropna(subset=['DFS_time']) # 再次删除转换后可能出现的 NaN

# 将 DataFrame 转换为字典
info_map = labels_df.set_index('cases.submitter_id')[['DFS_event', 'DFS_time']].to_dict('index')

print(f"有效标签数据 (非空) 共有 {len(info_map)} 条。")

# 3.3 对齐 PIDs
aligned_patients = []
aligned_events = []
aligned_times = []

for pid in original_patient_list:
    pid_clean = str(pid).strip()
    if pid_clean in info_map:
        aligned_patients.append(pid_clean)
        aligned_events.append(info_map[pid_clean]['DFS_event'])
        aligned_times.append(info_map[pid_clean]['DFS_time'])

patients = np.array(aligned_patients)
events = np.array(aligned_events)
times = np.array(aligned_times)

print(f"最终用于划分的对齐患者数: {len(patients)}")
if len(patients) == 0:
    raise ValueError("错误：对齐后没有剩余患者，请检查 ID 格式。")

# --- 4. 生成高级分层标签 (Time + Event) ---

# 4.1 计算 DFS_time 均值
time_mean = np.median(times)  # np.mean(times)
print(f"\n--- 时间统计 ---")
print(f"DFS_time 均值: {np.mean(times):.4f}")
print(f"DFS_time 中位数: {np.median(times):.4f}")

# 4.2 生成时间类别 (Time Class): > Mean 为 1, <= Mean 为 0
# 使用 astype(int) 将布尔值转为 0/1
time_classes = (times > time_mean).astype(int)

# 4.3 组合 Event 和 Time Class 生成用于分层的 "Stratify Label"
# 逻辑：
# 0: Low Time (0), Event (0)
# 1: Low Time (0), Event (1)
# 2: High Time (1), Event (0)
# 3: High Time (1), Event (1)
# 公式: time_class * 2 + event
stratify_labels = time_classes * 2 + events

# 打印组合分布情况
unique_labels, counts = np.unique(stratify_labels, return_counts=True)
label_meaning = {
    0: "Time<=Mean, Event=0",
    1: "Time<=Mean, Event=1",
    2: "Time>Mean,  Event=0",
    3: "Time>Mean,  Event=1"
}

print("\n--- 组合分层标签分布 (用于 StratifiedKFold) ---")
for label, count in zip(unique_labels, counts):
    print(f"  Label {label} ({label_meaning.get(label, 'Unknown')}): {count} 例")

# --- 5. 执行 5 折分层划分 (7:1:2) ---

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
folds_data = []
fold_idx = 1

# 关键：这里使用 stratify_labels 进行划分
for train_val_idx, test_idx in skf.split(patients, stratify_labels):
    
    # 1. 划分 Test Set (20%)
    test_patients = patients[test_idx]
    
    # 2. 获取 Train+Val Set (80%) 及其对应的 stratify_labels
    train_val_patients = patients[train_val_idx]
    train_val_strat_labels = stratify_labels[train_val_idx]
    
    # 3. 内部再划分 Train (70%) 和 Val (10%)
    # test_size = 0.25 (1/4)，因为 80% * 1/4 = 20% (总体的20%)
    # 关键：stratify 参数使用 train_val_strat_labels，保证内部验证集也符合时间+事件分布
    try:
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=0.25,
            random_state=RANDOM_SEED,
            stratify=train_val_strat_labels 
        )
    except ValueError as e:
        print(f"警告 (Fold {fold_idx}): 某类样本太少，无法严格分层，回退到随机划分。")
        train_patients, val_patients = train_test_split(
            train_val_patients,
            test_size=0.25,
            random_state=RANDOM_SEED
        )

    # 4. 存储结果
    folds_data.append({
        "fold": fold_idx,
        "train": train_patients.tolist(),
        "valid": val_patients.tolist(),
        "test": test_patients.tolist()
    })
    
    print(f"Fold {fold_idx} 完成: Train({len(train_patients)}), Val({len(val_patients)}), Test({len(test_patients)})")
    fold_idx += 1

# --- 6. 保存结果 ---
result = {
    "split_ratio": "6:2:2",
    "time_mean_threshold": float(time_mean),
    "random_seed": RANDOM_SEED,
    "n_folds": 5,
    "total_patients": len(patients),
    "folds": folds_data
}

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("-" * 30)
print(f"基于 (Time, Event) 的 5折分层划分完成！")
print(f"阈值均值: {time_mean}")
print(f"结果已保存到: {output_path}")
