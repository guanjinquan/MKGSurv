import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# --- 1. 定义文件路径 ---
patient_list_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/source/luad_patients.pkl"
labels_csv_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/luad_patient_labels.csv"
output_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/luad_patients_5fold.json"

RANDOM_SEED = 2026

# --- 2. 加载数据 ---
print("正在加载数据...")
try:
    with open(patient_list_path, 'rb') as f:
        original_patient_list = pickle.load(f)
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


# import pickle
# import json
# import numpy as np
# import pandas as pd  # 用于读取 CSV
# from sklearn.model_selection import StratifiedKFold, train_test_split

# # --- 1. 定义文件路径 ---
# # 包含患者ID列表的 Pickle 文件 (假设是 submitter_id 列表)
# patient_list_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/source/luad_patients.pkl"

# # 包含标签 (event) 的 CSV 文件
# labels_csv_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/luad_patient_labels.csv"

# # 最终输出的 JSON 文件
# output_path = "/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUAD/processed/luad_patients_5fold.json"

# RANDOM_SEED = 0

# # --- 2. 加载数据 ---
# # 加载患者ID列表 (来自 .pkl)
# try:
#     with open(patient_list_path, 'rb') as f:
#         original_patient_list = pickle.load(f)
# except FileNotFoundError:
#     print(f"错误: 找不到患者列表文件 {patient_list_path}")
#     exit()
# print(f"从 {patient_list_path} 加载了 {len(original_patient_list)} 个患者 ID。")

# # 加载标签数据 (来自 .csv)
# try:
#     labels_df = pd.read_csv(labels_csv_path)
# except FileNotFoundError:
#     print(f"错误: 找不到标签文件 {labels_csv_path}")
#     exit()
# print(f"从 {labels_csv_path} 加载了 {labels_df.shape[0]} 行标签数据。")

# # --- 3. 创建 PID -> Event 查找字典 ---
# # 使用您指定的列: 'cases.submitter_id' 和 'DFS_event'
# # .strip() 用于清除 'cases.submitter_id' 中可能存在的多余空格
# try:
#     labels_df['cases.submitter_id'] = labels_df['cases.submitter_id'].str.strip()
#     # 将 DataFrame 转换为字典以便快速查找
#     event_map = pd.Series(
#         labels_df.DFS_event.values, 
#         index=labels_df['cases.submitter_id']
#     ).to_dict()
# except KeyError:
#     print("错误: CSV 文件中未找到 'cases.submitter_id' 或 'DFS_event' 列。")
#     print(f"  CSV 中的列为: {labels_df.columns.tolist()}")
#     exit()
    
# print(f"创建了 {len(event_map)} 条 PID-Event 映射。")
# print("-" * 30)

# # --- 4. 对齐 PIDs 和 Events ---
# # 我们将只保留那些同时存在于 .pkl 列表和 .csv 标签文件中的患者
# aligned_patients = []
# aligned_events = []
# patients_not_found = 0

# for pid in original_patient_list:
#     pid_clean = str(pid).strip()  # 确保 pid 是干净的字符串
#     if pid_clean in event_map:
#         aligned_patients.append(pid_clean)
#         aligned_events.append(event_map[pid_clean])
#     else:
#         patients_not_found += 1

# if patients_not_found > 0:
#     print(f"警告：{patients_not_found} 个来自 .pkl 的患者在 .csv 标签中未找到，已被忽略。")

# print(f"总共 {len(aligned_patients)} 个患者将用于划分。")

# # 转换为 NumPy array 以便用于 sklearn
# patients = np.array(aligned_patients)
# events = np.array(aligned_events)

# if len(patients) == 0:
#     raise ValueError("没有可用的患者数据！请检查 .pkl 和 .csv 文件中的 ID 是否匹配。")

# print(f"最终 Event 分布: 0 (count={np.sum(events == 0)}), 1 (count={np.sum(events == 1)})")
# print("-" * 30)


# # --- 5. 创建 5 折分层划分 (7:1:2) ---
# # n_splits=5 将数据分为 80% (train+val) 和 20% (test)
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# folds_data = []
# fold_idx = 1

# for train_val_idx, test_idx in skf.split(patients, events):
    
#     # 1. 划分出 20% 的测试集 (Test)
#     test_patients = patients[test_idx]
#     test_events = events[test_idx]  # 用于验证分层
    
#     # 2. 剩余 80% 作为 训练+验证 (Train+Val) 集
#     train_val_patients = patients[train_val_idx]
#     train_val_events = events[train_val_idx]
    
#     # 3. 将 80% 的 (Train+Val) 集再次分层划分为 7:1
#     #    test_size = 10% / 80% = 1/8 = 0.125
    
#     # 检查 train_val_events 是否有足够的数据和类别来进行分层
#     if len(np.unique(train_val_events)) < 2 or len(train_val_patients) < 2:
#         print(f"警告: Fold {fold_idx} 的 train_val 集太小或类别单一，无法进行分层。")
#         # 无法分层时的备用方案（例如，如果所有 events 都是 0 或 1）
#         train_patients, val_patients, train_events, val_events = train_test_split(
#             train_val_patients,
#             train_val_events,
#             test_size=0.125,
#             random_state=RANDOM_SEED,
#             stratify=None  # 降级为非分层
#         )
#     else:
#         # 标准的分层划分
#         train_patients, val_patients, train_events, val_events = train_test_split(
#             train_val_patients,
#             train_val_events,
#             test_size=0.125, 
#             random_state=RANDOM_SEED,
#             stratify=train_val_events # 关键：在 80% 的数据内部再次分层
#         )
    
#     # 4. 存储这一折的结果
#     folds_data.append({
#         "fold": fold_idx,
#         "train": train_patients.tolist(),
#         "valid": val_patients.tolist(),
#         "test": test_patients.tolist()
#     })
    
#     # 5. (可选) 打印每折的统计信息
#     print(f"--- Fold {fold_idx} ---")
#     print(f"  Train: {len(train_patients)} (Event 1: {np.sum(train_events)} / {len(train_patients)})")
#     print(f"  Valid  : {len(val_patients)} (Event 1: {np.sum(val_events)} / {len(val_patients)})")
#     print(f"  Test : {len(test_patients)} (Event 1: {np.sum(test_events)} / {len(test_patients)})")
    
#     fold_idx += 1

# # --- 6. 准备并保存最终的 JSON 结果 ---
# result = {
#     "split_ratio": "7:1:2 (Train:Val:Test)",
#     "random_seed": RANDOM_SEED,
#     "n_folds": 5,
#     "total_patients_in_pkl": len(original_patient_list),
#     "total_patients_processed": len(aligned_patients),
#     "folds": folds_data
# }

# with open(output_path, 'w', encoding='utf-8') as f:
#     json.dump(result, f, ensure_ascii=False, indent=4)

# print("-" * 30)
# print(f"5折 (7:1:2) 分层划分完成！")
# print(f"结果已保存到: {output_path}")