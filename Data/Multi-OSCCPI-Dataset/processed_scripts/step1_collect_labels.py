import pandas as pd
import os
import json
import re
from collections import defaultdict

# --- Helper Function: 强力清洗字符串 ---
def clean_str(s):
    if not isinstance(s, str):
        return str(s)
    # 去除零宽空格、不间断空格、首尾空格
    return s.replace("\u200b", "").replace("\xa0", "").strip()

# --- 文件路径 (保持不变) ---
df_file = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/clinical_data.csv"
cn_df_file = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/ChineseClinicalData.xlsx"
pid_to_rid_file = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/pid_to_randomid.json"
class_file = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/surgery_classes.json"
split_file = "/home/Guanjq/NewWork/MedAlignFusion/Data/Multi-OSCCPI-Dataset/split_seed=2024.json"

with open(split_file, 'r') as f:
    split = json.load(f)

# --- 1. 加载手术分类 ---
with open(class_file, 'r') as f:
    class_map_list = json.load(f)["手术分类"]

all_cls = set()
surgery_to_class = {}

for dict_item in class_map_list:
    cls = clean_str(dict_item['类别'])
    all_cls.add(cls)
    item_list = dict_item['手术列表']
    for item in item_list:
        item = clean_str(item)
        if item:
            surgery_to_class[item] = cls

# 【修正点 1】: 添加 'Other' 分类并排序
all_cls.update(["Radiotherapy", "Chemotherapy", "Other"])
all_cls = sorted(list(all_cls))

# 【修正点 2】: 获取 'Other' 标签和对应的ID
other_label = "Other"
other_id = str(all_cls.index(other_label))

print("all_cls = ", all_cls)
print("all_cls len = ", len(all_cls))
print(f"Assigning 'other' label as ID: {other_id}")

# --- 2. 加载PID到RID的映射 ---
with open(pid_to_rid_file, 'r') as f:
    pid_to_rid = json.load(f)

# --- 3. 加载DataFrame ---
df = pd.read_csv(df_file)
cn_df = pd.read_excel(cn_df_file)

# --- 4. 从中文数据表 (cn_df) 构建 RID 到手术分类的映射 ---
pid_to_surgery = {}
for index, row in cn_df.iterrows():
    try:
        pid_raw = clean_str(str(row['病案号']))
        if not pid_raw: continue
        
        pid = int(float(pid_raw)) # Handle potentially float-like strings "123.0"
        
        if str(pid) not in pid_to_rid:
            continue

        rid = str(int(pid_to_rid[str(pid)]))
        surgery_str = str(row['手术方式'])
        
        if not surgery_str or surgery_str.lower() == 'nan':
            continue

        # 统一分隔符
        items = surgery_str.replace("＋","+").replace("、", "+").replace("，", "+").split("+")
        
        if rid not in pid_to_surgery:
            pid_to_surgery[rid] = set()
            
        for item in items:
            item = clean_str(item)
            if not item: continue
            
            if item in surgery_to_class:
                cls = surgery_to_class[item]
                # 【重要】断言：映射出的类别必须在总表中
                assert cls in all_cls, f"Class {cls} not found in all_cls list!"
                pid_to_surgery[rid].add(cls)

        if len(pid_to_surgery[rid]) == 0:
            print(f"[Info] PID={pid} has surgery string '{surgery_str}' but mapped to empty set.")
            
    except Exception as e:
        print(f"处理中文表行 {index} 出错: {e}")

# --- 5. 遍历主数据表 (df) 并添加新列 ---
cls_counter = defaultdict(int)

# 预先清理列类型
df['PID'] = df['PID'].astype(str).apply(lambda x: str(int(float(x))) if x.replace('.', '', 1).isdigit() else x)

for index, row in df.iterrows():
    rid = str(row["PID"]).strip()
    
    surgery_set = pid_to_surgery.get(rid, set())

    if len(surgery_set) == 0:
        # print(f"Rid = {rid} has no mapped surgeries.")
        pass
    
    # 获取辅助治疗信息
    is_radio = row.get("Radiotherapy(0no/1yes)", 0) == 1
    is_chemo = row.get("Chemotherapy(0no/1yes)", 0) == 1

    if is_radio:
        surgery_set.add("Radiotherapy")
    if is_chemo:
        surgery_set.add("Chemotherapy")

    # 【关键修正】: 严格排序，确保 "A+B" 和 "B+A" 是一样的
    sorted_surgeries = sorted(list(surgery_set))
    treatment_type_str = "+".join(sorted_surgeries)
    
    # 【关键修正】: 计算 ID 并严格按照 ID 数值排序，确保 ID 序列唯一
    # 比如 12 和 2，如果不按数值排序，字符串排序可能导致 12 在 2 前面或者后面不一致
    id_list = []
    for x in sorted_surgeries:
        if x in all_cls:
            id_list.append(all_cls.index(x))
        else:
            print(f"[Error] Class '{x}' for RID {rid} not in all_cls!")
    
    sorted_id_list = sorted(id_list) # 数值排序
    treatment_type_id_str = ",".join([str(i) for i in sorted_id_list])

    # 过滤 split
    if treatment_type_str:
        for key, values_list in split.items():
            if int(rid) in values_list:
                values_list.remove(int(rid))
                break 

    # 写入
    df.at[index, "12_treatment_type"] = treatment_type_str
    df.at[index, "12_treatment_type_id"] = treatment_type_id_str
    
    if treatment_type_id_str:
        cls_counter[treatment_type_id_str] += 1

# --- 5.5 添加缺失的行 ---
print("="*30)
print("开始处理在 split.json 中但不在 .csv 中的缺失RID...")

all_missing_rids = []
for split_name, rid_list in split.items():
    if rid_list:
        print(f"在 {split_name} 中找到 {len(rid_list)} 个缺失的RID。")
        all_missing_rids.extend(rid_list)

print(f"总共找到 {len(all_missing_rids)} 个缺失的RID。")

new_rows_list = []
if all_missing_rids:
    for rid in all_missing_rids:
        new_row = {
            "PID": str(rid),
            "12_treatment_type": other_label,
            "12_treatment_type_id": other_id
            # 注意：其他列会默认为 NaN
        }
        new_rows_list.append(new_row)
    
    new_df = pd.DataFrame(new_rows_list)
    df = pd.concat([df, new_df], ignore_index=True)
    
    cls_counter[other_id] += len(all_missing_rids)
    print(f"已成功添加 {len(new_rows_list)} 行，标签为 '{other_label}' (ID: {other_id})。")
else:
    print("没有在 split 文件中找到缺失的RID，无需添加新行。")

print("="*30)

# --- 打印和保存 ---
print("最终类别计数 (Top 10):")
for key, count in sorted(cls_counter.items(), key=lambda item: item[1], reverse=True)[:10]:
    print(f"ID(s) '{key}': {count} 个")

df.to_csv(df_file, index=False)
print("处理完成，文件已保存。")