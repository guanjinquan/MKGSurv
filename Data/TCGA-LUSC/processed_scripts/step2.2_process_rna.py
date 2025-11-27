import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
import sys

# --- 1. 配置路径和参数 ---
# (请根据您的实际环境确认路径)
HALLMARK_FILE = Path("/home/Guanjq/NewWork/MedAlignFusion/Data/MMP_hallmarks_signatures.csv")
RNA_FILE = Path("/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUSC/source/HiSeqV2_PANCAN")
PATIENT_FILE = Path("/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUSC/processed/lusc_patient_labels.csv")

# 输出文件路径
OUTPUT_FILE = Path("/home/Guanjq/NewWork/MedAlignFusion/Data/TCGA-LUSC/processed/features_rna.pkl")

# 参数设定
TARGET_DIM = 512  # 目标维度 (Padding/Truncate 到此长度)
NUM_PATHWAYS = 50 # 预期 Pathway 数量

print("=== 启动 RNA 特征提取 (格式修正版) ===")
print(f"输入 Hallmark: {HALLMARK_FILE}")
print(f"输入 RNA Data: {RNA_FILE}")
print(f"目标输出: {OUTPUT_FILE}")

# --- 2. 基础数据加载 ---
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 2.1 加载患者 ID
try:
    patient_df = pd.read_csv(PATIENT_FILE)
    # 确保 ID 是字符串格式
    patient_list = patient_df['cases.submitter_id'].astype(str).unique().tolist()
    print(f"-> 已加载 {len(patient_list)} 位目标患者。")
except Exception as e:
    print(f"Error loading patients: {e}")
    sys.exit(1)

# 2.2 加载 Hallmark 定义
try:
    hallmark_df = pd.read_csv(HALLMARK_FILE)
    pathway_names = hallmark_df.columns.tolist()
    
    # 构建 {Pathway: [Genes]} 字典
    hallmark_map = {}
    for pname in pathway_names:
        genes = hallmark_df[pname].dropna().unique().tolist()
        if genes:
            hallmark_map[pname] = genes
            
    print(f"-> 已加载 {len(hallmark_map)} 个 Pathway 定义。")
except Exception as e:
    print(f"Error loading hallmarks: {e}")
    sys.exit(1)

# --- 3. RNA 数据加载与预处理 ---
try:
    print("-> 正在加载 RNA-Seq 原始数据 (请稍候)...")
    # 假设是制表符分隔
    rna_df = pd.read_csv(RNA_FILE, delimiter='\t')
    
    if 'sample' in rna_df.columns:
        rna_df = rna_df.set_index('sample')
    
    # 转置：行变成样本，列变成基因
    rna_df = rna_df.transpose()
    
    # 索引处理：截取前12位作为 Patient ID
    rna_df.index = rna_df.index.str.slice(0, 12)
    
    # 过滤：只保留目标患者
    rna_df = rna_df[rna_df.index.isin(patient_list)]
    
    # 去重：如果有重复 ID，取平均值
    if rna_df.index.duplicated().any():
        print("   注意: 发现重复患者 ID，正在合并取均值...")
        rna_df = rna_df.groupby(rna_df.index).mean()

    # 取交集并排序，确保数据对其
    common_patients = sorted(list(set(patient_list) & set(rna_df.index)))
    rna_df = rna_df.loc[common_patients]

    print(f"-> RNA 数据预处理完成。有效矩阵形状: {rna_df.shape} (Patients x Genes)")

except Exception as e:
    print(f"Error processing RNA data: {e}")
    sys.exit(1)

# --- 4. 检查 Pathway 长度 & 构建索引 ---
print("\n=== 正在分析 Pathway 基因覆盖情况 ===")

valid_pathway_configs = []
max_len_found = 0

for pname in pathway_names:
    if pname not in hallmark_map:
        continue
        
    original_genes = hallmark_map[pname]
    # 只取 RNA 数据中实际存在的基因
    valid_genes = [g for g in original_genes if g in rna_df.columns]
    curr_len = len(valid_genes)
    
    if curr_len > max_len_found:
        max_len_found = curr_len
        
    # *** 检查是否超过 512 ***
    if curr_len > TARGET_DIM:
        # print(f"警告: Pathway '{pname}' 超过限制，将被截断。")
        valid_genes = valid_genes[:TARGET_DIM]
    
    valid_pathway_configs.append({
        'name': pname,
        'cols': valid_genes
    })

print(f"-> 最大 Pathway 长度为: {max_len_found}")
print(f"-> 有效 Pathway 数量: {len(valid_pathway_configs)}")

# --- 5. 构建 Tensor 矩阵 (Padding 逻辑) ---
print(f"\n=== 开始构建 Tensor (Padding 0 to {TARGET_DIM}) ===")

# 1. 建立一个大矩阵: [N_Patients, N_Pathways, 512]
n_patients = len(rna_df)
n_pathways = len(valid_pathway_configs)

# 使用 float32 节省内存并符合 PyTorch 默认格式
final_matrix = np.zeros((n_patients, n_pathways, TARGET_DIM), dtype=np.float32)

# 2. 填充数据
for i, config in enumerate(valid_pathway_configs):
    cols = config['cols']
    if not cols:
        continue
        
    # 提取所有患者在该 Pathway 下的基因表达值
    # shape: (n_patients, n_valid_genes)
    gene_data = rna_df[cols].values
    
    current_gene_count = gene_data.shape[1]
    
    # 填入大矩阵: 对应 Pathway 通道，前 k 个位置
    final_matrix[:, i, :current_gene_count] = gene_data

print(f"-> 矩阵构建完成。Shape: {final_matrix.shape}")

# --- 6. 转换为 PyTorch 并保存为 PID -> Tensor 字典 ---
print("\n=== 正在转换格式并保存 (Pickle) ===")

# 转换为 PyTorch Tensor
tensor_data_all = torch.from_numpy(final_matrix) # Shape: (N, 50, 512)
final_patient_ids = rna_df.index.tolist()

# 构建目标字典: { 'PID': Tensor(50, 512) }
features_dict = {}

for idx, pid in enumerate(final_patient_ids):
    # 提取该患者对应的 Tensor
    # clone() 确保它是一块独立的内存（可选，但在循环中也无伤大雅）
    patient_tensor = tensor_data_all[idx].clone() 
    features_dict[str(pid)] = patient_tensor

try:
    # 保存字典
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(features_dict, f)
        
    print(f"成功保存到: {OUTPUT_FILE}")
    print(f"字典大小 (患者数): {len(features_dict)}")
    
    # --- 验证部分 ---
    print("\n--- 最终验证 ---")
    # 读取第一个 Key 进行检查
    first_pid = list(features_dict.keys())[0]
    first_tensor = features_dict[first_pid]
    
    print(f"Key 类型: {type(first_pid)} (示例: '{first_pid}')")
    print(f"Value 类型: {type(first_tensor)}")
    print(f"Value 形状: {first_tensor.shape} (预期: ({n_pathways}, {TARGET_DIM}))")
    
    if first_tensor.shape == (n_pathways, TARGET_DIM):
        print("✅ 格式验证通过！")
    else:
        print("❌ 格式验证失败，请检查维度设置。")

except Exception as e:
    print(f"保存失败: {e}")