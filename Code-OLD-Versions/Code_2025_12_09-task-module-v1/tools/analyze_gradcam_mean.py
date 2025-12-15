import json
import numpy as np

file = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run002+medkgat_fusion_only_msa_view_attn/gradcam.jsonl"
# file = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run001+medkgat_fusion/gradcam.jsonl"
# file = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run003+medkgat_fusion_healnet_group/gradcam.jsonl"

def calculate_column_statistics(file_path):
    # 用于存储所有行的数据
    all_data = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        # 解析JSON行
                        data = json.loads(line)
                        
                        # 根据您的示例，数据可能以不同形式存储
                        # 情况1: 如果数据本身就是列表
                        if isinstance(data, list):
                            all_data.append(data)
                        # 情况2: 如果数据是字典，包含列表
                        elif isinstance(data, dict):
                            # 尝试找到包含列表的值
                            for value in data.values():
                                if isinstance(value, list):
                                    all_data.append(value)
                                    break
                        else:
                            print(f"跳过无法处理的行: {line[:50]}...")
                            
                    except json.JSONDecodeError:
                        print(f"JSON解析错误，跳过行: {line[:50]}...")
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    if not all_data:
        print("未找到有效数据")
        return None
    
    # 检查所有行的长度是否一致
    first_len = len(all_data[0])
    for i, row in enumerate(all_data):
        if len(row) != first_len:
            print(f"警告: 第{i}行的长度({len(row)})与第一行({first_len})不一致")
    
    # 将数据转换为numpy数组以便计算
    try:
        data_array = np.array(all_data, dtype=float)
    except ValueError:
        print("数据中包含无法转换为浮点数的值")
        return None
    
    # 计算每一列的均值和方差
    column_means = np.mean(data_array, axis=0)
    column_variances = np.var(data_array, axis=0, ddof=1)  # 样本方差 (n-1)
    column_stds = np.std(data_array, axis=0, ddof=1)  # 样本标准差
    
    return column_means, column_variances, column_stds, len(all_data)

# 执行计算
result = calculate_column_statistics(file)

if result:
    column_means, column_variances, column_stds, num_patients = result
    
    # 输出结果
    print(f"总共有 {num_patients} 个患者")
    print(f"每个患者有 {len(column_means)} 个模态")
    print("\n每一列的统计信息:")
    print(f"{'模态':<6} {'均值':<12} {'方差':<12} {'标准差':<12}")
    print("-" * 45)
    for i, (mean, var, std) in enumerate(zip(column_means, column_variances, column_stds)):
        print(f"{i+1:<6} {mean:<12.6f} {var:<12.6f} {std:<12.6f}")
    
    # 可选：保存结果到文件
    # with open("column_statistics.txt", "w") as f:
    #     f.write(f"总共有 {num_patients} 个患者\n")
    #     f.write(f"每个患者有 {len(column_means)} 个模态\n")
    #     f.write("\n每一列的统计信息:\n")
    #     f.write(f"{'模态':<6} {'均值':<12} {'方差':<12} {'标准差':<12}\n")
    #     f.write("-" * 45 + "\n")
    #     for i, (mean, var, std) in enumerate(zip(column_means, column_variances, column_stds)):
    #         f.write(f"{i+1:<6} {mean:<12.6f} {var:<12.6f} {std:<12.6f}\n")
    
else:
    print("未能计算结果")