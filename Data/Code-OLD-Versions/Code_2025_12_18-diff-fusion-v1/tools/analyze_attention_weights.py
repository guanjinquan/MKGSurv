import json
import argparse
import numpy as np
import os
import sys

def analyze_attention_scores(file_path):
    """
    读取保存的 attention scores jsonl 文件并计算每个 Group 的均值和方差。
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return

    data = []
    
    print(f"正在读取文件: {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    # 每一行应该是一个 list: [score_g0, score_g1, score_g2, ...]
                    sample_scores = json.loads(line)
                    data.append(sample_scores)
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_idx + 1} 行 JSON 解析失败，已跳过。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    if not data:
        print("错误: 未读取到有效数据。")
        return

    # 转换为 numpy 数组进行计算: (Samples, Groups)
    # 假设所有样本的 Group 数量是一致的
    try:
        matrix = np.array(data)
    except ValueError as e:
        print("错误: 数据格式不一致 (可能某些样本的 Group 数量不同):", e)
        return

    num_samples, num_groups = matrix.shape
    print(f"\n--- 数据统计 ---")
    print(f"样本总数: {num_samples}")
    print(f"Group 数量: {num_groups}")

    # 计算统计量 (沿着 axis 0，即跨样本计算)
    # mean: 每个 Group 在所有样本中的平均权重
    # var:  每个 Group 权重的波动程度
    # std:  标准差 (Standard Deviation)
    means = np.mean(matrix, axis=0)
    variances = np.var(matrix, axis=0)
    stds = np.std(matrix, axis=0)

    print(f"\n--- 分析结果 (Attention Score 统计) ---")
    print(f"{'Group ID':<10} | {'Mean (均值)':<15} | {'Variance (方差)':<15} | {'Std Dev (标准差)':<15}")
    print("-" * 65)

    for g_idx in range(num_groups):
        m = means[g_idx]
        v = variances[g_idx]
        s = stds[g_idx]
        print(f"{g_idx:<10} | {m:<15.6f} | {v:<15.6f} | {s:<15.6f}")

    print("-" * 65)
    
    # 找出关注度最高和最低的 Group
    max_idx = np.argmax(means)
    min_idx = np.argmin(means)
    
    print(f"\n总结:")
    print(f"平均关注度最高的组: Group {max_idx} (Mean: {means[max_idx]:.6f})")
    print(f"平均关注度最低的组: Group {min_idx} (Mean: {means[min_idx]:.6f})")
    print(f"波动最大(最不稳定)的组: Group {np.argmax(variances)} (Var: {np.max(variances):.6f})")

def main():
    parser = argparse.ArgumentParser(description="分析 Groups Attention Scores 的分布 (均值和方差)")
    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help='med_kgat_fusion.py 生成的 jsonl 文件路径 (例如: ./output/group_attention_score.jsonl)'
    )
    
    args = parser.parse_args()
    
    analyze_attention_scores(args.path)

if __name__ == "__main__":
    main()