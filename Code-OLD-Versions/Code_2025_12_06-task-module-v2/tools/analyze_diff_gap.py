import json
import torch
import numpy as np
from itertools import combinations, product # (*** 新增 product ***)
from tqdm import tqdm
import os 

def load_json(file_path):
    """加载 JSON 文件。"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误: 无法解析 JSON 文件 {file_path}")
        exit(1)

def calculate_risk_from_logits(logits):
    """
    根据您代码中的逻辑，从 N-N-Discrete 模型的 logits 计算风险评分。
    logits -> hazards -> survival_prob -> risk_score
    注意：这里的 risk_score 是 负的期望生存时间，所以分数越高，风险越高。
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
        
    hazards_tensor = torch.sigmoid(logits)
    # 计算生存函数 S (Survival function)
    # S[i] = (1-h[0]) * (1-h[1]) * ... * (1-h[i])
    S = torch.cumprod(1 - hazards_tensor, dim=0) # 假设 logits 是一维的
    
    # 风险评分 (Risk Score)
    # 您的代码段中使用了 S，所以我们用 -sum(S) 作为风险评分
    # 这样，风险评分越高，代表期望生存时间越短，即风险越高。
    risk_score = -torch.sum(S)
    
    return risk_score.item()

def check_concordance_status(label_a, label_b, risk_a, risk_b):
    """
    检查一个可评估对 (A, B) 是否一致。
    返回:
     1: 一致 (Concordant)
     0: 不一致 (Discordant)
    -1: 风险持平 (Tied Risk)
    """
    # 假设传入的已经是可评估对 (T_A < T_B, E_A = 1)
    # 真实情况：A 的风险 > B 的风险
    
    if risk_a > risk_b:
        return 1 # 一致: 模型预测 A 的风险 > B 的风险
    elif risk_a < risk_b:
        return 0 # 不一致: 模型预测 A 的风险 < B 的风险
    else:
        return -1 # 风险持平

def get_evaluable_pairs(pid_to_labels):
    """
    从标签数据中找出所有可评估的患者对。
    返回一个列表，每个元素是 (pid_a, pid_b)，其中 T_A < T_B 且 E_A = 1
    """
    pids = list(pid_to_labels.keys())
    evaluable_pairs = []
    
    # 遍历所有唯一的 PID 组合
    for pid_a, pid_b in combinations(pids, 2):
        label_a = pid_to_labels[pid_a] 
        label_b = pid_to_labels[pid_b]
        
        T_a, E_a = label_a['label_time'], label_a['label_event'] # 1=event, 0=censored
        T_b, E_b = label_b['label_time'], label_b['label_event']
        
        # C-Index 规则
        if (T_a < T_b) and (E_a == 1):
            evaluable_pairs.append((pid_a, pid_b, 'A_gt_B')) # A risk > B risk
        elif (T_b < T_a) and (E_b == 1):
            evaluable_pairs.append((pid_b, pid_a, 'B_gt_A')) # B risk > A risk
            
    return evaluable_pairs

if __name__ == "__main__":
    
    # --- 1. 定义文件路径 ---
    # (*** 更新: 定义单模态和多模态文件 ***)
    # files_to_analyze = {
    #     "A": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run005+msa/test_pid_to_data.json",
    #     "B": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run006+msa/test_pid_to_data.json",
    #     "C": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run007+msa/test_pid_to_data.json",
    #     "D": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run001_only_image+msa/test_pid_to_data.json",
    # }

    # multi_modal_file = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run002+msa/test_pid_to_data.json"
    # multi_modal_id = "Multi" # (*** 新增: 多模态模型的ID ***)


    files_to_analyze = {
        "A": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run001+msa/test_pid_to_data.json",
        "B": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run004+msa/test_pid_to_data.json",
        "C": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run006+msa/test_pid_to_data.json",
    }

    multi_modal_file = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run013+msa/test_pid_to_data.json"
    multi_modal_id = "Multi" # (*** 新增: 多模态模型的ID ***)


    # --- 2. 加载所有数据 ---
    print("正在加载单模态模型数据 (logits + labels)...")
    model_data = {} 
    for run_id, file_path in files_to_analyze.items():
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path} (模型 {run_id})，跳过此文件。")
            continue
        model_data[run_id] = load_json(file_path)
        print(f"  已加载 {run_id} ({len(model_data[run_id])} 个样本)")

    if not model_data:
        print("错误：没有成功加载任何单模态模型数据。退出。")
        exit(1)
        
    # (*** 新增: 加载多模态数据 ***)
    print(f"\n正在加载多模态模型数据 ({multi_modal_id})...")
    multi_modal_data = {}
    if not os.path.exists(multi_modal_file):
        print(f"错误: 找不到多模态文件 {multi_modal_file}。退出。")
        exit(1)
    multi_modal_data = load_json(multi_modal_file)
    print(f"  已加载 {multi_modal_id} ({len(multi_modal_data)} 个样本)")


    print(f"\n正在从 {list(model_data.keys())[0]} 的数据中提取标签...")
    pid_to_labels = {}
    first_model_id = list(model_data.keys())[0]
    for pid, data in model_data[first_model_id].items():
        pid_to_labels[pid] = data['label']
    print(f"  已提取 {len(pid_to_labels)} 个标签")

    model_ids = list(model_data.keys()) # (*** 现在是 ['A', 'B', 'C', 'D'] ***)
    num_models = len(model_ids)
    
    # --- 3. 找出所有可评估的配对 ---
    print("\n正在识别所有可评估的患者对...")
    evaluable_pairs = get_evaluable_pairs(pid_to_labels)
    total_evaluable = len(evaluable_pairs)
    if total_evaluable == 0:
        print("错误：未找到任何可评估的配对。请检查您的标签数据。")
        exit(1)
        
    print(f"  共找到 {total_evaluable} 个可评估的配对。")

    # --- 4. 分析每个配对在所有模型上的表现 ---
    print(f"正在分析所有配对在 {num_models} 个单模态 + 1 个多模态模型上的一致性...")
    
    # results_matrix 存储单模态结果
    results_matrix = np.zeros((total_evaluable, num_models), dtype=int)
    # multi_modal_results 存储多模态结果
    multi_modal_results = np.zeros(total_evaluable, dtype=int)
    
    # C-Index 追踪 (单模态)
    model_concordance_counts = {run_id: 0 for run_id in model_ids}
    model_valid_pairs = {run_id: 0 for run_id in model_ids}
    
    # C-Index 追踪 (多模态)
    multi_modal_concordance_count = 0
    multi_modal_valid_pairs = 0

    for i, (pid_a, pid_b, expected_order) in enumerate(tqdm(evaluable_pairs)):
        # pid_a 是生存时间更短且发生事件的患者
        # pid_b 是生存时间更长的患者
        # 期望：Risk(A) > Risk(B)
        
        # 4.1 分析单模态模型
        for j, run_id in enumerate(model_ids):
            if pid_a not in model_data[run_id] or pid_b not in model_data[run_id]:
                results_matrix[i, j] = -99 # 标记缺失数据
                continue 

            logits_a = model_data[run_id][pid_a]['logits']
            logits_b = model_data[run_id][pid_b]['logits']
            risk_a = calculate_risk_from_logits(logits_a)
            risk_b = calculate_risk_from_logits(logits_b)
            
            status = check_concordance_status(None, None, risk_a, risk_b)
            results_matrix[i, j] = status
            
            if status != -1: # 只要风险不持平
                model_valid_pairs[run_id] += 1
                if status == 1: # 如果一致
                    model_concordance_counts[run_id] += 1
                    
        # 4.2 (*** 新增: 分析多模态模型 ***)
        if pid_a not in multi_modal_data or pid_b not in multi_modal_data:
            multi_modal_results[i] = -99 # 标记缺失数据
        else:
            logits_a = multi_modal_data[pid_a]['logits']
            logits_b = multi_modal_data[pid_b]['logits']
            risk_a = calculate_risk_from_logits(logits_a)
            risk_b = calculate_risk_from_logits(logits_b)
            
            status = check_concordance_status(None, None, risk_a, risk_b)
            multi_modal_results[i] = status
            
            if status != -1:
                multi_modal_valid_pairs += 1
                if status == 1:
                    multi_modal_concordance_count += 1


    # --- 5. 打印 C-Index 验证结果 ---
    print("\n--- 模型 C-Index (验证) ---")
    for run_id in model_ids:
        if model_valid_pairs[run_id] > 0:
            c_index = model_concordance_counts[run_id] / model_valid_pairs[run_id]
            print(f"  [单] 模型 {run_id}: {c_index:.4f}  ({model_concordance_counts[run_id]} / {model_valid_pairs[run_id]})")
        else:
            print(f"  [单] 模型 {run_id}: N/A (没有有效的非持平配对)")
    
    # (*** 新增: 打印多模态 C-Index ***)
    if multi_modal_valid_pairs > 0:
        c_index = multi_modal_concordance_count / multi_modal_valid_pairs
        print(f"  [多] 模型 {multi_modal_id}: {c_index:.4f}  ({multi_modal_concordance_count} / {multi_modal_valid_pairs})")
    else:
        print(f"  [多] 模型 {multi_modal_id}: N/A (没有有效的非持平配对)")


    # --- 6. 寻找互补配对 (单模态 vs 单模态) ---
    print("\n--- 6. [单模态] 互补配对分析 (A 失败, B 成功) ---")
    for (idx_a, model_a), (idx_b, model_b) in combinations(enumerate(model_ids), 2):
        a_fails_b_succeeds = np.sum(
            (results_matrix[:, idx_a] == 0) & (results_matrix[:, idx_b] == 1)
        )
        b_fails_a_succeeds = np.sum(
            (results_matrix[:, idx_b] == 0) & (results_matrix[:, idx_a] == 1)
        )
        total_complementary = a_fails_b_succeeds + b_fails_a_succeeds
        
        print(f"\n  比较: {model_a} vs {model_b}")
        print(f"    {model_a} 失败, {model_b} 成功: {a_fails_b_succeeds} 个配对")
        print(f"    {model_b} 失败, {model_a} 成功: {b_fails_a_succeeds} 个配对")
        print(f"    总互补配对数: {total_complementary}")

    # --- 7. [单模态] 一致性模式分析 (Concordance Pattern Analysis) ---
    print("\n--- 7. [单模态] 一致性模式分析 (Concordance Pattern Analysis) ---")
    
    # 7.1. 筛选出 *单模态* 模型中都没有"风险持平"(-1)或缺失数据(-99)的配对
    valid_mask_step7 = np.all((results_matrix != -1) & (results_matrix != -99), axis=1)
    valid_results_step7 = results_matrix[valid_mask_step7]
    total_valid_for_analysis_step7 = len(valid_results_step7)
    
    if total_valid_for_analysis_step7 == 0:
        print("  错误: 找不到任何所有 *单模态* 模型均无'风险持平'或'缺失'的配对，无法计算模式。")
    else:
        print(f"  基于 {total_valid_for_analysis_step7} / {total_evaluable} 个有效配对进行分析。")

        # 7.2. 动态分析所有 2^N 种模式
        pattern_counts = {}
        for pattern in product([1, 0], repeat=num_models):
            count = np.sum(np.all(valid_results_step7 == pattern, axis=1))
            if count > 0: 
                pattern_counts[pattern] = count
        
        # 7.3. 格式化打印结果
        print("\n  --- 模式计数 (按数量排序) ---")
        sorted_patterns = sorted(pattern_counts.items(), key=lambda item: item[1], reverse=True)
        header_list = [f" {run_id:<5} " for run_id in model_ids]
        header = " | ".join(header_list)
        print(f"   {header} |  Count  |   (%)")
        print(f"  {'=' * (len(header) + 2)}|{'=' * 9}|{'=' * 9}")

        for pattern, count in sorted_patterns:
            pattern_str = " | ".join([f" {p:<5} " for p in pattern])
            percentage = (count / total_valid_for_analysis_step7) * 100
            print(f"   {pattern_str} |  {count:<7} |  ({percentage:5.1f}%)")

        # 7.4. 打印 "全部成功" 和 "全部失败" 的摘要
        all_success_pattern = tuple([1] * num_models)
        all_fail_pattern = tuple([0] * num_models)
        n_all_success = pattern_counts.get(all_success_pattern, 0)
        n_all_fail = pattern_counts.get(all_fail_pattern, 0)
        
        print("\n  --- 摘要 ---")
        print(f"  所有 {num_models} 个模型均正确 (All Success): {n_all_success} ({ (n_all_success/total_valid_for_analysis_step7)*100 :.1f}%)")
        print(f"  所有 {num_models} 个模型均错误 (All Fail):   {n_all_fail} ({ (n_all_fail/total_valid_for_analysis_step7)*100 :.1f}%)")

    # --- (*** 新增 ***) 8. 多模态 vs 单模态 融合分析 ---
    print(f"\n--- 8. 多模态 ({multi_modal_id}) vs 单模态 ({', '.join(model_ids)}) 融合分析 ---")
    
    # 8.1. 找到一个 *共同* 的有效集
    # 必须是所有 4 个单模态 和 1 个多模态 模型都给出了有效结果 (1 或 0) 的配对
    
    # 7.1 中已经计算了单模态的有效掩码: valid_mask_step7
    # 计算多模态的有效掩码
    multi_modal_valid_mask = (multi_modal_results != -1) & (multi_modal_results != -99)
    
    # 最终的共同掩码 (所有 5 个模型都有效)
    final_common_mask = valid_mask_step7 & multi_modal_valid_mask
    
    total_common_pairs = np.sum(final_common_mask)
    
    if total_common_pairs == 0:
        print(f"  错误: 找不到任何一个配对是所有 {num_models+1} 个模型都有效（非持平/非缺失）的。")
        print("  无法进行融合分析。")
    else:
        print(f"  基于 {total_common_pairs} / {total_evaluable} 个共同有效配对进行分析。")
        
        # 8.2. 提取共同有效集的结果
        common_single_modal_results = results_matrix[final_common_mask] # (N, 4) 数组
        common_multi_modal_results = multi_modal_results[final_common_mask]   # (N,) 数组
        
        # 8.3. 计算指标
        
        # S_multi: 多模态正确的配对集合 (布尔数组)
        s_multi_correct = (common_multi_modal_results == 1)
        num_multi_correct = np.sum(s_multi_correct)
        
        # S_union: 至少一个单模态正确的配对集合 (布尔数组)
        s_any_single_correct = np.any(common_single_modal_results == 1, axis=1)
        num_any_single_correct = np.sum(s_any_single_correct)
        
        # S_all_fail: 所有单模态都失败的配对集合 (布尔数组)
        s_all_single_fail = np.all(common_single_modal_results == 0, axis=1)
        num_all_single_fail = np.sum(s_all_single_fail)

        print(f"\n  --- 核心指标 ---")
        print(f"  A. 多模态模型正确配对数 (S_multi): {num_multi_correct}")
        print(f"  B. 至少一个单模态正确配对数 (S_union): {num_any_single_correct}")

        # (*** 指标 1: 回答你的问题 ***)
        if num_any_single_correct > 0:
            metric_1_ratio = num_multi_correct / num_any_single_correct
            print(f"\n  [指标 1] 融合比例 (A / B): {metric_1_ratio:.4f}")
            print(f"  > 解释: 多模态模型正确的配对数，占 '至少一个单模态能解决的' 配对总数的 {metric_1_ratio*100:.1f}%.")
        else:
            print("\n  [指标 1] 无法计算：没有一个单模态模型正确处理任何配对。")
            
        print(f"\n  --- 补充指标 (基于多模态正确的 {num_multi_correct} 个配对) ---")
        
        if num_multi_correct > 0:
            # (*** 指标 2: 独特贡献 ***)
            # 在多模态正确(S_multi)的配对中，有多少是所有单模态都失败(S_all_fail)的
            multi_correct_and_all_single_fail = s_multi_correct & s_all_single_fail
            num_multi_correct_unique = np.sum(multi_correct_and_all_single_fail)
            metric_2_ratio = num_multi_correct_unique / num_multi_correct

            print(f"  [指标 2] 独特贡献比例: {metric_2_ratio:.4f} ({num_multi_correct_unique} / {num_multi_correct})")
            print(f"  > 解释: 在多模态解决的配对中，有 {metric_2_ratio*100:.1f}% 是 *所有* 单模态模型都失败了的。")
            
            # (*** 指标 3: 重叠贡献 ***)
            # 在多模态正确(S_multi)的配对中，有多少是至少一个单模态也正确(S_any_single_correct)的
            multi_correct_and_any_single_correct = s_multi_correct & s_any_single_correct
            num_multi_correct_overlap = np.sum(multi_correct_and_any_single_correct)
            metric_3_ratio = num_multi_correct_overlap / num_multi_correct
            
            print(f"\n  [指标 3] 重叠贡献比例: {metric_3_ratio:.4f} ({num_multi_correct_overlap} / {num_multi_correct})")
            print(f"  > 解释: 在多模态解决的配对中，有 {metric_3_ratio*100:.1f}% 是 *至少一个* 单模态模型也能解决的。")
        else:
            print("  多模态模型没有正确处理任何配对，无法计算补充指标。")

    print("\n分析完成。")