import json
import torch
import numpy as np
from itertools import combinations, product
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
    files_to_analyze = {
        "A": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run005+msa/test_pid_to_data.json",
        "B": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run006+msa/test_pid_to_data.json",
        "C": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run007+msa/test_pid_to_data.json",
        "D": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run001_only_image+msa/test_pid_to_data.json",
    }
    multi_modal_file = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse/inhouse_run002+msa/test_pid_to_data.json"
    
    # files_to_analyze = {
    #     "A": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run001+msa/test_pid_to_data.json",
    #     "B": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run004+msa/test_pid_to_data.json",
    #     "C": "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run006+msa/test_pid_to_data.json",
    # }
    # multi_modal_file = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/tcga_luad/tcga_luad_run013+msa/test_pid_to_data.json"
    
    multi_modal_id = "Multi" 
    
    # (*** 新增: 定义输出文件 ***)
    output_conflict_file = "patient_risk_conflict_analysis.txt"


    # --- 2. 加载所有数据 ---
    print("正在加载单模态模型数据 (logits + labels)...")
    model_data = {} 
    for run_id, file_path in files_to_analyze.items():
        if not os.path.exists(file_path):
            print(f"警告: 找不到文件 {file_path} (模型 {run_id})，跳过此文件。")
            continue
        model_data[run_id] = load_json(file_path)
        print(f"   已加载 {run_id} ({len(model_data[run_id])} 个样本)")

    if not model_data:
        print("错误：没有成功加载任何单模态模型数据。退出。")
        exit(1)
        
    print(f"\n正在加载多模态模型数据 ({multi_modal_id})...")
    multi_modal_data = {}
    if not os.path.exists(multi_modal_file):
        print(f"错误: 找不到多模态文件 {multi_modal_file}。退出。")
        exit(1)
    multi_modal_data = load_json(multi_modal_file)
    print(f"   已加载 {multi_modal_id} ({len(multi_modal_data)} 个样本)")


    print(f"\n正在从 {list(model_data.keys())[0]} 的数据中提取标签...")
    pid_to_labels = {}
    first_model_id = list(model_data.keys())[0]
    all_pids = list(model_data[first_model_id].keys()) 
    for pid, data in model_data[first_model_id].items():
        pid_to_labels[pid] = data['label']
    print(f"   已提取 {len(pid_to_labels)} 个标签 (共 {len(all_pids)} 个 PIDs)")

    model_ids = list(model_data.keys()) 
    num_models = len(model_ids)
    
    # --- 2.5 预先计算所有风险评分 ---
    print("\n--- 2.5 正在预先计算所有患者的风险评分 ---")
    
    all_model_data_for_risks = model_data.copy()
    all_model_data_for_risks[multi_modal_id] = multi_modal_data
    
    pid_to_risks = {pid: {} for pid in all_pids} # 格式: {pid: {model_id: risk}}
    
    pbar = tqdm(total=len(all_pids) * len(all_model_data_for_risks), desc="计算风险")
    for model_id, data in all_model_data_for_risks.items():
        for pid in all_pids:
            if pid in data:
                logits = data[pid]['logits']
                risk = calculate_risk_from_logits(logits)
                pid_to_risks[pid][model_id] = risk
            else:
                pid_to_risks[pid][model_id] = np.nan # 标记缺失数据
            pbar.update(1)
    pbar.close()
    print("   所有风险评分计算完毕。")

    
    # --- 3. 找出所有可评估的配对 ---
    print("\n--- 3. 正在识别所有可评估的患者对 ---")
    evaluable_pairs = get_evaluable_pairs(pid_to_labels)
    total_evaluable = len(evaluable_pairs)
    if total_evaluable == 0:
        print("错误：未找到任何可评估的配对。请检查您的标签数据。")
        exit(1)
        
    print(f"   共找到 {total_evaluable} 个可评估的配对。")

    # --- 4. 分析每个配对在所有模型上的表现 ---
    print(f"\n--- 4. 正在分析所有配对在 {num_models} 个单模态 + 1 个多模态模型上的一致性 ---")
    
    results_matrix = np.zeros((total_evaluable, num_models), dtype=int)
    multi_modal_results = np.zeros(total_evaluable, dtype=int)
    
    model_concordance_counts = {run_id: 0 for run_id in model_ids}
    model_valid_pairs = {run_id: 0 for run_id in model_ids}
    
    multi_modal_concordance_count = 0
    multi_modal_valid_pairs = 0

    for i, (pid_a, pid_b, expected_order) in enumerate(tqdm(evaluable_pairs, desc="分析配对")):
        
        # 4.1 分析单模态模型
        for j, run_id in enumerate(model_ids):
            risk_a = pid_to_risks[pid_a][run_id]
            risk_b = pid_to_risks[pid_b][run_id]
            
            if np.isnan(risk_a) or np.isnan(risk_b):
                results_matrix[i, j] = -99 # 标记缺失数据
                continue 

            status = check_concordance_status(None, None, risk_a, risk_b)
            results_matrix[i, j] = status
            
            if status != -1: 
                model_valid_pairs[run_id] += 1
                if status == 1: 
                    model_concordance_counts[run_id] += 1
                        
        # 4.2 分析多模态模型
        risk_a = pid_to_risks[pid_a][multi_modal_id]
        risk_b = pid_to_risks[pid_b][multi_modal_id]

        if np.isnan(risk_a) or np.isnan(risk_b):
            multi_modal_results[i] = -99 # 标记缺失数据
        else:
            status = check_concordance_status(None, None, risk_a, risk_b)
            multi_modal_results[i] = status
            
            if status != -1:
                multi_modal_valid_pairs += 1
                if status == 1:
                    multi_modal_concordance_count += 1


    # --- 5. 打印 C-Index 验证结果 ---
    print("\n--- 5. 模型 C-Index (验证) ---")
    for run_id in model_ids:
        if model_valid_pairs[run_id] > 0:
            c_index = model_concordance_counts[run_id] / model_valid_pairs[run_id]
            print(f"   [单] 模型 {run_id}: {c_index:.4f}  ({model_concordance_counts[run_id]} / {model_valid_pairs[run_id]})")
        else:
            print(f"   [单] 模型 {run_id}: N/A (没有有效的非持平配对)")
    
    if multi_modal_valid_pairs > 0:
        c_index = multi_modal_concordance_count / multi_modal_valid_pairs
        print(f"   [多] 模型 {multi_modal_id}: {c_index:.4f}  ({multi_modal_concordance_count} / {multi_modal_valid_pairs})")
    else:
        print(f"   [多] 模型 {multi_modal_id}: N/A (没有有效的非持平配对)")


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
        
        print(f"\n   比较: {model_a} vs {model_b}")
        print(f"     {model_a} 失败, {model_b} 成功: {a_fails_b_succeeds} 个配对")
        print(f"     {model_b} 失败, {model_a} 成功: {b_fails_a_succeeds} 个配对")
        print(f"     总互补配对数: {total_complementary}")

    # --- 7. [单模态] 一致性模式分析 (Concordance Pattern Analysis) ---
    print("\n--- 7. [单模态] 一致性模式分析 (Concordance Pattern Analysis) ---")
    
    valid_mask_step7 = np.all((results_matrix != -1) & (results_matrix != -99), axis=1)
    valid_results_step7 = results_matrix[valid_mask_step7]
    total_valid_for_analysis_step7 = len(valid_results_step7)
    
    if total_valid_for_analysis_step7 == 0:
        print("   错误: 找不到任何所有 *单模态* 模型均无'风险持平'或'缺失'的配对，无法计算模式。")
    else:
        print(f"   基于 {total_valid_for_analysis_step7} / {total_evaluable} 个有效配对进行分析。")

        pattern_counts = {}
        for pattern in product([1, 0], repeat=num_models):
            count = np.sum(np.all(valid_results_step7 == pattern, axis=1))
            if count > 0: 
                pattern_counts[pattern] = count
        
        print("\n   --- 模式计数 (按数量排序) ---")
        sorted_patterns = sorted(pattern_counts.items(), key=lambda item: item[1], reverse=True)
        header_list = [f" {run_id:<5} " for run_id in model_ids]
        header = " | ".join(header_list)
        print(f"    {header} |  Count  |  (%)")
        print(f"   {'=' * (len(header) + 2)}|{'=' * 9}|{'=' * 9}")

        for pattern, count in sorted_patterns:
            pattern_str = " | ".join([f" {p:<5} " for p in pattern])
            percentage = (count / total_valid_for_analysis_step7) * 100
            print(f"    {pattern_str} |  {count:<7} |  ({percentage:5.1f}%)")

        all_success_pattern = tuple([1] * num_models)
        all_fail_pattern = tuple([0] * num_models)
        n_all_success = pattern_counts.get(all_success_pattern, 0)
        n_all_fail = pattern_counts.get(all_fail_pattern, 0)
        
        print("\n   --- 摘要 ---")
        print(f"   所有 {num_models} 个模型均正确 (All Success): {n_all_success} ({ (n_all_success/total_valid_for_analysis_step7)*100 :.1f}%)")
        print(f"   所有 {num_models} 个模型均错误 (All Fail):   {n_all_fail} ({ (n_all_fail/total_valid_for_analysis_step7)*100 :.1f}%)")

    # --- 8. 多模态 vs 单模态 融合分析 ---
    print(f"\n--- 8. 多模态 ({multi_modal_id}) vs 单模态 ({', '.join(model_ids)}) 融合分析 ---")
    
    multi_modal_valid_mask = (multi_modal_results != -1) & (multi_modal_results != -99)
    final_common_mask = valid_mask_step7 & multi_modal_valid_mask
    total_common_pairs = np.sum(final_common_mask)
    
    if total_common_pairs == 0:
        print(f"   错误: 找不到任何一个配对是所有 {num_models+1} 个模型都有效（非持平/非缺失）的。")
        print("   无法进行融合分析。")
    else:
        print(f"   基于 {total_common_pairs} / {total_evaluable} 个共同有效配对进行分析。")
        
        common_single_modal_results = results_matrix[final_common_mask]
        common_multi_modal_results = multi_modal_results[final_common_mask]  
        
        s_multi_correct = (common_multi_modal_results == 1)
        num_multi_correct = np.sum(s_multi_correct)
        
        s_any_single_correct = np.any(common_single_modal_results == 1, axis=1)
        num_any_single_correct = np.sum(s_any_single_correct)
        
        s_all_single_fail = np.all(common_single_modal_results == 0, axis=1)
        
        print(f"\n   --- 核心指标 ---")
        print(f"   A. 多模态模型正确配对数 (S_multi): {num_multi_correct}")
        print(f"   B. 至少一个单模态正确配对数 (S_union): {num_any_single_correct}")

        if num_any_single_correct > 0:
            metric_1_ratio = num_multi_correct / num_any_single_correct
            print(f"\n   [指标 1] 融合比例 (A / B): {metric_1_ratio:.4f}")
            print(f"   > 解释: 多模态模型正确的配对数，占 '至少一个单模态能解决的' 配对总数的 {metric_1_ratio*100:.1f}%.")
        else:
            print("\n   [指标 1] 无法计算：没有一个单模态模型正确处理任何配对。")
            
        print(f"\n   --- 补充指标 (基于多模态正确的 {num_multi_correct} 个配对) ---")
        
        if num_multi_correct > 0:
            multi_correct_and_all_single_fail = s_multi_correct & s_all_single_fail
            num_multi_correct_unique = np.sum(multi_correct_and_all_single_fail)
            metric_2_ratio = num_multi_correct_unique / num_multi_correct

            print(f"   [指标 2] 独特贡献比例: {metric_2_ratio:.4f} ({num_multi_correct_unique} / {num_multi_correct})")
            print(f"   > 解释: 在多模态解决的配对中，有 {metric_2_ratio*100:.1f}% 是 *所有* 单模态模型都失败了的。")
            
            multi_correct_and_any_single_correct = s_multi_correct & s_any_single_correct
            num_multi_correct_overlap = np.sum(multi_correct_and_any_single_correct)
            metric_3_ratio = num_multi_correct_overlap / num_multi_correct
            
            print(f"\n   [指标 3] 重叠贡献比例: {metric_3_ratio:.4f} ({num_multi_correct_overlap} / {num_multi_correct})")
            print(f"   > 解释: 在多模态解决的配对中，有 {metric_3_ratio*100:.1f}% 是 *至少一个* 单模态模型也能解决的。")
        else:
            print("   多模态模型没有正确处理任何配对，无法计算补充指标。")

    # --- 9. [单模态] 风险分布与冲突分析 ---
    print(f"\n--- 9. [单模态] 风险分布与冲突分析 ({', '.join(model_ids)}) ---")
    
    all_unimodal_risks = []
    for pid in all_pids:
        for model_id in model_ids:
            risk = pid_to_risks[pid].get(model_id, np.nan)
            if not np.isnan(risk):
                all_unimodal_risks.append(risk)
    
    if not all_unimodal_risks:
        print("   错误: 未能从单模态模型中收集到任何有效的风险评分。")
        # (*** 修改: 如果失败，则跳过后续步骤 ***)
        pid_conflict_list = [] # 确保列表为空
        all_pids_with_conflict = [] # 确保列表为空
    else:
        global_min_risk = np.min(all_unimodal_risks)
        global_max_risk = np.max(all_unimodal_risks)
        global_range = global_max_risk - global_min_risk
        
        print(f"   全局单模态风险分布: Min={global_min_risk:.4f}, Max={global_max_risk:.4f}, Range={global_range:.4f}")
        
        patient_conflict_stds = []
        pid_conflict_list = [] # 存储 (pid, conflict_std)
        all_pids_with_conflict = [] # (*** 新增: 存储所有计算了冲突的pid ***)

        for pid in all_pids:
            normalized_risks = []
            for model_id in model_ids:
                risk = pid_to_risks[pid].get(model_id, np.nan)
                if not np.isnan(risk):
                    if global_range > 1e-6: 
                        norm_risk = (risk - global_min_risk) / global_range
                    else:
                        norm_risk = 0.5 
                    normalized_risks.append(norm_risk)
            
            if len(normalized_risks) >= 2:
                std_dev = np.std(normalized_risks)
                patient_conflict_stds.append(std_dev)
                pid_conflict_list.append((pid, std_dev))
                all_pids_with_conflict.append(pid) # (*** 新增 ***)

        if not patient_conflict_stds:
            print("   错误: 没有找到任何具有至少2个有效单模态评分的患者，无法计算冲突。")
        else:
            avg_normalized_std = np.mean(patient_conflict_stds)
            avg_normalized_range = np.mean([np.max(patient_conflict_stds) - np.min(patient_conflict_stds) for stds in patient_conflict_stds]) # 这行计算似乎有误，修正
            
            # (*** 修正 9.3 中 avg_normalized_range 的计算 ***)
            # 之前的计算是错误的，我们应该在 9.2 中存储 p_range
            # 为简单起见，我们重新在 9.2 中添加 patient_conflict_ranges
            
            # --- 重做 9.2 和 9.3 ---
            patient_conflict_stds = []
            patient_conflict_ranges = [] # (*** 重新添加 ***)
            pid_conflict_list = []
            all_pids_with_conflict = []
            
            for pid in all_pids:
                normalized_risks = []
                for model_id in model_ids:
                    risk = pid_to_risks[pid].get(model_id, np.nan)
                    if not np.isnan(risk):
                        if global_range > 1e-6: 
                            norm_risk = (risk - global_min_risk) / global_range
                        else:
                            norm_risk = 0.5 
                        normalized_risks.append(norm_risk)
                
                if len(normalized_risks) >= 2:
                    std_dev = np.std(normalized_risks)
                    p_range = np.max(normalized_risks) - np.min(normalized_risks) # (*** 计算极差 ***)
                    
                    patient_conflict_stds.append(std_dev)
                    patient_conflict_ranges.append(p_range) # (*** 存储极差 ***)
                    pid_conflict_list.append((pid, std_dev))
                    all_pids_with_conflict.append(pid)
            
            if not patient_conflict_stds:
                 print("   错误: 没有找到任何具有至少2个有效单模态评分的患者，无法计算冲突。")
            else:
                avg_normalized_std = np.mean(patient_conflict_stds)
                avg_normalized_range = np.mean(patient_conflict_ranges) # (*** 现在这个计算是正确的 ***)
                
                print(f"\n   --- 聚合冲突指标 (基于 {len(patient_conflict_stds)}/{len(all_pids)} 个样本) ---")
                print(f"   平均归一化标准差 (Avg Conflict Std): {avg_normalized_std:.4f}")
                print(f"   平均归一化极差 (Avg Conflict Range):   {avg_normalized_range:.4f}")
                print(f"   > 解释: 归一化后的风险在 [0, 1] 之间。")
                print(f"   > 标准差(Std)越大 (Max约0.5)，冲突越大。极差(Range)越大 (Max为1.0)，冲突越大。")

                sorted_conflicts = sorted(pid_conflict_list, key=lambda x: x[1], reverse=True)
                
                # (*** 9.4 修改: 添加标签信息到打印 ***)
                print(f"\n   --- 冲突最严重的 5 个样本 (PID) ---")
                print(f"    {'PID':<30} | {'Conflict (Std)':<15} | {'Label (T/E)':<12} | 原始风险评分")
                print(f"   {'-'*30:s}-+--{'-'*15:s}-+--{'-'*12:s}-+--{'-'*20:s}")
                
                for pid, conflict_std in sorted_conflicts[:5]:
                    label = pid_to_labels.get(pid, {})
                    l_time = label.get('label_time', 'N/A')
                    l_event = label.get('label_event', 'N/A')
                    label_str = f"{l_time:>5} / {l_event:<5}"
                    
                    raw_risks_str_parts = []
                    for mid in model_ids:
                        r = pid_to_risks[pid].get(mid, 'N/A')
                        r_str = f"{r:8.2f}" if isinstance(r, float) else "  N/A   "
                        raw_risks_str_parts.append(f"{mid}: {r_str}")
                    
                    raw_risks_str = " | ".join(raw_risks_str_parts)
                    print(f"    {pid:<30} | {conflict_std:<15.4f} | {label_str:<12} | {raw_risks_str}")

                # (*** 9.5 新增: 导出所有样本到TXT文件 ***)
                print(f"\n   --- 导出冲突分析报告 ---")
                try:
                    with open(output_conflict_file, 'w') as f:
                        # 编写表头
                        header_parts = ["PID", "Conflict_Std_Norm", "Label_Time", "Label_Event"]
                        for mid in model_ids:
                            header_parts.append(f"Risk_{mid}")
                        header_parts.append(f"Risk_{multi_modal_id}")
                        f.write("\t".join(header_parts) + "\n")
                        
                        # 遍历所有排序后的冲突样本
                        for pid, conflict_std in sorted_conflicts:
                            label = pid_to_labels.get(pid, {})
                            l_time = label.get('label_time', 'N/A')
                            l_event = label.get('label_event', 'N/A')
                            
                            line_parts = [
                                str(pid),
                                f"{conflict_std:.6f}",
                                str(l_time),
                                str(l_event)
                            ]
                            
                            # 添加单模态风险
                            for mid in model_ids:
                                r = pid_to_risks[pid].get(mid, 'N/A')
                                line_parts.append(f"{r:.4f}" if isinstance(r, float) else "N/A")
                            
                            # 添加多模态风险
                            r_multi = pid_to_risks[pid].get(multi_modal_id, 'N/A')
                            line_parts.append(f"{r_multi:.4f}" if isinstance(r_multi, float) else "N/A")
                            
                            f.write("\t".join(line_parts) + "\n")
                            
                    print(f"   成功: 已将 {len(sorted_conflicts)} 条完整的患者冲突数据导出到 {output_conflict_file}")
                
                except Exception as e:
                    print(f"   错误: 导出到 {output_conflict_file} 失败。{e}")


    # (*** 10. 新增: 分析高冲突患者 与 C-Index配对正确性 的关系 ***)
    print(f"\n--- 10. [分析] 高冲突患者 vs 配对正确性 ({', '.join(model_ids)} + {multi_modal_id}) ---")
    
    if not pid_conflict_list:
        print("   跳过: 步骤9未能计算冲突指标，无法进行此分析。")
    elif total_valid_for_analysis_step7 == 0:
         print("   跳过: 步骤7未能找到有效的配对，无法进行此分析。")
    else:
        # 10.1 定义“高冲突”患者 (Top 20%)

        Top_percent = 50
        conflict_threshold = np.percentile([std for _, std in pid_conflict_list], 100 - Top_percent)
        high_conflict_pids = {pid for pid, std in pid_conflict_list if std >= conflict_threshold}
        print(f"   定义: '高冲突' = 冲突标准差排名前 {Top_percent}% (>= {conflict_threshold:.4f})。")
        print(f"   共找到 {len(high_conflict_pids)} / {len(all_pids_with_conflict)} 个高冲突患者。")
        
        # (*** 10.2 修改: 筛选出 *所有模型均有效* 且包含高冲突患者的配对 ***)
        
        # 我们使用在步骤8中定义的 final_common_mask，它确保所有单模态和多模态模型都有效(0或1)
        # final_common_mask = valid_mask_step7 & multi_modal_valid_mask (在步骤8已定义)
        
        # 如果步骤8失败了，我们需要在这里重新定义
        if 'final_common_mask' not in locals():
             multi_modal_valid_mask = (multi_modal_results != -1) & (multi_modal_results != -99)
             final_common_mask = valid_mask_step7 & multi_modal_valid_mask
             
        high_conflict_pair_indices = []
        
        # 找出 final_common_mask 为 True 的所有索引
        valid_common_indices = np.where(final_common_mask)[0]
        
        for i in valid_common_indices:
            # i 是 evaluable_pairs 中的原始索引
            pid_a, pid_b, _ = evaluable_pairs[i]
            # 检查这个 "所有模型均有效" 的配对是否涉及高冲突患者
            if (pid_a in high_conflict_pids or pid_b in high_conflict_pids):
                high_conflict_pair_indices.append(i)
        
        if not high_conflict_pair_indices:
            print("   错误: 在所有模型均有效的配对中，未找到任何包含高冲突患者的配对。")
        else:
            # 10.3 分析这些配对的正确性
            
            # (*** 修改: 同时提取单模态和多模态的结果 ***)
            valid_hc_single_modal_results = results_matrix[high_conflict_pair_indices]
            valid_hc_multi_modal_results = multi_modal_results[high_conflict_pair_indices]
            
            n_total_hc_pairs = len(valid_hc_single_modal_results)
            print(f"   共找到 {n_total_hc_pairs} 个涉及高冲突患者且所有模型均有效的C-Index配对。")
            
            # (*** 10.4 新增: 定义一个辅助函数用于安全计算百分比 ***)
            def get_percent_str(numerator, denominator, decimals=1):
                if denominator == 0:
                    return "(N/A)"
                percent = (numerator / denominator) * 100
                # 动态格式化字符串以处理对齐
                return f"({percent:5.{decimals}f}%)"

            # (*** 10.5 修改: 详细分析 ***)
            
            # 单模态正确性计数 (每对)
            correctness_counts_per_pair = np.sum(valid_hc_single_modal_results == 1, axis=1)
            
            # 多模态是否正确 (每对)
            multi_correct_mask = (valid_hc_multi_modal_results == 1)
            
            # 单模态分组的掩码
            mask_0_correct = (correctness_counts_per_pair == 0)
            mask_1_correct = (correctness_counts_per_pair == 1)
            mask_2plus_correct = (correctness_counts_per_pair >= 2)
            
            # 统计总数
            n_0_correct = np.sum(mask_0_correct)
            n_1_correct = np.sum(mask_1_correct)
            n_2plus_correct = np.sum(mask_2plus_correct)
            
            # 统计交叉结果
            n_0_correct_and_multi = np.sum(mask_0_correct & multi_correct_mask)
            n_1_correct_and_multi = np.sum(mask_1_correct & multi_correct_mask)
            n_2plus_correct_and_multi = np.sum(mask_2plus_correct & multi_correct_mask)
            
            print(f"\n   --- 高冲突配对的正确性分布 (共 {n_total_hc_pairs} 对) ---")
            
            # 类别 1: 0 个单模态正确
            percent_str_0 = get_percent_str(n_0_correct, n_total_hc_pairs)
            print(f"   {n_0_correct:>7} 个配对 {percent_str_0:>10} : 0 个单模态模型正确 (全体失败)")
            multi_percent_str_0 = get_percent_str(n_0_correct_and_multi, n_0_correct)
            print(f"       > 其中, {multi_modal_id} 正确: {n_0_correct_and_multi:>7} / {n_0_correct:<7} {multi_percent_str_0:>10}")

            # 类别 2: 1 个单模态正确
            percent_str_1 = get_percent_str(n_1_correct, n_total_hc_pairs)
            print(f"   {n_1_correct:>7} 个配对 {percent_str_1:>10} : 1 个单模态模型正确")
            multi_percent_str_1 = get_percent_str(n_1_correct_and_multi, n_1_correct)
            print(f"       > 其中, {multi_modal_id} 正确: {n_1_correct_and_multi:>7} / {n_1_correct:<7} {multi_percent_str_1:>10}")

            # 类别 3: 2+ 个单模态正确
            percent_str_2plus = get_percent_str(n_2plus_correct, n_total_hc_pairs)
            print(f"   {n_2plus_correct:>7} 个配对 {percent_str_2plus:>10} : 2个或更多 单模态模型正确")
            multi_percent_str_2plus = get_percent_str(n_2plus_correct_and_multi, n_2plus_correct)
            print(f"       > 其中, {multi_modal_id} 正确: {n_2plus_correct_and_multi:>7} / {n_2plus_correct:<7} {multi_percent_str_2plus:>10}")
            

    # (*** 11. 新增: 分析低冲突患者 与 C-Index配对正确性 的关系 ***)
    print(f"\n--- 11. [分析] 低冲突患者 vs 配对正确性 ({', '.join(model_ids)} + {multi_modal_id}) ---")
    
    if not pid_conflict_list:
        print("   跳过: 步骤9未能计算冲突指标，无法进行此分析。")
    elif total_valid_for_analysis_step7 == 0:
         print("   跳过: 步骤7未能找到有效的配对，无法进行此分析。")
    else:
        # 11.1 定义“低冲突”患者 (Bottom 20%)
        # (*** 注意: 这里使用 20 百分位数 (后20%) ***)
        Low_percent = 50
        low_conflict_threshold = np.percentile([std for _, std in pid_conflict_list], Low_percent)
        low_conflict_pids = {pid for pid, std in pid_conflict_list if std <= low_conflict_threshold}
        print(f"   定义: '低冲突' = 冲突标准差排名后 {Low_percent}% (<= {low_conflict_threshold:.4f})。")
        print(f"   共找到 {len(low_conflict_pids)} / {len(all_pids_with_conflict)} 个低冲突患者。")
        
        # 11.2 筛选出 *所有模型均有效* 且包含低冲突患者的配对
        
        # 我们使用在步骤8中定义的 final_common_mask
        if 'final_common_mask' not in locals():
             multi_modal_valid_mask = (multi_modal_results != -1) & (multi_modal_results != -99)
             final_common_mask = valid_mask_step7 & multi_modal_valid_mask
             
        low_conflict_pair_indices = []
        
        # 找出 final_common_mask 为 True 的所有索引
        valid_common_indices = np.where(final_common_mask)[0]
        
        for i in valid_common_indices:
            # i 是 evaluable_pairs 中的原始索引
            pid_a, pid_b, _ = evaluable_pairs[i]
            # (*** 注意: 检查 low_conflict_pids ***)
            if (pid_a in low_conflict_pids or pid_b in low_conflict_pids):
                low_conflict_pair_indices.append(i)
        
        if not low_conflict_pair_indices:
            print("   错误: 在所有模型均有效的配对中，未找到任何包含低冲突患者的配对。")
        else:
            # 11.3 分析这些配对的正确性
            
            valid_lc_single_modal_results = results_matrix[low_conflict_pair_indices]
            valid_lc_multi_modal_results = multi_modal_results[low_conflict_pair_indices]
            
            n_total_lc_pairs = len(valid_lc_single_modal_results)
            print(f"   共找到 {n_total_lc_pairs} 个涉及低冲突患者且所有模型均有效的C-Index配对。")
            
            # 11.4 定义一个辅助函数用于安全计算百分比
            def get_percent_str(numerator, denominator, decimals=1):
                if denominator == 0:
                    return "(N/A)"
                percent = (numerator / denominator) * 100
                return f"({percent:5.{decimals}f}%)"

            # 11.5 详细分析
            
            # 单模态正确性计数 (每对)
            correctness_counts_per_pair_lc = np.sum(valid_lc_single_modal_results == 1, axis=1)
            
            # 多模态是否正确 (每对)
            multi_correct_mask_lc = (valid_lc_multi_modal_results == 1)
            
            # 单模态分组的掩码
            mask_0_correct_lc = (correctness_counts_per_pair_lc == 0)
            mask_1_correct_lc = (correctness_counts_per_pair_lc == 1)
            mask_2plus_correct_lc = (correctness_counts_per_pair_lc >= 2)
            
            # 统计总数
            n_0_correct_lc = np.sum(mask_0_correct_lc)
            n_1_correct_lc = np.sum(mask_1_correct_lc)
            n_2plus_correct_lc = np.sum(mask_2plus_correct_lc)
            
            # 统计交叉结果
            n_0_correct_and_multi_lc = np.sum(mask_0_correct_lc & multi_correct_mask_lc)
            n_1_correct_and_multi_lc = np.sum(mask_1_correct_lc & multi_correct_mask_lc)
            n_2plus_correct_and_multi_lc = np.sum(mask_2plus_correct_lc & multi_correct_mask_lc)
            
            print(f"\n   --- 低冲突配对的正确性分布 (共 {n_total_lc_pairs} 对) ---")
            
            # 类别 1: 0 个单模态正确
            percent_str_0 = get_percent_str(n_0_correct_lc, n_total_lc_pairs)
            print(f"   {n_0_correct_lc:>7} 个配对 {percent_str_0:>10} : 0 个单模态模型正确 (全体失败)")
            multi_percent_str_0 = get_percent_str(n_0_correct_and_multi_lc, n_0_correct_lc)
            print(f"       > 其中, {multi_modal_id} 正确: {n_0_correct_and_multi_lc:>7} / {n_0_correct_lc:<7} {multi_percent_str_0:>10}")

            # 类别 2: 1 个单模态正确
            percent_str_1 = get_percent_str(n_1_correct_lc, n_total_lc_pairs)
            print(f"   {n_1_correct_lc:>7} 个配对 {percent_str_1:>10} : 1 个单模态模型正确")
            multi_percent_str_1 = get_percent_str(n_1_correct_and_multi_lc, n_1_correct_lc)
            print(f"       > 其中, {multi_modal_id} 正确: {n_1_correct_and_multi_lc:>7} / {n_1_correct_lc:<7} {multi_percent_str_1:>10}")

            # 类别 3: 2+ 个单模态正确
            percent_str_2plus = get_percent_str(n_2plus_correct_lc, n_total_lc_pairs)
            print(f"   {n_2plus_correct_lc:>7} 个配对 {percent_str_2plus:>10} : 2个或更多 单模态模型正确")
            multi_percent_str_2plus = get_percent_str(n_2plus_correct_and_multi_lc, n_2plus_correct_lc)
            print(f"       > 其中, {multi_modal_id} 正确: {n_2plus_correct_and_multi_lc:>7} / {n_2plus_correct_lc:<7} {multi_percent_str_2plus:>10}")

    print("\n分析完成。")