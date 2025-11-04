import json
import torch
import numpy as np
from itertools import combinations
from tqdm import tqdm
# import matplotlib.pyplot as plt  # (*** 已移除 ***)
# from matplotlib_venn import venn3 # (*** 已移除 ***)
import os # (*** 新增, 确保路径操作更可靠 ***)

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
    # cumprod 默认是累积乘积
    S = torch.cumprod(1 - hazards_tensor, dim=0) # 假设 logits 是一维的
    
    # 风险评分 (Risk Score) 可以是多种定义
    # 1. 累积风险: torch.sum(hazards_tensor)
    # 2. 期望生存时间: torch.sum(S) (分数越高，生存越好，风险越低)
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
        label_a = pid_to_labels[pid_a] # {'label_Y': time, 'label_c': event (1=censored, 0=event)}
        label_b = pid_to_labels[pid_b]
        
        T_a, E_a = label_a['label_Y'], 1 - label_a['label_c'] # 转换: 1=event, 0=censored
        T_b, E_b = label_b['label_Y'], 1 - label_b['label_c']
        
        # C-Index 规则 (参见 analysis_readme.md)
        if (T_a < T_b) and (E_a == 1):
            # 这是一个可评估对，A 应该比 B 有更高风险
            evaluable_pairs.append((pid_a, pid_b, 'A_gt_B')) # A risk > B risk
        elif (T_b < T_a) and (E_b == 1):
            # 这是一个可评估对，B 应该比 A 有更高风险
            evaluable_pairs.append((pid_b, pid_a, 'B_gt_A')) # B risk > A risk
        
        # 其他情况 (T_a == T_b, 或 T_a < T_b 但 E_a=0, 或 T_b < T_a 但 E_b=0, 或 E_a=0 且 E_b=0)
        # 都是不可评估对，直接忽略。
            
    return evaluable_pairs

if __name__ == "__main__":
    
    # --- 1. 定义文件路径 ---
    # !!! 注意：请根据您的实际路径修改这些值 !!!
    base_path = "/home/Guanjq/NewWork/MedAlignFusion/Checkpoints/oscc_inhouse"
    
    # (*** 修改: 指向新的 'test_pid_to_data.json' ***)
    files_to_analyze = {
        "001": f"{base_path}/inhouse_run001+msa/test_pid_to_data.json",
        "006": f"{base_path}/inhouse_run006+msa/test_pid_to_data.json",
        "007": f"{base_path}/inhouse_run007+msa/test_pid_to_data.json"
    }
    
    # (*** 移除: 不再需要单独的 'labels_file_path' ***)
    # labels_file_path = f"{base_path}/inhouse_run001+msa/test_pid_to_original_labels.json"

    # --- 2. 加载所有数据 ---
    print("正在加载模型数据 (logits + labels)...")
    model_data = {} # (*** 重命名: model_logits -> model_data ***)
    for run_id, file_path in files_to_analyze.items():
        model_data[run_id] = load_json(file_path)
        print(f"  已加载 {run_id} ({len(model_data[run_id])} 个样本)")

    # (*** 修改: 从加载的 model_data 中提取标签 ***)
    print(f"\n正在从 {list(files_to_analyze.keys())[0]} 的数据中提取标签...")
    pid_to_labels = {}
    first_model_id = list(model_data.keys())[0]
    # 从第一个模型的数据中构建 pid_to_labels 字典
    for pid, data in model_data[first_model_id].items():
        pid_to_labels[pid] = data['label']
    print(f"  已提取 {len(pid_to_labels)} 个标签")

    model_ids = list(model_data.keys()) # (*** 确保使用新字典名 ***)
    
    # --- 3. 找出所有可评估的配对 ---
    print("\n正在识别所有可评估的患者对...")
    # evaluable_pairs 列表: [(pid_shorter, pid_longer, expected_risk_order), ...]
    # e.g., [('patient_5', 'patient_10', 'A_gt_B')]
    evaluable_pairs = get_evaluable_pairs(pid_to_labels)
    total_evaluable = len(evaluable_pairs)
    if total_evaluable == 0:
        print("错误：未找到任何可评估的配对。请检查您的标签数据。")
        exit(1)
        
    print(f"  共找到 {total_evaluable} 个可评估的配对。")

    # --- 4. 分析每个配对在所有模型上的表现 ---
    print("正在分析所有配对的一致性...")
    
    # 存储每个 pair 在每个 model 上的结果: 1=一致, 0=不一致, -1=持平
    # results_matrix[pair_index][model_index]
    results_matrix = np.zeros((total_evaluable, len(model_ids)), dtype=int)
    
    # 存储每个模型的 C-Index (用于验证)
    model_concordance_counts = {run_id: 0 for run_id in model_ids}
    model_valid_pairs = {run_id: 0 for run_id in model_ids}

    for i, (pid_a, pid_b, expected_order) in enumerate(tqdm(evaluable_pairs)):
        # pid_a 是生存时间更短且发生事件的患者
        # pid_b 是生存时间更长的患者
        # 期望：Risk(A) > Risk(B)
        
        for j, run_id in enumerate(model_ids):
            # (*** 修改: 从 'model_data' 字典中提取 'logits' ***)
            logits_a = model_data[run_id][pid_a]['logits']
            logits_b = model_data[run_id][pid_b]['logits']
            
            risk_a = calculate_risk_from_logits(logits_a)
            risk_b = calculate_risk_from_logits(logits_b)
            
            # 检查一致性 (1=一致, 0=不一致, -1=持平)
            status = check_concordance_status(None, None, risk_a, risk_b)
            results_matrix[i, j] = status
            
            if status != -1: # 只要风险不持平
                model_valid_pairs[run_id] += 1
                if status == 1: # 如果一致
                    model_concordance_counts[run_id] += 1

    # --- 5. 打印 C-Index 验证结果 ---
    print("\n--- 模型 C-Index (验证) ---")
    for run_id in model_ids:
        # (*** 修改：添加除零检查 ***)
        if model_valid_pairs[run_id] > 0:
            c_index = model_concordance_counts[run_id] / model_valid_pairs[run_id]
            print(f"  模型 {run_id}: {c_index:.4f}  ({model_concordance_counts[run_id]} / {model_valid_pairs[run_id]})")
        else:
            print(f"  模型 {run_id}: N/A (没有有效的非持平配对)")

    # --- 6. 寻找互补配对 ---
    print("\n--- 互补配对分析 (A 失败, B 成功) ---")
    
    # 遍历所有模型组合
    for (idx_a, model_a), (idx_b, model_b) in combinations(enumerate(model_ids), 2):
        
        # A 失败 (0) 且 B 成功 (1)
        a_fails_b_succeeds = np.sum(
            (results_matrix[:, idx_a] == 0) & (results_matrix[:, idx_b] == 1)
        )
        
        # B 失败 (0) 且 A 成功 (1)
        b_fails_a_succeeds = np.sum(
            (results_matrix[:, idx_b] == 0) & (results_matrix[:, idx_a] == 1)
        )
        
        total_complementary = a_fails_b_succeeds + b_fails_a_succeeds
        
        print(f"\n  比较: {model_a} vs {model_b}")
        print(f"    {model_a} 失败, {model_b} 成功: {a_fails_b_succeeds} 个配对")
        print(f"    {model_b} 失败, {model_a} 成功: {b_fails_a_succeeds} 个配对")
        print(f"    总互补配对数: {total_complementary}")

    # --- (*** 修改 ***) 7. 韦恩图 (Venn Diagram) 分析 ---
    print("\n--- 韦恩图 (Venn Diagram) 分析 ---")
    
    # 7.1. 筛选出所有模型都没有"风险持平"(-1)的配对
    # 只有 0 (不一致) 和 1 (一致) 的配对才用于韦恩图
    valid_venn_mask = np.all(results_matrix != -1, axis=1)
    valid_results = results_matrix[valid_venn_mask]
    total_valid_for_venn = len(valid_results)
    
    if total_valid_for_venn == 0:
        print("  错误: 找不到任何所有模型均无'风险持平'的配对，无法计算韦恩图信息。")
    else:
        print(f"  基于 {total_valid_for_venn} 个有效配对 (所有模型均无'风险持平') 进行分析。")

        # 7.2. 为三个模型创建布尔掩码 (Correct / Incorrect)
        # 假设 model_ids 的顺序是 ['001', '006', '007']
        c_A = (valid_results[:, 0] == 1) # model '001' 正确
        c_B = (valid_results[:, 1] == 1) # model '006' 正确
        c_C = (valid_results[:, 2] == 1) # model '007' 正确
        
        i_A = ~c_A # model '001' 错误
        i_B = ~c_B # model '006' 错误
        i_C = ~c_C # model '007' 错误
        
        # 7.3. 计算韦恩图的8个区域
        n_ABC = np.sum(c_A & c_B & c_C) # A, B, C 均正确
        n_ABc = np.sum(c_A & c_B & i_C) # A, B 正确; C 错误
        n_AbC = np.sum(c_A & i_B & c_C) # A, C 正确; B 错误
        n_aBC = np.sum(i_A & c_B & c_C) # B, C 正确; A 错误
        n_Abc = np.sum(c_A & i_B & i_C) # 仅 A 正确
        n_aBc = np.sum(i_A & c_B & i_C) # 仅 B 正确
        n_abC = np.sum(i_A & i_B & c_C) # 仅 C 正确
        n_abc = np.sum(i_A & i_B & i_C) # A, B, C 均错误
        
        print("\n  --- 韦恩图区域计数 ---")
        print(f"  {model_ids[0]}, {model_ids[1]}, {model_ids[2]} 均正确 (ABC): {n_ABC}")
        print(f"  {model_ids[0]}, {model_ids[1]} 正确; {model_ids[2]} 错误 (ABc): {n_ABc}")
        print(f"  {model_ids[0]}, {model_ids[2]} 正确; {model_ids[1]} 错误 (AbC): {n_AbC}")
        print(f"  {model_ids[1]}, {model_ids[2]} 正确; {model_ids[0]} 错误 (aBC): {n_aBC}")
        print(f"  仅 {model_ids[0]} 正确 (Abc): {n_Abc}")
        print(f"  仅 {model_ids[1]} 正确 (aBc): {n_aBc}")
        print(f"  仅 {model_ids[2]} 正确 (abC): {n_abC}")
        print(f"  {model_ids[0]}, {model_ids[1]}, {model_ids[2]} 均错误 (abc): {n_abc}")
        
        # 7.4. (*** 已移除 ***) 绘制并保存韦恩图

    print("\n分析完成。")

