import torch.nn.functional as F
import torch
from sksurv.metrics import concordance_index_censored
import numpy as np
from pycox.evaluation import EvalSurv  # 替换 sksurv
import pandas as pd


def classification_metrics(logits, labels):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

    logits_np = np.array(logits)
    labels_np = np.array(labels)

    if labels_np.ndim == 2 and labels_np.shape[1] > 1:
        labels_for_metrics = np.argmax(labels_np, axis=1)
    else:
        labels_for_metrics = labels_np

    y_pred = np.argmax(logits_np, axis=1)

    logits_tensor = torch.tensor(logits_np, dtype=torch.float32)
    y_prob = F.softmax(logits_tensor, dim=1).numpy()

    acc = accuracy_score(labels_for_metrics, y_pred)
    macro_f1 = f1_score(labels_for_metrics, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(labels_for_metrics, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(labels_for_metrics, y_pred, average='macro', zero_division=0)

    num_classes = logits_np.shape[1] if logits_np.ndim == 2 else 2
    auc = 0.0
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels_for_metrics, y_prob[:, 1])
        elif num_classes > 2:
            auc = roc_auc_score(labels_for_metrics, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    return {"Acc": acc, "F1": macro_f1, "Recall": macro_recall, "Precision": macro_precision, "AUC": auc}



import torch
import numpy as np
import pandas as pd
from pycox.evaluation import EvalSurv
import scipy.integrate # <--- 新增导入

def survival_metrics(logits, labels):
    """
    根据模型输出和真实标签计算 C-index 和 Integrated Brier Score (IBS)。
    
    参数:
    logits (Tensor): 模型的原始输出, shape=(N_samples, n_time_bins)
    labels (list or Tensor): 真实标签
    
    假设:
    - 每个 bin 自动代表 3 个月 (例如, [0, 3, 6, 9, ...])
    """
    
    # --- 步骤 1: 处理模型输出 (logits) ---
    if isinstance(logits, list):
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
    else:
        logits_tensor = logits
    
    # --- 步骤 1b: 自动生成 time_bin_cutoffs ---
    # 根据 "每个bin代表3个月" 的规则自动生成
    n_time_bins = logits_tensor.shape[1]
    time_bin_cutoffs = np.arange(n_time_bins) * 3.0

    # --- 步骤 2: 处理标签 (labels) ---
    if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], dict):
        try:
            label_Y_list = [d['label_Y'] for d in labels] # 这是 bin 索引
            label_c_list = [d['label_c'] for d in labels]
            labels_tensor = torch.tensor([label_Y_list, label_c_list], dtype=torch.float32).T
        except KeyError:
            print("ERROR: Label dictionaries are missing 'label_Y' or 'label_c' keys.")
            return {"C-Index": 0.0, "IBS": 1.0}
    elif isinstance(labels, list):
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
    else:
        labels_tensor = labels
    
    # --- 步骤 3: 风险和生存函数计算 ---
    hazards_tensor = torch.sigmoid(logits_tensor)
    S = torch.cumprod(1 - hazards_tensor, dim=1)
    risk_scores_tensor = -torch.sum(S, dim=1)
    
    risk_scores = risk_scores_tensor.cpu().detach().numpy()
    labels_np = labels_tensor.cpu().detach().numpy()
    S_np = S.cpu().detach().numpy() # shape=(N_samples, n_time_bins)

    # --- 步骤 4: C-Index 计算 ---
    # C-Index 是基于排序的，所以使用 bin 索引或真实时间都可以
    # 我们在这里仍然使用 bin 索引 event_times_binned
    event_times_binned = labels_np[:, 0]
    event_observed = (1 - labels_np[:, 1]).astype(bool)

    ci_result_tuple = concordance_index_censored(event_observed, event_times_binned, risk_scores)
    ci = ci_result_tuple[0]
    
    if ci is None or np.isnan(ci):
        ci = 0.0

    # --- 步骤 5: 积分布里尔分数 (IBS) 计算 ---
    
    # [!! 新增修复 !!]
    # 解决 'scipy.integrate' has no attribute 'simps' 的问题
    # 新版 scipy (>=1.6.0) 将 simps 重命名为 simpson。
    # pycox 可能依赖了旧版的 simps。我们在这里“猴子补丁”它:
    if not hasattr(scipy.integrate, 'simps'):
        scipy.integrate.simps = scipy.integrate.simpson
        
    ibs = 1.0 
    try:
        # 1. S_df 必须使用 *真实时间* 作为索引
        # S_np 是 (n_samples, n_time_bins)，转置为 (n_time_bins, n_samples)
        # index 使用我们自动生成的 time_bin_cutoffs
        S_df = pd.DataFrame(S_np.T, index=time_bin_cutoffs)
        
        # 2. durations 必须是 *真实时间*
        # event_times_binned 是 bin 索引 (e.g., 0, 1, 5, 20)
        # 我们需要将它们映射回真实时间
        # 我们使用 bin 的起始时间作为事件时间
        real_event_times = time_bin_cutoffs[event_times_binned.astype(int)]

        # 3. 初始化 EvalSurv
        # 修复: 'surv_df=' 改为 'surv='
        ev = EvalSurv(surv=S_df, 
                      durations=real_event_times, # <-- 使用真实时间
                      events=event_observed, 
                      censor_surv='km') 
        
        # 4. time_grid 必须基于 *真实时间*
        # 我们在 *真实事件时间* 的范围内取100个点
        real_observed_times = real_event_times[event_observed]
        if len(real_observed_times) > 0:
            min_time = real_observed_times.min()
            max_time = real_observed_times.max()
        else: 
            min_time = real_event_times.min()
            max_time = real_event_times.max()
        
        if min_time >= max_time: 
            time_grid = np.array([min_time])
            drop_last_times = 0
        else:
            time_grid = np.linspace(min_time, max_time, 100)
            drop_last_times = 25 
        
        if drop_last_times > 0 and len(time_grid) > drop_last_times:
            time_grid = time_grid[:-drop_last_times]
        
        # 5. 计算 IBS
        ibs = ev.integrated_brier_score(time_grid) # <-- 现在在真实时间上积分
        
        if np.isnan(ibs):
            ibs = 1.0 

    except Exception as e:
        # 保留它是个好习惯
        print(f"ERROR: Could not calculate IBS: {e}")
        ibs = 1.0 

    # --- 步骤 6: 返回所有指标 ---
    return {"C-Index": ci, "IBS": ibs}

