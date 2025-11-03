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



def survival_metrics(logits, labels):
    """
    根据模型输出和真实标签计算 C-index 和 Integrated Brier Score (IBS)。
    """
    # --- 步骤 1: 处理模型输出 (logits) ---
    if isinstance(logits, list):
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
    else:
        logits_tensor = logits
    
    # --- 步骤 2: 处理标签 (labels) ---
    if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], dict):
        try:
            label_Y_list = [d['label_Y'] for d in labels]
            label_c_list = [d['label_c'] for d in labels]
            labels_tensor = torch.tensor([label_Y_list, label_c_list], dtype=torch.float32).T
        except KeyError:
            print("ERROR: Label dictionaries are missing 'label_Y' or 'label_c' keys.")
            return {"C-Index": 0.0, "IBS": 1.0} # 返回 IBS 的最差值
    elif isinstance(labels, list):
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
    else:
        labels_tensor = labels
    
    # --- 步骤 3: 风险和生存函数计算 ---
    hazards_tensor = torch.sigmoid(logits_tensor)
    # S 是生存函数 S(t) = P(T > t) 在每个时间点的估计
    S = torch.cumprod(1 - hazards_tensor, dim=1)
    risk_scores_tensor = -torch.sum(S, dim=1)
    
    # 转换为 numpy
    risk_scores = risk_scores_tensor.cpu().detach().numpy()
    labels_np = labels_tensor.cpu().detach().numpy()
    S_np = S.cpu().detach().numpy() # 生存概率, shape=(N_samples, n_time_bins)

    # 提取事件时间和状态
    event_times = labels_np[:, 0]
    # pycox EvalSurv 需要: 1=事件, 0=审查 (或 True/False)
    # 原始 label_c: 0=事件, 1=审查
    event_observed = (1 - labels_np[:, 1]).astype(bool)

    # --- 步骤 4: C-Index 计算 ---
    ci_result_tuple = concordance_index_censored(event_observed, event_times, risk_scores)
    ci = ci_result_tuple[0]
    
    if ci is None or np.isnan(ci):
        ci = 0.0

    # --- 步骤 5: 积分布里尔分数 (IBS) 计算 (新!! ) ---
    ibs = 1.0 # 默认Brier分数为1.0 (最差)
    try:
        # 1. 为 pycox 准备生存函数 DataFrame
        # pycox 需要 (n_time_bins, n_samples)
        # S_np 是 (n_samples, n_time_bins)，所以需要转置
        time_bins = np.arange(S_np.shape[1])
        S_df = pd.DataFrame(S_np.T, index=time_bins)
        
        # 2. 初始化 EvalSurv
        # durations = event_times, events = event_observed (bool or 0/1)
        ev = EvalSurv(surv_df=S_df, 
                      durations=event_times, 
                      events=event_observed, 
                      censor_surv='km') # 使用 Kaplan-Meier 估计审查分布
        
        # 3. 定义积分时间网格 (time_grid)
        # 我们模仿 Evaluation 类的逻辑：在事件发生的时间范围内取100个点，并丢弃不稳定的尾部
        observed_times = event_times[event_observed]
        if len(observed_times) > 0:
            min_time = observed_times.min()
            max_time = observed_times.max()
        else: # Fallback (如果没有事件发生)
            min_time = event_times.min()
            max_time = event_times.max()
        
        if min_time >= max_time: # 处理只有一个时间点或数据为空的情况
            time_grid = np.array([min_time])
            drop_last_times = 0
        else:
            time_grid = np.linspace(min_time, max_time, 100)
            drop_last_times = 25 # 丢弃最后25个点以保证稳定性
        
        if drop_last_times > 0 and len(time_grid) > drop_last_times:
            time_grid = time_grid[:-drop_last_times]
        
        # 4. 计算 IBS
        # pycox 会自动在 time_grid 上的点计算 BS 并进行积分
        ibs = ev.integrated_brier_score(time_grid)
        
        if np.isnan(ibs):
            ibs = 1.0 # 处理计算中可能出现的NaN

    except Exception as e:
        print(f"ERROR: Could not calculate IBS: {e}")
        ibs = 1.0 # 发生错误时，设为最差分数

    # --- 步骤 6: 返回所有指标 ---
    # 将 "Brier-Score" 重命名为 "IBS"
    return {"C-Index": ci, "IBS": ibs}

