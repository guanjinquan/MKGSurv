import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    hamming_loss
)

try:
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
except ImportError:
    concordance_index_censored = None
    concordance_index_ipcw = None

try:
    from lifelines.utils import concordance_index as lifelines_concordance_index
except ImportError:
    lifelines_concordance_index = None


def recall_top_k(batch_treat_risks, labels, k):
    """
    Calculates Recall@K.
    Args:
        batch_treat_risks: Risk scores (Lower is better).
        labels: Ground truth int.
        k: Top-K.
    """
    # 1. 转换格式
    if isinstance(batch_treat_risks, torch.Tensor):
        scores = batch_treat_risks.detach().cpu().numpy()
    else:
        scores = np.array(batch_treat_risks)
        
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.array(labels)

    if scores.ndim == 3:
        scores = scores.reshape(scores.shape[0], -1)
    assert scores.ndim == 2, f"Input scores shape must be (Batch_Size, Num_Classes), but got {scores.shape}"

    if len(scores) != len(labels):
        print(f"[Warning] Length mismatch: risks {len(scores)} vs labels {len(labels)}")
        return 0.0

    hits = 0
    num_samples = len(scores)
    
    # 如果 K 超过类别总数，取类别总数 (避免切片越界虽不报错但逻辑清晰)
    num_classes = scores.shape[1]
    real_k = min(k, num_classes)

    for i in range(num_samples):
        patient_risks = scores[i] # 现在这保证是一个 1D 数组 (num_classes,)
        gt_index = labels[i]
        
        # Risk 越低越好 -> 升序排列 (argsort 默认升序)
        sorted_indices = np.argsort(patient_risks)
        top_k_indices = sorted_indices[:real_k]
        
        # 现在 top_k_indices 是 1D 数组，'in' 操作符能正常工作
        if gt_index in top_k_indices:
            hits += 1
            
    return float(hits / num_samples) if num_samples > 0 else 0.0

def multiple_classification_metrics(logits, labels, with_sigmoid=True):
    """
    计算多标签分类指标 (Multi-Label Classification Metrics)。

    参数:
    logits (list or Tensor): 模型的原始输出 (logits)。
                             预期 shape=(N_samples, N_classes)
                             e.g., [[0.1, -0.2, 0.3], ...]
    labels (list): 真实标签列表, 每个元素是字典。
                   e.g., [{'treatment_type_onehot': [0, 1, 1]}, ...]
    
    返回:
    dict: 包含多项分类指标的字典。
    """
    
    # 1. 预处理输入 (Pre-process Inputs)
    if isinstance(logits, list):
        if not logits:
            # 处理空列表的情况
            logits_tensor = torch.empty(0, 0, dtype=torch.float32)
        else:
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
    else:
        logits_tensor = logits

    # 从字典列表中提取真实标签
    y_true_np = np.array([d['treatment_type_onehot'] for d in labels])

    # 如果输入为空, 返回0值
    if logits_tensor.shape[0] == 0 or y_true_np.shape[0] == 0:
        return {
            "Accuracy": 0.0,
            "Hamming Loss": 1.0,
            "Precision (Macro)": 0.0,
            "Recall (Macro)": 0.0,
            "F1 (Macro)": 0.0,
        }

    # 2. 处理 Logits (Process Logits)
    # 多标签分类：应用 sigmoid 获得每个类别的独立概率
    if with_sigmoid:
        probs = torch.sigmoid(logits_tensor)
    else:
        probs = logits_tensor
    
    # 基于 0.5 的阈值获取硬预测 (0 或 1)
    preds = (probs > 0.5).int()

    # 3. 转换为 NumPy 以便 sklearn 使用
    y_pred_np = preds.detach().cpu().numpy()

    # 4. 计算指标 (Calculate Metrics)
    
    # 准确率 (子集准确率): 预测的标签集与真实标签集完全匹配的比例
    acc = accuracy_score(y_true_np, y_pred_np)

    # 汉明损失: 预测错误的标签比例 (越低越好)
    ham_loss = hamming_loss(y_true_np, y_pred_np)

    # --- Macro-averaged metrics ---
    # (独立计算每个类别的指标, 然后取平均值，平等对待所有类别)
    prec_macro = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    rec_macro = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)

    # 5. 结构化输出
    metrics = {
        "Accuracy": acc,
        "Hamming Loss": ham_loss,
        "Precision (Macro)": prec_macro,
        "Recall (Macro)": rec_macro,
        "F1 (Macro)": f1_macro,
    }

    return metrics


def calculate_brier_score(probs, event_indicators):
    """
    计算 Brier Score (BS)。
    """
    predicted_probs = probs.flatten()
    # event_indicators (1=事件, 0=删失) 被当作二元分类的真实标签
    bs = np.mean((predicted_probs - event_indicators) ** 2)
    return bs


def to_structured_array(labels):
    """
    辅助函数：将 list of dicts 转换为 sksurv 要求的结构化数组
    dtype = [('event', bool), ('time', float)]
    """
    events = [bool(d['label_event']) for d in labels]
    times = [float(d['label_time']) for d in labels]
    
    y_structured = np.zeros(len(labels), dtype={'names': ('event', 'time'),
                                                'formats': ('?', '<f8')})
    y_structured['event'] = events
    y_structured['time'] = times
    return y_structured


def _concordance_from_arrays(event_observed, event_time, risk_scores):
    numerator = 0.0
    denominator = 0.0
    n_samples = len(event_time)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if event_time[i] < event_time[j] and event_observed[i]:
                case_idx, control_idx = i, j
            elif event_time[j] < event_time[i] and event_observed[j]:
                case_idx, control_idx = j, i
            elif event_time[i] == event_time[j] and event_observed[i] != event_observed[j]:
                case_idx = i if event_observed[i] else j
                control_idx = j if event_observed[i] else i
            else:
                continue

            denominator += 1.0
            if risk_scores[case_idx] > risk_scores[control_idx]:
                numerator += 1.0
            elif risk_scores[case_idx] == risk_scores[control_idx]:
                numerator += 0.5

    if denominator == 0:
        return 0.5
    return numerator / denominator


def _estimate_censoring_survival(train_event_observed, train_event_time, query_times):
    censor_events = ~train_event_observed.astype(bool)
    survival = 1.0
    survival_before_time = {}

    for time in np.sort(np.unique(train_event_time)):
        survival_before_time[time] = survival
        at_risk = np.sum(train_event_time >= time)
        censored_at_time = np.sum((train_event_time == time) & censor_events)
        if at_risk > 0:
            survival *= 1.0 - censored_at_time / at_risk

    return np.array([survival_before_time.get(time, survival) for time in query_times], dtype=float)


def _ipcw_concordance_from_arrays(train_event_observed, train_event_time, test_event_observed, test_event_time, risk_scores):
    censoring_survival = _estimate_censoring_survival(train_event_observed, train_event_time, test_event_time)
    numerator = 0.0
    denominator = 0.0
    n_samples = len(test_event_time)

    for i in range(n_samples):
        if not test_event_observed[i] or censoring_survival[i] <= 0:
            continue

        weight = 1.0 / (censoring_survival[i] ** 2)
        for j in range(n_samples):
            if i == j or not (test_event_time[i] < test_event_time[j]):
                continue

            denominator += weight
            if risk_scores[i] > risk_scores[j]:
                numerator += weight
            elif risk_scores[i] == risk_scores[j]:
                numerator += 0.5 * weight

    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_c_index(event_observed, event_time, risk_scores):
    """
    计算 Harrell's C-index (标准 C-index)
    """
    if not isinstance(event_observed, np.ndarray) or event_observed.dtype != bool:
        event_observed = event_observed.astype(int).astype(bool)

    if concordance_index_censored is None:
        if lifelines_concordance_index is None:
            return _concordance_from_arrays(event_observed, event_time, risk_scores)
        try:
            return lifelines_concordance_index(event_time, -risk_scores, event_observed)
        except Exception as e:
            print(f"[Harrell C-index Error] {e}")
            return _concordance_from_arrays(event_observed, event_time, risk_scores)

    try:
        ci_tuple = concordance_index_censored(event_observed, event_time, risk_scores)
        return ci_tuple[0]
    except Exception as e:
        print(f"[Harrell C-index Error] {e}")
        return 0.5


def calculate_ipcw_c_index(y_train, y_test, risk_scores):
    """
    计算 Uno's IPCW C-index
    """
    if concordance_index_ipcw is None:
        try:
            return _ipcw_concordance_from_arrays(
                y_train["event"],
                y_train["time"],
                y_test["event"],
                y_test["time"],
                risk_scores,
            )
        except Exception as e:
            print(f"[IPCW C-index Error] {e}")
            return 0.0

    try:
        # Uno's C-index
        uno_c, _, _, _, _ = concordance_index_ipcw(y_train, y_test, risk_scores)
        return uno_c
    except Exception as e:
        print(f"[IPCW C-index Error] {e}")
        return 0.0


def survival_metrics(logits, labels, train_labels=None):
    """
    计算生存分析指标，包含 C-index 和 IPCW C-index。
    """
    if isinstance(logits, list):
        logits = torch.tensor(logits, dtype=torch.float32)
    
    # 1. 准备风险分数
    probs_np = logits.detach().cpu().numpy().flatten()
    
    # 2. 准备 Harrell's C-index 需要的普通数组
    event_times = np.array([label["label_time"] for label in labels])
    event_indicators = np.array([label["label_event"] for label in labels]).astype(bool)
    
    # 计算 Harrell's C-index
    ci_harrell = calculate_c_index(event_indicators, event_times, probs_np)
    
    # 3. 准备 IPCW C-index 需要的结构化数组
    y_test_structured = to_structured_array(labels)
    
    if train_labels is not None:
        y_train_structured = to_structured_array(train_labels)
    else:
        # 如果没给训练集，勉强用测试集代替
        y_train_structured = y_test_structured

    # 计算 IPCW C-index
    ci_ipcw = calculate_ipcw_c_index(y_train_structured, y_test_structured, probs_np)
    
    # 4. 返回结果
    return {
        "C-Index": ci_harrell,
        "C-Index-IPCW": ci_ipcw
    }
