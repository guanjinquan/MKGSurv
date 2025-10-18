import torch.nn.functional as F
import torch
from sksurv.metrics import concordance_index_censored
import numpy as np


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
    根据模型输出和真实标签计算 C-index。
    """
    # --- 步骤 1: 处理模型输出 (logits) ---
    if isinstance(logits, list):
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
    else:
        logits_tensor = logits
    
    # --- 步骤 2: 处理标签 (labels) ---
    # 这是本次修改的核心！
    if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], dict):
        # 如果 labels 是一个字典列表, e.g., [{'label_Y': 5, 'label_c': 0}, ...]
        # 我们需要手动将它解析出来
        try:
            label_Y_list = [d['label_Y'] for d in labels]
            label_c_list = [d['label_c'] for d in labels]
            
            # 将两个列表组合成一个 (N, 2) 的张量
            labels_tensor = torch.tensor([label_Y_list, label_c_list], dtype=torch.float32).T
        except KeyError:
            print("ERROR: Label dictionaries are missing 'label_Y' or 'label_c' keys.")
            return {"C-Index": 0.0}
    elif isinstance(labels, list):
        # 兼容列表的列表的情况
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
    else:
        # 如果已经是张量，直接使用
        labels_tensor = labels
    
    # --- 步骤 3: 后续计算逻辑 ---
    hazards_tensor = torch.sigmoid(logits_tensor)
    S = torch.cumprod(1 - hazards_tensor, dim=1)
    risk_scores_tensor = -torch.sum(S, dim=1)
    
    risk_scores = risk_scores_tensor.cpu().detach().numpy()
    labels_np = labels_tensor.cpu().detach().numpy()

    event_times = labels_np[:, 0]
    event_observed = (1 - labels_np[:, 1]).astype(bool)

    ci_result_tuple = concordance_index_censored(event_observed, event_times, risk_scores)
    ci = ci_result_tuple[0]
    
    if ci is None or np.isnan(ci):
        ci = 0.0

    return {"C-Index": ci}