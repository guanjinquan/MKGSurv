import torch
import numpy as np
from sksurv.metrics import concordance_index_censored

def calculate_brier_score(probs, event_indicators):
    """
    计算 Brier Score (BS)。
    
    此函数针对 (n_samples, 1) 的风险预测输出进行了修改。
    它计算的是预测的事件概率与实际的事件指示器 (label_event) 之间的均方误差。
    
    注意：此方法将“删失” (event_indicator == 0) 视为“未发生事件”，
    这符合BS "average squared error between the observed survival status
    and the predicted survival probability" 的描述。
    
    参数:
    probs (np.array): 预测概率, shape=(n_samples, 1)
    event_indicators (np.array): 事件指示器 (1: 事件发生, 0: 删失)
    
    返回:
    float: Brier Score
    """
    # 将 (n_samples, 1) 的概率展平为 (n_samples,)
    predicted_probs = probs.flatten()
    
    # 计算 (预测概率 - 真实标签)^2 的均值
    # event_indicators (1=事件, 0=删失) 被当作二元分类的真实标签
    bs = np.mean((predicted_probs - event_indicators) ** 2)
    
    return bs

def calculate_c_index(event_observed, event_time, risk_scores):
    """
    封装 sksurv 的 concordance_index_censored 函数，计算 C-index。
    
    参数:
    event_observed (np.array): 事件是否发生 (bool 或 int: 1/True 为发生, 0/False 为删失)
    event_time (np.array): 事件时间
    risk_scores (np.array): 预测风险分数 (越高表示越高风险/越短生存时间)
    
    返回:
    float: C-index 值。如果计算失败或 NaN，返回 0.0
    """
    # 确保 event_observed 是 bool 类型
    if not isinstance(event_observed, np.ndarray) or event_observed.dtype != bool:
        event_observed = event_observed.astype(bool)
    
    # 调用 sksurv 函数
    ci_tuple = concordance_index_censored(event_observed, event_time, risk_scores)
    
    # 提取 C-index 值
    ci = ci_tuple[0]
    
    # 处理 NaN 或 None 的情况
    if ci is None or np.isnan(ci):
        ci = 0.0
    
    return ci

def survival_metrics(logits, labels):
    """
    根据模型输出和真实标签计算 C-index 和 Brier Score (BS)。
    
    参数:
    logits (Tensor or list): 模型的原始输出, shape=(N_samples, 1) # 风险 logits
    labels (list): 真实标签列表, 每个元素是字典 {"label_time": 真实事件 days, "label_event": 是否发生事件}
    
    返回:
    dict: 包含 C-index 和 BS 的字典
    """
    if isinstance(logits, list):
        logits = torch.tensor(logits, dtype=torch.float32)
    
    # 将logits转换为生存概率
    # logits shape=(N_samples, 1) -> probs shape=(N_samples, 1)
    probs = torch.sigmoid(logits) 
    probs_np = probs.detach().cpu().numpy()
    
    # 提取真实标签信息
    event_times = np.array([label["label_time"] for label in labels])
    event_indicators = np.array([label["label_event"] for label in labels])
    
    # 计算 C-index 使用封装函数
    ci = calculate_c_index(event_indicators, event_times, probs_np.flatten())
    
    # 计算 Brier Score
    bs = calculate_brier_score(probs_np, event_indicators)
    
    # 返回所有指标
    return {"C-Index": ci, "BS": bs}



if __name__ == "__main__":
    # 模拟数据
    n_samples = 100
    n_time_bins = 1  # 匹配 (n_samples, 1) 的输出
    
    # 模型输出 (logits)
    logits = torch.randn(n_samples, n_time_bins) 
    
    # 真实标签
    labels = [
        {
            "label_time": np.random.exponential(365),  # 生存时间（天）
            "label_event": np.random.choice([0, 1])    # (1: 事件发生, 0: 删失)
        }
        for _ in range(n_samples)
    ]
    
    # 计算指标
    metrics = survival_metrics(logits, labels)
    print(f"C-Index: {metrics['C-Index']:.4f}")
    print(f"Brier Score: {metrics['BS']:.4f}")