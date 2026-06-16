import numpy as np
import torch
import matplotlib.pyplot as plt

def convert_logits_to_risk_scores(logits_list: list) -> np.ndarray:
    """
    将模型输出的 logits 列表 (list of lists) 转换为单一的风险分数数组。
    
    对于离散时间模型，一个常见的风险分数是所有时间区间风险概率的总和。
    
    Args:
        logits_list (list): 每个元素是模型对一个样本输出的 logits 列表 (e.g., shape [N, 10])

    Returns:
        np.ndarray: 每个样本的单一风险分数 (e.g., shape [N,])
    """
    # 1. 转换为
    logits_tensor = torch.tensor(logits_list)
    
    # 2. 通过 sigmoid 转换为概率 (P(event in interval_i | alive at i))
    probabilities = torch.sigmoid(logits_tensor)
    
    # 3. 将所有区间的概率相加，作为一个总的风险分数
    # 分数越高，代表在整个时间段内发生事件的累积风险越高
    risk_scores = torch.sum(probabilities, dim=1)
    
    return risk_scores.numpy()


def plot_risk_stratified_km(
    risk_scores: np.ndarray,
    original_durations: list,
    original_events_observed: list,
    output_path: str
):
    """
    根据模型预测的风险分数对患者进行高/低风险分层，
    并绘制两条Kaplan-Meier生存曲线，同时计算Log-Rank检验的p-value。

    Args:
        risk_scores (np.ndarray): 每个患者的单一风险分数。
        original_durations (list): 每个患者的 *真实* 持续时间 (天数/月数)。
        original_events_observed (list): 每个患者的 *真实* 事件状态 (1=发生事件, 0=删失)。
        output_path (str): 图像的完整保存路径 (e.g., ".../Kaplan_Meier_Plot.png")。
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError as exc:
        raise ImportError("Kaplan-Meier plotting requires the optional lifelines package.") from exc

    print("  [KM Plotter] 开始生成KM图...")
    
    # --- 1. 数据准备 ---
    durations = np.array(original_durations)
    events = np.array(original_events_observed)
    
    # 检查数据
    if len(risk_scores) != len(durations) or len(risk_scores) != len(events):
        print(f"  [KM Plotter] 错误: 数据长度不匹配。Scores: {len(risk_scores)}, Durations: {len(durations)}, Events: {len(events)}")
        return

    # --- 2. 风险分层 ---
    # 使用风险分数的中位数作为阈值
    median_risk_score = np.median(risk_scores)
    
    # 分为高风险组和低风险组
    low_risk_mask = risk_scores < median_risk_score
    high_risk_mask = risk_scores >= median_risk_score

    # 分离两组的持续时间和事件
    low_risk_durations = durations[low_risk_mask]
    low_risk_events = events[low_risk_mask]
    
    high_risk_durations = durations[high_risk_mask]
    high_risk_events = events[high_risk_mask]
    
    print(f"  [KM Plotter] 分组完成: 低风险组 {len(low_risk_durations)} 人, 高风险组 {len(high_risk_durations)} 人。")

    # --- 3. 统计检验 (Log-Rank Test) ---
    # 比较两条曲线是否存在显著差异
    results = logrank_test(low_risk_durations, high_risk_durations, 
                           event_observed_A=low_risk_events, event_observed_B=high_risk_events)
    p_value = results.p_value
    
    print(f"  [KM Plotter] Log-Rank Test P-Value: {p_value:.4e}")

    # --- 4. 绘图 ---
    plt.figure(figsize=(10, 7))
    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    # 绘制低风险组
    kmf_low.fit(low_risk_durations, event_observed=low_risk_events, label=f"Low Risk (n={len(low_risk_durations)})")
    kmf_low.plot(ax=plt.gca(), color='green', ci_show=True, ci_alpha=0.2)

    # 绘制高风险组
    kmf_high.fit(high_risk_durations, event_observed=high_risk_events, label=f"High Risk (n={len(high_risk_durations)})")
    kmf_high.plot(ax=plt.gca(), color='red', ci_show=True, ci_alpha=0.2)
    
    # --- 5. 格式化图表 ---
    plt.title('Kaplan-Meier Survival Analysis (Test Set Stratified by Model Prediction)')
    plt.xlabel('Time (Days)') # 假设你的 'event_time' 是天数
    plt.ylabel('Survival Probability')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 在图上标注 P-value
    # 使用 .4e 来处理非常小的值 (例如 1.40e-07)
    plt.text(0.05, 0.05, f'Log-Rank p-value: {p_value:.4e}', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))
    
    plt.legend(loc='upper right')
    
    # --- 6. 保存 ---
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() # 关闭图像，释放内存
        print(f"  [KM Plotter] KM图已保存到: {output_path}")
    except Exception as e:
        print(f"  [KM Plotter] 错误: 保存KM图失败. {e}")
