import torch
import torch.nn as nn
from torch import Tensor


class CustomCoxPHLoss(nn.Module):
    """Cox Proportional Hazards loss with reduction support.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of events
            'sum': the output will be summed
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction
    
    def forward(
        self,
        log_h: Tensor,      # 预测的 log hazard (相当于你原来的 theta)
        durations: Tensor,  # 生存时间 T
        events: Tensor,     # 事件指示 E (1=event, 0=censor)
        eps: float = 1e-7
    ) -> Tensor:
        return cox_ph_loss_matrix(log_h, durations, events, self.reduction, eps)


def mean_by_event(losses, events):
    assert losses.dim() == 1, f"Expected losses have 1 dim, but got {losses.dim()}"
    return torch.mean(losses)

def cox_ph_loss_matrix(
    log_h: Tensor,
    durations: Tensor,
    events: Tensor,
    reduction: str = 'mean',
    eps: float = 1e-7
) -> Tensor:
    """
    Cox PH loss using explicit risk-set matrix (不排序的矩阵写法).

    对每个样本 i：
        R_i = { j | T_j >= T_i }
        loss_i = - d_i * ( log h_i - log sum_{j in R_i} h_j )
    其中 h = exp(log_h)，d_i 是事件指示 (events)。
    """
    # 展平成一维
    log_h = log_h.view(-1)
    durations = durations.view(-1)
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)

    device = log_h.device
    dtype = log_h.dtype

    # 构造风险集矩阵 R: R[i, j] = 1{ T_j >= T_i }
    # durations.unsqueeze(0): [1, n] -> T_j
    # durations.unsqueeze(1): [n, 1] -> T_i
    # 比较得到 [n, n]，第 (i, j) 元素 = (T_j >= T_i)
    risk_matrix = (durations.unsqueeze(0) >= durations.unsqueeze(1)).to(dtype).to(device)

    # h = exp(log_h)
    hazards = log_h.exp()  # [n]

    # 对每个 i，计算 Σ_{j∈R_i} h_j
    # risk_matrix: [n, n], hazards: [n] -> risk_sum: [n]
    risk_sum = risk_matrix.matmul(hazards)  # [n]

    # log Σ_{j∈R_i} h_j
    log_risk_sum = (risk_sum + eps).log()

    # log 部分似然：log h_i - log Σ_{j∈R_i} h_j
    log_partial_likelihood = log_h - log_risk_sum

    # 负对数部分似然，仅在有事件的样本上起作用
    losses = -log_partial_likelihood * events  # [n]

    # reduction
    if reduction == 'none':
        return losses
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'mean':
        n_events = events.sum()
        if n_events == 0:
            # 和你之前的实现类似，没事件时就直接返回总和（一般应接近 0）
            return losses.sum()
        return losses.sum() / n_events
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


# 如果你喜欢函数式接口，也可以这样用：
def cox_ph_loss(
    log_h: Tensor,
    durations: Tensor,
    events: Tensor,
    reduction: str = 'mean',
    eps: float = 1e-7
) -> Tensor:
    return cox_ph_loss_matrix(log_h, durations, events, reduction, eps)


# import torch
# import torch.nn as nn
# from torch import Tensor
# import pycox

# class CustomCoxPHLoss(nn.Module):
#     """Cox Proportional Hazards loss with reduction support.
    
#     Args:
#         reduction (str): Specifies the reduction to apply to the output:
#             'none': no reduction will be applied
#             'mean': the sum of the output will be divided by the number of events
#             'sum': the output will be summed
#     """
    
#     def __init__(self, reduction: str = 'mean'):
#         super().__init__()
#         assert reduction in ['none', 'mean', 'sum']
#         self.reduction = reduction
    
#     def forward(self, log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
#         """Compute Cox PH loss.
        
#         Args:
#             log_h: Log hazard ratios, shape (n_samples,)
#             durations: Survival times, shape (n_samples,)
#             events: Event indicators (1 for event, 0 for censored), shape (n_samples,)
#             eps: Small value for numerical stability
            
#         Returns:
#             Loss tensor with specified reduction
#         """
#         return cox_ph_loss_with_reduction(log_h, durations, events, self.reduction, eps)


# def mean_by_event(losses, events):
#     assert losses.dim() == 1, f"Expected losses have 1 dim, but got {losses.dim()}"
#     return torch.mean(losses)
#     # n_events = events.sum()
#     # if n_events == 0:
#     #     sum = losses.sum()
#     #     assert sum.item() < 0.1, f"Expected sum of loss is zero, but got {sum}"
#     #     return sum
#     # return losses.sum() / n_events


# def cox_ph_loss_with_reduction(log_h: Tensor, durations: Tensor, events: Tensor, 
#                               reduction: str = 'mean', eps: float = 1e-7) -> Tensor:
#     """Cox PH loss function with reduction support.
    
#     We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
#     where h = exp(log_h) are the hazards and R is the risk set, and d is event.
#     """
#     # Sort by descending duration
#     idx = durations.sort(descending=True)[1]
#     events_sorted = events[idx]
#     log_h_sorted = log_h[idx]
    
#     return cox_ph_loss_sorted_with_reduction(log_h_sorted, events_sorted, reduction, eps)


# def cox_ph_loss_sorted_with_reduction(log_h: Tensor, events: Tensor, 
#                                      reduction: str = 'mean', eps: float = 1e-7) -> Tensor:
#     """Cox PH loss for sorted data with reduction support.
    
#     Requires the input to be sorted by descending duration time.
#     """
#     if events.dtype is torch.bool:
#         events = events.float()
    
#     events = events.view(-1)
#     log_h = log_h.view(-1)
    
#     # Calculate log cumulative sum with numerical stability
#     gamma = log_h.max()
#     # log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
#     log_cumsum_h = log_h.exp().cumsum(0).add(eps).log()
    
#     # Calculate negative log partial likelihood for each sample
#     log_partial_likelihood = log_h.sub(log_cumsum_h)
#     losses = -log_partial_likelihood.mul(events)
    
#     # Apply reduction
#     if reduction == 'none':
#         return losses
#     elif reduction == 'sum':
#         return losses.sum()
#     elif reduction == 'mean':
#         n_events = events.sum()
#         if n_events == 0:
#             return losses.sum()
#         return losses.sum() / n_events
#     else:
#         raise ValueError(f"Invalid reduction: {reduction}")


# # Alternative implementation as a standalone function
# def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, 
#                 reduction: str = 'mean', eps: float = 1e-7) -> Tensor:
#     """Cox PH loss function with reduction support.
    
#     This is a standalone function version that can be used without class instantiation.
#     """
#     return cox_ph_loss_with_reduction(log_h, durations, events, reduction, eps)

