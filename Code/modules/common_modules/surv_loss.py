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
    
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        """Compute Cox PH loss.
        
        Args:
            log_h: Log hazard ratios, shape (n_samples,)
            durations: Survival times, shape (n_samples,)
            events: Event indicators (1 for event, 0 for censored), shape (n_samples,)
            eps: Small value for numerical stability
            
        Returns:
            Loss tensor with specified reduction
        """
        return cox_ph_loss_with_reduction(log_h, durations, events, self.reduction, eps)


def cox_ph_loss_with_reduction(log_h: Tensor, durations: Tensor, events: Tensor, 
                              reduction: str = 'mean', eps: float = 1e-7) -> Tensor:
    """Cox PH loss function with reduction support.
    
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    """
    # Sort by descending duration
    idx = durations.sort(descending=True)[1]
    events_sorted = events[idx]
    log_h_sorted = log_h[idx]
    
    return cox_ph_loss_sorted_with_reduction(log_h_sorted, events_sorted, reduction, eps)


def cox_ph_loss_sorted_with_reduction(log_h: Tensor, events: Tensor, 
                                     reduction: str = 'mean', eps: float = 1e-7) -> Tensor:
    """Cox PH loss for sorted data with reduction support.
    
    Requires the input to be sorted by descending duration time.
    """
    if events.dtype is torch.bool:
        events = events.float()
    
    events = events.view(-1)
    log_h = log_h.view(-1)
    
    # Calculate log cumulative sum with numerical stability
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    
    # Calculate negative log partial likelihood for each sample
    log_partial_likelihood = log_h.sub(log_cumsum_h)
    losses = -log_partial_likelihood.mul(events)
    
    # Apply reduction
    if reduction == 'none':
        return losses
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'mean':
        n_events = events.sum()
        if n_events == 0:
            return torch.tensor(0.0, device=losses.device)
        return losses.sum() / n_events
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


# Alternative implementation as a standalone function
def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, 
                reduction: str = 'mean', eps: float = 1e-7) -> Tensor:
    """Cox PH loss function with reduction support.
    
    This is a standalone function version that can be used without class instantiation.
    """
    return cox_ph_loss_with_reduction(log_h, durations, events, reduction, eps)


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    n_samples = 100
    log_h = torch.randn(n_samples, requires_grad=True)
    durations = torch.rand(n_samples) * 365  # 生存时间（天）
    events = torch.bernoulli(torch.ones(n_samples) * 0.7)  # 70% 事件发生
    
    # 使用不同的reduction方式
    loss_none = CoxPHLoss(reduction='none')(log_h, durations, events)
    loss_mean = CoxPHLoss(reduction='mean')(log_h, durations, events)
    loss_sum = CoxPHLoss(reduction='sum')(log_h, durations, events)
    
    print(f"Loss shape (none): {loss_none.shape}")
    print(f"Loss (mean): {loss_mean.item():.4f}")
    print(f"Loss (sum): {loss_sum.item():.4f}")
    print(f"Sum of individual losses: {loss_none.sum().item():.4f}")
    print(f"Number of events: {events.sum().item()}")
    
    # 验证数学关系
    print(f"Mean loss should equal sum / n_events: {loss_none.sum().item() / events.sum().item():.4f}")