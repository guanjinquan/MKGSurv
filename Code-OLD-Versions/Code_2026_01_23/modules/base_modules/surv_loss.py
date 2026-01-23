import torch
import torch.nn as nn
from torch import Tensor

class CustomCoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards loss with reduction support.
    
    Uses the Explicit Matrix method (Risk Set Matrix) for the Cox calculation.
    This implementation is completely order-invariant and requires NO sorting.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction

    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor, 
                sample_weights: Tensor = None, eps: float = 1e-7) -> Tensor:
        """
        Compute Cox PH loss with Matrix implementation.
        """
        return cox_ph_loss_matrix(log_h, durations, events, 
                                  self.reduction, eps, sample_weights)


def mean_by_event(losses: Tensor, events: Tensor) -> Tensor:
    """
    Helper to calculate mean loss only over observed events.
    """
    n_events = events.sum()
    if n_events == 0:
        return losses.sum()
    return losses.sum() / n_events


def cox_ph_loss_matrix(
    log_h: Tensor,
    durations: Tensor,
    events: Tensor,
    reduction: str = 'mean',
    eps: float = 1e-7,
    sample_weights: Tensor = None
) -> Tensor:
    """
    Cox PH loss using explicit risk-set matrix (Naive implementation).
    
    For every sample i, the risk set R_i = { j | T_j >= T_i }.
    loss_i = - d_i * ( log h_i - log sum_{j in R_i} h_j )
    """
    # 1. Flatten inputs
    log_h = log_h.view(-1)
    durations = durations.view(-1)
    events = events.view(-1)
    
    if sample_weights is not None:
        sample_weights = sample_weights.view(-1)

    device = log_h.device
    dtype = log_h.dtype

    # 2. Construct Risk Matrix R
    # R[i, j] = 1 if T_j >= T_i (j is in risk set of i)
    # shape: [N, 1] compared to [1, N] -> [N, N]
    risk_matrix = (durations.unsqueeze(1) <= durations.unsqueeze(0)).to(dtype)
    
    # 3. Compute Hazards
    # h = exp(log_h)
    hazards = log_h.exp()  # [n]

    # 4. Sum hazards in risk set for each i
    # [n, n] @ [n] -> [n]
    # risk_sum[i] = sum_{j} R[i, j] * h[j]
    risk_sum = risk_matrix.matmul(hazards)

    # 5. Log of risk sums
    # log sum_{j in R_i} h_j
    log_risk_sum = (risk_sum + eps).log()

    # 6. Log partial likelihood
    # log h_i - log sum_{j in R_i} h_j
    log_partial_likelihood = log_h - log_risk_sum

    # 7. Negative Log Likelihood, masked by events
    losses = -log_partial_likelihood * events  # [n]

    # Apply sample weights if provided
    if sample_weights is not None:
        losses = losses * sample_weights

    # 8. Reduction
    if reduction == 'none':
        return losses
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'mean':
        return mean_by_event(losses, events)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

# Functional wrapper
def cox_ph_loss(
    log_h: Tensor,
    durations: Tensor,
    events: Tensor,
    sample_weights: Tensor = None,
    reduction: str = 'mean',
    eps: float = 1e-7,
    **kwargs
) -> Tensor:
    return cox_ph_loss_matrix(log_h, durations, events, reduction, eps, sample_weights)