import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class CustomCoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards loss with reduction support and 
    augmentation regularization (Mixup logic).
    
    Uses the Explicit Matrix method (Risk Set Matrix) for the Cox calculation.
    This implementation is completely order-invariant and requires NO sorting.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of events
            'sum': the output will be summed
        reg_weight (float): Weight for the Regularization term.
                            Set to 0 to disable.
        target_delta (float): (Not used in the current exp loss logic, kept for API compatibility)
                              The parameter intended for margin definitions.
    """
    
    def __init__(self, reduction: str = 'mean', reg_weight: float = 0.5):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction
        self.reg_weight = reg_weight
    
    def _compute_margin_reg(self, log_h: Tensor, durations: Tensor, events: Tensor, is_augmented: Tensor) -> Tensor:
        """
        Computes an Exponential Regularization Loss.
        
        Enforces: log_h_i (Original) < log_h_j (Augmented)
        
        Finds ALL pairs (i, j) such that:
        - durations[i] == durations[j] (Same time)
        - events[i] == events[j]       (Same event status)
        - is_augmented[i] == 0         (i is Original)
        - is_augmented[j] == 1         (j is Augmented)
        
        Loss = exp(log_h_i - log_h_j)
        
        If log_h_i < log_h_j (as desired), the exponent is negative, loss is small (< 1).
        If log_h_i > log_h_j (violation), the exponent is positive, loss grows exponentially.
        """
        # Ensure vectors are flat
        log_h = log_h.view(-1)
        durations = durations.view(-1)
        events = events.view(-1)
        is_augmented = is_augmented.view(-1)
        
        if is_augmented.dtype is torch.bool:
            is_augmented = is_augmented.long()
            
        # Expand for broadcasting: Rows (i=Orig) vs Cols (j=Aug)
        # We look for i (Original) -> j (Augmented) relationships
        dur_i = durations.unsqueeze(1)   # [n, 1]
        dur_j = durations.unsqueeze(0)   # [1, n]
        
        evt_i = events.unsqueeze(1)      # [n, 1]
        evt_j = events.unsqueeze(0)      # [1, n]
        
        aug_i = is_augmented.unsqueeze(1) # [n, 1]
        aug_j = is_augmented.unsqueeze(0) # [1, n]
        
        # 1. Check Duration Match
        mask_dur = (dur_i == dur_j)
        
        # 2. Check Event Match
        mask_evt = (evt_i == evt_j)
        
        # 3. Check Augmentation: i must be Original (0), j must be Augmented (1)
        mask_aug_pair = (aug_i == 0) & (aug_j == 1)
        
        # Combine masks
        pair_mask = mask_dur & mask_evt & mask_aug_pair
        
        if pair_mask.sum() == 0:
            return torch.tensor(0.0, device=log_h.device)
        
        # Get log_h for valid pairs
        # Broadcast log_h to matrix:
        log_h_i = log_h.unsqueeze(1).exp()  # [n, 1] -> Rows (Original)
        log_h_j = log_h.unsqueeze(0).exp()  # [1, n] -> Cols (Augmented)
        
        # Calculate delta matrix: log_h[i] - log_h[j]
        # We want log_h_i < log_h_j, so we want this difference to be negative.
        delta_matrix = log_h_i - log_h_j 
        
        # Select only valid pairs
        valid_deltas = delta_matrix[pair_mask]
        
        # Exponential Loss
        # Minimizing exp(Orig - Aug) pushes Orig to be much smaller than Aug.
        violations = valid_deltas.exp()
        
        return violations.mean()

    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor, 
                is_augmented: Tensor = None, eps: float = 1e-7) -> Tensor:
        """
        Compute Cox PH loss with Matrix implementation + Margin Regularization.
        NO SORTING REQUIRED.
        """
        
        # 1. Compute Matrix Cox Loss (Order invariant)
        cox_loss = cox_ph_loss_matrix(log_h, durations, events, 
                                      self.reduction, eps, is_augmented)
        
        # 2. Compute Regularization (Matrix based, Order invariant)
        reg_loss = 0.0
        if self.reg_weight > 0 and is_augmented is not None and is_augmented.any():
            reg_loss = self._compute_margin_reg(
                log_h, durations, events, is_augmented
            )
            print("Reg Loss : ", reg_loss, "Augment Sum: ", is_augmented.sum())
            return cox_loss + self.reg_weight * reg_loss
        
        return cox_loss


def mean_by_event(losses, events):
    assert losses.dim() == 1, f"Expected losses have 1 dim, but got {losses.dim()}"
    # return torch.mean(losses)
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
    is_augmented: Tensor = None
) -> Tensor:
    """
    Cox PH loss using explicit risk-set matrix.
    
    For every sample i, the risk set R_i = { j | T_j >= T_i }.
    loss_i = - d_i * ( log h_i - log sum_{j in R_i} h_j )
    
    Enhanced Logic for Mixup/Augmentation:
    If is_augmented is provided, we assume 'Augmented' data (is_augmented=1) 
    has HIGHER RISK (effectively slightly shorter survival) than 'Original' data 
    (is_augmented=0) when durations are tied.
    
    Risk Set Condition (j in R_i?):
    - T_j > T_i: True
    - T_j < T_i: False
    - T_j == T_i: True only if Aug_j <= Aug_i
      (Meaning: Original(0) is in risk set of Augmented(1), but Augmented(1) 
       is NOT in risk set of Original(0)).
    """
    # Flatten
    log_h = log_h.view(-1)
    durations = durations.view(-1)
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)

    device = log_h.device
    dtype = log_h.dtype

    # Construct Risk Matrix R: R[i, j] = 1 if j is in risk set of i
    if is_augmented is not None:
        # Ensure is_augmented is flat and float
        if is_augmented.dtype is torch.bool:
            is_augmented = is_augmented.long()
        is_augmented = is_augmented.view(-1)
        
        # Expand dimensions for broadcast comparison
        # Rows (i) vs Columns (j)
        t_i = durations.unsqueeze(1)    # [n, 1]
        t_j = durations.unsqueeze(0)    # [1, n]
        aug_i = is_augmented.unsqueeze(1) # [n, 1]
        aug_j = is_augmented.unsqueeze(0) # [1, n]
        
        # Condition 1: Strict time inequality
        # If T_j > T_i, j is definitely in risk set of i
        cond_time_gt = t_j > t_i
        
        # Condition 2: Time equality with Tie-Breaking
        # If T_j == T_i, we check augmentation status
        # We want 'Original' (0) to be considered "longer lived" than 'Augmented' (1).
        # So j is in risk set of i if Aug_j <= Aug_i.
        cond_time_eq = t_j == t_i
        cond_aug_le = aug_j <= aug_i
        
        # Combine: (T_j > T_i) OR (T_j == T_i AND Aug_j <= Aug_i)
        risk_matrix = cond_time_gt | (cond_time_eq & cond_aug_le)
        
        risk_matrix = risk_matrix.to(dtype).to(device)
        
    else:
        # Standard definition: R[i, j] = 1 if T_j >= T_i
        risk_matrix = (durations.unsqueeze(0) >= durations.unsqueeze(1)).to(dtype).to(device)

    # h = exp(log_h)
    hazards = log_h.exp()  # [n]

    # Sum hazards in risk set for each i
    # [n, n] @ [n] -> [n]
    risk_sum = risk_matrix.matmul(hazards)

    # log sum_{j in R_i} h_j
    log_risk_sum = (risk_sum + eps).log()

    # log partial likelihood: log h_i - log sum h_j
    log_partial_likelihood = log_h - log_risk_sum

    # Negative Log Likelihood, masked by events
    losses = -log_partial_likelihood * events  # [n]

    # Reduction
    if reduction == 'none':
        return losses
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'mean':
        return mean_by_event(losses, events)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

# Functional interface
def cox_ph_loss(
    log_h: Tensor,
    durations: Tensor,
    events: Tensor,
    is_augmented: Tensor = None,
    reduction: str = 'mean',
    eps: float = 1e-7,
    reg_weight: float = 0.0,
    target_delta: float = 1.0
) -> Tensor:
    """
    Functional wrapper for the CustomCoxPHLoss logic.
    """
    # We instantiate the class to reuse the logic cleanly
    loss_fn = CustomCoxPHLoss(reduction=reduction, reg_weight=reg_weight, target_delta=target_delta)
    return loss_fn(log_h, durations, events, is_augmented, eps)