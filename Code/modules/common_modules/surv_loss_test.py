import torch
import torch.nn as nn
from torch import Tensor
import pycox

class CustomCoxPHLoss(nn.Module):
    """Cox Proportional Hazards loss with reduction support and 
    augmentation regularization.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of events
            'sum': the output will be summed
        reg_weight (float): Weight for the L2 regularization term.
                            Set to 0 to disable.
        target_delta (float): The target value for (log_h_aug - log_h_orig).
                              Represents how much higher the augmented risk
                              should be.
        reg_eps (float): Small value for numerical stability (if needed,
                         though not used in target L2 loss).
    """
    
    def __init__(self, reduction: str = 'mean', reg_weight: float = 1.0, 
                 target_delta: float = 1.0, reg_eps: float = 1e-7):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        self.reduction = reduction
        self.reg_weight = reg_weight
        self.target_delta = target_delta
        self.reg_eps = reg_eps # Kept for compatibility, but not used by L2
    
    def _compute_target_l2_reg(self, log_h: Tensor, durations: Tensor, events: Tensor, is_augmented: Tensor) -> Tensor:
        """
        Computes the Target L2 regularization term for aug/orig pairs.
        
        Finds pairs (i, j) where:
        - durations[i] == durations[j]
        - events[i] == events[j]
        - is_augmented[i] == True, is_augmented[j] == False
        
        And applies a loss:
        Loss = (delta - target_delta)^2
        where delta = log_h_i - log_h_j
        """
        
        # We need the same sorting as in cox_ph_loss_with_reduction
        # to find adjacent pairs.
        # Sort by 1. durations (desc), 2. events (asc), 3. is_augmented (asc)
        
        if is_augmented.dtype is torch.bool:
            is_augmented_float = is_augmented.float()
        else:
            is_augmented_float = is_augmented

        idx1 = is_augmented_float.argsort(descending=False, stable=True)
        idx2 = events[idx1].argsort(descending=False, stable=True)
        idx3 = durations[idx1[idx2]].argsort(descending=True, stable=True)
        idx = idx1[idx2[idx3]]

        # Get sorted tensors
        dur_s = durations[idx]
        evt_s = events[idx]
        aug_s = is_augmented[idx] # This will be bool
        log_h_s = log_h[idx]

        # Find adjacent pairs
        # Check where current sample (i) and next sample (i+1) match
        mask_dur = (dur_s[:-1] == dur_s[1:])
        mask_evt = (evt_s[:-1] == evt_s[1:])
        
        # Find where i is 'orig' (False) and i+1 is 'aug' (True)
        mask_aug_pair = (aug_s[:-1] == False) & (aug_s[1:] == True)
        
        # Final mask for pairs
        pair_mask = mask_dur & mask_evt & mask_aug_pair
        
        if pair_mask.sum() == 0:
            # No pairs found, no regularization loss
            return torch.tensor(0.0, device=log_h.device)

        # Get log_h for the pairs
        # log_h_s[:-1] is the array up to the second-to-last
        # log_h_s[1:] is the array from the second element
        orig_log_h = log_h_s[:-1][pair_mask] # This is log_h_j (is_augmented=False)
        aug_log_h = log_h_s[1:][pair_mask]   # This is log_h_i (is_augmented=True)
        
        # Calculate loss terms
        delta = aug_log_h - orig_log_h
        
        # Target L2 Loss: (delta - target_delta)^2
        pair_losses = (delta - self.target_delta).pow(2)
        
        # Return the mean loss over all found pairs
        return pair_losses.mean()

    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor, is_augmented: Tensor = None, eps: float = 1e-7) -> Tensor:
        """Compute Cox PH loss with optional regularization.
        
        Args:
            log_h: Log hazard ratios, shape (n_samples,)
            durations: Survival times, shape (n_samples,)
            events: Event indicators (1 for event, 0 for censored), shape (n_samples,)
            is_augmented: (Optional) Indicator for augmented data, shape (n_samples,)
            eps: Small value for numerical stability in Cox loss
            
        Returns:
            Loss tensor with specified reduction
        """
        
        # 1. Compute standard Cox PH loss
        cox_loss = cox_ph_loss_with_reduction(log_h, durations, events, self.reduction, is_augmented, eps)
        
        # 2. Compute regularization loss
        if self.reg_weight > 0 and is_augmented is not None:
            # Call the new function
            reg_loss = self._compute_target_l2_reg(log_h, durations, events, is_augmented)
            return cox_loss + self.reg_weight * reg_loss
        
        return cox_loss
    

def mean_by_event(losses, events):
    assert losses.dim() == 1, f"Expected losses have 1 dim, but got {losses.dim()}"

    n_events = events.sum()
    if n_events == 0:
        sum = losses.sum()
        assert sum.item() < 0.1, f"Expected sum of loss is zero, but got {sum}"
        return sum
    return losses.sum() / n_events


def cox_ph_loss_with_reduction(log_h: Tensor, durations: Tensor, events: Tensor, 
                              reduction: str = 'mean', is_augmented: Tensor = None, eps: float = 1e-7) -> Tensor:
    """Cox PH loss function with reduction support.
    
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    """
    
    if is_augmented is None:
        # Original behavior: Sort by descending duration only
        idx = durations.sort(descending=True)[1]
    else:
        # Sort by 1. durations (desc), 2. events (asc), 3. is_augmented (asc)
        # We use stable sort by applying keys in reverse order of priority.
        
        # Ensure is_augmented is in a sortable format (e.g., float or long)
        if is_augmented.dtype is torch.bool:
            is_augmented = is_augmented.float()

        # 1. Get indices from sorting by is_augmented (ascending)         # 时间相同，事件相同，默认增强的数据风险至少不低于原本的不增强数据
        # (Key 3: 'is_augmented' ascending - not augmented comes first)  
        idx1 = is_augmented.argsort(descending=False, stable=True)
        
        # 2. Get indices from sorting events[idx1] (ascending)            # 时间相同，发生事件的风险要高一些
        # (Key 2: 'events' ascending - event=0 comes first)
        idx2 = events[idx1].argsort(descending=False, stable=True)
        
        # 3. Get indices from sorting durations[idx1[idx2]] (descending)  # 时间短的风险高一些
        # (Key 1: 'durations' descending - longer duration comes first)
        idx3 = durations[idx1[idx2]].argsort(descending=True, stable=True)
        
        # The final index is the composition of these sorted indices
        idx = idx1[idx2[idx3]]

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
            return losses.sum()
        return losses.sum() / n_events
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


# Alternative implementation as a standalone function
def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, 
                reduction: str = 'mean', eps: float = 1e-7) -> Tensor:
    """Cox PH loss function with reduction support.
    
    This is a standalone function version that can be used without class instantiation.
    """
    # We must pass is_augmented=None here, or update the signature
    return cox_ph_loss_with_reduction(log_h, durations, events, reduction, is_augmented=None, eps=eps)