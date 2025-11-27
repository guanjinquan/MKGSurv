import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple



# --- Masked Mean Pooling Function ---
def masked_mean_pool(embedding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    带详细 Debug 逻辑的 Masked Mean Pooling。
    
    Args:
        embedding (torch.Tensor): (B, N, D)
        mask (Optional[torch.Tensor]): (B, N), 1为有效, 0为padding
        
    Returns:
        pooled_embedding: (B, D)
        valid_batch_mask: (B,) bool, 标记该样本是否有效（至少有一个token）
    """
    # --- Debug 1: 检查输入是否含有 NaN ---
    if torch.isnan(embedding).any():
        raise ValueError("Error: Input 'embedding' contains NaN values before pooling!")
    
    # 无 Mask 情况：直接求平均
    if mask is None:
        return embedding.mean(dim=1), torch.ones(embedding.shape[0], device=embedding.device, dtype=torch.bool)

    # 确保 mask 是 float 类型用于计算，且维度对齐
    # mask: (B, N)
    if mask.dim() != 2:
        raise ValueError(f"Error: Mask should be 2D (B, N), but got {mask.shape}")
        
    # --- Debug 2: 检查 Mask 是否全为 0 (Empty Sequence) ---
    # 计算每个样本有多少个有效 token
    seq_lengths = mask.sum(dim=1)  # (B,)
    
    # 找出全为 0 的样本索引（即该样本全是 Padding）
    empty_indices = (seq_lengths == 0).nonzero(as_tuple=False).squeeze()
    
    if empty_indices.numel() > 0:
        # 严重警告：如果存在全空的样本，除法会出问题，或者业务逻辑有误
        print(f"\n[DEBUG WARNING] Found {empty_indices.numel()} empty sequences in batch!")
        print(f"  Empty indices: {empty_indices}")
        print(f"  Embedding shape: {embedding.shape}")
        raise ValueError(f"Found empty sequences at indices {empty_indices}. Cannot pool all-padding sequences.")

    # 扩展 mask 维度以匹配 embedding: (B, N) -> (B, N, 1)
    mask_expanded = mask.float().unsqueeze(-1)

    # Masked Sum
    # sum_embeddings: (B, D)
    sum_embeddings = (embedding * mask_expanded).sum(dim=1)

    # --- Debug 3: 检查求和后是否产生 NaN/Inf (可能是数值溢出) ---
    if torch.isnan(sum_embeddings).any() or torch.isinf(sum_embeddings).any():
        raise ValueError("Error: NaN or Inf detected after summing embeddings.")

    # Safe Division
    # 将分母中的 0 替换为 1e-9 或 1，防止除零错误 (NaN)
    # clamp_min=1e-9 保证分母不为 0
    safe_lengths = torch.clamp(seq_lengths.unsqueeze(-1), min=1e-9)
    
    pooled_embedding = sum_embeddings / safe_lengths
    
    # --- Debug 4: 最终检查 ---
    if torch.isnan(pooled_embedding).any():
        # 打印详细现场以供调试
        nan_indices = torch.isnan(pooled_embedding).any(dim=1).nonzero().squeeze()
        print(f"\n[CRITICAL ERROR] NaN produced in output!")
        print(f"  Indices with NaN: {nan_indices}")
        print(f"  Corresponding seq_lengths: {seq_lengths[nan_indices]}")
        raise ValueError("Output contains NaN after division.")

    # 生成 valid_batch_mask: 只要序列长度 > 0，就是有效的样本
    valid_batch_mask = (seq_lengths > 0).bool()

    # 处理那些原本全空的样本：
    # 虽然计算结果是 0 (0 / 1e-9 = 0)，但为了安全，显式地将它们置为 0
    if (~valid_batch_mask).any():
        pooled_embedding[~valid_batch_mask] = 0.0

    return pooled_embedding, valid_batch_mask


# --- Weighted Aggregation Head ---
class AggregationHead(nn.Module):
    """
    An aggregation module that pools token-level embeddings into a single vector
    using a simple attention mechanism. It now also returns a mask indicating
    which samples in the batch had valid tokens.
    """
    def __init__(self, embed_dim: int):
        """
        Initializes the AggregationHead.
        Args:
            embed_dim (int): The dimensionality of the token embeddings.
        """
        super().__init__()
        # A simple attention network to compute a score for each token
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, token_embeddings: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_embeddings (torch.Tensor): Token-level embeddings of shape (batch_size, num_tokens, embed_dim).
            masks (torch.Tensor): A mask tensor of shape (batch_size, num_tokens) where 1 indicates a valid token and 0 an invalid one.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - aggregated_embedding (torch.Tensor): Aggregated embedding of shape (batch_size, embed_dim).
                - pooled_mask (torch.Tensor): A boolean mask of shape (batch_size,) indicating which samples were valid.
        """
        # Determine which samples in the batch are valid (have at least one unmasked token)
        # pooled_mask shape: (batch_size,)

        device = token_embeddings.device
        pooled_mask = torch.any(masks.bool(), dim=1).to(device)

        # Compute attention scores for each token
        # attention_scores shape: (batch_size, num_tokens, 1)
        attention_scores = self.attention_net(token_embeddings)
        
        # Apply the mask to the attention scores before softmax.
        # Set the scores of masked-out tokens to a very large negative number
        # so their softmax probability becomes negligible.
        if masks is not None:
            attention_scores = attention_scores.masked_fill(masks.unsqueeze(-1) == 0, -1e9)
        
        # Convert scores to weights using softmax along the token dimension
        # attention_weights shape: (batch_size, num_tokens, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Explicitly set the embeddings of invalid tokens to zero before the weighted sum.
        masked_embeddings = token_embeddings * masks.unsqueeze(-1)
        
        # Compute the weighted sum of token embeddings
        # (B, N, D) * (B, N, 1) -> sum over N -> (B, D)
        weighted_sum = torch.sum(masked_embeddings * attention_weights, dim=1)
        
        # Ensure that embeddings for fully masked samples are zero
        weighted_sum = weighted_sum * pooled_mask.unsqueeze(-1)
        
        return weighted_sum, pooled_mask


if __name__ == '__main__':
    # This block serves as a test case for the AggregationHead module.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running AggregationHead Test on device: {device} ---")

    # 1. Define parameters
    batch_size = 4
    num_tokens = 10
    embed_dim = 128

    # 2. Instantiate the module
    aggregation_head = AggregationHead(embed_dim=embed_dim).to(device)

    # 3. Create dummy data
    # Dummy token embeddings
    dummy_embeddings = torch.randn(batch_size, num_tokens, embed_dim).to(device)
    
    # Dummy mask: one sample will have all tokens masked out
    dummy_mask = torch.ones(batch_size, num_tokens, device=device)
    dummy_mask[0, -2:] = 0  # First item has 2 padded tokens
    dummy_mask[1, :] = 0      # Second item has all tokens masked out (missing modality)
    dummy_mask[2, -1:] = 0  # Third item has 1 padded token
    # Fourth item has no padded tokens

    print("\nDummy Mask:")
    print(dummy_mask.int())

    # 4. Perform a forward pass
    print("\nPerforming forward pass...")
    aggregated_embedding, pooled_mask = aggregation_head(dummy_embeddings, dummy_mask)

    # 5. Verify the output
    print(f"\nShape of aggregated embedding: {aggregated_embedding.shape}")
    print(f"Expected shape: ({batch_size}, {embed_dim})")
    assert aggregated_embedding.shape == (batch_size, embed_dim)

    print(f"\nGenerated Pooled Mask: {pooled_mask.int()}")
    expected_pooled_mask = torch.tensor([1, 0, 1, 1], device=device)
    print(f"Expected Pooled Mask:  {expected_pooled_mask.int()}")
    assert torch.equal(pooled_mask.int(), expected_pooled_mask.int())
    
    # Check if the embedding for the fully masked sample is zero
    assert torch.all(aggregated_embedding[1] == 0)
    print("\nEmbedding for the fully masked patient is all zeros.")

    print("\n--- Test PASSED ---")
