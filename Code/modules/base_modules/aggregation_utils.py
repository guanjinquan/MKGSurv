import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict



# --- Masked Mean Pooling Function ---
def masked_mean_pool(embedding: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Performs masked mean pooling on a sequence.
    This is used to aggregate variable-length sequences into a single feature vector.

    Args:
        embedding (torch.Tensor): Input sequence of shape (B, N, D).
        mask (Optional[torch.Tensor]): Boolean or binary mask of shape (B, N), with 1s for valid tokens.

    Returns:
        torch.Tensor: Pooled embedding of shape (B, D).
    """
    if mask is None:
        return embedding.mean(dim=1)
    
    pooled_mask = torch.any(mask.bool(), dim=1)  # Shape: (B,)
    
    # Ensure mask is float for multiplication and summation
    mask = mask.float().unsqueeze(-1)
    
    # Masked pooling
    summed = (embedding * mask).sum(dim=1)
    # Count non-zero elements in the mask for each batch item to get the sequence length
    count = mask.sum(dim=1)
    # Avoid division by zero for empty sequences
    count = torch.max(count, torch.ones_like(count))
    
    return summed / count, pooled_mask



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
