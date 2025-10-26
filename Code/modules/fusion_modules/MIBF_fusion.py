import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

# --- Helper Modules from your provided code & paper ---

class InformationBalancedFusionAttention(nn.Module):
    """
    This is the IBFA module from the paper, which your code
    implemented as MultiHeadCrossAttention_v2.
    
    It takes Q from modality X and K,V from [X, Y].
    """
    def __init__(self, dim, num_heads):
        super(InformationBalancedFusionAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # Linear layers for x (Query)
        self.toQ_x = nn.Linear(dim, dim)
        
        # Linear layers for x (Key, Value)
        self.toK_x = nn.Linear(dim, dim)
        self.toV_x = nn.Linear(dim, dim)

        # Linear layers for y (Key, Value)
        self.toK_y = nn.Linear(dim, dim)
        self.toV_y = nn.Linear(dim, dim)

        # Output linear layer
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, y):
        """
        x: Modality to use for Query, e.g., Image (B, seq_len_x, dim)
        y: Other modality, e.g., Text (B, seq_len_y, dim)
        
        Returns:
        output: Fused features with shape (B, seq_len_x, dim)
        """
        batch_size, seq_len_x, _ = x.shape
        _, seq_len_y, _ = y.shape

        # 1. Generate Q from x
        Qx = self.toQ_x(x) # (B, seq_len_x, dim)

        # 2. Generate K, V from x
        Kx = self.toK_x(x)
        Vx = self.toV_x(x)

        # 3. Generate K, V from y
        Ky = self.toK_y(y)
        Vy = self.toV_y(y)

        # 4. Split Qx into heads
        Qx = Qx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, num_heads, seq_len_x, head_dim)

        # 5. Split Kx, Vx, Ky, Vy into heads
        Kx = Kx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        Vx = Vx.view(batch_size, seq_len_x, self.num_heads, self.head_dim).transpose(1, 2)
        Ky = Ky.view(batch_size, seq_len_y, self.num_heads, self.head_dim).transpose(1, 2)
        Vy = Vy.view(batch_size, seq_len_y, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 6. Concatenate K and V from both modalities
        # Kcat shape: (B, num_heads, seq_len_x + seq_len_y, head_dim)
        # Vcat shape: (B, num_heads, seq_len_x + seq_len_y, head_dim)
        Kcat = torch.cat([Kx, Ky], dim=2) 
        Vcat = torch.cat([Vx, Vy], dim=2)

        # 7. Perform attention
        # (B, num_heads, seq_len_x, head_dim) @ (B, num_heads, head_dim, seq_len_x + seq_len_y)
        # -> (B, num_heads, seq_len_x, seq_len_x + seq_len_y)
        attention_scores = torch.matmul(Qx, Kcat.transpose(-2, -1))
        attention_scores = attention_scores / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # (B, num_heads, seq_len_x, seq_len_x + seq_len_y) @ (B, num_heads, seq_len_x + seq_len_y, head_dim)
        # -> (B, num_heads, seq_len_x, head_dim)
        output = torch.matmul(attention_weights, Vcat)
        
        # 8. Combine heads
        # (B, seq_len_x, num_heads, head_dim) -> (B, seq_len_x, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_x, self.dim)
        output = self.to_out(output)

        return output

def batch_js_divergence(p: torch.Tensor, q: torch.Tensor, min_val: float = 1e-10) -> torch.Tensor:
    """
    Calculates Jensen-Shannon divergence between two batches of distributions.
    p, q: Tensors of shape (batch_size, num_classes)
    """
    # Clamp for numerical stability
    p = p.clamp(min=min_val)
    q = q.clamp(min=min_val)
    
    # Calculate M
    m = 0.5 * (p + q)
    m_log = m.log()
    
    # Calculate KL(P || M) and KL(Q || M)
    kl_p_m = F.kl_div(m_log, p, reduction='none').sum(dim=-1)
    kl_q_m = F.kl_div(m_log, q, reduction='none').sum(dim=-1)
    
    # JSD is the average of the two KL divergences
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd # (batch_size,)

def _masked_mean_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Performs masked mean pooling on a batch of tokens.
    tokens: (batch_size, seq_len, dim)
    mask: (batch_size, seq_len) with 1s for valid tokens, 0s for padding
    """
    if mask is None:
        return torch.mean(tokens, dim=1)
        
    mask_expanded = mask.unsqueeze(-1).float() # (B, seq_len, 1)
    summed_tokens = torch.sum(tokens * mask_expanded, dim=1) # (B, dim)
    
    # Count non-zero elements in the mask for each batch item
    valid_token_count = mask_expanded.sum(dim=1) # (B, 1)
    
    # Avoid division by zero for empty sequences
    valid_token_count = torch.clamp(valid_token_count, min=1e-9)
    
    pooled = summed_tokens / valid_token_count
    return pooled # (B, dim)


# --- Main Fusion Class ---
class MIBF_fusion(nn.Module):
    
    def __init__(self, embed_dim: int, max_modalities: int, attn_heads: int = 8) -> None:
        """
        Initializes the MIBF-Net fusion module.
        
        Args:
            embed_dim: The embedding dimension of the input tokens.
            max_modalities: (Not used by MIBF) Max number of modalities.
            layers_num: (Not used by MIBF) MIBF-Net has a single fusion step.
            attn_heads: Number of attention heads for the IBFA modules.
            context_dim: (Not used)
            mlp_hidden_dim: (Not used)
        """
        super().__init__()
        
        self.attn_heads = attn_heads
        self.max_modalities = max_modalities
        print("Modalities Number: ", self.max_modalities)
        self.embed_dim = embed_dim


        # As per MIBF-Net paper (Eq. 2 & 3), we need two IBFA modules.
        
        # 1. Image-as-Query (Fi2t in paper)
        # Q = Image, K/V = [Image, Text]
        self.image_to_text_attention = InformationBalancedFusionAttention(
            dim=embed_dim, 
            num_heads=attn_heads
        )
        
        # 2. Text-as-Query (Ft2i in paper)
        # Q = Text, K/V = [Text, Image]
        self.text_to_image_attention = InformationBalancedFusionAttention(
            dim=embed_dim, 
            num_heads=attn_heads
        )
        
        self.concat_fusion = nn.Sequential(
            nn.Linear(self.embed_dim * self.max_modalities, self.embed_dim), # Use self.embed_dim
            nn.LayerNorm(self.embed_dim),  
            nn.ReLU(),
        )

    def forward(self, embeddings: List[Optional[torch.Tensor]], masks: List[Optional[torch.Tensor]], task_head: nn.Module, batch: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Implements the MIBF-Net fusion and loss calculation logic.
        
        *** Assumptions ***
        embeddings[0] is image_tokens (B, seq_len_img, dim)
        embeddings[1] is text_tokens (B, seq_len_txt, dim)
        masks[0] is image_mask (B, seq_len_img)
        masks[1] is text_mask (B, seq_len_txt)
        """
        
        # 1. Separate modalities
        image_tokens, text_tokens = embeddings[0], embeddings[1]
        image_mask, text_mask = masks[0], masks[1]

        # Ensure tensors are on the correct device
        device = next(self.parameters()).device
        if image_tokens is not None:
            image_tokens = image_tokens.to(device)
        if text_tokens is not None:
            text_tokens = text_tokens.to(device)
        if image_mask is not None:
            image_mask = image_mask.to(device)
        if text_mask is not None:
            text_mask = text_mask.to(device)

        # Initialize outputs
        loss_dict = {}
        deep_supervision_losses = []
        js_list = []

        # --- MIBF-Net Step 1: IoP, ToP, and Conflict Score ---
        # We need pooled features for single-modality predictions
        image_pooled = _masked_mean_pool(image_tokens, image_mask)
        text_pooled = _masked_mean_pool(text_tokens, text_mask)

        # 2.2 (Optional) Get logits and deep supervision loss (IoP, ToP)
        if batch is not None:
            # Training mode: use decode to get logits and loss
            # This matches the (α * ||y - f_i||^2 + β * ||y - f_t||^2) part of Eq. 4
            
            # IoP
            patient_mask_image = torch.any(image_mask, dim=1) if image_mask is not None else torch.ones(image_tokens.shape[0], dtype=torch.bool, device=device)
            supervision_output_image = task_head.decode(image_pooled, patient_mask_image, batch)
            image_logits = supervision_output_image['logits']
            deep_supervision_losses.append(supervision_output_image['loss'])

            # ToP
            patient_mask_text = torch.any(text_mask, dim=1) if text_mask is not None else torch.ones(text_tokens.shape[0], dtype=torch.bool, device=device)
            supervision_output_text = task_head.decode(text_pooled, patient_mask_text, batch)
            text_logits = supervision_output_text['logits']
            deep_supervision_losses.append(supervision_output_text['loss'])
        else:
            # Inference mode: only use classifier to get logits
            image_logits = task_head.classifier(image_pooled)
            text_logits = task_head.classifier(text_pooled)

        # 2.3 Calculate conflict score (KL Divergence in paper, JSD in your stub)
        # This is the (KL) term in Eq. 4
        image_probs = F.softmax(image_logits, dim=-1)
        text_probs = F.softmax(text_logits, dim=-1)
        
        # Detach as this score is used as a *weight* for the main loss (in the main model)
        # or just for logging.
        conflict_score = batch_js_divergence(image_probs, text_probs).detach() 
        js_list.append(conflict_score)

        # --- MIBF-Net Step 2: Information Balanced Fusion ---
        # Use the *unpooled* tokens for fusion, as per the paper.
        
        # Fi2t = IBFA(Qi, K[i,t], V[i,t]) (Eq. 2)
        fused_image_features = self.image_to_text_attention(image_tokens, text_tokens)
        
        # Ft2i = IBFA(Qt, K[t,i], V[t,i]) (Eq. 3)
        fused_text_features = self.text_to_image_attention(text_tokens, image_tokens)
        
        # --- MIBF-Net Step 3: Pool & Concatenate for Final Prediction ---
        # Pool the outputs of the fusion modules
        fused_image_pooled = _masked_mean_pool(fused_image_features, image_mask)
        fused_text_pooled = _masked_mean_pool(fused_text_features, text_mask)
        
        # Concatenate to create the final multimodal feature vector
        # This is the [Fi2t, Ft2i] input to the final MLP in Fig. 3
        fused_embedding_concat = torch.cat([fused_image_pooled, fused_text_pooled], dim=1)
        fused_embedding = self.concat_fusion(fused_embedding_concat)

        # Get the conflict score tensor (B,) to pass as weights
        # Add a fallback to ones() for safety, though js_list should not be empty
        final_conflict_score = js_list[0].reshape(-1, 1) if len(js_list) > 0 else torch.ones((image_tokens.shape[0], 1), dtype=torch.float32, device=device)

        # 4. Prepare output dictionary
        if batch is not None and len(deep_supervision_losses) > 0:
            # Aggregate the IoP and ToP losses
            loss_dict["total_loss"] = torch.mean(torch.stack(deep_supervision_losses))

        if len(js_list) > 0:
            # Log the average JS divergence
            loss_dict[f"js_divergence"] = torch.mean(js_list[0])

        return {
            "fused_embedding": fused_embedding, # (B, embed_dim) after concat_fusion
            "loss_dict": loss_dict,
            "weights": final_conflict_score  # (B,) tensor with JS scores
        }