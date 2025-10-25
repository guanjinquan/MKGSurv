import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Dict, Tuple, Any

# --- 辅助模块和函数 ---

def masked_mean_pool(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    对一个序列张量进行带掩码的平均池化。
    Args:
        tensor (torch.Tensor): 输入张量，形状为 (B, N, D)。
        mask (torch.Tensor): 掩码张量，形状为 (B, N)，1代表有效token，0代表填充token。
    Returns:
        torch.Tensor: 池化后的张量，形状为 (B, D)。
    """
    if mask is None:
        # 如果没有提供掩码，则执行标准平均池化
        return torch.mean(tensor, dim=1)
    
    # 检查掩码是否全为0（例如，在某些罕见情况下）
    # 注意：在单个患者级别，我们将在下面处理
    if not torch.any(mask):
        # 如果整个批次的掩码都为0，返回一个全零张量，形状为 (B, D)
        return torch.zeros(tensor.shape[0], tensor.shape[2], device=tensor.device, dtype=tensor.dtype)
    
    mask = mask.unsqueeze(-1).float()  # (B, N, 1)
    masked_tensor = tensor * mask
    sum_feats = torch.sum(masked_tensor, dim=1) # (B, D)
    num_valid_tokens = torch.sum(mask, dim=1) # (B, 1)
    
    # 防止除以零（关键！）。
    # 如果一个患者的 num_valid_tokens 为 0，clamp会将其变为 1e-9。
    # 对应的 sum_feats 也为 0，所以 0 / 1e-9 = 0。
    # 这就实现了“缺失模态 = 零向量”
    num_valid_tokens = num_valid_tokens.clamp(min=1e-9)
    
    return sum_feats / num_valid_tokens

def batch_js_divergence(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    为批次数据计算JS散度，为每个样本返回一个分数。
    Args:
        p (torch.Tensor): 第一个概率分布，形状为 (B, num_classes)。
        q (torch.Tensor): 第二个概率分布，形状为 (B, num_classes)。
        epsilon (float): 防止log(0)的微小值。
    Returns:
        torch.Tensor: 每个样本的JS散度分数，形状 (B,)。
    """
    m = 0.5 * (p + q)
    
    # 手动计算KL散度以获得每个样本的分数
    kl_p_m = torch.sum(p * (torch.log(p + epsilon) - torch.log(m + epsilon)), dim=1)
    kl_q_m = torch.sum(q * (torch.log(q + epsilon) - torch.log(m + epsilon)), dim=1)
    
    jsd = 0.5 * (kl_p_m + kl_q_m)
    return jsd

# --- 核心模块 ---

class TokenWiseGatedNetwork(nn.Module):
    """
    一个上下文感知的、为每个Token生成门控的模块。
    适用于不同序列长度的图像和文本特征。
    
    *** 已更新 ***
    现在可以正确处理患者级别的模态缺失（通过掩码）。
    """
    def __init__(self, feature_dim: int, context_dim: int, mlp_hidden_dim: int):
        super().__init__()
        # 两个模态的池化特征 + 1个冲突分数
        self.global_context_dim = feature_dim * 2 + 1 
        self.context_projector = nn.Linear(self.global_context_dim, context_dim)

        gate_mlp_input_dim = feature_dim + context_dim
        self.image_gate_mlp = nn.Sequential(
            nn.Linear(gate_mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )
        self.text_gate_mlp = nn.Sequential(
            nn.Linear(gate_mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, 
                image_tokens: torch.Tensor, 
                text_tokens: torch.Tensor, 
                image_mask: Optional[torch.Tensor], 
                text_mask: Optional[torch.Tensor], 
                conflict_score: torch.Tensor):
        """
        *** 已更新 ***
        添加了 image_mask 和 text_mask 作为输入。
        """
        B, N1, D = image_tokens.shape
        _, N2, _ = text_tokens.shape
        
        # --- 修改点：使用 masked_mean_pool ---
        # 如果一个患者的 image_mask 全为0，image_pooled[i] 将是一个全零向量
        image_pooled = masked_mean_pool(image_tokens, image_mask)
        # 如果一个患者的 text_mask 全为0，text_pooled[i] 将是一个全零向量
        text_pooled = masked_mean_pool(text_tokens, text_mask)
        # -------------------------------------
        
        if conflict_score.dim() == 1:
            conflict_score = conflict_score.unsqueeze(1)
        
        # global_context_raw 会接收到“零向量”作为缺失信号
        global_context_raw = torch.cat([image_pooled, text_pooled, conflict_score], dim=1)
        # context_projector (MLP) 将学会解释这个信号
        global_context = self.context_projector(global_context_raw)

        # 门控机制现在会感知到模态缺失
        context_expanded_for_image = global_context.unsqueeze(1).repeat(1, N1, 1)
        image_gate_input = torch.cat([image_tokens, context_expanded_for_image], dim=2)
        image_gates = self.image_gate_mlp(image_gate_input)
        
        context_expanded_for_text = global_context.unsqueeze(1).repeat(1, N2, 1)
        text_gate_input = torch.cat([text_tokens, context_expanded_for_text], dim=2)
        text_gates = self.text_gate_mlp(text_gate_input)

        # 对于缺失的模态（例如图像），image_tokens 本身就是填充（0）
        # 所以 gated_image_tokens 也会是 0，这没问题
        gated_image_tokens = image_tokens * torch.sigmoid(image_gates)
        # 对于存在的模态（例如文本），门控会基于“图像缺失”这一上下文来调整token
        gated_text_tokens = text_tokens * torch.sigmoid(text_gates)

        return gated_image_tokens, gated_text_tokens


class KLGatedFusion(nn.Module):
    def __init__(self, embed_dim: int, max_modalities: int, layers_num: int = 2, attn_heads: int = 8, context_dim: int = 128, mlp_hidden_dim: int = 256) -> None:
        super().__init__()
        
        self.layers_num = layers_num
        self.attn_heads = attn_heads
        self.in_dim = embed_dim
        self.out_dim = embed_dim
        self.max_modalities_num = max_modalities
        
        # 注意：这里的 in_dim 似乎没有在Transformer层中使用，
        # 而是假定TokenWiseGatedNetwork的feature_dim (out_dim) 与Transformer的d_model (out_dim) 匹配
        # 这是一个潜在的假设，但基于您的代码是合理的。
        assert self.out_dim % self.attn_heads == 0, f"out_dim ({self.out_dim}) must be a multiple of attn_heads ({self.attn_heads})"
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.out_dim,
            nhead=self.attn_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(self.layers_num)])

        self.gated_network = TokenWiseGatedNetwork(
            feature_dim=self.out_dim,  # 假设输入token的维度已经是 out_dim
            context_dim=context_dim,  
            mlp_hidden_dim=mlp_hidden_dim
        )

        # num_modalities 似乎在__init__中未定义，我假设它为 2
        # 如果您打算支持2个以上的模态，这里的concat_fusion也需要修改
        self.num_modalities = 2 
        self.concat_fusion = nn.Sequential(
            nn.Linear(self.out_dim * self.num_modalities, self.out_dim), # 使用 self.out_dim
            nn.LayerNorm(self.out_dim),  
            nn.ReLU(),
        )
        
        print(f"KLGatedFusion Block Initialized with {self.layers_num} layers and Deep Supervision. Params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")


    def forward(self, embeddings: List[Optional[torch.Tensor]], masks: List[Optional[torch.Tensor]], task_head: nn.Module, batch: Optional[Dict[str, Any]] = None) -> Dict:
        """
        具有迭代式冲突门控和(可选的)深度监督的前向传播函数。
        
        *** 假设 ***
        我们假设 embeddings 和 masks 列表包含[image, text, ...]，
        并且我们只使用前两个元素。
        我们还假设，如果 embeddings[0] 不是 None，那么 masks[0] 也不是 None。
        """
        
        # 1. 分离并预处理模态
        # 假设 embeddings[0] 是图像, embeddings[1] 是文本
        # 并且假设它们不是 None（这应该由 ModelInterface 保证）
        image_tokens, text_tokens = embeddings[0], embeddings[1]
        image_mask, text_mask = masks[0], masks[1]
        
        # 如果上游（ModelInterface）没有过滤掉None，我们在这里需要一个健壮的检查
        # 但基于您的问题，我们假设这里收到的 image_tokens 和 text_tokens 是张量
        # 并且 image_mask 和 text_mask 也是张量（可能是全False，但不是None）
        
        N1 = image_tokens.shape[1]
        N2 = text_tokens.shape[1]
        device = next(self.parameters()).device

        # 初始化用于收集深度监督损失的列表
        deep_supervision_losses = []
        js_list = []

        # 2. 迭代式融合、门控和深度监督
        for i, layer in enumerate(self.layers):
            # 2.1 基于当前token状态进行池化（使用掩码！）
            # 这里的池化是用于深度监督的
            image_pooled = masked_mean_pool(image_tokens, image_mask)
            text_pooled = masked_mean_pool(text_tokens, text_mask)
            
            # 2.2 (可选) 获取logits和深度监督损失
            if batch is not None:
                # 训练模式：使用decode获取logits和loss
                # patient_mask_image (B,)：标记哪些患者*有*图像
                patient_mask_image = torch.any(image_mask, dim=1) if image_mask is not None else None
                supervision_output_image = task_head.decode(image_pooled, patient_mask_image, batch)
                image_logits = supervision_output_image['logits']
                deep_supervision_losses.append(supervision_output_image['loss'])

                patient_mask_text = torch.any(text_mask, dim=1) if text_mask is not None else None
                supervision_output_text = task_head.decode(text_pooled, patient_mask_text, batch)
                text_logits = supervision_output_text['logits']
                deep_supervision_losses.append(supervision_output_text['loss'])
            else:
                # 推理模式：仅使用分类器获取logits
                image_logits = task_head.classifier(image_pooled)
                text_logits = task_head.classifier(text_pooled)

            # 2.3 计算冲突分数
            # 如果患者i没有图像，image_logits[i]会是0，image_probs[i]会是均匀分布
            image_probs = F.softmax(image_logits, dim=-1)
            text_probs = F.softmax(text_logits, dim=-1)
            # conflict_score 会正确地计算 JSD(均匀分布, 文本概率)
            conflict_score = batch_js_divergence(image_probs, text_probs).detach() # 计算冲突分数，不需要梯度
            js_list.append(conflict_score)

            # --- 修改点：将掩码传递给门控网络 ---
            gated_image_tokens, gated_text_tokens = self.gated_network(
                image_tokens, text_tokens, image_mask, text_mask, conflict_score
            )
            # -------------------------------------
            
            # 2.5 拼接序列以进行多模态融合
            fused_sequence = torch.cat([gated_image_tokens, gated_text_tokens], dim=1)
            
            # 创建Transformer需要的填充掩码（padding mask）
            # Transformer 期望 1/True 代表“被遮盖/忽略”
            if image_mask is not None and text_mask is not None:
                # ~image_mask (B, N1) -> True 代表填充
                # ~text_mask (B, N2) -> True 代表填充
                fused_padding_mask = torch.cat([~image_mask, ~text_mask], dim=1)
            else:
                fused_padding_mask = None

            # 2.6 通过Transformer层进行深度融合
            # src_key_padding_mask=True 的位置会被注意力机制忽略
            fused_sequence = layer(fused_sequence, src_key_padding_mask=fused_padding_mask)

            # 2.7 更新token序列，为下一次迭代做准备
            image_tokens, text_tokens = torch.split(fused_sequence, [N1, N2], dim=1)

        # 3. 最终融合
        # 使用掩码池化来获取最终的患者级别特征
        mask_image_feature = masked_mean_pool(image_tokens, image_mask)
        mask_text_feature = masked_mean_pool(text_tokens, text_mask)

        fused_features = self.concat_fusion(torch.cat([mask_image_feature, mask_text_feature], dim=1))

        loss_dict = {}
        # print("Deep Supervision Losses:", deep_supervision_losses)
        if len(deep_supervision_losses) > 0 and batch is not None: # 仅在训练时计算
            loss_dict["total_loss"] = torch.mean(torch.stack(deep_supervision_losses))
        
        if len(js_list) > 0:
            for i, js in enumerate(js_list):
                loss_dict[f"js_divergence_{i}"] = torch.mean(js)
            print("JSD:", js_list)


        return {
            "fused_embedding": fused_features,
            "loss_dict": loss_dict
        }
