import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Dict, Tuple, Any
from modules.base_modules.aggregation_utils import masked_mean_pool

def batch_js_divergence(prob_list: List[torch.Tensor], epsilon: float = 1e-10) -> torch.Tensor:
    """
    为批次数据计算广义JS散度 (Generalized JS Divergence)，为每个样本返回一个分数。
    这是列表中所有概率分布到它们平均分布的平均KL散度。
    
    Args:
        prob_list (List[torch.Tensor]): 概率分布的列表。
            列表中的每个张量形状都为 (B, num_classes)。
            (重要：这个列表应该只包含*存在*的模态的概率)
        epsilon (float): 防止log(0)的微小值。
    
    Returns:
        torch.Tensor: 每个样本的JS散度分数，形状 (B,)。
    """
    if not prob_list:
        # 列表为空，无法计算
        return torch.tensor(0.0) # 默认为 CPU tensor
        
    num_modalities = len(prob_list)
    if num_modalities < 2:
        # 只有一个分布，无散度
        return torch.tensor(0.0, device=prob_list[0].device).repeat(prob_list[0].shape[0])
        
    # 1. 堆叠并计算平均分布 M
    # (B, num_classes, M)
    prob_stack = torch.stack(prob_list, dim=2)
    # (B, num_classes)
    m = prob_stack.mean(dim=2)
    
    total_kl = 0.0
    
    # 预先计算 m + epsilon 以提高效率
    m_plus_epsilon = m + epsilon
    
    for p in prob_list:
        # p: (B, num_classes)
        # m: (B, num_classes)
        # D_KL(P || M)
        # sum over classes (dim=1)
        kl_p_m = torch.sum(p * (torch.log(p + epsilon) - torch.log(m_plus_epsilon)), dim=1)
        total_kl += kl_p_m
        
    # 2. 平均 KL 散度
    # (B,)
    jsd = total_kl / num_modalities
    return jsd

# --- 核心模块 ---

class SharedTokenWiseGatedNetwork(nn.Module):
    """
    一个轻量化的、上下文感知的门控模块。
    *** 它为所有模态共享同一个门控MLP。***
    
    门控的上下文由两部分组成：
    1. context: 所有*存在*模态的池化特征的平均值。
    2. conflict: 所有*存在*模态的预测概率之间的广义JS散度。
    """
    def __init__(self, feature_dim: int, context_dim: int, mlp_hidden_dim: int):
        super().__init__()
        
        # 上下文维度 = 特征维度 (来自 'context') + 1 (来自 'conflict')
        self.global_context_dim = feature_dim + 1
        self.context_projector = nn.Linear(self.global_context_dim, context_dim)

        gate_mlp_input_dim = feature_dim + context_dim
        
        # 需求3：所有模态共享一个轻量化的门控MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(mlp_hidden_dim),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, 
                all_tokens: List[torch.Tensor], 
                all_masks: List[torch.Tensor],
                context_vector: torch.Tensor,
                conflict_score: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            all_tokens (List[torch.Tensor]): 长度 M, 元素 (B, N_i, D)
            all_masks (List[torch.Tensor]): 长度 M, 元素 (B, N_i) (True=有效)
            context_vector (torch.Tensor): (B, D), 存在模态的平均池化特征
            conflict_score (torch.Tensor): (B,), 存在模态间的JSD
        Returns:
            gated_tokens_list (List[torch.Tensor]): 长度 M, 元素 (B, N_i, D)
        """
        
        # 1. 创建全局上下文
        # (B, D + 1)
        global_context_raw = torch.cat([context_vector, conflict_score.unsqueeze(1)], dim=1)
        # (B, context_dim)
        global_context = self.context_projector(global_context_raw)

        gated_tokens_list = []
        for i in range(len(all_tokens)):
            tokens_i = all_tokens[i]
            mask_i = all_masks[i]
            B, N_i, D = tokens_i.shape
            
            # 2. 广播上下文
            # (B, N_i, context_dim)
            context_expanded = global_context.unsqueeze(1).repeat(1, N_i, 1)
            
            # (B, N_i, D + context_dim)
            gate_input = torch.cat([tokens_i, context_expanded], dim=2)
            
            # 3. 计算门控 (使用共享的MLP)
            # (B, N_i, 1)
            gates = self.gate_mlp(gate_input)
            
            # 4. 应用门控
            # (B, N_i, D)
            gated_tokens = tokens_i * torch.sigmoid(gates)
            
            # 5. (重要) 确保被掩码的token保持为0
            gated_tokens = gated_tokens * mask_i.unsqueeze(-1).float()
            
            gated_tokens_list.append(gated_tokens)

        return gated_tokens_list


class KLGatedFusion(nn.Module):
    """
    重构的 KLGatedFusion 模块：
    1. 支持任意数量 (M) 的模态。
    2. **不使用 VAE**。融合仅基于存在的、经过门控的token。
    3. Transformer 的 padding_mask 负责处理缺失的模态。
    4. 使用一个共享的、基于冲突的门控网络 (`SharedTokenWiseGatedNetwork`)。
    5. 最终输出是**整个**融合序列的掩码平均池化 (无 `concat_fusion`)。
    """
    def __init__(self, 
                 embed_dim: int, 
                 max_modalities: int, 
                 layers_num: int = 2, 
                 attn_heads: int = 8, 
                 context_dim: int = 128, 
                 mlp_hidden_dim: int = 256) -> None:
        super().__init__()
        
        self.layers_num = layers_num
        self.attn_heads = attn_heads
        self.out_dim = embed_dim
        self.num_modalities = max_modalities
        
        assert self.out_dim % self.attn_heads == 0, f"out_dim ({self.out_dim}) must be a multiple of attn_heads ({self.attn_heads})"
        
        # 迭代融合的 Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.out_dim,
            nhead=self.attn_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(self.layers_num)])

        # 共享的、基于冲突的门控网络
        self.gated_network = SharedTokenWiseGatedNetwork(
            feature_dim=self.out_dim,
            context_dim=context_dim,
            mlp_hidden_dim=mlp_hidden_dim
        )
        
        # 需求2：不需要 self.concat_fusion
        
        print(f"Masked KLGatedFusion Block Initialized with {self.layers_num} layers, {self.num_modalities} modalities, and Shared Conflict-Gating.")


    def forward(self, 
                embeddings: List[torch.Tensor], 
                masks: List[torch.Tensor], 
                task_head: nn.Module, 
                batch: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Args:
            embeddings (List[torch.Tensor]): 长度 M, 元素 (B, N_i, D)
            masks (List[torch.Tensor]): 长度 M, 元素 (B, N_i) (布尔, True=有效)
            task_head (nn.Module): 用于深度监督的任务头
            batch (Optional[Dict[str, Any]]): 包含标签的批次数据
        """
        
        assert len(embeddings) == self.num_modalities, "embeddings 列表的长度必须等于 num_modalities"
        assert len(masks) == self.num_modalities, "masks 列表的长度必须等于 num_modalities"
        
        B = embeddings[0].shape[0]
        device = embeddings[0].device
        is_training = batch is not None
        loss_dict = {}

        # --- 步骤 0: 准备静态掩码和长度 ---
        
        # all_masks 在整个前向传播过程中保持不变
        all_masks = masks
        token_lengths = [t.shape[1] for t in embeddings]
        
        # (B, M) - 标记哪些患者*真正*拥有哪些模态
        patient_presence_list = [masked_mean_pool(t, m)[1] for t, m in zip(embeddings, all_masks)]
        patient_presence_mask = torch.stack(patient_presence_list, dim=1)
        
        # (B, M, 1) - 用于广播
        presence_mask_float_expanded = patient_presence_mask.float().unsqueeze(-1)
        # (B, 1) - 每个患者存在的模态数
        num_present_per_patient = presence_mask_float_expanded.sum(dim=1) + 1e-6
        # (B,) - 用于JSD损失的掩码，只在有多种模态时计算
        jsd_loss_mask = (patient_presence_mask.float().sum(dim=1) > 1).float()
        
        # current_tokens_list 将在循环中被更新
        current_tokens_list = list(embeddings)
        
        deep_supervision_losses = []
        jsd_losses = []
        
        fused_sequence = None
        fused_padding_mask = None # (True 代表 填充/遮盖)

        # --- 步骤 1: 迭代融合 ---
        for layer_idx, layer in enumerate(self.layers):
            
            # 1.1 池化以获取上下文、冲突和深度监督
            all_pooled_current = []
            all_logits = []
            all_probs = []
            
            for j in range(self.num_modalities):
                # (B, D)
                pooled_j, _ = masked_mean_pool(current_tokens_list[j], all_masks[j])
                all_pooled_current.append(pooled_j)
                
                # (B,)
                patient_mask_j = patient_presence_mask[:, j] # 使用原始的存在掩码
                
                if is_training:
                    supervision_output = task_head.decode(pooled_j, patient_mask_j, batch)
                    logits_j = supervision_output['logits']
                    deep_supervision_losses.append(supervision_output['loss'])
                else:
                    logits_j = task_head.prediction_head(pooled_j)

                all_logits.append(logits_j)
                all_probs.append(F.sigmoid(logits_j))

            # 1.2 计算 "Context" 和 "Conflict"
            
            # Context (B, D): 存在模态的平均池化特征
            # (B, M, D)
            pooled_stack = torch.stack(all_pooled_current, dim=1)
            # (B, D)
            context_vector = (pooled_stack * presence_mask_float_expanded).sum(dim=1) / num_present_per_patient
            
            # Conflict (B,): 存在模态间的JSD
            # (B, num_classes)
            # 找出每个患者的*有效*概率
            probs_for_jsd = []
            for b in range(B):
                present_probs = [all_probs[m][b] for m in range(self.num_modalities) if patient_presence_mask[b, m]]
                
                if len(present_probs) < 2:
                    probs_for_jsd.append(torch.tensor(0.0, device=device))
                else:
                    # (num_present, num_classes)
                    present_probs_stacked = torch.stack(present_probs, dim=0).unsqueeze(0) # 模拟 (1, num_present, num_classes)
                    # 将它们转换回列表以匹配 `batch_js_divergence` 的输入
                    present_probs_list = [p.squeeze(0) for p in torch.split(present_probs_stacked, 1, dim=1)]
                    
                    # (1,) -> 标量
                    jsd_b = batch_js_divergence(present_probs_list).squeeze()
                    probs_for_jsd.append(jsd_b)
            
            # (B,)
            jsd_per_patient = torch.stack(probs_for_jsd)

            if is_training:
                # JSD 损失只对有多模态的患者计算
                jsd_loss = (jsd_per_patient * jsd_loss_mask).sum() / (jsd_loss_mask.sum() + 1e-6)
                jsd_losses.append(jsd_loss)
            
            # (B,) - 分离梯度，JSD仅作为门控的*值*输入
            conflict_score_input = jsd_per_patient.detach()
            
            # 1.3 门控
            gated_tokens_list = self.gated_network(
                current_tokens_list, 
                all_masks,
                context_vector,
                conflict_score_input
            )
            
            # 1.4 拼接序列
            fused_sequence = torch.cat(gated_tokens_list, dim=1)
            
            # 1.5 创建Transformer填充掩码 (True=忽略)
            # 缺失模态的掩码 (all_masks[i]) 是全False,
            # ~m 之后会变成全True，Transformer会完全忽略它们。
            fused_padding_mask = torch.cat([~m for m in all_masks], dim=1)

            # 1.6 通过Transformer层进行深度融合
            fused_sequence = layer(fused_sequence, src_key_padding_mask=fused_padding_mask)

            # 1.7 更新token序列，为下一次迭代做准备
            current_tokens_list = list(torch.split(fused_sequence, token_lengths, dim=1))

        # --- 步骤 2: 最终池化 (无 concat_fusion) ---
        
        # 需求2：对*整个*融合后的序列进行掩码池化
        # fused_padding_mask: (B, N_total), True=填充
        # 我们需要 (B, N_total), True=有效
        fused_mask = ~fused_padding_mask
        
        fused_embedding, _ = masked_mean_pool(fused_sequence, fused_mask)

        # --- 步骤 3: 汇总损失 ---
        if is_training:
            if deep_supervision_losses:
                # 平均所有层和所有模态的深度监督损失
                loss_dict["deep_supervision_loss"] = torch.mean(torch.stack(deep_supervision_losses))
            if jsd_losses:
                # 平均所有层的JSD损失
                loss_dict["jsd_consensus_loss"] = torch.mean(torch.stack(jsd_losses))
            
            total_loss = 0
            for k, v in loss_dict.items():
                total_loss += v
            loss_dict["total_loss"] = total_loss

        return {
            "fused_embedding": fused_embedding,
            "loss_dict": loss_dict
        }



if __name__ == "__main__":
    # --- 模拟测试 ---
    
    # 1. 定义模拟的任务头
    class MockTaskHead(nn.Module):
        def __init__(self, embed_dim, num_classes):
            super().__init__()
            self.prediction_head = nn.Linear(embed_dim, num_classes)
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none') # 保持每个样本的损失
            print(f"MockTaskHead initialized with embed_dim={embed_dim}, num_classes={num_classes}")

        def decode(self, pooled_features, patient_mask, batch):
            """
            模拟的 decode 函数，用于深度监督。
            pooled_features: (B, D)
            patient_mask: (B,) 布尔张量, True=有效
            batch: 模拟的批次数据
            """
            logits = self.prediction_head(pooled_features)
            
            # (B, C)
            labels = batch["labels"]
            
            # (B, C)
            loss_all = self.loss_fn(logits, labels.float())
            
            # (B,)
            loss_per_patient = loss_all.mean(dim=1)
            
            # 只对*存在*的样本计算损失
            # (B,) * (B,) -> (B,) -> 标量
            mask_float = patient_mask.float()
            loss = (loss_per_patient * mask_float).sum() / (mask_float.sum() + 1e-6)
            
            return {"logits": logits, "loss": loss}

    # 2. 设置超参数
    B = 4
    D = 128 # embed_dim
    C = 5   # num_classes
    M = 3   # num_modalities
    
    # 模态的Token长度
    N_list = [10, 20, 15] 

    # 3. 创建模型
    model = KLGatedFusion(
        embed_dim=D,
        num_modalities=M,
        layers_num=2,
        attn_heads=4,
        context_dim=64,
        mlp_hidden_dim=128
    )
    
    task_head = MockTaskHead(D, C)
    
    # 4. 创建模拟输入数据
    embeddings_list = []
    masks_list = []
    
    # 模态 0: 所有人都有
    embeddings_list.append(torch.randn(B, N_list[0], D))
    masks_list.append(torch.ones(B, N_list[0]).bool())
    
    # 模态 1: 患者 1 缺失
    emb1 = torch.randn(B, N_list[1], D)
    mask1 = torch.ones(B, N_list[1]).bool()
    emb1[1, :, :] = 0.0 # 模拟缺失数据的输入 (全零)
    mask1[1, :] = False # 模拟缺失数据的掩码 (全False)
    embeddings_list.append(emb1)
    masks_list.append(mask1)
    
    # 模态 2: 患者 2 缺失
    emb2 = torch.randn(B, N_list[2], D)
    mask2 = torch.ones(B, N_list[2]).bool()
    emb2[2, :, :] = 0.0
    mask2[2, :] = False
    embeddings_list.append(emb2)
    masks_list.append(mask2)

    print(f"\n--- 准备测试数据 ---")
    print(f"Batch Size: {B}, Embed Dim: {D}, Num Classes: {C}, Num Modalities: {M}")
    print(f"Token 长度 (N): {N_list}")
    print(f"模态 1 缺失: 患者 1")
    print(f"模态 2 缺失: 患者 2")
    
    # 5. 模拟训练模式 (有 batch)
    mock_batch = {
        "labels": torch.randint(0, 2, (B, C))
    }
    
    print("\n--- 1. 测试训练模式 (is_training=True) ---")
    model.train()
    output_train = model(embeddings_list, masks_list, task_head, mock_batch)
    
    print(f"输出 'fused_embedding' 形状: {output_train['fused_embedding'].shape}")
    print(f"输出 'loss_dict':")
    for k, v in output_train['loss_dict'].items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")
            
    # 6. 模拟推理模式 (batch=None)
    print("\n--- 2. 测试推理模式 (is_training=False) ---")
    model.eval()
    with torch.no_grad():
        output_eval = model(embeddings_list, masks_list, task_head, None)

    print(f"输出 'fused_embedding' 形状: {output_eval['fused_embedding'].shape}")
    print(f"输出 'loss_dict':")
    for k, v in output_eval['loss_dict'].items():
        if isinstance(v, torch.Tensor):
             print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")

    # 7. 检查输出形状是否正确
    assert output_train['fused_embedding'].shape == (B, D)
    assert output_eval['fused_embedding'].shape == (B, D)
    print("\n--- 测试通过！---")