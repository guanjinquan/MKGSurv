import os
import sys
# 假设这个文件在 'modules/models' 目录下，调整路径以导入同级 'common_modules'
# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import torch
from torch import nn
import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel # <-- 在此代码中未被使用
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from nystrom_attention import NystromAttention


# ==========================================================================================
# TransMIL Components
# ==========================================================================================
class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        NystromAttention 期望 mask 形状为 [B, N],
        其中 True = 保持, False = 掩盖 (padding).
        如果 mask 为 None, 则不进行掩码。
        """
        x = x + self.attn(self.norm(x), mask=mask)
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, feat_token, H, W, mask: Optional[torch.Tensor] = None):
        """
        如果提供了掩码 (mask, [B, N_feat]),
        则在卷积前将 padding token 置零。
        如果 mask 为 None, 则不进行掩码。
        """
        B, N_feat, C = feat_token.shape # [B, H*W, C]

        if mask is not None:
            # 在卷积前将 padding token 置零
            # mask [B, H*W], feat_token [B, H*W, C]
            assert N_feat == mask.shape[1], \
                f"PPEG Mask 形状不匹配! feat_token 长度为 {N_feat}, mask 长度为 {mask.shape[1]}"
            
            # 应用掩码 (确保掩码可以广播)
            feat_token = feat_token * mask.unsqueeze(-1).float() # 乘法需要浮点型

        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        
        # PPEG 只返回处理过的特征 token
        return x


class AggregatingTransMIL(nn.Module):
    """
    一个通用的 MIL 聚合器，将 (B, N, input_dim) 聚合成 (B, num_aggregated_tokens, embed_dim)
    
    修改 (根据用户请求):
    - (B, N, D_in) -> (B, E, _H*_W)
    - 不再使用 0 填充 (pad0) 或 0 掩码 (masking out)。
    - 而是通过循环复制 (repeating) *有效*的 tokens 来填充到 _H*_W 的长度。
    - 注意力机制 (mask=None) 会处理所有 (有效的或复制的) tokens。
    """
    def __init__(self, input_dim=1024, embed_dim=512, num_aggregated_tokens: int = 16):
        super(AggregatingTransMIL, self).__init__()
        self.num_aggregated_tokens = num_aggregated_tokens
        
        self.pos_layer = PPEG(dim=embed_dim) 
        
        self._fc1 = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, self.num_aggregated_tokens, embed_dim)) # K 个 tokens
        self.layer1 = TransLayer(dim=embed_dim)
        self.layer2 = TransLayer(dim=embed_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, h, mask=None):
        """
        Args:
            h (torch.Tensor): 输入特征 [B, N, input_dim] (N 是 batch 中的最大长度)
            mask (torch.Tensor, optional): 布尔掩码 [B, N]. True=有效, False=padding.
        """
        B, N, _ = h.shape # N = N_max
        
        if mask is None:
            # 如果未提供掩码，则假设所有 token 都有效
            mask = torch.ones(B, N, device=h.device, dtype=torch.bool)
        
        # 确保掩码是布尔类型 (bool)。
        mask = mask.to(dtype=torch.bool)
            
        h = self.norm1(h)
        h = self._fc1(h)  # [B, N, embed_dim]
        E = h.shape[2]
        

        H_orig = h.shape[1] # N (N_max)
        _H, _W = int(np.ceil(np.sqrt(H_orig))), int(np.ceil(np.sqrt(H_orig)))
        target_len = _H * _W
        
        valid_counts = mask.sum(dim=1) # [B], 得到每个 item 的有效 token 数
        
        h_filled_list = []
        for i in range(B):
            valid_len = valid_counts[i].item()
            
            if valid_len == 0:
                # 边缘情况: 如果没有有效的 tokens, 则用 0 填充
                h_filled = torch.zeros(target_len, E, device=h.device, dtype=h.dtype)
            else:
                # 1. 获取有效的 tokens
                valid_tokens = h[i, :valid_len, :] # [valid_len, E]
                
                # 2. 计算需要重复多少次才能填满 target_len
                num_repeats = (target_len // valid_len) + 1
                
                # 3. 重复并裁剪
                repeated_tokens = valid_tokens.repeat(num_repeats, 1) # [num_repeats * valid_len, E]
                h_filled = repeated_tokens[:target_len, :] # [target_len, E]
            
            h_filled_list.append(h_filled)

        # 'h' 现在是特征张量, 完全由有效数据 (或其副本) 填充
        h = torch.stack(h_filled_list, dim=0) # [B, target_len, E]

        #---->cls_token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1) # [B, K + target_len, D]

        #---->Translayer x1
        h = self.layer1(h, mask=None)
        
        #---->PPEG
        # PPEG 只作用于特征 token, 所以我们先分离
        cls_token_out1, feat_token_out1 = h[:, :self.num_aggregated_tokens], h[:, self.num_aggregated_tokens:]
        
        # (修改) 移除掩码 (mask=None), PPEG 将处理所有 tokens
        # _H 和 _W 是基于 N_max (H_orig) 计算的, 保持不变
        feat_token_processed = self.pos_layer(feat_token_out1, _H, _W, mask=None) 
        
        # 重新组合 CLS token (未改变) 和处理过的特征 token
        h = torch.cat((cls_token_out1, feat_token_processed), dim=1)
        
        #---->Translayer x2
        # (修改) 移除掩码 (mask=None), Attention 将处理所有 tokens
        h = self.layer2(h, mask=None) 
        
        #---->Return K aggregated token embeddings
        h = self.norm2(h) 
        
        # 只返回 K 个 CLS token
        return h[:, 0:self.num_aggregated_tokens, :]


# ==========================================================================================
# 单元测试 (更新)
# ==========================================================================================

if __name__ == "__main__":
    
    # 1. 定义模型超参数
    in_dim_D = 1024  # D_in: 输入特征维度
    embed_dim_E = 512 # E: 内部嵌入维度
    n_tokens_K = 16   # K: 聚合后 token 的数量

    # 2. 定义测试参数 (模拟可变长度 batch)
    B_var = 4       # 批量大小
    n_max = 1400      # 整个 batch 的最大长度 (为了容纳 1300+)
    valid_lengths = [80, 1350, 1320, 1400] # 模拟的有效长度

    # 3. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 单元测试: AggregatingTransMIL (Loop Padding) ---")
    print(f"Using device: {device}")

    # 4. 创建模拟数据
    # 创建 (B, n_max, D_in) 的输入张量
    input_tensor_var = torch.rand(B_var, n_max, in_dim_D).to(device)
    
    # 5. 创建对应的 mask (布尔类型, True=valid, False=padding)
    mask_var_bool = torch.zeros(B_var, n_max, device=device, dtype=torch.bool)
    for i, length in enumerate(valid_lengths):
        mask_var_bool[i, :length] = True
    
    # 5b. 创建一个浮点型掩码来模拟
    mask_var_float = mask_var_bool.float()

    print(f"创建了一个 Batch (B={B_var}), 最大长度 n_max={n_max}")
    print(f"输入张量形状 (B, n, D_in): {input_tensor_var.shape}")
    print(f"Batch 中各项的有效 tokens 数量: {valid_lengths}")
    print("注意: 模型现在使用 '循环复制' 替换 0 填充, 并移除所有 Attention 掩码。")
    
    # 6. 实例化模型
    model = AggregatingTransMIL(
        input_dim=in_dim_D,
        embed_dim=embed_dim_E,
        num_aggregated_tokens=n_tokens_K
    ).to(device)

    # 7. 执行前向传播 (使用布尔掩码)
    print("\n正在执行前向传播 (使用 布尔型 掩码)...")
    output_bool = model(input_tensor_var, mask=mask_var_bool)
    
    expected_shape_var = (B_var, n_tokens_K, embed_dim_E)
    print(f"输出张量形状 (B, K, E): {output_bool.shape}")
    assert output_bool.shape == expected_shape_var
    print("[成功] 布尔型掩码测试成功!")

    # 8. 执行前向传播 (使用浮点型掩码)
    print("\n正在执行前向传播 (使用 浮点型 掩码)...")
    output_float = model(input_tensor_var, mask=mask_var_float)
    
    print(f"输出张量形状 (B, K, E): {output_float.shape}")
    assert output_float.shape == expected_shape_var
    print("[成功] 浮点型掩码测试成功!")
    
    # 9. 测试无掩码 (None) 的情况
    print("\n正在执行前向传播 (mask=None)...")
    output_none_mask = model(input_tensor_var, mask=None)
    
    assert output_none_mask.shape == expected_shape_var, \
        f"无掩码 (None) 测试失败! 预期: {expected_shape_var}, 得到: {output_none_mask.shape}"
        
    print("[成功] 无掩码 (None) 测试成功!")
    print(f"---------------------------------")