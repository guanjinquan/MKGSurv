import torch
import torch.nn as nn

def init_kaiming_norm(m):
    """
    对模块 'm' 应用权重初始化。
    
    - Kaiming Normal 用于 Linear 和 Conv2d 层 (假设激活函数为 ReLU)。
    - LayerNorm 权重初始化为 1，偏置初始化为 0。
    """
    
    # 检查 'm' 的具体类型
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # 对权重使用 Kaiming Normal 初始化
        # 'fan_in' 模式保留了前向传播中权重的方差
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
        # 如果存在偏置，则初始化为 0
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.LayerNorm):
        # LayerNorm 的权重 (gamma) 通常初始化为 1
        nn.init.constant_(m.weight, 1.0)
        # LayerNorm 的偏置 (beta) 通常初始化为 0
        nn.init.constant_(m.bias, 0.0)

# 假设 PPEG 和 TransLayer 已经定义
# from somewhere import PPEG, TransLayer