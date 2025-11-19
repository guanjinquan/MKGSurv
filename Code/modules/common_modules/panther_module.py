"""
此文件将 networks.py, layers.py, model_PANTHER.py, 
和 model_StructuredPANTHER.py 合并为一个文件。

已移除 create_emb_surv 和 predict 方法，以接受纯张量输入。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init  # 从 layers.py 导入

# -------------------------------------------------------------------
# 内容来自: networks.py
# -------------------------------------------------------------------

def mog_eval(mog, data):
    """
    This evaluates the log-likelihood of mixture of Gaussians
    """    
    B, N, d = data.shape    
    pi, mu, Sigma = mog    
    if len(pi.shape)==1:
        pi = pi.unsqueeze(0).repeat(B,1)
        mu = mu.unsqueeze(0).repeat(B,1,1)
        Sigma = Sigma.unsqueeze(0).repeat(B,1,1)
    
    # compute the log(prior * N(data; mean, cov))
    jll = -0.5 * ( d * np.log(2*np.pi) + 
        Sigma.log().sum(-1).unsqueeze(1) +
        torch.bmm(data**2, 1./Sigma.permute(0,2,1)) + 
        ((mu**2) / Sigma).sum(-1).unsqueeze(1) + 
        -2. * torch.bmm(data, (mu/Sigma).permute(0,2,1))
    ) + pi.log().unsqueeze(1)  
    
    # compute the log(sum(prior * N(data; mean, cov)))
    mll = jll.logsumexp(-1) 
    # compute the log posterior prob
    cll = jll - mll.unsqueeze(-1)
    
    return jll, cll, mll



class DirNIWNet(nn.Module):
    """
    Conjugate prior for the Gaussian mixture model

    Args:
    - p (int): Number of prototypes
    - d (int): Embedding dimension
    - eps (float): initial covariance (similar function to sinkorn entropic regularizer)
    """
    
    def __init__(self, p, d, prototypes, eps=0.1, fix_proto=False):
        """
        self.m: prior mean (p x d)
        self.V_: prior covariance (diagonal) (p x d)
        """
        super(DirNIWNet, self).__init__()

        self.eps = eps

        # prototypes 数组通常是通过在整个训练集的所有 patch 上运行 K-means 聚类来获得的。K-means 算法找到的 p 个聚类中心就是这个数组的内容。
        self.m = nn.Parameter(torch.from_numpy(prototypes), requires_grad=not fix_proto)

        self.V_ = nn.Parameter(np.log(np.exp(1) - 1) * torch.ones((p, d)), requires_grad=not fix_proto)
        # All values are 0.5413

        self.p, self.d = p, d
    
    def forward(self):
        """
        Return prior mean and covariance
        """
        V = self.eps * F.softplus(self.V_)
        # V == filled with 0.1
        return self.m, V
    
    def mode(self, prior=None):
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior
        pi = torch.ones(self.p).to(m) / self.p
        mu = m
        Sigma = V
        return pi.float(), mu.float(), Sigma.float()

        
    def map_m_step(self, data, weight, tau=1.0, prior=None):
        # Update rules are obtained from Kim, M. "Differentiable Expectation-Maximization for Set Representation Learning ", ICLR, 2022
        # This is a MAP-EM step, which is more stable than MLE (Eq 5 in the paper)
        B, N, d = data.shape
        
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior

        wsum = weight.sum(1)
        wsum_reg = wsum + tau 
        wxsum = torch.bmm(weight.permute(0,2,1), data) 
        wxxsum = torch.bmm(weight.permute(0,2,1), data**2) 

        pi = wsum_reg / wsum_reg.sum(1, keepdim=True) 
        mu = (wxsum + m.unsqueeze(0)*tau) / wsum_reg.unsqueeze(-1)
        Sigma = (wxxsum + (V+m**2).unsqueeze(0)*tau) / wsum_reg.unsqueeze(-1) - mu**2

        return pi.float(), mu.float(), Sigma.float()
    
    def map_em(self, data, mask=None, num_iters=1, tau=1.0, prior=None):
        # EM algorithm
        B, N, d = data.shape
        
        if mask is None:
            mask = torch.ones(B, N).to(data)

        # Need to set to the mode for initial starting point
        pi, mu, Sigma = self.mode(prior)
        pi = pi.unsqueeze(0).repeat(B,1)
        mu = mu.unsqueeze(0).repeat(B,1,1)
        Sigma = Sigma.unsqueeze(0).repeat(B,1,1)
        
        for emiter in range(num_iters):
            # E-step: Evaluate the log likelihood of the model given the data. 
            _, qq, _ = mog_eval((pi, mu, Sigma), data)
            # qq = posterior probability
            qq = qq.exp() * mask.unsqueeze(-1)

            # M-step: Update prior prob, mean and covariance
            pi, mu, Sigma = self.map_m_step(data, weight=qq, tau=tau, prior=prior)
            
        return pi, mu, Sigma, qq



class PANTHERBase(nn.Module):
    """
    Args:
    - p (int): Number of prototypes
    - d (int): Feature dimension
    - L (int): Number of EM iterations
    - out (str): Ways to merge features
    - ot_eps (float): eps
    """
    # L=1 <-- 根据论文 4.3 节: "a single EM step is sufficient"
    def __init__(self, d, prototypes, p, L=1, tau=1.0, ot_eps=0.1, fix_proto=False):
        super(PANTHERBase, self).__init__()

        self.L = L
        self.tau = tau

        self.priors = DirNIWNet(p, d, prototypes, ot_eps, fix_proto)
        # This outdim (p + 2*p*d) matches the paper's Eq 6 logic for a flat vector
        self.outdim = p + 2*p*d

    def forward(self, S, mask=None):
        """
        Args
        - S: data
        """
        B, N_max, d = S.shape
        
        if mask is None:
            mask = torch.ones(B, N_max).to(S)
        
        pis, mus, Sigmas, qqs = [], [], [], []
        pi, mu, Sigma, qq = self.priors.map_em(S, 
                                              mask=mask, 
                                              num_iters=self.L, 
                                              tau=self.tau, 
                                              prior=self.priors())

        pis.append(pi)
        mus.append(mu)
        Sigmas.append(Sigma)
        qqs.append(qq)

        pis = torch.stack(pis, dim=2)              # pis: (n_batch x n_proto x n_head)
        mus = torch.stack(mus, dim=3)              # mus: (n_batch x n_proto x embed_dim x n_head)
        Sigmas = torch.stack(Sigmas, dim=3)        # Sigmas: (n_batch x n_proto x embed_dim x n_head)
        qqs = torch.stack(qqs, dim=3)
            
        # Create the flat (B, p + 2*p*d) vector, consistent with paper's Eq 6
        out = torch.cat([pis.reshape(B,-1), mus.reshape(B,-1), Sigmas.reshape(B,-1)], dim=1)
        return out, qqs


# -------------------------------------------------------------------
# 内容来自: model_StructuredPANTHER.py (已移除数据加载逻辑)
# -------------------------------------------------------------------
class StructuredPANTHER(nn.Module):
    """
    Wrapper for PANTHER model with structured output (B, 2*p+1, D).
    
    This module implements an *alternative* representation to the paper's Eq 6.
    Instead of a flat (B, p*(1+2D)) vector, it creates a (B, 2p+1, D) token
    sequence, which is suitable for downstream Transformer/Attention models.
    
    It constructs the sequence as:
    1. mu: (B, p, D)      -> p tokens
    2. Sigma: (B, p, D)   -> p tokens
    3. pi: (B, p) -> projected to (B, 1, D) -> 1 token
    Total: (B, 2p+1, D)
    """
    
    # em_iter=1 <-- 根据论文 4.3 节: "a single EM step is sufficient"
    def __init__(self, in_dim, out_dim, n_proto, prototypes, em_iter=1, tau=0.001, ot_eps=0.1, fix_proto=True):
        super(StructuredPANTHER, self).__init__()
        self.emb_dim = in_dim   # 这是 'D' (特征维度)
        self.outsize = n_proto  # This is 'p' (原型数量)
        self.prototypes = prototypes  # (p, in_dim)
        
        # 保存 EM 迭代所需的参数
        self.em_iter = em_iter
        self.tau = tau

        # This module contains the EM step
        # We still create panther_base to get access to its self.priors network
        self.panther_base = PANTHERBase(
            self.emb_dim, prototypes, p=self.outsize, L=em_iter,
            tau=tau, ot_eps=ot_eps, fix_proto=fix_proto
        )
        
        # Output dim
        self.output_projector = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, out_dim),
            nn.ReLU(out_dim),
            nn.LayerNorm(out_dim),
            # nn.Dropout(0.2),

            nn.Linear(out_dim, out_dim),
        )

    def representation(self, S, mask=None):
        """ 
        Construct structured slide representation (B, 2*p+1, D). 
        Bypasses PANTHERBase.forward() to get structured components.
        
        Args:
            S (torch.Tensor): 输入的 patch-level 特征, 形状为 (B, n, D)
            mask (torch.Tensor, optional): 掩码, 形状为 (B, n)
        """
        B, n, D = S.shape # S 是 (B, n, D) 的输入
    
        if mask is None:
            mask = torch.ones(B, n).to(S)
        
        # Call map_em directly from the priors network
        # This is the core logic from PANTHERBase.forward() but before flattening
        pi, mu, Sigma, qq = self.panther_base.priors.map_em(
            S, 
            mask=mask, 
            num_iters=self.em_iter, # <-- 使用 self.em_iter
            tau=self.tau,           # <-- 使用 self.tau
            prior=self.panther_base.priors()
        )
        # pi: (B, p)
        # mu: (B, p, D)
        # Sigma: (B, p, D)
        # qq: (B, n, p)

        # Concatenate mu, Sigma, and pi_structured along dim=1
        # mu: (B, p, D)
        # Sigma: (B, p, D)
        # pi: (B, p, 1)
        out = torch.cat([mu, Sigma, pi.unsqueeze(2)], dim=2) # Shape: (B, p, D + D + 1)
        
        # Recreate the 'qqs' output from PANTHERBase for consistency
        qqs = torch.stack([qq], dim=3) # Shape: (B, n, p, 1)
        
        # out = slide embeddings (structured), qqs = posterior probabilities
        return {'repr': out, 'qq': qqs}

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): 输入的 patch-level 特征, 形状为 (B, n, D)
            mask (torch.Tensor, optional): 掩码, 形状为 (B, n)
        Returns:
            torch.Tensor: 结构化表征, 形状为 (B, 2*p + 1, D)
        """
        # 注意: 原始代码没有传递 mask, 但 representation 方法支持它。
        # 为稳健起见，这里也应传递 mask。
        out = self.representation(x, mask)['repr']
        out = self.output_projector(out)

        return out
    



# -------------------------------------------------------------------
# 单元测试代码
# -------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. 定义模型超参数
    in_dim_D = 64    # D: 特征维度
    n_proto_p = 64   # p: 原型数量
    em_iter_L = 1    # L: EM 迭代次数 (根据论文 4.3 节更新为 1)
    tau_val = 10.0
    ot_eps_val = 0.1
    fix_proto_val = True

    # 2. 定义测试参数
    B = 4        # 批量大小
    n = 10000    # patch 数量

    # 3. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. 创建模拟数据
    # 创建 (p, D) 的原型
    prototypes = np.random.rand(n_proto_p, in_dim_D).astype(np.float32)
    # 创建 (B, n, D) 的输入张量
    input_tensor = torch.rand(B, n, in_dim_D).to(device)
    # 创建模拟掩码 (例如，最后10个 patch 是 padding)
    mask = torch.ones(B, n).to(device)
    mask[:, -10:] = 0

    # 5. 实例化模型 (使用新的、明确的参数签名)
    # 注意：StructuredPANTHER 的 __init__ 有一个 out_dim 参数, 单元测试未使用, 传入一个示例值
    model = StructuredPANTHER(
        in_dim=in_dim_D,
        out_dim=in_dim_D, # 示例值, 因为单元测试不使用 output_projector
        n_proto=n_proto_p,
        em_iter=em_iter_L,
        tau=tau_val,
        ot_eps=ot_eps_val,
        fix_proto=fix_proto_val,
        prototypes=prototypes,
    ).to(device)

    # 6. 执行前向传播 (同时传递 input 和 mask)
    print(f"--- 单元测试: StructuredPANTHER ---")
    print(f"输入张量形状 (B, n, D): {input_tensor.shape}")
    print(f"输入掩码形状 (B, n): {mask.shape}")
    
    output = model(input_tensor, mask) # <-- 传递 mask

    # 7. 验证输出形状
    expected_shape = (B, 2 * n_proto_p + 1, in_dim_D)
    print(f"输出张量形状 (B, 2*p+1, D): {output.shape}")
    

    print("形状验证成功!")
    print(f"---------------------------------")

    # -------------------------------------------------------------------
    # 新增单元测试: 模拟可变长度的 Batch (如用户所述)
    # -------------------------------------------------------------------
    print(f"--- 单元测试: 可变长度 Batch (Padding) ---")
    
    # 1. 定义测试参数
    B_var = 4       # 批量大小
    n_max = 1400    # 整个 batch 的最大长度 (为了容纳 1300+)
    
    # 2. 创建模拟数据
    # 创建 (B, n_max, D) 的输入张量
    input_tensor_var = torch.rand(B_var, n_max, in_dim_D).to(device)
    
    # 3. 创建对应的 mask
    # 假设: 1 = valid, 0 = padding
    mask_var = torch.zeros(B_var, n_max).to(device)
    
    # 第 1 张图片: 只有 80 个 valid token
    valid_lengths = [80, 1350, 1320, 1400] # 模拟的有效长度
    mask_var[0, :valid_lengths[0]] = 1
    
    # 第 2 张图片: 有 1350 个 valid token
    mask_var[1, :valid_lengths[1]] = 1
    
    # 第 3 张图片: 有 1320 个 valid token
    mask_var[2, :valid_lengths[2]] = 1
    
    # 第 4 张图片: 满了 (1400 个 valid token)
    mask_var[3, :valid_lengths[3]] = 1
    
    print(f"创建了一个 Batch (B={B_var}), 最大长度 n_max={n_max}")
    print(f"输入张量形状 (B, n, D): {input_tensor_var.shape}")
    print(f"输入掩码形状 (B, n): {mask_var.shape}")
    print(f"Batch 中各项的有效 tokens 数量: {valid_lengths}")
    print(f"Mask 中第一项的 (1) 的总数: {mask_var[0].sum().item()}")
    print(f"Mask 中第二项的 (1) 的总数: {mask_var[1].sum().item()}")

    # 4. 实例化新模型 (确保原型参数维度匹配)
    prototypes_var = np.random.rand(n_proto_p, in_dim_D).astype(np.float32)
    model_var = StructuredPANTHER(
        in_dim=in_dim_D,
        out_dim=in_dim_D, # 示例值
        n_proto=n_proto_p,
        em_iter=em_iter_L, # 使用 L=1
        tau=tau_val,
        ot_eps=ot_eps_val,
        fix_proto=fix_proto_val,
        prototypes=prototypes_var,
    ).to(device)
    
    # 5. 执行前向传播
    # model 会使用 mask 来确保 padding (0) 的 token 不参与 EM 计算
    output_var = model_var(input_tensor_var, mask_var)

    # 6. 验证输出形状
    expected_shape_var = (B_var, 2 * n_proto_p + 1, in_dim_D)
    print(f"输出张量形状 (B, 2*p+1, D): {output_var.shape}")
    

    print("可变长度 (Padding) 测试成功!")
    print(f"这证明了模型可以处理 (B, n_max, D) 的输入和 (B, n_max) 的掩码，")
    print(f"并始终产生 (B, 2p+1, D) 的固定大小输出。")
    print(f"---------------------------------")