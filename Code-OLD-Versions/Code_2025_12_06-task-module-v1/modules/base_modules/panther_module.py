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
    
    def map_em(self, data, mask=None, num_iters=3, tau=1.0, prior=None):
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
    from (B, p, D + D + 1) -> (B, p, D)
    Total: (B, p, D)
    """
    
    # em_iter=1 <-- 根据论文 4.3 节: "a single EM step is sufficient" 但是我提高到3试一下
    def __init__(self, in_dim, n_proto, prototypes, em_iter=3, tau=0.001, ot_eps=0.1, fix_proto=True):
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
            torch.Tensor: 结构化表征, 形状为 (B, p, D+D+1)
        """
        # 注意: 原始代码没有传递 mask, 但 representation 方法支持它。
        # 为稳健起见，这里也应传递 mask。
        out = self.representation(x, mask)['repr']

        return out
    



