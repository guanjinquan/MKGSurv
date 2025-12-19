import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# 假设 masked_mean_pool 在您的路径中，尽管此版本的 VIB *不*使用它
# from modules.common_modules.aggregation_utils import masked_mean_pool


class TokenWiseMultiModalVIB(nn.Module):
    """
    一个基于 Transformer Decoder 和可学习 Query 的逐Token多模态变分信息瓶颈模块。

    [审稿人注]：
    此版本已从 nn.TransformerEncoder 修正为 nn.TransformerDecoder，
    以正确实现“Query token 压缩 Input token”的意图。
    
    [新增修改]：
    已为所有线性层(Linear)和可学习查询(Queries)添加 Kaiming Normal 初始化。
    """

    def __init__(self,
                 num_modalities: int,
                 embed_dim: int,
                 latent_dim: int = None,  # [新增] 允许瓶颈维度与 embed_dim 不同
                 num_queries: int = 16,
                 num_decoder_layers: int = 4,
                 num_attn_heads: int = 8):
        
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim if latent_dim else embed_dim
        self.num_queries = num_queries

        # 将 queries 存储在 nn.ParameterList 中
        self.modal_queries = nn.ParameterList()
        # 将 modules (decoder, fc) 存储在 nn.ModuleList[nn.ModuleDict] 中
        self.vib_modules = nn.ModuleList()

        for _ in range(num_modalities):
            # 1. 可学习的 Query Token
            # [修改] 使用 empty 并应用 Kaiming Normal 初始化
            modal_query = nn.Parameter(torch.empty(self.num_queries, embed_dim))
            nn.init.kaiming_normal_(modal_query.data, a=0, mode='fan_in', nonlinearity='relu')
            self.modal_queries.append(modal_query)

            # 2. 使用 TransformerDecoderLayer
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.embed_dim,
                nhead=num_attn_heads,
                dropout=0.1,
                batch_first=True
            )
            # 3. 使用 TransformerDecoder
            transformer_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_decoder_layers
            )

            # 4. VIB 头部 (从 D -> D_latent)
            fc_mu = nn.Linear(self.embed_dim, self.latent_dim)
            fc_log_var = nn.Linear(self.embed_dim, self.latent_dim)

            # [新增] 应用 Kaiming Normal 初始化
            # 这将递归地初始化 TransformerDecoder 和 VIB 头部中的所有 nn.Linear 层
            transformer_decoder.apply(self._init_weights)
            fc_mu.apply(self._init_weights)
            fc_log_var.apply(self._init_weights)

            # ModuleDict 现在只包含 nn.Module 子类
            self.vib_modules.append(nn.ModuleDict({
                'decoder': transformer_decoder,
                'mu': fc_mu,
                'log_var': fc_log_var
            }))

    # [新增] Kaiming Normal 初始化辅助函数
    def _init_weights(self, m: nn.Module):
        """将 Kaiming Normal 初始化应用于 nn.Linear 层。"""
        if isinstance(m, nn.Linear):
            # a=0 对应 ReLU 激活函数
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 注意: 这也会正确地递归到 TransformerDecoderLayer 
        # 内部的 MHA 和 FFN 中的 nn.Linear 层。
        # LayerNorm 层不会受到影响，因为它们不是 nn.Linear 的实例。

    def reparameterize(self,
                       mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def compute_kl_loss(self,
                        mu: torch.Tensor,
                        log_var: torch.Tensor,
                        patient_mask: torch.Tensor) -> torch.Tensor:
        """
        参数:
            mu (torch.Tensor): (B, num_queries, D_latent)
            log_var (torch.Tensor): (B, num_queries, D_latent)
            patient_mask (torch.Tensor): (B,) [True=有效, False=空患者]
        """
        
        # KL = -0.5 * (1 + log_var - mu^2 - exp(log_var))
        # 维度: (B, num_queries, D_latent)
        kl_element_wise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())

        # 在 token(N) 和 latent(D_latent) 维度上求平均
        # 这使得损失量级不依赖于 num_queries 和 latent_dim
        kl_per_sample = torch.mean(kl_element_wise, dim=[1, 2])  # (B,)

        # 只选择那些真正有数据的患者 (patient_mask 为 True) 的 KL 损失
        valid_kl_per_sample = kl_per_sample[patient_mask]

        # 边缘情况处理：如果批次中没有一个患者有此模态，返回 0.0
        if valid_kl_per_sample.numel() == 0:
            return torch.tensor(0.0, device=mu.device)
            
        # 在 batch 维度上只对有效样本取平均
        return torch.mean(valid_kl_per_sample)


    def forward(self,
                modal_features_list: List[torch.Tensor],
                modal_masks_list: List[Optional[torch.Tensor]]
                ) -> tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], dict]:
        """
        参数:
            modal_features_list (List[torch.Tensor]):
                包含 M 个模态的 token 序列的列表。每个元素: (B, N_i, D)
            modal_masks_list (List[Optional[torch.Tensor]]):
                包含 M 个模态的掩码的列表。每个元素: (B, N_i) (1 表示有效, 0 表示 padding)
                
        返回:
            tuple:
            - z_sequences (List[Tensor]): VIB 瓶颈特征
            - z_masks (List[Tensor]): VIB 瓶颈特征对应的掩码
            - kl_losses (Dict[str, Tensor]): KL 损失
        """

        assert len(modal_features_list) == self.num_modalities, \
            f"输入了 {len(modal_features_list)} 个模态, 但模块初始化为 {self.num_modalities} 个"
        assert len(modal_masks_list) == self.num_modalities, \
            "特征列表和掩码列表的长度必须一致"

        z_sequences = []
        z_masks = []  # 为 VIB 输出创建新的掩码列表
        kl_losses = {}  # 用于存储每个模态的损失
        mu_dict = {}
        var_dict = {}

        # 预先获取 device，以防第一个模态为 None
        device = None
        for features in modal_features_list:
            if features is not None:
                device = features.device
                break
        
        # 如果所有模态都是 None, 尝试获取默认 device
        if device is None:
            # 尝试从参数获取
            try:
                device = next(self.parameters()).device
            except StopIteration:
                # 如果没有参数, 回退到 cpu
                device = torch.device("cpu")


        for i in range(self.num_modalities):
            if modal_features_list[i] is None:
                z_sequences.append(None)
                z_masks.append(None)
                kl_losses[str(i)] = torch.tensor(0.0).to(device)
                continue

            tokens = modal_features_list[i]  # (B, N_i, D) [Memory]
            
            # [新增调试] 检查 VIB 的输入是否包含 NaN
            assert not torch.any(torch.isnan(tokens)), f"VIB Input 'tokens' contains NaN in modality {i}"
            
            mask = modal_masks_list[i]        # (B, N_i) or None
            B, N_i, _ = tokens.shape

            # --- [新策略：只处理有效患者] ---
            
            # 1. 立即计算哪些患者是有效的
            if mask is not None:
                # 检查原始 mask (B, N_i) 中，哪些患者至少有一个有效 token
                # mask.any(dim=1) -> (B,) e.g., [True, False, True]
                patient_has_data_mask = mask.any(dim=1) # (B,)
            else:
                # 如果 mask 为 None, 意味着所有 (B, N_i) token 都被假定为有效
                patient_has_data_mask = torch.ones(B, device=device, dtype=torch.bool)

            # 2. [新策略] 为 VIB 输出预分配全零张量
            # VIB head (h) 的输出
            final_h = torch.zeros(B, self.num_queries, self.embed_dim, device=device)
            # VIB mu/log_var 的输出
            final_mu = torch.zeros(B, self.num_queries, self.latent_dim, device=device)
            final_log_var = torch.zeros(B, self.num_queries, self.latent_dim, device=device)

            # 3. 检查是否 *至少有一个* 患者有此模态的数据
            if patient_has_data_mask.any():
                # [新策略] 只有存在有效数据时, 才运行 VIB 模块
                
                # 3a. 选择有效数据 (B_valid, N_i, D)
                valid_tokens = tokens[patient_has_data_mask] 
                
                # 3b. 获取模块并准备有效 query (B_valid, num_queries, D)
                modal_query_param = self.modal_queries[i]
                vib_encoder_parts = self.vib_modules[i]
                num_valid = valid_tokens.shape[0] # B_valid
                valid_query = modal_query_param.expand(num_valid, -1, -1) 

                # 3c. 选择有效 mask (B_valid, N_i)
                valid_memory_key_padding_mask = None
                if mask is not None:
                    # (B, N_i) -> (B_valid, N_i)
                    valid_mask_bool = mask[patient_has_data_mask]
                    # True 表示 *被掩码*
                    valid_memory_key_padding_mask = (valid_mask_bool == 0)
                    # (我们不再需要旧的 "fix-the-mask" 逻辑, 
                    # 因为 B_valid 中的每一行都保证至少有一个 False)

                # 3d. 运行 Decoder (只在子批次上)
                h_valid = vib_encoder_parts['decoder'](
                    tgt=valid_query,
                    memory=valid_tokens,
                    memory_key_padding_mask=valid_memory_key_padding_mask
                )
                
                # 3e. 运行 VIB heads (只在子批次上)
                mu_valid = vib_encoder_parts['mu'](h_valid)
                log_var_valid = vib_encoder_parts['log_var'](h_valid)

                mu_dict[str(i)] = torch.mean(mu_valid)
                var_dict[str(i)] = torch.mean(log_var_valid)
                
                # 3f. 将有效输出放回全零张量
                final_h[patient_has_data_mask] = h_valid
                final_mu[patient_has_data_mask] = mu_valid
                final_log_var[patient_has_data_mask] = log_var_valid

            # --- VIB 模块运行结束 ---
            # 此时, final_mu, final_log_var 要么是全零 (if no valid patient),
            # 要么包含了有效数据 (if some valid patients).

            # [新增调试] 检查 VIB 解码器的输出是否包含 NaN 或 Inf
            # (我们检查 final_h 而不是 h_valid, 因为它包含了完整批次)
            assert not torch.any(torch.isnan(final_h)), f"VIB Decoder output 'final_h' contains NaN in modality {i}"
            assert not torch.any(torch.isinf(final_h)), f"VIB Decoder output 'final_h' contains Inf in modality {i}" 

            # 瓶颈 token (h) 通过 VIB 头部 (现在使用 final_mu, final_log_var)
            mu = final_mu
            log_var = final_log_var

            # [新增调试] 检查 VIB 头部(mu/log_var)的输出
            assert not torch.any(torch.isnan(mu)), f"VIB head output 'mu' contains NaN in modality {i}"
            assert not torch.any(torch.isinf(mu)), f"VIB head output 'mu' contains Inf in modality {i}"
            assert not torch.any(torch.isnan(log_var)), f"VIB head output 'log_var' (pre-clamp) contains NaN in modality {i}"
            assert not torch.any(torch.isinf(log_var)), f"VIB head output 'log_var' (pre-clamp) contains Inf in modality {i}"

            # 稳定性修复：裁剪 log_var 以防止 NaN
            # [修改] 将 log_var 的上界从 10 降低到 6，以提高采样稳定性
            # log_var = torch.clamp(log_var, min=-1e5, max=1e5)
            log_var = torch.clamp(log_var, min=-10, max=10)

            # Samping
            z_seq = self.reparameterize(mu, log_var)  # (B, num_queries, D_latent)
            
            # [逻辑不变] 解决“部分患者缺失”问题
            # 我们已经有了 patient_has_data_mask，现在用它来创建 new_mask
            # (B,) -> (B, 1) -> (B, num_queries)
            new_mask = patient_has_data_mask.unsqueeze(-1).expand(B, self.num_queries)

            # 将 patient_has_data_mask 传递给 compute_kl_loss
            kl_losses[str(i)] = self.compute_kl_loss(mu, log_var, patient_mask=patient_has_data_mask)
            
            # 将"空"患者 (new_mask 为 False) 的 z_seq 向量显式归零
            # new_mask (B, num_queries) -> (B, num_queries, 1)
            # z_seq (B, num_queries, D_latent) * (B, num_queries, 1)
            z_seq = z_seq * new_mask.unsqueeze(-1)
            
            z_sequences.append(z_seq)

            # 确保掩码是 bool 类型
            z_masks.append(new_mask.bool())

        # 聚合 KL 损失并创建返回字典
        # 确保只对有效的 KL 损失值进行平均
        valid_kl_scalars = [kl for kl in kl_losses.values() if kl.numel() > 0 and kl > 0] # 过滤掉空的或0
        
        # 边缘情况：如果所有模态在所有患者上都为空
        if not valid_kl_scalars:
            total_kl_loss_scalar = torch.tensor(0.0).to(device)
        else:
            total_kl_loss_scalar = torch.stack(valid_kl_scalars).mean()
            
        kl_losses['total_loss'] = total_kl_loss_scalar

        # record mu and var
        for i in range(self.num_modalities):
            kl_losses[f"mu_{i}"] = mu_dict[str(i)]
            kl_losses[f"logvar_{i}"] = var_dict[str(i)]

        # 返回元组以匹配 model.py 中的解包
        return z_sequences, z_masks, kl_losses



