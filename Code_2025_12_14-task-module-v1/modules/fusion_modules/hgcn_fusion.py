import torch
import math
import random
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.layers import drop_path, trunc_normal_
from typing import List, Optional, Dict, Tuple, Any

# Note: Removed torch_scatter and torch_geometric imports
# Using standard nn.LayerNorm
from torch.nn import LayerNorm 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- GNN & Pooling Helpers (Refactored for Dense Batches) ---

class SimulatedSAGEConv(nn.Module):
    """
    Simulates SAGEConv with mean aggregation on a B-batched, dense, 
    fully-connected graph.
    Input: x (B, N, D_in), mask (B, N, 1)
    Output: (B, N, D_out)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # SAGEConv has two linear layers
        # lin_l is for aggregated neighbors
        self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
        # lin_r is for the node itself (self-loop)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features, shape (B, N, D_in)
            mask (torch.Tensor): Mask, shape (B, N, 1). True/1 for real, False/0 for padding.
        """
        # Ensure mask is float for (x * mask)
        mask_float = mask.float()
        
        # --- Aggregation (lin_l) ---
        # 1. Project all nodes as if they are neighbors
        x_l = self.lin_l(x) # (B, N, D_out)
        
        # 2. Apply mask *before* summing for mean calculation
        x_l_masked = x_l * mask_float # (B, N, D_out)
        
        # 3. Calculate masked sum and count for each graph in the batch
        #    This is the "aggregation" step.
        sum_l = x_l_masked.sum(dim=1, keepdim=True) # (B, 1, D_out)
        num_nodes = mask_float.sum(dim=1, keepdim=True) # (B, 1, 1)
        
        # 4. Calculate masked mean. Add eps for stability.
        #    This is our aggregated neighbor vector, same for all nodes in a graph.
        mean_l = sum_l / (num_nodes + 1e-6) # (B, 1, D_out)
        
        # --- Self-Loop (lin_r) ---
        # 1. Project all nodes for the self-loop
        x_r = self.lin_r(x) # (B, N, D_out)
        
        # --- Combine ---
        # SAGEConv logic: aggregated_neighbors + self_loop
        # We broadcast the (B, 1, D_out) mean vector to (B, N, D_out)
        # and add it to the self-loop term.
        out = mean_l + x_r # (B, N, D_out)
        
        # 6. Apply mask *after* combination to zero-out padding nodes
        return out * mask_float

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels})'

class DenseGlobalAttention(torch.nn.Module):
    """
    Global Attention pooling layer for dense, batched data.
    Input: x (B, N, D), mask (B, N, 1)
    Output: (B, D)
    """
    def __init__(self, gate_nn, nn=None):
        super(DenseGlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        def _reset(item):
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()
        
        if self.gate_nn is not None:
            self.gate_nn.apply(_reset)
        if self.nn is not None:
            self.nn.apply(_reset)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input features, shape (B, N, D)
            mask (torch.Tensor): Mask, shape (B, N, 1). True/1 for real, False/0 for padding.
        """
        # (B, N, D) -> (B, N, 1)
        gate = self.gate_nn(x)
        
        # Apply mask (True/1 for real, False/0 for padding)
        # Set padding nodes to -inf so they get zero weight in softmax
        # Ensure mask is boolean
        mask_bool = mask.bool()
        gate = gate.masked_fill(~mask_bool, -torch.inf)
        
        # Softmax over the node dimension (dim=1)
        # (B, N, 1)
        gate = F.softmax(gate, dim=1)
        
        # FIX: If all nodes were masked, softmax results in nan.
        # Replace nan with 0.0, so the weighted sum becomes 0.
        gate = torch.nan_to_num(gate, nan=0.0)
        
        # Apply transform if nn is provided
        # (B, N, D)
        x_nn = self.nn(x) if self.nn is not None else x
        
        # Masked weighted sum
        # (gate * x_nn) -> (B, N, D)
        # .sum(dim=1) -> (B, D)
        out = (gate * x_nn).sum(dim=1)
        
        return out, gate

    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)

def GNN_relu_Block(dim, dropout=0.3):
    """
    GNN activation/normalization block from your provided code.
    Note: Uses standard nn.LayerNorm
    """
    return nn.Sequential(
        nn.ReLU(),
        LayerNorm(dim),
        nn.Dropout(p=dropout)
    )

# --- MAE / ViT Helpers (from refactored code) ---
# (These helpers are mostly batch-agnostic and remain unchanged)

def generate_mae_mask(num_modalities, mask_ratio=0.75):
    """
    Generates a random mask for the MAE.
    False = Visible, True = Masked
    """
    mask_num = int(num_modalities * mask_ratio)
    visible_num = num_modalities - mask_num
    
    # Create an array with `visible_num` Falses and `mask_num` Trues
    mask = np.hstack([
        np.zeros(visible_num, dtype=bool), # Visible
        np.ones(mask_num, dtype=bool),     # Masked
    ])
    np.random.shuffle(mask)
    
    # Reshape to (1, 1, M) for broadcasting
    mask = np.expand_dims(mask, 0)
    mask = np.expand_dims(mask, 0)
    return torch.from_numpy(mask)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    """MLP as used in Vision Transformer blocks.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """Multi-head self-attention mechanism.
    """
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """Transformer Block.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

# --- MAE Vision Transformer (Encoder, Decoder, Main) ---
# (Unchanged, as they already operate on (B, N, C) data)
class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer Encoder for MAE
    """
    def __init__(self, embed_dim=512, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False, train_type_num=3):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = nn.Linear(embed_dim, embed_dim) # Simple linear projection
        num_patches = train_type_num
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, M, C = x.shape # (B, M, C)
        # `mask` is (1, 1, M), True for *masked*
        mask_bool = ~mask.squeeze(1) # (1, M), True for *visible*
        
        # Expand mask to (B, M) for batch-wise boolean indexing
        mask_bool = mask_bool.expand(B, M) # (B, M)
        
        x_vis = x[mask_bool].reshape(B, -1, C) # Select visible tokens
        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)
        return x_vis

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer Decoder for MAE
    """
    def __init__(self, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, train_type_num=3,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        
        x = self.head(self.norm(x))
        return x

class PretrainVisionTransformer(nn.Module):
    """ MAE Vision Transformer
    """
    def __init__(self,
                 encoder_embed_dim=512,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=512,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 train_type_num=3,
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            train_type_num=train_type_num)
        self.decoder = PretrainVisionTransformerDecoder(
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            train_type_num=train_type_num)
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = get_sinusoid_encoding_table(train_type_num, decoder_embed_dim)
        trunc_normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x, mask):
        x_vis = self.encoder(x, mask) # (B, N_vis, C_e)
        x_vis = self.encoder_to_decoder(x_vis) # (B, N_vis, C_d)
        B, N_vis, C = x_vis.shape
        M = self.pos_embed.shape[1] # Get M from pos_embed (e.g., 3)
        
        # mask is (1, 1, M), True for masked
        # We need (B, M) masks
        mask_bool_vis = ~mask.squeeze(1).expand(B, M) # (B, M), True for visible
        mask_bool_mask = mask.squeeze(1).expand(B, M) # (B, M), True for masked
        
        # Get positional embeddings
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach() # (B, M, C)
        pos_emd_vis = expand_pos_embed[mask_bool_vis].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask_bool_mask].reshape(B, -1, C)
        
        # Create the full sequence with mask tokens
        # x_full = (B, N_vis + N_mask, C)
        N_mask = pos_emd_mask.shape[1] # N_mask per sample
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        
        # Unshuffle tokens to original order
        ids_vis = mask_bool_vis.nonzero(as_tuple=False)
        ids_mask = mask_bool_mask.nonzero(as_tuple=False)
        
        # `ids_restore` maps from shuffled (vis, mask) order to original (0..M) order
        ids_restore = torch.empty(B, M, dtype=torch.long, device=x.device)
        
        # Create arange tensors that respect batching
        arange_vis = torch.arange(N_vis, device=x.device).repeat(B)
        arange_mask = torch.arange(N_vis, N_vis + N_mask, device=x.device).repeat(B)
        
        ids_restore[ids_vis[:, 0], ids_vis[:, 1]] = arange_vis
        ids_restore[ids_mask[:, 0], ids_mask[:, 1]] = arange_mask
        
        ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, C)
        
        # Unshuffle
        x_unshuffled = torch.gather(x_full, 1, ids_restore)
        
        # Decode the unshuffled full sequence
        x = self.decoder(x_unshuffled) # (B, M, C_d)
        return x

def Mix_mlp(dim):
    return nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim))

class MixerBlock(nn.Module):
    """
    MLP-Mixer block, modified to be B-compatible.
    Operates on (B, M, D)
    dim1 = M (modalities, token-mixing)
    dim2 = D (embed_dim, channel-mixing)
    """
    def __init__(self, dim1, dim2):
        super(MixerBlock,self).__init__()
        
        self.norm1 = LayerNorm(dim2)
        self.token_mixer = Mix_mlp(dim1) # Mixes across M dimension
        self.norm2 = LayerNorm(dim2)
        self.channel_mixer = Mix_mlp(dim2) # Mixes across D dimension
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, M, D)
        """
        B, M, D = x.shape
        
        # Token-mixing
        y = self.norm1(x)       # (B, M, D)
        y = y.transpose(1, 2)   # (B, D, M)
        y = self.token_mixer(y) # (B, D, M)
        y = y.transpose(1, 2)   # (B, M, D)
        x = x + y
        
        # Channel-mixing
        y = self.norm2(x)       # (B, M, D)
        y = self.channel_mixer(y) # (B, M, D)
        x = x + y
    
        return x

# --- Main HGCN_FUSION Module ---

class HGCNFusionModule(nn.Module):
    """
    HGCN Fusion Module, refactored for dense, B-batched data.
    
    Input `embeddings`: List[Optional[torch.Tensor]]
        - `embeddings[m]` is (B, N_m, D) or None
    Input `masks`: List[Optional[torch.Tensor]]
        - `masks[m]` is (B, N_m, 1) or None
        - Mask is True/1 for real nodes, False/0 for padding.
    """
    def __init__(self,
                 args, 
                 embed_dim: int, 
                 max_modalities: int = 3, 
                 dropout: float = 0.3,
                 mae_encoder_depth: int = 2,
                 mae_decoder_depth: int = 1,
                 mae_attn_heads: int = 8,
                 mae_mask_ratio: float = 0.75,
                 mae_loss_weight: float = 1.0
                 ) -> None:
        super().__init__()
        
        self.args = args
        self.embed_dim = embed_dim
        self.max_modalities = max_modalities
        self.mae_mask_ratio = mae_mask_ratio
        self.mae_loss_weight = mae_loss_weight
        
        # --- 1. Intra-Modal GNNs (Generalized) ---
        self.gnns = nn.ModuleList([
            SimulatedSAGEConv(in_channels=embed_dim, out_channels=embed_dim) 
            for _ in range(max_modalities)
        ])
        self.relus = nn.ModuleList([
            GNN_relu_Block(embed_dim, dropout=dropout) 
            for _ in range(max_modalities)
        ])
        
        # --- 2. Pooling (Generalized) ---
        self.pools = nn.ModuleList()
        for _ in range(max_modalities):
            att_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//4), 
                nn.ReLU(), 
                nn.Linear(embed_dim//4, 1)
            )
            self.pools.append(DenseGlobalAttention(att_net))
            
        # --- 3. Inter-Modal MAE ---
        self.mae = PretrainVisionTransformer(
            encoder_embed_dim=embed_dim,
            encoder_depth=mae_encoder_depth,
            encoder_num_heads=mae_attn_heads,
            decoder_embed_dim=embed_dim,
            decoder_depth=mae_decoder_depth,
            decoder_num_heads=mae_attn_heads,
            decoder_num_classes=embed_dim, # Reconstruct the embedding
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            train_type_num=max_modalities
        )
        
        # --- 4. Inter-Modal Mixer ---
        self.mixer = MixerBlock(max_modalities, embed_dim) # Dims are (M, D)
        
        # --- 7. Final Fusion MLP ---
        self.fusion_mlp = nn.Linear(max_modalities * embed_dim, embed_dim)
        
    def forward(self, 
                embeddings: List[Optional[torch.Tensor]], 
                masks: List[Optional[torch.Tensor]],
                task_head: nn.Module = None, 
                batch: Optional[Dict[str, Any]] = None
                ) -> Dict:
        """
        Forward pass for HGCN Fusion (Dense).
        
        Args:
            embeddings (List[Optional[torch.Tensor]]): 
                List[m] of (B, N_m, D) node feature tensors.
            masks (List[Optional[torch.Tensor]]): 
                List[m] of (B, N_m, 1) boolean/float masks (1=real, 0=padding).
        Returns:
            Dict: {
                "fused_embedding": torch.Tensor (B, D),
                "loss_dict": Dict[str, torch.Tensor],
            }
        """
        
        pooled_features_list = [] # List to store H_m (B, D)
        gnn_outputs = {}          # Dict to store {m: (G_intra_m, mask_m)}
        patient_modality_mask_list = [] # List to store (B, 1) patient masks
        loss_dict = {}

        # --- Determine Batch Size B and Device ---
        B = -1
        _dev = device
        for m in range(self.max_modalities):
            if m < len(embeddings) and embeddings[m] is not None:
                B = embeddings[m].shape[0]
                _dev = embeddings[m].device
                break
        
        if B == -1:
            # No data was provided at all
            raise ValueError("No valid embedding tensors provided to HGCN_FUSION")

        # --- Steps 1 & 2: Intra-Modal GNN + First Pooling ---
        for m in range(self.max_modalities):
            x_m = embeddings[m] if m < len(embeddings) else None
            mask_m = masks[m] if m < len(masks) else None
            
            # Ensure mask is (B, N_m, 1) and float
            if mask_m is not None:
                mask_m = mask_m.float().view(B, -1, 1)
            
            has_data = (
                x_m is not None and 
                mask_m is not None and 
                x_m.numel() > 0 and 
                mask_m.numel() > 0
            )
            
            if has_data:
                # Ensure mask matches x_m shape
                if mask_m.shape[1] != x_m.shape[1]:
                    raise ValueError(f"Modality {m}: x shape {x_m.shape} and mask shape {mask_m.shape} mismatch")
                
                # 1. GNN -> G_intra
                # (B, N_m, D)
                x_gnn = self.relus[m](self.gnns[m](x_m, mask_m))
                
                # 2. Pooling -> H_m
                # (B, D)
                pooled_m, _ = self.pools[m](x_gnn, mask_m)
                
                pooled_features_list.append(pooled_m)
                
                # Store G_intra and mask_m for Step 5
                gnn_outputs[m] = (x_gnn, mask_m) 
                
                # Patient mask: (B, 1). True if *any* node is present for this patient
                patient_mask_m = mask_m.any(dim=1) # (B, 1)
                patient_modality_mask_list.append(patient_mask_m)

            else:
                # Modality is missing
                pooled_features_list.append(
                    torch.zeros(B, self.embed_dim, device=_dev)
                )
                patient_modality_mask_list.append(
                    torch.zeros(B, 1, dtype=torch.bool, device=_dev)
                )
        
        # --- Step 3 & 4: Inter-Modality MAE + Mixer ---
        
        # stacked_features: H (B, M, D)
        stacked_features = torch.stack(pooled_features_list, dim=1) 
        
        # patient_modality_mask: (B, M), True if modality is present
        patient_modality_mask = torch.cat(patient_modality_mask_list, dim=1) # (B, M)

        # Generate random mask (True = Masked)
        # (1, 1, M) -> broadcasts to (B, 1, M)
        # 1) 训练时：用随机 mask 做 MAE
        if self.training and self.mae_mask_ratio > 0:
            mae_bool_mask = generate_mae_mask(
                self.max_modalities, self.mae_mask_ratio
            ).to(_dev)
        else:
            # 2) eval 时：不 mask，全部可见
            mae_bool_mask = torch.zeros(1, 1, self.max_modalities, dtype=torch.bool, device=_dev)
            
        # reconstructed_features: (B, M, D)
        reconstructed_features = self.mae(stacked_features, mae_bool_mask)
        
        # Apply mixer to the reconstructed features -> H_inter
        mixed_features = self.mixer(reconstructed_features) # (B, M, D)

        # --- Step 5 & 6: Broadcast Add and Second Readout ---
        
        final_pooled_features = [] # H_final (List of (B, D))
        
        for m in range(self.max_modalities):
            # Get H_inter_m (B, D)
            h_inter_m = mixed_features[:, m, :] 
            
            if m in gnn_outputs:
                # Modality was present
                g_intra_m, mask_m = gnn_outputs[m] # (B, N_m, D), (B, N_m, 1)
                N_m = g_intra_m.shape[1]
                
                # Step 5: G_out = G_intra + H_inter (B-aware Broadcast Add)
                # h_inter_m (B, D) -> (B, 1, D) -> (B, N_m, D)
                h_inter_m_bcast = h_inter_m.unsqueeze(1).expand(-1, N_m, -1)
                
                # Add and re-apply mask
                g_out_m = (g_intra_m + h_inter_m_bcast) * mask_m # (B, N_m, D)
                
                # Step 6: H_final = Pool(G_out)
                final_pooled_m, _ = self.pools[m](g_out_m, mask_m) # (B, D)
                final_pooled_features.append(final_pooled_m)
                
            else:
                # Modality was missing. 
                # Use the reconstructed and mixed hyperedge (H_inter_m)
                final_pooled_features.append(h_inter_m)

        # --- Step 7: Final Fusion MLP ---
        # h_final_stacked: (B, M, D)
        h_final_stacked = torch.stack(final_pooled_features, dim=1)
        
        # Flatten for MLP: (B, M*D)
        h_flattened = h_final_stacked.reshape(B, -1)
        
        # Fused embedding: (B, D)
        fused_embedding = self.fusion_mlp(h_flattened)
        
        # --- Calculate MAE Reconstruction Loss ---
        
        # loss_mask: (B, M), True for tokens that were *randomly masked*
        loss_mask = mae_bool_mask.squeeze().expand_as(patient_modality_mask) # (B, M)
        
        # final_loss_mask: (B, M), True for tokens that were *randomly masked* AND *actually present*
        final_loss_mask = loss_mask & patient_modality_mask
        
        # Calculate MSE loss against original pooled features (H)
        reconstruction_loss = torch.tensor(0.0, device=_dev)

        if self.training and self.mae_loss_weight > 0:
            loss_mask = mae_bool_mask.squeeze().expand_as(patient_modality_mask)  # (B, M)
            final_loss_mask = loss_mask & patient_modality_mask
            reconstruction_loss_all = F.mse_loss(
                reconstructed_features, stacked_features, reduction='none'
            )  # (B, M, D)
            if final_loss_mask.sum() > 0:
                reconstruction_loss = reconstruction_loss_all[final_loss_mask].mean()

        
        if final_loss_mask.sum() > 0:
            # Select only the losses for tokens that were present AND masked
            # reconstruction_loss_all[final_loss_mask] selects (K, D) elements
            # .mean() averages over all K*D elements.
            reconstruction_loss = reconstruction_loss_all[final_loss_mask].mean()
        else:
            # No masked tokens were present (or mask_ratio=0)
            reconstruction_loss = torch.tensor(0.0, device=_dev)
        
        # Deep Supervision
        loss_ds = torch.zeros((1, ), device=_dev, dtype=torch.float32)
        if task_head and batch:
            for m in range(self.max_modalities):
                modal_feat = final_pooled_features[m]       # (B, D)
                modal_mask = patient_modality_mask[:, m]    # (B, 1)
                supervision_output_text = task_head.decode(modal_feat, modal_mask, batch)
                loss_dict[f"modality_{m}"] = supervision_output_text['loss']
                loss_ds += supervision_output_text['loss']

        # Apply loss weight
        loss_dict['total_loss'] = reconstruction_loss * self.mae_loss_weight + 5 * loss_ds
        
        return {
            "fused_embedding": fused_embedding, # (B, D)
            "loss_dict": loss_dict,
        }
    
