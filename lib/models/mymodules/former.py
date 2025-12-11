import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from torch.nn.functional import pad
import numpy as np
from mamba_ssm.modules.mamba_simple import Mamba

###############################################################################
# ClassMamba
###############################################################################
class ClassMamba(nn.Module):
    """
    使用 mamba 核心机制实现的 token mixer，用于替换 ClassAttention。
    机制说明：
      1. 输入 x 的形状为 (B, N, dim)，其中 x[:, 0] 为 class token，其余为 patch tokens。
      2. 对 class token 进行线性映射得到 query 表示。
      3. 将 query 与 patch tokens 拼接后送入 Mamba 模块进行 token 混合，
         Mamba 模块内部基于状态空间模型（SSM）的思想对整个序列进行高效混合。
      4. 从混合后的序列中提取更新后的 class token，再经过投影和 dropout 得到最终输出，
         而 patch tokens 部分保持不变。
    """
    def __init__(self,
                 dim,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 bias=False,
                 use_fast_path=True,
                 drop=0.,
                 num_heads=8,
                 qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
        )

        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, inference_params=None):
        cls_token = x[:, :1, :]  # (B, 1, dim)
        patch_tokens = x[:, 1:, :]  # (B, N-1, dim)
        q = self.q(cls_token)  # (B, 1, dim)
        tokens = torch.cat([q, patch_tokens], dim=1)
        tokens = self.mamba(tokens, inference_params=inference_params)
        updated_cls = tokens[:, :1, :]
        updated_cls = self.proj(updated_cls)
        updated_cls = self.dropout(updated_cls)
        out = torch.cat([updated_cls, patch_tokens], dim=1)
        return out


class ClassRNN(nn.Module):
    """
    基于 RNN 的 token mixer 模块。

    参数:
        dim (int): token 的特征维度，输入和输出通道数均为 dim。
        drop (float): dropout 概率。
        num_layers (int, 可选): RNN 的层数，默认1层。
        bidirectional (bool, 可选): 是否采用双向 RNN，默认 False。
    """

    def __init__(self, dim, drop=0.0, num_layers=1, bidirectional=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

        self.rnn = nn.RNN(
            input_size=dim,
            hidden_size=dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        # 如果采用双向 RNN，则输出通道会变为 2*dim，因此需要投影回 dim
        if bidirectional:
            self.proj = nn.Linear(2 * dim, dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """
        x: Tensor, shape (B, tokens, dim)
        返回: Tensor, shape (B, tokens, dim)
        """
        # 通过 RNN 层（返回全部序列输出和隐藏状态，我们这里只用序列输出）
        out, _ = self.rnn(x)  # out 的形状为 (B, tokens, dim) 或 (B, tokens, 2*dim)
        out = self.proj(out)
        out = self.drop(out)
        return out


class ClassAttention(nn.Module):
    """
    使用传统 attention 更新 class token（假设 x[:,0] 为 class token）。
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        cls_token = x[:, :1, :]
        tokens = x[:, 1:, :]
        B = x.shape[0]
        q = self.q(cls_token).reshape(B, 1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        k = self.k(tokens).reshape(B, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        v = self.v(tokens).reshape(B, -1, self.num_heads, self.dim // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, 1, self.dim)
        out = self.proj(out)
        out = self.dropout(out)
        x = torch.cat([out, tokens], dim=1)
        return x
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Scale(nn.Module):
    """Learnable scaling factor."""
    def __init__(self, dim, init_value=1e-6):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(1, 1, dim))

    def forward(self, x):
        return x * self.scale


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

###############################################################################
# 辅助模块：用于 3D 特征图的 LayerNorm（归一化 channel 维度）
###############################################################################
class LayerNorm3d(nn.Module):
    """
    对 3D 特征图 (B, C, D, H, W) 使用 LayerNorm：
      先将通道维移到最后，然后归一化，再恢复原来顺序。
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, D, H, W) -> (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        # 恢复为 (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3)
        return x
