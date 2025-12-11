import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from einops import rearrange

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from lib.models.mymodules.former2d import Mlp, Scale, DropPath, LayerNorm2d, ClassMamba, ClassAttention, ClassRNN

# 调用 MONAI 的模块
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock


###############################################################################
# 辅助函数：窗口划分与恢复
###############################################################################
def window_partition(x, window_size):
    """
    将输入的 4D tensor (B, H, W, C) 按照 window_size 划分为局部窗口
    返回形状为 (num_windows*B, window_h*window_w, C)
    """
    B, H, W, C = x.shape
    wh, ww = window_size
    x = x.view(
        B,
        H // wh, wh,
        W // ww, ww,
        C
    )
    # 先交换维度，再合并局部窗口为一个维度
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = x.view(-1, wh * ww, C)
    return windows


def window_reverse(windows, window_size, B, H, W, C):
    """
    将划分后的窗口恢复为原始的 4D tensor (B, H, W, C)
    """
    wh, ww = window_size
    x = windows.view(
        B,
        H // wh,
        W // ww,
        wh, ww,
        C
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    return x


###############################################################################
# MetaFormerBlock（修改后，可精确控制剪枝）
###############################################################################
class MetaFormerBlock(nn.Module):
    def __init__(self, dim,
                 token_mixer=ClassAttention,
                 mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 res_scale_init_value=None,
                 prune_ln2_mlp=False,
                 prune_ln1_token_mixer=False,
                 **token_mixer_kwargs):
        super().__init__()
        self.prune_ln1_token_mixer = prune_ln1_token_mixer
        self.prune_ln2_mlp = prune_ln2_mlp

        # 如果剪枝 LN1+token_mixer，则将 norm1 和 token_mixer 均替换为 Identity
        self.norm1 = nn.Identity() if prune_ln1_token_mixer else norm_layer(dim)
        if prune_ln1_token_mixer:
            self.token_mixer = nn.Identity()
        else:
            if token_mixer == ClassMamba:
                d_state = token_mixer_kwargs.pop('d_state', 32)
                self.token_mixer = token_mixer(
                    dim=dim,
                    drop=drop,
                    d_state=d_state,
                    **token_mixer_kwargs
                )
            elif token_mixer == ClassAttention:
                num_heads = token_mixer_kwargs.pop('num_heads', 12)
                self.token_mixer = token_mixer(
                    dim=dim,
                    drop=drop,
                    num_heads=num_heads,
                    **token_mixer_kwargs
                )
            elif token_mixer == ClassRNN:
                token_mixer_kwargs.pop('num_heads', None)
                self.token_mixer = ClassRNN(
                    dim=dim,
                    drop=drop,
                    **token_mixer_kwargs
                )
            elif token_mixer.__name__ == "ClassLSTM":
                token_mixer_kwargs.pop("num_heads", None)
                token_mixer_kwargs.pop("window_size", None)
                self.token_mixer = token_mixer(dim=dim, drop=drop, **token_mixer_kwargs)
            else:
                self.token_mixer = token_mixer(dim=dim, drop=drop, **token_mixer_kwargs)

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale1 = Scale(dim=dim,
                                init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()

        # LN2 和 MLP 的剪枝控制
        self.norm2 = nn.Identity() if prune_ln2_mlp else norm_layer(dim)
        self.mlp = nn.Identity() if prune_ln2_mlp else mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale2 = Scale(dim=dim,
                                init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + self.drop_path1(self.token_mixer(self.norm1(x)))
        if not self.prune_ln2_mlp:
            x = self.res_scale2(x) + self.drop_path2(self.mlp(self.norm2(x)))
        return x


###############################################################################
# Patch Merging V2：基于 2×2 patch 合并，下采样空间尺寸同时翻倍通道数
# 当 small_input 模式下，我们不会使用该模块
###############################################################################
class PatchMergingV2(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, spatial_dims=2):
        """
        Args:
            dim: 输入通道数
            norm_layer: 归一化层
            spatial_dims: 2D
        """
        super().__init__()
        self.dim = dim
        # 对于 2D，下采样时将 2×2 个 patch 拼接，通道数由 dim 变为 4*dim，然后线性映射到 2*dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        # 分别采样 4 个相邻 patch
        x0 = x[:, 0::2, 0::2, :]  # (B, H//2, W//2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H//2, W//2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H//2, W//2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H//2, W//2, C)
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H//2, W//2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


###############################################################################
# EncoderStage：增加 prune_flags 参数以便精确控制每层的剪枝，同时支持 small_input 模式
###############################################################################
class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, token_mixer,
                 num_layers=2, drop=0.0, drop_path=0.0, window_size=(2, 2),
                 prune_flags=None, small_input=False):
        """
        Args:
            prune_flags: 可选列表，每个元素为 dict，包含键 "prune_ln2_mlp" 和 "prune_ln1_token_mixer"
                         列表长度应与该阶段的 MetaFormerBlock 数量相同。
            small_input: 如果为 True，则不进行空间下采样，只用 1×1 卷积调整通道数，同时 window_size 设为 (1,1)
        """
        super().__init__()
        self.small_input = small_input
        # 当 small_input 为 True，则 window_size 设为 (1,1) 避免空间降采样
        self.window_size = (1, 1) if small_input else window_size
        blocks = []
        for i in range(num_layers):
            if prune_flags is not None and i < len(prune_flags):
                pf = prune_flags[i]
                prune_ln2_mlp = pf.get("prune_ln2_mlp", False)
                prune_ln1_token_mixer = pf.get("prune_ln1_token_mixer", False)
            else:
                prune_ln2_mlp = False
                prune_ln1_token_mixer = False
            block = MetaFormerBlock(dim=in_channels,
                                    token_mixer=token_mixer,
                                    mlp=Mlp,
                                    norm_layer=nn.LayerNorm,
                                    drop=drop,
                                    drop_path=drop_path,
                                    res_scale_init_value=1.0,
                                    num_heads=num_heads,
                                    prune_ln2_mlp=prune_ln2_mlp,
                                    prune_ln1_token_mixer=prune_ln1_token_mixer)
            blocks.append(block)
        self.metaformer = nn.Sequential(*blocks)
        if not small_input:
            # 使用 Patch Merging 下采样：将 (B, H, W, C) -> (B, H//2, W//2, 2*C)
            self.patch_merging = PatchMergingV2(dim=in_channels, norm_layer=nn.LayerNorm, spatial_dims=2)
            if out_channels != 2 * in_channels:
                self.channel_proj = nn.Conv2d(2 * in_channels, out_channels, kernel_size=1)
            else:
                self.channel_proj = nn.Identity()
        else:
            # small_input 模式下不做空间下采样，直接用 1×1 卷积实现通道数的调整
            self.patch_merging = None
            if out_channels != in_channels:
                self.channel_proj_small = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.channel_proj_small = nn.Identity()
        # skip 分支：用 3×3 卷积（UnetrBasicBlock）保持空间尺寸和通道数，作为后续解码器的跳跃连接
        self.skip_conv = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True
        )

    def forward(self, x):
        # 输入 x: (B, C, H, W)，其中 C = in_channels
        B, C, H, W = x.shape
        # 调整为 (B, H, W, C) 进行 window partition
        x = rearrange(x, "b c h w -> b h w c")
        windows = window_partition(x, self.window_size)  # (num_windows*B, window_volume, C)
        windows = self.metaformer(windows)
        x = window_reverse(windows, self.window_size, B, H, W, C)  # (B, H, W, C)
        # skip 分支：恢复至 (B, C, H, W) 后用 3×3 卷积（保持通道数不变）
        x_skip = rearrange(x, "b h w c -> b c h w")
        x_skip = self.skip_conv(x_skip)
        if not self.small_input:
            # 下采样：先用 Patch Merging，再用 1x1 卷积调整（如需要）
            x_down = self.patch_merging(x)  # (B, H//2, W//2, 2*C)
            x_down = rearrange(x_down, "b h w c -> b c h w")
            x_down = self.channel_proj(x_down)
        else:
            # small_input 模式下，不下采样，直接用 1×1 卷积调整通道数
            x_down = rearrange(x, "b h w c -> b c h w")
            x_down = self.channel_proj_small(x_down)
        return x_down, x_skip


###############################################################################
# PMFSNet 主网络（去掉 encoder4 和 decoder4）- 2D 版本
###############################################################################
class SwinUNETR(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=35,
                 patch_size=2,  # 默认 2×2 的 patch
                 base_channels=48,  # PatchEmbed 输出通道数
                 num_heads=(4, 8, 16, 32),
                 token_mixer=ClassMamba,
                 drop=0.2,
                 drop_path=0.2,
                 scaling_version="TINY",
                 metaformer_layers=[2, 2, 2, 2],  # 去掉了第四阶段
                 # 各 EncoderStage 的剪枝配置字典列表（这里仅对前三阶段进行配置）
                 prune_flags_list=[
                                   [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False},
                                     {"prune_ln2_mlp": False, "prune_ln1_token_mixer": False}
                                   ],
                                    [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False},
                                     {"prune_ln2_mlp": False, "prune_ln1_token_mixer": False}
                                    ],
                                   [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False},
                                    {"prune_ln2_mlp": False, "prune_ln1_token_mixer": False}
                                    ],
                                   [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False},
                                    {"prune_ln2_mlp": False, "prune_ln1_token_mixer": False}
                                    ]
                                   ],
                 small_input=False   # 若 True，则支持最小空间尺寸为 4（例如最后一维为4）????????
                 ):
        super().__init__()
        self.small_input = small_input
        # 1. Patch embedding：若 small_input 则使用 patch_size=1 保持分辨率
        patch_size_tuple = (1, 1) if small_input else (patch_size, patch_size)
        self.patch_embed = PatchEmbed(
            patch_size=patch_size_tuple,
            in_chans=in_channels,
            embed_dim=base_channels,
            norm_layer=nn.LayerNorm,
            spatial_dims=2
        )
        self.pos_drop = nn.Dropout(p=drop)

        # 若未提供剪枝配置，则各阶段均不剪枝
        if prune_flags_list is None:
            prune_flags_list = [None, None, None]

        # 2. Encoder 阶段（3 层，每层下采样同时翻倍通道数）
        self.encoder1 = EncoderStage(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            num_heads=num_heads[0],
            token_mixer=token_mixer,
            num_layers=metaformer_layers[0],
            drop=drop,
            drop_path=drop_path,
            window_size=(2, 2),
            prune_flags=prune_flags_list[0],
            small_input=small_input
        )
        self.encoder2 = EncoderStage(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            num_heads=num_heads[1],
            token_mixer=token_mixer,
            num_layers=metaformer_layers[1],
            drop=drop,
            drop_path=drop_path,
            window_size=(2, 2),
            prune_flags=prune_flags_list[1],
            small_input=small_input
        )
        self.encoder3 = EncoderStage(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
            num_heads=num_heads[2],
            token_mixer=token_mixer,
            num_layers=metaformer_layers[2],
            drop=drop,
            drop_path=drop_path,
            window_size=(2, 2),
            prune_flags=prune_flags_list[2],
            small_input=small_input
        )
        self.encoder4 = EncoderStage(
            in_channels=base_channels * 8,
            out_channels=base_channels * 16,
            num_heads=num_heads[3],
            token_mixer=token_mixer,
            num_layers=metaformer_layers[3],
            drop=drop,
            drop_path=drop_path,
            window_size=(2, 2),
            prune_flags=prune_flags_list[3],
            small_input=small_input
        )
        # 3. Bottleneck：对 encoder3 下采样输出（通道数 base_channels*8）进行处理，保持通道数不变
        self.bottleneck = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=base_channels * 16,
            out_channels=base_channels * 16,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True
        )

        # 4. Decoder 阶段：逐步上采样并与对应的 skip 分支融合
        # 若 small_input，则不进行空间上采样，因此 upsample_kernel_size 设为1
        up_kernel = 1 if small_input else 2
        norm_name = "instance"
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=base_channels * 16,
            out_channels=base_channels * 8,
            kernel_size=3,
            upsample_kernel_size=up_kernel,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=base_channels * 8,
            out_channels=base_channels * 4,
            kernel_size=3,
            upsample_kernel_size=up_kernel,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            kernel_size=3,
            upsample_kernel_size=up_kernel,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=3,
            upsample_kernel_size=up_kernel,
            norm_name=norm_name,
            res_block=True
        )

        # 5. Segmentation Head：将 decoder1 输出映射到最终类别数
        self.seg_head = UnetOutBlock(
            spatial_dims=2,
            in_channels=base_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        # print("x shape:",x.shape)
        # 输入 x: (B, in_channels, H, W)
        # 1. Patch embedding 得到最高分辨率特征 x0（skip0）
        x0 = self.patch_embed(x)  # (B, base_channels, H0, W0)
        x0 = self.pos_drop(x0)

        # 2. Encoder 阶段，每层返回 (下采样输出, skip)
        x1, skip1 = self.encoder1(x0)       # x1: base_channels*2, skip1: base_channels
        x2, skip2 = self.encoder2(x1)       # x2: base_channels*4, skip2: base_channels*2
        x3, skip3 = self.encoder3(x2)       # x3: base_channels*8, skip3: base_channels*4
        x4, skip4 = self.encoder4(x3)
        x4 = self.bottleneck(x4)

        # 3. Decoder 阶段：逐层上采样并融合对应 skip 分支
        d4 = self.decoder4(x4, skip4)
        d3 = self.decoder3(d4, skip3)       # d3: base_channels*4
        d2 = self.decoder2(d3, skip2)       # d2: base_channels*2
        d1 = self.decoder1(d2, skip1)       # d1: base_channels

        seg = self.seg_head(d1)
        seg_up = F.interpolate(seg, size=x.shape[2:], mode='bilinear', align_corners=True)
        return seg_up


###### test ######
if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 测试两种情形：
    # 1. 常规输入
    x1 = torch.randn((1, 1, 256, 256)).to(device)
    model1 = SwinUNETR(in_channels=1, out_channels=4,
                    patch_size=2, base_channels=48, num_heads=(4, 8, 16, 32),
                    token_mixer=ClassMamba,
                    small_input=False
                    ).to(device)
    y1 = model1(x1)
    print("常规模式:")
    print("Input size:", x1.size())
    print("Output size:", y1.size())
    print("Params: {:.6f}M".format(count_parameters(model1)))


    # # 可选：计算 FLOPs 和参数量
    from thop import profile
    macs, params = profile(model1, inputs=(x1,), verbose=False)
    flops = 2 * macs  # FLOPs 通常为 2 * MACs
    print("常规模式 MACs: {:.3f} GFLOPs".format(macs / 1e9))
    print("常规模式 FLOPs: {:.3f} GFLOPs".format(flops / 1e9))