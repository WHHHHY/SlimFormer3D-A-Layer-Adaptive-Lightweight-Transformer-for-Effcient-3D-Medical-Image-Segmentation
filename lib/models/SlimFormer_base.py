import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from einops import rearrange

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from lib.models.mymodules.former import Mlp, Scale, DropPath, LayerNorm3d, ClassMamba, ClassAttention, ClassRNN

from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    wd, wh, ww = window_size
    x = x.view(
        B,
        D // wd, wd,
        H // wh, wh,
        W // ww, ww,
        C
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = x.view(-1, wd * wh * ww, C)
    return windows


def window_reverse(windows, window_size, B, D, H, W, C):
    wd, wh, ww = window_size
    x = windows.view(
        B,
        D // wd,
        H // wh,
        W // ww,
        wd, wh, ww,
        C
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, C)
    return x


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

        self.norm1 = nn.Identity() if prune_ln1_token_mixer else norm_layer(dim)
        if prune_ln1_token_mixer:
            self.token_mixer = nn.Identity()
        else:
            if token_mixer == ClassMamba:
                d_state = token_mixer_kwargs.pop('d_state', 16)
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


class PatchMergingV2(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, spatial_dims=3):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        B, D, H, W, C = x.shape
        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 1::2, 0::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, token_mixer,
                 num_layers=2, drop=0.0, drop_path=0.0, window_size=(2, 2, 2), prune_flags=None):
        super().__init__()
        self.window_size = window_size
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
        self.patch_merging = PatchMergingV2(dim=in_channels, norm_layer=nn.LayerNorm, spatial_dims=3)
        if out_channels != 2 * in_channels:
            self.channel_proj = nn.Conv3d(2 * in_channels, out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()
        self.skip_conv = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = rearrange(x, "b c d h w -> b d h w c")
        windows = window_partition(x, self.window_size)
        windows = self.metaformer(windows)
        x = window_reverse(windows, self.window_size, B, D, H, W, C)
        x_skip = rearrange(x, "b d h w c -> b c d h w")
        x_skip = self.skip_conv(x_skip)
        x_down = self.patch_merging(x)
        x_down = rearrange(x_down, "b d h w c -> b c d h w")
        x_down = self.channel_proj(x_down)
        return x_down, x_skip


class PMFSNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=35,
                 patch_size=2,
                 base_channels=48,
                 num_heads=(4, 8, 16, 32),
                 token_mixer=ClassAttention,
                 drop=0.2,
                 drop_path=0.2,
                 scaling_version="TINY",
                 metaformer_layers=[2, 2, 2, 2],
                 prune_flags_list=[ [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False},
                                    {"prune_ln2_mlp": False, "prune_ln1_token_mixer": False}
                                   ],
                                   [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False},
                                    {"prune_ln2_mlp": False, "prune_ln1_token_mixer": False}],
                                    [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False},
                                     {"prune_ln2_mlp": False, "prune_ln1_token_mixer": False}],
                                   [{"prune_ln2_mlp": False, "prune_ln1_token_mixer": False},
                                    {"prune_ln2_mlp": False, "prune_ln1_token_mixer": False}]
                                   ]
                 ):
        super().__init__()
        patch_size_tuple = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.patch_embed = PatchEmbed(
            patch_size=patch_size_tuple,
            in_chans=in_channels,
            embed_dim=base_channels,
            norm_layer=nn.LayerNorm,
            spatial_dims=3
        )
        self.pos_drop = nn.Dropout(p=drop)

        if prune_flags_list is None:
            prune_flags_list = [None, None, None, None]

        self.encoder1 = EncoderStage(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            num_heads=num_heads[0],
            token_mixer=token_mixer,
            num_layers=metaformer_layers[0],
            drop=drop,
            drop_path=drop_path,
            window_size=(2, 2, 2),
            prune_flags=prune_flags_list[0]
        )
        self.encoder2 = EncoderStage(
            in_channels=base_channels * 2,
            out_channels=base_channels * 4,
            num_heads=num_heads[1],
            token_mixer=token_mixer,
            num_layers=metaformer_layers[1],
            drop=drop,
            drop_path=drop_path,
            window_size=(2, 2, 2),
            prune_flags=prune_flags_list[1]
        )
        self.encoder3 = EncoderStage(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
            num_heads=num_heads[2],
            token_mixer=token_mixer,
            num_layers=metaformer_layers[2],
            drop=drop,
            drop_path=drop_path,
            window_size=(2, 2, 2),
            prune_flags=prune_flags_list[2]
        )
        self.encoder4 = EncoderStage(
            in_channels=base_channels * 8,
            out_channels=base_channels * 16,
            num_heads=num_heads[3],
            token_mixer=token_mixer,
            num_layers=metaformer_layers[3],
            drop=drop,
            drop_path=drop_path,
            window_size=(2, 2, 2),
            prune_flags=prune_flags_list[3]
        )

        self.bottleneck = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=base_channels * 16,
            out_channels=base_channels * 16,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True
        )

        norm_name = "instance"
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=base_channels * 16,
            out_channels=base_channels * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=base_channels * 8,
            out_channels=base_channels * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=base_channels * 4,
            out_channels=base_channels * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=base_channels * 2,
            out_channels=base_channels,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True
        )

        self.seg_head = UnetOutBlock(
            spatial_dims=3,
            in_channels=base_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)

        x1, skip1 = self.encoder1(x0)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)

        x4 = self.bottleneck(x4)

        d4 = self.decoder4(x4, skip4)
        d3 = self.decoder3(d4, skip3)
        d2 = self.decoder2(d3, skip2)
        d1 = self.decoder1(d2, skip1)

        seg = self.seg_head(d1)
        seg_up = F.interpolate(seg, size=x.shape[2:], mode='trilinear', align_corners=True)
        return seg_up


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn((1, 1, 96, 96, 96)).to(device)

    model = PMFSNet(in_channels=1, out_channels=4,
                    patch_size=2, base_channels=48, num_heads=(4, 8, 16, 32),
                    token_mixer=ClassAttention).to(device)
    y = model(x)
    print("Input size:", x.size())
    print("Output size:", y.size())
    print("Params: {:.6f}M".format(count_parameters(model)))
