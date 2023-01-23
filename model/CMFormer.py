import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.utils.MS_CAC import MS_CAC
from model.utils.GFA import GFA
from model.decoder.fpn_head import FPNHead
import math
from thop import clever_format
from thop import profile
from get_config import get_config


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, to_2tuple(3), to_2tuple(1), to_2tuple(1), bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, cfg, stage):
        super().__init__()
        self.in_features = cfg.in_features[stage]
        self.hidden_features = cfg.hidden_features[stage]
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.dwconv = DWConv(self.hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_features, self.in_features)
        self.drop = nn.Dropout(cfg.drop)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, cfg, stage):
        super().__init__()

        self.dim = cfg.dims[stage]
        self.num_heads = cfg.num_heads[stage]
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, f"dim {self.dim} should be divided by num_heads {self.num_heads}."
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(self.dim, self.dim, bias=cfg.qkv_bias)
        self.kv = nn.Linear(self.dim, self.dim * 2, bias=cfg.qkv_bias)
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(cfg.proj_drop)

        self.sr_ratio = cfg.sr_ratios[stage]
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(self.dim, self.dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(self.dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, cfg, stage, dpr):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dims[stage], eps=1e-6)
        self.attn = Attention(cfg.attn, stage)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(cfg.dims[stage], eps=1e-6)
        self.mlp = Mlp(cfg.mlp, stage)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, cfg, stage):
        super().__init__()
        self.proj = nn.Conv2d(cfg.channels[stage], cfg.channels[stage + 1],
                              kernel_size=to_2tuple(cfg.kernel_size[stage]),
                              stride=to_2tuple(cfg.stride[stage]),
                              padding=to_2tuple(cfg.padding[stage]))
        self.norm = nn.LayerNorm(cfg.channels[stage + 1])

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_stages = cfg.num_stages
        self.num = 0

        dpr = [x.item() for x in torch.linspace(0, cfg.trans.drop_path, sum(cfg.trans.blocks))]
        cur = 0
        for i in range(cfg.num_stages):
            setattr(self, f"rgb_patch_embed{i + 1}", OverlapPatchEmbed(cfg.embed, stage=i))
            setattr(self, f"rgb_block{i + 1}", nn.ModuleList([Block(cfg.trans, stage=i, dpr=dpr[cur + j])
                                                              for j in range(cfg.trans.blocks[i])]))
            setattr(self, f"rgb_norm{i + 1}", nn.LayerNorm(cfg.trans.dims[i], eps=1e-6))

            if i == 0:
                cfg.embed.channels[i] = 1
            setattr(self, f"depth_patch_embed{i + 1}", OverlapPatchEmbed(cfg.embed, stage=i))
            setattr(self, f"depth_block{i + 1}", nn.ModuleList([Block(cfg.trans, stage=i, dpr=dpr[cur + j])
                                                                for j in range(cfg.trans.blocks[i])]))
            setattr(self, f"depth_norm{i + 1}", nn.LayerNorm(cfg.trans.dims[i], eps=1e-6))
            cur += cfg.trans.blocks[i]

            setattr(self, f"cac{i + 1}", MS_CAC(cfg.MS_CAC, i))
            setattr(self, f"gfa{i + 1}", GFA(cfg.GFA, i))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_features(self, rgb, depth):
        B = rgb.shape[0]
        outs = []
        for i in range(self.num_stages):
            rgb_patch_embed = getattr(self, f"rgb_patch_embed{i + 1}")
            rgb_block = getattr(self, f"rgb_block{i + 1}")
            rgb_norm = getattr(self, f"rgb_norm{i + 1}")

            depth_patch_embed = getattr(self, f"depth_patch_embed{i + 1}")
            depth_block = getattr(self, f"depth_block{i + 1}")
            depth_norm = getattr(self, f"depth_norm{i + 1}")
            rgb, H, W = rgb_patch_embed(rgb)
            depth, _, _ = depth_patch_embed(depth)
            for rgb_blk, depth_blk in zip(rgb_block, depth_block):
                rgb = rgb_blk(rgb, H, W)
                depth = depth_blk(depth, H, W)
            rgb = rgb_norm(rgb)
            depth = depth_norm(depth)
            rgb = rgb.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            depth = depth.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            rgb, depth = getattr(self, f"cac{i + 1}")(rgb, depth)
            out = getattr(self, f"gfa{i + 1}")(rgb, depth)

            outs.append(out)

        return outs

    def forward(self, rgb, depth):
        x = self.forward_features(rgb, depth)

        return x


class CMFormer(nn.Module):
    def __init__(self, cfg):
        super(CMFormer, self).__init__()
        self.encoder = Encoder(cfg)
        self.decoder = FPNHead(cfg)

    def forward(self, rgb, depth):
        outs = self.encoder(rgb, depth)
        pre = self.decoder(outs)
        return pre


if __name__ == '__main__':
    Config = get_config()
    model = CMFormer(cfg=Config).to("cuda")
    model.eval()
    rgb_input = torch.rand(size=(1, 3, 480, 640), device="cuda")
    depth_input = torch.rand(size=(1, 1, 480, 640), device="cuda")

    flops, params = profile(model, inputs=(rgb_input, depth_input))
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:', flops)
    print('params:', params)
