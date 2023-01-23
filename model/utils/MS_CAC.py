import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import math


class SPP(nn.Module):
    def __init__(self, cfg):
        super(SPP, self).__init__()
        self.target_size = cfg.target_size
        for i in range(len(self.target_size)):
            setattr(self, 'avg_pool{}'.format(i), nn.AdaptiveAvgPool2d(self.target_size[i]))
            setattr(self, 'max_pool{}'.format(i), nn.AdaptiveMaxPool2d(self.target_size[i]))

    def forward(self, x):
        avg_scales = []
        max_scales = []
        for i in range(len(self.target_size)):
            avg_scale = getattr(self, 'avg_pool{}'.format(i))(x)
            max_scale = getattr(self, 'max_pool{}'.format(i))(x)
            avg_scale = rearrange(avg_scale, 'b c h w -> b c (h w)')
            max_scale = rearrange(max_scale, 'b c h w -> b c (h w)')
            avg_scales.append(avg_scale)
            max_scales.append(max_scale)
        x = torch.cat(avg_scales + max_scales, dim=2)
        return x


class I2V(nn.Module):
    def __init__(self, cfg, stage):
        super(I2V, self).__init__()
        self.spp = SPP(cfg)
        self.proj = nn.Linear(in_features=cfg.spp_dims * 2, out_features=cfg.out_dims[stage])
        self.rgb_class = nn.Parameter(torch.zeros(size=(1, 1, cfg.out_dims[stage])))
        self.depth_class = nn.Parameter(torch.zeros(size=(1, 1, cfg.out_dims[stage])))

    def forward(self, x):
        x = self.spp(x)
        x = self.proj(x)
        rgb, depth = x.chunk(2, 1)
        rgb = rgb + self.rgb_class
        depth = depth + self.depth_class
        x = torch.cat([rgb, depth], dim=1)
        return x


class Mlp(nn.Module):
    def __init__(self, cfg, stage):
        super().__init__()
        self.in_features = cfg.in_features[stage]
        self.hidden_features = cfg.hidden_features[stage]
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_features, self.in_features)
        self.drop = nn.Dropout(cfg.drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, cfg, stage):
        super().__init__()

        self.dim = cfg.dims[stage]
        self.num_heads = cfg.num_heads[stage]
        self.head_dim = self.dim // self.num_heads
        assert self.dim % self.num_heads == 0, f"dim {self.dim} should be divided by num_heads {self.num_heads}."
        self.scale = self.head_dim ** -0.5

        self.rgb_qkv = nn.Linear(self.dim, self.dim * 3, bias=cfg.qkv_bias)
        self.depth_qkv = nn.Linear(self.dim, self.dim * 3, bias=cfg.qkv_bias)
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(cfg.proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        rgb, depth = x.chunk(2, 1)
        rgb_qkv = self.rgb_qkv(rgb).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        rgb_q, rgb_k, rgb_v = rgb_qkv[0], rgb_qkv[1], rgb_qkv[2]
        depth_qkv = self.depth_qkv(depth).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        depth_q, depth_k, depth_v = depth_qkv[0], depth_qkv[1], depth_qkv[2]

        depth_attn = (rgb_q @ depth_k.transpose(-2, -1)) * self.scale
        depth_attn = depth_attn.softmax(dim=-1)
        depth_attn = self.attn_drop(depth_attn)
        depth_v = (depth_attn @ depth_v).transpose(1, 2).reshape(B, N // 2, C)

        rgb_attn = (depth_q @ rgb_k.transpose(-2, -1)) * self.scale
        rgb_attn = rgb_attn.softmax(dim=-1)
        rgb_attn = self.attn_drop(rgb_attn)
        rgb_v = (rgb_attn @ rgb_v).transpose(1, 2).reshape(B, N // 2, C)
        x = torch.cat([rgb_v, depth_v], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, cfg, stage, dpr):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dims[stage], eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.dims[stage], eps=1e-6)
        self.attn = CrossAttention(cfg.attn, stage)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.mlp = Mlp(cfg.mlp, stage)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MS_CAC(nn.Module):
    def __init__(self, cfg, stage):
        super(MS_CAC, self).__init__()
        self.i2v = I2V(cfg.I2V, stage)
        dpr = [x.item() for x in torch.linspace(0, cfg.trans.drop_path, sum(cfg.trans.blocks))]
        cur = sum(cfg.trans.blocks[:stage])
        self.blocks = nn.ModuleList([Block(cfg.trans, stage=stage, dpr=dpr[cur + j])
                                     for j in range(cfg.trans.blocks[stage])])
        self.proj = nn.Sequential(
            nn.Linear(cfg.I2V.out_dims[stage], cfg.I2V.out_dims[stage]),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.I2V.out_dims[stage], 1),
            nn.Sigmoid())
        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward(self, rgb, depth):
        B, C, _, _ = rgb.shape
        x = torch.cat([rgb, depth], dim=1)
        x = self.i2v(x)
        for block in self.blocks:
            x = block(x)

        x = self.proj(x).view(B, 2, C, 1, 1).permute(1, 0, 2, 3, 4)
        rgb_attn, depth_attn = x[0], x[1]
        rgb_out = rgb + depth * depth_attn
        depth_out = depth + rgb * rgb_attn

        return rgb_out, depth_out
