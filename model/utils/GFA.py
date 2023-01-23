import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, to_2tuple
import math


class CrossAttention(nn.Module):
    def __init__(self, cfg, stage):
        super().__init__()

        self.dim = cfg.dims[stage]
        self.num_heads = cfg.num_heads[stage]
        self.head_dim = self.dim // self.num_heads
        self.sr_ratio = cfg.sr_ratios[stage]
        assert self.dim % self.num_heads == 0, f"dim {self.dim} should be divided by num_heads {self.num_heads}."
        self.scale = self.head_dim ** -0.5

        self.rgb_q = nn.Linear(self.dim, self.dim, bias=cfg.qkv_bias)
        self.rgb_kv = nn.Linear(self.dim, self.dim * 2, bias=cfg.qkv_bias)
        self.depth_q = nn.Linear(self.dim, self.dim, bias=cfg.qkv_bias)
        self.depth_kv = nn.Linear(self.dim, self.dim * 2, bias=cfg.qkv_bias)
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(cfg.proj_drop)
        if self.sr_ratio > 1:
            self.rgb_sr = nn.Conv2d(self.dim, self.dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.rgb_norm = nn.LayerNorm(self.dim)
            self.depth_sr = nn.Conv2d(self.dim, self.dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.depth_norm = nn.LayerNorm(self.dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        rgb, depth = x.chunk(2, 1)
        rgb_q = self.rgb_q(rgb).reshape(B, N // 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        depth_q = self.depth_q(depth).reshape(B, N // 2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            rgb_ = rgb.permute(0, 2, 1).reshape(B, C, H, W)
            rgb_ = self.rgb_sr(rgb_).reshape(B, C, -1).permute(0, 2, 1)
            rgb_ = self.rgb_norm(rgb_)
            rgb_kv = self.rgb_kv(rgb_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            depth_ = depth.permute(0, 2, 1).reshape(B, C, H, W)
            depth_ = self.depth_sr(depth_).reshape(B, C, -1).permute(0, 2, 1)
            depth_ = self.depth_norm(depth_)
            depth_kv = self.depth_kv(depth_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
        else:
            rgb_kv = self.rgb_kv(rgb).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            depth_kv = self.depth_kv(depth).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                           4)
        rgb_k, rgb_v = rgb_kv[0], rgb_kv[1]
        depth_k, depth_v = depth_kv[0], depth_kv[1]

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


class GFA(nn.Module):
    def __init__(self, cfg, stage):
        super(GFA, self).__init__()
        self.dim = cfg.trans.dims[stage]
        self.rgb_class = nn.Parameter(torch.zeros(size=(1, 1, self.dim)))
        self.depth_class = nn.Parameter(torch.zeros(size=(1, 1, self.dim)))
        self.pos_encode = nn.Conv2d(self.dim * 2, self.dim * 2, to_2tuple(3), to_2tuple(1), to_2tuple(1),
                                    groups=self.dim * 2)
        self.ln_norm1 = nn.LayerNorm(self.dim, eps=1e-6)
        self.attn = CrossAttention(cfg.trans.attn, stage)

        self.proj = nn.Conv2d(self.dim * 2, self.dim, to_2tuple(1), bias=False)
        self.residual = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim, to_2tuple(1), bias=False),
            nn.Conv2d(self.dim, self.dim, to_2tuple(3), to_2tuple(1), to_2tuple(1), groups=self.dim,
                      bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim, self.dim, to_2tuple(1), bias=False),
            nn.BatchNorm2d(self.dim)
        )
        self.bn_norm = nn.BatchNorm2d(self.dim)
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
        B, C, H, W = rgb.shape
        x = torch.cat([rgb, depth], dim=1)
        residual = x
        x = self.pos_encode(x) + x
        rgb_ = x.chunk(2, 1)[0].reshape(B, C, -1).permute(0, 2, 1)
        depth_ = x.chunk(2, 1)[1].reshape(B, C, -1).permute(0, 2, 1)
        x = torch.cat([rgb_ + self.rgb_class, depth_ + self.depth_class], dim=1)
        x = self.ln_norm1(self.attn(x, H, W) + x)
        x = x.permute(0, 2, 1).reshape(B, C * 2, H, W)
        out = self.bn_norm(self.proj(x) + self.residual(residual))

        return out
