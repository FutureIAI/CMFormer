import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule


# from mmseg.ops import resize


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        # self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=(1, 1))
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    def __init__(self, cfg):
        super(SegFormerHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = cfg.trans.dims
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=cfg.SegFormer_head.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=cfg.SegFormer_head.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=cfg.SegFormer_head.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=cfg.SegFormer_head.embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=cfg.SegFormer_head.embedding_dim * 4,
            out_channels=cfg.SegFormer_head.embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.drop_out = nn.Dropout(cfg.SegFormer_head.drop_out)
        self.linear_pred = nn.Conv2d(cfg.SegFormer_head.embedding_dim, cfg.classes, kernel_size=(1, 1))

    def forward(self, x):  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        # _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # _c1 = self.linear_c1(c1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.drop_out(_c)
        x = self.linear_pred(x)
        x = F.interpolate(x, size=(c1.shape[2] * 4, c1.shape[3] * 4), mode='bilinear', align_corners=False)
        return x
