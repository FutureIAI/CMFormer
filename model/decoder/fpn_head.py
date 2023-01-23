import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize
import torch
import torch.nn.functional as F


class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, cfg):
        super(FPNHead, self).__init__()
        # -----------
        self.feature_strides = cfg.fpn_head.feature_strides
        self.input_transform = 'multiple_select'
        self.in_channels = cfg.fpn_head.dims
        self.channels = cfg.fpn_head.channels
        self.align_corners = False
        self.dropout_ratio = cfg.fpn_head.dropout_ratio
        self.in_index = cfg.fpn_head.in_index
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None
        self.conv_seg = nn.Conv2d(self.channels, cfg.classes, kernel_size=(1, 1))
        # -----------
        assert len(self.feature_strides) == len(self.in_channels)
        assert min(self.feature_strides) == self.feature_strides[0]

        self.scale_heads = nn.ModuleList()
        for i in range(len(self.feature_strides)):
            head_length = max(
                1,
                int(np.log2(self.feature_strides[i]) - np.log2(self.feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        norm_cfg=dict(type='BN', requires_grad=True)))
                if self.feature_strides[i] != self.feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)
        output = F.interpolate(output, size=(output.shape[2] * 4, output.shape[3] * 4), mode='bilinear',
                               align_corners=True)
        return output
