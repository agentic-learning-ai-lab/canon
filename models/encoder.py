"""
ResNet-18 visual encoder variants for CANON.

resnet18         : standard GAP encoder (512-D flat output). Used for LIBERO Goal.
resnet18_spatial : layer4 spatial-map encoder (gap + spatial tokens). Used for MetaWorld.
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 512,
        unit_norm: bool = False,
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unit_norm = unit_norm

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = F.normalize(out, p=2, dim=-1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        return out


class resnet18_spatial(resnet18):
    """ResNet-18 that returns (gap, spatial) for the SO(3) angle head.

    gap     : [N, 512]        global average pool output (same as resnet18)
    spatial : [N, 512, H, W]  layer4 feature map before avgpool

    At 224×224 input: H=W=7. The SO(3) angle head uses spatial for conv-based
    channel reduction (512→128) followed by flatten and linear projection.
    """

    def forward(self, x: torch.Tensor):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.normalize(x)
        # Sequential: conv1(0) bn1(1) relu(2) maxpool(3) layer1(4) layer2(5) layer3(6) layer4(7) avgpool(8)
        for i in range(8):
            x = self.resnet[i](x)
        spatial = x             # [N, 512, H, W]
        x = self.resnet[8](x)   # avgpool → [N, 512, 1, 1]
        gap = self.flatten(x)   # [N, 512]
        if self.unit_norm:
            gap = F.normalize(gap, p=2, dim=-1)
        if dims == 3:
            gap = gap.squeeze(0)
            spatial = spatial.squeeze(0)
        elif dims > 4:
            H, W = spatial.shape[-2:]
            gap     = gap.reshape(*orig_shape[:-3], -1)
            spatial = spatial.reshape(*orig_shape[:-3], 512, H, W)
        return gap, spatial
