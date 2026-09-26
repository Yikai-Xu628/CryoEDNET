"""Network reconstructed from manuscript Sections 3.1--3.4.

Input masking is an external ASPIRE preparation step. No mask, normalization,
clipping, extra DnCNN trunk or internal intensity transform is applied here.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MultiScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(64, 32, 3, padding=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
        )
        self.branch3 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.fusion = nn.Conv2d(112, 64, 3, padding=1)

    def forward(self, x):
        return self.fusion(torch.cat([
            self.branch1(x), self.branch2(x), self.branch3(x)
        ], dim=1))


class EdgeEnhancement(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("laplacian", torch.tensor(
            [[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]
        ).reshape(1, 1, 3, 3))
        self.features = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.weight = nn.Sequential(
            nn.Conv2d(64, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )
        self.fusion = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        edge = F.conv2d(x.mean(dim=1, keepdim=True), self.laplacian, padding=1)
        features = self.features(x)
        # The single-channel Laplacian response broadcasts over 64 channels.
        return x + self.fusion(features * self.weight(features) + edge)


class DetailPreservation(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(64, 32, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 1),
        )
        self.weight = nn.Sequential(
            nn.Conv2d(64, 8, 1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.Conv2d(8, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x + self.features(x) * self.weight(x)


class CryoEDNet(nn.Module):
    """Residual denoiser accepting (B,1,H,W), also compatible with DeepInv.

    Ablations bypass individual modules without parameter-count matching.
    Old depth=18/nf=144 EnhancedDnCNN checkpoints are NOT compatible.
    """

    def __init__(self, use_multiscale=True, use_edge=True, use_detail=True):
        super().__init__()
        self.config = dict(use_multiscale=use_multiscale, use_edge=use_edge,
                           use_detail=use_detail)
        self.input_conv = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU())
        self.multiscale = MultiScale() if use_multiscale else nn.Identity()
        self.edge = EdgeEnhancement() if use_edge else nn.Identity()
        self.detail = DetailPreservation() if use_detail else nn.Identity()
        self.output_conv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, physics=None, **kwargs):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected (batch, 1, height, width).")
        f = self.detail(self.edge(self.multiscale(self.input_conv(x))))
        return x - self.output_conv(f)
