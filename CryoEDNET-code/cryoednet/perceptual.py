"""Reconstructed lightweight extractor and VGG19 feature distillation.

No historical distillation program or weights were recovered. Distillation
settings are defined in distill.py and saved with each new training run.
"""

import torch
from torch import nn
from torch.nn import functional as F


class CBAM(nn.Module):
    def __init__(self, channels=128, ratio=8):
        super().__init__()
        self.channel = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        c = self.channel(F.adaptive_avg_pool2d(x, 1))
        c = c + self.channel(F.adaptive_max_pool2d(x, 1))
        x = x * c.sigmoid()
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * self.spatial(s).sigmoid()


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # The manuscript specifies Conv/ReLU, not the historical snippet's BN.
        self.block1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.cbam = CBAM()
        self.residual = nn.Conv2d(1, 128, 1)

    def stages(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.cbam(self.block3(f2)) + self.residual(x)
        return f1, f2, f3

    def forward(self, x):
        return self.stages(x)[-1]


class VGGTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        # Explicit pretrained weights. No fallback to a random teacher.
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:18]
        self.register_buffer("mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, grayscale_01):
        x = (grayscale_01.repeat(1, 3, 1, 1) - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (3, 8, 17):  # relu1_2, relu2_2, relu3_4
                out.append(x)
        return out


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.adapters = nn.ModuleList([
            nn.Conv2d(32, 64, 1), nn.Conv2d(64, 128, 1), nn.Conv2d(128, 256, 1)
        ])

    def forward(self, student_input, teacher_input):
        targets = self.teacher(teacher_input)
        losses = []
        for adapter, feat, target in zip(self.adapters, self.student.stages(student_input), targets):
            feat = F.adaptive_avg_pool2d(adapter(feat), target.shape[-2:])
            losses.append(F.l1_loss(feat, target))
        return torch.stack(losses).mean()


def load_extractor(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "cryoednet-distilled-v1":
        raise ValueError("Expected a checkpoint from distill.py, not random or legacy weights.")
    if checkpoint.get("teacher") != "VGG19_IMAGENET1K_V1" or checkpoint.get("steps", 0) < 1:
        raise ValueError("Checkpoint does not record completed VGG19 distillation steps.")
    student = FeatureExtractor()
    student.load_state_dict(checkpoint["state_dict"], strict=True)
    student.requires_grad_(False)
    student.eval()
    return student, checkpoint
