"""Section 3.5 losses. Inputs are already masked externally, no second mask."""

import torch
from torch import nn
from torch.nn import functional as F


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-coords.square() / (2 * sigma**2))
        window = torch.outer(g, g)
        self.register_buffer("window", (window / window.sum())[None, None])
        self.padding = window_size // 2

    def forward(self, a, b):
        def average(x):
            return F.conv2d(x, self.window, padding=self.padding)
        ma, mb = average(a), average(b)
        va, vb = average(a*a) - ma*ma, average(b*b) - mb*mb
        cov = average(a*b) - ma*mb
        score = ((2*ma*mb + .02**2) * (2*cov + .06**2)) / (
            (ma*ma + mb*mb + .02**2) * (va + vb + .06**2)).clamp_min(1e-12)
        return 1 - score.mean()


class CombinedLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor.requires_grad_(False).eval()
        self.ssim = SSIMLoss()

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()
        return self

    def components(self, prediction, target):
        # Gradients must flow through the frozen extractor into prediction.
        p = self.extractor(prediction)
        with torch.no_grad():
            t = self.extractor(target)
        fp = torch.fft.fft2(F.avg_pool2d(prediction, 3, stride=1, padding=1)).abs()
        ft = torch.fft.fft2(F.avg_pool2d(target, 3, stride=1, padding=1)).abs()
        return dict(rec=F.l1_loss(prediction, target), perc=F.mse_loss(p, t),
                    ssim=self.ssim(prediction, target), freq=F.l1_loss(fp, ft))

    def forward(self, x_net, x, **kwargs):
        c = self.components(x_net, x)
        total = c["rec"] + .1*c["perc"] + .5*c["ssim"] + .1*c["freq"]
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite training loss.")
        return total

    def adapt_model(self, model, **kwargs):
        return model
