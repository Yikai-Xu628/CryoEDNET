"""Evaluate paired synthetic images with a fixed, explicitly supplied PSNR peak.

No reference-volume resolution or historical table reproduction is claimed.
SSIM uses the paper's c1=.02^2 and c2=.06^2, without intensity rescaling.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from cryoednet.data import Stack
from cryoednet.losses import SSIMLoss


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", required=True)
    p.add_argument("--prediction", required=True)
    p.add_argument("--peak", required=True, type=float,
                   help="Fixed PSNR peak shared across all methods; not the maximum of each output.")
    p.add_argument("--output", required=True, help="New per-image CSV.")
    args = p.parse_args(argv)
    if not np.isfinite(args.peak) or args.peak <= 0:
        p.error("peak must be finite and positive.")
    clean, predicted = Stack(args.clean), Stack(args.prediction)
    if clean.data.shape != predicted.data.shape:
        raise ValueError("Evaluation stacks must have identical shapes and ordering.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ssim = SSIMLoss()
    with open(out, "x", newline="", encoding="utf-8") as f, torch.inference_mode():
        writer = csv.writer(f)
        writer.writerow(["index", "mse", "psnr_db", "ssim", "psnr_peak"])
        for i in range(len(clean)):
            a, b = clean[i], predicted[i]
            mse = float(np.mean(np.square(a.astype(np.float64)-b)))
            psnr = float("inf") if mse == 0 else 10*np.log10(args.peak**2/mse)
            score = 1-float(ssim(torch.from_numpy(a)[None, None], torch.from_numpy(b)[None, None]))
            writer.writerow([i, mse, psnr, score, args.peak])
    clean.close()
    predicted.close()
    print(f"Saved {out}; individual images are not independent training seeds.")


if __name__ == "__main__":
    main()
