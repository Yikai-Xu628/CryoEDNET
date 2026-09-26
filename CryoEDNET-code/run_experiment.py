"""Run the explicitly authorized as-provided data trial, then evaluate once.

The held-out test set is never used to choose parameters, epochs or checkpoints.
All scientific results belong to this reconstructed implementation and dataset
preparation, not to a verified reproduction of the manuscript tables.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch

from cryoednet.data import Stack, stack_stats
from cryoednet.losses import SSIMLoss
from cryoednet.utils import environment, new_output_dir, sha256, write_json


def now():
    return datetime.now().astimezone().isoformat()


def assess(clean_path, noisy_path, prediction_path, out, peak):
    """Report full-frame metrics and fixed-index, shared-scale visual comparisons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stacks = [Stack(p) for p in (clean_path, noisy_path, prediction_path)]
    clean, noisy, prediction = stacks
    if any(s.data.shape != clean.data.shape for s in stacks):
        raise ValueError("Test clean/noisy/prediction shapes differ.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ssim = SSIMLoss().to(device)
    scores = {"noisy": [], "denoised": []}
    per_image = []
    try:
        with torch.inference_mode():
            for i in range(len(clean)):
                reference = clean[i]
                ref_tensor = torch.from_numpy(reference)[None, None].to(device)
                row = [i]
                for label, stack in (("noisy", noisy), ("denoised", prediction)):
                    image = stack[i]
                    mse = float(np.mean((image.astype(np.float64)-reference)**2))
                    psnr = float("inf") if mse == 0 else 10*np.log10(peak**2/mse)
                    score = 1-float(ssim(torch.from_numpy(image)[None, None].to(device), ref_tensor))
                    scores[label].append([mse, psnr, score])
                    row.extend([mse, psnr, score])
                per_image.append(row)
        with open(out/"test_metrics.csv", "x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "noisy_mse", "noisy_psnr", "noisy_ssim",
                             "denoised_mse", "denoised_psnr", "denoised_ssim"])
            writer.writerows(per_image)
        summary = {"count": len(clean), "psnr_peak": peak,
                   "psnr_peak_source": "clean TRAINING maximum minus minimum",
                   "ssim": "raw intensities, paper constants .02^2/.06^2; window 11, sigma 1.5",
                   "region": "full original frame, no added mask, no intensity clipping",
                   "selection": "final epoch 100; no test-based checkpoint selection",
                   "warning": "As-provided train/test masking differs. One seed only; not manuscript reproduction."}
        for label, values in scores.items():
            a = np.asarray(values)
            summary[label] = dict(mean_mse=float(a[:, 0].mean()), mean_psnr_db=float(a[:, 1].mean()),
                                  mean_ssim=float(a[:, 2].mean()))
        summary["improvement"] = dict(
            psnr_db=summary["denoised"]["mean_psnr_db"]-summary["noisy"]["mean_psnr_db"],
            ssim=summary["denoised"]["mean_ssim"]-summary["noisy"]["mean_ssim"])
        write_json(out/"test_summary.json", summary)

        # Indices selected before viewing outputs, not selected for visual quality.
        indices = [i for i in (0, 100, 250) if i < len(clean)]
        fig, axes = plt.subplots(len(indices), 4, figsize=(13, 3.5*len(indices)), squeeze=False)
        for row, i in enumerate(indices):
            c, n, d = clean[i], noisy[i], prediction[i]
            vmin, vmax = float(c.min()), float(c.max())
            for col, (array, title) in enumerate(((n, "Noisy"), (d, "Denoised"), (c, "Clean reference"))):
                axes[row, col].imshow(array, cmap="gray", vmin=vmin, vmax=vmax)
                axes[row, col].set_title(f"{title} | image {i}")
            error = d-c
            bound = max(float(np.max(np.abs(error))), 1e-8)
            axes[row, 3].imshow(error, cmap="coolwarm", vmin=-bound, vmax=bound)
            axes[row, 3].set_title("Denoised minus reference")
            for ax in axes[row]:
                ax.axis("off")
        fig.suptitle("Fixed examples; noisy/denoised/clean share each row's intensity scale")
        fig.tight_layout()
        fig.savefig(out/"comparison.png", dpi=160)
        plt.close(fig)
        report = ["# As-provided training trial", "", json.dumps(summary, ensure_ascii=False, indent=2), "",
                  "These are final-checkpoint full-frame results. Training noisy inputs contain a masked perimeter,",
                  "while the provided test noisy inputs do not. Original data were not changed.",
                  "A PSNR increase does not by itself establish preservation of biological structure."]
        (out/"RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    finally:
        for s in stacks:
            s.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    source = Path(__file__).resolve().parent
    out = new_output_dir(args.out).resolve()
    data = Path(args.data_dir).resolve()
    status = dict(started=now(), pid=os.getpid(), state="running", stage="preflight")

    def record(**changes):
        status.update(changes, updated=now())
        write_json(out/"status.json", status)

    def run(stage, command):
        print(now(), stage, flush=True)
        record(stage=stage, command=command)
        with open(out/(stage+".log"), "x", encoding="utf-8") as log:
            subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT, check=True)

    snapshot = out/"source"
    try:
        record()
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU trial, but CUDA is unavailable.")
        paths = {"train_clean": data/"emd_65045clean1000.npy",
                 "train_noisy": data/"emd_65045SNR101000.npy",
                 "test_clean": data/"EMD65045_test_clean500.npy",
                 "test_noisy": data/"EMD65045_test_noisy500.npy"}
        stats = {}
        for name, path in paths.items():
            stack = Stack(path)
            stats[name] = stack_stats(stack)
            stats[name]["sha256"] = sha256(path)
            stats[name]["first_image_zero_fraction"] = float((stack[0] == 0).mean())
            stack.close()
        if stats["train_clean"]["count"] != 1000 or stats["test_clean"]["count"] != 500:
            raise ValueError("Unexpected train/test counts.")
        plan = dict(created=now(), paths={k:str(v) for k,v in paths.items()}, stats=stats,
                    environment=environment(), seed=42, distillation_epochs=20, training_epochs=100,
                    batch_size=4, inference_normalization="train", backend="deepinv",
                    data_policy="User authorized original files unchanged, including masking mismatch.",
                    evaluation="Final epoch only; no test-guided selection.")
        write_json(out/"experiment.json", plan)
        snapshot.mkdir()
        for name in ("distill.py", "train.py", "infer.py", "evaluate.py", "run_experiment.py", "requirements.txt"):
            shutil.copy2(source/name, snapshot/name)
        shutil.copytree(source/"cryoednet", snapshot/"cryoednet", ignore=shutil.ignore_patterns("__pycache__"))
        commands = [sys.executable, "-u", "-B"]
        run("distillation", commands+["distill.py", "--clean", str(paths["train_clean"]),
                                       "--out", str(out/"distillation"), "--epochs", "20",
                                       "--batch-size", "4", "--seed", "42", "--device", "cuda"])
        run("training", commands+["train.py", "--clean", str(paths["train_clean"]),
                                   "--noisy", str(paths["train_noisy"]),
                                   "--extractor", str(out/"distillation"/"extractor.pt"),
                                   "--out", str(out/"training"), "--epochs", "100",
                                   "--batch-size", "4", "--seed", "42", "--device", "cuda"])
        run("inference", commands+["infer.py", "--input", str(paths["test_noisy"]),
                                    "--checkpoint", str(out/"training"/"checkpoint.pt"),
                                    "--output", str(out/"test_denoised.npy"), "--normalization", "train",
                                    "--batch-size", "4", "--device", "cuda"])
        record(stage="evaluation")
        assess(paths["test_clean"], paths["test_noisy"], out/"test_denoised.npy", out,
               stats["train_clean"]["maximum"]-stats["train_clean"]["minimum"])
        record(state="completed", stage="completed", completed=now())
        print(now(), "Completed:", out, flush=True)
    except BaseException as exc:
        record(state="failed", error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
