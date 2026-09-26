"""Reconstruct the manuscript's VGG19 distillation stage using training data only."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from cryoednet.data import Stack, stack_stats
from cryoednet.perceptual import FeatureExtractor, VGGTeacher, Distiller
from cryoednet.utils import seed_everything, environment, sha256, new_output_dir, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", required=True, help="ASPIRE-prepared clean TRAINING stack.")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20, help="Reconstructed recipe, not a recovered paper value.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error("epochs, batch-size and lr must be positive.")
    seed_everything(args.seed)
    clean = Stack(args.clean)
    stats = stack_stats(clean)
    if min(stats["shape"]) < 16:
        p.error("Distillation requires images at least 16 by 16.")
    out = new_output_dir(args.out)
    clean_hash = sha256(args.clean)
    recipe = dict(vars(args), clean_sha256=clean_hash, stats=stats,
                  environment=environment(), provenance="manuscript-based reconstruction",
                  teacher_scaling="global clean-training minmax, then ImageNet mean/std",
                  student_scaling="global clean-training mean/std",
                  spatial_matching="adaptive average pooling of adapted student features")
    write_json(out / "recipe.json", recipe)
    teacher = VGGTeacher().to(args.device)
    student = FeatureExtractor().to(args.device)
    distiller = Distiller(student, teacher).to(args.device)
    optimizer = torch.optim.Adam([p for p in distiller.parameters() if p.requires_grad], lr=args.lr)
    rng = np.random.default_rng(args.seed)
    steps = 0
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "distillation_loss"])
        for epoch in range(args.epochs):
            indices = rng.permutation(len(clean))
            total = 0.
            for start in range(0, len(clean), args.batch_size):
                batch = clean[indices[start:start+args.batch_size]]
                raw = torch.from_numpy(batch[:, None]).to(args.device)
                # Only the teacher branch uses training-derived [0,1] scaling.
                # Student data use exactly the subsequent denoiser's target scale.
                x = (raw-stats["mean"])/stats["std"]
                t = (raw-stats["minimum"])/(stats["maximum"]-stats["minimum"])
                optimizer.zero_grad(set_to_none=True)
                loss = distiller(x, t)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite distillation loss.")
                loss.backward()
                optimizer.step()
                total += loss.item()*len(batch)
                steps += 1
            writer.writerow([epoch+1, total/len(clean)])
            log.flush()
            print(f"Distillation {epoch+1}/{args.epochs}: {total/len(clean):.6g}")
    # Projection adapters are discarded. The entire retained extractor was
    # supervised, including CBAM and the residual branch via stage three.
    torch.save(dict(format="cryoednet-distilled-v1", teacher="VGG19_IMAGENET1K_V1",
                    state_dict={k:v.detach().cpu() for k,v in student.state_dict().items()},
                    stats=stats, steps=steps, clean_sha256=clean_hash, recipe=recipe),
               out / "extractor.pt")
    clean.close()
    print(f"Saved {out / 'extractor.pt'}")


if __name__ == "__main__":
    main()
