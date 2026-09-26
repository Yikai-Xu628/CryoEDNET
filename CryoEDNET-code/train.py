"""Train the manuscript architecture on precomputed clean/noisy particle pairs."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cryoednet import CryoEDNet
from cryoednet.data import PairDataset
from cryoednet.losses import CombinedLoss
from cryoednet.perceptual import load_extractor
from cryoednet.utils import seed_everything, environment, sha256, new_output_dir, write_json


def train_epoch(model, loader, loss_fn, optimizer, scheduler, device):
    model.train()
    loss_fn.train()
    total, count = 0., 0
    for clean, noisy in loader:
        clean, noisy = clean.to(device), noisy.to(device)
        optimizer.zero_grad(set_to_none=True)
        value = loss_fn(model(noisy), clean)
        value.backward()
        optimizer.step()
        scheduler.step()  # OneCycleLR is a per-batch scheduler.
        total += value.item()*len(clean)
        count += len(clean)
    return total/count


def run_deepinv(model, loader, loss_fn, optimizer, scheduler, epochs, device, out):
    import deepinv as dinv
    from deepinv.training import Trainer

    class BatchScheduledTrainer(Trainer):
        def compute_loss(self, physics, x, y, train=True, epoch=None):
            result = super().compute_loss(physics, x, y, train=train, epoch=epoch)
            if train:
                scheduler.step()
            return result

    trainer = BatchScheduledTrainer(
        model=model, physics=dinv.physics.Denoising(), optimizer=optimizer,
        train_dataloader=loader, epochs=epochs, losses=loss_fn,
        scheduler=None, online_measurements=False, metrics=[],
        device=device, plot_images=False, wandb_vis=False,
        save_path=Path(out)/"deepinv_checkpoints", ckp_interval=10,
        compare_no_learning=False,
    )
    trainer.train()
    return [float(x) for x in trainer.loss_history]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", required=True)
    p.add_argument("--noisy", required=True)
    p.add_argument("--extractor", required=True, help="Required output from distill.py.")
    p.add_argument("--out", required=True)
    p.add_argument("--engine", choices=["deepinv", "torch"], default="deepinv")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", action="store_true", help="Optional paired flips/90-degree rotations; no extra noise.")
    p.add_argument("--ablation", choices=["none", "no-ms", "no-ee", "no-dp"], default="none")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        p.error("epochs and batch-size must be positive.")
    seed_everything(args.seed)
    extractor, ex_ckpt = load_extractor(args.extractor)
    clean_hash = sha256(args.clean)
    if clean_hash != ex_ckpt["clean_sha256"]:
        raise ValueError("Clean training file differs from the distillation training file.")
    dataset = PairDataset(args.clean, args.noisy, ex_ckpt["stats"], augment=args.augment)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    if len(loader)*args.epochs < 6:
        p.error("Use at least six optimizer steps for this OneCycle schedule.")
    out = new_output_dir(args.out)
    model = CryoEDNet(use_multiscale=args.ablation != "no-ms",
                      use_edge=args.ablation != "no-ee",
                      use_detail=args.ablation != "no-dp").to(args.device)
    loss_fn = CombinedLoss(extractor).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=args.epochs, steps_per_epoch=len(loader),
        pct_start=.2, div_factor=10, final_div_factor=100,
    )
    config = dict(vars(args), environment=environment(), clean_sha256=clean_hash,
                  noisy_sha256=sha256(args.noisy), extractor_sha256=sha256(args.extractor),
                  architecture=model.config, train_stats=ex_ckpt["stats"],
                  external_mask="ASPIRE-prepared inputs; no in-code input or loss mask",
                  loss_weights=dict(rec=1., perc=.1, ssim=.5, freq=.1),
                  lr=dict(initial=1e-4, maximum=1e-3, pct_start=.2, final_div_factor=100),
                  provenance="manuscript-based reconstruction; results not historically verified")
    write_json(out / "config.json", config)

    def save_checkpoint(epoch):
        torch.save(dict(format="cryoednet-reconstruction-v1", architecture=model.config,
                        state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        train_stats=ex_ckpt["stats"], config=config, epoch=epoch,
                        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()),
                   out / "checkpoint.pt")

    if args.engine == "deepinv":
        history = run_deepinv(model, loader, loss_fn, optimizer, scheduler,
                              args.epochs, args.device, out)
        save_checkpoint(args.epochs)
    else:
        history = []
        for epoch in range(args.epochs):
            value = train_epoch(model, loader, loss_fn, optimizer, scheduler, args.device)
            history.append(value)
            print(f"Epoch {epoch+1}/{args.epochs}: loss={value:.6g}")
            save_checkpoint(epoch+1)
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "mean_training_loss"])
        writer.writerows(enumerate(history, start=1))
    dataset.clean.close()
    dataset.noisy.close()
    print(f"Saved {out / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
