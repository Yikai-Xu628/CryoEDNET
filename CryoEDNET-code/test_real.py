import argparse
from pathlib import Path

import numpy as np
import torch

from cryoednet import CryoEDNet
from cryoednet.data import Stack
from cryoednet.utils import environment, sha256, write_json


def denoise(model, raw, train_mean, train_std, strength):
    image_mean = raw.mean(dim=(-2, -1), keepdim=True)
    image_std = raw.std(dim=(-2, -1), keepdim=True) + 1e-8
    scale = train_std * strength
    model_input = (raw - image_mean) / image_std * scale + train_mean
    return (model(model_input) - train_mean) / (scale + 1e-10) * image_std + image_mean


def main(argv=None):
    parser = argparse.ArgumentParser(description="Denoise a real particle stack with per-image intensity scaling.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strength", type=float, default=2.3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pixel-size", type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    output = Path(args.output)
    manifest = output.with_suffix(output.suffix + ".json")
    if output.suffix.lower() not in (".npy", ".mrcs"):
        parser.error("Output must end in .npy or .mrcs.")
    if output.exists() or manifest.exists():
        raise FileExistsError("Will not overwrite output or its metadata sidecar.")
    if not np.isfinite(args.strength) or args.strength <= 0:
        parser.error("strength must be finite and positive.")
    if args.batch_size < 1 or (args.pixel_size is not None and args.pixel_size <= 0):
        parser.error("batch-size and pixel-size must be positive.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "cryoednet-reconstruction-v1":
        raise ValueError("Expected a reconstructed-model checkpoint.")
    model = CryoEDNet(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    train_mean = checkpoint["train_stats"]["mean"]
    train_std = checkpoint["train_stats"]["std"]
    if not np.isfinite(train_mean) or not np.isfinite(train_std) or train_std <= 0:
        raise ValueError("Invalid training statistics in checkpoint.")

    stack = Stack(args.input)
    try:
        count = len(stack) if args.limit is None else args.limit
        if not 0 < count <= len(stack):
            parser.error("limit must be between 1 and the number of input particles.")
        pixel_size = args.pixel_size if args.pixel_size is not None else stack.voxel_size
        if output.suffix.lower() == ".mrcs" and pixel_size is None:
            parser.error("NPY to MRCS requires --pixel-size.")
        shape = (count, *stack.data.shape[1:])
        output.parent.mkdir(parents=True, exist_ok=True)
        mrc = None
        if output.suffix.lower() == ".npy":
            target = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=shape)
        else:
            import mrcfile

            mrc = mrcfile.new_mmap(output, shape=shape, mrc_mode=2, overwrite=False)
            mrc.set_image_stack()
            mrc.voxel_size = pixel_size
            target = mrc.data
        try:
            with torch.inference_mode():
                for start in range(0, count, args.batch_size):
                    stop = min(start + args.batch_size, count)
                    raw = torch.from_numpy(stack[start:stop, ...][:, None]).to(args.device)
                    result = denoise(model, raw, train_mean, train_std, args.strength)
                    if not torch.isfinite(result).all():
                        raise FloatingPointError("Non-finite denoising result; output is incomplete.")
                    target[start:stop] = result[:, 0].cpu().numpy()
                    print(f"Denoised {stop}/{count}")
            target.flush()
            if mrc is not None:
                mrc.update_header_stats()
        finally:
            if mrc is not None:
                mrc.close()
            else:
                target._mmap.close()
    finally:
        stack.close()

    write_json(manifest, dict(input=str(Path(args.input).resolve()),
                              checkpoint_sha256=sha256(args.checkpoint),
                              strength=args.strength,
                              normalization="per-image mean and sample standard deviation; checkpoint training statistics",
                              shape=list(shape), pixel_size=pixel_size,
                              clipping=False, blending=False, masking=False, resizing=False,
                              environment=environment(), complete=True))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
