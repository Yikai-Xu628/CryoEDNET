"""Denoise NPY/MRCS particle stacks without clipping or implicit normalization choices."""

import argparse
from pathlib import Path
import warnings

import numpy as np
import torch

from cryoednet import CryoEDNet
from cryoednet.data import Stack, stack_stats, normalize, denormalize
from cryoednet.utils import environment, write_json, sha256


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="New .npy or .mrcs path, particle axis first.")
    p.add_argument("--normalization", required=True,
                   choices=["train", "stack", "legacy-stack-rescale"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, help="Use only the first N particles, also for stack statistics.")
    p.add_argument("--pixel-size", type=float, help="Angstrom/pixel for MRCS; otherwise copy input metadata.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    output = Path(args.output)
    manifest = output.with_suffix(output.suffix+".json")
    if output.suffix.lower() not in (".npy", ".mrcs"):
        p.error("Output must end in .npy or .mrcs.")
    if output.exists() or manifest.exists():
        raise FileExistsError("Will not overwrite output or its metadata sidecar.")
    if args.batch_size < 1 or (args.pixel_size is not None and args.pixel_size <= 0):
        p.error("batch-size and pixel-size must be positive.")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if ckpt.get("format") != "cryoednet-reconstruction-v1":
        raise ValueError("Not a reconstructed-model checkpoint. Legacy depth/nf weights cannot be silently reused.")
    model = CryoEDNet(**ckpt["architecture"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(args.device).eval()
    stack = Stack(args.input)
    count = len(stack) if args.limit is None else args.limit
    if not 0 < count <= len(stack):
        p.error("limit must be between 1 and the number of input particles.")
    train_stats = ckpt["train_stats"]
    input_stats = stack_stats(stack, count=count) if args.normalization != "train" else None
    if args.normalization == "legacy-stack-rescale":
        warnings.warn("Legacy remapping matches the recovered real-data script/manuscript wording, "
                      "but differs from the denoiser's standardized training input scale. "
                      "This is not a validated reproduction setting.")
    voxel_size = args.pixel_size if args.pixel_size is not None else stack.voxel_size
    if output.suffix.lower() == ".mrcs" and voxel_size is None:
        p.error("NPY to MRCS requires --pixel-size; physical sampling must not be invented.")
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = (count, *stack.data.shape[1:])
    mrc = None
    if output.suffix.lower() == ".npy":
        target = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=shape)
    else:
        import mrcfile
        mrc = mrcfile.new_mmap(output, shape=shape, mrc_mode=2, overwrite=False)
        mrc.set_image_stack()
        mrc.voxel_size = voxel_size
        target = mrc.data
    try:
        with torch.inference_mode():
            for start in range(0, count, args.batch_size):
                stop = min(start+args.batch_size, count)
                raw = torch.from_numpy(stack[start:stop, ...][:, None]).to(args.device)
                result = model(normalize(raw, args.normalization, train_stats, input_stats))
                result = denormalize(result, args.normalization, train_stats, input_stats)
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
        stack.close()
    write_json(manifest, dict(input=str(Path(args.input).resolve()),
                              checkpoint_sha256=sha256(args.checkpoint),
                              normalization=args.normalization, train_stats=train_stats,
                              input_stats=input_stats, shape=list(shape),
                              pixel_size=voxel_size, clipping=False, mask="external ASPIRE",
                              environment=environment(), complete=True,
                              provenance=ckpt["config"].get("provenance")))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
