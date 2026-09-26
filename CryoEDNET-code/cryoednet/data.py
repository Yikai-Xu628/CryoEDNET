"""Memory-mapped particle stacks; axis order is explicitly (N,H,W)."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class Stack:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None
        if self.path.suffix.lower() == ".npy":
            data = np.load(self.path, mmap_mode="r", allow_pickle=False)
            self.voxel_size = None
        elif self.path.suffix.lower() in (".mrc", ".mrcs"):
            import mrcfile
            self.handle = mrcfile.mmap(self.path, mode="r", permissive=False)
            data = self.handle.data
            self.voxel_size = tuple(float(self.handle.voxel_size[k]) for k in ("x", "y", "z"))
        else:
            raise ValueError("Use .npy, .mrc or .mrcs with particle axis first.")
        if data.ndim == 4 and data.shape[1] == 1:
            data = data[:, 0]
        if data.ndim != 3 or min(data.shape) < 1:
            raise ValueError("Expected (N,H,W) or (N,1,H,W), not a transposed stack.")
        if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise ValueError("Particle intensities must be real numeric values.")
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = np.array(self.data[i], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite input values in {self.path}, item {i}.")
        return x

    def close(self):
        if self.handle is not None:
            self.handle.close()
        else:
            mapping = getattr(self.data, "_mmap", None)
            if mapping is not None:
                mapping.close()


def stack_stats(stack, count=None, batch_size=32):
    """Stable pooled population variance over all pixels, including background."""
    n_images = len(stack) if count is None else count
    if not 0 < n_images <= len(stack):
        raise ValueError("Invalid stack selection size.")
    total, mean, m2 = 0, 0., 0.
    minimum, maximum = float("inf"), -float("inf")
    for start in range(0, n_images, batch_size):
        a = stack[start:min(start+batch_size, n_images)].astype(np.float64)
        n, part_mean = a.size, float(a.mean())
        delta = part_mean - mean
        m2 += float(np.square(a-part_mean).sum()) + delta**2 * total*n/(total+n)
        mean += delta*n/(total+n)
        total += n
        minimum, maximum = min(minimum, float(a.min())), max(maximum, float(a.max()))
    std = float(np.sqrt(m2/total))
    if not np.isfinite(std) or std < 1e-8:
        raise ValueError("Constant or invalid stack; cannot standardize.")
    return dict(mean=mean, std=std, minimum=minimum, maximum=maximum,
                count=n_images, shape=list(stack.data.shape[1:]), pixels=total)


class PairDataset(Dataset):
    def __init__(self, clean_path, noisy_path, stats, augment=False):
        self.clean, self.noisy = Stack(clean_path), Stack(noisy_path)
        if self.clean.data.shape != self.noisy.data.shape:
            raise ValueError("Clean/noisy stacks must have identical shapes and ordering.")
        self.stats, self.augment = stats, augment

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        mean, std = self.stats["mean"], self.stats["std"]
        clean = torch.from_numpy(self.clean[i])[None]
        noisy = torch.from_numpy(self.noisy[i])[None]
        clean, noisy = (clean-mean)/std, (noisy-mean)/std
        if self.augment:
            if torch.rand(()) > .5:
                clean, noisy = clean.flip(-1), noisy.flip(-1)
            if torch.rand(()) > .5:
                clean, noisy = clean.rot90(1, (-2, -1)), noisy.rot90(1, (-2, -1))
        return clean, noisy


def normalize(x, mode, train_stats, input_stats):
    if mode == "train":
        return (x-train_stats["mean"])/train_stats["std"]
    z = (x-input_stats["mean"])/input_stats["std"]
    if mode == "stack":
        return z
    if mode == "legacy-stack-rescale":
        return z*train_stats["std"] + train_stats["mean"]
    raise ValueError(f"Unknown normalization: {mode}")


def denormalize(x, mode, train_stats, input_stats):
    if mode == "train":
        return x*train_stats["std"]+train_stats["mean"]
    if mode == "legacy-stack-rescale":
        x = (x-train_stats["mean"])/train_stats["std"]
    elif mode != "stack":
        raise ValueError(f"Unknown normalization: {mode}")
    return x*input_stats["std"]+input_stats["mean"]
