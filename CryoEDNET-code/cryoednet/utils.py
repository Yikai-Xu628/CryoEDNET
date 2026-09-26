import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def environment():
    versions = {}
    for name in ("torch", "torchvision", "numpy", "deepinv", "mrcfile"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return dict(python=sys.version, versions=versions, cuda=torch.version.cuda)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def new_output_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
