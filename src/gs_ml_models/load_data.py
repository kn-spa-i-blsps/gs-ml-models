from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import TORCH_DEVICE


def load_labels_csv(path="data/splatdata.csv") -> pd.DataFrame:
    """Loads the labels file, might potentially do more later so
    it's left as a call to basic pandas right now"""
    df = pd.read_csv(path)
    return df


def load_images_np(path="data/classification_data", verbose=False) -> np.ndarray:
    return np.concatenate(load_images_grouped_np(path, verbose=verbose), axis=0)


def load_images_grouped_np(
    path="data/classification_data", verbose=False
) -> list[np.ndarray]:
    result = []
    for dir in sorted(Path(path).iterdir()):
        if dir.is_dir():
            acc = []
            for img in sorted(dir.iterdir()):
                with Image.open(img) as image:
                    acc.append(np.array(image))
            result.append(np.stack(acc))
            if verbose:
                print(f"Loaded {len(acc)} images from directory {dir.name}")
    return result


def load_images_grouped_torch(
    path="data/classification_data", verbose=False
) -> list[torch.Tensor]:
    data = load_images_grouped_np(path, verbose=verbose)
    result = []
    acc = 0
    total = sum(d.shape[0] for d in data)
    if verbose:
        print("Converting images to tensors...")

    first = True
    for d in data:
        result.append(torch.from_numpy(d).to(TORCH_DEVICE))
        if verbose:
            acc += d.shape[0]
            if first:
                first = False
                print(f"{acc}/{total}")
            else:
                print("\033[1A\033[2K", end="\r")
                print(f"{acc}/{total}")
    return result


def load_images_torch(path="data/classification_data", verbose=False):
    data = load_images_np(path, verbose=verbose)
    return torch.from_numpy(data).to(TORCH_DEVICE)
