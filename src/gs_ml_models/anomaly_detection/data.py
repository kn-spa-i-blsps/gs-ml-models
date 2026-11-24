"""
Data for anomaly detection: the manifest, its checks, and image loading.

The manifest is a CSV with one row per image. Code never guesses from folder
names; it reads the manifest.

Required columns:
    path    image path, relative to the manifest's folder (or absolute)
    split   train | val | test | splat_input
            splat_input = real frames used to build a splat; never given to a
            detector, listed only so the checks below can prove it
    source  real | splat
    label   0 = normal, 1 = anomaly

Optional columns (used when present):
    run           recording / pass the frame comes from
    t             time of the frame in that recording, in seconds
    anomaly_type  e.g. holes, broken_piece (for per-type results)
    splat_id      which splat a render comes from
    session       recording session (for future multi-session data)

Typical use:
    df = load_manifest("data/basen_01/manifest.csv")
    train = df[(df.split == "train") & (df.source == "real")]
    X = load_images(train, size=256, resize="center_crop")
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REQUIRED_COLUMNS = ("path", "split", "source", "label")
SPLITS = {"train", "val", "test", "splat_input"}
SOURCES = {"real", "splat"}
RESIZE_MODES = {"squash", "center_crop"}


class ManifestError(ValueError):
    """The manifest breaks one of the data rules."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def load_manifest(path, validate: bool = True) -> pd.DataFrame:
    """Read a manifest CSV (and check it, unless ``validate=False``).

    The manifest's folder is stored in ``df.attrs["root"]`` and used by
    ``load_images`` to resolve relative image paths.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.attrs["root"] = str(path.parent)
    if validate:
        validate_manifest(df)
    return df


def validate_manifest(df: pd.DataFrame) -> None:
    """Check the data rules; raise ``ManifestError`` listing every violation.

    1. Every image appears once.
    2. Frames used to build a splat (split=splat_input) appear in no other split.
    3. Anomalies (label=1) appear only in test.
    4. Validation holds only normal images.
    5. Renders (source=splat) appear only in train.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"Manifest is missing columns: {missing}")

    problems: list[str] = []

    def report(mask: pd.Series, message: str) -> None:
        if mask.any():
            examples = df.loc[mask, "path"].head(5).tolist()
            problems.append(f"{message} ({int(mask.sum())} rows, e.g. {examples})")

    report(~df["split"].isin(SPLITS), f"unknown split (allowed: {sorted(SPLITS)})")
    report(~df["source"].isin(SOURCES), f"unknown source (allowed: {sorted(SOURCES)})")
    report(~df["label"].isin([0, 1]), "label must be 0 or 1")

    # 1. every image once
    report(df["path"].duplicated(keep=False), "rule 1: image listed more than once")

    # 2. splat input frames never used elsewhere (also by recording time, if known)
    if {"run", "t"} <= set(df.columns):
        splat_frames = set(
            map(tuple, df.loc[df["split"] == "splat_input", ["run", "t"]].dropna().to_numpy().tolist())
        )
        key = list(map(tuple, df[["run", "t"]].to_numpy().tolist()))
        reused = pd.Series([k in splat_frames for k in key], index=df.index) & (df["split"] != "splat_input")
        report(reused, "rule 2: frame used to build a splat is also in train/val/test")

    # 3. anomalies only in test
    report((df["label"] == 1) & (df["split"] != "test"), "rule 3: anomaly outside test")

    # 4. validation is normal-only
    report((df["split"] == "val") & (df["label"] != 0), "rule 4: anomaly in validation")

    # 5. renders only in train
    report((df["source"] == "splat") & (df["split"] != "train"), "rule 5: render outside train")

    if problems:
        raise ManifestError("Manifest breaks the data rules:\n- " + "\n- ".join(problems))


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def _target_size(size) -> tuple[int, int]:
    """int -> square (size, size); (H, W) kept as is."""
    if isinstance(size, int):
        return size, size
    height, width = size
    return int(height), int(width)


def _center_crop_box(width: int, height: int, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    """Largest centred box with the target aspect ratio."""
    target_ratio = target_w / target_h
    if width / height > target_ratio:  # too wide: cut the sides
        new_w = round(height * target_ratio)
        left = (width - new_w) // 2
        return left, 0, left + new_w, height
    new_h = round(width / target_ratio)  # too tall: cut top and bottom
    top = (height - new_h) // 2
    return 0, top, width, top + new_h


def load_images(images, size, resize: str = "squash", root=None, as_float: bool = False) -> torch.Tensor:
    """Load images into one tensor of shape (B, 3, H, W).

    Args:
        images: A manifest DataFrame (or a slice of one), or a list of paths.
        size: Output size: an int for a square (e.g. 256) or (H, W).
        resize: How to reach ``size`` from the original frame:
            "squash"      - resize the whole frame, aspect ratio not kept
                            (a 16:9 frame gets squeezed horizontally);
            "center_crop" - cut the largest centred region with the target
                            aspect ratio, then resize (edges are lost,
                            nothing is distorted).
        root: Folder that relative paths start from. Defaults to the
            manifest's folder (``df.attrs["root"]``), else the current folder.
        as_float: False (default) returns uint8 0-255, 4x smaller in memory;
            True returns float32 0-1. The models accept both.

    Warns if the original images have different aspect ratios, because the
    same resize would then distort them differently.
    """
    if resize not in RESIZE_MODES:
        raise ValueError(f"resize must be one of {sorted(RESIZE_MODES)}, got {resize!r}")
    target_h, target_w = _target_size(size)

    if isinstance(images, pd.DataFrame):
        paths = images["path"].tolist()
        root = root if root is not None else images.attrs.get("root")
    else:
        paths = list(images)
    root = Path(root) if root is not None else Path.cwd()

    out = np.empty((len(paths), 3, target_h, target_w), dtype=np.uint8)
    ratios: Counter = Counter()
    for k, rel in enumerate(paths):
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            ratios[round(width / height, 2)] += 1
            if resize == "center_crop":
                img = img.crop(_center_crop_box(width, height, target_w, target_h))
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            out[k] = np.asarray(img).transpose(2, 0, 1)

    if len(ratios) > 1:
        summary = ", ".join(f"{r}: {n} images" for r, n in sorted(ratios.items()))
        warnings.warn(
            f"Images have different aspect ratios ({summary}); "
            f"resize={resize!r} will treat them differently.",
            stacklevel=2,
        )

    tensor = torch.from_numpy(out)
    return tensor.float().div_(255.0) if as_float else tensor
