"""Adapter that converts images of any size to the size the models expect.

All models in this repo work on images of shape (B, 3, 480, 640) (4:3), which
is the size of the training datasets. Recordings come in other sizes, e.g.
1920x1080 (16:9). `to_standard_size` converts a batch of images (B, 3, H, W)
of any size to (B, 3, 480, 640), so it can be passed directly to `predict`.

Three modes are available (see `to_standard_size`):
    "crop"    (default) cut a 4:3 window out of the image, then rescale it.
              Nothing is distorted and there are no empty borders, but parts
              of the image are lost (for 16:9: a quarter of the width).
    "pad"     rescale the whole image to fit and fill the rest with a constant
              (letterbox). Nothing is lost, but the models never saw borders
              during training.
    "stretch" rescale the whole image to 480x640, ignoring the aspect ratio.
              Nothing is lost, but the content is squashed.
"""

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["STANDARD_SIZE", "to_standard_size"]

# (height, width) expected by the models.
STANDARD_SIZE = (480, 640)

_MODES = ("crop", "pad", "stretch")


class _Plan(NamedTuple):
    """Geometry of one transform: where to cut, what to resize to, where to paste."""

    crop_top: int
    crop_left: int
    crop_h: int
    crop_w: int
    resize_h: int
    resize_w: int
    pad_top: int
    pad_left: int


def _plan(h, w, size, mode, align) -> _Plan:
    """Compute the geometry for converting an (h, w) image to `size`.

    This only does arithmetic on integers (no image data), so it is cheap and
    the same plan is used for every image in a batch.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    align_v, align_h = align
    if not (0 <= align_v <= 1 and 0 <= align_h <= 1):
        raise ValueError(f"align values must be in [0, 1], got {align}")
    th, tw = size

    # Part of the source image that is used (the whole image, unless cropping).
    crop_h, crop_w = h, w
    if mode == "crop":
        # Source wider than the target aspect ratio -> cut the sides,
        # otherwise cut the top and bottom. (w / h > tw / th without floats.)
        if w * th > h * tw:
            crop_w = min(w, max(1, round(h * tw / th)))
        else:
            crop_h = min(h, max(1, round(w * th / tw)))
    crop_top = round((h - crop_h) * align_v)
    crop_left = round((w - crop_w) * align_h)

    # Size the used part is resized to.
    if mode == "pad":
        scale = min(th / crop_h, tw / crop_w)  # largest size that still fits
        resize_h = min(th, max(1, round(crop_h * scale)))
        resize_w = min(tw, max(1, round(crop_w * scale)))
    else:
        resize_h, resize_w = th, tw

    # Where the resized image is placed on the output canvas (only != 0 for "pad").
    pad_top = round((th - resize_h) * align_v)
    pad_left = round((tw - resize_w) * align_h)
    return _Plan(
        crop_top, crop_left, crop_h, crop_w, resize_h, resize_w, pad_top, pad_left
    )


def _to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 images (0-255) become float32 in [0, 1]; floats are kept as they are."""
    if x.dtype == torch.uint8:
        return x.to(torch.float32) / 255.0
    if x.is_floating_point():
        return x.to(torch.float32)
    raise TypeError(f"images must be uint8 or floating point, got {x.dtype}")


def _apply(chunk: torch.Tensor, plan: _Plan, size, pad_value: float) -> torch.Tensor:
    """Apply a plan to a chunk of images (n, 3, H, W) -> (n, 3, *size)."""
    top, left, ch, cw = plan.crop_top, plan.crop_left, plan.crop_h, plan.crop_w
    # Cut first and convert afterwards, so we never convert pixels we throw away.
    patch = _to_float(chunk[:, :, top : top + ch, left : left + cw])

    if (ch, cw) != (plan.resize_h, plan.resize_w):
        # Bilinear with antialiasing: averages source pixels when shrinking
        # (1080p -> 480p), so thin lines do not turn into aliasing artefacts.
        patch = F.interpolate(
            patch,
            size=(plan.resize_h, plan.resize_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    if (plan.resize_h, plan.resize_w) == tuple(size):
        return patch

    # "pad" mode: paste the resized image onto a constant-colour canvas.
    canvas = patch.new_full((patch.shape[0], 3, *size), pad_value)
    canvas[
        :,
        :,
        plan.pad_top : plan.pad_top + plan.resize_h,
        plan.pad_left : plan.pad_left + plan.resize_w,
    ] = patch
    return canvas


def to_standard_size(
    images,
    mode: str = "crop",
    align=(0.5, 0.5),
    size=STANDARD_SIZE,
    pad_value: float = 0.0,
    chunk_size: int = 16,
):
    """Convert images of any size to the size the models expect.

    Args:
        images: Batch of shape (B, 3, H, W), or a single image (3, H, W).
            NumPy array or torch tensor. uint8 values (0-255) are scaled to
            [0, 1]; floating point values are assumed to be in [0, 1] already.
            Images from OpenCV / PIL are usually (H, W, 3): convert them first,
            e.g. with `frames.transpose(0, 3, 1, 2)`.
        mode: "crop" (default), "pad" or "stretch", see the module docstring.
        align: (vertical, horizontal) position of the used window ("crop") or
            of the image on the canvas ("pad"), each in [0, 1]. 0.5 means
            centred, 0 means top / left, 1 means bottom / right. For example
            `align=(0.5, 0.0)` keeps the left part of a 16:9 frame.
        size: Output (height, width). Defaults to the models' (480, 640).
        pad_value: Fill value of the borders in "pad" mode.
        chunk_size: Number of images converted at once. It only limits memory
            use (a 1080p float32 image takes ~25 MB), the result is the same.

    Returns:
        float32 array / tensor with values in [0, 1] and shape (B, 3, *size)
        (or (3, *size) for a single image). The type (NumPy or torch) and the
        device of the input are preserved. Images that already have the right
        size are only converted to float32, their pixels are not changed.
    """
    is_numpy = isinstance(images, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(np.ascontiguousarray(images))
    elif isinstance(images, torch.Tensor):
        x = images
    else:
        raise TypeError(f"images must be a NumPy array or a torch tensor, got {type(images)}")

    single = x.ndim == 3
    if single:
        x = x[None]
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(
            f"expected images of shape (B, 3, H, W) or (3, H, W), got {tuple(images.shape)}"
        )
    batch, _, h, w = x.shape
    if h == 0 or w == 0:
        raise ValueError(f"images are empty: {tuple(images.shape)}")

    plan = _plan(h, w, size, mode, align)  # the same for every image in the batch
    out = torch.empty((batch, 3, *size), dtype=torch.float32, device=x.device)
    for start in range(0, batch, chunk_size):
        stop = start + chunk_size
        out[start:stop] = _apply(x[start:stop], plan, size, pad_value)

    if single:
        out = out[0]
    return out.numpy() if is_numpy else out