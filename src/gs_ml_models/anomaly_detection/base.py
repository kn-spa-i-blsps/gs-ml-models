"""
Shared adapter between the anomaly-detection API (see README.md) and Anomalib.

Every model in this package is a thin subclass of ``AnomalibDetector`` that only
says which Anomalib model to build and whether it is trained over epochs.
Everything the API promises lives here, written once:

- input checks (shape, dtype, the image size remembered from ``fit``),
- training through the Anomalib ``Engine`` on in-memory images,
- early stopping for trainable models, monitored on normal-only validation images,
- the decision threshold, set from normal-only validation images,
- ``save`` / ``load``.

Anomalib is imported lazily, so importing this module does not require it.
"""

from __future__ import annotations

import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# Name of the metric logged during validation and watched by early stopping:
# the mean anomaly score on the (normal-only) validation images.
MONITOR = "val_normal_score"


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _to_tensor(X) -> Tensor:
    """Validate a batch of images and return it as a CPU tensor.

    Accepts a NumPy array or a Torch tensor of shape (B, 3, H, W), either
    uint8 with values 0-255 or float with values in [0, 1]. The dtype is kept,
    so uint8 images stay 4x smaller in memory until they reach the model.
    """
    X = torch.as_tensor(X).cpu()
    if X.ndim != 4 or X.shape[1] != 3:
        raise ValueError(f"Images must have shape (B, 3, H, W), got {tuple(X.shape)}.")
    if X.dtype == torch.uint8:
        return X
    if not torch.is_floating_point(X):
        raise TypeError(f"Images must be uint8 (0-255) or float (0-1), got dtype {X.dtype}.")
    if X.numel() and (X.min() < 0 or X.max() > 1):
        raise ValueError("Float images must have values in [0, 1]. Pass uint8 for 0-255 values.")
    return X


def _to_float(batch: Tensor) -> Tensor:
    """Convert a (small) batch to float32 in [0, 1] right before the model."""
    if batch.dtype == torch.uint8:
        return batch.float().div_(255.0)
    return batch.float()


class _InMemoryImages(Dataset):
    """Serves in-memory images to Anomalib as ``ImageItem`` objects."""

    def __init__(self, X: Tensor) -> None:
        self.X = X

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        from anomalib.data.dataclasses.torch import ImageItem
        from torchvision.tv_tensors import Image

        return ImageItem(
            image=Image(_to_float(self.X[idx])),
            gt_label=torch.tensor(0),  # training / validation images are normal
            image_path=f"in_memory/{idx}",  # placeholder, never opened
        )


# ---------------------------------------------------------------------------
# Lightning helpers
# ---------------------------------------------------------------------------


def _make_score_monitor():
    """Callback that logs the mean anomaly score on the validation images.

    The validation images are all normal, so a lower mean score means the
    model fits normal data better. This is what early stopping watches; no
    anomalies are needed for it.
    """
    from lightning.pytorch.callbacks import Callback

    class _NormalScoreMonitor(Callback):
        def __init__(self) -> None:
            self._scores: list[Tensor] = []

        def on_validation_epoch_start(self, trainer, pl_module) -> None:
            self._scores = []

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
            score = getattr(outputs, "pred_score", None)
            if score is not None:
                self._scores.append(score.detach().flatten().float().cpu())

        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if self._scores:
                mean = torch.cat(self._scores).mean().to(pl_module.device)
                pl_module.log(MONITOR, mean, prog_bar=True)

    return _NormalScoreMonitor()


def _resolve_device(accelerator: str) -> torch.device:
    """Device used for inference, following the same rule as Lightning's 'auto'."""
    if accelerator == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if accelerator == "gpu":
        return torch.device("cuda")
    return torch.device(accelerator)


def _load_state_dict(module: torch.nn.Module, state_dict: dict[str, Tensor]) -> None:
    """``load_state_dict`` that also fills buffers created empty.

    Some models (e.g. PatchCore's memory bank) start with an empty buffer that
    is only filled during training, so a plain ``load_state_dict`` fails on the
    shape mismatch. Such buffers are resized before loading.
    """
    own = module.state_dict()
    for name, value in state_dict.items():
        if name in own and own[name].numel() == 0 and value.numel() > 0:
            *parents, leaf = name.split(".")
            owner = module
            for part in parents:
                owner = getattr(owner, part)
            if leaf in owner._buffers:
                owner._buffers[leaf] = torch.empty_like(value)
    module.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class AnomalibDetector(ABC):
    """Base class implementing the anomaly-detection API on top of Anomalib.

    Subclasses set ``trainable`` and implement ``_build_module`` (which
    Anomalib model to build) and extend ``get_config`` with their own
    settings.

    Args:
        threshold_percentile: Percentile of the validation scores used as the
            decision threshold. 95 means about 5% false alarms on normal images.
        batch_size: Training batch size.
        eval_batch_size: Batch size for validation and inference.
        max_epochs: Upper limit on training epochs (trainable models only).
        check_val_every_n_epoch: How often early stopping looks at the
            validation score (trainable models only).
        patience: Number of validation checks without improvement before
            training stops (trainable models only). With the defaults,
            3 checks x 5 epochs = 15 epochs.
        min_delta: Minimum decrease of the validation score that counts as
            an improvement.
        seed: Seed for all randomness (weight init, shuffling, sampling).
        accelerator: "auto" (CUDA > MPS > CPU), "cpu", "gpu"/"cuda" or "mps".
        work_dir: Where Lightning writes checkpoints during ``fit``. ``None``
            uses a temporary directory that is deleted afterwards.
    """

    #: True for models trained over many epochs (early stopping applies).
    trainable: bool = False

    def __init__(
        self,
        threshold_percentile: float = 95.0,
        batch_size: int = 32,
        eval_batch_size: int = 32,
        max_epochs: int = 200,
        check_val_every_n_epoch: int = 5,
        patience: int = 3,
        min_delta: float = 0.0,
        seed: int = 0,
        accelerator: str = "auto",
        work_dir: str | Path | None = None,
    ) -> None:
        self.threshold_percentile = threshold_percentile
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.patience = patience
        self.min_delta = min_delta
        self.seed = seed
        self.accelerator = accelerator
        self.work_dir = None if work_dir is None else str(work_dir)

        self.module = None  # the Anomalib model, built in fit() or load()
        self.input_size: tuple[int, int] | None = None  # (H, W) seen in fit()
        self.threshold: float | None = None

    # -- to be provided by subclasses ---------------------------------------

    @abstractmethod
    def _build_module(self, image_size: tuple[int, int], pre_trained: bool = True):
        """Return a fresh Anomalib model that works on images of ``image_size``.

        It must be built with ``post_processor=False`` (raw scores, thresholds
        are handled here) and with a pre-processor whose resize matches
        ``image_size``, so the model does not rescale the images again.
        """

    def get_config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this model (used by save)."""
        return {
            "threshold_percentile": self.threshold_percentile,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "max_epochs": self.max_epochs,
            "check_val_every_n_epoch": self.check_val_every_n_epoch,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "seed": self.seed,
            "accelerator": self.accelerator,
            "work_dir": self.work_dir,
        }

    # -- API ----------------------------------------------------------------

    def fit(self, X, X_val=None) -> None:
        """Train from scratch.

        Args:
            X: Normal training images, (B, 3, H, W), uint8 0-255 or float 0-1.
            X_val: Normal-only validation images of the same size. Used to set
                the decision threshold and, for trainable models, for early
                stopping. Without it, ``predict`` is unavailable until
                ``calibrate`` is called, and trainable models run for
                ``max_epochs``.
        """
        import lightning as L
        from anomalib.callbacks.checkpoint import ModelCheckpoint
        from anomalib.data.dataclasses.torch import ImageBatch
        from anomalib.engine import Engine
        from lightning.pytorch.callbacks import EarlyStopping

        X = _to_tensor(X)
        size = (int(X.shape[-2]), int(X.shape[-1]))
        if X_val is not None:
            X_val = _to_tensor(X_val)
            if tuple(X_val.shape[-2:]) != size:
                raise ValueError(
                    f"X_val images are {tuple(X_val.shape[-2:])}, but training images are {size}."
                )
        elif self.trainable:
            warnings.warn(
                f"No X_val given: {type(self).__name__} trains for max_epochs={self.max_epochs} "
                "without early stopping, and no threshold is set.",
                stacklevel=2,
            )

        L.seed_everything(self.seed, workers=True)
        self.input_size = size
        self.threshold = None
        self.module = self._build_module(size, pre_trained=True)

        train_loader = DataLoader(
            _InMemoryImages(X),
            batch_size=self.batch_size,
            shuffle=self.trainable,
            collate_fn=ImageBatch.collate,
        )

        val_loader = None
        callbacks = []
        checkpoint = None
        if self.trainable and X_val is not None:
            val_loader = DataLoader(
                _InMemoryImages(X_val),
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=ImageBatch.collate,
            )

        if self.work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="gs_ml_models_"))
            cleanup = True
        else:
            work_dir = Path(self.work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False

        if val_loader is not None:
            checkpoint = ModelCheckpoint(
                dirpath=str(work_dir / "checkpoints"),
                filename="best",
                monitor=MONITOR,
                mode="min",
                save_top_k=1,
                auto_insert_metric_name=False,
            )
            callbacks = [
                _make_score_monitor(),
                EarlyStopping(monitor=MONITOR, mode="min", patience=self.patience, min_delta=self.min_delta),
                checkpoint,
            ]

        engine = Engine(
            callbacks=callbacks,
            default_root_dir=str(work_dir),
            accelerator=self.accelerator,
            devices=1,
            max_epochs=self.max_epochs,  # memory-bank models override this to 1
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            deterministic="warn",
            enable_model_summary=False,
        )
        try:
            engine.fit(model=self.module, train_dataloaders=train_loader, val_dataloaders=val_loader)
            # Keep the weights of the best validation check, not the last epoch.
            if checkpoint is not None and checkpoint.best_model_path:
                best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
                self.module.load_state_dict(best["state_dict"])
        finally:
            if cleanup:
                shutil.rmtree(work_dir, ignore_errors=True)

        self.module.eval()
        if X_val is not None:
            self.threshold = float(np.percentile(self.anomaly_scores(X_val), self.threshold_percentile))

    def train(self, X) -> None:
        """Not supported yet: continuing training on new data only would make
        the model forget earlier data. Retrain on all data with ``fit``."""
        raise NotImplementedError(
            f"{type(self).__name__}.train() is not supported yet. Use fit() on all the data instead."
        )

    def anomaly_scores(self, X) -> np.ndarray:
        """Raw anomaly score per image, shape (B,). Higher = more anomalous."""
        scores, _ = self._infer(X, want_maps=False)
        return scores

    def predict(self, X) -> np.ndarray:
        """0/1 label per image, shape (B,): 1 if score >= threshold."""
        if self.threshold is None:
            raise RuntimeError(
                "No threshold set. Pass X_val (normal images) to fit(), or call calibrate()."
            )
        return (self.anomaly_scores(X) >= self.threshold).astype(np.int64)

    def anomaly_maps(self, X) -> np.ndarray:
        """Per-pixel anomaly map, shape (B, H, W), same size as the input images."""
        _, maps = self._infer(X, want_maps=True)
        return maps

    def calibrate(self, X, y) -> float:
        """Optional: replace the threshold with the one maximising F1 on (X, y).

        Needs labelled images with both classes (0 = normal, 1 = anomaly).
        Those images must not be reused for the final evaluation.

        Returns:
            The new threshold.
        """
        scores = self.anomaly_scores(X)
        y = np.asarray(y).astype(np.int64).ravel()
        if len(y) != len(scores):
            raise ValueError(f"Got {len(scores)} images but {len(y)} labels.")
        if not (y == 0).any() or not (y == 1).any():
            raise ValueError("calibrate() needs both normal (0) and anomalous (1) images.")

        best_threshold, best_f1 = None, -1.0
        for t in np.unique(scores):
            pred = scores >= t
            tp = np.sum(pred & (y == 1))
            fp = np.sum(pred & (y == 0))
            fn = np.sum(~pred & (y == 1))
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            if f1 > best_f1:
                best_threshold, best_f1 = float(t), f1
        self.threshold = best_threshold
        return best_threshold

    def save(self, filename) -> None:
        """Save weights, settings, threshold and input size to one file."""
        self._require_fitted()
        import anomalib

        torch.save(
            {
                "class": type(self).__name__,
                "config": self.get_config(),
                "input_size": list(self.input_size),
                "threshold": self.threshold,
                "anomalib_version": anomalib.__version__,
                "state_dict": {k: v.detach().cpu() for k, v in self.module.state_dict().items()},
            },
            filename,
        )

    @classmethod
    def load(cls, filename, **overrides) -> "AnomalibDetector":
        """Return a model restored from a file produced by ``save``.

        Keyword arguments override saved settings, e.g. ``accelerator="cpu"``
        when loading on another machine.
        """
        import anomalib

        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        if checkpoint["class"] != cls.__name__:
            raise ValueError(f"File holds a {checkpoint['class']}, not a {cls.__name__}.")
        if checkpoint["anomalib_version"] != anomalib.__version__:
            warnings.warn(
                f"Model saved with anomalib {checkpoint['anomalib_version']}, "
                f"loading with {anomalib.__version__}.",
                stacklevel=2,
            )

        model = cls(**{**checkpoint["config"], **overrides})
        model.input_size = tuple(checkpoint["input_size"])
        # pre_trained=False: all weights come from the file, nothing to download.
        model.module = model._build_module(model.input_size, pre_trained=False)
        _load_state_dict(model.module, checkpoint["state_dict"])
        model.module.eval()
        model.threshold = checkpoint["threshold"]
        return model

    # -- internals ------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self.module is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call fit() or load() first.")

    @torch.no_grad()
    def _infer(self, X, want_maps: bool) -> tuple[np.ndarray, np.ndarray | None]:
        self._require_fitted()
        X = _to_tensor(X)
        if tuple(X.shape[-2:]) != self.input_size:
            raise ValueError(
                f"{type(self).__name__} was fitted on {self.input_size[0]}x{self.input_size[1]} images, "
                f"got {X.shape[-2]}x{X.shape[-1]}. Load the images with the same size."
            )

        device = _resolve_device(self.accelerator)
        self.module.to(device).eval()
        scores, maps = [], []
        for start in range(0, len(X), self.eval_batch_size):
            batch = _to_float(X[start : start + self.eval_batch_size]).to(device)
            output = self.module(batch)
            scores.append(output.pred_score.detach().flatten().float().cpu())
            if want_maps:
                amap = output.anomaly_map.detach().float().cpu()
                maps.append(amap.reshape(len(batch), *amap.shape[-2:]))

        scores_np = torch.cat(scores).numpy() if scores else np.empty(0, dtype=np.float32)
        maps_np = torch.cat(maps).numpy() if want_maps and maps else None
        return scores_np, maps_np
