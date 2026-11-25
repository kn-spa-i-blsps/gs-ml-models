"""DINOv2 feature extractor + MLP head classifier.

Exposes the Splat Classifier API: fit / train / predict / save / load / predict_extra.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, TensorDataset

from gs_ml_models import TORCH_DEVICE

warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

DINO_REPO = "facebookresearch/dinov2"
DINO_DEFAULT_MODEL = "dinov2_vits14"
DINO_IMG_HEIGHT = 420
DINO_IMG_WIDTH = 560
EPSILON = 1e-6

LABELS = ["edge_fraction", "visgauss_fraction", "blur_fraction", "unreal_fraction"]


class DefectClassifierNet(nn.Module):
    """MLP network mapping 1152 DINOv2 features to 4 defect logits."""

    def __init__(self, in_features: int = 1152) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DinoV2DefectClassifier:
    """Wrapper exposing the Splat Classifier API for DINOv2 + MLP."""

    def __init__(
        self,
        dino_model_name: str = DINO_DEFAULT_MODEL,
        image_size: tuple[int, int] = (DINO_IMG_HEIGHT, DINO_IMG_WIDTH),
        epochs: int = 40,
        batch_size: int = 128,
        extract_batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        device=TORCH_DEVICE,
        seed: int = 67,
    ) -> None:
        self.dino_model_name = dino_model_name
        self.image_size = tuple(image_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.extract_batch_size = extract_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.seed = seed

        self.in_features = 1152
        self._dino: nn.Module | None = None
        self._reset()

    def _reset(self) -> None:
        self.net = DefectClassifierNet(in_features=self.in_features).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.thresholds = torch.full((len(LABELS),), 0.5, dtype=torch.float32)

    def _get_dino(self) -> nn.Module:
        if self._dino is None:
            dino = torch.hub.load(DINO_REPO, self.dino_model_name)
            dino = dino.to(self.device)
            dino.eval()
            for param in dino.parameters():
                param.requires_grad_(False)
            self._dino = dino
        return self._dino

    @torch.no_grad()
    def extract_features(self, X) -> torch.Tensor:
        """Extract [cls_token, mean_patch, max_patch] embeddings from (B, 3, H, W) RGB in [0, 1]."""
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        dino = self._get_dino()

        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)

        extracted = []
        for start in range(0, len(X_tensor), self.extract_batch_size):
            batch = X_tensor[start : start + self.extract_batch_size].to(self.device)
            batch = TF.resize(batch, list(self.image_size), antialias=True)
            batch = (batch - mean) / std

            feature_dict = dino.forward_features(batch)
            cls_token = feature_dict["x_norm_clstoken"]
            patch_tokens = feature_dict["x_norm_patchtokens"]

            mean_patch = patch_tokens.mean(dim=1)
            max_patch = patch_tokens.max(dim=1).values
            combined = torch.cat([cls_token, mean_patch, max_patch], dim=1)
            extracted.append(combined.float())

        return torch.cat(extracted, dim=0)

    @staticmethod
    @torch.no_grad()
    def _find_best_f1_thresholds(
        y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        y_true_bin = (y_true >= 0.5).float()
        candidate_thresholds = torch.linspace(0.2, 0.7, 51, device=y_pred.device)
        preds_bin = (
            y_pred.unsqueeze(0) >= candidate_thresholds.view(51, 1, 1)
        ).float()
        trues_bin = y_true_bin.unsqueeze(0)

        tp = ((trues_bin == 1) & (preds_bin == 1)).sum(dim=1)
        fp = ((trues_bin == 0) & (preds_bin == 1)).sum(dim=1)
        fn = ((trues_bin == 1) & (preds_bin == 0)).sum(dim=1)

        precision = tp / (tp + fp + EPSILON)
        recall = tp / (tp + fn + EPSILON)
        f1_matrix = (2.0 * precision * recall) / (precision + recall + EPSILON)

        best_indices = torch.argmax(f1_matrix, dim=0)
        return candidate_thresholds[best_indices].cpu()

    def fit_from_features(self, features: torch.Tensor, y: torch.Tensor) -> None:
        """Train the MLP head directly on pre-extracted DINOv2 features."""
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        self.in_features = features.shape[1]
        self._reset()
        self.train_from_features(features, y)

    def train_from_features(self, features: torch.Tensor, y: torch.Tensor) -> None:
        """Continue training the MLP head on pre-extracted DINOv2 features."""
        features_dev = features.to(self.device)
        y_dev = y.to(self.device)

        # Restore base learning rate so CosineAnnealingLR does not start from 0.0 on subsequent train() calls
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
            param_group.pop("initial_lr", None)

        loss_fn = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        dataset = TensorDataset(features_dev, y_dev)
        # Prevent BatchNorm1d failure when len(dataset) % batch_size == 1
        drop_last = len(dataset) > self.batch_size and (
            len(dataset) % self.batch_size == 1
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last
        )

        self.net.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                self.optimizer.zero_grad(set_to_none=False)
                logits = self.net(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                self.optimizer.step()
            scheduler.step()

        self.net.eval()
        with torch.no_grad():
            preds = torch.sigmoid(self.net(features_dev))
            self.thresholds = self._find_best_f1_thresholds(y_dev, preds)

    def fit(self, X, y) -> None:
        """Train from scratch on images X (B, 3, 480, 640) and labels y (B, 4)."""
        features = self.extract_features(X)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        self.fit_from_features(features, y_tensor)

    def train(self, X, y) -> None:
        """Continue training without resetting weights or optimizer."""
        features = self.extract_features(X)
        y_tensor = torch.as_tensor(y, dtype=torch.float32)
        self.train_from_features(features, y_tensor)

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        """Return an array of shape (B, 4) with values in [0, 1]."""
        self.net.eval()
        features = self.extract_features(X)
        probs = torch.sigmoid(self.net(features))
        return probs.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_extra(self, X) -> dict[str, Any]:
        """Return binary defect flags, F1 thresholds, and raw MLP logits."""
        self.net.eval()
        features = self.extract_features(X)
        logits = self.net(features)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        thresholds_np = self.thresholds.numpy()
        return {
            "binary_predictions": (probs >= thresholds_np).astype(int),
            "thresholds": thresholds_np.copy(),
            "logits": logits.cpu().numpy(),
        }

    def save(self, filename) -> None:
        """Save checkpoint in the exact format produced by the training script."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dinov2_backbone": self.dino_model_name,
                "image_size": self.image_size,
                "thresholds": self.thresholds.cpu(),
                "model_state_dict": {
                    k: v.cpu() for k, v in self.net.state_dict().items()
                },
            },
            path,
        )

    @classmethod
    def load(cls, filename, device=TORCH_DEVICE) -> "DinoV2DefectClassifier":
        """Restore a DinoV2DefectClassifier from a saved .pt checkpoint."""
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model_state_dict"]
        in_features = state_dict["net.0.weight"].shape[0]

        instance = cls(
            dino_model_name=checkpoint.get("dinov2_backbone", DINO_DEFAULT_MODEL),
            image_size=tuple(
                checkpoint.get("image_size", (DINO_IMG_HEIGHT, DINO_IMG_WIDTH))
            ),
            device=device,
        )
        instance.in_features = in_features
        instance._reset()
        instance.net.load_state_dict(state_dict)
        instance.net.eval()

        if "thresholds" in checkpoint:
            instance.thresholds = checkpoint["thresholds"].float().cpu()

        return instance