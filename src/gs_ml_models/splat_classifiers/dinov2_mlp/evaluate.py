"""GroupKFold cross-validation and final training for DinoV2DefectClassifier."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold

from gs_ml_models import TORCH_DEVICE
from gs_ml_models.load_data import load_labels_csv
from gs_ml_models.splat_classifiers.dinov2_mlp.model import (
    EPSILON,
    LABELS,
    DinoV2DefectClassifier,
)

PROJECT = Path.cwd()
DATA = next(
    (
        p
        for p in (PROJECT / "data", PROJECT / "src" / "gs_ml_models" / "data")
        if (p / "splatdata.csv").exists()
    ),
    None,
)
if DATA is None:
    raise FileNotFoundError(
        "Could not find splatdata.csv in <project>/data or <project>/src/gs_ml_models/data"
    )

IMG_DIR = DATA / "classification_data"
SAVE_PATH = PROJECT / "model" / "image-defect-dinov2.pt"


def _f1_per_label(y_true_bin: torch.Tensor, y_pred_bin: torch.Tensor) -> torch.Tensor:
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum(dim=0)
    fp = ((y_true_bin == 0) & (y_pred_bin == 1)).sum(dim=0)
    fn = ((y_true_bin == 1) & (y_pred_bin == 0)).sum(dim=0)
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    return (2.0 * precision * recall) / (precision + recall + EPSILON)


def _log_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_pred_clipped = torch.clip(y_pred, EPSILON, 1.0 - EPSILON)
    bce = -torch.mean(
        y_true * torch.log(y_pred_clipped)
        + (1.0 - y_true) * torch.log(1.0 - y_pred_clipped)
    )
    return float(bce.item())


def main() -> None:
    device = torch.device(TORCH_DEVICE)
    df = load_labels_csv(DATA / "splatdata.csv")
    groups = df["set"].to_numpy()

    X = np.empty((len(df), 3, 480, 640), dtype=np.float32)
    for idx, (img_set, img_id) in enumerate(zip(df["set"], df["id"])):
        with Image.open(IMG_DIR / img_set / img_id) as img:
            pixels = np.asarray(img.convert("RGB"), dtype=np.float32)
        X[idx] = pixels.transpose(2, 0, 1) / 255.0

    y = torch.tensor(df[LABELS].to_numpy(dtype=np.float32), device=device)

    extractor = DinoV2DefectClassifier(device=device)
    print("Extracting features using DINOv2...")
    features = extractor.extract_features(X)

    splitter = GroupKFold(n_splits=5)
    f1_scores = torch.zeros(5, dtype=torch.float32, device=device)
    bce_scores = torch.zeros(5, dtype=torch.float32, device=device)
    per_label_f1 = torch.zeros((5, len(LABELS)), dtype=torch.float32, device=device)
    per_label_mae = torch.zeros((5, len(LABELS)), dtype=torch.float32, device=device)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(features.cpu(), y.cpu(), groups=groups)
    ):
        tr = torch.tensor(train_idx, dtype=torch.long, device=device)
        te = torch.tensor(test_idx, dtype=torch.long, device=device)

        fold_clf = DinoV2DefectClassifier(device=device)
        fold_clf.fit_from_features(features[tr], y[tr])

        fold_clf.net.eval()
        with torch.no_grad():
            y_pred_test = torch.sigmoid(fold_clf.net(features[te]))

        thresholds = fold_clf.thresholds.to(device)
        y_test_bin = (y[te] >= 0.5).float()
        y_pred_bin = (y_pred_test >= thresholds).float()

        f1_lbl = _f1_per_label(y_test_bin, y_pred_bin)
        mae_lbl = torch.abs(y[te] - y_pred_test).mean(dim=0)
        bce = _log_loss(y[te], y_pred_test)

        f1_scores[fold] = f1_lbl.mean()
        bce_scores[fold] = bce
        per_label_f1[fold] = f1_lbl
        per_label_mae[fold] = mae_lbl

        print(f"\nFOLD {fold + 1}/5 | F1: {f1_lbl.mean().item():.4f} | BCE: {bce:.4f}")
        for i, label in enumerate(LABELS):
            print(
                f"  {label:<20} | F1: {f1_lbl[i].item():.4f} | MAE: {mae_lbl[i].item():.4f}"
            )

    print("\nSummary")
    print(
        f"Mean F1: {f1_scores.mean().item():.4f} (+- {f1_scores.std(correction=0).item():.4f}) | "
        f"Mean BCE: {bce_scores.mean().item():.4f} (+- {bce_scores.std(correction=0).item():.4f})"
    )

    print("\nTraining final model on full dataset...")
    final_clf = DinoV2DefectClassifier(device=device)
    final_clf.fit_from_features(features, y)
    final_clf.save(SAVE_PATH)
    print(f"Saved final model to {SAVE_PATH}")


if __name__ == "__main__":
    main()