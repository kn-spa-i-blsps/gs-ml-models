"""GroupKFold cross-validation and final training for XGBoostDefectClassifier."""

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold

from gs_ml_models.load_data import load_labels_csv
from gs_ml_models.splat_classifiers.xgboost_baseline.model import (
    EPSILON,
    LABELS,
    XGBoostDefectClassifier,
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
SAVE_PATH = PROJECT / "model" / "image-defect-baseline.json"


def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_clipped = np.clip(y_pred, EPSILON, 1.0 - EPSILON)
    return float(
        -np.mean(
            y_true * np.log(y_pred_clipped)
            + (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )
    )


def main() -> None:
    df = load_labels_csv(DATA / "splatdata.csv")
    y = df[LABELS].to_numpy(dtype=np.float32)
    groups = df["set"].to_numpy()

    X = np.empty((len(df), 3, 480, 640), dtype=np.float32)
    for idx, (img_set, img_id) in enumerate(zip(df["set"], df["id"])):
        with Image.open(IMG_DIR / img_set / img_id) as img:
            pixels = np.asarray(img.convert("RGB"), dtype=np.float32)
        X[idx] = pixels.transpose(2, 0, 1) / 255.0

    clf = XGBoostDefectClassifier()
    print("Extracting image features...")
    features = clf.extract_features(X)

    splitter = GroupKFold(n_splits=5)
    f1_scores, bce_scores = [], []
    per_label_f1 = np.zeros((5, len(LABELS)), dtype=np.float32)
    per_label_mae = np.zeros((5, len(LABELS)), dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, y, groups=groups)):
        fold_clf = XGBoostDefectClassifier()
        fold_clf.model.fit(features[train_idx], y[train_idx])

        y_pred_train = fold_clf.model.predict(features[train_idx])
        y_pred_test = fold_clf.model.predict(features[test_idx])

        thresholds = fold_clf._find_best_f1_thresholds(y[train_idx], y_pred_train)
        y_test_bin = (y[test_idx] >= 0.5).astype(int)
        y_pred_bin = (y_pred_test >= thresholds).astype(int)

        f1 = f1_score(y_test_bin, y_pred_bin, average="macro")
        bce = _log_loss(y[test_idx], y_pred_test)
        f1_scores.append(f1)
        bce_scores.append(bce)

        print(f"\nFOLD {fold + 1}/5 | F1: {f1:.4f} | BCE: {bce:.4f}")
        for i, label in enumerate(LABELS):
            per_label_f1[fold, i] = f1_score(y_test_bin[:, i], y_pred_bin[:, i], zero_division=0)
            per_label_mae[fold, i] = mean_absolute_error(y[test_idx, i], y_pred_test[:, i])
            print(f"  {label:<20} | F1: {per_label_f1[fold, i]:.4f} | MAE: {per_label_mae[fold, i]:.4f}")

    print("\nSummary")
    print(f"Mean F1: {np.mean(f1_scores):.4f} (+- {np.std(f1_scores):.4f})")
    print(f"Mean BCE: {np.mean(bce_scores):.4f} (+- {np.std(bce_scores):.4f})")

    print("\nTraining final model on full dataset...")
    final_clf = XGBoostDefectClassifier()
    final_clf.fit(X, y)
    final_clf.save(SAVE_PATH)
    print(f"Saved final model to {SAVE_PATH}")


if __name__ == "__main__":
    main()