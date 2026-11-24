"""XGBoost classifier using 27 handcrafted image quality features.

Exposes the Splat Classifier API: fit / train / predict / save / load / predict_extra.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score
from xgboost import XGBRegressor

GRID_WIDTH = 12
GRID_HEIGHT = 8
EPSILON = 1e-6

LABELS = ["edge_fraction", "visgauss_fraction", "blur_fraction", "unreal_fraction"]


def _make_line_kernels(length: int = 15) -> list[np.ndarray]:
    kernels = []
    c = length // 2
    for angle_idx in range(4):
        theta = angle_idx * np.pi / 4.0
        k = np.zeros((length, length), dtype=np.uint8)
        dx = int(round(c * np.cos(theta)))
        dy = int(round(c * np.sin(theta)))
        cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, thickness=1)
        kernels.append(k)
    return kernels


LINE_KERNELS_15 = _make_line_kernels(15)
DISK_KERNEL_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def _compute_single_image_features(img_bgr: np.ndarray) -> np.ndarray:
    """Compute the 27 image defect features from a BGR uint8 image (H, W, 3)."""
    height, width, _ = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_std = float(np.std(img_gray))

    lap = cv2.Laplacian(img_gray, cv2.CV_32F)
    lap_abs = np.abs(lap)

    laplacian_var = float(np.var(lap))
    laplacian_mean = float(np.mean(lap_abs))
    laplacian_mean_norm = float(laplacian_mean / (gray_std + EPSILON))

    patch_height, patch_width = height // GRID_HEIGHT, width // GRID_WIDTH

    patch_lap_vars = np.array(
        [
            np.var(
                lap[
                    r * patch_height : (r + 1) * patch_height,
                    c * patch_width : (c + 1) * patch_width,
                ]
            )
            for r in range(GRID_HEIGHT)
            for c in range(GRID_WIDTH)
        ],
        dtype=np.float32,
    )

    patch_lap_median = float(np.median(patch_lap_vars))
    patch_lap_p25 = float(np.percentile(patch_lap_vars, 25))
    patch_lap_p90 = float(np.percentile(patch_lap_vars, 90))
    smooth_patches_lt2 = float(np.mean(patch_lap_vars < 2.0))
    smooth_patches_lt10 = float(np.mean(patch_lap_vars < 10.0))

    img_blurred = cv2.GaussianBlur(img_gray, (0, 0), 2.0)

    grad_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    blurred_grad_x = cv2.Sobel(img_blurred, cv2.CV_32F, 1, 0, ksize=3)
    blurred_grad_y = cv2.Sobel(img_blurred, cv2.CV_32F, 0, 1, ksize=3)

    sobel_mag = np.sqrt(grad_x**2 + grad_y**2)
    blurred_sobel_mag = np.sqrt(blurred_grad_x**2 + blurred_grad_y**2)

    grad_retention_global = float(
        np.mean(blurred_sobel_mag) / (np.mean(sobel_mag) + EPSILON)
    )

    img_blurred_fine = cv2.GaussianBlur(img_gray, (0, 0), 1.5)
    img_blurred_wide = cv2.GaussianBlur(img_gray, (0, 0), 8.0)
    mf_signed = img_blurred_fine - img_blurred_wide
    mf_map = np.abs(mf_signed)

    patch_grad_retention = []
    # Preserved as zeros to match the exact feature vector used during pre-training
    grid_lap_var = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    grid_mf_mean = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)

    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
            r_0, r_1 = row * patch_height, (row + 1) * patch_height
            c_0, c_1 = column * patch_width, (column + 1) * patch_width

            patch_smag = sobel_mag[r_0:r_1, c_0:c_1]
            patch_bsmag = blurred_sobel_mag[r_0:r_1, c_0:c_1]
            patch_grad_retention.append(
                float(np.mean(patch_bsmag) / (np.mean(patch_smag) + EPSILON))
            )
            grid_mf_mean[row, column] = np.mean(mf_map[r_0:r_1, c_0:c_1])

    patch_grad_retention_arr = np.array(patch_grad_retention)
    grad_ret_p25 = float(np.percentile(patch_grad_retention_arr, 25))
    grad_ret_p90 = float(np.percentile(patch_grad_retention_arr, 90))

    border_ring_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    border_ring_mask[0, :] = True
    border_ring_mask[-1, :] = True
    border_ring_mask[:, 0] = True
    border_ring_mask[:, -1] = True

    border_mf_map = grid_mf_mean[border_ring_mask]
    inner_mf_map = grid_mf_mean[~border_ring_mask]

    border_lap_var = grid_lap_var[border_ring_mask]
    inner_lap_var = grid_lap_var[~border_ring_mask]

    inner_lap_med = float(np.median(inner_lap_var))
    inner_mf_med = float(np.median(inner_mf_map))

    border_smoothest_4_lap_med = float(np.median(np.sort(border_lap_var)[:4]))
    border_smoothest_4_mf_med = float(np.median(np.sort(border_mf_map)[:4]))

    border_min_4_lap_to_inner = float(
        border_smoothest_4_lap_med / (inner_lap_med + EPSILON)
    )
    border_p15_lap_to_inner = float(
        np.percentile(border_lap_var, 15) / (inner_lap_med + EPSILON)
    )
    border_min4_mf_to_inner = float(
        border_smoothest_4_mf_med / (inner_mf_med + EPSILON)
    )
    abs_dead_border_ratio = float(np.mean(border_lap_var < 2.0))

    border_h, border_w = int(0.11 * height), int(0.11 * width)
    border_laps = [
        lap_abs[:border_h, :],
        lap_abs[-border_h:, :],
        lap_abs[:, :border_w],
        lap_abs[:, -border_w:],
    ]
    border_smooth_lap_max = float(np.max([np.mean(b < 1.0) for b in border_laps]))

    hf_signed = img_gray - img_blurred
    hf_gx = cv2.Sobel(hf_signed, cv2.CV_32F, 1, 0, ksize=3)
    hf_gy = cv2.Sobel(hf_signed, cv2.CV_32F, 0, 1, ksize=3)
    hf_xx = cv2.GaussianBlur(hf_gx * hf_gx, (0, 0), 1.8)
    hf_yy = cv2.GaussianBlur(hf_gy * hf_gy, (0, 0), 1.8)
    hf_xy = cv2.GaussianBlur(hf_gx * hf_gy, (0, 0), 1.8)
    hf_coh = np.sqrt((hf_xx - hf_yy) ** 2 + 4.0 * (hf_xy**2)) / (
        hf_xx + hf_yy + 1.0
    )
    hf_coh_mean = float(np.mean(hf_coh))
    hf_coh_p90 = float(np.percentile(hf_coh, 90))

    mf_gx = cv2.Sobel(mf_signed, cv2.CV_32F, 1, 0, ksize=3)
    mf_gy = cv2.Sobel(mf_signed, cv2.CV_32F, 0, 1, ksize=3)
    mf_xx = cv2.GaussianBlur(mf_gx * mf_gx, (0, 0), 5.0)
    mf_yy = cv2.GaussianBlur(mf_gy * mf_gy, (0, 0), 5.0)
    mf_xy = cv2.GaussianBlur(mf_gx * mf_gy, (0, 0), 5.0)
    mf_coh = np.sqrt((mf_xx - mf_yy) ** 2 + 4.0 * (mf_xy**2)) / (
        mf_xx + mf_yy + 0.5
    )
    mf_coh_p90 = float(np.percentile(mf_coh, 90))

    open_disk = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, DISK_KERNEL_9)
    close_disk = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, DISK_KERNEL_9)
    tophat = img_gray - open_disk
    blackhat = close_disk - img_gray
    needle_resp = np.maximum(tophat, blackhat)
    needle_p99_to_mean = float(
        np.percentile(needle_resp, 99) / (np.mean(needle_resp) + EPSILON)
    )

    sup_open_line = np.max(
        [cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, k) for k in LINE_KERNELS_15],
        axis=0,
    )
    inf_close_line = np.min(
        [cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, k) for k in LINE_KERNELS_15],
        axis=0,
    )
    line_only = np.maximum(
        np.maximum(0.0, sup_open_line - open_disk),
        np.maximum(0.0, close_disk - inf_close_line),
    )
    line_to_disk_ratio = float(
        np.mean(line_only) / (np.mean(needle_resp) + EPSILON)
    )

    j_xx = cv2.GaussianBlur(grad_x * grad_x, (15, 15), 3.0)
    j_yy = cv2.GaussianBlur(grad_y * grad_y, (15, 15), 3.0)
    j_xy = cv2.GaussianBlur(grad_x * grad_y, (15, 15), 3.0)
    coherence = np.sqrt((j_xx - j_yy) ** 2 + 4.0 * (j_xy**2)) / (
        j_xx + j_yy + 1.0
    )
    st_coh_mean = float(np.mean(coherence))
    st_coh_p95 = float(np.percentile(coherence, 95))

    black_void = float(np.mean((img_gray < 10.0) & (sobel_mag < 2.5)))

    b_mean = float(np.mean(img_bgr[:, :, 0]))
    g_mean = float(np.mean(img_bgr[:, :, 1]))
    r_mean = float(np.mean(img_bgr[:, :, 2]))
    max_rgb = max(b_mean, g_mean, r_mean)
    min_rgb = min(b_mean, g_mean, r_mean)
    color_cast_ratio = float((max_rgb - min_rgb) / (max_rgb + min_rgb + 1.0))

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_centered = img_lab[:, :, 1] - 128.0
    b_centered = img_lab[:, :, 2] - 128.0
    chroma = np.sqrt(a_centered**2 + b_centered**2)
    chroma_p95 = float(np.percentile(chroma, 95))

    h_crop, w_crop = GRID_HEIGHT * patch_height, GRID_WIDTH * patch_width
    gray_blocks = img_gray[:h_crop, :w_crop].reshape(
        GRID_HEIGHT, patch_height, GRID_WIDTH, patch_width
    )
    grid_gray_std = gray_blocks.std(axis=(1, 3)).astype(np.float32)
    patch_std_cv = float(
        np.std(grid_gray_std) / (np.mean(grid_gray_std) + EPSILON)
    )

    return np.array(
        [
            laplacian_var,
            laplacian_mean,
            laplacian_mean_norm,
            patch_lap_median,
            patch_lap_p25,
            patch_lap_p90,
            smooth_patches_lt2,
            smooth_patches_lt10,
            grad_retention_global,
            grad_ret_p25,
            grad_ret_p90,
            border_min_4_lap_to_inner,
            border_p15_lap_to_inner,
            abs_dead_border_ratio,
            border_smooth_lap_max,
            border_min4_mf_to_inner,
            st_coh_mean,
            st_coh_p95,
            hf_coh_mean,
            hf_coh_p90,
            mf_coh_p90,
            needle_p99_to_mean,
            line_to_disk_ratio,
            color_cast_ratio,
            chroma_p95,
            patch_std_cv,
            black_void,
        ],
        dtype=np.float32,
    )


class XGBoostDefectClassifier:
    """Wrapper exposing the Splat Classifier API for the 27-feature XGBoost model."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 4,
        learning_rate: float = 0.03,
        min_child_weight: int = 10,
        reg_lambda: float = 5.0,
        reg_alpha: float = 1.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 67,
        n_jobs: int = -1,
    ) -> None:
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        self._reset()

    def _reset(self) -> None:
        self.model = XGBRegressor(**self.params)
        self.thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
        self._is_fitted = False

    def extract_features(self, X) -> np.ndarray:
        """Convert (B, 3, H, W) RGB images in [0, 1] into (B, 27) tabular features."""
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float32)

        # (B, 3, H, W) RGB [0, 1] -> (B, H, W, 3) BGR uint8 [0, 255]
        imgs_bgr = np.transpose(X_np, (0, 2, 3, 1))[:, :, :, ::-1]
        imgs_uint8 = np.clip(np.round(imgs_bgr * 255.0), 0, 255).astype(np.uint8)

        if len(imgs_uint8) <= 8:
            features = [_compute_single_image_features(img) for img in imgs_uint8]
        else:
            try:
                with ProcessPoolExecutor() as executor:
                    features = list(
                        executor.map(
                            _compute_single_image_features, imgs_uint8, chunksize=32
                        )
                    )
            except RuntimeError:
                features = [_compute_single_image_features(img) for img in imgs_uint8]

        return np.vstack(features)

    @staticmethod
    def _find_best_f1_thresholds(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true_bin = (y_true >= 0.5).astype(int)
        best_thresholds = np.full(y_true_bin.shape[1], 0.5, dtype=np.float32)
        candidates = np.linspace(0.2, 0.7, 51)

        for i in range(y_true_bin.shape[1]):
            best_score = -1.0
            for threshold in candidates:
                score = f1_score(
                    y_true_bin[:, i],
                    (y_pred[:, i] >= threshold).astype(int),
                    zero_division=0,
                )
                if score > best_score:
                    best_score = score
                    best_thresholds[i] = threshold

        return best_thresholds

    def fit(self, X, y) -> None:
        """Train from scratch on images X (B, 3, 480, 640) and labels y (B, 4)."""
        self._reset()
        self.train(X, y)

    def train(self, X, y) -> None:
        """Continue boosting on additional data without resetting existing trees."""
        y_np = (
            y.detach().cpu().numpy()
            if isinstance(y, torch.Tensor)
            else np.asarray(y, dtype=np.float32)
        )
        features = self.extract_features(X)
        booster = self.model.get_booster() if self._is_fitted else None

        self.model.fit(features, y_np, xgb_model=booster)
        self._is_fitted = True

        preds = self.model.predict(features)
        self.thresholds = self._find_best_f1_thresholds(y_np, preds)

    def predict(self, X) -> np.ndarray:
        """Return an array of shape (B, 4) with values in [0, 1]."""
        features = self.extract_features(X)
        preds = self.model.predict(features)
        return np.clip(preds, 0.0, 1.0).astype(np.float32)

    def predict_extra(self, X) -> dict[str, Any]:
        """Return binary defect flags, F1 thresholds, and extracted tabular features."""
        features = self.extract_features(X)
        probs = np.clip(self.model.predict(features), 0.0, 1.0).astype(np.float32)
        return {
            "binary_predictions": (probs >= self.thresholds).astype(int),
            "thresholds": self.thresholds.copy(),
            "features": features,
        }

    def save(self, filename) -> None:
        """Save the XGBoost model in JSON format compatible with load()."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    @classmethod
    def load(cls, filename) -> "XGBoostDefectClassifier":
        """Restore an XGBoostDefectClassifier from a saved JSON file."""
        instance = cls()
        instance.model.load_model(str(filename))
        instance._is_fitted = True
        return instance