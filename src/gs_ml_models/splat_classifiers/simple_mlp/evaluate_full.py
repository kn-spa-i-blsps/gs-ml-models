"""Full evaluation of SimpleMLPClassifier, configured for the RTX 5090 machine.

Protocol:
    1. The data is split BY SCENE (set) into train / validation / test, because
       images from one scene are very similar to each other.
    2. Several configurations (including different input resolutions) are
       compared on the validation scenes, each one with several random seeds.
    3. The best configuration is evaluated ONCE on the test scenes and compared
       with trivial baselines (constant mean / median predictions).
    4. All results and the settings of the run are saved in "results_full/".
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from gs_ml_models.load_data import load_labels_csv
from gs_ml_models.splat_classifiers.simple_mlp.model import SimpleMLPClassifier


PROJECT = Path.cwd()
RESULTS = PROJECT / "results_full"

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
        "Could not find splatdata.csv in <project>/data or "
        "<project>/src/gs_ml_models/data"
    )
ROOT = DATA / "classification_data"

COLS = ["edge_fraction", "visgauss_fraction", "blur_fraction", "unreal_fraction"]

# Scene split. The test scenes are used only once, at the very end.
TRAIN_SETS = ["d1-1", "d2-1", "japanese-1", "574-1", "738-1", "curasao-1", "panama-1"]
VAL_SETS = ["marine-1", "642-1"]
TEST_SETS = ["d3-1", "saltpond-1"]

# Max images per scene. 150 gives ~1050 train / 300 val / 300 test images
# (about 6 GB of RAM). Use None to load every image of these scenes (needs
# roughly 25 GB of RAM).
N_PER_SET = 150

EPOCHS = 30
BATCH_SIZE = 16
SEEDS = [0, 1, 2]

# Configurations compared on the validation scenes. pooled_size is the size
# the image is downscaled to inside the network (480, 640) means no downscaling.
CONFIGS = [
    {"lr": 1e-3},  # 48x64

]


def load_sets(df, names, n_per_set):
    """Load up to n_per_set random images from each of the given scenes.

    The array is allocated once and filled image by image, which avoids the
    extra copies that np.stack / np.concatenate would make.

    Returns:
        X: float32 array (B, 3, 480, 640), values in [0, 1].
        y: float32 array (B, 4), the matching labels.
    """
    selected = []
    for name in names:
        rows = df[df["set"] == name]
        if n_per_set is not None and len(rows) > n_per_set:
            rows = rows.sample(n_per_set, random_state=0)
        selected.append(rows)
    rows = pd.concat(selected)

    X = np.empty((len(rows), 3, 480, 640), dtype=np.float32)
    for k, (set_name, img_id) in enumerate(zip(rows["set"], rows["id"])):
        with Image.open(ROOT / set_name / img_id) as img:
            pixels = np.asarray(img.convert("RGB"), dtype=np.float32)
        X[k] = pixels.transpose(2, 0, 1) / 255.0  # (H, W, 3) -> (3, H, W)
        if (k + 1) % 200 == 0:
            print(f"  loaded {k + 1}/{len(rows)}")
    y = rows[COLS].to_numpy(dtype=np.float32)
    return X, y


def evaluate(pred, y):
    """Return overall MAE and MSE plus the MAE of each label."""
    err = np.abs(pred - y)
    out = {"MAE": err.mean(), "MSE": ((pred - y) ** 2).mean()}
    for j, col in enumerate(COLS):
        out[f"MAE_{col}"] = err[:, j].mean()
    return out


def baseline_metrics(y_train, y):
    """Metrics of constant predictions (training mean / median) on labels y."""
    mean_pred = np.broadcast_to(y_train.mean(axis=0), y.shape)
    median_pred = np.broadcast_to(np.median(y_train, axis=0), y.shape)
    return {
        "baseline_mean": evaluate(mean_pred, y),
        "baseline_median": evaluate(median_pred, y),
    }


def train_model(cfg, seed, X, y):
    """Train a fresh model with the given config and random seed."""
    torch.manual_seed(seed)  # fixes the weight init and the batch shuffling
    model = SimpleMLPClassifier(epochs=EPOCHS, batch_size=BATCH_SIZE, **cfg)
    model.fit(X, y)
    return model


def main():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"PyTorch {torch.__version__}, device: {gpu_name}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, training will be very slow.")

    df = load_labels_csv(DATA / "splatdata.csv")
    RESULTS.mkdir(exist_ok=True)

    print("Loading data...")
    print("train scenes:")
    X_train, y_train = load_sets(df, TRAIN_SETS, N_PER_SET)
    print("validation scenes:")
    X_val, y_val = load_sets(df, VAL_SETS, N_PER_SET)
    print("test scenes:")
    X_test, y_test = load_sets(df, TEST_SETS, N_PER_SET)
    print(f"train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)} images")

    # Save the settings so the run can be reproduced and described later.
    with open(RESULTS / "run_settings.json", "w") as f:
        json.dump(
            {
                "date": datetime.now().isoformat(timespec="seconds"),
                "torch": torch.__version__,
                "device": gpu_name,
                "data": str(DATA),
                "train_sets": TRAIN_SETS,
                "val_sets": VAL_SETS,
                "test_sets": TEST_SETS,
                "n_per_set": N_PER_SET,
                "n_images": {
                    "train": len(X_train),
                    "val": len(X_val),
                    "test": len(X_test),
                },
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seeds": SEEDS,
                "configs": CONFIGS,
            },
            f,
            indent=2,
        )

    # Step 1: compare configurations on the validation scenes

    search_rows = []
    for cfg_id, cfg in enumerate(CONFIGS):
        for seed in SEEDS:
            print(f"\n=== config {cfg_id} {cfg}, seed {seed} ===")
            start = time.time()
            model = train_model(cfg, seed, X_train, y_train)
            search_rows.append(
                {
                    "cfg_id": cfg_id,
                    "config": str(cfg),
                    "seed": seed,
                    "train_MAE": evaluate(model.predict(X_train), y_train)["MAE"],
                    "val_MAE": evaluate(model.predict(X_val), y_val)["MAE"],
                    "seconds": round(time.time() - start),
                }
            )
            # Save after every run so nothing is lost if the script is interrupted.
            pd.DataFrame(search_rows).to_csv(RESULTS / "validation_runs.csv", index=False)
            del model
            torch.cuda.empty_cache()

    search = pd.DataFrame(search_rows)

    summary = search.groupby(["cfg_id", "config"])[["train_MAE", "val_MAE"]].agg(
        ["mean", "std"]
    )
    summary.to_csv(RESULTS / "validation_summary.csv")
    print("\nValidation results (mean and std over seeds):")
    print(summary.round(3))

    val_base = baseline_metrics(y_train, y_val)
    print("\nValidation baselines (MAE):")
    print({k: round(float(v["MAE"]), 3) for k, v in val_base.items()})

    best_id = int(search.groupby("cfg_id")["val_MAE"].mean().idxmin())
    best_cfg = CONFIGS[best_id]
    print(f"\nBest configuration on validation: {best_cfg}")

    # Step 2: evaluate the best configuration once on the test scenes

    test_rows = []
    for seed in SEEDS:
        print(f"\n=== TEST run, seed {seed} ===")
        model = train_model(best_cfg, seed, X_train, y_train)
        test_rows.append(
            {
                "seed": seed,
                "train_MAE": evaluate(model.predict(X_train), y_train)["MAE"],
                **evaluate(model.predict(X_test), y_test),
            }
        )
        if seed == SEEDS[0]:
            model.save(RESULTS / "best_model.pt")
        del model
        torch.cuda.empty_cache()

    test = pd.DataFrame(test_rows)
    test.to_csv(RESULTS / "test_runs.csv", index=False)

    test_summary = test.drop(columns="seed").agg(["mean", "std"]).round(3).T
    test_summary.to_csv(RESULTS / "test_summary.csv")
    print(f"\nTest results of {best_cfg} (mean and std over seeds):")
    print(test_summary)

    test_base = pd.DataFrame(baseline_metrics(y_train, y_test)).round(3)
    test_base.to_csv(RESULTS / "test_baselines.csv")
    print("\nTest baselines (constant predictions from the training labels):")
    print(test_base)

    print(f"\nDone. Results saved in {RESULTS}")


if __name__ == "__main__":
    main()
