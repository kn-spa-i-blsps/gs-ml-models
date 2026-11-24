# Anomaly Detection

One-class (unsupervised) image-level anomaly detection: models learn from
normal images only and score how unusual a new image is.

## Setup

The models are built on [Anomalib](https://github.com/open-edge-platform/anomalib),
installed as an optional extra:

```bash
uv pip install -e ".[anomaly]"
```

## Quick start

```python
from gs_ml_models.anomaly_detection.data import load_manifest, load_images
from gs_ml_models.anomaly_detection.registry import get_model

df = load_manifest("data/<session>/manifest.csv")
X_train = load_images(df[(df.split == "train") & (df.source == "real")], size=256, resize="center_crop")
X_val   = load_images(df[df.split == "val"],  size=256, resize="center_crop")
X_test  = load_images(df[df.split == "test"], size=256, resize="center_crop")

model = get_model("reverse_distillation", seed=0)
model.fit(X_train, X_val)          # early stopping + threshold from X_val
scores = model.anomaly_scores(X_test)
labels = model.predict(X_test)
model.save("rd.pt")
```

## API

Every model is a class with the methods below.

### Images

`X` is a batch of shape `(B, 3, H, W)`, as a NumPy array or Torch tensor, either
**uint8 with values 0-255** or **float with values in [0, 1]**. uint8 takes 4x
less memory; it is converted to float in small batches right before the model.

Any `H, W` is accepted. A model remembers the size it was fitted on and raises
an error if later calls use another size. The size is chosen when loading the
images (`load_images(..., size=...)`), not by the model, so the image is
resized only once.

### Methods

- `fit(X, X_val=None)` trains from scratch on normal images `X`.
  `X_val` are **normal-only** validation images of the same size. They are used
  1. to set the decision threshold (see below), and
  2. for trainable models, for early stopping: the mean anomaly score on
     `X_val` is checked every `check_val_every_n_epoch` epochs (default 5);
     training stops after `patience` checks without improvement (default 3,
     i.e. 15 epochs) or at `max_epochs` (default 200). The weights of the best
     check are kept.

  Without `X_val`, no threshold is set and trainable models run for
  `max_epochs`.

- `train(X)` is **not supported yet** and raises `NotImplementedError`.
  Continuing training on new data only makes a model forget earlier data; use
  `fit` on all the data. To be revisited when multi-session data exist.

- `anomaly_scores(X)` returns raw scores, shape `(B,)`. Higher = more anomalous.
  Scores are not normalised and are only comparable within one model.

- `predict(X)` returns 0/1 labels, shape `(B,)`: 1 if `score >= threshold`.
  Raises an error if no threshold is set.

- `save(filename)` stores weights, settings, threshold and input size in one
  file. `load(filename, **overrides)` restores it (e.g.
  `load("rd.pt", accelerator="cpu")` on another machine).

### Threshold

By default the threshold is the **95th percentile of the scores on `X_val`**,
i.e. about 5% false alarms on normal images (`threshold_percentile`
parameter). No anomalies are needed, which is how a detector would be set up
on a new site.

Fix the percentile **before** looking at test results. It is cheap to report
results at several percentiles (e.g. 95 and 99).

### Optional methods

- `anomaly_maps(X)` returns per-pixel anomaly maps, shape `(B, H, W)`, same
  size as the input. Useful to check *where* a model looks.
- `calibrate(X, y)` replaces the threshold with the one maximising F1 on
  labelled images `y` (0 = normal, 1 = anomaly). Needs both classes. The images
  used here must not be reused for evaluation.

### Common settings

All models accept (see `base.AnomalibDetector`): `threshold_percentile`,
`batch_size`, `eval_batch_size`, `max_epochs`, `check_val_every_n_epoch`,
`patience`, `min_delta`, `seed`, `accelerator` (`"auto"` = CUDA > MPS > CPU)
and `work_dir` (where Lightning checkpoints go during `fit`; a temporary
folder by default).

## Data

`data.py` reads a **manifest**: a CSV with one row per image.

| column | required | values |
|---|---|---|
| `path` | yes | relative to the manifest's folder, or absolute |
| `split` | yes | `train`, `val`, `test`, `splat_input` |
| `source` | yes | `real`, `splat` |
| `label` | yes | `0` normal, `1` anomaly |
| `run`, `t` | no | recording and time in it (s) |
| `anomaly_type` | no | e.g. `holes`, `broken_piece` |
| `splat_id` | no | which splat a render comes from |
| `session` | no | recording session |

`splat_input` marks real frames used to build a splat. They are never given to
a detector; they are listed so the checks can prove it.

`load_manifest` checks these rules and stops with a list of violations:

1. every image appears once,
2. frames used to build a splat appear in no other split (also matched by
   `run` + `t` when present),
3. anomalies appear only in `test`,
4. `val` holds only normal images,
5. renders (`source=splat`) appear only in `train`.

`load_images(df, size, resize)` loads images to `(B, 3, H, W)`:

- `resize="squash"` resizes the whole frame (16:9 gets squeezed),
- `resize="center_crop"` cuts the largest centred region with the target
  aspect ratio, then resizes (edges lost, nothing distorted).

It warns if the original images have different aspect ratios.

## Models

| name | class | trained over epochs |
|---|---|---|
| `patchcore` | `PatchCoreDetector` | no (memory bank, single pass) |
| `reverse_distillation` | `ReverseDistillationDetector` | yes |

Defaults follow the ECCV Marine Vision paper (see each `model.py`).

## Adding a model

1. Create `<name>/model.py` with a subclass of `AnomalibDetector`:
   - set `trainable`,
   - implement `_build_module(image_size, pre_trained)` returning the Anomalib
     model built with `post_processor=False` and
     `pre_processor=<Model>.configure_pre_processor(image_size=image_size)`,
   - extend `get_config()` with the model's own arguments.
2. Register it in `registry.py`.

Candidates already reviewed: Gaussian AD / DFM, AnomalyDINO, EfficientAD,
Dinomaly, DRAEM / GLASS. Notes:

- AnomalyDINO and Dinomaly use DINOv2: image sizes must be multiples of 14.
  Dinomaly also centre-crops inside its pre-processor.
- EfficientAD (ImageNette) and DRAEM / GLASS (DTD textures) download extra
  datasets on first use. Download them beforehand on machines without
  internet access.
