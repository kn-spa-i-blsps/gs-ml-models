"""
Model registry: name -> detector class.

Experiments pick models by name, so adding a model means adding one line here.
"""

from gs_ml_models.anomaly_detection.base import AnomalibDetector
from gs_ml_models.anomaly_detection.patchcore.model import PatchCoreDetector
from gs_ml_models.anomaly_detection.reverse_distillation.model import ReverseDistillationDetector

MODELS: dict[str, type[AnomalibDetector]] = {
    "patchcore": PatchCoreDetector,
    "reverse_distillation": ReverseDistillationDetector,
}


def get_model(name: str, **kwargs) -> AnomalibDetector:
    """Create a detector by name, e.g. ``get_model("patchcore", seed=1)``."""
    try:
        cls = MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(MODELS)}") from None
    return cls(**kwargs)
