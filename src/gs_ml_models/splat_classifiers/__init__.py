from .dinov2_mlp import DinoV2DefectClassifier
from .simple_mlp.model import SimpleMLPClassifier
from .xgboost_baseline import XGBoostDefectClassifier

__all__ = [
    "SimpleMLPClassifier",
    "XGBoostDefectClassifier",
    "DinoV2DefectClassifier",
]