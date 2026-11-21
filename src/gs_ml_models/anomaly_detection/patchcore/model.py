"""
PatchCore (Roth et al., CVPR 2022): memory-bank anomaly detector.

Cuts every normal training image into patches, describes each patch with a
pretrained network and stores a representative subset (coreset) in a memory
bank. A test patch far from every stored patch is anomalous. No network is
trained; the bank is built in a single pass.

Defaults follow the ECCV Marine Vision paper: wide_resnet50_2, layers 2-3,
1% coreset, 9 neighbours.
"""

from collections.abc import Sequence

from gs_ml_models.anomaly_detection.base import AnomalibDetector


class PatchCoreDetector(AnomalibDetector):
    """PatchCore exposing the anomaly-detection API (see README.md).

    Args:
        backbone: Pretrained feature extractor.
        layers: Backbone layers whose features are stored.
        coreset_sampling_ratio: Fraction of patch features kept in the bank.
        num_neighbors: Neighbours used when scoring.
        **kwargs: Common settings, see ``AnomalibDetector``.
    """

    trainable = False

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.01,
        num_neighbors: int = 9,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone = backbone
        self.layers = tuple(layers)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "backbone": self.backbone,
            "layers": list(self.layers),
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
        }

    def _build_module(self, image_size, pre_trained=True):
        from anomalib.models import Patchcore

        return Patchcore(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=pre_trained,
            coreset_sampling_ratio=self.coreset_sampling_ratio,
            num_neighbors=self.num_neighbors,
            pre_processor=Patchcore.configure_pre_processor(image_size=image_size),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
