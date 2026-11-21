"""
Reverse Distillation (Deng & Li, CVPR 2022): teacher-student anomaly detector.

A frozen pretrained encoder (teacher) describes the image; a trained decoder
(student) learns to reproduce the teacher's features from a compressed
bottleneck, on normal images only. Where the student fails to reproduce them,
the image is anomalous. The student is trained over epochs, with early stopping
on normal validation images (see ``AnomalibDetector``).

Defaults follow the ECCV Marine Vision paper / Deng & Li recipe:
wide_resnet50_2, layers 1-3, batch size 16. Optimiser (Adam, lr 0.005,
betas 0.5/0.999) is Anomalib's default for this model.
"""

from collections.abc import Sequence

from gs_ml_models.anomaly_detection.base import AnomalibDetector


class ReverseDistillationDetector(AnomalibDetector):
    """Reverse Distillation exposing the anomaly-detection API (see README.md).

    Args:
        backbone: Pretrained teacher encoder.
        layers: Encoder layers distilled by the student.
        batch_size: Training batch size (16 as in the reference recipe).
        **kwargs: Common settings, see ``AnomalibDetector``.
    """

    trainable = True

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: Sequence[str] = ("layer1", "layer2", "layer3"),
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(batch_size=batch_size, **kwargs)
        self.backbone = backbone
        self.layers = tuple(layers)

    def get_config(self) -> dict:
        return {
            **super().get_config(),
            "backbone": self.backbone,
            "layers": list(self.layers),
        }

    def _build_module(self, image_size, pre_trained=True):
        from anomalib.models import ReverseDistillation

        return ReverseDistillation(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=pre_trained,
            # The model reads its input size from this resize.
            pre_processor=ReverseDistillation.configure_pre_processor(image_size=image_size),
            post_processor=False,
            evaluator=False,
            visualizer=False,
        )
