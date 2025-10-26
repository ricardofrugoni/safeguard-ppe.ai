"""
Testes unitários para o detector.
"""

import pytest
import numpy as np
from src.detector import DetectionResult, PredictionResult
from src.config import ModelConfig


class TestDetectionResult:
    """Testes para DetectionResult."""

    def test_creation(self):
        """Testa criação de detecção."""
        detection = DetectionResult(
            class_id=0,
            class_name="helmet",
            confidence=0.95,
            bbox=(10, 20, 100, 200)
        )

        assert detection.class_id == 0
        assert detection.class_name == "helmet"
        assert detection.confidence == 0.95
        assert detection.bbox == (10, 20, 100, 200)

    def test_repr(self):
        """Testa representação em string."""
        detection = DetectionResult(
            class_id=0,
            class_name="helmet",
            confidence=0.856,
            bbox=(10, 20, 100, 200)
        )

        assert "helmet" in repr(detection)
        assert "85.6%" in repr(detection) or "86%" in repr(detection)


class TestPredictionResult:
    """Testes para PredictionResult."""

    def setup_method(self):
        """Setup para cada teste."""
        self.detections = [
            DetectionResult(0, "helmet", 0.9, (10, 20, 100, 200)),
            DetectionResult(0, "helmet", 0.85, (150, 50, 250, 300)),
            DetectionResult(1, "head", 0.75, (300, 100, 400, 350)),
        ]

        self.result = PredictionResult(
            detections=self.detections,
            image_shape=(640, 480, 3),
            inference_time=15.5
        )

    def test_count(self):
        """Testa contagem de detecções."""
        assert self.result.count == 3

    def test_get_detections_by_class(self):
        """Testa agrupamento por classe."""
        grouped = self.result.get_detections_by_class()

        assert "helmet" in grouped
        assert "head" in grouped
        assert len(grouped["helmet"]) == 2
        assert len(grouped["head"]) == 1

    def test_get_class_statistics(self):
        """Testa estatísticas por classe."""
        stats = self.result.get_class_statistics()

        # Helmet
        assert stats["helmet"]["count"] == 2
        assert stats["helmet"]["avg_confidence"] == pytest.approx(0.875, 0.01)
        assert stats["helmet"]["max_confidence"] == 0.9

        # Head
        assert stats["head"]["count"] == 1
        assert stats["head"]["avg_confidence"] == 0.75

    def test_empty_result(self):
        """Testa resultado vazio."""
        empty_result = PredictionResult(
            detections=[],
            image_shape=(640, 480, 3),
            inference_time=10.0
        )

        assert empty_result.count == 0
        assert empty_result.get_detections_by_class() == {}
        assert empty_result.get_class_statistics() == {}


class TestModelConfig:
    """Testes para configuração do modelo."""

    def test_default_config(self):
        """Testa configuração padrão."""
        config = ModelConfig()

        assert config.name == "yolov8n.pt"
        assert config.epochs == 20
        assert config.batch_size == 64

    def test_custom_config(self):
        """Testa configuração customizada."""
        config = ModelConfig(
            name="yolov8s.pt",
            epochs=50,
            batch_size=32
        )

        assert config.name == "yolov8s.pt"
        assert config.epochs == 50
        assert config.batch_size == 32


# Nota: Testes do PPEDetector completo requerem modelo YOLO
# e são melhor executados como testes de integração
