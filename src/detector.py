"""Detector de EPIs usando YOLOv8."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass

from src.config import ModelConfig


@dataclass
class DetectionResult:
    """Resultado de uma detecção."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple

    def __repr__(self) -> str:
        return f"{self.class_name} ({self.confidence:.1%})"


@dataclass
class PredictionResult:
    """Resultado completo de uma predição."""
    detections: List[DetectionResult]
    image_shape: tuple
    inference_time: float

    @property
    def count(self) -> int:
        return len(self.detections)

    def get_detections_by_class(self) -> Dict[str, List[DetectionResult]]:
        """Agrupa detecções por classe."""
        grouped = {}
        for detection in self.detections:
            if detection.class_name not in grouped:
                grouped[detection.class_name] = []
            grouped[detection.class_name].append(detection)
        return grouped

    def get_class_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calcula estatísticas por classe."""
        grouped = self.get_detections_by_class()
        stats = {}

        for class_name, detections in grouped.items():
            confidences = [d.confidence for d in detections]
            stats[class_name] = {
                'count': len(detections),
                'avg_confidence': np.mean(confidences),
                'max_confidence': max(confidences),
                'min_confidence': min(confidences),
            }

        return stats


class PPEDetector:
    """Detector de EPIs usando YOLOv8."""

    def __init__(self, config: ModelConfig):
        self._config = config
        self._model: Optional[YOLO] = None
        self._class_names: Dict[int, str] = {}

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def model(self) -> YOLO:
        if self._model is None:
            raise RuntimeError("Modelo não carregado. Use load_model() ou train() primeiro.")
        return self._model

    @property
    def class_names(self) -> Dict[int, str]:
        if self._model is not None:
            return self._model.names
        return self._class_names

    def load_pretrained(self) -> None:
        """Carrega modelo pré-treinado do YOLOv8."""
        print(f"Carregando modelo pré-treinado: {self._config.name}")
        self._model = YOLO(self._config.name)
        print("Modelo carregado com sucesso")

    def load_model(self, model_path: str) -> None:
        """Carrega modelo treinado de um arquivo."""
        print(f"Carregando modelo de: {model_path}")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        self._model = YOLO(model_path)
        self._class_names = self._model.names
        print(f"Modelo carregado com {len(self._class_names)} classes")

    def train(self, data_yaml_path: str, project_dir: str, project_name: str) -> None:
        """Treina o modelo."""
        print("Iniciando treinamento...")

        if self._model is None:
            self.load_pretrained()

        results = self._model.train(
            data=data_yaml_path,
            epochs=self._config.epochs,
            imgsz=self._config.image_size,
            batch=self._config.batch_size,
            device=self._config.device,
            workers=self._config.workers,
            cache=self._config.cache,
            project=project_dir,
            name=project_name,
            patience=self._config.patience,
            save=True,
            exist_ok=True,
            verbose=False,
            plots=False
        )

        print("Treinamento concluído!")
        self._class_names = self._model.names

    def predict(self, image: np.ndarray, confidence: Optional[float] = None) -> PredictionResult:
        """Realiza predição em uma imagem."""
        conf_threshold = confidence or self._config.confidence_threshold

        results = self.model.predict(image, conf=conf_threshold, verbose=False)[0]

        detections = self._process_results(results)

        return PredictionResult(
            detections=detections,
            image_shape=image.shape,
            inference_time=results.speed['inference'] if hasattr(results, 'speed') else 0.0
        )

    def _process_results(self, results) -> List[DetectionResult]:
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.class_names[class_id]

            detection = DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2)
            )

            detections.append(detection)

        return detections

    def validate(self, data_yaml_path: str) -> Dict[str, float]:
        """Valida o modelo no dataset de validação."""
        print("Validando modelo...")

        metrics = self.model.val(data=data_yaml_path)

        results = {
            'map50': float(metrics.box.map50),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'map50_95': float(metrics.box.map),
        }

        print(f"mAP50: {results['map50']:.1%}")
        print(f"Precision: {results['precision']:.1%}")
        print(f"Recall: {results['recall']:.1%}")

        return results

    def test_inference(self, image_path: str) -> None:
        """Testa inferência em uma imagem."""
        print(f"\nTestando inferência em: {image_path}")

        import cv2
        image = cv2.imread(image_path)
        result = self.predict(image)

        print(f"Detecções: {result.count}")

        for detection in result.detections[:5]:
            print(f"  - {detection}")

    def __repr__(self) -> str:
        status = "carregado" if self._model else "não carregado"
        return f"PPEDetector(model={status}, classes={len(self.class_names)})"
