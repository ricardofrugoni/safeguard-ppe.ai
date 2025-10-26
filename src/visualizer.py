"""Visualizador de detecções de EPIs."""

import cv2
import numpy as np
from typing import Tuple
from PIL import Image

from src.config import VisualizationConfig
from src.detector import PredictionResult, DetectionResult


class Visualizer:
    """Visualizador de detecções."""

    def __init__(self, config: VisualizationConfig):
        self._config = config

    @property
    def config(self) -> VisualizationConfig:
        return self._config

    def annotate_image(
        self,
        image: np.ndarray,
        prediction: PredictionResult,
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Anota uma imagem com as detecções."""
        annotated = image.copy()

        for detection in prediction.detections:
            self._draw_detection(annotated, detection, show_labels, show_confidence)

        return annotated

    def _draw_detection(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        show_labels: bool,
        show_confidence: bool
    ) -> None:
        x1, y1, x2, y2 = detection.bbox
        color = self._config.get_color(detection.class_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, self._config.box_thickness)

        if show_labels or show_confidence:
            self._draw_label(image, detection, (x1, y1), color, show_labels, show_confidence)

    def _draw_label(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        show_name: bool,
        show_conf: bool
    ) -> None:
        label_parts = []
        if show_name:
            label_parts.append(detection.class_name)
        if show_conf:
            label_parts.append(f"{detection.confidence:.0%}")

        text = " ".join(label_parts)

        if not text:
            return

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self._config.font_scale, self._config.font_thickness
        )

        x, y = position

        cv2.rectangle(
            image,
            (x, y - self._config.label_height),
            (x + text_width + self._config.label_padding, y),
            color,
            -1
        )

        cv2.putText(
            image, text, (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, self._config.font_scale,
            (255, 255, 255), self._config.font_thickness
        )

    def create_summary_text(self, prediction: PredictionResult, confidence_threshold: float) -> str:
        """Cria texto resumo das detecções."""
        lines = [
            f"Confiança mínima: {confidence_threshold:.0%}",
            f"Detecções: {prediction.count}",
            ""
        ]

        if prediction.count > 0:
            lines.append("Por classe:")
            stats = prediction.get_class_statistics()

            for class_name, class_stats in sorted(stats.items()):
                count = int(class_stats['count'])
                avg_conf = class_stats['avg_confidence']
                lines.append(f"  {class_name}: {count}x (média: {avg_conf:.1%})")
        else:
            lines.append("Nenhuma detecção")

        return "\n".join(lines)

    def numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Converte imagem numpy para PIL."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)

    def pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Converte imagem PIL para numpy."""
        image_array = np.array(image)

        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return image_array

    def __repr__(self) -> str:
        return f"Visualizer(colors={len(self._config.colors)})"
