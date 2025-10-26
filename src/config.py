"""Configurações do sistema de detecção de EPIs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class ModelConfig:
    """Configurações do modelo YOLO."""

    name: str = "yolov8n.pt"
    epochs: int = 20
    image_size: int = 640
    batch_size: int = 64
    device: int = 0
    workers: int = 8
    cache: bool = True
    patience: int = 8
    confidence_threshold: float = 0.4

    def __post_init__(self):
        """Validação após inicialização."""
        if self.epochs < 1:
            raise ValueError("Epochs deve ser maior que 0")
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("Confidence threshold deve estar entre 0 e 1")


@dataclass
class DatasetConfig:
    """Configurações do dataset."""

    base_path: str = "/content/datasets/ppe"
    augmented_path: str = "/content/datasets/ppe_augmented"
    validation_split: float = 0.2
    min_validation_samples: int = 10

    api_key: str = "tSGbJRy40ipX4Xo7rYhw"
    workspace: str = "joseph-nelson"
    project_name: str = "hard-hat-workers"
    version: int = 2

    @property
    def train_images_path(self) -> Path:
        """Retorna o caminho das imagens de treino."""
        return Path(self.base_path) / "train" / "images"

    @property
    def valid_images_path(self) -> Path:
        """Retorna o caminho das imagens de validação."""
        return Path(self.base_path) / "valid" / "images"

    @property
    def train_labels_path(self) -> Path:
        """Retorna o caminho dos labels de treino."""
        return Path(self.base_path) / "train" / "labels"

    @property
    def valid_labels_path(self) -> Path:
        """Retorna o caminho dos labels de validação."""
        return Path(self.base_path) / "valid" / "labels"

    @property
    def data_yaml_path(self) -> str:
        """Retorna o caminho do arquivo data.yaml."""
        return f"{self.base_path}/data.yaml"


@dataclass
class VisualizationConfig:
    """Configurações de visualização."""

    colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (255, 0, 0), (0, 255, 0), (255, 165, 0), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (0, 128, 0), (0, 0, 255), (128, 128, 128),
    ])

    box_thickness: int = 3
    font_scale: float = 0.6
    font_thickness: int = 2
    label_padding: int = 10
    label_height: int = 25

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Retorna a cor para uma classe específica."""
        return self.colors[class_id % len(self.colors)]


@dataclass
class AppConfig:
    """Configuração principal da aplicação."""

    save_dir: str = "/content/drive/MyDrive/PPE_Detection"
    project_name: str = "ppe_model"

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @property
    def model_save_path(self) -> str:
        """Retorna o caminho onde o modelo será salvo."""
        return f"{self.save_dir}/{self.project_name}"

    @property
    def best_model_path(self) -> str:
        """Retorna o caminho do melhor modelo treinado."""
        return f"{self.model_save_path}/weights/best.pt"

    def __str__(self) -> str:
        """Representação em string da configuração."""
        return f"""
AppConfig:
  Save Dir: {self.save_dir}
  Project: {self.project_name}
  Model: {self.model.name}
  Dataset: {self.dataset.base_path}
"""
