"""
Testes unitários para configurações.
Demonstra como OOP facilita testes.
"""

import pytest
from src.config import ModelConfig, DatasetConfig, VisualizationConfig, AppConfig


class TestModelConfig:
    """Testes para ModelConfig."""

    def test_default_values(self):
        """Testa valores padrão."""
        config = ModelConfig()

        assert config.name == "yolov8n.pt"
        assert config.epochs == 20
        assert config.batch_size == 64
        assert config.confidence_threshold == 0.4

    def test_custom_values(self):
        """Testa valores customizados."""
        config = ModelConfig(
            epochs=50,
            batch_size=32,
            confidence_threshold=0.6
        )

        assert config.epochs == 50
        assert config.batch_size == 32
        assert config.confidence_threshold == 0.6

    def test_validation_epochs(self):
        """Testa validação de epochs."""
        with pytest.raises(ValueError, match="Epochs deve ser maior que 0"):
            ModelConfig(epochs=0)

    def test_validation_confidence(self):
        """Testa validação de confidence."""
        with pytest.raises(ValueError, match="Confidence threshold"):
            ModelConfig(confidence_threshold=1.5)


class TestDatasetConfig:
    """Testes para DatasetConfig."""

    def test_paths(self):
        """Testa propriedades de caminho."""
        config = DatasetConfig(base_path="/test/path")

        assert str(config.train_images_path) == "/test/path/train/images"
        assert str(config.valid_images_path) == "/test/path/valid/images"
        assert config.data_yaml_path == "/test/path/data.yaml"

    def test_default_split(self):
        """Testa split padrão."""
        config = DatasetConfig()

        assert config.validation_split == 0.2
        assert config.min_validation_samples == 10


class TestVisualizationConfig:
    """Testes para VisualizationConfig."""

    def test_colors(self):
        """Testa cores."""
        config = VisualizationConfig()

        assert len(config.colors) == 10
        assert config.colors[0] == (255, 0, 0)  # Vermelho

    def test_get_color(self):
        """Testa obtenção de cor por classe."""
        config = VisualizationConfig()

        # Classe 0 -> cor 0
        assert config.get_color(0) == config.colors[0]

        # Classe > len(colors) deve fazer wrap
        assert config.get_color(15) == config.colors[5]


class TestAppConfig:
    """Testes para AppConfig."""

    def test_composition(self):
        """Testa composição de configs."""
        config = AppConfig()

        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.dataset, DatasetConfig)
        assert isinstance(config.visualization, VisualizationConfig)

    def test_model_paths(self):
        """Testa paths do modelo."""
        config = AppConfig(
            save_dir="/test",
            project_name="test_project"
        )

        assert config.model_save_path == "/test/test_project"
        assert config.best_model_path == "/test/test_project/weights/best.pt"

    def test_str_representation(self):
        """Testa representação em string."""
        config = AppConfig()
        str_repr = str(config)

        assert "AppConfig" in str_repr
        assert "Save Dir" in str_repr
