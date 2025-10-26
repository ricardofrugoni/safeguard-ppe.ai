"""Aplicação principal para detecção de EPIs."""

from pathlib import Path
from typing import Optional

from src.config import AppConfig
from src.dataset_manager import DatasetManager
from src.detector import PPEDetector
from src.visualizer import Visualizer
from src.gradio_interface import GradioInterface


class PPEDetectionApp:
    """Aplicação principal de detecção de EPIs."""

    def __init__(self, config: Optional[AppConfig] = None):
        self._config = config or AppConfig()
        self._dataset_manager: Optional[DatasetManager] = None
        self._detector: Optional[PPEDetector] = None
        self._visualizer: Optional[Visualizer] = None
        self._interface: Optional[GradioInterface] = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        print("Inicializando componentes...")
        self._dataset_manager = DatasetManager(self._config.dataset)
        self._detector = PPEDetector(self._config.model)
        self._visualizer = Visualizer(self._config.visualization)
        print("Componentes inicializados")

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def dataset_manager(self) -> DatasetManager:
        return self._dataset_manager

    @property
    def detector(self) -> PPEDetector:
        return self._detector

    @property
    def visualizer(self) -> Visualizer:
        return self._visualizer

    def setup_dataset(self) -> None:
        """Configura o dataset."""
        print("\n" + "=" * 60)
        print("CONFIGURANDO DATASET")
        print("=" * 60)
        self._dataset_manager.prepare_dataset()

    def train_model(self) -> None:
        """Treina o modelo."""
        print("\n" + "=" * 60)
        print("TREINANDO MODELO")
        print("=" * 60)

        self._detector.train(
            data_yaml_path=self._config.dataset.data_yaml_path,
            project_dir=self._config.save_dir,
            project_name=self._config.project_name
        )

        print(f"\nModelo salvo em: {self._config.best_model_path}")

    def load_trained_model(self, model_path: Optional[str] = None) -> None:
        """Carrega modelo treinado."""
        path = model_path or self._config.best_model_path

        print("\n" + "=" * 60)
        print("CARREGANDO MODELO")
        print("=" * 60)

        self._detector.load_model(path)

    def validate_model(self) -> dict:
        """Valida o modelo."""
        print("\n" + "=" * 60)
        print("VALIDANDO MODELO")
        print("=" * 60)

        metrics = self._detector.validate(self._config.dataset.data_yaml_path)
        return metrics

    def test_inference(self) -> None:
        """Testa inferência em uma imagem de validação."""
        print("\n" + "=" * 60)
        print("TESTANDO INFERÊNCIA")
        print("=" * 60)

        valid_images = list(self._config.dataset.valid_images_path.glob('*.jpg'))

        if not valid_images:
            print("Nenhuma imagem de validação encontrada")
            return

        test_image = str(valid_images[0])
        self._detector.test_inference(test_image)

    def launch_interface(self, share: bool = True, debug: bool = False) -> None:
        """Inicia a interface web."""
        print("\n" + "=" * 60)
        print("INICIANDO INTERFACE WEB")
        print("=" * 60)

        train_count, valid_count = self._dataset_manager.get_dataset_stats()
        dataset_info = f"""
- Dataset treino: {train_count} imagens
- Dataset validação: {valid_count} imagens
- Modelo: {self._config.model.name}
"""

        self._interface = GradioInterface(
            detector=self._detector,
            visualizer=self._visualizer,
            dataset_info=dataset_info
        )

        self._interface.launch(share=share, debug=debug)

    def run_full_pipeline(self, skip_training: bool = False, launch_ui: bool = True) -> None:
        """Executa pipeline completo."""
        print("\n" + "=" * 60)
        print("INICIANDO PIPELINE COMPLETO")
        print("=" * 60)

        self.setup_dataset()

        if skip_training:
            print("\nPulando treinamento, carregando modelo existente...")
            self.load_trained_model()
        else:
            self.train_model()
            self.load_trained_model()

        self.validate_model()
        self.test_inference()

        if launch_ui:
            self.launch_interface()
        else:
            print("\nPipeline concluído! Use launch_interface() para iniciar a UI.")

    def get_system_info(self) -> str:
        """Retorna informações do sistema."""
        import torch

        info = f"""
{'=' * 60}
INFORMAÇÕES DO SISTEMA
{'=' * 60}

PyTorch: {torch.__version__}
CUDA Disponível: {torch.cuda.is_available()}
"""

        if torch.cuda.is_available():
            info += f"GPU: {torch.cuda.get_device_name(0)}\n"

        info += f"\n{self._config}"
        info += f"\nDataset Manager: {self._dataset_manager}"
        info += f"Detector: {self._detector}"
        info += f"Visualizer: {self._visualizer}"

        return info

    def __repr__(self) -> str:
        return f"PPEDetectionApp(project={self._config.project_name})"

    def __str__(self) -> str:
        return self.get_system_info()
