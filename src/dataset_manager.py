"""Gerenciador de datasets."""

import shutil
from pathlib import Path
from typing import List, Tuple
from roboflow import Roboflow

from src.config import DatasetConfig


class DatasetManager:
    """Gerencia operações de dataset."""

    def __init__(self, config: DatasetConfig):
        self._config = config
        self._roboflow_client = None

    @property
    def config(self) -> DatasetConfig:
        return self._config

    def download_dataset(self) -> None:
        """Baixa o dataset do Roboflow."""
        print("Iniciando download do dataset...")

        rf = Roboflow(api_key=self._config.api_key)
        project = rf.workspace(self._config.workspace).project(self._config.project_name)
        dataset = project.version(self._config.version).download(
            "yolov8",
            location=self._config.base_path
        )

        print(f"Dataset baixado em: {self._config.base_path}")
        self._roboflow_client = rf

    def get_dataset_stats(self) -> Tuple[int, int]:
        """Retorna (num_treino, num_validacao)."""
        train_count = len(list(self._config.train_images_path.glob('*.jpg')))
        valid_count = len(list(self._config.valid_images_path.glob('*.jpg')))
        return train_count, valid_count

    def validate_structure(self) -> bool:
        """Valida se a estrutura de diretórios está correta."""
        required_paths = [
            self._config.train_images_path,
            self._config.train_labels_path,
            self._config.valid_images_path,
            self._config.valid_labels_path,
        ]

        for path in required_paths:
            if not path.exists():
                return False

        return True

    def create_validation_split(self) -> None:
        """Cria split de validação se não existir."""
        train_count, valid_count = self.get_dataset_stats()

        print(f"Estrutura atual:")
        print(f"  Treino: {train_count} imagens")
        print(f"  Validação: {valid_count} imagens")

        if valid_count == 0:
            self._perform_split()
        else:
            print("Split de validação já existe")

    def _perform_split(self) -> None:
        print("\nCriando split de validação...")

        self._config.valid_images_path.mkdir(parents=True, exist_ok=True)
        self._config.valid_labels_path.mkdir(parents=True, exist_ok=True)

        train_images = list(self._config.train_images_path.glob('*.jpg'))

        n_valid = max(
            self._config.min_validation_samples,
            int(len(train_images) * self._config.validation_split)
        )

        moved_count = self._move_files(train_images[:n_valid])

        print(f"Movidas {moved_count} imagens para validação")

        train_final, valid_final = self.get_dataset_stats()
        print(f"\nEstatísticas finais:")
        print(f"  Treino: {train_final} imagens")
        print(f"  Validação: {valid_final} imagens")

    def _move_files(self, image_paths: List[Path]) -> int:
        moved_count = 0

        for img_path in image_paths:
            dest_img = self._config.valid_images_path / img_path.name
            shutil.move(str(img_path), str(dest_img))

            label_path = self._config.train_labels_path / f"{img_path.stem}.txt"
            if label_path.exists():
                dest_label = self._config.valid_labels_path / label_path.name
                shutil.move(str(label_path), str(dest_label))

            moved_count += 1

        return moved_count

    def prepare_dataset(self) -> None:
        """Prepara o dataset completo (download + split)."""
        if not self._config.train_images_path.exists():
            self.download_dataset()
        else:
            print("Dataset já existe, pulando download")

        self.create_validation_split()

        if not self.validate_structure():
            raise RuntimeError("Estrutura do dataset inválida após preparação")

        print("\nDataset preparado com sucesso!")

    def __repr__(self) -> str:
        train_count, valid_count = self.get_dataset_stats()
        return f"DatasetManager(train={train_count}, valid={valid_count})"
