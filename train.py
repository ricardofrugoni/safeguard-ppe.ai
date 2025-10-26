#!/usr/bin/env python3
"""Script de treinamento."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.app import PPEDetectionApp


def main():
    print("Iniciando treinamento...")

    app = PPEDetectionApp()
    app.setup_dataset()
    app.train_model()
    app.validate_model()

    print("\nTreinamento concluído!")
    print(f"Modelo salvo em: {app.config.best_model_path}")


if __name__ == "__main__":
    main()
