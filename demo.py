#!/usr/bin/env python3
"""Script de demonstração."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.app import PPEDetectionApp


def main():
    print("Iniciando demonstração...")

    app = PPEDetectionApp()
    app.load_trained_model()
    app.launch_interface(share=True, debug=False)


if __name__ == "__main__":
    main()
