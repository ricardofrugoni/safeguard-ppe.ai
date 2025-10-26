#!/usr/bin/env python3
"""Sistema de Detecção de EPIs com YOLOv8."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.app import PPEDetectionApp
from src.config import AppConfig


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de EPIs com YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s                           # Pipeline completo
  %(prog)s --skip-training           # Carregar modelo existente
  %(prog)s --no-ui                   # Treinar sem interface
  %(prog)s --validate-only           # Apenas validar
        """
    )

    parser.add_argument('--skip-training', action='store_true',
                        help='Pula treinamento e carrega modelo existente')
    parser.add_argument('--no-ui', action='store_true',
                        help='Não inicia interface web')
    parser.add_argument('--validate-only', action='store_true',
                        help='Apenas valida modelo existente')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Caminho customizado para o modelo')
    parser.add_argument('--config', type=str, default=None,
                        help='Caminho para arquivo de configuração customizado')
    parser.add_argument('--no-share', action='store_true',
                        help='Não cria link público do Gradio')
    parser.add_argument('--debug', action='store_true',
                        help='Ativa modo debug')

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = AppConfig()
    app = PPEDetectionApp(config)

    print(app.get_system_info())

    try:
        if args.validate_only:
            app.load_trained_model(args.model_path)
            app.validate_model()
            app.test_inference()
        else:
            app.run_full_pipeline(
                skip_training=args.skip_training,
                launch_ui=not args.no_ui
            )

    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")
        sys.exit(0)

    except Exception as e:
        print(f"\n\nErro: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
