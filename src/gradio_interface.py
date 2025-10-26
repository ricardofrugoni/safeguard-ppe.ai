"""Interface Gradio para detecção de EPIs."""

import gradio as gr
from typing import Tuple
from PIL import Image

from src.detector import PPEDetector
from src.visualizer import Visualizer


class GradioInterface:
    """Interface web usando Gradio."""

    def __init__(self, detector: PPEDetector, visualizer: Visualizer, dataset_info: str = ""):
        self._detector = detector
        self._visualizer = visualizer
        self._dataset_info = dataset_info
        self._interface = None

    def _detect_ppe(
        self, image: Image.Image, confidence: float, show_labels: bool, show_conf: bool
    ) -> Tuple[Image.Image, str]:
        if image is None:
            return None, "Por favor, carregue uma imagem"

        img_array = self._visualizer.pil_to_numpy(image)
        prediction = self._detector.predict(img_array, confidence)
        annotated = self._visualizer.annotate_image(img_array, prediction, show_labels, show_conf)
        summary = self._visualizer.create_summary_text(prediction, confidence)
        annotated_pil = self._visualizer.numpy_to_pil(annotated)

        return annotated_pil, summary

    def build_interface(self) -> gr.Blocks:
        """Constrói a interface Gradio."""
        with gr.Blocks(title="Detecção EPIs") as interface:
            gr.Markdown("# Detecção de EPIs com YOLOv8")
            gr.Markdown("Sistema de detecção de equipamentos de proteção individual")

            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Carregar Imagem")
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.4, step=0.05, label="Confiança Mínima"
                    )

                    with gr.Row():
                        labels_check = gr.Checkbox(value=True, label="Mostrar Labels")
                        conf_check = gr.Checkbox(value=True, label="Mostrar Confiança")

                    detect_btn = gr.Button("Detectar EPIs", variant="primary", size="lg")

                with gr.Column():
                    image_output = gr.Image(label="Resultado")
                    text_output = gr.Textbox(label="Informações", lines=12)

            detect_btn.click(
                fn=self._detect_ppe,
                inputs=[image_input, conf_slider, labels_check, conf_check],
                outputs=[image_output, text_output]
            )

            gr.Markdown(f"""
            ### Informações do Sistema
            {self._dataset_info}
            - Classes detectadas: {len(self._detector.class_names)}
            - Classes: {', '.join(self._detector.class_names.values())}
            """)

        self._interface = interface
        return interface

    def launch(
        self, share: bool = True, debug: bool = False,
        server_name: str = "0.0.0.0", server_port: int = 7860
    ) -> None:
        """Inicia a interface Gradio."""
        if self._interface is None:
            self.build_interface()

        print("Iniciando interface Gradio...")
        print(f"Servidor: {server_name}:{server_port}")

        self._interface.launch(
            share=share, debug=debug, server_name=server_name, server_port=server_port
        )

    def close(self) -> None:
        """Fecha a interface Gradio."""
        if self._interface is not None:
            self._interface.close()
            print("Interface Gradio fechada")

    def __repr__(self) -> str:
        status = "iniciada" if self._interface else "não iniciada"
        return f"GradioInterface(status={status})"
