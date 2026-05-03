#!/usr/bin/env python3
"""
Gradio Web UI for Armenian Video Dubbing — Phase 3

Web interface with:
  - Video upload
  - Speaker selection (reference audio for voice cloning)
  - Emotion control
  - Language pair selection
  - Lip-sync toggle
  - Background audio toggle
  - Output download

Usage:
    python src/ui/gradio_app.py [--port 7860]
"""

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from loguru import logger

from src.pipeline import DubbingPipeline
from src.utils.logger import setup_logger


class GradioDubbingApp:
    """Gradio interface for dubbing pipeline."""

    def __init__(self, config_path: Optional[str] = None, config_override_path: Optional[str] = None):
        setup_logger()
        self.pipeline = DubbingPipeline(
            config_path=config_path,
            config_override_path=config_override_path,
        )
        self.temp_dir = Path(tempfile.gettempdir()) / "armtts_gradio"
        self.temp_dir.mkdir(exist_ok=True)

    def process_video(
        self,
        video_file,
        speaker_audio_file: Optional[str],
        emotion: str,
        src_lang: str,
        tgt_lang: str,
        dialect: str,
        skip_lipsync: bool,
        no_background: bool,
        progress=gr.Progress(),
    ) -> Tuple[str, str]:
        """Process video through dubbing pipeline."""
        try:
            if video_file is None:
                return None, "Please upload a video file."

            video_path = Path(video_file) if isinstance(video_file, str) else Path(video_file.name)
            logger.info("Processing video: {}", video_path.name)

            output_name = f"{video_path.stem}_dubbed_{dialect}_{emotion}.mp4"
            output_path = self.temp_dir / output_name

            progress(0.05, desc="Initializing pipeline...")

            result = self.pipeline.dub_video(
                video_path=str(video_path),
                reference_speaker_audio=speaker_audio_file,
                emotion=emotion,
                output_path=str(output_path),
                keep_background=not no_background,
                skip_lipsync=skip_lipsync,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                dialect=dialect,
            )

            progress(0.95, desc="Finalizing...")

            if "error" in result:
                return None, f"Error: {result['error']}"

            summary = f"""
**Dubbing Complete!**

| | |
|---|---|
| **Input** | {video_path.name} |
| **Output** | {output_name} |
| **Direction** | {src_lang} → {tgt_lang} |
| **Emotion** | {emotion} |
| **Segments** | {result.get('n_segments', '?')} |
| **Duration** | {result.get('duration_sec', 0):.1f}s |
| **Lip-sync** | {'On' if not skip_lipsync else 'Off'} |
| **Background** | {'Kept' if not no_background else 'Removed'} |

**Original**: {result.get('transcription', '')[:120]}...

**Translated**: {result.get('translated_text', '')[:120]}...
            """

            return str(output_path), summary

        except Exception as e:
            logger.error("Processing failed: {}", e)
            return None, f"Error: {str(e)}"

    def build_app(self):
        """Build Gradio interface."""
        with gr.Blocks(title="Armenian Video Dubbing AI", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
# Armenian Video Dubbing AI

Upload a video and get it dubbed into Armenian with AI-powered speech synthesis and lip-sync.

**Pipeline**: ASR (Whisper) → Translation (SeamlessM4T) → TTS (edge-tts/Fish-Speech) → Lip-sync (MuseTalk) → Audio mix
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")

                    video_input = gr.File(
                        label="Upload Video",
                        file_types=["video"],
                        type="filepath",
                    )

                    speaker_input = gr.File(
                        label="Reference Speaker Audio (Optional)",
                        file_types=["audio"],
                        type="filepath",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Settings")

                    with gr.Row():
                        src_lang = gr.Dropdown(
                            choices=["eng", "rus", "fra", "deu", "spa", "tur", "hye"],
                            value="eng",
                            label="Source Language",
                        )
                        tgt_lang = gr.Dropdown(
                            choices=["hye", "eng", "rus"],
                            value="hye",
                            label="Target Language",
                        )

                    dialect = gr.Radio(
                        choices=["eastern", "western"],
                        value="eastern",
                        label="Armenian Dialect",
                    )

                    emotion = gr.Radio(
                        choices=["neutral", "happy", "sad", "angry", "excited", "calm"],
                        value="neutral",
                        label="Emotion Style",
                    )

                    with gr.Row():
                        skip_lipsync = gr.Checkbox(
                            value=False,
                            label="Skip Lip-Sync",
                        )
                        no_background = gr.Checkbox(
                            value=False,
                            label="Remove Background Audio",
                        )

                    submit_btn = gr.Button("Start Dubbing", variant="primary", scale=2)

            gr.Markdown("### Output")

            with gr.Row():
                with gr.Column():
                    video_output = gr.File(label="Dubbed Video", type="filepath")
                with gr.Column():
                    summary_output = gr.Markdown(label="Summary")

            submit_btn.click(
                fn=self.process_video,
                inputs=[
                    video_input,
                    speaker_input,
                    emotion,
                    src_lang,
                    tgt_lang,
                    dialect,
                    skip_lipsync,
                    no_background,
                ],
                outputs=[video_output, summary_output],
            )

            gr.Markdown("""
---
- Supported: MP4, WebM, MOV (video); WAV, MP3 (audio reference)
- All processing is local — no data leaves your machine
- Voice cloning requires speaker consent

Powered by Whisper, SeamlessM4T, Fish-Speech/edge-tts, and MuseTalk
            """)

        return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gradio UI for Video Dubbing")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config-override", type=str, default=None)

    args = parser.parse_args()

    app_builder = GradioDubbingApp(
        config_path=args.config,
        config_override_path=args.config_override,
    )
    demo = app_builder.build_app()

    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_api=False,
    )


if __name__ == "__main__":
    main()
