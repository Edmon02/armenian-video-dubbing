#!/usr/bin/env python3
"""Phase 2 Step 7: Export Models to ONNX"""
import argparse
from pathlib import Path
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-model", default="models/asr/whisper-hy-full")
    parser.add_argument("--tts-model", default="models/tts/fish-speech-hy")
    parser.add_argument("--output-dir", default="models/onnx")
    parser.add_argument("--quantize", action="store_true", help="Export int8 quantized versions")
    args = parser.parse_args()

    logger.info("Model Export to ONNX")
    logger.info("Status: Stub (implementation in Phase 2.7)")
    logger.info("Output: {}", args.output_dir)
    logger.info("Quantize: {}", args.quantize)

if __name__ == "__main__":
    main()
