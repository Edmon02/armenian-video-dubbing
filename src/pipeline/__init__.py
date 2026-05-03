"""
End-to-End Dubbing Orchestrator package.

Re-exports from the pipeline module for backward-compatible imports like:
    from src.pipeline import DubbingPipeline
"""

# The real implementation lives in src/pipeline/pipeline.py (was src/pipeline.py).
from src.pipeline.pipeline import DubbingPipeline, main

__all__ = ["DubbingPipeline", "main"]
