"""Runtime wiring — the streaming pipeline the sensor team and app consume (§1)."""

from .pipeline import PipelineOutput, SleepPipeline

__all__ = ["SleepPipeline", "PipelineOutput"]
