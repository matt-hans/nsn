"""Data models for slot timing orchestration.

Defines dataclasses for slot results, timing breakdowns, and metadata used
throughout the orchestration pipeline.
"""

from dataclasses import dataclass

import torch


@dataclass
class SlotMetadata:
    """Metadata for slot generation.

    Attributes:
        slot_id: Unique slot identifier (from recipe)
        start_time: Generation start timestamp (time.monotonic())
        end_time: Generation end timestamp (time.monotonic())
        deadline: Absolute deadline timestamp (start_time + duration_sec)
    """

    slot_id: int
    start_time: float
    end_time: float
    deadline: float


@dataclass
class GenerationBreakdown:
    """Timing breakdown of generation stages.

    Note: total_ms may not equal sum of individual stages due to parallel execution.
    For example, audio (2s) ∥ image (12s) = 12s total, not 14s.

    Attributes:
        audio_ms: Audio generation time (milliseconds)
        image_ms: Actor image generation time (milliseconds)
        video_ms: Video warping time (milliseconds)
        clip_ms: CLIP verification time (milliseconds)
        total_ms: Total end-to-end generation time (milliseconds)
    """

    audio_ms: int
    image_ms: int
    video_ms: int
    clip_ms: int
    total_ms: int


@dataclass
class SlotResult:
    """Result of a single slot generation.

    Returned by SlotScheduler.execute() after completing all generation phases.

    Attributes:
        video_frames: Video tensor [num_frames, channels, height, width]
        audio_waveform: Audio tensor [num_samples]
        clip_embedding: Combined CLIP embedding from dual ensemble [512]
        metadata: Slot metadata (id, timestamps, deadline)
        breakdown: Timing breakdown per stage
        deadline_met: Whether generation completed before deadline
    """

    video_frames: torch.Tensor
    audio_waveform: torch.Tensor
    clip_embedding: torch.Tensor
    metadata: SlotMetadata
    breakdown: GenerationBreakdown
    deadline_met: bool
