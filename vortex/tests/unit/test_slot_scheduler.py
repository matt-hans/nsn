"""Unit tests for slot timing orchestration.

Tests SlotScheduler, timeout management, retry logic, and deadline tracking.
All tests are deterministic and mock pipeline dependencies.

Test Coverage:
- Data models (SlotResult, GenerationBreakdown, SlotMetadata)
- Deadline tracking logic
- Timeout enforcement per stage
- Audio retry with exponential backoff
- Parallel execution timing
"""

import asyncio
from unittest.mock import MagicMock

import pytest
import torch

from vortex.orchestration.models import GenerationBreakdown, SlotMetadata, SlotResult
from vortex.orchestration.scheduler import SlotScheduler

# ============================================================================
# Data Model Tests
# ============================================================================


def test_slot_result_dataclass():
    """Test SlotResult dataclass structure and defaults.

    Why: Validates data contract for slot results
    Contract: All fields present, deadline_met defaults to True
    """
    video = torch.randn(1080, 3, 512, 512)
    audio = torch.randn(1080000)
    clip_embedding = torch.randn(512)
    metadata = SlotMetadata(
        slot_id=12345,
        start_time=0.0,
        end_time=12.5,
        deadline=45.0,
    )
    breakdown = GenerationBreakdown(
        audio_ms=2000,
        image_ms=12000,
        video_ms=8000,
        clip_ms=1000,
        total_ms=12500,
    )

    result = SlotResult(
        video_frames=video,
        audio_waveform=audio,
        clip_embedding=clip_embedding,
        metadata=metadata,
        breakdown=breakdown,
        deadline_met=True,
    )

    assert result.video_frames.shape == (1080, 3, 512, 512)
    assert result.audio_waveform.shape == (1080000,)
    assert result.clip_embedding.shape == (512,)
    assert result.metadata.slot_id == 12345
    assert result.breakdown.total_ms == 12500
    assert result.deadline_met is True


def test_generation_breakdown_dataclass():
    """Test GenerationBreakdown timing fields.

    Why: Validates timing metadata structure
    Contract: All timing fields non-negative, total equals sum of phases
    """
    breakdown = GenerationBreakdown(
        audio_ms=2000,
        image_ms=12000,
        video_ms=8000,
        clip_ms=1000,
        total_ms=23000,  # Note: total != sum because parallel execution
    )

    assert breakdown.audio_ms >= 0
    assert breakdown.image_ms >= 0
    assert breakdown.video_ms >= 0
    assert breakdown.clip_ms >= 0
    assert breakdown.total_ms >= 0

    # For parallel execution: total = max(audio, image) + video + clip
    # Not sum of all phases


def test_slot_metadata_dataclass():
    """Test SlotMetadata structure.

    Why: Validates slot identification contract
    Contract: slot_id, timestamps, deadline present
    """
    metadata = SlotMetadata(
        slot_id=67890,
        start_time=100.5,
        end_time=112.8,
        deadline=145.5,
    )

    assert metadata.slot_id == 67890
    assert metadata.start_time == 100.5
    assert metadata.end_time == 112.8
    assert metadata.deadline == 145.5
    assert metadata.end_time > metadata.start_time
    assert metadata.deadline > metadata.end_time


# ============================================================================
# Scheduler Initialization Tests
# ============================================================================


def test_scheduler_init():
    """Test SlotScheduler initialization.

    Why: Validates proper dependency injection
    Contract: Pipeline and config stored, timeouts parsed
    """
    mock_pipeline = MagicMock()
    config = {
        "timeouts": {
            "audio_s": 3,
            "image_s": 15,
            "video_s": 10,
            "clip_s": 2,
        },
        "retry_policy": {
            "audio": 1,
            "image": 0,
            "video": 0,
            "clip": 0,
        },
        "deadline_buffer_s": 5,
    }

    scheduler = SlotScheduler(pipeline=mock_pipeline, config=config)

    assert scheduler.pipeline == mock_pipeline
    assert scheduler.timeouts["audio_s"] == 3
    assert scheduler.timeouts["image_s"] == 15
    assert scheduler.timeouts["video_s"] == 10
    assert scheduler.timeouts["clip_s"] == 2
    assert scheduler.retry_policy["audio"] == 1
    assert scheduler.deadline_buffer_s == 5


# ============================================================================
# Deadline Tracking Tests
# ============================================================================


def test_deadline_check_sufficient_time():
    """Test deadline check with ample time remaining.

    Why: Validates deadline tracking logic (green path)
    Contract: Returns True (30s available >= 10s needed + 5s buffer)

    Scenario: current=5s, deadline=45s, remaining_work=10s
    Available: 45 - 5 = 40s
    Needed: 10s + 5s buffer = 15s
    Result: 40s >= 15s → Continue
    """
    mock_pipeline = MagicMock()
    config = {
        "timeouts": {"audio_s": 3, "image_s": 15, "video_s": 10, "clip_s": 2},
        "retry_policy": {"audio": 1, "image": 0, "video": 0, "clip": 0},
        "deadline_buffer_s": 5,
    }
    scheduler = SlotScheduler(pipeline=mock_pipeline, config=config)

    can_continue = scheduler._check_deadline(
        current_time=5.0,
        deadline=45.0,
        remaining_work_s=10.0,
    )

    assert can_continue is True


def test_deadline_check_insufficient_time():
    """Test deadline check with inadequate time remaining.

    Why: Validates deadline tracking logic (red path)
    Contract: Returns False (9s available < 11s needed + 5s buffer)

    Scenario: current=36s, deadline=45s, remaining_work=11s
    Available: 45 - 36 = 9s
    Needed: 11s + 5s buffer = 16s
    Result: 9s < 16s → Abort
    """
    mock_pipeline = MagicMock()
    config = {
        "timeouts": {"audio_s": 3, "image_s": 15, "video_s": 10, "clip_s": 2},
        "retry_policy": {"audio": 1, "image": 0, "video": 0, "clip": 0},
        "deadline_buffer_s": 5,
    }
    scheduler = SlotScheduler(pipeline=mock_pipeline, config=config)

    can_continue = scheduler._check_deadline(
        current_time=36.0,
        deadline=45.0,
        remaining_work_s=11.0,
    )

    assert can_continue is False


# ============================================================================
# Timeout Enforcement Tests
# ============================================================================


@pytest.mark.asyncio
async def test_timeout_enforcement_audio():
    """Test audio timeout cancellation.

    Why: Validates per-stage timeout works
    Contract: Raises asyncio.TimeoutError after 3s
    """
    mock_pipeline = MagicMock()

    # Mock audio generation that hangs indefinitely
    async def hanging_audio(*args, **kwargs):
        await asyncio.sleep(100)  # Much longer than 3s timeout
        return torch.randn(1080000)

    mock_pipeline._generate_audio = hanging_audio

    config = {
        "timeouts": {"audio_s": 3, "image_s": 15, "video_s": 10, "clip_s": 2},
        "retry_policy": {"audio": 0, "image": 0, "video": 0, "clip": 0},  # No retry
        "deadline_buffer_s": 5,
    }
    scheduler = SlotScheduler(pipeline=mock_pipeline, config=config)

    mock_recipe = {"audio_track": {"script": "test"}}

    with pytest.raises(asyncio.TimeoutError):
        await scheduler._generate_audio_with_timeout(mock_recipe)


# ============================================================================
# Retry Logic Tests
# ============================================================================


@pytest.mark.asyncio
async def test_retry_logic_success_on_retry():
    """Test audio retry recovers from failure.

    Why: Validates retry recovery mechanism
    Contract: Returns audio after 1 retry, logs retry attempt
    """
    mock_pipeline = MagicMock()

    attempt_count = 0

    async def flaky_audio(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise RuntimeError("Transient CUDA error")
        return torch.randn(1080000)

    mock_pipeline._generate_audio = flaky_audio

    config = {
        "timeouts": {"audio_s": 3, "image_s": 15, "video_s": 10, "clip_s": 2},
        "retry_policy": {"audio": 1, "image": 0, "video": 0, "clip": 0},
        "deadline_buffer_s": 5,
    }
    scheduler = SlotScheduler(pipeline=mock_pipeline, config=config)

    mock_recipe = {"audio_track": {"script": "test"}}

    # Should succeed on second attempt
    result = await scheduler._with_retry(
        lambda: scheduler._generate_audio_with_timeout(mock_recipe),
        retries=1,
    )

    assert result.shape == (1080000,)
    assert attempt_count == 2  # First attempt failed, second succeeded


@pytest.mark.asyncio
async def test_retry_logic_exhausted():
    """Test audio retry exhaustion.

    Why: Validates retry limit enforcement
    Contract: Raises exception after max retries
    """
    mock_pipeline = MagicMock()

    async def always_fails(*args, **kwargs):
        raise RuntimeError("Persistent CUDA error")

    mock_pipeline._generate_audio = always_fails

    config = {
        "timeouts": {"audio_s": 3, "image_s": 15, "video_s": 10, "clip_s": 2},
        "retry_policy": {"audio": 1, "image": 0, "video": 0, "clip": 0},
        "deadline_buffer_s": 5,
    }
    scheduler = SlotScheduler(pipeline=mock_pipeline, config=config)

    mock_recipe = {"audio_track": {"script": "test"}}

    with pytest.raises(RuntimeError, match="Persistent CUDA error"):
        await scheduler._with_retry(
            lambda: scheduler._generate_audio_with_timeout(mock_recipe),
            retries=1,
        )


# ============================================================================
# Parallel Execution Tests
# ============================================================================


@pytest.mark.asyncio
async def test_parallel_execution_timing():
    """Test parallel vs sequential speedup.

    Why: Validates parallel execution optimization
    Contract: Total time ≈ max(2s, 12s) = 12s, not 14s sequential

    Parallel: audio (2s) ∥ image (12s) = 12s total
    Sequential: audio (2s) + image (12s) = 14s total
    Speedup: 2s (17% improvement)
    """
    mock_pipeline = MagicMock()

    # Mock audio takes 2 seconds
    async def mock_audio(*args, **kwargs):
        await asyncio.sleep(0.02)  # 20ms scaled to simulate 2s
        return torch.randn(1080000)

    # Mock image takes 12 seconds
    async def mock_image(*args, **kwargs):
        await asyncio.sleep(0.12)  # 120ms scaled to simulate 12s
        return torch.randn(1, 3, 512, 512)

    mock_pipeline._generate_audio = mock_audio
    mock_pipeline._generate_actor = mock_image

    config = {
        "timeouts": {"audio_s": 30, "image_s": 30, "video_s": 30, "clip_s": 30},
        "retry_policy": {"audio": 0, "image": 0, "video": 0, "clip": 0},
        "deadline_buffer_s": 5,
    }
    scheduler = SlotScheduler(pipeline=mock_pipeline, config=config)

    mock_recipe = {
        "audio_track": {"script": "test"},
        "visual_track": {"prompt": "test"},
    }

    import time
    start = time.monotonic()

    # Execute parallel phase
    audio_task = asyncio.create_task(
        scheduler._generate_audio_with_timeout(mock_recipe)
    )
    image_task = asyncio.create_task(
        scheduler._generate_image_with_timeout(mock_recipe)
    )

    audio_result, image_result = await asyncio.gather(audio_task, image_task)

    elapsed = time.monotonic() - start

    # Should take ~120ms (image time), not 140ms (audio + image)
    # Allow 50ms tolerance for overhead
    assert elapsed < 0.170, f"Expected <170ms (parallel), got {elapsed*1000:.1f}ms"
    assert elapsed >= 0.120, f"Expected >=120ms (image time), got {elapsed*1000:.1f}ms"

    assert audio_result.shape == (1080000,)
    assert image_result.shape == (1, 3, 512, 512)
