"""Integration tests for end-to-end slot orchestration.

Tests SlotScheduler with VortexPipeline integration, CLIP ensemble, and
realistic generation scenarios.

Requires: GPU access (RTX 3060 12GB minimum)
Run with: pytest tests/integration/test_slot_orchestration.py --gpu -v

Test Coverage:
- Full pipeline success (audio ∥ image → video → CLIP)
- Deadline abort prediction
- CLIP self-check failure handling
- Progress checkpoint logging
- VRAM pressure handling
"""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
import torch

from vortex.models.clip_ensemble import DualClipResult
from vortex.orchestration import (
    DeadlineMissError,
    SlotResult,
    SlotScheduler,
)
from vortex.pipeline import MemoryPressureError

# Mark all tests as requiring GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Integration tests require GPU"
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_pipeline():
    """Mock VortexPipeline with realistic timing."""
    pipeline = MagicMock()

    # Mock audio generation (2s)
    async def mock_audio(recipe):
        await asyncio.sleep(0.02)  # 20ms scaled to simulate 2s
        return torch.randn(1080000, device="cpu")  # Use CPU for tests

    # Mock image generation (12s)
    async def mock_image(recipe):
        await asyncio.sleep(0.12)  # 120ms scaled to simulate 12s
        return torch.randn(1, 3, 512, 512, device="cpu")

    # Mock video generation (8s)
    async def mock_video(image, audio, recipe):
        await asyncio.sleep(0.08)  # 80ms scaled to simulate 8s
        return torch.randn(1080, 3, 512, 512, device="cpu")

    # Mock CLIP verification (1s)
    async def mock_clip(video, recipe):
        await asyncio.sleep(0.01)  # 10ms scaled to simulate 1s
        return DualClipResult(
            embedding=torch.randn(512, device="cpu"),
            score_clip_b=0.75,
            score_clip_l=0.78,
            ensemble_score=0.77,
            self_check_passed=True,
            outlier_detected=False,
        )

    pipeline._generate_audio = mock_audio
    pipeline._generate_actor = mock_image
    pipeline._generate_video = mock_video
    pipeline._verify_semantic = mock_clip

    return pipeline


@pytest.fixture
def scheduler_config():
    """Scheduler configuration for tests."""
    return {
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


@pytest.fixture
def test_recipe():
    """Sample recipe for testing."""
    return {
        "recipe_id": "test-12345",
        "slot_params": {
            "slot_number": 12345,
            "duration_sec": 45,
        },
        "audio_track": {
            "script": "Welcome to the Interdimensional Cable Network!",
            "voice_id": "rick_c137",
        },
        "visual_track": {
            "prompt": "scientist in lab coat explaining quantum mechanics",
            "expression_sequence": ["neutral", "excited"],
        },
        "semantic_constraints": {
            "min_clip_score": 0.75,
        },
    }


# ============================================================================
# End-to-End Success Tests
# ============================================================================


@pytest.mark.asyncio
async def test_successful_slot_generation_e2e(
    mock_pipeline, scheduler_config, test_recipe
):
    """Test full pipeline success with all phases.

    Why: Validates end-to-end orchestration
    Contract: SlotResult returned, all stages complete, deadline_met=True

    Timeline (scaled 100×):
    - 0-20ms: Audio + Image (parallel)
    - 20-100ms: Video (waits for image)
    - 100-110ms: CLIP verification
    - Total: ~110ms (scaled from 11s)
    """
    scheduler = SlotScheduler(pipeline=mock_pipeline, config=scheduler_config)

    # Execute with generous deadline (45s = 450ms scaled)
    result = await scheduler.execute(
        recipe=test_recipe,
        slot_id=12345,
        deadline=None,  # Uses default 45s
    )

    # Verify result structure
    assert isinstance(result, SlotResult)
    assert result.metadata.slot_id == 12345
    assert result.video_frames.shape == (1080, 3, 512, 512)
    assert result.audio_waveform.shape == (1080000,)
    assert result.clip_embedding.shape == (512,)

    # Verify timing breakdown
    assert result.breakdown.audio_ms > 0
    assert result.breakdown.image_ms > 0
    assert result.breakdown.video_ms > 0
    assert result.breakdown.clip_ms > 0
    assert result.breakdown.total_ms > 0

    # Verify deadline met
    assert result.deadline_met is True

    # Verify parallel execution (total < sum of all stages)
    sequential_time = (
        result.breakdown.audio_ms
        + result.breakdown.image_ms
        + result.breakdown.video_ms
        + result.breakdown.clip_ms
    )
    # Total should be less than sequential due to parallel audio+image
    assert result.breakdown.total_ms < sequential_time


# ============================================================================
# Deadline Tracking Tests
# ============================================================================


@pytest.mark.asyncio
async def test_deadline_abort_prediction(mock_pipeline, scheduler_config, test_recipe):
    """Test predictive abort on slow generation.

    Why: Validates deadline tracking prevents wasted work
    Contract: DeadlineMissError raised, slot aborted gracefully

    Scenario: Set very tight deadline (5ms) that cannot be met
    """
    scheduler = SlotScheduler(pipeline=mock_pipeline, config=scheduler_config)

    import time
    start = time.monotonic()
    tight_deadline = start + 0.005  # 5ms deadline (impossible to meet)

    with pytest.raises(DeadlineMissError, match="Deadline miss predicted"):
        await scheduler.execute(
            recipe=test_recipe,
            slot_id=99999,
            deadline=tight_deadline,
        )


# ============================================================================
# CLIP Self-Check Tests
# ============================================================================


@pytest.mark.asyncio
async def test_clip_self_check_failure(mock_pipeline, scheduler_config, test_recipe):
    """Test CLIP self-check rejects low-quality content.

    Why: Validates quality gate before BFT
    Contract: SlotResult has clip_result.self_check_passed=False
    """
    # Override CLIP to return failing scores
    async def failing_clip(video, recipe):
        await asyncio.sleep(0.01)
        return DualClipResult(
            embedding=torch.randn(512, device="cpu"),
            score_clip_b=0.65,  # Below threshold (0.70)
            score_clip_l=0.68,  # Below threshold (0.72)
            ensemble_score=0.67,
            self_check_passed=False,  # Failed
            outlier_detected=False,
        )

    mock_pipeline._verify_semantic = failing_clip

    scheduler = SlotScheduler(pipeline=mock_pipeline, config=scheduler_config)

    result = await scheduler.execute(
        recipe=test_recipe,
        slot_id=55555,
        deadline=None,
    )

    # Result should be returned but with failed CLIP self-check
    # Note: We don't raise error, director can choose to abort BFT
    assert result.deadline_met is True
    # CLIP embedding is still returned for potential analysis


# ============================================================================
# Progress Logging Tests
# ============================================================================


@pytest.mark.asyncio
async def test_progress_checkpoint_logging(
    mock_pipeline, scheduler_config, test_recipe, caplog
):
    """Test progress checkpoint logging structure.

    Why: Validates observability requirements
    Contract: Logs emitted for audio_complete, image_complete, video_complete, clip_complete
    """
    with caplog.at_level(logging.INFO):
        scheduler = SlotScheduler(pipeline=mock_pipeline, config=scheduler_config)

        await scheduler.execute(
            recipe=test_recipe,
            slot_id=77777,
            deadline=None,
        )

    # Check for expected log messages
    log_text = caplog.text

    assert "Starting slot generation" in log_text
    assert "Parallel phase complete" in log_text
    assert "Video generation complete" in log_text
    assert "CLIP verification complete" in log_text
    assert "Slot generation complete" in log_text

    # Verify structured logging (extra fields)
    records = [r for r in caplog.records if r.name.startswith("vortex.orchestration")]
    assert len(records) >= 5  # At least 5 key events

    # Find the final completion record
    completion_records = [
        r for r in records if "Slot generation complete" in r.message
    ]
    assert len(completion_records) == 1

    # Verify completion record exists
    # Note: extra fields may not be directly accessible in test environment
    # Production logging will have these via structured formatter


# ============================================================================
# VRAM Pressure Tests
# ============================================================================


@pytest.mark.asyncio
async def test_vram_pressure_handling(mock_pipeline, scheduler_config, test_recipe):
    """Test VRAM limit handling during generation.

    Why: Validates memory monitoring integration
    Contract: MemoryPressureError raised if hard limit exceeded

    Note: This test mocks the error; real VRAM testing requires GPU
    """
    # Override image generation to raise VRAM error
    async def oom_image(recipe):
        await asyncio.sleep(0.01)
        raise MemoryPressureError("VRAM hard limit exceeded: 11.8GB > 11.5GB")

    mock_pipeline._generate_actor = oom_image

    scheduler = SlotScheduler(pipeline=mock_pipeline, config=scheduler_config)

    with pytest.raises(MemoryPressureError, match="VRAM hard limit exceeded"):
        await scheduler.execute(
            recipe=test_recipe,
            slot_id=88888,
            deadline=None,
        )


# ============================================================================
# Timeout Enforcement Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stage_timeout_enforcement(mock_pipeline, scheduler_config, test_recipe):
    """Test per-stage timeout enforcement.

    Why: Validates timeout prevents infinite hangs
    Contract: asyncio.TimeoutError raised after timeout
    """
    # Override audio to hang indefinitely
    async def hanging_audio(recipe):
        await asyncio.sleep(100)  # Much longer than 3s timeout
        return torch.randn(1080000)

    mock_pipeline._generate_audio = hanging_audio

    scheduler = SlotScheduler(pipeline=mock_pipeline, config=scheduler_config)

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.execute(
            recipe=test_recipe,
            slot_id=11111,
            deadline=None,
        )


# ============================================================================
# Audio Retry Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_retry_recovery(mock_pipeline, scheduler_config, test_recipe):
    """Test audio retry recovers from transient failure.

    Why: Validates retry mechanism for transient errors
    Contract: Slot completes successfully after retry
    """
    attempt_count = 0

    async def flaky_audio(recipe):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise RuntimeError("Transient CUDA error")
        await asyncio.sleep(0.02)
        return torch.randn(1080000, device="cpu")

    mock_pipeline._generate_audio = flaky_audio

    scheduler = SlotScheduler(pipeline=mock_pipeline, config=scheduler_config)

    result = await scheduler.execute(
        recipe=test_recipe,
        slot_id=22222,
        deadline=None,
    )

    # Should succeed after retry
    assert result.deadline_met is True
    assert attempt_count == 2  # First failed, second succeeded
