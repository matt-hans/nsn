"""Slot timing orchestration scheduler.

Orchestrates AI generation pipeline with deadline tracking, parallel execution,
timeout enforcement, and retry logic.

Key Features:
- Parallel audio + image generation (saves ~2s vs sequential)
- Per-stage timeout enforcement (audio: 3s, image: 15s, video: 10s, CLIP: 2s)
- Predictive deadline abort (prevents wasted work on doomed slots)
- Audio retry with exponential backoff (recovers from transient failures)
- Progress checkpoint logging for observability

Timeline (45-second slot):
- 0-12s: GENERATION PHASE (audio ∥ image → video → CLIP)
  - 0-2s: Audio (Kokoro) - parallel with Flux
  - 0-12s: Actor image (Flux) - parallel with audio
  - 12-20s: Video warping (LivePortrait) - waits for audio
  - 20-21s: CLIP verification (dual ensemble)
- 21-26s: BFT PHASE (off-chain, separate task)
- 26-40s: PROPAGATION PHASE (off-chain, separate task)
- 40-45s: PLAYBACK BUFFER

This module implements the GENERATION PHASE (0-21s) orchestration.
"""

import asyncio
import logging
import time
from typing import Any

import torch

from vortex.models.clip_ensemble import DualClipResult
from vortex.orchestration.models import GenerationBreakdown, SlotMetadata, SlotResult

logger = logging.getLogger(__name__)


class DeadlineMissError(RuntimeError):
    """Raised when generation cannot meet deadline."""

    pass


class SlotScheduler:
    """Orchestrate AI generation pipeline with deadline tracking.

    Manages parallel execution of audio + image generation, sequential video
    warping, and CLIP verification. Enforces per-stage timeouts and tracks
    deadline to abort doomed slots early.

    Example:
        >>> scheduler = SlotScheduler(pipeline, config)
        >>> result = await scheduler.execute(recipe, slot_id=12345, deadline=45.0)
        >>> print(f"Generation took {result.breakdown.total_ms}ms")
    """

    def __init__(self, pipeline: Any, config: dict[str, Any]):
        """Initialize slot scheduler.

        Args:
            pipeline: VortexPipeline instance with generation methods
            config: Configuration dict with timeouts, retry_policy, deadline_buffer_s

        Raises:
            ValueError: If config missing required keys
        """
        # Validate required config keys
        required_keys = ["timeouts", "retry_policy", "deadline_buffer_s"]
        missing_keys = [k for k in required_keys if k not in config]
        if missing_keys:
            raise ValueError(
                f"Config missing required keys: {missing_keys}. "
                f"Required: {required_keys}"
            )

        self.pipeline = pipeline
        self.timeouts = config["timeouts"]
        self.retry_policy = config["retry_policy"]
        self.deadline_buffer_s = config["deadline_buffer_s"]

        logger.info(
            "SlotScheduler initialized",
            extra={
                "timeouts": self.timeouts,
                "retry_policy": self.retry_policy,
                "deadline_buffer_s": self.deadline_buffer_s,
            },
        )

    async def execute(
        self,
        recipe: dict[str, Any],
        slot_id: int,
        deadline: float | None = None,
    ) -> SlotResult:
        """Execute slot generation with deadline tracking.

        Orchestrates:
        1. Parallel: Audio (Kokoro) + Actor image (Flux)
        2. Sequential: Video warping (LivePortrait, waits for audio)
        3. Verification: Dual CLIP embedding

        Args:
            recipe: Recipe dict with audio_track, visual_track, semantic_constraints
            slot_id: Unique slot identifier
            deadline: Optional absolute deadline timestamp (default: start + 45s)

        Returns:
            SlotResult with video, audio, CLIP embedding, metadata, deadline_met

        Raises:
            DeadlineMissError: If deadline cannot be met
            asyncio.TimeoutError: If stage exceeds timeout
            RuntimeError: If generation fails after retries

        Example:
            >>> recipe = {"audio_track": {...}, "visual_track": {...}}
            >>> result = await scheduler.execute(recipe, slot_id=12345)
        """
        start_time = time.monotonic()

        # Default deadline: start + 45s (full slot duration)
        if deadline is None:
            deadline = start_time + 45.0

        metadata = SlotMetadata(
            slot_id=slot_id,
            start_time=start_time,
            end_time=0,  # Set after completion
            deadline=deadline,
        )

        logger.info(
            "Starting slot generation",
            extra={
                "slot_id": slot_id,
                "deadline_s": deadline - start_time,
                "buffer_s": self.deadline_buffer_s,
            },
        )

        try:
            # PHASE 1: Parallel audio + image generation
            audio_start = time.monotonic()
            audio_task = asyncio.create_task(
                self._with_retry(
                    lambda: self._generate_audio_with_timeout(recipe),
                    retries=self.retry_policy["audio"],
                )
            )

            image_start = time.monotonic()
            image_task = asyncio.create_task(
                self._generate_image_with_timeout(recipe)
            )

            # Wait for both to complete
            audio_waveform, actor_image = await asyncio.gather(
                audio_task, image_task
            )

            audio_time_ms = int((time.monotonic() - audio_start) * 1000)
            image_time_ms = int((time.monotonic() - image_start) * 1000)

            logger.info(
                "Parallel phase complete",
                extra={
                    "slot_id": slot_id,
                    "audio_ms": audio_time_ms,
                    "image_ms": image_time_ms,
                },
            )

            # Check deadline before continuing (video + CLIP = ~10s remaining)
            if not self._check_deadline(
                current_time=time.monotonic(),
                deadline=deadline,
                remaining_work_s=10.0,
            ):
                elapsed = time.monotonic() - start_time
                raise DeadlineMissError(
                    f"Deadline miss predicted after parallel phase: "
                    f"elapsed={elapsed:.1f}s, deadline={deadline-start_time:.1f}s, "
                    f"remaining_work=10s"
                )

            # PHASE 2: Video warping (depends on audio + image)
            video_start = time.monotonic()
            video_frames = await self._generate_video_with_timeout(
                recipe, actor_image, audio_waveform
            )
            video_time_ms = int((time.monotonic() - video_start) * 1000)

            logger.info(
                "Video generation complete",
                extra={"slot_id": slot_id, "video_ms": video_time_ms},
            )

            # Check deadline before CLIP (CLIP = ~2s remaining)
            if not self._check_deadline(
                current_time=time.monotonic(),
                deadline=deadline,
                remaining_work_s=2.0,
            ):
                elapsed = time.monotonic() - start_time
                raise DeadlineMissError(
                    f"Deadline miss predicted before CLIP: "
                    f"elapsed={elapsed:.1f}s, deadline={deadline-start_time:.1f}s, "
                    f"remaining_work=2s"
                )

            # PHASE 3: CLIP verification
            clip_start = time.monotonic()
            clip_result = await self._verify_with_clip(
                video_frames, recipe.get("visual_track", {}).get("prompt", "")
            )
            clip_time_ms = int((time.monotonic() - clip_start) * 1000)

            logger.info(
                "CLIP verification complete",
                extra={
                    "slot_id": slot_id,
                    "clip_ms": clip_time_ms,
                    "ensemble_score": clip_result.ensemble_score,
                    "self_check_passed": clip_result.self_check_passed,
                },
            )

            # Check CLIP self-check
            if not clip_result.self_check_passed:
                logger.warning(
                    "CLIP self-check failed",
                    extra={
                        "slot_id": slot_id,
                        "score_b": clip_result.score_clip_b,
                        "score_l": clip_result.score_clip_l,
                        "threshold_b": 0.70,
                        "threshold_l": 0.72,
                    },
                )
                # Continue but mark in result (director can choose to abort BFT)

            # Finalize
            end_time = time.monotonic()
            metadata.end_time = end_time
            total_ms = int((end_time - start_time) * 1000)
            deadline_met = end_time <= deadline

            breakdown = GenerationBreakdown(
                audio_ms=audio_time_ms,
                image_ms=image_time_ms,
                video_ms=video_time_ms,
                clip_ms=clip_time_ms,
                total_ms=total_ms,
            )

            logger.info(
                "Slot generation complete",
                extra={
                    "slot_id": slot_id,
                    "total_ms": total_ms,
                    "deadline_met": deadline_met,
                    "breakdown": {
                        "audio_ms": audio_time_ms,
                        "image_ms": image_time_ms,
                        "video_ms": video_time_ms,
                        "clip_ms": clip_time_ms,
                    },
                },
            )

            return SlotResult(
                video_frames=video_frames,
                audio_waveform=audio_waveform,
                clip_embedding=clip_result.embedding,
                metadata=metadata,
                breakdown=breakdown,
                deadline_met=deadline_met,
            )

        except asyncio.CancelledError:
            logger.warning(f"Slot {slot_id} generation cancelled")
            raise

        except Exception as e:
            logger.error(
                f"Slot {slot_id} generation failed",
                exc_info=True,
                extra={"slot_id": slot_id, "error": str(e)},
            )
            raise

    def _check_deadline(
        self, current_time: float, deadline: float, remaining_work_s: float
    ) -> bool:
        """Check if remaining work can complete before deadline.

        Args:
            current_time: Current timestamp (time.monotonic())
            deadline: Absolute deadline timestamp
            remaining_work_s: Estimated remaining work time (seconds)

        Returns:
            True if sufficient time remaining, False otherwise

        Example:
            >>> can_continue = scheduler._check_deadline(
            ...     current_time=5.0,
            ...     deadline=45.0,
            ...     remaining_work_s=10.0
            ... )
            >>> # Available: 45 - 5 = 40s
            >>> # Needed: 10s + 5s buffer = 15s
            >>> # Result: 40s >= 15s → True
        """
        time_remaining = deadline - current_time
        buffer = self.deadline_buffer_s
        sufficient = time_remaining - buffer >= remaining_work_s

        if not sufficient:
            logger.warning(
                "Insufficient time to meet deadline",
                extra={
                    "time_remaining_s": time_remaining,
                    "remaining_work_s": remaining_work_s,
                    "buffer_s": buffer,
                },
            )

        return bool(sufficient)

    async def _generate_audio_with_timeout(
        self, recipe: dict[str, Any]
    ) -> torch.Tensor:
        """Generate audio with timeout.

        Args:
            recipe: Recipe with audio_track section

        Returns:
            Audio waveform tensor

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        return await asyncio.wait_for(
            self.pipeline._generate_audio(recipe),
            timeout=self.timeouts["audio_s"],
        )

    async def _generate_image_with_timeout(
        self, recipe: dict[str, Any]
    ) -> torch.Tensor:
        """Generate actor image with timeout.

        Args:
            recipe: Recipe with visual_track section

        Returns:
            Actor image tensor

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        return await asyncio.wait_for(
            self.pipeline._generate_actor(recipe),
            timeout=self.timeouts["image_s"],
        )

    async def _generate_video_with_timeout(
        self,
        recipe: dict[str, Any],
        image: torch.Tensor,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Generate video with timeout.

        Args:
            recipe: Recipe with visual_track section
            image: Base actor image
            audio: Audio waveform for lip sync

        Returns:
            Video frames tensor

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        return await asyncio.wait_for(
            self.pipeline._generate_video(image, audio, recipe),
            timeout=self.timeouts["video_s"],
        )

    async def _verify_with_clip(
        self, video: torch.Tensor, prompt: str
    ) -> DualClipResult:
        """Verify video with CLIP ensemble.

        Args:
            video: Generated video frames
            prompt: Text prompt to verify against

        Returns:
            DualClipResult with scores, embedding, self-check status

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        return await asyncio.wait_for(
            self.pipeline._verify_semantic(video, {"visual_track": {"prompt": prompt}}),
            timeout=self.timeouts["clip_s"],
        )

    async def _with_retry(
        self, coro_func: Any, retries: int = 1
    ) -> torch.Tensor:
        """Retry async function on failure with exponential backoff.

        Args:
            coro_func: Async function (or coroutine) to execute
            retries: Maximum number of retries (default: 1)

        Returns:
            Result from successful execution

        Raises:
            Exception: If all retries exhausted

        Example:
            >>> result = await scheduler._with_retry(
            ...     lambda: scheduler._generate_audio_with_timeout(recipe),
            ...     retries=1
            ... )
        """
        result: torch.Tensor | None = None
        for attempt in range(retries + 1):
            try:
                # Support both coroutines and async callables
                if asyncio.iscoroutine(coro_func):
                    # If passed a coroutine directly (first attempt only)
                    if attempt > 0:
                        raise RuntimeError(
                            "Cannot retry coroutine (already awaited). "
                            "Pass a callable instead."
                        )
                    result = await coro_func
                else:
                    # Callable that returns a coroutine
                    result = await coro_func()
                return result
            except Exception as e:
                if attempt < retries:
                    backoff_s = 0.5 * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt+1}/{retries+1} failed, retrying after {backoff_s}s",
                        extra={"error": str(e), "backoff_s": backoff_s},
                    )
                    await asyncio.sleep(backoff_s)
                else:
                    logger.error(
                        f"All {retries+1} attempts exhausted",
                        exc_info=True,
                        extra={"error": str(e)},
                    )
                    raise

        # This should never be reached due to exception raising above
        raise RuntimeError("Retry loop completed without return or exception")
