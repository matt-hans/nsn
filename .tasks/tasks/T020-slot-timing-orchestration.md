---
id: T020
title: Slot Timing Orchestration - 45-Second Pipeline Scheduler
status: pending
priority: 1
agent: ai-ml
dependencies: [T014, T015, T016, T017, T018]
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [vortex, ai-ml, python, scheduling, orchestration, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - prd.md#section-12.3-slot-timing-budget
  - architecture.md#section-4.2.1-director-node-components

est_tokens: 13000
actual_tokens: null
---

## Description

Implement slot timing orchestration to ensure the Vortex pipeline meets the 45-second glass-to-glass deadline. This system schedules AI generation, BFT coordination, and content propagation within strict time budgets.

**Critical Timeline (45-second slot)**:
- **0-12s: GENERATION PHASE**
  - 0-2s: Audio (Kokoro) — parallel with Flux
  - 0-12s: Actor image (Flux) — parallel with audio
  - Audio completes → 12-20s: Video warping (LivePortrait)
  - 20-21s: CLIP verification (dual ensemble)
- **21-26s: BFT PHASE** (off-chain, separate task)
  - Directors exchange CLIP embeddings
  - Compute 3-of-5 agreement
  - Submit BFT result to ICN Chain
- **26-40s: PROPAGATION PHASE** (off-chain, separate task)
  - Super-nodes download from canonical director
  - Regional relays download from super-nodes
  - Edge viewers download + buffer
- **40-45s: PLAYBACK BUFFER**
  - Viewers have 5-second grace period before deadline

**This task focuses on the GENERATION PHASE (0-21s) orchestration.**

## Business Context

**User Story**: As a Director node, I want the generation pipeline to automatically coordinate audio, image, video, and verification within the 12-second budget, so that I have enough time for BFT coordination and avoid slot misses that result in 150 ICN reputation penalties.

**Why This Matters**:
- Slot timing is the most critical constraint in ICN - miss the 45s deadline = automatic slot rejection
- Generation phase (12s) is the bottleneck - must optimize parallelization
- Poor scheduling leads to sequential execution (audio → image → video = 22s, MISS)
- Good scheduling enables parallelization (audio ∥ image → video = 12s, SUCCESS)
- Deadline tracking prevents wasted work (abort generation if can't meet deadline)

**What It Unblocks**:
- Production Director node deployment
- Off-chain BFT coordination (T021, separate task)
- Integration with P2P propagation layer
- End-to-end ICN slot generation workflow

**Priority Justification**: Priority 1 (Critical Path) - This is the final integration task for the Vortex Engine. Without timing orchestration, the pipeline cannot meet deadlines, making ICN non-functional.

## Acceptance Criteria

- [ ] SlotScheduler class orchestrates audio ∥ image → video → CLIP pipeline
- [ ] Audio (Kokoro) and image (Flux) generation run in parallel via asyncio.create_task()
- [ ] Video warping (LivePortrait) starts immediately after audio completes (dependency)
- [ ] CLIP verification starts immediately after video completes
- [ ] Total generation phase completes in <12s P99 on RTX 3060
- [ ] Deadline tracking: abort generation if current_time + remaining_work > deadline
- [ ] Progress checkpoints logged: audio_complete, image_complete, video_complete, clip_complete
- [ ] Timeout handling: all async tasks have max timeout (audio: 3s, image: 15s, video: 10s, CLIP: 2s)
- [ ] Error recovery: if audio fails, retry once; if image/video/CLIP fails, abort slot
- [ ] Slot metadata includes: generation_time_ms, breakdown{audio, image, video, clip}, deadline_met: bool
- [ ] Integration with VortexPipeline.generate_slot() method
- [ ] Configuration: timeouts, retry policy, deadline buffer (from config.yaml)

## Test Scenarios

**Test Case 1: Successful Parallel Execution**
- Given: VortexPipeline with all models loaded
  And Recipe with 45-second slot deadline
- When: SlotScheduler.execute(recipe) is called
- Then: Audio and image generation start simultaneously (t=0s)
  And audio completes at t=2s
  And image completes at t=12s
  And video warping starts at t=12s (waits for image)
  And video completes at t=20s
  And CLIP verification starts at t=20s
  And CLIP completes at t=21s
  And total generation time is 21s (within 12s budget after optimization)
  And deadline_met = True

**Test Case 2: Sequential vs. Parallel Performance**
- Given: Same Recipe
- When: SlotScheduler runs with parallelization disabled (sequential mode)
- Then: Audio runs 0-2s, then image 2-14s, then video 14-22s, then CLIP 22-23s
  And total time is 23s (vs 21s parallel)
  And parallelization saves 2 seconds

**Test Case 3: Deadline Abort**
- Given: Slow GPU (RTX 2060) or high load
  And image generation at t=18s, video not started
  And deadline is t=45s
- When: Scheduler checks: current_time(18s) + remaining_work(video:10s + clip:1s) = 29s vs. deadline 45s
- Then: Remaining budget is 45 - 18 = 27s, needed is 11s, OK to continue
  BUT if current_time = 36s, remaining = 9s < needed 11s
  And scheduler aborts generation
  And DirectorSlotMissed event is emitted
  And error logged: "Deadline miss predicted: 36s + 11s = 47s > 45s deadline"

**Test Case 4: Audio Failure with Retry**
- Given: Audio generation fails due to transient CUDA error
- When: Scheduler detects audio failure
- Then: Audio is retried once (with exponential backoff)
  And if retry succeeds, pipeline continues
  And if retry fails, entire slot is aborted
  And error logged: "Audio generation failed after 1 retry, aborting slot"

**Test Case 5: Timeout Enforcement**
- Given: Image generation (Flux) hangs at 15s (normally completes in 12s)
- When: Timeout (15s) is reached
- Then: asyncio.CancelledError is raised
  And image generation task is cancelled
  And CUDA kernels are synchronized
  And slot is aborted
  And error logged: "Image generation timeout (15s), aborting slot"

**Test Case 6: Progress Checkpoint Logging**
- Given: Slot generation in progress
- When: Each stage completes
- Then: Progress logs are emitted:
  ```json
  {"event": "audio_complete", "elapsed_ms": 2000}
  {"event": "image_complete", "elapsed_ms": 12000}
  {"event": "video_complete", "elapsed_ms": 20000}
  {"event": "clip_complete", "elapsed_ms": 21000}
  {"event": "generation_complete", "total_ms": 21000, "deadline_met": true}
  ```

**Test Case 7: CLIP Self-Check Failure**
- Given: Video generation completes
  And CLIP self-check returns self_check_passed = False (score <0.70)
- When: Scheduler receives CLIP result
- Then: Slot is marked as failed (not submitted to BFT)
  And error logged: "CLIP self-check failed: score_b=0.65, score_l=0.68, aborting BFT submission"
  And Director can optionally regenerate with different Recipe parameters

## Technical Implementation

**Required Components**:

1. **vortex/orchestration/scheduler.py** (Core scheduler)
   - `SlotScheduler` class with `execute(recipe: Recipe) -> SlotResult`
   - `_orchestrate_parallel()` method for audio ∥ image
   - `_wait_for_audio()` method (video dependency)
   - Deadline tracking and abort logic
   - Progress checkpoint logging

2. **vortex/orchestration/models.py** (Data models)
   - `SlotResult` dataclass: video, audio, clip_embedding, metadata, deadline_met
   - `GenerationBreakdown` dataclass: audio_ms, image_ms, video_ms, clip_ms
   - `SlotMetadata` dataclass: slot_id, start_time, end_time, deadline

3. **vortex/orchestration/timeouts.py** (Timeout management)
   - `with_timeout(coro, timeout_s)` async context manager
   - Configurable timeouts per stage
   - Grace period for cleanup after cancellation

4. **vortex/config.yaml** (Configuration)
   - `timeouts: {audio_s: 3, image_s: 15, video_s: 10, clip_s: 2}`
   - `retry_policy: {audio: 1, image: 0, video: 0, clip: 0}`
   - `deadline_buffer_s: 5` (safety margin)

5. **vortex/tests/integration/test_slot_orchestration.py** (Integration tests)
   - Test cases 1-7 from above
   - End-to-end slot generation
   - Deadline simulation

6. **vortex/benchmarks/slot_timing.py** (Performance benchmark)
   - Measure breakdown times over 100 slots
   - Plot Gantt chart of parallel execution
   - Identify bottlenecks

**Validation Commands**:
```bash
# Integration test (full pipeline)
pytest vortex/tests/integration/test_slot_orchestration.py --gpu -v

# Timing benchmark
python vortex/benchmarks/slot_timing.py --slots 100

# Deadline simulation (slow GPU)
python vortex/tests/stress/deadline_simulation.py --gpu-slowdown 1.5x

# Parallel vs. sequential comparison
python vortex/benchmarks/parallel_vs_sequential.py
```

**Code Patterns**:
```python
# From vortex/orchestration/scheduler.py
import asyncio
import time
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class SlotResult:
    """Result of slot generation."""
    video_frames: torch.Tensor
    audio_waveform: torch.Tensor
    clip_result: DualClipResult
    metadata: SlotMetadata
    breakdown: GenerationBreakdown
    deadline_met: bool

@dataclass
class GenerationBreakdown:
    """Timing breakdown of generation stages."""
    audio_ms: int
    image_ms: int
    video_ms: int
    clip_ms: int
    total_ms: int

@dataclass
class SlotMetadata:
    """Metadata for slot generation."""
    slot_id: int
    start_time: float
    end_time: float
    deadline: float

class SlotScheduler:
    """Orchestrate AI generation pipeline with deadline tracking."""

    def __init__(self, pipeline: VortexPipeline, config: dict):
        self.pipeline = pipeline
        self.timeouts = config["timeouts"]
        self.retry_policy = config["retry_policy"]
        self.deadline_buffer_s = config["deadline_buffer_s"]

    async def execute(self, recipe: Recipe) -> SlotResult:
        """Execute slot generation with deadline tracking."""
        start_time = time.monotonic()
        deadline = start_time + recipe.slot_params.duration_sec

        metadata = SlotMetadata(
            slot_id=recipe.slot_id,
            start_time=start_time,
            end_time=0,  # Set later
            deadline=deadline
        )

        # PHASE 1: Parallel audio + image generation
        audio_task = asyncio.create_task(
            self._with_retry(
                self._generate_audio_with_timeout(recipe),
                retries=self.retry_policy["audio"]
            )
        )
        image_task = asyncio.create_task(
            self._generate_image_with_timeout(recipe)
        )

        # Wait for both to complete
        try:
            audio_waveform, image_result = await asyncio.gather(
                audio_task, image_task
            )
        except asyncio.CancelledError:
            logger.error("Generation cancelled during parallel phase")
            raise
        except Exception as e:
            logger.error(f"Generation failed during parallel phase: {e}")
            raise

        audio_time = time.monotonic() - start_time
        image_time = time.monotonic() - start_time
        logger.info("Parallel phase complete", extra={
            "audio_ms": int(audio_time * 1000),
            "image_ms": int(image_time * 1000)
        })

        # Check deadline before continuing
        if not self._check_deadline(time.monotonic(), deadline, remaining_work_s=10):
            raise DeadlineMissError("Deadline miss predicted after parallel phase")

        # PHASE 2: Video warping (depends on audio + image)
        video_start = time.monotonic()
        video_frames = await self._generate_video_with_timeout(
            recipe, image_result, audio_waveform
        )
        video_time = (time.monotonic() - video_start) * 1000
        logger.info("Video generation complete", extra={"video_ms": int(video_time)})

        # Check deadline before CLIP
        if not self._check_deadline(time.monotonic(), deadline, remaining_work_s=2):
            raise DeadlineMissError("Deadline miss predicted before CLIP")

        # PHASE 3: CLIP verification
        clip_start = time.monotonic()
        clip_result = await self._verify_with_clip(video_frames, recipe.visual_track.prompt)
        clip_time = (time.monotonic() - clip_start) * 1000
        logger.info("CLIP verification complete", extra={
            "clip_ms": int(clip_time),
            "ensemble_score": clip_result.ensemble_score,
            "self_check_passed": clip_result.self_check_passed
        })

        # Check CLIP self-check
        if not clip_result.self_check_passed:
            logger.warning(f"CLIP self-check failed: score_b={clip_result.score_clip_b:.3f}, "
                         f"score_l={clip_result.score_clip_l:.3f}")
            # Director can choose to abort or retry with different params
            # For now, mark as failed but return result

        # Finalize
        end_time = time.monotonic()
        metadata.end_time = end_time
        total_ms = int((end_time - start_time) * 1000)
        deadline_met = end_time <= deadline

        breakdown = GenerationBreakdown(
            audio_ms=int(audio_time * 1000),
            image_ms=int(image_time * 1000),
            video_ms=int(video_time),
            clip_ms=int(clip_time),
            total_ms=total_ms
        )

        logger.info("Slot generation complete", extra={
            "total_ms": total_ms,
            "deadline_met": deadline_met,
            "breakdown": breakdown.__dict__
        })

        return SlotResult(
            video_frames=video_frames,
            audio_waveform=audio_waveform,
            clip_result=clip_result,
            metadata=metadata,
            breakdown=breakdown,
            deadline_met=deadline_met
        )

    def _check_deadline(self, current_time: float, deadline: float,
                        remaining_work_s: float) -> bool:
        """Check if remaining work can complete before deadline."""
        time_remaining = deadline - current_time
        buffer = self.deadline_buffer_s
        return time_remaining - buffer >= remaining_work_s

    async def _generate_audio_with_timeout(self, recipe: Recipe) -> torch.Tensor:
        """Generate audio with timeout."""
        return await asyncio.wait_for(
            self.pipeline._generate_audio(recipe.audio_track),
            timeout=self.timeouts["audio_s"]
        )

    async def _generate_image_with_timeout(self, recipe: Recipe) -> torch.Tensor:
        """Generate actor image with timeout."""
        return await asyncio.wait_for(
            self.pipeline._generate_actor(recipe.visual_track),
            timeout=self.timeouts["image_s"]
        )

    async def _generate_video_with_timeout(self, recipe: Recipe,
                                           image: torch.Tensor,
                                           audio: torch.Tensor) -> torch.Tensor:
        """Generate video with timeout."""
        return await asyncio.wait_for(
            self.pipeline._warp_video(
                actor_image=image,
                driving_audio=audio,
                expression=recipe.visual_track.expression_sequence[0],
                duration_sec=recipe.slot_params.duration_sec
            ),
            timeout=self.timeouts["video_s"]
        )

    async def _verify_with_clip(self, video: torch.Tensor, prompt: str) -> DualClipResult:
        """Verify video with CLIP ensemble."""
        return await asyncio.wait_for(
            self.pipeline._compute_clip_embedding(video, prompt),
            timeout=self.timeouts["clip_s"]
        )

    async def _with_retry(self, coro, retries: int = 1):
        """Retry async coroutine on failure."""
        for attempt in range(retries + 1):
            try:
                return await coro
            except Exception as e:
                if attempt < retries:
                    logger.warning(f"Attempt {attempt+1}/{retries+1} failed, retrying: {e}")
                    await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                else:
                    raise

class DeadlineMissError(RuntimeError):
    """Raised when generation cannot meet deadline."""
    pass
```

## Dependencies

**Hard Dependencies** (must be complete first):
- [T014] Vortex Core Pipeline - provides VortexPipeline base
- [T015] Flux-Schnell Integration - provides _generate_actor()
- [T016] LivePortrait Integration - provides _warp_video()
- [T017] Kokoro-82M TTS Integration - provides _generate_audio()
- [T018] Dual CLIP Ensemble - provides _compute_clip_embedding()

**Soft Dependencies** (nice to have):
- [T019] VRAM Manager - tracks memory during orchestration

**External Dependencies**:
- Python 3.11
- asyncio (built-in)
- PyTorch 2.1+ (async CUDA stream support)

## Design Decisions

**Decision 1: Parallelization (Audio ∥ Image) vs. Sequential**
- **Rationale**: Audio takes 2s, image takes 12s. Running sequentially = 14s total. Running parallel = max(2s, 12s) = 12s, saves 2 seconds (17% speedup).
- **Alternatives**:
  - Sequential (rejected: wastes 2 seconds)
  - Parallel video + CLIP (rejected: video depends on audio, can't start early)
- **Trade-offs**: (+) Faster, meets deadline. (-) More complex error handling for concurrent tasks.

**Decision 2: Deadline Abort (Predictive) vs. Hard Timeout**
- **Rationale**: Predictive abort (check if current + remaining > deadline) allows graceful cleanup. Hard timeout causes abrupt cancellation mid-CUDA kernel.
- **Alternatives**:
  - Hard timeout only (rejected: messy CUDA state)
  - No deadline checking (rejected: wastes GPU time on doomed slots)
- **Trade-offs**: (+) Clean shutdown, reusable resources. (-) Slightly pessimistic (may abort when could squeeze in).

**Decision 3: Audio Retry (1×) vs. No Retry**
- **Rationale**: Audio failures are often transient (CUDA memory fragmentation, brief GPU contention). One retry (2s overhead) has 80% success rate.
- **Alternatives**:
  - No retry (rejected: wastes entire slot on transient failures)
  - Multiple retries (rejected: eats into deadline budget)
- **Trade-offs**: (+) Recovers from transient failures. (-) 2s overhead if retry needed.

**Decision 4: Per-Stage Timeouts vs. Total Timeout**
- **Rationale**: Per-stage timeouts (audio: 3s, image: 15s, video: 10s, CLIP: 2s) allow early detection of stuck stages. Total timeout would delay detection.
- **Alternatives**:
  - Total timeout only (rejected: delayed detection)
  - No timeouts (rejected: infinite hangs)
- **Trade-offs**: (+) Early detection, granular control. (-) More config parameters.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Parallel tasks deadlock | High (generation hangs) | Low (simple DAG) | Timeout all tasks, log task dependency graph, test stress scenarios |
| Deadline prediction too conservative | Medium (aborts winnable slots) | Medium (uncertain GPU variance) | Tune deadline buffer (5s default), monitor abort vs. success ratio, adaptive buffer |
| Retry logic causes deadline miss | Medium (wasted time) | Low (1 retry = 2s) | Limit retries to fast stages (audio only), exponential backoff, cancel retries if deadline risk |
| CUDA async stream errors | High (corrupted state) | Low (PyTorch handles) | Synchronize CUDA before stage transitions, verify tensor validity, catch CUDA errors |
| Progress logging overhead | Low (performance) | Low (structured logging fast) | Use async logging, batch logs, disable debug logs in production |

## Context7 Enrichment

> **Sources**: Context7 `/huggingface/diffusers`, Python asyncio documentation

### Python asyncio Parallel Execution

**Parallel Audio + Image Generation**:
```python
import asyncio

async def execute_parallel_generation(recipe):
    # Create concurrent tasks
    audio_task = asyncio.create_task(generate_audio(recipe))
    image_task = asyncio.create_task(generate_image(recipe))
    
    # Wait for both to complete
    audio_result, image_result = await asyncio.gather(
        audio_task, 
        image_task
    )
    
    return audio_result, image_result
```

**Timeout Enforcement**:
```python
async def generate_with_timeout(recipe, timeout_s):
    try:
        result = await asyncio.wait_for(
            pipeline.generate(recipe),
            timeout=timeout_s
        )
        return result
    except asyncio.TimeoutError:
        logging.error(f"Generation timeout after {timeout_s}s")
        raise DeadlineMissError(f"Timeout exceeded: {timeout_s}s")
```

**Exponential Backoff Retry**:
```python
async def with_retry(coro, retries=1):
    for attempt in range(retries + 1):
        try:
            return await coro
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(0.5 * (2 ** attempt))
            else:
                raise
```

### Diffusers Memory Optimization for Orchestration

**Sequential CPU Offload** (for memory-constrained environments):
```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
# Moves submodules between CPU/GPU during inference
pipeline.enable_sequential_cpu_offload()
```

**Group Offloading** (advanced memory management):
```python
from diffusers.utils import apply_group_offloading

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

# Leaf-level offloading for transformer
pipeline.transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="leaf_level",
    use_stream=True
)

# Block-level offloading for text encoder
apply_group_offloading(
    pipeline.text_encoder,
    onload_device=onload_device,
    offload_type="block_level",
    num_blocks_per_group=2
)
```

**Deadline Check Pattern**:
```python
def check_deadline(current_time, deadline, remaining_work_s, buffer_s=5):
    time_remaining = deadline - current_time
    return (time_remaining - buffer_s) >= remaining_work_s
```

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive Vortex Engine tasks per PRD sections 12.1-12.3
**Dependencies:** T014, T015, T016, T017, T018 (all Vortex components)
**Estimated Complexity:** Standard (13,000 tokens estimated)

**Notes**: Slot timing orchestration is the final integration task for Vortex Engine. Enables parallelization (audio ∥ image) to save 2 seconds. Deadline tracking prevents wasted work. Critical for meeting 45-second glass-to-glass target.

## Completion Checklist

**Code Complete**:
- [ ] vortex/orchestration/scheduler.py implemented with SlotScheduler.execute()
- [ ] Parallel audio + image generation (asyncio.gather)
- [ ] Video warping dependency (waits for audio)
- [ ] CLIP verification after video
- [ ] Deadline tracking and predictive abort
- [ ] Per-stage timeout enforcement
- [ ] Audio retry logic (1× retry)
- [ ] Progress checkpoint logging
- [ ] SlotResult, GenerationBreakdown, SlotMetadata dataclasses

**Testing**:
- [ ] Integration test runs full pipeline
- [ ] Parallel vs. sequential comparison shows 2s savings
- [ ] Deadline abort test prevents doomed slots
- [ ] Audio retry test recovers from transient failures
- [ ] Timeout test cancels hung stages
- [ ] Progress logging test validates checkpoint structure
- [ ] CLIP self-check failure test aborts BFT submission

**Documentation**:
- [ ] Docstrings for SlotScheduler, execute()
- [ ] vortex/orchestration/README.md with timing diagram
- [ ] Timeout configuration explained
- [ ] Retry policy documented
- [ ] Deadline buffer rationale (5s default)

**Performance**:
- [ ] Total generation time <12s P99 on RTX 3060
- [ ] Parallelization achieves 17% speedup (14s → 12s)
- [ ] Deadline abort triggers <1% false positives (aborting winnable slots)
- [ ] Audio retry success rate >80% on transient failures

**Definition of Done:**
Task is complete when ALL acceptance criteria met, ALL validations pass, and slot orchestration reliably completes generation phase in <12s P99 on RTX 3060 with deadline tracking and parallel execution.
