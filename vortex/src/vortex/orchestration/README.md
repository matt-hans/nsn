## Slot Timing Orchestration

Orchestrates Vortex AI generation pipeline within strict 45-second glass-to-glass deadline with parallel execution, timeout enforcement, and deadline tracking.

### Overview

The SlotScheduler coordinates 4 AI generation phases to meet ICN's critical timing requirements:

1. **Parallel Phase (0-12s)**: Audio (Kokoro) ∥ Image (Flux) generation
2. **Video Phase (12-20s)**: LivePortrait warping (waits for audio completion)
3. **Verification Phase (20-21s)**: Dual CLIP ensemble semantic verification
4. **BFT Phase (21-26s)**: Off-chain consensus (separate task)

**Key Optimization**: Parallel audio + image saves ~2 seconds vs. sequential execution (17% speedup).

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SlotScheduler                                │
│                                                                 │
│  execute(recipe, slot_id, deadline) -> SlotResult              │
│  ├── Phase 1: Parallel (audio ∥ image) [0-12s]                 │
│  │   ├─► asyncio.create_task(audio_task)                       │
│  │   ├─► asyncio.create_task(image_task)                       │
│  │   └─► asyncio.gather(audio, image)                          │
│  │                                                              │
│  ├── Phase 2: Video warping [12-20s]                           │
│  │   └─► waits for audio completion (dependency)               │
│  │                                                              │
│  ├── Phase 3: CLIP verification [20-21s]                       │
│  │   └─► dual ensemble self-check                              │
│  │                                                              │
│  └── Deadline tracking at each checkpoint                      │
│       └─► abort if current + remaining > deadline              │
└─────────────────────────────────────────────────────────────────┘
```

### Timeline Diagram

```
0s        2s        12s        20s       21s       26s       45s
│─────────┼─────────┼──────────┼─────────┼─────────┼─────────┤
│                                                             │
├─────────┤ Audio (Kokoro TTS)                               │
│         └─► 2s target, 3s timeout                          │
│                                                             │
├───────────────────┤ Image (Flux-Schnell)                   │
│                   └─► 12s target, 15s timeout              │
│                                                             │
                    ├────────┤ Video (LivePortrait)          │
                    │        └─► 8s target, 10s timeout      │
                    │                                         │
                             ├─┤ CLIP Verification           │
                             │ └─► 1s target, 2s timeout     │
                             │                                │
                             └──────┤ BFT Phase (off-chain)  │
                                    └─► 5s target            │
                                                              │
                                          ├─────────┤ Propagation
                                                    └─► 14s  │
                                                              │
                                                    ├─────┤ Buffer
                                                          └─► 5s
```

### Usage

```python
from vortex.orchestration import SlotScheduler
from vortex.pipeline import VortexPipeline
import yaml

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize pipeline and scheduler
pipeline = VortexPipeline(config_path="config.yaml")
scheduler = SlotScheduler(
    pipeline=pipeline,
    config=config["orchestration"]
)

# Execute slot generation
recipe = {
    "recipe_id": "test-12345",
    "slot_params": {"slot_number": 12345, "duration_sec": 45},
    "audio_track": {
        "script": "Welcome to the Interdimensional Cable Network!",
        "voice_id": "rick_c137",
    },
    "visual_track": {
        "prompt": "scientist in lab coat explaining quantum mechanics",
        "expression_sequence": ["neutral", "excited"],
    },
}

result = await scheduler.execute(
    recipe=recipe,
    slot_id=12345,
    deadline=None,  # Uses default 45s from recipe
)

print(f"Generation time: {result.breakdown.total_ms}ms")
print(f"Deadline met: {result.deadline_met}")
print(f"Breakdown: audio={result.breakdown.audio_ms}ms, "
      f"image={result.breakdown.image_ms}ms, "
      f"video={result.breakdown.video_ms}ms, "
      f"clip={result.breakdown.clip_ms}ms")
```

### Configuration

Add to `config.yaml`:

```yaml
orchestration:
  # Per-stage timeouts (seconds)
  timeouts:
    audio_s: 3     # Kokoro TTS timeout
    image_s: 15    # Flux-Schnell timeout
    video_s: 10    # LivePortrait timeout
    clip_s: 2      # Dual CLIP ensemble timeout

  # Retry policy per stage
  retry_policy:
    audio: 1   # Retry audio once (recovers from transient CUDA errors)
    image: 0   # No image retry (too expensive)
    video: 0   # No video retry (too expensive)
    clip: 0    # No CLIP retry (fast, rarely fails)

  # Deadline buffer (seconds) - safety margin
  deadline_buffer_s: 5
```

### Data Models

#### SlotResult

```python
@dataclass
class SlotResult:
    video_frames: torch.Tensor       # [1080, 3, 512, 512]
    audio_waveform: torch.Tensor     # [1080000]
    clip_embedding: torch.Tensor     # [512]
    metadata: SlotMetadata           # Slot ID, timestamps, deadline
    breakdown: GenerationBreakdown   # Timing per stage
    deadline_met: bool               # Whether deadline was met
```

#### GenerationBreakdown

```python
@dataclass
class GenerationBreakdown:
    audio_ms: int    # Audio generation time
    image_ms: int    # Image generation time
    video_ms: int    # Video warping time
    clip_ms: int     # CLIP verification time
    total_ms: int    # Total end-to-end time
```

Note: `total_ms` may not equal sum of stages due to parallel execution.

#### SlotMetadata

```python
@dataclass
class SlotMetadata:
    slot_id: int        # Unique slot identifier
    start_time: float   # Start timestamp (monotonic)
    end_time: float     # End timestamp (monotonic)
    deadline: float     # Absolute deadline timestamp
```

### Deadline Tracking

The scheduler uses **predictive abort** to prevent wasted work on slots that cannot meet the deadline:

```python
def _check_deadline(current_time, deadline, remaining_work_s):
    """Check if remaining work can complete before deadline."""
    time_remaining = deadline - current_time
    buffer = deadline_buffer_s  # 5s default safety margin
    return time_remaining - buffer >= remaining_work_s
```

**Example**:
- Current time: 36s
- Deadline: 45s
- Remaining work: 11s (video + CLIP)
- Available: 45 - 36 = 9s
- Needed: 11s + 5s buffer = 16s
- **Result**: 9s < 16s → Abort (DeadlineMissError)

### Timeout Enforcement

Each stage has a maximum timeout to prevent infinite hangs:

| Stage | Target Time | Timeout | Retry |
|-------|-------------|---------|-------|
| Audio (Kokoro) | 2s | 3s | 1× |
| Image (Flux) | 12s | 15s | None |
| Video (LivePortrait) | 8s | 10s | None |
| CLIP Verification | 1s | 2s | None |

Timeouts use `asyncio.wait_for()`:

```python
async def _generate_audio_with_timeout(recipe):
    return await asyncio.wait_for(
        pipeline._generate_audio(recipe),
        timeout=timeouts["audio_s"]
    )
```

### Retry Logic

Audio generation supports **1× retry** with exponential backoff to recover from transient CUDA errors:

```python
async def _with_retry(coro_func, retries=1):
    for attempt in range(retries + 1):
        try:
            return await coro_func()
        except Exception as e:
            if attempt < retries:
                backoff_s = 0.5 * (2 ** attempt)
                await asyncio.sleep(backoff_s)
            else:
                raise
```

**Backoff schedule**:
- Attempt 1: Immediate
- Attempt 2: 0.5s delay
- Attempt 3: 1.0s delay (if more retries configured)

### Progress Logging

The scheduler emits structured log events at key checkpoints:

```json
{"event": "starting_slot_generation", "slot_id": 12345, "deadline_s": 45}
{"event": "parallel_phase_complete", "audio_ms": 2000, "image_ms": 12000}
{"event": "video_generation_complete", "video_ms": 8000}
{"event": "clip_verification_complete", "clip_ms": 1000, "ensemble_score": 0.77}
{"event": "slot_generation_complete", "total_ms": 21000, "deadline_met": true}
```

### Error Handling

| Error Type | Cause | Recovery |
|------------|-------|----------|
| `DeadlineMissError` | Predicted deadline violation | Abort slot, log error, return failure |
| `asyncio.TimeoutError` | Stage timeout exceeded | Abort slot, synchronize CUDA, raise error |
| `MemoryPressureError` | VRAM hard limit exceeded | Abort slot, clear cache, raise error |
| `RuntimeError` (CUDA) | CUDA kernel failure | Retry (audio only), else abort |

### Performance Targets

| Metric | Target | Verification |
|--------|--------|--------------|
| Total generation time | <12s P99 | Run benchmark on RTX 3060 |
| Parallel speedup | 2s savings | Compare parallel vs. sequential |
| Deadline abort false positives | <1% | Monitor production metrics |
| Audio retry success rate | >80% | Log transient failures |

### Testing

**Unit Tests** (10 tests):
```bash
pytest tests/unit/test_slot_scheduler.py -v
```

**Integration Tests** (7 tests, requires GPU):
```bash
pytest tests/integration/test_slot_orchestration.py --gpu -v
```

**Coverage**:
```bash
pytest tests/unit/test_slot_scheduler.py \
    --cov=src/vortex/orchestration \
    --cov-report=term-missing
```

### Benchmark

```bash
# Measure breakdown times over 100 slots
python vortex/benchmarks/slot_timing.py --slots 100

# Compare parallel vs sequential
python vortex/benchmarks/parallel_vs_sequential.py
```

### Known Limitations

1. **CUDA Cancellation**: Timeout cancellation may leave GPU in inconsistent state. Mitigated by `torch.cuda.synchronize()` after cancellation.

2. **Deadline Buffer Tuning**: Default 5s buffer may be too conservative (aborts winnable slots) or too aggressive (misses deadline). Tune based on production metrics.

3. **Single-Node Only**: Scheduler assumes single-node execution. Multi-node BFT coordination is handled by separate off-chain layer (T021-T027).

4. **No Async CLIP**: CLIP verification is currently synchronous. Future optimization: async CLIP encoding during video generation.

### Future Enhancements

- **Adaptive Deadline Buffer**: Adjust buffer based on historical generation times and GPU load
- **Async CLIP**: Overlap CLIP encoding with video generation for further speedup
- **Dynamic Timeout Adjustment**: Adjust timeouts based on GPU performance characteristics
- **Graceful Degradation**: Fall back to lower-quality models if deadline risk detected

### Related Tasks

- **T014**: Vortex Core Pipeline - Base VortexPipeline implementation
- **T015**: Flux-Schnell Integration - Image generation
- **T016**: LivePortrait Integration - Video warping
- **T017**: Kokoro-82M TTS Integration - Audio generation
- **T018**: Dual CLIP Ensemble - Semantic verification
- **T019**: VRAM Manager - Memory monitoring (soft dependency)
- **T021-T027**: Off-chain BFT coordination (next phase)

### References

- PRD §10.3: Slot Timing Budget (45-second breakdown)
- Architecture §4.2: Director Node Components (Vortex Engine)
- Task Specification: `.tasks/tasks/T020-slot-timing-orchestration.md`
