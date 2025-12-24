---
id: T014
title: Vortex Core Pipeline - Static VRAM Manager & Generation Orchestration
status: pending
priority: 1
agent: ai-ml
dependencies: []
blocked_by: []
created: 2025-12-24T00:00:00Z
updated: 2025-12-24T00:00:00Z
tags: [vortex, ai-ml, python, gpu, phase1]

context_refs:
  - context/project.md
  - context/architecture.md
  - context/acceptance-templates.md

docs_refs:
  - prd.md#section-12.1-static-resident-vram-layout
  - prd.md#section-12.2-generation-pipeline
  - architecture.md#section-4.2.1-director-node-components

est_tokens: 15000
actual_tokens: null
---

## Description

Implement the core Vortex pipeline orchestration system that manages static GPU-resident AI models and coordinates multi-phase video generation. This is the foundational component that all other Vortex modules depend on.

**Critical Constraint**: All models must remain loaded in VRAM at all times (no swapping). Total VRAM budget is 11.8GB to fit RTX 3060 12GB cards. The pipeline must pre-allocate output buffers at startup to avoid memory fragmentation during generation.

**Key Architecture Patterns**:
- Python 3.11 async/await for concurrent audio + image generation
- PyO3 FFI bridge for Rust off-chain node integration
- Static model registry with lifetime guarantees
- Pre-allocated tensor buffers to prevent CUDA OOM
- VRAM pressure monitoring with soft/hard limits

## Business Context

**User Story**: As a Director node operator, I want my GPU to efficiently generate 45-second video slots without model loading delays or memory fragmentation, so that I can consistently meet the 12-second generation deadline and earn reputation rewards.

**Why This Matters**:
- The Vortex pipeline is the performance bottleneck for the entire ICN system
- Static VRAM residency is the only way to achieve <15s generation on consumer GPUs
- Memory fragmentation failures would cause slot misses, leading to 150 ICN slashing penalties
- This component directly determines Director profitability and network reliability

**What It Unblocks**:
- T015 (Flux integration), T016 (LivePortrait), T017 (Kokoro), T018 (CLIP ensemble)
- Off-chain Director node implementation (depends on Vortex via PyO3)
- Slot timing orchestration (T020)

**Priority Justification**: Priority 1 (Critical Path) - All AI generation tasks depend on this foundation. Without it, no video generation is possible, making the entire ICN protocol non-functional.

## Acceptance Criteria

- [ ] VortexPipeline class loads all models at __init__ and never unloads them
- [ ] Total VRAM usage stays ≤11.5GB (500MB safety margin) under all generation scenarios
- [ ] Pre-allocated output buffers exist for: actor_buffer (512x512x3), video_buffer (45s×24fps×512x512x3), audio_buffer (45s×24kHz)
- [ ] GenerationResult dataclass contains: video_frames, audio_waveform, clip_embedding, generation_time_ms, slot_id
- [ ] generate_slot() method completes in <15s P99 on RTX 3060 (12GB)
- [ ] VRAM pressure monitor raises MemoryPressureWarning at 11.0GB, MemoryPressureError at 11.5GB
- [ ] Memory is NOT freed between slot generations (models stay resident)
- [ ] All async operations use asyncio.create_task() for parallelization
- [ ] Pipeline gracefully handles CUDA OOM errors with structured logging
- [ ] Model registry exposes get_model(name: str) -> torch.nn.Module for child components
- [ ] Configuration loaded from vortex/config.yaml (device, precision, buffer sizes)
- [ ] Python logging outputs JSON-structured logs to stdout (for Vector aggregation)

## Test Scenarios

**Test Case 1: Cold Start Model Loading**
- Given: Fresh Python process with no models loaded
- When: VortexPipeline() is instantiated
- Then: All 5 models (Flux, LivePortrait, Kokoro, CLIP-B, CLIP-L) are loaded
  And total VRAM usage is 10.8-11.8GB
  And no CUDA OOM errors occur
  And model_registry contains all 5 models

**Test Case 2: Parallel Audio + Image Generation**
- Given: VortexPipeline is initialized
- When: generate_slot(recipe) is called with valid Recipe
- Then: _generate_audio() and _generate_actor() run concurrently
  And both complete within 12 seconds
  And audio_task finishes before video warping starts (dependency)
  And no race conditions or deadlocks occur

**Test Case 3: Memory Pressure Detection**
- Given: VortexPipeline with VRAM at 10.5GB
- When: A generation temporarily allocates 1.5GB for intermediate tensors
- Then: VRAM monitor detects usage >11.0GB
  And MemoryPressureWarning is logged with current usage details
  And generation continues (soft limit)
  And if usage exceeds 11.5GB, MemoryPressureError is raised

**Test Case 4: Buffer Reuse Across Generations**
- Given: VortexPipeline has completed slot N generation
- When: generate_slot() is called for slot N+1
- Then: actor_buffer, video_buffer, audio_buffer are reused (not reallocated)
  And VRAM usage does NOT increase between slots
  And buffer contents are overwritten (not appended)

**Test Case 5: CUDA OOM Graceful Failure**
- Given: System VRAM artificially limited to 10GB (via nvidia-smi)
- When: VortexPipeline() attempts to load all models
- Then: torch.cuda.OutOfMemoryError is caught
  And structured error log includes: model_name, requested_vram, available_vram
  And pipeline raises VortexInitializationError with remediation suggestion
  And partial models are unloaded/garbage collected

**Test Case 6: Async Cancellation**
- Given: generate_slot() is running (8s elapsed)
- When: asyncio.CancelledError is raised (e.g., slot deadline expired)
- Then: All pending async tasks (_generate_audio, _generate_actor) are cancelled
  And CUDA kernels are synchronized (no orphaned GPU work)
  And partial results are NOT returned
  And next generate_slot() call succeeds

## Technical Implementation

**Required Components**:

1. **vortex/pipeline.py** (Core module)
   - `VortexPipeline` class with `__init__`, `generate_slot()`, async helper methods
   - `ModelRegistry` class for model lifecycle management
   - `VRAMMonitor` class for memory pressure detection
   - `GenerationResult` dataclass for output packaging

2. **vortex/config.yaml** (Configuration)
   - Device selection: `cuda:0` or `cpu` (fallback)
   - Model precision overrides (e.g., force FP16 for debugging)
   - Buffer sizes: actor, video, audio
   - VRAM limits: soft (11.0GB), hard (11.5GB)

3. **vortex/models/__init__.py** (Model loader registry)
   - Factory functions: `load_flux()`, `load_liveportrait()`, etc.
   - Lazy import to avoid startup overhead for non-Director nodes

4. **vortex/utils/memory.py** (VRAM utilities)
   - `get_current_vram_usage() -> int` (bytes)
   - `log_vram_snapshot(label: str)` for debugging
   - `clear_cuda_cache()` for emergency cleanup

**Validation Commands**:
```bash
# Install dependencies
pip install -r vortex/requirements.txt

# Unit tests (mocked models)
pytest vortex/tests/unit/test_pipeline.py -v

# Integration test (real models, requires GPU)
pytest vortex/tests/integration/test_full_generation.py --gpu -v

# VRAM profiling
python vortex/benchmarks/vram_profile.py

# Generation benchmark
python vortex/benchmarks/slot_generation.py --slots 5 --max-time 15

# Async stress test
python vortex/tests/stress/async_cancellation.py
```

**Code Patterns**:
```python
# Static model loading (from PRD example)
class VortexPipeline:
    def __init__(self, device: str = "cuda"):
        self.device = device

        # All models loaded once - NEVER unloaded
        self.flux = self._load_flux()
        self.live_portrait = self._load_live_portrait()
        self.kokoro = self._load_kokoro()
        self.clip_b = self._load_clip_b()
        self.clip_l = self._load_clip_l()

        # Pre-allocate output buffers
        self.actor_buffer = torch.zeros(1, 3, 512, 512, device=device)
        self.audio_buffer = torch.zeros(24000 * 45, device=device)
        self.video_buffer = torch.zeros(45 * 24, 3, 512, 512, device=device)

        # VRAM monitor
        self.vram_monitor = VRAMMonitor(soft_limit=11.0e9, hard_limit=11.5e9)

        log.info("Vortex pipeline initialized", extra={
            "vram_usage_gb": torch.cuda.memory_allocated() / 1e9,
            "models_loaded": 5
        })
```

## Dependencies

**Hard Dependencies** (must be complete first):
- None - This is a foundational task with no on-chain dependencies

**Soft Dependencies** (nice to have):
- Off-chain Rust bridge (for PyO3 integration testing)

**External Dependencies**:
- Python 3.11
- PyTorch 2.1+ with CUDA 12.1+ support
- NVIDIA GPU driver 535+ (for RTX 3060)
- bitsandbytes 0.41+ (for NF4 quantization support)
- accelerate 0.25+ (for model loading)
- transformers 4.36+ (for CLIP models)

## Design Decisions

**Decision 1: Static VRAM Residency (No Model Swapping)**
- **Rationale**: PCIe model swapping takes 1-2s per model, which exceeds our 12s generation budget. Static residency is the only way to achieve target latency.
- **Alternatives**:
  - Dynamic loading (rejected: too slow)
  - CPU offload with pinned memory (rejected: even slower due to PCIe bandwidth)
  - Model distillation to smaller models (future: may reduce VRAM budget)
- **Trade-offs**: (+) Predictable latency, no thrashing. (-) RTX 3060 12GB is hard minimum, excludes lower-end GPUs.

**Decision 2: Pre-Allocated Output Buffers**
- **Rationale**: CUDA memory allocation is expensive (~50ms) and can cause fragmentation. Pre-allocating at startup amortizes this cost.
- **Alternatives**:
  - Allocate per-generation (rejected: fragmentation risk)
  - Memory pool with dynamic resizing (rejected: adds complexity)
- **Trade-offs**: (+) Zero allocation overhead during generation. (-) Wastes ~500MB if buffer size exceeds actual output.

**Decision 3: Async/Await for Parallelization**
- **Rationale**: Audio (Kokoro, 2s) and image (Flux, 12s) generation can run concurrently on GPU via CUDA streams.
- **Alternatives**:
  - Sequential pipeline (rejected: adds 2s to total time)
  - Threading (rejected: Python GIL, complex GPU context management)
- **Trade-offs**: (+) ~15% faster overall. (-) More complex error handling for concurrent tasks.

**Decision 4: JSON Structured Logging**
- **Rationale**: Logs must be machine-parseable for Vector → Loki aggregation and Grafana dashboards.
- **Alternatives**:
  - Plain text logs (rejected: hard to query)
  - Binary logs (rejected: debugging difficulty)
- **Trade-offs**: (+) Easy integration with monitoring stack. (-) Slightly larger log size.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| CUDA OOM during model loading | High (pipeline unusable) | Medium (on lower-end GPUs) | Pre-flight VRAM check with clear error message, suggest upgrade path |
| Memory fragmentation over time | High (slot misses) | Low (mitigated by pre-allocation) | Monitor VRAM allocator stats, restart Director node weekly if needed |
| Async task deadlock | High (generation hangs) | Low (simple DAG) | Timeout on all async tasks (20s max), log task dependency graph on failure |
| PyTorch version incompatibility | Medium (crashes) | Low (pinned versions) | CI matrix tests PyTorch 2.1.0, 2.1.1, 2.1.2, 2.2.0 |
| NVIDIA driver regressions | Medium (crashes) | Low (production drivers) | Document minimum driver version (535+), test on multiple driver versions in CI |

## Progress Log

### [2025-12-24T00:00:00Z] - Task Created

**Created By:** task-creator agent
**Reason:** User request to create comprehensive Vortex Engine tasks per PRD sections 12.1-12.3
**Dependencies:** None (foundational task)
**Estimated Complexity:** Standard (15,000 tokens estimated)

**Notes**: This task is the critical path for all AI generation. Once complete, it unblocks T015 (Flux), T016 (LivePortrait), T017 (Kokoro), T018 (CLIP ensemble), and T020 (timing orchestration).

## Completion Checklist

**Code Complete**:
- [ ] VortexPipeline class implemented with static model loading
- [ ] ModelRegistry with get_model() interface
- [ ] VRAMMonitor with soft/hard limit detection
- [ ] Pre-allocated buffers for actor, video, audio
- [ ] generate_slot() async orchestration
- [ ] JSON structured logging configured
- [ ] Configuration loaded from vortex/config.yaml

**Testing**:
- [ ] Unit tests pass (mocked models)
- [ ] Integration test with real models on GPU
- [ ] VRAM profiling shows ≤11.5GB usage
- [ ] Generation benchmark completes 5 slots in <75s (15s average)
- [ ] Async cancellation test passes
- [ ] CUDA OOM test shows graceful error handling

**Documentation**:
- [ ] Docstrings for all public methods
- [ ] vortex/README.md updated with usage examples
- [ ] VRAM budget documented in comments
- [ ] Async task DAG documented

**Performance**:
- [ ] P99 generate_slot() latency <15s on RTX 3060
- [ ] VRAM usage stable across 100 consecutive generations
- [ ] No memory leaks (VRAM delta <100MB after 100 slots)

**Definition of Done:**
Task is complete when ALL acceptance criteria met, ALL validations pass, and integration tests succeed on RTX 3060 12GB GPU with PyTorch 2.1+.
