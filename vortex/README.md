# Vortex - ICN AI Generation Engine

GPU-resident AI pipeline for video generation. All models remain loaded in VRAM at all times.

## VRAM Layout (~11.8 GB)

| Component | Model | Precision | VRAM |
|-----------|-------|-----------|------|
| Actor Generation | Flux-Schnell | NF4 | ~6.0 GB |
| Video Warping | LivePortrait | FP16 | ~3.5 GB |
| Text-to-Speech | Kokoro-82M | FP32 | ~0.4 GB |
| Semantic Verify | CLIP-ViT-B-32 | INT8 | ~0.3 GB |
| Semantic Verify | CLIP-ViT-L-14 | INT8 | ~0.6 GB |
| System Overhead | PyTorch/CUDA | - | ~1.0 GB |

**Minimum GPU:** RTX 3060 12GB

## Pipeline Timing (45s slot)

1. **Parallel Phase (0-12s):** Audio (Kokoro) + Actor image (Flux) generated simultaneously
2. **Sequential Phase (12-15s):** Video warping (LivePortrait)
3. **Verification (15-17s):** Dual CLIP embedding + self-check
4. **BFT (17-30s):** Exchange embeddings, consensus
5. **Propagation (30-45s):** Distribution to network

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Usage

### Initialize Pipeline

```python
from vortex.pipeline import VortexPipeline

# Initialize with default config
pipeline = VortexPipeline()

# Or with custom config
pipeline = VortexPipeline(config_path="custom_config.yaml")
```

### Generate a Slot

```python
import asyncio

recipe = {
    "audio_track": {
        "script": "Welcome to the Interdimensional Cable Network!",
        "voice_id": "rick_c137",
    },
    "visual_track": {
        "prompt": "a mad scientist in a laboratory",
        "negative_prompt": "blurry, low quality",
    },
}

# Async generation
result = await pipeline.generate_slot(recipe=recipe, slot_id=12345)

print(f"Success: {result.success}")
print(f"Generation time: {result.generation_time_ms}ms")
print(f"Video shape: {result.video_frames.shape}")
print(f"Audio shape: {result.audio_waveform.shape}")
```

### Monitor VRAM

```python
from vortex.utils.memory import get_vram_stats, log_vram_snapshot

# Get current stats
stats = get_vram_stats()
print(f"Allocated: {stats['allocated_gb']:.2f} GB")
print(f"Total: {stats['total_gb']:.2f} GB")

# Log snapshot with label
log_vram_snapshot("after_model_load")
```

## Plugins (Custom Renderers)

Vortex supports pluggable renderers that declare input/output schemas and
resource/latency guarantees. Plugins are discovered from a directory of
subfolders with `manifest.yaml`.

Quickstart:

```python
from vortex.plugins import PluginHost

host = PluginHost.from_config()
print(host.registry.list_plugins())
```

See `vortex/src/vortex/plugins/README.md` for the manifest schema and interface.

## Architecture

### Core Components

**VortexPipeline** - Main orchestrator
- Loads all 5 models once at initialization (static VRAM residency)
- Pre-allocates output buffers to prevent fragmentation
- Orchestrates async generation with parallel audio + actor phase
- Returns GenerationResult with video, audio, CLIP embedding

**ModelRegistry** - Model lifecycle manager
- Stores all loaded models in dictionary
- Exposes `get_model(name)` interface for child components
- Handles CUDA OOM gracefully during initialization

**VRAMMonitor** - Memory pressure detection
- Soft limit (11.0GB): Log warning, continue
- Hard limit (11.5GB): Raise MemoryPressureError, abort generation
- Prevents CUDA OOM crashes

**GenerationResult** - Output dataclass
- video_frames: Tensor (num_frames, channels, height, width)
- audio_waveform: Tensor (num_samples,)
- clip_embedding: Tensor (embedding_dim,)
- generation_time_ms: float
- slot_id: int
- success: bool
- error_msg: Optional[str]

### File Structure

```
vortex/
├── config.yaml                 # Configuration (VRAM limits, precision, buffers)
├── src/vortex/
│   ├── __init__.py
│   ├── pipeline.py             # VortexPipeline, ModelRegistry, VRAMMonitor
│   ├── models/
│   │   └── __init__.py         # Model loaders (load_flux, load_kokoro, etc.)
│   ├── utils/
│   │   └── memory.py           # VRAM utilities (get_vram_stats, log_snapshot)
│   └── pipeline/
│       └── __init__.py
└── tests/
    └── unit/
        ├── test_memory.py      # VRAM utility tests
        └── test_pipeline.py    # Pipeline orchestration tests
```

## Testing

### Unit Tests (CPU/GPU)

Unit tests use **real CLIP models on CPU** for higher confidence (reduced mock ratio to ~40%).

```bash
# Run all unit tests with real CLIP (default)
pytest tests/unit/ -v

# Disable real CLIP for faster tests (mocks only)
export CLIP_UNIT_TEST_REAL=false
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_clip_ensemble.py -v

# Run with coverage
pytest tests/unit/ --cov=vortex --cov-report=term-missing
```

### Integration Tests (GPU or CPU Fallback)

Integration tests run on **GPU if available, else CPU fallback** (89% CPU-compatible).

```bash
# Run all integration tests (auto-detects GPU/CPU)
pytest tests/integration/ -v

# Run only CPU-compatible tests (skip GPU-only)
pytest tests/integration/ -v -m "not gpu_only"

# Override latency threshold for CI (default: 1.0s)
export CLIP_CI_LATENCY_THRESHOLD=3.0
pytest tests/integration/test_clip_ensemble.py::test_verification_latency -v
```

### Mutation Testing

Verify tests catch bugs by introducing code mutations:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run mutation testing
mutmut run

# View results
mutmut results

# Generate HTML report
mutmut html
open html/index.html
```

**Configuration:** `vortex/mutmut_config.py` targets `src/vortex/models/clip_ensemble.py`

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CLIP_UNIT_TEST_REAL` | `true` | Use real CLIP models on CPU for unit tests |
| `CLIP_CI_LATENCY_THRESHOLD` | `1.0` | P99 latency threshold in seconds (CI: set to `3.0`) |

### Test Categories

| Category | Count | GPU Required | Mock Ratio | Edge Cases |
|----------|-------|--------------|------------|------------|
| Unit tests | 15 | No | ~40% | N/A |
| Integration tests | 19 | 2 tests only | 0% | 55% |
| **Total** | **34** | **11% GPU-only** | **~20% overall** | **55%** |

**Edge case tests:**
- Adversarial prompt injection (SQL, XSS, instruction injection)
- FGSM perturbation attacks (ε=0.01)
- Numerical stability (NaN, Inf, denormal floats)
- OpenCLIP token truncation (real tokenizer, >77 tokens)
- Concurrent verification (4 threads, thread safety)
```

### Test Coverage

- **VRAM utilities** (test_memory.py): Mocked CUDA, format_bytes
- **VRAMMonitor** (test_pipeline.py): Soft/hard limit detection
- **ModelRegistry** (test_pipeline.py): Model loading, get_model interface
- **VortexPipeline** (test_pipeline.py): Buffer allocation, async orchestration
- **Async generation** (test_pipeline.py): Parallel tasks, cancellation, errors

## Configuration

See `config.yaml` for full options:

```yaml
device:
  name: "cuda:0"          # or "cpu" for testing
  allow_tf32: true

vram:
  soft_limit_gb: 11.0     # Warning threshold
  hard_limit_gb: 11.5     # Error threshold

models:
  precision:
    flux: "nf4"           # NF4 quantization
    liveportrait: "fp16"  # Half precision
    kokoro: "fp32"        # Full precision
    clip_b: "int8"        # 8-bit quantization
    clip_l: "int8"

buffers:
  actor: {height: 512, width: 512, channels: 3}
  video: {frames: 1080, height: 512, width: 512, channels: 3}
  audio: {sample_rate: 24000, duration_sec: 45}

pipeline:
  generation_timeout_sec: 20.0
  parallel_audio_actor: true
```

## Task Status

- **T014 (Core Pipeline):** ✅ Complete - Static VRAM manager, async orchestration
- **T015 (Flux-Schnell):** ✅ Complete - NF4 quantized image generation (~6GB VRAM)
- **T016 (LivePortrait):** Pending - Real LivePortrait integration
- **T017 (Kokoro TTS):** ✅ Complete - FP32 text-to-speech with voice/emotion control (~0.4GB VRAM)
- **T018 (Dual CLIP):** Pending - Real CLIP ensemble integration
- **T019 (VRAM Manager):** ✅ Complete (integrated in T014)
- **T020 (Slot Timing):** Pending - Slot scheduler and deadline management

### T017 Implementation Details

**Kokoro-82M TTS Integration** (See `T017_IMPLEMENTATION_SUMMARY.md` for full details)

Features:
- ✅ KokoroWrapper class with voice/emotion mapping
- ✅ 24kHz mono audio output, up to 45 seconds per slot
- ✅ 3+ character voices (rick_c137, morty, summer, jerry, beth)
- ✅ Speed control (0.8-1.2×)
- ✅ Emotion modulation (neutral, excited, sad, angry, manic)
- ✅ Pre-allocated buffer output (no VRAM fragmentation)
- ✅ Deterministic generation with seed control
- ✅ Comprehensive test suite (unit + integration)
- ✅ Benchmark and download scripts

Installation:
```bash
pip install kokoro soundfile
python scripts/download_kokoro.py --test-synthesis
```

Usage:
```python
from vortex.models.kokoro import load_kokoro

kokoro = load_kokoro(device="cuda:0")
audio = kokoro.synthesize(
    text="Wubba lubba dub dub!",
    voice_id="rick_c137",
    speed=1.1,
    emotion="manic"
)
# Returns: torch.Tensor (num_samples,) at 24kHz
```

Testing:
```bash
# Unit tests (no GPU required)
pytest tests/unit/test_kokoro.py -v

# Integration tests (requires GPU + kokoro package)
pytest tests/integration/test_kokoro_synthesis.py --gpu -v

# Benchmark latency
python benchmarks/kokoro_latency.py --iterations 50
```

## Performance Targets

| Metric | Target | Implementation Status |
|--------|--------|----------------------|
| Cold start (model loading) | <30s | ✅ Parallelized loaders |
| Actor generation (P99) | <12s | ✅ Flux-Schnell NF4 (T015) |
| Slot generation (P99) | <15s | 🔄 Pending (T016-T018) |
| VRAM usage | ≤11.5GB | ✅ Monitored with hard limit |
| Buffer allocation overhead | 0ms (pre-allocated) | ✅ Allocated at init |
| Async parallelization | Audio ‖ Actor | ✅ asyncio.gather |

## Known Limitations

1. **Partial Model Integration:** Flux-Schnell (T015) integrated. LivePortrait, Kokoro, CLIP pending (T016-T018).
2. **CPU Fallback:** Tests run on CPU. GPU required for production (RTX 3060+ 12GB).
3. **Integration Testing:** Full end-to-end benchmarks deferred to T020 (requires all models).

## Flux-Schnell Usage (T015)

### Basic Generation

```python
from vortex.models.flux import load_flux_schnell

# Load Flux model with NF4 quantization
flux = load_flux_schnell(device="cuda:0", quantization="nf4")

# Generate 512×512 actor image
image = flux.generate(
    prompt="a scientist in a laboratory",
    negative_prompt="blurry, low quality, watermark",
    num_inference_steps=4,
    guidance_scale=0.0,
    seed=42  # Optional: for deterministic output
)
```

### Pre-allocated Buffer (Recommended)

```python
import torch

# Pre-allocate buffer to prevent fragmentation
actor_buffer = torch.zeros(3, 512, 512, device="cuda:0", dtype=torch.float32)

# Generate directly to buffer (in-place write)
image = flux.generate(
    prompt="scientist",
    output=actor_buffer
)
```

### Download Model Weights

```bash
# One-time download (~12GB)
python vortex/scripts/download_flux.py

# Verify visual quality
python vortex/scripts/visual_check_flux.py \
    --prompt "scientist" \
    --output /tmp/test.png
```

### Benchmarking

```bash
# VRAM profiling (5.5-6.5GB target)
python vortex/benchmarks/flux_vram_profile.py

# Latency benchmark (P99 <12s target)
python vortex/benchmarks/flux_latency.py --iterations 50 --plot
```

## Next Steps

1. **T016:** Integrate LivePortrait with FP16 (3.5GB VRAM)
2. **T017:** Integrate Kokoro-82M TTS with FP32 (0.4GB VRAM)
3. **T018:** Integrate dual CLIP ensemble (0.9GB VRAM combined)
4. **T020:** Implement slot scheduler with deadline management
