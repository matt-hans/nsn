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
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Tasks

- T014: Core pipeline orchestration
- T015: Flux-Schnell integration
- T016: LivePortrait integration
- T017: Kokoro TTS integration
- T018: Dual CLIP ensemble
- T019: VRAM manager
- T020: Slot timing orchestration
