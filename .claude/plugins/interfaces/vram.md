# VRAM Budget Constraints

Plugin interface for vram-oracle validation.

---

## ICN Fixed Allocations

| Model | Precision | VRAM | Residency |
|-------|-----------|------|-----------|
| Flux-Schnell | NF4 | 6.0 GB | Static |
| LivePortrait | FP16 | 3.5 GB | Static |
| Kokoro-82M | FP32 | 0.4 GB | Static |
| CLIP Ensemble | INT8 | 0.9 GB | Static |
| **TOTAL** | - | **10.8 GB** | - |
| **Buffer** | - | **1.2 GB** | Dynamic |

**Target Hardware**: RTX 3060 12GB (minimum)

---

## L0 Blocking Constraints

- Total model VRAM MUST NOT exceed 10.8 GB
- Models MUST remain GPU-resident (no dynamic loading/unloading)
- No `.to(device)` copies of already GPU-resident tensors

## L1 Critical Constraints

- All model loading MUST include `torch.cuda.memory_allocated()` monitoring
- Inference MUST use `torch.no_grad()` context
- Static residency pattern required (all models load at startup)

## L2 Mandatory Checks

- VRAM budget validated with profiling output
- OOM prevention test must exist (100 consecutive slots)
- Peak usage must remain < 11.5 GB

## L3 Standard Guidance

- Stage-by-stage memory breakdown documented
- Memory profiling hooks at pipeline stages

---

## Validation Commands

| Phase | Trigger | Check |
|-------|---------|-------|
| Planning | Before L1 plan creation | Verify new models fit within 1.2GB buffer |
| Implementation | Write to `vortex/**/*.py` | Check for memory leak patterns |
| Final | Before L4 completion | 5-slot benchmark, peak < 11.5GB |

## Anti-Patterns to Detect

- Tensor accumulation in loops without `.detach()`
- Gradient retention during inference
- Dynamic model loading between slots
- Unbounded cache growth

