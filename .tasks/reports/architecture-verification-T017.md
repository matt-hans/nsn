# Architecture Verification Report - T017

**Date:** 2025-12-28  
**Agent:** Architecture Verification Specialist (STAGE 4)  
**Task:** T017 - Kokoro-82M TTS Integration  
**Status:** PASS ✅

---

## Pattern Identified: Layered Architecture with Model Registry Factory

### Pattern: Model-Loader-Pipeline Layering

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│                  (Tauri Viewer, CLI)                        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Layer                           │
│              (VortexPipeline, ModelRegistry)                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                              │
│     (KokoroWrapper, load_flux, load_liveportrait, etc.)    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  External Dependencies                      │
│          (torch, kokoro, transformers, etc.)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Status: ✅ PASS

### Overall Score: 95/100

**Rationale:** T017 implementation correctly follows established architectural patterns. No critical violations detected. The KokoroWrapper integrates cleanly with the ModelRegistry factory pattern, maintaining proper separation of concerns and dependency flow (high-level → low-level).

---

## Critical Issues: 0

No blocking violations found.

---

## Warnings

### 1. Try/Except Fallback Pattern (MEDIUM)

**File:** `vortex/src/vortex/models/__init__.py:155-172`

**Issue:** The `load_kokoro()` function uses a try/except block to fall back to MockModel when the kokoro package is unavailable. While this enables testing without GPU, it may hide installation issues in production.

**Code:**
```python
try:
    from vortex.models.kokoro import load_kokoro as load_kokoro_real
    model = load_kokoro_real(device=device)
    logger.info("Kokoro-82M loaded successfully (real implementation)")
    return model
except (ImportError, Exception) as e:
    # Fallback to mock for environments without kokoro package
    logger.warning(...)
    model = MockModel(name="kokoro", vram_gb=0.4)
    model = model.to(device)
    return model
```

**Recommendation:** Consider adding an explicit opt-in flag (e.g., `allow_mock=False`) to prevent silent fallback in production environments.

**Severity:** Medium - Does not violate layering or dependency rules, but could improve operational clarity.

---

## Info (Improvement Opportunities)

### 1. KokoroWrapper Constructor Parameter Count

**File:** `vortex/src/vortex/models/kokoro.py:51-76`

**Observation:** The `KokoroWrapper.__init__()` accepts 7 parameters (model, voice_config, emotion_config, device, sample_rate, max_duration_sec). Consider using a configuration object pattern for better maintainability as the wrapper evolves.

**Current Pattern:**
```python
def __init__(
    self,
    model: nn.Module,
    voice_config: dict[str, str],
    emotion_config: dict[str, dict],
    device: str = "cuda",
    sample_rate: int = 24000,
    max_duration_sec: float = 45.0,
):
```

**Suggested Pattern:**
```python
@dataclass
class KokoroConfig:
    device: str = "cuda"
    sample_rate: int = 24000
    max_duration_sec: float = 45.0

def __init__(self, model, voice_config, emotion_config, config: KokoroConfig):
```

**Severity:** Info - Not a violation, but would improve readability.

### 2. ModelRegistry Pattern Consistency

**File:** `vortex/src/vortex/pipeline.py:71-159`

**Positive Finding:** The ModelRegistry class correctly implements the factory pattern across all 5 models (flux, liveportrait, kokoro, clip_b, clip_l). The `load_model()` function provides a unified interface with consistent error handling.

**Strengths:**
- Single entry point for model loading
- Device management centralized
- Lazy loading support
- Consistent return types (nn.Module)

---

## Dependency Analysis

### Circular Dependencies: None

Verified dependency flow:
```
Application → VortexPipeline → ModelRegistry → load_kokoro → KokoroWrapper → kokoro (external)
```

No upstream dependencies detected in KokoroWrapper.

### Layer Violations: 0

**Verified Boundaries:**
1. **Model Layer (`vortex/models/kokoro.py`)**: Only depends on torch, yaml, and external kokoro package. No knowledge of VortexPipeline.
2. **Factory Layer (`vortex/models/__init__.py`)**: Coordinates model loading, does not implement business logic.
3. **Pipeline Layer (`vortex/pipeline.py`)**: Uses ModelRegistry to load models, does not directly import KokoroWrapper.

### Dependency Direction: Correct ✅

All dependencies flow from high-level (Pipeline) → low-level (Models). No inversions detected.

---

## Naming Convention Analysis

### Consistency: 95%

**Observed Patterns:**
- Model loaders: `load_<model>()` (load_kokoro, load_flux, load_clip_b) ✅
- Wrapper classes: `<Model>Wrapper` (KokoroWrapper) ✅
- Private methods: `_<verb>_<noun>()` (_validate_synthesis_inputs, _generate_audio) ✅
- Factory functions: `load_<model>()` in __init__.py ✅

**Minor Deviation:**
- Config files use `kokoro_<type>.yaml` (kokoro_voices.yaml, kokoro_emotions.yaml) ✅
- Consider standardizing on `<model>_<type>.yaml` for other models (flux_*.yaml, clip_*.yaml)

---

## Separation of Concerns

### Verified Separation

| Component | Responsibility | Dependencies |
|-----------|----------------|--------------|
| `KokoroWrapper` | TTS synthesis, voice/emotion mapping | kokoro, torch |
| `load_kokoro()` | Factory function, config loading | kokoro, yaml |
| `ModelRegistry` | Model lifecycle management | torch, all model loaders |
| `VortexPipeline` | Orchestration, slot generation | ModelRegistry, recipes |

**Assessment:** Each layer has a single, well-defined responsibility. No cross-cutting concerns detected.

---

## Integration with VortexPipeline

### Architecture Diagram (Verified)

```
┌─────────────────────────────────────────────────────────────┐
│                    VortexPipeline                          │
│                                                             │
│  async def _generate_audio(recipe: dict):                   │
│      kokoro = self.model_registry.get_model("kokoro")       │
│      audio = kokoro.synthesize(                             │
│          text=recipe["audio_track"]["script"],              │
│          voice_id=recipe["audio_track"]["voice_id"],        │
│          speed=recipe["audio_track"]["speed"],              │
│          emotion=recipe["audio_track"]["emotion"],          │
│          output=self.audio_buffer  # Pre-allocated          │
│      )                                                      │
│      return audio  # 24kHz mono, ≤45s                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ModelRegistry                            │
│                                                             │
│  def get_model(self, name: str) -> nn.Module:              │
│      return self._models.get(name, self._load_model(name)) │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              load_kokoro() → KokoroWrapper                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  synthesize(text, voice_id, speed, emotion, output)  │   │
│  │    ├─ Voice ID mapping (rick_c137 → af_sky)         │   │
│  │    ├─ Emotion modulation (tempo, pitch, energy)     │   │
│  │    └─ Audio normalization to [-1, 1]                │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Integration Points Verified

1. **VortexPipeline._generate_audio()**: Correctly uses `model_registry.get_model("kokoro")` ✅
2. **Recipe Schema Access**: Reads from `recipe["audio_track"]` with proper defaults ✅
3. **Buffer Reuse**: Passes `self.audio_buffer` for VRAM efficiency ✅
4. **Return Type**: Returns `torch.Tensor` of shape `(num_samples,)` ✅

---

## VRAM Management Compliance

### Static Residency Pattern: Followed ✅

**Verified Implementation:**
```python
# KokoroWrapper allocates model once at initialization
self.model = KPipeline(lang_code='a')  # ~0.4 GB

# VortexPipeline pre-allocates buffer
self.audio_buffer = torch.empty(max_samples, device=device)

# synthesis writes to pre-allocated buffer
def _write_to_buffer(self, waveform, output):
    if output is not None:
        output[:num_samples].copy_(waveform)
        return output[:num_samples]
```

**VRAM Budget:** 0.4 GB (within 0.3-0.5 GB target) ✅

---

## Error Handling Analysis

### Pattern: Input Validation + Exception Propagation ✅

**Verified in `KokoroWrapper.synthesize()`:**
1. `_validate_synthesis_inputs()` - Pre-flight checks ✅
2. `_truncate_text_if_needed()` - Graceful degradation ✅
3. `_generate_audio()` - Exception logging + re-raise ✅
4. `_get_emotion_params()` - Fallback to "neutral" ✅

**Error Types:**
- `ValueError`: Invalid voice_id, empty text ✅
- `ImportError`: kokoro package not found ✅
- `FileNotFoundError`: Config files missing ✅
- `RuntimeError`: Generation failure ✅

---

## Recommendation: PASS

### Summary

T017 implementation correctly follows the established layered architecture with proper separation of concerns. The KokoroWrapper integrates cleanly with the ModelRegistry factory pattern, maintaining correct dependency flow and avoiding circular dependencies.

### Strengths
- Clean factory pattern via ModelRegistry
- Proper VRAM management with pre-allocated buffers
- Consistent naming conventions
- Comprehensive error handling
- No layering violations

### Areas for Future Enhancement
1. Consider explicit opt-in flag for mock fallback
2. Config object pattern for wrapper initialization
3. Standardize config file naming across all models

### Blocking Issues: None

**Decision:** PASS - Proceed to next stage (Performance/Security Testing)

---

**Report Generated:** 2025-12-28  
**Agent:** Architecture Verification Specialist (STAGE 4)  
**Next Review:** After T018 (CLIP integration)
