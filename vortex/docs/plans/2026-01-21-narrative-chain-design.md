# Vortex v2: Narrative Chain Pipeline Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Vortex from a distorted 2-second clip generator into a 10-15 second "Interdimensional Cable" episode generator with working audio and fluid animation.

**Architecture:** LLM-driven script generation → Kokoro TTS → Flux keyframe → CogVideoX autoregressive video chaining → CLIP verification. Fully deprecates ComfyUI, LivePortrait, F5-TTS, and audio-gated driver.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, diffusers, transformers, Ollama (external), Kokoro TTS, Flux-Schnell (NF4), CogVideoX-5B (INT8), CLIP ViT-B-32/ViT-L-14

---

## 1. Problem Statement

The current Vortex pipeline produces:
- **Warped geometry** instead of animation (LivePortrait misapplied to non-face content)
- **Silent/tone audio** instead of speech (TTS pipeline broken)
- **2-second clips** instead of 10-15 second episodes

Root causes:
1. LivePortrait is a face-reenactment model, not a general animation engine
2. ComfyUI integration adds complexity without solving the core problem
3. Audio pipeline has cascading failures (F5-TTS unavailable, Kokoro not triggered)

---

## 2. Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vortex v2: Narrative Chain                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Showrunner  │───>│    Kokoro    │───>│    Flux      │       │
│  │   (Ollama)   │    │     TTS      │    │  Schnell     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         │ Script            │ Audio             │ Keyframe       │
│         ▼                   ▼                   ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CogVideoX-5B (INT8)                        │    │
│  │                                                          │    │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐  │    │
│  │  │ Chunk 1 │──>│ Chunk 2 │──>│ Chunk 3 │──>│ Chunk 4 │  │    │
│  │  │  (4s)   │   │  (4s)   │   │  (4s)   │   │  (4s)   │  │    │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘  │    │
│  │       │             │             │             │        │    │
│  │       └─────────────┴─────────────┴─────────────┘        │    │
│  │                  Last Frame → Next Input                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│                    ┌──────────────┐                              │
│                    │  Dual CLIP   │                              │
│                    │  Verification│                              │
│                    └──────────────┘                              │
│                              │                                   │
│                              ▼                                   │
│                    [10-15s Animated Video + Audio]              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Components Being Deleted

| File | Reason |
|------|--------|
| `vortex/models/liveportrait.py` | Causes warped geometry on non-face content |
| `vortex/models/liveportrait_features.py` | Wav2Vec2 processor for LivePortrait |
| `vortex/models/audio_driver.py` | Audio-gated jaw animation for LivePortrait |
| `vortex/engine/client.py` | ComfyUI WebSocket client |
| `vortex/engine/payload.py` | ComfyUI workflow builder |
| `vortex/templates/*.json` | All ComfyUI workflow templates |
| F5-TTS fallback in `core/audio.py` | Unnecessary complexity |

---

## 4. Components Being Created

### 4.1 Showrunner (LLM Script Generator)

**File:** `vortex/models/showrunner.py`

**Purpose:** Generate surreal joke scripts via Ollama

**Interface:**
```python
class Showrunner:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        ...

    async def generate_script(
        self,
        theme: str,
        tone: str = "absurd"
    ) -> Script:
        """
        Returns:
            Script(
                setup="A man made of ham is running for mayor.",
                punchline="He promises to cure world hunger.",
                visual_prompt="A ham-man giving a speech at a podium, surreal, cartoon"
            )
        """
```

**Ollama Prompt Template:**
```
You are a writer for "Interdimensional Cable", an absurdist comedy show.

Write a SHORT surreal commercial/scene about: {theme}
Tone: {tone}

Format your response as JSON:
{
  "setup": "The premise (1 sentence)",
  "punchline": "The absurd conclusion (1 sentence)",
  "visual_prompt": "Scene description for image generation"
}

Keep it weird. Keep it short. No more than 15 seconds when spoken aloud.
```

### 4.2 CogVideoX Wrapper

**File:** `vortex/models/cogvideox.py`

**Purpose:** Generate 4-second video chunks with autoregressive chaining

**Interface:**
```python
class CogVideoXModel:
    def __init__(
        self,
        model_id: str = "THUDM/CogVideoX-5b",
        device: str = "cuda",
        dtype: torch.dtype = torch.int8
    ):
        ...

    def load(self) -> None:
        """Load model with INT8 quantization"""

    def unload(self) -> None:
        """Free VRAM"""

    async def generate_chunk(
        self,
        image: torch.Tensor,      # [3, H, W] input keyframe
        prompt: str,
        num_frames: int = 49,     # CogVideoX default
        guidance_scale: float = 6.0
    ) -> torch.Tensor:            # [T, 3, H, W] video frames
        """Generate one 4-second chunk from input image"""

    async def generate_chain(
        self,
        keyframe: torch.Tensor,
        prompt: str,
        target_duration: float,   # seconds
        chunk_duration: float = 4.0
    ) -> torch.Tensor:            # [T, 3, H, W] full video
        """Generate multiple chunks via autoregressive chaining"""
```

**VRAM Strategy:**
- Use `diffusers` with `enable_model_cpu_offload()`
- INT8 quantization via `torchao` or `bitsandbytes`
- Peak VRAM: ~10-11GB during generation

---

## 5. VRAM Budget

**Target Hardware:** RTX 3060 12GB

**Sequential Execution Flow:**

| Phase | Model | VRAM | Duration |
|-------|-------|------|----------|
| 1 | Ollama LLM | 0 (external) | ~2s |
| 2 | Kokoro TTS | ~0.4GB | ~3s |
| 3 | Flux-Schnell | ~6GB | ~5s |
| - | Unload Flux | - | - |
| 4 | CogVideoX-5B INT8 | ~10-11GB | ~60-90s |
| - | Unload CogVideoX | - | - |
| 5 | CLIP Ensemble | ~0.6GB | ~2s |

**Peak VRAM:** ~11GB (CogVideoX phase)
**Total Time:** ~90-120s for 12-second episode

---

## 6. New Recipe Schema

```python
{
  "slot_params": {
    "slot_id": int,           # Unique ID
    "seed": int,              # Deterministic seed
    "target_duration": float, # 10-15 seconds
    "fps": int                # Default 24
  },

  "narrative": {
    "theme": str,             # e.g., "surreal commercial", "alien talk show"
    "tone": str,              # e.g., "absurd", "deadpan", "manic"
    "auto_script": bool,      # True = LLM generates
    "script": {               # Optional if auto_script=True
      "setup": str,
      "punchline": str,
      "visual_prompt": str
    }
  },

  "audio": {
    "voice_id": str,          # Kokoro voice
    "speed": float,           # Speech speed
    "bgm": str | None,        # Background music
    "bgm_volume": float       # 0.0-1.0
  },

  "video": {
    "style_prompt": str,      # Style additions
    "negative_prompt": str,   # Avoid list
    "chunk_duration": float,  # Default 4.0
    "guidance_scale": float   # Default 6.0
  },

  "quality": {
    "clip_threshold": float,  # Default 0.70
    "max_retries": int        # Default 3
  }
}
```

---

## 7. File Structure After Refactoring

```
vortex/src/vortex/
├── models/
│   ├── flux.py              # KEEP (keyframe generation)
│   ├── kokoro.py            # KEEP (TTS)
│   ├── cogvideox.py         # NEW (video generation)
│   ├── showrunner.py        # NEW (LLM script)
│   └── clip_ensemble.py     # KEEP (verification)
├── core/
│   ├── audio.py             # SIMPLIFIED (Kokoro only)
│   └── mixer.py             # KEEP (audio mixing)
├── renderers/default/
│   ├── renderer.py          # REWRITTEN (new pipeline)
│   ├── recipe_schema.py     # UPDATED (new schema)
│   └── manifest.yaml        # UPDATED (new deps)
├── utils/
│   ├── memory.py            # KEEP (VRAM monitoring)
│   ├── offloader.py         # KEEP (model offloading)
│   └── render_output.py     # KEEP (save results)
├── pipeline.py              # SIMPLIFIED
├── orchestrator.py          # SIMPLIFIED (no ComfyUI)
└── config.yaml              # UPDATED
```

---

## 8. Error Handling

| Error | Strategy |
|-------|----------|
| Ollama unavailable | Fall back to hardcoded templates |
| Kokoro fails | Retry once, then fail |
| Flux OOM | Enable CPU offload, retry |
| CogVideoX OOM | Reduce guidance, retry |
| CLIP below threshold | Regenerate with new seed (up to max_retries) |

---

## 9. Testing Strategy

**Unit Tests:**
- `test_showrunner.py` - LLM integration, response parsing
- `test_cogvideox.py` - Model loading, quantization, chaining
- `test_audio_kokoro.py` - TTS synthesis

**Integration Tests:**
- `test_script_to_audio.py` - LLM → Kokoro
- `test_image_to_video_chain.py` - Flux → CogVideoX
- `test_full_pipeline.py` - Complete recipe → video

**E2E Verification:**
```bash
python scripts/e2e_narrative_test.py --theme "bizarre infomercial" --duration 12
```

**Manual Checklist:**
- [ ] Video shows actual motion (frames differ significantly)
- [ ] Audio has audible speech
- [ ] Duration matches target (±2s)
- [ ] Style matches "Interdimensional Cable" aesthetic
- [ ] No warping/distortion artifacts

---

## 10. Implementation Phases

### Phase 1: Cleanup & Foundation
1. Delete deprecated files (LivePortrait, ComfyUI, audio driver)
2. Simplify audio.py to Kokoro-only
3. Update config.yaml and manifest.yaml

### Phase 2: Showrunner Integration
1. Create showrunner.py with Ollama integration
2. Add fallback template system
3. Write unit tests

### Phase 3: CogVideoX Integration
1. Create cogvideox.py with INT8 quantization
2. Implement autoregressive chaining
3. Test VRAM behavior on 12GB card

### Phase 4: Pipeline Assembly
1. Rewrite renderer.py with new flow
2. Update recipe schema
3. Implement error handling and recovery

### Phase 5: Verification & Polish
1. Run full E2E tests
2. Tune parameters (guidance, chunk duration)
3. Verify CLIP thresholds work correctly

---

## 11. Success Criteria

- [ ] 10-15 second episodes with fluid animation
- [ ] Audible TTS speech (not silence)
- [ ] No warped geometry artifacts
- [ ] Runs on RTX 3060 12GB without OOM
- [ ] CLIP verification passing at 0.70 threshold
- [ ] "Interdimensional Cable" aesthetic achieved

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CogVideoX too slow | Pre-generate keyframes, batch processing |
| INT8 quality loss | Test FP16 comparison, adjust guidance |
| Ollama latency | Async call, cache common themes |
| Chaining artifacts | Overlap frames, smooth transitions |

---

*Design validated: 2026-01-21*
*Status: Ready for implementation*
