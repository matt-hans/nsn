# I2V Pipeline Restoration Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Restore I2V (Image-to-Video) pipeline with resolution alignment and domain gap fixes to eliminate blurring/swirling artifacts.

**Architecture:** Audio-first generation flow using Flux-Schnell for 720×480 keyframes with texture anchoring, fed to CogVideoX-5B-I2V for 3×5-second montage clips.

**Tech Stack:** Flux-Schnell (NF4), CogVideoX-5B-I2V (INT8), Bark TTS, diffusers, PyTorch

---

## Root Cause Analysis

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Blurring/Swirling | Resolution mismatch (512×512 → 720×480 resize) | Flux output 720×480 native |
| VAE artifacts | Domain gap (Flux "smooth" vs video "grainy") | Texture anchoring in prompts |
| Sync issues | Video generated before audio duration known | Audio-first architecture |
| Subject inconsistency | Varied seeds per keyframe | Fixed seed for all keyframes |

---

## Architecture

```
Showrunner (Ollama) → Script with 3 scenes + subject_visual
       ↓
Bark TTS (Audio First) → Audio track (defines exact duration)
       ↓
Flux-Schnell (NF4) → 3 keyframes @ 720x480 (texture anchored)
       ↓
   [Unload Flux]
       ↓
CogVideoX-5B-I2V (INT8) → 3 clips using motion prompts
       ↓
Concatenate → Final 15-second video with audio
```

### VRAM Budget (24GB available)

| Stage | Models Loaded | VRAM Used |
|-------|--------------|-----------|
| Script generation | Ollama (external) | 0 GB |
| Audio generation | Bark | ~1 GB |
| Keyframe generation | Flux NF4 | ~6 GB |
| Video generation | CogVideoX INT8 | ~10 GB |
| Verification | CLIP | ~0.6 GB |
| **Peak** | | **~10 GB** |

Sequential loading ensures ~10GB peak, leaving 14GB headroom.

---

## Task 1: Restore Flux Model with Resolution Fix

**Files:**
- Restore: `src/vortex/models/flux.py` (from git history)
- Modify: Resolution defaults

**Step 1: Restore flux.py from git history**

```bash
git show 589504a~1:vortex/src/vortex/models/flux.py > src/vortex/models/flux.py
```

**Step 2: Update FluxConfig resolution**

Change defaults from 512×512 to 720×480:

```python
@dataclass
class FluxConfig:
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    height: int = 480   # CHANGED: Was 512
    width: int = 720    # CHANGED: Was 512
    num_inference_steps: int = 4
    guidance_scale: float = 0.0
    max_sequence_length: int = 256
```

**Step 3: Add texture anchoring constant**

```python
# Domain gap fix: Add texture for CogVideoX VAE to "grip"
TEXTURE_ANCHOR_SUFFIX = ", film grain, detailed texture, 4k, high definition"
```

**Step 4: Update docstrings**

Update module docstring and FluxModel docstring to reflect 720×480 output.

**Step 5: Run Flux unit tests**

```bash
pytest tests/unit/test_flux.py -v
```

**Step 6: Commit**

```bash
git add src/vortex/models/flux.py
git commit -m "feat(flux): restore with 720x480 resolution for CogVideoX I2V"
```

---

## Task 2: Update CogVideoX to I2V Pipeline

**Files:**
- Modify: `src/vortex/models/cogvideox.py`

**Step 1: Update model ID and imports**

```python
# Change import
from diffusers import (
    CogVideoXImageToVideoPipeline,  # Was: CogVideoXPipeline
    PipelineQuantizationConfig,
    TorchAoConfig,
)

# Change model ID in dataclass
model_id: str = "THUDM/CogVideoX-5b-I2V"  # Was: CogVideoX-5b
```

**Step 2: Add _to_pil_image method**

```python
def _to_pil_image(self, tensor: torch.Tensor) -> "Image.Image":
    """Convert tensor [C, H, W] or [B, C, H, W] to PIL Image."""
    from PIL import Image

    if tensor.dim() == 4:
        tensor = tensor[0]

    img_np = (tensor.permute(1, 2, 0) * 255).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(img_np, mode="RGB")
```

**Step 3: Update generate_chunk signature**

Add `image` parameter as first required argument:

```python
async def generate_chunk(
    self,
    image: torch.Tensor | "Image.Image",  # NEW: Required keyframe
    prompt: str,
    config: VideoGenerationConfig | None = None,
    seed: int | None = None,
) -> torch.Tensor:
```

**Step 4: Update _generate_sync to pass image**

```python
def _generate_sync(
    self,
    image: "Image.Image",
    prompt: str,
    config: VideoGenerationConfig,
    generator: torch.Generator | None,
) -> list:
    result = self._pipe(
        image=image,  # NEW: Pass keyframe
        prompt=prompt,
        # ... rest unchanged
    )
    return result.frames[0]
```

**Step 5: Update generate_montage signature**

Add `keyframes` parameter:

```python
async def generate_montage(
    self,
    keyframes: list[torch.Tensor],  # NEW: Required keyframes
    prompts: list[str],
    config: VideoGenerationConfig | None = None,
    seed: int | None = None,
    trim_frames: int = 40,
    progress_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
```

**Step 6: Update load() to use I2V pipeline**

```python
self._pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    self.model_id,
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    cache_dir=self.cache_dir,
)
```

**Step 7: Remove or deprecate generate_chain**

Mark as deprecated (already done) or remove entirely.

**Step 8: Update module docstring**

Change "Text-to-Video" to "Image-to-Video" throughout.

**Step 9: Run CogVideoX unit tests**

```bash
pytest tests/unit/test_cogvideox.py -v
```

**Step 10: Commit**

```bash
git add src/vortex/models/cogvideox.py
git commit -m "feat(cogvideox): restore I2V pipeline with image input"
```

---

## Task 3: Update Renderer for Audio-First Flow

**Files:**
- Modify: `src/vortex/renderers/default/renderer.py`

**Step 1: Add Flux import**

```python
from vortex.models.flux import FluxModel, FluxConfig, TEXTURE_ANCHOR_SUFFIX
```

**Step 2: Update style constants**

```python
VISUAL_STYLE_PROMPT = (
    "2D cel-shaded cartoon, flat colors, expressive linework, "
    "detailed texture, film grain, 4k, high definition"
)

MOTION_STYLE_SUFFIX = ", smooth animation, continuous motion, fluid movement"
```

**Step 3: Update buffer allocation to 720×480**

```python
def _allocate_buffers(self, config: dict[str, Any]) -> None:
    # ...
    self._actor_buffer = torch.zeros(
        1,
        actor_cfg.get("channels", 3),
        actor_cfg.get("height", 480),   # Was: 512
        actor_cfg.get("width", 720),    # Was: 512
        device=self._device,
        dtype=torch.float32,
    )
```

**Step 4: Add _generate_keyframes method**

```python
async def _generate_keyframes(
    self,
    script: Script,
    seed: int,
) -> list[torch.Tensor]:
    """Generate 3 keyframes with texture anchoring and fixed seed."""
    flux = self._model_registry.get_flux()
    keyframes = []

    for i, scene in enumerate(script.storyboard):
        # FIXED SEED for subject consistency
        scene_seed = seed  # NOT seed + i

        visual_prompt = f"{script.subject_visual}, {scene}, {VISUAL_STYLE_PROMPT}"

        keyframe = flux.generate(
            prompt=visual_prompt,
            seed=scene_seed,
            output=self._actor_buffer,
        )

        keyframes.append(keyframe.clone())

    return keyframes
```

**Step 5: Add _unload_flux helper**

```python
def _unload_flux(self) -> None:
    """Unload Flux to free VRAM before CogVideoX."""
    flux = self._model_registry.get_flux()
    if flux.is_loaded:
        flux.unload()
        logger.info("Flux unloaded to free VRAM for CogVideoX")
```

**Step 6: Update _generate_video for I2V**

```python
async def _generate_video(
    self,
    script: Script,
    keyframes: list[torch.Tensor],
    frames_per_scene: int,
    seed: int,
) -> torch.Tensor:
    """Generate video from keyframes using I2V."""
    cogvideox = self._model_registry.get_cogvideox()

    motion_prompts = [
        f"{scene}{MOTION_STYLE_SUFFIX}"
        for scene in script.storyboard
    ]

    config = VideoGenerationConfig(
        num_frames=49,
        guidance_scale=3.5,
        use_dynamic_cfg=True,
        fps=8,
    )

    trim_frames = min(frames_per_scene, 40)

    video = await cogvideox.generate_montage(
        keyframes=keyframes,
        prompts=motion_prompts,
        config=config,
        seed=seed,
        trim_frames=trim_frames,
    )

    return video
```

**Step 7: Update render() for audio-first flow**

Reorder to: Script → Audio → Keyframes → Unload Flux → Video → Verify

**Step 8: Run renderer unit tests**

```bash
pytest tests/unit/test_renderer_montage.py -v
```

**Step 9: Commit**

```bash
git add src/vortex/renderers/default/renderer.py
git commit -m "feat(renderer): audio-first flow with I2V keyframe generation"
```

---

## Task 4: Update Manifest

**Files:**
- Modify: `src/vortex/renderers/default/manifest.yaml`

**Step 1: Add flux-schnell-nf4 to dependencies**

```yaml
model_dependencies:
  - flux-schnell-nf4
  - cogvideox-5b-i2v-int8
  - bark-tts
  - clip-vit-b-32-fp16
  - clip-vit-l-14-fp16
```

**Step 2: Update description**

```yaml
description: "NSN Lane 0 I2V Montage renderer using Showrunner for scripts, Bark for TTS, Flux-Schnell for 720x480 keyframes, CogVideoX I2V for video, and dual CLIP for verification."
```

**Step 3: Commit**

```bash
git add src/vortex/renderers/default/manifest.yaml
git commit -m "docs(manifest): update for I2V pipeline with Flux keyframes"
```

---

## Task 5: Update Unit Tests

**Files:**
- Restore/Update: `tests/unit/test_flux.py`
- Update: `tests/unit/test_cogvideox.py`
- Update: `tests/unit/test_renderer_montage.py`

**Step 1: Restore test_flux.py from git history**

```bash
git show 589504a~1:vortex/tests/unit/test_flux.py > tests/unit/test_flux.py
```

**Step 2: Update test_flux.py assertions**

Change all 512 references to 720×480:
- `assert config.height == 480`
- `assert config.width == 720`
- `assert result.shape == (3, 480, 720)`

**Step 3: Update test_cogvideox.py for I2V**

- Add test for model ID containing "I2V"
- Add test that generate_chunk requires image parameter
- Add test that generate_montage requires keyframes parameter
- Update mock signatures

**Step 4: Update test_renderer_montage.py**

- Update buffer shape assertions to (1, 3, 480, 720)
- Add test for fixed seed in keyframe generation
- Update keyframe shape assertions to (3, 480, 720)

**Step 5: Run all unit tests**

```bash
pytest tests/unit/ -v
```

**Step 6: Commit**

```bash
git add tests/unit/
git commit -m "test: update unit tests for I2V pipeline with 720x480 resolution"
```

---

## Task 6: Run E2E Verification

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```

**Step 2: Run E2E generation test**

```bash
python scripts/e2e_narrative_test.py --theme "bizarre infomercial" --duration 15 --verbose
```

**Step 3: Verify metrics**

Expected results:
- `avg_motion > 0.05` (was 0.0861 in previous I2V)
- No blurring/swirling in visual inspection
- Audio-video sync (audio duration ≈ video duration)

**Step 4: Final commit if all passes**

```bash
git add -A
git commit -m "feat(pipeline): complete I2V restoration with resolution and domain gap fixes"
```

---

## Verification Checklist

- [ ] Flux resolution: 720×480
- [ ] Flux has TEXTURE_ANCHOR_SUFFIX constant
- [ ] CogVideoX model ID contains "I2V"
- [ ] CogVideoX generate_chunk accepts image parameter
- [ ] CogVideoX generate_montage accepts keyframes parameter
- [ ] Renderer buffer: (1, 3, 480, 720)
- [ ] Renderer uses fixed seed for keyframes
- [ ] Renderer uses audio-first flow
- [ ] Manifest includes flux-schnell-nf4
- [ ] Unit tests pass
- [ ] E2E test avg_motion > 0.05
- [ ] Visual inspection: no swirling artifacts

---

## Prompt Strategy Summary

| Stage | Prompt Components | Purpose |
|-------|------------------|---------|
| Flux (visual) | `{subject_visual}, {scene}, {VISUAL_STYLE_PROMPT}` | High-detail keyframe with texture anchoring |
| CogVideoX (motion) | `{scene}{MOTION_STYLE_SUFFIX}` | Action-focused prompt for fluid animation |

**VISUAL_STYLE_PROMPT:** "2D cel-shaded cartoon, flat colors, expressive linework, detailed texture, film grain, 4k, high definition"

**MOTION_STYLE_SUFFIX:** ", smooth animation, continuous motion, fluid movement"

---

## Rollback Plan

If I2V still exhibits artifacts after these fixes:

1. Revert to T2V: `git revert HEAD~N` (where N = number of I2V commits)
2. The T2V implementation is preserved in commit `589504a`
3. Alternative: Try CogVideoX-2B-I2V (smaller model, different training data)
