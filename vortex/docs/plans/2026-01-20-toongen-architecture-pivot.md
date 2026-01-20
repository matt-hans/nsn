# ToonGen Architecture Pivot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Vortex Legacy (custom PyTorch facial animation) with ToonGen (ComfyUI-based workflow orchestration) to generate wacky cartoon clips in the style of "Interdimensional Cable."

**Architecture:** Python Orchestrator generates audio (F5-TTS/Kokoro + FFmpeg), dispatches visual generation jobs to headless ComfyUI (Flux → LivePortrait → AnimateDiff), and returns MP4. VRAM managed via sequential execution with explicit unloading. Two Docker containers: `vortex-orchestrator` and `comfyui-server`.

**Tech Stack:** Python 3.11, F5-TTS, Kokoro-82M, FFmpeg, ComfyUI, Flux.1-Schnell (NF4), LivePortrait, AnimateDiff (SD1.5), Docker, WebSocket

---

## Prerequisites

Before starting, ensure you have:
- RTX 3060 12GB (or equivalent) with CUDA installed
- Docker and docker-compose installed
- Python 3.11+ with pip
- ~50GB free disk space for models

---

## Phase 0: Legacy Archive

Archive the existing Vortex implementation before the purge.

### Task 0.1: Create Legacy Archive Branch

**Files:**
- None (git operations only)

**Step 1: Create archive branch from current state**

```bash
cd /home/matt/nsn/vortex
git checkout main
git pull origin main
git checkout -b legacy/vortex-v1
git push -u origin legacy/vortex-v1
```

**Step 2: Return to main branch**

```bash
git checkout main
```

**Step 3: Verify archive exists**

```bash
git branch -a | grep legacy
```

Expected: `legacy/vortex-v1` appears in list

**Step 4: Commit (none needed - branch operation only)**

---

### Task 0.2: Delete Legacy Model Files

**Files:**
- Delete: `vortex/src/vortex/models/audio_driver.py`
- Delete: `vortex/src/vortex/models/liveportrait.py`
- Delete: `vortex/src/vortex/models/liveportrait_features.py`
- Delete: `vortex/src/vortex/models/flux.py`
- Delete: `vortex/src/vortex/models/clip_ensemble.py`

**Step 1: Remove legacy model files**

```bash
cd /home/matt/nsn/vortex
rm -f src/vortex/models/audio_driver.py
rm -f src/vortex/models/liveportrait.py
rm -f src/vortex/models/liveportrait_features.py
rm -f src/vortex/models/flux.py
rm -f src/vortex/models/clip_ensemble.py
```

**Step 2: Remove legacy test files**

```bash
rm -f tests/test_audio_driver.py
rm -f tests/test_liveportrait.py
rm -f tests/test_flux.py
```

**Step 3: Update models __init__.py**

Replace `vortex/src/vortex/models/__init__.py` with:

```python
"""ToonGen model wrappers.

This package provides wrappers for:
- F5-TTS: Zero-shot voice cloning
- Kokoro: Fallback TTS engine
"""

from vortex.models.kokoro import KokoroWrapper

__all__ = ["KokoroWrapper"]
```

**Step 4: Commit the purge**

```bash
git add -A
git commit -m "refactor: remove legacy Vortex models (archived in legacy/vortex-v1)

BREAKING CHANGE: Removes audio_driver, liveportrait, flux, clip_ensemble
These are replaced by ComfyUI workflow orchestration in ToonGen architecture"
```

---

### Task 0.3: Delete Legacy Utilities

**Files:**
- Delete: `vortex/src/vortex/utils/lipsync.py`
- Delete: `vortex/src/vortex/utils/face_landmarks.py`
- Delete: `vortex/src/vortex/utils/offloader.py`
- Keep: `vortex/src/vortex/utils/memory.py` (still useful for VRAM monitoring)

**Step 1: Remove legacy utility files**

```bash
cd /home/matt/nsn/vortex
rm -f src/vortex/utils/lipsync.py
rm -f src/vortex/utils/face_landmarks.py
rm -f src/vortex/utils/offloader.py
```

**Step 2: Update utils __init__.py**

Replace `vortex/src/vortex/utils/__init__.py` with:

```python
"""ToonGen utility modules."""

__all__ = []
```

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: remove legacy utility modules (lipsync, face_landmarks, offloader)"
```

---

## Phase 1: Manual Proof (ComfyUI Workflow Validation)

Validate the visual stack works on RTX 3060 before writing any orchestration code.

### Task 1.1: Install ComfyUI Locally

**Files:**
- None (local installation)

**Step 1: Clone ComfyUI**

```bash
cd /home/matt/nsn
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

**Step 2: Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Step 3: Verify ComfyUI starts**

```bash
python main.py --listen 0.0.0.0 --port 8188
```

Expected: Server starts, accessible at `http://localhost:8188`

**Step 4: Stop server (Ctrl+C) and continue**

---

### Task 1.2: Install Required Custom Nodes

**Files:**
- None (ComfyUI Manager operations)

**Step 1: Install ComfyUI Manager**

```bash
cd /home/matt/nsn/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

**Step 2: Start ComfyUI and open browser**

```bash
cd /home/matt/nsn/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

**Step 3: Install custom nodes via Manager UI**

Open `http://localhost:8188`, click "Manager" button, install:
- `ComfyUI-LivePortraitKJ`
- `ComfyUI-AnimateDiff-Evolved`
- `ComfyUI-VideoHelperSuite`
- `ComfyUI_IPAdapter_plus` (optional)

**Step 4: Restart ComfyUI after installations**

---

### Task 1.3: Download Required Models

**Files:**
- Create: `vortex/scripts/download_models.sh`

**Step 1: Create model download script**

Create `vortex/scripts/download_models.sh`:

```bash
#!/bin/bash
# ToonGen Model Download Script
# Downloads all required models for the ComfyUI workflow

set -e

COMFY_DIR="${COMFY_DIR:-/home/matt/nsn/ComfyUI}"
MODELS_DIR="$COMFY_DIR/models"

echo "=== ToonGen Model Downloader ==="
echo "Target: $MODELS_DIR"

# Create directories
mkdir -p "$MODELS_DIR/checkpoints"
mkdir -p "$MODELS_DIR/liveportrait"
mkdir -p "$MODELS_DIR/animatediff_models"
mkdir -p "$MODELS_DIR/clip"

# Flux.1-Schnell NF4 (Quantized for 12GB VRAM)
echo "Downloading Flux.1-Schnell NF4..."
if [ ! -f "$MODELS_DIR/checkpoints/flux1-schnell-bnb-nf4.safetensors" ]; then
    wget -O "$MODELS_DIR/checkpoints/flux1-schnell-bnb-nf4.safetensors" \
        "https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-schnell-bnb-nf4.safetensors"
fi

# ToonYou SD1.5 Checkpoint (Cartoon Style)
echo "Downloading ToonYou Beta6..."
if [ ! -f "$MODELS_DIR/checkpoints/toonyou_beta6.safetensors" ]; then
    wget -O "$MODELS_DIR/checkpoints/toonyou_beta6.safetensors" \
        "https://civitai.com/api/download/models/125771"
fi

# AnimateDiff Motion Module
echo "Downloading AnimateDiff v3..."
if [ ! -f "$MODELS_DIR/animatediff_models/mm_sd_v15_v3.ckpt" ]; then
    wget -O "$MODELS_DIR/animatediff_models/mm_sd_v15_v3.ckpt" \
        "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v3.ckpt"
fi

# LivePortrait weights (downloaded by node on first use, but we can pre-fetch)
echo "LivePortrait weights will be downloaded on first use by the ComfyUI node"

echo "=== Download Complete ==="
echo "Models installed to: $MODELS_DIR"
```

**Step 2: Make executable and run**

```bash
chmod +x /home/matt/nsn/vortex/scripts/download_models.sh
/home/matt/nsn/vortex/scripts/download_models.sh
```

**Step 3: Commit script**

```bash
cd /home/matt/nsn/vortex
git add scripts/download_models.sh
git commit -m "feat: add model download script for ToonGen"
```

---

### Task 1.4: Build and Test ComfyUI Workflow

**Files:**
- Create: `vortex/templates/cartoon_workflow.json` (exported from ComfyUI)

**Step 1: Build the workflow in ComfyUI GUI**

Open `http://localhost:8188` and construct this node graph:

**Source Image (Flux):**
1. Add `CheckpointLoaderNF4` → Load `flux1-schnell-bnb-nf4.safetensors`
2. Add `EmptyLatentImage` → Width: 1024, Height: 576, Batch: 1
3. Add `CLIPTextEncode` (Positive) → Connect to checkpoint CLIP output
4. Add `KSampler` → Steps: 4, Scheduler: simple, Denoise: 1.0
5. Add `VAEDecode` → Connect to KSampler output

**Audio Input:**
6. Add `LoadAudio` → Will be parameterized

**Motion (LivePortrait):**
7. Add `LivePortraitLoadCropper`
8. Add `LivePortraitLoadModels` → Load human or animal model
9. Add `LivePortraitProcess`:
   - Image: From VAEDecode
   - Audio: From LoadAudio
   - `lip_ratio`: 1.2
   - `expression_scale`: 1.2
   - `crop_factor`: 1.7

**Style (AnimateDiff):**
10. Add `CheckpointLoaderSimple` → Load `toonyou_beta6.safetensors`
11. Add `AnimateDiffLoaderGen1` → Load `mm_sd_v15_v3.ckpt`
12. Add `ADE_AnimateDiffUniformContextOptions`:
    - Context Length: 16
    - Context Overlap: 4
    - Closed Loop: False
13. Add `VAEEncode` → Encode LivePortrait output
14. Add `KSampler`:
    - Denoise: 0.35
    - Steps: 20
15. Add `VAEDecode`

**Output:**
16. Add `VHS_VideoCombine`:
    - Images: From AnimateDiff VAEDecode
    - Audio: From LoadAudio
    - Format: video/h264-mp4

**Step 2: Test with sample inputs**

- Set a test prompt: "Medium shot, cartoon style, a nervous scientist in a lab coat, talking to camera"
- Load a test WAV file (any speech audio)
- Click "Queue Prompt"

**Step 3: Verify VRAM usage**

Monitor with: `watch -n 1 nvidia-smi`

Expected: Peak VRAM < 11GB (safe margin for 12GB card)

**Step 4: Export workflow as API format**

- Settings → Enable Dev Mode Options
- Click "Save (API Format)"
- Save as `cartoon_workflow.json`

**Step 5: Move to vortex templates**

```bash
mkdir -p /home/matt/nsn/vortex/templates
mv ~/Downloads/cartoon_workflow.json /home/matt/nsn/vortex/templates/
```

**Step 6: Identify node IDs for parameterization**

Open `templates/cartoon_workflow.json` and note:
- Prompt node ID (CLIPTextEncode): e.g., `"6"`
- Audio node ID (LoadAudio): e.g., `"40"`
- Seed node ID (KSampler): e.g., `"10"`

Document these in a comment at the top of the JSON or in a separate config.

**Step 7: Commit workflow template**

```bash
cd /home/matt/nsn/vortex
git add templates/cartoon_workflow.json
git commit -m "feat: add validated ComfyUI workflow template for ToonGen"
```

---

## Phase 2: Audio Engine

Build the audio generation pipeline (F5-TTS + Kokoro + FFmpeg mixing).

### Task 2.1: Create Audio Engine Module Structure

**Files:**
- Create: `vortex/src/vortex/core/__init__.py`
- Create: `vortex/src/vortex/core/audio.py`

**Step 1: Create core package**

```bash
mkdir -p /home/matt/nsn/vortex/src/vortex/core
touch /home/matt/nsn/vortex/src/vortex/core/__init__.py
```

**Step 2: Write core __init__.py**

Create `vortex/src/vortex/core/__init__.py`:

```python
"""ToonGen core modules.

- audio: Voice synthesis (F5-TTS, Kokoro) with graceful degradation
- mixer: Audio compositing (FFmpeg-based BGM/SFX mixing)
"""

from vortex.core.audio import AudioEngine
from vortex.core.mixer import AudioCompositor

__all__ = ["AudioEngine", "AudioCompositor"]
```

**Step 3: Commit structure**

```bash
cd /home/matt/nsn/vortex
git add src/vortex/core/
git commit -m "feat: create core package structure for ToonGen audio"
```

---

### Task 2.2: Implement AudioEngine with Graceful Degradation

**Files:**
- Create: `vortex/src/vortex/core/audio.py`
- Create: `vortex/tests/unit/test_audio_engine.py`

**Step 1: Write the failing test**

Create `vortex/tests/unit/test_audio_engine.py`:

```python
"""Unit tests for AudioEngine with graceful degradation."""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestAudioEngineSelection:
    """Test engine selection logic."""

    def test_auto_uses_f5_when_reference_exists(self, tmp_path):
        """Auto mode should use F5-TTS when voice reference file exists."""
        from vortex.core.audio import AudioEngine

        # Create mock reference file
        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()
        ref_file = refs_dir / "manic.wav"
        ref_file.write_bytes(b"fake audio data")

        with patch.object(AudioEngine, '_load_f5') as mock_load:
            with patch.object(AudioEngine, '_generate_f5') as mock_gen:
                mock_gen.return_value = str(tmp_path / "output.wav")

                engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))
                result = engine.generate(
                    script="Hello world",
                    engine="auto",
                    voice_style="manic"
                )

                mock_gen.assert_called_once()

    def test_auto_falls_back_to_kokoro_when_reference_missing(self, tmp_path):
        """Auto mode should fall back to Kokoro when reference file is missing."""
        from vortex.core.audio import AudioEngine

        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()
        # No reference file created

        with patch.object(AudioEngine, '_load_kokoro') as mock_load:
            with patch.object(AudioEngine, '_generate_kokoro') as mock_gen:
                mock_gen.return_value = str(tmp_path / "output.wav")

                engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))
                result = engine.generate(
                    script="Hello world",
                    engine="auto",
                    voice_style="nonexistent"
                )

                mock_gen.assert_called_once()

    def test_explicit_f5_raises_when_reference_missing(self, tmp_path):
        """Explicit f5_tts engine should raise FileNotFoundError when reference missing."""
        from vortex.core.audio import AudioEngine

        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()

        engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))

        with pytest.raises(FileNotFoundError):
            engine.generate(
                script="Hello world",
                engine="f5_tts",
                voice_style="nonexistent"
            )

    def test_explicit_kokoro_skips_f5(self, tmp_path):
        """Explicit kokoro engine should skip F5 entirely."""
        from vortex.core.audio import AudioEngine

        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()
        # Create reference that would be used by F5
        ref_file = refs_dir / "manic.wav"
        ref_file.write_bytes(b"fake audio data")

        with patch.object(AudioEngine, '_generate_f5') as mock_f5:
            with patch.object(AudioEngine, '_generate_kokoro') as mock_kokoro:
                mock_kokoro.return_value = str(tmp_path / "output.wav")

                engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))
                result = engine.generate(
                    script="Hello world",
                    engine="kokoro",
                    voice_style="manic",  # Should be ignored
                    voice_id="af_heart"
                )

                mock_f5.assert_not_called()
                mock_kokoro.assert_called_once()


class TestAudioEngineVRAM:
    """Test VRAM management."""

    def test_unload_clears_models(self, tmp_path):
        """Unload should clear model references and call empty_cache."""
        from vortex.core.audio import AudioEngine

        refs_dir = tmp_path / "voices"
        refs_dir.mkdir()

        engine = AudioEngine(device="cpu", assets_dir=str(refs_dir))
        engine._f5_model = Mock()
        engine._kokoro_model = Mock()

        with patch('torch.cuda.empty_cache') as mock_cache:
            with patch('gc.collect') as mock_gc:
                engine.unload()

                assert engine._f5_model is None
                assert engine._kokoro_model is None
                mock_gc.assert_called()
```

**Step 2: Run test to verify it fails**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_audio_engine.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'vortex.core.audio'"

**Step 3: Write minimal implementation**

Create `vortex/src/vortex/core/audio.py`:

```python
"""Audio generation engine with F5-TTS/Kokoro graceful degradation.

This module provides voice synthesis with automatic fallback:
- F5-TTS: Zero-shot voice cloning from reference audio (primary)
- Kokoro: Standard TTS (fallback when reference missing)

VRAM Management:
- Models are lazy-loaded on first use
- Call unload() before visual pipeline to free GPU memory
"""

from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class AudioEngine:
    """Voice synthesis engine with graceful degradation.

    Supports three modes via the `engine` parameter:
    - "auto": Try F5-TTS if reference exists, fall back to Kokoro
    - "f5_tts": Force F5-TTS (raises if reference missing)
    - "kokoro": Force Kokoro (skip F5 entirely)
    """

    def __init__(
        self,
        device: str = "cuda",
        assets_dir: str = "assets/voices",
        output_dir: str = "temp/audio",
    ):
        """Initialize audio engine.

        Args:
            device: PyTorch device for inference ("cuda" or "cpu")
            assets_dir: Directory containing voice reference WAVs
            output_dir: Directory for generated audio files
        """
        self.device = device
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded models
        self._f5_model = None
        self._kokoro_model = None

    def _load_f5(self) -> None:
        """Lazy-load F5-TTS model."""
        if self._f5_model is None:
            logger.info("Loading F5-TTS model...")
            # TODO: Import and load actual F5-TTS
            # from f5_tts import F5TTS
            # self._f5_model = F5TTS(device=self.device)
            self._f5_model = object()  # Placeholder
            logger.info("F5-TTS model loaded")

    def _load_kokoro(self) -> None:
        """Lazy-load Kokoro model."""
        if self._kokoro_model is None:
            logger.info("Loading Kokoro model...")
            from vortex.models.kokoro import KokoroWrapper
            self._kokoro_model = KokoroWrapper(device=self.device)
            logger.info("Kokoro model loaded")

    def _generate_f5(
        self, script: str, ref_path: Path, output_path: Path
    ) -> str:
        """Generate audio using F5-TTS.

        Args:
            script: Text to synthesize
            ref_path: Path to voice reference WAV
            output_path: Path to save generated audio

        Returns:
            Path to generated WAV file
        """
        self._load_f5()
        # TODO: Actual F5-TTS inference
        # self._f5_model.infer(
        #     ref_file=str(ref_path),
        #     text=script,
        #     output_file=str(output_path)
        # )
        logger.info(f"F5-TTS generated: {output_path}")
        return str(output_path)

    def _generate_kokoro(
        self, script: str, voice_id: str, output_path: Path
    ) -> str:
        """Generate audio using Kokoro.

        Args:
            script: Text to synthesize
            voice_id: Kokoro voice identifier
            output_path: Path to save generated audio

        Returns:
            Path to generated WAV file
        """
        self._load_kokoro()
        result = self._kokoro_model.generate(
            text=script,
            voice=voice_id,
            output_path=str(output_path),
        )
        logger.info(f"Kokoro generated: {output_path}")
        return str(output_path)

    def generate(
        self,
        script: str,
        engine: Literal["auto", "f5_tts", "kokoro"] = "auto",
        voice_style: str | None = None,
        voice_id: str = "af_heart",
    ) -> str:
        """Generate voice audio with graceful degradation.

        Args:
            script: Text to synthesize
            engine: Engine selection mode
            voice_style: F5-TTS reference filename (without .wav)
            voice_id: Kokoro voice ID (used if engine=kokoro or as fallback)

        Returns:
            Path to generated WAV file

        Raises:
            FileNotFoundError: If engine=f5_tts and reference file missing
        """
        output_path = self.output_dir / f"voice_{uuid.uuid4().hex[:8]}.wav"

        # Explicit Kokoro mode - skip F5 entirely
        if engine == "kokoro":
            logger.info(f"Using Kokoro (explicit) with voice: {voice_id}")
            return self._generate_kokoro(script, voice_id, output_path)

        # Try F5-TTS if reference exists
        if voice_style:
            ref_path = self.assets_dir / f"{voice_style}.wav"

            if ref_path.exists():
                try:
                    logger.info(f"Using F5-TTS with style: {voice_style}")
                    return self._generate_f5(script, ref_path, output_path)
                except Exception as e:
                    if engine == "f5_tts":
                        raise  # Explicit F5 mode - don't fall back
                    logger.warning(f"F5-TTS failed: {e}, falling back to Kokoro")
            else:
                if engine == "f5_tts":
                    raise FileNotFoundError(
                        f"Voice reference not found: {ref_path}"
                    )
                logger.warning(
                    f"Voice style '{voice_style}' not found, falling back to Kokoro"
                )

        # Fallback to Kokoro
        logger.info(f"Using Kokoro (fallback) with voice: {voice_id}")
        return self._generate_kokoro(script, voice_id, output_path)

    def unload(self) -> None:
        """Unload models and free VRAM.

        Call this before starting the visual pipeline to ensure
        maximum GPU memory is available for ComfyUI.
        """
        import torch

        if self._f5_model is not None:
            logger.info("Unloading F5-TTS model...")
            del self._f5_model
            self._f5_model = None

        if self._kokoro_model is not None:
            logger.info("Unloading Kokoro model...")
            del self._kokoro_model
            self._kokoro_model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Audio models unloaded, VRAM freed")
```

**Step 4: Run test to verify it passes**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_audio_engine.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/vortex/core/audio.py tests/unit/test_audio_engine.py
git commit -m "feat: add AudioEngine with F5-TTS/Kokoro graceful degradation"
```

---

### Task 2.3: Implement AudioCompositor (FFmpeg Mixing)

**Files:**
- Create: `vortex/src/vortex/core/mixer.py`
- Create: `vortex/tests/unit/test_audio_mixer.py`

**Step 1: Write the failing test**

Create `vortex/tests/unit/test_audio_mixer.py`:

```python
"""Unit tests for AudioCompositor (FFmpeg mixing)."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestAudioCompositor:
    """Test FFmpeg mixing logic."""

    def test_mix_voice_only(self, tmp_path):
        """Mixing with voice only should just copy the voice file."""
        from vortex.core.mixer import AudioCompositor

        voice_path = tmp_path / "voice.wav"
        voice_path.write_bytes(b"fake voice data")
        output_path = tmp_path / "output.wav"

        compositor = AudioCompositor(assets_dir=str(tmp_path / "assets"))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = compositor.mix(
                voice_path=str(voice_path),
                output_path=str(output_path),
                bgm_name=None,
                sfx_name=None,
            )

            mock_run.assert_called_once()
            # Verify FFmpeg was called
            call_args = mock_run.call_args[0][0]
            assert 'ffmpeg' in call_args[0]

    def test_mix_with_bgm_applies_volume(self, tmp_path):
        """Mixing with BGM should apply volume filter."""
        from vortex.core.mixer import AudioCompositor

        voice_path = tmp_path / "voice.wav"
        voice_path.write_bytes(b"fake voice data")

        assets_dir = tmp_path / "assets" / "audio" / "bgm"
        assets_dir.mkdir(parents=True)
        bgm_path = assets_dir / "elevator.wav"
        bgm_path.write_bytes(b"fake bgm data")

        output_path = tmp_path / "output.wav"

        compositor = AudioCompositor(assets_dir=str(tmp_path / "assets"))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = compositor.mix(
                voice_path=str(voice_path),
                output_path=str(output_path),
                bgm_name="elevator",
                mix_ratio=0.3,
            )

            # Verify FFmpeg command includes volume filter
            call_args = str(mock_run.call_args)
            assert 'volume' in call_args or 'amix' in call_args

    def test_mix_raises_on_ffmpeg_failure(self, tmp_path):
        """Should raise RuntimeError if FFmpeg fails."""
        from vortex.core.mixer import AudioCompositor

        voice_path = tmp_path / "voice.wav"
        voice_path.write_bytes(b"fake voice data")
        output_path = tmp_path / "output.wav"

        compositor = AudioCompositor(assets_dir=str(tmp_path / "assets"))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")

            with pytest.raises(RuntimeError):
                compositor.mix(
                    voice_path=str(voice_path),
                    output_path=str(output_path),
                )


class TestAudioDurationCalculation:
    """Test audio duration utilities."""

    def test_calculate_frame_count_from_duration(self):
        """Should calculate correct frame count from audio duration."""
        from vortex.core.mixer import calculate_frame_count

        # 10 seconds at 24fps = 240 frames
        with patch('soundfile.SoundFile') as mock_sf:
            mock_file = MagicMock()
            mock_file.__len__ = MagicMock(return_value=240000)  # 10s at 24kHz
            mock_file.samplerate = 24000
            mock_sf.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_sf.return_value.__exit__ = MagicMock(return_value=False)

            frames = calculate_frame_count("dummy.wav", fps=24)
            assert frames == 240

    def test_calculate_frame_count_pads_short_audio(self):
        """Should pad to minimum 16 frames for AnimateDiff context."""
        from vortex.core.mixer import calculate_frame_count

        # 0.5 seconds at 24fps = 12 frames, should pad to 16
        with patch('soundfile.SoundFile') as mock_sf:
            mock_file = MagicMock()
            mock_file.__len__ = MagicMock(return_value=12000)  # 0.5s at 24kHz
            mock_file.samplerate = 24000
            mock_sf.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_sf.return_value.__exit__ = MagicMock(return_value=False)

            frames = calculate_frame_count("dummy.wav", fps=24, min_frames=16)
            assert frames == 16
```

**Step 2: Run test to verify it fails**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_audio_mixer.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'vortex.core.mixer'"

**Step 3: Write minimal implementation**

Create `vortex/src/vortex/core/mixer.py`:

```python
"""Audio compositor for mixing voice, BGM, and SFX.

Uses FFmpeg for all mixing operations (CPU-based, zero VRAM).
Supports:
- Voice + BGM mixing with configurable volume ratio
- SFX overlay at specific timestamps
- Duration-based frame count calculation for video sync
"""

from __future__ import annotations

import logging
import math
import subprocess
import uuid
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)


def calculate_frame_count(
    audio_path: str,
    fps: int = 24,
    min_frames: int = 16,
) -> int:
    """Calculate video frame count from audio duration.

    Args:
        audio_path: Path to audio file
        fps: Target video frame rate
        min_frames: Minimum frames (AnimateDiff context window)

    Returns:
        Number of frames needed for video
    """
    with sf.SoundFile(audio_path) as f:
        duration_sec = len(f) / f.samplerate

    frames = math.ceil(duration_sec * fps)

    # Pad to minimum context window for AnimateDiff
    if frames < min_frames:
        logger.warning(
            f"Audio too short ({frames} frames), padding to {min_frames}"
        )
        return min_frames

    return frames


class AudioCompositor:
    """FFmpeg-based audio compositor.

    Mixes voice tracks with background music and sound effects.
    All operations are CPU-based with zero VRAM usage.
    """

    def __init__(
        self,
        assets_dir: str = "assets",
        output_dir: str = "temp/audio",
    ):
        """Initialize compositor.

        Args:
            assets_dir: Root directory for audio assets (bgm/, sfx/)
            output_dir: Directory for mixed output files
        """
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bgm_dir = self.assets_dir / "audio" / "bgm"
        self.sfx_dir = self.assets_dir / "audio" / "sfx"

    def mix(
        self,
        voice_path: str,
        output_path: str | None = None,
        bgm_name: str | None = None,
        sfx_name: str | None = None,
        mix_ratio: float = 0.3,
    ) -> str:
        """Mix voice with optional BGM and SFX.

        Args:
            voice_path: Path to voice WAV file
            output_path: Path for output (auto-generated if None)
            bgm_name: BGM filename (without .wav) from assets/audio/bgm/
            sfx_name: SFX filename (without .wav) from assets/audio/sfx/
            mix_ratio: BGM volume relative to voice (0.0-1.0)

        Returns:
            Path to mixed audio file

        Raises:
            FileNotFoundError: If voice or specified asset files don't exist
            RuntimeError: If FFmpeg fails
        """
        if output_path is None:
            output_path = str(
                self.output_dir / f"mixed_{uuid.uuid4().hex[:8]}.wav"
            )

        voice_path = Path(voice_path)
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")

        # Build FFmpeg command
        inputs = ["-i", str(voice_path)]
        filter_parts = []
        input_count = 1

        # Add BGM if specified
        if bgm_name:
            bgm_path = self.bgm_dir / f"{bgm_name}.wav"
            if not bgm_path.exists():
                logger.warning(f"BGM not found: {bgm_path}, skipping")
            else:
                inputs.extend(["-i", str(bgm_path)])
                input_count += 1

        # Add SFX if specified
        if sfx_name:
            sfx_path = self.sfx_dir / f"{sfx_name}.wav"
            if not sfx_path.exists():
                logger.warning(f"SFX not found: {sfx_path}, skipping")
            else:
                inputs.extend(["-i", str(sfx_path)])
                input_count += 1

        # Build filter graph
        if input_count == 1:
            # Voice only - just convert/copy
            cmd = [
                "ffmpeg", "-y",
                *inputs,
                "-c:a", "pcm_s16le",
                output_path,
            ]
        else:
            # Mix multiple inputs
            # Voice at full volume, BGM at mix_ratio, SFX at 0.6
            filter_complex = []

            if input_count == 2 and bgm_name:
                # Voice + BGM
                filter_complex = [
                    "-filter_complex",
                    f"[1:a]volume={mix_ratio},aloop=loop=-1:size=2e9[bgm];"
                    f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2",
                ]
            elif input_count == 2 and sfx_name:
                # Voice + SFX
                filter_complex = [
                    "-filter_complex",
                    f"[1:a]volume=0.6[sfx];"
                    f"[0:a][sfx]amix=inputs=2:duration=first",
                ]
            else:
                # Voice + BGM + SFX
                filter_complex = [
                    "-filter_complex",
                    f"[1:a]volume={mix_ratio},aloop=loop=-1:size=2e9[bgm];"
                    f"[2:a]volume=0.6[sfx];"
                    f"[0:a][bgm][sfx]amix=inputs=3:duration=first:dropout_transition=2",
                ]

            cmd = [
                "ffmpeg", "-y",
                *inputs,
                *filter_complex,
                "-c:a", "pcm_s16le",
                output_path,
            ]

        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        # Execute FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg mixing failed: {result.stderr}")

        logger.info(f"Mixed audio saved to: {output_path}")
        return output_path


def mix_final_audio(
    voice_path: str,
    output_path: str,
    bgm_name: str | None = None,
    sfx_name: str | None = None,
    mix_ratio: float = 0.3,
    assets_dir: str = "assets",
) -> str:
    """Convenience function for one-shot mixing.

    Args:
        voice_path: Path to voice WAV
        output_path: Path for output WAV
        bgm_name: Background music name
        sfx_name: Sound effect name
        mix_ratio: BGM volume ratio
        assets_dir: Assets directory root

    Returns:
        Path to mixed audio file
    """
    compositor = AudioCompositor(assets_dir=assets_dir)
    return compositor.mix(
        voice_path=voice_path,
        output_path=output_path,
        bgm_name=bgm_name,
        sfx_name=sfx_name,
        mix_ratio=mix_ratio,
    )
```

**Step 4: Run test to verify it passes**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_audio_mixer.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/vortex/core/mixer.py tests/unit/test_audio_mixer.py
git commit -m "feat: add AudioCompositor for FFmpeg-based audio mixing"
```

---

### Task 2.4: Create Voice Reference Assets Structure

**Files:**
- Create: `vortex/assets/voices/.gitkeep`
- Create: `vortex/assets/audio/bgm/.gitkeep`
- Create: `vortex/assets/audio/sfx/.gitkeep`
- Create: `vortex/assets/README.md`

**Step 1: Create asset directories**

```bash
cd /home/matt/nsn/vortex
mkdir -p assets/voices
mkdir -p assets/audio/bgm
mkdir -p assets/audio/sfx
touch assets/voices/.gitkeep
touch assets/audio/bgm/.gitkeep
touch assets/audio/sfx/.gitkeep
```

**Step 2: Create assets README**

Create `vortex/assets/README.md`:

```markdown
# ToonGen Audio Assets

This directory contains audio assets for the ToonGen pipeline.

## Directory Structure

```
assets/
├── voices/              # F5-TTS voice reference clips (5-10 seconds)
│   ├── manic_salesman.wav
│   ├── nervous_morty.wav
│   ├── monotone_bot.wav
│   └── ...
├── audio/
│   ├── bgm/            # Background music loops
│   │   ├── cheesy_elevator.wav
│   │   ├── 80s_synth.wav
│   │   └── ...
│   └── sfx/            # Sound effects
│       ├── static_glitch.wav
│       ├── explosion_short.wav
│       └── ...
```

## Voice References (F5-TTS)

Voice reference files should be:
- Format: WAV (16-bit PCM)
- Duration: 5-10 seconds of clear speech
- Quality: Clean recording, minimal background noise
- Content: Representative of the target voice style

### MVP Voice Styles (10 required)

1. `manic_salesman.wav` - Fast, loud, infomercial energy
2. `nervous_morty.wav` - Stuttering, cracking voice
3. `monotone_bot.wav` - Flat, robotic delivery
4. `whispering_creep.wav` - Quiet, breathy
5. `excited_host.wav` - Game show energy
6. `deadpan_narrator.wav` - Documentary style
7. `angry_chef.wav` - Gordon Ramsay energy
8. `surfer_dude.wav` - Laid back, California
9. `old_timey.wav` - 1920s radio announcer
10. `alien_visitor.wav` - Confused, formal

## Background Music

BGM files should be:
- Format: WAV (16-bit PCM)
- Style: Loopable (seamless loop points preferred)
- Duration: 30+ seconds (will be looped automatically)

## Sound Effects

SFX files should be:
- Format: WAV (16-bit PCM)
- Duration: 1-5 seconds
- Used for: Intro/outro stings, transitions, emphasis
```

**Step 3: Commit**

```bash
git add assets/
git commit -m "feat: create audio assets directory structure"
```

---

## Phase 3: Orchestrator (ComfyUI Client)

Build the Python wrapper for ComfyUI WebSocket API.

### Task 3.1: Implement WorkflowBuilder (JSON Injection)

**Files:**
- Create: `vortex/src/vortex/engine/__init__.py`
- Create: `vortex/src/vortex/engine/payload.py`
- Create: `vortex/tests/unit/test_payload.py`

**Step 1: Create engine package**

```bash
mkdir -p /home/matt/nsn/vortex/src/vortex/engine
touch /home/matt/nsn/vortex/src/vortex/engine/__init__.py
```

**Step 2: Write the failing test**

Create `vortex/tests/unit/test_payload.py`:

```python
"""Unit tests for WorkflowBuilder (JSON injection)."""

import json
import pytest
from pathlib import Path


class TestWorkflowBuilder:
    """Test JSON template injection."""

    def test_injects_prompt(self):
        """Should inject prompt into correct node."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "6": {"inputs": {"text": "__PROMPT__"}},
            "10": {"inputs": {"seed": 0}},
        }

        builder = WorkflowBuilder(template_data=template)
        result = builder.build(prompt="A cyberpunk cat")

        assert result["6"]["inputs"]["text"] == "A cyberpunk cat"

    def test_injects_audio_path(self):
        """Should inject audio path into correct node."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "40": {"inputs": {"audio": "__AUDIO__"}},
        }

        builder = WorkflowBuilder(
            template_data=template,
            node_map={"audio": "40"}
        )
        result = builder.build(audio_path="/tmp/voice.wav")

        assert result["40"]["inputs"]["audio"] == "/tmp/voice.wav"

    def test_injects_seed(self):
        """Should inject seed into KSampler node."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "10": {"inputs": {"seed": 0}},
        }

        builder = WorkflowBuilder(
            template_data=template,
            node_map={"seed": "10"}
        )
        result = builder.build(seed=12345)

        assert result["10"]["inputs"]["seed"] == 12345

    def test_generates_random_seed_if_not_provided(self):
        """Should generate random seed if not specified."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "10": {"inputs": {"seed": 0}},
        }

        builder = WorkflowBuilder(
            template_data=template,
            node_map={"seed": "10"}
        )
        result1 = builder.build()
        result2 = builder.build()

        # Seeds should be different (random)
        assert result1["10"]["inputs"]["seed"] != 0
        assert result2["10"]["inputs"]["seed"] != 0

    def test_loads_template_from_file(self, tmp_path):
        """Should load template from JSON file."""
        from vortex.engine.payload import WorkflowBuilder

        template = {"6": {"inputs": {"text": "default"}}}
        template_path = tmp_path / "workflow.json"
        template_path.write_text(json.dumps(template))

        builder = WorkflowBuilder(template_path=str(template_path))
        result = builder.build(prompt="Test prompt")

        assert result["6"]["inputs"]["text"] == "Test prompt"

    def test_preserves_unmodified_nodes(self):
        """Should not modify nodes that aren't in the injection map."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "6": {"inputs": {"text": "__PROMPT__"}},
            "99": {"inputs": {"special": "value", "other": 123}},
        }

        builder = WorkflowBuilder(template_data=template)
        result = builder.build(prompt="Test")

        assert result["99"]["inputs"]["special"] == "value"
        assert result["99"]["inputs"]["other"] == 123
```

**Step 3: Run test to verify it fails**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_payload.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'vortex.engine.payload'"

**Step 4: Write minimal implementation**

Create `vortex/src/vortex/engine/payload.py`:

```python
"""Workflow payload builder for ComfyUI API.

Handles JSON template loading and parameter injection for the
ComfyUI workflow API format.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default node ID mappings (update these after exporting your workflow)
DEFAULT_NODE_MAP = {
    "prompt": "6",      # CLIPTextEncode (Positive)
    "audio": "40",      # LoadAudio
    "seed": "10",       # KSampler (Flux)
    "seed_ad": "14",    # KSampler (AnimateDiff)
}


class WorkflowBuilder:
    """Builds ComfyUI API payloads from templates.

    Loads a workflow JSON template and injects runtime parameters
    (prompt, audio path, seed) into the appropriate nodes.
    """

    def __init__(
        self,
        template_path: str | None = None,
        template_data: dict[str, Any] | None = None,
        node_map: dict[str, str] | None = None,
    ):
        """Initialize builder.

        Args:
            template_path: Path to workflow JSON file
            template_data: Workflow dict (alternative to file)
            node_map: Mapping of parameter names to node IDs

        Raises:
            ValueError: If neither template_path nor template_data provided
        """
        if template_data is not None:
            self._template = template_data
        elif template_path is not None:
            self._template = self._load_template(template_path)
        else:
            raise ValueError("Must provide template_path or template_data")

        self._node_map = node_map or DEFAULT_NODE_MAP

    def _load_template(self, path: str) -> dict[str, Any]:
        """Load workflow template from JSON file."""
        template_path = Path(path)
        if not template_path.exists():
            raise FileNotFoundError(f"Workflow template not found: {path}")

        with open(template_path) as f:
            return json.load(f)

    def build(
        self,
        prompt: str | None = None,
        audio_path: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build workflow payload with injected parameters.

        Args:
            prompt: Text prompt for image generation
            audio_path: Path to audio file for LivePortrait
            seed: Deterministic seed (random if not provided)

        Returns:
            Complete workflow dict ready for ComfyUI API
        """
        # Deep copy to avoid mutating template
        workflow = copy.deepcopy(self._template)

        # Inject prompt
        if prompt is not None:
            prompt_node = self._node_map.get("prompt", "6")
            if prompt_node in workflow:
                workflow[prompt_node]["inputs"]["text"] = prompt
                logger.debug(f"Injected prompt into node {prompt_node}")

        # Inject audio path
        if audio_path is not None:
            audio_node = self._node_map.get("audio", "40")
            if audio_node in workflow:
                workflow[audio_node]["inputs"]["audio"] = audio_path
                logger.debug(f"Injected audio path into node {audio_node}")

        # Inject seed (generate random if not provided)
        actual_seed = seed if seed is not None else random.randint(1, 2**31 - 1)

        seed_node = self._node_map.get("seed", "10")
        if seed_node in workflow:
            workflow[seed_node]["inputs"]["seed"] = actual_seed
            logger.debug(f"Injected seed {actual_seed} into node {seed_node}")

        # Also set AnimateDiff seed if present
        seed_ad_node = self._node_map.get("seed_ad")
        if seed_ad_node and seed_ad_node in workflow:
            workflow[seed_ad_node]["inputs"]["seed"] = actual_seed
            logger.debug(f"Injected seed {actual_seed} into AnimateDiff node")

        return workflow

    def get_node_ids(self) -> dict[str, str]:
        """Return current node ID mappings."""
        return self._node_map.copy()


def load_workflow(template_path: str) -> WorkflowBuilder:
    """Convenience function to create a WorkflowBuilder from file.

    Args:
        template_path: Path to workflow JSON

    Returns:
        Configured WorkflowBuilder instance
    """
    return WorkflowBuilder(template_path=template_path)
```

**Step 5: Run test to verify it passes**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_payload.py -v
```

Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/vortex/engine/ tests/unit/test_payload.py
git commit -m "feat: add WorkflowBuilder for ComfyUI JSON injection"
```

---

### Task 3.2: Implement ComfyClient (WebSocket Wrapper)

**Files:**
- Create: `vortex/src/vortex/engine/client.py`
- Create: `vortex/tests/unit/test_comfy_client.py`

**Step 1: Write the failing test**

Create `vortex/tests/unit/test_comfy_client.py`:

```python
"""Unit tests for ComfyClient (WebSocket wrapper)."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock


class TestComfyClientConnection:
    """Test connection handling."""

    @pytest.mark.asyncio
    async def test_connects_to_server(self):
        """Should establish WebSocket connection."""
        from vortex.engine.client import ComfyClient

        with patch('websockets.connect', new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value.__aenter__ = AsyncMock()
            mock_ws.return_value.__aexit__ = AsyncMock()

            client = ComfyClient(host="localhost", port=8188)

            # Connection happens on queue_prompt, not init
            assert client.host == "localhost"
            assert client.port == 8188

    def test_generates_unique_client_id(self):
        """Should generate unique client ID for session tracking."""
        from vortex.engine.client import ComfyClient

        client1 = ComfyClient()
        client2 = ComfyClient()

        assert client1.client_id != client2.client_id
        assert len(client1.client_id) > 0


class TestComfyClientQueueing:
    """Test job queuing."""

    @pytest.mark.asyncio
    async def test_queue_prompt_sends_workflow(self):
        """Should POST workflow to /prompt endpoint."""
        from vortex.engine.client import ComfyClient

        workflow = {"6": {"inputs": {"text": "test"}}}

        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"prompt_id": "abc123"})

            mock_session.return_value.__aenter__ = AsyncMock(
                return_value=mock_session.return_value
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            mock_session.return_value.post = AsyncMock(return_value=mock_response)
            mock_response.__aenter__ = AsyncMock(return_value=mock_response)
            mock_response.__aexit__ = AsyncMock()

            client = ComfyClient()
            prompt_id = await client.queue_prompt(workflow)

            assert prompt_id == "abc123"


class TestComfyClientResults:
    """Test result retrieval."""

    def test_parses_execution_success_message(self):
        """Should correctly parse execution_success WebSocket message."""
        from vortex.engine.client import ComfyClient

        client = ComfyClient()

        message = json.dumps({
            "type": "executed",
            "data": {
                "node": "99",
                "output": {
                    "gifs": [{"filename": "output.mp4", "subfolder": "", "type": "output"}]
                }
            }
        })

        result = client._parse_message(message)

        assert result is not None
        assert result["type"] == "executed"

    def test_parses_progress_message(self):
        """Should parse progress updates."""
        from vortex.engine.client import ComfyClient

        client = ComfyClient()

        message = json.dumps({
            "type": "progress",
            "data": {"value": 50, "max": 100}
        })

        result = client._parse_message(message)

        assert result["type"] == "progress"
        assert result["data"]["value"] == 50
```

**Step 2: Run test to verify it fails**

```bash
cd /home/matt/nsn/vortex
pip install pytest-asyncio aiohttp websockets
python -m pytest tests/unit/test_comfy_client.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'vortex.engine.client'"

**Step 3: Write minimal implementation**

Create `vortex/src/vortex/engine/client.py`:

```python
"""ComfyUI WebSocket client for job dispatch and monitoring.

Provides async interface for:
- Queuing workflow prompts
- Monitoring execution progress
- Retrieving output file paths
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import aiohttp

logger = logging.getLogger(__name__)


class ComfyClient:
    """Async client for ComfyUI WebSocket API.

    Handles job queuing, progress monitoring, and result retrieval.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        timeout: float = 300.0,
    ):
        """Initialize client.

        Args:
            host: ComfyUI server hostname
            port: ComfyUI server port
            timeout: Maximum seconds to wait for job completion
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

        self._base_url = f"http://{host}:{port}"
        self._ws_url = f"ws://{host}:{port}/ws?clientId={self.client_id}"

    async def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Queue a workflow for execution.

        Args:
            workflow: ComfyUI workflow dict (API format)

        Returns:
            Prompt ID for tracking execution

        Raises:
            RuntimeError: If queuing fails
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/prompt",
                json=payload,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"Failed to queue prompt: {text}")

                data = await response.json()
                prompt_id = data.get("prompt_id")

                if not prompt_id:
                    raise RuntimeError(f"No prompt_id in response: {data}")

                logger.info(f"Queued prompt: {prompt_id}")
                return prompt_id

    def _parse_message(self, raw: str) -> dict[str, Any] | None:
        """Parse WebSocket message.

        Args:
            raw: Raw message string

        Returns:
            Parsed message dict or None if not JSON
        """
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Non-JSON message: {raw[:100]}")
            return None

    async def _listen_progress(
        self,
        prompt_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Listen for execution progress via WebSocket.

        Args:
            prompt_id: Prompt ID to monitor

        Yields:
            Progress update dicts
        """
        import websockets

        async with websockets.connect(self._ws_url) as ws:
            async for raw_message in ws:
                if isinstance(raw_message, bytes):
                    # Binary message (preview image) - skip
                    continue

                message = self._parse_message(raw_message)
                if message is None:
                    continue

                msg_type = message.get("type")
                data = message.get("data", {})

                # Filter to our prompt
                if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                    continue

                yield message

                # Check for completion
                if msg_type == "executed":
                    logger.info(f"Execution completed for node: {data.get('node')}")
                elif msg_type == "execution_error":
                    raise RuntimeError(f"Execution error: {data}")
                elif msg_type == "execution_complete":
                    logger.info("Workflow execution complete")
                    return

    async def wait_for_completion(
        self,
        prompt_id: str,
        output_node: str = "99",
    ) -> str:
        """Wait for job completion and return output path.

        Args:
            prompt_id: Prompt ID to wait for
            output_node: Node ID that produces the output file

        Returns:
            Path to output file (MP4)

        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If execution fails
        """
        output_path = None

        try:
            async with asyncio.timeout(self.timeout):
                async for message in self._listen_progress(prompt_id):
                    msg_type = message.get("type")
                    data = message.get("data", {})

                    if msg_type == "progress":
                        value = data.get("value", 0)
                        max_val = data.get("max", 100)
                        logger.debug(f"Progress: {value}/{max_val}")

                    elif msg_type == "executed":
                        node = data.get("node")
                        output = data.get("output", {})

                        # Check for video output (VHS_VideoCombine)
                        if "gifs" in output:
                            for gif in output["gifs"]:
                                filename = gif.get("filename")
                                subfolder = gif.get("subfolder", "")
                                if filename:
                                    output_path = str(
                                        Path(self._get_output_dir())
                                        / subfolder
                                        / filename
                                    )
                                    logger.info(f"Output file: {output_path}")

                    elif msg_type == "execution_complete":
                        break

        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Job {prompt_id} did not complete within {self.timeout}s"
            )

        if output_path is None:
            raise RuntimeError(f"No output file found for job {prompt_id}")

        return output_path

    def _get_output_dir(self) -> str:
        """Get ComfyUI output directory path."""
        # Default ComfyUI output location
        return f"{self.host == '127.0.0.1' and '/home/matt/nsn/ComfyUI' or '/ComfyUI'}/output"

    async def generate(
        self,
        workflow: dict[str, Any],
        output_node: str = "99",
    ) -> str:
        """Queue workflow and wait for output.

        Convenience method combining queue_prompt and wait_for_completion.

        Args:
            workflow: ComfyUI workflow dict
            output_node: Node ID for output

        Returns:
            Path to output file
        """
        prompt_id = await self.queue_prompt(workflow)
        return await self.wait_for_completion(prompt_id, output_node)

    async def check_health(self) -> bool:
        """Check if ComfyUI server is reachable.

        Returns:
            True if server responds, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._base_url}/system_stats",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"ComfyUI health check failed: {e}")
            return False
```

**Step 4: Run test to verify it passes**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_comfy_client.py -v
```

Expected: All tests PASS

**Step 5: Update engine __init__.py**

```python
"""ToonGen engine modules.

- client: ComfyUI WebSocket client
- payload: Workflow JSON builder
"""

from vortex.engine.client import ComfyClient
from vortex.engine.payload import WorkflowBuilder, load_workflow

__all__ = ["ComfyClient", "WorkflowBuilder", "load_workflow"]
```

**Step 6: Commit**

```bash
git add src/vortex/engine/ tests/unit/test_comfy_client.py
git commit -m "feat: add ComfyClient for ComfyUI WebSocket API"
```

---

### Task 3.3: Implement VideoOrchestrator (Main Pipeline)

**Files:**
- Create: `vortex/src/vortex/orchestrator.py`
- Create: `vortex/tests/unit/test_orchestrator.py`

**Step 1: Write the failing test**

Create `vortex/tests/unit/test_orchestrator.py`:

```python
"""Unit tests for VideoOrchestrator."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path


class TestVideoOrchestrator:
    """Test main orchestration pipeline."""

    @pytest.mark.asyncio
    async def test_generate_calls_audio_then_visual(self, tmp_path):
        """Should generate audio first, then dispatch to ComfyUI."""
        from vortex.orchestrator import VideoOrchestrator

        with patch('vortex.orchestrator.AudioEngine') as mock_audio_cls:
            with patch('vortex.orchestrator.AudioCompositor') as mock_mixer_cls:
                with patch('vortex.orchestrator.ComfyClient') as mock_comfy_cls:
                    with patch('vortex.orchestrator.WorkflowBuilder') as mock_builder_cls:
                        # Setup mocks
                        mock_audio = mock_audio_cls.return_value
                        mock_audio.generate.return_value = str(tmp_path / "voice.wav")
                        mock_audio.unload = Mock()

                        mock_mixer = mock_mixer_cls.return_value
                        mock_mixer.mix.return_value = str(tmp_path / "mixed.wav")

                        mock_builder = mock_builder_cls.return_value
                        mock_builder.build.return_value = {"workflow": "data"}

                        mock_comfy = mock_comfy_cls.return_value
                        mock_comfy.generate = AsyncMock(
                            return_value=str(tmp_path / "output.mp4")
                        )

                        # Run orchestrator
                        orchestrator = VideoOrchestrator(
                            template_path=str(tmp_path / "workflow.json"),
                            assets_dir=str(tmp_path / "assets"),
                        )

                        result = await orchestrator.generate(
                            prompt="A cool cat",
                            script="Hello world",
                        )

                        # Verify call order
                        mock_audio.generate.assert_called_once()
                        mock_audio.unload.assert_called_once()  # VRAM freed before ComfyUI
                        mock_comfy.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_both_audio_paths(self, tmp_path):
        """Should return clean audio (for lip-sync) and mixed audio (for broadcast)."""
        from vortex.orchestrator import VideoOrchestrator

        with patch('vortex.orchestrator.AudioEngine') as mock_audio_cls:
            with patch('vortex.orchestrator.AudioCompositor') as mock_mixer_cls:
                with patch('vortex.orchestrator.ComfyClient') as mock_comfy_cls:
                    with patch('vortex.orchestrator.WorkflowBuilder') as mock_builder_cls:
                        mock_audio = mock_audio_cls.return_value
                        mock_audio.generate.return_value = str(tmp_path / "voice.wav")
                        mock_audio.unload = Mock()

                        mock_mixer = mock_mixer_cls.return_value
                        mock_mixer.mix.return_value = str(tmp_path / "mixed.wav")

                        mock_builder = mock_builder_cls.return_value
                        mock_builder.build.return_value = {}

                        mock_comfy = mock_comfy_cls.return_value
                        mock_comfy.generate = AsyncMock(
                            return_value=str(tmp_path / "output.mp4")
                        )

                        orchestrator = VideoOrchestrator(
                            template_path=str(tmp_path / "workflow.json"),
                        )

                        result = await orchestrator.generate(
                            prompt="Test",
                            script="Test script",
                            bgm_name="elevator",
                        )

                        assert "video_path" in result
                        assert "clean_audio_path" in result
                        assert "mixed_audio_path" in result
```

**Step 2: Run test to verify it fails**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_orchestrator.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'vortex.orchestrator'"

**Step 3: Write minimal implementation**

Create `vortex/src/vortex/orchestrator.py`:

```python
"""ToonGen Video Orchestrator.

Main pipeline that coordinates:
1. Audio generation (F5-TTS/Kokoro)
2. Audio mixing (FFmpeg)
3. Visual generation (ComfyUI)
4. VRAM management (sequential execution)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vortex.core.audio import AudioEngine
from vortex.core.mixer import AudioCompositor, calculate_frame_count
from vortex.engine.client import ComfyClient
from vortex.engine.payload import WorkflowBuilder

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of video generation."""

    video_path: str
    clean_audio_path: str
    mixed_audio_path: str
    frame_count: int
    seed: int


class VideoOrchestrator:
    """Orchestrates the ToonGen video generation pipeline.

    Pipeline flow:
    1. Generate voice audio (F5-TTS or Kokoro)
    2. Mix with BGM/SFX (FFmpeg)
    3. Unload audio models (free VRAM)
    4. Build ComfyUI workflow payload
    5. Dispatch to ComfyUI and wait for completion
    6. Return paths to output files
    """

    def __init__(
        self,
        template_path: str = "templates/cartoon_workflow.json",
        assets_dir: str = "assets",
        output_dir: str = "outputs",
        comfy_host: str = "127.0.0.1",
        comfy_port: int = 8188,
        device: str = "cuda",
    ):
        """Initialize orchestrator.

        Args:
            template_path: Path to ComfyUI workflow JSON
            assets_dir: Root directory for audio assets
            output_dir: Directory for output files
            comfy_host: ComfyUI server hostname
            comfy_port: ComfyUI server port
            device: PyTorch device for audio models
        """
        self.template_path = template_path
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._audio_engine = AudioEngine(
            device=device,
            assets_dir=str(self.assets_dir / "voices"),
        )
        self._audio_mixer = AudioCompositor(
            assets_dir=str(self.assets_dir),
        )
        self._workflow_builder = WorkflowBuilder(template_path=template_path)
        self._comfy_client = ComfyClient(host=comfy_host, port=comfy_port)

    async def generate(
        self,
        prompt: str,
        script: str,
        voice_style: str | None = None,
        voice_id: str = "af_heart",
        engine: str = "auto",
        bgm_name: str | None = None,
        sfx_name: str | None = None,
        mix_ratio: float = 0.3,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate a video clip.

        Args:
            prompt: Visual prompt for Flux image generation
            script: Text to synthesize as speech
            voice_style: F5-TTS voice reference name
            voice_id: Kokoro voice ID
            engine: Audio engine selection ("auto", "f5_tts", "kokoro")
            bgm_name: Background music asset name
            sfx_name: Sound effect asset name
            mix_ratio: BGM volume relative to voice
            seed: Deterministic seed (random if not provided)

        Returns:
            Dict with video_path, clean_audio_path, mixed_audio_path, etc.
        """
        logger.info(f"Starting generation: prompt='{prompt[:50]}...'")

        # ===== PHASE 1: Audio Generation =====
        logger.info("Phase 1: Generating audio...")

        # Generate clean voice (for lip-sync)
        clean_audio_path = self._audio_engine.generate(
            script=script,
            engine=engine,
            voice_style=voice_style,
            voice_id=voice_id,
        )
        logger.info(f"Clean audio generated: {clean_audio_path}")

        # Mix with BGM/SFX (for broadcast)
        mixed_audio_path = self._audio_mixer.mix(
            voice_path=clean_audio_path,
            bgm_name=bgm_name,
            sfx_name=sfx_name,
            mix_ratio=mix_ratio,
        )
        logger.info(f"Mixed audio generated: {mixed_audio_path}")

        # Calculate frame count from audio duration
        frame_count = calculate_frame_count(clean_audio_path)
        logger.info(f"Calculated frame count: {frame_count}")

        # ===== VRAM HANDOFF =====
        # CRITICAL: Unload audio models BEFORE ComfyUI starts
        logger.info("Unloading audio models for VRAM handoff...")
        self._audio_engine.unload()

        # ===== PHASE 2: Visual Generation =====
        logger.info("Phase 2: Generating visuals via ComfyUI...")

        # Build workflow payload
        workflow = self._workflow_builder.build(
            prompt=prompt,
            audio_path=clean_audio_path,  # Clean audio for lip-sync
            seed=seed,
        )

        # Dispatch to ComfyUI
        video_path = await self._comfy_client.generate(workflow)
        logger.info(f"Video generated: {video_path}")

        # ===== PHASE 3: Post-processing =====
        # TODO: Mux mixed audio into final video (replace ComfyUI audio track)
        # For MVP, ComfyUI's VHS_VideoCombine uses the clean audio directly

        return {
            "video_path": video_path,
            "clean_audio_path": clean_audio_path,
            "mixed_audio_path": mixed_audio_path,
            "frame_count": frame_count,
            "seed": seed,
        }

    async def health_check(self) -> dict[str, bool]:
        """Check health of all components.

        Returns:
            Dict with component health status
        """
        comfy_ok = await self._comfy_client.check_health()

        return {
            "orchestrator": True,
            "comfyui": comfy_ok,
        }
```

**Step 4: Run test to verify it passes**

```bash
cd /home/matt/nsn/vortex
python -m pytest tests/unit/test_orchestrator.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/vortex/orchestrator.py tests/unit/test_orchestrator.py
git commit -m "feat: add VideoOrchestrator for ToonGen pipeline"
```

---

## Phase 4: Containerization

Package everything in Docker containers.

### Task 4.1: Create ComfyUI Dockerfile

**Files:**
- Create: `vortex/docker/Dockerfile.comfyui`

**Step 1: Create docker directory**

```bash
mkdir -p /home/matt/nsn/vortex/docker
```

**Step 2: Write Dockerfile**

Create `vortex/docker/Dockerfile.comfyui`:

```dockerfile
# ToonGen ComfyUI Server
# GPU-enabled container for visual generation pipeline

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="NSN Team"
LABEL description="ComfyUI server for ToonGen visual generation"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

WORKDIR /app/ComfyUI

# Install ComfyUI requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for custom nodes
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    insightface \
    mediapipe \
    opencv-python-headless

# Install ComfyUI Manager
RUN cd custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# Install required custom nodes
RUN cd custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-LivePortraitKJ.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# Install custom node requirements
RUN cd custom_nodes/ComfyUI-LivePortraitKJ && \
    pip install --no-cache-dir -r requirements.txt || true
RUN cd custom_nodes/ComfyUI-AnimateDiff-Evolved && \
    pip install --no-cache-dir -r requirements.txt || true
RUN cd custom_nodes/ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r requirements.txt || true

# Create model directories (will be mounted as volumes)
RUN mkdir -p /app/ComfyUI/models/checkpoints \
    /app/ComfyUI/models/liveportrait \
    /app/ComfyUI/models/animatediff_models \
    /app/ComfyUI/models/clip \
    /app/ComfyUI/input \
    /app/ComfyUI/output

# Expose port
EXPOSE 8188

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8188/system_stats || exit 1

# Start ComfyUI in headless mode
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
```

**Step 3: Commit**

```bash
git add docker/Dockerfile.comfyui
git commit -m "feat: add ComfyUI Dockerfile for ToonGen"
```

---

### Task 4.2: Create Orchestrator Dockerfile

**Files:**
- Create: `vortex/docker/Dockerfile.orchestrator`

**Step 1: Write Dockerfile**

Create `vortex/docker/Dockerfile.orchestrator`:

```dockerfile
# ToonGen Orchestrator
# Python service for audio generation and ComfyUI coordination

FROM python:3.11-slim

LABEL maintainer="NSN Team"
LABEL description="ToonGen orchestrator for audio generation and pipeline coordination"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY templates/ ./templates/
COPY assets/ ./assets/

# Create output directories
RUN mkdir -p /app/outputs /app/temp

# Set Python path
ENV PYTHONPATH=/app/src

# Expose API port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:50051/health')" || exit 1

# Start server
CMD ["python", "-m", "vortex.server", "--host", "0.0.0.0", "--port", "50051"]
```

**Step 2: Commit**

```bash
git add docker/Dockerfile.orchestrator
git commit -m "feat: add orchestrator Dockerfile for ToonGen"
```

---

### Task 4.3: Create Docker Compose Configuration

**Files:**
- Create: `vortex/docker-compose.yml`

**Step 1: Write docker-compose.yml**

Create `vortex/docker-compose.yml`:

```yaml
# ToonGen Docker Compose Configuration
# Two-container architecture: orchestrator + ComfyUI

version: '3.8'

services:
  # ComfyUI Visual Generation Server
  comfyui:
    build:
      context: .
      dockerfile: docker/Dockerfile.comfyui
    container_name: toongen-comfyui

    # GPU access
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    # Port mapping
    ports:
      - "8188:8188"

    # Model volumes (external, not baked into image)
    volumes:
      - ./models/checkpoints:/app/ComfyUI/models/checkpoints:ro
      - ./models/liveportrait:/app/ComfyUI/models/liveportrait:ro
      - ./models/animatediff_models:/app/ComfyUI/models/animatediff_models:ro
      - ./models/clip:/app/ComfyUI/models/clip:ro
      # Input/Output (shared with orchestrator)
      - ./temp/comfy_input:/app/ComfyUI/input
      - ./outputs:/app/ComfyUI/output

    # Restart policy
    restart: unless-stopped

    # Health check
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8188/system_stats"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  # ToonGen Orchestrator
  orchestrator:
    build:
      context: .
      dockerfile: docker/Dockerfile.orchestrator
    container_name: toongen-orchestrator

    # GPU access (for F5-TTS audio generation)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    # Port mapping
    ports:
      - "50051:50051"

    # Volumes
    volumes:
      - ./assets:/app/assets:ro
      - ./templates:/app/templates:ro
      - ./temp/comfy_input:/app/temp/comfy_input
      - ./outputs:/app/outputs

    # Environment
    environment:
      - COMFY_HOST=comfyui
      - COMFY_PORT=8188
      - VORTEX_DEVICE=cuda:0

    # Dependencies
    depends_on:
      comfyui:
        condition: service_healthy

    # Restart policy
    restart: unless-stopped

# Named volumes for persistence
volumes:
  models:
  outputs:
```

**Step 2: Create .dockerignore**

Create `vortex/.dockerignore`:

```
# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
.pytest_cache
.mypy_cache
.ruff_cache
*.egg-info
.eggs
dist
build
.venv
venv

# IDE
.vscode
.idea
*.swp

# Testing
tests/
htmlcov/
.coverage

# Documentation
docs/
*.md
!README.md

# Large files (use volumes instead)
models/
*.safetensors
*.ckpt
*.pth
*.bin

# Outputs
outputs/
temp/
*.mp4
*.wav
*.npy
```

**Step 3: Commit**

```bash
git add docker-compose.yml .dockerignore
git commit -m "feat: add docker-compose configuration for ToonGen"
```

---

## Phase 5: Content & Integration

Add scenario content and update the API.

### Task 5.1: Create Scenario Library

**Files:**
- Create: `vortex/assets/scenarios.json`

**Step 1: Create scenarios file**

Create `vortex/assets/scenarios.json`:

```json
{
  "version": "1.0.0",
  "description": "ToonGen MVP Scenario Library - 25 Interdimensional Cable scenarios",
  "scenarios": [
    {
      "id": "ants_in_eyes",
      "name": "Ants in My Eyes Johnson",
      "prompt": "Medium shot, 90s commercial style, a frantic bald man with bulging eyes surrounded by floating price tags, in an electronics store, talking directly to camera, cartoon style",
      "script": "I'm Ants in My Eyes Johnson! Everything's black, I can't see a thing! And also I can't feel anything either! But that's not as catchy!",
      "voice_style": "manic_salesman",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "fake_doors",
      "name": "Real Fake Doors",
      "prompt": "Medium shot, infomercial style, an enthusiastic salesman in a polo shirt gesturing at rows of doors, warehouse setting, cartoon style, bright lighting",
      "script": "Hey, are you tired of real doors cluttering up your house? Come on down to Real Fake Doors! That's us!",
      "voice_style": "excited_host",
      "bgm": "80s_synth",
      "sfx": "static_glitch"
    },
    {
      "id": "personal_space",
      "name": "Personal Space Show",
      "prompt": "Close up shot, talk show set, a nervous host in a suit with wide eyes, studio background with 'Personal Space' logo, cartoon style",
      "script": "Hi, I'm Phillip Jacobs and this is Personal Space. One: Personal Space. Two: Personal Space. Three: Stay out of my personal space.",
      "voice_style": "nervous_morty",
      "bgm": "static_void_hum",
      "sfx": null
    },
    {
      "id": "gazorpazorp",
      "name": "Gazorpazorpfield",
      "prompt": "Medium shot, sitcom living room, an angry orange alien cat-creature on a couch pointing at the viewer, retro TV frame border, cartoon style",
      "script": "Hey Jon, it's me, Gazorpazorpfield. Boy, do I hate Mondays. Now give me my enchiladas.",
      "voice_style": "angry_chef",
      "bgm": null,
      "sfx": null
    },
    {
      "id": "turbulent_juice",
      "name": "Turbulent Juice",
      "prompt": "Medium shot, extreme sports commercial, an excited spokesman holding a glowing bottle, mountain backdrop with explosions, 90s aesthetic, cartoon style",
      "script": "Drink Turbulent Juice! It's got what plants crave! Turbulence! Side effects may include turbulence.",
      "voice_style": "excited_host",
      "bgm": "80s_synth",
      "sfx": "explosion_short"
    },
    {
      "id": "baby_legs",
      "name": "Baby Legs Detective",
      "prompt": "Medium shot, noir detective office, a serious detective with comically tiny legs sitting at a desk, dim lighting, rain on window, cartoon style",
      "script": "The name's Baby Legs. I'm the best damn detective on the force... but I got these tiny baby legs.",
      "voice_style": "deadpan_narrator",
      "bgm": "static_void_hum",
      "sfx": null
    },
    {
      "id": "plumbus",
      "name": "How They Do It: Plumbus",
      "prompt": "Wide shot, factory documentary style, a narrator in a lab coat gesturing at strange machinery, industrial background, educational TV aesthetic, cartoon style",
      "script": "First, they take the dinglebop and smooth it out with a bunch of schleem. Everyone has a plumbus in their home.",
      "voice_style": "monotone_bot",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "scary_terry",
      "name": "Scary Terry Nightmares",
      "prompt": "Medium shot, horror movie set, a tiny sweater-wearing monster with knife fingers pointing aggressively, dark misty background, 80s horror poster style, cartoon",
      "script": "Welcome to your nightmare, bitch! I'm Scary Terry, bitch! You can run but you can't hide, bitch!",
      "voice_style": "angry_chef",
      "bgm": "static_void_hum",
      "sfx": "static_glitch"
    },
    {
      "id": "ball_fondlers",
      "name": "Ball Fondlers Trailer",
      "prompt": "Action movie poster shot, four tough mercenaries in tactical gear posing with weapons, explosion background, 80s action movie style, cartoon",
      "script": "This summer, the Ball Fondlers are back. And this time, it's personal. Ball Fondlers Three: The Re-Fondling.",
      "voice_style": "deadpan_narrator",
      "bgm": "80s_synth",
      "sfx": "explosion_short"
    },
    {
      "id": "eyehole_man",
      "name": "Eyehole Man Warning",
      "prompt": "Medium shot, 50s PSA style, a creepy figure in an eyeball costume lurking in shadows, warning sign aesthetic, grainy film look, cartoon style",
      "script": "Get up on outta here with my Eyeholes! I'm the Eyehole Man! I'm the only one that's allowed to have Eyeholes!",
      "voice_style": "whispering_creep",
      "bgm": null,
      "sfx": "static_glitch"
    },
    {
      "id": "interdimensional_ad_1",
      "name": "Glip Glop Cereal",
      "prompt": "Medium shot, breakfast cereal commercial, an alien creature happily eating from a bowl, kitchen table setting, bright morning light, cartoon style",
      "script": "New Glip Glop cereal! Part of a complete breakfast in dimensions C-137 through C-500!",
      "voice_style": "excited_host",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "interdimensional_ad_2",
      "name": "Schmeckles Bank",
      "prompt": "Medium shot, bank commercial, a friendly alien banker at a desk with stacks of strange coins, professional office setting, cartoon style",
      "script": "At Schmeckles Bank, your money is always worth exactly what we say it is. Trust us. We're a bank.",
      "voice_style": "monotone_bot",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "interdimensional_ad_3",
      "name": "Interdimensional Travel Agency",
      "prompt": "Medium shot, travel agency commercial, an enthusiastic agent pointing at a portal, colorful dimension thumbnails on wall, cartoon style",
      "script": "Tired of your dimension? Visit ours! Interdimensional Travel Agency - where every trip is a one-way adventure!",
      "voice_style": "manic_salesman",
      "bgm": "80s_synth",
      "sfx": "static_glitch"
    },
    {
      "id": "interdimensional_ad_4",
      "name": "Blips and Chitz Arcade",
      "prompt": "Wide shot, neon arcade interior, an excited gamer at a Roy cabinet, flashing lights everywhere, 80s arcade aesthetic, cartoon style",
      "script": "Blips and Chitz! Where you can be Roy! A man! And live an entire life! Then do it again!",
      "voice_style": "excited_host",
      "bgm": "80s_synth",
      "sfx": null
    },
    {
      "id": "interdimensional_ad_5",
      "name": "Simple Rick's Wafers",
      "prompt": "Close up, premium snack commercial, a wafer cookie with a small Rick figure inside glowing, warm nostalgic lighting, cartoon style",
      "script": "Come home to Simple Rick's. Come home to the impossible flavor of your own completion.",
      "voice_style": "deadpan_narrator",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "news_broadcast_1",
      "name": "Interdimensional News Update",
      "prompt": "Medium shot, news desk setting, a professional alien news anchor with multiple eyes, news studio background with monitors, cartoon style",
      "script": "Breaking news from Dimension J-19-Zeta-7. The war is over. Both sides have agreed to disagree. More at eleven.",
      "voice_style": "monotone_bot",
      "bgm": null,
      "sfx": "static_glitch"
    },
    {
      "id": "cooking_show",
      "name": "Cooking with Squanchy",
      "prompt": "Medium shot, cooking show set, a cat-like creature in a chef hat holding a pan, kitchen background with flames, cartoon style",
      "script": "Today we're gonna squanch up some eggs. You gotta squanch 'em real good. That's the squanchy way.",
      "voice_style": "surfer_dude",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "lawyer_ad",
      "name": "Morty's Law Firm",
      "prompt": "Medium shot, lawyer commercial, a nervous young man in an oversized suit at a desk, law office backdrop, cheesy local ad aesthetic, cartoon style",
      "script": "Have you been injured in an interdimensional accident? Call Morty's Law Firm. I-I-I'll probably lose your case.",
      "voice_style": "nervous_morty",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "mattress_store",
      "name": "Sleepy Gary's Mattresses",
      "prompt": "Medium shot, mattress store commercial, a suspiciously friendly salesman next to a bed, warehouse full of mattresses, cartoon style",
      "script": "I'm Sleepy Gary! You don't remember me, but I've always been here. Come to Sleepy Gary's. We have beds.",
      "voice_style": "whispering_creep",
      "bgm": "static_void_hum",
      "sfx": null
    },
    {
      "id": "car_dealership",
      "name": "Wacky Waving Dealership",
      "prompt": "Wide shot, car dealership lot, an inflatable tube man next to used spaceships, colorful flags and banners, cartoon style",
      "script": "Wacky Waving Inflatable Arm Flailing Tube Men! Now with legs! Come get your used spacecraft today!",
      "voice_style": "manic_salesman",
      "bgm": "80s_synth",
      "sfx": null
    },
    {
      "id": "therapy_ad",
      "name": "Dr. Wong's Therapy",
      "prompt": "Medium shot, therapy office, a calm professional therapist with glasses taking notes, peaceful office setting, cartoon style",
      "script": "I'm Dr. Wong. I'm not going to tell you what you want to hear. I'm going to tell you the truth. Call now.",
      "voice_style": "deadpan_narrator",
      "bgm": "static_void_hum",
      "sfx": null
    },
    {
      "id": "fast_food",
      "name": "Shoney's All Day Breakfast",
      "prompt": "Medium shot, fast food commercial, a greasy breakfast plate with eggs and bacon, diner booth setting, nostalgic americana, cartoon style",
      "script": "Shoney's. Where the food is real. Unlike everything else. Shoney's: We're pretty sure we exist.",
      "voice_style": "old_timey",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "alien_greeting",
      "name": "Earth Welcome Message",
      "prompt": "Medium shot, PSA style, a confused alien diplomat in formal robes reading from a paper, UN-style podium, cartoon style",
      "script": "Greetings, Earth people. We come in peace. Please stop sending us your reality television. It is confusing.",
      "voice_style": "alien_visitor",
      "bgm": "static_void_hum",
      "sfx": null
    },
    {
      "id": "dating_app",
      "name": "Glorp Dating App",
      "prompt": "Medium shot, dating app commercial, a lonely blob creature holding a phone with hearts floating around, apartment setting, cartoon style",
      "script": "Glorp: The dating app for beings who might be carbon-based. Swipe right if you have hands!",
      "voice_style": "excited_host",
      "bgm": "cheesy_elevator",
      "sfx": null
    },
    {
      "id": "finale",
      "name": "Stay Tuned",
      "prompt": "Close up, TV static transitioning to test pattern, vintage TV frame, interdimensional cable logo, retro broadcast aesthetic, cartoon style",
      "script": "You're watching Interdimensional Cable. Don't touch that dial. Actually, do whatever you want. We're not your parents.",
      "voice_style": "monotone_bot",
      "bgm": "static_void_hum",
      "sfx": "static_glitch"
    }
  ]
}
```

**Step 2: Commit**

```bash
git add assets/scenarios.json
git commit -m "feat: add 25 MVP scenarios for Interdimensional Cable"
```

---

### Task 5.2: Update Recipe Schema for ToonGen

**Files:**
- Modify: `vortex/src/vortex/renderers/recipe_schema.py`

**Step 1: Update schema**

Replace `vortex/src/vortex/renderers/recipe_schema.py` with ToonGen-compatible schema:

```python
"""Standardized recipe schema for ToonGen video generation.

This schema defines the API contract between clients and the ToonGen
orchestrator. It replaces the legacy Vortex schema with fields
appropriate for ComfyUI-based workflow orchestration.
"""

from __future__ import annotations

from typing import Any

# JSON Schema for ToonGen recipe validation
RECIPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["slot_params", "audio_track", "visual_track"],
    "properties": {
        "slot_params": {
            "type": "object",
            "required": ["slot_id"],
            "properties": {
                "slot_id": {
                    "type": "integer",
                    "description": "Unique slot identifier",
                },
                "fps": {
                    "type": "integer",
                    "default": 24,
                    "description": "Frames per second",
                },
                "seed": {
                    "type": "integer",
                    "description": "Deterministic seed (random if not provided)",
                },
            },
        },
        "audio_track": {
            "type": "object",
            "required": ["script"],
            "properties": {
                "script": {
                    "type": "string",
                    "description": "Text to synthesize as speech",
                },
                "engine": {
                    "type": "string",
                    "enum": ["auto", "f5_tts", "kokoro"],
                    "default": "auto",
                    "description": "TTS engine selection (auto tries F5 first, falls back to Kokoro)",
                },
                "voice_style": {
                    "type": "string",
                    "description": "F5-TTS voice reference filename (without .wav)",
                },
                "voice_id": {
                    "type": "string",
                    "default": "af_heart",
                    "description": "Kokoro voice ID (used if engine=kokoro or as fallback)",
                },
            },
        },
        "audio_environment": {
            "type": "object",
            "properties": {
                "bgm": {
                    "type": "string",
                    "description": "Background music filename (without .wav)",
                },
                "sfx": {
                    "type": "string",
                    "description": "Sound effect filename (without .wav)",
                },
                "mix_ratio": {
                    "type": "number",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "BGM volume relative to voice",
                },
            },
        },
        "visual_track": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Full scene prompt for Flux image generation",
                },
                "negative_prompt": {
                    "type": "string",
                    "default": "blurry, low quality, distorted face, extra limbs",
                    "description": "Negative prompt for image generation",
                },
            },
        },
    },
}


def validate_recipe(recipe: dict[str, Any]) -> list[str]:
    """Validate recipe against ToonGen schema, returning list of errors.

    Args:
        recipe: Recipe dict to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Check required top-level fields
    for field in ("slot_params", "audio_track", "visual_track"):
        if field not in recipe:
            errors.append(f"Missing required field: {field}")
            continue
        if not isinstance(recipe[field], dict):
            errors.append(f"Field '{field}' must be an object")

    if errors:
        return errors

    # Validate slot_params
    slot_params = recipe["slot_params"]
    if "slot_id" not in slot_params:
        errors.append("Missing required field: slot_params.slot_id")
    elif not isinstance(slot_params.get("slot_id"), int):
        errors.append("slot_params.slot_id must be an integer")

    # Validate audio_track
    audio_track = recipe["audio_track"]
    if "script" not in audio_track:
        errors.append("Missing required field: audio_track.script")
    elif not isinstance(audio_track.get("script"), str):
        errors.append("audio_track.script must be a string")

    # Validate visual_track
    visual_track = recipe["visual_track"]
    if "prompt" not in visual_track:
        errors.append("Missing required field: visual_track.prompt")
    elif not isinstance(visual_track.get("prompt"), str):
        errors.append("visual_track.prompt must be a string")

    return errors


def get_recipe_defaults() -> dict[str, Any]:
    """Return a recipe dict with all defaults filled in.

    Returns:
        Recipe dict with default values for all optional fields
    """
    return {
        "slot_params": {
            "slot_id": 0,
            "fps": 24,
        },
        "audio_track": {
            "script": "",
            "engine": "auto",
            "voice_id": "af_heart",
        },
        "audio_environment": {
            "bgm": None,
            "sfx": None,
            "mix_ratio": 0.3,
        },
        "visual_track": {
            "prompt": "",
            "negative_prompt": "blurry, low quality, distorted face, extra limbs",
        },
    }


def merge_with_defaults(recipe: dict[str, Any]) -> dict[str, Any]:
    """Merge recipe with defaults for missing optional fields.

    Args:
        recipe: Partial recipe dict

    Returns:
        Complete recipe dict with defaults applied
    """
    defaults = get_recipe_defaults()
    result: dict[str, Any] = {}

    for section in defaults:
        if section not in recipe:
            result[section] = defaults[section].copy()
        else:
            result[section] = defaults[section].copy()
            result[section].update(recipe[section])

    return result
```

**Step 2: Commit**

```bash
git add src/vortex/renderers/recipe_schema.py
git commit -m "refactor: update recipe schema for ToonGen architecture"
```

---

### Task 5.3: Update Server API for ToonGen

**Files:**
- Modify: `vortex/src/vortex/server.py`

**Step 1: Update server to use VideoOrchestrator**

This is a significant refactor. The key changes:
- Replace `VortexPipeline` with `VideoOrchestrator`
- Change response format from NPY paths to MP4 path
- Add health check for ComfyUI

(Full implementation would be similar to existing server.py but using the new orchestrator. For brevity, showing the key changes.)

**Step 2: Commit**

```bash
git add src/vortex/server.py
git commit -m "refactor: update server API for ToonGen orchestrator"
```

---

### Task 5.4: Create Integration Test

**Files:**
- Create: `vortex/tests/integration/test_comfy_connection.py`
- Create: `vortex/scripts/test_e2e.py`

**Step 1: Create integration test**

Create `vortex/tests/integration/test_comfy_connection.py`:

```python
"""Integration test for ComfyUI connection."""

import pytest
import requests


@pytest.mark.integration
def test_comfyui_is_running():
    """Verify ComfyUI server is accessible."""
    try:
        response = requests.get(
            "http://127.0.0.1:8188/system_stats",
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("ComfyUI is not running on port 8188")
```

**Step 2: Create E2E smoke test**

Create `vortex/scripts/test_e2e.py`:

```python
#!/usr/bin/env python3
"""End-to-end smoke test for ToonGen pipeline.

Usage:
    python scripts/test_e2e.py

Requires:
    - ComfyUI running on localhost:8188
    - Audio models available
    - Workflow template configured
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vortex.orchestrator import VideoOrchestrator


async def main():
    """Run E2E smoke test."""
    print("=" * 60)
    print("ToonGen E2E Smoke Test")
    print("=" * 60)

    orchestrator = VideoOrchestrator(
        template_path="templates/cartoon_workflow.json",
        assets_dir="assets",
        output_dir="outputs/test",
    )

    # Health check
    print("\n[1/3] Checking component health...")
    health = await orchestrator.health_check()
    print(f"  Orchestrator: {'OK' if health['orchestrator'] else 'FAIL'}")
    print(f"  ComfyUI: {'OK' if health['comfyui'] else 'FAIL'}")

    if not health["comfyui"]:
        print("\nERROR: ComfyUI is not running. Start it with:")
        print("  docker-compose up comfyui")
        return 1

    # Generate test video
    print("\n[2/3] Generating test video...")
    print("  Prompt: 'A cartoon cat in a lab coat'")
    print("  Script: 'This is a test of the ToonGen system.'")

    result = await orchestrator.generate(
        prompt="Medium shot, cartoon style, a friendly cat in a lab coat waving at camera, bright laboratory background",
        script="This is a test of the ToonGen system. If you can hear this, everything is working correctly.",
        engine="kokoro",  # Use Kokoro for quick test
        seed=12345,
    )

    print(f"  Video: {result['video_path']}")
    print(f"  Frames: {result['frame_count']}")

    # Verify output
    print("\n[3/3] Verifying output...")
    video_path = Path(result["video_path"])

    if not video_path.exists():
        print(f"  ERROR: Video file not found: {video_path}")
        return 1

    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")

    if size_mb < 0.1:
        print("  WARNING: Video file seems too small")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
```

**Step 3: Make executable**

```bash
chmod +x /home/matt/nsn/vortex/scripts/test_e2e.py
```

**Step 4: Commit**

```bash
git add tests/integration/ scripts/test_e2e.py
git commit -m "feat: add integration and E2E tests for ToonGen"
```

---

## Summary

This plan implements the ToonGen architecture pivot in 5 phases:

1. **Phase 0: Legacy Archive** - Create archive branch, delete old code
2. **Phase 1: Manual Proof** - Install ComfyUI, build and validate workflow
3. **Phase 2: Audio Engine** - F5-TTS/Kokoro with graceful degradation, FFmpeg mixing
4. **Phase 3: Orchestrator** - ComfyClient, WorkflowBuilder, VideoOrchestrator
5. **Phase 4: Containerization** - Docker setup with volume-mounted models
6. **Phase 5: Content & Integration** - 25 scenarios, updated API, tests

### Verification Commands

```bash
# Unit tests
cd /home/matt/nsn/vortex
python -m pytest tests/unit/ -v

# Integration tests (requires ComfyUI running)
python -m pytest tests/integration/ -v -m integration

# E2E smoke test
python scripts/test_e2e.py

# Full Docker deployment
docker-compose up --build
```

### Success Criteria

- [ ] All unit tests pass
- [ ] ComfyUI workflow generates video on RTX 3060
- [ ] Audio engine produces clean voice + mixed output
- [ ] E2E test completes successfully
- [ ] Docker containers start and communicate
- [ ] Recognizable lip-sync in output video
- [ ] 95%+ reliability across 20 test generations
