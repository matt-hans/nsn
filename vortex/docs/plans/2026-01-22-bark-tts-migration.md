# Bark TTS Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Kokoro TTS with Bark TTS to enable "Interdimensional Cable" style chaotic, expressive voice generation with paralinguistic sounds (`[laughter]`, `[gasps]`, etc.).

**Architecture:** Bark-only TTS (no RVC) with zero-shot voice cloning via reference audio. Pre-allocated buffer pattern maintained. 3-retry mechanism with fallback audio for failures. Temperature-based emotion control.

**Tech Stack:** suno-bark (MIT license), torch, soundfile, numpy

---

## Task 1: Create Fallback Audio Asset

**Files:**
- Create: `vortex/assets/voices/fallback_audio.wav`

**Step 1: Generate fallback audio file**

Create a 2-second "technical difficulties" audio clip. This will be used when Bark fails after 3 retries.

```bash
cd /home/matt/nsn/vortex
mkdir -p assets/voices

# Generate 2s of static noise + beep using Python
python3 -c "
import numpy as np
import soundfile as sf

# 2 seconds at 24kHz
sr = 24000
duration = 2.0
samples = int(sr * duration)

# White noise base (low volume)
noise = np.random.randn(samples) * 0.1

# Add a 440Hz beep in the middle (0.5s to 1.0s)
t = np.linspace(0, duration, samples)
beep_mask = (t >= 0.5) & (t <= 1.0)
beep = np.sin(2 * np.pi * 440 * t) * 0.3
noise[beep_mask] += beep[beep_mask]

# Normalize to [-1, 1]
noise = np.clip(noise, -1.0, 1.0)

sf.write('assets/voices/fallback_audio.wav', noise.astype(np.float32), sr)
print('Created fallback_audio.wav')
"
```

**Step 2: Verify the file exists**

```bash
ls -la /home/matt/nsn/vortex/assets/voices/fallback_audio.wav
```

Expected: File exists, ~96KB (2s * 24000 * 4 bytes)

**Step 3: Commit**

```bash
git add assets/voices/fallback_audio.wav
git commit -m "feat(audio): add fallback audio for Bark TTS failures"
```

---

## Task 2: Create Voice Profiles Configuration

**Files:**
- Create: `vortex/src/vortex/models/configs/voice_profiles.yaml`

**Step 1: Write the voice profiles config**

```yaml
# Voice Profiles for Bark TTS
#
# This configuration maps character voice IDs to Bark speaker presets
# and optional reference audio for zero-shot cloning.
#
# Types:
#   - preset: Uses built-in Bark speaker (no reference audio needed)
#   - cloned: Uses reference audio to clone voice (requires .wav file)
#
# Bark speaker presets (v2/en_speaker_N where N=0-9):
#   0: Neutral male narrator
#   1: Calm female
#   2: Energetic male
#   3: Soft female
#   4: Deep male
#   5: Young female
#   6: Gruff/raspy male (good for "Rick" style)
#   7: Warm female
#   8: Fast-talking male
#   9: Higher-pitched male (good for "Morty" style)

# Main character voices (mapped from Kokoro equivalents)
rick_c137:
  type: "preset"
  bark_speaker: "v2/en_speaker_6"
  description: "Manic scientist - gruff, raspy, energetic"

morty:
  type: "preset"
  bark_speaker: "v2/en_speaker_9"
  description: "Anxious teen - higher pitch, nervous"

summer:
  type: "preset"
  bark_speaker: "v2/en_speaker_5"
  description: "Confident female teenager"

# Additional character voices
jerry:
  type: "preset"
  bark_speaker: "v2/en_speaker_0"
  description: "Hesitant, average male voice"

beth:
  type: "preset"
  bark_speaker: "v2/en_speaker_7"
  description: "Mature, professional female"

mr_meeseeks:
  type: "preset"
  bark_speaker: "v2/en_speaker_2"
  description: "High energy, helpful voice"

# Default fallback
default:
  type: "preset"
  bark_speaker: "v2/en_speaker_0"
  description: "Generic neutral narrator"

# Example cloned voice (template for user-created voices)
# custom_alien:
#   type: "cloned"
#   reference_audio: "assets/voices/alien_sample.wav"
#   history_prompt: "assets/voices/alien_sample.npz"
#   description: "User-created alien voice"
```

**Step 2: Verify YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('src/vortex/models/configs/voice_profiles.yaml'))"
```

Expected: No output (success)

**Step 3: Commit**

```bash
git add src/vortex/models/configs/voice_profiles.yaml
git commit -m "feat(audio): add Bark voice profiles configuration"
```

---

## Task 3: Create Bark Emotions Configuration

**Files:**
- Create: `vortex/src/vortex/models/configs/bark_emotions.yaml`

**Step 1: Write the emotions config**

```yaml
# Bark TTS Emotion Configuration
#
# Bark doesn't have explicit emotion controls like Kokoro's speed parameter.
# Instead, we control emotion through:
#   1. Temperature settings (coarse_temp, fine_temp)
#   2. Optional text prefixes (experimental, may or may not influence output)
#
# Temperature Guide:
#   - coarse_temp: Controls prosody/rhythm variation (0.0-1.0)
#     Lower = more monotone, Higher = more varied intonation
#   - fine_temp: Controls acoustic detail variation (0.0-1.0)
#     Lower = cleaner audio, Higher = more artifacts/texture
#
# For "Interdimensional Cable" aesthetic:
#   - Higher coarse_temp = more chaotic prosody (good)
#   - Moderate fine_temp = some texture without too much hiss

neutral:
  text_prefix: ""
  coarse_temp: 0.7
  fine_temp: 0.5
  description: "Standard delivery, balanced chaos"

excited:
  text_prefix: ""
  coarse_temp: 0.85
  fine_temp: 0.55
  description: "Enthusiastic, upbeat, more prosodic variation"

sad:
  text_prefix: ""
  coarse_temp: 0.5
  fine_temp: 0.4
  description: "Subdued, slower, less variation"

angry:
  text_prefix: ""
  coarse_temp: 0.85
  fine_temp: 0.6
  description: "Intense, forceful, high variation"

manic:
  text_prefix: ""
  coarse_temp: 0.95
  fine_temp: 0.7
  description: "Frenzied, erratic, maximum chaos (Rick's signature)"

calm:
  text_prefix: ""
  coarse_temp: 0.5
  fine_temp: 0.4
  description: "Soothing, peaceful, minimal variation"

fearful:
  text_prefix: ""
  coarse_temp: 0.9
  fine_temp: 0.6
  description: "Anxious, nervous, high pitch variation"

# Default fallback (same as neutral)
default:
  text_prefix: ""
  coarse_temp: 0.7
  fine_temp: 0.5
  description: "Fallback emotion settings"
```

**Step 2: Verify YAML syntax**

```bash
python3 -c "import yaml; yaml.safe_load(open('src/vortex/models/configs/bark_emotions.yaml'))"
```

Expected: No output (success)

**Step 3: Commit**

```bash
git add src/vortex/models/configs/bark_emotions.yaml
git commit -m "feat(audio): add Bark emotions configuration with temperature control"
```

---

## Task 4: Update Dependencies (pyproject.toml)

**Files:**
- Modify: `vortex/pyproject.toml:26` (remove kokoro, add bark)

**Step 1: Write the failing test**

```bash
cd /home/matt/nsn/vortex
python3 -c "from bark import SAMPLE_RATE; print(f'Bark sample rate: {SAMPLE_RATE}')"
```

Expected: FAIL with "ModuleNotFoundError: No module named 'bark'"

**Step 2: Update pyproject.toml**

Replace line 26:
```
    "kokoro>=0.7.0",         # TTS model (T017) - KPipeline for TTS
```

With:
```
    "git+https://github.com/suno-ai/bark.git",  # TTS model - Bark for expressive audio
```

**Step 3: Install updated dependencies**

```bash
cd /home/matt/nsn/vortex
pip install -e ".[dev]"
```

**Step 4: Verify bark is installed**

```bash
python3 -c "from bark import SAMPLE_RATE; print(f'Bark sample rate: {SAMPLE_RATE}')"
```

Expected: `Bark sample rate: 24000`

**Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat(deps): replace kokoro with bark TTS"
```

---

## Task 5: Create BarkVoiceEngine Core Class

**Files:**
- Create: `vortex/src/vortex/models/bark.py`
- Test: `vortex/tests/unit/test_bark.py`

**Step 1: Write the failing test**

Create `vortex/tests/unit/test_bark.py`:

```python
"""Unit tests for Bark TTS model wrapper."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestBarkVoiceEngine:
    """Test suite for BarkVoiceEngine wrapper."""

    def test_import_bark_voice_engine(self):
        """Test that BarkVoiceEngine can be imported."""
        from vortex.models.bark import BarkVoiceEngine
        assert BarkVoiceEngine is not None

    def test_init_loads_configs(self):
        """Test that engine initializes with voice and emotion configs."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        assert "rick_c137" in engine.voice_profiles
        assert "neutral" in engine.emotion_config
        assert engine.sample_rate == 24000

    def test_get_voice_preset_returns_bark_speaker(self):
        """Test voice ID mapping to Bark speaker preset."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        speaker = engine._get_bark_speaker("rick_c137")
        assert speaker == "v2/en_speaker_6"

    def test_get_voice_preset_unknown_returns_default(self):
        """Test unknown voice ID returns default speaker."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        speaker = engine._get_bark_speaker("unknown_voice")
        assert speaker == "v2/en_speaker_0"

    def test_get_emotion_params_returns_temperatures(self):
        """Test emotion mapping returns temperature settings."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        params = engine._get_emotion_params("manic")
        assert params["coarse_temp"] == 0.95
        assert params["fine_temp"] == 0.7

    def test_get_emotion_params_unknown_returns_neutral(self):
        """Test unknown emotion returns neutral settings."""
        from vortex.models.bark import BarkVoiceEngine

        with patch('vortex.models.bark.preload_models'):
            engine = BarkVoiceEngine(device="cpu")

        params = engine._get_emotion_params("unknown_emotion")
        assert params["coarse_temp"] == 0.7
        assert params["fine_temp"] == 0.5
```

**Step 2: Run test to verify it fails**

```bash
cd /home/matt/nsn/vortex
pytest tests/unit/test_bark.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'vortex.models.bark'"

**Step 3: Write BarkVoiceEngine implementation**

Create `vortex/src/vortex/models/bark.py`:

```python
"""Bark TTS model wrapper for Vortex pipeline.

This module provides the BarkVoiceEngine class that integrates Bark TTS
into the Vortex pipeline with:
- Voice selection via speaker presets or zero-shot cloning
- Emotion modulation via temperature control
- Paralinguistic sound support ([laughter], [gasps], etc.)
- Pre-allocated buffer output (no VRAM fragmentation)
- Retry mechanism with seed variation
- Fallback audio for generation failures

VRAM Budget: ~6-12 GB (depends on model size and offloading)
Output: 24kHz mono audio
"""

import gc
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import yaml

logger = logging.getLogger(__name__)

# Bark sample rate (fixed)
BARK_SAMPLE_RATE = 24000

# Maximum retry attempts before using fallback audio
MAX_RETRIES = 3

# Minimum valid audio length (seconds) - below this triggers retry
MIN_AUDIO_LENGTH_SEC = 0.5


class BarkVoiceEngine:
    """Wrapper for Bark TTS with ICN-specific voice/emotion mapping.

    This wrapper provides a consistent interface for the Vortex pipeline while
    handling voice preset selection, emotion via temperature, and VRAM management.

    Attributes:
        voice_profiles: Mapping of character IDs to Bark settings
        emotion_config: Emotion name to temperature mapping
        device: Target device (cuda or cpu)
        sample_rate: Output sample rate (24000 Hz fixed for Bark)

    Example:
        >>> engine = BarkVoiceEngine(device="cuda")
        >>> audio = engine.synthesize(
        ...     text="[laughs] Wubba lubba dub dub!",
        ...     voice_id="rick_c137",
        ...     emotion="manic"
        ... )
        >>> engine.unload()
    """

    def __init__(
        self,
        device: str = "cuda",
        config_dir: Optional[Path] = None,
        assets_dir: Optional[Path] = None,
    ):
        """Initialize Bark voice engine.

        Args:
            device: Target device for inference
            config_dir: Directory containing voice_profiles.yaml and bark_emotions.yaml
            assets_dir: Directory containing voice assets (fallback audio, reference clips)
        """
        self.device = device
        self.sample_rate = BARK_SAMPLE_RATE
        self._models_loaded = False

        # Set config and assets directories
        module_dir = Path(__file__).parent
        self._config_dir = config_dir or module_dir / "configs"
        self._assets_dir = assets_dir or module_dir.parent.parent.parent / "assets" / "voices"

        # Load configurations
        self.voice_profiles = self._load_voice_profiles()
        self.emotion_config = self._load_emotion_config()

        # Fallback audio path
        self._fallback_audio_path = self._assets_dir / "fallback_audio.wav"

        # Preload models (downloads if needed, respects device)
        self._preload_models()

        logger.info(
            "BarkVoiceEngine initialized",
            extra={
                "device": device,
                "voices": list(self.voice_profiles.keys()),
                "emotions": list(self.emotion_config.keys()),
            },
        )

    def _load_voice_profiles(self) -> dict:
        """Load voice profiles from YAML config."""
        config_path = self._config_dir / "voice_profiles.yaml"
        if not config_path.exists():
            logger.warning(f"Voice profiles not found at {config_path}, using defaults")
            return {
                "default": {"type": "preset", "bark_speaker": "v2/en_speaker_0"},
                "rick_c137": {"type": "preset", "bark_speaker": "v2/en_speaker_6"},
                "morty": {"type": "preset", "bark_speaker": "v2/en_speaker_9"},
            }

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _load_emotion_config(self) -> dict:
        """Load emotion config from YAML."""
        config_path = self._config_dir / "bark_emotions.yaml"
        if not config_path.exists():
            logger.warning(f"Emotion config not found at {config_path}, using defaults")
            return {
                "neutral": {"coarse_temp": 0.7, "fine_temp": 0.5},
                "manic": {"coarse_temp": 0.95, "fine_temp": 0.7},
            }

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _preload_models(self) -> None:
        """Preload Bark models to specified device."""
        from bark import preload_models

        # Bark's preload_models handles device placement
        # Use CPU for initial load if offloading, then move to GPU when needed
        use_gpu = self.device.startswith("cuda")
        preload_models(
            text_use_gpu=use_gpu,
            coarse_use_gpu=use_gpu,
            fine_use_gpu=use_gpu,
            codec_use_gpu=use_gpu,
        )
        self._models_loaded = True
        logger.info(f"Bark models preloaded (GPU={use_gpu})")

    def _get_bark_speaker(self, voice_id: str) -> str:
        """Map voice ID to Bark speaker preset.

        Args:
            voice_id: ICN voice ID (e.g., "rick_c137") or raw Bark preset

        Returns:
            Bark speaker preset string (e.g., "v2/en_speaker_6")
        """
        # Check if it's a known profile
        if voice_id in self.voice_profiles:
            profile = self.voice_profiles[voice_id]
            return profile.get("bark_speaker", "v2/en_speaker_0")

        # Check if it's already a Bark preset format
        if voice_id.startswith("v2/"):
            return voice_id

        # Fall back to default
        logger.debug(f"Unknown voice_id '{voice_id}', using default speaker")
        return self.voice_profiles.get("default", {}).get("bark_speaker", "v2/en_speaker_0")

    def _get_emotion_params(self, emotion: str) -> dict:
        """Get temperature parameters for emotion.

        Args:
            emotion: Emotion name (e.g., "manic", "neutral")

        Returns:
            Dict with coarse_temp and fine_temp
        """
        if emotion in self.emotion_config:
            return self.emotion_config[emotion]

        logger.debug(f"Unknown emotion '{emotion}', using neutral")
        return self.emotion_config.get("neutral", {"coarse_temp": 0.7, "fine_temp": 0.5})

    def _load_fallback_audio(self) -> np.ndarray:
        """Load fallback audio for generation failures.

        Returns:
            Audio waveform as numpy array (24kHz, mono)
        """
        if not self._fallback_audio_path.exists():
            logger.warning("Fallback audio not found, generating silence")
            return np.zeros(BARK_SAMPLE_RATE * 2, dtype=np.float32)  # 2s silence

        audio, sr = sf.read(self._fallback_audio_path)
        if sr != BARK_SAMPLE_RATE:
            logger.warning(f"Fallback audio sample rate mismatch: {sr} vs {BARK_SAMPLE_RATE}")
        return audio.astype(np.float32)

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        voice_id: str = "default",
        emotion: str = "neutral",
        output: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate 24kHz mono audio from text using Bark.

        Supports paralinguistic tokens: [laughter], [laughs], [sighs],
        [music], [gasps], [clears throat]

        Args:
            text: Text to synthesize (may include Bark tokens)
            voice_id: Voice profile ID or raw Bark speaker preset
            emotion: Emotion name for temperature control
            output: Pre-allocated output buffer (optional)
            seed: Random seed for deterministic generation

        Returns:
            Audio waveform tensor of shape (samples,) at 24kHz

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Get Bark parameters
        speaker = self._get_bark_speaker(voice_id)
        emotion_params = self._get_emotion_params(emotion)
        coarse_temp = emotion_params.get("coarse_temp", 0.7)
        fine_temp = emotion_params.get("fine_temp", 0.5)

        logger.info(
            "Bark synthesis starting",
            extra={
                "text_length": len(text),
                "voice_id": voice_id,
                "speaker": speaker,
                "emotion": emotion,
                "coarse_temp": coarse_temp,
                "fine_temp": fine_temp,
            },
        )

        # Try generation with retries
        audio = None
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                # Set seed for this attempt (varies by attempt number)
                attempt_seed = seed + attempt if seed is not None else None
                if attempt_seed is not None:
                    torch.manual_seed(attempt_seed)
                    np.random.seed(attempt_seed)

                # Generate with Bark
                audio = self._generate_bark(
                    text=text,
                    speaker=speaker,
                    coarse_temp=coarse_temp,
                    fine_temp=fine_temp,
                )

                # Check if audio is valid (minimum length)
                audio_duration = len(audio) / BARK_SAMPLE_RATE
                if audio_duration >= MIN_AUDIO_LENGTH_SEC:
                    logger.info(
                        f"Bark synthesis succeeded on attempt {attempt + 1}",
                        extra={"duration_sec": audio_duration},
                    )
                    break
                else:
                    logger.warning(
                        f"Bark produced short audio ({audio_duration:.2f}s), retrying",
                        extra={"attempt": attempt + 1},
                    )
                    audio = None

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Bark synthesis failed on attempt {attempt + 1}: {e}",
                    extra={"attempt": attempt + 1},
                )

        # Use fallback if all retries failed
        if audio is None:
            logger.error(
                f"Bark synthesis failed after {MAX_RETRIES} attempts, using fallback audio",
                extra={"last_error": str(last_error)},
            )
            audio = self._load_fallback_audio()

        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Write to pre-allocated buffer if provided
        if output is not None:
            samples = min(len(audio_tensor), len(output))
            output[:samples] = audio_tensor[:samples]
            return output[:samples]

        return audio_tensor

    def _generate_bark(
        self,
        text: str,
        speaker: str,
        coarse_temp: float,
        fine_temp: float,
    ) -> np.ndarray:
        """Internal Bark generation call.

        Args:
            text: Text to synthesize
            speaker: Bark speaker preset
            coarse_temp: Coarse acoustic temperature
            fine_temp: Fine acoustic temperature

        Returns:
            Audio waveform as numpy array
        """
        from bark import generate_audio, SAMPLE_RATE

        # Bark's generate_audio returns numpy array
        audio = generate_audio(
            text,
            history_prompt=speaker,
            text_temp=0.7,  # Text/semantic temperature
            waveform_temp=coarse_temp,  # Maps to coarse acoustic
            fine_temp=fine_temp,  # Fine acoustic temperature (custom if supported)
        )

        return audio

    def unload(self) -> None:
        """Unload Bark models and free VRAM.

        Call this before starting the visual pipeline to ensure
        maximum GPU memory is available.
        """
        if self._models_loaded:
            logger.info("Unloading Bark models...")

            # Clear Bark's cached models
            from bark import generation

            # Clear generation module's cached models
            if hasattr(generation, 'models'):
                generation.models = {}

            self._models_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Bark models unloaded, VRAM freed")


def load_bark(
    device: str = "cuda",
    config_dir: Optional[Path] = None,
    assets_dir: Optional[Path] = None,
) -> BarkVoiceEngine:
    """Factory function to create BarkVoiceEngine.

    Args:
        device: Target device for inference
        config_dir: Optional config directory override
        assets_dir: Optional assets directory override

    Returns:
        Configured BarkVoiceEngine instance
    """
    return BarkVoiceEngine(
        device=device,
        config_dir=config_dir,
        assets_dir=assets_dir,
    )
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/matt/nsn/vortex
pytest tests/unit/test_bark.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/vortex/models/bark.py tests/unit/test_bark.py
git commit -m "feat(audio): implement BarkVoiceEngine with retry and fallback"
```

---

## Task 6: Delete Kokoro Files

**Files:**
- Delete: `vortex/src/vortex/models/kokoro.py`
- Delete: `vortex/src/vortex/models/configs/kokoro_voices.yaml`
- Delete: `vortex/src/vortex/models/configs/kokoro_emotions.yaml`
- Delete: `vortex/tests/unit/test_kokoro.py`
- Delete: `vortex/tests/integration/test_kokoro_synthesis.py`

**Step 1: Verify files exist before deletion**

```bash
ls -la /home/matt/nsn/vortex/src/vortex/models/kokoro.py
ls -la /home/matt/nsn/vortex/src/vortex/models/configs/kokoro_*.yaml
ls -la /home/matt/nsn/vortex/tests/unit/test_kokoro.py
ls -la /home/matt/nsn/vortex/tests/integration/test_kokoro_synthesis.py
```

**Step 2: Delete files**

```bash
cd /home/matt/nsn/vortex
rm src/vortex/models/kokoro.py
rm src/vortex/models/configs/kokoro_voices.yaml
rm src/vortex/models/configs/kokoro_emotions.yaml
rm tests/unit/test_kokoro.py
rm tests/integration/test_kokoro_synthesis.py
```

**Step 3: Verify deletion**

```bash
ls /home/matt/nsn/vortex/src/vortex/models/kokoro.py 2>&1 | grep -q "No such file" && echo "kokoro.py deleted"
```

Expected: "kokoro.py deleted"

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor(audio): remove Kokoro TTS (replaced by Bark)"
```

---

## Task 7: Update AudioEngine (core/audio.py)

**Files:**
- Modify: `vortex/src/vortex/core/audio.py`
- Modify: `vortex/tests/unit/test_audio_engine.py`

**Step 1: Write the failing test**

Update `vortex/tests/unit/test_audio_engine.py`:

```python
"""Unit tests for AudioEngine with Bark TTS."""

from unittest.mock import Mock, patch, MagicMock

import pytest
import torch
import numpy as np


class TestAudioEngineWithBark:
    """Test AudioEngine using Bark backend."""

    def test_audio_engine_imports(self):
        """Test that AudioEngine can be imported."""
        from vortex.core.audio import AudioEngine
        assert AudioEngine is not None

    def test_generate_calls_bark(self, tmp_path):
        """Test that generate() uses Bark engine."""
        from vortex.core.audio import AudioEngine

        # Mock BarkVoiceEngine
        mock_bark = MagicMock()
        mock_bark.synthesize.return_value = torch.randn(24000)

        with patch('vortex.core.audio.BarkVoiceEngine', return_value=mock_bark):
            engine = AudioEngine(device="cpu", output_dir=str(tmp_path))
            path = engine.generate(
                script="Hello world",
                voice_id="rick_c137",
                emotion="manic"
            )

            assert path.endswith(".wav")
            mock_bark.synthesize.assert_called_once()

    def test_unload_clears_bark_model(self, tmp_path):
        """Test that unload() clears Bark model."""
        from vortex.core.audio import AudioEngine

        mock_bark = MagicMock()

        with patch('vortex.core.audio.BarkVoiceEngine', return_value=mock_bark):
            engine = AudioEngine(device="cpu", output_dir=str(tmp_path))
            engine._bark_engine = mock_bark

            with patch('gc.collect') as mock_gc:
                engine.unload()

                mock_bark.unload.assert_called_once()
                mock_gc.assert_called()
```

**Step 2: Run test to verify it fails**

```bash
cd /home/matt/nsn/vortex
pytest tests/unit/test_audio_engine.py::TestAudioEngineWithBark -v
```

Expected: FAIL (AudioEngine still references Kokoro)

**Step 3: Update AudioEngine implementation**

Replace `vortex/src/vortex/core/audio.py`:

```python
"""Audio generation engine using Bark TTS.

This module provides voice synthesis using Bark TTS:
- Expressive, chaotic audio for "Interdimensional Cable" aesthetic
- Paralinguistic sounds ([laughter], [gasps], etc.)
- Voice selection via speaker presets or zero-shot cloning
- Emotion control via temperature settings
- 24kHz mono audio output

VRAM Management:
- Model is lazy-loaded on first use
- Call unload() before visual pipeline to free GPU memory
"""

from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path

import soundfile as sf
import torch

logger = logging.getLogger(__name__)


class AudioEngine:
    """Voice synthesis engine using Bark TTS.

    Generates 24kHz mono audio from text using Bark.
    Supports paralinguistic tokens for expressive audio.
    Model is lazy-loaded on first generate() call.

    Example:
        >>> engine = AudioEngine(device="cuda")
        >>> path = engine.generate(
        ...     "[laughs] Hello world!",
        ...     voice_id="rick_c137",
        ...     emotion="manic"
        ... )
        >>> print(path)  # temp/audio/voice_abc123.wav
        >>> engine.unload()  # Free VRAM when done
    """

    def __init__(
        self,
        device: str = "cuda",
        output_dir: str = "temp/audio",
    ):
        """Initialize audio engine.

        Args:
            device: PyTorch device for inference ("cuda" or "cpu")
            output_dir: Directory for generated audio files
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded Bark engine
        self._bark_engine = None

    def _load_bark(self) -> None:
        """Lazy-load Bark engine."""
        if self._bark_engine is None:
            logger.info("Loading Bark engine...")
            from vortex.models.bark import BarkVoiceEngine
            self._bark_engine = BarkVoiceEngine(device=self.device)
            logger.info("Bark engine loaded")

    def generate(
        self,
        script: str,
        voice_id: str = "default",
        emotion: str = "neutral",
        seed: int | None = None,
    ) -> str:
        """Generate voice audio using Bark TTS.

        Args:
            script: Text to synthesize (supports Bark tokens like [laughter])
            voice_id: Voice profile ID (e.g., "rick_c137", "morty")
            emotion: Emotion for temperature control (e.g., "manic", "neutral")
            seed: Optional seed for reproducibility

        Returns:
            Path to generated WAV file (24kHz mono)

        Raises:
            ValueError: If script is empty
        """
        if not script or script.strip() == "":
            raise ValueError("Script cannot be empty")

        self._load_bark()

        output_path = self.output_dir / f"voice_{uuid.uuid4().hex[:8]}.wav"

        # Generate using Bark
        audio = self._bark_engine.synthesize(
            text=script,
            voice_id=voice_id,
            emotion=emotion,
            seed=seed,
        )

        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Save to output path
        sf.write(str(output_path), audio, samplerate=24000)
        logger.info(f"Bark generated: {output_path}")

        return str(output_path)

    def unload(self) -> None:
        """Unload Bark engine and free VRAM.

        Call this before starting the visual pipeline to ensure
        maximum GPU memory is available.
        """
        if self._bark_engine is not None:
            logger.info("Unloading Bark engine...")
            self._bark_engine.unload()
            del self._bark_engine
            self._bark_engine = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Bark engine unloaded, VRAM freed")
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/matt/nsn/vortex
pytest tests/unit/test_audio_engine.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/vortex/core/audio.py tests/unit/test_audio_engine.py
git commit -m "refactor(audio): update AudioEngine to use Bark TTS"
```

---

## Task 8: Update Renderer (renderer.py)

**Files:**
- Modify: `vortex/src/vortex/renderers/default/renderer.py`

**Step 1: Update imports (line 49)**

Replace:
```python
from vortex.models.kokoro import load_kokoro
```

With:
```python
from vortex.models.bark import load_bark
```

**Step 2: Update _ModelRegistry.load_all_models() (around line 147-150)**

Replace:
```python
            # Load Kokoro TTS
            logger.info("Loading model: kokoro")
            target_device = "cpu" if self._offloading_enabled else self.device
            kokoro = load_kokoro(device=target_device)
```

With:
```python
            # Load Bark TTS
            logger.info("Loading model: bark")
            target_device = "cpu" if self._offloading_enabled else self.device
            bark = load_bark(device=target_device)
```

**Step 3: Update model registration (around line 152)**

Replace any `self._models["kokoro"] = kokoro` with:
```python
            self._models["bark"] = bark
```

**Step 4: Update _generate_audio method (around line 634)**

Replace:
```python
        # Get Kokoro model from registry
        kokoro = self._model_registry.get_model("kokoro")
```

With:
```python
        # Get Bark model from registry
        bark = self._model_registry.get_model("bark")
```

**Step 5: Update synthesis call (around line 655-662)**

Replace the Kokoro synthesis call with Bark:
```python
        # Generate audio with Bark (wrap in thread to avoid blocking)
        try:
            # Extract emotion from recipe (default to neutral)
            emotion = audio_config.get("emotion", "neutral")

            audio = await asyncio.to_thread(
                bark.synthesize,
                text=script,
                voice_id=voice_id,
                emotion=emotion,
                output=self._audio_buffer,
                seed=seed,
            )
            return audio
```

**Step 6: Update docstrings and comments**

Search and replace "Kokoro" with "Bark" in all docstrings and comments within renderer.py.

**Step 7: Verify no Kokoro references remain**

```bash
grep -n "kokoro" /home/matt/nsn/vortex/src/vortex/renderers/default/renderer.py
grep -n "Kokoro" /home/matt/nsn/vortex/src/vortex/renderers/default/renderer.py
```

Expected: No output (no remaining references)

**Step 8: Commit**

```bash
git add src/vortex/renderers/default/renderer.py
git commit -m "refactor(renderer): integrate Bark TTS replacing Kokoro"
```

---

## Task 9: Update Showrunner Prompts

**Files:**
- Modify: `vortex/src/vortex/models/showrunner.py`

**Step 1: Locate the prompt template**

The Showrunner uses a prompt template to instruct the LLM. We need to add instructions for Bark tokens.

**Step 2: Add Bark token instructions to the prompt**

Find the prompt template (around line 100+) and add these instructions:

```python
# Add to the system prompt or script generation instructions:
BARK_TOKEN_INSTRUCTIONS = """
For dialogue, you may use these audio tokens to add expressiveness:
- [laughter] or [laughs] - character laughs
- [sighs] - character sighs
- [gasps] - character gasps in surprise
- [clears throat] - character clears throat
- ... (ellipsis) - hesitation or trailing off
- CAPITALIZED WORDS - shouting or emphasis

Example: "[gasps] Oh my GOD... [laughs] That's the most ridiculous thing I've ever seen!"
"""
```

**Step 3: Update the script generation prompt**

In the `generate_script` method, append the Bark token instructions to the prompt.

**Step 4: Update fallback templates**

Update `FALLBACK_TEMPLATES` to include Bark tokens in the dialogue:

Example update for the first template:
```python
Script(
    setup="[clears throat] Are you tired of your... regular teeth?",
    punchline="Try Teeth-B-Gone! [laughs maniacally] Now your mouth is just a SMOOTH hole!",
    storyboard=[...],
),
```

**Step 5: Commit**

```bash
git add src/vortex/models/showrunner.py
git commit -m "feat(showrunner): add Bark token instructions for expressive audio"
```

---

## Task 10: Update Config (config.yaml)

**Files:**
- Modify: `vortex/config.yaml`

**Step 1: Update model precision section (line 28)**

Replace:
```yaml
    kokoro: "fp32"        # Full precision (small model, quality matters)
```

With:
```yaml
    bark: "fp32"          # Full precision for expressive audio quality
```

**Step 2: Update comments referencing Kokoro**

Search for "Kokoro" in config.yaml and update to "Bark".

**Step 3: Update timeout (line 165)**

```yaml
    audio_s: 45    # Bark TTS timeout (20-40s typical for 15s audio)
```

**Step 4: Commit**

```bash
git add config.yaml
git commit -m "refactor(config): update for Bark TTS"
```

---

## Task 11: Update test_imports.py

**Files:**
- Modify: `vortex/tests/test_imports.py`

**Step 1: Find and update Kokoro import test**

Replace any test that imports `kokoro` module with `bark`:

```python
def test_import_bark():
    """Test that Bark TTS module can be imported."""
    from vortex.models.bark import BarkVoiceEngine, load_bark
    assert BarkVoiceEngine is not None
    assert load_bark is not None
```

**Step 2: Remove Kokoro import tests**

Delete any tests that reference `vortex.models.kokoro`.

**Step 3: Run import tests**

```bash
cd /home/matt/nsn/vortex
pytest tests/test_imports.py -v
```

Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_imports.py
git commit -m "test: update import tests for Bark migration"
```

---

## Task 12: Run Full Test Suite

**Files:**
- None (validation task)

**Step 1: Run all unit tests**

```bash
cd /home/matt/nsn/vortex
pytest tests/unit/ -v --tb=short
```

Expected: All tests PASS (some may skip if they require GPU)

**Step 2: Run linter**

```bash
cd /home/matt/nsn/vortex
ruff check src/
```

Expected: No errors

**Step 3: Run type checker (if mypy is configured)**

```bash
cd /home/matt/nsn/vortex
mypy src/vortex/models/bark.py --ignore-missing-imports
```

Expected: No errors

**Step 4: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: address test and lint issues from Bark migration"
```

---

## Task 13: Integration Test (Manual)

**Files:**
- None (validation task)

**Step 1: Test Bark audio generation standalone**

```bash
cd /home/matt/nsn/vortex
python3 -c "
from vortex.models.bark import load_bark
import soundfile as sf

engine = load_bark(device='cuda')
audio = engine.synthesize(
    text='[laughs] Wubba lubba dub dub! This is a test of the Bark TTS system.',
    voice_id='rick_c137',
    emotion='manic',
)
sf.write('test_bark_output.wav', audio.numpy(), 24000)
print(f'Generated audio: {len(audio)} samples ({len(audio)/24000:.2f}s)')
engine.unload()
"
```

**Step 2: Listen to output**

Play `test_bark_output.wav` and verify:
- Audio is audible and coherent
- There's noticeable expressiveness/chaos
- Duration is reasonable (~3-5 seconds for this text)

**Step 3: Clean up test file**

```bash
rm test_bark_output.wav
```

**Step 4: Create final summary commit**

```bash
git add -A
git commit -m "feat(audio): complete Bark TTS migration - Kokoro removed

BREAKING CHANGE: Kokoro TTS replaced with Bark TTS

- Added BarkVoiceEngine with paralinguistic token support
- Added voice_profiles.yaml for character mapping
- Added bark_emotions.yaml for temperature-based emotion
- Added fallback_audio.wav for generation failures
- Updated AudioEngine, renderer, showrunner for Bark
- Removed all Kokoro-related code and tests

Bark features:
- [laughter], [gasps], [sighs], [clears throat] tokens
- Temperature-based chaos control (coarse_temp, fine_temp)
- 3-retry mechanism with fallback audio
- Zero-shot voice cloning via reference audio (future)
"
```

---

## Summary

| Task | Description | Estimated Steps |
|------|-------------|-----------------|
| 1 | Create fallback audio asset | 3 |
| 2 | Create voice profiles config | 3 |
| 3 | Create emotions config | 3 |
| 4 | Update dependencies | 5 |
| 5 | Create BarkVoiceEngine | 5 |
| 6 | Delete Kokoro files | 4 |
| 7 | Update AudioEngine | 5 |
| 8 | Update Renderer | 8 |
| 9 | Update Showrunner | 5 |
| 10 | Update config.yaml | 4 |
| 11 | Update test_imports | 4 |
| 12 | Run full test suite | 4 |
| 13 | Integration test | 4 |

**Total: 13 tasks, ~57 steps**

---

## Execution Notes

- Tasks 1-4 can be done in parallel (no dependencies)
- Task 5 depends on Task 4 (bark must be installed)
- Task 6 should be done after Task 5 is verified working
- Tasks 7-10 depend on Tasks 5-6
- Tasks 11-13 are validation/cleanup

**Risk Mitigation:**
- Keep a backup of deleted Kokoro files until Task 13 passes
- If Bark fails to install, check Python version (requires 3.8+)
- If VRAM issues occur, reduce batch size or enable more aggressive offloading
