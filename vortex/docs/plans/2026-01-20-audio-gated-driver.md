# Audio-Gated Motion Driver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the failed JoyVASA integration with a robust, dependency-light Audio-Gated Motion Driver that uses real human motion templates gated by audio energy.

**Architecture:** Extract source face keypoints from Flux-generated actor image, load a pre-recorded human motion template (d7.pkl), and blend between source (rest) and template (motion) based on audio RMS energy. This "VTuber-style" approach ensures smooth, natural motion without phoneme-level complexity.

**Tech Stack:** librosa (audio analysis), scipy (gaussian smoothing), LivePortrait (rendering), existing Vortex pipeline

---

## Background

### Why JoyVASA Failed
- **Manifold Collapse**: JoyVASA outputs absolute motion in its own coordinate space (values ~[-5, 5])
- **LivePortrait expects**: Relative deltas in implicit keypoint space (values ~[-0.05, 0.05])
- **Result**: 100x magnitude mismatch caused face smearing/explosion

### The Audio-Gated Solution
- Use **real human motion data** from LivePortrait templates (smooth, natural)
- **Gate motion intensity** by audio RMS energy (loud = animate, quiet = rest)
- **Source-anchored blending**: `Final = Source + (Template_Delta × Energy)`
- **No new AI models**: Just signal processing (~50 lines of code)

### Verified Template Structure (`d7.pkl`)
```python
{
    'n_frames': 178,
    'output_fps': 30,
    'motion': [
        {
            'scale': np.ndarray([1, 1]),      # Face scale
            'R_d':   np.ndarray([1, 3, 3]),   # Rotation matrix
            'exp':   np.ndarray([1, 21, 3]),  # Expression keypoints
            't':     np.ndarray([1, 3]),      # Translation
        },
        ...  # 178 frames
    ],
    'c_d_eyes_lst': [np.ndarray([1, 2]), ...],  # Eye closure per frame
    'c_d_lip_lst':  [np.ndarray([1, 1]), ...],  # Lip closure per frame
}
```

---

## Task 1: Add Dependencies

**Files:**
- Modify: `vortex/pyproject.toml` (add librosa, scipy)

**Step 1: Update pyproject.toml**

Add to dependencies section:

```toml
"librosa>=0.10.0",
"scipy>=1.10.0",
```

**Step 2: Install dependencies**

Run: `cd /home/matt/nsn/vortex && pip install librosa scipy`
Expected: Successfully installed librosa scipy

**Step 3: Verify installation**

Run: `python3 -c "import librosa; import scipy; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add vortex/pyproject.toml
git commit -m "deps: add librosa and scipy for audio-gated driver"
```

---

## Task 2: Create AudioGatedDriver Class

**Files:**
- Create: `vortex/src/vortex/models/audio_driver.py`
- Test: `vortex/tests/test_audio_driver.py`

**Step 1: Write the failing test**

Create `vortex/tests/test_audio_driver.py`:

```python
"""Tests for AudioGatedDriver."""

import numpy as np
import pytest
import tempfile
import os

# We'll create a simple test audio file
def create_test_audio(path: str, duration_sec: float = 2.0, sr: int = 24000):
    """Create a test audio file with varying volume."""
    import soundfile as sf

    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    # Create audio with silence, then loud, then silence
    audio = np.zeros_like(t)
    # Loud section in the middle (0.5s to 1.5s)
    start_idx = int(0.5 * sr)
    end_idx = int(1.5 * sr)
    audio[start_idx:end_idx] = 0.5 * np.sin(2 * np.pi * 440 * t[start_idx:end_idx])

    sf.write(path, audio, sr)
    return path


class TestAudioGatedDriver:
    """Test suite for AudioGatedDriver."""

    def test_driver_initialization(self, tmp_path):
        """Driver initializes with valid template path."""
        from vortex.models.audio_driver import AudioGatedDriver

        # Create a mock template
        template_path = tmp_path / "test_template.pkl"
        mock_template = {
            'n_frames': 10,
            'output_fps': 30,
            'motion': [
                {
                    'scale': np.ones((1, 1), dtype=np.float32),
                    'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                    'exp': np.zeros((1, 21, 3), dtype=np.float32),
                    't': np.zeros((1, 3), dtype=np.float32),
                }
                for _ in range(10)
            ],
            'c_d_eyes_lst': [np.zeros((1, 2), dtype=np.float32) for _ in range(10)],
            'c_d_lip_lst': [np.zeros((1, 1), dtype=np.float32) for _ in range(10)],
        }

        import pickle
        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        driver = AudioGatedDriver(str(template_path), device="cpu")
        assert driver.template_len == 10
        assert driver.output_fps == 30

    def test_drive_returns_correct_structure(self, tmp_path):
        """Drive method returns motion dict with correct keys and shapes."""
        from vortex.models.audio_driver import AudioGatedDriver
        import torch

        # Create mock template
        template_path = tmp_path / "test_template.pkl"
        n_template_frames = 30
        mock_template = {
            'n_frames': n_template_frames,
            'output_fps': 30,
            'motion': [
                {
                    'scale': np.ones((1, 1), dtype=np.float32),
                    'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                    'exp': np.random.randn(1, 21, 3).astype(np.float32) * 0.01,
                    't': np.zeros((1, 3), dtype=np.float32),
                }
                for _ in range(n_template_frames)
            ],
            'c_d_eyes_lst': [np.random.rand(1, 2).astype(np.float32) for _ in range(n_template_frames)],
            'c_d_lip_lst': [np.random.rand(1, 1).astype(np.float32) for _ in range(n_template_frames)],
        }

        import pickle
        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        # Create test audio
        audio_path = tmp_path / "test_audio.wav"
        create_test_audio(str(audio_path), duration_sec=2.0)

        # Create mock source_info (what LivePortrait extracts from source image)
        source_info = {
            'exp': torch.zeros(1, 21, 3),
            'scale': torch.ones(1, 1),
            'R': torch.eye(3).unsqueeze(0),
            't': torch.zeros(1, 3),
        }

        driver = AudioGatedDriver(str(template_path), device="cpu")
        result = driver.drive(str(audio_path), source_info, fps=24)

        # Check structure
        assert 'motion' in result
        assert 'c_d_eyes_lst' in result
        assert 'c_d_lip_lst' in result
        assert 'n_frames' in result

        # Check motion list structure
        assert len(result['motion']) == result['n_frames']
        assert 'exp' in result['motion'][0]
        assert 'scale' in result['motion'][0]
        assert 'R_d' in result['motion'][0]
        assert 't' in result['motion'][0]

    def test_energy_gates_expression(self, tmp_path):
        """Expression magnitude correlates with audio energy."""
        from vortex.models.audio_driver import AudioGatedDriver
        import torch

        # Create template with non-zero expression
        template_path = tmp_path / "test_template.pkl"
        n_template_frames = 60
        mock_template = {
            'n_frames': n_template_frames,
            'output_fps': 30,
            'motion': [
                {
                    'scale': np.ones((1, 1), dtype=np.float32),
                    'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                    'exp': np.ones((1, 21, 3), dtype=np.float32) * 0.05,  # Non-zero template
                    't': np.zeros((1, 3), dtype=np.float32),
                }
                for _ in range(n_template_frames)
            ],
            'c_d_eyes_lst': [np.ones((1, 2), dtype=np.float32) * 0.5 for _ in range(n_template_frames)],
            'c_d_lip_lst': [np.ones((1, 1), dtype=np.float32) * 0.5 for _ in range(n_template_frames)],
        }

        import pickle
        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        # Create audio with silence-loud-silence pattern
        audio_path = tmp_path / "test_audio.wav"
        create_test_audio(str(audio_path), duration_sec=2.0)

        source_info = {
            'exp': torch.zeros(1, 21, 3),
            'scale': torch.ones(1, 1),
            'R': torch.eye(3).unsqueeze(0),
            't': torch.zeros(1, 3),
        }

        driver = AudioGatedDriver(str(template_path), device="cpu")
        result = driver.drive(str(audio_path), source_info, fps=24)

        # Get expression magnitudes for silent vs loud frames
        # Frames 0-12 should be silent (first 0.5s at 24fps)
        # Frames 12-36 should be loud (0.5s-1.5s)
        # Frames 36-48 should be silent again

        silent_exp = result['motion'][5]['exp']
        loud_exp = result['motion'][24]['exp']

        silent_mag = torch.abs(silent_exp).mean().item()
        loud_mag = torch.abs(loud_exp).mean().item()

        # Loud frames should have larger expression magnitude
        assert loud_mag > silent_mag, f"Expected loud ({loud_mag}) > silent ({silent_mag})"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/matt/nsn/vortex && .venv/bin/pytest tests/test_audio_driver.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'vortex.models.audio_driver'"

**Step 3: Write the implementation**

Create `vortex/src/vortex/models/audio_driver.py`:

```python
"""Audio-Gated Motion Driver for LivePortrait.

This module implements a "VTuber-style" motion driver that uses real human
motion templates gated by audio energy. Instead of synthesizing motion from
scratch (which causes artifacts), we blend between a source rest pose and
recorded motion data based on audio volume.

Architecture:
    1. Load pre-recorded motion template (d7.pkl from LivePortrait)
    2. Extract audio RMS energy envelope
    3. For each frame: blend = source + (template_delta × energy)
    4. Output motion dict compatible with LivePortrait's warp pipeline

This approach guarantees:
    - Smooth motion (from real human recording)
    - Mouth opens when speaking (template contains open-mouth frames)
    - Natural "alive" appearance during silence (20% baseline motion)
    - No coordinate space mismatches (uses LivePortrait's native format)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class AudioGatedDriver:
    """Audio-gated motion driver using template blending.

    Uses a pre-recorded motion template and gates it with audio energy
    to create natural-looking speech animation.

    Attributes:
        template_len: Number of frames in the motion template
        output_fps: Frame rate of the template (typically 30)
        device: Torch device for tensor operations
    """

    def __init__(
        self,
        template_path: str,
        device: str = "cuda",
    ):
        """Initialize the audio-gated driver.

        Args:
            template_path: Path to LivePortrait motion template (.pkl)
            device: Torch device for output tensors
        """
        self.device = device
        self.template_path = Path(template_path)

        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        # Load template
        with open(self.template_path, 'rb') as f:
            self.template = pickle.load(f)

        self.template_len = self.template['n_frames']
        self.output_fps = self.template['output_fps']

        # Pre-extract template tensors for efficiency
        self._cache_template_tensors()

        logger.info(
            f"AudioGatedDriver initialized",
            extra={
                "template_frames": self.template_len,
                "template_fps": self.output_fps,
            }
        )

    def _cache_template_tensors(self) -> None:
        """Pre-convert template to tensors for efficient access."""
        motion = self.template['motion']

        # Stack all frames into tensors
        self.t_exp = torch.stack([
            torch.tensor(m['exp'], dtype=torch.float32)
            for m in motion
        ])  # [N, 1, 21, 3]

        self.t_scale = torch.stack([
            torch.tensor(m['scale'], dtype=torch.float32)
            for m in motion
        ])  # [N, 1, 1]

        self.t_R = torch.stack([
            torch.tensor(m['R_d'], dtype=torch.float32)
            for m in motion
        ])  # [N, 1, 3, 3]

        self.t_t = torch.stack([
            torch.tensor(m['t'], dtype=torch.float32)
            for m in motion
        ])  # [N, 1, 3]

        # Eye and lip controls
        self.t_eyes = torch.stack([
            torch.tensor(e, dtype=torch.float32)
            for e in self.template['c_d_eyes_lst']
        ])  # [N, 1, 2]

        self.t_lips = torch.stack([
            torch.tensor(l, dtype=torch.float32)
            for l in self.template['c_d_lip_lst']
        ])  # [N, 1, 1]

        # Compute template neutral (frame 0) for delta calculation
        self.t_exp_neutral = self.t_exp[0]  # [1, 21, 3]
        self.t_R_neutral = self.t_R[0]      # [1, 3, 3]
        self.t_t_neutral = self.t_t[0]      # [1, 3]

    def _extract_audio_energy(
        self,
        audio_path: str,
        fps: int,
        sample_rate: int = 24000,
    ) -> np.ndarray:
        """Extract smoothed RMS energy envelope from audio.

        Args:
            audio_path: Path to audio file (WAV)
            fps: Target frame rate for energy envelope
            sample_rate: Expected sample rate of audio

        Returns:
            Normalized energy envelope [0, 1] per frame
        """
        import librosa

        # Load audio
        y, sr = librosa.load(audio_path, sr=sample_rate)

        # Calculate RMS energy per video frame
        hop_length = int(sr / fps)
        rms = librosa.feature.rms(
            y=y,
            frame_length=hop_length,
            hop_length=hop_length
        )[0]

        # Smooth to prevent twitching (sigma=1.5 frames)
        rms_smooth = gaussian_filter1d(rms, sigma=1.5)

        # Normalize: clip noise floor (0.02) and saturate at typical speech (0.25)
        # This ensures silence is truly 0 and normal speech reaches 1.0
        energy = np.clip((rms_smooth - 0.02) / 0.23, 0.0, 1.0)

        return energy

    def _ping_pong_index(self, frame_idx: int) -> int:
        """Convert linear frame index to ping-pong looping index.

        This prevents visible loop seams by playing forward then backward.

        Args:
            frame_idx: Linear frame index

        Returns:
            Template index with ping-pong looping
        """
        cycle_len = (self.template_len - 1) * 2
        t_idx = frame_idx % cycle_len

        if t_idx >= self.template_len:
            t_idx = cycle_len - t_idx

        return t_idx

    def drive(
        self,
        audio_path: str,
        source_info: dict[str, torch.Tensor],
        fps: int = 24,
    ) -> dict[str, Any]:
        """Generate motion sequence from audio and source face.

        Uses source-anchored blending: Final = Source + (Template_Delta × Energy)

        Args:
            audio_path: Path to audio file (WAV, 24kHz)
            source_info: Dict with source face parameters:
                - 'exp': Expression tensor [1, 21, 3]
                - 'scale': Scale tensor [1, 1]
                - 'R': Rotation matrix [1, 3, 3]
                - 't': Translation tensor [1, 3]
            fps: Output frame rate

        Returns:
            Motion dict compatible with LivePortrait:
                - 'n_frames': int
                - 'output_fps': int
                - 'motion': list of frame dicts
                - 'c_d_eyes_lst': list of eye tensors
                - 'c_d_lip_lst': list of lip tensors
        """
        # Extract audio energy envelope
        energy = self._extract_audio_energy(audio_path, fps)
        num_frames = len(energy)

        logger.debug(
            f"Driving motion from audio",
            extra={
                "audio_path": audio_path,
                "num_frames": num_frames,
                "fps": fps,
                "energy_mean": float(energy.mean()),
                "energy_max": float(energy.max()),
            }
        )

        # Get source anchor tensors
        src_exp = source_info['exp'].to(self.device)      # [1, 21, 3]
        src_scale = source_info['scale'].to(self.device)  # [1, 1]
        src_R = source_info['R'].to(self.device)          # [1, 3, 3]
        src_t = source_info['t'].to(self.device)          # [1, 3]

        # Move template to device
        t_exp = self.t_exp.to(self.device)
        t_scale = self.t_scale.to(self.device)
        t_R = self.t_R.to(self.device)
        t_t = self.t_t.to(self.device)
        t_eyes = self.t_eyes.to(self.device)
        t_lips = self.t_lips.to(self.device)
        t_exp_neutral = self.t_exp_neutral.to(self.device)
        t_R_neutral = self.t_R_neutral.to(self.device)
        t_t_neutral = self.t_t_neutral.to(self.device)

        # Generate motion for each frame
        motion_list = []
        eyes_list = []
        lips_list = []

        for i in range(num_frames):
            # Ping-pong loop through template
            t_idx = self._ping_pong_index(i)

            # Current audio energy [0, 1]
            e = float(energy[i])

            # Motion scaling:
            # - 20% baseline keeps character "alive" during silence
            # - 100% at full energy
            motion_scale = 0.2 + (0.8 * e)

            # Lip-specific scaling (more aggressive for visible mouth movement)
            lip_scale = 0.1 + (0.9 * e)

            # === EXPRESSION ===
            # Delta from template neutral
            exp_delta = t_exp[t_idx] - t_exp_neutral
            # Apply scaled delta to source
            final_exp = src_exp + (exp_delta * motion_scale)

            # === ROTATION ===
            # For rotation matrices, we interpolate toward identity
            # then apply to source rotation
            R_delta = t_R[t_idx]
            R_identity = torch.eye(3, device=self.device).unsqueeze(0)
            R_blend = R_identity * (1 - motion_scale * 0.5) + R_delta * (motion_scale * 0.5)
            # Simplified: just use source rotation with slight template influence
            final_R = src_R  # Keep source rotation stable for now

            # === TRANSLATION ===
            t_delta = t_t[t_idx] - t_t_neutral
            final_t = src_t + (t_delta * motion_scale * 0.3)  # Dampen translation

            # === SCALE ===
            # Keep scale close to source
            scale_delta = t_scale[t_idx] - 1.0
            final_scale = src_scale + (scale_delta * motion_scale * 0.1)

            # === EYE/LIP CONTROLS ===
            # Eyes: always use template (natural blinking)
            eyes = t_eyes[t_idx]

            # Lips: gate by audio energy
            lips = t_lips[t_idx] * lip_scale

            # Append frame
            motion_list.append({
                'exp': final_exp.cpu().numpy(),
                'scale': final_scale.cpu().numpy(),
                'R_d': final_R.cpu().numpy(),
                't': final_t.cpu().numpy(),
            })
            eyes_list.append(eyes.cpu().numpy())
            lips_list.append(lips.cpu().numpy())

        return {
            'n_frames': num_frames,
            'output_fps': fps,
            'motion': motion_list,
            'c_d_eyes_lst': eyes_list,
            'c_d_lip_lst': lips_list,
        }
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/matt/nsn/vortex && .venv/bin/pytest tests/test_audio_driver.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add vortex/src/vortex/models/audio_driver.py vortex/tests/test_audio_driver.py
git commit -m "feat: add AudioGatedDriver for template-based motion synthesis"
```

---

## Task 3: Add animate_gated Method to LivePortrait

**Files:**
- Modify: `vortex/src/vortex/models/liveportrait.py`
- Test: `vortex/tests/test_liveportrait_gated.py`

**Step 1: Write the failing test**

Create `vortex/tests/test_liveportrait_gated.py`:

```python
"""Tests for LivePortrait animate_gated method."""

import pytest
import torch
import numpy as np
import tempfile
import soundfile as sf


class TestAnimateGated:
    """Test suite for animate_gated method."""

    def test_animate_gated_returns_tensor(self):
        """animate_gated returns video tensor with correct shape."""
        pytest.skip("Integration test - requires GPU and models")

        # This would be run manually with:
        # from vortex.models.liveportrait import LivePortraitModel
        # model = LivePortraitModel(...)
        # result = model.animate_gated(source_image, audio_path)
        # assert result.shape[0] > 0  # Has frames
        # assert result.shape[1] == 3  # RGB
        # assert result.shape[2] == 512
        # assert result.shape[3] == 512

    def test_animate_gated_method_exists(self):
        """LivePortraitModel has animate_gated method."""
        from vortex.models.liveportrait import LivePortraitModel

        assert hasattr(LivePortraitModel, 'animate_gated')
```

**Step 2: Run test to verify it fails**

Run: `cd /home/matt/nsn/vortex && .venv/bin/pytest tests/test_liveportrait_gated.py::TestAnimateGated::test_animate_gated_method_exists -v`
Expected: FAIL with "AssertionError" (method doesn't exist yet)

**Step 3: Add animate_gated method to LivePortraitModel**

Modify `vortex/src/vortex/models/liveportrait.py`. Add this method after `animate_from_motion`:

```python
    def animate_gated(
        self,
        source_image: torch.Tensor,
        audio_path: str,
        fps: int = 24,
        template_name: str = "d7",
    ) -> torch.Tensor:
        """Generate video using audio-gated motion from template.

        This method uses the AudioGatedDriver to blend between the source
        face (rest pose) and a pre-recorded motion template based on audio
        energy. This produces smooth, natural animation without the artifacts
        caused by direct audio-to-motion synthesis.

        Args:
            source_image: Actor image tensor [3, 512, 512] or [1, 3, 512, 512]
            audio_path: Path to audio file (WAV, 24kHz)
            fps: Output frame rate (default 24)
            template_name: Motion template to use (default "d7")

        Returns:
            Video frames tensor [num_frames, 3, 512, 512]
        """
        import cv2
        from vortex.models.audio_driver import AudioGatedDriver

        # Ensure GPU backend is loaded
        gpu_backend = getattr(self.pipeline, "_gpu_backend", None)
        if gpu_backend is None or not gpu_backend._loaded:
            if gpu_backend is not None:
                if not gpu_backend.load():
                    raise RuntimeError("Failed to load GPU backend")
            else:
                raise RuntimeError("GPU backend required for animate_gated")

        lp_pipeline = gpu_backend._pipeline
        wrapper = lp_pipeline.live_portrait_wrapper

        # Handle input shape
        if source_image.dim() == 4:
            source_image = source_image.squeeze(0)

        # Convert source image to LivePortrait format
        img_np = source_image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Extract source face info (the "anchor")
        crop_info = lp_pipeline.cropper.crop_source_image(
            img_bgr,
            lp_pipeline.cropper.crop_cfg
        )
        if crop_info is None:
            raise RuntimeError("Failed to detect face in source image")

        img_crop_256 = crop_info['img_crop_256x256']
        I_s = wrapper.prepare_source(img_crop_256)
        x_s_info = wrapper.get_kp_info(I_s)

        # Build source_info dict for driver
        source_info = {
            'exp': x_s_info['exp'].cpu(),
            'scale': x_s_info['scale'].cpu(),
            'R': self._euler_to_rotation_matrix_torch(
                x_s_info['pitch'],
                x_s_info['yaw'],
                x_s_info['roll']
            ).cpu(),
            't': x_s_info['t'].cpu(),
        }

        # Find template path
        template_path = self.repo_path / f"assets/examples/driving/{template_name}.pkl"
        if not template_path.exists():
            raise FileNotFoundError(f"Motion template not found: {template_path}")

        # Initialize audio-gated driver
        driver = AudioGatedDriver(str(template_path), device=str(self.device))

        # Generate motion from audio
        logger.info(
            "Generating audio-gated motion",
            extra={
                "template": template_name,
                "audio_path": audio_path,
                "fps": fps,
            }
        )
        motion_data = driver.drive(audio_path, source_info, fps=fps)

        # Build driving template in LivePortrait format
        driving_template = {
            'n_frames': motion_data['n_frames'],
            'output_fps': motion_data['output_fps'],
            'motion': motion_data['motion'],
            'c_eyes_lst': motion_data['c_d_eyes_lst'],
            'c_lip_lst': motion_data['c_d_lip_lst'],
        }

        # Use existing template animation path
        logger.info(
            "Rendering video from gated motion",
            extra={
                "num_frames": driving_template['n_frames'],
                "fps": fps,
            }
        )

        video = self._animate_with_template(
            source_image=source_image.unsqueeze(0) if source_image.dim() == 3 else source_image,
            driving_template=driving_template,
            num_frames=driving_template['n_frames'],
            fps=fps,
        )

        return video

    @staticmethod
    def _euler_to_rotation_matrix_torch(
        pitch: torch.Tensor,
        yaw: torch.Tensor,
        roll: torch.Tensor,
    ) -> torch.Tensor:
        """Convert euler angles to rotation matrix.

        Args:
            pitch: Rotation around X axis
            yaw: Rotation around Y axis
            roll: Rotation around Z axis

        Returns:
            Rotation matrix [1, 3, 3]
        """
        # Ensure scalar tensors
        if pitch.dim() > 0:
            pitch = pitch.squeeze()
        if yaw.dim() > 0:
            yaw = yaw.squeeze()
        if roll.dim() > 0:
            roll = roll.squeeze()

        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)

        # R = Rz @ Ry @ Rx
        R = torch.zeros(3, 3, device=pitch.device, dtype=pitch.dtype)

        R[0, 0] = cos_y * cos_r
        R[0, 1] = sin_p * sin_y * cos_r - cos_p * sin_r
        R[0, 2] = cos_p * sin_y * cos_r + sin_p * sin_r
        R[1, 0] = cos_y * sin_r
        R[1, 1] = sin_p * sin_y * sin_r + cos_p * cos_r
        R[1, 2] = cos_p * sin_y * sin_r - sin_p * cos_r
        R[2, 0] = -sin_y
        R[2, 1] = sin_p * cos_y
        R[2, 2] = cos_p * cos_y

        return R.unsqueeze(0)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/matt/nsn/vortex && .venv/bin/pytest tests/test_liveportrait_gated.py::TestAnimateGated::test_animate_gated_method_exists -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vortex/src/vortex/models/liveportrait.py vortex/tests/test_liveportrait_gated.py
git commit -m "feat: add animate_gated method to LivePortraitModel"
```

---

## Task 4: Update Renderer to Use Audio-Gated Driver

**Files:**
- Modify: `vortex/src/vortex/renderers/default/renderer.py`

**Step 1: Add new render method using audio-gated driver**

Replace the `_generate_motion` and `_generate_video_from_motion` calls with a single `_generate_video_gated` method.

In `renderer.py`, add this new method:

```python
    async def _generate_video_gated(
        self,
        actor_img: torch.Tensor,
        audio: torch.Tensor,
        recipe: dict[str, Any],
        seed: int,
    ) -> torch.Tensor:
        """Generate video using audio-gated motion driver.

        This replaces the JoyVASA motion generation with a simpler,
        more robust template-based approach.

        Args:
            actor_img: Actor image tensor [1, 3, 512, 512] or [3, 512, 512]
            audio: Audio waveform tensor at 24kHz
            recipe: Recipe with slot_params
            seed: Deterministic seed

        Returns:
            Video frames tensor [num_frames, 3, 512, 512]
        """
        import tempfile
        import soundfile as sf

        assert self._model_registry is not None
        assert self._video_buffer is not None

        liveportrait = self._model_registry.get_model("liveportrait")

        slot_params = recipe.get("slot_params", {})
        fps = slot_params.get("fps", 24)

        # Ensure correct shape
        if actor_img.dim() == 4 and actor_img.shape[0] == 1:
            actor_img = actor_img[0]

        if not hasattr(liveportrait, "animate_gated"):
            logger.warning(
                "LivePortrait missing animate_gated(); falling back to viseme-based"
            )
            return await self._generate_video(
                actor_img.unsqueeze(0),
                audio,
                recipe,
                seed,
            )

        # Save audio to temp file for librosa
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
            sf.write(audio_path, audio.cpu().numpy(), 24000)

        try:
            logger.info(
                "Generating video with audio-gated driver",
                extra={"fps": fps, "audio_samples": audio.shape[0]},
            )

            video = liveportrait.animate_gated(
                source_image=actor_img,
                audio_path=audio_path,
                fps=fps,
                template_name="d7",
            )

            if not isinstance(video, torch.Tensor):
                logger.warning("animate_gated returned non-tensor")
                return self._video_buffer

            return video

        finally:
            # Cleanup temp file
            import os
            if os.path.exists(audio_path):
                os.unlink(audio_path)
```

**Step 2: Update render() to use new method**

In the `render()` method, replace the motion generation section. Find this block (around line 482-496):

```python
            # Phase 2: Motion generation (JoyVASA) - audio-to-motion
            if offloading_enabled:
                self._model_registry.prepare_for_stage("motion")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            motion_data = await self._generate_motion(audio_result, recipe, seed)

            # Phase 3: Video rendering (LivePortrait) with JoyVASA motion
            if offloading_enabled:
                self._model_registry.prepare_for_stage("video")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            video_result = await self._generate_video_from_motion(
                actor_result, motion_data, recipe, seed
            )
```

Replace with:

```python
            # Phase 2: Video rendering with audio-gated driver
            # (Combines motion generation + rendering in one step)
            if offloading_enabled:
                self._model_registry.prepare_for_stage("video")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            video_result = await self._generate_video_gated(
                actor_result, audio_result, recipe, seed
            )
```

**Step 3: Update _ModelRegistry to remove JoyVASA**

In `_ModelRegistry.load_all_models()`, change:

```python
        # All required models (JoyVASA is required - viseme fallback deprecated)
        single_models: list[ModelName] = ["flux", "liveportrait", "kokoro", "joyvasa"]
```

To:

```python
        # All required models (audio-gated driver replaces JoyVASA)
        single_models: list[ModelName] = ["flux", "liveportrait", "kokoro"]
```

Also update `prepare_for_stage` to remove the "motion" stage:

```python
        stage_models = {
            "audio": "kokoro",
            "image": "flux",
            "video": "liveportrait",
            "clip": "clip_ensemble",
        }
```

**Step 4: Commit**

```bash
git add vortex/src/vortex/renderers/default/renderer.py
git commit -m "feat: replace JoyVASA with audio-gated driver in renderer"
```

---

## Task 5: Deprecate and Remove JoyVASA

**Files:**
- Delete: `vortex/src/vortex/models/joyvasa.py`
- Modify: `vortex/src/vortex/models/__init__.py`
- Modify: `vortex/config.yaml`

**Step 1: Remove JoyVASA from model registry**

In `vortex/src/vortex/models/__init__.py`, remove JoyVASA imports and registry entries:

Remove these lines:
```python
from vortex.models.joyvasa import JoyVASAWrapper, load_joyvasa
```

Remove from `ModelName` literal:
```python
ModelName = Literal["flux", "liveportrait", "kokoro", "clip_b", "clip_l"]
```

Remove from `load_model` function the joyvasa case.

**Step 2: Update config.yaml**

In `vortex/config.yaml`, remove any joyvasa-specific configuration.

**Step 3: Delete joyvasa.py**

Run: `rm vortex/src/vortex/models/joyvasa.py`

**Step 4: Verify no remaining references**

Run: `grep -r "joyvasa" vortex/src/ --include="*.py" | grep -v __pycache__`
Expected: No matches (or only comments)

**Step 5: Commit**

```bash
git add -u
git commit -m "refactor: remove JoyVASA integration (replaced by audio-gated driver)

BREAKING CHANGE: JoyVASA model is no longer supported. Use audio-gated
driver via LivePortrait.animate_gated() instead.

Reasoning:
- JoyVASA coordinate space was incompatible with LivePortrait
- Caused 'manifold collapse' artifacts (face smearing)
- Audio-gated driver provides smoother, more reliable animation"
```

---

## Task 6: End-to-End Verification

**Step 1: Run the E2E test**

Run: `cd /home/matt/nsn/vortex && .venv/bin/python scripts/test_e2e.py --recipe test.json`

Expected output should include:
```
Using audio-gated driver for video generation
Generating audio-gated motion
Rendering video from gated motion
E2E TEST PASSED
```

**Step 2: Visual verification**

Open the generated video at `outputs/render_slot1001_*.mp4` and verify:
- Face remains intact (no smearing)
- Mouth opens when audio plays
- Mouth closes during silence
- Natural head sway/blinking
- No twitching or jitter

**Step 3: Commit verification results**

```bash
git add -A
git commit -m "test: verify audio-gated driver E2E"
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Add dependencies | pyproject.toml |
| 2 | Create AudioGatedDriver | audio_driver.py, test_audio_driver.py |
| 3 | Add animate_gated to LivePortrait | liveportrait.py |
| 4 | Update renderer pipeline | renderer.py |
| 5 | Remove JoyVASA | joyvasa.py (deleted), __init__.py, config.yaml |
| 6 | E2E verification | test_e2e.py run |

---

## Rollback Strategy

If the audio-gated driver produces unacceptable results:

1. Revert to JoyVASA with the coordinate-space adapter we implemented earlier
2. Git command: `git revert HEAD~5..HEAD` (reverts tasks 2-6)
3. The JoyVASA code exists in git history and can be restored

---

## Tuning Guide

If motion needs adjustment after implementation:

| Parameter | Location | Effect |
|-----------|----------|--------|
| `motion_scale` baseline | audio_driver.py:180 | Increase 0.2→0.3 for more "alive" idle |
| `lip_scale` | audio_driver.py:183 | Increase 0.9→1.2 for more mouth opening |
| Gaussian sigma | audio_driver.py:95 | Increase 1.5→2.5 for smoother motion |
| Energy normalization | audio_driver.py:98 | Adjust 0.23 divisor for sensitivity |
