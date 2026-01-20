"""Audio-Gated Motion Driver for LivePortrait.

This module implements a "VTuber-style" motion driver that uses real human
motion templates gated by audio energy. Instead of synthesizing motion from
scratch (which causes artifacts), we blend between a source rest pose and
recorded motion data based on audio volume.

Architecture:
    1. Load pre-recorded motion template (d7.pkl from LivePortrait)
    2. Extract audio RMS energy envelope
    3. For each frame: blend = source + (template_delta * energy)
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
            "AudioGatedDriver initialized",
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
            torch.tensor(lip, dtype=torch.float32)
            for lip in self.template['c_d_lip_lst']
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
        if self.template_len <= 1:
            return 0

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

        Uses source-anchored blending: Final = Source + (Template_Delta * Energy)

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
        # Validate source_info structure
        required_keys = {'exp', 'scale', 'R', 't'}
        missing_keys = required_keys - set(source_info.keys())
        if missing_keys:
            raise ValueError(
                f"source_info missing required keys: {missing_keys}"
            )

        # Extract audio energy envelope
        energy = self._extract_audio_energy(audio_path, fps)
        num_frames = len(energy)

        logger.debug(
            "Driving motion from audio",
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
        src_rot = source_info['R'].to(self.device)        # [1, 3, 3]
        src_t = source_info['t'].to(self.device)          # [1, 3]

        # Move template to device
        t_exp = self.t_exp.to(self.device)
        t_scale = self.t_scale.to(self.device)
        t_t = self.t_t.to(self.device)
        t_eyes = self.t_eyes.to(self.device)
        t_lips = self.t_lips.to(self.device)
        t_exp_neutral = self.t_exp_neutral.to(self.device)
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
            # For rotation matrices, keep source rotation stable
            # (Template rotation blending can introduce artifacts)
            final_rot = src_rot

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
                'R_d': final_rot.cpu().numpy(),
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
