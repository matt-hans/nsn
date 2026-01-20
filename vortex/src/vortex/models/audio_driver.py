"""Audio-Gated Motion Driver for LivePortrait.

Implements "VTuber-style" animation with Procedural Jaw Override.
1. Head/Eyes: Blended from human template (natural).
2. Mouth: Overdriven procedurally by Audio RMS + Spectral Centroid.

This approach decouples head/eye motion (template-driven, subtle) from
mouth motion (audio-driven, forced open) to solve the "sealed mouth" bug.
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
    """Audio-gated motion driver using procedural jaw override.

    Uses a pre-recorded motion template for head/eye motion and
    procedurally overrides jaw keypoints based on audio features.

    Attributes:
        template_len: Number of frames in the motion template
        output_fps: Frame rate of the template (typically 30)
        device: Torch device for tensor operations
    """

    # Procedural Override Parameters
    JAW_OPEN_STRENGTH = 0.08   # How much jaw drops at max volume
    LIP_WIDEN_STRENGTH = 0.03  # How much lips widen for high frequencies

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

        with open(self.template_path, 'rb') as f:
            self.template = pickle.load(f)

        self.template_len = self.template['n_frames']
        self.output_fps = self.template['output_fps']

        self._cache_template_tensors()

        logger.info(
            "AudioGatedDriver initialized (procedural jaw override)",
            extra={
                "template_frames": self.template_len,
                "template_fps": self.output_fps,
            }
        )

    def _cache_template_tensors(self) -> None:
        """Pre-convert template to tensors for efficient access."""
        motion = self.template['motion']

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

        # Compute template neutral (frame 0) for delta calculation
        self.t_exp_neutral = self.t_exp[0]  # [1, 21, 3]

    def _extract_audio_features(
        self,
        audio_path: str,
        fps: int,
        sample_rate: int = 24000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract Volume (RMS) and Tone (Spectral Centroid) for lip shaping.

        Args:
            audio_path: Path to audio file (WAV)
            fps: Target frame rate for feature extraction
            sample_rate: Expected sample rate of audio

        Returns:
            Tuple of (energy, tone) arrays:
                - energy: Normalized RMS [0, 1] for jaw opening
                - tone: Normalized spectral centroid [0, 1] for lip width
        """
        import librosa

        y, sr = librosa.load(audio_path, sr=sample_rate)
        hop_length = int(sr / fps)

        # 1. RMS Energy -> Jaw Opening
        rms = librosa.feature.rms(
            y=y,
            frame_length=hop_length,
            hop_length=hop_length
        )[0]
        rms_smooth = gaussian_filter1d(rms, sigma=1.0)
        energy = np.clip((rms_smooth - 0.03) / 0.25, 0.0, 1.0)

        # 2. Spectral Centroid -> Lip Width (low=round "O", high=wide "E")
        cent = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=2048,
            hop_length=hop_length
        )[0]
        cent_smooth = gaussian_filter1d(cent, sigma=2.0)
        tone = np.clip((cent_smooth - 1000) / 4000, 0.0, 1.0)

        # Ensure same length
        if len(tone) < len(energy):
            tone = np.pad(tone, (0, len(energy) - len(tone)), 'edge')

        return energy, tone[:len(energy)]

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

        Uses procedural jaw override: template drives head/eye motion,
        audio features directly modify jaw keypoint indices.

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
                - 'c_d_eyes_lst': empty list (baked into expression)
                - 'c_d_lip_lst': empty list (baked into expression)
        """
        # Validate source_info structure
        required_keys = {'exp', 'scale', 'R', 't'}
        missing_keys = required_keys - set(source_info.keys())
        if missing_keys:
            raise ValueError(
                f"source_info missing required keys: {missing_keys}"
            )

        # Extract audio features
        energy, tone = self._extract_audio_features(audio_path, fps)
        num_frames = len(energy)

        logger.debug(
            "Driving motion from audio (procedural jaw)",
            extra={
                "audio_path": audio_path,
                "num_frames": num_frames,
                "fps": fps,
                "energy_mean": float(energy.mean()),
                "energy_max": float(energy.max()),
                "tone_mean": float(tone.mean()),
            }
        )

        # Get source anchor tensors
        src_exp = source_info['exp'].to(self.device)      # [1, 21, 3]
        src_scale = source_info['scale'].to(self.device)  # [1, 1]
        src_rot = source_info['R'].to(self.device)        # [1, 3, 3]
        src_t = source_info['t'].to(self.device)          # [1, 3]

        # Move template to device
        t_exp = self.t_exp.to(self.device)
        t_exp_neutral = self.t_exp_neutral.to(self.device)

        motion_list = []

        for i in range(num_frames):
            t_idx = self._ping_pong_index(i)
            e = float(energy[i])
            t = float(tone[i])

            # Base motion from template (subtle: 30% baseline + 20% scaled by energy)
            base_scale = 0.3 + (0.2 * e)
            exp_delta = t_exp[t_idx] - t_exp_neutral
            current_exp = src_exp.clone() + (exp_delta * base_scale)

            # PROCEDURAL JAW OVERRIDE
            # Indices 19, 20 = lower face, Y-axis (dim 1) = vertical opening
            # Negative offset = open jaw (downward movement)
            jaw_offset = -1.0 * e * self.JAW_OPEN_STRENGTH
            current_exp[:, 19, 1] += jaw_offset
            current_exp[:, 20, 1] += jaw_offset

            # Lip shaping: Index 17 = mouth corners, X-axis (dim 0)
            # Positive = wider, negative = narrower
            width_offset = (t - 0.5) * self.LIP_WIDEN_STRENGTH
            current_exp[:, 17, 0] += width_offset

            # Translation bobbing from template (subtle)
            t_delta = (self.t_t[t_idx].to(self.device) - self.t_t[0].to(self.device)) * 0.1

            motion_list.append({
                'exp': current_exp.cpu().numpy(),
                'scale': src_scale.cpu().numpy(),
                'R_d': src_rot.cpu().numpy(),
                't': (src_t + t_delta).cpu().numpy(),
            })

        return {
            'n_frames': num_frames,
            'output_fps': fps,
            'motion': motion_list,
            'c_d_eyes_lst': [],  # Baked into expression
            'c_d_lip_lst': [],   # Baked into expression
        }
