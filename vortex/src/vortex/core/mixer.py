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
                    "[1:a]volume=0.6[sfx];"
                    "[0:a][sfx]amix=inputs=2:duration=first",
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
