"""Utilities to persist render outputs to disk."""

from __future__ import annotations

import logging
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_render_result(
    result: Any,
    output_dir: Path | str,
    fps: int = 24,
    sample_rate: int = 24000,
    slot_id: int | None = None,
    seed: int | None = None,
    prefix: str | None = None,
    include_audio_in_mp4: bool = True,
) -> dict[str, Path]:
    """Persist RenderResult tensors to .mp4 and .wav files.

    Returns a dict with "video_path" and "audio_path".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if prefix:
        base = prefix
    else:
        parts = ["render"]
        if slot_id is not None:
            parts.append(f"slot{slot_id}")
        if seed is not None:
            parts.append(f"seed{seed}")
        parts.append(timestamp)
        base = "_".join(parts)

    video_path = output_dir / f"{base}.mp4"
    audio_path = output_dir / f"{base}.wav"

    _write_wav(audio_path, result.audio_waveform, sample_rate)
    try:
        _write_mp4(
            video_path,
            result.video_frames,
            fps=fps,
            audio_waveform=result.audio_waveform if include_audio_in_mp4 else None,
            sample_rate=sample_rate,
        )
    except Exception as exc:
        logger.warning("Failed to write mp4 with audio: %s", exc)
        _write_mp4(video_path, result.video_frames, fps=fps, audio_waveform=None)

    return {"video_path": video_path, "audio_path": audio_path}


def _tensor_to_uint8_video(video_frames: torch.Tensor) -> np.ndarray:
    frames = video_frames.detach().clamp(0.0, 1.0).cpu()
    if frames.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W], got shape {tuple(frames.shape)}")
    frames = frames.permute(0, 2, 3, 1).numpy()
    return (frames * 255.0).round().astype(np.uint8)


def _tensor_to_int16_audio(audio_waveform: torch.Tensor) -> np.ndarray:
    audio = audio_waveform.detach().cpu().flatten()
    if audio.numel() == 0:
        return np.zeros(0, dtype=np.int16)
    if audio.dtype in (torch.int16, torch.int32, torch.int64):
        return audio.to(torch.int16).numpy()
    audio = audio.to(torch.float32).clamp(-1.0, 1.0)
    return (audio * 32767.0).round().to(torch.int16).numpy()


def _write_wav(path: Path, audio_waveform: torch.Tensor, sample_rate: int) -> None:
    audio_int16 = _tensor_to_int16_audio(audio_waveform)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def _write_mp4(
    path: Path,
    video_frames: torch.Tensor,
    fps: int = 24,
    audio_waveform: torch.Tensor | None = None,
    sample_rate: int = 24000,
) -> None:
    try:
        import imageio_ffmpeg

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_bin = "ffmpeg"

    frames = _tensor_to_uint8_video(video_frames)
    if frames.shape[0] == 0:
        raise ValueError("Video tensor contains no frames")

    height, width = frames.shape[1], frames.shape[2]
    audio_path: Path | None = None
    if audio_waveform is not None and audio_waveform.numel() > 0:
        audio_path = path.with_suffix(".wav")
        _write_wav(audio_path, audio_waveform, sample_rate)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if audio_path is not None:
        cmd.extend(
            [
                "-i",
                str(audio_path),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
            ]
        )
    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
    )

    process = subprocess.run(
        cmd,
        input=frames.tobytes(),
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg mp4 encode failed: {stderr}")
