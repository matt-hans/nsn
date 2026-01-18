"""VP9 and Opus encoding utilities for Lane 0 video pipeline.

This module provides functions to encode raw video frames and audio waveforms
into VP9 (IVF container) and Opus formats suitable for P2P streaming and
WebCodecs playback.
"""

from __future__ import annotations

import io
import struct
from typing import Any

import av
import numpy as np


class EncodingError(Exception):
    """Raised when video/audio encoding fails."""

    pass


def encode_video_frames(
    frames: np.ndarray,
    fps: int = 24,
    bitrate: int = 3_000_000,
    keyframe_interval: int = 24,
) -> bytes:
    """Encode video frames to VP9 IVF container.

    Args:
        frames: Video frames as numpy array with shape [T, H, W, C] where
            T=frames, H=height, W=width, C=channels (3 for RGB).
            Must be uint8 dtype.
        fps: Frames per second (default 24).
        bitrate: Target bitrate in bits/second (default 3 Mbps).
        keyframe_interval: Frames between keyframes (default 24 = 1 second).

    Returns:
        VP9 encoded video in IVF container format as bytes.

    Raises:
        ValueError: If frame dimensions are invalid.
        EncodingError: If encoding fails.
    """
    # Validate input shape
    if frames.ndim != 4:
        raise ValueError(
            f"Expected 4D array [T, H, W, C], got shape {frames.shape}"
        )

    t, h, w, c = frames.shape

    if c != 3:
        raise ValueError(
            f"Expected 3 channels (RGB), got {c}. "
            f"If shape is [T, C, H, W], transpose to [T, H, W, C] first."
        )

    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(
            f"Height ({h}) and width ({w}) must be even for VP9 encoding"
        )

    # Create in-memory buffer for IVF output
    buffer = io.BytesIO()

    try:
        # Open container with IVF format
        container = av.open(buffer, mode="w", format="ivf")

        # Add VP9 video stream
        stream = container.add_stream("vp9", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"  # Standard format, Profile 0 compatible
        stream.bit_rate = bitrate
        stream.gop_size = keyframe_interval  # Keyframe interval

        # Encode each frame
        for i, frame_data in enumerate(frames):
            # Create VideoFrame from numpy array
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            frame.pts = i

            # Encode frame
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush encoder
        for packet in stream.encode():
            container.mux(packet)

        container.close()

    except Exception as e:
        raise EncodingError(f"VP9 encoding failed: {e}") from e

    return buffer.getvalue()


def encode_audio(
    waveform: np.ndarray,
    sample_rate: int = 24000,
    bitrate: int = 64000,
) -> bytes:
    """Encode audio waveform to Opus format.

    Args:
        waveform: Audio samples as 1D numpy array, float32 in range [-1, 1].
        sample_rate: Sample rate in Hz (default 24000 for Kokoro TTS).
        bitrate: Target bitrate in bits/second (default 64 kbps).

    Returns:
        Opus encoded audio as raw packet bytes.

    Raises:
        EncodingError: If encoding fails.
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")

    # Convert float32 [-1, 1] to int16
    if waveform.dtype == np.float32:
        waveform_int = (waveform * 32767).astype(np.int16)
    elif waveform.dtype == np.int16:
        waveform_int = waveform
    else:
        raise ValueError(f"Expected float32 or int16, got {waveform.dtype}")

    buffer = io.BytesIO()

    try:
        # Create container (ogg for opus)
        container = av.open(buffer, mode="w", format="ogg")

        # Add Opus audio stream
        stream = container.add_stream("libopus", rate=sample_rate)
        stream.bit_rate = bitrate
        stream.layout = "mono"

        # Create audio frame
        # Opus typically wants 960 samples per frame at 48kHz, or 480 at 24kHz
        frame_size = 480 if sample_rate == 24000 else 960

        # Pad waveform to multiple of frame_size
        pad_len = (frame_size - len(waveform_int) % frame_size) % frame_size
        if pad_len > 0:
            waveform_int = np.pad(waveform_int, (0, pad_len))

        # Encode in chunks
        for i in range(0, len(waveform_int), frame_size):
            chunk = waveform_int[i : i + frame_size]
            frame = av.AudioFrame.from_ndarray(
                chunk.reshape(1, -1), format="s16", layout="mono"
            )
            frame.sample_rate = sample_rate
            frame.pts = i

            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush
        for packet in stream.encode():
            container.mux(packet)

        container.close()

    except Exception as e:
        raise EncodingError(f"Opus encoding failed: {e}") from e

    return buffer.getvalue()


def parse_ivf_frames(ivf_data: bytes) -> list[dict[str, Any]]:
    """Parse IVF container to extract frame information.

    This is primarily for testing and debugging. The Rust side
    will do the actual parsing for P2P chunking.

    Args:
        ivf_data: IVF container bytes.

    Returns:
        List of dicts with keys: offset, size, is_keyframe, timestamp
    """
    if len(ivf_data) < 32:
        raise ValueError("IVF data too short for header")

    # Verify IVF magic
    if ivf_data[:4] != b"DKIF":
        raise ValueError(f"Invalid IVF magic: {ivf_data[:4]!r}")

    # Parse file header (32 bytes)
    # Bytes 0-3: signature "DKIF"
    # Bytes 4-5: version (should be 0)
    # Bytes 6-7: header length (should be 32)
    # Bytes 8-11: codec FourCC ("VP90" for VP9)
    # Bytes 12-13: width
    # Bytes 14-15: height
    # Bytes 16-19: frame rate denominator
    # Bytes 20-23: frame rate numerator
    # Bytes 24-27: number of frames
    # Bytes 28-31: unused

    frames = []
    pos = 32  # Start after file header

    while pos + 12 <= len(ivf_data):
        # Frame header: 12 bytes
        # Bytes 0-3: frame size (little endian)
        # Bytes 4-11: timestamp (little endian)
        frame_size = struct.unpack_from("<I", ivf_data, pos)[0]
        timestamp = struct.unpack_from("<Q", ivf_data, pos + 4)[0]

        frame_start = pos + 12
        frame_end = frame_start + frame_size

        if frame_end > len(ivf_data):
            break

        # VP9 keyframe detection: check bit 2 of first payload byte
        # VP9 uncompressed header format (profile 0-2):
        # - bits 7-6: frame_marker (0b10)
        # - bit 5: profile_low_bit
        # - bit 4: show_existing_frame (0 for normal frames)
        # - bit 3: reserved/error_resilient for keyframe
        # - bit 2: frame_type (0=keyframe, 1=non-keyframe)
        payload_byte = ivf_data[frame_start]
        is_keyframe = (payload_byte & 0x04) == 0

        frames.append(
            {
                "offset": frame_start,
                "size": frame_size,
                "is_keyframe": is_keyframe,
                "timestamp": timestamp,
            }
        )

        pos = frame_end

    return frames
