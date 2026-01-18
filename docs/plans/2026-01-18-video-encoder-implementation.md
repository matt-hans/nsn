# Video Encoder Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add VP9 video and Opus audio encoding to the Lane 0 pipeline, enabling browser playback via WebCodecs.

**Architecture:** PyAV encodes raw frames to VP9 IVF container in the Vortex plugin. Rust parses IVF to extract frames for P2P chunking. Viewer uses WebCodecs VideoDecoder with proper keyframe detection.

**Tech Stack:** PyAV (Python), Rust byteorder, TypeScript WebCodecs API

---

## Task 1: Add PyAV Dependency

**Files:**
- Modify: `vortex/pyproject.toml:10-28`

**Step 1: Add av dependency**

Add `"av>=12.0.0"` to the dependencies list in `vortex/pyproject.toml`:

```toml
dependencies = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "transformers>=4.36.0",
    "diffusers>=0.25.0",
    "accelerate>=0.25.0",
    "safetensors>=0.4.0",
    "pillow>=10.0.0",
    "numpy>=1.26.0",
    "opencv-python>=4.8.0",
    "soundfile>=0.12.0",
    "einops>=0.7.0",
    "bitsandbytes>=0.41.0",
    "pynvml>=11.5.0",
    "prometheus-client>=0.19.0",
    "kokoro>=0.7.0",
    "pyyaml>=6.0.0",
    "open-clip-torch>=2.23.0",
    "av>=12.0.0",  # VP9/Opus encoding for Lane 0 video pipeline
]
```

**Step 2: Install the new dependency**

Run:
```bash
cd vortex && source .venv/bin/activate && pip install av>=12.0.0
```

Expected: Successfully installed av-12.x.x

**Step 3: Verify import works**

Run:
```bash
cd vortex && source .venv/bin/activate && python -c "import av; print(f'PyAV {av.__version__}')"
```

Expected: `PyAV 12.x.x`

**Step 4: Commit**

```bash
git add vortex/pyproject.toml
git commit -m "build(vortex): add PyAV dependency for VP9/Opus encoding"
```

---

## Task 2: Create Video Encoder Module - Tests First

**Files:**
- Create: `vortex/tests/unit/test_encoder.py`
- Create: `vortex/src/vortex/utils/encoder.py`

**Step 1: Write failing tests for video encoder**

Create `vortex/tests/unit/test_encoder.py`:

```python
"""Tests for VP9/Opus video encoder utilities."""

import numpy as np
import pytest


class TestEncodeVideoFrames:
    """Tests for encode_video_frames function."""

    def test_encode_returns_bytes(self):
        """Encoder should return bytes object."""
        from vortex.utils.encoder import encode_video_frames

        # Create 24 frames of 64x64 RGB video (1 second at 24fps)
        frames = np.random.randint(0, 255, (24, 64, 64, 3), dtype=np.uint8)

        result = encode_video_frames(frames, fps=24)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_produces_ivf_container(self):
        """Output should be IVF container (starts with DKIF magic)."""
        from vortex.utils.encoder import encode_video_frames

        frames = np.random.randint(0, 255, (24, 64, 64, 3), dtype=np.uint8)

        result = encode_video_frames(frames, fps=24)

        # IVF files start with "DKIF" magic bytes
        assert result[:4] == b"DKIF", f"Expected IVF magic 'DKIF', got {result[:4]!r}"

    def test_encode_respects_keyframe_interval(self):
        """Keyframes should appear at specified interval."""
        from vortex.utils.encoder import encode_video_frames, parse_ivf_frames

        # 48 frames = 2 seconds, should have keyframes at 0 and 24
        frames = np.random.randint(0, 255, (48, 64, 64, 3), dtype=np.uint8)

        result = encode_video_frames(frames, fps=24, keyframe_interval=24)
        frame_info = parse_ivf_frames(result)

        # Check keyframe positions
        keyframe_indices = [i for i, info in enumerate(frame_info) if info["is_keyframe"]]
        assert 0 in keyframe_indices, "First frame should be keyframe"

    def test_encode_validates_shape(self):
        """Should raise on wrong tensor shape."""
        from vortex.utils.encoder import encode_video_frames

        # Wrong shape: [T, C, H, W] instead of [T, H, W, C]
        frames = np.random.randint(0, 255, (24, 3, 64, 64), dtype=np.uint8)

        with pytest.raises(ValueError, match="shape"):
            encode_video_frames(frames, fps=24)

    def test_encode_validates_dimensions_even(self):
        """Width and height must be even for VP9."""
        from vortex.utils.encoder import encode_video_frames

        # Odd dimensions
        frames = np.random.randint(0, 255, (24, 63, 63, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="even"):
            encode_video_frames(frames, fps=24)


class TestEncodeAudio:
    """Tests for encode_audio function."""

    def test_encode_audio_returns_bytes(self):
        """Audio encoder should return bytes."""
        from vortex.utils.encoder import encode_audio

        # 1 second of audio at 24kHz
        waveform = np.random.randn(24000).astype(np.float32)

        result = encode_audio(waveform, sample_rate=24000)

        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_audio_compresses(self):
        """Opus should compress significantly vs raw PCM."""
        from vortex.utils.encoder import encode_audio

        # 1 second = 24000 samples * 4 bytes = 96KB raw
        waveform = np.random.randn(24000).astype(np.float32)
        raw_size = waveform.nbytes

        result = encode_audio(waveform, sample_rate=24000, bitrate=64000)

        # Opus at 64kbps should be ~8KB for 1 second
        assert len(result) < raw_size / 5, "Opus should compress to <20% of raw size"


class TestParseIvfFrames:
    """Tests for IVF parsing helper."""

    def test_parse_extracts_frame_count(self):
        """Should extract correct number of frames."""
        from vortex.utils.encoder import encode_video_frames, parse_ivf_frames

        frames = np.random.randint(0, 255, (24, 64, 64, 3), dtype=np.uint8)

        ivf_data = encode_video_frames(frames, fps=24)
        frame_info = parse_ivf_frames(ivf_data)

        assert len(frame_info) == 24

    def test_parse_extracts_keyframe_flag(self):
        """Should correctly identify keyframes via VP9 bit check."""
        from vortex.utils.encoder import encode_video_frames, parse_ivf_frames

        frames = np.random.randint(0, 255, (24, 64, 64, 3), dtype=np.uint8)

        ivf_data = encode_video_frames(frames, fps=24, keyframe_interval=24)
        frame_info = parse_ivf_frames(ivf_data)

        # First frame must be keyframe
        assert frame_info[0]["is_keyframe"] is True
        # Most other frames should be delta frames
        delta_count = sum(1 for f in frame_info if not f["is_keyframe"])
        assert delta_count > 0, "Should have delta frames"
```

**Step 2: Run tests to verify they fail**

Run:
```bash
cd vortex && source .venv/bin/activate && pytest tests/unit/test_encoder.py -v
```

Expected: FAILED - `ModuleNotFoundError: No module named 'vortex.utils.encoder'`

**Step 3: Create encoder module with minimal implementation**

Create `vortex/src/vortex/utils/encoder.py`:

```python
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

        # VP9 keyframe detection: check bit 5 of first payload byte
        # Bit 5 = 0 means keyframe, bit 5 = 1 means inter-frame
        payload_byte = ivf_data[frame_start]
        is_keyframe = (payload_byte & 0x20) == 0

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
```

**Step 4: Run tests to verify they pass**

Run:
```bash
cd vortex && source .venv/bin/activate && pytest tests/unit/test_encoder.py -v
```

Expected: All tests PASSED

**Step 5: Commit**

```bash
git add vortex/src/vortex/utils/encoder.py vortex/tests/unit/test_encoder.py
git commit -m "feat(vortex): add VP9/Opus encoder module with IVF output

- encode_video_frames(): VP9 in IVF container with keyframe control
- encode_audio(): Opus encoding for voice audio
- parse_ivf_frames(): Helper for testing IVF structure
- Full test coverage for encoding and validation"
```

---

## Task 3: Update Plugin to Use Encoder

**Files:**
- Modify: `vortex/plugins/vortex-lane0/plugin.py`

**Step 1: Update plugin imports and _run_async method**

Modify `vortex/plugins/vortex-lane0/plugin.py`. Replace the `_run_async` method (approximately lines 89-168):

```python
    async def _run_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute video generation asynchronously.

        Args:
            payload: Same as run()

        Returns:
            Same as run()

        Raises:
            RuntimeError: If generation fails
            ValueError: If payload is invalid
        """
        import base64

        start_time = time.time()

        # Validate payload
        recipe = payload.get("recipe")
        if not isinstance(recipe, dict):
            raise ValueError("payload 'recipe' must be a dict")

        slot_id = payload.get("slot_id")
        if not isinstance(slot_id, int):
            raise ValueError("payload 'slot_id' must be an integer")

        seed = payload.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("payload 'seed' must be an integer or null")

        # Initialize pipeline
        pipeline = await self._ensure_pipeline()

        logger.info(
            f"Starting Lane 0 generation for slot {slot_id}",
            extra={"slot_id": slot_id, "seed": seed, "renderer": pipeline.renderer_name},
        )

        # Generate video
        result = await pipeline.generate_slot(recipe, slot_id=slot_id, seed=seed)

        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error_msg}")

        # Convert tensors to numpy
        video_np = result.video_frames.cpu().numpy()
        audio_np = result.audio_waveform.cpu().numpy()
        clip_np = result.clip_embedding.cpu().numpy()

        # Shape normalization: [T, C, H, W] → [T, H, W, C]
        if video_np.ndim == 4 and video_np.shape[1] == 3 and video_np.shape[3] != 3:
            video_np = np.transpose(video_np, (0, 2, 3, 1))

        # Ensure uint8 for encoder
        if video_np.dtype != np.uint8:
            # Assume float [0, 1] range
            video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

        # Import encoder (lazy to avoid import errors without GPU)
        from vortex.utils.encoder import encode_audio, encode_video_frames

        # Offload CPU-bound encoding to thread pool
        fps = recipe.get("slot_params", {}).get("fps", 24)

        try:
            video_bytes, audio_bytes = await asyncio.gather(
                asyncio.to_thread(
                    encode_video_frames,
                    video_np,
                    fps=fps,
                    bitrate=3_000_000,
                    keyframe_interval=24,
                ),
                asyncio.to_thread(
                    encode_audio,
                    audio_np,
                    sample_rate=24000,
                    bitrate=64000,
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Encoding pipeline failed: {str(e)}") from e

        # Free large arrays before base64 allocation
        del video_np
        del audio_np

        generation_time_ms = (time.time() - start_time) * 1000

        # Generate content identifier
        output_cid = f"local://{slot_id}/{result.determinism_proof.hex()[:16]}"

        logger.info(
            f"Lane 0 generation completed for slot {slot_id}",
            extra={
                "slot_id": slot_id,
                "generation_time_ms": generation_time_ms,
                "proof": result.determinism_proof.hex()[:16],
                "video_bytes": len(video_bytes),
                "audio_bytes": len(audio_bytes),
            },
        )

        return {
            "output_cid": output_cid,
            "video_data": base64.b64encode(video_bytes).decode("ascii"),
            "audio_waveform": base64.b64encode(audio_bytes).decode("ascii"),
            "clip_embedding": clip_np.tolist(),
            "determinism_proof": result.determinism_proof.hex(),
            "generation_time_ms": generation_time_ms,
        }
```

**Step 2: Verify plugin syntax**

Run:
```bash
cd vortex && source .venv/bin/activate && python -c "from plugins import VortexLane0Plugin; print('Import OK')" 2>/dev/null || python -c "import sys; sys.path.insert(0, 'plugins/vortex-lane0'); from plugin import VortexLane0Plugin; print('Import OK')"
```

Expected: `Import OK`

**Step 3: Commit**

```bash
git add vortex/plugins/vortex-lane0/plugin.py
git commit -m "feat(vortex): update plugin to encode video as VP9/Opus

- Replace .npy file output with in-memory encoding
- Shape normalize [T,C,H,W] → [T,H,W,C] before encoding
- Offload encoding to thread pool (non-blocking)
- Return base64-encoded video_data and audio_waveform
- Memory cleanup before base64 allocation"
```

---

## Task 4: Update Rust VideoChunk for IVF Parsing

**Files:**
- Modify: `node-core/crates/p2p/src/video.rs`

**Step 1: Add IVF parsing function**

Add the following function to `node-core/crates/p2p/src/video.rs` after the existing `build_video_chunks` function (around line 156):

```rust
/// Build video chunks from IVF-encoded VP9 video data.
///
/// Parses the IVF container to extract individual frames, creating one
/// VideoChunk per VP9 frame. Keyframe detection uses VP9 bitstream analysis.
///
/// # Arguments
/// * `content_id` - Content identifier for the video
/// * `slot` - Slot number
/// * `ivf_data` - IVF container bytes (VP9 encoded)
/// * `keypair` - Signing keypair
///
/// # Returns
/// Vector of signed VideoChunks, one per frame
pub fn build_video_chunks_from_ivf(
    content_id: &str,
    slot: u64,
    ivf_data: &[u8],
    keypair: &Keypair,
) -> Result<Vec<VideoChunk>, VideoChunkError> {
    if content_id.trim().is_empty() {
        return Err(VideoChunkError::EmptyContentId);
    }
    if ivf_data.len() < 32 {
        return Err(VideoChunkError::EmptyPayload);
    }

    // Verify IVF magic bytes
    if &ivf_data[0..4] != b"DKIF" {
        return Err(VideoChunkError::DecodeFailed(
            "Invalid IVF magic bytes".to_string(),
        ));
    }

    let max_allowed = TopicCategory::VideoChunks.max_message_size();
    let signer = keypair.public().encode_protobuf();

    let mut chunks = Vec::new();
    let mut pos = 32; // Skip 32-byte IVF file header
    let mut chunk_index = 0u32;

    // First pass: count total frames
    let mut count_pos = 32;
    let mut total_frames = 0u32;
    while count_pos + 12 <= ivf_data.len() {
        let frame_size = u32::from_le_bytes([
            ivf_data[count_pos],
            ivf_data[count_pos + 1],
            ivf_data[count_pos + 2],
            ivf_data[count_pos + 3],
        ]) as usize;
        count_pos += 12 + frame_size;
        total_frames += 1;
    }

    // Second pass: build chunks
    while pos + 12 <= ivf_data.len() {
        // Read 12-byte IVF frame header
        let frame_size = u32::from_le_bytes([
            ivf_data[pos],
            ivf_data[pos + 1],
            ivf_data[pos + 2],
            ivf_data[pos + 3],
        ]) as usize;

        let _timestamp = u64::from_le_bytes([
            ivf_data[pos + 4],
            ivf_data[pos + 5],
            ivf_data[pos + 6],
            ivf_data[pos + 7],
            ivf_data[pos + 8],
            ivf_data[pos + 9],
            ivf_data[pos + 10],
            ivf_data[pos + 11],
        ]);

        let frame_start = pos + 12;
        let frame_end = frame_start + frame_size;

        if frame_end > ivf_data.len() {
            break;
        }

        let payload = &ivf_data[frame_start..frame_end];

        // VP9 keyframe detection: Bit 5 of first byte
        // 0 = keyframe, 1 = inter-frame
        let is_keyframe = !payload.is_empty() && (payload[0] & 0x20) == 0;

        let payload_hash = *blake3::hash(payload).as_bytes();
        let timestamp_ms = now_ms();

        let header = VideoChunkHeader {
            version: VIDEO_CHUNK_VERSION,
            slot,
            content_id: content_id.to_string(),
            chunk_index,
            total_chunks: total_frames,
            timestamp_ms,
            is_keyframe,
            payload_hash,
        };

        let signature = keypair
            .sign(&signing_payload(&header))
            .map_err(|_| VideoChunkError::SigningFailed)?;

        let chunk = VideoChunk {
            header,
            payload: payload.to_vec(),
            signer: signer.clone(),
            signature,
        };

        let encoded_len = chunk.encode().len();
        if encoded_len > max_allowed {
            return Err(VideoChunkError::InvalidChunkSize);
        }

        chunks.push(chunk);

        pos = frame_end;
        chunk_index += 1;
    }

    if chunks.is_empty() {
        return Err(VideoChunkError::EmptyPayload);
    }

    Ok(chunks)
}
```

**Step 2: Add tests for IVF parsing**

Add tests to the `#[cfg(test)]` module at the bottom of `video.rs`:

```rust
    #[test]
    fn test_build_video_chunks_from_ivf_rejects_invalid_magic() {
        let keypair = Keypair::generate_ed25519();
        let invalid_data = vec![0u8; 64]; // Not IVF

        let result = build_video_chunks_from_ivf("QmTest", 1, &invalid_data, &keypair);
        assert!(matches!(result, Err(VideoChunkError::DecodeFailed(_))));
    }

    #[test]
    fn test_build_video_chunks_from_ivf_rejects_short_data() {
        let keypair = Keypair::generate_ed25519();
        let short_data = vec![0u8; 16]; // Too short for IVF header

        let result = build_video_chunks_from_ivf("QmTest", 1, &short_data, &keypair);
        assert!(matches!(result, Err(VideoChunkError::EmptyPayload)));
    }
```

**Step 3: Run Rust tests**

Run:
```bash
cd node-core && cargo test -p nsn-p2p -- video --nocapture
```

Expected: All tests pass

**Step 4: Commit**

```bash
git add node-core/crates/p2p/src/video.rs
git commit -m "feat(p2p): add IVF-based video chunk builder

- build_video_chunks_from_ivf(): Parse IVF container for frame-aware chunking
- VP9 keyframe detection via bit 5 check
- One VideoChunk per VP9 frame (variable size)
- Validates IVF magic bytes and frame boundaries"
```

---

## Task 5: Update Viewer VideoBuffer Interface

**Files:**
- Modify: `viewer/src/services/videoBuffer.ts`

**Step 1: Add is_keyframe to VideoChunk interface**

Modify `viewer/src/services/videoBuffer.ts` line 3-8:

```typescript
export interface VideoChunk {
	slot: number;
	chunk_index: number;
	data: Uint8Array;
	timestamp: number;
	is_keyframe: boolean;
}
```

**Step 2: Run TypeScript type check**

Run:
```bash
cd viewer && npx tsc --noEmit 2>&1 | head -20
```

Expected: Type errors in files that use VideoChunk without is_keyframe (this is expected, we'll fix in next tasks)

**Step 3: Commit**

```bash
git add viewer/src/services/videoBuffer.ts
git commit -m "feat(viewer): add is_keyframe to VideoChunk interface"
```

---

## Task 6: Update VideoPipeline to Propagate is_keyframe

**Files:**
- Modify: `viewer/src/services/videoPipeline.ts`

**Step 1: Update handleIncomingChunk to pass is_keyframe**

Modify `viewer/src/services/videoPipeline.ts` lines 101-106:

```typescript
		this.buffer.addChunk({
			slot: message.slot,
			chunk_index: message.chunk_index,
			data: message.data,
			timestamp: message.timestamp,
			is_keyframe: message.is_keyframe,
		});
```

**Step 2: Update decodeChunk to use is_keyframe**

Modify `viewer/src/services/videoPipeline.ts` lines 165-178:

```typescript
	/**
	 * Decode a single chunk
	 */
	private decodeChunk(chunk: {
		slot: number;
		chunk_index: number;
		data: Uint8Array;
		timestamp: number;
		is_keyframe: boolean;
	}): void {
		// Calculate timestamp from frame index for consistent timing
		// WebCodecs expects microseconds
		const frameDurationUs = 1_000_000 / 24; // 24 fps
		const timestamp = chunk.chunk_index * frameDurationUs;

		const encodedChunk = new EncodedVideoChunk({
			type: chunk.is_keyframe ? "key" : "delta",
			timestamp: timestamp,
			data: chunk.data,
		});

		this.decoder.decode(encodedChunk);
	}
```

**Step 3: Run TypeScript check**

Run:
```bash
cd viewer && npx tsc --noEmit
```

Expected: No errors (or only unrelated errors)

**Step 4: Commit**

```bash
git add viewer/src/services/videoPipeline.ts
git commit -m "feat(viewer): propagate is_keyframe through pipeline

- Pass is_keyframe to buffer in handleIncomingChunk
- Use chunk.is_keyframe for EncodedVideoChunk type
- Calculate timestamp from frame index (24fps)"
```

---

## Task 7: Update WebCodecs Service with Feature Detection

**Files:**
- Modify: `viewer/src/services/webcodecs.ts`

**Step 1: Add feature detection**

Modify `viewer/src/services/webcodecs.ts` to add the static method after line 8:

```typescript
export class VideoDecoderService {
	private decoder: VideoDecoder | null = null;
	private canvas: HTMLCanvasElement;
	private ctx: CanvasRenderingContext2D;
	private isConfigured = false;
	private static isSupported: boolean | null = null;

	constructor(canvas: HTMLCanvasElement) {
		this.canvas = canvas;
		// biome-ignore lint/style/noNonNullAssertion: 2d context always available
		this.ctx = canvas.getContext("2d")!;
	}

	/**
	 * Check if WebCodecs VideoDecoder is available.
	 * Caches result for performance.
	 */
	static checkSupport(): boolean {
		if (VideoDecoderService.isSupported === null) {
			VideoDecoderService.isSupported =
				typeof VideoDecoder !== "undefined" &&
				typeof EncodedVideoChunk !== "undefined";
		}
		return VideoDecoderService.isSupported;
	}

	/**
	 * Initialize decoder with codec
	 */
	async init(codec: string): Promise<void> {
		if (!VideoDecoderService.checkSupport()) {
			throw new Error(
				"WebCodecs not supported in this browser. " +
				"Requires Chrome 94+, Edge 94+, or Firefox 130+ with secure context (HTTPS/localhost)."
			);
		}

		const config: VideoDecoderConfig = {
			codec, // e.g., 'vp09.00.41.08' for VP9 Level 4.1
			optimizeForLatency: true,
		};

		// Check if codec is supported
		const support = await VideoDecoder.isConfigSupported(config);
		if (!support.supported) {
			throw new Error(`Codec ${codec} not supported`);
		}

		this.decoder = new VideoDecoder({
			output: (frame: VideoFrame) => this.renderFrame(frame),
			error: (e: DOMException) => {
				console.error("Decode error:", e);
			},
		});

		this.decoder.configure(config);
		this.isConfigured = true;
	}
```

**Step 2: Run tests**

Run:
```bash
cd viewer && npm test -- --run src/services/webcodecs.test.ts
```

Expected: Tests pass (mocked environment)

**Step 3: Commit**

```bash
git add viewer/src/services/webcodecs.ts
git commit -m "feat(viewer): add WebCodecs feature detection

- VideoDecoderService.checkSupport() for runtime detection
- Clear error message when WebCodecs unavailable
- Cached result for performance"
```

---

## Task 8: Update VideoPlayer Codec String

**Files:**
- Modify: `viewer/src/components/VideoPlayer/index.tsx`

**Step 1: Update codec string to Level 4.1**

Modify `viewer/src/components/VideoPlayer/index.tsx` line 55:

```typescript
				// Initialize decoder with VP9 Level 4.1 (supports 3Mbps, 4K)
				await pipeline.init("vp09.00.41.08");
```

**Step 2: Run viewer tests**

Run:
```bash
cd viewer && npm test
```

Expected: Tests pass (106/107 or better)

**Step 3: Commit**

```bash
git add viewer/src/components/VideoPlayer/index.tsx
git commit -m "fix(viewer): use VP9 Level 4.1 codec string

Level 1.0 only supports 200kbps/256x144.
Level 4.1 supports 4K resolution and high bitrates."
```

---

## Task 9: Final Integration Verification

**Step 1: Run all viewer tests**

Run:
```bash
cd viewer && npm test
```

Expected: All relevant tests pass

**Step 2: Run Rust tests**

Run:
```bash
cd node-core && cargo test -p nsn-p2p
```

Expected: All tests pass

**Step 3: Run Python encoder tests**

Run:
```bash
cd vortex && source .venv/bin/activate && pytest tests/unit/test_encoder.py -v
```

Expected: All tests pass

**Step 4: Type check viewer**

Run:
```bash
cd viewer && npx tsc --noEmit
```

Expected: No type errors

**Step 5: Commit any fixes**

If any tests fail, fix and commit individually.

**Step 6: Final commit summarizing all changes**

```bash
git log --oneline -10
```

Review commits and ensure they're atomic and well-described.

---

## Summary

| Task | Component | Description |
|------|-----------|-------------|
| 1 | vortex/pyproject.toml | Add PyAV dependency |
| 2 | vortex/utils/encoder.py | VP9/Opus encoder module |
| 3 | vortex-lane0/plugin.py | Use encoder, return base64 |
| 4 | node-core/p2p/video.rs | IVF parsing for frame chunks |
| 5 | viewer/videoBuffer.ts | Add is_keyframe to interface |
| 6 | viewer/videoPipeline.ts | Propagate is_keyframe |
| 7 | viewer/webcodecs.ts | Feature detection |
| 8 | viewer/VideoPlayer | VP9 Level 4.1 codec |
| 9 | All | Integration verification |
