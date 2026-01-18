# Video Encoder Pipeline Design

**Date:** 2026-01-18
**Status:** Approved
**Author:** Claude + User collaborative design session

## Overview

Add VP9 video and Opus audio encoding to the Vortex Lane 0 pipeline, enabling browser playback via WebCodecs. This closes the gap between Vortex's raw tensor output and the viewer's expectation of encoded video streams.

## Problem Statement

The current pipeline has three critical gaps:

1. **No video encoding**: Vortex outputs raw `torch.Tensor` frames, but the viewer expects VP9 encoded video
2. **WebCodecs unavailable**: The viewer crashes with `VideoDecoder is not defined` in Tauri's WebKitGTK
3. **Keyframe detection broken**: `build_video_chunks()` uses index-based math, incompatible with variable-size encoded frames

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Video codec | VP9 | Royalty-free, WebCodecs native, viewer already expects it |
| Audio codec | Opus | WebCodecs native, 10x compression vs PCM |
| Encoding library | PyAV | Direct NumPy→VP9, mature FFmpeg bindings |
| Output format | IVF container | Frame-delimited, easy to parse for chunking |
| Keyframe interval | 24 frames (1 sec) | Fast mid-stream join for P2P viewers |
| Encoding location | Plugin layer | Clean separation from AI generation |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           VORTEX (Python)                                │
├─────────────────────────────────────────────────────────────────────────┤
│  DefaultRenderer                                                         │
│  ├── Flux-Schnell → actor image [1,3,512,512]                           │
│  ├── LivePortrait → video frames [T,3,512,512]                          │
│  ├── Kokoro TTS → audio waveform [samples]                              │
│  └── CLIP ensemble → embedding [512]                                    │
│           ↓                                                              │
│  Plugin (vortex-lane0/plugin.py)                                        │
│  ├── Shape normalize: [T,C,H,W] → [T,H,W,C]                            │
│  ├── encode_video_frames() → VP9 IVF bytes                             │
│  ├── encode_audio() → Opus bytes                                        │
│  └── base64 encode → JSON response                                      │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓ gRPC
┌─────────────────────────────────────────────────────────────────────────┐
│                         NODE-CORE (Rust)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  VortexClient                                                            │
│  └── Deserialize base64 video_data/audio_waveform                       │
│           ↓                                                              │
│  build_video_chunks() [MODIFIED]                                        │
│  ├── Parse 32-byte IVF file header                                      │
│  ├── Loop: parse 12-byte frame headers                                  │
│  ├── Extract frame payload per chunk                                    │
│  ├── VP9 keyframe detection: (payload[0] & 0x20) == 0                  │
│  └── 1 VideoChunk = 1 VP9 frame                                         │
│           ↓                                                              │
│  GossipSub publish (≤16MB per message)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓ P2P
┌─────────────────────────────────────────────────────────────────────────┐
│                          VIEWER (TypeScript)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  P2PClient → SCALE decode → VideoChunkMessage                           │
│           ↓                                                              │
│  VideoPipeline                                                           │
│  ├── handleIncomingChunk() - pass is_keyframe to buffer                │
│  ├── VideoBuffer - store chunks with is_keyframe                        │
│  └── decodeChunk() - create EncodedVideoChunk with correct type        │
│           ↓                                                              │
│  VideoDecoderService (WebCodecs)                                        │
│  ├── checkSupport() - feature detection                                 │
│  ├── init("vp09.00.41.08") - Level 4.1 for 3Mbps                       │
│  └── decode() → render to canvas                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Changes

### 1. New: `vortex/src/vortex/utils/encoder.py`

```python
def encode_video_frames(
    frames: np.ndarray,          # [T, H, W, C] uint8
    fps: int = 24,
    bitrate: int = 3_000_000,
    keyframe_interval: int = 24,
) -> bytes:
    """Encode frames to VP9 IVF container."""

def encode_audio(
    waveform: np.ndarray,        # [samples] float32 24kHz
    sample_rate: int = 24000,
    bitrate: int = 64000,
) -> bytes:
    """Encode audio to Opus stream."""
```

**Implementation notes:**
- Use `av.open(buffer, format='ivf')` for IVF output
- Set `stream.pix_fmt = 'yuv420p'` for Profile 0 compatibility
- Handle exceptions with clear `EncodingError` messages

### 2. Modified: `vortex/plugins/vortex-lane0/plugin.py`

```python
async def _run_async(self, payload: dict[str, Any]) -> dict[str, Any]:
    # ... generation ...

    # Shape normalize: [T,C,H,W] → [T,H,W,C]
    video_np = result.video_frames.cpu().numpy()
    if video_np.ndim == 4 and video_np.shape[1] == 3 and video_np.shape[3] != 3:
        video_np = np.transpose(video_np, (0, 2, 3, 1))

    # Offload encoding to thread pool (non-blocking)
    video_bytes, audio_bytes = await asyncio.gather(
        asyncio.to_thread(encode_video_frames, video_np, fps=fps, ...),
        asyncio.to_thread(encode_audio, audio_np, ...),
    )

    # Free arrays before base64 allocation
    del video_np, audio_np

    return {
        "video_data": base64.b64encode(video_bytes).decode("ascii"),
        "audio_waveform": base64.b64encode(audio_bytes).decode("ascii"),
        ...
    }
```

### 3. Modified: `node-core/crates/p2p/src/video.rs`

Replace fixed-size chunking with IVF frame parsing:

```rust
pub fn build_video_chunks_from_ivf(
    content_id: &str,
    slot: u64,
    ivf_data: &[u8],
    keypair: &Keypair,
) -> Result<Vec<VideoChunk>, VideoChunkError> {
    // Skip 32-byte IVF file header
    let mut pos = 32;
    let mut chunk_index = 0;

    while pos + 12 <= ivf_data.len() {
        // Read 12-byte frame header
        let frame_size = u32::from_le_bytes(...);
        let _timestamp = u64::from_le_bytes(...);

        let frame_start = pos + 12;
        let frame_end = frame_start + frame_size as usize;
        let payload = &ivf_data[frame_start..frame_end];

        // VP9 keyframe detection: Bit 5 of first byte
        let is_keyframe = (payload[0] & 0x20) == 0;

        // Build VideoChunk with actual frame data
        chunks.push(VideoChunk { ... });

        pos = frame_end;
        chunk_index += 1;
    }
}
```

### 4. Modified: `viewer/src/services/videoBuffer.ts`

```typescript
export interface VideoChunk {
    slot: number;
    chunk_index: number;
    data: Uint8Array;
    timestamp: number;
    is_keyframe: boolean;  // ADD
}
```

### 5. Modified: `viewer/src/services/videoPipeline.ts`

```typescript
// handleIncomingChunk - pass is_keyframe
this.buffer.addChunk({
    ...
    is_keyframe: message.is_keyframe,
});

// decodeChunk - use actual keyframe flag
const encodedChunk = new EncodedVideoChunk({
    type: chunk.is_keyframe ? "key" : "delta",
    timestamp: chunk.chunk_index * (1_000_000 / 24), // µs from index
    data: chunk.data,
});
```

### 6. Modified: `viewer/src/components/VideoPlayer/index.tsx`

```typescript
// Change codec string to Level 4.1
await pipeline.init("vp09.00.41.08");
```

### 7. Modified: `viewer/src/services/webcodecs.ts`

```typescript
static checkSupport(): boolean {
    if (VideoDecoderService.isSupported === null) {
        VideoDecoderService.isSupported =
            typeof VideoDecoder !== "undefined" &&
            typeof EncodedVideoChunk !== "undefined";
    }
    return VideoDecoderService.isSupported;
}
```

## New Dependency

```toml
# vortex/pyproject.toml
dependencies = [
    ...
    "av>=12.0.0",  # PyAV for VP9/Opus encoding
]
```

## Encoding Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Video codec | VP9 Profile 0 | `vp09.00.41.08` |
| Video bitrate | 3 Mbps | Good quality at 512x512 |
| Video pixel format | YUV420P | 8-bit, standard |
| Keyframe interval | 24 frames | 1 second at 24fps |
| Audio codec | Opus | Native WebCodecs support |
| Audio sample rate | 24 kHz | Matches Kokoro TTS |
| Audio bitrate | 64 kbps | Voice quality |

## Transport Limits

| Topic | Max Size | VP9 Frame Size | Status |
|-------|----------|----------------|--------|
| VideoChunks | 16 MB | ~50-150 KB (keyframe) | Safe |

## Integration Checklist

- [ ] Add `av>=12.0.0` to pyproject.toml
- [ ] Create `vortex/src/vortex/utils/encoder.py`
- [ ] Update plugin to use encoder + base64
- [ ] Update Rust `build_video_chunks` for IVF parsing
- [ ] Add `is_keyframe` to viewer VideoChunk interface
- [ ] Propagate `is_keyframe` through pipeline
- [ ] Fix `decodeChunk` to use actual keyframe flag
- [ ] Update codec string to Level 4.1
- [ ] Add WebCodecs feature detection
- [ ] Unit test encoder with known tensor shape
- [ ] End-to-end test: Vortex → P2P → Viewer playback

## Verification

1. **Hex dump test**: Verify IVF output starts with `DKIF` magic bytes
2. **Keyframe test**: Check `(frame[0] & 0x20) == 0` for first frame
3. **Browser test**: Confirm no `DOMException` on decoder configure
4. **Playback test**: Video plays at correct speed, no artifacts

## Risks Mitigated

| Risk | Mitigation |
|------|------------|
| Tensor shape mismatch | Explicit transpose [T,C,H,W]→[T,H,W,C] |
| Event loop blocking | `asyncio.to_thread()` for encoding |
| Memory spikes | Early `del` of intermediate arrays |
| VP9 level mismatch | Level 4.1 supports 4K/high bitrate |
| Keyframe detection wrong | VP9 bit 5 check, not index math |
| Timestamp units | Calculate from index × frame duration |
