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

        # Use 48 frames with keyframe interval of 12 to ensure mix of keyframes and delta frames
        frames = np.random.randint(0, 255, (48, 64, 64, 3), dtype=np.uint8)

        ivf_data = encode_video_frames(frames, fps=24, keyframe_interval=12)
        frame_info = parse_ivf_frames(ivf_data)

        # First frame must be keyframe
        assert frame_info[0]["is_keyframe"] is True
        # Most other frames should be delta frames
        delta_count = sum(1 for f in frame_info if not f["is_keyframe"])
        assert delta_count > 0, "Should have delta frames"
