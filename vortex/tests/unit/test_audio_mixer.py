"""Unit tests for AudioCompositor (FFmpeg mixing)."""

from unittest.mock import MagicMock, patch

import pytest


class TestAudioCompositor:
    """Test FFmpeg mixing logic."""

    def test_mix_voice_only(self, tmp_path):
        """Mixing with voice only should just copy the voice file."""
        from vortex.core.mixer import AudioCompositor

        voice_path = tmp_path / "voice.wav"
        voice_path.write_bytes(b"fake voice data")
        output_path = tmp_path / "output.wav"

        compositor = AudioCompositor(assets_dir=str(tmp_path / "assets"))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            compositor.mix(
                voice_path=str(voice_path),
                output_path=str(output_path),
                bgm_name=None,
                sfx_name=None,
            )

            mock_run.assert_called_once()
            # Verify FFmpeg was called
            call_args = mock_run.call_args[0][0]
            assert 'ffmpeg' in call_args[0]

    def test_mix_with_bgm_applies_volume(self, tmp_path):
        """Mixing with BGM should apply volume filter."""
        from vortex.core.mixer import AudioCompositor

        voice_path = tmp_path / "voice.wav"
        voice_path.write_bytes(b"fake voice data")

        assets_dir = tmp_path / "assets" / "audio" / "bgm"
        assets_dir.mkdir(parents=True)
        bgm_path = assets_dir / "elevator.wav"
        bgm_path.write_bytes(b"fake bgm data")

        output_path = tmp_path / "output.wav"

        compositor = AudioCompositor(assets_dir=str(tmp_path / "assets"))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            compositor.mix(
                voice_path=str(voice_path),
                output_path=str(output_path),
                bgm_name="elevator",
                mix_ratio=0.3,
            )

            # Verify FFmpeg command includes volume filter
            call_args = str(mock_run.call_args)
            assert 'volume' in call_args or 'amix' in call_args

    def test_mix_raises_on_ffmpeg_failure(self, tmp_path):
        """Should raise RuntimeError if FFmpeg fails."""
        from vortex.core.mixer import AudioCompositor

        voice_path = tmp_path / "voice.wav"
        voice_path.write_bytes(b"fake voice data")
        output_path = tmp_path / "output.wav"

        compositor = AudioCompositor(assets_dir=str(tmp_path / "assets"))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")

            with pytest.raises(RuntimeError):
                compositor.mix(
                    voice_path=str(voice_path),
                    output_path=str(output_path),
                )


class TestAudioDurationCalculation:
    """Test audio duration utilities."""

    def test_calculate_frame_count_from_duration(self):
        """Should calculate correct frame count from audio duration."""
        from vortex.core.mixer import calculate_frame_count

        # 10 seconds at 24fps = 240 frames
        with patch('soundfile.SoundFile') as mock_sf:
            mock_file = MagicMock()
            mock_file.__len__ = MagicMock(return_value=240000)  # 10s at 24kHz
            mock_file.samplerate = 24000
            mock_sf.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_sf.return_value.__exit__ = MagicMock(return_value=False)

            frames = calculate_frame_count("dummy.wav", fps=24)
            assert frames == 240

    def test_calculate_frame_count_pads_short_audio(self):
        """Should pad to minimum 16 frames for AnimateDiff context."""
        from vortex.core.mixer import calculate_frame_count

        # 0.5 seconds at 24fps = 12 frames, should pad to 16
        with patch('soundfile.SoundFile') as mock_sf:
            mock_file = MagicMock()
            mock_file.__len__ = MagicMock(return_value=12000)  # 0.5s at 24kHz
            mock_file.samplerate = 24000
            mock_sf.return_value.__enter__ = MagicMock(return_value=mock_file)
            mock_sf.return_value.__exit__ = MagicMock(return_value=False)

            frames = calculate_frame_count("dummy.wav", fps=24, min_frames=16)
            assert frames == 16
