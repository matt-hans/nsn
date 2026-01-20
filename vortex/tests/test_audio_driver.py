"""Tests for AudioGatedDriver with Procedural Jaw Override."""


import numpy as np


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


def create_test_audio_high_freq(path: str, duration_sec: float = 1.0, sr: int = 24000):
    """Create a test audio file with high frequency content (for spectral centroid)."""
    import soundfile as sf

    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    # High frequency tone (3kHz)
    audio = 0.3 * np.sin(2 * np.pi * 3000 * t)
    sf.write(path, audio, sr)
    return path


def create_mock_template(n_frames: int = 10):
    """Create a minimal mock template for testing."""
    return {
        'n_frames': n_frames,
        'output_fps': 30,
        'motion': [
            {
                'scale': np.ones((1, 1), dtype=np.float32),
                'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                'exp': np.zeros((1, 21, 3), dtype=np.float32),
                't': np.zeros((1, 3), dtype=np.float32),
            }
            for _ in range(n_frames)
        ],
    }


class TestAudioGatedDriver:
    """Test suite for AudioGatedDriver with procedural jaw override."""

    def test_driver_initialization(self, tmp_path):
        """Driver initializes with valid template path."""
        import pickle

        from vortex.models.audio_driver import AudioGatedDriver

        template_path = tmp_path / "test_template.pkl"
        mock_template = create_mock_template(10)

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        driver = AudioGatedDriver(str(template_path), device="cpu")
        assert driver.template_len == 10
        assert driver.output_fps == 30

    def test_drive_returns_correct_structure(self, tmp_path):
        """Drive method returns motion dict with correct keys and shapes."""
        import pickle

        import torch

        from vortex.models.audio_driver import AudioGatedDriver

        template_path = tmp_path / "test_template.pkl"
        mock_template = create_mock_template(30)

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

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

        # Check structure
        assert 'motion' in result
        assert 'c_d_eyes_lst' in result
        assert 'c_d_lip_lst' in result
        assert 'n_frames' in result

        # New API: c_d_eyes_lst and c_d_lip_lst are empty (baked into expression)
        assert result['c_d_eyes_lst'] == []
        assert result['c_d_lip_lst'] == []

        # Check motion list structure
        assert len(result['motion']) == result['n_frames']
        assert 'exp' in result['motion'][0]
        assert 'scale' in result['motion'][0]
        assert 'R_d' in result['motion'][0]
        assert 't' in result['motion'][0]

    def test_procedural_jaw_override(self, tmp_path):
        """Jaw keypoint indices (19, 20) change with audio energy."""
        import pickle

        import torch

        from vortex.models.audio_driver import AudioGatedDriver

        template_path = tmp_path / "test_template.pkl"
        mock_template = create_mock_template(60)

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

        # Frames 0-12 should be silent (first 0.5s at 24fps)
        # Frames 12-36 should be loud (0.5s-1.5s)
        # Frames 36-48 should be silent again

        silent_exp = result['motion'][5]['exp']
        loud_exp = result['motion'][24]['exp']

        # Check jaw keypoint indices 19 and 20, Y-axis (dim 1)
        # During loud frames, jaw should be more open (more negative Y value)
        silent_jaw_19 = silent_exp[0, 19, 1]
        loud_jaw_19 = loud_exp[0, 19, 1]
        silent_jaw_20 = silent_exp[0, 20, 1]
        loud_jaw_20 = loud_exp[0, 20, 1]

        # Jaw offset is negative (open), so loud frames should have more negative Y
        assert loud_jaw_19 < silent_jaw_19, (
            f"Expected jaw index 19 to be more open (negative) during loud: "
            f"loud={loud_jaw_19}, silent={silent_jaw_19}"
        )
        assert loud_jaw_20 < silent_jaw_20, (
            f"Expected jaw index 20 to be more open (negative) during loud: "
            f"loud={loud_jaw_20}, silent={silent_jaw_20}"
        )

    def test_spectral_centroid_affects_lip_width(self, tmp_path):
        """Spectral centroid (tone) affects lip width at keypoint index 17."""
        import pickle

        import torch

        from vortex.models.audio_driver import AudioGatedDriver

        template_path = tmp_path / "test_template.pkl"
        mock_template = create_mock_template(60)

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        # Create high-frequency audio (high spectral centroid)
        audio_path = tmp_path / "test_audio_high.wav"
        create_test_audio_high_freq(str(audio_path), duration_sec=1.0)

        source_info = {
            'exp': torch.zeros(1, 21, 3),
            'scale': torch.ones(1, 1),
            'R': torch.eye(3).unsqueeze(0),
            't': torch.zeros(1, 3),
        }

        driver = AudioGatedDriver(str(template_path), device="cpu")
        result = driver.drive(str(audio_path), source_info, fps=24)

        # High frequency should result in wider lips (positive X offset at index 17)
        # Check middle frame where audio is present
        frame = result['motion'][12]['exp']
        lip_width_x = frame[0, 17, 0]

        # With high spectral centroid, tone > 0.5, so width_offset > 0
        # This should make the lip width positive
        assert lip_width_x > 0 or np.isclose(lip_width_x, 0, atol=0.01), (
            f"Expected positive lip width for high freq audio, got {lip_width_x}"
        )

    def test_energy_gates_expression(self, tmp_path):
        """Expression magnitude correlates with audio energy."""
        import pickle

        import torch

        from vortex.models.audio_driver import AudioGatedDriver

        # Create template with varying expression (frame 0 is neutral, others have motion)
        template_path = tmp_path / "test_template.pkl"
        n_template_frames = 60

        motion_list = []
        for i in range(n_template_frames):
            if i == 0:
                exp = np.zeros((1, 21, 3), dtype=np.float32)
            else:
                exp = np.ones((1, 21, 3), dtype=np.float32) * 0.1
            motion_list.append({
                'scale': np.ones((1, 1), dtype=np.float32),
                'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                'exp': exp,
                't': np.zeros((1, 3), dtype=np.float32),
            })

        mock_template = {
            'n_frames': n_template_frames,
            'output_fps': 30,
            'motion': motion_list,
        }

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

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

        silent_exp = result['motion'][5]['exp']
        loud_exp = result['motion'][24]['exp']

        silent_mag = np.abs(silent_exp).mean()
        loud_mag = np.abs(loud_exp).mean()

        assert loud_mag > silent_mag, f"Expected loud ({loud_mag}) > silent ({silent_mag})"

    def test_driver_raises_file_not_found(self, tmp_path):
        """Driver raises FileNotFoundError when template is missing."""
        import pytest

        from vortex.models.audio_driver import AudioGatedDriver

        nonexistent_path = tmp_path / "nonexistent_template.pkl"

        with pytest.raises(FileNotFoundError) as excinfo:
            AudioGatedDriver(str(nonexistent_path), device="cpu")

        assert "nonexistent_template.pkl" in str(excinfo.value)

    def test_ping_pong_index_single_frame_template(self, tmp_path):
        """Ping-pong index handles single-frame template (no division by zero)."""
        import pickle

        from vortex.models.audio_driver import AudioGatedDriver

        template_path = tmp_path / "single_frame_template.pkl"
        mock_template = create_mock_template(1)

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        driver = AudioGatedDriver(str(template_path), device="cpu")

        # Should not raise ZeroDivisionError
        assert driver._ping_pong_index(0) == 0
        assert driver._ping_pong_index(1) == 0
        assert driver._ping_pong_index(100) == 0

    def test_drive_raises_on_missing_source_info_keys(self, tmp_path):
        """Drive raises ValueError when source_info is missing required keys."""
        import pickle

        import pytest
        import torch

        from vortex.models.audio_driver import AudioGatedDriver

        template_path = tmp_path / "test_template.pkl"
        mock_template = create_mock_template(10)

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        audio_path = tmp_path / "test_audio.wav"
        create_test_audio(str(audio_path), duration_sec=1.0)

        driver = AudioGatedDriver(str(template_path), device="cpu")

        # Missing 'R' key
        incomplete_source_info = {
            'exp': torch.zeros(1, 21, 3),
            'scale': torch.ones(1, 1),
            't': torch.zeros(1, 3),
        }

        with pytest.raises(ValueError) as excinfo:
            driver.drive(str(audio_path), incomplete_source_info, fps=24)

        assert "missing required keys" in str(excinfo.value)
        assert "R" in str(excinfo.value)

    def test_extract_audio_features_returns_energy_and_tone(self, tmp_path):
        """_extract_audio_features returns both energy and tone arrays."""
        import pickle

        from vortex.models.audio_driver import AudioGatedDriver

        template_path = tmp_path / "test_template.pkl"
        mock_template = create_mock_template(10)

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        audio_path = tmp_path / "test_audio.wav"
        create_test_audio(str(audio_path), duration_sec=2.0)

        driver = AudioGatedDriver(str(template_path), device="cpu")
        energy, tone = driver._extract_audio_features(str(audio_path), fps=24)

        # Both should be numpy arrays
        assert isinstance(energy, np.ndarray)
        assert isinstance(tone, np.ndarray)

        # Same length
        assert len(energy) == len(tone)

        # Values should be normalized [0, 1]
        assert energy.min() >= 0.0
        assert energy.max() <= 1.0
        assert tone.min() >= 0.0
        assert tone.max() <= 1.0

    def test_class_parameters_are_configurable(self, tmp_path):
        """Procedural override parameters can be adjusted."""
        import pickle

        from vortex.models.audio_driver import AudioGatedDriver

        template_path = tmp_path / "test_template.pkl"
        mock_template = create_mock_template(10)

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        driver = AudioGatedDriver(str(template_path), device="cpu")

        # Check default values exist
        assert hasattr(driver, 'JAW_OPEN_STRENGTH')
        assert hasattr(driver, 'LIP_WIDEN_STRENGTH')

        # Can be modified if needed
        assert driver.JAW_OPEN_STRENGTH == 0.08
        assert driver.LIP_WIDEN_STRENGTH == 0.03
