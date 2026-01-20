"""Tests for AudioGatedDriver."""


import numpy as np


# We'll create a simple test audio file
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


class TestAudioGatedDriver:
    """Test suite for AudioGatedDriver."""

    def test_driver_initialization(self, tmp_path):
        """Driver initializes with valid template path."""
        from vortex.models.audio_driver import AudioGatedDriver

        # Create a mock template
        template_path = tmp_path / "test_template.pkl"
        mock_template = {
            'n_frames': 10,
            'output_fps': 30,
            'motion': [
                {
                    'scale': np.ones((1, 1), dtype=np.float32),
                    'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                    'exp': np.zeros((1, 21, 3), dtype=np.float32),
                    't': np.zeros((1, 3), dtype=np.float32),
                }
                for _ in range(10)
            ],
            'c_d_eyes_lst': [np.zeros((1, 2), dtype=np.float32) for _ in range(10)],
            'c_d_lip_lst': [np.zeros((1, 1), dtype=np.float32) for _ in range(10)],
        }

        import pickle
        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        driver = AudioGatedDriver(str(template_path), device="cpu")
        assert driver.template_len == 10
        assert driver.output_fps == 30

    def test_drive_returns_correct_structure(self, tmp_path):
        """Drive method returns motion dict with correct keys and shapes."""
        import torch

        from vortex.models.audio_driver import AudioGatedDriver

        # Create mock template
        template_path = tmp_path / "test_template.pkl"
        n_template_frames = 30
        mock_template = {
            'n_frames': n_template_frames,
            'output_fps': 30,
            'motion': [
                {
                    'scale': np.ones((1, 1), dtype=np.float32),
                    'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                    'exp': np.random.randn(1, 21, 3).astype(np.float32) * 0.01,
                    't': np.zeros((1, 3), dtype=np.float32),
                }
                for _ in range(n_template_frames)
            ],
            'c_d_eyes_lst': [
                np.random.rand(1, 2).astype(np.float32)
                for _ in range(n_template_frames)
            ],
            'c_d_lip_lst': [
                np.random.rand(1, 1).astype(np.float32)
                for _ in range(n_template_frames)
            ],
        }

        import pickle
        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        # Create test audio
        audio_path = tmp_path / "test_audio.wav"
        create_test_audio(str(audio_path), duration_sec=2.0)

        # Create mock source_info (what LivePortrait extracts from source image)
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

        # Check motion list structure
        assert len(result['motion']) == result['n_frames']
        assert 'exp' in result['motion'][0]
        assert 'scale' in result['motion'][0]
        assert 'R_d' in result['motion'][0]
        assert 't' in result['motion'][0]

    def test_energy_gates_expression(self, tmp_path):
        """Expression magnitude correlates with audio energy."""
        import torch

        from vortex.models.audio_driver import AudioGatedDriver

        # Create template with varying expression (frame 0 is neutral, others have motion)
        # This models real human motion where frame 0 is the rest pose
        template_path = tmp_path / "test_template.pkl"
        n_template_frames = 60

        # Frame 0 is neutral (zero), other frames have non-zero expression
        motion_list = []
        for i in range(n_template_frames):
            if i == 0:
                # Neutral frame (rest pose)
                exp = np.zeros((1, 21, 3), dtype=np.float32)
            else:
                # Motion frames with non-zero expression
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
            'c_d_eyes_lst': [
                np.ones((1, 2), dtype=np.float32) * 0.5
                for _ in range(n_template_frames)
            ],
            'c_d_lip_lst': [
                np.ones((1, 1), dtype=np.float32) * 0.5
                for _ in range(n_template_frames)
            ],
        }

        import pickle
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

        # Get expression magnitudes for silent vs loud frames
        # Frames 0-12 should be silent (first 0.5s at 24fps)
        # Frames 12-36 should be loud (0.5s-1.5s)
        # Frames 36-48 should be silent again

        silent_exp = result['motion'][5]['exp']
        loud_exp = result['motion'][24]['exp']

        silent_mag = torch.abs(torch.tensor(silent_exp)).mean().item()
        loud_mag = torch.abs(torch.tensor(loud_exp)).mean().item()

        # Loud frames should have larger expression magnitude
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

        # Create a single-frame template
        template_path = tmp_path / "single_frame_template.pkl"
        mock_template = {
            'n_frames': 1,
            'output_fps': 30,
            'motion': [
                {
                    'scale': np.ones((1, 1), dtype=np.float32),
                    'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                    'exp': np.zeros((1, 21, 3), dtype=np.float32),
                    't': np.zeros((1, 3), dtype=np.float32),
                }
            ],
            'c_d_eyes_lst': [np.zeros((1, 2), dtype=np.float32)],
            'c_d_lip_lst': [np.zeros((1, 1), dtype=np.float32)],
        }

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

        # Create mock template
        template_path = tmp_path / "test_template.pkl"
        mock_template = {
            'n_frames': 10,
            'output_fps': 30,
            'motion': [
                {
                    'scale': np.ones((1, 1), dtype=np.float32),
                    'R_d': np.eye(3, dtype=np.float32).reshape(1, 3, 3),
                    'exp': np.zeros((1, 21, 3), dtype=np.float32),
                    't': np.zeros((1, 3), dtype=np.float32),
                }
                for _ in range(10)
            ],
            'c_d_eyes_lst': [np.zeros((1, 2), dtype=np.float32) for _ in range(10)],
            'c_d_lip_lst': [np.zeros((1, 1), dtype=np.float32) for _ in range(10)],
        }

        with open(template_path, 'wb') as f:
            pickle.dump(mock_template, f)

        # Create test audio
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
