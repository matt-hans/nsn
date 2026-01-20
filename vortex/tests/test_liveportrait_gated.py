"""Tests for LivePortrait animate_gated method."""

import pytest
import torch
import numpy as np
import tempfile
import soundfile as sf


class TestAnimateGated:
    """Test suite for animate_gated method."""

    def test_animate_gated_returns_tensor(self):
        """animate_gated returns video tensor with correct shape."""
        pytest.skip("Integration test - requires GPU and models")

        # This would be run manually with:
        # from vortex.models.liveportrait import LivePortraitModel
        # model = LivePortraitModel(...)
        # result = model.animate_gated(source_image, audio_path)
        # assert result.shape[0] > 0  # Has frames
        # assert result.shape[1] == 3  # RGB
        # assert result.shape[2] == 512
        # assert result.shape[3] == 512

    def test_animate_gated_method_exists(self):
        """LivePortraitModel has animate_gated method."""
        from vortex.models.liveportrait import LivePortraitModel

        assert hasattr(LivePortraitModel, 'animate_gated')

    def test_euler_to_rotation_matrix_identity(self):
        """Zero angles should produce identity matrix."""
        from vortex.models.liveportrait import LivePortraitModel

        pitch = torch.tensor(0.0)
        yaw = torch.tensor(0.0)
        roll = torch.tensor(0.0)

        R = LivePortraitModel._euler_to_rotation_matrix_torch(pitch, yaw, roll)

        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(R, expected, atol=1e-6)
        assert R.shape == (1, 3, 3)

    def test_euler_to_rotation_matrix_handles_tensor_shapes(self):
        """Method handles various tensor input shapes."""
        from vortex.models.liveportrait import LivePortraitModel

        # Test with different tensor shapes
        pitch = torch.tensor([[0.1]])  # [1, 1] shape
        yaw = torch.tensor([0.2])       # [1] shape
        roll = torch.tensor(0.3)        # scalar

        R = LivePortraitModel._euler_to_rotation_matrix_torch(pitch, yaw, roll)

        assert R.shape == (1, 3, 3)
        # Verify orthogonality (R @ R.T = I)
        RTR = R @ R.transpose(-2, -1)
        assert torch.allclose(RTR, torch.eye(3).unsqueeze(0), atol=1e-5)
