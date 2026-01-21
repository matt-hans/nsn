"""Integration test for ComfyUI connection."""

import pytest
import requests


@pytest.mark.integration
def test_comfyui_is_running():
    """Verify ComfyUI server is accessible."""
    try:
        response = requests.get(
            "http://127.0.0.1:8188/system_stats",
            timeout=5,
        )
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
    except requests.exceptions.ConnectionError:
        pytest.skip("ComfyUI is not running on port 8188")
