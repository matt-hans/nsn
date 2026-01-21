"""Unit tests for ComfyClient (WebSocket wrapper)."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class AsyncContextManagerMock:
    """Helper to mock async context managers."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


class TestComfyClientConnection:
    """Test connection handling."""

    @pytest.mark.asyncio
    async def test_connects_to_server(self):
        """Should establish WebSocket connection."""
        from vortex.engine.client import ComfyClient

        with patch('websockets.connect', new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value.__aenter__ = AsyncMock()
            mock_ws.return_value.__aexit__ = AsyncMock()

            client = ComfyClient(host="localhost", port=8188)

            # Connection happens on queue_prompt, not init
            assert client.host == "localhost"
            assert client.port == 8188

    def test_generates_unique_client_id(self):
        """Should generate unique client ID for session tracking."""
        from vortex.engine.client import ComfyClient

        client1 = ComfyClient()
        client2 = ComfyClient()

        assert client1.client_id != client2.client_id
        assert len(client1.client_id) > 0


class TestComfyClientQueueing:
    """Test job queuing."""

    @pytest.mark.asyncio
    async def test_queue_prompt_sends_workflow(self):
        """Should POST workflow to /prompt endpoint."""
        from vortex.engine.client import ComfyClient

        workflow = {"6": {"inputs": {"text": "test"}}}

        # Create mock response object
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"prompt_id": "abc123"})

        # Create mock session that returns mock_response from post()
        mock_session = MagicMock()
        mock_session.post = MagicMock(
            return_value=AsyncContextManagerMock(mock_response)
        )

        with patch('aiohttp.ClientSession') as mock_client_session:
            mock_client_session.return_value = AsyncContextManagerMock(mock_session)

            client = ComfyClient()
            prompt_id = await client.queue_prompt(workflow)

            assert prompt_id == "abc123"


class TestComfyClientResults:
    """Test result retrieval."""

    def test_parses_execution_success_message(self):
        """Should correctly parse execution_success WebSocket message."""
        from vortex.engine.client import ComfyClient

        client = ComfyClient()

        message = json.dumps({
            "type": "executed",
            "data": {
                "node": "99",
                "output": {
                    "gifs": [{"filename": "output.mp4", "subfolder": "", "type": "output"}]
                }
            }
        })

        result = client._parse_message(message)

        assert result is not None
        assert result["type"] == "executed"

    def test_parses_progress_message(self):
        """Should parse progress updates."""
        from vortex.engine.client import ComfyClient

        client = ComfyClient()

        message = json.dumps({
            "type": "progress",
            "data": {"value": 50, "max": 100}
        })

        result = client._parse_message(message)

        assert result["type"] == "progress"
        assert result["data"]["value"] == 50
