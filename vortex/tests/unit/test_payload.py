"""Unit tests for WorkflowBuilder (JSON injection)."""

import json
import pytest
from pathlib import Path


class TestWorkflowBuilder:
    """Test JSON template injection."""

    def test_injects_prompt(self):
        """Should inject prompt into correct node."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "6": {"inputs": {"text": "__PROMPT__"}},
            "10": {"inputs": {"seed": 0}},
        }

        builder = WorkflowBuilder(template_data=template)
        result = builder.build(prompt="A cyberpunk cat")

        assert result["6"]["inputs"]["text"] == "A cyberpunk cat"

    def test_injects_audio_path(self):
        """Should inject audio path into correct node."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "40": {"inputs": {"audio": "__AUDIO__"}},
        }

        builder = WorkflowBuilder(
            template_data=template,
            node_map={"audio": "40"}
        )
        result = builder.build(audio_path="/tmp/voice.wav")

        assert result["40"]["inputs"]["audio"] == "/tmp/voice.wav"

    def test_injects_seed(self):
        """Should inject seed into KSampler node."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "10": {"inputs": {"seed": 0}},
        }

        builder = WorkflowBuilder(
            template_data=template,
            node_map={"seed": "10"}
        )
        result = builder.build(seed=12345)

        assert result["10"]["inputs"]["seed"] == 12345

    def test_generates_random_seed_if_not_provided(self):
        """Should generate random seed if not specified."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "10": {"inputs": {"seed": 0}},
        }

        builder = WorkflowBuilder(
            template_data=template,
            node_map={"seed": "10"}
        )
        result1 = builder.build()
        result2 = builder.build()

        # Seeds should be different (random)
        assert result1["10"]["inputs"]["seed"] != 0
        assert result2["10"]["inputs"]["seed"] != 0

    def test_loads_template_from_file(self, tmp_path):
        """Should load template from JSON file."""
        from vortex.engine.payload import WorkflowBuilder

        template = {"6": {"inputs": {"text": "default"}}}
        template_path = tmp_path / "workflow.json"
        template_path.write_text(json.dumps(template))

        builder = WorkflowBuilder(template_path=str(template_path))
        result = builder.build(prompt="Test prompt")

        assert result["6"]["inputs"]["text"] == "Test prompt"

    def test_preserves_unmodified_nodes(self):
        """Should not modify nodes that aren't in the injection map."""
        from vortex.engine.payload import WorkflowBuilder

        template = {
            "6": {"inputs": {"text": "__PROMPT__"}},
            "99": {"inputs": {"special": "value", "other": 123}},
        }

        builder = WorkflowBuilder(template_data=template)
        result = builder.build(prompt="Test")

        assert result["99"]["inputs"]["special"] == "value"
        assert result["99"]["inputs"]["other"] == 123
