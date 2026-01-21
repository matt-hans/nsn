"""Workflow payload builder for ComfyUI API.

Handles JSON template loading and parameter injection for the
ComfyUI workflow API format.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default node ID mappings (update these after exporting your workflow)
DEFAULT_NODE_MAP = {
    "prompt": "6",      # CLIPTextEncode (Positive)
    "audio": "40",      # LoadAudio
    "seed": "10",       # KSampler (Flux)
    "seed_ad": "14",    # KSampler (AnimateDiff)
}


class WorkflowBuilder:
    """Builds ComfyUI API payloads from templates.

    Loads a workflow JSON template and injects runtime parameters
    (prompt, audio path, seed) into the appropriate nodes.
    """

    def __init__(
        self,
        template_path: str | None = None,
        template_data: dict[str, Any] | None = None,
        node_map: dict[str, str] | None = None,
    ):
        """Initialize builder.

        Args:
            template_path: Path to workflow JSON file
            template_data: Workflow dict (alternative to file)
            node_map: Mapping of parameter names to node IDs

        Raises:
            ValueError: If neither template_path nor template_data provided
        """
        if template_data is not None:
            self._template = template_data
        elif template_path is not None:
            self._template = self._load_template(template_path)
        else:
            raise ValueError("Must provide template_path or template_data")

        self._node_map = node_map or DEFAULT_NODE_MAP

    def _load_template(self, path: str) -> dict[str, Any]:
        """Load workflow template from JSON file."""
        template_path = Path(path)
        if not template_path.exists():
            raise FileNotFoundError(f"Workflow template not found: {path}")

        with open(template_path) as f:
            return json.load(f)

    def build(
        self,
        prompt: str | None = None,
        audio_path: str | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build workflow payload with injected parameters.

        Args:
            prompt: Text prompt for image generation
            audio_path: Path to audio file for LivePortrait
            seed: Deterministic seed (random if not provided)

        Returns:
            Complete workflow dict ready for ComfyUI API
        """
        # Deep copy to avoid mutating template
        workflow = copy.deepcopy(self._template)

        # Inject prompt
        if prompt is not None:
            prompt_node = self._node_map.get("prompt", "6")
            if prompt_node in workflow:
                workflow[prompt_node]["inputs"]["text"] = prompt
                logger.debug(f"Injected prompt into node {prompt_node}")

        # Inject audio path
        if audio_path is not None:
            audio_node = self._node_map.get("audio", "40")
            if audio_node in workflow:
                workflow[audio_node]["inputs"]["audio"] = audio_path
                logger.debug(f"Injected audio path into node {audio_node}")

        # Inject seed (generate random if not provided)
        actual_seed = seed if seed is not None else random.randint(1, 2**31 - 1)

        seed_node = self._node_map.get("seed", "10")
        if seed_node in workflow:
            workflow[seed_node]["inputs"]["seed"] = actual_seed
            logger.debug(f"Injected seed {actual_seed} into node {seed_node}")

        # Also set AnimateDiff seed if present
        seed_ad_node = self._node_map.get("seed_ad")
        if seed_ad_node and seed_ad_node in workflow:
            workflow[seed_ad_node]["inputs"]["seed"] = actual_seed
            logger.debug(f"Injected seed {actual_seed} into AnimateDiff node")

        return workflow

    def get_node_ids(self) -> dict[str, str]:
        """Return current node ID mappings."""
        return self._node_map.copy()


def load_workflow(template_path: str) -> WorkflowBuilder:
    """Convenience function to create a WorkflowBuilder from file.

    Args:
        template_path: Path to workflow JSON

    Returns:
        Configured WorkflowBuilder instance
    """
    return WorkflowBuilder(template_path=template_path)
