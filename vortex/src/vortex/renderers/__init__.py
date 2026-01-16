"""Lane 0 video renderer system.

This module provides the infrastructure for modular video generation backends.
All Lane 0 renderers must implement the DeterministicVideoRenderer interface
to guarantee reproducible output for BFT consensus verification.

Example:
    from vortex.renderers import RendererRegistry, RendererPolicy

    # Load renderers from directory
    policy = RendererPolicy(max_vram_gb=11.5, max_latency_ms=15000)
    registry = RendererRegistry.from_directory(Path("renderers"), policy)

    # Get and initialize a renderer
    renderer = registry.get("default-flux-liveportrait")
    await renderer.initialize("cuda:0", config)

    # Render a slot
    result = await renderer.render(recipe, slot_id=1, seed=42, deadline=time.time() + 45)
"""

from vortex.renderers.base import DeterministicVideoRenderer
from vortex.renderers.errors import (
    DeterminismError,
    RecipeValidationError,
    RendererError,
    RendererLoadError,
    RendererNotFoundError,
    RendererPolicyError,
)
from vortex.renderers.policy import RendererPolicy, policy_from_config
from vortex.renderers.recipe_schema import (
    RECIPE_SCHEMA,
    get_recipe_defaults,
    merge_with_defaults,
    validate_recipe,
)
from vortex.renderers.registry import RendererRegistry, load_manifest, load_renderer
from vortex.renderers.types import (
    LANE0_MAX_LATENCY_MS,
    LANE0_MAX_VRAM_GB,
    RendererManifest,
    RendererResources,
    RenderResult,
)

__all__ = [
    # Base class
    "DeterministicVideoRenderer",
    # Types
    "RendererManifest",
    "RendererResources",
    "RenderResult",
    # Registry
    "RendererRegistry",
    "load_manifest",
    "load_renderer",
    # Policy
    "RendererPolicy",
    "policy_from_config",
    # Recipe
    "RECIPE_SCHEMA",
    "validate_recipe",
    "get_recipe_defaults",
    "merge_with_defaults",
    # Errors
    "RendererError",
    "RendererLoadError",
    "RendererNotFoundError",
    "RendererPolicyError",
    "RecipeValidationError",
    "DeterminismError",
    # Constants
    "LANE0_MAX_VRAM_GB",
    "LANE0_MAX_LATENCY_MS",
]
