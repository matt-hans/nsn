"""
Vortex - ICN GPU-resident AI video generation pipeline.

Components:
- models/: Model loaders for Flux-Schnell, CogVideoX, Kokoro, CLIP
- pipeline/: Generation orchestration and slot timing
- plugins/: Pluggable renderer interface for custom pipelines
- utils/: VRAM management and monitoring
"""

__version__ = "0.1.0"
