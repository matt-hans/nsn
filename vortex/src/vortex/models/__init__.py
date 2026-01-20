"""ToonGen model wrappers.

This package provides wrappers for:
- F5-TTS: Zero-shot voice cloning
- Kokoro: Fallback TTS engine
"""

from vortex.models.kokoro import KokoroWrapper

__all__ = ["KokoroWrapper"]
