"""ToonGen core modules.

- audio: Voice synthesis (F5-TTS, Kokoro) with graceful degradation
- mixer: Audio compositing (FFmpeg-based BGM/SFX mixing)
"""

from vortex.core.audio import AudioEngine
from vortex.core.mixer import AudioCompositor, calculate_frame_count

__all__ = ["AudioEngine", "AudioCompositor", "calculate_frame_count"]
