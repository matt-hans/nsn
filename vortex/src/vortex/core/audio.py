"""Audio generation engine using Bark TTS.

This module provides voice synthesis using Bark TTS:
- Expressive, chaotic audio for "Interdimensional Cable" aesthetic
- Paralinguistic sounds ([laughter], [gasps], etc.)
- Voice selection via speaker presets or zero-shot cloning
- Emotion control via temperature settings
- 24kHz mono audio output

VRAM Management:
- Model is lazy-loaded on first use
- Call unload() before visual pipeline to free GPU memory
"""

from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path

import soundfile as sf
import torch

from vortex.models.bark import BarkVoiceEngine

logger = logging.getLogger(__name__)


class AudioEngine:
    """Voice synthesis engine using Bark TTS.

    Generates 24kHz mono audio from text using Bark.
    Supports paralinguistic tokens for expressive audio.
    Model is lazy-loaded on first generate() call.

    Example:
        >>> engine = AudioEngine(device="cuda")
        >>> path = engine.generate(
        ...     "[laughs] Hello world!",
        ...     voice_id="rick_c137",
        ...     emotion="manic"
        ... )
        >>> print(path)  # temp/audio/voice_abc123.wav
        >>> engine.unload()  # Free VRAM when done
    """

    def __init__(
        self,
        device: str = "cuda",
        output_dir: str = "temp/audio",
    ):
        """Initialize audio engine.

        Args:
            device: PyTorch device for inference ("cuda" or "cpu")
            output_dir: Directory for generated audio files
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded Bark engine
        self._bark_engine = None

    def _load_bark(self) -> None:
        """Lazy-load Bark engine."""
        if self._bark_engine is None:
            logger.info("Loading Bark engine...")
            self._bark_engine = BarkVoiceEngine(device=self.device)
            logger.info("Bark engine loaded")

    def generate(
        self,
        script: str,
        voice_id: str = "default",
        emotion: str = "neutral",
        seed: int | None = None,
    ) -> str:
        """Generate voice audio using Bark TTS.

        Args:
            script: Text to synthesize (supports Bark tokens like [laughter])
            voice_id: Voice profile ID (e.g., "rick_c137", "morty")
            emotion: Emotion for temperature control (e.g., "manic", "neutral")
            seed: Optional seed for reproducibility

        Returns:
            Path to generated WAV file (24kHz mono)

        Raises:
            ValueError: If script is empty
        """
        if not script or script.strip() == "":
            raise ValueError("Script cannot be empty")

        self._load_bark()

        output_path = self.output_dir / f"voice_{uuid.uuid4().hex[:8]}.wav"

        # Generate using Bark
        audio = self._bark_engine.synthesize(
            text=script,
            voice_id=voice_id,
            emotion=emotion,
            seed=seed,
        )

        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Save to output path
        sf.write(str(output_path), audio, samplerate=24000)
        logger.info(f"Bark generated: {output_path}")

        return str(output_path)

    def unload(self) -> None:
        """Unload Bark engine and free VRAM.

        Call this before starting the visual pipeline to ensure
        maximum GPU memory is available.
        """
        if self._bark_engine is not None:
            logger.info("Unloading Bark engine...")
            self._bark_engine.unload()
            del self._bark_engine
            self._bark_engine = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Bark engine unloaded, VRAM freed")
