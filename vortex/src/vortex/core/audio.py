"""Audio generation engine using Kokoro TTS.

This module provides voice synthesis using Kokoro-82M TTS:
- Lightweight (~0.4 GB VRAM)
- 24kHz mono audio output
- Voice selection via voice_id parameter
- Speed control for speech rate

VRAM Management:
- Model is lazy-loaded on first use
- Call unload() before visual pipeline to free GPU memory
"""

from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class AudioEngine:
    """Voice synthesis engine using Kokoro TTS.

    Generates 24kHz mono audio from text using Kokoro-82M.
    Model is lazy-loaded on first generate() call.

    Example:
        >>> engine = AudioEngine(device="cuda")
        >>> path = engine.generate("Hello world", voice_id="af_heart")
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

        # Lazy-loaded model
        self._kokoro_model = None

    def _load_kokoro(self) -> None:
        """Lazy-load Kokoro model."""
        if self._kokoro_model is None:
            logger.info("Loading Kokoro model...")
            from vortex.models.kokoro import load_kokoro
            self._kokoro_model = load_kokoro(device=self.device)
            logger.info("Kokoro model loaded")

    def generate(
        self,
        script: str,
        voice_id: str = "af_heart",
        speed: float = 1.0,
    ) -> str:
        """Generate voice audio using Kokoro TTS.

        Args:
            script: Text to synthesize
            voice_id: Kokoro voice ID (e.g., "af_heart", "af_bella")
            speed: Speech speed multiplier (0.8-1.2 recommended)

        Returns:
            Path to generated WAV file (24kHz mono)

        Raises:
            ValueError: If script is empty
        """
        if not script or script.strip() == "":
            raise ValueError("Script cannot be empty")

        self._load_kokoro()

        output_path = self.output_dir / f"voice_{uuid.uuid4().hex[:8]}.wav"

        # Use Kokoro's synthesize method
        audio = self._kokoro_model.synthesize(
            text=script,
            voice_id=voice_id,
            speed=speed,
        )

        # Save to output path
        import soundfile as sf

        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        sf.write(str(output_path), audio, samplerate=24000)
        logger.info(f"Kokoro generated: {output_path}")
        return str(output_path)

    def unload(self) -> None:
        """Unload Kokoro model and free VRAM.

        Call this before starting the visual pipeline to ensure
        maximum GPU memory is available for ComfyUI.
        """
        if self._kokoro_model is not None:
            logger.info("Unloading Kokoro model...")
            del self._kokoro_model
            self._kokoro_model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Kokoro model unloaded, VRAM freed")
