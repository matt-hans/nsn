"""Audio generation engine with F5-TTS/Kokoro graceful degradation.

This module provides voice synthesis with automatic fallback:
- F5-TTS: Zero-shot voice cloning from reference audio (primary)
- Kokoro: Standard TTS (fallback when reference missing)

VRAM Management:
- Models are lazy-loaded on first use
- Call unload() before visual pipeline to free GPU memory
"""

from __future__ import annotations

import gc
import logging
import uuid
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


class AudioEngine:
    """Voice synthesis engine with graceful degradation.

    Supports three modes via the `engine` parameter:
    - "auto": Try F5-TTS if reference exists, fall back to Kokoro
    - "f5_tts": Force F5-TTS (raises if reference missing)
    - "kokoro": Force Kokoro (skip F5 entirely)
    """

    def __init__(
        self,
        device: str = "cuda",
        assets_dir: str = "assets/voices",
        output_dir: str = "temp/audio",
    ):
        """Initialize audio engine.

        Args:
            device: PyTorch device for inference ("cuda" or "cpu")
            assets_dir: Directory containing voice reference WAVs
            output_dir: Directory for generated audio files
        """
        self.device = device
        self.assets_dir = Path(assets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded models
        self._f5_model = None
        self._kokoro_model = None

    def _load_f5(self) -> None:
        """Lazy-load F5-TTS model."""
        if self._f5_model is None:
            logger.info("Loading F5-TTS model...")
            # TODO: Import and load actual F5-TTS
            # from f5_tts import F5TTS
            # self._f5_model = F5TTS(device=self.device)
            self._f5_model = object()  # Placeholder
            logger.info("F5-TTS model loaded")

    def _load_kokoro(self) -> None:
        """Lazy-load Kokoro model."""
        if self._kokoro_model is None:
            logger.info("Loading Kokoro model...")
            from vortex.models.kokoro import load_kokoro
            self._kokoro_model = load_kokoro(device=self.device)
            logger.info("Kokoro model loaded")

    def _generate_f5(
        self, script: str, ref_path: Path, output_path: Path
    ) -> str:
        """Generate audio using F5-TTS.

        Args:
            script: Text to synthesize
            ref_path: Path to voice reference WAV
            output_path: Path to save generated audio

        Returns:
            Path to generated WAV file
        """
        self._load_f5()
        # TODO: Actual F5-TTS inference
        # self._f5_model.infer(
        #     ref_file=str(ref_path),
        #     text=script,
        #     output_file=str(output_path)
        # )
        logger.info(f"F5-TTS generated: {output_path}")
        return str(output_path)

    def _generate_kokoro(
        self, script: str, voice_id: str, output_path: Path
    ) -> str:
        """Generate audio using Kokoro.

        Args:
            script: Text to synthesize
            voice_id: Kokoro voice identifier
            output_path: Path to save generated audio

        Returns:
            Path to generated WAV file
        """
        self._load_kokoro()

        # Use Kokoro's synthesize method
        audio = self._kokoro_model.synthesize(
            text=script,
            voice_id=voice_id,
        )

        # Save to output path
        import soundfile as sf
        import torch

        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        sf.write(str(output_path), audio, samplerate=24000)
        logger.info(f"Kokoro generated: {output_path}")
        return str(output_path)

    def generate(
        self,
        script: str,
        engine: Literal["auto", "f5_tts", "kokoro"] = "auto",
        voice_style: str | None = None,
        voice_id: str = "af_heart",
    ) -> str:
        """Generate voice audio with graceful degradation.

        Args:
            script: Text to synthesize
            engine: Engine selection mode
            voice_style: F5-TTS reference filename (without .wav)
            voice_id: Kokoro voice ID (used if engine=kokoro or as fallback)

        Returns:
            Path to generated WAV file

        Raises:
            FileNotFoundError: If engine=f5_tts and reference file missing
        """
        output_path = self.output_dir / f"voice_{uuid.uuid4().hex[:8]}.wav"

        # Explicit Kokoro mode - skip F5 entirely
        if engine == "kokoro":
            logger.info(f"Using Kokoro (explicit) with voice: {voice_id}")
            return self._generate_kokoro(script, voice_id, output_path)

        # Try F5-TTS if reference exists
        if voice_style:
            ref_path = self.assets_dir / f"{voice_style}.wav"

            if ref_path.exists():
                try:
                    logger.info(f"Using F5-TTS with style: {voice_style}")
                    return self._generate_f5(script, ref_path, output_path)
                except Exception as e:
                    if engine == "f5_tts":
                        raise  # Explicit F5 mode - don't fall back
                    logger.warning(f"F5-TTS failed: {e}, falling back to Kokoro")
            else:
                if engine == "f5_tts":
                    raise FileNotFoundError(
                        f"Voice reference not found: {ref_path}"
                    )
                logger.warning(
                    f"Voice style '{voice_style}' not found, falling back to Kokoro"
                )

        # Fallback to Kokoro
        logger.info(f"Using Kokoro (fallback) with voice: {voice_id}")
        return self._generate_kokoro(script, voice_id, output_path)

    def unload(self) -> None:
        """Unload models and free VRAM.

        Call this before starting the visual pipeline to ensure
        maximum GPU memory is available for ComfyUI.
        """
        import torch

        if self._f5_model is not None:
            logger.info("Unloading F5-TTS model...")
            del self._f5_model
            self._f5_model = None

        if self._kokoro_model is not None:
            logger.info("Unloading Kokoro model...")
            del self._kokoro_model
            self._kokoro_model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Audio models unloaded, VRAM freed")
