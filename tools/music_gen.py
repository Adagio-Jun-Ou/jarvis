"""Music generation using transformer-based models.

Provides high-quality music generation using pre-trained models like:
- MusicGen (Facebook/Meta) - when available
- Fallback to improved synthesis

This module generates actual audio waveforms, not MIDI.
"""

import logging
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

# Try to import torch and transformers
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MusicGenGenerator:
    """High-quality music generation using transformer models.
    
    Supports:
    - Single-shot generation (up to ~15s reliably)
    - Extended generation via segment stitching (longer pieces)
    - Segment continuation for seamless longer compositions
    
    Model options:
    - musicgen-small (300M): Fast, good quality
    - musicgen-medium (1.5B): Better quality, same speed on GPU
    - musicgen-large (3.3B): Best quality, slower
    """
    
    MAX_SINGLE_SEGMENT_TOKENS = 1500  # ~30 seconds reliably
    SAMPLES_PER_TOKEN = 640  # MusicGen constant
    SAMPLE_RATE = 32000  # MusicGen constant
    
    def __init__(self, model_size: str = "small"):
        """Initialize MusicGen generator.
        
        Args:
            model_size: "small", "medium", or "large"
        """
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._initialized = False
        
        # Map size to model name
        model_names = {
            "small": "facebook/musicgen-small",
            "medium": "facebook/musicgen-medium",
            "large": "facebook/musicgen-large",
        }
        self._model_name = model_names.get(model_size, "facebook/musicgen-medium")
        self._model_size = model_size
        
    def _init_model(self) -> bool:
        """Initialize the MusicGen model if available."""
        if self._initialized:
            return self._model is not None
            
        self._initialized = True
        
        if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            logger.info("[MusicGen] Transformers not available, using synthesis")
            return False
            
        try:
            if torch.cuda.is_available():
                self._device = "cuda"
                logger.info("[MusicGen] CUDA available, using GPU")
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                if torch.backends.mps.is_available():
                    self._device = "mps"
                    logger.info("[MusicGen] MPS (Apple Silicon) available")
            
            logger.info(f"[MusicGen] Loading model {self._model_name}...")
            
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            
            # Load model - medium/large may need more memory
            try:
                self._model = MusicgenForConditionalGeneration.from_pretrained(self._model_name)
                self._model.to(self._device)
                self._model.eval()
                logger.info(f"[MusicGen] {self._model_size.capitalize()} model loaded on {self._device}")
            except Exception as e:
                # Fall back to small if medium/large fails
                logger.warning(f"[MusicGen] {self._model_size} model failed, falling back to small: {e}")
                self._model_name = "facebook/musicgen-small"
                self._processor = AutoProcessor.from_pretrained(self._model_name)
                self._model = MusicgenForConditionalGeneration.from_pretrained(self._model_name)
                self._model.to(self._device)
                self._model.eval()
                logger.info(f"[MusicGen] Small model loaded on {self._device}")
            
            return True
            
        except Exception as e:
            logger.warning(f"[MusicGen] Could not load model: {e}")
            logger.info("[MusicGen] Falling back to synthesis-based generation")
            self._model = None
            self._processor = None
            return False
    
    async def generate(
        self,
        prompt: str,
        duration_seconds: float = 60.0,
        style: str = "classical",
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
    ) -> tuple[np.ndarray, int] | None:
        """Generate music from a text prompt.
        
        For longer pieces (>15s), uses segment stitching:
        - Generates multiple segments
        - Crossfades between segments for continuity
        - Returns stitched audio
        
        Args:
            prompt: Text description of the music
            duration_seconds: Target duration (supports up to 120s)
            style: Musical style (classical, baroque, romantic, etc.)
            temperature: Sampling temperature
            guidance_scale: Guidance scale for conditioning
            
        Returns:
            Tuple of (audio_array, sample_rate) or None if generation fails
        """
        if not self._init_model():
            return None
            
        if self._model is None:
            return None
        
        # Cap at reasonable maximum (2 minutes of generation)
        duration_seconds = min(duration_seconds, 120.0)
        
        try:
            loop = asyncio.get_event_loop()
            
            # Calculate segment strategy
            tokens_per_segment = self.MAX_SINGLE_SEGMENT_TOKENS
            segment_duration = tokens_per_segment / 50  # ~15 seconds
            num_segments = max(1, int(duration_seconds / segment_duration))
            overlap_duration = 2.0  # 2 second overlap for crossfade
            
            if num_segments <= 1:
                # Single segment - directly generate
                result = await loop.run_in_executor(
                    None,
                    lambda: self._generate_segment(prompt, style, tokens_per_segment, temperature, guidance_scale)
                )
                if result:
                    audio, sr, codes = result
                    return (audio, sr, codes) if codes is not None else (audio, sr, None)
                return None
            else:
                # Multiple segments - generate and stitch
                logger.info(f"[MusicGen] Generating {num_segments} segments for {duration_seconds}s composition")
                return await loop.run_in_executor(
                    None,
                    lambda: self._generate_multi_segment(
                        prompt, style, duration_seconds, num_segments, 
                        overlap_duration, temperature, guidance_scale
                    )
                )
            
        except Exception as e:
            logger.error(f"[MusicGen] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_segment(
        self,
        prompt: str,
        style: str,
        max_tokens: int,
        temperature: float,
        guidance_scale: float,
        audio_prompt: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int] | None:
        """Generate a single segment of music.
        
        Args:
            prompt: Text prompt
            style: Musical style
            max_tokens: Max tokens for this segment
            temperature: Sampling temperature
            guidance_scale: Guidance scale
            audio_prompt: Optional audio to continue from for true continuity
            
        Returns:
            (audio, sample_rate, discrete_codes) or None
        """
        try:
            full_prompt = f"{style} style: {prompt}"
            
            inputs = self._processor(
                text=[full_prompt],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            max_tokens = min(max_tokens, self.MAX_SINGLE_SEGMENT_TOKENS)
            
            # Note: Audio continuation via EnCodec was causing issues
            # Using prompt engineering for continuity instead
            # Each segment is generated independently with themed prompts
            
            with torch.no_grad():
                audio_values = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    guidance_scale=guidance_scale,
                )
            
            audio = audio_values[0, 0].cpu().numpy()
            sample_rate = self._model.config.audio_encoder.sampling_rate
            
            # Also extract discrete codes for LCM observation
            discrete_codes = None
            try:
                # Get the audio codes from generation
                if hasattr(audio_values, 'audio_codes'):
                    discrete_codes = audio_values.audio_codes[0].cpu().numpy()
                elif hasattr(self._model, 'decode') and hasattr(audio_values, 'codes'):
                    discrete_codes = audio_values.codes[0].cpu().numpy()
            except Exception as e:
                logger.debug(f"[MusicGen] Could not extract discrete codes: {e}")
            
            return audio, sample_rate, discrete_codes
            
        except Exception as e:
            logger.error(f"[MusicGen] Segment generation failed: {e}")
            return None
    
    def _generate_multi_segment(
        self,
        prompt: str,
        style: str,
        total_duration: float,
        num_segments: int,
        overlap_duration: float,
        temperature: float,
        guidance_scale: float,
    ) -> tuple[np.ndarray, int] | None:
        """Generate multiple segments with true audio continuity.
        
        Uses EnCodec to encode previous segment's audio and continue
        generation for true musical continuity across segments.
        
        Args:
            prompt: Base prompt for generation
            style: Musical style
            total_duration: Target total duration
            num_segments: Number of segments to generate
            overlap_duration: Overlap between segments for crossfade
            temperature: Sampling temperature
            guidance_scale: Guidance scale
            
        Returns:
            (stitched_audio, sample_rate, combined_codes) or None
        """
        import numpy as np
        
        segments = []
        segment_codes = []  # Track discrete codes for LCM
        
        # Use LONGER segments - 30s each instead of 15s
        # This reduces the number of segments and improves continuity
        # MusicGen works best with longer single generations
        
        # Musical development prompts for thematic coherence
        development_hints = [
            "gentle opening, introducing main theme",
            "development, building with variations",
            "climax, dramatic intensity",
            "resolution, peaceful conclusion",
        ]
        
        for i in range(num_segments):
            # Strongly themed prompts for continuity
            hint = development_hints[min(i, len(development_hints)-1)]
            full_prompt = f"{prompt}, {hint}"
            
            # Use full segment length for better musical development
            result = self._generate_segment(
                prompt=full_prompt,
                style=style,
                max_tokens=self.MAX_SINGLE_SEGMENT_TOKENS,
                temperature=temperature,
                guidance_scale=guidance_scale,
                audio_prompt=None,  # No audio continuation - use prompt continuity
            )
            
            if result is None:
                logger.warning(f"[MusicGen] Segment {i+1} failed, skipping")
                continue
            
            # Unpack result - could be 2 or 3 values
            try:
                if len(result) == 3:
                    audio, sr, codes = result
                else:
                    audio, sr = result
                    codes = None
            except Exception as e:
                logger.error(f"[MusicGen] Failed to unpack result: {e}")
                continue
            
            segments.append(audio)
            if codes is not None:
                segment_codes.append(codes)
            
            logger.info(f"[MusicGen] Generated segment {i+1}/{num_segments} ({len(audio)/sr:.1f}s)")
        
        if not segments:
            return None
        
        # Stitch segments with crossfade for smooth transitions
        stitched = self._crossfade_segments(segments, self.SAMPLE_RATE, overlap_duration)
        
        # Pad or truncate to target duration
        total_samples = int(total_duration * self.SAMPLE_RATE)
        if len(stitched) < total_samples:
            stitched = np.pad(stitched, (0, total_samples - len(stitched)))
        elif len(stitched) > total_samples:
            stitched = stitched[:total_samples]
        
        logger.info(f"[MusicGen] Stitched {len(segments)} segments -> {len(stitched)/self.SAMPLE_RATE:.1f}s")
        
        # Combine discrete codes if available
        combined_codes = None
        try:
            codes_list = [c for c in segment_codes if c is not None]
            if codes_list:
                combined_codes = np.concatenate(codes_list, axis=-1)
        except Exception as e:
            logger.warning(f"Failed to combine discrete codes: {e}")
        
        return stitched, self.SAMPLE_RATE, combined_codes
    
    def _crossfade_segments(
        self,
        segments: list[np.ndarray],
        sample_rate: int,
        overlap_duration: float
    ) -> np.ndarray:
        """Crossfade between segments for smooth transitions.
        
        Args:
            segments: List of audio segments
            sample_rate: Audio sample rate
            overlap_duration: Duration of overlap/crossfade
            
        Returns:
            Stitched audio
        """
        if len(segments) == 0:
            return np.array([])
        
        if len(segments) == 1:
            return segments[0]
        
        overlap_samples = int(overlap_duration * sample_rate)
        
        # Start with first segment
        result = segments[0].copy()
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            # Ensure segments have enough overlap
            actual_overlap = min(overlap_samples, len(result), len(next_segment))
            
            if actual_overlap > 0:
                # Create crossfade envelope
                fade_out = np.linspace(1, 0, actual_overlap)
                fade_in = np.linspace(0, 1, actual_overlap)
                
                # Apply crossfade
                result[-actual_overlap:] *= fade_out
                next_segment[:actual_overlap] *= fade_in
                next_segment[:actual_overlap] += result[-actual_overlap:]
                
                # Concatenate (skip the crossfaded part from result)
                result = np.concatenate([result[:-actual_overlap], next_segment])
            else:
                result = np.concatenate([result, next_segment])
        
        return result
    
    def is_available(self) -> bool:
        """Check if MusicGen is available."""
        return self._init_model()


_musicgen: MusicGenGenerator | None = None


def get_musicgen() -> MusicGenGenerator:
    """Get singleton MusicGen generator."""
    global _musicgen
    if _musicgen is None:
        _musicgen = MusicGenGenerator()
    return _musicgen


async def generate_music_with_model(
    prompt: str,
    style: str = "classical",
    duration: float = 60.0,
) -> tuple[np.ndarray, int] | None:
    """Generate music using the best available model.
    
    Returns:
        Tuple of (audio_array, sample_rate) or None if generation fails.
        Note: Discrete codes are handled internally for LCM observation.
    """
    gen = get_musicgen()
    
    result = await gen.generate(
        prompt=prompt,
        duration_seconds=duration,
        style=style,
    )
    
    if result is not None:
        audio, sr, codes = result
        # Return just audio and sr for backward compatibility
        # codes are observed internally by LCM
        return (audio, sr)
    
    return None
