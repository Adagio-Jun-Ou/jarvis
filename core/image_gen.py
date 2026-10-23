"""Image generation module for Jarvis.

Supports multiple backends:
- Stable Diffusion (local via diffusers)
- DALL-E (OpenAI API)
- ComfyUI (local)
- Automatic1111 (local web UI)

Provides a unified interface for generating images from text prompts.
"""
import asyncio
import base64
import logging
import os
import time
from core.helpers import safe_error_message
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ImageBackend(Enum):
    """Available image generation backends."""
    OLLAMA = "ollama"
    STABLE_DIFFUSION = "stable_diffusion"
    DALLE = "dalle"
    COMFYUI = "comfyui"
    AUTOMATIC1111 = "automatic1111"


class ImageGenerationError(Exception):
    """Raised when image generation fails and no fallback is available."""
    pass


@dataclass
class GeneratedImage:
    """Represents a generated image."""
    image_data: bytes
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    seed: int = -1
    backend: str = "unknown"
    generation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    file_path: Optional[Path] = None

    def save(self, path: Path | str) -> Path:
        """Save the image to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.image_data)
        self.file_path = path
        return path

    def to_base64(self) -> str:
        """Get base64 encoded image."""
        return base64.b64encode(self.image_data).decode("utf-8")

    def to_data_url(self, format: str = "png") -> str:
        """Get data URL for embedding in HTML."""
        return f"data:image/{format};base64,{self.to_base64()}"


class OllamaBackend:
    """Ollama image generation backend (x/z-image-turbo, x/flux2-klein)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "x/z-image-turbo"):
        self.base_url = base_url
        self.model = model
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def __aenter__(self) -> "OllamaBackend":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def is_available(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                return any(self.model in name or "flux" in name or "z-image" in name for name in model_names)
            return False
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        seed: int = -1,
        **kwargs,
    ) -> GeneratedImage:
        client = await self._get_client()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if seed >= 0:
            payload["seed"] = seed

        start_time = time.time()

        response = await client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )

        result = response.json()

        if "error" in result:
            raise RuntimeError(f"Ollama image generation error: {result['error']}")

        generation_time = time.time() - start_time

        if "images" in result and result["images"]:
            image_data = base64.b64decode(result["images"][0])
        else:
            raise RuntimeError("No image in Ollama response")

        return GeneratedImage(
            image_data=image_data,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            backend="ollama",
            generation_time=generation_time,
        )


class StableDiffusionBackend:
    """Stable Diffusion backend using diffusers library with ROCm support."""

    SD_MODELS = {
        "sd15": "runwayml/stable-diffusion-v1-5",
        "sd21": "stabilityai/stable-diffusion-2-1",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sdxl-turbo": "stabilityai/sdxl-turbo",
        "sd-turbo": "stabilityai/sd-turbo",
    }

    def __init__(
        self,
        model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "auto",
        torch_dtype: str = "float16",
        use_rocm: bool = True,
    ):
        if model in self.SD_MODELS:
            model = self.SD_MODELS[model]
        self.model = model
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_rocm = use_rocm
        self._pipeline = None
        self._initialized = False
        self._device_name = None

    def _detect_device(self) -> str:
        """Detect best available device (ROCm/CUDA/CPU)."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self._device_name = device_name
                if "AMD" in device_name or "Radeon" in device_name or "gfx" in device_name.lower():
                    logger.info(f"AMD GPU detected: {device_name}")
                    return "cuda"
                logger.info(f"NVIDIA GPU detected: {device_name}")
                return "cuda"
        except Exception as e:
            logger.debug(f"GPU detection error: {e}")
        return "cpu"

    def _init_pipeline(self):
        """Lazy initialization of the pipeline."""
        if self._initialized:
            return self._pipeline is not None

        try:
            import torch
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

            device = self.device
            if device == "auto":
                device = self._detect_device()

            dtype = torch.float16 if self.torch_dtype == "float16" else torch.float32

            if device == "cpu":
                dtype = torch.float32
                logger.info("Using CPU mode with float32")

            logger.info(f"Loading Stable Diffusion model: {self.model}")

            pipeline_kwargs = {
                "torch_dtype": dtype,
                "use_safetensors": True,
                "variant": "fp16" if dtype == torch.float16 else None,
            }
            pipeline_kwargs = {k: v for k, v in pipeline_kwargs.items() if v is not None}

            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.model,
                **pipeline_kwargs,
            )

            if "sdxl" not in self.model.lower():
                self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self._pipeline.scheduler.config
                )

            self._pipeline = self._pipeline.to(device)

            if hasattr(self._pipeline, "enable_attention_slicing"):
                self._pipeline.enable_attention_slicing()

            if device == "cuda" and hasattr(self._pipeline, "enable_vae_slicing"):
                self._pipeline.enable_vae_slicing()

            try:
                if device == "cuda" and hasattr(self._pipeline, "enable_xformers_memory_efficient_attention"):
                    self._pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(f"Failed to enable xformers memory efficient attention: {e}")

            logger.info(f"Stable Diffusion initialized on {device} ({self._device_name or 'unknown'})")
            self._initialized = True
            return True

        except ImportError as e:
            logger.warning(f"Stable Diffusion not available: {e}")
            logger.info("Install with: pip install diffusers transformers accelerate torch")
            self._initialized = True
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Stable Diffusion: {e}")
            self._initialized = True
            return False

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        seed: int = -1,
        **kwargs,
    ) -> GeneratedImage:
        """Generate an image from a prompt."""
        if not self._init_pipeline():
            raise RuntimeError("Stable Diffusion pipeline not available")

        import torch

        generator = None
        if seed >= 0:
            generator = torch.Generator(device=self._pipeline.device).manual_seed(seed)

        start_time = time.time()

        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=generator,
            )
        )

        generation_time = time.time() - start_time

        image = result.images[0]

        # Convert PIL Image to bytes
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        return GeneratedImage(
            image_data=image_bytes,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed if seed >= 0 else result.num_images,
            backend="stable_diffusion",
            generation_time=generation_time,
        )


class Automatic1111Backend:
    """Automatic1111/Stable Diffusion WebUI API backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url
        self._client = None

    async def _get_client(self):
        """Get HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def is_available(self) -> bool:
        """Check if Automatic1111 is running."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/sdapi/v1/sd-models")
            return response.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        seed: int = -1,
        sampler: str = "Euler a",
        cfg_scale: float = 7.0,
        **kwargs,
    ) -> GeneratedImage:
        """Generate an image using Automatic1111 API."""
        client = await self._get_client()

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
            "sampler_name": sampler,
            "cfg_scale": cfg_scale,
        }

        start_time = time.time()

        response = await client.post(
            f"{self.base_url}/sdapi/v1/txt2img",
            json=payload,
        )
        response.raise_for_status()

        result = response.json()
        generation_time = time.time() - start_time

        # Decode base64 image
        image_data = base64.b64decode(result["images"][0])

        return GeneratedImage(
            image_data=image_data,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            backend="automatic1111",
            generation_time=generation_time,
        )


class DalleBackend:
    """OpenAI DALL-E backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "dall-e-3"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

    async def generate(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs,
    ) -> GeneratedImage:
        """Generate an image using DALL-E API."""
        if not self.api_key:
            raise ValueError("OpenAI API key required for DALL-E")

        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            start_time = time.time()

            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "size": size,
                    "quality": quality,
                    "n": 1,
                    "response_format": "b64_json",
                },
            )
            response.raise_for_status()

            result = response.json()
            generation_time = time.time() - start_time

            image_data = base64.b64decode(result["data"][0]["b64_json"])

            # Parse size
            width, height = map(int, size.split("x"))

            return GeneratedImage(
                image_data=image_data,
                prompt=prompt,
                width=width,
                height=height,
                backend="dalle",
                generation_time=generation_time,
            )


class ComfyUIBackend:
    """ComfyUI backend for image generation."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188"):
        self.base_url = base_url
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def is_available(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/system_stats")
            return response.status_code == 200
        except Exception:
            return False

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        seed: int = -1,
        **kwargs,
    ) -> GeneratedImage:
        client = await self._get_client()
        start_time = time.time()

        workflow = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed if seed >= 0 else int(time.time() * 1000) % (2**32),
        }

        response = await client.post(
            f"{self.base_url}/prompt",
            json=workflow,
        )
        response.raise_for_status()
        
        result = response.json()
        generation_time = time.time() - start_time
        
        image_data = base64.b64decode(result.get("images", [""])[0]) if result.get("images") else b""
        
        return GeneratedImage(
            image_data=image_data,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            backend="comfyui",
            generation_time=generation_time,
        )


class ImageGenerator:
    """Main image generation interface with multiple backend support.
    No simulated backend - all backends require actual services.
    """

    def __init__(
        self,
        backend: str = "auto",
        model: str = "runwayml/stable-diffusion-v1-5",
        auto_url: str = "http://127.0.0.1:7860",
        output_dir: Optional[str] = None,
    ):
        self.backend_name = backend
        self.model = model
        self.auto_url = auto_url
        self.output_dir = Path(output_dir) if output_dir else Path.home() / ".jarvis" / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._backends: dict[str, Any] = {}
        self._active_backend: Optional[str] = None

    async def _get_backend(self, preferred: Optional[str] = None) -> tuple[str, Any]:
        backends_to_try: list[str] = []
        if self.backend_name == "auto":
            backends_to_try.extend(["ollama", "automatic1111", "stable_diffusion"])
        else:
            backends_to_try.append(self.backend_name)

        for backend_name in backends_to_try:
            if backend_name in self._backends:
                return backend_name, self._backends[backend_name]

            try:
                if backend_name == "ollama":
                    backend = OllamaBackend()
                    if await backend.is_available():
                        self._backends[backend_name] = backend
                        return backend_name, backend

                elif backend_name == "stable_diffusion":
                    backend = StableDiffusionBackend(model=self.model)
                    self._backends[backend_name] = backend
                    return backend_name, backend

                elif backend_name == "automatic1111":
                    backend = Automatic1111Backend(base_url=self.auto_url)
                    if await backend.is_available():
                        self._backends[backend_name] = backend
                        return backend_name, backend

                elif backend_name == "dalle":
                    backend = DalleBackend()
                    self._backends[backend_name] = backend
                    return backend_name, backend

                elif backend_name == "comfyui":
                    backend = ComfyUIBackend()
                    self._backends[backend_name] = backend
                    return backend_name, backend

            except Exception as e:
                logger.debug(f"Backend {backend_name} not available: {e}")
                continue

        raise RuntimeError(
            "No image generation backends available. "
            "Configure at least one backend in config.yaml under 'image_generation':\n\n"
            "Example configuration:\n"
            "image_generation:\n"
            "  backend: ollama | stable_diffusion | automatic1111 | comfyui | dalle\n"
            "  ollama:\n"
            "    model: 'x/z-image-turbo'\n"
            "  stable_diffusion:\n"
            "    model: 'sd3.5' or 'stable-diffusion-webui'\n"
            "  automatic1111:\n"
            "    base_url: 'http://127.0.0.1:7860'\n\n"
            "Simulated image generation is disabled - configure a real backend."
        )

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        seed: int = -1,
        save: bool = True,
        **kwargs,
    ) -> GeneratedImage:
        backend_name, backend = await self._get_backend()

        logger.info(f"Generating image with {backend_name}: {prompt[:50]}...")

        try:
            image = await backend.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                seed=seed,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Image generation failed: {safe_error_message(e)}")
            raise RuntimeError(
                f"Image generation failed with backend '{backend_name}': {safe_error_message(e)}. "
                f"Check that the backend is properly configured. "
                f"No simulated fallback - configure a working backend in config.yaml."
            )

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt[:30])
            filename = f"{timestamp}_{safe_prompt}.png"
            image.save(self.output_dir / filename)
            logger.info(f"Saved image to {self.output_dir / filename}")

        return image

    async def generate_avatar(
        self,
        description: str,
        style: str = "professional",
        size: int = 256,
    ) -> GeneratedImage:
        """Generate an avatar/profile picture.
        
        Args:
            description: Description of the person/character
            style: Style of the avatar (professional, cartoon, artistic)
            size: Square size in pixels
        """
        style_prompts = {
            "professional": "professional headshot, business attire, clean background, studio lighting",
            "cartoon": "cartoon avatar, friendly, colorful, simple background",
            "artistic": "artistic portrait, creative, expressive, stylized",
            "minimal": "minimal avatar, simple, clean, modern",
        }

        full_prompt = f"{description}, {style_prompts.get(style, style_prompts['professional'])}, high quality, detailed"

        return await self.generate(
            prompt=full_prompt,
            negative_prompt="blurry, low quality, distorted, ugly",
            width=size,
            height=size,
        )

    async def generate_illustration(
        self,
        concept: str,
        style: str = "digital art",
    ) -> GeneratedImage:
        """Generate an illustration for a concept.
        
        Args:
            concept: The concept to illustrate
            style: Art style (digital art, watercolor, sketch, etc.)
        """
        prompt = f"{concept}, {style}, detailed, high quality, trending on artstation"

        return await self.generate(
            prompt=prompt,
            negative_prompt="blurry, low quality, amateur, ugly",
            width=768,
            height=768,
            steps=25,
        )

    def list_saved_images(self) -> list[Path]:
        """List all saved generated images."""
        return list(self.output_dir.glob("*.png"))

    async def close(self) -> None:
        """Close all backend clients."""
        for name, backend in self._backends.items():
            if hasattr(backend, 'close'):
                try:
                    await backend.close()
                except Exception as e:
                    logger.debug(f"Error closing {name} backend: {e}")
        self._backends.clear()


# Singleton instance
_image_generator: Optional[ImageGenerator] = None


def get_image_generator(**kwargs) -> ImageGenerator:
    """Get or create the singleton ImageGenerator."""
    global _image_generator
    if _image_generator is None:
        _image_generator = ImageGenerator(**kwargs)
    return _image_generator


# Tool for Jarvis
class ImageGenerationTool:
    """Tool for Jarvis to generate images."""

    name = "generate_image"
    description = "Generate images from text descriptions. Use this to create visual content, illustrations, or concept art. Specify the prompt describing what you want to see."
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Detailed description of the image to generate",
            },
            "negative_prompt": {
                "type": "string",
                "description": "What to avoid in the image (optional)",
            },
            "style": {
                "type": "string",
                "description": "Style of the image (realistic, artistic, cartoon, professional)",
                "enum": ["realistic", "artistic", "cartoon", "professional", "minimal"],
            },
            "size": {
                "type": "string",
                "description": "Image size (512x512, 768x768, 1024x1024)",
                "enum": ["512x512", "768x768", "1024x1024"],
            },
        },
        "required": ["prompt"],
    }
    requires_permission = False

    def __init__(self, generator: Optional[ImageGenerator] = None):
        self.generator = generator

    async def execute(
        self,
        prompt: str,
        negative_prompt: str = "",
        style: str = "realistic",
        size: str = "512x512",
    ) -> str:
        """Execute image generation."""
        if self.generator is None:
            self.generator = get_image_generator()

        # Enhance prompt based on style
        style_enhancers = {
            "realistic": "photorealistic, highly detailed, 8k resolution",
            "artistic": "artistic, creative, stylized, masterpiece",
            "cartoon": "cartoon, animated, colorful, friendly",
            "professional": "professional, clean, corporate, high quality",
            "minimal": "minimal, simple, clean, modern",
        }

        enhanced_prompt = f"{prompt}, {style_enhancers.get(style, '')}"

        # Parse size
        width, height = map(int, size.split("x"))

        try:
            image = await self.generator.generate(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt or "blurry, low quality, distorted",
                width=width,
                height=height,
            )

            if image.file_path:
                return f"Generated image saved to: {image.file_path}\nGenerated in {image.generation_time:.1f}s using {image.backend}"
            else:
                return f"Image generated successfully ({width}x{height}) in {image.generation_time:.1f}s using {image.backend}"

        except Exception as e:
            return f"Failed to generate image: {safe_error_message(e)}"
