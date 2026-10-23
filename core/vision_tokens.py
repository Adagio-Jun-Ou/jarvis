"""Visual token encoder for image understanding.

Implements a vision tokenizer that converts images into discrete tokens,
similar to how semantic/acoustic tokens work for speech.

This enables:
- Unified multimodal token sequences
- Image understanding without direct pixel processing
- Vision-language model integration
"""
import asyncio
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image


@dataclass
class VisualTokenConfig:
    """Configuration for visual tokenizer."""
    image_size: int = 224          # Input image size (square)
    patch_size: int = 16           # ViT patch size
    vocab_size: int = 8192         # Visual vocabulary size
    num_patches: int = 196         # (224/16)^2 = 196 patches
    hidden_dim: int = 768          # Hidden dimension
    num_layers: int = 12           # Transformer layers
    num_heads: int = 12            # Attention heads


@dataclass
class VisualTokens:
    """Container for visual tokens."""
    tokens: np.ndarray             # Shape: [num_tokens]
    patch_embeddings: np.ndarray   # Shape: [num_patches, hidden_dim]
    spatial_grid: tuple            # (height_patches, width_patches)
    original_size: tuple           # (original_height, original_width)
    image_hash: str                # Hash for caching


class VisualTokenizer:
    """Tokenizes images into discrete visual tokens.

    Uses a Vision Transformer (ViT) style architecture to:
    1. Split image into patches
    2. Encode patches with transformer
    3. Quantize into discrete tokens

    This allows images to be processed as token sequences,
    compatible with language models.
    """

    def __init__(
        self,
        config: Optional[VisualTokenConfig] = None,
        model_path: Optional[str] = None,
    ):
        self.config = config or VisualTokenConfig()
        self.model_path = model_path
        self._model = None
        self._processor = None
        self._codebook = None
        self._loaded = False

    async def load(self) -> bool:
        """Load the visual tokenizer model."""
        try:
            # Try to load a pre-trained vision model
            import torch
            from transformers import ViTModel, ViTImageProcessor

            model_name = self.model_path or "google/vit-base-patch16-224"

            self._processor = ViTImageProcessor.from_pretrained(model_name)
            self._model = ViTModel.from_pretrained(model_name)

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

            self._model.eval()

            # Initialize codebook for quantization
            self._init_codebook()

            self._loaded = True
            return True

        except ImportError:
            print("Warning: transformers/torch not installed. Using fallback tokenizer.")
            return self._load_fallback()
        except Exception as e:
            print(f"Error loading visual tokenizer: {e}")
            return self._load_fallback()

    def _init_codebook(self) -> None:
        """Initialize the visual codebook for quantization."""
        import torch

        # Create a learnable codebook for visual tokens
        vocab_size = self.config.vocab_size
        hidden_dim = self.config.hidden_dim

        # Random initialization (in production, this would be learned)
        self._codebook = torch.randn(vocab_size, hidden_dim)
        self._codebook = self._codebook / self._codebook.norm(dim=-1, keepdim=True)

        if torch.cuda.is_available():
            self._codebook = self._codebook.cuda()

    def _load_fallback(self) -> bool:
        """Load fallback tokenizer using basic image features."""
        self._loaded = True
        return True

    def _preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Preprocess image for tokenization."""
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        original_size = image.size  # (width, height)

        # Resize to model input size
        image = image.resize(
            (self.config.image_size, self.config.image_size),
            Image.Resampling.BILINEAR
        )

        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std

        return img_array, original_size

    async def encode(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
    ) -> VisualTokens:
        """Encode an image into visual tokens.

        Args:
            image: Image as file path, PIL Image, or numpy array

        Returns:
            VisualTokens containing discrete tokens and embeddings
        """
        if not self._loaded:
            await self.load()

        # Preprocess
        img_array, original_size = self._preprocess_image(image)

        # Generate hash for caching
        import hashlib
        image_hash = hashlib.sha256(img_array.tobytes()).hexdigest()[:16]

        if self._model is not None:
            tokens, embeddings = await self._encode_with_model(img_array)
        else:
            tokens, embeddings = self._encode_fallback(img_array)

        spatial_grid = (
            self.config.image_size // self.config.patch_size,
            self.config.image_size // self.config.patch_size
        )

        return VisualTokens(
            tokens=tokens,
            patch_embeddings=embeddings,
            spatial_grid=spatial_grid,
            original_size=original_size,
            image_hash=image_hash,
        )

    async def _encode_with_model(
        self,
        img_array: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode using the ViT model."""
        import torch

        # Prepare input
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Get patch embeddings
        with torch.no_grad():
            outputs = self._model(img_tensor)
            # Get the patch embeddings (excluding CLS token)
            patch_embeddings = outputs.last_hidden_state[0, 1:]  # [num_patches, hidden_dim]

        # Quantize to discrete tokens
        # Find nearest codebook entry for each patch embedding
        patch_embeddings_np = patch_embeddings.cpu().numpy()

        # Normalize embeddings
        embeddings_norm = patch_embeddings_np / (
            np.linalg.norm(patch_embeddings_np, axis=-1, keepdims=True) + 1e-8
        )
        codebook_norm = self._codebook.cpu().numpy()

        # Find nearest neighbors
        similarities = embeddings_norm @ codebook_norm.T
        tokens = np.argmax(similarities, axis=-1)

        return tokens.astype(np.int32), patch_embeddings_np

    def _encode_fallback(
        self,
        img_array: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback encoding using basic features."""
        # Split into patches
        patch_size = self.config.patch_size
        patches = []

        for i in range(0, self.config.image_size, patch_size):
            for j in range(0, self.config.image_size, patch_size):
                patch = img_array[i:i+patch_size, j:j+patch_size]
                patches.append(patch.flatten())

        patches = np.array(patches)  # [num_patches, patch_size^2 * 3]

        # Simple quantization using color histograms
        tokens = []
        for patch in patches:
            # Compute average color and texture features
            avg_color = np.mean(patch.reshape(-1, 3), axis=0)
            variance = np.var(patch)

            # Hash to token
            feature = np.concatenate([avg_color, [variance]])
            feature_hash = hash(tuple(feature.round(4)))
            token = abs(feature_hash) % self.config.vocab_size
            tokens.append(token)

        tokens = np.array(tokens, dtype=np.int32)

        # Create dummy embeddings
        embeddings = np.random.randn(len(tokens), self.config.hidden_dim).astype(np.float32)

        return tokens, embeddings

    async def decode(
        self,
        tokens: VisualTokens,
    ) -> np.ndarray:
        """Decode visual tokens back to an image (reconstruction).

        Note: Reconstruction from tokens alone is lossy.
        For best results, use a dedicated image decoder model.

        Args:
            tokens: VisualTokens to decode

        Returns:
            Reconstructed image as numpy array
        """
        if not self._loaded:
            await self.load()

        if self._codebook is not None and self._model is not None:
            return await self._decode_with_codebook(tokens)
        else:
            return self._decode_fallback(tokens)

    async def _decode_with_codebook(
        self,
        tokens: VisualTokens,
    ) -> np.ndarray:
        """Decode using the codebook embeddings."""
        import torch

        # Get embeddings from codebook
        token_tensor = torch.from_numpy(tokens.tokens).long()
        if torch.cuda.is_available():
            token_tensor = token_tensor.cuda()
            codebook = self._codebook.cuda()
        else:
            codebook = self._codebook.cpu()

        embeddings = codebook[token_tensor]  # [num_patches, hidden_dim]

        # Use a simple upsampling to reconstruct
        # In production, use a proper image decoder
        patch_embeddings = embeddings.cpu().numpy()

        # Reshape to spatial grid
        h, w = tokens.spatial_grid
        patch_embeddings.shape[-1]

        # Simple reconstruction: average pool embeddings to get patch colors
        # This is a very rough approximation
        patches = np.random.rand(h * w, self.config.patch_size, self.config.patch_size, 3)

        for i, emb in enumerate(patch_embeddings):
            # Use embedding statistics to set patch color
            color = np.abs(emb[:3])  # Use first 3 dims as RGB
            color = color / (np.max(color) + 1e-8)
            patches[i] = patches[i] * 0.3 + color * 0.7

        # Reconstruct image
        image = np.zeros((
            self.config.image_size,
            self.config.image_size,
            3
        ))

        patch_idx = 0
        for i in range(h):
            for j in range(w):
                y = i * self.config.patch_size
                x = j * self.config.patch_size
                image[y:y+self.config.patch_size, x:x+self.config.patch_size] = patches[patch_idx]
                patch_idx += 1

        return (image * 255).astype(np.uint8)

    def _decode_fallback(
        self,
        tokens: VisualTokens,
    ) -> np.ndarray:
        """Fallback decoding - generates abstract image from tokens."""
        h, w = tokens.spatial_grid
        image = np.zeros((
            self.config.image_size,
            self.config.image_size,
            3
        ), dtype=np.uint8)

        for idx, token in enumerate(tokens.tokens):
            i = idx // w
            j = idx % w

            # Use token to generate patch color
            np.random.seed(token)
            color = np.random.randint(0, 256, 3)

            y = i * self.config.patch_size
            x = j * self.config.patch_size
            image[y:y+self.config.patch_size, x:x+self.config.patch_size] = color

        return image

    def get_token_sequence(
        self,
        tokens: VisualTokens,
        include_spatial: bool = True
    ) -> np.ndarray:
        """Get token sequence for language model input.

        Args:
            tokens: VisualTokens to convert
            include_spatial: Whether to include spatial position encoding

        Returns:
            Token sequence as numpy array
        """
        if include_spatial:
            # Add spatial position tokens
            h, w = tokens.spatial_grid
            position_tokens = []

            for idx, token in enumerate(tokens.tokens):
                row = idx // w
                col = idx % w
                # Encode position as offset
                position_token = self.config.vocab_size + row * w + col
                position_tokens.append(position_token)

            # Interleave visual and position tokens
            sequence = []
            for t, p in zip(tokens.tokens, position_tokens):
                sequence.extend([t, p])

            return np.array(sequence, dtype=np.int32)
        else:
            return tokens.tokens

    def get_attention_mask(
        self,
        tokens: VisualTokens,
    ) -> np.ndarray:
        """Get attention mask for visual tokens.

        Returns:
            Binary mask (all 1s for valid tokens)
        """
        return np.ones(len(tokens.tokens), dtype=np.int32)


class WebcamCapture:
    """Real-time webcam capture for Jarvis.

    Captures frames from webcam and can process them
    as visual tokens or send to vision model.
    """

    def __init__(
        self,
        device: int = 0,
        fps: int = 30,
        resolution: tuple = (640, 480),
    ):
        self.device = device
        self.fps = fps
        self.resolution = resolution
        self._capture = None
        self._running = False
        self._frame_callback = None

    async def start(self) -> bool:
        """Start webcam capture (non-blocking; cv2 device open runs in a thread)."""
        try:
            import cv2
        except ImportError:
            print("OpenCV not installed. Run: pip install opencv-python")
            return False

        def _open():
            capture = cv2.VideoCapture(self.device)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            return capture

        try:
            self._capture = await asyncio.to_thread(_open)

            if not self._capture.isOpened():
                await self._release_capture()
                print(f"Failed to open webcam device {self.device}")
                return False

            self._running = True
            return True
        except Exception as e:
            print(f"Error starting webcam: {e}")
            self._capture = None
            return False

    async def _release_capture(self) -> None:
        """Release the OS camera handle from a worker thread (cv2 is blocking)."""
        capture, self._capture = self._capture, None
        self._running = False
        if capture is not None:
            try:
                await asyncio.to_thread(capture.release)
            except Exception as e:
                print(f"Error releasing webcam: {e}")

    async def stop(self) -> None:
        """Stop webcam capture and release the camera (turns off the LED)."""
        await self._release_capture()

    async def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the webcam.

        Returns:
            Frame as numpy array (RGB format) or None if failed
        """
        if not self._capture or not self._running:
            return None

        import cv2

        def _grab():
            ret, frame = self._capture.read()
            if not ret:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            return await asyncio.to_thread(_grab)
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    async def capture_and_encode(
        self,
        tokenizer: VisualTokenizer,
    ) -> Optional[VisualTokens]:
        """Capture a frame and encode it to visual tokens.

        Args:
            tokenizer: VisualTokenizer to use

        Returns:
            VisualTokens or None if capture failed
        """
        frame = await self.capture_frame()
        if frame is None:
            return None

        return await tokenizer.encode(frame)

    async def stream_frames(
        self,
        callback,
        interval: float = 0.1,
    ):
        """Stream frames to a callback function.

        Args:
            callback: Async function to call with each frame
            interval: Seconds between frames
        """
        while self._running:
            frame = await self.capture_frame()
            if frame is not None:
                await callback(frame)
            await asyncio.sleep(interval)

    async def stream_tokens(
        self,
        tokenizer: VisualTokenizer,
        callback,
        interval: float = 0.5,
    ):
        """Stream visual tokens to a callback function.

        Args:
            tokenizer: VisualTokenizer to use
            callback: Async function to call with tokens
            interval: Seconds between tokenizations
        """
        while self._running:
            tokens = await self.capture_and_encode(tokenizer)
            if tokens is not None:
                await callback(tokens)
            await asyncio.sleep(interval)

    def list_devices(self) -> list[dict]:
        """List available webcam devices (blocking; callers should to_thread it)."""
        devices = []

        try:
            import cv2

            for i in range(10):  # Check first 10 devices
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    devices.append({
                        "id": i,
                        "name": f"Camera {i}",
                        "resolution": (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        ),
                        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
                    })
                    cap.release()

        except ImportError:
            pass

        return devices


class WebcamTool:
    """Tool for Jarvis to use the webcam.

    Permission policy (privacy-sensitive device):
    - "list" is free (no capture, just device enumeration)
    - "capture"/"stream" always require permission — seeing the user is
      never an implicit action. needs_permission() encodes this so both
      agent paths (execute_tool contract) gate identically.
    """

    name = "webcam"
    description = "Capture images from the webcam for analysis"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["capture", "list", "stream"],
                "description": "What to do with the webcam",
            },
            "description": {
                "type": "string",
                "description": "Optional description of what to look for",
            },
        },
        "required": ["action"],
    }
    requires_permission = True  # Conservative default for unknown actions

    def __init__(
        self,
        webcam: WebcamCapture,
        tokenizer: Optional[VisualTokenizer] = None,
        vision_processor=None,
    ):
        self.webcam = webcam
        self.tokenizer = tokenizer
        self.vision = vision_processor

    def needs_permission(self, action: str = "", **kwargs) -> bool:
        """Only device listing is free; anything that opens the camera needs a grant."""
        return action != "list"

    async def execute(
        self,
        action: str,
        description: Optional[str] = None,
    ) -> str:
        """Execute webcam action."""
        if action == "list":
            devices = await asyncio.to_thread(self.webcam.list_devices)
            if not devices:
                return "No webcam devices found."
            return "Available webcams:\n" + "\n".join(
                f"  - {d['name']}: {d['resolution'][0]}x{d['resolution'][1]} @ {d['fps']}fps"
                for d in devices
            )

        elif action == "capture":
            # Start webcam if not running
            if not self.webcam._running:
                success = await self.webcam.start()
                if not success:
                    return "Failed to start webcam."

            try:
                # Capture frame
                frame = await self.webcam.capture_frame()
                if frame is None:
                    return "Failed to capture frame from webcam."

                # Analyze if vision processor available
                if self.vision:
                    # Convert frame to base64
                    import base64
                    from io import BytesIO
                    from PIL import Image

                    pil_image = Image.fromarray(frame)
                    buffer = BytesIO()
                    pil_image.save(buffer, format="JPEG", quality=85)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()

                    prompt = description or "Describe what you see in this webcam capture."
                    analysis = await self.vision.analyze_image(img_base64, prompt=prompt)
                    return f"Webcam capture analysis:\n{analysis}"
                else:
                    return f"Captured frame: {frame.shape[1]}x{frame.shape[0]} pixels. (No vision processor available for analysis)"
            finally:
                # Privacy: always release the camera after a one-shot capture
                # so the device LED goes off. Streaming (when implemented)
                # would hold the camera for its duration instead.
                await self.webcam.stop()

        elif action == "stream":
            return "Streaming mode not yet implemented. Use 'capture' for single frames."

        else:
            return f"Unknown action: {action}"
