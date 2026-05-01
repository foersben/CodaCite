"""Infrastructure implementation for Local VLM (Vision Language Model) via llama.cpp."""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler
except ImportError:
    Llama = None  # type: ignore
    Llava15ChatHandler = None  # type: ignore

from app.config import settings

logger = logging.getLogger(__name__)


class LocalVLM:
    """Local Vision Language Model using llama-cpp-python.

    Implementation Details:
        - Uses Llava or compatible vision models.
        - Optimized for CPU inference.
    """

    def __init__(self) -> None:
        """Initialize the local VLM."""
        self.model_path = settings.local_vlm_path
        self.clip_path = ""  # Often vision models have a separate clip model, or it's integrated
        # In many newer GGUFs (like Moondream), the vision part is integrated or uses a specific handler
        # For simplicity, we'll try to load it if the path exists.
        self.llm: Any = None
        if not self.model_path:
            logger.warning("[VLM] No local_vlm_path configured. VLM features will be disabled.")
            return

        if not Path(self.model_path).exists():
            logger.error("[VLM] Model path does not exist: %s", self.model_path)
            return

        try:
            # This is a stub for Llava-style models.
            # If the user uses Moondream or other, the initialization might differ.
            # We'll use a basic Llama init with a chat handler if possible.
            if Llama is None:
                logger.error("[VLM] llama-cpp-python is not installed correctly.")
                return

            # Check if there is a mmproj file for clip
            clip_path = list(Path(self.model_path).parent.glob("*mmproj*.gguf"))
            chat_handler = None
            if clip_path:
                logger.info("[VLM] Found clip model at %s", clip_path[0])
                chat_handler = Llava15ChatHandler(clip_model_path=str(clip_path[0]))

            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=chat_handler,
                n_ctx=2048,
                n_threads=6,
                verbose=False,
            )
            logger.info("[VLM] Local VLM initialized from %s", self.model_path)
        except Exception as e:
            logger.error("[VLM] Failed to load local VLM: %s", e)

    def describe_image(self, image: Image.Image) -> str:
        """Generate a text description for a given PIL Image.

        Args:
            image: The PIL Image object to describe.

        Returns:
            A detailed text description of the image content.
        """
        if not self.llm:
            return "[VLM Error: Model not initialized or configured]"

        try:
            # Convert PIL image to base64 for the chat handler
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{img_str}"

            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this technical drawing or image in detail. Focus on structural elements and text content if any.",
                            },
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ]
            )
            return str(response["choices"][0]["message"]["content"]).strip()
        except Exception as e:
            logger.error("[VLM] Generation failed: %s", e)
            return f"[VLM Error: {e}]"
