"""Infrastructure implementation for local Vision Language Models (VLM) via llama.cpp."""

from __future__ import annotations

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

from app.core.config import settings

logger = logging.getLogger(__name__)


class LocalVLM:
    """Local Vision Language Model (VLM) for multimodal technical grounding.

    This class provides a local interface for executing vision-language tasks
    (e.g., image captioning, technical diagram analysis) using quantized GGUF
    models. It utilizes the llama-cpp-python library to manage the inference
    lifecycle on CPU, leveraging specialized chat handlers for multimodal
    tokenization.

    Architecture:
        - Engine: llama-cpp-python with CLIP/Llava chat handlers.
        - Precision: Typically 4-bit or 8-bit quantized GGUF.
        - Optimization: Multi-threaded CPU execution with local context windows.
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

    def describe_image(self, image: Any) -> str:
        """Generate a technical description for an image.

        Processes a raw image through the VLM to extract semantic meaning,
        structural layouts, or textual information from diagrams. The image
        is encoded to a base64 Data URI before being passed to the model's
        multimodal projector (CLIP).

        Args:
            image: A PIL Image object or compatible pixel array.

        Returns:
            A string containing the model's textual analysis. If inference
            fails or the model is not loaded, returns an error message.
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
