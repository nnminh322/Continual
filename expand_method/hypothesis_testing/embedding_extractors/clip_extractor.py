"""
Extract embeddings from frozen CLIP Vision Transformer.

GPU-accelerated: model on CUDA, returns torch tensors directly for routing.
Can also return numpy arrays for compatibility.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CLIPVisionExtractor:
    """
    Extract CLS embeddings from frozen CLIP ViT model.

    GPU-accelerated: runs on CUDA, can return torch tensors directly
    to avoid CPU<->GPU transfer during routing.

    Args:
        model_name: HuggingFace model name.
            Default: "openai/clip-vit-large-patch14-336" (matches LLaVA-1.5)
        device: 'cuda', 'cpu', or 'auto'.
        dtype: Model weight dtype. 'float16', 'bfloat16', or 'float32'.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14-336",
        device: str = "auto",
        dtype: str = "float16",
    ):
        if not HAS_TORCH:
            raise ImportError("torch and transformers are required. Install: pip install torch transformers")

        self.model_name = model_name

        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(dtype, torch.float16)

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model = self.model.to(self._device).eval()

        self.embedding_dim = getattr(
            self.model.config, "hidden_size",
            getattr(self.model.config, "vision_config", None) and
            getattr(self.model.config.vision_config, "hidden_size", None)
            or getattr(self.model.config, "projection_dim", 768)
        )

    @property
    def device(self) -> str:
        return self._device

    def extract(
        self,
        images: List[Image.Image],
        batch_size: int = 8,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract CLS embeddings from a list of PIL Images.

        Args:
            images: List of PIL Images.
            batch_size: Batch size for extraction.
            return_torch: If True, returns torch.Tensor on CUDA (recommended for routing).
                          If False, returns np.ndarray on CPU.

        Returns:
            (N, d) embeddings as numpy array or torch tensor.
        """
        all_embs = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i: i + batch_size]

            with torch.no_grad():
                inputs = self.processor(
                    images=batch_imgs,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                outputs = self.model(**inputs, output_hidden_states=True)
                cls_emb = outputs.hidden_states[-1][:, 0, :]  # (B, d)

            if return_torch:
                all_embs.append(cls_emb)
            else:
                all_embs.append(cls_emb.float().cpu().numpy())

        if return_torch:
            return torch.cat(all_embs, dim=0)
        return np.concatenate(all_embs, axis=0).astype(np.float32)

    def extract_from_paths(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract CLS embeddings from image file paths.

        Args:
            image_paths: List of image file paths.
            batch_size: Batch size.
            return_torch: If True, returns torch.Tensor on CUDA.

        Returns:
            (N, d) embeddings.
        """
        images = [Image.open(p).convert("RGB") for p in image_paths]
        return self.extract(images, batch_size=batch_size, return_torch=return_torch)

    def extract_single(
        self,
        image: Image.Image,
        return_torch: bool = False,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Extract CLS embedding for a single image.

        Args:
            image: PIL Image.
            return_torch: If True, returns (d,) torch.Tensor on CUDA.

        Returns:
            (d,) embedding.
        """
        result = self.extract([image], batch_size=1, return_torch=return_torch)
        return result[0]

    def get_embedding_dim(self) -> int:
        return self.embedding_dim

    def __repr__(self) -> str:
        gpu_tag = " [GPU]" if self._device == "cuda" else " [CPU]"
        return (
            f"CLIPVisionExtractor("
            f"model={self.model_name}, "
            f"dim={self.embedding_dim}, "
            f"device={self.device}{gpu_tag})"
        )
