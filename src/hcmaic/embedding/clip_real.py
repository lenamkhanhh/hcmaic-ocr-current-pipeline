"""Real CLIP provider (optional; requires `uv sync --extra clip`).

Lightweight ViT-B/32-class model via transformers. CPU path is mandatory;
CUDA is used when available with a batch size safe for 4 GB VRAM.
Never imported by tests; tests must not download weights.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
CUDA_BATCH = 8  # safe for 4 GB VRAM with ViT-B/32
CPU_BATCH = 16


class RealClipEmbeddingProvider(EmbeddingProvider):
    name = "clip"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        *,
        local_files_only: bool = True,
        revision: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Real CLIP provider needs torch+transformers. Install with: "
                "uv sync --extra clip  (tests never require this)."
            ) from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        default_batch = CUDA_BATCH if self._device.startswith("cuda") else CPU_BATCH
        self._batch = batch_size or default_batch
        if self._batch < 1:
            raise ValueError("batch_size must be >= 1")
        self._model_name = model_name
        self._local_files_only = local_files_only
        load_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            load_kwargs["revision"] = revision
        try:
            self._model = (
                CLIPModel.from_pretrained(model_name, **load_kwargs).to(self._device).eval()
            )
            self._processor = CLIPProcessor.from_pretrained(model_name, **load_kwargs)
        except OSError as exc:
            mode = "local cache" if local_files_only else "configured model source"
            raise RuntimeError(
                f"Real CLIP model {model_name!r} is unavailable from {mode}; "
                "cache the model first or pass local_files_only=False explicitly."
            ) from exc
        self._revision = getattr(self._model.config, "_commit_hash", None) or revision or "main"
        self.version = f"clip:{model_name}@{self._revision}"
        self._dimension = int(self._model.config.projection_dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    def info(self) -> dict[str, Any]:
        data = super().info()
        preprocessing = "CLIPProcessor RGB resize/crop 224px"
        data.update(
            {
                "model_name": self._model_name,
                "model_revision": self._revision,
                "revision": self._revision,
                "device": self._device,
                "batch_size": self._batch,
                "dtype": str(next(self._model.parameters()).dtype),
                "local_files_only": self._local_files_only,
                "processor_revision": self._revision,
                "preprocessing": preprocessing,
                "preprocess_hash": hashlib.sha256(preprocessing.encode("utf-8")).hexdigest(),
                "evidence_level": "REAL_PROVIDER",
            }
        )
        return data

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self._batch):
            batch_paths = paths[start : start + self._batch]
            images = []
            for p in batch_paths:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            inputs = self._processor(images=images, return_tensors="pt").to(self._device)
            with self._torch.no_grad():
                feats = self._model.get_image_features(**inputs)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        if not chunks:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return l2_normalize(np.vstack(chunks))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            inputs = self._processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self._device)
            with self._torch.no_grad():
                feats = self._model.get_text_features(**inputs)
            chunks.append(feats.cpu().numpy().astype(np.float32))
        if not chunks:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return l2_normalize(np.vstack(chunks))
