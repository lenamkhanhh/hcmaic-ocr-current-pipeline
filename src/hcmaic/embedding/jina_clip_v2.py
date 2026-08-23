"""Local-cache-only Jina CLIP v2 image/text embedding provider.

The adapter follows the model's native ``encode_image`` and ``encode_text``
API.  It never falls back to another provider and never downloads weights when
``local_files_only=True``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize

DEFAULT_MODEL = "jinaai/jina-clip-v2"
CPU_BATCH = 8
CUDA_BATCH = 4


def _as_matrix(value: Any, *, label: str) -> np.ndarray:
    """Convert torch/numpy/list model output to a two-dimensional float matrix."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise RuntimeError(f"Jina {label} output must be 2-D, got {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"Jina {label} output contains non-finite values")
    return array


def _configured_dimension(config: Any) -> int:
    candidates = [
        config,
        getattr(config, "text_config", None),
        getattr(config, "vision_config", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        for field in ("projection_dim", "embedding_dim", "hidden_size"):
            value = getattr(candidate, field, None)
            if value is not None:
                dimension = int(value)
                if dimension > 0:
                    return dimension
    raise RuntimeError("Jina CLIP v2 model config has no embedding dimension")


class RealJinaClipV2EmbeddingProvider(EmbeddingProvider):
    """Concrete Jina CLIP v2 provider with explicit local/network provenance."""

    name = "jina-clip-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        *,
        local_files_only: bool = True,
        revision: str | None = None,
        batch_size: int | None = None,
        truncate_dim: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Jina CLIP v2 requires torch+transformers. Install with: uv sync --extra clip"
            ) from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        default_batch = CUDA_BATCH if self._device.startswith("cuda") else CPU_BATCH
        self._batch = batch_size or default_batch
        if self._batch < 1:
            raise ValueError("batch_size must be >= 1")
        self._model_name = model_name
        self._local_files_only = local_files_only
        self._trust_remote_code = True
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "trust_remote_code": self._trust_remote_code,
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        try:
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
            self._model = model.to(self._device).eval()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            mode = "local cache" if local_files_only else "configured model source"
            raise RuntimeError(
                f"Jina CLIP v2 model {model_name!r} is unavailable from {mode}; "
                "cache the model first or pass local_files_only=False explicitly."
            ) from exc

        configured_dimension = _configured_dimension(self._model.config)
        if truncate_dim is not None and not 1 <= truncate_dim <= configured_dimension:
            raise ValueError(
                f"truncate_dim must be in [1, {configured_dimension}], got {truncate_dim}"
            )
        self._truncate_dim = truncate_dim or configured_dimension
        self._configured_dimension = configured_dimension
        self._revision = getattr(self._model.config, "_commit_hash", None) or revision or "main"
        self.version = f"jina-clip-v2:{model_name}@{self._revision}"

    @property
    def dimension(self) -> int:
        return self._truncate_dim

    def info(self) -> dict[str, Any]:
        data = super().info()
        preprocessing = "JinaCLIPModel encode_image local-path; encode_text task=retrieval.query"
        data.update(
            {
                "model_name": self._model_name,
                "model_revision": self._revision,
                "revision": self._revision,
                "device": self._device,
                "batch_size": self._batch,
                "dtype": str(next(self._model.parameters()).dtype),
                "local_files_only": self._local_files_only,
                "trust_remote_code": self._trust_remote_code,
                "processor_revision": self._revision,
                "configured_dimension": self._configured_dimension,
                "truncate_dim": self._truncate_dim,
                "preprocessing": preprocessing,
                "preprocess_hash": hashlib.sha256(preprocessing.encode("utf-8")).hexdigest(),
                "evidence_level": "REAL_PROVIDER",
            }
        )
        return data

    def _normalize(self, value: Any, *, label: str) -> np.ndarray:
        array = _as_matrix(value, label=label)
        if array.shape[1] != self.dimension:
            raise RuntimeError(
                f"Jina {label} output dimension {array.shape[1]} != {self.dimension}"
            )
        return l2_normalize(array)

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(paths), self._batch):
            batch = [str(path) for path in paths[start : start + self._batch]]
            with self._torch.inference_mode():
                features = self._model.encode_image(batch, truncate_dim=self._truncate_dim)
            chunks.append(self._normalize(features, label="image"))
        if not chunks:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack(chunks).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            with self._torch.inference_mode():
                features = self._model.encode_text(
                    batch,
                    task="retrieval.query",
                    truncate_dim=self._truncate_dim,
                )
            chunks.append(self._normalize(features, label="text"))
        if not chunks:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack(chunks).astype(np.float32)
