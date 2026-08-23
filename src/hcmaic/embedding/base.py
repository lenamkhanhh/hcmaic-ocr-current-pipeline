"""EmbeddingProvider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingProvider(ABC):
    """Encodes images and texts into one shared, L2-normalized space."""

    #: short registry key, e.g. "mock" / "clip"
    name: str = "base"
    #: version string recorded in every index manifest
    version: str = "unversioned"

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def embed_images(self, paths: list[Path]) -> np.ndarray:
        """Return float32 [len(paths), dimension], rows L2-normalized."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return float32 [len(texts), dimension], rows L2-normalized."""

    def embed_query_image(self, path: Path) -> np.ndarray:
        """Encode one query image with the same image tower as catalog images."""
        result = self.embed_images([Path(path)])
        if result.shape != (1, self.dimension):
            raise ValueError(
                f"query image provider returned {result.shape}; expected (1, {self.dimension})"
            )
        return result

    def info(self) -> dict[str, Any]:
        """Provider description recorded in index_manifest.json."""
        return {
            "provider": self.name,
            "version": self.version,
            "dimension": self.dimension,
            "normalization": "l2",
        }


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize; zero rows become a deterministic unit basis vector."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = norms[:, 0] == 0.0
    if zero_rows.any():
        matrix = matrix.copy()
        matrix[zero_rows, 0] = 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / norms).astype(np.float32)


def get_provider(name: str, **kwargs: Any) -> EmbeddingProvider:
    """Instantiate a provider by registry key, with lazy optional adapters."""
    if name == "mock":
        from hcmaic.embedding.mock import DeterministicMockEmbeddingProvider

        return DeterministicMockEmbeddingProvider(**kwargs)
    if name == "clip":
        from hcmaic.embedding.clip_real import RealClipEmbeddingProvider

        return RealClipEmbeddingProvider(**kwargs)
    from hcmaic.embedding.registry import create_provider

    return create_provider(name, **kwargs)
