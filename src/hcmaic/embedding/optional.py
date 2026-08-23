"""Lazy, interface-only adapters for optional multilingual image/text models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from hcmaic.embedding.base import EmbeddingProvider


class DeferredOptionalEmbeddingProvider(EmbeddingProvider):
    """A safe placeholder that never downloads a model during construction."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_revision: str,
        device: str = "cpu",
        dimension: int | None = None,
    ) -> None:
        self.name = provider_name
        self.version = f"{provider_name}:{model_revision}"
        self.model_revision = model_revision
        self.device = device
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError(
                f"{self.name} adapter is interface-only and has no discovered "
                "dimension. Supply an approved concrete adapter/model revision "
                "before building an index."
            )
        return self._dimension

    def _unavailable(self) -> NoReturn:
        raise RuntimeError(
            f"{self.name} ({self.model_revision}) is interface-only. "
            "Install/configure the concrete provider and run its local smoke "
            "test before enabling it."
        )

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        del paths
        self._unavailable()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        del texts
        self._unavailable()

    def info(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "version": self.version,
            "model_revision": self.model_revision,
            "device": self.device,
            "dimension": self._dimension,
            "normalization": "l2",
            "evidence_level": "INTERFACE_ONLY",
        }
