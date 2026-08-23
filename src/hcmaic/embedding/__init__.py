"""Embedding providers (deterministic mock mandatory, real CLIP optional)."""

from hcmaic.embedding.base import EmbeddingProvider, get_provider
from hcmaic.embedding.mock import DeterministicMockEmbeddingProvider

__all__ = [
    "DeterministicMockEmbeddingProvider",
    "EmbeddingProvider",
    "get_provider",
]
