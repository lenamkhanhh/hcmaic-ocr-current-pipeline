"""Deterministic mock embedding provider (mandatory test provider).

Design (DECISIONS.md D3): images are quantized onto a fixed 8-color palette
(8x8 downsample, nearest palette color per cell); the palette histogram is
projected through fixed seeded unit vectors. Text tokens matching the palette
vocabulary project through the same vectors; all tokens also add a tiny
hashed component for deterministic tie-breaking. Result: a text query naming
a color genuinely retrieves keyframes of that color — real cross-modal signal
with zero network, weights, or nondeterminism.

The mock provider validates plumbing only; it proves nothing about
competition retrieval quality.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image

from hcmaic.embedding.base import EmbeddingProvider, l2_normalize

_SEED = 20260726
_TOKEN_WEIGHT = 0.05

# name -> RGB; synonyms map into the same anchor.
PALETTE: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (40, 70, 220),
    "yellow": (230, 220, 50),
    "cyan": (60, 200, 210),
    "magenta": (200, 60, 200),
    "white": (240, 240, 240),
    "black": (15, 15, 15),
}
SYNONYMS: dict[str, str] = {
    "crimson": "red",
    "scarlet": "red",
    "navy": "blue",
    "azure": "blue",
    "lime": "green",
    "gold": "yellow",
    "purple": "magenta",
    "pink": "magenta",
    "teal": "cyan",
    "dark": "black",
    "bright": "white",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicMockEmbeddingProvider(EmbeddingProvider):
    name = "mock"
    version = "mock-palette-v1"

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension
        rng = np.random.default_rng(_SEED)
        self._anchors: dict[str, np.ndarray] = {}
        for color in PALETTE:  # dict order is fixed -> deterministic draws
            vec = rng.normal(size=dimension)
            self._anchors[color] = (vec / np.linalg.norm(vec)).astype(np.float32)
        self._palette_matrix = np.array(list(PALETTE.values()), dtype=np.float32)
        self._palette_names = list(PALETTE.keys())

    @property
    def dimension(self) -> int:
        return self._dimension

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        vec = rng.normal(size=self._dimension)
        return (vec / np.linalg.norm(vec)).astype(np.float32)

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        rows = np.zeros((len(paths), self._dimension), dtype=np.float32)
        for i, path in enumerate(paths):
            with Image.open(path) as im:
                small = np.asarray(
                    im.convert("RGB").resize((8, 8), Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ).reshape(-1, 3)
            # nearest palette color per cell
            dists = np.linalg.norm(small[:, None, :] - self._palette_matrix[None, :, :], axis=2)
            nearest = dists.argmin(axis=1)
            counts = np.bincount(nearest, minlength=len(self._palette_names))
            for color_idx, count in enumerate(counts):
                if count:
                    rows[i] += float(count) * self._anchors[self._palette_names[color_idx]]
        return l2_normalize(rows)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                color = SYNONYMS.get(token, token)
                if color in self._anchors:
                    rows[i] += self._anchors[color]
                rows[i] += _TOKEN_WEIGHT * self._token_vector(token)
        return l2_normalize(rows)
