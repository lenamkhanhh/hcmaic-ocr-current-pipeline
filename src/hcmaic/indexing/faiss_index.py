"""Optional FAISS index provider (requires `uv sync --extra faiss`).

Same contract and deterministic ordering as ExactNumpyIndex: candidates are
over-fetched from FAISS and re-sorted by (score desc, frame_id asc).
ExactNumpyIndex remains the verified fallback; FAISS must never block the
mission-critical path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hcmaic.indexing.base import SearchIndex


class FaissIndex(SearchIndex):
    name = "faiss-flat-ip"

    def __init__(self) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS not installed. Install with: uv sync --extra faiss "
                "(ExactNumpyIndex is the always-available fallback)."
            ) from exc
        self._faiss = faiss
        self._index: Any = None
        self._frame_ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def build(self, embeddings: np.ndarray, frame_ids: list[str]) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(frame_ids):
            raise ValueError(f"row/id mismatch: {embeddings.shape} vs {len(frame_ids)} ids")
        index = self._faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._index = index
        self._frame_ids = list(frame_ids)
        self._matrix = embeddings

    @property
    def size(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed_rows: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        if self._index is None or self._matrix is None:
            raise RuntimeError("Index not built")
        if top_k < 1:
            return []
        query = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if allowed_rows is not None:
            # Filtered search: exact scan over the allowed subset (correctness
            # over speed; identical to ExactNumpyIndex semantics).
            allowed_idx = np.flatnonzero(np.asarray(allowed_rows, dtype=bool))
            if allowed_idx.size == 0:
                return []
            scores = self._matrix[allowed_idx] @ query.reshape(-1)
            pairs = sorted(
                ((self._frame_ids[i], float(s)) for i, s in zip(allowed_idx, scores, strict=True)),
                key=lambda p: (-p[1], p[0]),
            )
            return pairs[:top_k]
        fetch = min(max(top_k * 2, top_k + 8), self.size)
        scores, indices = self._index.search(query, fetch)
        pairs = [
            (self._frame_ids[i], float(s))
            for i, s in zip(indices[0], scores[0], strict=True)
            if i != -1
        ]
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return pairs[:top_k]
