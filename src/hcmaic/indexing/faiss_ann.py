"""Optional FAISS HNSW inner-product index for scalable retrieval experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from hcmaic.indexing.base import SearchIndex


class FaissHNSWIndex(SearchIndex):
    name = "faiss-hnsw-ip"

    def __init__(self, *, m: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS HNSW needs faiss-cpu. Install with: uv sync --extra faiss."
            ) from exc
        if min(m, ef_construction, ef_search) < 1:
            raise ValueError("HNSW m/ef parameters must be >= 1")
        self._faiss = faiss
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index: Any = None
        self._frame_ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def build(self, embeddings: np.ndarray, frame_ids: list[str]) -> None:
        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(frame_ids):
            raise ValueError(f"row/id mismatch: {matrix.shape} vs {len(frame_ids)} ids")
        index = self._faiss.IndexHNSWFlat(matrix.shape[1], self.m, self._faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(matrix)
        self._index = index
        self._frame_ids = list(frame_ids)
        self._matrix = matrix

    @property
    def size(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    @property
    def parameters(self) -> dict[str, int]:
        return {
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        }

    def serialized_size_bytes(self) -> int:
        if self._index is None:
            raise RuntimeError("Index not built")
        return int(len(self._faiss.serialize_index(self._index)))

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
        vector = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
        if vector.shape[1] != self._matrix.shape[1]:
            raise ValueError("query dimension does not match index dimension")
        if allowed_rows is not None:
            allowed = np.flatnonzero(np.asarray(allowed_rows, dtype=bool))
            pairs = sorted(
                (
                    (self._frame_ids[index], float(self._matrix[index] @ vector[0]))
                    for index in allowed
                ),
                key=lambda item: (-item[1], item[0]),
            )
            return pairs[:top_k]
        scores, indices = self._index.search(vector, min(top_k, self.size))
        pairs = [
            (self._frame_ids[index], float(score))
            for index, score in zip(indices[0], scores[0], strict=True)
            if index >= 0
        ]
        return sorted(pairs, key=lambda item: (-item[1], item[0]))[:top_k]
