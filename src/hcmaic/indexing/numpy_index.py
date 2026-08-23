"""ExactNumpyIndex: mandatory, always-available exact search."""

from __future__ import annotations

import numpy as np

from hcmaic.indexing.base import SearchIndex


class ExactNumpyIndex(SearchIndex):
    name = "exact-numpy"

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._frame_ids: list[str] = []

    def build(self, embeddings: np.ndarray, frame_ids: list[str]) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(frame_ids):
            raise ValueError(
                f"row/id mismatch: {embeddings.shape[0]} embedding rows vs "
                f"{len(frame_ids)} frame ids"
            )
        self._matrix = embeddings
        self._frame_ids = list(frame_ids)

    @property
    def size(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed_rows: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        if self._matrix is None:
            raise RuntimeError("Index not built")
        if top_k < 1:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"query dimension {query.shape[0]} != index dimension {self._matrix.shape[1]}"
            )
        scores = self._matrix @ query
        if allowed_rows is not None:
            allowed_rows = np.asarray(allowed_rows, dtype=bool)
            if allowed_rows.shape[0] != scores.shape[0]:
                raise ValueError("allowed_rows mask length != index size")
            candidate_idx = np.flatnonzero(allowed_rows)
        else:
            candidate_idx = np.arange(scores.shape[0])
        if candidate_idx.size == 0:
            return []
        # Deterministic ordering: score desc, then frame_id asc.
        pairs = sorted(
            ((self._frame_ids[i], float(scores[i])) for i in candidate_idx),
            key=lambda p: (-p[1], p[0]),
        )
        return pairs[:top_k]
