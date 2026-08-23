"""SearchIndex interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SearchIndex(ABC):
    """Inner-product top-K search over L2-normalized vectors.

    Row i of the built matrix corresponds to frame_ids[i]; the id mapping is
    owned by the caller (id_map.json) and verified by tests — a wrong row
    must never silently return the wrong frame.
    """

    name: str = "base"

    @abstractmethod
    def build(self, embeddings: np.ndarray, frame_ids: list[str]) -> None: ...

    @property
    @abstractmethod
    def size(self) -> int: ...

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed_rows: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to top_k (frame_id, score), deterministically ordered.

        Ordering: score descending, then frame_id ascending (tie-break).
        `allowed_rows`: optional boolean mask over rows (video filtering).
        """
