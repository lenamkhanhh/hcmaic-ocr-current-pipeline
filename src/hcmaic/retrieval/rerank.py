"""Bounded reranker protocol with a deterministic no-op default."""

from __future__ import annotations

from typing import Protocol

from hcmaic.retrieval.candidates import FusedCandidate


class Reranker(Protocol):
    def rerank(self, candidates: list[FusedCandidate], *, top_k: int) -> list[FusedCandidate]: ...


class PassthroughReranker:
    def rerank(self, candidates: list[FusedCandidate], *, top_k: int) -> list[FusedCandidate]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        return candidates[:top_k]
