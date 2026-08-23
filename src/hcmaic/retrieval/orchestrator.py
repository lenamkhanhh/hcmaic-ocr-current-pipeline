"""Multi-channel retrieval orchestration with isolated optional failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hcmaic.retrieval.candidates import ChannelHit, FusedCandidate
from hcmaic.retrieval.fusion import reciprocal_rank_fusion, weighted_late_fusion
from hcmaic.retrieval.rerank import PassthroughReranker, Reranker


class ChannelUnavailableError(RuntimeError):
    """A channel is disabled/unavailable; other channels may continue."""


class RetrievalChannel(Protocol):
    def search(self, text: str, top_k: int) -> list[ChannelHit]: ...


@dataclass(frozen=True)
class OrchestrationOutput:
    candidates: list[FusedCandidate]
    unavailable_channels: dict[str, str]


class RetrievalOrchestrator:
    def __init__(
        self,
        channels: dict[str, RetrievalChannel],
        *,
        fusion_method: str = "rrf",
        weights: dict[str, float] | None = None,
        reranker: Reranker | None = None,
        candidate_multiplier: int = 5,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        self.channels = channels
        self.fusion_method = fusion_method
        self.weights = weights or {}
        self.reranker = reranker or PassthroughReranker()
        self.candidate_multiplier = candidate_multiplier

    def search(self, text: str, *, top_k: int) -> OrchestrationOutput:
        channel_hits: dict[str, list[ChannelHit]] = {}
        unavailable: dict[str, str] = {}
        channel_top_k = top_k * self.candidate_multiplier
        for modality, channel in self.channels.items():
            try:
                channel_hits[modality] = channel.search(text, channel_top_k)
            except ChannelUnavailableError as exc:
                unavailable[modality] = str(exc)
        if self.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(channel_hits, top_k=channel_top_k)
        elif self.fusion_method == "weighted":
            fused = weighted_late_fusion(
                channel_hits,
                weights=self.weights,
                top_k=channel_top_k,
            )
        else:
            raise ValueError(
                f"Unknown fusion_method {self.fusion_method!r}; expected rrf or weighted."
            )
        return OrchestrationOutput(
            candidates=self.reranker.rerank(fused, top_k=top_k),
            unavailable_channels=unavailable,
        )
