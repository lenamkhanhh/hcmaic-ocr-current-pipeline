"""Hybrid KIS orchestration: channels, RRF/weighted fusion, dedup and rerank."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from hcmaic.contracts.kis import Evidence, KISQuery, KISResult
from hcmaic.retrieval.asr import ASRArtifactError, ASRUnavailableError
from hcmaic.retrieval.candidates import ChannelHit, FusedCandidate
from hcmaic.retrieval.fusion import reciprocal_rank_fusion, weighted_late_fusion
from hcmaic.retrieval.object_retrieval import ObjectArtifactError, ObjectUnavailableError
from hcmaic.retrieval.ocr_bm25 import OCRArtifactError, OCRUnavailableError
from hcmaic.retrieval.ocr_elasticsearch import ElasticsearchOCRError
from hcmaic.skillpixel.retrieval import SkillPixelRetriever


class TextRetrievalChannel(Protocol):
    provider: str
    revision: str

    def search(self, text: str, top_k: int = 100) -> list[ChannelHit]: ...


@dataclass(frozen=True)
class KISHybridOutput:
    query: KISQuery
    results: list[KISResult]
    executed_channels: tuple[str, ...]
    unavailable_channels: dict[str, str]
    candidate_count: int


def _channel_failure(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _hit_from_visual(result: KISResult, provider: str) -> ChannelHit:
    return ChannelHit(
        entity_id=result.frame_uid,
        video_id=result.video_id,
        timestamp_ms=result.timestamp_ms,
        modality="visual",
        score=float(result.channel_scores.get("visual", result.fused_score)),
        rank=result.rank,
        provider=provider,
        evidence_text=None,
        frame_uid=result.frame_uid,
        video_filename=result.video_filename,
        source_frame_idx=result.source_frame_idx,
        evidence={"evidence_level": result.evidence_level},
    )


def _source_key(candidate: FusedCandidate) -> tuple[str, int | str]:
    if candidate.source_frame_idx is not None:
        return (candidate.video_id, candidate.source_frame_idx)
    return (candidate.video_id, candidate.entity_id)


def deduplicate_source_frames(candidates: list[FusedCandidate]) -> list[FusedCandidate]:
    """Merge candidates that point to the same official source frame."""
    deduplicated: dict[tuple[str, int | str], FusedCandidate] = {}
    for candidate in candidates:
        key = _source_key(candidate)
        current = deduplicated.get(key)
        if current is None:
            deduplicated[key] = candidate
            continue
        current.final_score += candidate.final_score
        current.signal_scores.update(candidate.signal_scores)
        current.normalized_scores.update(candidate.normalized_scores)
        current.evidence_texts.update(candidate.evidence_texts)
        current.evidence.update(candidate.evidence)
        for provider in candidate.contributing_providers:
            if provider not in current.contributing_providers:
                current.contributing_providers.append(provider)
    return sorted(
        deduplicated.values(),
        key=lambda item: (
            -item.final_score,
            item.video_id,
            str(item.source_frame_idx),
            item.entity_id,
        ),
    )


def diversify_source_frames(
    candidates: list[FusedCandidate], *, top_k: int, max_per_video: int | None
) -> list[FusedCandidate]:
    """Apply a bounded per-video quota, then backfill deterministically."""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if max_per_video is None:
        return candidates[:top_k]
    if max_per_video < 1:
        raise ValueError("max_per_video must be >= 1")
    selected: list[FusedCandidate] = []
    counts: dict[str, int] = {}
    deferred: list[FusedCandidate] = []
    for candidate in candidates:
        count = counts.get(candidate.video_id, 0)
        if count < max_per_video:
            selected.append(candidate)
            counts[candidate.video_id] = count + 1
        else:
            deferred.append(candidate)
        if len(selected) == top_k:
            return selected
    for candidate in deferred:
        if len(selected) == top_k:
            break
        selected.append(candidate)
    return selected


def bounded_rerank(
    candidates: list[FusedCandidate], *, top_k: int, candidate_limit: int, timeout_ms: int
) -> list[FusedCandidate]:
    """Deterministic bounded rerank based only on measured channel evidence."""
    if candidate_limit < top_k or timeout_ms < 1:
        raise ValueError("candidate_limit must cover top_k and timeout_ms must be >= 1")
    started = time.perf_counter()
    bounded = candidates[:candidate_limit]
    for candidate in bounded:
        if (time.perf_counter() - started) * 1000.0 >= timeout_ms:
            break
        agreement = max(0, len(candidate.signal_scores) - 1)
        visual_support = candidate.normalized_scores.get("visual", 0.0)
        candidate.rerank_score = candidate.final_score + 0.01 * agreement + 0.001 * visual_support
        candidate.explanation.update(
            {
                "reranker": "bounded-v1",
                "rerank_agreement_bonus": float(agreement),
                "rerank_candidate_limit": float(candidate_limit),
            }
        )
    for candidate in bounded:
        if candidate.rerank_score is None:
            candidate.rerank_score = candidate.final_score
    ranked = sorted(
        bounded,
        key=lambda item: (
            -(item.rerank_score or item.final_score),
            -item.final_score,
            item.video_id,
            str(item.source_frame_idx),
            item.entity_id,
        ),
    )
    return ranked[:top_k]


class KISHybridOrchestrator:
    """Run visual plus optional raw-derived channels for canonical KIS results."""

    _OPTIONAL_CHANNELS = ("ocr", "object", "asr")

    def __init__(
        self,
        visual_retriever: SkillPixelRetriever,
        *,
        optional_channels: dict[str, TextRetrievalChannel | None] | None = None,
        fusion_method: str = "rrf",
        fusion_weights: dict[str, float] | None = None,
        rank_constant: int = 60,
        candidate_multiplier: int = 5,
        max_per_video: int | None = 5,
        reranker: str = "bounded-v1",
        rerank_timeout_ms: int = 50,
        asr_enabled: bool = False,
        channel_status: Mapping[str, Any] | None = None,
    ) -> None:
        if fusion_method not in {"rrf", "weighted"}:
            raise ValueError("fusion_method must be 'rrf' or 'weighted'")
        if rank_constant < 1 or candidate_multiplier < 1:
            raise ValueError("rank_constant/candidate_multiplier must be >= 1")
        if reranker not in {"bounded-v1", "none"}:
            raise ValueError("reranker must be 'bounded-v1' or 'none'")
        if rerank_timeout_ms < 1:
            raise ValueError("rerank_timeout_ms must be >= 1")
        self.visual_retriever = visual_retriever
        self.optional_channels = optional_channels or {}
        unknown = set(self.optional_channels) - set(self._OPTIONAL_CHANNELS)
        if unknown:
            raise ValueError(f"unsupported KIS optional channels: {sorted(unknown)}")
        self.fusion_method = fusion_method
        self.fusion_weights = fusion_weights or {}
        self.rank_constant = rank_constant
        self.candidate_multiplier = candidate_multiplier
        self.max_per_video = max_per_video
        self.reranker = reranker
        self.rerank_timeout_ms = rerank_timeout_ms
        self.asr_enabled = asr_enabled
        self.channel_status = dict(channel_status or {})

    def _status_reason(self, name: str, default: str) -> str:
        raw = self.channel_status.get(name)
        if isinstance(raw, Mapping):
            reason = raw.get("reason")
            if reason:
                return str(reason)
        if raw is not None:
            value = str(raw)
            if value.startswith("unavailable:"):
                return value.split(":", 1)[1].strip() or default
            if value.startswith("disabled"):
                return "disabled_until_qrels_ablation_gain"
            if value and value != "ready":
                return value
        return default

    def _optional_hits(
        self, query: KISQuery, channel_top_k: int
    ) -> tuple[dict[str, list[ChannelHit]], dict[str, str], list[str]]:
        hits: dict[str, list[ChannelHit]] = {}
        unavailable: dict[str, str] = {}
        executed: list[str] = []
        for name in self._OPTIONAL_CHANNELS:
            channel = self.optional_channels.get(name)
            if query.task == "VKIS":
                unavailable[name] = "not_applicable_to_vkis_visual_query"
                continue
            if name == "asr" and not self.asr_enabled:
                unavailable[name] = self._status_reason(name, "disabled_until_qrels_ablation_gain")
                continue
            if channel is None:
                unavailable[name] = self._status_reason(
                    name, "not_configured_or_artifact_unavailable"
                )
                continue
            try:
                channel_hits = channel.search(query.text or "", channel_top_k)
            except (
                OCRArtifactError,
                OCRUnavailableError,
                ElasticsearchOCRError,
                ObjectArtifactError,
                ObjectUnavailableError,
                ASRArtifactError,
                ASRUnavailableError,
            ) as exc:
                unavailable[name] = _channel_failure(exc)
                continue
            hits[name] = channel_hits
            executed.append(name)
        return hits, unavailable, executed

    def _fuse(self, channels: dict[str, list[ChannelHit]], *, top_k: int) -> list[FusedCandidate]:
        pool_size = top_k * self.candidate_multiplier
        if self.fusion_method == "rrf":
            fused = reciprocal_rank_fusion(
                channels,
                rank_constant=self.rank_constant,
                top_k=pool_size,
            )
        else:
            fused = weighted_late_fusion(
                channels,
                weights=self.fusion_weights,
                top_k=pool_size,
            )
        deduplicated = deduplicate_source_frames(fused)
        if self.reranker == "bounded-v1":
            return bounded_rerank(
                deduplicated,
                top_k=top_k,
                candidate_limit=max(top_k, pool_size),
                timeout_ms=self.rerank_timeout_ms,
            )
        return deduplicated[:top_k]

    def _to_results(
        self,
        query: KISQuery,
        candidates: list[FusedCandidate],
        *,
        executed_channels: list[str],
        unavailable_channels: dict[str, str],
    ) -> list[KISResult]:
        results: list[KISResult] = []
        for rank, candidate in enumerate(
            diversify_source_frames(
                candidates,
                top_k=query.top_k,
                max_per_video=self.max_per_video,
            ),
            start=1,
        ):
            if (
                candidate.frame_uid is None
                or candidate.video_filename is None
                or candidate.source_frame_idx is None
            ):
                raise ValueError("fused candidate has incomplete source-frame mapping")
            evidence: list[Evidence] = []
            for channel_name, payload in sorted(candidate.evidence.items()):
                if not isinstance(payload, dict):
                    continue
                evidence.append(
                    Evidence(
                        channel=channel_name,
                        frame_uid=str(payload.get("frame_uid") or candidate.frame_uid),
                        video_id=candidate.video_id,
                        video_filename=str(
                            payload.get("video_filename") or candidate.video_filename
                        ),
                        source_frame_idx=int(
                            payload.get("source_frame_idx", candidate.source_frame_idx)
                        ),
                        timestamp_ms=int(payload.get("timestamp_ms", candidate.timestamp_ms)),
                        score=float(payload.get("score", 0.0)),
                        rank=int(payload.get("rank", rank)),
                        evidence_level="REAL_PROVIDER",
                        text=(str(payload["text"]) if payload.get("text") else None),
                        metadata={
                            "provider": payload.get("provider"),
                            **dict(payload.get("metadata") or {}),
                        },
                    )
                )
            results.append(
                KISResult(
                    query_id=query.query_id,
                    task=query.task,
                    rank=rank,
                    frame_uid=candidate.frame_uid,
                    video_id=candidate.video_id,
                    video_filename=candidate.video_filename,
                    source_frame_idx=candidate.source_frame_idx,
                    timestamp_ms=candidate.timestamp_ms,
                    channel_scores=dict(candidate.signal_scores),
                    fused_score=candidate.final_score,
                    rerank_score=candidate.rerank_score,
                    evidence=tuple(evidence),
                    executed_channels=tuple(executed_channels),
                    unavailable_channels=dict(unavailable_channels),
                    evidence_level="REAL_PROVIDER",
                    quality_status="UNVALIDATED_ON_HCMAIC",
                )
            )
        return results

    def search(self, query: KISQuery) -> KISHybridOutput:
        """Run one canonical query through visual and configured optional channels."""
        visual_results = self.visual_retriever.search_kis(query)
        channel_top_k = query.top_k * self.candidate_multiplier
        channels = {
            "visual": [
                _hit_from_visual(result, self.visual_retriever.provider.name)
                for result in visual_results
            ]
        }
        optional_hits, unavailable, optional_executed = self._optional_hits(query, channel_top_k)
        channels.update(optional_hits)
        executed = ["visual", *optional_executed]
        fused = self._fuse(channels, top_k=query.top_k)
        results = self._to_results(
            query,
            fused,
            executed_channels=executed,
            unavailable_channels=unavailable,
        )
        return KISHybridOutput(query, results, tuple(executed), unavailable, len(fused))

    def search_queries(self, queries: list[KISQuery]) -> dict[str, KISHybridOutput]:
        """Preserve mixed-query order while isolating optional-channel failures."""
        query_ids = [query.query_id for query in queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("KIS query_id values must be unique")
        outputs = [self.search(query) for query in queries]
        return {output.query.query_id: output for output in outputs}
