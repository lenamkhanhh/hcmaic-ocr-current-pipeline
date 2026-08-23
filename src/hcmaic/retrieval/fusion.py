"""Deterministic score-fusion primitives used by retrieval stages."""

from __future__ import annotations

import math

from hcmaic.retrieval.candidates import ChannelHit, FusedCandidate


def _candidate_for(hit: ChannelHit) -> FusedCandidate:
    return FusedCandidate(
        entity_id=hit.entity_id,
        video_id=hit.video_id,
        timestamp_ms=hit.timestamp_ms,
        final_score=0.0,
        frame_uid=hit.frame_uid,
        video_filename=hit.video_filename,
        source_frame_idx=hit.source_frame_idx,
    )


def _record_hit(
    candidate: FusedCandidate, modality: str, hit: ChannelHit, normalized: float
) -> None:
    candidate.signal_scores[modality] = hit.score
    candidate.normalized_scores[modality] = normalized
    if hit.evidence_text:
        candidate.evidence_texts[modality] = hit.evidence_text
    if hit.provider not in candidate.contributing_providers:
        candidate.contributing_providers.append(hit.provider)
    if candidate.frame_uid is None and hit.frame_uid is not None:
        candidate.frame_uid = hit.frame_uid
    if candidate.video_filename is None and hit.video_filename is not None:
        candidate.video_filename = hit.video_filename
    if candidate.source_frame_idx is None and hit.source_frame_idx is not None:
        candidate.source_frame_idx = hit.source_frame_idx
    candidate.evidence[modality] = {
        "frame_uid": hit.frame_uid,
        "video_filename": hit.video_filename,
        "source_frame_idx": hit.source_frame_idx,
        "timestamp_ms": hit.timestamp_ms,
        "provider": hit.provider,
        "score": hit.score,
        "rank": hit.rank,
        "text": hit.evidence_text,
        "metadata": dict(hit.evidence),
    }


def _rank(candidates: dict[str, FusedCandidate], top_k: int) -> list[FusedCandidate]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    return sorted(candidates.values(), key=lambda item: (-item.final_score, item.entity_id))[:top_k]


def reciprocal_rank_fusion(
    channels: dict[str, list[ChannelHit]], *, rank_constant: int = 60, top_k: int = 100
) -> list[FusedCandidate]:
    if rank_constant < 1:
        raise ValueError("rank_constant must be >= 1")
    candidates: dict[str, FusedCandidate] = {}
    for modality, hits in channels.items():
        seen_frame_uids: set[str] = set()
        for hit in hits:
            if hit.rank < 1:
                raise ValueError("hit rank must be >= 1")
            identity = str(hit.frame_uid or hit.entity_id)
            if identity in seen_frame_uids:
                raise ValueError(f"duplicate frame_uid in channel {modality}: {identity}")
            seen_frame_uids.add(identity)
            if hit.frame_uid is not None and hit.frame_uid != hit.entity_id:
                raise ValueError(
                    f"channel {modality} entity_id/frame_uid mismatch: "
                    f"{hit.entity_id!r} != {hit.frame_uid!r}"
                )
            candidate = candidates.setdefault(hit.entity_id, _candidate_for(hit))
            contribution = 1.0 / (rank_constant + hit.rank)
            candidate.final_score += contribution
            _record_hit(candidate, modality, hit, contribution)
    for candidate in candidates.values():
        candidate.explanation = {
            "method": "rrf",
            "rank_constant": float(rank_constant),
        }
    return _rank(candidates, top_k)


def harmonic_mean_fusion(
    channels: dict[str, list[ChannelHit]],
    *,
    epsilon: float = 1e-8,
    top_k: int = 100,
) -> list[FusedCandidate]:
    """Fuse available channel scores after independent min-max scaling.

    A candidate contributes only the modalities that returned a score for its
    identity.  This is intentional: an unavailable or unqueried channel is
    not silently converted into a zero.  A constant channel is mapped to one,
    matching the stage-level contract that a single available signal remains
    usable while still being on the common [0, 1] scale.
    """

    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite value > 0")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    candidates: dict[str, FusedCandidate] = {}
    for modality, hits in channels.items():
        if not hits:
            continue
        seen_frame_uids: set[str] = set()
        values: list[float] = []
        for hit in hits:
            if hit.rank < 1:
                raise ValueError("hit rank must be >= 1")
            identity = str(hit.frame_uid or hit.entity_id)
            if identity in seen_frame_uids:
                raise ValueError(f"duplicate frame_uid in channel {modality}: {identity}")
            seen_frame_uids.add(identity)
            if hit.frame_uid is not None and hit.frame_uid != hit.entity_id:
                raise ValueError(
                    f"channel {modality} entity_id/frame_uid mismatch: "
                    f"{hit.entity_id!r} != {hit.frame_uid!r}"
                )
            score = float(hit.score)
            if not math.isfinite(score):
                raise ValueError(f"channel {modality} returned a non-finite score")
            values.append(score)

        low, high = min(values), max(values)
        for hit in hits:
            normalized = (
                1.0 if high == low else max(0.0, min(1.0, (float(hit.score) - low) / (high - low)))
            )
            candidate = candidates.setdefault(hit.entity_id, _candidate_for(hit))
            _record_hit(candidate, modality, hit, normalized)

    for candidate in candidates.values():
        scores = list(candidate.normalized_scores.values())
        if not scores:
            continue
        candidate.final_score = min(
            1.0,
            len(scores) / sum(1.0 / (score + epsilon) for score in scores),
        )
        candidate.explanation = {
            "method": "harmonic-mean-minmax",
            "epsilon": float(epsilon),
        }
    return _rank(candidates, top_k)


def weighted_late_fusion(
    channels: dict[str, list[ChannelHit]],
    *,
    weights: dict[str, float],
    top_k: int = 100,
) -> list[FusedCandidate]:
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("fusion weight must be >= 0")
    candidates: dict[str, FusedCandidate] = {}
    for modality, hits in channels.items():
        weight = weights.get(modality, 0.0)
        if not hits or weight == 0:
            continue
        values = [hit.score for hit in hits]
        low, high = min(values), max(values)
        for hit in hits:
            normalized = 1.0 if high == low else (hit.score - low) / (high - low)
            candidate = candidates.setdefault(hit.entity_id, _candidate_for(hit))
            candidate.final_score += weight * normalized
            _record_hit(candidate, modality, hit, normalized)
    for candidate in candidates.values():
        candidate.explanation = {"method": "weighted-late-fusion"}
    return _rank(candidates, top_k)
