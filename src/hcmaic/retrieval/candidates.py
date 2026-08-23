"""Canonical per-channel and fused retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChannelHit:
    entity_id: str
    video_id: str
    timestamp_ms: int
    modality: str
    score: float
    rank: int
    provider: str
    evidence_text: str | None = None
    frame_uid: str | None = None
    video_filename: str | None = None
    source_frame_idx: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedCandidate:
    entity_id: str
    video_id: str
    timestamp_ms: int
    final_score: float
    signal_scores: dict[str, float] = field(default_factory=dict)
    normalized_scores: dict[str, float] = field(default_factory=dict)
    evidence_texts: dict[str, str] = field(default_factory=dict)
    contributing_providers: list[str] = field(default_factory=list)
    explanation: dict[str, str | float] = field(default_factory=dict)
    frame_uid: str | None = None
    video_filename: str | None = None
    source_frame_idx: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None
