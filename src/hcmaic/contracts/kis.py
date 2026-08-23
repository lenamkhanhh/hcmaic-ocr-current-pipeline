"""Canonical contracts shared by every HCMAIC KIS channel.

The contracts deliberately keep both internal and official identities.  A
``faiss_row`` or ``keyframe_id`` is never enough to create a submission; the
only exported frame number is ``source_frame_idx``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskType = Literal["TKIS", "VKIS"]
QualityStatus = Literal["UNVALIDATED_ON_HCMAIC", "VALIDATED_ON_HCMAIC"]


@dataclass(frozen=True)
class Evidence:
    """One channel's explainable evidence for a canonical frame."""

    channel: str
    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int
    score: float
    rank: int
    evidence_level: str = "VALIDATED_LOCAL"
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channel.strip():
            raise ValueError("evidence channel must not be blank")
        if not self.frame_uid.strip() or not self.video_id.strip():
            raise ValueError("evidence frame_uid and video_id must not be blank")
        if not self.video_filename.strip():
            raise ValueError("evidence video_filename must not be blank")
        if self.source_frame_idx < 0 or self.timestamp_ms < 0:
            raise ValueError("evidence frame/timestamp must be non-negative")
        if self.rank < 1:
            raise ValueError("evidence rank must be >= 1")
        if not math.isfinite(self.score):
            raise ValueError("evidence score must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "frame_uid": self.frame_uid,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "source_frame_idx": self.source_frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "score": self.score,
            "rank": self.rank,
            "evidence_level": self.evidence_level,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class KISQuery:
    """One TKIS/VKIS request while preserving raw user text."""

    query_id: str
    task: TaskType | str
    text: str | None = None
    image_path: Path | None = None
    top_k: int = 100
    raw_text: str | None = None

    def __post_init__(self) -> None:
        original_text = self.text
        query_id = self.query_id.strip()
        task = self.task.upper()
        if not query_id:
            raise ValueError("query_id must not be blank")
        if task not in {"TKIS", "VKIS"}:
            raise ValueError(f"unsupported KIS task {task!r}")
        if self.top_k < 1 or self.top_k > 500:
            raise ValueError("top_k must be in [1, 500]")
        if task == "TKIS" and not (self.text or "").strip():
            raise ValueError("TKIS text must not be blank")
        if task == "VKIS" and self.image_path is None:
            raise ValueError("VKIS image_path is required")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "task", task)
        if self.text is not None:
            object.__setattr__(self, "text", self.text.strip())
        if self.raw_text is None:
            object.__setattr__(self, "raw_text", original_text)

    @property
    def is_text(self) -> bool:
        return self.task == "TKIS"


@dataclass(frozen=True)
class KISResult:
    """Canonical result/evidence envelope returned by all KIS paths."""

    query_id: str
    task: TaskType | str
    rank: int
    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int
    channel_scores: Mapping[str, float] = field(default_factory=dict)
    fused_score: float = 0.0
    rerank_score: float | None = None
    evidence: tuple[Evidence, ...] = ()
    executed_channels: tuple[str, ...] = ()
    unavailable_channels: Mapping[str, str] = field(default_factory=dict)
    evidence_level: str = "VALIDATED_LOCAL"
    quality_status: QualityStatus = "UNVALIDATED_ON_HCMAIC"

    def __post_init__(self) -> None:
        task = self.task.upper()
        if task not in {"TKIS", "VKIS"}:
            raise ValueError(f"unsupported KIS task {task!r}")
        if self.rank < 1:
            raise ValueError("result rank must be >= 1")
        if not self.query_id.strip() or not self.frame_uid.strip():
            raise ValueError("result query_id and frame_uid must not be blank")
        if not self.video_id.strip() or not self.video_filename.strip():
            raise ValueError("result video identity must not be blank")
        if self.source_frame_idx < 0 or self.timestamp_ms < 0:
            raise ValueError("result frame/timestamp must be non-negative")
        if not math.isfinite(self.fused_score):
            raise ValueError("result fused_score must be finite")
        if self.rerank_score is not None and not math.isfinite(self.rerank_score):
            raise ValueError("result rerank_score must be finite")
        object.__setattr__(self, "task", task)

    @property
    def answer_cell(self) -> str:
        return f"{self.video_filename},{self.source_frame_idx}"

    def sort_key(self) -> tuple[float, str, int, str]:
        return (-self.fused_score, self.video_id, self.source_frame_idx, self.frame_uid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "task": self.task,
            "rank": self.rank,
            "frame_uid": self.frame_uid,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "source_frame_idx": self.source_frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "channel_scores": dict(self.channel_scores),
            "fused_score": self.fused_score,
            "rerank_score": self.rerank_score,
            "evidence": [item.to_dict() for item in self.evidence],
            "executed_channels": list(self.executed_channels),
            "unavailable_channels": dict(self.unavailable_channels),
            "evidence_level": self.evidence_level,
            "quality_status": self.quality_status,
        }


@dataclass(frozen=True)
class KISChannelConfig:
    """Typed configuration for one optional retrieval channel."""

    name: str
    provider: str = "unavailable"
    enabled: bool = True
    artifact_path: Path | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("channel name must not be blank")
        if not self.provider.strip():
            raise ValueError(f"channel {self.name!r} provider must not be blank")
        if self.provider.lower() == "mock":
            raise ValueError(f"channel {self.name!r} cannot use mock provider")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "enabled": self.enabled,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "params": dict(self.params),
        }


@dataclass(frozen=True)
class KISPipelineConfig:
    """Production-safe, reproducible configuration for the full KIS path."""

    dataset_root: Path
    visual_provider: str = "siglip2"
    index_provider: str = "faiss-flat-ip"
    sampling_policy: str = "uniform_stride_10_v1"
    channels: tuple[KISChannelConfig, ...] = (
        KISChannelConfig(name="visual", provider="siglip2"),
        KISChannelConfig(name="ocr", provider="paddleocr", enabled=False),
        KISChannelConfig(name="object", provider="ultralytics", enabled=False),
        KISChannelConfig(name="asr", provider="whisper", enabled=False),
    )
    fusion_method: str = "rrf"
    fusion_rank_constant: int = 60
    fusion_weights: Mapping[str, float] = field(default_factory=dict)
    reranker: str = "bounded-v1"
    output_path: Path | None = None
    benchmark_root: Path | None = None
    seed: int = 0
    quality_status: QualityStatus = "UNVALIDATED_ON_HCMAIC"

    def __post_init__(self) -> None:
        if not str(self.dataset_root).strip():
            raise ValueError("dataset_root must not be blank")
        if self.visual_provider.lower() == "mock":
            raise ValueError("mock visual provider is not allowed in production KIS")
        if self.index_provider not in {"faiss-flat-ip", "exact-numpy", "faiss-hnsw"}:
            raise ValueError(f"unsupported KIS index provider {self.index_provider!r}")
        if self.fusion_method not in {"rrf", "weighted"}:
            raise ValueError("fusion_method must be 'rrf' or 'weighted'")
        if self.fusion_rank_constant < 1:
            raise ValueError("fusion_rank_constant must be >= 1")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.quality_status not in {"UNVALIDATED_ON_HCMAIC", "VALIDATED_ON_HCMAIC"}:
            raise ValueError(f"unsupported quality_status {self.quality_status!r}")
        names = [channel.name for channel in self.channels]
        if len(names) != len(set(names)):
            raise ValueError("KIS channel names must be unique")
        if any(weight < 0 or not math.isfinite(weight) for weight in self.fusion_weights.values()):
            raise ValueError("fusion weights must be finite and non-negative")
        object.__setattr__(self, "dataset_root", Path(self.dataset_root))

    @property
    def enabled_channels(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels if channel.enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "visual_provider": self.visual_provider,
            "index_provider": self.index_provider,
            "sampling_policy": self.sampling_policy,
            "channels": [channel.to_dict() for channel in self.channels],
            "fusion_method": self.fusion_method,
            "fusion_rank_constant": self.fusion_rank_constant,
            "fusion_weights": dict(self.fusion_weights),
            "reranker": self.reranker,
            "output_path": str(self.output_path) if self.output_path else None,
            "benchmark_root": str(self.benchmark_root) if self.benchmark_root else None,
            "seed": self.seed,
            "quality_status": self.quality_status,
        }
