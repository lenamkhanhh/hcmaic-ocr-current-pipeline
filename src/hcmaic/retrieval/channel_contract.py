"""Shared channel contract helpers for HCMAIC retrieval adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

ExecutionStatus = Literal[
    "ENGINEERING_PROXY",
    "READY_LOCAL_PRECHECK",
    "VALIDATED_LOCAL",
    "DISABLED_BY_CONFIG",
    "DISABLED_BY_POLICY",
    "UNAVAILABLE",
]
QualityStatus = Literal[
    "UNVALIDATED",
    "UNVALIDATED_ON_HCMAIC",
    "VALIDATED_ON_HCMAIC",
]
ChannelStatus = Literal[
    "ready",
    "disabled_by_config",
    "disabled_by_policy",
    "unavailable",
    "schema_mismatch",
    "mapping_mismatch",
    "timeout",
]


@dataclass(frozen=True)
class ChannelContract:
    """Metadata that every HCMAIC channel should expose."""

    channel: str
    provider: str
    revision: str
    execution_status: ExecutionStatus
    quality_status: QualityStatus
    dataset_manifest_hash: str | None = None
    artifact_hash: str | None = None
    status: ChannelStatus = "ready"
    reason: str | None = None
    configured: bool | None = None
    ready: bool | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channel.strip():
            raise ValueError("channel must not be blank")
        if not self.provider.strip():
            raise ValueError("provider must not be blank")
        if not self.revision.strip():
            raise ValueError("revision must not be blank")
        if not self.execution_status.strip():
            raise ValueError("execution_status must not be blank")
        if not self.quality_status.strip():
            raise ValueError("quality_status must not be blank")
        if self.status not in {
            "ready",
            "disabled_by_config",
            "disabled_by_policy",
            "unavailable",
            "schema_mismatch",
            "mapping_mismatch",
            "timeout",
        }:
            raise ValueError("unsupported channel status")
        configured = self.configured
        ready = self.ready
        if configured is None:
            configured = self.status in {
                "ready",
                "disabled_by_policy",
                "schema_mismatch",
                "mapping_mismatch",
                "timeout",
            }
        if ready is None:
            ready = self.status == "ready"
        object.__setattr__(self, "configured", bool(configured))
        object.__setattr__(self, "ready", bool(ready))

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "configured": bool(self.configured),
            "ready": bool(self.ready),
            "status": self.status,
            "reason": self.reason,
            "provider": self.provider,
            "revision": self.revision,
            "execution_status": self.execution_status,
            "quality_status": self.quality_status,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "artifact_hash": self.artifact_hash,
        }

    def to_raw_provenance(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "revision": self.revision,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "artifact_hash": self.artifact_hash,
        }


@runtime_checkable
class ChannelAdapter(Protocol):
    provider: str
    revision: str
    execution_status: ExecutionStatus
    quality_status: QualityStatus
    dataset_manifest_hash: str | None
    artifact_hash: str | None

    def channel_contract(self) -> ChannelContract: ...

    def search(self, query: str, top_k: int = 100) -> list[Any]: ...


def build_channel_evidence(
    *,
    channel: str,
    provider: str,
    revision: str,
    execution_status: ExecutionStatus,
    quality_status: QualityStatus,
    frame_uid: str,
    video_id: str,
    video_filename: str,
    source_frame_idx: int,
    timestamp_ms: int,
    score: float,
    rank: int,
    dataset_manifest_hash: str | None = None,
    artifact_hash: str | None = None,
    channel_specific: Mapping[str, Any] | None = None,
    raw_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not channel.strip():
        raise ValueError("channel must not be blank")
    if not provider.strip() or not revision.strip():
        raise ValueError("provider and revision must not be blank")
    if not frame_uid.strip() or not video_id.strip() or not video_filename.strip():
        raise ValueError("frame/video identity must not be blank")
    if source_frame_idx < 0 or timestamp_ms < 0:
        raise ValueError("source_frame_idx and timestamp_ms must be non-negative")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    provenance = {
        "provider": provider,
        "revision": revision,
        "dataset_manifest_hash": dataset_manifest_hash,
        "artifact_hash": artifact_hash,
    }
    if raw_provenance is not None:
        provenance.update(dict(raw_provenance))
    payload: dict[str, Any] = {
        "entity_id": frame_uid,
        "channel": channel,
        "provider": provider,
        "revision": revision,
        "execution_status": execution_status,
        "quality_status": quality_status,
        "dataset_manifest_hash": dataset_manifest_hash,
        "artifact_hash": artifact_hash,
        "frame_uid": frame_uid,
        "video_id": video_id,
        "video_filename": video_filename,
        "source_frame_idx": source_frame_idx,
        "timestamp_ms": timestamp_ms,
        "score": score,
        "rank": rank,
        "raw_provenance": provenance,
        "channel_specific": dict(channel_specific or {}),
    }
    payload.update(dict(channel_specific or {}))
    return payload
