"""Small, fail-closed adapter for the BFE operator UI contract.

The BFE UI is a presentation client.  It must consume the existing dual visual
service without changing the artifact identity or pretending that the two
embedding spaces are one vector space.  This module contains only serialization
and request-validation helpers; retrieval and artifact validation stay in
``hcmaic.retrieval.dual_visual``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QUALITY_STATUS = "UNVALIDATED_ON_HCMAIC"
EXECUTION_STATUS = "ENGINEERING_PROXY"
ACTIVE_CHANNELS = ("siglip2", "qwen")
OPTIONAL_CHANNELS = ("ocr", "object", "asr", "trake", "qa")


class BFEKISSearchRequest(BaseModel):
    """JSON body sent by the BFE text-search client."""

    model_config = ConfigDict(extra="forbid")

    query_id: str | None = Field(default=None, min_length=1, max_length=256)
    task: Literal["TKIS"] = "TKIS"
    text: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=100, ge=1, le=500)
    ocr_query: str | None = Field(default=None, max_length=5000)
    object_query: str | None = Field(default=None, max_length=5000)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value.strip()

    @field_validator("object_query")
    @classmethod
    def _object_query_is_optional_and_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("ocr_query")
    @classmethod
    def _ocr_query_is_optional_and_trimmed(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class BFEInteractionEvent(BaseModel):
    """Bounded session telemetry accepted by the local operator console."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    query_revision: int | None = Field(default=None, ge=1)
    event_type: Literal[
        "query_submitted",
        "first_result",
        "feedback_recorded",
        "inspector_opened",
        "candidate_selected",
        "validation_failed",
        "download_completed",
    ]
    frame_uid: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    occurred_at: datetime
    safe_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("safe_metadata")
    @classmethod
    def _safe_metadata(
        cls, value: dict[str, str | int | float | bool | None]
    ) -> dict[str, str | int | float | bool | None]:
        allowed = {
            "task",
            "view",
            "rank",
            "result_count",
            "revision",
            "duration_bucket",
            "reason_code",
            "source",
        }
        if len(value) > 12 or any(key not in allowed for key in value):
            raise ValueError("interaction metadata contains an unsupported field")
        for key, item in value.items():
            if isinstance(item, str) and ("\\" in item or "/" in item or len(item) > 128):
                raise ValueError(f"interaction metadata {key!r} is not safe")
        return value

    @model_validator(mode="after")
    def _canonical_frame_uid(self) -> BFEInteractionEvent:
        if self.frame_uid is None:
            return self
        try:
            video_id, source_idx = self.frame_uid.rsplit(":", 1)
            if not video_id or int(source_idx) < 0:
                raise ValueError
        except (ValueError, TypeError) as exc:
            raise ValueError("frame_uid must be canonical") from exc
        return self


def frame_media_url(frame_uid: str) -> str:
    """Return a same-origin URL while keeping the canonical UID unchanged."""

    return f"/v1/frames/{quote(frame_uid, safe='')}/image"


def video_media_url(video_id: str) -> str:
    return f"/v1/videos/{quote(video_id, safe='')}/stream"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_entry(name: str, raw: Any, *, default_reason: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        payload = dict(raw)
        status = str(payload.get("status") or "unavailable")
        if status.startswith("ready"):
            status = "ready"
        elif status in {"disabled_by_policy", "disabled_by_config"}:
            pass
        elif status.startswith("disabled"):
            status = "disabled_by_policy"
        elif status not in {
            "unavailable",
            "schema_mismatch",
            "mapping_mismatch",
            "timeout",
            "disabled_by_policy",
            "disabled_by_config",
        }:
            status = "unavailable"
        payload["channel"] = name
        payload["status"] = status
        payload["ready"] = bool(payload.get("ready", status == "ready"))
        payload["configured"] = bool(payload.get("configured", payload["ready"]))
        payload.setdefault("reason", None if payload["ready"] else default_reason)
        payload.setdefault("provider", None)
        payload.setdefault("revision", None)
        return payload
    value = str(raw or "")
    if value.startswith("ready"):
        status, configured, reason = "ready", True, None
    elif value == "disabled_by_config":
        status, configured, reason = "disabled_by_config", False, "disabled_by_config"
    elif value.startswith("disabled"):
        status, configured, reason = (
            "disabled_by_policy",
            False,
            "disabled_until_qrels_ablation_gain",
        )
    elif value.startswith("unavailable:"):
        status, configured, reason = (
            "unavailable",
            True,
            value.split(":", 1)[1].strip() or default_reason,
        )
    else:
        status, configured, reason = "unavailable", False, value or default_reason
    return {
        "channel": name,
        "configured": configured,
        "ready": status == "ready",
        "status": status,
        "reason": reason,
        "provider": None,
        "revision": None,
    }


def channel_status_for_service(service: Any) -> dict[str, dict[str, Any]]:
    """Return the runtime's optional-channel capability contract.

    Optional channels are reported from the attached runtime only; an
    unconfigured channel remains fail-closed and never produces synthetic hits.
    """

    health = service.health() if hasattr(service, "health") else {}
    raw: Any = None
    channel_status_attr = getattr(service, "channel_status", None)
    if callable(channel_status_attr):
        raw = channel_status_attr()
    elif isinstance(channel_status_attr, Mapping):
        raw = channel_status_attr
    if raw is None:
        raw = health.get("channel_status") or health.get("channels")
    raw = raw if isinstance(raw, Mapping) else {}
    statuses: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        statuses[str(name)] = _status_entry(
            str(name), value, default_reason="optional_channel_unavailable"
        )

    provider_health = health.get("providers", {})
    if not isinstance(provider_health, Mapping):
        provider_health = {}
    if "visual" not in statuses:
        statuses["visual"] = {
            "channel": "visual",
            "configured": True,
            "ready": True,
            "status": "ready",
            "reason": None,
            "provider": "visual",
            "revision": str(health.get("index_version") or "unknown"),
        }
    for name in ACTIVE_CHANNELS:
        if name in statuses:
            continue
        details = provider_health.get(name, {})
        details = details if isinstance(details, Mapping) else {}
        statuses[name] = {
            "channel": name,
            "configured": True,
            "ready": True,
            "status": "ready",
            "reason": None,
            "provider": str(details.get("name") or name),
            "revision": str(details.get("version") or "unknown"),
        }
    for name in OPTIONAL_CHANNELS:
        if name in statuses:
            continue
        if name == "asr":
            statuses[name] = _status_entry(
                name,
                "disabled_by_policy",
                default_reason="disabled_until_qrels_ablation_gain",
            )
        else:
            statuses[name] = _status_entry(
                name,
                None,
                default_reason=(
                    "channel_not_attached_to_dual_visual_runtime"
                    if name in {"ocr", "object"}
                    else "channel_not_attached_to_local_runtime"
                ),
            )
    return statuses


def serialize_catalog_frame(row: dict[str, Any]) -> dict[str, Any]:
    """Map one catalog row to BFE's ``FrameCandidate`` shape."""

    frame_uid = str(row["frame_uid"])
    video_id = str(row["video_id"])
    feature_row = _optional_int(row.get("feature_row"))
    return {
        "frame_uid": frame_uid,
        "video_id": video_id,
        "video_filename": str(row.get("video_filename") or f"{video_id}.mp4"),
        "keyframe_id": row.get("keyframe_id") or frame_uid,
        "keyframe_path": row.get("keyframe_path"),
        "shot_id": row.get("shot_id"),
        "source_frame_idx": int(row["source_frame_idx"]),
        "timestamp_ms": int(row["timestamp_ms"]),
        # FAISS row is deliberately optional: it is index-local, never identity.
        "faiss_row": _optional_int(row.get("faiss_row")),
        "feature_row": feature_row,
        "rank": None,
        "score": None,
        "final_score": None,
        "evidence": {
            "identity_key": "frame_uid",
            "feature_row": feature_row,
            "quality_status": QUALITY_STATUS,
        },
        "signal_scores": {},
        "providers": [],
        "image_url": frame_media_url(frame_uid),
    }


def serialize_result(result: Any, service: Any) -> dict[str, Any]:
    """Map a canonical ``KISResult`` to the BFE result-wall contract."""

    row = service.get_frame(result.frame_uid)
    payload = serialize_catalog_frame(row)
    final_score = float(
        result.rerank_score if result.rerank_score is not None else result.fused_score
    )
    providers: list[str] = []
    evidence_rows: list[dict[str, Any]] = []
    for evidence in result.evidence:
        metadata = dict(evidence.metadata)
        provider = str(metadata.get("provider") or evidence.channel)
        if provider not in providers:
            providers.append(provider)
        evidence_row = evidence.to_dict()
        for field in (
            "label",
            "label_raw",
            "raw_labels",
            "normalized_label",
            "confidence",
            "bbox",
            "source_record_count",
            "duplicate_group",
            "duplicate_extra_instance_count",
            "instances",
            "source_records",
            "source_shard_ids",
            "frame_status",
            "model_id",
            "model_weights_sha256",
            "label_source",
            "provider_execution",
            "keyframe_paths",
            "matching_tokens",
            "revision",
            "segment_id",
            "window_id",
            "window_uid",
            "asr_window_uid",
            "start_ms",
            "end_ms",
            "text_raw",
            "manifest_sha256",
            "objects_sha256",
            "frame_status_sha256",
            "failure_ledger_sha256",
            "identity_hash",
            "channel",
            "execution_status",
            "quality_status",
            "dataset_manifest_hash",
            "artifact_hash",
            "raw_provenance",
            "channel_specific",
        ):
            if field in metadata:
                evidence_row[field] = metadata[field]
        evidence_rows.append(evidence_row)
    payload.update(
        {
            "rank": int(result.rank),
            "score": final_score,
            "final_score": final_score,
            "evidence": {
                "identity_key": "frame_uid",
                "channels": dict(result.channel_scores),
                "items": evidence_rows,
                "quality_status": QUALITY_STATUS,
            },
            "signal_scores": {name: float(score) for name, score in result.channel_scores.items()},
            "providers": providers or list(result.executed_channels),
            "executed_channels": list(result.executed_channels),
            "unavailable_channels": dict(result.unavailable_channels),
        }
    )
    if hasattr(service, "video_media_status"):
        video_status = service.video_media_status(str(result.video_id))
        payload.update(
            {
                "video_available": bool(video_status["available"]),
                "video_status": str(video_status["status"]),
                "video_stream_available": bool(
                    video_status.get("stream_available", video_status["available"])
                ),
                "video_stream_status": str(
                    video_status.get("stream_status", video_status["status"])
                ),
                "video_stream_reason": video_status.get("stream_reason"),
                "video_backend": video_status.get("backend"),
                "video_bytes": video_status.get("bytes"),
                "video_range_capable": video_status.get("range_capable", False),
                "video_provenance_status": video_status.get("provenance_status"),
            }
        )
    if hasattr(service, "frame_image_status"):
        payload.update(service.frame_image_status(result.frame_uid))
    return payload


def serialize_kis_response(
    query_id: str, task: str, results: list[Any], service: Any
) -> dict[str, Any]:
    health = service.health()
    statuses = channel_status_for_service(service)
    enabled_indexes = list(health.get("enabled_indexes") or [])
    disabled_indexes = list(health.get("disabled_indexes") or [])
    executed: list[str] = []
    for result in results:
        for channel in getattr(result, "executed_channels", ()):
            if channel not in executed:
                executed.append(channel)
    unavailable: dict[str, str] = {}
    for result in results:
        unavailable.update(dict(getattr(result, "unavailable_channels", {})))
    for name, status in statuses.items():
        if name in OPTIONAL_CHANNELS and not status.get("ready"):
            unavailable.setdefault(name, str(status.get("reason") or "channel_unavailable"))
    active = [name for name in ACTIVE_CHANNELS if statuses.get(name, {}).get("ready")]
    disabled = [name for name in OPTIONAL_CHANNELS if not statuses.get(name, {}).get("ready")]
    if not enabled_indexes:
        enabled_indexes = active
    if not disabled_indexes:
        disabled_indexes = [name for name in ACTIVE_CHANNELS if name not in enabled_indexes]
    return {
        "query_id": query_id,
        "task": task,
        "results": [serialize_result(result, service) for result in results],
        "configured_indexes": list(health.get("enabled_indexes") or []),
        "enabled_indexes": enabled_indexes,
        "disabled_indexes": disabled_indexes,
        "active_channels": active,
        "disabled_channels": disabled,
        "executed_channels": executed,
        "unavailable_channels": unavailable,
        "channel_status": statuses,
        "channel_contracts": health.get("channel_contracts", {}),
        "execution_status": EXECUTION_STATUS,
        "quality_status": QUALITY_STATUS,
        "index_version": service.artifacts.index_version,
    }


def serialize_providers(service: Any) -> dict[str, Any]:
    health = service.health()
    channel_health = health.get("providers", {})
    enabled_indexes = list(health.get("enabled_indexes") or [])
    disabled_indexes = list(health.get("disabled_indexes") or [])
    artifacts = getattr(service, "artifacts", None)
    visual_version = str(
        getattr(artifacts, "index_version", None)
        or health.get("index_version")
        or getattr(getattr(service, "index", None), "index_manifest", {}).get("index_version")
        or "unknown"
    )
    visual_provider_name = (
        getattr(getattr(service, "provider", None), "name", None)
        or health.get("embedding_provider")
    )
    if not visual_provider_name:
        visual_provider_name = "siglip2+qwen-rrf"
    statuses = channel_status_for_service(service)
    providers: dict[str, dict[str, Any]] = {
        "visual": {
            "provider": str(visual_provider_name),
            "version": visual_version,
            "available": True,
            "ready": True,
            "configured": True,
            "status": "ready",
            "reason": None,
            "execution_status": health.get("execution_status", EXECUTION_STATUS),
            "quality_status": health.get("quality_status", QUALITY_STATUS),
        },
    }
    for channel in ACTIVE_CHANNELS:
        details = channel_health.get(channel, {})
        status = statuses[channel]
        providers[channel] = {
            "provider": str(details.get("name", status.get("provider") or channel)),
            "version": str(details.get("version", "unknown")),
            "dimension": _optional_int(details.get("dimension")),
            "available": bool(status["ready"]),
            "ready": bool(status["ready"]),
            "configured": bool(status["configured"]),
            "status": status["status"],
            "reason": status["reason"],
            "execution_status": status.get("execution_status")
            or health.get("execution_status", EXECUTION_STATUS),
            "quality_status": status.get("quality_status")
            or health.get("quality_status", QUALITY_STATUS),
        }
    for channel in OPTIONAL_CHANNELS:
        status = statuses[channel]
        providers[channel] = {
            "provider": status.get("provider") or "optional",
            "version": status.get("revision"),
            "available": bool(status["ready"]),
            "ready": bool(status["ready"]),
            "configured": bool(status["configured"]),
            "status": status["status"],
            "reason": status["reason"],
            "execution_status": status.get("execution_status")
            or health.get("execution_status", EXECUTION_STATUS),
            "quality_status": status.get("quality_status")
            or health.get("quality_status", QUALITY_STATUS),
        }
    providers["enabled_indexes"] = enabled_indexes
    providers["disabled_indexes"] = disabled_indexes
    providers["configured_indexes"] = list(enabled_indexes)
    providers["channel_status"] = statuses
    providers["channel_contracts"] = health.get("channel_contracts", {})
    return providers
