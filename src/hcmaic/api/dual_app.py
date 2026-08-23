"""FastAPI viewer for the validated local SigLIP2 + Qwen dual index."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from hcmaic.api.bfe_adapter import (
    BFEInteractionEvent,
    BFEKISSearchRequest,
    channel_status_for_service,
    frame_media_url,
    serialize_catalog_frame,
    serialize_kis_response,
    serialize_providers,
    video_media_url,
)
from hcmaic.retrieval.asr_elasticsearch import ASRElasticsearchConfig, ASRElasticsearchError
from hcmaic.retrieval.dual_visual import (
    EXPECTED_CHANNELS,
    DualVisualService,
    load_dual_visual_service,
)
from hcmaic.retrieval.feedback import FeedbackEvent
from hcmaic.retrieval.image_thumbnail import (
    DEFAULT_IMAGE_THUMBNAIL_QUALITY,
    DEFAULT_IMAGE_THUMBNAIL_WIDTH,
    ImageThumbnailError,
)
from hcmaic.retrieval.media_resolver import (
    MediaRangeRequestError,
    MediaRangeUnsupportedError,
    MediaResolutionError,
    RemoteMediaResolver,
)
from hcmaic.retrieval.ocr_elasticsearch import ElasticsearchOCRError
from hcmaic.retrieval.remote_image_query import (
    MAX_REMOTE_IMAGE_URL_LENGTH,
    RemoteImageFetchError,
    fetch_remote_image,
)
from hcmaic.retrieval.temporal_selection import (
    SelectionError,
    TemporalSelectionService,
)
from hcmaic.retrieval.trake import build_stage_bundles, build_trake_tracks
from hcmaic.retrieval.video_frame import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_THUMBNAIL_WIDTH,
    VideoFrameDecodeError,
    decode_exact_video_frame,
    decode_exact_video_frame_url,
    decode_video_frame,
)
from hcmaic.submission.aic26_csv import (
    AIC26SubmissionError,
    generate_aic26_rows,
    render_aic26_csv,
)

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}
_MAX_BFE_UPLOAD_BYTES = 10 * 1024 * 1024
_MAX_QUERY_IMAGE_WIDTH = 8_192
_MAX_QUERY_IMAGE_HEIGHT = 8_192
_MAX_QUERY_IMAGE_PIXELS = 40_000_000
_LEGACY_UI_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"
_BFE_DIST_DIR = Path(__file__).resolve().parent.parent / "ui" / "dist"
_STAGE_CHANNELS = ("text", "ocr", "asr", "image", "object")
_BUNDLE_CANDIDATE_SCHEDULE = (500, 1_500, 3_000)
_ALL_HITS_LIMIT = 500
_ALL_HITS_DEFAULT_MIN_GAP_MS = 3_000
# All Hits may expose several complete temporal bundles from one video, but
# keep the per-video beam bounded so a dense candidate set cannot explode.
_ALL_HITS_MAX_BUNDLES_PER_VIDEO = 20
_ALL_HITS_BUNDLE_BEAM_WIDTH = 5_000
_MEDIA_CACHE_CONTROL = "public, max-age=3600"
_THUMBNAIL_CACHE_CONTROL = "public, max-age=86400, immutable"


def _has_explicit_whitespace_only_stage_query(
    stage: StageDefinition | TrakeStageDefinition,
) -> bool:
    explicit_values = [
        str(getattr(stage.channels, field_name))
        for field_name in stage.channels.model_fields_set
    ]
    return bool(explicit_values) and any(value != "" for value in explicit_values) and not any(
        value.strip() for value in explicit_values
    )


class DualSearchRequest(BaseModel):
    query_id: str | None = None
    text: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=500)
    video_ids: list[str] | None = None
    visual_indexes: list[str] | None = None
    ocr_query: str | None = Field(default=None, max_length=5000)
    object_query: str | None = Field(default=None, max_length=5000)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("visual_indexes")
    @classmethod
    def _visual_indexes_are_non_empty_known_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("visual_indexes must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("visual_indexes must not contain duplicates")
        unknown = sorted(set(cleaned) - set(EXPECTED_CHANNELS))
        if unknown:
            raise ValueError(f"visual_indexes contains unknown value(s): {unknown}")
        return [name for name in EXPECTED_CHANNELS if name in cleaned]

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


class StageImageFileReference(BaseModel):
    model_config = {"extra": "forbid"}

    file_key: str = Field(pattern=r"^s[1-5]_image$", min_length=8, max_length=16)


def _stage_channel_value(value: Any) -> str:
    if isinstance(value, StageImageFileReference):
        return value.file_key
    return str(value or "")


class StageChannelPayload(BaseModel):
    """Exactly five user-visible channel inputs; empty means disabled."""

    model_config = {"extra": "forbid"}

    text: str = Field(default="", max_length=5000)
    ocr: str = Field(default="", max_length=5000)
    asr: str = Field(default="", max_length=5000)
    image: str | StageImageFileReference = Field(default="")
    # Transport-only field for a user-supplied remote query image. It is
    # materialized to a local file reference before retrieval is executed.
    image_url: str = Field(default="", max_length=MAX_REMOTE_IMAGE_URL_LENGTH)
    object: str = Field(default="", max_length=5000)


class StageDefinition(BaseModel):
    model_config = {"extra": "forbid"}

    stage_id: Literal["S1", "S2", "S3", "S4", "S5"]
    channels: StageChannelPayload
    asr_mode: Literal["pho", "whisper_v3", "rrf"] = "rrf"
    top_k: int = Field(default=50, ge=1, le=500)


class TrakeStageDefinition(BaseModel):
    model_config = {"extra": "forbid"}

    stage_id: Literal["S1", "S2", "S3", "S4", "S5"]
    channels: StageChannelPayload
    asr_mode: Literal["pho", "whisper_v3", "rrf"] = "rrf"
    top_k: int = Field(default=50, ge=1, le=500)


def _stage_image_url(stage: StageDefinition | TrakeStageDefinition) -> str | None:
    explicit_url = str(getattr(stage.channels, "image_url", "") or "").strip()
    image = stage.channels.image
    if isinstance(image, StageImageFileReference):
        if explicit_url:
            raise HTTPException(status_code=400, detail="IMAGE_URL_CONFLICT")
        return None
    image_value = str(image or "").strip()
    if explicit_url and image_value:
        raise HTTPException(status_code=400, detail="IMAGE_URL_CONFLICT")
    # Keep accepting a string in the image slot for direct API clients while
    # the UI uses the explicit image_url transport field.
    return explicit_url or image_value or None


def _materialize_stage_image_url(stage: StageDefinition | TrakeStageDefinition) -> None:
    stage.channels.image = StageImageFileReference(file_key=f"{stage.stage_id.lower()}_image")
    stage.channels.image_url = ""


class StagedSearchRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query_id: str | None = Field(default=None, min_length=1, max_length=200)
    video_ids: list[str] | None = None
    stages: list[StageDefinition] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def _ordered_staged_contract(self) -> StagedSearchRequest:
        stage_ids = [stage.stage_id for stage in self.stages]
        stage_numbers = [int(stage_id[1:]) for stage_id in stage_ids]
        if len(set(stage_ids)) != len(stage_ids) or stage_numbers != sorted(stage_numbers):
            raise ValueError("staged stages must be ordered by physical stage_id S1 through S5")
        for stage in self.stages:
            if _has_explicit_whitespace_only_stage_query(stage):
                raise ValueError(
                    f"{stage.stage_id} requires at least one non-whitespace channel value"
                )
        return self


class TrakeSearchRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query_id: str | None = Field(default=None, min_length=1, max_length=200)
    video_ids: list[str] | None = None
    stages: list[TrakeStageDefinition] = Field(min_length=2, max_length=5)
    max_delta_ms: int = Field(default=60_000, ge=0, le=60_000)
    top_k: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def _ordered_trake_stage_contract(self) -> TrakeSearchRequest:
        stage_ids = [stage.stage_id for stage in self.stages]
        stage_numbers = [int(stage_id[1:]) for stage_id in stage_ids]
        if len(set(stage_ids)) != len(stage_ids) or stage_numbers != sorted(stage_numbers):
            raise ValueError("Trake stages must be ordered by physical stage_id S1 through S5")
        for stage in self.stages:
            if _has_explicit_whitespace_only_stage_query(stage):
                raise ValueError(
                    f"{stage.stage_id} requires at least one non-whitespace channel value"
                )
        return self


class UnifiedBundleSearchRequest(BaseModel):
    """Single UI search contract for ordinary and temporal stage bundles."""

    model_config = {"extra": "forbid"}

    query_id: str | None = Field(default=None, min_length=1, max_length=200)
    video_ids: list[str] | None = None
    stages: list[StageDefinition] = Field(min_length=1, max_length=5)
    temporal_enabled: bool = False
    max_delta_ms: int = Field(default=60_000, ge=0, le=60_000)
    view_mode: Literal["grouped", "all_hits"] = "grouped"
    all_hits_min_gap_ms: int = Field(
        default=_ALL_HITS_DEFAULT_MIN_GAP_MS,
        ge=0,
        le=600_000,
    )
    top_k: int = Field(default=_ALL_HITS_LIMIT, ge=1, le=_ALL_HITS_LIMIT)

    @model_validator(mode="after")
    def _ordered_bundle_stage_contract(self) -> UnifiedBundleSearchRequest:
        stage_ids = [stage.stage_id for stage in self.stages]
        stage_numbers = [int(stage_id[1:]) for stage_id in stage_ids]
        if (
            len(set(stage_ids)) != len(stage_ids)
            or stage_numbers != sorted(stage_numbers)
        ):
            raise ValueError("bundle stages must be ordered from S1 through S5")
        return self


class ReviewQueueRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query_id: str = Field(min_length=1, max_length=200)
    stage_id: Literal["S1", "S2", "S3", "S4", "S5"]
    video_id: str = Field(min_length=1, max_length=200)
    bundle_id: str | None = Field(default=None, max_length=300)
    frame_uid: str = Field(min_length=3, max_length=300)
    source_frame_idx: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    shot_id: str | None = Field(default=None, max_length=300)
    rank: int = Field(ge=1)
    scores: dict[str, float] = Field(default_factory=dict)
    selection_reason: str = Field(default="selected", min_length=1, max_length=1000)
    origin: Literal["search_result", "manual_seek"] = "search_result"
    requested_timestamp_ms: int | None = Field(default=None, ge=0)
    mapping_status: str | None = Field(default=None, max_length=100)
    provenance_status: str | None = Field(default=None, max_length=100)
    submission_task: Literal["KIS", "QA", "TRAKE"] | None = None
    chain_id: str | None = Field(default=None, max_length=300)
    event_step: int | None = Field(default=None, ge=0)
    selection_kind: str | None = Field(default=None, max_length=100)
    qa_answer: str | None = Field(default=None, max_length=100)
    bundle_temporal_enabled: bool | None = None

    @model_validator(mode="after")
    def _validate_submission_metadata(self) -> ReviewQueueRequest:
        self.bundle_id = str(self.bundle_id or "").strip() or None
        if self.submission_task == "QA" and (
            self.qa_answer is None
            or not isinstance(self.qa_answer, str)
            or not self.qa_answer.strip()
        ):
            raise ValueError("QA queue items require a non-whitespace qa_answer")
        if self.submission_task != "TRAKE":
            if any(
                value is not None
                for value in (self.chain_id, self.event_step, self.selection_kind)
            ):
                raise ValueError(
                    "KIS/QA queue items must not include TRAKE chain_id, "
                    "event_step, or selection_kind"
                )
            return self

        chain_id = str(self.chain_id or "").strip()
        if not chain_id:
            raise ValueError("TRAKE queue items require a stable chain_id")
        if self.event_step is None:
            raise ValueError("TRAKE queue items require a zero-based event_step")
        expected_selection_kind = f"E{self.event_step + 1}"
        if self.selection_kind != expected_selection_kind:
            raise ValueError(
                "TRAKE selection_kind must match event_step "
                f"({expected_selection_kind} for event_step={self.event_step})"
            )
        self.chain_id = chain_id
        return self


class ReviewQueuePatch(BaseModel):
    model_config = {"extra": "forbid"}

    selection_reason: str | None = Field(default=None, min_length=1, max_length=1000)
    queue_position: int | None = Field(default=None, ge=0)
    qa_answer: str | None = Field(default=None, max_length=100)
    bundle_temporal_enabled: bool | None = None


class ReviewQueueReorderRequest(BaseModel):
    """One authoritative item or group order for one query scope."""

    model_config = {"extra": "forbid"}

    query_id: str = Field(min_length=1, max_length=200)
    ordered_item_ids: list[str] | None = Field(default=None, max_length=500)
    ordered_group_ids: list[str] | None = Field(default=None, max_length=500)
    ordered_ids: list[str] | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _one_authoritative_order(self) -> ReviewQueueReorderRequest:
        provided = [
            ("ordered_item_ids", self.ordered_item_ids),
            ("ordered_group_ids", self.ordered_group_ids),
            ("ordered_ids", self.ordered_ids),
        ]
        active = [(name, values) for name, values in provided if values is not None]
        if len(active) != 1:
            raise ValueError(
                "provide exactly one of ordered_item_ids, ordered_group_ids, or ordered_ids"
            )
        name, values = active[0]
        normalized = [str(item).strip() for item in values or []]
        if not normalized:
            raise ValueError(f"{name} must contain at least one value")
        if any(not item for item in normalized):
            raise ValueError(f"{name} must not contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must not contain duplicates")
        if name in {"ordered_ids", "ordered_item_ids"}:
            self.ordered_item_ids = normalized
        else:
            self.ordered_group_ids = normalized
        return self


class AIC26SubmissionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    query_id: str = Field(min_length=1, max_length=200)
    task: Literal["KIS", "QA", "TRAKE"]
    filename: str | None = Field(default=None, min_length=1, max_length=200)
    target_rows: int = Field(default=100, ge=1, le=100)
    answer: str | None = Field(default=None, max_length=100)
    event_count: int | None = Field(default=None, ge=1, le=100)
    delta: int = Field(default=3, ge=1, le=1000)

    @model_validator(mode="after")
    def _qa_answer_is_non_whitespace(self) -> AIC26SubmissionRequest:
        if self.task == "QA" and (self.answer is None or not self.answer.strip()):
            raise ValueError("QA answer is required and must not be whitespace-only")
        return self


_EVENT_LABEL_RE = re.compile(r"^E[1-9][0-9]*$")


class TemporalSelectionBase(BaseModel):
    event_step: int = Field(default=0, ge=0)
    video_id: str = Field(min_length=1)
    selected_time_ms: int = Field(ge=0)
    candidate_frame_uid: str | None = None
    nearest_keyframe_uid: str | None = None
    mapping_mode: Literal["nearest_pts"] = "nearest_pts"
    session_id: str | None = Field(default=None, max_length=200)
    operator_id: str | None = Field(default=None, max_length=200)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=300)
    task: Literal["KIS", "TRAKE"] | None = None
    selection_kind: str | None = None

    @field_validator("selection_kind")
    @classmethod
    def _normalize_selection_kind(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if normalized != "KIS" and not _EVENT_LABEL_RE.fullmatch(normalized):
            raise ValueError("selection_kind must be KIS or E<number>")
        return normalized

    @model_validator(mode="after")
    def _validate_event_contract(self) -> TemporalSelectionBase:
        kind = self.selection_kind
        if kind is None:
            kind = "KIS" if self.task in {None, "KIS"} else f"E{self.event_step + 1}"
        if kind == "KIS":
            if self.task == "TRAKE":
                raise ValueError("TRAKE selections must use E<number>")
            if self.event_step != 0:
                raise ValueError("KIS event_step must be 0")
        else:
            expected_step = int(kind[1:]) - 1
            if self.task == "KIS":
                raise ValueError("KIS selections must use selection_kind=KIS")
            if self.event_step != expected_step:
                raise ValueError(f"{kind} maps to zero-based event_step={expected_step}")
        return self


class TemporalSelectionRequest(TemporalSelectionBase):
    query_id: str = Field(min_length=1)


class TemporalReplacementRequest(TemporalSelectionBase):
    pass


class TemporalValidationRequest(BaseModel):
    query_id: str = Field(min_length=1)
    task: Literal["KIS", "TRAKE"]


def _video_filter(video_ids: list[str] | None) -> list[str] | None:
    if video_ids is None:
        return None
    cleaned = [value.strip() for value in video_ids if value.strip()]
    return cleaned or None


def _aic26_filename(query_id: str, filename: str | None) -> str:
    candidate = str(filename or query_id).strip()
    if candidate.lower().endswith(".csv"):
        candidate = candidate[:-4]
    if not candidate:
        raise AIC26SubmissionError("AIC26 filename must not be empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", candidate):
        raise AIC26SubmissionError(
            "AIC26 filename must contain only letters, numbers, dot, underscore, or hyphen"
        )
    return f"{candidate}.csv"


def _result_payload(
    result: Any,
    service: DualVisualService,
    *,
    frame_status_cache: dict[str, dict[str, Any]] | None = None,
    video_status_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frame_status_cache = frame_status_cache if frame_status_cache is not None else {}
    video_status_cache = video_status_cache if video_status_cache is not None else {}
    item = result.to_dict()
    catalog_row = service.get_frame(result.frame_uid)
    item["shot_id"] = catalog_row.get("shot_id")
    item["keyframe_path"] = catalog_row.get("keyframe_path")
    item["image_url"] = f"/frames/{result.frame_uid}/image"
    item["thumbnail_url"] = f"/frames/{result.frame_uid}/thumbnail"
    item["video_url"] = f"/videos/{result.video_id}/stream"
    item["signal_scores"] = {name: float(score) for name, score in result.channel_scores.items()}
    frame_uid = str(result.frame_uid)
    if frame_uid not in frame_status_cache:
        frame_status_cache[frame_uid] = service.frame_image_status(frame_uid)
    item.update(frame_status_cache[frame_uid])
    video_id = str(result.video_id)
    if video_id not in video_status_cache:
        video_status_cache[video_id] = service.video_media_status(video_id)
    video_status = video_status_cache[video_id]
    item.update(
        {
            "video_available": bool(video_status["available"]),
            "video_status": str(video_status["status"]),
            "video_stream_available": bool(
                video_status.get("stream_available", video_status["available"])
            ),
            "video_stream_status": str(video_status.get("stream_status", video_status["status"])),
            "video_stream_reason": video_status.get("stream_reason"),
            "video_backend": video_status.get("backend"),
            "video_bytes": video_status.get("bytes"),
            "video_range_capable": video_status.get("range_capable", False),
            "video_provenance_status": video_status.get("provenance_status"),
            "video_sha256_status": video_status.get("sha256_status"),
            "video_source_path": video_status.get("source_path"),
            "video_member_path": video_status.get("member_path"),
            "video_media_info_id": video_status.get("media_info_id"),
            "video_dataset_id": video_status.get("dataset_id"),
            "video_range_probe_status": video_status.get("range_probe_status"),
            "video_range_probe_attempts": video_status.get("range_probe_attempts"),
            "video_source_manifest_id": video_status.get("source_manifest_id"),
            "video_source_fingerprint": video_status.get("source_fingerprint"),
            "video_remote_content_fingerprint": video_status.get("remote_content_fingerprint"),
            "video_join_method": video_status.get("join_method"),
        }
    )
    item["index_version"] = service.artifacts.index_version
    return item


def _result_payloads(results: list[Any], service: DualVisualService) -> list[dict[str, Any]]:
    """Serialize one response with shared frame/video metadata lookups."""

    frame_status_cache: dict[str, dict[str, Any]] = {}
    video_status_cache: dict[str, dict[str, Any]] = {}
    return [
        _result_payload(
            result,
            service,
            frame_status_cache=frame_status_cache,
            video_status_cache=video_status_cache,
        )
        for result in results
    ]


def _stage_result_score(item: dict[str, Any]) -> float:
    raw = item.get("fusion_score", item.get("final_score", item.get("score", 0.0)))
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return score if score == score else 0.0


def _flatten_all_hits(
    stage_results: dict[str, list[dict[str, Any]]],
    stage_ids: list[str],
    *,
    limit: int = _ALL_HITS_LIMIT,
) -> list[dict[str, Any]]:
    """Return the raw stage-hit union without video or cross-stage dedup."""

    stage_order = {stage_id: index for index, stage_id in enumerate(stage_ids)}
    flattened: list[dict[str, Any]] = []
    for stage_id in stage_ids:
        for raw_item in stage_results.get(stage_id, []):
            frame_uid = str(raw_item.get("frame_uid") or "")
            if not frame_uid:
                continue
            item = dict(raw_item)
            item["stage_id"] = str(item.get("stage_id") or stage_id)
            flattened.append(item)

    flattened.sort(
        key=lambda item: (
            -_stage_result_score(item),
            stage_order.get(str(item.get("stage_id")), len(stage_order)),
            int(item.get("rank_in_stage", item.get("rank", 2**31 - 1))),
            str(item.get("frame_uid") or ""),
        )
    )
    limited = flattened[: max(0, int(limit))]
    for rank, item in enumerate(limited, start=1):
        item["all_hits_rank"] = rank
        item["final_rank"] = rank
    return limited


def _diversify_all_hits(
    items: list[dict[str, Any]],
    *,
    min_gap_ms: int = _ALL_HITS_DEFAULT_MIN_GAP_MS,
) -> list[dict[str, Any]]:
    """Suppress near-duplicate positions without breaking complete bundles.

    Complete bundle members are selected as a unit.  This prevents a
    stage-level spacing filter from returning S1 from one temporal bundle and
    S2 from another, which is an invalid All Hits result.  Legacy unbundled
    rows keep the previous per-video/per-stage behavior.
    """

    gap_ms = max(0, int(min_gap_ms))

    bundle_groups: dict[str, list[dict[str, Any]]] = {}
    bundle_order: list[str] = []
    unbundled: list[dict[str, Any]] = []
    for raw_item in items:
        bundle_id = str(raw_item.get("bundle_id") or "").strip()
        if bundle_id:
            if bundle_id not in bundle_groups:
                bundle_groups[bundle_id] = []
                bundle_order.append(bundle_id)
            bundle_groups[bundle_id].append(dict(raw_item))
        else:
            unbundled.append(dict(raw_item))

    selected_bundles: list[dict[str, Any]] = []
    selected_bundle_positions: dict[str, list[tuple[str, int]]] = {}
    for bundle_id in bundle_order:
        group = bundle_groups[bundle_id]
        video_id = str(group[0].get("video_id") or "")
        positions: list[tuple[str, int]] = []
        for item in group:
            stage_id = str(item.get("stage_id") or "S1")
            try:
                timestamp_ms = int(item["timestamp_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            positions.append((stage_id, timestamp_ms))
        duplicate = False
        if gap_ms > 0 and positions:
            for selected_video, selected_positions in selected_bundle_positions.items():
                if selected_video != video_id:
                    continue
                for stage_id, timestamp_ms in positions:
                    if any(
                        selected_stage == stage_id
                        and abs(timestamp_ms - selected_timestamp) < gap_ms
                        for selected_stage, selected_timestamp in selected_positions
                    ):
                        duplicate = True
                        break
                if duplicate:
                    break
        if duplicate:
            continue
        selected_bundles.extend(group)
        selected_bundle_positions.setdefault(video_id, []).extend(positions)

    selected: list[dict[str, Any]] = []
    selected_timestamps: dict[tuple[str, str], list[int]] = {}
    exact_seen: set[tuple[str, str, str]] = set()
    for raw_item in [*selected_bundles, *unbundled]:
        item = dict(raw_item)
        frame_uid = str(item.get("frame_uid") or "")
        if not frame_uid:
            continue
        stage_id = str(item.get("stage_id") or "S1")
        bundle_id = str(item.get("bundle_id") or "").strip()
        exact_key = (bundle_id, stage_id, frame_uid)
        if exact_key in exact_seen:
            continue
        exact_seen.add(exact_key)

        video_id = str(item.get("video_id") or "")
        raw_timestamp = item.get("timestamp_ms")
        try:
            timestamp_ms = int(raw_timestamp) if raw_timestamp is not None else None
        except (TypeError, ValueError):
            timestamp_ms = None
        position_key = (video_id, stage_id)
        if (
            gap_ms > 0
            and timestamp_ms is not None
            and any(
                abs(timestamp_ms - existing) < gap_ms
                for existing in selected_timestamps.get(position_key, [])
            )
        ):
            continue

        selected.append(item)
        if timestamp_ms is not None:
            selected_timestamps.setdefault(position_key, []).append(timestamp_ms)

    for rank, item in enumerate(selected, start=1):
        item["all_hits_rank"] = rank
        item["final_rank"] = rank
    return selected


def _flatten_complete_bundle_hits(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten complete bundles in bundle/event order without mutating them."""

    flattened: list[dict[str, Any]] = []
    for bundle in bundles:
        bundle_id = bundle.get("bundle_id")
        bundle_rank = bundle.get("bundle_rank")
        bundle_score = float(bundle.get("bundle_score", bundle.get("score", 0.0)))
        for stage_item in bundle.get("stages", []):
            item = dict(stage_item)
            item.update(
                {
                    "bundle_id": bundle_id,
                    "bundle_rank": bundle_rank,
                    "bundle_score": bundle_score,
                    "fusion_score": float(item.get("stage_score", _stage_result_score(item))),
                }
            )
            flattened.append(item)

    for rank, item in enumerate(flattened, start=1):
        item["all_hits_rank"] = rank
        item["final_rank"] = rank
    return flattened


def _merge_independent_stage_results(
    stage_results: dict[str, list[dict[str, Any]]], stage_ids: list[str]
) -> list[dict[str, Any]]:
    """Deduplicate stage hits by frame_uid and order the visible union.

    Each stage is searched against the full corpus. The first stage that owns a
    frame keeps the frame's stage label, so the same image is never rendered in
    multiple stage buckets. Video groups are ordered by their best score; cards
    inside a group keep S1..S5 order and then score descending.
    """

    stage_order = {stage_id: index for index, stage_id in enumerate(stage_ids)}
    owned_by_frame: dict[str, dict[str, Any]] = {}
    for stage_id in stage_ids:
        for raw_item in stage_results.get(stage_id, []):
            frame_uid = str(raw_item.get("frame_uid") or "")
            if not frame_uid or frame_uid in owned_by_frame:
                continue
            item = dict(raw_item)
            item["stage_id"] = str(item.get("stage_id") or stage_id)
            owned_by_frame[frame_uid] = item

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in owned_by_frame.values():
        groups.setdefault(str(item.get("video_id") or ""), []).append(item)

    ordered_groups = sorted(
        groups.items(),
        key=lambda pair: (
            -max((_stage_result_score(item) for item in pair[1]), default=0.0),
            pair[0],
        ),
    )
    merged: list[dict[str, Any]] = []
    for _video_id, items in ordered_groups:
        items.sort(
            key=lambda item: (
                stage_order.get(str(item.get("stage_id")), len(stage_order)),
                -_stage_result_score(item),
                int(item.get("rank_in_stage", item.get("rank", 2**31 - 1))),
                str(item.get("frame_uid") or ""),
            )
        )
        for item in items:
            enriched = dict(item)
            enriched["final_rank"] = len(merged) + 1
            merged.append(enriched)
    return merged


def _channel_payload(
    service: DualVisualService,
    results: list[Any],
    visual_indexes: Any | None = None,
) -> dict[str, Any]:
    active_indexes = service.resolve_visual_indexes(visual_indexes)
    statuses = channel_status_for_service(service)
    for name in EXPECTED_CHANNELS:
        if name in service.enabled_indexes and name not in active_indexes:
            status = dict(statuses.get(name) or {})
            status.update(
                {
                    "channel": name,
                    "configured": True,
                    "ready": False,
                    "status": "disabled_by_request",
                    "reason": "disabled_by_request",
                    "execution_status": "DISABLED_BY_REQUEST",
                    "quality_status": "UNVALIDATED",
                }
            )
            statuses[name] = status
    health = service.health()
    executed: list[str] = []
    for result in results:
        for channel in getattr(result, "executed_channels", ()):
            if channel not in executed:
                executed.append(channel)
    unavailable: dict[str, str] = {}
    for result in results:
        unavailable.update(dict(getattr(result, "unavailable_channels", {})))
    for name, status in statuses.items():
        if name in {"ocr", "object", "asr"} and not status.get("ready"):
            unavailable.setdefault(name, str(status.get("reason") or "channel_unavailable"))
    return {
        "executed_channels": executed,
        "unavailable_channels": unavailable,
        "channel_status": statuses,
        "configured_indexes": list(health.get("enabled_indexes") or []),
        "enabled_indexes": list(active_indexes),
        "disabled_indexes": [name for name in EXPECTED_CHANNELS if name not in active_indexes],
    }


def _cleanup_query_images(paths: Mapping[str, Path]) -> None:
    for path in paths.values():
        path.unlink(missing_ok=True)


def _validated_query_image(payload: bytes, content_type: str) -> Path:
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="file must be image/jpeg or image/png")
    if not payload:
        raise HTTPException(status_code=400, detail="image body must not be empty")
    if len(payload) > _MAX_BFE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="UPLOAD_TOO_LARGE")
    expected_format = "JPEG" if normalized_type == "image/jpeg" else "PNG"
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            if (
                width < 1
                or height < 1
                or width > _MAX_QUERY_IMAGE_WIDTH
                or height > _MAX_QUERY_IMAGE_HEIGHT
                or width * height > _MAX_QUERY_IMAGE_PIXELS
            ):
                raise HTTPException(status_code=422, detail="IMAGE_DIMENSIONS_UNSUPPORTED")
            actual_format = str(image.format or "").upper()
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="invalid image payload") from exc
    if actual_format != expected_format:
        raise HTTPException(status_code=415, detail="image MIME does not match image content")
    suffix = ".png" if normalized_type == "image/png" else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
        handle.write(payload)
    return path


async def _validated_remote_query_image(url: str) -> Path:
    try:
        payload, content_type = await asyncio.to_thread(fetch_remote_image, url)
    except RemoteImageFetchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from exc
    except Exception as exc:
        # Do not surface transport exceptions, connection details, or the URL.
        raise HTTPException(status_code=502, detail="REMOTE_IMAGE_FETCH_FAILED") from exc
    return _validated_query_image(payload, content_type)


async def _resolve_stage_image_urls(
    body: StagedSearchRequest | TrakeSearchRequest | UnifiedBundleSearchRequest,
    paths: dict[str, Path],
) -> None:
    for stage in body.stages:
        image_url = _stage_image_url(stage)
        if not image_url:
            continue
        paths[stage.stage_id] = await _validated_remote_query_image(image_url)
        _materialize_stage_image_url(stage)


def _safe_stage_validation_detail(exc: ValidationError) -> str:
    if any("image_url" in error.get("loc", ()) for error in exc.errors()):
        return "IMAGE_URL_REJECTED"
    return str(exc)


async def _parse_stage_request(
    request: Request,
    model_type: type[BaseModel],
) -> tuple[BaseModel, dict[str, Path]]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "multipart/form-data":
        try:
            raw = await request.json()
            body = model_type.model_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            detail = (
                _safe_stage_validation_detail(exc)
                if isinstance(exc, ValidationError)
                else "invalid JSON body"
            )
            raise HTTPException(status_code=422, detail=detail) from exc
        paths: dict[str, Path] = {}
        try:
            await _resolve_stage_image_urls(body, paths)
        except Exception:
            _cleanup_query_images(paths)
            raise
        return body, paths

    try:
        form = await request.form()
    except AssertionError as exc:
        raise HTTPException(status_code=503, detail="MULTIPART_RUNTIME_UNAVAILABLE") from exc
    metadata = form.get("metadata")
    if metadata is None:
        raise HTTPException(status_code=400, detail="multipart field 'metadata' is required")
    if hasattr(metadata, "read"):
        metadata = (await metadata.read()).decode("utf-8")
    try:
        body = model_type.model_validate(json.loads(str(metadata)))
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        detail = (
            _safe_stage_validation_detail(exc)
            if isinstance(exc, ValidationError)
            else "invalid multipart metadata JSON"
        )
        raise HTTPException(status_code=422, detail=detail) from exc

    paths: dict[str, Path] = {}
    try:
        for stage in body.stages:
            image_url = _stage_image_url(stage)
            if image_url:
                paths[stage.stage_id] = await _validated_remote_query_image(image_url)
                _materialize_stage_image_url(stage)
                continue
            image = stage.channels.image
            if isinstance(image, str):
                if image.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="staged image metadata must use a file_key or image_url",
                    )
                continue
            upload = form.get(image.file_key)
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(
                    status_code=400,
                    detail=f"missing multipart image file: {image.file_key}",
                )
            payload = await upload.read()
            paths[stage.stage_id] = _validated_query_image(
                payload, str(getattr(upload, "content_type", ""))
            )
    except Exception:
        _cleanup_query_images(paths)
        raise
    return body, paths


def create_dual_app(
    artifacts_dir: Path | None = None,
    *,
    visual_indexes: tuple[str, ...] | None = None,
    asr_only: bool = False,
    image_root: Path | None = None,
    video_root: Path | None = None,
    siglip_device: str | None = None,
    qwen_device: str | None = None,
    local_files_only: bool = True,
    siglip_model_path: Path | None = None,
    qwen_model_path: Path | None = None,
    media_manifest: Path | None = None,
    media_cache_root: Path | None = None,
    media_http_hosts: set[str] | None = None,
    media_resolver: RemoteMediaResolver | None = None,
    object_sidecar: Path | None = None,
    object_enabled: bool = False,
    allow_engineering_proxy: bool = False,
    asr_config: ASRElasticsearchConfig | None = None,
    ocr_es_url: str | None = None,
    ocr_es_index: str | None = None,
    ocr_es_manifest: Path | None = None,
    ocr_es_include_low_conf: bool = True,
    ocr_es_api_key_env: str = "ELASTIC_API_KEY",
    ocr_es_username_env: str | None = None,
    ocr_es_password_env: str | None = None,
    ui_dir: Path | None = None,
    service: DualVisualService | None = None,
    temporal_root: Path | None = None,
    temporal_service: TemporalSelectionService | None = None,
) -> FastAPI:
    if service is None:
        if artifacts_dir is None:
            raise ValueError("artifacts_dir is required when service is not injected")
        if media_resolver is not None:
            raise ValueError("media_resolver can only be passed with an injected service")
        service = load_dual_visual_service(
            artifacts_dir,
            visual_indexes=() if asr_only else visual_indexes,
            asr_only=asr_only,
            image_root=image_root,
            video_root=video_root,
            siglip_device=siglip_device,
            qwen_device=qwen_device,
            local_files_only=local_files_only,
            siglip_model_path=siglip_model_path,
            model_path=qwen_model_path,
            media_manifest=media_manifest,
            media_cache_root=media_cache_root,
            media_http_hosts=media_http_hosts,
            object_sidecar=object_sidecar,
            object_enabled=object_enabled,
            allow_engineering_proxy=allow_engineering_proxy,
            asr_config=asr_config,
            ocr_es_url=ocr_es_url,
            ocr_es_index=ocr_es_index,
            ocr_es_manifest=ocr_es_manifest,
            ocr_es_include_low_conf=ocr_es_include_low_conf,
            ocr_es_api_key_env=ocr_es_api_key_env,
            ocr_es_username_env=ocr_es_username_env,
            ocr_es_password_env=ocr_es_password_env,
        )
    elif media_resolver is not None and service.media_resolver is None:
        service.media_resolver = media_resolver

    app = FastAPI(
        title="HCMAIC visual + OCR retrieval",
        version="0.2.0",
        description=(
            "Local SigLIP2 + Qwen3-VL viewer with optional crop-level OCR Elasticsearch "
            "late fusion. Execution is engineering evidence; retrieval quality remains "
            "UNVALIDATED without qrels."
        ),
    )
    app.state.service = service
    app.state.feedback_events: list[dict[str, Any]] = []
    app.state.interaction_events: list[dict[str, Any]] = []
    app.state.review_queue: dict[str, dict[str, Any]] = {}
    app.state.review_queue_lock = RLock()

    def _queue_sort_key(item: dict[str, Any], fallback: int) -> tuple[int, int, int]:
        try:
            position = int(item.get("queue_position", fallback))
        except (TypeError, ValueError):
            position = fallback
        try:
            rank = int(item.get("rank", fallback))
        except (TypeError, ValueError):
            rank = fallback
        return position, rank, fallback

    def _ordered_queue_items(query_id: str | None = None) -> list[dict[str, Any]]:
        with app.state.review_queue_lock:
            items = list(app.state.review_queue.values())
            if query_id is not None:
                items = [item for item in items if item["query_id"] == query_id]
            return [
                item
                for _, item in sorted(
                    enumerate(items),
                    key=lambda pair: _queue_sort_key(pair[1], pair[0]),
                )
            ]

    def _queue_item_payload(item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item)
        # Keep the persisted tri-state contract explicit.  Legacy in-memory
        # rows may predate this field; they remain unknown (None), not false.
        payload.setdefault("bundle_temporal_enabled", None)
        if not str(payload.get("queue_group_id") or "").strip():
            payload["queue_group_id"] = _queue_group_id(payload)
        return payload

    def _queue_snapshot(query_id: str | None = None) -> list[dict[str, Any]]:
        with app.state.review_queue_lock:
            return [_queue_item_payload(item) for item in _ordered_queue_items(query_id)]

    def _reorder_queue_item(queue_item_id: str, queue_position: int) -> dict[str, Any]:
        with app.state.review_queue_lock:
            item = app.state.review_queue[queue_item_id]
            siblings = _ordered_queue_items(str(item["query_id"]))
            siblings = [candidate for candidate in siblings if candidate is not item]
            insert_at = min(max(queue_position, 0), len(siblings))
            siblings.insert(insert_at, item)
            for position, candidate in enumerate(siblings):
                candidate["queue_position"] = position
            return _queue_item_payload(item)

    def _queue_group_id(item: dict[str, Any]) -> str:
        existing = str(item.get("queue_group_id") or "").strip()
        if existing:
            return existing
        task = str(item.get("submission_task") or "KIS").strip().upper()
        if task == "TRAKE" and item.get("chain_id"):
            return (
                f"{item.get('query_id', '')}|TRAKE|{item.get('chain_id', '')}|"
                f"{item.get('video_id', '')}"
            )
        bundle_id = str(item.get("bundle_id") or "").strip()
        if task in {"KIS", "QA"} and bundle_id:
            return (
                f"{item.get('query_id', '')}|{task}|{bundle_id}|"
                f"{item.get('video_id', '')}"
            )
        return str(item.get("queue_item_id") or "")

    def _queue_group_member_sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
        event_step = item.get("event_step")
        if event_step is not None:
            try:
                return 0, int(event_step), int(item.get("queue_position", 0)), str(
                    item.get("queue_item_id") or ""
                )
            except (TypeError, ValueError):
                pass
        try:
            queue_position = int(item.get("queue_position", 0))
        except (TypeError, ValueError):
            queue_position = 0
        return 1, queue_position, int(item.get("rank", 0) or 0), str(
            item.get("queue_item_id") or ""
        )

    def _bulk_reorder_queue(body: ReviewQueueReorderRequest) -> list[dict[str, Any]]:
        with app.state.review_queue_lock:
            siblings = _ordered_queue_items(body.query_id)
            if body.ordered_group_ids is not None:
                groups: dict[str, list[dict[str, Any]]] = {}
                for item in siblings:
                    groups.setdefault(_queue_group_id(item), []).append(item)
                expected_group_ids = set(groups)
                requested_group_ids = list(body.ordered_group_ids)
                missing_groups = sorted(expected_group_ids - set(requested_group_ids))
                extra_groups = sorted(set(requested_group_ids) - expected_group_ids)
                if (
                    missing_groups
                    or extra_groups
                    or len(requested_group_ids) != len(expected_group_ids)
                ):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "ordered_group_ids must exactly match queue group membership for "
                            f"query_id={body.query_id!r}; missing={missing_groups}, "
                            f"extra={extra_groups}"
                        ),
                    )
                requested_ids = [
                    str(item["queue_item_id"])
                    for group_id in requested_group_ids
                    for item in sorted(groups[group_id], key=_queue_group_member_sort_key)
                ]
            else:
                requested_ids = list(body.ordered_item_ids or [])
            expected_ids = {str(item["queue_item_id"]) for item in siblings}
            requested_set = set(requested_ids)
            foreign_ids = [
                item_id
                for item_id in requested_ids
                if item_id in app.state.review_queue
                and str(app.state.review_queue[item_id]["query_id"]) != body.query_id
            ]
            if foreign_ids:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"ordered_item_ids must stay within query_id={body.query_id!r}; "
                        f"foreign item(s): {foreign_ids}"
                    ),
                )
            missing = sorted(expected_ids - requested_set)
            extra = sorted(requested_set - expected_ids)
            if missing or extra or len(requested_ids) != len(expected_ids):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "ordered_item_ids must exactly match the queue membership for "
                        f"query_id={body.query_id!r}; missing={missing}, extra={extra}"
                    ),
                )
            for position, item_id in enumerate(requested_ids):
                app.state.review_queue[item_id]["queue_position"] = position
            return _queue_snapshot(body.query_id)

    if temporal_service is not None and temporal_root is not None:
        raise ValueError("pass temporal_root or temporal_service, not both")
    app.state.temporal_selection = temporal_service or (
        TemporalSelectionService(temporal_root) if temporal_root is not None else None
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return service.health()

    @app.get("/system/info")
    def system_info() -> dict[str, Any]:
        health = service.health()
        return {
            "merge_manifest": service.artifacts.merge_manifest,
            "fusion_contract": service.runtime_fusion_contract,
            "artifact_fusion_contract": service.artifacts.fusion_contract,
            "health": health,
            "video_ids": service.video_ids(),
            # Compatibility shape for the bundled static UI and older BFE
            # operator clients.  The manifest/fusion contract remains
            # authoritative for machine checks.
            "runtime": {
                "fusion": service.artifacts.fusion_contract,
                "index_version": service.artifacts.index_version,
            },
        }

    @app.get("/system/providers")
    def system_providers() -> dict[str, Any]:
        """BFE capability contract; unavailable channels stay fail-closed."""

        return serialize_providers(service)

    @app.get("/object/aliases")
    def object_aliases() -> dict[str, Any]:
        """Return the versioned lightweight object alias catalog for the UI."""

        return service.object_aliases()

    def _search_text(body: DualSearchRequest) -> dict[str, Any]:
        query_id = body.query_id or f"dual-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        try:
            results = service.search_text(
                query_id,
                body.text,
                top_k=body.top_k,
                video_ids=_video_filter(body.video_ids),
                visual_indexes=body.visual_indexes,
                object_query=body.object_query,
                ocr_query=body.ocr_query,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        payload = {
            "query_id": query_id,
            "task": "TKIS",
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED",
            "results": _result_payloads(results, service),
        }
        payload.update(_channel_payload(service, results, body.visual_indexes))
        return payload

    @app.post("/search")
    def search(body: DualSearchRequest) -> dict[str, Any]:
        return _search_text(body)

    @app.post("/search/text")
    def search_text_compat(body: DualSearchRequest) -> dict[str, Any]:
        """Backward-compatible alias used by the legacy static UI."""

        return _search_text(body)

    def _stage_channel_status(
        channel: str,
        value: Any,
        *,
        asr_mode: str = "rrf",
        image_path: Path | None = None,
    ) -> dict[str, Any]:
        normalized = _stage_channel_value(value).strip()
        base = {
            "channel": channel,
            "configured": bool(normalized),
            "ready": False,
            "quality_status": "UNVALIDATED",
        }
        if not normalized:
            status = {
                **base,
                "status": "disabled_by_empty_input",
                "reason": "disabled_by_empty_input",
                "execution_status": "DISABLED_BY_EMPTY_INPUT",
                "action": "no_call",
            }
            if channel == "asr":
                status["mode"] = asr_mode
            return status
        if channel == "text":
            visual = dict(service.channel_status().get("siglip2") or {})
            if visual.get("ready"):
                return {
                    **base,
                    "configured": True,
                    "ready": True,
                    "status": "ready",
                    "reason": None,
                    "execution_status": "ENGINEERING_PROXY",
                    "action": "execute_siglip2",
                    "provider": visual.get("provider"),
                    "revision": visual.get("revision"),
                }
            return {
                **base,
                "status": "unavailable",
                "reason": "visual SigLIP2 adapter unavailable",
                "execution_status": "UNAVAILABLE",
                "action": "no_call",
            }
        if channel == "ocr":
            ocr_status = dict(service.channel_status().get("ocr") or {})
            if ocr_status.get("ready"):
                return {
                    **base,
                    "configured": True,
                    "ready": True,
                    "status": "ready",
                    "reason": None,
                    "execution_status": str(
                        ocr_status.get("execution_status") or "ENGINEERING_PROXY"
                    ),
                    "action": "execute_ocr",
                    "provider": ocr_status.get("provider"),
                    "revision": ocr_status.get("revision"),
                    "index": ocr_status.get("index"),
                    "include_low_conf": ocr_status.get("include_low_conf"),
                }
            return {
                **base,
                "configured": bool(ocr_status.get("configured")),
                "status": str(ocr_status.get("status") or "unavailable"),
                "reason": (
                    "OCR artifact unavailable: optional adapter is not attached"
                    if ocr_status.get("reason")
                    == "channel_not_attached_to_dual_visual_runtime"
                    else str(ocr_status.get("reason") or "ocr adapter unavailable")
                ),
                "execution_status": str(ocr_status.get("execution_status") or "UNAVAILABLE"),
                "action": "no_call",
                "provider": ocr_status.get("provider"),
                "revision": ocr_status.get("revision"),
                "index": ocr_status.get("index"),
            }
        if channel == "object":
            object_status = dict(service.channel_status().get("object") or {})
            if object_status.get("ready"):
                return {
                    **base,
                    "configured": True,
                    "ready": True,
                    "status": "ready",
                    "reason": None,
                    "execution_status": str(
                        object_status.get("execution_status") or "ENGINEERING_PROXY"
                    ),
                    "action": "execute_object",
                    "provider": object_status.get("provider"),
                    "revision": object_status.get("revision"),
                }
            return {
                **base,
                "status": "unavailable",
                "reason": str(object_status.get("reason") or "object adapter unavailable"),
                "execution_status": "UNAVAILABLE",
                "action": "no_call",
            }
        if channel == "asr":
            asr_status = dict(service.channel_status().get("asr") or {})
            if asr_status.get("ready"):
                return {
                    **base,
                    "configured": True,
                    "ready": True,
                    "status": "ready",
                    "reason": None,
                    "execution_status": str(
                        asr_status.get("execution_status") or "ENGINEERING_PROXY"
                    ),
                    "action": "execute_asr",
                    "provider": asr_status.get("provider"),
                    "revision": asr_status.get("revision"),
                    "mode": asr_mode,
                    "fuzziness": asr_status.get("fuzziness"),
                    "index": asr_status.get("index"),
                }
            return {
                **base,
                "configured": bool(asr_status.get("configured")),
                "status": str(asr_status.get("status") or "unavailable"),
                "reason": str(asr_status.get("reason") or "asr adapter unavailable"),
                "execution_status": str(asr_status.get("execution_status") or "UNAVAILABLE"),
                "action": "no_call",
                "provider": asr_status.get("provider"),
                "revision": asr_status.get("revision"),
                "mode": asr_mode,
            }
        if channel == "image":
            visual = dict(service.channel_status().get("siglip2") or {})
            if image_path is not None and visual.get("ready"):
                provider = service.channel_status().get("siglip2", {})
                return {
                    **base,
                    "configured": True,
                    "ready": True,
                    "status": "ready",
                    "reason": None,
                    "execution_status": "ENGINEERING_PROXY",
                    "action": "execute_image",
                    "provider": provider.get("provider"),
                    "revision": provider.get("revision"),
                    "preprocessing": "SigLIP2 P14-384 RGB AutoProcessor",
                    "index": "siglip2",
                    "identity_key": "frame_uid",
                }
            return {
                **base,
                "status": "unavailable",
                "reason": "image query file is not attached or SigLIP2 is unavailable",
                "execution_status": "UNAVAILABLE",
                "action": "no_call",
            }
        reasons = {
            "asr": "ASR artifact unavailable: optional adapter is not attached",
            "image": "image adapter unavailable: stage payload is not attached",
            "object": "object artifact unavailable: optional adapter is not attached",
        }
        return {
            **base,
            "status": "unavailable",
            "reason": reasons[channel],
            "execution_status": "UNAVAILABLE",
            "action": "no_call",
        }

    def _run_stage(
        query_id: str,
        definition: StageDefinition | TrakeStageDefinition,
        candidate_frame_uids: set[str] | None,
        *,
        video_ids: list[str] | None = None,
        fusion_method: str = "harmonic",
        frame_status_cache: dict[str, dict[str, Any]] | None = None,
        video_status_cache: dict[str, dict[str, Any]] | None = None,
        stage_image_paths: Mapping[str, Path] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], set[str], dict[str, str]]:
        values = definition.channels.model_dump()
        image_path = (stage_image_paths or {}).get(definition.stage_id)
        statuses = {
            channel: _stage_channel_status(
                channel,
                values[channel],
                asr_mode=definition.asr_mode,
                image_path=image_path if channel == "image" else None,
            )
            for channel in _STAGE_CHANNELS
        }
        stage_unavailable: dict[str, str] = {
            channel: str(status["reason"])
            for channel, status in statuses.items()
            if status["status"] not in {"ready", "disabled_by_empty_input"}
        }
        results: list[dict[str, Any]] = []

        def append_found(found: list[Any]) -> None:
            for rank, result in enumerate(found, start=1):
                item = _result_payload(
                    result,
                    service,
                    frame_status_cache=frame_status_cache,
                    video_status_cache=video_status_cache,
                )
                item.update(
                    {
                        "stage_id": definition.stage_id,
                        "rank_in_stage": rank,
                        "final_rank": rank,
                        "fusion_score": float(result.fused_score),
                    }
                )
                results.append(item)

        def disable_for_empty_pool(channel: str) -> None:
            if str(values[channel]).strip() and statuses[channel]["status"] == "ready":
                statuses[channel].update(
                    {
                        "status": "disabled_by_empty_candidate_pool",
                        "ready": False,
                        "reason": "disabled_by_empty_candidate_pool",
                        "execution_status": "DISABLED_BY_CANDIDATE_POOL",
                        "action": "no_call",
                    }
                )

        def mark_asr_failure() -> None:
            runtime = dict(service.channel_status().get("asr") or {})
            statuses["asr"].update(
                {
                    "configured": bool(runtime.get("configured")),
                    "ready": False,
                    "status": str(runtime.get("status") or "unavailable"),
                    "reason": str(runtime.get("reason") or "asr_es_request_failed"),
                    "execution_status": str(runtime.get("execution_status") or "UNAVAILABLE"),
                    "action": "no_call",
                }
            )
            stage_unavailable["asr"] = str(statuses["asr"]["reason"])

        def mark_ocr_failure() -> None:
            runtime = dict(service.channel_status().get("ocr") or {})
            statuses["ocr"].update(
                {
                    "configured": bool(runtime.get("configured")),
                    "ready": False,
                    "status": str(runtime.get("status") or "unavailable"),
                    "reason": str(runtime.get("reason") or "ocr_es_request_failed"),
                    "execution_status": str(runtime.get("execution_status") or "UNAVAILABLE"),
                    "action": "no_call",
                }
            )
            stage_unavailable["ocr"] = str(statuses["ocr"]["reason"])

        text = _stage_channel_value(values["text"])
        ocr_query = _stage_channel_value(values["ocr"])
        asr_query = _stage_channel_value(values["asr"])
        object_query = _stage_channel_value(values["object"])
        text_ready = bool(text.strip()) and statuses["text"]["status"] == "ready"
        ocr_ready = bool(ocr_query.strip()) and statuses["ocr"]["status"] == "ready"
        asr_ready = bool(asr_query.strip()) and statuses["asr"]["status"] == "ready"
        object_ready = bool(object_query.strip()) and statuses["object"]["status"] == "ready"
        image_ready = image_path is not None and statuses["image"]["status"] == "ready"
        if text_ready:
            if candidate_frame_uids is not None and not candidate_frame_uids:
                disable_for_empty_pool("text")
                disable_for_empty_pool("ocr")
                disable_for_empty_pool("asr")
                disable_for_empty_pool("object")
                disable_for_empty_pool("image")
            else:
                try:
                    found = service.search_text(
                        f"{query_id}:{definition.stage_id}",
                        text,
                        top_k=definition.top_k,
                        video_ids=video_ids,
                        visual_indexes=("siglip2",),
                        candidate_frame_uids=candidate_frame_uids,
                        ocr_query=ocr_query if ocr_ready else None,
                        asr_query=asr_query if asr_ready else None,
                        asr_mode=definition.asr_mode if asr_ready else None,
                        object_query=object_query if object_ready else None,
                        image_path=image_path if image_ready else None,
                        fusion_method=fusion_method,
                        allow_large_top_k=definition.top_k > 500,
                    )
                except ElasticsearchOCRError:
                    mark_ocr_failure()
                except ASRElasticsearchError:
                    mark_asr_failure()
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    statuses["text"].update(
                        {
                            "status": "unavailable",
                            "ready": False,
                            "reason": f"visual adapter unavailable: {exc}",
                            "execution_status": "UNAVAILABLE",
                            "action": "no_call",
                        }
                    )
                    stage_unavailable["text"] = str(statuses["text"]["reason"])
                else:
                    append_found(found)
        elif ocr_ready:
            if candidate_frame_uids is not None and not candidate_frame_uids:
                disable_for_empty_pool("ocr")
                disable_for_empty_pool("asr")
                disable_for_empty_pool("object")
                disable_for_empty_pool("image")
            else:
                try:
                    found = service.search_ocr(
                        f"{query_id}:{definition.stage_id}",
                        ocr_query,
                        top_k=definition.top_k,
                        video_ids=video_ids,
                        candidate_frame_uids=candidate_frame_uids,
                        object_query=object_query if object_ready else None,
                        asr_query=asr_query if asr_ready else None,
                        asr_mode=definition.asr_mode if asr_ready else None,
                        fusion_method=fusion_method,
                        visual_indexes=("siglip2",) if image_ready else None,
                        image_path=image_path if image_ready else None,
                        allow_large_top_k=definition.top_k > 500,
                    )
                except ElasticsearchOCRError:
                    mark_ocr_failure()
                except ASRElasticsearchError:
                    mark_asr_failure()
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    statuses["ocr"].update(
                        {
                            "status": "unavailable",
                            "ready": False,
                            "reason": f"OCR adapter unavailable: {exc}",
                            "execution_status": "UNAVAILABLE",
                            "action": "no_call",
                        }
                    )
                    stage_unavailable["ocr"] = str(statuses["ocr"]["reason"])
                else:
                    append_found(found)
        elif asr_ready:
            if candidate_frame_uids is not None and not candidate_frame_uids:
                disable_for_empty_pool("asr")
                disable_for_empty_pool("object")
                disable_for_empty_pool("image")
            else:
                try:
                    found = service.search_asr(
                        f"{query_id}:{definition.stage_id}",
                        asr_query,
                        top_k=definition.top_k,
                        video_ids=video_ids,
                        candidate_frame_uids=candidate_frame_uids,
                        object_query=object_query if object_ready else None,
                        asr_mode=definition.asr_mode if asr_ready else None,
                        visual_indexes=("siglip2",) if image_ready else None,
                        image_path=image_path if image_ready else None,
                        fusion_method=fusion_method,
                        allow_large_top_k=definition.top_k > 500,
                    )
                except ASRElasticsearchError:
                    mark_asr_failure()
                except (FileNotFoundError, RuntimeError, ValueError):
                    statuses["asr"].update(
                        {
                            "status": "unavailable",
                            "ready": False,
                            "reason": "asr_es_request_failed",
                            "execution_status": "UNAVAILABLE",
                            "action": "no_call",
                        }
                    )
                    stage_unavailable["asr"] = str(statuses["asr"]["reason"])
                else:
                    append_found(found)
        elif object_ready:
            if candidate_frame_uids is not None and not candidate_frame_uids:
                disable_for_empty_pool("object")
                disable_for_empty_pool("image")
            else:
                try:
                    found = service.search_object(
                        f"{query_id}:{definition.stage_id}",
                        object_query,
                        top_k=definition.top_k,
                        video_ids=video_ids,
                        candidate_frame_uids=candidate_frame_uids,
                        visual_indexes=("siglip2",) if image_ready else None,
                        image_path=image_path if image_ready else None,
                        fusion_method=fusion_method,
                        allow_large_top_k=definition.top_k > 500,
                    )
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    statuses["object"].update(
                        {
                            "status": "unavailable",
                            "ready": False,
                            "reason": f"object adapter unavailable: {exc}",
                            "execution_status": "UNAVAILABLE",
                            "action": "no_call",
                        }
                    )
                    stage_unavailable["object"] = str(statuses["object"]["reason"])
                else:
                    append_found(found)
        elif image_ready:
            try:
                found = service.search_image(
                    f"{query_id}:{definition.stage_id}",
                    image_path,
                    top_k=definition.top_k,
                    video_ids=video_ids,
                    visual_indexes=("siglip2",),
                    candidate_frame_uids=candidate_frame_uids,
                    allow_large_top_k=definition.top_k > 500,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                statuses["image"].update(
                    {
                        "status": "unavailable",
                        "ready": False,
                        "reason": f"image adapter unavailable: {exc}",
                        "execution_status": "UNAVAILABLE",
                        "action": "no_call",
                    }
                )
                stage_unavailable["image"] = str(statuses["image"]["reason"])
            else:
                append_found(found)
        return results, statuses, {str(item["frame_uid"]) for item in results}, stage_unavailable

    def _execute_independent_stage_search(
        query_id: str,
        definitions: list[StageDefinition | TrakeStageDefinition],
        *,
        query_action: str,
        candidate_top_k: int | None = None,
        video_ids: list[str] | None = None,
        stage_image_paths: Mapping[str, Path] | None = None,
    ) -> dict[str, Any]:
        stage_results: dict[str, list[dict[str, Any]]] = {}
        stage_status: dict[str, dict[str, dict[str, Any]]] = {}
        unavailable_channels: dict[str, str] = {}
        executed_channels: list[str] = []
        planner_stages: dict[str, dict[str, Any]] = {}
        frame_status_cache: dict[str, dict[str, Any]] = {}
        video_status_cache: dict[str, dict[str, Any]] = {}

        for definition in definitions:
            values = definition.channels.model_dump()
            breakdown = {
                channel: {
                    "enabled": bool(str(values[channel]).strip()),
                    "value": str(values[channel]),
                    "planned_value": str(values[channel]),
                    "action": (
                        query_action
                        if str(values[channel]).strip()
                        else "disabled_by_empty_input"
                    ),
                }
                for channel in _STAGE_CHANNELS
            }
            planner_stages[definition.stage_id] = {
                "original_query": dict(values),
                "planned_query": dict(values),
                "asr_mode": definition.asr_mode,
                "fusion_method": "harmonic-mean-minmax",
                "breakdown": breakdown,
            }
            search_definition = (
                definition
                if candidate_top_k is None
                else definition.model_copy(update={"top_k": candidate_top_k})
            )
            found, statuses, _, stage_unavailable = _run_stage(
                query_id,
                search_definition,
                None,
                video_ids=video_ids,
                fusion_method="harmonic",
                frame_status_cache=frame_status_cache,
                video_status_cache=video_status_cache,
                stage_image_paths=stage_image_paths,
            )
            stage_results[definition.stage_id] = found
            stage_status[definition.stage_id] = statuses
            unavailable_channels.update(
                {
                    f"{definition.stage_id}:{name}": reason
                    for name, reason in stage_unavailable.items()
                }
            )
            for channel in ("siglip2", "qwen", "image", "ocr", "object", "asr"):
                if channel not in executed_channels and any(
                    item.get("executed_channels") and channel in item["executed_channels"]
                    for item in found
                ):
                    executed_channels.append(channel)

        stage_ids = [definition.stage_id for definition in definitions]
        first_stage_id = stage_ids[0] if stage_ids else "S1"
        return {
            "stage_ids": stage_ids,
            "stage_results": stage_results,
            "stage_result_counts": {
                stage_id: len(stage_results[stage_id]) for stage_id in stage_ids
            },
            "planner_stages": planner_stages,
            "stage_status": stage_status,
            "channel_status": stage_status.get(first_stage_id, {}),
            "executed_channels": executed_channels,
            "unavailable_channels": unavailable_channels,
        }

    @app.post("/search/bundles")
    async def search_bundles(request: Request) -> dict[str, Any]:
        body, image_paths = await _parse_stage_request(request, UnifiedBundleSearchRequest)
        try:
            return _search_bundles(body, image_paths)
        finally:
            _cleanup_query_images(image_paths)

    def _search_bundles(
        body: UnifiedBundleSearchRequest,
        stage_image_paths: Mapping[str, Path] | None = None,
    ) -> dict[str, Any]:
        """Run active stages independently, then rank complete video bundles."""

        query_id = body.query_id or f"bundle-{uuid.uuid4().hex[:12]}"
        video_ids = _video_filter(body.video_ids)
        active_definitions = [
            definition
            for definition in body.stages
            if any(str(value).strip() for value in definition.channels.model_dump().values())
        ]
        if not active_definitions:
            raise HTTPException(
                status_code=422,
                detail="at least one stage must have an enabled non-empty channel",
            )

        if body.view_mode == "all_hits":
            executed = _execute_independent_stage_search(
                query_id,
                active_definitions,
                query_action="independent_stage_search",
                candidate_top_k=_ALL_HITS_LIMIT,
                video_ids=video_ids,
                stage_image_paths=stage_image_paths,
            )
            stage_ids = executed["stage_ids"]
            all_hits_source_count = sum(
                len(executed["stage_results"].get(stage_id, [])) for stage_id in stage_ids
            )
            all_hits_raw = _flatten_all_hits(
                executed["stage_results"],
                stage_ids,
                limit=_ALL_HITS_LIMIT,
            )
            bundles = build_stage_bundles(
                executed["stage_results"],
                stage_ids=stage_ids,
                max_delta_ms=body.max_delta_ms if body.temporal_enabled else None,
                top_k=body.top_k,
                beam_width=_ALL_HITS_BUNDLE_BEAM_WIDTH,
                max_bundles_per_video=_ALL_HITS_MAX_BUNDLES_PER_VIDEO,
            )
            all_hits_eligible_raw = _flatten_complete_bundle_hits(bundles)
            all_hits = _diversify_all_hits(
                all_hits_eligible_raw,
                min_gap_ms=body.all_hits_min_gap_ms,
            )
            return {
                "query_id": query_id,
                "mode": "all_hits",
                "temporal_enabled": body.temporal_enabled,
                "max_delta_ms": body.max_delta_ms if body.temporal_enabled else None,
                "stage_ids": stage_ids,
                "active_stage_ids": stage_ids,
                "stage_results": executed["stage_results"],
                "stage_result_counts": executed["stage_result_counts"],
                "bundles": bundles,
                "bundle_count": len(bundles),
                "all_hits_limit": _ALL_HITS_LIMIT,
                "all_hits_min_gap_ms": body.all_hits_min_gap_ms,
                "all_hits_source_count": all_hits_source_count,
                "all_hits_raw_count": len(all_hits_raw),
                "all_hits_raw": all_hits_raw,
                "all_hits_eligible_raw_count": len(all_hits_eligible_raw),
                "all_hits_eligible_raw": all_hits_eligible_raw,
                "all_hits": all_hits,
                "fused_results": all_hits,
                "results": all_hits,
                "planner": {
                    "mode": "ordered_bundle_all_hits_v3",
                    "fusion_method": "harmonic-mean-minmax",
                    "temporal_enabled": body.temporal_enabled,
                    "max_delta_ms": body.max_delta_ms if body.temporal_enabled else None,
                    "requested_top_k": body.top_k,
                    "candidate_top_k": _ALL_HITS_LIMIT,
                    "candidate_rounds": 1,
                    "candidate_stop_reason": "raw_top_500",
                    "all_hits_min_gap_ms": body.all_hits_min_gap_ms,
                    "all_hits_contract": "complete_ordered_bundle_members_multiple_per_video",
                    "stages": executed["planner_stages"],
                },
                "stage_status": executed["stage_status"],
                "channel_status": executed["channel_status"],
                "channel_status_by_stage": executed["stage_status"],
                "executed_channels": executed["executed_channels"],
                "unavailable_channels": executed["unavailable_channels"],
                "execution_status": "ENGINEERING_PROXY",
                "quality_status": "UNVALIDATED",
            }

        candidate_schedule_index = 0
        candidate_top_k = _BUNDLE_CANDIDATE_SCHEDULE[candidate_schedule_index]
        candidate_rounds = 0
        previous_stage_result_counts: tuple[int, ...] | None = None
        candidate_stop_reason = "source_stalled"
        while True:
            candidate_rounds += 1
            executed = _execute_independent_stage_search(
                query_id,
                active_definitions,
                query_action="independent_stage_search",
                candidate_top_k=candidate_top_k,
                video_ids=video_ids,
                stage_image_paths=stage_image_paths,
            )
            stage_ids = executed["stage_ids"]
            bundles = build_stage_bundles(
                executed["stage_results"],
                stage_ids=stage_ids,
                max_delta_ms=body.max_delta_ms if body.temporal_enabled else None,
                top_k=body.top_k,
            )
            if len(bundles) >= body.top_k:
                candidate_stop_reason = "target_reached"
                break

            current_stage_result_counts = tuple(
                int(executed["stage_result_counts"].get(stage_id, 0))
                for stage_id in stage_ids
            )
            # If the provider returned no additional rows, more rounds cannot
            # produce another complete bundle. This is the exhaustion guard.
            if current_stage_result_counts == previous_stage_result_counts:
                candidate_stop_reason = "source_stalled"
                break
            previous_stage_result_counts = current_stage_result_counts
            if candidate_schedule_index + 1 >= len(_BUNDLE_CANDIDATE_SCHEDULE):
                candidate_stop_reason = "candidate_schedule_exhausted"
                break
            candidate_schedule_index += 1
            candidate_top_k = _BUNDLE_CANDIDATE_SCHEDULE[candidate_schedule_index]
        all_hits_source_count = sum(
            len(executed["stage_results"].get(stage_id, [])) for stage_id in stage_ids
        )
        all_hits_raw = _flatten_all_hits(
            executed["stage_results"],
            stage_ids,
            limit=_ALL_HITS_LIMIT,
        )
        all_hits = _diversify_all_hits(
            all_hits_raw,
            min_gap_ms=body.all_hits_min_gap_ms,
        )
        flat_results = [
            {
                **dict(stage_item),
                "bundle_id": bundle["bundle_id"],
                "bundle_rank": bundle["bundle_rank"],
                "bundle_score": float(bundle["bundle_score"]),
                "fusion_score": float(stage_item["stage_score"]),
            }
            for bundle in bundles
            for stage_item in bundle["stages"]
        ]
        return {
            "query_id": query_id,
            "mode": "bundle",
            "temporal_enabled": body.temporal_enabled,
            "max_delta_ms": body.max_delta_ms if body.temporal_enabled else None,
            "stage_ids": stage_ids,
            "active_stage_ids": stage_ids,
            "stage_results": executed["stage_results"],
            "stage_result_counts": executed["stage_result_counts"],
            "bundles": bundles,
            "bundle_count": len(bundles),
            "all_hits_limit": _ALL_HITS_LIMIT,
            "all_hits_min_gap_ms": body.all_hits_min_gap_ms,
            "all_hits_source_count": all_hits_source_count,
            "all_hits_raw_count": len(all_hits_raw),
            "all_hits_raw": all_hits_raw,
            "all_hits": all_hits,
            "fused_results": flat_results,
            "results": flat_results,
            "planner": {
                "mode": "independent_bundle_search_v1",
                "fusion_method": "harmonic-mean-minmax",
                "temporal_enabled": body.temporal_enabled,
                "max_delta_ms": body.max_delta_ms if body.temporal_enabled else None,
                "requested_bundle_top_k": body.top_k,
                "requested_top_k": body.top_k,
                "candidate_top_k": candidate_top_k,
                "candidate_schedule": list(_BUNDLE_CANDIDATE_SCHEDULE),
                "candidate_rounds": candidate_rounds,
                "candidate_stop_reason": candidate_stop_reason,
                "stages": executed["planner_stages"],
            },
            "stage_status": executed["stage_status"],
            "channel_status": executed["channel_status"],
            "channel_status_by_stage": executed["stage_status"],
            "executed_channels": executed["executed_channels"],
            "unavailable_channels": executed["unavailable_channels"],
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED",
        }

    @app.post("/search/stages")
    async def search_stages(request: Request) -> dict[str, Any]:
        body, image_paths = await _parse_stage_request(request, StagedSearchRequest)
        try:
            return _search_stages(body, image_paths)
        finally:
            _cleanup_query_images(image_paths)

    def _search_stages(
        body: StagedSearchRequest,
        stage_image_paths: Mapping[str, Path] | None = None,
    ) -> dict[str, Any]:
        query_id = body.query_id or f"stage-{uuid.uuid4().hex[:12]}"
        video_ids = _video_filter(body.video_ids)
        stage_results: dict[str, list[dict[str, Any]]] = {}
        stage_status: dict[str, dict[str, dict[str, Any]]] = {}
        unavailable_channels: dict[str, str] = {}
        executed_channels: list[str] = []
        planner_stages: dict[str, dict[str, Any]] = {}
        frame_status_cache: dict[str, dict[str, Any]] = {}
        video_status_cache: dict[str, dict[str, Any]] = {}
        for definition in body.stages:
            values = definition.channels.model_dump()
            breakdown = {
                channel: {
                    "enabled": bool(str(values[channel]).strip()),
                    "value": str(values[channel]),
                    "planned_value": str(values[channel]),
                    "action": (
                        "identity_passthrough"
                        if str(values[channel]).strip()
                        else "disabled_by_empty_input"
                    ),
                }
                for channel in _STAGE_CHANNELS
            }
            planner_stages[definition.stage_id] = {
                "original_query": dict(values),
                "planned_query": dict(values),
                "asr_mode": definition.asr_mode,
                "fusion_method": "harmonic-mean-minmax",
                "breakdown": breakdown,
            }
            found, statuses, _found_uids, stage_unavailable = _run_stage(
                query_id,
                definition,
                None,
                video_ids=video_ids,
                fusion_method="harmonic",
                frame_status_cache=frame_status_cache,
                video_status_cache=video_status_cache,
                stage_image_paths=stage_image_paths,
            )
            stage_results[definition.stage_id] = found
            stage_status[definition.stage_id] = statuses
            unavailable_channels.update(
                {
                    f"{definition.stage_id}:{name}": reason
                    for name, reason in stage_unavailable.items()
                }
            )
            for channel in ("siglip2", "qwen", "image", "ocr", "object", "asr"):
                if channel not in executed_channels and any(
                    item.get("executed_channels") and channel in item["executed_channels"]
                    for item in found
                ):
                    executed_channels.append(channel)
        # Staged Search intentionally runs every stage against the full
        # identity-preserving corpus.  Later stages are not refinements of an
        # S1 candidate pool; the UI groups the union by video and keeps the
        # stage owner on each frame for inspection and queue selection.
        stage_ids = [definition.stage_id for definition in body.stages]
        fused_results = _merge_independent_stage_results(stage_results, stage_ids)
        return {
            "query_id": query_id,
            "planner": {"mode": "independent_stage_search_v1", "stages": planner_stages},
            "stage_ids": stage_ids,
            "stage_results": stage_results,
            "stage_result_counts": {
                stage_id: len(stage_results[stage_id]) for stage_id in stage_ids
            },
            "fused_results": fused_results,
            "results": fused_results,
            "stage_status": stage_status,
            "channel_status": stage_status["S1"],
            "executed_channels": executed_channels,
            "unavailable_channels": unavailable_channels,
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED",
        }

    @app.post("/search/trake")
    async def search_trake(request: Request) -> dict[str, Any]:
        body, image_paths = await _parse_stage_request(request, TrakeSearchRequest)
        try:
            return _search_trake(body, image_paths)
        finally:
            _cleanup_query_images(image_paths)

    def _search_trake(
        body: TrakeSearchRequest,
        stage_image_paths: Mapping[str, Path] | None = None,
    ) -> dict[str, Any]:
        """Search every Trake stage independently, then link temporal tracks."""

        query_id = body.query_id or f"trake-{uuid.uuid4().hex[:12]}"
        video_ids = _video_filter(body.video_ids)
        stage_results: dict[str, list[dict[str, Any]]] = {}
        stage_status: dict[str, dict[str, dict[str, Any]]] = {}
        unavailable_channels: dict[str, str] = {}
        executed_channels: list[str] = []
        planner_stages: dict[str, dict[str, Any]] = {}
        frame_status_cache: dict[str, dict[str, Any]] = {}
        video_status_cache: dict[str, dict[str, Any]] = {}

        for definition in body.stages:
            values = definition.channels.model_dump()
            planner_stages[definition.stage_id] = {
                "original_query": dict(values),
                "planned_query": dict(values),
                "asr_mode": definition.asr_mode,
                "fusion_method": "harmonic-mean-minmax",
                "max_delta_ms": body.max_delta_ms,
                "breakdown": {
                    channel: {
                        "enabled": bool(str(values[channel]).strip()),
                        "value": str(values[channel]),
                        "action": (
                            "independent_stage_search"
                            if str(values[channel]).strip()
                            else "disabled_by_empty_input"
                        ),
                    }
                    for channel in _STAGE_CHANNELS
                },
            }
            found, statuses, _, stage_unavailable = _run_stage(
                query_id,
                definition,
                None,
                video_ids=video_ids,
                fusion_method="harmonic",
                frame_status_cache=frame_status_cache,
                video_status_cache=video_status_cache,
                stage_image_paths=stage_image_paths,
            )
            stage_results[definition.stage_id] = found
            stage_status[definition.stage_id] = statuses
            unavailable_channels.update(
                {
                    f"{definition.stage_id}:{name}": reason
                    for name, reason in stage_unavailable.items()
                }
            )
            for channel in ("siglip2", "qwen", "image", "ocr", "object", "asr"):
                if channel not in executed_channels and any(
                    item.get("executed_channels") and channel in item["executed_channels"]
                    for item in found
                ):
                    executed_channels.append(channel)

        tracks = build_trake_tracks(
            stage_results,
            stage_ids=[stage.stage_id for stage in body.stages],
            max_delta_ms=body.max_delta_ms,
            top_k=body.top_k,
        )
        fused_results = [
            {
                **dict(stage_item),
                "track_id": track["track_id"],
                "track_rank": track["track_rank"],
                "track_score": float(track["score"]),
            }
            for track in tracks
            for stage_item in track["stages"]
        ]
        return {
            "query_id": query_id,
            "mode": "trake",
            "stage_ids": [stage.stage_id for stage in body.stages],
            "max_delta_ms": body.max_delta_ms,
            "planner": {
                "mode": "independent_temporal_tracks_v1",
                "fusion_method": "harmonic-mean-minmax",
                "stages": planner_stages,
            },
            "stage_results": stage_results,
            "tracks": tracks,
            "fused_results": fused_results,
            "stage_status": stage_status,
            "channel_status": stage_status.get("S1", {}),
            "channel_status_by_stage": stage_status,
            "executed_channels": executed_channels,
            "unavailable_channels": unavailable_channels,
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED",
        }

    def _queue_item(body: ReviewQueueRequest) -> tuple[dict[str, Any], bool]:
        with app.state.review_queue_lock:
            try:
                row = service.get_frame(body.frame_uid)
            except KeyError as exc:
                raise HTTPException(
                    status_code=404, detail=f"Unknown frame_uid: {body.frame_uid}"
                ) from exc
            if str(row["video_id"]) != body.video_id:
                raise HTTPException(status_code=422, detail="video_id does not match frame_uid")
            if int(row["source_frame_idx"]) != body.source_frame_idx:
                raise HTTPException(
                    status_code=422,
                    detail="source_frame_idx does not match frame_uid",
                )
            if body.origin == "search_result" and int(row["timestamp_ms"]) != body.timestamp_ms:
                raise HTTPException(
                    status_code=422, detail="timestamp_ms does not match canonical frame"
                )
            if body.shot_id is not None and str(row.get("shot_id")) != body.shot_id:
                raise HTTPException(
                    status_code=422,
                    detail="shot_id does not match canonical frame",
                )
            key = (
                f"{body.query_id}|{body.submission_task or 'KIS'}|{body.chain_id or ''}|"
                f"{body.event_step if body.event_step is not None else ''}|"
                f"{body.stage_id}|{body.bundle_id or ''}|{body.frame_uid}"
            ).encode()
            queue_item_id = f"queue_{hashlib.sha256(key).hexdigest()[:16]}"
            existing = app.state.review_queue.get(queue_item_id)
            if existing is not None:
                return _queue_item_payload(existing), True
            if body.submission_task == "TRAKE":
                chain_items = [
                    candidate
                    for candidate in app.state.review_queue.values()
                    if candidate.get("query_id") == body.query_id
                    and str(candidate.get("submission_task") or "").upper() == "TRAKE"
                    and str(candidate.get("chain_id") or "").strip() == body.chain_id
                ]
                if any(
                    str(candidate.get("video_id")) != body.video_id
                    for candidate in chain_items
                ):
                    raise HTTPException(
                        status_code=422,
                        detail=f"TRAKE chain {body.chain_id!r} must stay on one video",
                    )
                existing_steps = [int(candidate["event_step"]) for candidate in chain_items]
                expected_step = len(existing_steps)
                if (
                    sorted(existing_steps) != list(range(expected_step))
                    or body.event_step != expected_step
                ):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"TRAKE chain {body.chain_id!r} requires contiguous zero-based "
                            f"event_step values; next event_step={expected_step}"
                        ),
                    )
                existing_stage_numbers = [
                    int(str(candidate["stage_id"])[1:]) for candidate in chain_items
                ]
                current_stage_number = int(body.stage_id[1:])
                if existing_stage_numbers and (
                    current_stage_number <= max(existing_stage_numbers)
                    or current_stage_number in existing_stage_numbers
                ):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"TRAKE chain {body.chain_id!r} physical stage_id values must be "
                            "strictly ordered and unique"
                        ),
                    )
            item = body.model_dump(mode="json")
            item.update(
                {
                    "queue_item_id": queue_item_id,
                    "created_at": datetime.now(UTC).isoformat(),
                    "queue_position": len(_ordered_queue_items(body.query_id)),
                    "mapping_status": body.mapping_status or "RESOLVED_EXACT",
                    "provenance_status": body.provenance_status or "ENGINEERING_PROXY",
                }
            )
            item["queue_group_id"] = _queue_group_id(item)
            app.state.review_queue[queue_item_id] = item
            return _queue_item_payload(item), False

    @app.post("/review/queue")
    def add_review_queue_item(body: ReviewQueueRequest) -> dict[str, Any]:
        item, duplicate = _queue_item(body)
        return {
            "item": item,
            "duplicate": duplicate,
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED",
        }

    @app.get("/review/queue")
    def list_review_queue(query_id: str | None = Query(default=None)) -> dict[str, Any]:
        items = _queue_snapshot(query_id)
        return {"items": items, "count": len(items), "quality_status": "UNVALIDATED"}

    @app.put("/review/queue/reorder")
    @app.post("/review/queue/reorder")
    def bulk_reorder_review_queue(body: ReviewQueueReorderRequest) -> dict[str, Any]:
        items = _bulk_reorder_queue(body)
        return {
            "query_id": body.query_id,
            "items": items,
            "count": len(items),
            "quality_status": "UNVALIDATED",
        }

    @app.patch("/review/queue/{queue_item_id}")
    def patch_review_queue_item(queue_item_id: str, body: ReviewQueuePatch) -> dict[str, Any]:
        with app.state.review_queue_lock:
            item = app.state.review_queue.get(queue_item_id)
            if item is None:
                raise HTTPException(status_code=404, detail="Unknown queue_item_id")
            if (
                body.qa_answer is not None
                and item.get("submission_task") == "QA"
                and not body.qa_answer.strip()
            ):
                raise HTTPException(
                    status_code=422,
                    detail="QA queue items require a non-whitespace qa_answer",
                )
            if body.selection_reason is not None:
                item["selection_reason"] = body.selection_reason
            if body.qa_answer is not None:
                item["qa_answer"] = body.qa_answer
            if "bundle_temporal_enabled" in body.model_fields_set:
                item["bundle_temporal_enabled"] = body.bundle_temporal_enabled
            if body.queue_position is not None:
                item = _reorder_queue_item(queue_item_id, body.queue_position)
            return {"item": _queue_item_payload(item), "quality_status": "UNVALIDATED"}

    @app.delete("/review/queue/{queue_item_id}")
    def delete_review_queue_item(queue_item_id: str) -> dict[str, Any]:
        with app.state.review_queue_lock:
            if app.state.review_queue.pop(queue_item_id, None) is None:
                raise HTTPException(status_code=404, detail="Unknown queue_item_id")
            return {"queue_item_id": queue_item_id, "deleted": True}

    @app.get("/review/queue/export")
    def export_review_queue(format: Literal["json", "jsonl"] = Query(default="json")) -> Response:
        items = _queue_snapshot()
        if format == "jsonl":
            content = "".join(
                json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items
            )
            return Response(content=content, media_type="application/jsonl")
        return JSONResponse(
            content={"items": items, "count": len(items), "quality_status": "UNVALIDATED"}
        )

    @app.post("/search/image")
    async def search_image(
        request: Request,
        top_k: int = Query(default=100, ge=1, le=500),
        query_id: str | None = Query(default=None),
        visual_indexes: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_query_id = query_id or f"dual-{uuid.uuid4().hex[:12]}"
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        upload_content_type = content_type
        if content_type == "multipart/form-data":
            try:
                form = await request.form()
            except AssertionError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="MULTIPART_RUNTIME_UNAVAILABLE",
                ) from exc
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(status_code=400, detail="multipart field 'file' is required")
            resolved_query_id = str(form.get("query_id") or resolved_query_id)
            upload_content_type = str(getattr(upload, "content_type", ""))
            payload = await upload.read()
        else:
            if content_type not in {"image/jpeg", "image/png"}:
                raise HTTPException(
                    status_code=415,
                    detail="content-type must be image/jpeg or image/png",
                )
            payload = await request.body()
        temp_path = _validated_query_image(payload, upload_content_type)
        try:
            results = service.search_image(
                resolved_query_id,
                temp_path,
                top_k=top_k,
                visual_indexes=visual_indexes,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)
        payload = {
            "query_id": resolved_query_id,
            "task": "VKIS",
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED",
            "results": _result_payloads(results, service),
        }
        payload.update(_channel_payload(service, results, visual_indexes))
        return payload

    @app.post("/v1/kis/search")
    def bfe_search(body: BFEKISSearchRequest) -> dict[str, Any]:
        query_id = body.query_id or f"dual-{uuid.uuid4().hex[:12]}"
        try:
            results = service.search_text(
                query_id,
                body.text,
                top_k=body.top_k,
                ocr_query=body.ocr_query,
                object_query=body.object_query,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return serialize_kis_response(query_id, "TKIS", results, service)

    @app.post("/v1/kis/search/image")
    async def bfe_search_image(request: Request) -> dict[str, Any]:
        """Accept BFE's multipart file upload and the raw-image fallback."""

        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        query_id = request.query_params.get("query_id")
        top_k_raw = request.query_params.get("top_k", "100")
        payload: bytes
        suffix: str
        upload_content_type = content_type
        if content_type == "multipart/form-data":
            try:
                form = await request.form()
            except AssertionError as exc:  # python-multipart missing/misconfigured
                raise HTTPException(
                    status_code=503, detail="MULTIPART_RUNTIME_UNAVAILABLE"
                ) from exc
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(status_code=400, detail="multipart field 'file' is required")
            query_id = str(form.get("query_id") or query_id or "")
            top_k_raw = str(form.get("top_k") or top_k_raw)
            upload_content_type = str(getattr(upload, "content_type", "") or "").lower()
            if upload_content_type not in {"image/jpeg", "image/png"}:
                raise HTTPException(status_code=415, detail="file must be image/jpeg or image/png")
            payload = await upload.read()
        else:
            if content_type not in {"image/jpeg", "image/png"}:
                raise HTTPException(
                    status_code=415,
                    detail="content-type must be multipart/form-data, image/jpeg or image/png",
                )
            payload = await request.body()
        if not payload:
            raise HTTPException(status_code=400, detail="request body must contain image bytes")
        if len(payload) > _MAX_BFE_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="UPLOAD_TOO_LARGE")
        try:
            top_k = int(top_k_raw)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="top_k must be an integer") from exc
        if top_k < 1 or top_k > 500:
            raise HTTPException(status_code=422, detail="top_k must be in [1, 500]")
        query_id = query_id or f"dual-{uuid.uuid4().hex[:12]}"
        suffix = ".png" if upload_content_type == "image/png" else ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
        try:
            results = service.search_image(query_id, temp_path, top_k=top_k)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temp_path.unlink(missing_ok=True)
        return serialize_kis_response(query_id, "VKIS", results, service)

    @app.get("/frames/{frame_uid}")
    def get_frame(frame_uid: str, window: int = Query(default=5, ge=0, le=50)) -> dict[str, Any]:
        try:
            row = service.get_frame(frame_uid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_uid: {frame_uid}") from exc
        neighbors = service.timeline_window(str(row["video_id"]), frame_uid, window)
        video_status = service.video_media_status(str(row["video_id"]))
        image_status = service.frame_image_status(frame_uid)
        return {
            "frame": {
                **row,
                "image_url": f"/frames/{frame_uid}/image",
                "thumbnail_url": f"/frames/{frame_uid}/thumbnail",
                **image_status,
            },
            "neighbors": [
                {
                    **item,
                    "image_url": f"/frames/{item['frame_uid']}/image",
                    "thumbnail_url": f"/frames/{item['frame_uid']}/thumbnail",
                    **service.frame_image_status(str(item["frame_uid"])),
                }
                for item in neighbors
            ],
            "image_url": f"/frames/{frame_uid}/image",
            "thumbnail_url": f"/frames/{frame_uid}/thumbnail",
            **image_status,
            "video_url": f"/videos/{row['video_id']}/stream",
            "video_available": bool(video_status["available"]),
            "video_status": str(video_status["status"]),
            "video_stream_available": bool(
                video_status.get("stream_available", video_status["available"])
            ),
            "video_stream_status": str(video_status.get("stream_status", video_status["status"])),
            "video_stream_reason": video_status.get("stream_reason"),
            "quality_status": "UNVALIDATED",
        }

    @app.get("/v1/frames/{frame_uid}")
    def bfe_get_frame(frame_uid: str) -> dict[str, Any]:
        try:
            row = service.get_frame(frame_uid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_uid: {frame_uid}") from exc
        video_status = service.video_media_status(str(row["video_id"]))
        image_status = service.frame_image_status(frame_uid)
        return {
            **serialize_catalog_frame(row),
            "frame": serialize_catalog_frame(row),
            "frame_uid": frame_uid,
            "image_url": frame_media_url(frame_uid),
            **image_status,
            "video_url": video_media_url(str(row["video_id"])),
            "video_available": bool(video_status["available"]),
            "video_status": str(video_status["status"]),
            "video_stream_available": bool(
                video_status.get("stream_available", video_status["available"])
            ),
            "video_stream_status": str(video_status.get("stream_status", video_status["status"])),
            "video_stream_reason": video_status.get("stream_reason"),
            "quality_status": "UNVALIDATED_ON_HCMAIC",
        }

    @app.get("/frames/{frame_uid}/image")
    def get_frame_image(frame_uid: str) -> Response:
        try:
            path = service.frame_image_path(frame_uid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_uid: {frame_uid}") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MediaResolutionError as exc:
            reason = str(exc)
            status = (
                "UNAVAILABLE_DEPENDENCY_MISSING"
                if reason.startswith("UNAVAILABLE_DEPENDENCY_MISSING")
                else "REMOTE_MEDIA_UNAVAILABLE"
            )
            return JSONResponse(
                status_code=503 if status == "UNAVAILABLE_DEPENDENCY_MISSING" else 502,
                content={"frame_uid": frame_uid, "status": status, "reason": reason},
            )
        return FileResponse(
            path,
            media_type=_MEDIA_TYPES.get(path.suffix.lower()),
            headers={"Cache-Control": _MEDIA_CACHE_CONTROL},
        )

    @app.get("/v1/frames/{frame_uid}/image")
    def bfe_get_frame_image(frame_uid: str) -> Response:
        try:
            path = service.frame_image_path(frame_uid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_uid: {frame_uid}") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MediaResolutionError as exc:
            reason = str(exc)
            status = (
                "UNAVAILABLE_DEPENDENCY_MISSING"
                if reason.startswith("UNAVAILABLE_DEPENDENCY_MISSING")
                else "REMOTE_MEDIA_UNAVAILABLE"
            )
            return JSONResponse(
                status_code=503 if status == "UNAVAILABLE_DEPENDENCY_MISSING" else 502,
                content={"frame_uid": frame_uid, "status": status, "reason": reason},
            )
        return FileResponse(
            path,
            media_type=_MEDIA_TYPES.get(path.suffix.lower()),
            headers={"Cache-Control": _MEDIA_CACHE_CONTROL},
        )

    def _frame_thumbnail_response(
        request: Request,
        frame_uid: str,
        width: int,
        quality: int,
    ) -> Response:
        try:
            path = service.frame_thumbnail_path(frame_uid, width=width, quality=quality)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_uid: {frame_uid}") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ImageThumbnailError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except MediaResolutionError as exc:
            reason = str(exc)
            status = (
                "UNAVAILABLE_DEPENDENCY_MISSING"
                if reason.startswith("UNAVAILABLE_DEPENDENCY_MISSING")
                else "REMOTE_MEDIA_UNAVAILABLE"
            )
            return JSONResponse(
                status_code=503 if status == "UNAVAILABLE_DEPENDENCY_MISSING" else 502,
                content={"frame_uid": frame_uid, "status": status, "reason": reason},
            )
        etag = f'"{path.stem}"'
        if_none_match = request.headers.get("if-none-match", "")
        if if_none_match == "*" or etag in {
            value.strip().removeprefix("W/") for value in if_none_match.split(",")
        }:
            return Response(
                status_code=304,
                headers={
                    "Cache-Control": _THUMBNAIL_CACHE_CONTROL,
                    "ETag": etag,
                },
            )
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={
                "Cache-Control": _THUMBNAIL_CACHE_CONTROL,
                "X-HCMAIC-Image-Variant": f"thumbnail-{width}",
                "ETag": etag,
            },
        )

    @app.get("/frames/{frame_uid}/thumbnail")
    def get_frame_thumbnail(
        request: Request,
        frame_uid: str,
        width: int = Query(default=DEFAULT_IMAGE_THUMBNAIL_WIDTH, ge=160, le=640),
        quality: int = Query(default=DEFAULT_IMAGE_THUMBNAIL_QUALITY, ge=40, le=95),
    ) -> Response:
        return _frame_thumbnail_response(request, frame_uid, width, quality)

    @app.get("/v1/frames/{frame_uid}/thumbnail")
    def bfe_get_frame_thumbnail(
        request: Request,
        frame_uid: str,
        width: int = Query(default=DEFAULT_IMAGE_THUMBNAIL_WIDTH, ge=160, le=640),
        quality: int = Query(default=DEFAULT_IMAGE_THUMBNAIL_QUALITY, ge=40, le=95),
    ) -> Response:
        return _frame_thumbnail_response(request, frame_uid, width, quality)

    @app.get("/videos/{video_id}/first-keyframe")
    def get_first_keyframe(video_id: str) -> dict[str, Any]:
        """Return one catalog-backed keyframe for direct video browsing.

        This is metadata-only and deliberately separate from retrieval.  The
        service keeps each video's catalog rows canonically ordered, so the
        response is deterministic without loading a full timeline into the
        HTTP response or invoking any channel/model.
        """

        try:
            row = service.first_keyframe(video_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}") from exc
        frame_uid = str(row["frame_uid"])
        image_status = service.frame_image_status(frame_uid)
        video_status = service.video_media_status(video_id)
        frame = {
            **row,
            "image_url": f"/frames/{frame_uid}/image",
            "thumbnail_url": f"/frames/{frame_uid}/thumbnail",
            **image_status,
        }
        return {
            "video_id": video_id,
            "frame_uid": frame_uid,
            "source_frame_idx": int(row["source_frame_idx"]),
            "timestamp_ms": int(row["timestamp_ms"]),
            "frame": frame,
            "image_url": frame["image_url"],
            "thumbnail_url": frame["thumbnail_url"],
            **image_status,
            "video_url": f"/videos/{video_id}/stream",
            "video_available": bool(video_status["available"]),
            "video_status": str(video_status["status"]),
            "video_stream_available": bool(
                video_status.get("stream_available", video_status["available"])
            ),
            "video_stream_status": str(video_status.get("stream_status", video_status["status"])),
            "video_stream_reason": video_status.get("stream_reason"),
            "quality_status": "UNVALIDATED",
        }

    @app.get("/videos/{video_id}/timeline")
    def get_timeline(video_id: str) -> dict[str, Any]:
        try:
            rows = service.timeline(video_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}") from exc
        return {
            "video_id": video_id,
            "n_frames": len(rows),
            **service.video_media_status(video_id),
            "quality_status": "UNVALIDATED",
            "frames": [{**row, "image_url": f"/frames/{row['frame_uid']}/image"} for row in rows],
        }

    @app.get("/v1/videos/{video_id}/timeline")
    def bfe_get_timeline(video_id: str) -> dict[str, Any]:
        temporal = app.state.temporal_selection
        if temporal is not None:
            try:
                rows = temporal.timeline(video_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=404, detail=f"Unknown video_id: {video_id}"
                ) from exc
            return {
                "video_id": video_id,
                "n_frames": len(rows),
                **temporal.availability(video_id),
                "quality_status": "UNVALIDATED",
                "frames": rows,
            }
        try:
            rows = service.timeline(video_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}") from exc
        return {
            "video_id": video_id,
            "n_frames": len(rows),
            **service.video_media_status(video_id),
            "quality_status": "UNVALIDATED_ON_HCMAIC",
            "frames": [serialize_catalog_frame(row) for row in rows],
        }

    @app.get("/v1/videos/{video_id}/availability")
    def video_availability(video_id: str) -> dict[str, Any]:
        temporal = app.state.temporal_selection
        if temporal is None:
            status = service.video_media_status(video_id)
            return {
                **status,
                "video_id": video_id,
                "source_available": bool(status["available"]),
                "stream_available": bool(status.get("stream_available", False)),
                "stream_status": str(status.get("stream_status", status["status"])),
                "pts_available": False,
                "preview_only": True,
                "status": "PREVIEW_TIMESTAMP_ONLY",
                "quality_status": "UNVALIDATED",
            }
        try:
            temporal_status = temporal.availability(video_id)
            media_status = service.video_media_status(video_id)
            return {
                **temporal_status,
                **{
                    key: media_status.get(key)
                    for key in (
                        "backend",
                        "bytes",
                        "range_capable",
                        "provenance_status",
                        "sha256_status",
                        "source_path",
                        "member_path",
                        "media_info_id",
                        "dataset_id",
                        "range_probe_status",
                        "range_probe_attempts",
                        "source_manifest_id",
                        "source_fingerprint",
                        "remote_content_fingerprint",
                        "join_method",
                        "stream_available",
                        "stream_status",
                    )
                },
                "media_status": media_status.get("status"),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}") from exc

    def _resolve_video_reference(
        video_id: str,
        frame_uid: str | None,
        source_frame_idx: int | None,
        timestamp_ms: int | None,
    ) -> dict[str, Any]:
        temporal = app.state.temporal_selection
        catalog_backed = True
        if timestamp_ms is not None and temporal is not None:
            try:
                pts_rows = temporal.timeline(video_id)
            except KeyError:
                pts_rows = []
            if pts_rows:
                selected = min(
                    pts_rows,
                    key=lambda row: (
                        abs(int(row["resolved_timestamp_ms"]) - timestamp_ms),
                        int(row["resolved_timestamp_ms"]),
                        int(row["source_frame_idx"]),
                        str(row["resolved_frame_uid"]),
                    ),
                )
                try:
                    resolved = service.resolve_frame_reference(
                        video_id, frame_uid=str(selected["resolved_frame_uid"])
                    )
                except KeyError:
                    # PTS is the canonical source timeline and can contain a
                    # presentation frame absent from the sparse retrieval
                    # catalog.  Keep that immutable identity instead of
                    # converting an exact-extraction boundary into a 500.
                    catalog_backed = False
                    resolved = {
                        "video_id": video_id,
                        "frame_uid": str(selected["resolved_frame_uid"]),
                        "resolved_frame_uid": str(selected["resolved_frame_uid"]),
                        "source_frame_idx": int(selected["source_frame_idx"]),
                        "timestamp_ms": int(selected["resolved_timestamp_ms"]),
                        "shot_id": None,
                        "keyframe_path": None,
                    }
                resolved.update(
                    {
                        "selected_time_ms": timestamp_ms,
                        "delta_ms": abs(int(selected["resolved_timestamp_ms"]) - timestamp_ms),
                        "resolved_timestamp_ms": int(selected["resolved_timestamp_ms"]),
                        "mapping_mode": "nearest_pts",
                        "mapping_method": "PTS_NEAREST_PRESENTATION_ORDER",
                        "mapping_status": "RESOLVED_CANONICAL",
                    }
                )
            else:
                resolved = service.resolve_frame_reference(
                    video_id,
                    frame_uid=frame_uid,
                    source_frame_idx=source_frame_idx,
                    timestamp_ms=timestamp_ms,
                )
        else:
            resolved = service.resolve_frame_reference(
                video_id,
                frame_uid=frame_uid,
                source_frame_idx=source_frame_idx,
                timestamp_ms=timestamp_ms,
            )
        media_status = service.video_media_status(video_id)
        resolved.update(
            {
                "image_url": (
                    f"/frames/{resolved['frame_uid']}/image" if catalog_backed else None
                ),
                "video_url": f"/videos/{video_id}/stream",
                "video_available": bool(media_status["available"]),
                "video_stream_available": bool(media_status.get("stream_available", False)),
                "video_stream_status": media_status.get("stream_status"),
                "backend": media_status.get("backend"),
                "bytes": media_status.get("bytes"),
                "range_capable": media_status.get("range_capable", False),
                "provenance_status": media_status.get("provenance_status"),
                "sha256_status": media_status.get("sha256_status"),
                "source_path": media_status.get("source_path"),
                "member_path": media_status.get("member_path"),
                "media_info_id": media_status.get("media_info_id"),
                "dataset_id": media_status.get("dataset_id"),
                "range_probe_status": media_status.get("range_probe_status"),
                "range_probe_attempts": media_status.get("range_probe_attempts"),
                "source_manifest_id": media_status.get("source_manifest_id"),
                "source_fingerprint": media_status.get("source_fingerprint"),
                "remote_content_fingerprint": media_status.get("remote_content_fingerprint"),
                "join_method": media_status.get("join_method"),
                "execution_status": "ENGINEERING_PROXY",
                "quality_status": "UNVALIDATED",
            }
        )
        return resolved

    @app.get("/videos/{video_id}/resolve")
    @app.get("/v1/videos/{video_id}/resolve")
    def resolve_video_reference(
        video_id: str,
        frame_uid: str | None = Query(default=None),
        source_frame_idx: int | None = Query(default=None, ge=0),
        timestamp_ms: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        try:
            return _resolve_video_reference(video_id, frame_uid, source_frame_idx, timestamp_ms)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown media reference: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/videos/{video_id}/raw-frame")
    @app.get("/v1/videos/{video_id}/raw-frame")
    def raw_video_frame(
        video_id: str,
        timestamp_ms: int = Query(..., ge=0),
        thumbnail_width: int = Query(default=DEFAULT_THUMBNAIL_WIDTH, ge=160, le=640),
    ) -> dict[str, Any]:
        """Decode one raw frame and return a verified source-frame identity.

        A materialized canonical PTS timeline remains the preferred path.  If
        it is absent, the endpoint resolves only this requested video into the
        verified media cache and performs a bounded presentation-order decode;
        it never derives an index from FPS or rebuilds a corpus artifact.
        """

        if not service.has_video_id(video_id):
            raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}")
        temporal = app.state.temporal_selection
        pts_rows: list[dict[str, Any]] = []
        if temporal is not None:
            try:
                pts_rows = temporal.timeline(video_id)
            except KeyError:
                # The catalog may know a video before the optional temporal
                # artifact does.  That is an on-demand fallback, not an
                # unknown video identity.
                pts_rows = []

        if pts_rows:
            timestamps = [int(row["resolved_timestamp_ms"]) for row in pts_rows]
            lower_bound_ms = min(timestamps)
            upper_bound_ms = max(timestamps)
            target_timestamp_ms = min(max(timestamp_ms, lower_bound_ms), upper_bound_ms)

            try:
                local_path = service.local_video_path(video_id)
            except (KeyError, PermissionError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

            if local_path is not None:
                try:
                    decoded = decode_video_frame(
                        local_path,
                        target_timestamp_ms,
                        thumbnail_width=thumbnail_width,
                        jpeg_quality=DEFAULT_JPEG_QUALITY,
                    )
                except VideoFrameDecodeError as exc:
                    raise HTTPException(status_code=502, detail=str(exc)) from exc
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                decode_status = "LOCAL_RAW_DECODE"
                mapping_method = "CANONICAL_PTS_NEAREST_DECODED_TIMESTAMP"
                extraction_mode = "LOCAL_VIDEO"
                remote_url_configured = False
                remote_fast_path_error = None
            else:
                # A canonical PTS timeline does not imply that the MP4 is
                # local.  The pinned HF URL is still an authoritative,
                # bounded decode source; map its decoded timestamp back to
                # the canonical PTS row instead of rejecting the request.
                try:
                    remote_url = service.remote_video_url(video_id)
                except (FileNotFoundError, KeyError):
                    remote_url = None
                if not remote_url:
                    raise HTTPException(
                        status_code=503,
                        detail="RAW_DECODE_LOCAL_VIDEO_UNAVAILABLE",
                    )
                try:
                    decoded = decode_exact_video_frame_url(
                        remote_url,
                        target_timestamp_ms,
                        thumbnail_width=thumbnail_width,
                        jpeg_quality=DEFAULT_JPEG_QUALITY,
                    )
                except VideoFrameDecodeError as exc:
                    raise HTTPException(
                        status_code=502,
                        detail=f"RAW_DECODE_REMOTE_URL_FAILED: {exc}",
                    ) from exc
                decode_status = "REMOTE_URL_RAW_DECODE"
                mapping_method = "REMOTE_URL_DECODER_CANONICAL_PTS"
                extraction_mode = "REMOTE_URL_FFMPEG"
                remote_url_configured = True
                remote_fast_path_error = None

            mapping_timestamp_ms = int(decoded["decoded_timestamp_ms"])
            resolved = _resolve_video_reference(
                video_id,
                frame_uid=None,
                source_frame_idx=None,
                timestamp_ms=mapping_timestamp_ms,
            )
            was_clamped = timestamp_ms != target_timestamp_ms
            decoded_mapping_delta_ms = abs(
                int(resolved["resolved_timestamp_ms"]) - mapping_timestamp_ms
            )
            raw_verification = (
                "PASS"
                if (
                    not was_clamped
                    and bool(decoded.get("target_reached", True))
                    and resolved["mapping_status"] == "RESOLVED_CANONICAL"
                )
                else "BLOCKED"
            )
            return {
                **decoded,
                "video_id": video_id,
                "requested_timestamp_ms": timestamp_ms,
                "target_timestamp_ms": target_timestamp_ms,
                "frame_uid": resolved["frame_uid"],
                "source_frame_idx": resolved["source_frame_idx"],
                "resolved_timestamp_ms": resolved["resolved_timestamp_ms"],
                "mapping_timestamp_ms": mapping_timestamp_ms,
                "mapping_delta_ms": abs(
                    int(resolved["resolved_timestamp_ms"]) - int(timestamp_ms)
                ),
                "decoded_mapping_delta_ms": decoded_mapping_delta_ms,
                "mapping_status": resolved["mapping_status"],
                "decode_status": decode_status,
                "extraction_mode": extraction_mode,
                "mapping_method": mapping_method,
                "remote_fast_path": "USED" if remote_url_configured else "NOT_CONFIGURED",
                "remote_fast_path_error": remote_fast_path_error,
                "raw_verification": raw_verification,
                "was_clamped": was_clamped,
                "execution_status": "ENGINEERING_PROXY",
                "quality_status": "UNVALIDATED",
            }

        # No PTS rows: prefer the pinned remote URL so FFmpeg can seek without
        # materializing the whole MP4.  A verified local/cache decode remains
        # the fallback when the remote backend cannot seek safely.
        decoded: dict[str, Any] | None = None
        decode_status = "LOCAL_RAW_DECODE_ON_DEMAND"
        mapping_method = "ON_DEMAND_DECODER_PRESENTATION_PTS"
        extraction_mode = "LOCAL_CACHE_SEQUENTIAL"
        remote_url_configured = False
        remote_fast_path_error: str | None = None
        try:
            remote_url = service.remote_video_url(video_id)
        except (FileNotFoundError, KeyError):
            remote_url = None
        if remote_url:
            remote_url_configured = True
            try:
                decoded = decode_exact_video_frame_url(
                    remote_url,
                    timestamp_ms,
                    thumbnail_width=thumbnail_width,
                    jpeg_quality=DEFAULT_JPEG_QUALITY,
                )
                decode_status = "REMOTE_URL_RAW_DECODE"
                mapping_method = "REMOTE_URL_DECODER_PRESENTATION_PTS"
                extraction_mode = "REMOTE_URL_FFMPEG"
            except VideoFrameDecodeError as exc:
                remote_fast_path_error = str(exc)

        if decoded is None:
            try:
                local_path = service.local_video_path(video_id)
                if local_path is None:
                    local_path = service.video_path(video_id)
            except (KeyError, PermissionError) as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except FileNotFoundError as exc:
                raise HTTPException(
                    status_code=503, detail="RAW_DECODE_MEDIA_UNAVAILABLE"
                ) from exc
            except MediaResolutionError as exc:
                raise HTTPException(
                    status_code=503, detail=f"RAW_DECODE_MEDIA_UNAVAILABLE: {exc}"
                ) from exc

            try:
                decoded = decode_exact_video_frame(
                    local_path,
                    timestamp_ms,
                    thumbnail_width=thumbnail_width,
                    jpeg_quality=DEFAULT_JPEG_QUALITY,
                )
            except VideoFrameDecodeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not bool(decoded.get("target_reached")):
            raise HTTPException(status_code=422, detail="EXACT_SOURCE_FRAME_OUT_OF_RANGE")
        try:
            source_frame_idx = int(decoded["source_frame_idx"])
            mapping_timestamp_ms = int(decoded["decoded_timestamp_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502, detail="EXACT_SOURCE_FRAME_IDENTITY_UNAVAILABLE"
            ) from exc
        if source_frame_idx < 0 or mapping_timestamp_ms < 0:
            raise HTTPException(status_code=502, detail="EXACT_SOURCE_FRAME_IDENTITY_INVALID")

        frame_uid = f"{video_id}:{source_frame_idx}"
        catalog_frame: dict[str, Any] | None = None
        try:
            catalog_frame = service.get_frame(frame_uid)
        except KeyError:
            # A raw presentation frame need not be one of the sparse indexed
            # keyframes.  It still has a valid immutable video_id:idx identity.
            catalog_frame = None
        media_status = service.video_media_status(video_id)
        return {
            **decoded,
            "video_id": video_id,
            "requested_timestamp_ms": timestamp_ms,
            "target_timestamp_ms": timestamp_ms,
            "frame_uid": frame_uid,
            "resolved_frame_uid": frame_uid,
            "source_frame_idx": source_frame_idx,
            "selected_time_ms": timestamp_ms,
            "resolved_timestamp_ms": mapping_timestamp_ms,
            "timestamp_ms": mapping_timestamp_ms,
            "mapping_timestamp_ms": mapping_timestamp_ms,
            "mapping_delta_ms": abs(mapping_timestamp_ms - int(timestamp_ms)),
            "decoded_mapping_delta_ms": 0,
            "mapping_mode": "exact_decoder_timestamp",
            "mapping_status": "RESOLVED_EXACT",
            "mapping_method": mapping_method,
            "decode_status": decode_status,
            "extraction_mode": extraction_mode,
            "remote_fast_path": (
                "USED"
                if remote_url_configured and decode_status == "REMOTE_URL_RAW_DECODE"
                else "FALLBACK"
                if remote_url_configured
                else "NOT_CONFIGURED"
            ),
            "remote_fast_path_error": remote_fast_path_error,
            "raw_verification": "PASS",
            "was_clamped": False,
            "image_url": f"/frames/{frame_uid}/image" if catalog_frame is not None else None,
            "video_url": f"/videos/{video_id}/stream",
            "video_available": bool(media_status.get("available")),
            "video_stream_available": bool(media_status.get("stream_available", False)),
            "video_stream_status": media_status.get("stream_status"),
            "backend": media_status.get("backend"),
            "bytes": media_status.get("bytes"),
            "range_capable": media_status.get("range_capable", False),
            "provenance_status": media_status.get("provenance_status"),
            "sha256_status": media_status.get("sha256_status"),
            "source_path": media_status.get("source_path"),
            "member_path": media_status.get("member_path"),
            "media_info_id": media_status.get("media_info_id"),
            "dataset_id": media_status.get("dataset_id"),
            "range_probe_status": media_status.get("range_probe_status"),
            "range_probe_attempts": media_status.get("range_probe_attempts"),
            "source_manifest_id": media_status.get("source_manifest_id"),
            "source_fingerprint": media_status.get("source_fingerprint"),
            "remote_content_fingerprint": media_status.get("remote_content_fingerprint"),
            "join_method": media_status.get("join_method"),
            "shot_id": catalog_frame.get("shot_id") if catalog_frame is not None else None,
            "keyframe_path": catalog_frame.get("keyframe_path")
            if catalog_frame is not None
            else None,
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED",
        }

    @app.get("/videos/{video_id}/stream")
    @app.get("/v1/videos/{video_id}/stream")
    def stream_video(video_id: str, request: Request) -> Response:
        """Serve one bounded range from local or allowlisted remote video media."""

        if not service.has_video_id(video_id):
            return JSONResponse(
                status_code=404,
                content={"video_id": video_id, "status": "VIDEO_NOT_FOUND"},
            )
        try:
            local_path = service.local_video_path(video_id)
        except PermissionError as exc:
            return JSONResponse(
                status_code=403,
                content={
                    "video_id": video_id,
                    "status": "VIDEO_PATH_FORBIDDEN",
                    "reason": str(exc),
                },
            )
        if local_path is not None:
            return FileResponse(
                local_path,
                media_type=_MEDIA_TYPES.get(local_path.suffix.lower(), "video/mp4"),
                headers={
                    "Accept-Ranges": "bytes",
                    "Cache-Control": _MEDIA_CACHE_CONTROL,
                },
            )
        try:
            fetched = service.stream_video_range(video_id, request.headers.get("range"))
        except MediaRangeRequestError as exc:
            headers = {"Accept-Ranges": "bytes"}
            if exc.total is not None:
                headers["Content-Range"] = f"bytes */{exc.total}"
            return JSONResponse(
                status_code=416,
                content={"video_id": video_id, "status": exc.code},
                headers=headers,
            )
        except MediaRangeUnsupportedError as exc:
            return JSONResponse(
                status_code=501,
                content={"video_id": video_id, "status": exc.code},
            )
        except FileNotFoundError:
            return JSONResponse(
                status_code=404,
                content={"video_id": video_id, "status": "VIDEO_STREAM_UNAVAILABLE"},
            )
        except MediaResolutionError:
            return JSONResponse(
                status_code=502,
                content={"video_id": video_id, "status": "REMOTE_MEDIA_UNAVAILABLE"},
            )
        response_headers = dict(fetched.headers)
        response_headers.setdefault("Cache-Control", _MEDIA_CACHE_CONTROL)
        return Response(content=fetched.body, status_code=206, headers=response_headers)

    def _require_temporal() -> TemporalSelectionService:
        temporal = app.state.temporal_selection
        if temporal is None:
            raise HTTPException(status_code=503, detail="TEMPORAL_SELECTION_ARTIFACT_UNAVAILABLE")
        return temporal

    @app.post("/v1/selections/resolve")
    def resolve_selection(body: TemporalSelectionRequest) -> dict[str, Any]:
        try:
            return _require_temporal().resolve(body.model_dump())
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/selections/validate")
    def validate_selection(body: TemporalValidationRequest) -> dict[str, Any]:
        return _require_temporal().validate(body.query_id, body.task)

    @app.get("/v1/selections/{query_id}")
    def get_selection(query_id: str) -> dict[str, Any]:
        return {"query_id": query_id, "events": _require_temporal().selections(query_id)}

    @app.post("/v1/selections/{query_id}/replace")
    def replace_selection(query_id: str, body: TemporalReplacementRequest) -> dict[str, Any]:
        try:
            return _require_temporal().replace(query_id, body.model_dump())
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/submissions/preview")
    def preview_submission(query_id: str = Query(min_length=1)) -> dict[str, Any]:
        try:
            return _require_temporal().export_preview(query_id)
        except SelectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/videos/{video_id}/file")
    def get_video(video_id: str) -> FileResponse:
        try:
            path = service.video_path(video_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}") from exc
        except (PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MediaResolutionError as exc:
            raise HTTPException(status_code=502, detail="REMOTE_MEDIA_UNAVAILABLE") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Video file not found: {path}")
        return FileResponse(
            path,
            media_type=_MEDIA_TYPES.get(path.suffix.lower(), "video/mp4"),
            headers={"Cache-Control": _MEDIA_CACHE_CONTROL},
        )

    @app.get("/v1/videos/{video_id}/file")
    def bfe_get_video(video_id: str) -> FileResponse:
        try:
            path = service.video_path(video_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown video_id: {video_id}") from exc
        except (PermissionError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MediaResolutionError as exc:
            raise HTTPException(status_code=502, detail="REMOTE_MEDIA_UNAVAILABLE") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Video file not found: {path}")
        return FileResponse(
            path,
            media_type=_MEDIA_TYPES.get(path.suffix.lower(), "video/mp4"),
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": _MEDIA_CACHE_CONTROL,
            },
        )

    @app.post("/feedback")
    @app.post("/v1/feedback")
    def bfe_feedback(body: FeedbackEvent) -> dict[str, Any]:
        positive = set(body.positive_ids)
        negative = set(body.negative_ids)
        if positive & negative:
            raise HTTPException(status_code=422, detail="feedback labels must be disjoint")
        prior = set(body.prior_result_ids)
        if not positive.issubset(prior) or not negative.issubset(prior):
            raise HTTPException(status_code=422, detail="feedback IDs must be prior result IDs")
        event = body.model_dump()
        app.state.feedback_events.append(event)
        return {
            "status": "recorded",
            "record_count": len(app.state.feedback_events),
            "event": event,
            "evidence_level": "SESSION_LOCAL",
        }

    @app.post("/v1/interactions")
    def bfe_interaction(body: BFEInteractionEvent) -> dict[str, Any]:
        event = body.model_dump(mode="json")
        app.state.interaction_events.append(event)
        return {"status": "recorded", "record_count": len(app.state.interaction_events)}

    def _aic26_draft(body: AIC26SubmissionRequest) -> tuple[str, Any, str]:
        queue_items = _queue_snapshot(body.query_id)
        if not queue_items:
            raise AIC26SubmissionError(f"no queue items for query {body.query_id}")

        def queue_task_label(item: dict[str, Any]) -> str:
            return str(item.get("submission_task") or "").strip().upper() or "UNSPECIFIED"

        queued_tasks = sorted({queue_task_label(item) for item in queue_items})
        incompatible_tasks = [
            item for item in queue_items
            if queue_task_label(item) != body.task
            and not (body.task in {"KIS", "QA"} and queue_task_label(item) == "UNSPECIFIED")
        ]
        if incompatible_tasks or (
            body.task == "TRAKE"
            and any(queue_task_label(item) == "UNSPECIFIED" for item in queue_items)
        ):
            raise AIC26SubmissionError(
                f"queue task mismatch for query {body.query_id}: requested {body.task}, "
                f"queued {', '.join(queued_tasks)}. "
                "Select the queued task or clear incompatible items."
            )

        if body.task == "TRAKE":
            queue_items = [
                item
                for item in queue_items
                if item.get("event_step") is not None
                and str(item.get("submission_task") or "TRAKE").upper() == "TRAKE"
            ]
            if not queue_items and app.state.temporal_selection is not None:
                # Temporal selections are canonical evidence, but queue order
                # remains authoritative when the user explicitly queued the
                # events.  This fallback supports an existing marked chain
                # without inventing a queue rank.
                try:
                    drafts = app.state.temporal_selection.selections(body.query_id)
                except (KeyError, SelectionError):
                    drafts = []
                queue_items = [
                    {
                        "query_id": body.query_id,
                        "video_id": row.get("video_id"),
                        "stage_id": row.get("stage_id") or f"S{int(row.get('event_step', 0)) + 1}",
                        "frame_uid": row.get("frame_uid"),
                        "source_frame_idx": row.get("source_frame_idx"),
                        "event_step": row.get("event_step"),
                        "chain_id": row.get("video_id"),
                        "queue_position": int(row.get("event_step", 0)),
                    }
                    for row in drafts
                    if row.get("source_frame_idx") is not None
                ]
        else:
            queue_items = [
                item
                for item in queue_items
                if item.get("submission_task") in (None, body.task)
            ]
        if not queue_items:
            raise AIC26SubmissionError(
                f"queue task mismatch for query {body.query_id}: requested {body.task}, "
                f"queued {', '.join(queued_tasks)}. "
                "Select the queued task or clear incompatible items."
            )

        videos = sorted({str(item.get("video_id") or "") for item in queue_items})
        catalog = {}
        for video_id in videos:
            if not video_id:
                raise AIC26SubmissionError("queue item is missing video_id")
            try:
                catalog[video_id] = service.timeline(video_id)
            except KeyError as exc:
                raise AIC26SubmissionError(
                    f"queue video_id is not in canonical catalog: {video_id}"
                ) from exc

        generated = generate_aic26_rows(
            body.task,
            queue_items,
            catalog,
            target_rows=body.target_rows,
            answer=body.answer,
            event_count=body.event_count,
            delta=body.delta,
        )
        csv_text = render_aic26_csv(body.task, generated.rows)
        filename = _aic26_filename(body.query_id, body.filename)
        return filename, generated, csv_text

    @app.post("/v1/submissions/aic26/preview")
    def aic26_submission_preview(body: AIC26SubmissionRequest) -> dict[str, Any]:
        try:
            filename, generated, csv_text = _aic26_draft(body)
        except (AIC26SubmissionError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "query_id": body.query_id,
            "task": body.task,
            "filename": filename,
            "row_count": len(generated.rows),
            "manual_count": generated.manual_count,
            "generated_count": generated.generated_count,
            "max_radius_used": generated.max_radius_used,
            "delta": body.delta,
            "rows": list(generated.rows),
            "csv": csv_text,
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED",
        }

    @app.post("/v1/submissions/aic26/download")
    def aic26_submission_download(body: AIC26SubmissionRequest) -> Response:
        try:
            filename, _generated, csv_text = _aic26_draft(body)
        except (AIC26SubmissionError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=csv_text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/v1/submissions/download")
    def bfe_submission_download() -> None:
        """Keep the UI fail-closed until the official submission contract is attached."""

        raise HTTPException(
            status_code=503,
            detail="SUBMISSION_GATEWAY_UNAVAILABLE_UNTIL_OFFICIAL_CONTRACT",
        )

    selected_ui_dir = Path(ui_dir).expanduser().resolve() if ui_dir else None
    if selected_ui_dir is None:
        selected_ui_dir = (
            _BFE_DIST_DIR if (_BFE_DIST_DIR / "index.html").is_file() else _LEGACY_UI_DIR
        )
    if selected_ui_dir.is_dir():
        app.mount("/", StaticFiles(directory=selected_ui_dir, html=True), name="ui")

    return app
