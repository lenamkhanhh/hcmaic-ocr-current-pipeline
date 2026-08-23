"""Fail-closed bridge between existing frame/video media inventories.

The bridge reads metadata only. Stable identity is ``frame_uid``; the
``source_fingerprint`` carried by ``vd/videos`` is advisory and is never used
as a raw-video SHA-256 gate.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class MediaBridgeError(ValueError):
    """A bridge input violates the canonical identity/provenance contract."""


@dataclass(frozen=True)
class VideoBridgeRecord:
    video_id: str
    dataset: str
    member_path: str
    bytes: int
    media_type: str
    range_capable: bool
    provenance_status: str
    watch_url: str | None
    source_manifest_id: str | None
    source_fingerprint: str | None
    sha256: str | None
    remote_content_fingerprint: dict[str, str]


@dataclass(frozen=True)
class ResolvedMedia:
    frame_uid: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    shot_id: str
    keyframe_path: str
    image_backend: str
    image_dataset: str
    image_media_path: str
    image_bytes: int
    image_sha256: str
    video_member_path: str
    video_bytes: int
    video_media_type: str
    video_range_capable: bool
    video_provenance_status: str
    video_sha256: str | None
    video_source_manifest_id: str | None
    watch_url: str | None
    watch_status: str
    source_fingerprint: str | None
    remote_content_fingerprint: dict[str, str]

    @property
    def video_stream_path(self) -> str:
        return f"/v1/videos/{quote(self.video_id, safe='')}/stream"


@dataclass(frozen=True)
class _FrameRecord:
    frame_uid: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    shot_id: str
    keyframe_path: str
    source_fingerprint: str | None
    image_backend: str
    image_dataset: str
    image_media_path: str
    image_bytes: int
    image_sha256: str


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MediaBridgeError(f"cannot read bridge input: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MediaBridgeError(f"invalid JSON in {path} line {line_number}") from exc
        if not isinstance(row, dict):
            raise MediaBridgeError(f"row in {path} line {line_number} must be an object")
        yield line_number, row


def _require_text(row: dict[str, Any], key: str, context: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise MediaBridgeError(f"missing {key} in {context}")
    return value


def _require_nonnegative_int(row: dict[str, Any], key: str, context: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaBridgeError(f"invalid {key} in {context}") from exc
    if value < 0:
        raise MediaBridgeError(f"invalid {key} in {context}")
    return value


def _candidate_fingerprint(value: Any, context: str) -> str | None:
    if value in {None, ""}:
        return None
    fingerprint = str(value).lower()
    if not _HEX64.fullmatch(fingerprint):
        raise MediaBridgeError(f"invalid advisory source_fingerprint in {context}")
    return fingerprint


def _load_vd(vd_root: Path) -> tuple[dict[str, _FrameRecord], set[str], dict[str, str | None]]:
    files = sorted(vd_root.rglob("*.jsonl")) if vd_root.is_dir() else []
    if not files:
        raise MediaBridgeError(f"no vd JSONL files found: {vd_root}")
    frames: dict[str, _FrameRecord] = {}
    video_fingerprints: dict[str, str | None] = {}
    keyframe_ids: set[str] = set()
    for path in files:
        for line_number, row in _iter_jsonl(path):
            context = f"{path}:{line_number}"
            payload = row.get("payload")
            metadata = row.get("metadata")
            if not isinstance(payload, dict) or not isinstance(metadata, dict):
                raise MediaBridgeError(f"vd row missing payload/metadata in {context}")
            if str(payload.get("schema_version")) != "2.2":
                raise MediaBridgeError(f"unsupported vd schema_version in {context}")
            video_id = _require_text(payload, "video_id", context)
            frame_idx = _require_nonnegative_int(payload, "frame_id", context)
            frame_uid = f"{video_id}:{frame_idx}"
            keyframe_id = _require_text(payload, "keyframe_id", context)
            if keyframe_id in keyframe_ids or frame_uid in frames:
                raise MediaBridgeError(f"duplicate vd identity in {context}")
            keyframe_ids.add(keyframe_id)
            fingerprint = _candidate_fingerprint(metadata.get("source_fingerprint"), context)
            prior = video_fingerprints.setdefault(video_id, fingerprint)
            if prior != fingerprint:
                raise MediaBridgeError(f"source_fingerprint changed within video_id {video_id}")
            frames[frame_uid] = _FrameRecord(
                frame_uid=frame_uid,
                video_id=video_id,
                source_frame_idx=frame_idx,
                timestamp_ms=_require_nonnegative_int(payload, "timestamp_ms", context),
                shot_id=_require_text(payload, "shot_id", context),
                keyframe_path=_require_text(payload, "keyframe_path", context),
                source_fingerprint=fingerprint,
                image_backend="",
                image_dataset="",
                image_media_path="",
                image_bytes=0,
                image_sha256="",
            )
    return frames, set(video_fingerprints), video_fingerprints


def _load_frame_catalog(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, row in _iter_jsonl(path):
        context = f"{path}:{line_number}"
        frame_uid = _require_text(row, "frame_uid", context)
        if frame_uid in rows:
            raise MediaBridgeError(f"duplicate frame_uid in catalog: {frame_uid}")
        rows[frame_uid] = row
    return rows


def _load_frame_media(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, row in _iter_jsonl(path):
        context = f"{path}:{line_number}"
        if str(row.get("kind") or "").lower() != "frame":
            raise MediaBridgeError(f"non-frame row in frame media manifest: {context}")
        frame_uid = _require_text(row, "frame_uid", context)
        if frame_uid in rows:
            raise MediaBridgeError(f"duplicate frame_uid in media manifest: {frame_uid}")
        rows[frame_uid] = row
    return rows


def _load_video_rows(path: Path, *, range_manifest: bool) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_number, row in _iter_jsonl(path):
        context = f"{path}:{line_number}"
        if range_manifest and str(row.get("kind") or "").lower() != "video":
            raise MediaBridgeError(f"non-video row in range manifest: {context}")
        video_id = _require_text(row, "video_id", context)
        if video_id in rows:
            raise MediaBridgeError(f"duplicate video_id in {path}: {video_id}")
        rows[video_id] = row
    return rows


def _video_dataset(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset_id") or "").strip()
    if dataset:
        return dataset
    return f"{row.get('dataset_owner')}/{row.get('dataset_slug')}"


class MediaBridge:
    """Validated metadata bridge keyed by canonical frame/video identity."""

    def __init__(
        self,
        frames: dict[str, ResolvedMedia],
        videos: dict[str, VideoBridgeRecord],
        *,
        summary: dict[str, int],
    ) -> None:
        self._frames = frames
        self._videos = videos
        self._summary = summary

    def resolve_frame(self, frame_uid: str) -> ResolvedMedia:
        try:
            return self._frames[frame_uid]
        except KeyError as exc:
            raise KeyError(f"unknown frame_uid: {frame_uid}") from exc

    def video(self, video_id: str) -> VideoBridgeRecord:
        try:
            return self._videos[video_id]
        except KeyError as exc:
            raise KeyError(f"unknown video_id: {video_id}") from exc

    def summary(self) -> dict[str, int]:
        return dict(self._summary)


def load_media_bridge(
    *,
    vd_root: Path,
    frame_catalog: Path,
    frame_media: Path,
    canonical_video: Path,
    range_manifest: Path,
) -> MediaBridge:
    """Load and cross-check existing inventories without copying their rows."""

    vd_frames, vd_video_ids, source_fingerprints = _load_vd(Path(vd_root))
    catalog_rows = _load_frame_catalog(Path(frame_catalog))
    media_rows = _load_frame_media(Path(frame_media))
    if set(vd_frames) != set(catalog_rows) or set(vd_frames) != set(media_rows):
        raise MediaBridgeError("frame_uid set mismatch across vd/catalog/media manifests")

    frames: dict[str, ResolvedMedia] = {}
    for frame_uid, vd_frame in vd_frames.items():
        catalog = catalog_rows[frame_uid]
        media = media_rows[frame_uid]
        for key in ("video_id", "source_frame_idx", "timestamp_ms", "shot_id"):
            if catalog.get(key) != media.get(key):
                raise MediaBridgeError(f"frame field mismatch for {frame_uid}: {key}")
        if str(catalog.get("video_id")) != vd_frame.video_id:
            raise MediaBridgeError(f"frame video_id mismatch for {frame_uid}")
        if int(catalog.get("source_frame_idx", -1)) != vd_frame.source_frame_idx:
            raise MediaBridgeError(f"frame source_frame_idx mismatch for {frame_uid}")
        if int(catalog.get("timestamp_ms", -1)) != vd_frame.timestamp_ms:
            raise MediaBridgeError(f"frame timestamp mismatch for {frame_uid}")
        frames[frame_uid] = ResolvedMedia(
            frame_uid=frame_uid,
            video_id=vd_frame.video_id,
            source_frame_idx=vd_frame.source_frame_idx,
            timestamp_ms=vd_frame.timestamp_ms,
            shot_id=vd_frame.shot_id,
            keyframe_path=_require_text(catalog, "keyframe_path", f"catalog:{frame_uid}"),
            image_backend=_require_text(media, "backend", f"media:{frame_uid}"),
            image_dataset=_require_text(media, "dataset", f"media:{frame_uid}"),
            image_media_path=_require_text(media, "path", f"media:{frame_uid}"),
            image_bytes=_require_nonnegative_int(media, "bytes", f"media:{frame_uid}"),
            image_sha256=_require_text(media, "sha256", f"media:{frame_uid}"),
            video_member_path="",
            video_bytes=0,
            video_media_type="",
            video_range_capable=False,
            video_provenance_status="",
            video_sha256=None,
            video_source_manifest_id=None,
            watch_url=None,
            watch_status="PREVIEW_REFERENCE_ONLY",
            source_fingerprint=vd_frame.source_fingerprint,
            remote_content_fingerprint={},
        )

    canonical_rows = _load_video_rows(Path(canonical_video), range_manifest=False)
    range_rows = _load_video_rows(Path(range_manifest), range_manifest=True)
    if vd_video_ids != set(canonical_rows) or vd_video_ids != set(range_rows):
        raise MediaBridgeError("video_id set mismatch across vd/video manifests")
    videos: dict[str, VideoBridgeRecord] = {}
    for video_id in sorted(vd_video_ids):
        canonical = canonical_rows[video_id]
        ranged = range_rows[video_id]
        dataset = _video_dataset(canonical)
        member_path = _require_text(canonical, "member_path", f"canonical:{video_id}")
        bytes_count = _require_nonnegative_int(canonical, "bytes", f"canonical:{video_id}")
        if (
            str(ranged.get("dataset")) != dataset
            or str(ranged.get("path")) != member_path
            or int(ranged.get("bytes", -1)) != bytes_count
        ):
            raise MediaBridgeError(f"video source mismatch for {video_id}")
        backend = _require_text(ranged, "backend", f"range:{video_id}")
        if backend != "kaggle_http_range":
            raise MediaBridgeError(f"unsupported range backend for {video_id}: {backend}")
        videos[video_id] = VideoBridgeRecord(
            video_id=video_id,
            dataset=dataset,
            member_path=member_path,
            bytes=bytes_count,
            media_type=str(canonical.get("media_type") or "video/mp4"),
            range_capable=bool(ranged.get("range_capable")),
            provenance_status=_require_text(ranged, "provenance_status", f"range:{video_id}"),
            watch_url=str(canonical.get("watch_url") or "") or None,
            source_manifest_id=str(canonical.get("source_manifest_id") or "") or None,
            source_fingerprint=source_fingerprints.get(video_id),
            sha256=str(ranged.get("sha256") or "") or None,
            remote_content_fingerprint={
                str(key): str(value)
                for key, value in (ranged.get("remote_content_fingerprint") or {}).items()
                if key in {"etag", "x-goog-hash"}
            },
        )

    for frame_uid, resolved in list(frames.items()):
        video = videos[resolved.video_id]
        frames[frame_uid] = ResolvedMedia(
            **{
                **resolved.__dict__,
                "video_member_path": video.member_path,
                "video_bytes": video.bytes,
                "video_media_type": video.media_type,
                "video_range_capable": video.range_capable,
                "video_provenance_status": video.provenance_status,
                "video_sha256": video.sha256,
                "video_source_manifest_id": video.source_manifest_id,
                "watch_url": video.watch_url,
                "remote_content_fingerprint": video.remote_content_fingerprint,
            }
        )
    return MediaBridge(
        frames,
        videos,
        summary={
            "vd_frame_count": len(vd_frames),
            "catalog_frame_count": len(catalog_rows),
            "media_frame_count": len(media_rows),
            "video_count": len(videos),
            "range_capable_video_count": sum(video.range_capable for video in videos.values()),
        },
    )
