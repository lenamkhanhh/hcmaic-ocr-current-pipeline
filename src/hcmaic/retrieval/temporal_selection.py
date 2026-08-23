"""Fail-closed temporal selection over canonical video PTS metadata.

The temporal layer treats ``frame_uid=video_id:source_frame_idx`` as the only
portable frame identity. PTS rows are ordered by presentation order; decode
order is retained as provenance and is never used to resolve a timestamp.
YouTube metadata is a preview reference only.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class SelectionError(ValueError):
    """A temporal selection cannot be resolved or exported safely."""


_EVENT_LABEL_RE = re.compile(r"^E[1-9][0-9]*$")
_MAPPING_MODES = {"nearest_pts"}
_CANONICAL_READY = {"CANONICAL_SOURCE_READY", "CANONICAL_VERIFIED"}
_PTS_COLUMNS = [
    ("video_id", pa.string()),
    ("canonical_video_sha256", pa.string()),
    ("source_frame_idx", pa.int64()),
    ("presentation_order_idx", pa.int64()),
    ("decode_order_idx", pa.int64()),
    ("presentation_order", pa.int64()),
    ("decode_order", pa.int64()),
    ("timebase_num", pa.int64()),
    ("timebase_den", pa.int64()),
    ("pts_ticks", pa.int64()),
    ("pts", pa.float64()),
    ("timestamp_ms", pa.int64()),
    ("duration_ms", pa.int64()),
    ("trim_offset_ms", pa.int64()),
    ("tie_break_policy", pa.string()),
    ("timestamp_convention", pa.string()),
    ("source_manifest_id", pa.string()),
    ("bytes", pa.int64()),
    ("source_kind", pa.string()),
    ("decoder", pa.string()),
    ("ffmpeg_version", pa.string()),
    ("extractor_code_sha", pa.string()),
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _selection_contract(request: dict[str, Any]) -> tuple[str, str]:
    event_step = int(request.get("event_step", 0))
    if event_step < 0:
        raise SelectionError("event_step must be non-negative")
    task = str(request.get("task") or "").upper() or None
    if task not in {None, "KIS", "TRAKE"}:
        raise SelectionError("task must be KIS or TRAKE")
    kind_value = request.get("selection_kind")
    kind = str(kind_value).strip().upper() if kind_value is not None else None
    if kind is None:
        kind = "KIS" if task in {None, "KIS"} else f"E{event_step + 1}"
    if kind != "KIS" and not _EVENT_LABEL_RE.fullmatch(kind):
        raise SelectionError("selection_kind must be KIS or E<number>")
    if kind == "KIS":
        if task == "TRAKE" or event_step != 0:
            raise SelectionError("KIS selection requires event_step=0")
        return "KIS", kind
    expected_step = int(kind[1:]) - 1
    if task == "KIS" or event_step != expected_step:
        raise SelectionError(f"{kind} maps to zero-based event_step={expected_step}")
    return task or "TRAKE", kind


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SelectionError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise SelectionError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_id(watch_url: str) -> str | None:
    match = re.search(r"(?:[?&]v=|youtu\.be/)([A-Za-z0-9_-]{6,})", watch_url)
    return match.group(1) if match else None


def _preview_row(video_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    watch_url = str(raw.get("watch_url") or "").strip()
    duration = raw.get("duration_s", raw.get("length"))
    try:
        duration_s = None if duration is None else float(duration)
    except (TypeError, ValueError) as exc:
        raise SelectionError(f"invalid duration for {video_id}") from exc
    return {
        "video_id": video_id,
        "source_kind": "youtube_preview",
        "external_id": str(raw.get("external_id") or _external_id(watch_url) or "") or None,
        "watch_url": watch_url or None,
        "duration_s": duration_s,
        "status": str(raw.get("status") or "PREVIEW_ONLY"),
        "identity_policy": "watch_url_is_preview_reference_only",
    }


def _canonical_row(raw: dict[str, Any], source_allowlist: tuple[Path, ...]) -> dict[str, Any]:
    video_id = str(raw.get("video_id") or "").strip()
    if not video_id:
        raise SelectionError("canonical source video_id is required")
    url = str(raw.get("url") or raw.get("watch_url") or "")
    if "drive.google.com" in url and "/view" in url:
        raise ValueError("Drive share /view URL cannot be a file identity")
    source_kind = str(raw.get("source_kind") or raw.get("backend") or "local").lower()
    if source_kind not in {"local", "drive", "kaggle", "object"}:
        raise SelectionError(f"unsupported canonical source_kind: {source_kind}")
    raw_path = raw.get("path")
    if not raw_path:
        return {
            "video_id": video_id,
            "source_kind": source_kind,
            "status": "UNAVAILABLE_NO_CANONICAL_SOURCE",
            "path": None,
            "sha256": None,
            "canonical_video_sha256": None,
            "bytes": None,
            "media_type": raw.get("media_type"),
            "source_manifest_id": raw.get("source_manifest_id"),
        }
    if source_kind != "local":
        normalized = str(raw_path).replace("\\", "/")
        if normalized.startswith("/") or any(part == ".." for part in normalized.split("/")):
            raise SelectionError("remote canonical path must be relative and traversal-free")
        supplied_sha = str(raw.get("sha256") or raw.get("canonical_video_sha256") or "").lower()
        supplied_bytes = raw.get("bytes")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", supplied_sha)
            or not isinstance(supplied_bytes, int)
            or supplied_bytes < 1
        ):
            raise SelectionError("remote canonical source requires sha256 and positive bytes")
        return {
            "video_id": video_id,
            "source_kind": source_kind,
            "status": "CANONICAL_SOURCE_DECLARED",
            "path": normalized,
            "sha256": supplied_sha,
            "canonical_video_sha256": supplied_sha,
            "bytes": supplied_bytes,
            "media_type": str(raw.get("media_type") or "video/mp4"),
            "source_manifest_id": raw.get("source_manifest_id")
            or f"canonical-source:{video_id}:{supplied_sha[:16]}",
        }
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        if len(source_allowlist) != 1:
            raise SelectionError("relative canonical paths require one source allowlist root")
        path = source_allowlist[0] / path
    resolved = path.resolve()
    if not any(resolved == root or root in resolved.parents for root in source_allowlist):
        raise SelectionError(f"canonical source path is outside allowlist: {resolved}")
    if not resolved.is_file():
        raise SelectionError(f"canonical source is missing: {resolved}")
    digest = _sha256(resolved)
    supplied_sha = raw.get("sha256") or raw.get("canonical_video_sha256")
    supplied_bytes = raw.get("bytes")
    if supplied_sha is not None and str(supplied_sha).lower() != digest:
        raise SelectionError(f"canonical source sha256 mismatch: {video_id}")
    if supplied_bytes is not None and int(supplied_bytes) != resolved.stat().st_size:
        raise SelectionError(f"canonical source byte count mismatch: {video_id}")
    return {
        "video_id": video_id,
        "source_kind": source_kind,
        "status": "CANONICAL_SOURCE_READY",
        "path": str(resolved),
        "sha256": digest,
        "canonical_video_sha256": digest,
        "bytes": resolved.stat().st_size,
        "media_type": str(raw.get("media_type") or "video/mp4"),
        "source_manifest_id": raw.get("source_manifest_id")
        or f"canonical-source:{video_id}:{digest[:16]}",
        "decoder": raw.get("decoder"),
        "ffmpeg_version": raw.get("ffmpeg_version"),
        "extractor_code_sha": raw.get("extractor_code_sha"),
    }


def _value(row: dict[str, Any], modern: str, legacy: str | None = None) -> Any:
    if modern in row:
        return row[modern]
    if legacy is not None and legacy in row:
        return row[legacy]
    return None


def _normalize_pts_row(
    row: dict[str, Any],
    *,
    canonical: dict[str, Any] | None = None,
    index_base: int = 0,
) -> dict[str, Any]:
    declared_base = int(row.get("index_base", index_base))
    if declared_base != 0:
        raise SelectionError("PTS source_frame_idx must use explicit zero-based indexing")
    video_id = str(row.get("video_id") or "").strip()
    if not video_id:
        raise SelectionError("PTS video_id is required")
    try:
        source_idx = int(row["source_frame_idx"])
        presentation = int(_value(row, "presentation_order_idx", "presentation_order"))
        decode = int(_value(row, "decode_order_idx", "decode_order"))
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionError("PTS row is missing explicit presentation/decode order") from exc
    if source_idx < 0 or presentation < 0 or decode < 0:
        raise SelectionError("PTS indexes use zero-based non-negative values")
    num = int(row.get("timebase_num", 1))
    den = int(row.get("timebase_den", 1000))
    if num <= 0 or den <= 0:
        raise SelectionError("PTS timebase must be positive")
    pts_ticks_value = row.get("pts_ticks")
    if pts_ticks_value is None:
        if row.get("timestamp_ms") is not None:
            pts_ticks_value = round(float(row["timestamp_ms"]) * den / (num * 1000))
        elif row.get("pts") is not None:
            pts_ticks_value = round(float(row["pts"]) * den / num)
        else:
            raise SelectionError("PTS row requires pts_ticks or timestamp_ms")
    pts_ticks = int(pts_ticks_value)
    trim_offset_ms = int(row.get("trim_offset_ms", 0))
    calculated_ms = round(pts_ticks * num * 1000 / den) + trim_offset_ms
    timestamp_ms = int(row.get("timestamp_ms", calculated_ms))
    if abs(timestamp_ms - calculated_ms) > 1:
        raise SelectionError("timestamp_ms disagrees with pts_ticks/timebase/trim offset")
    duration_value = row.get("duration_ms")
    duration_ms = None if duration_value is None else int(duration_value)
    if duration_ms is not None and duration_ms <= 0:
        raise SelectionError("duration_ms must be positive")
    if timestamp_ms < 0 or (duration_ms is not None and timestamp_ms >= duration_ms):
        raise SelectionError("PTS timestamp is outside [0,duration_ms)")
    canonical_sha = row.get("canonical_video_sha256") or row.get("sha256")
    if canonical is not None:
        canonical_sha = canonical_sha or canonical.get("canonical_video_sha256")
        if (
            canonical.get("canonical_video_sha256")
            and canonical_sha != canonical["canonical_video_sha256"]
        ):
            raise SelectionError(f"PTS canonical hash mismatch: {video_id}")
    canonical_sha = None if canonical_sha is None else str(canonical_sha)
    source_kind = row.get("source_kind") or (canonical or {}).get("source_kind")
    source_manifest_id = row.get("source_manifest_id") or (canonical or {}).get(
        "source_manifest_id"
    )
    bytes_value = row.get("bytes", (canonical or {}).get("bytes"))
    bytes_value = None if bytes_value is None else int(bytes_value)
    return {
        "video_id": video_id,
        "canonical_video_sha256": canonical_sha,
        "source_frame_idx": source_idx,
        "presentation_order_idx": presentation,
        "decode_order_idx": decode,
        "presentation_order": presentation,
        "decode_order": decode,
        "timebase_num": num,
        "timebase_den": den,
        "pts_ticks": pts_ticks,
        "pts": pts_ticks * num / den,
        "timestamp_ms": timestamp_ms,
        "duration_ms": duration_ms,
        "trim_offset_ms": trim_offset_ms,
        "tie_break_policy": str(
            row.get("tie_break_policy") or "abs_delta_ms,presentation_order_idx,source_frame_idx"
        ),
        "timestamp_convention": str(
            row.get("timestamp_convention")
            or "source_frame_idx=zero-based; presentation PTS; interval=[0,duration_ms)"
        ),
        "source_manifest_id": None if source_manifest_id is None else str(source_manifest_id),
        "bytes": bytes_value,
        "source_kind": None if source_kind is None else str(source_kind),
        "decoder": row.get("decoder") or (canonical or {}).get("decoder"),
        "ffmpeg_version": row.get("ffmpeg_version") or (canonical or {}).get("ffmpeg_version"),
        "extractor_code_sha": row.get("extractor_code_sha")
        or (canonical or {}).get("extractor_code_sha"),
    }


def _write_empty_pts(path: Path) -> None:
    pq.write_table(pa.Table.from_pylist([], schema=pa.schema(_PTS_COLUMNS)), path)


def build_temporal_artifacts(
    output_root: Path,
    *,
    media_info_dir: Path | None,
    video_references: list[dict[str, Any]] | None = None,
    canonical_sources: list[dict[str, Any]] | None = None,
    source_allowlist: list[Path] | None = None,
    pts_rows: list[dict[str, Any]] | None = None,
    artifact_version: str = "temporal-selection-v1",
    index_base: int = 0,
) -> Path:
    """Build a versioned metadata-only selection bundle; never downloads media."""

    if index_base != 0:
        raise SelectionError("artifact index_base must be zero")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    references = list(video_references or [])
    if media_info_dir is not None:
        references = []
        for path in sorted(Path(media_info_dir).glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            references.append(_preview_row(path.stem, raw))
    references = [
        row
        if row.get("source_kind") == "youtube_preview"
        else _preview_row(str(row["video_id"]), row)
        for row in references
    ]
    refs_by_id = {str(row["video_id"]): row for row in references}
    allowlist = tuple(path.expanduser().resolve() for path in (source_allowlist or []))
    canonical = [_canonical_row(row, allowlist) for row in (canonical_sources or [])]
    canonical_by_id = {row["video_id"]: row for row in canonical}
    for video_id, preview in refs_by_id.items():
        canonical_by_id.setdefault(
            video_id,
            {
                **preview,
                "status": "UNAVAILABLE_NO_CANONICAL_SOURCE",
                "path": None,
                "sha256": None,
                "canonical_video_sha256": None,
                "bytes": None,
                "media_type": "video/mp4",
                "source_manifest_id": None,
            },
        )
    pts: list[dict[str, Any]] = []
    seen_source: set[tuple[str, int]] = set()
    seen_presentation: set[tuple[str, int]] = set()
    for raw_row in list(pts_rows or []):
        video_id = str(raw_row.get("video_id") or "")
        normalized = _normalize_pts_row(
            raw_row, canonical=canonical_by_id.get(video_id), index_base=index_base
        )
        identity = (video_id, normalized["source_frame_idx"])
        if identity in seen_source:
            raise SelectionError(f"duplicate PTS source identity: {identity[0]}:{identity[1]}")
        presentation_identity = (video_id, normalized["presentation_order_idx"])
        if presentation_identity in seen_presentation:
            raise SelectionError(
                f"duplicate presentation order: {video_id}:{presentation_identity[1]}"
            )
        seen_source.add(identity)
        seen_presentation.add(presentation_identity)
        pts.append(normalized)
    for video_id in {row["video_id"] for row in pts}:
        durations = {row["duration_ms"] for row in pts if row["video_id"] == video_id}
        durations.discard(None)
        if len(durations) > 1:
            raise SelectionError(f"duration mismatch in PTS rows: {video_id}")
    (root / "video_reference_manifest.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in sorted(refs_by_id.values(), key=lambda x: x["video_id"])
        ),
        encoding="utf-8",
    )
    (root / "canonical_video_inventory.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in sorted(canonical_by_id.values(), key=lambda x: x["video_id"])
        ),
        encoding="utf-8",
    )
    if pts:
        pq.write_table(pa.Table.from_pylist(pts), root / "frame_pts_index.parquet")
    else:
        _write_empty_pts(root / "frame_pts_index.parquet")
    for name in ("selection_events.jsonl", "selection_drafts.jsonl"):
        (root / name).touch()
    (root / "selection_failure_ledger.json").write_text(
        json.dumps({"failures": []}, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "artifact_version": artifact_version,
        "status": "ENGINEERING_PROXY",
        "quality_status": "UNVALIDATED",
        "identity": "frame_uid=video_id:source_frame_idx",
        "timestamp_policy": (
            "PTS ticks/timebase -> timestamp_ms; presentation order only; no time*fps inference"
        ),
        "timestamp_convention": (
            "source_frame_idx zero-based; selected interval [0,duration_ms); "
            "trim_offset_ms included"
        ),
        "index_base": 0,
        "pts_schema": [name for name, _ in _PTS_COLUMNS],
        "tie_break_policy": "abs_delta_ms,presentation_order_idx,source_frame_idx",
        "audit_event_schema": {
            "event_uid": "string",
            "selection_uid": "string",
            "query_id": "string",
            "event_step": "integer|null, zero-based",
            "event_type": "resolve|replace|reject|validate",
            "created_at_utc": "RFC3339 UTC",
            "session_id": "optional string",
            "operator_id": "optional string",
            "supersedes_event_uid": "optional string",
            "status": "ACTIVE|SUPERSEDED|REJECTED",
            "idempotency_key": "string",
        },
        "video_count": len(canonical_by_id),
        "pts_row_count": len(pts),
        "source_video_count": sum(
            row.get("status") in _CANONICAL_READY for row in canonical_by_id.values()
        ),
    }
    (root / "temporal_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return root


def _row_presentation_order(row: dict[str, Any]) -> int:
    return int(_value(row, "presentation_order_idx", "presentation_order") or 0)


def map_timestamp_to_pts(
    rows: list[dict[str, Any]], selected_time_ms: int, *, mapping_mode: str = "nearest_pts"
) -> dict[str, Any]:
    if mapping_mode not in _MAPPING_MODES:
        raise SelectionError(f"unsupported mapping_mode: {mapping_mode}")
    if selected_time_ms < 0:
        raise SelectionError("selected_time_ms must be non-negative")
    if not rows:
        raise SelectionError("no PTS rows are available")
    ordered = sorted(
        rows,
        key=lambda row: (
            _row_presentation_order(row),
            int(row["source_frame_idx"]),
            int(_value(row, "decode_order_idx", "decode_order") or 0),
        ),
    )
    durations = {row.get("duration_ms") for row in ordered if row.get("duration_ms") is not None}
    if len(durations) > 1:
        raise SelectionError("duration mismatch in PTS rows")
    duration_ms = next(iter(durations), None)
    if duration_ms is not None and not 0 <= selected_time_ms < int(duration_ms):
        raise SelectionError("selected timestamp is out of range [0,duration_ms)")
    if duration_ms is None:
        low = min(int(row["timestamp_ms"]) for row in ordered)
        high = max(int(row["timestamp_ms"]) for row in ordered)
        if selected_time_ms < low or selected_time_ms > high:
            raise SelectionError("selected timestamp is out of range")
    chosen = min(
        ordered,
        key=lambda row: (
            abs(int(row["timestamp_ms"]) - selected_time_ms),
            _row_presentation_order(row),
            int(row["source_frame_idx"]),
        ),
    )
    resolved = int(chosen["timestamp_ms"])
    frame_uid = f"{chosen['video_id']}:{int(chosen['source_frame_idx'])}"
    return {
        "video_id": str(chosen["video_id"]),
        "source_frame_idx": int(chosen["source_frame_idx"]),
        "frame_uid": frame_uid,
        "resolved_frame_uid": frame_uid,
        "candidate_frame_uid": None,
        "nearest_keyframe_uid": None,
        "selected_time_ms": selected_time_ms,
        "resolved_timestamp_ms": resolved,
        "delta_ms": resolved - selected_time_ms,
        "mapping_mode": mapping_mode,
        "mapping_method": "PTS_NEAREST_PRESENTATION_ORDER",
        "mapping_status": "RESOLVED_CANONICAL",
    }


def validate_selection_rows(rows: list[dict[str, Any]], *, task: str) -> dict[str, Any]:
    task = str(task).upper()
    issues: list[dict[str, str]] = []
    if task not in {"KIS", "TRAKE"}:
        issues.append({"code": "UNKNOWN_TASK", "message": "task must be KIS or TRAKE"})
    if task == "KIS" and len(rows) != 1:
        issues.append(
            {"code": "KIS_ONE_MARKER_REQUIRED", "message": "KIS requires exactly one marker/query"}
        )
    if task == "TRAKE":
        ordered = sorted(rows, key=lambda row: int(row.get("event_step", -1)))
        steps = [int(row.get("event_step", -1)) for row in ordered]
        indexes = [
            int(row["source_frame_idx"]) if row.get("source_frame_idx") is not None else -1
            for row in ordered
        ]
        if len(steps) != len(set(steps)):
            issues.append(
                {
                    "code": "TRAKE_EVENT_STEP_DUPLICATE",
                    "message": "event_step values must be unique",
                }
            )
        if steps != list(range(len(steps))):
            issues.append(
                {
                    "code": "TRAKE_EVENT_STEP_NOT_CONTIGUOUS",
                    "message": "event_step must be contiguous from zero (E1..EN)",
                }
            )
        if len({str(row.get("video_id")) for row in ordered}) > 1:
            issues.append({"code": "TRAKE_VIDEO_MISMATCH", "message": "TRAKE must use one video"})
        if any(a >= b for a, b in zip(indexes, indexes[1:], strict=False)):
            issues.append(
                {
                    "code": "TRAKE_SOURCE_FRAME_IDX_NOT_STRICTLY_INCREASING",
                    "message": "source_frame_idx must be strictly increasing",
                }
            )
        if any(index < 0 for index in indexes):
            issues.append(
                {
                    "code": "TRAKE_MISSING_OR_REJECTED_MARKER",
                    "message": "missing/rejected marker blocks validation",
                }
            )
        if any(
            row.get("mapping_status") in {"PREVIEW_TIMESTAMP_ONLY", "SOURCE_FRAME_UNAVAILABLE"}
            or row.get("source_frame_idx") is None
            for row in ordered
        ):
            issues.append(
                {
                    "code": "TRAKE_CANONICAL_PTS_REQUIRED",
                    "message": "TRAKE cannot export without canonical PTS resolution",
                }
            )
    elif task == "KIS" and rows:
        row = rows[0]
        if (
            row.get("mapping_status") in {"PREVIEW_TIMESTAMP_ONLY", "SOURCE_FRAME_UNAVAILABLE"}
            or row.get("source_frame_idx") is None
        ):
            issues.append(
                {
                    "code": "KIS_CANONICAL_PTS_REQUIRED",
                    "message": "KIS cannot export without canonical PTS resolution",
                }
            )
    return {"status": "BLOCKED" if issues else "VALID", "task": task, "issues": issues}


@dataclass
class TemporalSelectionService:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.references = {
            str(row["video_id"]): row
            for row in _jsonl(self.root / "video_reference_manifest.jsonl")
        }
        self.inventory = {
            str(row["video_id"]): row
            for row in _jsonl(self.root / "canonical_video_inventory.jsonl")
        }
        self._pts: dict[str, list[dict[str, Any]]] = {}
        table = pq.read_table(self.root / "frame_pts_index.parquet")
        for row in table.to_pylist():
            self._pts.setdefault(str(row["video_id"]), []).append(row)

    def availability(self, video_id: str) -> dict[str, Any]:
        if video_id not in self.references and video_id not in self.inventory:
            raise KeyError(video_id)
        inventory = self.inventory.get(video_id, {})
        rows = self._pts.get(video_id, [])
        canonical_ready = inventory.get("status") in _CANONICAL_READY and bool(
            inventory.get("path") or inventory.get("source_kind") != "local"
        )
        return {
            "video_id": video_id,
            "external_id": self.references.get(video_id, {}).get("external_id"),
            "watch_url": self.references.get(video_id, {}).get("watch_url"),
            "duration_s": self.references.get(video_id, {}).get("duration_s"),
            "source_available": canonical_ready,
            "pts_available": bool(rows),
            "preview_only": not bool(rows) or not canonical_ready,
            "status": "READY_CANONICAL_PTS"
            if rows and canonical_ready
            else "PREVIEW_TIMESTAMP_ONLY",
            "quality_status": "UNVALIDATED",
        }

    def timeline(self, video_id: str) -> list[dict[str, Any]]:
        if video_id not in self.references and video_id not in self.inventory:
            raise KeyError(video_id)
        ordered = sorted(
            self._pts.get(video_id, []),
            key=lambda row: (
                _row_presentation_order(row),
                int(row["source_frame_idx"]),
            ),
        )
        return [
            {
                "video_id": str(row["video_id"]),
                "source_frame_idx": int(row["source_frame_idx"]),
                "frame_uid": f"{row['video_id']}:{int(row['source_frame_idx'])}",
                "resolved_frame_uid": f"{row['video_id']}:{int(row['source_frame_idx'])}",
                "candidate_frame_uid": None,
                "nearest_keyframe_uid": None,
                "selected_time_ms": int(row["timestamp_ms"]),
                "resolved_timestamp_ms": int(row["timestamp_ms"]),
                "delta_ms": 0,
                "mapping_mode": "nearest_pts",
                "mapping_method": "PTS_NEAREST_PRESENTATION_ORDER",
                "mapping_status": "RESOLVED_CANONICAL",
            }
            for row in ordered
        ]

    def _events(self) -> list[dict[str, Any]]:
        return _jsonl(self.root / "selection_events.jsonl")

    def _find_by_idempotency(self, key: str | None) -> dict[str, Any] | None:
        if not key:
            return None
        return next(
            (event for event in self._events() if event.get("idempotency_key") == key), None
        )

    def _latest_event(self, query_id: str, event_step: int) -> dict[str, Any] | None:
        candidates = [
            event
            for event in self._events()
            if event.get("query_id") == query_id
            and event.get("event_step") == event_step
            and event.get("event_type") in {"resolve", "replace"}
        ]
        return candidates[-1] if candidates else None

    def _resolve_event(
        self,
        request: dict[str, Any],
        *,
        event_type: str,
        supersedes_event_uid: str | None = None,
    ) -> dict[str, Any]:
        if event_type not in {"resolve", "replace"}:
            raise SelectionError("invalid selection event type")
        video_id = str(request.get("video_id") or "")
        selected = int(request.get("selected_time_ms", -1))
        task, selection_kind = _selection_contract(request)
        mapping_mode = str(request.get("mapping_mode") or "nearest_pts")
        if mapping_mode not in _MAPPING_MODES:
            raise SelectionError(f"unsupported mapping_mode: {mapping_mode}")
        query_id = str(request.get("query_id") or "")
        idempotency_key = request.get("idempotency_key")
        previous = self._find_by_idempotency(str(idempotency_key) if idempotency_key else None)
        if previous is not None:
            return previous
        if video_id not in self.references and video_id not in self.inventory:
            raise SelectionError("unknown video_id")
        rows = self._pts.get(video_id, [])
        inventory = self.inventory.get(video_id, {})
        canonical_ready = inventory.get("status") in _CANONICAL_READY and bool(
            inventory.get("path") or inventory.get("source_kind") != "local"
        )
        candidate = request.get("candidate_frame_uid")
        if candidate is not None:
            candidate = str(candidate)
            if not re.fullmatch(re.escape(video_id) + r":[0-9]+", candidate):
                raise SelectionError("candidate_frame_uid belongs to another video or is invalid")
        nearest_keyframe = request.get("nearest_keyframe_uid")
        if nearest_keyframe is not None:
            nearest_keyframe = str(nearest_keyframe)
            if not re.fullmatch(re.escape(video_id) + r":[0-9]+", nearest_keyframe):
                raise SelectionError("nearest_keyframe_uid belongs to another video or is invalid")
        if not rows:
            result = {
                "video_id": video_id,
                "source_frame_idx": None,
                "frame_uid": None,
                "resolved_frame_uid": None,
                "candidate_frame_uid": candidate,
                "nearest_keyframe_uid": nearest_keyframe,
                "selected_time_ms": selected,
                "resolved_timestamp_ms": None,
                "delta_ms": None,
                "mapping_mode": mapping_mode,
                "mapping_method": "YOUTUBE_PLAYER_TIMESTAMP",
                "mapping_status": "PREVIEW_TIMESTAMP_ONLY",
            }
        elif not canonical_ready:
            result = {
                "video_id": video_id,
                "source_frame_idx": None,
                "frame_uid": None,
                "resolved_frame_uid": None,
                "candidate_frame_uid": candidate,
                "nearest_keyframe_uid": nearest_keyframe,
                "selected_time_ms": selected,
                "resolved_timestamp_ms": None,
                "delta_ms": None,
                "mapping_mode": mapping_mode,
                "mapping_method": "PTS_MAPPING_BLOCKED",
                "mapping_status": "SOURCE_FRAME_UNAVAILABLE",
            }
        else:
            result = map_timestamp_to_pts(rows, selected, mapping_mode=mapping_mode)
            result["candidate_frame_uid"] = candidate
            result["nearest_keyframe_uid"] = nearest_keyframe
        event_uid = _uid("event")
        selection_uid = _uid("selection")
        event = {
            **result,
            "event_uid": event_uid,
            "selection_uid": selection_uid,
            "query_id": query_id,
            "event_step": int(request.get("event_step", 0)),
            "task": task,
            "selection_kind": selection_kind,
            "event_type": event_type,
            "created_at_utc": _utc_now(),
            "session_id": request.get("session_id"),
            "operator_id": request.get("operator_id"),
            "supersedes_event_uid": supersedes_event_uid,
            "status": "ACTIVE",
            "idempotency_key": str(idempotency_key or event_uid),
        }
        _append_jsonl(self.root / "selection_events.jsonl", event)
        _append_jsonl(self.root / "selection_drafts.jsonl", event)
        return event

    def resolve(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._resolve_event(request, event_type="resolve")

    def replace(self, query_id: str, request: dict[str, Any]) -> dict[str, Any]:
        event_step = int(request.get("event_step", 0))
        previous = self._latest_event(query_id, event_step)
        return self._resolve_event(
            {**request, "query_id": query_id},
            event_type="replace",
            supersedes_event_uid=previous.get("event_uid") if previous else None,
        )

    def selections(self, query_id: str) -> list[dict[str, Any]]:
        return [
            row
            for row in _jsonl(self.root / "selection_drafts.jsonl")
            if row.get("query_id") == query_id
        ]

    def validate(self, query_id: str, task: str) -> dict[str, Any]:
        report = validate_selection_rows(self.selections(query_id), task=task)
        report.update({"query_id": query_id, "quality_status": "UNVALIDATED"})
        event_uid = _uid("event")
        _append_jsonl(
            self.root / "selection_events.jsonl",
            {
                "event_uid": event_uid,
                "selection_uid": None,
                "query_id": query_id,
                "event_step": None,
                "event_type": "validate",
                "created_at_utc": _utc_now(),
                "session_id": None,
                "operator_id": None,
                "supersedes_event_uid": None,
                "status": "ACTIVE" if report["status"] == "VALID" else "REJECTED",
                "idempotency_key": event_uid,
                "report": report,
            },
        )
        return report

    def export_preview(self, query_id: str) -> dict[str, Any]:
        rows = self.selections(query_id)
        if not rows:
            raise SelectionError("no selection exists")
        if any(
            row.get("mapping_status") in {"PREVIEW_TIMESTAMP_ONLY", "SOURCE_FRAME_UNAVAILABLE"}
            or row.get("source_frame_idx") is None
            for row in rows
        ):
            raise SelectionError(
                "preview/source video without canonical PTS cannot export a source frame"
            )
        return {"query_id": query_id, "status": "PREVIEW_ONLY", "selections": rows}
