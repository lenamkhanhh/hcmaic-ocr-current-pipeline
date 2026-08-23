"""Build and validate the metadata-only mixed frame/video media contract.

The artifact produced here is a new, versioned bridge.  It does not rewrite
the existing frame manifest or video inventories and it never reads media
payloads.  Frame identity remains ``frame_uid=video_id:source_frame_idx``;
FAISS row numbers are deliberately absent from this contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from collections import Counter
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "hcmaic-media-manifest-full-range-v2"
RAW_VIDEO_SHA256_STATUS = "NOT_REQUIRED_RAW_VIDEO"
FRAME_UID_RE = re.compile(r"^[^:\s]+:[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SHARD_RE = re.compile(r"^shard_[0-9]{4}$")


class MediaManifestV2Error(ValueError):
    """Source inventories cannot produce a safe mixed media contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MediaManifestV2Error(f"manifest input not found: {source}")
    try:
        handle = source.open(encoding="utf-8")
    except OSError as exc:
        raise MediaManifestV2Error(f"cannot read manifest input: {source}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MediaManifestV2Error(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(row, dict):
                raise MediaManifestV2Error(f"JSONL row must be an object at {source}:{line_number}")
            yield line_number, row


def _text(row: dict[str, Any], key: str, context: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise MediaManifestV2Error(f"missing {key} in {context}")
    return value


def _nonnegative_int(row: dict[str, Any], key: str, context: str) -> int:
    value = row.get(key)
    if isinstance(value, bool):
        raise MediaManifestV2Error(f"invalid {key} in {context}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise MediaManifestV2Error(f"invalid {key} in {context}") from exc
    if result < 0:
        raise MediaManifestV2Error(f"invalid {key} in {context}")
    return result


def _positive_int(row: dict[str, Any], key: str, context: str) -> int:
    result = _nonnegative_int(row, key, context)
    if result < 1:
        raise MediaManifestV2Error(f"invalid {key} in {context}")
    return result


def _safe_relative_path(value: Any, field: str, context: str) -> str:
    path = str(value or "").replace("\\", "/")
    if not path or path.startswith("/") or re.match(r"^[A-Za-z]:", path):
        raise MediaManifestV2Error(f"{field} must be relative in {context}")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise MediaManifestV2Error(f"{field} must not contain traversal in {context}")
    return "/".join(parsed.parts)


def _frame_uid(row: dict[str, Any], context: str) -> tuple[str, str, int]:
    frame_uid = _text(row, "frame_uid", context)
    if not FRAME_UID_RE.fullmatch(frame_uid):
        raise MediaManifestV2Error(f"invalid frame_uid in {context}")
    video_id = _text(row, "video_id", context)
    source_frame_idx = _nonnegative_int(row, "source_frame_idx", context)
    if frame_uid != f"{video_id}:{source_frame_idx}":
        raise MediaManifestV2Error(f"frame_uid identity mismatch in {context}")
    return frame_uid, video_id, source_frame_idx


def _normal_video_id(value: Any, context: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise MediaManifestV2Error(f"missing video_id in {context}")
    return result.casefold()


def _sha256_or_none(value: Any, context: str) -> str | None:
    if value in {None, ""}:
        return None
    digest = str(value)
    if not SHA256_RE.fullmatch(digest):
        raise MediaManifestV2Error(f"invalid sha256 in {context}")
    return digest.lower()


def _safe_preview_url(value: Any, context: str) -> str | None:
    if value in {None, ""}:
        return None
    url = str(value).strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise MediaManifestV2Error(f"watch_url must be a credential-free https URL in {context}")
    return url


def _basename_key(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).stem.casefold()


def _index_video_rows(
    path: Path,
    *,
    label: str,
    require_kind: bool,
) -> dict[str, tuple[str, dict[str, Any]]]:
    indexed: dict[str, tuple[str, dict[str, Any]]] = {}
    for line_number, row in _read_jsonl(path):
        context = f"{label}:{line_number}"
        if require_kind and str(row.get("kind") or "").lower() != "video":
            raise MediaManifestV2Error(f"expected kind=video in {context}")
        video_id = _text(row, "video_id", context)
        normalized = _normal_video_id(video_id, context)
        if normalized in indexed:
            raise MediaManifestV2Error(f"duplicate normalized video_id in {label}: {video_id}")
        member_key = "path" if require_kind else "member_path"
        _safe_relative_path(_text(row, member_key, context), member_key, context)
        _positive_int(row, "bytes", context)
        indexed[normalized] = (video_id, row)
    if not indexed:
        raise MediaManifestV2Error(f"{label} is empty")
    return indexed


def _index_by_basename(
    rows: dict[str, tuple[str, dict[str, Any]]], *, path_key: str
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    result: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for video_id, row in rows.values():
        key = _basename_key(str(row[path_key]))
        result.setdefault(key, []).append((video_id, row))
    return result


def _validate_frame_row(
    row: dict[str, Any],
    *,
    context: str,
    frame_ids: dict[str, str],
    frame_video_ids: dict[str, str],
    shards: set[str],
) -> tuple[str, str]:
    if str(row.get("kind") or "").lower() != "frame":
        raise MediaManifestV2Error(f"expected kind=frame in {context}")
    frame_uid, video_id, _ = _frame_uid(row, context)
    if frame_uid in frame_ids:
        raise MediaManifestV2Error(f"duplicate frame_uid in {context}: {frame_uid}")
    normalized_video_id = _normal_video_id(video_id, context)
    prior_video_id = frame_video_ids.get(normalized_video_id)
    if prior_video_id is not None and prior_video_id != video_id:
        raise MediaManifestV2Error(f"duplicate normalized video_id in frame manifest: {video_id}")
    frame_ids[frame_uid] = video_id
    frame_video_ids[normalized_video_id] = video_id
    shard = _text(row, "image_shard_id", context)
    if not SHARD_RE.fullmatch(shard):
        raise MediaManifestV2Error(f"invalid image_shard_id in {context}")
    shards.add(shard)
    _safe_relative_path(_text(row, "path", context), "path", context)
    _positive_int(row, "bytes", context)
    _sha256_or_none(row.get("sha256"), context)
    _text(row, "backend", context)
    _text(row, "dataset", context)
    _text(row, "media_type", context)
    _nonnegative_int(row, "timestamp_ms", context)
    _text(row, "shot_id", context)
    return frame_uid, video_id


def _joined_video_row(
    *,
    frame_video_id: str,
    canonical: tuple[str, dict[str, Any]],
    ranged: tuple[str, dict[str, Any]],
    join_method: str,
) -> dict[str, Any]:
    canonical_id, canonical_row = canonical
    ranged_id, range_row = ranged
    context = f"video:{frame_video_id}"
    canonical_path = _safe_relative_path(
        _text(canonical_row, "member_path", context), "member_path", context
    )
    range_path = _safe_relative_path(_text(range_row, "path", context), "path", context)
    canonical_bytes = _positive_int(canonical_row, "bytes", context)
    range_bytes = _positive_int(range_row, "bytes", context)
    if canonical_bytes != range_bytes:
        raise MediaManifestV2Error(f"bytes mismatch for {frame_video_id}")
    if canonical_path != range_path:
        raise MediaManifestV2Error(f"source path mismatch for {frame_video_id}")
    canonical_dataset = str(canonical_row.get("dataset_id") or "").strip()
    range_dataset = _text(range_row, "dataset", context)
    if canonical_dataset and canonical_dataset != range_dataset:
        raise MediaManifestV2Error(f"dataset mismatch for {frame_video_id}")
    if str(range_row.get("backend") or "").strip().lower() != "kaggle_http_range":
        raise MediaManifestV2Error(f"unsupported range backend for {frame_video_id}")
    range_capable = range_row.get("range_capable")
    if type(range_capable) is not bool:
        raise MediaManifestV2Error(f"range_capable must be boolean for {frame_video_id}")
    probe_status = _text(range_row, "range_probe_status", context).lower()
    probe_attempts = _nonnegative_int(range_row, "range_probe_attempts", context)
    provenance = str(range_row.get("provenance_status") or "ENGINEERING_PROXY").strip().upper()
    source_path = str(canonical_row.get("metadata_video_path") or canonical_path).strip()
    source_path = _safe_relative_path(source_path, "source_path", context)
    fingerprint = _sha256_or_none(canonical_row.get("source_fingerprint"), context)
    remote_fingerprint = range_row.get("remote_content_fingerprint")
    if remote_fingerprint is None or remote_fingerprint == "":
        remote_fingerprint = None
    elif not isinstance(remote_fingerprint, dict):
        raise MediaManifestV2Error(
            f"remote_content_fingerprint must be an object for {frame_video_id}"
        )
    else:
        remote_fingerprint = {
            key: str(value)
            for key, value in remote_fingerprint.items()
            if key in {"etag", "x-goog-hash"} and value not in {None, ""}
        }
        remote_fingerprint = remote_fingerprint or None
    watch_url = _safe_preview_url(canonical_row.get("watch_url"), context)
    media_type = (
        str(canonical_row.get("media_type") or range_row.get("media_type") or "video/mp4")
        .strip()
        .lower()
    )
    if "/" not in media_type:
        raise MediaManifestV2Error(f"invalid media_type for {frame_video_id}")
    return {
        "backend": "kaggle_http_range",
        "bytes": canonical_bytes,
        "canonical_source_path": canonical_path,
        "dataset": range_dataset,
        "dataset_id": canonical_dataset or range_dataset,
        "join_method": join_method,
        "kind": "video",
        "media_info_id": canonical_id,
        "media_type": media_type,
        "member_path": canonical_path,
        "normalized_media_info_id": _normal_video_id(canonical_id, context),
        "path": range_path,
        "provenance_status": provenance,
        "range_capable": range_capable,
        "range_probe_attempts": probe_attempts,
        "range_probe_status": probe_status,
        "remote_content_fingerprint": remote_fingerprint,
        "sha256": None,
        "sha256_status": RAW_VIDEO_SHA256_STATUS,
        "source_fingerprint": fingerprint,
        "source_fingerprint_semantics": str(
            canonical_row.get("source_fingerprint_semantics")
            or "advisory_candidate_not_raw_mp4_sha256"
        ),
        "source_manifest_id": str(
            range_row.get("source_manifest_id") or canonical_row.get("source_manifest_id") or ""
        )
        or None,
        "source_path": source_path,
        "video_id": frame_video_id,
        "watch_url": watch_url,
    }


def _resolve_video_sources(
    frame_video_ids: dict[str, str],
    canonical: dict[str, tuple[str, dict[str, Any]]],
    ranged: dict[str, tuple[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    canonical_by_base = _index_by_basename(canonical, path_key="member_path")
    range_by_base = _index_by_basename(ranged, path_key="path")
    joined: dict[str, dict[str, Any]] = {}
    join_counts: Counter[str] = Counter()
    used_canonical: set[str] = set()
    used_range: set[str] = set()
    for normalized_frame_id, frame_video_id in sorted(frame_video_ids.items()):
        canonical_source = canonical.get(normalized_frame_id)
        range_source = ranged.get(normalized_frame_id)
        join_method = "video_id"
        if canonical_source is None or range_source is None:
            candidates_c = canonical_by_base.get(normalized_frame_id, [])
            candidates_r = range_by_base.get(normalized_frame_id, [])
            if len(candidates_c) != 1 or len(candidates_r) != 1:
                raise MediaManifestV2Error(f"missing or ambiguous video join for {frame_video_id}")
            canonical_source = candidates_c[0]
            range_source = candidates_r[0]
            join_method = "basename_unambiguous"
        canonical_id, _ = canonical_source
        range_id, _ = range_source
        canonical_key = _normal_video_id(canonical_id, f"canonical:{canonical_id}")
        range_key = _normal_video_id(range_id, f"range:{range_id}")
        if canonical_key in used_canonical or range_key in used_range:
            raise MediaManifestV2Error(f"video join reused by {frame_video_id}")
        used_canonical.add(canonical_key)
        used_range.add(range_key)
        joined[frame_video_id] = _joined_video_row(
            frame_video_id=frame_video_id,
            canonical=canonical_source,
            ranged=range_source,
            join_method=join_method,
        )
        join_counts[join_method] += 1
    if used_canonical != set(canonical) or used_range != set(ranged):
        raise MediaManifestV2Error("video_id set mismatch across frame/canonical/range inventories")
    return joined, join_counts


def validate_mixed_media_manifest(
    manifest_path: Path,
    *,
    expected_frame_count: int | None = 146121,
    expected_video_count: int | None = 873,
    expected_shard_count: int | None = 6,
) -> dict[str, Any]:
    """Validate one mixed JSONL without fetching any image or video."""

    path = Path(manifest_path).expanduser().resolve()
    frame_ids: set[str] = set()
    frame_video_ids: set[str] = set()
    video_ids: set[str] = set()
    shards: set[str] = set()
    kind_counts: Counter[str] = Counter()
    probe_statuses: Counter[str] = Counter()
    raw_sha_statuses: Counter[str] = Counter()
    for line_number, row in _read_jsonl(path):
        context = f"{path}:{line_number}"
        kind = str(row.get("kind") or "").lower()
        if kind not in {"frame", "video"}:
            raise MediaManifestV2Error(f"invalid kind in {context}")
        kind_counts[kind] += 1
        if kind == "frame":
            frame_uid, video_id, _ = _frame_uid(row, context)
            if frame_uid in frame_ids:
                raise MediaManifestV2Error(f"duplicate frame_uid in {context}")
            frame_ids.add(frame_uid)
            frame_video_ids.add(video_id)
            shard = _text(row, "image_shard_id", context)
            if not SHARD_RE.fullmatch(shard):
                raise MediaManifestV2Error(f"invalid image_shard_id in {context}")
            shards.add(shard)
            _safe_relative_path(_text(row, "path", context), "path", context)
            _positive_int(row, "bytes", context)
            digest = _sha256_or_none(row.get("sha256"), context)
            if digest is None:
                raise MediaManifestV2Error(f"frame sha256 is required in {context}")
            _text(row, "backend", context)
            _text(row, "dataset", context)
            _text(row, "media_type", context)
            _text(row, "shot_id", context)
            _nonnegative_int(row, "timestamp_ms", context)
        else:
            video_id = _text(row, "video_id", context)
            if video_id in video_ids:
                raise MediaManifestV2Error(f"duplicate video_id in {context}")
            video_ids.add(video_id)
            _safe_relative_path(_text(row, "path", context), "path", context)
            _safe_relative_path(_text(row, "source_path", context), "source_path", context)
            _safe_relative_path(_text(row, "member_path", context), "member_path", context)
            _positive_int(row, "bytes", context)
            raw_range = row.get("range_capable")
            if type(raw_range) is not bool:
                raise MediaManifestV2Error(f"range_capable must be boolean in {context}")
            probe_statuses[_text(row, "range_probe_status", context)] += 1
            raw_sha_status = _text(row, "sha256_status", context)
            if raw_sha_status != RAW_VIDEO_SHA256_STATUS:
                raise MediaManifestV2Error(f"raw video sha256 policy mismatch in {context}")
            raw_sha_statuses[raw_sha_status] += 1
            if row.get("sha256") is not None:
                raise MediaManifestV2Error(f"raw video sha256 must be null in {context}")
            if str(row.get("backend") or "").lower() != "kaggle_http_range":
                raise MediaManifestV2Error(f"unsupported video backend in {context}")
            _text(row, "dataset_id", context)
            _text(row, "media_info_id", context)
            _text(row, "normalized_media_info_id", context)
            _text(row, "join_method", context)
            _text(row, "provenance_status", context)
            if row.get("watch_url") not in {None, ""}:
                _safe_preview_url(row["watch_url"], context)
    if expected_frame_count is not None and len(frame_ids) != expected_frame_count:
        raise MediaManifestV2Error(
            f"frame count mismatch: expected {expected_frame_count}, got {len(frame_ids)}"
        )
    if expected_video_count is not None and len(video_ids) != expected_video_count:
        raise MediaManifestV2Error(
            f"video count mismatch: expected {expected_video_count}, got {len(video_ids)}"
        )
    if expected_shard_count is not None:
        expected_shards = {f"shard_{index:04d}" for index in range(expected_shard_count)}
        if shards != expected_shards:
            raise MediaManifestV2Error(
                f"shard set mismatch: expected {sorted(expected_shards)}, got {sorted(shards)}"
            )
    if frame_video_ids != video_ids:
        raise MediaManifestV2Error("frame/video video_id set mismatch")
    return {
        "status": "GREEN",
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(path),
        "manifest_sha256": _sha256_file(path),
        "frame_count": len(frame_ids),
        "video_count": len(video_ids),
        "total_count": sum(kind_counts.values()),
        "unique_frame_uid_count": len(frame_ids),
        "unique_video_id_count": len(video_ids),
        "shards": sorted(shards),
        "range_probe_status_counts": dict(sorted(probe_statuses.items())),
        "raw_video_sha256_status_counts": dict(sorted(raw_sha_statuses.items())),
    }


def build_mixed_media_manifest(
    *,
    frame_manifest: Path,
    range_manifest: Path,
    canonical_inventory: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    expected_frame_count: int | None = 146121,
    expected_video_count: int | None = 873,
    expected_shard_count: int | None = 6,
) -> dict[str, Any]:
    """Write a new mixed manifest from existing metadata-only inventories."""

    frame_manifest = Path(frame_manifest).expanduser().resolve()
    range_manifest = Path(range_manifest).expanduser().resolve()
    canonical_inventory = Path(canonical_inventory).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise MediaManifestV2Error(f"refusing to overwrite existing artifact: {output_path}")
    canonical = _index_video_rows(
        canonical_inventory, label="canonical inventory", require_kind=False
    )
    ranged = _index_video_rows(range_manifest, label="range manifest", require_kind=True)
    if expected_video_count is not None and (
        len(canonical) != expected_video_count or len(ranged) != expected_video_count
    ):
        raise MediaManifestV2Error(
            f"video input count mismatch: canonical={len(canonical)} range={len(ranged)}"
        )

    frame_ids: dict[str, str] = {}
    frame_video_ids: dict[str, str] = {}
    shards: set[str] = set()
    frame_count = 0
    output_video_ids: set[str] = set()
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for line_number, row in _read_jsonl(frame_manifest):
                context = f"frame manifest:{line_number}"
                _validate_frame_row(
                    row,
                    context=context,
                    frame_ids=frame_ids,
                    frame_video_ids=frame_video_ids,
                    shards=shards,
                )
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                frame_count += 1
        if expected_frame_count is not None and frame_count != expected_frame_count:
            raise MediaManifestV2Error(
                f"frame count mismatch: expected {expected_frame_count}, got {frame_count}"
            )
        if expected_shard_count is not None:
            expected_shards = {f"shard_{index:04d}" for index in range(expected_shard_count)}
            if shards != expected_shards:
                raise MediaManifestV2Error(
                    f"shard set mismatch: expected {sorted(expected_shards)}, got {sorted(shards)}"
                )
        joined, join_counts = _resolve_video_sources(frame_video_ids, canonical, ranged)
        with temporary.open("a", encoding="utf-8", newline="\n") as handle:
            for video_id in sorted(joined):
                handle.write(
                    json.dumps(joined[video_id], ensure_ascii=False, sort_keys=True) + "\n"
                )
                output_video_ids.add(video_id)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    validation = validate_mixed_media_manifest(
        output_path,
        expected_frame_count=expected_frame_count,
        expected_video_count=expected_video_count,
        expected_shard_count=expected_shard_count,
    )
    if metadata_path is None:
        metadata_path = output_path.with_name(f"{output_path.stem}.meta.json")
    metadata_path = Path(metadata_path).expanduser().resolve()
    range_status_counts: Counter[str] = Counter(
        row["range_probe_status"] for row in joined.values()
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "status": "ENGINEERING_PROXY",
        "quality_status": "UNVALIDATED",
        "identity": "frame_uid=video_id:source_frame_idx",
        "faiss_identity_policy": "faiss_row_is_index_local_only",
        "raw_video_sha256_policy": RAW_VIDEO_SHA256_STATUS,
        "manifest_path": str(output_path),
        "manifest_sha256": validation["manifest_sha256"],
        "frame_count": frame_count,
        "video_count": len(output_video_ids),
        "total_count": frame_count + len(output_video_ids),
        "unique_frame_uid_count": len(frame_ids),
        "unique_video_id_count": len(output_video_ids),
        "shards": sorted(shards),
        "join_counts": dict(sorted(join_counts.items())),
        "range_capable_video_count": sum(bool(row["range_capable"]) for row in joined.values()),
        "range_probe_status_counts": dict(sorted(range_status_counts.items())),
        "skipped_range_probe_video_ids": sorted(
            row["video_id"] for row in joined.values() if row["range_probe_status"] == "skipped"
        ),
        "input_paths": {
            "frame_manifest": str(frame_manifest),
            "range_manifest": str(range_manifest),
            "canonical_inventory": str(canonical_inventory),
        },
        "input_sha256": {
            "frame_manifest": _sha256_file(frame_manifest),
            "range_manifest": _sha256_file(range_manifest),
            "canonical_inventory": _sha256_file(canonical_inventory),
        },
        "input_row_counts": {
            "frame_manifest": frame_count,
            "range_manifest": len(ranged),
            "canonical_inventory": len(canonical),
        },
        "authoritative_inputs_unchanged": True,
        "payload_status": "METADATA_ONLY_NO_MEDIA_PAYLOAD_COPIED",
        "source_fingerprint_policy": "advisory_candidate_not_raw_mp4_sha256",
        "note": "New mixed bridge; authoritative frame/video inputs remain immutable.",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-manifest", required=True, type=Path)
    parser.add_argument("--range-manifest", required=True, type=Path)
    parser.add_argument("--canonical-inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    report = build_mixed_media_manifest(
        frame_manifest=args.frame_manifest,
        range_manifest=args.range_manifest,
        canonical_inventory=args.canonical_inventory,
        output_path=args.output,
        metadata_path=args.metadata,
    )
    print(
        json.dumps(
            {
                "manifest_path": report["manifest_path"],
                "manifest_sha256": report["manifest_sha256"],
                "frame_count": report["frame_count"],
                "video_count": report["video_count"],
                "total_count": report["total_count"],
                "status": report["status"],
                "quality_status": report["quality_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
