"""Build a manifest for the public NHANGIOI/AIC2026 Hugging Face dataset.

The builder consumes only metadata: the canonical video inventory, the
canonical frame catalog, and the Hugging Face file listing.  It never downloads
an image or video payload.  The resulting rows are consumed by
``RemoteMediaResolver`` using the fixed ``huggingface_http_range`` backend.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

HUGGINGFACE_RANGE_BACKEND = "huggingface_http_range"
SCHEMA_VERSION = "hcmaic-huggingface-media-v1"
DEFAULT_KEYFRAME_PREFIX = "processed/keyframes"
DEFAULT_CATALOG_KEYFRAME_PREFIX = "processed/keyframes"
_DATASET_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HF_API_ROOT = "https://huggingface.co/api/datasets"
_MAX_API_BYTES = 32 * 1024 * 1024
_HF_TREE_PAGE_LIMIT = 1000
_HF_TREE_MAX_PAGES = 1000
_NEXT_LINK_RE = re.compile(r"<([^>]+)>;\s*rel=\"next\"")


class HuggingFaceMediaManifestError(ValueError):
    """The canonical inputs cannot produce a safe HF media contract."""


def _dataset_id(value: Any) -> str:
    dataset = str(value or "").strip()
    if not _DATASET_RE.fullmatch(dataset):
        raise HuggingFaceMediaManifestError("Hugging Face dataset must be owner/name")
    return dataset


def _revision(value: Any) -> str:
    revision = str(value or "").strip()
    if not _REVISION_RE.fullmatch(revision):
        raise HuggingFaceMediaManifestError(
            "Hugging Face revision must be a simple branch, tag, or commit identifier"
        )
    return revision


def _safe_member(value: Any, *, field: str = "member path") -> str:
    member = str(value or "").replace("\\", "/")
    parsed = PurePosixPath(member)
    if (
        not member
        or parsed.is_absolute()
        or re.match(r"^[A-Za-z]:", member)
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise HuggingFaceMediaManifestError(f"unsafe {field}")
    return "/".join(parsed.parts)


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise HuggingFaceMediaManifestError(f"{label} not found: {source}")
    rows: list[dict[str, Any]] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise HuggingFaceMediaManifestError(f"cannot read {label}: {source}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HuggingFaceMediaManifestError(
                f"invalid JSON in {label} at line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise HuggingFaceMediaManifestError(
                f"{label} row at line {line_number} must be an object"
            )
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_catalog_path(value: Any) -> str:
    path = _safe_member(value, field="catalog keyframe path")
    if path.startswith("data/"):
        path = path[5:]
    return path


def _load_video_inventory(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(
        _read_jsonl(path, label="canonical video inventory"), start=1
    ):
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise HuggingFaceMediaManifestError(
                f"canonical video inventory missing video_id at row {line_number}"
            )
        if video_id in indexed:
            raise HuggingFaceMediaManifestError(f"duplicate video_id: {video_id}")
        try:
            bytes_count = int(row.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise HuggingFaceMediaManifestError(
                f"invalid bytes for canonical video {video_id}"
            ) from exc
        if bytes_count < 1:
            raise HuggingFaceMediaManifestError(
                f"canonical video bytes must be positive for {video_id}"
            )
        member_path = _safe_member(row.get("member_path"), field="canonical video path")
        indexed[video_id] = {
            **row,
            "video_id": video_id,
            "bytes": bytes_count,
            "member_path": member_path,
        }
    if not indexed:
        raise HuggingFaceMediaManifestError("canonical video inventory is empty")
    return indexed


def _load_frame_catalog(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(_read_jsonl(path, label="frame catalog"), start=1):
        frame_uid = str(row.get("frame_uid") or "").strip()
        video_id = str(row.get("video_id") or "").strip()
        try:
            source_frame_idx = int(row.get("source_frame_idx"))
        except (TypeError, ValueError) as exc:
            raise HuggingFaceMediaManifestError(
                f"invalid source_frame_idx at frame catalog row {line_number}"
            ) from exc
        if (
            not frame_uid
            or not video_id
            or source_frame_idx < 0
            or frame_uid != f"{video_id}:{source_frame_idx}"
        ):
            raise HuggingFaceMediaManifestError(
                f"frame identity mismatch at frame catalog row {line_number}"
            )
        if frame_uid in indexed:
            raise HuggingFaceMediaManifestError(f"duplicate frame_uid: {frame_uid}")
        catalog_path = _normalize_catalog_path(row.get("keyframe_path"))
        indexed[catalog_path] = {
            **row,
            "frame_uid": frame_uid,
            "video_id": video_id,
            "source_frame_idx": source_frame_idx,
            "_normalized_path": catalog_path,
        }
    return indexed


def _load_base_video_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Load video rows from an existing manifest without rewriting them."""

    indexed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(
        _read_jsonl(path, label="base video media manifest"), start=1
    ):
        if str(row.get("kind") or "").strip().lower() != "video":
            continue
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            raise HuggingFaceMediaManifestError(
                f"base video media manifest missing video_id at row {line_number}"
            )
        if video_id in indexed:
            raise HuggingFaceMediaManifestError(
                f"duplicate base video media row for video_id: {video_id}"
            )
        if str(row.get("path") or row.get("member_path") or "").strip() == "":
            raise HuggingFaceMediaManifestError(
                f"base video media manifest missing path for {video_id}"
            )
        indexed[video_id] = dict(row)
    if not indexed:
        raise HuggingFaceMediaManifestError("base video media manifest has no video rows")
    return indexed


def _normalize_file_paths(file_paths: Iterable[str]) -> list[str]:
    normalized: set[str] = set()
    for value in file_paths:
        path = _safe_member(value, field="Hugging Face file path")
        normalized.add(path)
    return sorted(normalized)


def _normalize_tree_prefix(value: Any, *, field: str) -> str:
    prefix = _safe_member(value, field=field).rstrip("/")
    if not prefix or prefix == "raw_video" or prefix.startswith("raw_video/"):
        raise HuggingFaceMediaManifestError(
            f"{field} must identify a keyframe tree, not raw_video"
        )
    return prefix


def _is_tree_member(path: str, prefix: str) -> bool:
    return path.startswith(f"{prefix}/")


def _catalog_path_for_member(
    member_path: str,
    *,
    member_prefix: str,
    catalog_prefix: str,
) -> str:
    if not _is_tree_member(member_path, member_prefix):
        raise HuggingFaceMediaManifestError(
            f"keyframe member is outside configured prefix: {member_path}"
        )
    return f"{catalog_prefix}{member_path[len(member_prefix):]}"


def _next_huggingface_tree_url(
    headers: Any,
    *,
    dataset: str,
) -> str | None:
    """Return a validated HF tree pagination URL, if one is advertised."""

    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        values = get_all("Link") or []
        link_header = ", ".join(str(value) for value in values)
    else:
        link_header = str(headers.get("Link", "") or "")
    match = _NEXT_LINK_RE.search(link_header)
    if match is None:
        return None
    candidate = match.group(1)
    parsed = urllib.parse.urlparse(candidate)
    expected_path_prefix = f"/api/datasets/{urllib.parse.quote(dataset, safe='/')}/tree/"
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or not parsed.path.startswith(expected_path_prefix)
    ):
        raise HuggingFaceMediaManifestError(
            "Hugging Face tree pagination returned an unsafe next URL"
        )
    return candidate


def _fetch_huggingface_tree_paths(
    dataset: str,
    *,
    revision: str,
    prefix: str,
    timeout_seconds: float,
) -> list[str]:
    """Read one paginated HF tree prefix without reading media payloads."""

    encoded_dataset = urllib.parse.quote(dataset, safe="/")
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_prefix = urllib.parse.quote(prefix, safe="")
    query = urllib.parse.urlencode(
        {"recursive": "true", "expand": "false", "limit": _HF_TREE_PAGE_LIMIT}
    )
    next_url = f"{_HF_API_ROOT}/{encoded_dataset}/tree/{encoded_revision}/{encoded_prefix}?{query}"
    paths: list[str] = []
    seen_urls: set[str] = set()
    page_count = 0
    while next_url:
        if next_url in seen_urls:
            raise HuggingFaceMediaManifestError(
                f"Hugging Face tree pagination loop for prefix {prefix}"
            )
        seen_urls.add(next_url)
        page_count += 1
        if page_count > _HF_TREE_MAX_PAGES:
            raise HuggingFaceMediaManifestError(
                f"Hugging Face tree pagination exceeds safety limit for {prefix}"
            )
        request = urllib.request.Request(
            next_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "hcmaic-hf-manifest/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read(_MAX_API_BYTES + 1)
                next_url = _next_huggingface_tree_url(
                    response.headers,
                    dataset=dataset,
                )
        except urllib.error.HTTPError as exc:
            # The raw-video tree is optional when a pinned base video manifest
            # is supplied; a missing keyframe tree remains a hard failure.
            if exc.code == 404 and prefix == "raw_video":
                return []
            raise HuggingFaceMediaManifestError(
                f"cannot read Hugging Face tree prefix: {prefix}"
            ) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise HuggingFaceMediaManifestError(
                f"cannot read Hugging Face tree prefix: {prefix}"
            ) from exc
        if len(body) > _MAX_API_BYTES:
            raise HuggingFaceMediaManifestError(
                f"Hugging Face tree page exceeds safety limit for {prefix}"
            )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HuggingFaceMediaManifestError(
                f"Hugging Face tree page is not valid JSON for {prefix}"
            ) from exc
        if not isinstance(payload, list):
            raise HuggingFaceMediaManifestError(
                f"Hugging Face tree page is not an array for {prefix}"
            )
        for item in payload:
            if (
                isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and item.get("type") == "file"
            ):
                paths.append(item["path"])
    return paths


def fetch_huggingface_file_paths(
    dataset_id: str,
    *,
    revision: str = "main",
    timeout_seconds: float = 30.0,
    keyframe_prefix: str = DEFAULT_KEYFRAME_PREFIX,
) -> list[str]:
    """Read paginated media paths without reading payload bytes.

    The dataset-info ``siblings`` array is not sufficient for large repos: it
    can be truncated before all keyframe files are listed. Prefix-scoped tree
    pagination keeps the manifest identity-complete while remaining metadata-
    only.
    """

    dataset = _dataset_id(dataset_id)
    ref = _revision(revision)
    keyframe_tree_prefix = _normalize_tree_prefix(
        keyframe_prefix,
        field="Hugging Face keyframe prefix",
    )
    paths: list[str] = []
    for prefix in (keyframe_tree_prefix, "raw_video"):
        paths.extend(
            _fetch_huggingface_tree_paths(
                dataset,
                revision=ref,
                prefix=prefix,
                timeout_seconds=timeout_seconds,
            )
        )
    if not paths:
        raise HuggingFaceMediaManifestError("Hugging Face media tree listing is empty")
    return _normalize_file_paths(paths)


def _video_file_map(file_paths: list[str]) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for path in file_paths:
        match = re.fullmatch(r"raw_video/[^/]+/([^/]+)\.mp4", path)
        if not match:
            continue
        video_id = match.group(1)
        if video_id in indexed:
            raise HuggingFaceMediaManifestError(
                f"duplicate HF raw video member for video_id: {video_id}"
            )
        indexed[video_id] = path
    return indexed


def _frame_row(
    catalog_row: dict[str, Any],
    *,
    member_path: str,
    dataset_id: str,
    revision: str,
) -> dict[str, Any]:
    return {
        "backend": HUGGINGFACE_RANGE_BACKEND,
        "bytes": None,
        "canonical_source_path": str(catalog_row["keyframe_path"]).replace("\\", "/"),
        "dataset": dataset_id,
        "dataset_id": dataset_id,
        "frame_uid": catalog_row["frame_uid"],
        "join_method": "catalog_keyframe_path",
        "kind": "frame",
        "member_path": member_path,
        "media_type": "image/jpeg",
        "path": member_path,
        "provenance_status": "ENGINEERING_PROXY",
        "range_capable": False,
        "revision": revision,
        "sha256": None,
        "sha256_status": "NOT_PROVIDED_BY_HF_METADATA",
        "source_path": str(catalog_row["keyframe_path"]).replace("\\", "/"),
        "source_frame_idx": catalog_row["source_frame_idx"],
        "shot_id": catalog_row.get("shot_id"),
        "timestamp_ms": catalog_row.get("timestamp_ms"),
        "video_id": catalog_row["video_id"],
    }


def _video_row(
    canonical_row: dict[str, Any],
    *,
    member_path: str,
    dataset_id: str,
    revision: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "backend": HUGGINGFACE_RANGE_BACKEND,
        "bytes": canonical_row["bytes"],
        "canonical_source_path": canonical_row["member_path"],
        "dataset": dataset_id,
        "dataset_id": dataset_id,
        "join_method": "video_id",
        "kind": "video",
        "member_path": member_path,
        "media_type": str(canonical_row.get("media_type") or "video/mp4").lower(),
        "path": member_path,
        "provenance_status": "ENGINEERING_PROXY",
        "range_capable": True,
        "range_probe_attempts": 0,
        "range_probe_status": "platform_contract",
        "revision": revision,
        "sha256": None,
        "sha256_status": "NOT_REQUIRED_RAW_VIDEO",
        "source_kind": "huggingface",
        "source_manifest_id": canonical_row.get("source_manifest_id"),
        "source_path": canonical_row["member_path"],
        "source_fingerprint_semantics": canonical_row.get("source_fingerprint_semantics"),
        "video_id": canonical_row["video_id"],
    }
    fingerprint = canonical_row.get("source_fingerprint")
    if isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint):
        row["source_fingerprint"] = fingerprint.lower()
    return row


def build_huggingface_media_manifest(
    *,
    canonical_video_inventory: Path,
    frame_catalog: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    dataset_id: str = "NHANGIOI/AIC2026",
    revision: str = "main",
    file_paths: Iterable[str] | None = None,
    base_video_manifest: Path | None = None,
    keyframe_prefix: str = DEFAULT_KEYFRAME_PREFIX,
    catalog_keyframe_prefix: str = DEFAULT_CATALOG_KEYFRAME_PREFIX,
) -> dict[str, Any]:
    """Create a deterministic HF video + available-keyframe manifest.

    ``base_video_manifest`` is used when a newer HF revision contains the
    processed keyframes but not the raw videos.  Its video rows are copied
    unchanged, preserving their pinned video revision and range contract.
    """

    dataset = _dataset_id(dataset_id)
    ref = _revision(revision)
    keyframe_tree_prefix = _normalize_tree_prefix(
        keyframe_prefix,
        field="Hugging Face keyframe prefix",
    )
    catalog_tree_prefix = _normalize_tree_prefix(
        catalog_keyframe_prefix,
        field="catalog keyframe prefix",
    )
    canonical_path = Path(canonical_video_inventory).expanduser().resolve()
    catalog_path = Path(frame_catalog).expanduser().resolve()
    canonical = _load_video_inventory(canonical_path)
    catalog = _load_frame_catalog(catalog_path)
    files = _normalize_file_paths(
        fetch_huggingface_file_paths(
            dataset,
            revision=ref,
            keyframe_prefix=keyframe_tree_prefix,
        )
        if file_paths is None
        else file_paths
    )
    raw_videos = _video_file_map(files)
    base_videos: dict[str, dict[str, Any]] | None = None
    base_video_manifest_path: Path | None = None
    if base_video_manifest is not None:
        base_video_manifest_path = Path(base_video_manifest).expanduser().resolve()
        base_videos = _load_base_video_manifest(base_video_manifest_path)
        missing = sorted(set(canonical) - set(base_videos))
        if missing:
            sample = ", ".join(missing[:5])
            raise HuggingFaceMediaManifestError(
                f"base video manifest is missing {len(missing)} canonical video(s): {sample}"
            )
    else:
        missing = sorted(set(canonical) - set(raw_videos))
        if missing:
            sample = ", ".join(missing[:5])
            raise HuggingFaceMediaManifestError(
                f"missing HF raw video members for {len(missing)} video(s): {sample}"
            )

    frame_rows: list[dict[str, Any]] = []
    matched_catalog_paths: set[str] = set()
    for member_path in files:
        if not _is_tree_member(member_path, keyframe_tree_prefix):
            continue
        catalog_member_path = _catalog_path_for_member(
            member_path,
            member_prefix=keyframe_tree_prefix,
            catalog_prefix=catalog_tree_prefix,
        )
        catalog_row = catalog.get(catalog_member_path)
        if catalog_row is None:
            continue
        frame_rows.append(
            _frame_row(
                catalog_row,
                member_path=member_path,
                dataset_id=dataset,
                revision=ref,
            )
        )
        matched_catalog_paths.add(catalog_member_path)

    if base_videos is not None:
        video_rows = [dict(base_videos[video_id]) for video_id in sorted(canonical)]
    else:
        video_rows = [
            _video_row(
                canonical[video_id],
                member_path=raw_videos[video_id],
                dataset_id=dataset,
                revision=ref,
            )
            for video_id in sorted(canonical)
        ]
    frame_rows.sort(key=lambda row: row["frame_uid"])
    rows = [*frame_rows, *video_rows]
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    report: dict[str, Any] = {
        "authoritative_inputs_unchanged": True,
        "backend": HUGGINGFACE_RANGE_BACKEND,
        "canonical_video_inventory_sha256": _sha256_file(canonical_path),
        "canonical_video_inventory_row_count": len(canonical),
        "dataset_id": dataset,
        "dataset_file_listing_sha256": hashlib.sha256(
            ("\n".join(files) + "\n").encode("utf-8")
        ).hexdigest(),
        "dataset_file_listing_count": len(files),
        "frame_catalog_sha256": _sha256_file(catalog_path),
        "frame_catalog_row_count": len(catalog),
        "frame_count": len(frame_rows),
        "identity": "frame_uid=video_id:source_frame_idx",
        "keyframe_prefix": keyframe_tree_prefix,
        "catalog_keyframe_prefix": catalog_tree_prefix,
        "keyframe_catalog_coverage_count": len(matched_catalog_paths),
        "keyframe_catalog_coverage_status": (
            "PARTIAL_DATASET_LISTING" if len(matched_catalog_paths) < len(catalog) else "COMPLETE"
        ),
        "manifest_path": str(output),
        "manifest_sha256": manifest_sha256,
        "base_video_manifest_sha256": (
            _sha256_file(base_video_manifest_path) if base_video_manifest_path is not None else None
        ),
        "base_video_manifest_path": (
            str(base_video_manifest_path) if base_video_manifest_path is not None else None
        ),
        "payload_status": "METADATA_ONLY_REMOTE_PAYLOAD_NOT_DOWNLOADED",
        "quality_status": "UNVALIDATED",
        "raw_video_count_in_dataset": len(raw_videos),
        "raw_video_join_status": (
            "PRESERVED_BASE_VIDEO_MANIFEST" if base_videos is not None else "GREEN"
        ),
        "revision": ref,
        "schema_version": SCHEMA_VERSION,
        "status": "ENGINEERING_PROXY",
        "video_count": len(video_rows),
        "video_bytes_source": (
            "base_video_manifest"
            if base_videos is not None
            else "canonical_video_inventory; HF tree listing has no byte sizes"
        ),
        "video_range_contract": (
            "Hugging Face resolve endpoint observed with bounded HTTP Range on one sample; "
            "runtime remains fail-closed on non-206 responses"
        ),
    }
    if metadata_path is None:
        metadata_path = output.with_name(f"{output.stem}.meta.json")
    metadata = Path(metadata_path).expanduser().resolve()
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-video-inventory", required=True, type=Path)
    parser.add_argument("--frame-catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--dataset", default="NHANGIOI/AIC2026")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--keyframe-prefix",
        default=DEFAULT_KEYFRAME_PREFIX,
        help="HF tree prefix containing the keyframe JPEG members",
    )
    parser.add_argument(
        "--catalog-keyframe-prefix",
        default=DEFAULT_CATALOG_KEYFRAME_PREFIX,
        help="Canonical catalog prefix used to map the selected HF keyframe tree",
    )
    parser.add_argument(
        "--base-video-manifest",
        type=Path,
        help=(
            "Existing JSONL manifest whose video rows are preserved when this revision "
            "has keyframes only"
        ),
    )
    args = parser.parse_args()
    report = build_huggingface_media_manifest(
        canonical_video_inventory=args.canonical_video_inventory,
        frame_catalog=args.frame_catalog,
        output_path=args.output,
        metadata_path=args.metadata,
        dataset_id=args.dataset,
        revision=args.revision,
        base_video_manifest=args.base_video_manifest,
        keyframe_prefix=args.keyframe_prefix,
        catalog_keyframe_prefix=args.catalog_keyframe_prefix,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
