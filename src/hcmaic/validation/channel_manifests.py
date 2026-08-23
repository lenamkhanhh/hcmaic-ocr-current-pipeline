"""Metadata-only validation for visual/OCR channel manifests.

The validator accepts the normalized ``hcmaic-channel-manifest-v1`` contract
and can also audit older execution manifests without rewriting them.  A raw
OCR crop output may legitimately contain several crops for one frame; only a
frame-level index map is required to have unique ``frame_uid`` values.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "hcmaic-channel-manifest-validation-v1"
_FRAME_UID_RE = re.compile(r"^[^:\s]+:[0-9]+$")
_HASH_KEY_RE = re.compile(r"(?:sha256|hash)$", re.IGNORECASE)


class ChannelManifestValidationError(ValueError):
    """Raised when a manifest cannot be read as a JSON object."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChannelManifestValidationError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ChannelManifestValidationError(f"manifest must be a JSON object: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value = value.get("rows", value.get("items", value))
            if not isinstance(value, list):
                raise ChannelManifestValidationError(f"row JSON must be a list: {path}")
            rows = value
        else:
            rows = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ChannelManifestValidationError(
                        f"invalid JSONL row at {path}:{line_number}"
                    ) from exc
    except OSError as exc:
        raise ChannelManifestValidationError(f"cannot read rows {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ChannelManifestValidationError(f"all rows must be objects: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first(value: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        candidate = value.get(name)
        if candidate is not None and candidate != "":
            return candidate
    return None


def _first_int(value: Mapping[str, Any], names: Sequence[str]) -> int | None:
    candidate = _first(value, names)
    if candidate is None or isinstance(candidate, bool):
        return None
    try:
        parsed = int(candidate)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _nested_model_values(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for key in ("model", "recognizer", "embedding", "encoder"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            values.append(value)
    return values


def _model_fields(manifest: Mapping[str, Any]) -> tuple[str | None, str | None, int | None]:
    model_id = _first(
        manifest,
        ("model_id", "recognizer_model", "embedding_model", "encoder_model"),
    )
    revision = _first(
        manifest,
        ("model_revision", "recognizer_revision", "revision", "code_revision"),
    )
    dimension = _first_int(
        manifest,
        ("dimension", "embedding_dimension", "vector_dimension", "embedding_dim"),
    )
    if isinstance(manifest.get("model"), str) and model_id is None:
        model_id = manifest["model"]
    for nested in _nested_model_values(manifest):
        model_id = model_id or _first(nested, ("model_id", "model", "name"))
        revision = revision or _first(nested, ("model_revision", "revision", "version"))
        dimension = (
            dimension
            if dimension is not None
            else _first_int(nested, ("dimension", "embedding_dimension", "vector_dimension"))
        )
    return (
        None if model_id is None else str(model_id),
        None if revision is None else str(revision),
        dimension,
    )


def _row_count(manifest: Mapping[str, Any]) -> int | None:
    return _first_int(
        manifest,
        (
            "row_count",
            "ntotal",
            "output_rows",
            "parseq_row_count",
            "line_count",
            "status_rows",
            "statuses",
            "crops",
            "crop_count",
            "lines",
        ),
    )


def _hash_evidence(value: Any, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            if _HASH_KEY_RE.search(str(key)) and isinstance(child, str) and child:
                found[label] = child
            elif isinstance(child, Mapping):
                found.update(_hash_evidence(child, label))
    return found


def _identity_text(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("identity_key") or manifest.get("identity") or "").strip()


def _identity_check(identity: str) -> str:
    normalized = identity.lower().replace(" ", "")
    if normalized == "frame_uid" or "frame_uid=video_id:source_frame_idx" in normalized:
        return "PASS"
    if "frame_uid" in normalized:
        return "WARN"
    return "FAIL"


def _issue(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


def _sidecar_from_manifest(
    manifest_path: Path, manifest: Mapping[str, Any], key: str
) -> Path | None:
    raw = manifest.get(key)
    if not raw:
        return None
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    return candidate.resolve()


def _uid_rows(rows: list[dict[str, Any]]) -> tuple[list[str], int, list[int]]:
    uids: list[str] = []
    missing: list[int] = []
    for index, row in enumerate(rows):
        value = str(row.get("frame_uid") or "").strip()
        if not value or not _FRAME_UID_RE.fullmatch(value):
            missing.append(index)
            continue
        uids.append(value)
    return uids, len(uids) - len(set(uids)), missing


def _validate_channel(
    channel: str,
    manifest_path: Path,
    *,
    row_path: Path | None,
    index_map_path: Path | None,
    expected_dimension: int | None,
    catalog_uids: set[str] | None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    model_id, model_revision, dimension = _model_fields(manifest)
    declared_rows = _row_count(manifest)
    identity = _identity_text(manifest)
    hashes = _hash_evidence(manifest)
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not manifest.get("schema_version"):
        issues.append(_issue("MISSING_SCHEMA_VERSION", "schema_version is required"))
    if model_id is None:
        issues.append(_issue("MISSING_MODEL", "model_id/recognizer_model is required"))
    if model_revision is None:
        issues.append(_issue("MISSING_MODEL_REVISION", "model revision is required"))
    if declared_rows is None:
        issues.append(_issue("MISSING_ROW_COUNT", "manifest row count is required"))
    if not hashes:
        issues.append(_issue("MISSING_HASH_EVIDENCE", "no sha256/hash field was declared"))
    if not manifest.get("quality_status"):
        issues.append(_issue("MISSING_QUALITY_STATUS", "quality_status must be explicit"))

    dimension_required = channel == "visual" or expected_dimension is not None
    if dimension_required and dimension is None:
        issues.append(
            _issue("MISSING_DIMENSION", "embedding dimension is required for this channel")
        )
    if expected_dimension is not None and dimension != expected_dimension:
        issues.append(
            _issue(
                "DIMENSION_MISMATCH",
                f"declared dimension {dimension!r} != expected {expected_dimension}",
            )
        )
    identity_status = _identity_check(identity)
    if identity_status == "FAIL":
        issues.append(_issue("IDENTITY_CONTRACT_INVALID", "frame_uid identity contract is missing"))
    elif identity_status == "WARN":
        warnings.append(
            _issue(
                "IDENTITY_PROVENANCE_ONLY",
                "manifest mentions frame_uid but does not declare "
                "frame_uid=video_id:source_frame_idx",
            )
        )

    row_uids: list[str] = []
    row_uid_set: set[str] = set()
    row_checks: dict[str, Any] = {"status": "NOT_CHECKED"}
    if row_path is not None:
        rows = _read_rows(row_path)
        row_uids, duplicate_count, missing_rows = _uid_rows(rows)
        row_uid_set = set(row_uids)
        crop_rows = str(manifest.get("row_granularity", "frame")) == "crop"
        row_checks = {
            "status": (
                "PASS" if not missing_rows and (duplicate_count == 0 or crop_rows) else "FAIL"
            ),
            "path": str(row_path),
            "row_count": len(rows),
            "missing_frame_uid_rows": missing_rows,
            "duplicate_frame_uid_count": duplicate_count,
        }
        if declared_rows is not None and len(rows) != declared_rows:
            issues.append(
                _issue(
                    "ROW_COUNT_MISMATCH",
                    f"rows contain {len(rows)} entries but manifest declares {declared_rows}",
                )
            )
        # Crop-level OCR output can repeat a frame UID.  A caller that supplies
        # it as an index map is still checked strictly below.
        if missing_rows:
            issues.append(
                _issue("MISSING_FRAME_UID", f"rows missing canonical UID at {missing_rows}")
            )
        if duplicate_count and not crop_rows:
            issues.append(
                _issue("DUPLICATE_FRAME_UID", "frame-level rows contain duplicate frame_uid")
            )

    index_checks: dict[str, Any] = {"status": "NOT_CHECKED"}
    if index_map_path is not None:
        index_rows = _read_rows(index_map_path)
        index_uids, index_duplicate_count, index_missing_rows = _uid_rows(index_rows)
        index_uid_set = set(index_uids)
        reference_uids = row_uid_set or catalog_uids
        index_issues: list[dict[str, str]] = []
        if index_missing_rows:
            index_issues.append(
                _issue(
                    "MISSING_INDEX_FRAME_UID",
                    f"index map missing canonical UID at {index_missing_rows}",
                )
            )
        if index_duplicate_count:
            index_issues.append(
                _issue("DUPLICATE_INDEX_FRAME_UID", "index map contains duplicate frame_uid")
            )
        if declared_rows is not None and len(index_rows) != declared_rows:
            index_issues.append(
                _issue(
                    "INDEX_ROW_COUNT_MISMATCH",
                    f"index map contains {len(index_rows)} entries but manifest "
                    f"declares {declared_rows}",
                )
            )
        missing_mapping: list[str] = []
        extra_mapping: list[str] = []
        coverage_status = "NOT_CHECKED_NO_REFERENCE"
        if reference_uids is not None:
            missing_mapping = sorted(reference_uids - index_uid_set)
            extra_mapping = sorted(index_uid_set - reference_uids)
            coverage_status = "PASS" if not missing_mapping and not extra_mapping else "FAIL"
            if missing_mapping:
                index_issues.append(
                    _issue(
                        "MISSING_INDEX_MAPPING",
                        f"missing {len(missing_mapping)} frame_uid mappings",
                    )
                )
            if extra_mapping:
                index_issues.append(
                    _issue(
                        "EXTRA_INDEX_MAPPING",
                        f"found {len(extra_mapping)} unmapped frame_uid values",
                    )
                )
        if index_issues:
            issues.extend(index_issues)
        index_checks = {
            "status": "PASS" if not index_issues else "FAIL",
            "path": str(index_map_path),
            "row_count": len(index_rows),
            "coverage": coverage_status,
            "missing_frame_uid": index_missing_rows,
            "duplicate_frame_uid_count": index_duplicate_count,
            "missing_mapping_count": len(missing_mapping),
            "extra_mapping_count": len(extra_mapping),
            "faiss_row_policy": "DIAGNOSTIC_ONLY",
        }

    checks = {
        "schema": "PASS" if manifest.get("schema_version") else "FAIL",
        "model": "PASS" if model_id else "FAIL",
        "model_revision": "PASS" if model_revision else "FAIL",
        "dimension": "PASS" if not dimension_required or dimension is not None else "FAIL",
        "row_count": "PASS" if declared_rows is not None else "FAIL",
        "hashes": "PASS" if hashes else "FAIL",
        "identity": identity_status,
        "rows": row_checks["status"],
        "index_to_manifest_coverage": index_checks.get("coverage", index_checks["status"]),
    }
    return {
        "channel": channel,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "schema_version": manifest.get("schema_version"),
        "status_declared": manifest.get("status") or manifest.get("execution_status"),
        "quality_status": manifest.get("quality_status", "UNVALIDATED"),
        "model_id": model_id,
        "model_revision": model_revision,
        "dimension": dimension,
        "row_count": declared_rows,
        "identity_key": "frame_uid",
        "identity_text": identity,
        "hash_evidence": hashes,
        "faiss_row_policy": "DIAGNOSTIC_ONLY",
        "checks": checks,
        "rows": row_checks,
        "index": index_checks,
        "issues": issues,
        "warnings": warnings,
    }


def validate_channel_manifests(
    manifest_paths: Mapping[str, Path],
    *,
    row_paths: Mapping[str, Path] | None = None,
    index_map_paths: Mapping[str, Path] | None = None,
    expected_dimensions: Mapping[str, int] | None = None,
    catalog_uids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate channel manifests and optional frame-level row/index maps.

    Inputs are read-only.  ``catalog_uids`` is accepted as a precomputed set so
    callers can audit a bounded fixture without forcing a full index/catalog
    load.  The report is always an engineering artifact and never a quality
    promotion.
    """

    if not manifest_paths:
        raise ChannelManifestValidationError("at least one channel manifest is required")
    rows_by_channel = row_paths or {}
    indexes_by_channel = index_map_paths or {}
    dimensions = expected_dimensions or {}
    channels: dict[str, Any] = {}
    for raw_channel, raw_path in manifest_paths.items():
        channel = str(raw_channel).strip().lower()
        if channel not in {"visual", "ocr"}:
            raise ChannelManifestValidationError(f"unsupported channel: {raw_channel!r}")
        manifest_path = Path(raw_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise ChannelManifestValidationError(f"manifest not found: {manifest_path}")
        channel_manifest = _read_json(manifest_path)
        row_path = (
            Path(rows_by_channel[channel]).expanduser().resolve()
            if channel in rows_by_channel
            else _sidecar_from_manifest(manifest_path, channel_manifest, "rows_path")
        )
        index_path = (
            Path(indexes_by_channel[channel]).expanduser().resolve()
            if channel in indexes_by_channel
            else _sidecar_from_manifest(manifest_path, channel_manifest, "index_map_path")
        )
        channels[channel] = _validate_channel(
            channel,
            manifest_path,
            row_path=row_path if row_path is not None and row_path.is_file() else None,
            index_map_path=index_path if index_path is not None and index_path.is_file() else None,
            expected_dimension=dimensions.get(channel),
            catalog_uids=catalog_uids,
        )
        if row_path is not None and not row_path.is_file():
            channels[channel]["issues"].append(
                _issue("ROWS_PATH_NOT_FOUND", f"declared rows path is missing: {row_path}")
            )
        if index_path is not None and not index_path.is_file():
            channels[channel]["issues"].append(
                _issue(
                    "INDEX_MAP_PATH_NOT_FOUND",
                    f"declared index map path is missing: {index_path}",
                )
            )

    issues = [
        {"channel": channel, **item}
        for channel, payload in channels.items()
        for item in payload["issues"]
    ]
    return {
        "format": SCHEMA_VERSION,
        "status": "PASS" if not issues else "FAIL",
        "execution_status": "ENGINEERING_PROXY",
        "quality_status": "UNVALIDATED",
        "identity_policy": {
            "primary": "frame_uid",
            "format": "video_id:source_frame_idx",
            "faiss_row": "diagnostic_only",
        },
        "channels": channels,
        "issues": issues,
        "provenance": {
            "read_only": True,
            "source_manifests": {
                name: str(Path(path).expanduser().resolve())
                for name, path in manifest_paths.items()
            },
        },
    }


def write_validation_report(report: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write a versioned report and open failure ledger without touching inputs."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "channel_manifest_validation_v1.json"
    ledger_path = output / "failure_ledger.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failures = list(report.get("issues", []))
    ledger = {
        "schema_version": "hcmaic-channel-manifest-failure-ledger-v1",
        "status": "OPEN" if failures else "CLEAN",
        "quality_status": "UNVALIDATED",
        "failure_count": len(failures),
        "unresolved_count": len(failures),
        "failures": failures,
    }
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"report": report_path, "failure_ledger": ledger_path}
