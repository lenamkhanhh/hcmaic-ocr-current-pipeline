"""RAM-bounded merger for crop-level DeepSolo/PARSeq OCR shard artifacts.

The merger deliberately produces a separate crop-level artifact.  It must not
be passed to the older frame-level ``ocr_bm25`` loader and it never changes the
immutable keyframe-v1 artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from hcmaic.retrieval.ocr_text import fold_ocr_text, normalize_ocr_nfc

OCR_MERGED_FORMAT = "hcmaic-dstext-parseq-ocr-merged-v1"
QUALITY_STATUS = "UNVALIDATED_ON_HCMAIC"
KNOWN_FAILURE_TYPES = {"NO_TEXT", "READ_FAILED", "INFERENCE_FAILED", "PARSE_ERROR"}


class OCRMergeError(RuntimeError):
    """Raised when a shard cannot be safely merged."""


@dataclass(frozen=True)
class _Shard:
    root: Path
    shard_id: str
    source_key: str
    manifest: dict[str, Any]
    manifest_sha256: str
    lines_path: Path
    status_path: Path
    ledger_path: Path
    model_signature: tuple[tuple[str, str, str], tuple[str, str, str]]
    postflight_status: str | None


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _to_python(value: object) -> object:
    """Convert Arrow scalars/containers into JSON-safe Python values."""
    if value is None:
        return None
    if hasattr(value, "as_py"):
        return _to_python(value.as_py())
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _to_python(value.item())
        except (AttributeError, TypeError, ValueError):
            pass
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OCRMergeError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise OCRMergeError(f"JSON artifact must be an object: {path}")
    return payload


def _resolve_shard_root(value: Path) -> Path:
    value = Path(value).expanduser()
    if value.is_file():
        value = value.parent
    if not value.is_dir():
        raise FileNotFoundError(f"OCR shard directory does not exist: {value}")
    direct = value / "final_manifest.json"
    if direct.is_file():
        return value.resolve()
    candidates = sorted(path.parent.resolve() for path in value.rglob("final_manifest.json"))
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise OCRMergeError(f"final_manifest.json not found under {value}")
    if len(unique) > 1:
        raise OCRMergeError(
            f"ambiguous OCR shard root {value}; found {len(unique)} final_manifest.json files"
        )
    return unique[0]


def _find_artifact(root: Path, names: Sequence[str], label: str) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise OCRMergeError(f"{label} missing under {root}; expected one of {list(names)}")


def _artifact_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    for key in ("artifact_hashes", "artifact_sha256"):
        payload = manifest.get(key)
        if isinstance(payload, dict):
            return {str(name).replace("\\", "/"): str(value) for name, value in payload.items()}
    return {}


def _validate_optional_hash(path: Path, manifest: dict[str, Any]) -> None:
    hashes = _artifact_hashes(manifest)
    expected = hashes.get(path.name)
    if expected is None:
        expected = hashes.get(path.as_posix())
    if expected is None:
        return
    actual = _sha256_file(path)
    if actual != expected:
        raise OCRMergeError(
            f"artifact hash mismatch for {path.name}: expected {expected}, actual {actual}"
        )


def _model_signature(
    manifest: dict[str, Any], shard_id: str
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    detector = manifest.get("detector")
    recognizer = manifest.get("recognizer")
    if not isinstance(detector, dict) or not isinstance(recognizer, dict):
        raise OCRMergeError(f"{shard_id}: detector/recognizer provenance is missing")
    detector_signature = (
        str(detector.get("model", "")),
        str(detector.get("revision", "")),
        str(detector.get("weight_sha256", "")),
    )
    recognizer_signature = (
        str(recognizer.get("model", "")),
        str(recognizer.get("revision", "")),
        str(recognizer.get("checkpoint_sha256", "")),
    )
    if not all(detector_signature) or not all(recognizer_signature):
        raise OCRMergeError(f"{shard_id}: incomplete detector/recognizer provenance")
    return detector_signature, recognizer_signature


def _partition_contract(manifest: dict[str, Any], shard_id: str) -> dict[str, Any] | None:
    """Validate and return a partition contract, if this is a partition artifact."""
    if manifest.get("full_shard") is True:
        if manifest.get("partition") is not None:
            raise OCRMergeError(f"{shard_id}: full shard must not carry a partition contract")
        return None
    partition = manifest.get("partition")
    if not isinstance(partition, dict):
        raise OCRMergeError(
            f"{shard_id}: full_shard is false but the partition contract is missing"
        )
    try:
        count = int(partition.get("count"))
        index = int(partition.get("index"))
        global_frame_count = int(partition.get("global_frame_count"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise OCRMergeError(f"{shard_id}: partition contract has invalid integer fields") from exc
    if count < 2 or not 0 <= index < count or global_frame_count < 1:
        raise OCRMergeError(f"{shard_id}: partition contract bounds are invalid")
    if not str(partition.get("strategy", "")).strip():
        raise OCRMergeError(f"{shard_id}: partition strategy is missing")
    expected_frames = manifest.get("expected_frame_count")
    if expected_frames is not None and int(expected_frames) > global_frame_count:
        raise OCRMergeError(f"{shard_id}: partition expected frame count exceeds global count")
    return {
        "count": count,
        "index": index,
        "global_frame_count": global_frame_count,
        "strategy": str(partition["strategy"]),
    }


def _validate_shard(
    value: Path,
    *,
    require_postflight_green: bool,
    allow_report_failures: bool,
) -> _Shard:
    root = _resolve_shard_root(value)
    manifest_path = root / "final_manifest.json"
    manifest = _read_json(manifest_path)
    shard_id = str(manifest.get("target_shard_id", "")).strip()
    if not shard_id:
        raise OCRMergeError(f"{root}: target_shard_id is missing")
    status = str(manifest.get("status", ""))
    accepted_statuses = {"ENGINEERING_ARTIFACT_COMPLETE"}
    if allow_report_failures:
        accepted_statuses.add("ENGINEERING_ARTIFACT_COMPLETE_REPORT_FAILED")
    if status not in accepted_statuses:
        raise OCRMergeError(f"{shard_id}: final status is not mergeable: {status!r}")
    if manifest.get("execution_status") not in {None, "COMPLETE", "COMPLETE_WITH_REPORT_FAILURE"}:
        raise OCRMergeError(f"{shard_id}: execution_status is not complete")
    if manifest.get("quality_status") not in {"UNVALIDATED", "UNVALIDATED_ON_HCMAIC"}:
        raise OCRMergeError(
            f"{shard_id}: refusing quality status {manifest.get('quality_status')!r}"
        )
    partition = _partition_contract(manifest, shard_id)
    identity = str(manifest.get("identity", ""))
    if "frame_uid=video_id:source_frame_idx" not in identity or "faiss_row" not in identity:
        raise OCRMergeError(f"{shard_id}: identity contract is missing or unsafe")
    postflight = manifest.get("postflight")
    postflight_status = postflight.get("status") if isinstance(postflight, dict) else None
    if require_postflight_green and postflight_status != "POSTFLIGHT_GREEN":
        raise OCRMergeError(
            f"{shard_id}: postflight must be POSTFLIGHT_GREEN, got {postflight_status!r}"
        )

    lines_path = _find_artifact(
        root,
        (
            "ocr_lines.parquet",
            "ocr_lines.jsonl",
            "parseq_ocr_lines.parquet",
            "parseq_ocr_lines.jsonl",
        ),
        "OCR line artifact",
    )
    status_path = _find_artifact(
        root,
        (
            "detection_status.parquet",
            "detection_status.jsonl",
            "parseq_detection_status.parquet",
            "parseq_detection_status.jsonl",
        ),
        "detection status artifact",
    )
    ledger_path = root / "failure_ledger.json"
    if not ledger_path.is_file():
        raise OCRMergeError(f"{shard_id}: failure_ledger.json is missing")
    _validate_optional_hash(lines_path, manifest)
    _validate_optional_hash(status_path, manifest)
    _validate_optional_hash(ledger_path, manifest)
    source_key = (
        f"{shard_id}#part{partition['index']}/{partition['count']}"
        if partition is not None
        else f"{shard_id}#full"
    )
    return _Shard(
        root=root,
        shard_id=shard_id,
        source_key=source_key,
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
        lines_path=lines_path,
        status_path=status_path,
        ledger_path=ledger_path,
        model_signature=_model_signature(manifest, shard_id),
        postflight_status=postflight_status,
    )


def _validate_partition_sets(shards: Sequence[_Shard]) -> None:
    """Require all partitions for a logical shard, exactly once each."""
    grouped: dict[str, list[_Shard]] = {}
    for shard in shards:
        grouped.setdefault(shard.shard_id, []).append(shard)
    for shard_id, members in grouped.items():
        partitioned = [member for member in members if member.manifest.get("full_shard") is not True]
        full = [member for member in members if member.manifest.get("full_shard") is True]
        if full and partitioned:
            raise OCRMergeError(f"{shard_id}: partition set cannot mix full and partition artifacts")
        if len(full) > 1:
            raise OCRMergeError(f"{shard_id}: duplicate full-shard artifacts")
        if not partitioned:
            continue
        contracts = [member.manifest["partition"] for member in partitioned]
        counts = {int(contract["count"]) for contract in contracts}
        globals_ = {int(contract["global_frame_count"]) for contract in contracts}
        indices = [int(contract["index"]) for contract in contracts]
        if len(counts) != 1 or len(globals_) != 1:
            raise OCRMergeError(f"{shard_id}: partition set contract differs across sources")
        expected_count = next(iter(counts))
        expected_global = next(iter(globals_))
        if len(partitioned) != expected_count or set(indices) != set(range(expected_count)):
            raise OCRMergeError(
                f"{shard_id}: incomplete partition set; expected indices "
                f"0..{expected_count - 1}, got {sorted(indices)}"
            )
        expected_sum = sum(
            int(member.manifest.get("expected_frame_count", -1)) for member in partitioned
        )
        if expected_sum != expected_global:
            raise OCRMergeError(
                f"{shard_id}: partition expected frame counts {expected_sum} "
                f"!= global count {expected_global}"
            )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise OCRMergeError(f"cannot open JSONL artifact: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OCRMergeError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise OCRMergeError(f"JSONL row is not an object at {path}:{line_number}")
            yield {str(key): _to_python(value) for key, value in row.items()}


def _iter_parquet(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        parquet_file = pq.ParquetFile(path)
        batches = parquet_file.iter_batches(batch_size=batch_size, use_threads=False)
        for batch in batches:
            for row in batch.to_pylist():
                yield {str(key): _to_python(value) for key, value in row.items()}
    except OCRMergeError:
        raise
    except Exception as exc:
        raise OCRMergeError(f"cannot stream Parquet artifact: {path}") from exc


def iter_artifact_rows(path: Path, *, batch_size: int = 50_000) -> Iterator[dict[str, Any]]:
    """Yield rows without materialising a shard in RAM."""
    if path.suffix.casefold() == ".parquet":
        yield from _iter_parquet(path, batch_size)
    else:
        yield from _iter_jsonl(path)


def _require_int(value: object, field: str, context: str) -> int:
    try:
        converted = int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise OCRMergeError(f"{context}: {field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise OCRMergeError(f"{context}: {field} must be an integer")
    return converted


def _canonical_line(row: dict[str, Any], shard: _Shard) -> dict[str, Any]:
    context = f"{shard.shard_id} OCR line"
    crop_uid = str(row.get("crop_uid", "")).strip()
    frame_uid = str(row.get("frame_uid", "")).strip()
    video_id = str(row.get("video_id", "")).strip()
    if not crop_uid or not frame_uid or not video_id:
        raise OCRMergeError(f"{context}: crop_uid/frame_uid/video_id must not be blank")
    source_frame_idx = _require_int(row.get("source_frame_idx"), "source_frame_idx", context)
    if source_frame_idx < 0:
        raise OCRMergeError(f"{context}: source_frame_idx must be non-negative")
    expected_frame_uid = f"{video_id}:{source_frame_idx}"
    if frame_uid != expected_frame_uid:
        raise OCRMergeError(
            f"{context}: frame_uid identity mismatch: {frame_uid!r} != {expected_frame_uid!r}"
        )
    result = {str(key): _to_python(value) for key, value in row.items()}
    raw_text = normalize_ocr_nfc(result.get("ocr_text_raw", result.get("text_raw", "")))
    nfc_text = normalize_ocr_nfc(result.get("ocr_text_nfc", raw_text))
    result.update(
        {
            "crop_uid": crop_uid,
            "line_uid": str(result.get("line_uid") or crop_uid),
            "frame_uid": frame_uid,
            "video_id": video_id,
            "source_frame_idx": source_frame_idx,
            "ocr_text_raw": raw_text,
            "ocr_text_nfc": nfc_text,
            "ocr_text_folded": normalize_ocr_nfc(
                result.get("ocr_text_folded", fold_ocr_text(nfc_text))
            ),
            "source_shard_id": shard.shard_id,
            "source_manifest_sha256": shard.manifest_sha256,
            "source_quality_status": str(shard.manifest.get("quality_status")),
            "source_execution_status": str(shard.manifest.get("execution_status", "COMPLETE")),
            "quality_status": QUALITY_STATUS,
            "execution_status": "ENGINEERING_ARTIFACT_COMPLETE",
        }
    )
    if not result["ocr_text_folded"] and nfc_text:
        result["ocr_text_folded"] = fold_ocr_text(nfc_text)
    return result


def _canonical_frame_status(row: dict[str, Any], shard: _Shard) -> dict[str, Any]:
    frame_uid = str(row.get("frame_uid", "")).strip()
    if not frame_uid:
        raise OCRMergeError(f"{shard.shard_id} detection status: frame_uid is blank")
    result = {str(key): _to_python(value) for key, value in row.items()}
    result.update(
        {
            "frame_uid": frame_uid,
            "source_shard_id": shard.shard_id,
            "source_manifest_sha256": shard.manifest_sha256,
            "quality_status": QUALITY_STATUS,
            "execution_status": "ENGINEERING_ARTIFACT_COMPLETE",
        }
    )
    return result


def _parquet_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Stable, Arrow-friendly projection; JSONL remains the lossless row view."""
    strings = (
        "crop_uid",
        "line_uid",
        "frame_uid",
        "video_id",
        "shot_id",
        "ocr_text_raw",
        "ocr_text_nfc",
        "ocr_text_folded",
        "confidence_status",
        "detector_model",
        "detector_revision",
        "recognizer_model",
        "recognizer_revision",
        "candidate_policy",
        "source_shard_id",
        "source_manifest_sha256",
        "source_quality_status",
        "source_execution_status",
        "quality_status",
        "execution_status",
    )
    integers = (
        "source_frame_idx",
        "timestamp_ms",
        "line_index",
        "detector_line_index",
        "word_index",
    )
    floats = ("det_score", "rec_score")
    json_fields = ("bbox", "polygon", "detector_polygon", "ocr_candidates")
    result: dict[str, Any] = {}
    for field in strings:
        value = row.get(field)
        result[field] = None if value is None else str(value)
    for field in integers:
        value = row.get(field)
        result[field] = None if value is None else _require_int(value, field, "Parquet projection")
    for field in floats:
        value = row.get(field)
        result[field] = None if value is None else float(value)
    for field in json_fields:
        result[f"{field}_json"] = json.dumps(
            _to_python(row.get(field)),
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
    result["raw_row_json"] = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default
    )
    return result


def _write_parquet_batch(writer: Any, rows: list[dict[str, Any]], schema: Any) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]

    writer.write_table(
        pa.Table.from_pylist([_parquet_projection(row) for row in rows], schema=schema)
    )


def _parquet_schema() -> Any:
    import pyarrow as pa  # type: ignore[import-untyped]

    fields = [
        *(
            pa.field(name, pa.string())
            for name in (
                "crop_uid",
                "line_uid",
                "frame_uid",
                "video_id",
                "shot_id",
                "ocr_text_raw",
                "ocr_text_nfc",
                "ocr_text_folded",
                "confidence_status",
                "detector_model",
                "detector_revision",
                "recognizer_model",
                "recognizer_revision",
                "candidate_policy",
                "source_shard_id",
                "source_manifest_sha256",
                "source_quality_status",
                "source_execution_status",
                "quality_status",
                "execution_status",
                "bbox_json",
                "polygon_json",
                "detector_polygon_json",
                "ocr_candidates_json",
                "raw_row_json",
            )
        ),
        *(
            pa.field(name, pa.int64())
            for name in (
                "source_frame_idx",
                "timestamp_ms",
                "line_index",
                "detector_line_index",
                "word_index",
            )
        ),
        *(pa.field(name, pa.float64()) for name in ("det_score", "rec_score")),
    ]
    return pa.schema(fields)


class _FailureWriter:
    def __init__(self, json_path: Path, jsonl_path: Path) -> None:
        self._json = json_path.open("w", encoding="utf-8")
        self._jsonl = jsonl_path.open("w", encoding="utf-8")
        self._json.write('{"status":"CLOSED","failures":[\n')
        self._first = True
        self.count = 0
        self.unresolved = 0
        self.by_type: Counter[str] = Counter()

    def add(self, failure: dict[str, Any]) -> None:
        failure = {str(key): _to_python(value) for key, value in failure.items()}
        failure_type = str(failure.get("failure_type") or "OTHER")
        self.by_type[failure_type] += 1
        if not bool(failure.get("resolved", False)):
            self.unresolved += 1
        encoded = json.dumps(failure, ensure_ascii=False, sort_keys=True, default=_json_default)
        if not self._first:
            self._json.write(",\n")
        self._json.write(encoded)
        self._jsonl.write(encoded + "\n")
        self._first = False
        self.count += 1

    def close(self) -> None:
        self._json.write(
            f'\n],"failure_count":{self.count},'
            f'"unresolved_count":{self.unresolved},'
            f'"counts_by_type":{json.dumps(dict(sorted(self.by_type.items())))}\n}}\n'
        )
        self._json.close()
        self._jsonl.close()


def _copy_failures(shard: _Shard, writer: _FailureWriter) -> None:
    ledger = _read_json(shard.ledger_path)
    failures = ledger.get("failures", [])
    if not isinstance(failures, list):
        raise OCRMergeError(f"{shard.shard_id}: failure_ledger.failures is not a list")
    reported_count = ledger.get("failure_count")
    if reported_count is not None and int(reported_count) != len(failures):
        raise OCRMergeError(f"{shard.shard_id}: failure ledger count mismatch")
    reported_unresolved = ledger.get("unresolved_count")
    if reported_unresolved is not None:
        actual_unresolved = sum(
            not bool(item.get("resolved", False)) for item in failures if isinstance(item, dict)
        )
        if int(reported_unresolved) != actual_unresolved:
            raise OCRMergeError(f"{shard.shard_id}: unresolved failure count mismatch")
    for failure in failures:
        if not isinstance(failure, dict):
            raise OCRMergeError(f"{shard.shard_id}: failure ledger row is not an object")
        failure_type = str(failure.get("failure_type") or failure.get("type") or "OTHER")
        if failure_type not in KNOWN_FAILURE_TYPES:
            failure_type = "OTHER"
        writer.add(
            {
                **failure,
                "failure_type": failure_type,
                "source_shard_id": shard.shard_id,
                "source_manifest_sha256": shard.manifest_sha256,
            }
        )


def _insert_frame_uid(conn: sqlite3.Connection, uid: str, shard_id: str) -> None:
    try:
        conn.execute("INSERT INTO frame_uids(uid, shard_id) VALUES (?, ?)", (uid, shard_id))
    except sqlite3.IntegrityError as exc:
        raise OCRMergeError(f"duplicate frame_uid: {uid}") from exc


def _insert_crop_uid(conn: sqlite3.Connection, uid: str, frame_uid: str, shard_id: str) -> None:
    try:
        conn.execute(
            "INSERT INTO crop_uids(uid, frame_uid, shard_id) VALUES (?, ?, ?)",
            (uid, frame_uid, shard_id),
        )
    except sqlite3.IntegrityError as exc:
        raise OCRMergeError(f"duplicate crop_uid: {uid}") from exc


def _frame_exists(conn: sqlite3.Connection, frame_uid: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM frame_uids WHERE uid = ?", (frame_uid,)).fetchone() is not None
    )


def merge_ocr_shards(
    shard_roots: Iterable[Path],
    output_dir: Path,
    *,
    batch_size: int = 50_000,
    require_postflight_green: bool = True,
    allow_report_failures: bool = False,
) -> dict[str, Any]:
    """Validate and merge full-shard OCR artifacts with bounded memory.

    The destination must be new or empty.  A successful run atomically moves a
    temporary directory into place; a failed run does not leave a misleading
    ``ocr_manifest.json`` at the requested destination.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    values = list(shard_roots)
    if not values:
        raise ValueError("at least one OCR shard is required")
    shards = [
        _validate_shard(
            Path(value),
            require_postflight_green=require_postflight_green,
            allow_report_failures=allow_report_failures,
        )
        for value in values
    ]
    _validate_partition_sets(shards)
    shard_ids = sorted({shard.shard_id for shard in shards})
    expected_signature = shards[0].model_signature
    if any(shard.model_signature != expected_signature for shard in shards[1:]):
        raise OCRMergeError("detector/recognizer model provenance differs across shards")
    source_has_report_failures = any(
        shard.manifest.get("status") != "ENGINEERING_ARTIFACT_COMPLETE"
        for shard in shards
    )

    output_dir = Path(output_dir).expanduser()
    if output_dir.exists():
        raise OCRMergeError(
            f"output directory already exists; use a new versioned path: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    db_path = temporary_dir / "identity.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("CREATE TABLE frame_uids(uid TEXT PRIMARY KEY, shard_id TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE crop_uids(uid TEXT PRIMARY KEY, frame_uid TEXT NOT NULL, "
        "shard_id TEXT NOT NULL)"
    )
    conn.commit()
    frame_writer = None
    line_writer = None
    failure_writer = None
    parquet_writer = None
    parquet_schema = None
    parquet_batch: list[dict[str, Any]] = []
    frame_count = 0
    line_count = 0
    crop_count = 0
    source_frame_counts: dict[str, int] = {}
    source_line_counts: dict[str, int] = {}
    source_status_counts: dict[str, int] = {}
    try:
        frame_writer = (temporary_dir / "frame_status.jsonl").open("w", encoding="utf-8")
        line_writer = (temporary_dir / "ocr_lines.jsonl").open("w", encoding="utf-8")
        failure_writer = _FailureWriter(
            temporary_dir / "failure_ledger.json", temporary_dir / "failure_ledger.jsonl"
        )
        import pyarrow.parquet as pq

        parquet_schema = _parquet_schema()
        parquet_writer = pq.ParquetWriter(temporary_dir / "ocr_lines.parquet", parquet_schema)
        for shard in shards:
            shard_frame_count = 0
            status_counter: Counter[str] = Counter()
            for raw_status in iter_artifact_rows(shard.status_path, batch_size=batch_size):
                status = _canonical_frame_status(raw_status, shard)
                frame_uid = str(status["frame_uid"])
                _insert_frame_uid(conn, frame_uid, shard.shard_id)
                frame_writer.write(
                    json.dumps(status, ensure_ascii=False, sort_keys=True, default=_json_default)
                    + "\n"
                )
                frame_count += 1
                shard_frame_count += 1
                status_counter[str(status.get("status", "UNKNOWN"))] += 1
            source_frame_counts[shard.source_key] = shard_frame_count
            source_status_counts[shard.source_key] = sum(status_counter.values())
            expected_frames = shard.manifest.get("expected_frame_count")
            if expected_frames is not None and int(expected_frames) != shard_frame_count:
                raise OCRMergeError(
                    f"{shard.shard_id}: detection status count {shard_frame_count} "
                    f"!= expected {expected_frames}"
                )
            declared_status_count = shard.manifest.get("frame_status_count")
            if (
                declared_status_count is not None
                and int(declared_status_count) != shard_frame_count
            ):
                raise OCRMergeError(f"{shard.shard_id}: frame_status_count mismatch")
            for raw_row in iter_artifact_rows(shard.lines_path, batch_size=batch_size):
                row = _canonical_line(raw_row, shard)
                frame_uid = str(row["frame_uid"])
                if not _frame_exists(conn, frame_uid):
                    raise OCRMergeError(
                        f"{shard.shard_id}: OCR crop references unknown frame_uid {frame_uid}"
                    )
                crop_uid = str(row["crop_uid"])
                _insert_crop_uid(conn, crop_uid, frame_uid, shard.shard_id)
                encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default)
                line_writer.write(encoded + "\n")
                parquet_batch.append(row)
                line_count += 1
                crop_count += 1
                if len(parquet_batch) >= batch_size:
                    _write_parquet_batch(parquet_writer, parquet_batch, parquet_schema)
                    parquet_batch.clear()
            source_line_counts[shard.source_key] = line_count - sum(source_line_counts.values())
            declared_crop_count = shard.manifest.get("crop_count")
            declared_rec_count = shard.manifest.get("recognition_count")
            actual_source_lines = source_line_counts[shard.source_key]
            for label, declared in (
                ("crop_count", declared_crop_count),
                ("recognition_count", declared_rec_count),
            ):
                if declared is not None and int(declared) != actual_source_lines:
                    raise OCRMergeError(
                        f"{shard.shard_id}: {label} {declared} != OCR line rows "
                        f"{actual_source_lines}"
                    )
            _copy_failures(shard, failure_writer)
        if parquet_batch:
            _write_parquet_batch(parquet_writer, parquet_batch, parquet_schema)
            parquet_batch.clear()
        conn.commit()
        frame_writer.close()
        line_writer.close()
        failure_writer.close()
        parquet_writer.close()
        conn.close()
        db_path.unlink(missing_ok=True)
        wal_path = temporary_dir / "identity.sqlite3-wal"
        shm_path = temporary_dir / "identity.sqlite3-shm"
        wal_path.unlink(missing_ok=True)
        shm_path.unlink(missing_ok=True)
        artifact_hashes = {
            path.relative_to(temporary_dir).as_posix(): _sha256_file(path)
            for path in sorted(temporary_dir.iterdir())
            if path.is_file()
        }
        source_manifest_revisions = sorted(
            {
                str(shard.manifest.get("code_revision", {}).get("pipeline_code_sha256"))
                for shard in shards
                if isinstance(shard.manifest.get("code_revision"), dict)
                and shard.manifest.get("code_revision", {}).get("pipeline_code_sha256")
            }
        )
        manifest = {
            "format": OCR_MERGED_FORMAT,
            "status": (
                "ENGINEERING_ARTIFACT_PARTIAL"
                if source_has_report_failures or failure_writer.unresolved
                else "ENGINEERING_ARTIFACT_COMPLETE"
            ),
            "execution_status": (
                "COMPLETE_WITH_REPORT_FAILURE"
                if source_has_report_failures or failure_writer.unresolved
                else "COMPLETE"
            ),
            "provenance_class": "ENGINEERING_PROXY",
            "quality_status": QUALITY_STATUS,
            "human_review_required": True,
            "identity": (
                "frame_uid=video_id:source_frame_idx; crop_uid is OCR line identity; "
                "faiss_row is not identity"
            ),
            "immutable_keyframe_v1": True,
            "channel": "ocr",
            "source_shard_ids": shard_ids,
            "source_shards": [
                {
                    "source_key": shard.source_key,
                    "target_shard_id": shard.shard_id,
                    "root": str(shard.root),
                    "final_manifest_sha256": shard.manifest_sha256,
                    "full_shard": shard.manifest.get("full_shard"),
                    "partition": shard.manifest.get("partition"),
                    "postflight_status": shard.postflight_status,
                    "selection_sha256": shard.manifest.get("selection_sha256"),
                    "expected_frame_count": shard.manifest.get("expected_frame_count"),
                    "actual_frame_count": source_frame_counts[shard.source_key],
                    "actual_line_count": source_line_counts[shard.source_key],
                    "status_counts": source_status_counts[shard.source_key],
                    "quality_status": shard.manifest.get("quality_status"),
                    "code_revision": shard.manifest.get("code_revision", {}),
                }
                for shard in shards
            ],
            "model_contract": {
                "detector": {
                    "model": expected_signature[0][0],
                    "revision": expected_signature[0][1],
                    "weight_sha256": expected_signature[0][2],
                },
                "recognizer": {
                    "model": expected_signature[1][0],
                    "revision": expected_signature[1][1],
                    "checkpoint_sha256": expected_signature[1][2],
                },
            },
            "code_revisions": source_manifest_revisions,
            "counts": {
                "frame_count": frame_count,
                "crop_count": crop_count,
                "line_count": line_count,
                "failure_count": failure_writer.count,
                "unresolved_failure_count": failure_writer.unresolved,
                "failure_counts_by_type": dict(sorted(failure_writer.by_type.items())),
            },
            "runtime": {
                "merge_batch_size": batch_size,
                "identity_validation": "sqlite_disk_backed_unique_frame_and_crop_uids",
                "jsonl_output": "lossless_canonical_rows_with_provenance",
                "parquet_output": "stable_projection_plus_raw_row_json",
                "partition_validation": "complete_index_set_per_target_shard",
            },
            "postflight": {
                "status": "POSTFLIGHT_GREEN",
                "checkpoint": "ocr_manifest.json",
                "counts": {
                    "frame_rows": frame_count,
                    "line_rows": line_count,
                    "failure_rows": failure_writer.count,
                    "unresolved_failure_count": failure_writer.unresolved,
                },
            },
            "artifact_hashes": artifact_hashes,
        }
        _write_json(temporary_dir / "ocr_manifest.json", manifest)
        if output_dir.exists():
            raise OCRMergeError(f"output directory became occupied: {output_dir}")
        temporary_dir.replace(output_dir)
        return {
            "output": str(output_dir.resolve()),
            "format": OCR_MERGED_FORMAT,
            "status": manifest["status"],
            "quality_status": QUALITY_STATUS,
            "frame_count": frame_count,
            "crop_count": crop_count,
            "line_count": line_count,
            "failure_count": failure_writer.count,
            "unresolved_failure_count": failure_writer.unresolved,
        }
    except Exception:
        for handle in (frame_writer, line_writer, failure_writer, parquet_writer):
            if handle is not None:
                with suppress(Exception):
                    handle.close()
        with suppress(Exception):
            conn.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
