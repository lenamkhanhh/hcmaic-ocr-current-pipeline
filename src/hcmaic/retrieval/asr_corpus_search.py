"""Bounded full-corpus execution for the optional ASR Elasticsearch channel."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hcmaic.retrieval.asr_elasticsearch import (
    ASRElasticsearchError,
    ASRElasticsearchTimeoutError,
    ASRESchemaMismatchError,
    ASRMappingMismatchError,
)

ASR_CORPUS_SEARCH_SCHEMA = "hcmaic-asr-corpus-search-v1"
_QUERY_FIELDS = (
    "phowhisper_raw",
    "whisper_v3_raw",
    "phowhisper_folded",
    "whisper_v3_folded",
)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _query_text(row: Mapping[str, Any]) -> tuple[str, str] | None:
    for field in _QUERY_FIELDS:
        value = str(row.get(field) or "").strip()
        if value:
            return field, value
    return None


def _failure_category(error: BaseException) -> str:
    if isinstance(error, ASRElasticsearchTimeoutError):
        return "timeout"
    if isinstance(error, ASRMappingMismatchError):
        return "mapping_mismatch"
    if isinstance(error, ASRESchemaMismatchError):
        return "schema_mismatch"
    if isinstance(error, ASRElasticsearchError):
        return "unavailable"
    if isinstance(error, (TimeoutError,)):
        return "timeout"
    return "runtime_error"


def _failure(
    *,
    query_index: int,
    segment_id: str,
    category: str,
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query_index": query_index,
        "segment_id": segment_id,
        "category": category,
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
    return payload


def _hit_payload(hit: Any) -> dict[str, Any]:
    evidence = _json_safe(getattr(hit, "evidence", {}))
    return {
        "rank": int(getattr(hit, "rank", 0)),
        "score": float(getattr(hit, "score", 0.0)),
        "frame_uid": str(getattr(hit, "frame_uid", None) or getattr(hit, "entity_id", "")),
        "identity_key": "frame_uid",
        "video_id": str(getattr(hit, "video_id", "")),
        "source_frame_idx": getattr(hit, "source_frame_idx", None),
        "timestamp_ms": getattr(hit, "timestamp_ms", None),
        "provider": str(getattr(hit, "provider", "")),
        "evidence_text": getattr(hit, "evidence_text", None),
        "evidence": evidence,
        # The adapter's evidence is the validated, bounded raw ES segment
        # response after parsing. Keep a named copy for audit consumers.
        "raw_response": evidence,
    }


def _status_payload(
    *,
    adapter_status: Mapping[str, Any],
    index: str,
    mode: str,
    top_k: int,
    total: int,
    completed: int,
    succeeded: int,
    failed: int,
    phase: str,
    status: str,
    started_at: str,
    last_segment_id: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": ASR_CORPUS_SEARCH_SCHEMA,
        "status": status,
        "phase": phase,
        "execution_status": "ENGINEERING_PROXY",
        "quality_status": "UNVALIDATED",
        "index": index,
        "mode": mode,
        "top_k": top_k,
        "query_count_total": total,
        "query_count_completed": completed,
        "success_count": succeeded,
        "failure_count": failed,
        "last_segment_id": last_segment_id,
        "started_at": started_at,
        "updated_at": _now(),
        "adapter": _json_safe(adapter_status),
    }


def run_full_asr_corpus_search(
    adapter: Any,
    queries: Iterable[Mapping[str, Any]],
    output_dir: Path,
    *,
    top_k: int = 10,
    heartbeat_every: int = 500,
    max_failures: int = 100,
    expected_query_count: int | None = None,
    expected_edge_count: int | None = None,
    expected_frame_count: int | None = None,
    expected_unmapped_count: int | None = None,
) -> dict[str, Any]:
    """Run every transcript query through the bounded adapter contract.

    ``queries`` is normally the full segment corpus from the dataset's Bulk
    NDJSON. The function materializes only the query metadata list, never
    fetches unbounded ES results, and writes each bounded result row to a
    replace-on-success JSONL artifact.
    """

    if top_k < 1 or top_k > 500:
        raise ValueError("top_k must be in [1, 500]")
    if heartbeat_every < 1:
        raise ValueError("heartbeat_every must be positive")
    if max_failures < 1:
        raise ValueError("max_failures must be positive")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    query_rows = [dict(row) for row in queries]
    adapter_status = dict(adapter.status_dict())
    config = getattr(adapter, "config", None)
    index = str(adapter_status.get("index") or getattr(config, "index", ""))
    mode = str(adapter_status.get("mode") or getattr(config, "mode", ""))
    started_at = _now()
    status_path = destination / "asr_corpus_status.json"
    manifest_path = destination / "asr_corpus_manifest.json"
    ledger_path = destination / "asr_corpus_failure_ledger.json"
    result_path = destination / "asr_corpus_results.jsonl"
    temporary_result = destination / ".asr_corpus_results.jsonl.part"

    failures: list[dict[str, Any]] = []
    completed = 0
    succeeded = 0
    total_hits = 0
    unique_frame_uids: set[str] = set()
    seen_segment_ids: set[str] = set()
    last_segment_id: str | None = None

    def write_status(phase: str, status: str) -> None:
        _write_json(
            status_path,
            _status_payload(
                adapter_status=adapter_status,
                index=index,
                mode=mode,
                top_k=top_k,
                total=len(query_rows),
                completed=completed,
                succeeded=succeeded,
                failed=len(failures),
                phase=phase,
                status=status,
                started_at=started_at,
                last_segment_id=last_segment_id,
            ),
        )

    def write_ledger(status: str) -> None:
        _write_json(
            ledger_path,
            {
                "schema_version": ASR_CORPUS_SEARCH_SCHEMA,
                "status": status,
                "failure_count": len(failures),
                "failures": failures,
            },
        )

    write_status("START", "RUNNING")
    write_ledger("RUNNING")
    started_monotonic = time.monotonic()
    with temporary_result.open("w", encoding="utf-8", newline="\n") as output:
        for query_index, row in enumerate(query_rows):
            segment_id = str(row.get("segment_id") or "").strip()
            last_segment_id = segment_id or None
            query_choice = _query_text(row)
            query_payload = {
                "segment_id": segment_id,
                "video_id": str(row.get("video_id") or ""),
                "phowhisper_raw": str(row.get("phowhisper_raw") or ""),
                "whisper_v3_raw": str(row.get("whisper_v3_raw") or ""),
                "phowhisper_folded": str(row.get("phowhisper_folded") or ""),
                "whisper_v3_folded": str(row.get("whisper_v3_folded") or ""),
            }
            record: dict[str, Any] = {
                "query_index": query_index,
                "query": query_payload,
                "hits": [],
            }
            if not segment_id:
                failures.append(
                    _failure(
                        query_index=query_index,
                        segment_id="",
                        category="blank_segment_id",
                    )
                )
            elif segment_id in seen_segment_ids:
                failures.append(
                    _failure(
                        query_index=query_index,
                        segment_id=segment_id,
                        category="duplicate_segment_id",
                    )
                )
            elif query_choice is None:
                failures.append(
                    _failure(
                        query_index=query_index,
                        segment_id=segment_id,
                        category="blank_query",
                    )
                )
            else:
                seen_segment_ids.add(segment_id)
                query_field, query_text = query_choice
                record["query"]["selected_field"] = query_field
                record["query"]["query_text"] = query_text
                try:
                    hits = adapter.search(query_text, top_k=top_k)
                    record["hits"] = [_hit_payload(hit) for hit in hits]
                    total_hits += len(record["hits"])
                    unique_frame_uids.update(
                        str(hit["frame_uid"])
                        for hit in record["hits"]
                        if str(hit.get("frame_uid") or "")
                    )
                    succeeded += 1
                except Exception as error:  # noqa: BLE001 - ledger every query failure
                    failures.append(
                        _failure(
                            query_index=query_index,
                            segment_id=segment_id,
                            category=_failure_category(error),
                            error=error,
                        )
                    )
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            completed += 1
            if completed % heartbeat_every == 0 or completed == len(query_rows):
                elapsed = max(time.monotonic() - started_monotonic, 0.001)
                print(
                    "HEARTBEAT "
                    f"phase=QUERY_CORPUS completed={completed}/{len(query_rows)} "
                    f"succeeded={succeeded} failed={len(failures)} "
                    f"rate={completed / elapsed:.2f}/s",
                    flush=True,
                )
                write_status("QUERY_CORPUS", "RUNNING")
                write_ledger("RUNNING")
            if len(failures) >= max_failures:
                failures.append(
                    _failure(
                        query_index=query_index,
                        segment_id=segment_id,
                        category="failure_limit_reached",
                    )
                )
                break

    if completed == len(query_rows):
        temporary_result.replace(result_path)
    else:
        temporary_result.replace(result_path)
        failures.append(
            _failure(
                query_index=completed,
                segment_id=last_segment_id or "",
                category="stopped_before_query_corpus_complete",
            )
        )

    mapping = adapter_status.get("mapping")
    if not isinstance(mapping, Mapping):
        mapping = {}
    mapping_checks = {
        "edge_count": (expected_edge_count, mapping.get("edge_count")),
        "frame_count": (expected_frame_count, mapping.get("frame_count")),
        "unmapped_count": (expected_unmapped_count, mapping.get("unmapped_count")),
    }
    for field, (expected, actual) in mapping_checks.items():
        if expected is not None and actual != expected:
            failures.append(
                _failure(
                    query_index=-1,
                    segment_id="",
                    category=f"{field}_mismatch",
                )
            )
    if expected_query_count is not None and len(query_rows) != expected_query_count:
        failures.append(
            _failure(
                query_index=-1,
                segment_id="",
                category="query_count_mismatch",
            )
        )

    final_status = "GREEN" if not failures and completed == len(query_rows) else "FAIL"
    manifest = {
        "schema_version": ASR_CORPUS_SEARCH_SCHEMA,
        "status": final_status,
        "operation": "bounded_adapter_query_corpus",
        "execution_status": "ENGINEERING_PROXY",
        "quality_status": "UNVALIDATED",
        "identity_key": "frame_uid",
        "index": index,
        "mode": mode,
        "top_k": top_k,
        "query_count": len(query_rows),
        "completed_count": completed,
        "success_count": succeeded,
        "failure_count": len(failures),
        "hit_count": total_hits,
        "unique_frame_uid_count": len(unique_frame_uids),
        "expected_query_count": expected_query_count,
        "expected_mapping": {
            "edge_count": expected_edge_count,
            "frame_count": expected_frame_count,
            "unmapped_count": expected_unmapped_count,
        },
        "adapter": _json_safe(adapter_status),
        "outputs": {
            "results": result_path.name,
            "status": status_path.name,
            "failure_ledger": ledger_path.name,
        },
        "started_at": started_at,
        "finished_at": _now(),
    }
    _write_json(manifest_path, manifest)
    write_status("DONE", final_status)
    write_ledger(final_status)
    return manifest
