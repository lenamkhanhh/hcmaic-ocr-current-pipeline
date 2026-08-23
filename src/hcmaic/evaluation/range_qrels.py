"""Build a reproducible qrels bundle from the HCMAIC range review export.

This module is deliberately model- and index-independent.  It maps the
human-reviewed inclusive ``[left, right]`` source-frame intervals onto a
validated keyframe catalog and keeps the older R1 anchor labels as provenance.
The stable identity is always ``frame_uid=video_id:source_frame_idx``;
``feature_row``/``faiss_row`` values are never emitted as qrel identities.

The generated bundle is an engineering evaluation artifact.  It carries the
human-review evidence level and remains ``quality_status=UNVALIDATED`` until
the project promotes the review and accepts a benchmark protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EVIDENCE_LEVEL = "HUMAN_REVIEW_DRAFT"
QUALITY_STATUS = "UNVALIDATED"
ACCEPTED_STATUSES = {"accepted", "edited"}
REJECTED_STATUS = "rejected"
STEP_RE = re.compile(r":event:(?P<step>[0-9]+)$")
CATALOG_COLUMNS = (
    "frame_uid",
    "video_id",
    "source_frame_idx",
    "timestamp_ms",
    "pts_time",
    "feature_row",
)


class RangeQrelsError(ValueError):
    """Raised when qrels cannot be built without guessing."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RangeQrelsError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RangeQrelsError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RangeQrelsError(f"JSONL row {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RangeQrelsError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RangeQrelsError(f"{name} must be an integer") from exc
    return result


def _frame_uid(video_id: str, source_frame_idx: int) -> str:
    return f"{video_id}:{source_frame_idx}"


def _catalog_row(row: Mapping[str, Any], path: Path, row_number: int) -> dict[str, Any]:
    video_id = str(row.get("video_id") or "").strip()
    if not video_id:
        raise RangeQrelsError(f"{path}:{row_number}: catalog video_id is blank")
    source_frame_idx = _int(
        row.get("source_frame_idx", row.get("frame_idx")),
        f"{path}:{row_number}: source_frame_idx",
    )
    if source_frame_idx < 0:
        raise RangeQrelsError(f"{path}:{row_number}: source_frame_idx is negative")
    timestamp_value = row.get("timestamp_ms")
    if timestamp_value is None and row.get("pts_time") is not None:
        timestamp_value = round(float(row["pts_time"]) * 1000)
    if timestamp_value is None:
        raise RangeQrelsError(f"{path}:{row_number}: timestamp_ms/pts_time is missing")
    timestamp_ms = _int(timestamp_value, f"{path}:{row_number}: timestamp_ms")
    if timestamp_ms < 0:
        raise RangeQrelsError(f"{path}:{row_number}: timestamp_ms is negative")
    expected_uid = _frame_uid(video_id, source_frame_idx)
    raw_uid = str(row.get("frame_uid") or expected_uid)
    if raw_uid != expected_uid:
        raise RangeQrelsError(
            f"{path}:{row_number}: frame_uid {raw_uid!r} != expected {expected_uid!r}"
        )
    return {
        "frame_uid": raw_uid,
        "video_id": video_id,
        "source_frame_idx": source_frame_idx,
        "timestamp_ms": timestamp_ms,
        "feature_row": row.get("feature_row"),
    }


def load_catalog(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load and validate one or more catalog Parquets.

    ``pyarrow`` is imported lazily so the rest of the evaluation package stays
    usable in the lightweight runtime.  The full catalog is required only
    while producing this bundle.
    """

    if not paths:
        raise RangeQrelsError("at least one --catalog path is required")
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RangeQrelsError(
            "pyarrow is required to read catalog Parquet files; install it in the eval environment"
        ) from exc

    rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_source: set[tuple[str, int]] = set()
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise RangeQrelsError(f"missing catalog: {path}")
        schema_names = set(parquet.read_schema(path).names)
        columns = [name for name in CATALOG_COLUMNS if name in schema_names]
        required = {"video_id", "source_frame_idx"} - schema_names
        if required:
            raise RangeQrelsError(f"{path}: catalog missing columns {sorted(required)}")
        for row_number, raw in enumerate(parquet.read_table(path, columns=columns).to_pylist(), 1):
            row = _catalog_row(raw, path, row_number)
            uid = row["frame_uid"]
            source_key = (row["video_id"], row["source_frame_idx"])
            if uid in seen_uids:
                raise RangeQrelsError(f"duplicate frame_uid in catalogs: {uid}")
            if source_key in seen_source:
                raise RangeQrelsError(
                    f"duplicate source identity in catalogs: {source_key[0]}:{source_key[1]}"
                )
            seen_uids.add(uid)
            seen_source.add(source_key)
            rows.append(row)
    rows.sort(key=lambda row: (row["video_id"], row["source_frame_idx"], row["frame_uid"]))
    return rows


def canonical_catalog_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash canonical source identity/timestamps, independent of shard order."""

    payload = "".join(
        _json(
            {
                "frame_uid": row["frame_uid"],
                "video_id": row["video_id"],
                "source_frame_idx": row["source_frame_idx"],
                "timestamp_ms": row["timestamp_ms"],
            }
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    return _sha256_bytes(payload)


def catalog_uid_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(f"{row['frame_uid']}\n" for row in rows).encode("utf-8")
    return _sha256_bytes(payload)


def _load_review_items(path: Path) -> dict[str, dict[str, Any]]:
    items = _read_jsonl(path)
    by_uid: dict[str, dict[str, Any]] = {}
    for item in items:
        uid = str(item.get("review_uid") or item.get("range_review_uid") or "")
        if not uid:
            raise RangeQrelsError(f"{path}: review item missing review_uid")
        if uid in by_uid:
            raise RangeQrelsError(f"duplicate review_uid in review items: {uid}")
        by_uid[uid] = item
    return by_uid


def _load_decisions(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    export = _read_json(path)
    decisions = export.get("decisions")
    if not isinstance(decisions, list):
        raise RangeQrelsError(f"{path}: decisions must be a list")
    by_uid: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise RangeQrelsError(f"{path}: decision is not an object")
        uid = str(decision.get("review_uid") or "")
        if not uid:
            raise RangeQrelsError(f"{path}: decision missing review_uid")
        if uid in by_uid:
            raise RangeQrelsError(f"duplicate review_uid in decisions: {uid}")
        by_uid[uid] = decision
    if export.get("item_count") != len(by_uid):
        raise RangeQrelsError(
            f"{path}: item_count={export.get('item_count')} but decisions={len(by_uid)}"
        )
    return export, by_uid


def _step_from_review_uid(review_uid: str) -> int | None:
    match = STEP_RE.search(review_uid)
    return int(match.group("step")) if match else None


def _task_from_query_uid(query_uid: str) -> str:
    parts = query_uid.split(":")
    if len(parts) < 3:
        raise RangeQrelsError(f"invalid query_uid: {query_uid}")
    task = parts[1]
    return "kis" if task == "kis" else task


def _build_catalog_by_video(
    catalog: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in catalog:
        by_video[str(row["video_id"])].append(dict(row))
    for rows in by_video.values():
        rows.sort(key=lambda row: (row["source_frame_idx"], row["timestamp_ms"], row["frame_uid"]))
    return by_video


def _map_range(
    decision: Mapping[str, Any],
    catalog_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    video_id = str(decision.get("video_id") or "")
    left = _int(decision.get("left"), f"{decision.get('review_uid')}:left")
    right = _int(decision.get("right"), f"{decision.get('review_uid')}:right")
    anchor = _int(decision.get("anchor"), f"{decision.get('review_uid')}:anchor")
    if left > right or not left <= anchor <= right:
        raise RangeQrelsError(
            f"{decision.get('review_uid')}: invalid inclusive range {left},{anchor},{right}"
        )
    rows = [
        dict(row)
        for row in catalog_by_video.get(video_id, [])
        if left <= int(row["source_frame_idx"]) <= right
    ]
    rows.sort(key=lambda row: (row["source_frame_idx"], row["timestamp_ms"], row["frame_uid"]))
    if not rows:
        raise RangeQrelsError(
            f"{decision.get('review_uid')}: no catalog frame in {video_id} [{left},{right}]"
        )
    interval = {
        "review_uid": str(decision["review_uid"]),
        "query_uid": str(decision["query_uid"]),
        "video_id": video_id,
        "status": str(decision["status"]),
        "left": left,
        "anchor": anchor,
        "right": right,
        "relevant_frame_ids": [row["frame_uid"] for row in rows],
        "relevant_frame_count": len(rows),
    }
    return interval, rows


def _query_from_item(item: Mapping[str, Any], query_uid: str, task_type: str) -> dict[str, Any]:
    return {
        "query_id": query_uid,
        "text": str(item.get("query") or ""),
        "task_type": task_type,
        "video_id": str(item.get("video_id") or ""),
        "source": str(item.get("source") or ""),
        "source_testcase_id": str(item.get("source_testcase_id") or ""),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_json(dict(row)) + "\n")
            count += 1
    return count


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _output_hashes(output: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name == "qrels_manifest.json":
            continue
        result[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def build_qrels_bundle(
    *,
    review_export_path: Path,
    review_items_path: Path,
    kis_qa_r1_path: Path,
    trake_r1_path: Path,
    catalog_paths: Sequence[Path],
    output_dir: Path,
    compare_catalog_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Build task-specific qrels and a provenance manifest.

    The range is inclusive in ``source_frame_idx``.  Rejected range decisions
    are excluded from scoreable qrels.  For TRAKE, a chain is scoreable only
    when every event is reviewed and mapped; this prevents a partially rejected
    sequence from being silently treated as a complete chain.
    """

    review_export_path = review_export_path.expanduser().resolve()
    review_items_path = review_items_path.expanduser().resolve()
    kis_qa_r1_path = kis_qa_r1_path.expanduser().resolve()
    trake_r1_path = trake_r1_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    export, decisions_by_uid = _load_decisions(review_export_path)
    items_by_uid = _load_review_items(review_items_path)
    catalog = load_catalog(catalog_paths)
    catalog_by_video = _build_catalog_by_video(catalog)
    catalog_uids = {row["frame_uid"] for row in catalog}

    comparison: dict[str, Any] = {"checked": False}
    if compare_catalog_paths:
        compare = load_catalog(compare_catalog_paths)
        primary_uids = {row["frame_uid"] for row in catalog}
        compare_uids = {row["frame_uid"] for row in compare}
        comparison = {
            "checked": True,
            "primary_rows": len(catalog),
            "compare_rows": len(compare),
            "primary_uid_hash": catalog_uid_hash(catalog),
            "compare_uid_hash": catalog_uid_hash(compare),
            "uid_set_equal": primary_uids == compare_uids,
            "primary_only_count": len(primary_uids - compare_uids),
            "compare_only_count": len(compare_uids - primary_uids),
        }
        if primary_uids != compare_uids:
            raise RangeQrelsError("primary and comparison catalogs have different frame_uid sets")

    kis_qa_r1 = _read_jsonl(kis_qa_r1_path)
    trake_r1 = _read_jsonl(trake_r1_path)
    if len(kis_qa_r1) != len({str(row.get("query_uid")) for row in kis_qa_r1}):
        raise RangeQrelsError("KIS/QA R1 contains duplicate query_uid")

    range_records: list[dict[str, Any]] = []
    mapping_issues: list[dict[str, Any]] = []
    mapped_by_review_uid: dict[str, dict[str, Any]] = {}
    for review_uid in sorted(decisions_by_uid):
        decision = decisions_by_uid[review_uid]
        item = items_by_uid.get(review_uid)
        if item is None:
            raise RangeQrelsError(f"decision has no review item: {review_uid}")
        record = {
            "review_uid": review_uid,
            "query_uid": str(decision.get("query_uid") or ""),
            "task_type": str(
                item.get("task") or _task_from_query_uid(str(decision.get("query_uid")))
            ),
            "video_id": str(decision.get("video_id") or item.get("video_id") or ""),
            "status": str(decision.get("status") or ""),
            "left": decision.get("left"),
            "anchor": decision.get("anchor"),
            "right": decision.get("right"),
            "reviewer": decision.get("reviewer"),
            "note": decision.get("note"),
            "scoreable": False,
        }
        if record["status"] in ACCEPTED_STATUSES:
            try:
                interval, _ = _map_range(decision, catalog_by_video)
            except RangeQrelsError as exc:
                mapping_issues.append(
                    {
                        "review_uid": review_uid,
                        "query_uid": record["query_uid"],
                        "code": "RANGE_MAPPING_FAILED",
                        "message": str(exc),
                    }
                )
            else:
                record.update(
                    {
                        "scoreable": True,
                        "relevant_frame_ids": interval["relevant_frame_ids"],
                        "relevant_frame_count": interval["relevant_frame_count"],
                    }
                )
                mapped_by_review_uid[review_uid] = interval
        elif record["status"] == REJECTED_STATUS:
            record["excluded_reason"] = "human_review_rejected"
        else:
            record["excluded_reason"] = f"unsupported_review_status:{record['status']}"
        range_records.append(record)

    if mapping_issues:
        raise RangeQrelsError(
            f"{len(mapping_issues)} range mappings failed; see mapping_issues.jsonl"
        )

    query_info: dict[str, dict[str, Any]] = {}
    for item in items_by_uid.values():
        query_uid = str(item.get("query_uid") or "")
        if not query_uid:
            continue
        task_type = str(item.get("task") or _task_from_query_uid(query_uid))
        if task_type == "kis":
            task_type = "kis"
        current = _query_from_item(item, query_uid, task_type)
        previous = query_info.get(query_uid)
        if previous is None:
            if task_type == "trake":
                current["_event_texts"] = {
                    _step_from_review_uid(str(item.get("review_uid") or "")) or 0: current["text"]
                }
            query_info[query_uid] = current
            continue
        if (
            previous["video_id"] != current["video_id"]
            or previous["task_type"] != current["task_type"]
        ):
            raise RangeQrelsError(f"inconsistent query metadata across review items: {query_uid}")
        if task_type != "trake" and previous["text"] != current["text"]:
            raise RangeQrelsError(f"inconsistent query text across review items: {query_uid}")
        if task_type == "trake":
            event_texts = previous.setdefault("_event_texts", {})
            step = _step_from_review_uid(str(item.get("review_uid") or "")) or 0
            event_texts[step] = current["text"]
            previous["text"] = "\n".join(
                f"event {event_step}: {event_texts[event_step]}"
                for event_step in sorted(event_texts)
                if event_texts[event_step]
            )

    for query in query_info.values():
        query.pop("_event_texts", None)

    kis_qrels: list[dict[str, Any]] = []
    qa_qrels: list[dict[str, Any]] = []
    r1_anchor_rows: list[dict[str, Any]] = []
    for r1_row in sorted(kis_qa_r1, key=lambda row: str(row["query_uid"])):
        query_uid = str(r1_row["query_uid"])
        decisions = [
            decision
            for decision in decisions_by_uid.values()
            if str(decision.get("query_uid")) == query_uid
            and _step_from_review_uid(str(decision["review_uid"])) is None
        ]
        if len(decisions) != 1:
            raise RangeQrelsError(
                f"expected one reviewed range for KIS/QA query {query_uid}, got {len(decisions)}"
            )
        decision = decisions[0]
        review_uid = str(decision["review_uid"])
        interval = mapped_by_review_uid.get(review_uid)
        positive_uid = str(r1_row.get("positive_frame_uid") or r1_row.get("frame_uid") or "")
        r1_anchor_rows.append(
            {
                "query_id": query_uid,
                "task_type": str(r1_row.get("task") or query_info[query_uid]["task_type"]),
                "event_step": None,
                "r1_frame_uid": positive_uid,
                "r1_frame_in_catalog": positive_uid in catalog_uids,
                "review_uid": review_uid,
                "review_status": str(decision.get("status")),
                "range_left": decision.get("left"),
                "range_anchor": decision.get("anchor"),
                "range_right": decision.get("right"),
            }
        )
        if interval is None:
            continue
        qrel = {
            "query_id": query_uid,
            "task_type": str(r1_row.get("task") or query_info[query_uid]["task_type"]),
            "relevant_frame_ids": interval["relevant_frame_ids"],
            "relevant_video_ids": [str(decision["video_id"])],
            "range_review_uids": [review_uid],
            "r1_positive_frame_uids": [positive_uid],
            "qrels_policy": "inclusive_source_frame_idx_range_plus_r1_provenance",
        }
        if qrel["task_type"] == "qa":
            answer = r1_row.get("reviewed_qa_answer")
            qrel["reference_answers"] = [answer] if answer else []
            qa_qrels.append(qrel)
        else:
            kis_qrels.append(qrel)

    trake_event_queries: list[dict[str, Any]] = []
    trake_event_qrels: list[dict[str, Any]] = []
    trake_qrels: list[dict[str, Any]] = []
    excluded_queries: list[dict[str, Any]] = []
    for chain in sorted(trake_r1, key=lambda row: str(row["query_uid"])):
        query_uid = str(chain["query_uid"])
        chain_events: list[dict[str, Any]] = []
        chain_exclusions: list[dict[str, Any]] = []
        for event in sorted(
            chain.get("events", []), key=lambda row: _int(row.get("step"), "trake step")
        ):
            step = _int(event.get("step"), f"{query_uid}:step")
            candidates = [
                decision
                for decision in decisions_by_uid.values()
                if str(decision.get("query_uid")) == query_uid
                and _step_from_review_uid(str(decision["review_uid"])) == step
            ]
            if len(candidates) != 1:
                chain_exclusions.append(
                    {
                        "step": step,
                        "reason": "missing_or_duplicate_review_decision",
                        "count": len(candidates),
                    }
                )
                continue
            decision = candidates[0]
            review_uid = str(decision["review_uid"])
            interval = mapped_by_review_uid.get(review_uid)
            r1_uid = str(event.get("frame_uid") or "")
            r1_anchor_rows.append(
                {
                    "query_id": query_uid,
                    "task_type": "trake",
                    "event_step": step,
                    "r1_frame_uid": r1_uid,
                    "r1_frame_in_catalog": r1_uid in catalog_uids,
                    "review_uid": review_uid,
                    "review_status": str(decision.get("status")),
                    "range_left": decision.get("left"),
                    "range_anchor": decision.get("anchor"),
                    "range_right": decision.get("right"),
                }
            )
            if interval is None:
                chain_exclusions.append(
                    {
                        "step": step,
                        "review_uid": review_uid,
                        "reason": "human_review_rejected_or_unmapped",
                    }
                )
                continue
            event_qrel = {
                # Event qrels must have a unique query_id. Reusing the chain
                # query_id would make the standard evaluator overwrite all
                # but one event when it loads JSONL qrels into a dict.
                "query_id": review_uid,
                "parent_query_id": query_uid,
                "event_id": review_uid,
                "event_step": step,
                "task_type": "trake",
                "relevant_frame_ids": interval["relevant_frame_ids"],
                "relevant_video_ids": [str(decision["video_id"])],
                "range_review_uid": review_uid,
                "r1_positive_frame_uid": r1_uid,
                "qrels_policy": "inclusive_source_frame_idx_range_plus_r1_provenance",
            }
            event_item = items_by_uid[review_uid]
            trake_event_queries.append(
                {
                    "query_id": review_uid,
                    "parent_query_id": query_uid,
                    "event_step": step,
                    "text": str(event_item.get("query") or ""),
                    "task_type": "trake_event",
                    "video_id": str(decision["video_id"]),
                    "source": str(event_item.get("source") or ""),
                    "source_testcase_id": str(event_item.get("source_testcase_id") or ""),
                }
            )
            trake_event_qrels.append(event_qrel)
            chain_events.append(
                {
                    "event_id": review_uid,
                    "step": step,
                    "relevant_frame_ids": interval["relevant_frame_ids"],
                    "r1_positive_frame_uid": r1_uid,
                    "range_left": decision["left"],
                    "range_anchor": decision["anchor"],
                    "range_right": decision["right"],
                }
            )
        if chain_exclusions:
            excluded_queries.append(
                {
                    "query_id": query_uid,
                    "task_type": "trake",
                    "reason": "chain_not_complete_after_human_review",
                    "excluded_events": chain_exclusions,
                }
            )
            continue
        if not chain_events:
            excluded_queries.append(
                {
                    "query_id": query_uid,
                    "task_type": "trake",
                    "reason": "empty_chain",
                }
            )
            continue
        trake_qrels.append(
            {
                "query_id": query_uid,
                "task_type": "trake",
                "expected_video_id": str(
                    chain.get("video_id") or query_info[query_uid]["video_id"]
                ),
                "events": chain_events,
                "strict_order": True,
                "qrels_policy": "inclusive_source_frame_idx_range_plus_r1_provenance",
            }
        )

    if not mapping_issues and len(kis_qrels) != 75:
        raise RangeQrelsError(f"expected 75 KIS qrels from R1, got {len(kis_qrels)}")
    if not mapping_issues and len(qa_qrels) != 11:
        raise RangeQrelsError(f"expected 11 QA qrels from R1, got {len(qa_qrels)}")

    scoreable_query_ids = {row["query_id"] for row in kis_qrels + qa_qrels + trake_qrels}
    queries = [query_info[query_id] for query_id in sorted(scoreable_query_ids)]
    kis_qa_queries = [query for query in queries if query["task_type"] in {"kis", "qa"}]
    trake_chain_queries = [query for query in queries if query["task_type"] == "trake"]
    kis_qa_qrels = sorted(kis_qrels + qa_qrels, key=lambda row: row["query_id"])
    trake_event_queries.sort(key=lambda row: (row["parent_query_id"], row["event_step"]))
    trake_event_qrels.sort(key=lambda row: (row["query_id"], row["event_step"]))
    trake_qrels.sort(key=lambda row: row["query_id"])
    range_records.sort(key=lambda row: row["review_uid"])
    r1_anchor_rows.sort(
        key=lambda row: (
            row["query_id"],
            row["event_step"] is not None,
            row["event_step"] or 0,
        )
    )

    files_rows: dict[str, int] = {}
    files_rows["queries.jsonl"] = _write_jsonl(output_dir / "queries.jsonl", queries)
    files_rows["kis_qa_queries.jsonl"] = _write_jsonl(
        output_dir / "kis_qa_queries.jsonl", kis_qa_queries
    )
    files_rows["trake_chain_queries.jsonl"] = _write_jsonl(
        output_dir / "trake_chain_queries.jsonl", trake_chain_queries
    )
    files_rows["qrels.jsonl"] = _write_jsonl(output_dir / "qrels.jsonl", kis_qa_qrels)
    files_rows["kis_qrels.jsonl"] = _write_jsonl(output_dir / "kis_qrels.jsonl", kis_qrels)
    files_rows["qa_qrels.jsonl"] = _write_jsonl(output_dir / "qa_qrels.jsonl", qa_qrels)
    files_rows["trake_event_queries.jsonl"] = _write_jsonl(
        output_dir / "trake_event_queries.jsonl", trake_event_queries
    )
    files_rows["trake_event_qrels.jsonl"] = _write_jsonl(
        output_dir / "trake_event_qrels.jsonl", trake_event_qrels
    )
    files_rows["trake_qrels.jsonl"] = _write_jsonl(output_dir / "trake_qrels.jsonl", trake_qrels)
    files_rows["range_review_records.jsonl"] = _write_jsonl(
        output_dir / "range_review_records.jsonl", range_records
    )
    files_rows["r1_anchor_provenance.jsonl"] = _write_jsonl(
        output_dir / "r1_anchor_provenance.jsonl", r1_anchor_rows
    )
    files_rows["excluded_queries.jsonl"] = _write_jsonl(
        output_dir / "excluded_queries.jsonl", excluded_queries
    )
    files_rows["mapping_issues.jsonl"] = _write_jsonl(
        output_dir / "mapping_issues.jsonl", mapping_issues
    )

    catalog_source_hashes = [
        {"path": str(path.expanduser().resolve()), "sha256": sha256_file(path)}
        for path in catalog_paths
    ]
    compare_source_hashes = [
        {"path": str(path.expanduser().resolve()), "sha256": sha256_file(path)}
        for path in compare_catalog_paths
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "hcmaic_range_qrels_bundle",
        "status": "ENGINEERING_ARTIFACT_COMPLETE",
        "qrels_status": "DRAFT_READY_FOR_INDEX_EVAL",
        "quality_status": QUALITY_STATUS,
        "evidence_level": EVIDENCE_LEVEL,
        "policy": {
            "frame_identity": "frame_uid=video_id:source_frame_idx",
            "interval_semantics": "inclusive source_frame_idx [left,right]",
            "rejected_decisions": "excluded",
            "trake_chain": "exclude chain when any event is rejected or unmapped",
            "trake_event_eval": (
                "event qrels use the unique range review_uid as query_id; "
                "chain qrels use the parent query_id"
            ),
            "standard_evaluator_inputs": (
                "kis_qa_queries.jsonl + qrels.jsonl, or "
                "trake_event_queries.jsonl + trake_event_qrels.jsonl"
            ),
            "r1_role": (
                "anchor provenance and query-set contract; not a substitute for catalog membership"
            ),
            "faiss_row_policy": "never used as identity",
        },
        "inputs": {
            "review_export": {
                "path": str(review_export_path),
                "sha256": sha256_file(review_export_path),
            },
            "review_items": {
                "path": str(review_items_path),
                "sha256": sha256_file(review_items_path),
            },
            "kis_qa_r1": {
                "path": str(kis_qa_r1_path),
                "sha256": sha256_file(kis_qa_r1_path),
            },
            "trake_r1": {
                "path": str(trake_r1_path),
                "sha256": sha256_file(trake_r1_path),
            },
        },
        "catalog": {
            "primary_sources": catalog_source_hashes,
            "comparison_sources": compare_source_hashes,
            "rows": len(catalog),
            "videos": len({row["video_id"] for row in catalog}),
            "identity_hash": canonical_catalog_hash(catalog),
            "uid_hash": catalog_uid_hash(catalog),
            "cross_channel_parity": comparison,
        },
        "counts": {
            "review_decisions": len(decisions_by_uid),
            "review_scoreable_ranges": sum(1 for row in range_records if row["scoreable"]),
            "review_rejected_ranges": sum(
                1 for row in range_records if row["status"] == REJECTED_STATUS
            ),
            "kis_qrels": len(kis_qrels),
            "qa_qrels": len(qa_qrels),
            "trake_event_queries": len(trake_event_queries),
            "trake_event_qrels": len(trake_event_qrels),
            "trake_chain_qrels": len(trake_qrels),
            "kis_qa_queries": len(kis_qa_queries),
            "trake_chain_queries": len(trake_chain_queries),
            "excluded_queries": len(excluded_queries),
            "mapping_issues": len(mapping_issues),
            "scoreable_queries": len(queries),
        },
        "r1": {
            "kis_qa_rows": len(kis_qa_r1),
            "trake_chain_rows": len(trake_r1),
            "anchor_provenance_rows": len(r1_anchor_rows),
            "r1_frame_uids_in_catalog": sum(
                1 for row in r1_anchor_rows if row["r1_frame_in_catalog"]
            ),
        },
        "evaluation_sets": {
            "kis_qa": {
                "queries": "kis_qa_queries.jsonl",
                "qrels": "qrels.jsonl",
                "query_count": len(kis_qa_queries),
                "protocol": "standard_frame_recall_mrr",
            },
            "kis": {
                "queries": "kis_qa_queries.jsonl",
                "qrels": "kis_qrels.jsonl",
                "query_count": len(kis_qrels),
                "protocol": "standard_frame_recall_mrr",
            },
            "qa": {
                "queries": "kis_qa_queries.jsonl",
                "qrels": "qa_qrels.jsonl",
                "query_count": len(qa_qrels),
                "protocol": "standard_frame_recall_mrr_plus_reference_answers",
            },
            "trake_event": {
                "queries": "trake_event_queries.jsonl",
                "qrels": "trake_event_qrels.jsonl",
                "query_count": len(trake_event_queries),
                "protocol": "standard_frame_recall_mrr_per_event",
            },
            "trake_chain": {
                "queries": "trake_chain_queries.jsonl",
                "qrels": "trake_qrels.jsonl",
                "query_count": len(trake_qrels),
                "protocol": "custom_strict_order_chain_eval",
            },
        },
        "files": files_rows,
    }
    _write_json(output_dir / "qrels_manifest.json", manifest)
    manifest["output_hashes"] = _output_hashes(output_dir)
    _write_json(output_dir / "qrels_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-export", type=Path, required=True)
    parser.add_argument("--review-items", type=Path, required=True)
    parser.add_argument("--kis-qa-r1", type=Path, required=True)
    parser.add_argument("--trake-r1", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, action="append", required=True)
    parser.add_argument("--compare-catalog", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_qrels_bundle(
        review_export_path=args.review_export,
        review_items_path=args.review_items,
        kis_qa_r1_path=args.kis_qa_r1,
        trake_r1_path=args.trake_r1,
        catalog_paths=args.catalog,
        compare_catalog_paths=args.compare_catalog,
        output_dir=args.output,
    )
    print(
        "RANGE_QRELS_READY "
        f"catalog={manifest['catalog']['rows']} "
        f"queries={manifest['counts']['scoreable_queries']} "
        f"kis={manifest['counts']['kis_qrels']} "
        f"qa={manifest['counts']['qa_qrels']} "
        f"trake_events={manifest['counts']['trake_event_qrels']} "
        f"trake_chains={manifest['counts']['trake_chain_qrels']} "
        f"excluded={manifest['counts']['excluded_queries']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
