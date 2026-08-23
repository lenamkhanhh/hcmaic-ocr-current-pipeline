"""Offline evaluator scaffold for canonical frame-level retrieval evidence.

This module is intentionally independent of model loading and FAISS.  It reads
already-produced ranked evidence, joins only on ``frame_uid``, and labels all
outputs ``ENGINEERING_PROXY``/``UNVALIDATED`` unless a caller explicitly
provides an approved official qrels source.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.fusion import reciprocal_rank_fusion

FORMAT = "hcmaic-offline-evaluation-v1"
_FRAME_UID_RE = re.compile(r"^[^:\s]+:[0-9]+$")
_CUTOFFS = (1, 5, 10, 20, 50, 100)


class OfflineEvaluationError(ValueError):
    """Raised when offline evidence cannot be joined by canonical identity."""


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value = value.get("rows", value.get("qrels", value))
            if not isinstance(value, list):
                raise OfflineEvaluationError(f"JSON rows must be a list: {path}")
            rows = value
        else:
            rows = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise OfflineEvaluationError(f"invalid JSONL at {path}:{line_number}") from exc
    except OSError as exc:
        raise OfflineEvaluationError(f"cannot read {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise OfflineEvaluationError(f"all rows must be JSON objects: {path}")
    return rows


def load_qrels_jsonl(path: Path) -> dict[str, set[str]]:
    """Read qrels without modifying or promoting the source artifact."""

    qrels: dict[str, set[str]] = {}
    for row_number, row in enumerate(_read_json_rows(Path(path)), 1):
        query_id = str(row.get("query_id", row.get("qid", ""))).strip()
        if not query_id:
            raise OfflineEvaluationError(f"qrels row {row_number}: query_id is empty")
        if query_id in qrels:
            raise OfflineEvaluationError(f"duplicate qrels query_id: {query_id}")
        values: Any = row.get("relevant_frame_uids", row.get("relevant_frame_ids"))
        if values is None:
            values = row.get("relevant", [])
        if isinstance(values, (str, int)):
            values = [values]
        if not isinstance(values, Iterable) or isinstance(values, Mapping):
            raise OfflineEvaluationError(f"qrels row {query_id}: relevant values must be a list")
        identities: set[str] = set()
        for value in values:
            if isinstance(value, Mapping):
                value = value.get("frame_uid", value.get("frame_id", ""))
            uid = str(value or "").strip()
            if uid:
                if not _FRAME_UID_RE.fullmatch(uid):
                    raise OfflineEvaluationError(
                        f"qrels row {query_id}: non-canonical frame_uid {uid!r}"
                    )
                identities.add(uid)
        qrels[query_id] = identities
    return qrels


def _uid_parts(frame_uid: str) -> tuple[str, int]:
    if not _FRAME_UID_RE.fullmatch(frame_uid):
        raise OfflineEvaluationError(
            "frame_uid must use the canonical video_id:source_frame_idx format"
        )
    video_id, raw_idx = frame_uid.rsplit(":", 1)
    return video_id, int(raw_idx)


def _as_hit(channel: str, raw: Any, position: int) -> ChannelHit:
    if isinstance(raw, ChannelHit):
        frame_uid = str(raw.frame_uid or raw.entity_id)
        video_id, source_frame_idx = _uid_parts(frame_uid)
        return ChannelHit(
            entity_id=frame_uid,
            video_id=video_id,
            timestamp_ms=int(raw.timestamp_ms),
            modality=channel,
            score=float(raw.score),
            rank=int(raw.rank),
            provider=str(raw.provider),
            evidence_text=raw.evidence_text,
            frame_uid=frame_uid,
            video_filename=raw.video_filename,
            source_frame_idx=source_frame_idx,
            evidence=dict(raw.evidence),
        )
    if not isinstance(raw, Mapping):
        raise OfflineEvaluationError(f"{channel} evidence row must be an object")
    frame_uid = str(raw.get("frame_uid") or "").strip()
    if not frame_uid:
        raise OfflineEvaluationError(f"{channel} evidence row is missing frame_uid")
    video_id, source_frame_idx = _uid_parts(frame_uid)
    supplied_video = str(raw.get("video_id") or video_id)
    if supplied_video != video_id:
        raise OfflineEvaluationError(
            f"{channel} frame_uid/video_id mismatch: {frame_uid!r} != {supplied_video!r}"
        )
    try:
        rank = int(raw.get("rank", position))
        timestamp_ms = int(raw.get("timestamp_ms", 0))
        score = float(raw.get("score", raw.get("final_score", 0.0)))
    except (TypeError, ValueError) as exc:
        raise OfflineEvaluationError(
            f"{channel} evidence has invalid rank/score/timestamp"
        ) from exc
    if rank < 1 or timestamp_ms < 0:
        raise OfflineEvaluationError(f"{channel} evidence rank/timestamp is out of range")
    return ChannelHit(
        entity_id=frame_uid,
        video_id=video_id,
        timestamp_ms=timestamp_ms,
        modality=channel,
        score=score,
        rank=rank,
        provider=str(raw.get("provider") or channel),
        evidence_text=str(raw.get("evidence_text")) if raw.get("evidence_text") else None,
        frame_uid=frame_uid,
        video_filename=(str(raw["video_filename"]) if raw.get("video_filename") else None),
        source_frame_idx=source_frame_idx,
        evidence={
            "model_id": raw.get("model_id"),
            "model_revision": raw.get("model_revision"),
            "dimension": raw.get("dimension"),
            # Retain this only as provenance; it is never passed as identity.
            "faiss_row": raw.get("faiss_row"),
        },
    )


def _retrieved_uids(hits: Iterable[ChannelHit]) -> list[str]:
    return [str(hit.frame_uid or hit.entity_id) for hit in hits]


def _metrics(
    retrieved: Mapping[str, list[str]],
    relevant: Mapping[str, set[str]] | None,
) -> dict[str, Any]:
    if relevant is None:
        return {
            "n_scored": 0,
            "recall_at": {str(cutoff): None for cutoff in _CUTOFFS},
            "mrr": None,
        }
    n_scored = len(retrieved)
    hits_at = {cutoff: 0 for cutoff in _CUTOFFS}
    reciprocal_ranks: list[float] = []
    for query_id, values in retrieved.items():
        targets = relevant.get(query_id, set())
        first_rank = next(
            (rank for rank, uid in enumerate(values, 1) if uid in targets),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for cutoff in _CUTOFFS:
            if first_rank is not None and first_rank <= cutoff:
                hits_at[cutoff] += 1
    return {
        "n_scored": n_scored,
        "recall_at": {
            str(cutoff): round(hits_at[cutoff] / n_scored, 6) if n_scored else None
            for cutoff in _CUTOFFS
        },
        "mrr": round(sum(reciprocal_ranks) / n_scored, 6) if n_scored else None,
    }


def _provenance(rows: Iterable[ChannelHit]) -> dict[str, Any]:
    model_ids: set[str] = set()
    revisions: set[str] = set()
    dimensions: set[int] = set()
    providers: set[str] = set()
    for row in rows:
        providers.add(row.provider)
        if row.evidence.get("model_id"):
            model_ids.add(str(row.evidence["model_id"]))
        if row.evidence.get("model_revision"):
            revisions.add(str(row.evidence["model_revision"]))
        if row.evidence.get("dimension") is not None:
            dimensions.add(int(row.evidence["dimension"]))
    return {
        "providers": sorted(providers),
        "model_ids": sorted(model_ids),
        "model_revisions": sorted(revisions),
        "dimensions": sorted(dimensions),
        "faiss_row": "diagnostic_only",
    }


def evaluate_offline(
    queries: Mapping[str, Mapping[str, Iterable[Any]]],
    *,
    qrels: Mapping[str, Iterable[str]] | None = None,
    qrels_source: str | None = None,
    qrels_approved: bool = False,
    top_k: int = 100,
    rank_constant: int = 60,
) -> dict[str, Any]:
    """Evaluate already-ranked channel evidence using the RRF contract."""

    if not queries:
        raise OfflineEvaluationError("at least one query is required")
    if top_k < 1 or top_k > 500:
        raise OfflineEvaluationError("top_k must be in [1, 500]")
    if rank_constant < 1:
        raise OfflineEvaluationError("rank_constant must be >= 1")
    normalized_qrels: dict[str, set[str]] | None = None
    if qrels is not None:
        normalized_qrels = {
            str(query_id): {str(uid) for uid in values} for query_id, values in qrels.items()
        }
        missing = sorted(set(queries) - set(normalized_qrels))
        extra = sorted(set(normalized_qrels) - set(queries))
        if missing or extra:
            raise OfflineEvaluationError(f"qrels/query mismatch: missing={missing}, extra={extra}")
        for _query_id, values in normalized_qrels.items():
            for uid in values:
                _uid_parts(uid)

    channel_retrieved: dict[str, dict[str, list[str]]] = {}
    channel_provenance: dict[str, list[dict[str, Any]]] = {}
    per_query: list[dict[str, Any]] = []
    all_channels: set[str] = set()
    for query_id, raw_channels in queries.items():
        if not isinstance(raw_channels, Mapping):
            raise OfflineEvaluationError(f"query {query_id} channels must be a mapping")
        hits_by_channel: dict[str, list[ChannelHit]] = {}
        query_payload: dict[str, Any] = {"query_id": str(query_id), "channels": {}}
        for raw_channel, raw_hits in raw_channels.items():
            channel = str(raw_channel).strip().lower()
            if channel in {"", "fusion"}:
                raise OfflineEvaluationError(f"invalid channel name: {raw_channel!r}")
            try:
                hit_list = list(raw_hits)
            except TypeError as exc:
                raise OfflineEvaluationError(
                    f"query {query_id} channel {channel} is not iterable"
                ) from exc
            hits = [_as_hit(channel, raw, position) for position, raw in enumerate(hit_list, 1)]
            # RRF validates rank and duplicate identity within a channel while
            # preserving the existing score semantics for valid inputs.
            hits_by_channel[channel] = hits
            all_channels.add(channel)
            channel_retrieved.setdefault(channel, {})[str(query_id)] = _retrieved_uids(
                sorted(hits, key=lambda hit: (hit.rank, hit.frame_uid or hit.entity_id))[:top_k]
            )
            channel_provenance.setdefault(channel, []).append(_provenance(hits))
            query_payload["channels"][channel] = {
                "retrieved_frame_uids": channel_retrieved[channel][str(query_id)],
                "row_count": len(hits),
            }
        fused = reciprocal_rank_fusion(hits_by_channel, rank_constant=rank_constant, top_k=top_k)
        fused_uids = _retrieved_uids(fused)
        query_payload["fusion"] = {
            "retrieved_frame_uids": fused_uids,
            "row_count": len(fused_uids),
            "rank_constant": rank_constant,
        }
        per_query.append(query_payload)
        channel_retrieved.setdefault("fusion", {})[str(query_id)] = fused_uids

    channel_reports: dict[str, Any] = {}
    for channel in sorted(all_channels):
        all_rows = [
            row
            for provenance_rows in [channel_provenance.get(channel, [])]
            for row in provenance_rows
        ]
        channel_reports[channel] = {
            "metrics": _metrics(channel_retrieved[channel], normalized_qrels),
            "provenance": {
                "providers": sorted(
                    {provider for row in all_rows for provider in row["providers"]}
                ),
                "model_ids": sorted({model for row in all_rows for model in row["model_ids"]}),
                "model_revisions": sorted(
                    {revision for row in all_rows for revision in row["model_revisions"]}
                ),
                "dimensions": sorted(
                    {dimension for row in all_rows for dimension in row["dimensions"]}
                ),
                "faiss_row": "diagnostic_only",
            },
        }
    channel_reports["fusion"] = {
        "metrics": _metrics(channel_retrieved["fusion"], normalized_qrels),
        "provenance": {
            "method": "rrf",
            "rank_constant": rank_constant,
            "identity_key": "frame_uid",
        },
    }
    quality_status = (
        "VALIDATED_ON_HCMAIC"
        if qrels_approved and qrels_source == "hcmaic-official"
        else "UNVALIDATED"
    )
    return {
        "format": FORMAT,
        "status": "ENGINEERING_PROXY",
        "execution_status": "ENGINEERING_PROXY",
        "quality_status": quality_status,
        "quality_note": (
            "Metrics are wiring evidence only; quality remains UNVALIDATED unless "
            "approved official qrels and protocol are supplied."
        ),
        "identity_policy": {
            "primary": "frame_uid",
            "format": "video_id:source_frame_idx",
            "faiss_row": "never_identity",
        },
        "fusion_contract": {"method": "rrf", "rank_constant": rank_constant, "top_k": top_k},
        "qrels": {
            "present": qrels is not None,
            "source": qrels_source,
            "approved": bool(qrels_approved and qrels_source == "hcmaic-official"),
        },
        "query_count": len(queries),
        "channels": channel_reports,
        "per_query": per_query,
    }


def write_offline_report(report: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write report, per-query rows and an open ledger as versioned outputs."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "offline_evaluation_report_v1.json"
    per_query_path = output / "offline_evaluation_per_query_v1.jsonl"
    ledger_path = output / "failure_ledger.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = report.get("per_query", [])
    per_query_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "hcmaic-offline-evaluation-failure-ledger-v1",
                "status": "OPEN",
                "quality_status": "UNVALIDATED",
                "failure_count": 0,
                "unresolved_count": 0,
                "failures": [],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"report": report_path, "per_query": per_query_path, "failure_ledger": ledger_path}
