"""HCMAIC KIS qrels adapter, metrics and channel ablation harness."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmaic.contracts.kis import KISQuery
from hcmaic.runtime.kis import KISRuntime
from hcmaic.skillpixel.retrieval import SkillPixelQuestion

CUTOFFS = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class KISQrel:
    query_id: str
    relevant_frame_uids: frozenset[str] = frozenset()
    relevant_answer_cells: frozenset[str] = frozenset()
    relevant_video_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class KISQrelSet:
    qrels: dict[str, KISQrel]
    source: str = "unknown"

    @property
    def quality_status(self) -> str:
        return (
            "VALIDATED_ON_HCMAIC" if self.source == "hcmaic-official" else "UNVALIDATED_ON_HCMAIC"
        )


class KISEvaluationError(ValueError):
    """Raised when qrels or KIS evaluation inputs are inconsistent."""


def _as_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [value]
    if isinstance(value, list | tuple | set | frozenset):
        return list(value)
    raise KISEvaluationError(f"qrel value must be scalar or sequence, got {type(value).__name__}")


def _read_qrel_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise KISEvaluationError(f"{path}:{line_number}: invalid JSON qrel") from exc
                if not isinstance(data, dict):
                    raise KISEvaluationError(
                        f"{path}:{line_number}: qrel must be an object"
                    ) from None
                rows.append(data)
        return rows
    if isinstance(payload, dict):
        if "qrels" in payload:
            payload = payload["qrels"]
        elif "query_id" in payload or "qid" in payload:
            payload = [payload]
        else:
            payload = [dict(value, query_id=key) for key, value in payload.items()]
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise KISEvaluationError("qrels JSON must be a list/object or JSONL objects")
    return payload


def _answer_cell(value: Any) -> str | None:
    if isinstance(value, str):
        cell = value.strip()
        return cell or None
    if not isinstance(value, dict):
        return None
    filename = str(value.get("video_filename", value.get("video", ""))).strip()
    raw_idx = value.get("source_frame_idx", value.get("frame_idx"))
    if not filename or raw_idx is None:
        return None
    try:
        source_frame_idx = int(raw_idx)
    except (TypeError, ValueError) as exc:
        raise KISEvaluationError(f"invalid qrel source_frame_idx {raw_idx!r}") from exc
    if source_frame_idx < 0:
        raise KISEvaluationError("qrel source_frame_idx must be non-negative")
    return f"{filename},{source_frame_idx}"


def load_kis_qrels(path: Path, *, source: str = "unknown") -> KISQrelSet:
    """Load official-like qrels without treating fixture files as official evidence."""
    rows = _read_qrel_rows(Path(path))
    qrels: dict[str, KISQrel] = {}
    for row_number, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id", row.get("qid", ""))).strip()
        if not query_id:
            raise KISEvaluationError(f"qrel row {row_number}: query_id is empty")
        if query_id in qrels:
            raise KISEvaluationError(f"duplicate qrel query_id {query_id!r}")
        frame_values = _as_values(row.get("relevant_frame_uids", row.get("relevant_frame_ids")))
        answer_values = _as_values(
            row.get(
                "relevant_answer_cells",
                row.get("relevant_answers", row.get("answers")),
            )
        )
        video_values = _as_values(row.get("relevant_video_ids", row.get("video_ids")))
        relevant_answers = {cell for value in answer_values if (cell := _answer_cell(value))}
        for value in _as_values(row.get("relevant")):
            if isinstance(value, dict):
                cell = _answer_cell(value)
                if cell:
                    relevant_answers.add(cell)
                frame_values.extend(_as_values(value.get("frame_uid", value.get("frame_id"))))
                video_values.extend(_as_values(value.get("video_id")))
        qrels[query_id] = KISQrel(
            query_id=query_id,
            relevant_frame_uids=frozenset(
                str(value).strip() for value in frame_values if str(value).strip()
            ),
            relevant_answer_cells=frozenset(relevant_answers),
            relevant_video_ids=frozenset(
                str(value).strip() for value in video_values if str(value).strip()
            ),
        )
    return KISQrelSet(qrels=qrels, source=source)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _query_from_question(
    question: SkillPixelQuestion, *, query_root: Path | None, top_k: int
) -> KISQuery:
    image_path = Path(question.query_image)
    if question.task == "VKIS" and not image_path.is_absolute():
        image_path = (query_root or Path.cwd()) / image_path
    return KISQuery(
        query_id=question.query_id,
        task=question.task,
        text=question.text or None,
        image_path=image_path if question.task == "VKIS" else None,
        top_k=top_k,
    )


def _is_relevant(result: Any, qrel: KISQrel, frame_tolerance: int) -> bool:
    if result.frame_uid in qrel.relevant_frame_uids:
        return True
    if result.answer_cell in qrel.relevant_answer_cells:
        return True
    if result.video_id in qrel.relevant_video_ids:
        return True
    for cell in qrel.relevant_answer_cells:
        filename, separator, raw_idx = cell.rpartition(",")
        if not separator or filename != result.video_filename:
            continue
        try:
            target_idx = int(raw_idx)
        except ValueError:
            continue
        if abs(target_idx - result.source_frame_idx) <= frame_tolerance:
            return True
    return False


def evaluate_kis_runtime(
    runtime: KISRuntime,
    questions: list[SkillPixelQuestion],
    qrels: KISQrelSet | None,
    *,
    top_k: int = 100,
    query_root: Path | None = None,
    frame_tolerance: int = 12,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate KIS with explicit qrels; absent qrels yield null quality metrics."""
    if top_k < max(CUTOFFS) or frame_tolerance < 0:
        raise ValueError("top_k must be >= 100 and frame_tolerance must be >= 0")
    if not questions:
        raise ValueError("KIS evaluation requires at least one question")
    query_ids = [question.query_id for question in questions]
    if len(query_ids) != len(set(query_ids)):
        raise KISEvaluationError("questions contain duplicate query_id values")
    if qrels is not None:
        missing = sorted(set(query_ids) - set(qrels.qrels))
        extra = sorted(set(qrels.qrels) - set(query_ids))
        if missing or extra:
            raise KISEvaluationError(f"qrels/question mismatch: missing={missing}, extra={extra}")

    latencies: list[float] = []
    per_query: list[dict[str, Any]] = []
    hits_at = {cutoff: 0 for cutoff in CUTOFFS}
    reciprocal_ranks: list[float] = []
    n_empty = 0
    n_invalid = 0
    n_scored = 0
    for question in questions:
        query = _query_from_question(question, query_root=query_root, top_k=top_k)
        started = time.perf_counter()
        output = runtime.search(query)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        results = output.results
        if not results:
            n_empty += 1
        invalid = [
            result for result in results if result.source_frame_idx < 0 or not result.video_filename
        ]
        n_invalid += len(invalid)
        row: dict[str, Any] = {
            "query_id": question.query_id,
            "task": question.task,
            "latency_ms": round(latency_ms, 3),
            "retrieved": [result.answer_cell for result in results],
            "executed_channels": list(output.executed_channels),
            "unavailable_channels": dict(output.unavailable_channels),
        }
        if qrels is None:
            row["quality_status"] = "UNVALIDATED_ON_HCMAIC"
            per_query.append(row)
            continue
        qrel = qrels.qrels[question.query_id]
        n_scored += 1
        first_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if _is_relevant(result, qrel, frame_tolerance)
            ),
            None,
        )
        reciprocal = 1.0 / first_rank if first_rank else 0.0
        reciprocal_ranks.append(reciprocal)
        for cutoff in CUTOFFS:
            if first_rank is not None and first_rank <= cutoff:
                hits_at[cutoff] += 1
        row.update(
            {
                "first_relevant_rank": first_rank,
                "reciprocal_rank": reciprocal,
                "quality_status": qrels.quality_status,
            }
        )
        per_query.append(row)

    quality_status = qrels.quality_status if qrels is not None else "UNVALIDATED_ON_HCMAIC"
    report = {
        "format": "hcmaic-kis-evaluation-v1",
        "quality_status": quality_status,
        "qrels_source": qrels.source if qrels is not None else None,
        "n_queries": len(questions),
        "n_scored": n_scored,
        "n_empty_results": n_empty,
        "n_invalid_results": n_invalid,
        "top_k": top_k,
        "frame_tolerance": frame_tolerance,
        "recall_at": {
            str(cutoff): (hits_at[cutoff] / n_scored if n_scored else None) for cutoff in CUTOFFS
        },
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
        "query_score": None,
        "query_score_note": "Official HCMAIC QueryScore definition is not available in checkout.",
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "mean": statistics.mean(latencies) if latencies else None,
        },
    }
    return report, per_query


def run_kis_ablation(
    runtime: KISRuntime,
    questions: list[SkillPixelQuestion],
    qrels: KISQrelSet | None,
    *,
    top_k: int = 100,
    query_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run visual baseline and configured channel ablations with identical inputs."""
    available = runtime.orchestrator.optional_channels
    modes = {
        "visual": set(),
        "visual+ocr": {"ocr"},
        "visual+object": {"object"},
        "visual+asr": {"asr"},
        "all-configured": set(available),
    }
    reports: dict[str, dict[str, Any]] = {}
    for mode, enabled in modes.items():
        channels = {
            name: channel if name in enabled else None for name, channel in available.items()
        }
        from hcmaic.retrieval.kis_orchestrator import KISHybridOrchestrator

        orchestrator = KISHybridOrchestrator(
            runtime.retriever,
            optional_channels=channels,
            fusion_method=runtime.orchestrator.fusion_method,
            fusion_weights=runtime.orchestrator.fusion_weights,
            rank_constant=runtime.orchestrator.rank_constant,
            candidate_multiplier=runtime.orchestrator.candidate_multiplier,
            max_per_video=runtime.orchestrator.max_per_video,
            reranker=runtime.orchestrator.reranker,
            rerank_timeout_ms=runtime.orchestrator.rerank_timeout_ms,
            asr_enabled=runtime.orchestrator.asr_enabled and "asr" in enabled,
        )
        ablation_runtime = KISRuntime(
            index=runtime.index,
            provider=runtime.provider,
            retriever=runtime.retriever,
            orchestrator=orchestrator,
            provider_selection=runtime.provider_selection,
            channel_status=runtime.channel_status,
        )
        report, _ = evaluate_kis_runtime(
            ablation_runtime,
            questions,
            qrels,
            top_k=top_k,
            query_root=query_root,
        )
        report["mode"] = mode
        report["enabled_optional_channels"] = sorted(enabled)
        reports[mode] = report
    return reports


def write_kis_evaluation_report(
    report: dict[str, Any], per_query: Iterable[dict[str, Any]], out_dir: Path
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "kis_evaluation_report.json"
    per_query_path = out_dir / "kis_per_query.jsonl"
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with per_query_path.open("w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"summary": summary_path, "per_query": per_query_path}
