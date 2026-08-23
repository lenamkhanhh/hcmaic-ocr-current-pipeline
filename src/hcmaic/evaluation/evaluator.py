"""Evaluator over queries.jsonl + qrels.jsonl.

queries.jsonl lines: {"query_id": str, "text": str, "task_type"?: str}
qrels.jsonl lines:   {"query_id": str, "relevant_frame_ids": [str, ...],
                      "relevant_video_ids"?: [str, ...]}

Metrics: Recall@1/5/10/100 (fraction of queries with >=1 relevant frame in the
cutoff), MRR (over frame relevance), p50/p95 search latency, invalid/missing
counts. The report clearly labels the evaluation mode; deterministic-mock
results validate plumbing only and never competition retrieval quality.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hcmaic.contracts.models import SearchRequest
from hcmaic.retrieval.service import RetrievalService

CUTOFFS = (1, 5, 10, 100)


@dataclass
class EvalQuery:
    query_id: str
    text: str
    task_type: str = "kis"


@dataclass
class Qrel:
    query_id: str
    relevant_frame_ids: set[str] = field(default_factory=set)
    relevant_video_ids: set[str] = field(default_factory=set)


def load_queries(path: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "query_id" not in data or "text" not in data:
                raise ValueError(f"{path}:{lineno}: query needs 'query_id' and 'text', got {data}")
            queries.append(
                EvalQuery(
                    query_id=str(data["query_id"]),
                    text=str(data["text"]),
                    task_type=str(data.get("task_type", "kis")),
                )
            )
    return queries


def load_qrels(path: Path) -> dict[str, Qrel]:
    qrels: dict[str, Qrel] = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "query_id" not in data:
                raise ValueError(f"{path}:{lineno}: qrel needs 'query_id'")
            qrels[str(data["query_id"])] = Qrel(
                query_id=str(data["query_id"]),
                relevant_frame_ids=set(map(str, data.get("relevant_frame_ids", []))),
                relevant_video_ids=set(map(str, data.get("relevant_video_ids", []))),
            )
    return qrels


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = rank - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def evaluate(
    service: RetrievalService,
    queries: list[EvalQuery],
    qrels: dict[str, Qrel],
    top_k: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (evaluation_report, per_query_results)."""
    top_k = max(top_k, max(CUTOFFS))
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    reciprocal_ranks: list[float] = []
    hits_at = {c: 0 for c in CUTOFFS}
    n_missing_qrels = 0
    n_empty_results = 0

    for query in queries:
        qrel = qrels.get(query.query_id)
        request = SearchRequest(
            query_id=query.query_id,
            task_type=query.task_type,
            text=query.text,
            top_k=top_k,
        )
        start = time.perf_counter()
        results = service.search(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(latency_ms)

        retrieved = [r.frame_id for r in results]
        if not retrieved:
            n_empty_results += 1
        if qrel is None or not qrel.relevant_frame_ids:
            n_missing_qrels += 1
            per_query.append(
                {
                    "query_id": query.query_id,
                    "text": query.text,
                    "latency_ms": round(latency_ms, 3),
                    "retrieved": retrieved,
                    "error": "no qrels for this query",
                }
            )
            continue

        first_rank: int | None = None
        for rank, frame_id in enumerate(retrieved, start=1):
            if frame_id in qrel.relevant_frame_ids:
                first_rank = rank
                break
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        for cutoff in CUTOFFS:
            if first_rank is not None and first_rank <= cutoff:
                hits_at[cutoff] += 1

        per_query.append(
            {
                "query_id": query.query_id,
                "text": query.text,
                "latency_ms": round(latency_ms, 3),
                "retrieved": retrieved,
                "relevant": sorted(qrel.relevant_frame_ids),
                "first_relevant_rank": first_rank,
                "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            }
        )

    n_scored = len(reciprocal_ranks)
    latencies_sorted = sorted(latencies)
    provider = service.text_provider.name
    mode = "deterministic-mock" if provider == "mock" else f"real-{provider}-smoke"
    report: dict[str, Any] = {
        "mode": mode,
        "disclaimer": (
            "Fixture/mock evaluations validate plumbing only and must never "
            "be presented as competition retrieval quality."
        ),
        "index_version": service.index_version,
        "top_k": top_k,
        "n_queries": len(queries),
        "n_scored": n_scored,
        "n_missing_qrels": n_missing_qrels,
        "n_empty_results": n_empty_results,
        "recall_at": {str(c): (hits_at[c] / n_scored if n_scored else 0.0) for c in CUTOFFS},
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "latency_ms": {
            "p50": round(_percentile(latencies_sorted, 0.50), 3),
            "p95": round(_percentile(latencies_sorted, 0.95), 3),
            "mean": round(statistics.mean(latencies), 3) if latencies else 0.0,
        },
    }
    return report, per_query


def write_reports(
    report: dict[str, Any],
    per_query: list[dict[str, Any]],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "evaluation_report.json"
    per_query_path = out_dir / "per_query_results.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(per_query_path, "w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report_path, per_query_path


def format_summary(report: dict[str, Any]) -> str:
    recall = report["recall_at"]
    latency = report["latency_ms"]
    lines = [
        f"Evaluation mode: {report['mode']}  (index {report['index_version']})",
        f"Queries: {report['n_queries']}  scored: {report['n_scored']}  "
        f"missing qrels: {report['n_missing_qrels']}  empty results: "
        f"{report['n_empty_results']}",
        f"Recall@1: {recall['1']:.3f}  Recall@5: {recall['5']:.3f}  "
        f"Recall@10: {recall['10']:.3f}  MRR: {report['mrr']:.3f}",
        f"Latency p50: {latency['p50']} ms  p95: {latency['p95']} ms",
        f"NOTE: {report['disclaimer']}",
    ]
    return "\n".join(lines)
