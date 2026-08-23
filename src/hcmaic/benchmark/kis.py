"""Reproducible visual TKIS/VKIS benchmark without synthetic production scores."""

from __future__ import annotations

import datetime as dt
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from hcmaic.skillpixel.retrieval import (
    SkillPixelHit,
    SkillPixelQuestion,
    SkillPixelRetriever,
)


class KISBenchmarkError(RuntimeError):
    """Raised when a KIS benchmark cannot be run safely."""


def _code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _answer_cell(hit: SkillPixelHit) -> str:
    """Keep benchmark identity identical to the official submission contract."""
    return f"{hit.video_filename},{hit.source_frame_idx}"


def _normalise_qrels(qrels: dict[str, Any]) -> dict[str, set[str]]:
    normalised: dict[str, set[str]] = {}
    for query_id, values in qrels.items():
        if isinstance(values, str):
            items = [values]
        elif isinstance(values, (list, tuple, set, frozenset)):
            items = list(values)
        else:
            raise KISBenchmarkError(
                f"qrels for {query_id!r} must be a string or a sequence of answer cells"
            )
        cells = {str(item).strip() for item in items if str(item).strip()}
        if not cells:
            raise KISBenchmarkError(f"qrels for {query_id!r} is empty")
        normalised[str(query_id)] = cells
    return normalised


def _metric_values(
    results: dict[str, list[SkillPixelHit]],
    qrels: dict[str, Any] | None,
) -> dict[str, float | None]:
    keys = (1, 5, 20, 50, 100)
    metrics: dict[str, float | None] = {
        **{f"recall@{key}": None for key in keys},
        "mrr": None,
        "query_score": None,
    }
    if qrels is None:
        return metrics
    expected = _normalise_qrels(qrels)
    query_ids = list(results)
    if set(query_ids) != set(expected):
        missing = sorted(set(query_ids) - set(expected))
        extra = sorted(set(expected) - set(query_ids))
        raise KISBenchmarkError(f"qrels/query mismatch: missing={missing}, extra={extra}")

    reciprocal_ranks: list[float] = []
    query_scores: list[float] = []
    for query_id in query_ids:
        answers = expected[query_id]
        retrieved = [_answer_cell(hit) for hit in results[query_id]]
        first_rank = next(
            (rank for rank, answer in enumerate(retrieved, start=1) if answer in answers),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank is not None else 0.0)
        query_scores.append(1.0 if first_rank is not None else 0.0)
        for key in keys:
            if any(answer in answers for answer in retrieved[:key]):
                metric_name = f"recall@{key}"
                metrics[metric_name] = (metrics[metric_name] or 0.0) + 1.0
    n_queries = len(query_ids)
    for key in keys:
        value = metrics[f"recall@{key}"]
        metrics[f"recall@{key}"] = None if value is None else value / n_queries
    metrics["mrr"] = mean(reciprocal_ranks)
    metrics["query_score"] = mean(query_scores)
    return metrics


@dataclass(frozen=True)
class VisualBenchmarkReport:
    """Measured visual retrieval evidence for one immutable index/provider pair."""

    created_at: str
    code_version: str
    n_queries: int
    n_tkis: int
    n_vkis: int
    top_k: int
    provider: dict[str, Any]
    index: dict[str, Any]
    latency_ms: dict[str, float | None]
    metrics: dict[str, float | None]
    qrels_present: bool
    quality_status: str
    evidence_level: str = "VALIDATED_LOCAL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "hcmaic-kis-visual-benchmark-v1",
            "created_at": self.created_at,
            "code_version": self.code_version,
            "n_queries": self.n_queries,
            "n_tkis": self.n_tkis,
            "n_vkis": self.n_vkis,
            "top_k": self.top_k,
            "provider": dict(self.provider),
            "index": dict(self.index),
            "latency_ms": dict(self.latency_ms),
            "metrics": dict(self.metrics),
            "qrels_present": self.qrels_present,
            "quality_status": self.quality_status,
            "evidence_level": self.evidence_level,
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor() or "unknown",
            },
        }


def benchmark_visual_retrieval(
    retriever: SkillPixelRetriever,
    questions: list[SkillPixelQuestion],
    *,
    top_k: int = 100,
    query_root: Path | None = None,
    qrels: dict[str, Any] | None = None,
    qrels_source: str | None = None,
) -> VisualBenchmarkReport:
    """Benchmark batched TKIS/VKIS search while preserving input query IDs.

    The benchmark accepts qrels only as an explicitly supplied mapping.  No
    fixture or BTC artifact is discovered implicitly, and no quality metric is
    emitted when qrels are absent.  ``qrels_source`` must identify official
    HCMAIC qrels before the report can claim validated quality.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not questions:
        raise KISBenchmarkError("visual benchmark requires at least one question")
    if retriever.provider.name.lower() == "mock":
        raise KISBenchmarkError("mock embedding providers are not allowed in KIS benchmark")
    query_ids = [item.query_id for item in questions]
    if len(query_ids) != len(set(query_ids)):
        raise KISBenchmarkError("benchmark questions contain duplicate query_id values")
    if qrels is not None and qrels_source is None:
        raise KISBenchmarkError("qrels_source is required when qrels are supplied")

    tkis = [(item.query_id, item.text) for item in questions if item.task == "TKIS"]
    root = Path(query_root) if query_root is not None else None
    vkis: list[tuple[str, Path]] = []
    for item in questions:
        if item.task != "VKIS":
            continue
        image_path = Path(item.query_image)
        if not image_path.is_absolute():
            image_path = (root or Path.cwd()) / image_path
        vkis.append((item.query_id, image_path))

    task_latencies: dict[str, float] = {}
    results: dict[str, list[SkillPixelHit]] = {}
    if tkis:
        started = time.perf_counter()
        results.update(retriever.search_text_queries(tkis, top_k=top_k))
        task_latencies["tkis_batch"] = (time.perf_counter() - started) * 1000.0
    if vkis:
        started = time.perf_counter()
        results.update(retriever.search_image_queries(vkis, top_k=top_k))
        task_latencies["vkis_batch"] = (time.perf_counter() - started) * 1000.0
    if list(results) != query_ids:
        raise KISBenchmarkError("retriever did not preserve query order or query IDs")

    total_latency = sum(task_latencies.values())
    latency_ms: dict[str, float | None] = {
        "tkis_batch": task_latencies.get("tkis_batch"),
        "vkis_batch": task_latencies.get("vkis_batch"),
        "total_batch": total_latency,
        "mean_per_query": total_latency / len(questions),
        "p50_per_query": None,
        "p95_per_query": None,
    }
    official_qrels = qrels is not None and qrels_source == "hcmaic-official"
    quality_status = "VALIDATED_ON_HCMAIC" if official_qrels else "UNVALIDATED_ON_HCMAIC"
    index_manifest = retriever.index.index_manifest
    index_summary = {
        "format": index_manifest.get("format"),
        "index_version": index_manifest.get("index_version"),
        "index_provider": index_manifest.get("index_provider"),
        "exact": index_manifest.get("index_parameters", {}).get("exact"),
        "metric": index_manifest.get("index_parameters", {}).get("metric"),
        "n_frames": retriever.index.size,
        "dimension": retriever.index.dimension,
        "dataset_manifest_hash": index_manifest.get("dataset_manifest_hash"),
    }
    return VisualBenchmarkReport(
        created_at=dt.datetime.now(dt.UTC).isoformat(),
        code_version=_code_version(),
        n_queries=len(questions),
        n_tkis=len(tkis),
        n_vkis=len(vkis),
        top_k=top_k,
        provider=retriever.provider.info(),
        index=index_summary,
        latency_ms={
            key: round(value, 3) if value is not None else None for key, value in latency_ms.items()
        },
        metrics=_metric_values(results, qrels),
        qrels_present=qrels is not None,
        quality_status=quality_status,
    )


def write_visual_benchmark_report(report: VisualBenchmarkReport, path: Path) -> Path:
    """Persist a benchmark report without writing any retrieval artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
