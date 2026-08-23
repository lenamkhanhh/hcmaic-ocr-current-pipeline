"""Freeze benchmark inputs, build artifacts, evaluate, and write evidence."""

from __future__ import annotations

import datetime as dt
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from hcmaic.config import config_hash, load_config
from hcmaic.contracts.models import SearchRequest
from hcmaic.embedding.base import get_provider
from hcmaic.evaluation.evaluator import evaluate, load_qrels, load_queries
from hcmaic.indexing.artifacts import build_index_artifacts
from hcmaic.ingestion.catalog import build_catalog
from hcmaic.ingestion.manifest import build_dataset_manifest, sha256_file
from hcmaic.ingestion.validator import validate_dataset
from hcmaic.retrieval.service import load_service


def _params(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    return dict(value)


def _resolve(config_path: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _code_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_benchmark(config_path: Path, out_dir: Path) -> dict[str, Path]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    params = _params(config.benchmark_inputs.params)
    required = ("dataset_root", "queries", "qrels")
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"benchmark_inputs.params missing required fields: {missing}")

    dataset_root = _resolve(config_path, params["dataset_root"])
    query_path = _resolve(config_path, params["queries"])
    qrels_path = _resolve(config_path, params["qrels"])
    warmups = int(params.get("warmups", 1))
    repeats = int(params.get("repeats", 3))
    top_k = int(params.get("top_k", 100))
    if warmups < 0 or repeats < 1 or top_k < 1:
        raise ValueError("warmups >= 0, repeats >= 1, and top_k >= 1 are required")

    validation = validate_dataset(dataset_root, check_images=True)
    if not validation.ok:
        raise ValueError(f"benchmark dataset has {len(validation.errors)} validation error(s)")
    queries = load_queries(query_path)
    qrels = load_qrels(qrels_path)
    dataset_manifest = build_dataset_manifest(dataset_root)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_dir = out_dir / "_index"
    provider = get_provider(config.embedding_provider.name)
    catalog = build_catalog(dataset_root)
    indexing_start = time.perf_counter()
    build_index_artifacts(
        dataset_root,
        catalog,
        provider,
        index_dir,
        index_provider=config.index_provider.name,
        foundation_config=config,
    )
    indexing_time_ms = (time.perf_counter() - indexing_start) * 1000
    service = load_service(index_dir, dataset_root=dataset_root)

    for _ in range(warmups):
        for query in queries:
            service.search(
                SearchRequest(
                    query_id=f"warmup-{query.query_id}",
                    task_type=query.task_type,
                    text=query.text,
                    top_k=top_k,
                )
            )

    run_reports: list[dict[str, Any]] = []
    all_per_query: list[dict[str, Any]] = []
    for repeat in range(repeats):
        report, per_query = evaluate(service, queries, qrels, top_k=top_k)
        run_reports.append(report)
        for row in per_query:
            all_per_query.append({"repeat": repeat + 1, **row})

    last_report = run_reports[-1]
    summary = {
        **last_report,
        "mode": "proxy_fixture_non_competitive",
        "evidence_level": "FIXTURE_VERIFIED",
        "indexing_time_ms": round(indexing_time_ms, 3),
        "repeat_count": repeats,
        "warmup_count": warmups,
        "repeat_mrr_mean": round(statistics.mean(report["mrr"] for report in run_reports), 6),
        "timestamp_error_ms": None,
        "slice_metrics": {
            "unsliced_fixture": {
                "n_queries": len(queries),
                "recall_at_1": last_report["recall_at"]["1"],
            }
        },
        "ann_recall_at_k": None,
        "disclaimer": (
            "Deterministic proxy fixture validates benchmark plumbing only; "
            "these are not BTC scores or competition-quality evidence."
        ),
    }
    repo_root = Path(__file__).resolve().parents[3]
    run_manifest = {
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "evidence_level": "FIXTURE_VERIFIED",
        "dataset_hash": dataset_manifest["dataset_hash"],
        "query_hash": sha256_file(query_path),
        "qrels_hash": sha256_file(qrels_path),
        "config_hash": config_hash(config),
        "code_commit_sha": _code_sha(repo_root),
        "configuration": config.to_dict(),
        "provider": provider.info(),
        "index_provider": config.index_provider.to_dict(),
        "fusion": config.fusion.to_dict(),
        "reranker": config.reranker.to_dict(),
        "seed": config.seed,
        "warmups": warmups,
        "repeats": repeats,
        "top_k": top_k,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
    }

    summary_path = out_dir / "benchmark_summary.json"
    per_query_path = out_dir / "per_query_results.jsonl"
    manifest_path = out_dir / "run_manifest.json"
    failure_path = out_dir / "failure_cases.md"
    _write_json(summary_path, summary)
    with open(per_query_path, "w", encoding="utf-8") as stream:
        for row in all_per_query:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(manifest_path, run_manifest)
    failures = [row for row in all_per_query if row.get("first_relevant_rank") is None]
    lines = [
        "# Benchmark failure cases",
        "",
        "> Proxy fixture evidence only; not BTC competition evidence.",
        "",
        f"Failures: {len(failures)} / {len(all_per_query)} repeated query runs.",
    ]
    for row in failures:
        lines.extend(
            [
                "",
                f"## {row['query_id']} (repeat {row['repeat']})",
                "",
                f"- Query: {row.get('text', '')}",
                f"- Retrieved: {row.get('retrieved', [])}",
            ]
        )
    failure_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "benchmark_summary": summary_path,
        "per_query_results": per_query_path,
        "run_manifest": manifest_path,
        "failure_cases": failure_path,
    }
