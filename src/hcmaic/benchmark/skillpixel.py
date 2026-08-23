"""SkillPixel-only visual/provider/channel benchmark and artifact packaging."""

from __future__ import annotations

import csv
import ctypes
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

from hcmaic.skillpixel.index import load_skillpixel_index
from hcmaic.skillpixel.raw import validate_raw_dataset
from hcmaic.skillpixel.retrieval import (
    SkillPixelHit,
    SkillPixelQuestion,
    SkillPixelRetriever,
    load_skillpixel_questions,
)
from hcmaic.skillpixel.submission import (
    export_skillpixel_submission,
    validate_submission_csv,
    write_results_jsonl,
)

QUALITY_UNVALIDATED = "UNVALIDATED_ON_SKILLPIXEL_QRELS"
QUALITY_VALIDATED = "VALIDATED_ON_SKILLPIXEL_QRELS"
VISUAL_VARIANTS = ("V0", "V1", "V2")
CHANNEL_VARIANTS = ("C0", "C1", "C2", "C3", "C4", "C5")
EVIDENCE_FIELDS = (
    "query_id",
    "query_type",
    "query_order",
    "rank",
    "video_id",
    "video_filename",
    "keyframe_id",
    "frame_uid",
    "source_frame_idx",
    "timestamp_ms",
    "preview_path",
    "image_path",
    "visual_score",
    "ocr_score",
    "object_score",
    "asr_score",
    "rrf_score",
    "rerank_score",
    "provider",
    "model",
    "revision",
    "faiss_row",
    "feature_row",
)


class SkillPixelBenchmarkError(RuntimeError):
    """Raised when a benchmark would produce incomplete or misleading evidence."""


@dataclass(frozen=True)
class SkillPixelBenchmarkConfig:
    raw_root: Path
    index_dir: Path
    questions_path: Path
    corpus_path: Path
    output_dir: Path
    top_k: int = 100
    qrels: dict[str, Any] | None = None
    qrels_source: str | None = None
    ocr_artifact: Path | None = None
    object_artifact: Path | None = None

    def __post_init__(self) -> None:
        if self.top_k < 100 or self.top_k > 500:
            raise ValueError("SkillPixel submission benchmark top_k must be in [100, 500]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return _sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(child.relative_to(path).as_posix().encode("utf-8"))
            digest.update(_sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def _code_sha() -> str:
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


def _dir_size_mb(path: Path) -> float:
    return round(
        sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())
        / (1024 * 1024),
        3,
    )


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 3)
    except (ImportError, OSError):
        if sys.platform != "win32":
            return None
        try:
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("page_fault_count", ctypes.c_ulong),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            process = get_current_process()
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            get_process_memory_info.restype = wintypes.BOOL
            ok = get_process_memory_info(process, ctypes.byref(counters), counters.cb)
            if ok:
                return round(counters.working_set_size / (1024 * 1024), 3)
        except (AttributeError, OSError, TypeError, ctypes.ArgumentError):
            return None
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 3)


def _resolve_query_image(question: SkillPixelQuestion, query_root: Path) -> Path:
    path = Path(question.query_image)
    return path if path.is_absolute() else query_root / path


def _question_batches(
    questions: list[SkillPixelQuestion], query_root: Path
) -> tuple[list[tuple[str, str]], list[tuple[str, Path]]]:
    tkis = [(item.query_id, item.text) for item in questions if item.task == "TKIS"]
    vkis = [
        (item.query_id, _resolve_query_image(item, query_root))
        for item in questions
        if item.task == "VKIS"
    ]
    return tkis, vkis


def _answer_cell(hit: SkillPixelHit) -> str:
    return f"{hit.video_filename},{hit.source_frame_idx}"


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _model_registry_entry(
    *, provider: Any, requested_provider: str, selection: Mapping[str, Any]
) -> dict[str, Any]:
    info = provider.info()
    model_id = str(info.get("model_name", provider.name))
    model_path: str | None = None
    try:
        candidate = Path(model_id)
        if candidate.is_absolute() and candidate.exists():
            model_path = str(candidate.resolve())
    except (OSError, ValueError):
        model_path = None
    return {
        "requested_provider": requested_provider,
        "selected_provider": provider.name,
        "model_id": model_id,
        "revision": info.get("model_revision", info.get("revision", provider.version)),
        "weights_path": model_path,
        "weights_available": model_path is not None,
        "device": info.get("device"),
        "dtype": info.get("dtype"),
        "dimension": provider.dimension,
        "preprocess_hash": info.get("preprocess_hash"),
        "dependencies": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
        "network_required": not bool(info.get("local_files_only", True)),
        "fallback": selection.get("fallback"),
        "provider_execution": "validated-local",
        "training_status": "not_run",
    }


def _mapping_validation(index: Any) -> dict[str, Any]:
    errors: list[str] = []
    for row, entry in enumerate(index.id_map):
        record = index.catalog[row]
        source_frame_idx = (
            record.source_frame_idx if record.source_frame_idx is not None else record.frame_idx
        )
        expected = {
            "faiss_row": row,
            "feature_row": row,
            "frame_uid": record.frame_id,
            "video_id": record.video_id,
            "source_frame_idx": source_frame_idx,
            "timestamp_ms": record.timestamp_ms,
        }
        for field, value in expected.items():
            if entry.get(field) != value:
                errors.append(f"row {row}: {field}={entry.get(field)!r} != {value!r}")
        frame_count = entry.get("frame_count")
        if frame_count is not None and not 0 <= int(source_frame_idx) < int(frame_count):
            errors.append(f"row {row}: source_frame_idx is outside frame_count")
    faiss_ntotal = int(getattr(index.faiss_index, "ntotal", -1))
    if faiss_ntotal != len(index.id_map):
        errors.append(f"faiss ntotal {faiss_ntotal} != id_map rows {len(index.id_map)}")
    return {
        "ok": not errors,
        "n_checked": len(index.id_map),
        "n_errors": len(errors),
        "errors": errors[:20],
        "faiss_ntotal": faiss_ntotal,
        "id_map_rows": len(index.id_map),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_evidence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EVIDENCE_FIELDS))
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in EVIDENCE_FIELDS} for row in rows)


def _write_checksums(output_dir: Path) -> Path:
    checksum_path = Path(output_dir) / "checksums.sha256"
    lines: list[str] = []
    for path in sorted(Path(output_dir).rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        lines.append(f"{_sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return checksum_path


def _write_visual_run_contracts(
    *,
    config: SkillPixelBenchmarkConfig,
    candidate_dir: Path,
    questions: list[SkillPixelQuestion],
    results: Mapping[str, list[SkillPixelHit]],
    provider: Any,
    requested_provider: str,
    selection: Mapping[str, Any],
    index: Any,
    raw_manifest: Mapping[str, Any],
) -> dict[str, Path]:
    model = _model_registry_entry(
        provider=provider, requested_provider=requested_provider, selection=selection
    )
    evidence_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    query_root = Path(config.questions_path).parent
    for query_order, question in enumerate(questions):
        hits = list(results.get(question.query_id, []))
        status_rows.append(
            {
                "query_id": question.query_id,
                "query_type": question.task,
                "query_order": query_order,
                "status": "ok" if hits else "empty",
                "error": None if hits else "retrieval returned no hits",
                "n_results": len(hits),
                "top_k": config.top_k,
                "channels": ["visual"],
                "provider": provider.name,
                "model": model["model_id"],
                "revision": model["revision"],
            }
        )
        for hit in hits[: config.top_k]:
            preview_path = (Path(config.raw_root) / hit.image_path).resolve()
            evidence_rows.append(
                {
                    "query_id": hit.query_id,
                    "query_type": hit.task,
                    "query_order": query_order,
                    "rank": hit.rank,
                    "video_id": hit.video_id,
                    "video_filename": hit.video_filename,
                    "keyframe_id": hit.keyframe_id,
                    "frame_uid": hit.frame_uid,
                    "source_frame_idx": hit.source_frame_idx,
                    "timestamp_ms": hit.timestamp_ms,
                    "preview_path": str(preview_path),
                    "image_path": hit.image_path,
                    "visual_score": hit.visual_score,
                    "ocr_score": None,
                    "object_score": None,
                    "asr_score": None,
                    "rrf_score": None,
                    "rerank_score": None,
                    "provider": provider.name,
                    "model": model["model_id"],
                    "revision": model["revision"],
                    "faiss_row": hit.faiss_row,
                    "feature_row": hit.feature_row,
                }
            )

    evidence_top100_path = candidate_dir / "retrieval_evidence_top100.jsonl"
    evidence_top20_path = candidate_dir / "retrieval_evidence_top20.jsonl"
    evidence_top100_csv = candidate_dir / "retrieval_evidence_top100.csv"
    evidence_top20_csv = candidate_dir / "retrieval_evidence_top20.csv"
    _write_jsonl(evidence_top100_path, evidence_rows)
    top20_rows = [row for row in evidence_rows if int(row["rank"]) <= 20]
    _write_jsonl(evidence_top20_path, top20_rows)
    _write_evidence_csv(evidence_top100_csv, evidence_rows)
    _write_evidence_csv(evidence_top20_csv, top20_rows)
    query_status_path = candidate_dir / "query_status.jsonl"
    _write_jsonl(query_status_path, status_rows)

    raw_stats = validate_raw_dataset(config.raw_root)
    missing_query_images = [
        question.query_id
        for question in questions
        if question.task == "VKIS" and not _resolve_query_image(question, query_root).is_file()
    ]
    corpus_videos = 0
    if Path(config.corpus_path).is_file():
        with Path(config.corpus_path).open(newline="", encoding="utf-8-sig") as handle:
            corpus_videos = sum(1 for _ in csv.DictReader(handle))
    mapping = _mapping_validation(index)
    preflight = {
        "format": "hcmaic-skillpixel-kis-preflight-v1",
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "code_sha": _code_sha(),
        "raw_video_source": True,
        "btc_artifacts_used": False,
        "training_status": "not_run",
        "quality_status": (QUALITY_VALIDATED if config.qrels else "UNVALIDATED_ON_HCMAIC"),
        "raw_root": str(Path(config.raw_root).resolve()),
        "dataset_hash": raw_manifest.get("dataset_hash"),
        "sampling_policy": raw_manifest.get("sampling_policy"),
        "n_videos": raw_stats.n_videos,
        "n_sampled_frames": raw_stats.n_frames,
        "questions_path": str(Path(config.questions_path).resolve()),
        "query_file_hash": _sha256_file(config.questions_path),
        "n_queries": len(questions),
        "n_tkis": sum(item.task == "TKIS" for item in questions),
        "n_vkis": sum(item.task == "VKIS" for item in questions),
        "query_order": [item.query_id for item in questions],
        "query_order_preserved": list(results) == [item.query_id for item in questions],
        "missing_query_images": missing_query_images,
        "corpus_path": str(Path(config.corpus_path).resolve()),
        "corpus_file_hash": (
            _sha256_file(config.corpus_path) if Path(config.corpus_path).is_file() else None
        ),
        "n_corpus_videos": corpus_videos,
        "index_dir": str(Path(config.index_dir).resolve()),
        "index_type": "IndexFlatIP",
        "n_vectors": index.size,
        "embedding_dimension": index.dimension,
        "mapping_validation": mapping,
        "model": model,
    }
    preflight_path = candidate_dir / "preflight_report.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    model_registry_path = candidate_dir / "model_registry.json"
    model_registry_path.write_text(
        json.dumps(
            {
                "format": "hcmaic-skillpixel-kis-model-registry-v1",
                "training_status": "not_run",
                "quality_status": preflight["quality_status"],
                "candidates": [model],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checksums_path = _write_checksums(candidate_dir)
    return {
        "evidence_top100": evidence_top100_path,
        "evidence_top20": evidence_top20_path,
        "evidence_top100_csv": evidence_top100_csv,
        "evidence_top20_csv": evidence_top20_csv,
        "query_status": query_status_path,
        "preflight": preflight_path,
        "model_registry": model_registry_path,
        "checksums": checksums_path,
    }


def _metrics(
    results: Mapping[str, list[SkillPixelHit]], qrels: dict[str, Any] | None
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        **{f"recall@{cutoff}": None for cutoff in (1, 5, 20, 50, 100)},
        "mrr": None,
    }
    if qrels is None:
        return metrics
    expected: dict[str, set[str]] = {}
    for query_id, values in qrels.items():
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple, set, frozenset)):
            raise SkillPixelBenchmarkError(f"qrels for {query_id!r} must be a sequence")
        expected[str(query_id)] = {str(value).strip() for value in values if str(value).strip()}
    if set(expected) != set(results):
        raise SkillPixelBenchmarkError("qrels/query IDs do not match")
    reciprocal: list[float] = []
    counts = {cutoff: 0 for cutoff in (1, 5, 20, 50, 100)}
    for query_id, hits in results.items():
        retrieved = [_answer_cell(hit) for hit in hits]
        first = next(
            (rank for rank, value in enumerate(retrieved, 1) if value in expected[query_id]),
            None,
        )
        reciprocal.append(1.0 / first if first else 0.0)
        for cutoff in counts:
            if first is not None and first <= cutoff:
                counts[cutoff] += 1
    n_queries = len(results)
    for cutoff, count in counts.items():
        metrics[f"recall@{cutoff}"] = count / n_queries
    metrics["mrr"] = mean(reciprocal)
    return metrics


def _base_row(
    config: SkillPixelBenchmarkConfig,
    *,
    variant: str,
    kind: str,
    dataset_hash: str,
    query_file_hash: str,
    sampling_policy: str,
) -> dict[str, Any]:
    return {
        "run_id": Path(config.output_dir).name,
        "variant": variant,
        "kind": kind,
        "dataset_hash": dataset_hash,
        "query_file_hash": query_file_hash,
        "sampling_policy": sampling_policy,
        "channels": "",
        "provider": "",
        "model": "",
        "revision": "",
        "provider_execution": "",
        "fallback": None,
        "index_type": "IndexFlatIP",
        "embedding_dimension": None,
        "mapping_errors": None,
        "empty_error_queries": None,
        "build_time_s": None,
        "query_batch_ms": None,
        "query_p50_ms": None,
        "query_p95_ms": None,
        "ram_mb": _rss_mb(),
        "vram_mb": None,
        "disk_mb": None,
        "official_skillpixel_score": None,
        "metrics": {},
        "submission_validation": "not-run",
        "status": "unavailable",
        "error": None,
    }


def unavailable_candidate_row(
    *,
    variant: str,
    requested_provider: str,
    error: str,
    dataset_hash: str,
    query_file_hash: str,
) -> dict[str, Any]:
    """Create a matrix row that cannot be mistaken for a scored benchmark."""
    return {
        "run_id": "unavailable",
        "variant": variant,
        "kind": "visual",
        "dataset_hash": dataset_hash,
        "query_file_hash": query_file_hash,
        "sampling_policy": "",
        "channels": "visual",
        "provider": requested_provider,
        "model": "",
        "revision": "",
        "provider_execution": "unavailable",
        "fallback": None,
        "index_type": "IndexFlatIP",
        "embedding_dimension": None,
        "mapping_errors": None,
        "empty_error_queries": None,
        "build_time_s": None,
        "query_batch_ms": None,
        "query_p50_ms": None,
        "query_p95_ms": None,
        "ram_mb": _rss_mb(),
        "vram_mb": None,
        "disk_mb": None,
        "official_skillpixel_score": None,
        "metrics": {},
        "submission_validation": "not-run",
        "status": "unavailable",
        "model_registry": {
            "requested_provider": requested_provider,
            "selected_provider": None,
            "model_id": None,
            "revision": None,
            "weights_path": None,
            "weights_available": False,
            "provider_execution": "unavailable",
            "fallback": None,
            "training_status": "not_run",
        },
        "error": error,
    }


def benchmark_visual_candidate(
    config: SkillPixelBenchmarkConfig,
    provider: Any,
    *,
    requested_provider: str,
    selection: Mapping[str, Any],
    variant: str = "V0",
    build_time_s: float | None = None,
) -> dict[str, Any]:
    """Run one real provider/index pair, export a validated submission, and measure it."""
    validate_raw_dataset(config.raw_root)
    raw_manifest = json.loads(
        (Path(config.raw_root) / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    questions = load_skillpixel_questions(config.questions_path)
    index = load_skillpixel_index(config.index_dir)
    dataset_hash = str(raw_manifest.get("dataset_hash", ""))
    if index.index_manifest.get("dataset_manifest_hash") != dataset_hash:
        raise SkillPixelBenchmarkError("raw/index dataset hash mismatch")
    if provider.name != requested_provider:
        raise SkillPixelBenchmarkError(
            f"provider fallback detected: requested={requested_provider!r}, "
            f"selected={provider.name!r}"
        )
    if provider.name != str(index.index_manifest.get("provider_id", provider.name)):
        raise SkillPixelBenchmarkError("provider/index identity mismatch")

    query_root = Path(config.questions_path).parent
    tkis, vkis = _question_batches(questions, query_root)
    retriever = SkillPixelRetriever(index, provider)
    started = time.perf_counter()
    results: dict[str, list[SkillPixelHit]] = {}
    if tkis:
        results.update(retriever.search_text_queries(tkis, top_k=config.top_k))
    if vkis:
        results.update(retriever.search_image_queries(vkis, top_k=config.top_k))
    batch_ms = (time.perf_counter() - started) * 1000.0
    if list(results) != [item.query_id for item in questions]:
        raise SkillPixelBenchmarkError("batch retrieval changed query order or IDs")

    warm_latencies: list[float] = []
    for question in questions:
        started = time.perf_counter()
        if question.task == "TKIS":
            retriever.search_text(question.query_id, question.text, top_k=config.top_k)
        else:
            retriever.search_image(
                question.query_id,
                _resolve_query_image(question, query_root),
                top_k=config.top_k,
            )
        warm_latencies.append((time.perf_counter() - started) * 1000.0)

    candidate_dir = Path(config.output_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    results_path = candidate_dir / "results.jsonl"
    write_results_jsonl(results, results_path)
    submission_path = candidate_dir / "submission.csv"
    export_skillpixel_submission(
        config.questions_path, results_path, config.corpus_path, submission_path
    )
    validation = validate_submission_csv(submission_path, config.questions_path, config.corpus_path)
    (candidate_dir / "validation.json").write_text(
        json.dumps(
            {
                "valid": validation.ok,
                "n_queries": validation.n_queries,
                "errors": list(validation.errors),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if not validation.ok:
        raise SkillPixelBenchmarkError(f"submission validation failed: {list(validation.errors)}")

    info = provider.info()
    contract_paths = _write_visual_run_contracts(
        config=config,
        candidate_dir=candidate_dir,
        questions=questions,
        results=results,
        provider=provider,
        requested_provider=requested_provider,
        selection=selection,
        index=index,
        raw_manifest=raw_manifest,
    )
    row = _base_row(
        config,
        variant=variant,
        kind="visual",
        dataset_hash=dataset_hash,
        query_file_hash=_sha256_file(config.questions_path),
        sampling_policy=str(raw_manifest.get("sampling_policy", "")),
    )
    row.update(
        {
            "channels": "visual",
            "provider": provider.name,
            "model": info.get("model_name", provider.name),
            "revision": info.get("model_revision", info.get("revision", provider.version)),
            "provider_execution": "validated-local",
            "fallback": selection.get("fallback"),
            "embedding_dimension": provider.dimension,
            "mapping_errors": 0,
            "empty_error_queries": 0,
            "build_time_s": build_time_s,
            "query_batch_ms": round(batch_ms, 3),
            "query_p50_ms": _percentile(warm_latencies, 0.50),
            "query_p95_ms": _percentile(warm_latencies, 0.95),
            "disk_mb": _dir_size_mb(candidate_dir),
            "metrics": _metrics(results, config.qrels),
            "submission_validation": "pass",
            "status": "validated-local",
            "error": None,
            "artifact_dir": str(candidate_dir),
            "index_dir": str(config.index_dir),
            "n_queries": len(questions),
            "n_tkis": len(tkis),
            "n_vkis": len(vkis),
            "n_vectors": index.size,
            "model_registry": _model_registry_entry(
                provider=provider,
                requested_provider=requested_provider,
                selection=selection,
            ),
            "evidence_top100": str(contract_paths["evidence_top100"]),
            "evidence_top20": str(contract_paths["evidence_top20"]),
            "query_status": str(contract_paths["query_status"]),
            "preflight_report": str(contract_paths["preflight"]),
            "checksums": str(contract_paths["checksums"]),
        }
    )
    return row


def _channel_row(
    base: dict[str, Any], *, variant: str, channels: str, status: str, error: str | None = None
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "variant": variant,
            "kind": "channel",
            "channels": channels,
            "status": status,
            "provider_execution": (
                "validated-local" if status == "validated-local" else "unavailable"
            ),
            "submission_validation": "not-run",
            "error": error,
        }
    )
    return row


def benchmark_channel_ablation(
    config: SkillPixelBenchmarkConfig, base_row: dict[str, Any], provider: Any
) -> list[dict[str, Any]]:
    """Benchmark optional raw-derived channels without inventing unavailable scores."""
    rows = [_channel_row(base_row, variant="C0", channels="visual", status="validated-local")]
    from hcmaic.evaluation.kis import evaluate_kis_runtime
    from hcmaic.runtime.kis import KISRuntime

    def run_hybrid(
        variant: str, channel_names: str, optional_channels: dict[str, Any]
    ) -> dict[str, Any]:
        runtime = KISRuntime.from_components(
            load_skillpixel_index(config.index_dir),
            provider,
            optional_channels=optional_channels,
            max_per_video=5,
        )
        questions = load_skillpixel_questions(config.questions_path)
        report, _ = evaluate_kis_runtime(
            runtime,
            questions,
            None,
            top_k=config.top_k,
            query_root=Path(config.questions_path).parent,
        )
        row = _channel_row(
            base_row,
            variant=variant,
            channels=channel_names,
            status="validated-local",
        )
        row.update(
            {
                "query_p50_ms": report["latency_ms"].get("p50"),
                "query_p95_ms": report["latency_ms"].get("p95"),
                "empty_error_queries": report["n_empty_results"] + report["n_invalid_results"],
                "metrics": {
                    f"recall@{key}": report["recall_at"].get(str(key))
                    for key in (1, 5, 20, 50, 100)
                }
                | {"mrr": report.get("mrr")},
                "error": None,
            }
        )
        return row

    def load_channels(names: set[str]) -> dict[str, Any]:
        channels: dict[str, Any] = {}
        dataset_hash = str(base_row["dataset_hash"])
        if "ocr" in names:
            from hcmaic.retrieval.ocr_bm25 import BM25OCRChannel, load_ocr_artifact

            if config.ocr_artifact is None:
                raise FileNotFoundError("OCR artifact not configured")
            channels["ocr"] = BM25OCRChannel(
                load_ocr_artifact(config.ocr_artifact, dataset_manifest_hash=dataset_hash)
            )
        if "object" in names:
            from hcmaic.retrieval.object_retrieval import (
                ObjectRetrievalChannel,
                load_object_artifact,
            )

            if config.object_artifact is None:
                raise FileNotFoundError("object artifact not configured")
            channels["object"] = ObjectRetrievalChannel(
                load_object_artifact(config.object_artifact, dataset_manifest_hash=dataset_hash)
            )
        return channels

    for variant, channel_names, required in (
        ("C1", "visual+ocr", {"ocr"}),
        ("C2", "visual+object", {"object"}),
        ("C3", "visual+ocr+object", {"ocr", "object"}),
    ):
        try:
            rows.append(run_hybrid(variant, channel_names, load_channels(required)))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            rows.append(
                _channel_row(
                    base_row,
                    variant=variant,
                    channels=channel_names,
                    status="unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    try:
        rows.append(
            run_hybrid(
                "C4",
                "visual+bounded-rerank",
                {},
            )
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        rows.append(
            _channel_row(
                base_row,
                variant="C4",
                channels="visual+bounded-rerank",
                status="unavailable",
                error=f"bounded rerank failed: {type(exc).__name__}: {exc}",
            )
        )
    rows.append(
        _channel_row(
            base_row,
            variant="C5",
            channels="visual+text-reranker",
            status="unavailable",
            error="BGE text reranker artifact/provider not configured",
        )
    )
    return rows


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_benchmark_outputs(
    config: SkillPixelBenchmarkConfig,
    rows: list[dict[str, Any]],
    *,
    promotion_decision: str,
) -> dict[str, Path]:
    """Write the handoff-required CSV/Markdown/manifest/environment files."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    csv_path = output_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})

    raw_manifest = {}
    raw_manifest_path = Path(config.raw_root) / "dataset_manifest.json"
    if raw_manifest_path.is_file():
        raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    quality_status = QUALITY_VALIDATED if config.qrels else QUALITY_UNVALIDATED
    manifest = {
        "format": "hcmaic-skillpixel-kis-benchmark-v1",
        "run_id": output_dir.name,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "code_sha": _code_sha(),
        "dataset_hash": raw_manifest.get("dataset_hash"),
        "query_file_hash": _sha256_file(config.questions_path)
        if Path(config.questions_path).is_file()
        else None,
        "corpus_file_hash": _sha256_file(config.corpus_path)
        if Path(config.corpus_path).is_file()
        else None,
        "sampling_policy": raw_manifest.get("sampling_policy"),
        "quality_status": quality_status,
        "qrels_source": config.qrels_source,
        "promotion_decision": promotion_decision,
        "rows": rows,
        "raw_video_source": True,
        "btc_artifacts_used": False,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_registry_path = output_dir / "model_registry.json"
    model_registry_path.write_text(
        json.dumps(
            {
                "format": "hcmaic-skillpixel-kis-model-registry-v1",
                "training_status": "not_run",
                "quality_status": (QUALITY_VALIDATED if config.qrels else "UNVALIDATED_ON_HCMAIC"),
                "candidates": [row.get("model_registry") for row in rows],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preflight_path = output_dir / "preflight_report.json"
    if not preflight_path.is_file():
        preflight_path.write_text(
            json.dumps(
                {
                    "format": "hcmaic-skillpixel-kis-preflight-v1",
                    "code_sha": manifest["code_sha"],
                    "raw_video_source": True,
                    "btc_artifacts_used": False,
                    "training_status": "not_run",
                    "quality_status": (
                        QUALITY_VALIDATED if config.qrels else "UNVALIDATED_ON_HCMAIC"
                    ),
                    "dataset_hash": manifest["dataset_hash"],
                    "query_file_hash": manifest["query_file_hash"],
                    "mapping_validation": {"ok": False, "n_errors": None},
                    "note": (
                        "candidate-specific preflight is copied when a visual candidate "
                        "is validated"
                    ),
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    report_path = output_dir / "benchmark_report.md"
    lines = [
        "# SkillPixel KIS benchmark report",
        "",
        f"- quality_status: `{quality_status}`",
        f"- promotion_decision: {promotion_decision}",
        f"- code_sha: `{manifest['code_sha']}`",
        f"- dataset_hash: `{manifest['dataset_hash']}`",
        "",
        "| variant | kind | provider | execution | status | Recall@100 | MRR | p95 ms | error |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = row.get("metrics", {})
        lines.append(
            "| {variant} | {kind} | {provider} | {execution} | {status} | {recall} | "
            "{mrr} | {p95} | {error} |".format(
                variant=row.get("variant", ""),
                kind=row.get("kind", ""),
                provider=row.get("provider", ""),
                execution=row.get("provider_execution", ""),
                status=row.get("status", ""),
                recall=metrics.get("recall@100"),
                mrr=metrics.get("mrr"),
                p95=row.get("query_p95_ms"),
                error=str(row.get("error") or "").replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "Scores are emitted only for validated local providers or explicit qrels.",
            "Unavailable/fallback candidates have null quality metrics and are not promoted.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    environment_path = output_dir / "environment.txt"
    environment_path.write_text(
        "\n".join(
            [
                f"python={sys.version}",
                f"platform={platform.platform()}",
                f"processor={platform.processor() or 'unknown'}",
                f"code_sha={manifest['code_sha']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    checksums_path = _write_checksums(output_dir)
    return {
        "csv": csv_path,
        "report": report_path,
        "manifest": manifest_path,
        "environment": environment_path,
        "model_registry": model_registry_path,
        "preflight": preflight_path,
        "checksums": checksums_path,
    }


def run_skillpixel_benchmark(
    config: SkillPixelBenchmarkConfig,
    *,
    provider_ids: tuple[str, ...] = ("clip", "siglip2", "jina-clip-v2"),
    device: str = "cpu",
    batch_size: int = 32,
    allow_network: bool = False,
    build_missing: bool = True,
    model_paths: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    """Run V0/V1/V2, optional channel ablation, and package one matrix."""
    raw_manifest = json.loads(
        (Path(config.raw_root) / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    dataset_hash = str(raw_manifest.get("dataset_hash", ""))
    query_hash = _sha256_file(config.questions_path)
    rows: list[dict[str, Any]] = []
    actual: list[tuple[dict[str, Any], Any, Path]] = []
    index_for_clip = Path(config.index_dir)
    variant_for_provider = {"clip": "V0", "siglip2": "V1", "jina-clip-v2": "V2"}

    from hcmaic.embedding.factory import get_real_visual_provider
    from hcmaic.skillpixel.index import build_skillpixel_index

    for provider_id in provider_ids:
        variant = variant_for_provider.get(provider_id, provider_id)
        try:
            model_kwargs: dict[str, str] = {}
            model_path = (model_paths or {}).get(provider_id)
            if model_path:
                model_key = {
                    "siglip2": "siglip2_model",
                    "clip": "clip_model",
                    "jina-clip-v2": "jina_model",
                }[provider_id]
                model_kwargs[model_key] = model_path
            provider, selection = get_real_visual_provider(
                prefer=provider_id,
                device=device,
                local_files_only=not allow_network,
                batch_size=batch_size,
                allow_fallback=False,
                **model_kwargs,
            )
            if provider.name != provider_id:
                raise SkillPixelBenchmarkError(
                    f"provider fallback detected: {provider_id} -> {provider.name}"
                )
            index_dir = index_for_clip if provider_id == "clip" else None
            build_time_s: float | None = None
            if index_dir is None:
                index_dir = Path(config.output_dir) / "visual" / provider_id
                if index_dir.exists() and any(index_dir.iterdir()):
                    pass
                elif not build_missing:
                    raise SkillPixelBenchmarkError("candidate index is not configured")
                else:
                    started = time.perf_counter()
                    build_skillpixel_index(config.raw_root, index_dir, provider)
                    build_time_s = time.perf_counter() - started
            candidate_config = replace(
                config,
                index_dir=index_dir,
                output_dir=Path(config.output_dir) / "candidates" / provider_id,
            )
            row = benchmark_visual_candidate(
                candidate_config,
                provider,
                requested_provider=provider_id,
                selection=selection,
                variant=variant,
                build_time_s=build_time_s,
            )
            rows.append(row)
            actual.append((row, provider, index_dir))
        except (FileNotFoundError, RuntimeError, ValueError, SkillPixelBenchmarkError) as exc:
            rows.append(
                unavailable_candidate_row(
                    variant=variant,
                    requested_provider=provider_id,
                    error=f"{type(exc).__name__}: {exc}",
                    dataset_hash=dataset_hash,
                    query_file_hash=query_hash,
                )
            )

    if actual:
        best_row, best_provider, best_index_dir = actual[0]
        channel_config = replace(
            config,
            index_dir=best_index_dir,
        )
        rows.extend(benchmark_channel_ablation(channel_config, best_row, best_provider))
        promotion = (
            "promote benchmark candidate by qrels"
            if config.qrels
            else "retain V0; no SkillPixel qrels, so no model/channel promotion claim"
        )
        source_submission = Path(str(best_row["artifact_dir"])) / "submission.csv"
        if source_submission.is_file():
            target_submission = (
                Path(config.output_dir) / f"submission_{Path(config.output_dir).name}.csv"
            )
            shutil.copyfile(source_submission, target_submission)
        source_contract_dir = Path(str(best_row["artifact_dir"]))
        for contract_name in (
            "retrieval_evidence_top100.jsonl",
            "retrieval_evidence_top20.jsonl",
            "retrieval_evidence_top100.csv",
            "retrieval_evidence_top20.csv",
            "query_status.jsonl",
            "preflight_report.json",
        ):
            source_contract = source_contract_dir / contract_name
            if source_contract.is_file():
                shutil.copyfile(source_contract, Path(config.output_dir) / contract_name)
    else:
        promotion = "blocked: no validated visual provider/index pair"
    paths = write_benchmark_outputs(config, rows, promotion_decision=promotion)
    return rows, paths
