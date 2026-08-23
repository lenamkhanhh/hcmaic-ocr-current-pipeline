"""Deterministic synthetic exact-versus-ANN engineering benchmark."""

from __future__ import annotations

import dataclasses
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hcmaic.embedding.base import l2_normalize
from hcmaic.indexing.faiss_ann import FaissHNSWIndex
from hcmaic.indexing.numpy_index import ExactNumpyIndex


@dataclass(frozen=True)
class ScaleBenchmarkConfig:
    vector_count: int = 10_000
    dimension: int = 512
    query_count: int = 100
    top_k: int = 100
    seed: int = 7
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 128

    def __post_init__(self) -> None:
        if (
            min(
                self.vector_count,
                self.dimension,
                self.query_count,
                self.top_k,
                self.hnsw_m,
                self.ef_construction,
                self.ef_search,
            )
            < 1
        ):
            raise ValueError("scale benchmark counts and HNSW parameters must be >= 1")


def run_scale_benchmark(config: ScaleBenchmarkConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.seed)
    vectors = l2_normalize(
        rng.normal(size=(config.vector_count, config.dimension)).astype(np.float32)
    )
    queries = l2_normalize(
        rng.normal(size=(config.query_count, config.dimension)).astype(np.float32)
    )
    frame_ids = [f"F{index:09d}" for index in range(config.vector_count)]

    exact = ExactNumpyIndex()
    exact_start = time.perf_counter()
    exact.build(vectors, frame_ids)
    exact_build_ms = (time.perf_counter() - exact_start) * 1000

    ann = FaissHNSWIndex(
        m=config.hnsw_m,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
    )
    ann_start = time.perf_counter()
    ann.build(vectors, frame_ids)
    ann_build_ms = (time.perf_counter() - ann_start) * 1000

    recalls: list[float] = []
    latencies: list[float] = []
    for query in queries:
        exact_ids = {item[0] for item in exact.search(query, config.top_k)}
        start = time.perf_counter()
        ann_ids = {item[0] for item in ann.search(query, config.top_k)}
        latencies.append((time.perf_counter() - start) * 1000)
        recalls.append(len(exact_ids & ann_ids) / max(1, len(exact_ids)))

    ordered_latency = sorted(latencies)
    total_seconds = sum(latencies) / 1000
    return {
        "config": dataclasses.asdict(config),
        "vector_bytes": int(vectors.nbytes),
        "index_size_bytes": ann.serialized_size_bytes(),
        "exact_build_ms": round(exact_build_ms, 3),
        "ann_build_ms": round(ann_build_ms, 3),
        "ann_recall_at_k": round(statistics.mean(recalls), 6),
        "p50_latency_ms": round(float(np.percentile(ordered_latency, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(ordered_latency, 95)), 3),
        "throughput_qps": round(config.query_count / max(total_seconds, 1e-9), 3),
        "ann_parameters": ann.parameters,
        "evidence_level": "SYNTHETIC_SCALE_VERIFIED",
        "disclaimer": "Synthetic engineering benchmark; not BTC competition evidence.",
    }
