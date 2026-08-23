"""Evaluation harness: Recall@K, MRR, latency percentiles."""

from hcmaic.evaluation.evaluator import evaluate, load_qrels, load_queries
from hcmaic.evaluation.offline import evaluate_offline, load_qrels_jsonl, write_offline_report

__all__ = [
    "evaluate",
    "evaluate_offline",
    "load_qrels",
    "load_qrels_jsonl",
    "load_queries",
    "write_offline_report",
]
