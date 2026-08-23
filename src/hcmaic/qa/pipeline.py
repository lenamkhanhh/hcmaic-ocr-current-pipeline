"""QA retrieval -> exact evidence -> answer provider -> post-hoc scorer.

The retrieval and answer-provider contracts intentionally contain no qrels,
answers, or aliases. Official semantic qrels are loaded only by the scorer
after inference has completed.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

FORBIDDEN_QA_KEYS = frozenset(
    {
        "accepted_aliases",
        "answer_aliases",
        "canonical_answer",
        "expected_answer",
        "human_approved_aliases",
        "reference_answers",
        "relevant_frame_ids",
        "relevant_video_ids",
        "reviewed_answer",
        "qrel",
        "qrels",
    }
)


class ForbiddenQADataError(ValueError):
    """Inference input or provider payload contains post-hoc label data."""


class MissingExactEvidenceError(ValueError):
    """An exact frame evidence row or its verified image is missing."""


@dataclass(frozen=True)
class QAEvidenceFrame:
    query_id: str
    rank: int
    frame_uid: str
    video_id: str
    source_frame_idx: int
    timestamp_ms: int
    score: float
    image_path: Path | None = None
    image_sha256: str | None = None


@dataclass(frozen=True)
class QAQuery:
    query_id: str
    question: str
    evidence: tuple[QAEvidenceFrame, ...]


@dataclass(frozen=True)
class QARetrievalInput:
    queries: tuple[QAQuery, ...]
    catalog_identity_hash: str
    catalog_rows: int
    video_count: int
    dataset_slug: str
    dataset_version: int
    manifest_sha256: str
    ranked_results_sha256: str


@dataclass(frozen=True)
class ProviderRequest:
    query_id: str
    question: str
    evidence: tuple[QAEvidenceFrame, ...]


class AnswerProvider(Protocol):
    model_id: str
    revision: str

    def answer(self, request: ProviderRequest) -> str:
        """Return an answer generated from request question and evidence only."""


class Qwen3VLAnswerProvider:
    """Thin Qwen3-VL adapter with no qrels-aware runtime path.

    The model and processor are injected so local tests remain offline. The
    optional ``from_pretrained`` constructor is the only place that imports
    Transformers and may download public model weights in an explicitly
    provisioned runtime such as Kaggle.
    """

    def __init__(
        self,
        processor: Any,
        model: Any,
        *,
        model_id: str,
        revision: str,
        max_new_tokens: int = 128,
    ) -> None:
        self.processor = processor
        self.model = model
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        revision: str = "main",
        *,
        device_map: str = "auto",
        max_new_tokens: int = 128,
    ) -> Qwen3VLAnswerProvider:
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - depends on optional Kaggle runtime
            raise RuntimeError("Qwen3-VL requires the pinned Transformers runtime") from exc
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, device_map=device_map, torch_dtype="auto"
        )
        return cls(
            processor,
            model,
            model_id=model_id,
            revision=revision,
            max_new_tokens=max_new_tokens,
        )

    def answer(self, request: ProviderRequest) -> str:
        from PIL import Image

        if any(frame.image_path is None for frame in request.evidence):
            raise MissingExactEvidenceError(
                f"{request.query_id}: Qwen requires exact image evidence"
            )
        opened_images = [
            Image.open(frame.image_path) for frame in request.evidence if frame.image_path
        ]
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": image} for image in opened_images),
                        {
                            "type": "text",
                            "text": (
                                "Answer the question using only the supplied evidence images. "
                                f"Question: {request.question}"
                            ),
                        },
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model_device())
            elif isinstance(inputs, dict):
                inputs = {
                    key: value.to(self._model_device()) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            input_ids = inputs["input_ids"]
            trimmed = [
                output[len(input_row) :]
                for input_row, output in zip(input_ids, generated, strict=True)
            ]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)
            return str(decoded[0]).strip() if decoded else ""
        finally:
            for image in opened_images:
                image.close()
            try:
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def _model_device(self) -> str:
        device = getattr(self.model, "device", None)
        return str(device) if device is not None else "cpu"


@dataclass(frozen=True)
class QAInferenceResult:
    query_id: str
    answer: str
    status: str
    model_id: str
    revision: str
    prompt_hash: str
    evidence_hash: str
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True)
class OfficialQrel:
    query_id: str
    canonical_answer: str
    accepted_aliases: tuple[str, ...]


@dataclass(frozen=True)
class ScoreReport:
    n_scored: int
    n_unscorable: int
    per_query: tuple[dict[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest().upper()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno}: expected object")
            rows.append(value)
    return rows


def _reject_forbidden(value: object, path: str = "input") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_QA_KEYS:
                raise ForbiddenQADataError(f"forbidden QA label field at {path}.{key}")
            _reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}[{index}]")


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def load_retrieval_input(
    manifest_path: Path,
    queries_path: Path,
    ranked_results_path: Path,
    *,
    expected_video_count: int = 873,
    expected_catalog_rows: int = 233062,
    expected_dataset_slug: str = "qadeptrai123/aicthegay",
    expected_dataset_version: int = 5,
    expected_top_k: int = 12,
) -> QARetrievalInput:
    """Load and validate the authoritative QA retrieval-only bundle."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "hcmaic_ranked_eval":
        raise ValueError("unexpected ranked-eval artifact_type")
    bundle = manifest.get("bundle")
    if not isinstance(bundle, dict):
        raise ValueError("manifest.bundle is required")
    video_count = _require_int(bundle.get("video_count"), "bundle.video_count")
    catalog_rows = _require_int(bundle.get("catalog_rows"), "bundle.catalog_rows")
    if video_count != expected_video_count:
        raise ValueError(f"bundle.video_count={video_count}, expected {expected_video_count}")
    if catalog_rows != expected_catalog_rows:
        raise ValueError(f"bundle.catalog_rows={catalog_rows}, expected {expected_catalog_rows}")
    dataset = bundle.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("slug") != expected_dataset_slug:
        raise ValueError("bundle.dataset.slug is not the authoritative dataset")
    if dataset.get("version") != expected_dataset_version:
        raise ValueError("bundle.dataset.version is not the authoritative version")
    catalog_hash = bundle.get("catalog_identity_hash")
    if not isinstance(catalog_hash, str) or not catalog_hash:
        raise ValueError("bundle.catalog_identity_hash is required")

    query_rows = [row for row in _read_jsonl(queries_path) if row.get("task") == "qa"]
    query_map: dict[str, str] = {}
    for row in query_rows:
        query_id = row.get("query_id")
        question = row.get("text")
        if (
            not isinstance(query_id, str)
            or not query_id
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise ValueError("QA query requires non-empty query_id and text")
        if query_id in query_map:
            raise ValueError(f"duplicate QA query_id {query_id}")
        query_map[query_id] = question

    ranked_rows = _read_jsonl(ranked_results_path)
    _reject_forbidden(ranked_rows, "ranked_results")
    if len(query_map) != 11 or len(ranked_rows) != 11:
        raise ValueError(
            f"expected 11 QA queries, got query_file={len(query_map)} ranked={len(ranked_rows)}"
        )
    seen_query_ids: set[str] = set()
    queries: list[QAQuery] = []
    for row in ranked_rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id not in query_map:
            raise ValueError(f"ranked result has unknown query_id {query_id!r}")
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate ranked query_id {query_id}")
        seen_query_ids.add(query_id)
        raw_evidence = row.get("evidence")
        if not isinstance(raw_evidence, list) or len(raw_evidence) != expected_top_k:
            raise ValueError(f"{query_id}: expected exactly {expected_top_k} evidence rows")
        evidence: list[QAEvidenceFrame] = []
        seen_ranks: set[int] = set()
        for raw in raw_evidence:
            if not isinstance(raw, dict):
                raise ValueError(f"{query_id}: evidence row must be an object")
            rank = _require_int(raw.get("rank"), f"{query_id}.rank", minimum=1)
            if rank in seen_ranks:
                raise ValueError(f"{query_id}: duplicate rank {rank}")
            seen_ranks.add(rank)
            video_id = raw.get("video_id")
            frame_uid = raw.get("frame_uid")
            source_frame_idx = _require_int(
                raw.get("source_frame_idx"), f"{query_id}.source_frame_idx"
            )
            timestamp_ms = _require_int(raw.get("timestamp_ms"), f"{query_id}.timestamp_ms")
            if not isinstance(video_id, str) or not video_id:
                raise ValueError(f"{query_id}: video_id is required")
            expected_frame_uid = f"{video_id}:{source_frame_idx}"
            if frame_uid != expected_frame_uid:
                raise ValueError(f"{query_id}: frame_uid identity mismatch")
            score = raw.get("score", 0.0)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{query_id}: score must be numeric")
            evidence.append(
                QAEvidenceFrame(
                    query_id=query_id,
                    rank=rank,
                    frame_uid=frame_uid,
                    video_id=video_id,
                    source_frame_idx=source_frame_idx,
                    timestamp_ms=timestamp_ms,
                    score=float(score),
                )
            )
        if sorted(seen_ranks) != list(range(1, expected_top_k + 1)):
            raise ValueError(f"{query_id}: ranks must be contiguous from 1")
        queries.append(QAQuery(query_id, query_map[query_id], tuple(evidence)))
    if seen_query_ids != set(query_map):
        raise ValueError("ranked results and QA query file have different coverage")
    return QARetrievalInput(
        queries=tuple(queries),
        catalog_identity_hash=catalog_hash,
        catalog_rows=catalog_rows,
        video_count=video_count,
        dataset_slug=expected_dataset_slug,
        dataset_version=expected_dataset_version,
        manifest_sha256=_sha256(manifest_path),
        ranked_results_sha256=_sha256(ranked_results_path),
    )


def resolve_exact_evidence(
    retrieval_input: QARetrievalInput,
    exact_evidence_path: Path,
    image_root: Path,
) -> list[QAEvidenceFrame]:
    """Attach only hash-verified exact images to the retrieval identities."""
    rows = _read_jsonl(exact_evidence_path)
    _reject_forbidden(rows, "exact_evidence")
    expected = {
        (query.query_id, evidence.rank): evidence
        for query in retrieval_input.queries
        for evidence in query.evidence
    }
    if len(rows) != len(expected):
        raise MissingExactEvidenceError(
            f"expected {len(expected)} exact evidence rows, got {len(rows)}"
        )
    root = image_root.resolve()
    resolved: list[QAEvidenceFrame] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        row_query_id = row.get("query_id")
        row_rank = row.get("rank")
        if not isinstance(row_query_id, str) or not isinstance(row_rank, int):
            raise MissingExactEvidenceError("exact evidence key requires query_id and integer rank")
        key: tuple[str, int] = (row_query_id, row_rank)
        source = expected.get(key)
        if source is None or key in seen:
            raise MissingExactEvidenceError(f"unexpected or duplicate exact evidence key {key}")
        seen.add(key)
        identity_fields = ("frame_uid", "video_id", "source_frame_idx", "timestamp_ms")
        if any(row.get(field) != getattr(source, field) for field in identity_fields):
            raise MissingExactEvidenceError(f"exact evidence identity mismatch for {key}")
        relative_path = row.get("image_path")
        declared_hash = row.get("image_sha256")
        if not isinstance(relative_path, str) or not isinstance(declared_hash, str):
            raise MissingExactEvidenceError(f"missing image path/hash for {key}")
        image_path = (root / relative_path).resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise MissingExactEvidenceError(f"missing or escaping image for {key}")
        actual_hash = _sha256(image_path)
        if actual_hash != declared_hash.upper():
            raise MissingExactEvidenceError(f"image hash mismatch for {key}")
        resolved.append(
            QAEvidenceFrame(
                query_id=source.query_id,
                rank=source.rank,
                frame_uid=source.frame_uid,
                video_id=source.video_id,
                source_frame_idx=source.source_frame_idx,
                timestamp_ms=source.timestamp_ms,
                score=source.score,
                image_path=image_path,
                image_sha256=actual_hash,
            )
        )
    if seen != set(expected):
        raise MissingExactEvidenceError("exact evidence coverage differs from retrieval input")
    return sorted(resolved, key=lambda item: (item.query_id, item.rank))


def _prompt_payload(request: ProviderRequest) -> dict[str, Any]:
    return {
        "instruction": "Answer the question using only the supplied retrieved evidence images.",
        "question": request.question,
        "evidence": [
            {
                "rank": frame.rank,
                "frame_uid": frame.frame_uid,
                "video_id": frame.video_id,
                "source_frame_idx": frame.source_frame_idx,
                "timestamp_ms": frame.timestamp_ms,
                "image_path": str(frame.image_path) if frame.image_path else None,
            }
            for frame in request.evidence
        ],
    }


def _prompt_hash(request: ProviderRequest) -> str:
    return _stable_hash(_prompt_payload(request))


def _evidence_hash(evidence: Sequence[QAEvidenceFrame]) -> str:
    return _stable_hash(
        [
            {
                "rank": frame.rank,
                "frame_uid": frame.frame_uid,
                "timestamp_ms": frame.timestamp_ms,
                "image_sha256": frame.image_sha256,
            }
            for frame in evidence
        ]
    )


def run_qa_inference(
    retrieval_input: QARetrievalInput,
    provider: AnswerProvider,
    exact_evidence: Iterable[QAEvidenceFrame] | None = None,
) -> list[QAInferenceResult]:
    """Run provider inference only with complete exact evidence coverage."""
    if exact_evidence is None:
        raise MissingExactEvidenceError("run_qa_inference requires exact evidence")
    exact_frames = list(exact_evidence)
    exact_by_key = {(frame.query_id, frame.rank): frame for frame in exact_frames}
    expected_keys = {
        (query.query_id, frame.rank)
        for query in retrieval_input.queries
        for frame in query.evidence
    }
    if len(exact_frames) != len(exact_by_key) or set(exact_by_key) != expected_keys:
        raise MissingExactEvidenceError("provider exact evidence coverage is incomplete")
    results: list[QAInferenceResult] = []
    for query in retrieval_input.queries:
        evidence = tuple(
            exact_by_key.get((query.query_id, frame.rank), frame) for frame in query.evidence
        )
        request = ProviderRequest(query.query_id, query.question, evidence)
        _reject_forbidden(_prompt_payload(request), f"provider_request[{query.query_id}]")
        prompt_hash = _prompt_hash(request)
        evidence_hash = _evidence_hash(evidence)
        started = time.perf_counter()
        try:
            answer = provider.answer(request)
            answer = answer.strip() if isinstance(answer, str) else ""
            status = "INFERRED" if answer else "ERROR"
            error = None if answer else "EMPTY_ANSWER"
        except Exception as exc:  # provider errors become explicit rows, never fake answers
            answer = ""
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            QAInferenceResult(
                query_id=query.query_id,
                answer=answer,
                status=status,
                model_id=str(provider.model_id),
                revision=str(provider.revision),
                prompt_hash=prompt_hash,
                evidence_hash=evidence_hash,
                latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
                error=error,
            )
        )
    if len({row.query_id for row in results}) != len(retrieval_input.queries):
        raise ValueError("inference result query coverage is not unique")
    return results


def load_official_qrels(path: Path) -> dict[str, OfficialQrel]:
    """Load approved qrels for post-hoc scoring only."""
    rows = _read_jsonl(path)
    qrels: dict[str, OfficialQrel] = {}
    for row in rows:
        query_id = row.get("query_id")
        aliases = row.get("accepted_aliases")
        if not isinstance(query_id, str) or not isinstance(row.get("canonical_answer"), str):
            raise ValueError("official qrel requires query_id and canonical_answer")
        if (
            row.get("promotion_status") != "OFFICIAL"
            or row.get("review_status") != "HUMAN_REVIEWED"
        ):
            raise ValueError(f"qrel {query_id} is not human-reviewed official")
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(alias, str) for alias in aliases)
        ):
            raise ValueError(f"qrel {query_id} has no approved aliases")
        if query_id in qrels:
            raise ValueError(f"duplicate official qrel {query_id}")
        qrels[query_id] = OfficialQrel(query_id, row["canonical_answer"], tuple(aliases))
    return qrels


def _normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def score_qa_answers(
    results: Sequence[QAInferenceResult], qrels: Mapping[str, OfficialQrel]
) -> ScoreReport:
    """Deterministically score predictions against approved aliases after inference."""
    per_query: list[dict[str, Any]] = []
    n_scored = 0
    for result in results:
        qrel = qrels.get(result.query_id)
        if qrel is None or result.status != "INFERRED" or not result.answer:
            per_query.append({"query_id": result.query_id, "status": "UNSCORABLE", "match": None})
            continue
        accepted = {
            _normalize_answer(qrel.canonical_answer),
            *(_normalize_answer(alias) for alias in qrel.accepted_aliases),
        }
        match = _normalize_answer(result.answer) in accepted
        n_scored += 1
        per_query.append({"query_id": result.query_id, "status": "SCORED", "match": match})
    return ScoreReport(
        n_scored=n_scored,
        n_unscorable=len(results) - n_scored,
        per_query=tuple(per_query),
    )
