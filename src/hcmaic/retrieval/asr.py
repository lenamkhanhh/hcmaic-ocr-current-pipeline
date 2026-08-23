"""Optional timestamped ASR artifact/channel with an explicit promotion gate."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.channel_contract import ChannelContract, build_channel_evidence
from hcmaic.retrieval.ocr_bm25 import normalize_ocr_text

ASR_RECORDS_NAME = "asr.jsonl"
ASR_MANIFEST_NAME = "asr_manifest.json"
ASR_FORMAT = "hcmaic-raw-asr-v1"


def _is_mock_provider(provider: str) -> bool:
    return "mock" in provider.casefold()


class ASRArtifactError(RuntimeError):
    """Raised when an ASR artifact is missing, malformed or unsafe."""


class ASRUnavailableError(ASRArtifactError):
    """Raised when the optional ASR channel cannot execute locally."""


class ASRProvider(Protocol):
    """Provider metadata contract for timestamped local ASR output."""

    name: str
    version: str
    execution: str


@dataclass(frozen=True)
class ASRRecord:
    """One transcript segment anchored to a raw source frame."""

    segment_id: str
    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int
    start_ms: int
    end_ms: int
    text: str
    provider: str
    revision: str
    confidence: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.segment_id.strip() or not self.frame_uid.strip():
            raise ValueError("ASR segment_id and frame_uid must not be blank")
        if not self.video_id.strip() or not self.video_filename.strip():
            raise ValueError("ASR video identity must not be blank")
        try:
            frame_video, frame_index = self.frame_uid.rsplit(":", 1)
            if (
                frame_video != self.video_id
                or not frame_index.isdigit()
                or int(frame_index) != self.source_frame_idx
            ):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("ASR frame_uid must match video_id:source_frame_idx") from exc
        if min(self.source_frame_idx, self.timestamp_ms, self.start_ms) < 0:
            raise ValueError("ASR frame/timestamp/start must be non-negative")
        if self.end_ms < self.start_ms:
            raise ValueError("ASR end_ms must be >= start_ms")
        if not self.text.strip():
            raise ValueError("ASR text must not be blank")
        if not self.provider.strip() or _is_mock_provider(self.provider):
            raise ValueError("ASR provider must be a real non-mock provider")
        if not self.revision.strip():
            raise ValueError("ASR revision must not be blank")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ASR confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "frame_uid": self.frame_uid,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "source_frame_idx": self.source_frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "provider": self.provider,
            "revision": self.revision,
            "confidence": self.confidence,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ASRArtifact:
    artifact_dir: Path
    records: tuple[ASRRecord, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ASRPromotionDecision:
    """Evidence gate for enabling ASR in a production fusion configuration."""

    enabled: bool
    reason: str
    baseline_score: float | None
    asr_score: float | None
    gain: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "baseline_score": self.baseline_score,
            "asr_score": self.asr_score,
            "gain": self.gain,
        }


def decide_asr_promotion(
    *,
    qrels_available: bool,
    baseline_score: float | None,
    asr_score: float | None,
    minimum_gain: float = 0.01,
) -> ASRPromotionDecision:
    """Enable ASR only after paired qrels evidence shows a meaningful gain."""
    if minimum_gain < 0:
        raise ValueError("minimum_gain must be >= 0")
    if not qrels_available:
        return ASRPromotionDecision(
            False,
            "disabled_without_hcmaic_qrels",
            baseline_score,
            asr_score,
            None,
        )
    if baseline_score is None or asr_score is None:
        return ASRPromotionDecision(
            False,
            "disabled_without_paired_benchmark_scores",
            baseline_score,
            asr_score,
            None,
        )
    if not math.isfinite(baseline_score) or not math.isfinite(asr_score):
        raise ValueError("ASR promotion scores must be finite")
    gain = asr_score - baseline_score
    if gain < minimum_gain:
        return ASRPromotionDecision(
            False,
            "disabled_without_minimum_benchmark_gain",
            baseline_score,
            asr_score,
            gain,
        )
    return ASRPromotionDecision(
        True, "enabled_after_paired_benchmark_gain", baseline_score, asr_score, gain
    )


def _canonical_bytes(records: list[ASRRecord]) -> bytes:
    lines = [json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) for record in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_asr_artifact(
    records: list[ASRRecord],
    artifact_dir: Path,
    *,
    dataset_manifest_hash: str,
    provider_execution: str = "validated-local",
) -> ASRArtifact:
    """Persist timestamped ASR output; runtime enablement remains policy-gated."""
    if not dataset_manifest_hash.strip():
        raise ASRArtifactError("dataset_manifest_hash must not be blank")
    if provider_execution != "validated-local":
        raise ASRArtifactError(
            "ASR artifact execution must be validated-local; refusing unverified runtime"
        )
    if not records:
        raise ASRArtifactError("ASR artifact cannot be empty")
    records = sorted(
        records,
        key=lambda record: (record.video_id, record.start_ms, record.segment_id),
    )
    providers = {record.provider for record in records}
    revisions = {record.revision for record in records}
    if len(providers) != 1 or len(revisions) != 1:
        raise ASRArtifactError("ASR artifact must contain one provider and revision")
    if any(_is_mock_provider(provider) for provider in providers):
        raise ASRArtifactError("mock ASR provider is not allowed")
    segment_ids = [record.segment_id for record in records]
    if len(segment_ids) != len(set(segment_ids)):
        raise ASRArtifactError("ASR artifact has duplicate segment_id values")
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ASRArtifactError(
            f"Artifact directory {artifact_dir} is not empty; use a new versioned path"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / ASR_RECORDS_NAME
    payload = _canonical_bytes(records)
    records_path.write_bytes(payload)
    manifest = {
        "format": ASR_FORMAT,
        "records": ASR_RECORDS_NAME,
        "records_sha256": _sha256(payload),
        "n_records": len(records),
        "provider": next(iter(providers)),
        "revision": next(iter(revisions)),
        "dataset_manifest_hash": dataset_manifest_hash,
        "raw_video_source": True,
        "btc_artifacts_used": False,
        "provider_execution": provider_execution,
        "runtime_policy": "disabled-unless-hcmaic-qrels-ablation-gain",
        "evidence_level": "REAL_PROVIDER_ARTIFACT",
        "quality_status": "UNVALIDATED_ON_HCMAIC",
    }
    _write_json(artifact_dir / ASR_MANIFEST_NAME, manifest)
    return load_asr_artifact(artifact_dir, dataset_manifest_hash=dataset_manifest_hash)


def _parse_records(payload: bytes) -> list[ASRRecord]:
    records: list[ASRRecord] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ASRArtifactError("ASR artifact is not valid UTF-8") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("record is not an object")
            records.append(ASRRecord(**data))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ASRArtifactError(f"invalid ASR record at line {line_number}") from exc
    return records


def load_asr_artifact(
    artifact_dir: Path, *, dataset_manifest_hash: str | None = None
) -> ASRArtifact:
    """Load ASR only when raw provenance and hashes validate."""
    artifact_dir = Path(artifact_dir).resolve()
    manifest_path = artifact_dir / ASR_MANIFEST_NAME
    records_path = artifact_dir / ASR_RECORDS_NAME
    if not manifest_path.is_file() or not records_path.is_file():
        raise ASRUnavailableError(f"ASR artifact is unavailable in {artifact_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = records_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ASRArtifactError(f"cannot read ASR artifact in {artifact_dir}") from exc
    if manifest.get("format") != ASR_FORMAT:
        raise ASRArtifactError("unsupported ASR artifact format")
    if manifest.get("raw_video_source") is not True:
        raise ASRArtifactError("ASR artifact is not marked raw_video_source")
    if manifest.get("btc_artifacts_used") is not False:
        raise ASRArtifactError("ASR artifact claims BTC artifacts")
    if manifest.get("provider_execution") != "validated-local":
        raise ASRArtifactError("ASR provider execution is not validated-local")
    if _is_mock_provider(str(manifest.get("provider", ""))):
        raise ASRArtifactError("mock ASR provider is not allowed")
    if (
        dataset_manifest_hash is not None
        and manifest.get("dataset_manifest_hash") != dataset_manifest_hash
    ):
        raise ASRArtifactError("ASR dataset manifest hash mismatch")
    if manifest.get("records_sha256") != _sha256(payload):
        raise ASRArtifactError("ASR records hash mismatch")
    records = _parse_records(payload)
    if len(records) != int(manifest.get("n_records", -1)):
        raise ASRArtifactError("ASR manifest record count mismatch")
    if any(_is_mock_provider(record.provider) for record in records):
        raise ASRArtifactError("mock ASR provider is not allowed")
    if {record.provider for record in records} != {manifest.get("provider")}:
        raise ASRArtifactError("ASR record provider mismatch")
    if {record.revision for record in records} != {manifest.get("revision")}:
        raise ASRArtifactError("ASR record revision mismatch")
    if _canonical_bytes(records) != payload:
        raise ASRArtifactError("ASR artifact is not canonical")
    if len({record.segment_id for record in records}) != len(records):
        raise ASRArtifactError("ASR artifact has duplicate segment_id values")
    return ASRArtifact(artifact_dir, tuple(records), manifest)


@dataclass(frozen=True)
class _ASRSegment:
    record: ASRRecord
    tokens: tuple[str, ...]


class ASRRetrievalChannel:
    """Timestamped transcript search; caller must enforce promotion enablement."""

    def __init__(self, artifact: ASRArtifact) -> None:
        self.artifact = artifact
        self._segments = tuple(
            _ASRSegment(record, tuple(normalize_ocr_text(record.text).split()))
            for record in artifact.records
        )
        self._postings: dict[str, list[int]] = defaultdict(list)
        for segment_idx, segment in enumerate(self._segments):
            for token in set(segment.tokens):
                self._postings[token].append(segment_idx)
        self._df = Counter(token for segment in self._segments for token in set(segment.tokens))

    @classmethod
    def from_artifact(
        cls, artifact_dir: Path, *, dataset_manifest_hash: str | None = None
    ) -> ASRRetrievalChannel:
        return cls(load_asr_artifact(artifact_dir, dataset_manifest_hash=dataset_manifest_hash))

    @property
    def provider(self) -> str:
        return str(self.artifact.manifest["provider"])

    @property
    def revision(self) -> str:
        return str(self.artifact.manifest["revision"])

    @property
    def execution_status(self) -> str:
        return "ENGINEERING_PROXY"

    @property
    def quality_status(self) -> str:
        return str(self.artifact.manifest.get("quality_status", "UNVALIDATED_ON_HCMAIC"))

    @property
    def dataset_manifest_hash(self) -> str | None:
        value = self.artifact.manifest.get("dataset_manifest_hash")
        return str(value) if isinstance(value, str) and value else None

    @property
    def artifact_hash(self) -> str | None:
        value = self.artifact.manifest.get("records_sha256")
        return str(value) if isinstance(value, str) and value else None

    def channel_contract(self) -> ChannelContract:
        return ChannelContract(
            channel="asr",
            provider=self.provider,
            revision=self.revision,
            execution_status=self.execution_status,
            quality_status=self.quality_status,
            dataset_manifest_hash=self.dataset_manifest_hash,
            artifact_hash=self.artifact_hash,
            status="ready",
        )

    def search(self, text: str, top_k: int = 100) -> list[ChannelHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_tokens = normalize_ocr_text(text).split()
        if not query_tokens:
            return []
        candidates = sorted(
            {segment_idx for token in query_tokens for segment_idx in self._postings.get(token, [])}
        )
        if not candidates:
            return []
        n_segments = len(self._segments)
        scored: list[tuple[_ASRSegment, float, list[str]]] = []
        for segment_idx in candidates:
            segment = self._segments[segment_idx]
            matching_tokens = [token for token in query_tokens if token in segment.tokens]
            if not matching_tokens:
                continue
            score = 0.0
            for token in set(matching_tokens):
                idf = math.log(1.0 + (n_segments + 1.0) / (self._df[token] + 1.0))
                score += idf * (segment.record.confidence or 1.0)
            scored.append((segment, score, matching_tokens))
        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].record.video_id,
                item[0].record.source_frame_idx,
                item[0].record.segment_id,
            )
        )
        return [
            ChannelHit(
                entity_id=segment.record.frame_uid,
                video_id=segment.record.video_id,
                timestamp_ms=segment.record.timestamp_ms,
                modality="asr",
                score=float(score),
                rank=rank,
                provider=self.provider,
                evidence_text=segment.record.text,
                frame_uid=segment.record.frame_uid,
                video_filename=segment.record.video_filename,
                source_frame_idx=segment.record.source_frame_idx,
                evidence=build_channel_evidence(
                    channel="asr",
                    provider=self.provider,
                    revision=self.revision,
                    execution_status=self.execution_status,
                    quality_status=self.quality_status,
                    dataset_manifest_hash=self.dataset_manifest_hash,
                    artifact_hash=self.artifact_hash,
                    frame_uid=segment.record.frame_uid,
                    video_id=segment.record.video_id,
                    video_filename=segment.record.video_filename,
                    source_frame_idx=segment.record.source_frame_idx,
                    timestamp_ms=segment.record.timestamp_ms,
                    score=float(score),
                    rank=rank,
                    channel_specific={
                        "segment_id": segment.record.segment_id,
                        "start_ms": segment.record.start_ms,
                        "end_ms": segment.record.end_ms,
                        "text": segment.record.text,
                        "text_raw": str(
                            (segment.record.metadata or {}).get("text_raw", segment.record.text)
                        ),
                        "confidence": segment.record.confidence,
                        "matched_tokens": matching_tokens,
                    },
                    raw_provenance=self.channel_contract().to_raw_provenance(),
                ),
            )
            for rank, (segment, score, matching_tokens) in enumerate(scored[:top_k], start=1)
        ]
