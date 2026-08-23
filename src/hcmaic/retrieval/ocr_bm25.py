"""Raw-derived OCR artifact contract and deterministic BM25 retrieval channel."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.channel_contract import ChannelContract, build_channel_evidence

OCR_RECORDS_NAME = "ocr.jsonl"
OCR_MANIFEST_NAME = "ocr_manifest.json"
OCR_FORMAT = "hcmaic-raw-ocr-v1"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _is_mock_provider(provider: str) -> bool:
    return "mock" in provider.casefold()


class OCRArtifactError(RuntimeError):
    """Raised when OCR evidence is missing, malformed or not raw-derived."""


class OCRUnavailableError(OCRArtifactError):
    """Raised when the optional OCR channel cannot execute locally."""


class OCRProvider(Protocol):
    """Provider contract for a real local OCR implementation."""

    name: str
    version: str
    execution: str


@dataclass(frozen=True)
class OCRRecord:
    """One frame-level OCR observation with official source-frame identity."""

    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int
    text: str
    provider: str
    revision: str
    confidence: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.frame_uid.strip() or not self.video_id.strip():
            raise ValueError("OCR frame_uid and video_id must not be blank")
        if not self.video_filename.strip():
            raise ValueError("OCR video_filename must not be blank")
        if self.source_frame_idx < 0 or self.timestamp_ms < 0:
            raise ValueError("OCR source_frame_idx/timestamp_ms must be non-negative")
        if not self.text.strip():
            raise ValueError("OCR text must not be blank")
        if not self.provider.strip() or _is_mock_provider(self.provider):
            raise ValueError("OCR provider must be a real non-mock provider")
        if not self.revision.strip():
            raise ValueError("OCR revision must not be blank")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OCR confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_uid": self.frame_uid,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "source_frame_idx": self.source_frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "text": self.text,
            "provider": self.provider,
            "revision": self.revision,
            "confidence": self.confidence,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class OCRArtifact:
    artifact_dir: Path
    records: tuple[OCRRecord, ...]
    manifest: dict[str, Any]


def _canonical_bytes(records: list[OCRRecord]) -> bytes:
    lines = [json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) for record in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_ocr_artifact(
    records: list[OCRRecord],
    artifact_dir: Path,
    *,
    dataset_manifest_hash: str,
    provider_execution: str = "validated-local",
) -> OCRArtifact:
    """Persist a canonical raw-derived OCR artifact and its provenance manifest."""
    if not dataset_manifest_hash.strip():
        raise OCRArtifactError("dataset_manifest_hash must not be blank")
    if provider_execution != "validated-local":
        raise OCRArtifactError(
            "OCR artifact execution must be validated-local; refusing unverified runtime"
        )
    if not records:
        raise OCRArtifactError("OCR artifact cannot be empty")
    records = sorted(
        records,
        key=lambda record: (record.video_id, record.source_frame_idx, record.frame_uid),
    )
    frame_uids = [record.frame_uid for record in records]
    if len(frame_uids) != len(set(frame_uids)):
        raise OCRArtifactError("OCR artifact has duplicate frame_uid values")
    providers = {record.provider for record in records}
    revisions = {record.revision for record in records}
    if len(providers) != 1 or len(revisions) != 1:
        raise OCRArtifactError("OCR artifact must contain one provider and revision")
    if any(_is_mock_provider(provider) for provider in providers):
        raise OCRArtifactError("mock OCR provider is not allowed")

    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise OCRArtifactError(
            f"Artifact directory {artifact_dir} is not empty; use a new versioned path"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / OCR_RECORDS_NAME
    payload = _canonical_bytes(records)
    records_path.write_bytes(payload)
    manifest = {
        "format": OCR_FORMAT,
        "records": OCR_RECORDS_NAME,
        "records_sha256": _sha256(payload),
        "n_records": len(records),
        "provider": next(iter(providers)),
        "revision": next(iter(revisions)),
        "dataset_manifest_hash": dataset_manifest_hash,
        "raw_video_source": True,
        "btc_artifacts_used": False,
        "provider_execution": provider_execution,
        "evidence_level": "REAL_PROVIDER_ARTIFACT",
        "quality_status": "UNVALIDATED_ON_HCMAIC",
    }
    _write_json(artifact_dir / OCR_MANIFEST_NAME, manifest)
    return load_ocr_artifact(artifact_dir, dataset_manifest_hash=dataset_manifest_hash)


def _parse_records(payload: bytes) -> list[OCRRecord]:
    records: list[OCRRecord] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise OCRArtifactError("OCR artifact is not valid UTF-8") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("record is not an object")
            records.append(OCRRecord(**data))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OCRArtifactError(f"invalid OCR record at line {line_number}") from exc
    return records


def load_ocr_artifact(
    artifact_dir: Path, *, dataset_manifest_hash: str | None = None
) -> OCRArtifact:
    """Load OCR records only when raw provenance and hashes validate."""
    artifact_dir = Path(artifact_dir).resolve()
    manifest_path = artifact_dir / OCR_MANIFEST_NAME
    records_path = artifact_dir / OCR_RECORDS_NAME
    if not manifest_path.is_file() or not records_path.is_file():
        raise OCRUnavailableError(f"OCR artifact is unavailable in {artifact_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = records_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OCRArtifactError(f"cannot read OCR artifact in {artifact_dir}") from exc
    if manifest.get("format") != OCR_FORMAT:
        raise OCRArtifactError("unsupported OCR artifact format")
    if manifest.get("raw_video_source") is not True:
        raise OCRArtifactError("OCR artifact is not marked raw_video_source")
    if manifest.get("btc_artifacts_used") is not False:
        raise OCRArtifactError("OCR artifact claims BTC artifacts")
    if manifest.get("provider_execution") != "validated-local":
        raise OCRArtifactError("OCR artifact provider execution is not validated-local")
    if _is_mock_provider(str(manifest.get("provider", ""))):
        raise OCRArtifactError("mock OCR provider is not allowed")
    expected_dataset_hash = manifest.get("dataset_manifest_hash")
    if dataset_manifest_hash is not None and expected_dataset_hash != dataset_manifest_hash:
        raise OCRArtifactError("OCR dataset manifest hash mismatch")
    if manifest.get("records_sha256") != _sha256(payload):
        raise OCRArtifactError("OCR records hash mismatch")
    records = _parse_records(payload)
    if len(records) != int(manifest.get("n_records", -1)):
        raise OCRArtifactError("OCR manifest record count mismatch")
    if any(_is_mock_provider(record.provider) for record in records):
        raise OCRArtifactError("mock OCR provider is not allowed")
    if {record.provider for record in records} != {manifest.get("provider")}:
        raise OCRArtifactError("OCR record provider mismatch")
    if {record.revision for record in records} != {manifest.get("revision")}:
        raise OCRArtifactError("OCR record revision mismatch")
    if _canonical_bytes(records) != payload:
        raise OCRArtifactError("OCR artifact is not canonical")
    if len({record.frame_uid for record in records}) != len(records):
        raise OCRArtifactError("OCR artifact has duplicate frame_uid values")
    return OCRArtifact(artifact_dir, tuple(records), manifest)


def normalize_ocr_text(text: str) -> str:
    """Normalize case and diacritics while retaining token boundaries."""
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    without_diacritics = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(_TOKEN_RE.findall(without_diacritics))


def compact_ocr_text(text: str) -> str:
    """Return a boundary-free normalized form for OCR spacing errors."""
    return "".join(_TOKEN_RE.findall(normalize_ocr_text(text)))


def _tokens(text: str) -> list[str]:
    normalized = normalize_ocr_text(text)
    tokens = normalized.split()
    compact = compact_ocr_text(text)
    if compact and compact not in tokens:
        tokens.append(compact)
    return tokens


def _phrase_tokens(text: str) -> list[str]:
    return normalize_ocr_text(text).split()


@dataclass(frozen=True)
class _OCRDocument:
    record: OCRRecord
    tokens: tuple[str, ...]
    phrase_tokens: tuple[str, ...]


class BM25OCRChannel:
    """In-memory BM25 over an explicit raw-derived OCR artifact."""

    def __init__(
        self,
        artifact: OCRArtifact,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        phrase_boost: float = 1.0,
        proximity_boost: float = 0.25,
        proximity_window: int = 8,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b in [0, 1]")
        if phrase_boost < 0 or proximity_boost < 0 or proximity_window < 1:
            raise ValueError("BM25 boosts must be non-negative and window must be >= 1")
        self.artifact = artifact
        self.k1 = k1
        self.b = b
        self.phrase_boost = phrase_boost
        self.proximity_boost = proximity_boost
        self.proximity_window = proximity_window
        self._documents = tuple(
            _OCRDocument(record, tuple(_tokens(record.text)), tuple(_phrase_tokens(record.text)))
            for record in artifact.records
        )
        self._postings: dict[str, list[int]] = defaultdict(list)
        for document_idx, document in enumerate(self._documents):
            for token in set(document.tokens):
                self._postings[token].append(document_idx)
        self._avgdl = sum(len(document.tokens) for document in self._documents) / len(
            self._documents
        )

    @classmethod
    def from_artifact(
        cls, artifact_dir: Path, *, dataset_manifest_hash: str | None = None, **kwargs: Any
    ) -> BM25OCRChannel:
        artifact = load_ocr_artifact(artifact_dir, dataset_manifest_hash=dataset_manifest_hash)
        return cls(artifact, **kwargs)

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
            channel="ocr",
            provider=self.provider,
            revision=self.revision,
            execution_status=self.execution_status,
            quality_status=self.quality_status,
            dataset_manifest_hash=self.dataset_manifest_hash,
            artifact_hash=self.artifact_hash,
            status="ready",
        )

    def _phrase_bonus(self, document: _OCRDocument, query_phrase: list[str]) -> float:
        if not query_phrase or len(query_phrase) == 1:
            return 0.0
        for start in range(len(document.phrase_tokens) - len(query_phrase) + 1):
            if list(document.phrase_tokens[start : start + len(query_phrase)]) == query_phrase:
                return self.phrase_boost
        positions: list[list[int]] = [
            [idx for idx, token in enumerate(document.phrase_tokens) if token == term]
            for term in query_phrase
        ]
        if any(not values for values in positions):
            return 0.0
        best_window = min(max(window) - min(window) for window in _position_windows(positions))
        if best_window >= self.proximity_window:
            return 0.0
        return self.proximity_boost / (1.0 + best_window)

    def search(self, text: str, top_k: int = 100) -> list[ChannelHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_tokens = _tokens(text)
        query_phrase = _phrase_tokens(text)
        if not query_tokens:
            return []
        candidates = sorted(
            {
                document_idx
                for token in query_tokens
                for document_idx in self._postings.get(token, [])
            }
        )
        if not candidates:
            return []
        n_documents = len(self._documents)
        scored: list[tuple[_OCRDocument, float]] = []
        for document_idx in candidates:
            document = self._documents[document_idx]
            frequencies = Counter(document.tokens)
            length = len(document.tokens)
            score = 0.0
            for token in query_tokens:
                term_frequency = frequencies[token]
                if term_frequency == 0:
                    continue
                document_frequency = len(self._postings[token])
                idf = math.log(
                    1.0 + (n_documents - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length / self._avgdl
                )
                score += idf * (term_frequency * (self.k1 + 1.0)) / denominator
            score += self._phrase_bonus(document, query_phrase)
            scored.append((document, score))
        scored.sort(
            key=lambda item: (
                -item[1],
                item[0].record.video_id,
                item[0].record.source_frame_idx,
                item[0].record.frame_uid,
            )
        )
        return [
            ChannelHit(
                entity_id=document.record.frame_uid,
                video_id=document.record.video_id,
                timestamp_ms=document.record.timestamp_ms,
                modality="ocr",
                score=float(score),
                rank=rank,
                provider=self.provider,
                evidence_text=document.record.text,
                frame_uid=document.record.frame_uid,
                video_filename=document.record.video_filename,
                source_frame_idx=document.record.source_frame_idx,
                evidence=build_channel_evidence(
                    channel="ocr",
                    provider=self.provider,
                    revision=self.revision,
                    execution_status=self.execution_status,
                    quality_status=self.quality_status,
                    dataset_manifest_hash=self.dataset_manifest_hash,
                    artifact_hash=self.artifact_hash,
                    frame_uid=document.record.frame_uid,
                    video_id=document.record.video_id,
                    video_filename=document.record.video_filename,
                    source_frame_idx=document.record.source_frame_idx,
                    timestamp_ms=document.record.timestamp_ms,
                    score=float(score),
                    rank=rank,
                    channel_specific={
                        "confidence": document.record.confidence,
                        "normalized_query": normalize_ocr_text(text),
                        "compact_query": compact_ocr_text(text),
                    },
                    raw_provenance=self.channel_contract().to_raw_provenance(),
                ),
            )
            for rank, (document, score) in enumerate(scored[:top_k], start=1)
        ]


def _position_windows(positions: list[list[int]]) -> list[list[int]]:
    """Enumerate one occurrence of each term for the small OCR query sizes."""
    if not positions:
        return []
    windows = [[position] for position in positions[0]]
    for term_positions in positions[1:]:
        windows = [window + [position] for window in windows for position in term_positions]
    return windows
