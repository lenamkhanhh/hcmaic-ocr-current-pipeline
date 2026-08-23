"""Raw-derived object detection artifact and label retrieval channel."""

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

OBJECT_RECORDS_NAME = "objects.jsonl"
OBJECT_MANIFEST_NAME = "object_manifest.json"
OBJECT_FORMAT = "hcmaic-raw-objects-v1"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _is_mock_provider(provider: str) -> bool:
    return "mock" in provider.casefold()


class ObjectArtifactError(RuntimeError):
    """Raised when an object artifact is missing, malformed or unsafe."""


class ObjectUnavailableError(ObjectArtifactError):
    """Raised when the optional object channel cannot execute locally."""


class ObjectDetectionProvider(Protocol):
    """Provider metadata contract for team-generated raw-frame detections."""

    name: str
    version: str
    execution: str


@dataclass(frozen=True)
class ObjectRecord:
    """One detected label mapped to an official raw source frame."""

    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int
    label: str
    confidence: float
    provider: str
    revision: str
    bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.frame_uid.strip() or not self.video_id.strip():
            raise ValueError("object frame_uid and video_id must not be blank")
        try:
            frame_video, frame_index = self.frame_uid.rsplit(":", 1)
            if (
                frame_video != self.video_id
                or not frame_index.isdigit()
                or int(frame_index) != self.source_frame_idx
            ):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("object frame_uid must match video_id:source_frame_idx") from exc
        if not self.video_filename.strip():
            raise ValueError("object video_filename must not be blank")
        if self.source_frame_idx < 0 or self.timestamp_ms < 0:
            raise ValueError("object source_frame_idx/timestamp_ms must be non-negative")
        if not self.label.strip():
            raise ValueError("object label must not be blank")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("object confidence must be finite and in [0, 1]")
        if not self.provider.strip() or _is_mock_provider(self.provider):
            raise ValueError("object provider must be a real non-mock provider")
        if not self.revision.strip():
            raise ValueError("object revision must not be blank")
        if self.bbox is not None and (
            len(self.bbox) != 4 or any(not math.isfinite(value) or value < 0 for value in self.bbox)
        ):
            raise ValueError("object bbox must have four finite non-negative values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_uid": self.frame_uid,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "source_frame_idx": self.source_frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "label": self.label,
            "confidence": self.confidence,
            "provider": self.provider,
            "revision": self.revision,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True)
class ObjectArtifact:
    artifact_dir: Path
    records: tuple[ObjectRecord, ...]
    manifest: dict[str, Any]


def _canonical_bytes(records: list[ObjectRecord]) -> bytes:
    lines = [json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) for record in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_object_artifact(
    records: list[ObjectRecord],
    artifact_dir: Path,
    *,
    dataset_manifest_hash: str,
    provider_execution: str = "validated-local",
) -> ObjectArtifact:
    """Persist canonical team-generated object detections with provenance."""
    if not dataset_manifest_hash.strip():
        raise ObjectArtifactError("dataset_manifest_hash must not be blank")
    if provider_execution != "validated-local":
        raise ObjectArtifactError(
            "object artifact execution must be validated-local; refusing unverified runtime"
        )
    if not records:
        raise ObjectArtifactError("object artifact cannot be empty")
    records = sorted(
        records,
        key=lambda record: (
            record.video_id,
            record.source_frame_idx,
            record.frame_uid,
            normalize_object_label(record.label),
        ),
    )
    providers = {record.provider for record in records}
    revisions = {record.revision for record in records}
    if len(providers) != 1 or len(revisions) != 1:
        raise ObjectArtifactError("object artifact must contain one provider and revision")
    if any(_is_mock_provider(provider) for provider in providers):
        raise ObjectArtifactError("mock object provider is not allowed")
    keys = [(record.frame_uid, normalize_object_label(record.label)) for record in records]
    if len(keys) != len(set(keys)):
        raise ObjectArtifactError("object artifact has duplicate frame_uid/label values")

    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ObjectArtifactError(
            f"Artifact directory {artifact_dir} is not empty; use a new versioned path"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    records_path = artifact_dir / OBJECT_RECORDS_NAME
    payload = _canonical_bytes(records)
    records_path.write_bytes(payload)
    manifest = {
        "format": OBJECT_FORMAT,
        "records": OBJECT_RECORDS_NAME,
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
    _write_json(artifact_dir / OBJECT_MANIFEST_NAME, manifest)
    return load_object_artifact(artifact_dir, dataset_manifest_hash=dataset_manifest_hash)


def _parse_records(payload: bytes) -> list[ObjectRecord]:
    records: list[ObjectRecord] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ObjectArtifactError("object artifact is not valid UTF-8") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("record is not an object")
            if data.get("bbox") is not None:
                data["bbox"] = tuple(data["bbox"])
            records.append(ObjectRecord(**data))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ObjectArtifactError(f"invalid object record at line {line_number}") from exc
    return records


def load_object_artifact(
    artifact_dir: Path, *, dataset_manifest_hash: str | None = None
) -> ObjectArtifact:
    """Load object detections only when raw provenance and hashes validate."""
    artifact_dir = Path(artifact_dir).resolve()
    manifest_path = artifact_dir / OBJECT_MANIFEST_NAME
    records_path = artifact_dir / OBJECT_RECORDS_NAME
    if not manifest_path.is_file() or not records_path.is_file():
        raise ObjectUnavailableError(f"object artifact is unavailable in {artifact_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = records_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObjectArtifactError(f"cannot read object artifact in {artifact_dir}") from exc
    if manifest.get("format") != OBJECT_FORMAT:
        raise ObjectArtifactError("unsupported object artifact format")
    if manifest.get("raw_video_source") is not True:
        raise ObjectArtifactError("object artifact is not marked raw_video_source")
    if manifest.get("btc_artifacts_used") is not False:
        raise ObjectArtifactError("object artifact claims BTC artifacts")
    if manifest.get("provider_execution") != "validated-local":
        raise ObjectArtifactError("object provider execution is not validated-local")
    if _is_mock_provider(str(manifest.get("provider", ""))):
        raise ObjectArtifactError("mock object provider is not allowed")
    if (
        dataset_manifest_hash is not None
        and manifest.get("dataset_manifest_hash") != dataset_manifest_hash
    ):
        raise ObjectArtifactError("object dataset manifest hash mismatch")
    if manifest.get("records_sha256") != _sha256(payload):
        raise ObjectArtifactError("object records hash mismatch")
    records = _parse_records(payload)
    if len(records) != int(manifest.get("n_records", -1)):
        raise ObjectArtifactError("object manifest record count mismatch")
    if any(_is_mock_provider(record.provider) for record in records):
        raise ObjectArtifactError("mock object provider is not allowed")
    if {record.provider for record in records} != {manifest.get("provider")}:
        raise ObjectArtifactError("object record provider mismatch")
    if {record.revision for record in records} != {manifest.get("revision")}:
        raise ObjectArtifactError("object record revision mismatch")
    if _canonical_bytes(records) != payload:
        raise ObjectArtifactError("object artifact is not canonical")
    keys = [(record.frame_uid, normalize_object_label(record.label)) for record in records]
    if len(keys) != len(set(keys)):
        raise ObjectArtifactError("object artifact has duplicate frame_uid/label values")
    return ObjectArtifact(artifact_dir, tuple(records), manifest)


def normalize_object_label(label: str) -> str:
    """Normalize detector labels and query text into comparable tokens."""
    decomposed = unicodedata.normalize("NFKD", label).casefold()
    without_diacritics = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(_TOKEN_RE.findall(without_diacritics))


@dataclass(frozen=True)
class _ObjectFrame:
    record: ObjectRecord
    tokens: tuple[str, ...]


class ObjectRetrievalChannel:
    """Deterministic label posting-list search over a real object artifact."""

    def __init__(self, artifact: ObjectArtifact) -> None:
        self.artifact = artifact
        self._frames = tuple(
            _ObjectFrame(record, tuple(normalize_object_label(record.label).split()))
            for record in artifact.records
        )
        self._postings: dict[str, list[int]] = defaultdict(list)
        for record_idx, frame in enumerate(self._frames):
            for token in set(frame.tokens):
                self._postings[token].append(record_idx)
        self._document_frequency = Counter(
            token for frame in self._frames for token in set(frame.tokens)
        )

    @classmethod
    def from_artifact(
        cls, artifact_dir: Path, *, dataset_manifest_hash: str | None = None
    ) -> ObjectRetrievalChannel:
        return cls(load_object_artifact(artifact_dir, dataset_manifest_hash=dataset_manifest_hash))

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
            channel="object",
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
        query_tokens = normalize_object_label(text).split()
        if not query_tokens:
            return []
        candidates = sorted(
            {record_idx for token in query_tokens for record_idx in self._postings.get(token, [])}
        )
        if not candidates:
            return []
        n_frames = len(self._frames)
        scored: dict[str, tuple[ObjectRecord, float, list[str]]] = {}
        for record_idx in candidates:
            frame = self._frames[record_idx]
            matching_labels = [token for token in query_tokens if token in frame.tokens]
            if not matching_labels:
                continue
            score = 0.0
            for token in set(matching_labels):
                df = self._document_frequency[token]
                idf = math.log(1.0 + (n_frames + 1.0) / (df + 1.0))
                score += frame.record.confidence * idf
            current = scored.get(frame.record.frame_uid)
            if current is None or score > current[1]:
                scored[frame.record.frame_uid] = (frame.record, score, matching_labels)
        ranked = sorted(
            scored.values(),
            key=lambda item: (
                -item[1],
                item[0].video_id,
                item[0].source_frame_idx,
                item[0].frame_uid,
            ),
        )
        return [
            ChannelHit(
                entity_id=record.frame_uid,
                video_id=record.video_id,
                timestamp_ms=record.timestamp_ms,
                modality="object",
                score=float(score),
                rank=rank,
                provider=self.provider,
                evidence_text=record.label,
                frame_uid=record.frame_uid,
                video_filename=record.video_filename,
                source_frame_idx=record.source_frame_idx,
                evidence=build_channel_evidence(
                    channel="object",
                    provider=self.provider,
                    revision=self.revision,
                    execution_status=self.execution_status,
                    quality_status=self.quality_status,
                    dataset_manifest_hash=self.dataset_manifest_hash,
                    artifact_hash=self.artifact_hash,
                    frame_uid=record.frame_uid,
                    video_id=record.video_id,
                    video_filename=record.video_filename,
                    source_frame_idx=record.source_frame_idx,
                    timestamp_ms=record.timestamp_ms,
                    score=float(score),
                    rank=rank,
                    channel_specific={
                        "label": record.label,
                        "confidence": record.confidence,
                        "matched_tokens": matching_labels,
                        "bbox": list(record.bbox) if record.bbox is not None else None,
                    },
                    raw_provenance=self.channel_contract().to_raw_provenance(),
                ),
            )
            for rank, (record, score, matching_labels) in enumerate(ranked[:top_k], start=1)
        ]
