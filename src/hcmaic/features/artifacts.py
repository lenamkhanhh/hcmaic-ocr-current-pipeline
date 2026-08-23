"""Hash-addressed JSONL feature artifacts with fail-closed loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hcmaic.features.models import FeatureRecord


class FeatureArtifactError(ValueError):
    """Raised when a feature artifact is malformed or tampered."""


def _canonical_bytes(records: list[FeatureRecord]) -> bytes:
    lines = [
        json.dumps(record.model_dump(), sort_keys=True, ensure_ascii=False) for record in records
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def write_feature_records(records: list[FeatureRecord], path: Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(records)
    path.write_bytes(payload)
    modalities = sorted({record.modality for record in records})
    return {
        "path": path.name,
        "n_records": len(records),
        "modality": modalities[0] if len(modalities) == 1 else "mixed",
        "content_hash": hashlib.sha256(payload).hexdigest(),
        "evidence_level": "FIXTURE_VERIFIED",
    }


def load_feature_records(
    path: Path, *, expected_content_hash: str | None = None
) -> list[FeatureRecord]:
    payload = Path(path).read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if expected_content_hash is not None and actual_hash != expected_content_hash:
        raise FeatureArtifactError(
            f"feature artifact hash mismatch: expected {expected_content_hash}, got {actual_hash}"
        )
    records: list[FeatureRecord] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            record = FeatureRecord.model_validate_json(line)
        except (UnicodeDecodeError, ValueError) as exc:
            raise FeatureArtifactError(
                f"invalid feature record at line {line_number}: {exc}"
            ) from exc
        records.append(record)
    if _canonical_bytes(records) != payload:
        raise FeatureArtifactError(
            "feature artifact is not canonical; refusing mixed or rewritten records"
        )
    return records
