"""Canonical records for visual/OCR/ASR/caption/metadata features."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Modality = Literal["visual", "ocr", "asr", "caption", "metadata", "segment"]


class FeatureRecord(BaseModel):
    video_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    modality: Modality
    provider: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    text: str | None = None
    artifact_ref: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    content_hash: str = Field(min_length=64, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_range(self) -> FeatureRecord:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        if self.text is None and self.artifact_ref is None:
            raise ValueError("feature needs text or artifact_ref")
        return self

    @classmethod
    def from_content(
        cls,
        *,
        video_id: str,
        entity_id: str,
        start_ms: int,
        end_ms: int,
        modality: Modality,
        provider: str,
        revision: str,
        text: str | None = None,
        artifact_ref: str | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureRecord:
        payload = {
            "video_id": video_id,
            "entity_id": entity_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "modality": modality,
            "provider": provider,
            "revision": revision,
            "text": text,
            "artifact_ref": artifact_ref,
            "confidence": confidence,
            "metadata": metadata or {},
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return cls(
            video_id=video_id,
            entity_id=entity_id,
            start_ms=start_ms,
            end_ms=end_ms,
            modality=modality,
            provider=provider,
            revision=revision,
            text=text,
            artifact_ref=artifact_ref,
            confidence=confidence,
            metadata=metadata or {},
            content_hash=hashlib.sha256(encoded).hexdigest(),
        )
