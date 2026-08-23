"""Provider protocol and safe extraction context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hcmaic.features.models import FeatureRecord


@dataclass(frozen=True)
class FeatureContext:
    video_id: str
    entity_id: str
    start_ms: int
    end_ms: int
    text_hint: str = ""

    def __post_init__(self) -> None:
        if self.start_ms < 0:
            raise ValueError("start_ms must be >= 0")
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")


class FeatureProvider(Protocol):
    modality: str
    provider: str
    revision: str

    def extract(self, context: FeatureContext) -> list[FeatureRecord]: ...
