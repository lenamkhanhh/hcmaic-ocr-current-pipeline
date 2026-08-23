"""Deterministic OCR/ASR/caption fixtures; no external models."""

from __future__ import annotations

from hcmaic.features.base import FeatureContext
from hcmaic.features.models import FeatureRecord, Modality


class _MockTextProvider:
    def __init__(self, modality: Modality, provider: str) -> None:
        self.modality = modality
        self.provider = provider
        self.revision = "fixture-v1"

    def extract(self, context: FeatureContext) -> list[FeatureRecord]:
        text = context.text_hint.strip() or f"{self.modality} for {context.entity_id}"
        return [
            FeatureRecord.from_content(
                video_id=context.video_id,
                entity_id=context.entity_id,
                start_ms=context.start_ms,
                end_ms=context.end_ms,
                modality=self.modality,
                provider=self.provider,
                revision=self.revision,
                text=text,
                confidence=1.0,
            )
        ]


class MockOCRProvider(_MockTextProvider):
    def __init__(self) -> None:
        super().__init__("ocr", "mock-ocr")


class MockASRProvider(_MockTextProvider):
    def __init__(self) -> None:
        super().__init__("asr", "mock-asr")


class MockCaptionProvider(_MockTextProvider):
    def __init__(self) -> None:
        super().__init__("caption", "mock-caption")
