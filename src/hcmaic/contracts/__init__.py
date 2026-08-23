"""Typed data contracts shared by every module."""

from hcmaic.contracts.kis import (
    Evidence,
    KISChannelConfig,
    KISPipelineConfig,
    KISQuery,
    KISResult,
)
from hcmaic.contracts.models import (
    CanonicalSubmission,
    FrameRecord,
    SearchRequest,
    SearchResult,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "CanonicalSubmission",
    "FrameRecord",
    "SearchRequest",
    "SearchResult",
    "ValidationIssue",
    "ValidationReport",
    "Evidence",
    "KISChannelConfig",
    "KISPipelineConfig",
    "KISQuery",
    "KISResult",
]
