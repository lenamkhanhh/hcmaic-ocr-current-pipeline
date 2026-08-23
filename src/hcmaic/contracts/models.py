"""Mandatory data contracts (see CLAUDE.md / GOAL.md).

`frame_id` is globally unique and built as ``{video_id}:{keyframe_id}``,
e.g. ``L01_V001:001``. It is URL-safe for path parameters.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

FRAME_ID_SEP = ":"


def make_frame_id(video_id: str, keyframe_id: str) -> str:
    return f"{video_id}{FRAME_ID_SEP}{keyframe_id}"


class FrameRecord(BaseModel):
    """One indexed keyframe, fully mappable back to its video and time."""

    frame_id: str
    video_id: str
    keyframe_id: str
    frame_idx: int = Field(ge=0)
    source_frame_idx: int | None = Field(default=None, ge=0)
    pts: float | None = None
    timestamp_ms: int = Field(ge=0)
    shot_id: str | None = None
    shot_start_ms: int | None = Field(default=None, ge=0)
    shot_end_ms: int | None = Field(default=None, ge=0)
    image_path: str  # relative to the dataset root, POSIX separators
    video_filename: str | None = None
    frame_count: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_version: str | None = None


class SearchRequest(BaseModel):
    query_id: str = Field(min_length=1)
    session_id: str | None = None
    task_type: str = "kis"
    text: str = Field(min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=500)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class SearchResult(BaseModel):
    rank: int = Field(ge=1)
    final_score: float
    signal_scores: dict[str, float]
    video_id: str
    frame_id: str
    frame_idx: int
    timestamp_ms: int
    image_url: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    index_version: str


class CanonicalSubmission(BaseModel):
    """Internal submission contract.

    This is NOT a real HCMAIC/DRES protocol; a SubmissionAdapter maps this
    to the official format once BTC publishes it.
    """

    query_id: str
    task_type: str
    video_id: str
    frame_id: str
    timestamp_ms: int
    answer: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    video_id: str | None = None
    frame_id: str | None = None
    path: str | None = None


class ValidationReport(BaseModel):
    dataset_root: str
    n_videos: int
    n_frames: int
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors
