"""Deterministic shot-detector and within-shot sampling contracts.

The no-shot implementation is the mandatory offline fallback. Optional
PySceneDetect and TransNetV2 adapters are intentionally interface-only until
their dependencies and weights are explicitly installed and benchmarked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ShotDetectorUnavailable(RuntimeError):
    """Raised when an optional shot detector is not installed or enabled."""


@dataclass(frozen=True)
class Shot:
    shot_id: str
    video_id: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s < self.start_s:
            raise ValueError("shot timestamps must satisfy 0 <= start_s <= end_s")


class ShotDetector(Protocol):
    def detect(self, video_id: str, duration_s: float) -> list[Shot]:
        """Return deterministic, ordered shots for one video."""


class NoShotDetector:
    """One full-video shot; safe fallback when no detector is available."""

    def detect(self, video_id: str, duration_s: float) -> list[Shot]:
        if duration_s < 0:
            raise ValueError("duration_s must be >= 0")
        return [Shot(f"{video_id}:shot-000", video_id, 0.0, duration_s)]


class PySceneDetectShotDetector:
    """Reserved adapter; no dependency or model is imported automatically."""

    def detect(self, video_id: str, duration_s: float) -> list[Shot]:
        del video_id, duration_s
        raise ShotDetectorUnavailable(
            "PySceneDetect shot detection is not installed. "
            "Install the optional adapter and run a benchmark before enabling it."
        )


class TransNetV2ShotDetector:
    """Reserved adapter; weights are never downloaded by this package."""

    def detect(self, video_id: str, duration_s: float) -> list[Shot]:
        del video_id, duration_s
        raise ShotDetectorUnavailable(
            "TransNetV2 shot detection is interface-only; provide approved local "
            "weights and an adapter before enabling it."
        )


def sample_shot_times(shots: list[Shot], *, interval_s: float, max_frames: int) -> list[float]:
    """Sample each shot's start plus deterministic extra times.

    The first timestamp of every shot is emitted before extra timestamps from
    earlier shots, so a global cap never silently removes a shot representative.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    timestamps = [shot.start_s for shot in shots[:max_frames]]
    if len(timestamps) >= max_frames:
        return timestamps
    for shot in shots:
        extra = shot.start_s + interval_s
        while extra < shot.end_s - 1e-9 and len(timestamps) < max_frames:
            timestamps.append(extra)
            extra += interval_s
    return timestamps
