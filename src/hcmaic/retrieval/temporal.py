"""Stable temporal expansion over keyframes and shots."""

from __future__ import annotations

from hcmaic.contracts.models import FrameRecord


def expand_temporal(
    seed_frame_ids: list[str],
    frames: list[FrameRecord],
    *,
    window_ms: int = 0,
    include_same_shot: bool = True,
    adjacent_shots: int = 0,
) -> list[str]:
    if window_ms < 0 or adjacent_shots < 0:
        raise ValueError("temporal window and adjacent_shots must be >= 0")
    by_id = {frame.frame_id: frame for frame in frames}
    ordered = sorted(frames, key=lambda frame: (frame.timestamp_ms, frame.frame_id))
    shot_order: list[str] = []
    for frame in ordered:
        if frame.shot_id and frame.shot_id not in shot_order:
            shot_order.append(frame.shot_id)

    result: list[str] = []

    def add(frame_id: str) -> None:
        if frame_id in by_id and frame_id not in result:
            result.append(frame_id)

    for seed_id in seed_frame_ids:
        add(seed_id)
    for seed_id in seed_frame_ids:
        seed = by_id.get(seed_id)
        if seed is None:
            continue
        for frame in ordered:
            if frame.video_id != seed.video_id:
                continue
            if abs(frame.timestamp_ms - seed.timestamp_ms) <= window_ms:
                add(frame.frame_id)
        if include_same_shot and seed.shot_id:
            for frame in ordered:
                if frame.video_id == seed.video_id and frame.shot_id == seed.shot_id:
                    add(frame.frame_id)
        if adjacent_shots and seed.shot_id in shot_order:
            shot_index = shot_order.index(seed.shot_id)
            lower = max(0, shot_index - adjacent_shots)
            upper = shot_index + adjacent_shots + 1
            wanted_shots = set(shot_order[lower:upper])
            for frame in ordered:
                if frame.video_id == seed.video_id and frame.shot_id in wanted_shots:
                    add(frame.frame_id)
    return result
