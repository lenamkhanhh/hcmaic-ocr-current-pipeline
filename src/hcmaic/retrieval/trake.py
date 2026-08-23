"""Deterministic temporal-track assembly for the Trake UI/search contract."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _harmonic_mean(scores: Sequence[float], *, epsilon: float) -> float:
    if not scores:
        raise ValueError("at least one score is required")
    return len(scores) / sum(1.0 / (float(score) + epsilon) for score in scores)


def _stage_score(item: Mapping[str, Any]) -> float:
    for key in ("fusion_score", "fused_score", "final_score", "score"):
        if key in item and item[key] is not None:
            score = float(item[key])
            if not math.isfinite(score) or score < 0:
                raise ValueError("stage score must be finite and >= 0")
            return score
    raise ValueError("stage result is missing a score")


def _normalise_stage_rows(stage_id: str, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        frame_uid = str(row.get("frame_uid") or "").strip()
        if not video_id or not frame_uid:
            raise ValueError(f"{stage_id} result is missing video_id or frame_uid")
        raw_source_frame_idx = row.get("source_frame_idx")
        if isinstance(raw_source_frame_idx, bool) or raw_source_frame_idx is None:
            raise ValueError(f"{stage_id} result is missing source_frame_idx")
        try:
            source_frame_idx = int(raw_source_frame_idx)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{stage_id} result source_frame_idx must be an integer") from exc
        if source_frame_idx < 0:
            raise ValueError(f"{stage_id} result source_frame_idx must be >= 0")
        expected_frame_uid = f"{video_id}:{source_frame_idx}"
        if frame_uid != expected_frame_uid:
            raise ValueError(
                f"{stage_id} result frame_uid identity mismatch: "
                f"expected {expected_frame_uid!r}, got {frame_uid!r}"
            )
        if "timestamp_ms" not in row:
            raise ValueError(f"{stage_id} result is missing timestamp_ms")
        timestamp_ms = int(row["timestamp_ms"])
        if timestamp_ms < 0:
            raise ValueError("timestamp_ms must be >= 0")
        item = dict(row)
        item.update(
            {
                "stage_id": stage_id,
                "video_id": video_id,
                "frame_uid": frame_uid,
                "source_frame_idx": source_frame_idx,
                "timestamp_ms": timestamp_ms,
                "stage_score": _stage_score(row),
            }
        )
        previous = unique.get(frame_uid)
        if previous is None or (-float(item["stage_score"]), frame_uid) < (
            -float(previous["stage_score"]),
            frame_uid,
        ):
            unique[frame_uid] = item
    return sorted(
        unique.values(),
        key=lambda item: (
            -float(item["stage_score"]),
            str(item["video_id"]),
            int(item["timestamp_ms"]),
            str(item["frame_uid"]),
        ),
    )


def build_stage_bundles(
    stage_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    stage_ids: Sequence[str] | None = None,
    max_delta_ms: int | None = 60_000,
    top_k: int = 20,
    beam_width: int | None = None,
    max_bundles_per_video: int = 1,
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    """Build complete, independently searched bundles.

    ``max_delta_ms=None`` is the ordinary staged-search mode: stage candidates
    must share a video and follow the requested forward temporal order, without
    a maximum gap.  A numeric value enables Trake temporal mode, where every
    adjacent stage pair must also be within that positive maximum delta.
    ``max_bundles_per_video`` keeps grouped mode at one bundle per video by
    default, while allowing All Hits to retain several complete bundles from
    the same video.  The cap is intentionally bounded so a dense candidate
    set cannot create an unbounded Cartesian product.  Every returned bundle
    still contains one distinct frame per active stage.  The input mappings
    and rows are copied and never mutated.
    """

    ids = list(stage_ids or stage_results.keys())
    stage_numbers = []
    for stage_id in ids:
        match = str(stage_id).upper().removeprefix("S")
        if not match.isdigit() or int(match) not in range(1, 6):
            raise ValueError("stage_ids must be ordered S1 through S5")
        stage_numbers.append(int(match))
    if (
        not ids
        or len(set(stage_numbers)) != len(stage_numbers)
        or stage_numbers != sorted(stage_numbers)
    ):
        raise ValueError("stage_ids must be ordered S1 through S5")
    if any(stage_id not in stage_results for stage_id in ids):
        raise ValueError("stage_ids must exist in stage_results")
    if max_delta_ms is not None and max_delta_ms < 0:
        raise ValueError("max_delta_ms must be >= 0")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if max_bundles_per_video < 1 or max_bundles_per_video > 100:
        raise ValueError("max_bundles_per_video must be between 1 and 100")
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite value > 0")

    rows_by_stage = {
        stage_id: _normalise_stage_rows(stage_id, stage_results[stage_id]) for stage_id in ids
    }

    width = beam_width if beam_width is not None else max(100, top_k * 50)
    if width < 1:
        raise ValueError("beam_width must be >= 1")

    video_ids = sorted(
        {
            str(row["video_id"])
            for rows in rows_by_stage.values()
            for row in rows
        }
    )
    bundles: list[dict[str, Any]] = []
    for video_id in video_ids:
        rows_for_video = {
            stage_id: [row for row in rows_by_stage[stage_id] if row["video_id"] == video_id]
            for stage_id in ids
        }
        if any(not rows_for_video[stage_id] for stage_id in ids):
            continue

        partials: list[dict[str, Any]] = [
            {
                "video_id": video_id,
                "stages": [row],
                "stage_scores": [float(row["stage_score"])],
                "deltas_ms": [],
            }
            for row in rows_for_video[ids[0]]
        ]
        for stage_id in ids[1:]:
            next_partials: list[dict[str, Any]] = []
            for partial in partials:
                previous = partial["stages"][-1]
                used_frame_uids = {str(item["frame_uid"]) for item in partial["stages"]}
                for row in rows_for_video[stage_id]:
                    if row["frame_uid"] in used_frame_uids:
                        continue
                    delta = int(row["timestamp_ms"]) - int(previous["timestamp_ms"])
                    if delta <= 0:
                        continue
                    if max_delta_ms is not None and delta > max_delta_ms:
                        continue
                    next_partials.append(
                        {
                            "video_id": video_id,
                            "stages": [*partial["stages"], row],
                            "stage_scores": [
                                *partial["stage_scores"],
                                float(row["stage_score"]),
                            ],
                            "deltas_ms": [
                                *partial["deltas_ms"],
                                delta if max_delta_ms is not None else None,
                            ],
                        }
                    )
            next_partials.sort(
                key=lambda partial: (
                    -_harmonic_mean(partial["stage_scores"], epsilon=epsilon),
                    tuple(str(item["frame_uid"]) for item in partial["stages"]),
                )
            )
            partials = next_partials[:width]
            if not partials:
                break
        if not partials:
            continue

        for partial in partials[:max_bundles_per_video]:
            stage_rows = [dict(row) for row in partial["stages"]]
            frame_uids = [str(row["frame_uid"]) for row in stage_rows]
            bundle_score = _harmonic_mean(partial["stage_scores"], epsilon=epsilon)
            bundle_id = f"bundle:{video_id}:{'|'.join(frame_uids)}"
            bundles.append(
                {
                    "bundle_id": bundle_id,
                    "video_id": video_id,
                    "score": bundle_score,
                    "bundle_score": bundle_score,
                    "fusion_score": bundle_score,
                    "stage_ids": list(ids),
                    "stage_scores": list(partial["stage_scores"]),
                    "deltas_ms": list(partial["deltas_ms"]),
                    "temporal_enabled": max_delta_ms is not None,
                    "stages": stage_rows,
                }
            )

    bundles.sort(
        key=lambda bundle: (
            -float(bundle["score"]),
            str(bundle["video_id"]),
            tuple(str(item["frame_uid"]) for item in bundle["stages"]),
        )
    )
    for rank, bundle in enumerate(bundles[:top_k], start=1):
        bundle["bundle_rank"] = rank
        for event_step, (row, delta) in enumerate(
            zip(bundle["stages"], [None, *bundle["deltas_ms"]], strict=True)
        ):
            row.update(
                {
                    "bundle_id": bundle["bundle_id"],
                    "bundle_rank": rank,
                    "bundle_score": bundle["score"],
                    "event_step": event_step,
                    "selection_kind": f"E{event_step + 1}",
                    "delta_from_previous_ms": delta,
                }
            )
    return bundles[:top_k]


def build_trake_tracks(
    stage_results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    stage_ids: Sequence[str] | None = None,
    max_delta_ms: int = 60_000,
    top_k: int = 20,
    beam_width: int | None = None,
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for the temporal bundle builder."""

    ids = list(stage_ids or stage_results.keys())
    stage_numbers = [str(stage_id).upper().removeprefix("S") for stage_id in ids]
    if (
        len(ids) < 2
        or any(not value.isdigit() or int(value) not in range(1, 6) for value in stage_numbers)
        or len(set(stage_numbers)) != len(stage_numbers)
        or stage_numbers != sorted(stage_numbers, key=int)
    ):
        raise ValueError(
            "stage_ids must be unique and ordered by physical stage_id S1 through S5"
        )

    tracks = build_stage_bundles(
        stage_results,
        stage_ids=ids,
        max_delta_ms=max_delta_ms,
        top_k=top_k,
        beam_width=beam_width,
        epsilon=epsilon,
    )
    for rank, track in enumerate(tracks, start=1):
        track_id = track["bundle_id"].replace("bundle:", "track:", 1)
        track["track_id"] = track_id
        track["track_rank"] = rank
        track["track_score"] = track["score"]
        for event_step, (row, delta) in enumerate(
            zip(track["stages"], [None, *track["deltas_ms"]], strict=True)
        ):
            row.update(
                {
                    "track_id": track_id,
                    "track_rank": rank,
                    "track_score": track["score"],
                    "event_step": event_step,
                    "selection_kind": f"E{event_step + 1}",
                    "delta_from_previous_ms": delta,
                    "stage_score": row["stage_score"],
                    "fusion_score": row["stage_score"],
                }
            )
    return tracks
