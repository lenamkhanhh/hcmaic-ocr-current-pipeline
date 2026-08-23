"""AIC26 CSV generation from an ordered review queue.

The AIC26 organizer format is intentionally kept separate from the older
SkillPixel submission contract.  Queue order is the source of truth for
manual seeds; generated rows expand the raw ``source_frame_idx`` timeline of
the same video.  The canonical keyframe catalog is used for identity and for
a conservative upper bound when an exact frame count is not present.  No
model or retrieval call is made here.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_AIC26_DELTA = 3
TRAKE_MAX_DELTA_MS = 60_000


class AIC26SubmissionError(ValueError):
    """A draft cannot be rendered without violating the AIC26 contract."""


@dataclass(frozen=True)
class AIC26GenerationResult:
    task: str
    rows: tuple[dict[str, Any], ...]
    manual_count: int
    generated_count: int
    max_radius_used: int | None
    target_rows: int


def _task(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in {"KIS", "QA", "TRAKE"}:
        raise AIC26SubmissionError("task must be KIS, QA, or TRAKE")
    return normalized


def _int_value(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise AIC26SubmissionError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AIC26SubmissionError(f"{label} must be an integer") from exc
    if parsed < minimum:
        raise AIC26SubmissionError(f"{label} must be >= {minimum}")
    return parsed


def _canonical_catalog(
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for raw_video_id, raw_rows in catalog.items():
        video_id = str(raw_video_id).strip()
        if not video_id:
            raise AIC26SubmissionError("catalog contains an empty video_id")
        rows: list[dict[str, Any]] = []
        seen_uids: set[str] = set()
        for raw in raw_rows:
            row_video_id = str(raw.get("video_id") or video_id).strip()
            if row_video_id != video_id:
                raise AIC26SubmissionError("catalog row video_id does not match its bucket")
            source_frame_idx = _int_value(
                raw.get("source_frame_idx"),
                f"{video_id}.source_frame_idx",
            )
            frame_uid = f"{video_id}:{source_frame_idx}"
            declared_uid = raw.get("frame_uid")
            if declared_uid is not None and str(declared_uid) != frame_uid:
                raise AIC26SubmissionError(
                    f"catalog frame_uid identity mismatch for {video_id}:{source_frame_idx}"
                )
            if frame_uid in seen_uids:
                raise AIC26SubmissionError(f"duplicate catalog frame_uid: {frame_uid}")
            seen_uids.add(frame_uid)
            row = {
                "video_id": video_id,
                "source_frame_idx": source_frame_idx,
                "frame_uid": frame_uid,
                "timestamp_ms": _int_value(
                    raw.get("timestamp_ms", 0),
                    f"{frame_uid}.timestamp_ms",
                ),
            }
            raw_frame_count = raw.get("frame_count")
            if raw_frame_count is None:
                raw_frame_count = raw.get("source_frame_count")
            if raw_frame_count is not None:
                row["source_frame_count"] = _int_value(
                    raw_frame_count,
                    f"{video_id}.source_frame_count",
                    minimum=1,
                )
            rows.append(row)
        if not rows:
            raise AIC26SubmissionError(f"catalog has no keyframes for video {video_id}")
        declared_counts = {
            int(row["source_frame_count"])
            for row in rows
            if row.get("source_frame_count") is not None
        }
        if len(declared_counts) > 1:
            raise AIC26SubmissionError(
                f"catalog has inconsistent source_frame_count values for {video_id}"
            )
        if declared_counts:
            frame_count = next(iter(declared_counts))
            if any(int(row["source_frame_idx"]) >= frame_count for row in rows):
                raise AIC26SubmissionError(
                    f"catalog source_frame_idx exceeds source_frame_count for {video_id}"
                )
        rows.sort(key=lambda row: (int(row["timestamp_ms"]), int(row["source_frame_idx"])))
        result[video_id] = tuple(rows)
    if not result:
        raise AIC26SubmissionError("canonical catalog is empty")
    return result


def _queue_order(item: Mapping[str, Any], fallback: int) -> tuple[int, int, int]:
    queue_position = item.get("queue_position")
    rank = item.get("rank")
    try:
        position = int(queue_position) if queue_position is not None else fallback
    except (TypeError, ValueError):
        position = fallback
    try:
        item_rank = int(rank) if rank is not None else fallback
    except (TypeError, ValueError):
        item_rank = fallback
    return position, item_rank, fallback


def _ordered_queue(queue_items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        item
        for _, item in sorted(
            enumerate(queue_items),
            key=lambda pair: _queue_order(pair[1], pair[0]),
        )
    ]


def _resolve_seed(
    item: Mapping[str, Any],
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, tuple[dict[str, Any], ...], int, bool]:
    video_id = str(item.get("video_id") or "").strip()
    if not video_id:
        raw_uid = str(item.get("frame_uid") or "")
        video_id = raw_uid.split(":", 1)[0].strip()
    if video_id not in catalog:
        raise AIC26SubmissionError(f"queue video_id is not in canonical catalog: {video_id!r}")
    rows = tuple(catalog[video_id])

    raw_source_idx = item.get("source_frame_idx")
    if raw_source_idx is None:
        raw_uid = str(item.get("frame_uid") or "")
        try:
            raw_source_idx = raw_uid.rsplit(":", 1)[1]
        except IndexError as exc:
            raise AIC26SubmissionError("queue item is missing source_frame_idx/frame_uid") from exc
    # A stale/manual queue item may be outside the video range.  Parse it as
    # a signed integer first; the resolver below clips it to a valid raw
    # source-frame index instead of allowing an out-of-range export.
    if isinstance(raw_source_idx, bool):
        raise AIC26SubmissionError(f"queue source_frame_idx for {video_id} must be an integer")
    try:
        source_idx = int(raw_source_idx)
    except (TypeError, ValueError) as exc:
        raise AIC26SubmissionError(
            f"queue source_frame_idx for {video_id} must be an integer"
        ) from exc

    exact_uid = str(item.get("frame_uid") or "").strip()
    if exact_uid and exact_uid != f"{video_id}:{source_idx}":
        raise AIC26SubmissionError(
            f"queue frame_uid identity mismatch for {video_id}: "
            f"frame_uid={exact_uid!r}, source_frame_idx={source_idx}"
        )
    lower, upper = _source_bounds(rows)
    clipped_source_idx = min(max(source_idx, lower), upper)
    return video_id, rows, clipped_source_idx, clipped_source_idx != source_idx


def _source_bounds(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Return the valid raw source-frame interval for one video.

    ``frame_count`` is preferred when serving metadata provides it.  The
    current canonical dual-index catalog does not carry that field, so the
    largest indexed source frame is used as a conservative upper bound. This
    never fabricates an index beyond the known source range; generated frames
    may intentionally be absent from the sparse keyframe catalog.
    """

    if not rows:
        raise AIC26SubmissionError("video has no canonical rows")
    declared_counts = {
        int(row["source_frame_count"])
        for row in rows
        if row.get("source_frame_count") is not None
    }
    if len(declared_counts) > 1:
        video_id = str(rows[0].get("video_id") or "")
        raise AIC26SubmissionError(
            f"inconsistent source_frame_count values for {video_id}"
        )
    if declared_counts:
        upper = next(iter(declared_counts)) - 1
    else:
        upper = max(int(row["source_frame_idx"]) for row in rows)
    if upper < 0:
        video_id = str(rows[0].get("video_id") or "")
        raise AIC26SubmissionError(f"video has no valid source-frame range: {video_id}")
    return 0, upper


def _clip_source_idx(rows: Sequence[Mapping[str, Any]], source_idx: int) -> int:
    lower, upper = _source_bounds(rows)
    return min(max(source_idx, lower), upper)


def _simple_row(
    video_id: str,
    source_frame_idx: int,
    *,
    generated: bool,
    seed_rank: int,
    delta: int,
    answer: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "video_id": video_id,
        "source_frame_idx": source_frame_idx,
        "frame_uid": f"{video_id}:{source_frame_idx}",
        "generated": generated,
        "seed_rank": seed_rank,
        "delta": delta,
    }
    if answer is not None:
        payload["answer"] = answer
    return payload


def _validate_answer(answer: str | None) -> str:
    if answer is None or not isinstance(answer, str) or not answer.strip():
        raise AIC26SubmissionError("QA answer is required")
    if len(answer) > 100:
        raise AIC26SubmissionError("QA answer must be at most 100 characters")
    return answer


def _canonical_timestamp_for_seed(
    item: Mapping[str, Any],
    video_id: str,
    rows: Sequence[Mapping[str, Any]],
    source_frame_idx: int,
) -> int:
    canonical_rows = [
        row
        for row in rows
        if int(row["source_frame_idx"]) == source_frame_idx
    ]
    provided_timestamp = item.get("timestamp_ms")
    if provided_timestamp is not None:
        provided_timestamp = _int_value(
            provided_timestamp,
            f"{video_id}:{source_frame_idx}.timestamp_ms",
        )
    if canonical_rows:
        canonical_timestamp = _int_value(
            canonical_rows[0].get("timestamp_ms"),
            f"{video_id}:{source_frame_idx}.timestamp_ms",
        )
        if provided_timestamp is not None and provided_timestamp != canonical_timestamp:
            raise AIC26SubmissionError(
                f"TRAKE timestamp_ms does not match canonical frame {video_id}:{source_frame_idx}"
            )
        return canonical_timestamp
    if provided_timestamp is None:
        raise AIC26SubmissionError(
            f"TRAKE queue item {video_id}:{source_frame_idx} requires canonical timestamp_ms"
        )
    return provided_timestamp


def _max_source_radius(
    seeds: Sequence[tuple[tuple[dict[str, Any], ...], int]],
) -> int:
    return max(
        max(source_idx - _source_bounds(rows)[0], _source_bounds(rows)[1] - source_idx)
        for rows, source_idx in seeds
    )


def _generate_simple_rows(
    task: str,
    queue_items: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_rows: int,
    answer: str | None,
    delta: int,
) -> AIC26GenerationResult:
    if task == "QA":
        answer = _validate_answer(answer)
    ordered = _ordered_queue(queue_items)
    if not ordered:
        raise AIC26SubmissionError("queue is empty")
    if len(ordered) > target_rows:
        raise AIC26SubmissionError(
            f"queue has {len(ordered)} manual rows, more than target_rows={target_rows}"
        )

    seeds: list[tuple[str, tuple[dict[str, Any], ...], int, bool]] = []
    for item in ordered:
        seeds.append(_resolve_seed(item, catalog))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        video_id: str,
        source_frame_idx: int,
        *,
        generated: bool,
        seed_rank: int,
        delta: int,
    ) -> bool:
        uid = f"{video_id}:{source_frame_idx}"
        if uid in seen:
            return False
        seen.add(uid)
        rows.append(
            _simple_row(
                video_id,
                source_frame_idx,
                generated=generated,
                seed_rank=seed_rank,
                delta=delta,
                answer=answer,
            )
        )
        return True

    for seed_rank, (video_id, _rows_for_video, source_idx, _clipped) in enumerate(seeds, 1):
        add(video_id, source_idx, generated=False, seed_rank=seed_rank, delta=0)

    if len(rows) == target_rows:
        return AIC26GenerationResult(
            task,
            tuple(rows),
            len(seeds),
            0,
            None,
            target_rows,
        )

    max_source_radius = _max_source_radius(
        [(rows_for_video, source_idx) for _, rows_for_video, source_idx, _ in seeds]
    )
    max_radius = (max_source_radius + delta - 1) // delta
    max_radius_used: int | None = None
    for radius in range(1, max_radius + 1):
        for seed_rank, (video_id, rows_for_video, source_idx, _clipped) in enumerate(seeds, 1):
            for signed_delta in (-radius * delta, radius * delta):
                candidate_idx = _clip_source_idx(rows_for_video, source_idx + signed_delta)
                if add(
                    video_id,
                    candidate_idx,
                    generated=True,
                    seed_rank=seed_rank,
                    delta=candidate_idx - source_idx,
                ):
                    max_radius_used = max(max_radius_used or 0, abs(candidate_idx - source_idx))
                if len(rows) == target_rows:
                    return AIC26GenerationResult(
                        task,
                        tuple(rows),
                        len(seeds),
                        len(rows) - len(seeds),
                        max_radius_used,
                        target_rows,
                    )

    # The preferred source-frame stride visits only one modulo-delta sequence.
    # Walk the remaining raw offsets deterministically so a sparse queue can
    # still reach exactly target_rows without using keyframe ordinal positions
    # or random sampling.
    for source_radius in range(1, max_source_radius + 1):
        for seed_rank, (video_id, rows_for_video, source_idx, _clipped) in enumerate(seeds, 1):
            for signed_delta in (-source_radius, source_radius):
                candidate_idx = _clip_source_idx(rows_for_video, source_idx + signed_delta)
                if add(
                    video_id,
                    candidate_idx,
                    generated=True,
                    seed_rank=seed_rank,
                    delta=candidate_idx - source_idx,
                ):
                    max_radius_used = max(max_radius_used or 0, abs(candidate_idx - source_idx))
                if len(rows) == target_rows:
                    return AIC26GenerationResult(
                        task,
                        tuple(rows),
                        len(seeds),
                        len(rows) - len(seeds),
                        max_radius_used,
                        target_rows,
                    )

    raise AIC26SubmissionError(
        f"canonical catalog has only {len(rows)} unique rows; "
        f"cannot reach target_rows={target_rows}"
    )


def _trake_seeds(
    queue_items: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    event_count: int | None,
) -> list[tuple[str, str, list[tuple[tuple[dict[str, Any], ...], int]]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    group_order: dict[str, int] = {}
    for index, item in enumerate(_ordered_queue(queue_items)):
        event_step = item.get("event_step")
        if event_step is None:
            raise AIC26SubmissionError("TRAKE queue items require event_step and chain_id metadata")
        chain_id = str(item.get("chain_id") or "").strip()
        if not chain_id:
            raise AIC26SubmissionError("TRAKE queue items require a stable chain_id")
        groups.setdefault(chain_id, []).append(item)
        group_order.setdefault(chain_id, index)

    ordered_groups = sorted(groups, key=lambda key: group_order[key])
    seeds: list[tuple[str, str, list[tuple[tuple[dict[str, Any], ...], int]]]] = []
    inferred_event_count: int | None = event_count
    for chain_id in ordered_groups:
        items = sorted(
            groups[chain_id],
            key=lambda item: _int_value(item.get("event_step"), "TRAKE event_step"),
        )
        steps = [_int_value(item.get("event_step"), "TRAKE event_step") for item in items]
        if steps != list(range(len(steps))):
            raise AIC26SubmissionError(f"TRAKE chain {chain_id!r} event steps must be E1..EN")
        selection_kinds = [str(item.get("selection_kind") or "").strip().upper() for item in items]
        expected_selection_kinds = [f"E{step + 1}" for step in steps]
        if selection_kinds != expected_selection_kinds:
            raise AIC26SubmissionError(
                f"TRAKE chain {chain_id!r} selection_kind must match "
                "contiguous E1..EN event_step values"
            )
        query_ids = {str(item.get("query_id") or "").strip() for item in items}
        if len(query_ids) != 1 or "" in query_ids:
            raise AIC26SubmissionError(
                f"TRAKE chain {chain_id!r} requires one non-empty query_id"
            )
        if inferred_event_count is None:
            inferred_event_count = len(items)
        if len(items) != inferred_event_count:
            raise AIC26SubmissionError(
                f"TRAKE chain {chain_id!r} has {len(items)} events; expected {inferred_event_count}"
            )
        resolved: list[tuple[tuple[dict[str, Any], ...], int]] = []
        resolved_timestamps: list[int] = []
        video_id: str | None = None
        physical_stage_ids: list[str] = []
        for item in items:
            current_video_id, rows, source_idx, _clipped = _resolve_seed(item, catalog)
            if video_id is None:
                video_id = current_video_id
            elif video_id != current_video_id:
                raise AIC26SubmissionError(f"TRAKE chain {chain_id!r} spans multiple videos")
            physical_stage_id = str(item.get("stage_id") or "").strip().upper()
            if not physical_stage_id.startswith("S") or not physical_stage_id[1:].isdigit():
                raise AIC26SubmissionError(
                    f"TRAKE chain {chain_id!r} requires physical stage_id metadata"
                )
            physical_stage_ids.append(physical_stage_id)
            resolved_timestamps.append(
                _canonical_timestamp_for_seed(item, current_video_id, rows, source_idx)
            )
            resolved.append((rows, source_idx))
        physical_stage_numbers = [int(stage_id[1:]) for stage_id in physical_stage_ids]
        if (
            any(number not in range(1, 6) for number in physical_stage_numbers)
            or len(set(physical_stage_ids)) != len(physical_stage_ids)
            or physical_stage_numbers != sorted(physical_stage_numbers)
        ):
            raise AIC26SubmissionError(
                f"TRAKE chain {chain_id!r} physical stage_id values must be strictly ordered"
            )
        source_indexes = [source_idx for _rows, source_idx in resolved]
        if any(
            left >= right for left, right in zip(source_indexes, source_indexes[1:], strict=False)
        ):
            raise AIC26SubmissionError(
                f"TRAKE chain {chain_id!r} events must be strictly increasing"
            )
        if any(
            left >= right
            for left, right in zip(resolved_timestamps, resolved_timestamps[1:], strict=False)
        ):
            raise AIC26SubmissionError(
                f"TRAKE chain {chain_id!r} canonical timestamps must be strictly increasing"
            )
        temporal_enabled = any(
            item.get("bundle_temporal_enabled") is True
            for item in items
        )
        if temporal_enabled and any(
            right - left > TRAKE_MAX_DELTA_MS
            for left, right in zip(resolved_timestamps, resolved_timestamps[1:], strict=False)
        ):
            raise AIC26SubmissionError(
                f"TRAKE chain {chain_id!r} temporal mode requires adjacent timestamp gaps "
                "within 60s"
            )
        seeds.append((chain_id, video_id or "", resolved))
    if not seeds:
        raise AIC26SubmissionError("TRAKE queue is empty")
    return seeds


def _generate_trake_rows(
    queue_items: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_rows: int,
    event_count: int | None,
    delta: int,
) -> AIC26GenerationResult:
    seeds = _trake_seeds(queue_items, catalog, event_count=event_count)
    if len(seeds) > target_rows:
        raise AIC26SubmissionError(
            f"queue has {len(seeds)} manual TRAKE rows, more than target_rows={target_rows}"
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add(
        chain_id: str,
        video_id: str,
        resolved: Sequence[tuple[tuple[dict[str, Any], ...], int]],
        *,
        generated: bool,
        delta: int,
    ) -> bool:
        source_indexes = [source_idx for _rows_for_video, source_idx in resolved]
        frame_uids = tuple(f"{video_id}:{source_idx}" for source_idx in source_indexes)
        if frame_uids in seen:
            return False
        if any(
            left >= right for left, right in zip(source_indexes, source_indexes[1:], strict=False)
        ):
            return False
        seen.add(frame_uids)
        rows.append(
            {
                "video_id": video_id,
                "source_frame_idx": source_indexes,
                "frame_uid": list(frame_uids),
                "generated": generated,
                "chain_id": chain_id,
                "delta": delta,
                "event_count": len(source_indexes),
            }
        )
        return True

    for chain_id, video_id, resolved in seeds:
        add(chain_id, video_id, resolved, generated=False, delta=0)

    if len(rows) == target_rows:
        return AIC26GenerationResult(
            "TRAKE",
            tuple(rows),
            len(seeds),
            0,
            None,
            target_rows,
        )

    max_seed_radius = max(
        _max_source_radius(resolved)
        for _, _, resolved in seeds
    )
    max_radius = (max_seed_radius + delta - 1) // delta
    max_radius_used: int | None = None
    for radius in range(1, max_radius + 1):
        for chain_id, video_id, resolved in seeds:
            # First perturb one event at a time.  Then shift the complete
            # event chain in both directions.  Every emitted row is still a
            # complete E1..EN chain, never a partial event list.
            for event_index in range(len(resolved)):
                for signed_delta in (-radius * delta, radius * delta):
                    variant = list(resolved)
                    rows_for_video, source_idx = variant[event_index]
                    candidate_idx = _clip_source_idx(
                        rows_for_video,
                        source_idx + signed_delta,
                    )
                    variant[event_index] = (rows_for_video, candidate_idx)
                    if add(
                        chain_id,
                        video_id,
                        variant,
                        generated=True,
                        delta=candidate_idx - source_idx,
                    ):
                        max_radius_used = max(max_radius_used or 0, abs(candidate_idx - source_idx))
                    if len(rows) == target_rows:
                        return AIC26GenerationResult(
                            "TRAKE",
                            tuple(rows),
                            len(seeds),
                            len(rows) - len(seeds),
                            max_radius_used,
                            target_rows,
                        )
            for signed_delta in (-radius * delta, radius * delta):
                variant = [
                    (
                        rows_for_video,
                        _clip_source_idx(rows_for_video, source_idx + signed_delta),
                    )
                    for rows_for_video, source_idx in resolved
                ]
                actual_deltas = [
                    variant_idx - source_idx
                    for (_rows_for_video, variant_idx), (_original_rows, source_idx) in zip(
                        variant,
                        resolved,
                        strict=True,
                    )
                ]
                if add(
                    chain_id,
                    video_id,
                    variant,
                    generated=True,
                    delta=actual_deltas[0] if actual_deltas else signed_delta,
                ):
                    max_radius_used = max(
                        max_radius_used or 0,
                        max((abs(value) for value in actual_deltas), default=0),
                    )
                if len(rows) == target_rows:
                    return AIC26GenerationResult(
                        "TRAKE",
                        tuple(rows),
                        len(seeds),
                        len(rows) - len(seeds),
                        max_radius_used,
                        target_rows,
                    )

    # Complete the row count with unvisited raw source-frame neighbors when
    # the preferred delta stride cannot cover enough valid complete chains.
    for source_radius in range(1, max_seed_radius + 1):
        for chain_id, video_id, resolved in seeds:
            for event_index in range(len(resolved)):
                for signed_delta in (-source_radius, source_radius):
                    variant = list(resolved)
                    rows_for_video, source_idx = variant[event_index]
                    candidate_idx = _clip_source_idx(
                        rows_for_video,
                        source_idx + signed_delta,
                    )
                    variant[event_index] = (rows_for_video, candidate_idx)
                    if add(
                        chain_id,
                        video_id,
                        variant,
                        generated=True,
                        delta=candidate_idx - source_idx,
                    ):
                        max_radius_used = max(max_radius_used or 0, abs(candidate_idx - source_idx))
                    if len(rows) == target_rows:
                        return AIC26GenerationResult(
                            "TRAKE",
                            tuple(rows),
                            len(seeds),
                            len(rows) - len(seeds),
                            max_radius_used,
                            target_rows,
                        )
            for signed_delta in (-source_radius, source_radius):
                variant = [
                    (
                        rows_for_video,
                        _clip_source_idx(rows_for_video, source_idx + signed_delta),
                    )
                    for rows_for_video, source_idx in resolved
                ]
                actual_deltas = [
                    variant_idx - source_idx
                    for (_rows_for_video, variant_idx), (_original_rows, source_idx) in zip(
                        variant,
                        resolved,
                        strict=True,
                    )
                ]
                if add(
                    chain_id,
                    video_id,
                    variant,
                    generated=True,
                    delta=actual_deltas[0] if actual_deltas else signed_delta,
                ):
                    max_radius_used = max(
                        max_radius_used or 0,
                        max((abs(value) for value in actual_deltas), default=0),
                    )
                if len(rows) == target_rows:
                    return AIC26GenerationResult(
                        "TRAKE",
                        tuple(rows),
                        len(seeds),
                        len(rows) - len(seeds),
                        max_radius_used,
                        target_rows,
                    )

    raise AIC26SubmissionError(
        f"canonical catalog has only {len(rows)} unique TRAKE rows; "
        f"cannot reach target_rows={target_rows}"
    )


def generate_aic26_rows(
    task: str,
    queue_items: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_rows: int = 100,
    answer: str | None = None,
    event_count: int | None = None,
    delta: int = DEFAULT_AIC26_DELTA,
) -> AIC26GenerationResult:
    """Generate an ordered AIC26 draft, filling missing rows from local frames."""

    normalized_task = _task(task)
    if target_rows < 1 or target_rows > 100:
        raise AIC26SubmissionError("target_rows must be between 1 and 100")
    normalized_delta = _int_value(delta, "delta", minimum=1)
    normalized_catalog = _canonical_catalog(catalog)
    if normalized_task == "TRAKE":
        return _generate_trake_rows(
            queue_items,
            normalized_catalog,
            target_rows=target_rows,
            event_count=event_count,
            delta=normalized_delta,
        )
    return _generate_simple_rows(
        normalized_task,
        queue_items,
        normalized_catalog,
        target_rows=target_rows,
        answer=answer,
        delta=normalized_delta,
    )


def _video_id_for_csv(value: Any) -> str:
    video_id = str(value)
    if not video_id or video_id.lower().endswith(".mp4"):
        raise AIC26SubmissionError("CSV video_id must be a filename without .mp4")
    if "\r" in video_id or "\n" in video_id:
        raise AIC26SubmissionError("CSV video_id must not contain line breaks")
    return video_id


def _quote_csv_field(value: str) -> str:
    """Quote one CSV field while escaping embedded double quotes."""

    return '"' + value.replace('"', '""') + '"'


def render_aic26_csv(task: str, rows: Sequence[Mapping[str, Any]]) -> str:
    """Render validated rows as headerless UTF-8-compatible CSV text."""

    normalized_task = _task(task)
    if not rows:
        raise AIC26SubmissionError("cannot render an empty CSV")
    if len(rows) > 100:
        raise AIC26SubmissionError("AIC26 CSV cannot contain more than 100 rows")

    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=",", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        video_id = _video_id_for_csv(row.get("video_id"))
        if normalized_task in {"KIS", "QA"}:
            source_frame_idx = _int_value(row.get("source_frame_idx"), "CSV source_frame_idx")
            if normalized_task == "QA":
                answer = _validate_answer(row.get("answer"))
                prefix = io.StringIO(newline="")
                csv.writer(
                    prefix,
                    delimiter=",",
                    lineterminator="",
                    quoting=csv.QUOTE_MINIMAL,
                ).writerow([video_id, source_frame_idx])
                output.write(prefix.getvalue())
                output.write(f",{_quote_csv_field(answer)}\r\n")
            else:
                writer.writerow([video_id, source_frame_idx])
            continue

        frame_indexes = row.get("source_frame_idx")
        if not isinstance(frame_indexes, (list, tuple)) or not frame_indexes:
            raise AIC26SubmissionError("TRAKE CSV row requires a non-empty event frame list")
        indexes = [_int_value(value, "TRAKE source_frame_idx") for value in frame_indexes]
        if any(left >= right for left, right in zip(indexes, indexes[1:], strict=False)):
            raise AIC26SubmissionError("TRAKE CSV event frame indexes must be strictly increasing")
        raw_timestamps = row.get("timestamp_ms")
        if raw_timestamps is not None:
            if not isinstance(raw_timestamps, (list, tuple)) or len(raw_timestamps) != len(indexes):
                raise AIC26SubmissionError(
                    "TRAKE CSV row timestamp_ms must match the event frame list"
                )
            timestamps = [_int_value(value, "TRAKE timestamp_ms") for value in raw_timestamps]
            if any(left >= right for left, right in zip(timestamps, timestamps[1:], strict=False)):
                raise AIC26SubmissionError(
                    "TRAKE CSV event timestamps must be strictly increasing"
                )
        writer.writerow([video_id, *indexes])
    return output.getvalue()
