"""Normalized deterministic catalog of FrameRecords."""

from __future__ import annotations

import json
from pathlib import Path

from hcmaic.contracts.models import FrameRecord, make_frame_id
from hcmaic.ingestion.mapping import (
    find_keyframe_image,
    keyframe_id_for,
    load_mapping_rows,
)
from hcmaic.ingestion.validator import load_media_info

CATALOG_NAME = "catalog.jsonl"

# Metadata keys copied from media-info JSON when present.
_METADATA_KEYS = (
    "title",
    "author",
    "length",
    "publish_date",
    "watch_url",
    "keywords",
    "width",
    "height",
    "frame_count",
    "video_filename",
    "duration_seconds",
)


def build_catalog(root: Path) -> list[FrameRecord]:
    """Build FrameRecords in deterministic order: (video_id, n) ascending.

    Assumes the dataset already passed validation; rows whose image is
    missing are skipped defensively rather than crashing.
    """
    root = Path(root)
    rows = sorted(load_mapping_rows(root), key=lambda r: (r.video_id, r.n))
    media_cache: dict[str, dict | None] = {}
    records: list[FrameRecord] = []
    for row in rows:
        image = find_keyframe_image(root, row.video_id, row.n)
        if image is None:
            continue
        if row.video_id not in media_cache:
            media_cache[row.video_id] = load_media_info(root, row.video_id)
        info = media_cache[row.video_id] or {}
        metadata = {k: info[k] for k in _METADATA_KEYS if k in info}
        if row.fps > 0:
            metadata["fps"] = row.fps
        if row.width is not None:
            metadata["width"] = row.width
        if row.height is not None:
            metadata["height"] = row.height
        metadata["timestamp_source"] = row.timestamp_source
        if row.ingestion_provider:
            metadata["ingestion_provider"] = row.ingestion_provider
        if row.sampling_policy:
            metadata["sampling_policy"] = row.sampling_policy
        records.append(
            FrameRecord(
                frame_id=row.frame_id or make_frame_id(row.video_id, keyframe_id_for(row.n)),
                video_id=row.video_id,
                keyframe_id=keyframe_id_for(row.n),
                frame_idx=row.frame_idx,
                source_frame_idx=(
                    row.source_frame_idx if row.source_frame_idx is not None else row.frame_idx
                ),
                pts=row.pts_time,
                timestamp_ms=(
                    row.timestamp_ms if row.timestamp_ms is not None else round(row.pts_time * 1000)
                ),
                shot_id=row.shot_id,
                shot_start_ms=(
                    round(row.shot_start * 1000) if row.shot_start is not None else None
                ),
                shot_end_ms=(round(row.shot_end * 1000) if row.shot_end is not None else None),
                image_path=image.relative_to(root).as_posix(),
                video_filename=(
                    row.video_filename
                    or (info.get("video_filename") if isinstance(info, dict) else None)
                ),
                frame_count=(
                    row.frame_count
                    or (int(info["frame_count"]) if info.get("frame_count") else None)
                ),
                metadata=metadata,
            )
        )
    return records


def write_catalog(records: list[FrameRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False, sort_keys=True))
            f.write("\n")


def load_catalog(path: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(FrameRecord.model_validate_json(line))
    return records
