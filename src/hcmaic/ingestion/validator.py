"""Dataset validation with actionable errors.

Checks (mission-mandated minimum):
- required mapping columns          -> mapping-columns (error)
- missing keyframe image            -> missing-image (error)
- duplicate frame/keyframe id       -> duplicate-frame (error)
- invalid video id                  -> invalid-video-id (error)
- negative timestamp                -> negative-timestamp (error)
- timestamp outside known duration  -> timestamp-out-of-range (error)
- invalid/unreadable image          -> unreadable-image (error)
- path traversal / escaping root    -> path-escape (error)
- orphan image without mapping row  -> orphan-image (warning)
- missing optional metadata         -> missing-metadata (warning)
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from hcmaic.contracts.models import ValidationIssue, ValidationReport, make_frame_id
from hcmaic.ingestion.mapping import (
    MappingError,
    MappingRow,
    find_keyframe_image,
    keyframe_id_for,
    load_mapping_rows,
)

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
MEDIA_INFO_DIR = "media-info"


def _err(code: str, message: str, **kw: Any) -> ValidationIssue:
    return ValidationIssue(severity="error", code=code, message=message, **kw)


def _warn(code: str, message: str, **kw: Any) -> ValidationIssue:
    return ValidationIssue(severity="warning", code=code, message=message, **kw)


def load_media_info(root: Path, video_id: str) -> dict[str, Any] | None:
    path = Path(root) / MEDIA_INFO_DIR / f"{video_id}.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _check_image_readable(path: Path) -> str | None:
    """Return an error string if the image cannot be decoded."""
    try:
        with Image.open(path) as im:
            im.verify()
        # verify() invalidates the file object; re-open for a real decode
        with Image.open(path) as im:
            im.convert("RGB")
        return None
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        return f"{type(exc).__name__}: {exc}"


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_dataset(root: Path, check_images: bool = True) -> ValidationReport:
    root = Path(root)
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    try:
        rows: list[MappingRow] = load_mapping_rows(root)
    except MappingError as exc:
        report = ValidationReport(dataset_root=str(root), n_videos=0, n_frames=0)
        report.errors.append(_err("mapping-columns", str(exc)))
        return report

    seen: dict[str, str] = {}
    videos: set[str] = set()
    mapped_images: set[Path] = set()
    durations: dict[str, float | None] = {}

    for row in rows:
        vid = row.video_id
        videos.add(vid)

        if not VIDEO_ID_RE.match(vid):
            errors.append(
                _err(
                    "invalid-video-id",
                    f"{row.source}: video_id {vid!r} is not a valid id "
                    f"(letters/digits/underscore/hyphen only, no separators). "
                    f"Fix the mapping CSV or rename the video.",
                    video_id=vid,
                )
            )
            continue  # dependent checks would produce noise

        frame_id = make_frame_id(vid, keyframe_id_for(row.n))
        if frame_id in seen:
            errors.append(
                _err(
                    "duplicate-frame",
                    f"{row.source}: duplicate keyframe n={row.n} for video "
                    f"{vid} (first seen at {seen[frame_id]}). Remove or "
                    f"renumber the duplicate row.",
                    video_id=vid,
                    frame_id=frame_id,
                )
            )
            continue
        seen[frame_id] = row.source

        if row.pts_time < 0:
            errors.append(
                _err(
                    "negative-timestamp",
                    f"{row.source}: pts_time={row.pts_time} is negative for "
                    f"{frame_id}. Timestamps must be >= 0 seconds.",
                    video_id=vid,
                    frame_id=frame_id,
                )
            )
        if not math.isfinite(row.pts_time) or not math.isfinite(row.fps):
            errors.append(
                _err(
                    "non-finite-timing",
                    f"{row.source}: pts_time and fps must be finite numbers for "
                    f"{frame_id}; got pts_time={row.pts_time}, fps={row.fps}.",
                    video_id=vid,
                    frame_id=frame_id,
                )
            )
        elif row.fps <= 0:
            errors.append(
                _err(
                    "invalid-fps",
                    f"{row.source}: fps={row.fps} must be > 0 for {frame_id}.",
                    video_id=vid,
                    frame_id=frame_id,
                )
            )
        if row.frame_idx < 0:
            errors.append(
                _err(
                    "negative-timestamp",
                    f"{row.source}: frame_idx={row.frame_idx} is negative for {frame_id}.",
                    video_id=vid,
                    frame_id=frame_id,
                )
            )

        if row.shot_start is not None or row.shot_end is not None:
            shot_valid = (
                row.shot_start is not None
                and row.shot_end is not None
                and math.isfinite(row.shot_start)
                and math.isfinite(row.shot_end)
                and row.shot_start >= 0
                and row.shot_end >= row.shot_start
                and row.shot_start <= row.pts_time <= row.shot_end
            )
            if not shot_valid:
                errors.append(
                    _err(
                        "invalid-shot-range",
                        f"{row.source}: shot range "
                        f"[{row.shot_start}, {row.shot_end}] is invalid for "
                        f"pts_time={row.pts_time} ({frame_id}).",
                        video_id=vid,
                        frame_id=frame_id,
                    )
                )

        if vid not in durations:
            info = load_media_info(root, vid)
            length = info.get("length") if info else None
            durations[vid] = float(length) if isinstance(length, (int, float)) else None
            if info is None:
                warnings.append(
                    _warn(
                        "missing-metadata",
                        f"No {MEDIA_INFO_DIR}/{vid}.json metadata (optional).",
                        video_id=vid,
                    )
                )
        duration = durations[vid]
        if duration is not None and row.pts_time > duration:
            errors.append(
                _err(
                    "timestamp-out-of-range",
                    f"{row.source}: pts_time={row.pts_time}s exceeds the "
                    f"declared duration {duration}s of {vid} "
                    f"({MEDIA_INFO_DIR}/{vid}.json 'length'). Fix the mapping "
                    f"or the metadata.",
                    video_id=vid,
                    frame_id=frame_id,
                )
            )
        if duration is not None and row.shot_end is not None and row.shot_end > duration:
            errors.append(
                _err(
                    "invalid-shot-range",
                    f"{row.source}: shot_end={row.shot_end}s exceeds "
                    f"declared duration {duration}s of {vid}.",
                    video_id=vid,
                    frame_id=frame_id,
                )
            )

        image = find_keyframe_image(root, vid, row.n)
        if image is None:
            errors.append(
                _err(
                    "missing-image",
                    f"{row.source}: no keyframe image for {frame_id}; expected "
                    f"keyframes/{vid}/{row.n:03d}.jpg (or .jpeg/.png). Add the "
                    f"image or remove the mapping row.",
                    video_id=vid,
                    frame_id=frame_id,
                )
            )
            continue
        if not _is_within(root, image):
            errors.append(
                _err(
                    "path-escape",
                    f"{row.source}: image path {image} resolves outside the "
                    f"dataset root {root}. Refusing to use it.",
                    video_id=vid,
                    frame_id=frame_id,
                    path=str(image),
                )
            )
            continue
        mapped_images.add(image.resolve())
        if check_images:
            problem = _check_image_readable(image)
            if problem:
                errors.append(
                    _err(
                        "unreadable-image",
                        f"{frame_id}: image {image.relative_to(root)} cannot "
                        f"be decoded ({problem}). Re-export the keyframe.",
                        video_id=vid,
                        frame_id=frame_id,
                        path=str(image),
                    )
                )

    # Orphan images: files on disk that no mapping row references.
    keyframes_dir = root / "keyframes"
    if keyframes_dir.is_dir():
        for image in sorted(keyframes_dir.rglob("*")):
            if (
                image.is_file()
                and image.suffix.lower() in (".jpg", ".jpeg", ".png")
                and image.resolve() not in mapped_images
            ):
                warnings.append(
                    _warn(
                        "orphan-image",
                        f"Image {image.relative_to(root)} has no mapping row; "
                        f"it will not be indexed.",
                        path=str(image),
                    )
                )

    report = ValidationReport(
        dataset_root=str(root),
        n_videos=len(videos),
        n_frames=len(seen),
        errors=errors,
        warnings=warnings,
    )
    return report


def write_validation_report(report: ValidationReport, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
