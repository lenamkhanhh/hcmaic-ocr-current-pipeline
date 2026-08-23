"""Keyframe-mapping CSV parsing.

Two supported layouts (both BTC-compatible, see UPSTREAM.md):

1. Single file ``<root>/keyframe_mapping.csv`` with columns
   ``video_id,n,pts_time,fps,frame_idx``.
2. Per-video files ``<root>/map-keyframes/<video_id>.csv`` with columns
   ``n,pts_time,fps,frame_idx`` (upstream/BTC convention).

``n`` is the keyframe number; the image is ``keyframes/<video_id>/<nnn>.jpg``
(zero-padded to 3, .jpg/.jpeg/.png probed in that order).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("n", "pts_time", "fps", "frame_idx")
OPTIONAL_COLUMNS = (
    "shot_id",
    "frame_id",
    "shot_start",
    "shot_end",
    "width",
    "height",
    "timestamp_source",
    "ingestion_provider",
    "sampling_policy",
    "source_frame_idx",
    "timestamp_ms",
    "frame_count",
    "video_filename",
)
SINGLE_FILE_NAME = "keyframe_mapping.csv"
PER_VIDEO_DIR = "map-keyframes"


class MappingError(ValueError):
    """Raised when the mapping layout/columns are unusable."""


@dataclass(frozen=True)
class MappingRow:
    video_id: str
    n: int
    pts_time: float
    fps: float
    frame_idx: int
    source: str  # file the row came from, for error messages
    source_frame_idx: int | None = None
    timestamp_ms: int | None = None
    frame_count: int | None = None
    video_filename: str | None = None
    shot_id: str | None = None
    frame_id: str | None = None
    shot_start: float | None = None
    shot_end: float | None = None
    width: int | None = None
    height: int | None = None
    timestamp_source: str = "legacy_mapping"
    ingestion_provider: str | None = None
    sampling_policy: str | None = None


def _optional_float(raw: dict[str, str], key: str) -> float | None:
    value = (raw.get(key) or "").strip()
    return None if not value else float(value)


def _optional_int(raw: dict[str, str], key: str) -> int | None:
    value = (raw.get(key) or "").strip()
    return None if not value else int(value)


def _parse_rows(csv_path: Path, video_id: str | None, root: Path) -> list[MappingRow]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = [c.strip() for c in (reader.fieldnames or [])]
        required = list(REQUIRED_COLUMNS) + ([] if video_id else ["video_id"])
        missing = [c for c in required if c not in fields]
        if missing:
            raise MappingError(
                f"{csv_path.relative_to(root)}: missing required column(s) "
                f"{missing}; found {fields}. Expected columns: {required}."
            )
        rows: list[MappingRow] = []
        for lineno, raw in enumerate(reader, start=2):
            src = f"{csv_path.relative_to(root)}:{lineno}"
            vid = video_id if video_id else (raw.get("video_id") or "").strip()
            try:
                rows.append(
                    MappingRow(
                        video_id=vid,
                        n=int(str(raw["n"]).strip()),
                        pts_time=float(str(raw["pts_time"]).strip()),
                        fps=float(str(raw["fps"]).strip()),
                        frame_idx=int(str(raw["frame_idx"]).strip()),
                        source=src,
                        source_frame_idx=_optional_int(raw, "source_frame_idx"),
                        timestamp_ms=_optional_int(raw, "timestamp_ms"),
                        frame_count=_optional_int(raw, "frame_count"),
                        video_filename=(raw.get("video_filename") or "").strip() or None,
                        shot_id=(raw.get("shot_id") or "").strip() or None,
                        frame_id=(raw.get("frame_id") or "").strip()
                        or f"{vid}:{int(str(raw['n']).strip()):03d}",
                        shot_start=_optional_float(raw, "shot_start"),
                        shot_end=_optional_float(raw, "shot_end"),
                        width=_optional_int(raw, "width"),
                        height=_optional_int(raw, "height"),
                        timestamp_source=(
                            (raw.get("timestamp_source") or "").strip() or "legacy_mapping"
                        ),
                        ingestion_provider=(raw.get("ingestion_provider") or "").strip() or None,
                        sampling_policy=(raw.get("sampling_policy") or "").strip() or None,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise MappingError(f"{src}: unparsable row {dict(raw)!r}: {exc}") from exc
        return rows


def load_mapping_rows(root: Path) -> list[MappingRow]:
    """Load all mapping rows for a dataset root, in file order."""
    root = Path(root)
    single = root / SINGLE_FILE_NAME
    per_video_dir = root / PER_VIDEO_DIR
    if single.is_file():
        return _parse_rows(single, video_id=None, root=root)
    if per_video_dir.is_dir():
        rows: list[MappingRow] = []
        csv_files = sorted(per_video_dir.glob("*.csv"))
        if not csv_files:
            raise MappingError(f"{PER_VIDEO_DIR}/ exists but contains no .csv files in {root}")
        for csv_path in csv_files:
            rows.extend(_parse_rows(csv_path, video_id=csv_path.stem, root=root))
        return rows
    raise MappingError(
        f"No keyframe mapping found in {root}: expected either "
        f"'{SINGLE_FILE_NAME}' or a '{PER_VIDEO_DIR}/' directory of per-video CSVs."
    )


def keyframe_id_for(n: int) -> str:
    return f"{n:03d}"


def find_keyframe_image(root: Path, video_id: str, n: int) -> Path | None:
    """Locate the keyframe image for a mapping row, or None."""
    base = Path(root) / "keyframes" / video_id
    for name in (f"{n:03d}", str(n)):
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = base / f"{name}{ext}"
            if candidate.is_file():
                return candidate
    return None
