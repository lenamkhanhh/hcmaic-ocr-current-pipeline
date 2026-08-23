"""Deterministic raw-video ingestion for the SkillPixel submission slice.

This module deliberately owns its generated dataset contract.  The source
video is decoded in order and every saved image keeps the original decoded
frame number in ``source_frame_idx``.  BTC keyframes, feature files and
mapping files are never read here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hcmaic.ingestion.video import SUPPORTED_EXTENSIONS

RAW_MANIFEST_NAME = "dataset_manifest.json"
RAW_COVERAGE_NAME = "coverage_report.json"
RAW_SCHEMA_VERSION = "skillpixel-raw-v1"
MAPPING_COLUMNS = (
    "n",
    "pts_time",
    "fps",
    "frame_idx",
    "source_frame_idx",
    "timestamp_ms",
    "sampling_policy",
    "timestamp_source",
    "video_filename",
    "frame_count",
    "width",
    "height",
)
DEFAULT_STRIDE_FRAMES = 10
_JPEG_QUALITY = 92


class RawIngestError(RuntimeError):
    """Raised when raw ingestion or generated-data validation fails."""


@dataclass(frozen=True)
class RawVideoInfo:
    video_id: str
    video_filename: str
    source_path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    sha256: str
    timestamp_source: str


@dataclass(frozen=True)
class RawIngestReport:
    videos: tuple[RawVideoInfo, ...]
    sampling_policy: str
    frame_count: int

    @property
    def n_videos(self) -> int:
        return len(self.videos)

    @property
    def n_frames(self) -> int:
        return self.frame_count


@dataclass(frozen=True)
class RawDatasetStats:
    n_videos: int
    n_frames: int


@dataclass(frozen=True)
class RawSourceValidationReport:
    """Read-only validation report for a raw-video source directory."""

    input_path: Path
    videos: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def n_videos(self) -> int:
        return len(self.videos)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.videos)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "n_videos": self.n_videos,
            "videos": [dict(item) for item in self.videos],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "valid": self.ok,
        }


@dataclass(frozen=True)
class RawCoverageReport:
    """Coverage/density evidence for one generated raw-video dataset."""

    n_videos: int
    n_source_frames: int
    n_sampled_frames: int
    sampling_ratio: float
    max_nearest_frame_error: int
    per_video: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_videos": self.n_videos,
            "n_source_frames": self.n_source_frames,
            "n_sampled_frames": self.n_sampled_frames,
            "sampling_ratio": self.sampling_ratio,
            "max_nearest_frame_error": self.max_nearest_frame_error,
            "per_video": [dict(item) for item in self.per_video],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_files(input_path: Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise RawIngestError(f"Raw video input does not exist: {input_path}")
    if not files:
        raise RawIngestError(
            f"No raw videos with extensions {sorted(SUPPORTED_EXTENSIONS)} found in {input_path}"
        )
    return files


def _video_id(path: Path) -> str:
    raw = path.stem
    cleaned = "".join(char if char.isalnum() or char in "_-" else "_" for char in raw)
    cleaned = cleaned.strip("_-")
    if not cleaned:
        raise RawIngestError(f"Cannot derive a video_id from raw video {path.name!r}")
    return cleaned


def validate_raw_video_source(input_path: Path) -> RawSourceValidationReport:
    """Probe and hash raw videos before any generated artifact is consumed."""
    input_path = Path(input_path)
    errors: list[str] = []
    videos: list[dict[str, Any]] = []
    try:
        sources = _video_files(input_path)
    except RawIngestError as exc:
        return RawSourceValidationReport(input_path.resolve(), (), (str(exc),))

    seen_ids: set[str] = set()
    for source in sources:
        try:
            video_id = _video_id(source)
            if video_id in seen_ids:
                errors.append(f"duplicate video_id {video_id!r} from {source.name!r}")
                continue
            seen_ids.add(video_id)
            width, height, fps, frame_count = _probe(source)
            videos.append(
                {
                    "video_id": video_id,
                    "video_filename": source.name,
                    "source_path": str(source.resolve()),
                    "sha256": _sha256(source),
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "frame_count": frame_count,
                }
            )
        except (OSError, RawIngestError, ValueError) as exc:
            errors.append(f"{source.name}: {exc}")
    return RawSourceValidationReport(input_path.resolve(), tuple(videos), tuple(errors))


def _probe(path: Path) -> tuple[int, int, float, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - project extra is installed in tests
        raise RawIngestError(
            "Raw video ingestion requires OpenCV. Install with: uv sync --extra video"
        ) from exc

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RawIngestError(f"{path.name}: OpenCV cannot open the raw video")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RawIngestError(
            f"{path.name}: invalid OpenCV metadata width={width}, height={height}, fps={fps}"
        )
    return width, height, fps, max(frame_count, 0)


def _write_mapping(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mapping_count(path: Path) -> int:
    return len(_read_mapping(path))


def _nearest_frame_error(sampled: list[int], frame_count: int) -> int:
    if not sampled or frame_count < 1:
        return frame_count
    sampled_array = np.asarray(sampled, dtype=np.int64)
    source = np.arange(frame_count, dtype=np.int64)
    positions = np.searchsorted(sampled_array, source, side="left")
    right = np.minimum(positions, len(sampled_array) - 1)
    left = np.maximum(positions - 1, 0)
    distances = np.minimum(
        np.abs(source - sampled_array[left]),
        np.abs(source - sampled_array[right]),
    )
    return int(distances.max())


def coverage_report(root: Path) -> RawCoverageReport:
    """Calculate deterministic sample density and nearest-frame coverage."""
    root = Path(root)
    manifest_path = root / RAW_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RawIngestError(f"Missing {RAW_MANIFEST_NAME} in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    per_video: list[dict[str, Any]] = []
    total_source = 0
    total_sampled = 0
    global_error = 0
    for item in manifest.get("raw_videos", []):
        video_id = str(item.get("video_id", ""))
        frame_count = int(item.get("frame_count", 0))
        mapping_path = root / "map-keyframes" / f"{video_id}.csv"
        rows = _read_mapping(mapping_path)
        sampled = sorted(int(row["source_frame_idx"]) for row in rows)
        gaps = np.diff(np.asarray(sampled, dtype=np.int64)) if len(sampled) > 1 else np.array([])
        nearest_error = _nearest_frame_error(sampled, frame_count)
        payload = {
            "video_id": video_id,
            "video_filename": str(item.get("video_filename", f"{video_id}.mp4")),
            "frame_count": frame_count,
            "sampled_frames": len(sampled),
            "sampling_ratio": len(sampled) / frame_count if frame_count else 0.0,
            "min_gap_frames": int(gaps.min()) if gaps.size else None,
            "max_gap_frames": int(gaps.max()) if gaps.size else None,
            "mean_gap_frames": float(gaps.mean()) if gaps.size else None,
            "max_nearest_frame_error": nearest_error,
        }
        per_video.append(payload)
        total_source += frame_count
        total_sampled += len(sampled)
        global_error = max(global_error, nearest_error)
    return RawCoverageReport(
        n_videos=len(per_video),
        n_source_frames=total_source,
        n_sampled_frames=total_sampled,
        sampling_ratio=total_sampled / total_source if total_source else 0.0,
        max_nearest_frame_error=global_error,
        per_video=tuple(per_video),
    )


def _write_coverage_report(root: Path) -> RawCoverageReport:
    report = coverage_report(root)
    (root / RAW_COVERAGE_NAME).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _extract_one(source: Path, output_root: Path, stride_frames: int) -> RawVideoInfo:
    import cv2

    width, height, fps, probed_frame_count = _probe(source)
    video_id = _video_id(source)
    frames_dir = output_root / "keyframes" / video_id
    mapping_path = output_root / "map-keyframes" / f"{video_id}.csv"
    media_path = output_root / "media-info" / f"{video_id}.json"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    rows: list[dict[str, Any]] = []
    decoded_count = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            source_frame_idx = decoded_count
            if source_frame_idx % stride_frames == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                n = len(rows)
                pts_time = source_frame_idx / fps
                image_path = frames_dir / f"{n:03d}.jpg"
                Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8)).save(
                    image_path, quality=_JPEG_QUALITY
                )
                rows.append(
                    {
                        "n": n,
                        "pts_time": round(pts_time, 6),
                        "fps": round(fps, 6),
                        "frame_idx": source_frame_idx,
                        "source_frame_idx": source_frame_idx,
                        "timestamp_ms": round(pts_time * 1000),
                        "sampling_policy": f"uniform_stride_{stride_frames}_v1",
                        "timestamp_source": "cfr_frame_index",
                        "video_filename": source.name,
                        "frame_count": probed_frame_count,
                        "width": width,
                        "height": height,
                    }
                )
            decoded_count += 1
    finally:
        capture.release()

    if decoded_count == 0 or not rows:
        raise RawIngestError(f"{source.name}: no decodable frames were produced")
    frame_count = decoded_count
    if probed_frame_count and probed_frame_count != decoded_count:
        frame_count = decoded_count
        for row in rows:
            row["frame_count"] = frame_count

    _write_mapping(mapping_path, rows)
    info = RawVideoInfo(
        video_id=video_id,
        video_filename=source.name,
        source_path=source.resolve(),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=frame_count / fps,
        sha256=_sha256(source),
        timestamp_source="cfr_frame_index",
    )
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_text(
        json.dumps(
            {
                "video_id": info.video_id,
                "video_filename": info.video_filename,
                "source_file": info.video_filename,
                "source_video_path": str(info.source_path),
                "width": info.width,
                "height": info.height,
                "fps": info.fps,
                "frame_count": info.frame_count,
                "length": round(info.duration_s, 6),
                "duration_seconds": round(info.duration_s, 6),
                "timestamp_source": info.timestamp_source,
                "sampling_policy": f"uniform_stride_{stride_frames}_v1",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return info


def _generated_file_hashes(root: Path) -> dict[str, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != RAW_MANIFEST_NAME
        and path.suffix.lower() in {".csv", ".json", ".jsonl", ".jpg", ".jpeg", ".png"}
    )
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def _write_raw_catalog(root: Path) -> None:
    """Persist the canonical catalog beside raw mappings for audit/package use."""
    from hcmaic.ingestion.catalog import build_catalog, write_catalog

    write_catalog(build_catalog(root), root / "catalog.jsonl")


def _write_manifest(root: Path, videos: list[RawVideoInfo], stride_frames: int) -> None:
    policy = f"uniform_stride_{stride_frames}_v1"
    raw_videos = [
        {
            "video_id": item.video_id,
            "video_filename": item.video_filename,
            "source_path": str(item.source_path),
            "sha256": item.sha256,
            "frame_count": item.frame_count,
            "fps": item.fps,
            "width": item.width,
            "height": item.height,
            "duration_seconds": item.duration_s,
            "timestamp_source": item.timestamp_source,
        }
        for item in videos
    ]
    payload: dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "sampling_policy": policy,
        "stride_frames": stride_frames,
        "raw_videos": raw_videos,
        "n_videos": len(videos),
        "n_frames": sum(
            _mapping_count(root / "map-keyframes" / f"{item.video_id}.csv") for item in videos
        ),
        "coverage_report": RAW_COVERAGE_NAME,
        "files": _generated_file_hashes(root),
    }
    payload["dataset_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    (root / RAW_MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def ingest_raw_videos(
    input_path: Path,
    output_root: Path,
    *,
    stride_frames: int = DEFAULT_STRIDE_FRAMES,
    force: bool = False,
) -> RawIngestReport:
    """Decode raw videos in source order and write a versioned raw dataset."""
    if stride_frames < 1:
        raise RawIngestError(f"stride_frames must be >= 1, got {stride_frames}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _video_files(Path(input_path))
    existing_manifest = output_root / RAW_MANIFEST_NAME
    if existing_manifest.is_file() and not force:
        try:
            manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
            expected_sources = [
                (_video_id(source), source.name, _sha256(source)) for source in sources
            ]
            actual_sources = [
                (
                    str(item.get("video_id", "")),
                    str(item.get("video_filename", "")),
                    str(item.get("sha256", "")),
                )
                for item in manifest.get("raw_videos", [])
            ]
            if (
                int(manifest.get("stride_frames", -1)) == stride_frames
                and actual_sources == expected_sources
                and (output_root / RAW_COVERAGE_NAME).is_file()
            ):
                stats = validate_raw_dataset(output_root)
                cached_infos = tuple(
                    RawVideoInfo(
                        video_id=str(item["video_id"]),
                        video_filename=str(item["video_filename"]),
                        source_path=Path(str(item.get("source_path", item["video_filename"]))),
                        width=int(item["width"]),
                        height=int(item["height"]),
                        fps=float(item["fps"]),
                        frame_count=int(item["frame_count"]),
                        duration_s=float(item.get("duration_seconds", 0.0)),
                        sha256=str(item["sha256"]),
                        timestamp_source=str(item.get("timestamp_source", "unknown")),
                    )
                    for item in manifest["raw_videos"]
                )
                if not (output_root / "catalog.jsonl").is_file():
                    _write_raw_catalog(output_root)
                    _write_manifest(output_root, list(cached_infos), stride_frames)
                return RawIngestReport(
                    cached_infos,
                    str(manifest.get("sampling_policy", f"uniform_stride_{stride_frames}_v1")),
                    stats.n_frames,
                )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            pass
        if any((output_root / name).exists() for name in ("keyframes", "map-keyframes")):
            raise RawIngestError(
                f"Existing raw dataset {output_root} does not match the requested source/stride; "
                "use a new versioned output path or explicit force=True/--force"
            )
    infos: list[RawVideoInfo] = []
    for source in sources:
        video_id = _video_id(source)
        frames_dir = output_root / "keyframes" / video_id
        if frames_dir.exists() and any(frames_dir.iterdir()):
            if not force:
                raise RawIngestError(
                    f"{video_id}: generated frames already exist in {output_root}; "
                    "use force=True/--force to replace them"
                )
            shutil.rmtree(frames_dir)
            (output_root / "map-keyframes" / f"{video_id}.csv").unlink(missing_ok=True)
            (output_root / "media-info" / f"{video_id}.json").unlink(missing_ok=True)
        infos.append(_extract_one(source, output_root, stride_frames))
    _write_manifest(output_root, infos, stride_frames)
    _write_raw_catalog(output_root)
    _write_coverage_report(output_root)
    # Include catalog and coverage artifacts in the final manifest hashes.
    _write_manifest(output_root, infos, stride_frames)
    return RawIngestReport(
        tuple(infos),
        f"uniform_stride_{stride_frames}_v1",
        sum(
            _mapping_count(output_root / "map-keyframes" / f"{item.video_id}.csv") for item in infos
        ),
    )


def validate_raw_dataset(root: Path) -> RawDatasetStats:
    """Fail closed on source-frame, image, count, or manifest mismatches."""
    root = Path(root)
    manifest_path = root / RAW_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RawIngestError(f"Missing {RAW_MANIFEST_NAME} in {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RawIngestError(f"Invalid {manifest_path}: {exc}") from exc
    raw_videos = manifest.get("raw_videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise RawIngestError("dataset_manifest.json has no raw_videos")

    total_frames = 0
    for video in raw_videos:
        video_id = str(video.get("video_id", ""))
        frame_count = int(video.get("frame_count", -1))
        if not video_id or frame_count < 1:
            raise RawIngestError(f"Invalid frame_count/video_id for {video_id!r}")
        mapping_path = root / "map-keyframes" / f"{video_id}.csv"
        if not mapping_path.is_file():
            raise RawIngestError(f"Missing mapping for {video_id}")
        rows = _read_mapping(mapping_path)
        fields = list(rows[0]) if rows else []
        if not fields:
            with mapping_path.open(newline="", encoding="utf-8") as handle:
                fields = list(csv.DictReader(handle).fieldnames or [])
        missing = [column for column in MAPPING_COLUMNS[:7] if column not in fields]
        if missing:
            raise RawIngestError(f"{mapping_path.name}: missing required columns {missing}")
        seen_source: set[int] = set()
        for row in rows:
            try:
                n = int(row["n"])
                source_idx = int(row["source_frame_idx"])
                frame_idx = int(row["frame_idx"])
                timestamp_ms = int(row["timestamp_ms"])
                pts_time = float(row["pts_time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RawIngestError(f"{mapping_path.name}: invalid mapping row {row}") from exc
            if source_idx in seen_source:
                raise RawIngestError(f"duplicate source_frame_idx={source_idx} for {video_id}")
            seen_source.add(source_idx)
            if source_idx < 0 or source_idx >= frame_count:
                raise RawIngestError(
                    f"source_frame_idx={source_idx} out of range for {video_id} [0, {frame_count})"
                )
            if frame_idx != source_idx:
                raise RawIngestError(
                    f"frame_idx/source_frame_idx mismatch for {video_id} n={n}: "
                    f"{frame_idx} != {source_idx}"
                )
            if timestamp_ms != round(pts_time * 1000):
                raise RawIngestError(f"timestamp mismatch for {video_id} n={n}")
            image = root / "keyframes" / video_id / f"{n:03d}.jpg"
            if not image.is_file():
                raise RawIngestError(f"missing image for {video_id} n={n}: {image}")
            try:
                with Image.open(image) as handle:
                    handle.verify()
            except (OSError, SyntaxError) as exc:
                raise RawIngestError(f"unreadable image for {video_id} n={n}") from exc
        if not rows:
            raise RawIngestError(f"empty mapping for {video_id}")
        total_frames += len(rows)

    expected = int(manifest.get("n_frames", -1))
    if expected != total_frames:
        raise RawIngestError(
            f"dataset manifest frame count {expected} != mapping count {total_frames}"
        )
    return RawDatasetStats(n_videos=len(raw_videos), n_frames=total_frames)
