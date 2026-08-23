"""Raw video ingestion: MP4/MKV/AVI/MOV -> keyframes + mapping + metadata.

Produces the exact dataset layout the rest of the pipeline already consumes
(`keyframes/<video_id>/NNN.jpg`, `keyframe_mapping.csv`,
`media-info/<video_id>.json`), so `validate-data` / `build-index` / `search`
work on ingested output unchanged.

Backend selection (explicit, no silent magic):

1. ``ffmpeg``  — used when both ``ffmpeg`` and ``ffprobe`` are on PATH.
2. ``opencv`` — pure-pip fallback (``uv sync --extra video``); the wheel
   bundles its own codecs, no system FFmpeg needed.

If neither is available, ingestion fails with an actionable error naming
both options. Keyframes are sampled at a fixed time interval (uniform
sampling — the documented fallback when no shot detector is installed;
PySceneDetect integration is a follow-up ticket, see FINAL_HANDOFF.md).
Near-duplicate consecutive keyframes are dropped via a deterministic
grayscale difference threshold.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hcmaic.ingestion.mapping import SINGLE_FILE_NAME
from hcmaic.ingestion.validator import MEDIA_INFO_DIR, VIDEO_ID_RE

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
DEFAULT_INTERVAL_S = 2.0
DEFAULT_MAX_FRAMES = 500
#: mean absolute grayscale difference (0-255) below which a frame is a duplicate
DEDUP_THRESHOLD = 2.0
_JPEG_QUALITY = 92

MAPPING_COLUMNS = [
    "video_id",
    "n",
    "pts_time",
    "fps",
    "frame_idx",
    "shot_id",
    "frame_id",
    "shot_start",
    "shot_end",
    "width",
    "height",
    "timestamp_source",
    "ingestion_provider",
    "sampling_policy",
]


class IngestError(RuntimeError):
    """Raised when a video cannot be ingested; message is actionable."""


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    source_path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    backend: str


@dataclass
class IngestResult:
    info: VideoInfo
    n_candidates: int
    n_kept: int
    n_duplicates: int
    warnings: list[str] = field(default_factory=list)


def sanitize_video_id(name: str) -> str:
    """Turn a filename stem into a valid video_id or raise IngestError."""
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in name.strip())
    cleaned = cleaned.strip("_-")
    if not cleaned or not VIDEO_ID_RE.match(cleaned):
        raise IngestError(
            f"Cannot derive a valid video_id from {name!r}. Rename the file to "
            f"letters/digits/underscore/hyphen or pass --video-id explicitly."
        )
    return cleaned


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


def _ffmpeg_binaries() -> tuple[str, str] | None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    return None


def _have_opencv() -> bool:
    try:
        import cv2  # noqa: F401

        return True
    except ImportError:
        return False


def available_backend() -> str:
    if _ffmpeg_binaries():
        return "ffmpeg"
    if _have_opencv():
        return "opencv"
    raise IngestError(
        "No video backend available. Either install FFmpeg (ffmpeg+ffprobe on "
        "PATH) or run: uv sync --extra video  (pure-pip OpenCV fallback)."
    )


def _probe_ffprobe(path: Path, ffprobe: str) -> dict[str, Any]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
    except subprocess.CalledProcessError as exc:
        raise IngestError(
            f"{path.name}: ffprobe failed ({exc.stderr.strip()[:200]}). "
            f"The file is likely corrupt or not a video."
        ) from exc
    data = json.loads(out.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise IngestError(f"{path.name}: no video stream found (ffprobe).")
    stream = streams[0]
    num, _, den = str(stream.get("r_frame_rate", "0/1")).partition("/")
    fps = (float(num) / float(den)) if float(den or 1) else 0.0
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    frame_count = int(stream.get("nb_frames") or 0) or int(round(duration * fps))
    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration,
    }


def _probe_opencv(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IngestError(
                f"{path.name}: OpenCV cannot open this file. It is likely "
                f"corrupt, empty, or uses an unsupported codec."
            )
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        if width <= 0 or height <= 0 or frame_count <= 0:
            raise IngestError(
                f"{path.name}: OpenCV reports no decodable video frames "
                f"(width={width}, height={height}, frames={frame_count})."
            )
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_s": duration,
        }
    finally:
        cap.release()


def probe_video(path: Path, video_id: str | None = None) -> VideoInfo:
    """Read container metadata using the best available backend."""
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"Video file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestError(
            f"{path.name}: unsupported extension {path.suffix!r}. Supported: "
            f"{sorted(SUPPORTED_EXTENSIONS)}."
        )
    if video_id is not None and not VIDEO_ID_RE.match(video_id):
        raise IngestError(
            f"--video-id {video_id!r} is invalid: letters/digits/underscore/"
            f"hyphen only (no path separators or dots)."
        )
    backend = available_backend()
    binaries = _ffmpeg_binaries()
    if backend == "ffmpeg" and binaries:
        raw = _probe_ffprobe(path, binaries[1])
    else:
        raw = _probe_opencv(path)
    return VideoInfo(
        video_id=video_id or sanitize_video_id(path.stem),
        source_path=path,
        backend=backend,
        **raw,
    )


# --------------------------------------------------------------------------
# Frame iteration (candidate frames at a fixed interval)
# --------------------------------------------------------------------------


@dataclass
class _Candidate:
    pts_time: float
    frame_idx: int
    image: np.ndarray  # RGB uint8
    timestamp_source: str


def _iter_candidates_opencv(
    info: VideoInfo, interval_s: float, max_frames: int, warnings: list[str]
) -> Iterator[_Candidate]:
    import cv2

    cap = cv2.VideoCapture(str(info.source_path))
    try:
        next_sample = 0.0
        frame_idx = -1
        emitted = 0
        pts_reliable = True
        while emitted < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pts_ms <= 0 and frame_idx > 0:
                # Some codecs give no PTS; fall back to CFR arithmetic once.
                if pts_reliable:
                    warnings.append(
                        f"{info.video_id}: backend gave no PTS; using "
                        f"frame_idx/fps (CFR assumption, VFR videos may drift)."
                    )
                    pts_reliable = False
                pts_s = frame_idx / info.fps if info.fps > 0 else 0.0
            else:
                pts_s = pts_ms / 1000.0
            if pts_s + 1e-9 < next_sample:
                continue
            yield _Candidate(
                pts_time=pts_s,
                frame_idx=frame_idx,
                image=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                timestamp_source=(
                    "best_effort_pts" if pts_ms > 0 or frame_idx == 0 else "cfr_fallback"
                ),
            )
            emitted += 1
            next_sample += interval_s
    finally:
        cap.release()


def _iter_candidates_ffmpeg(
    info: VideoInfo,
    interval_s: float,
    max_frames: int,
    warnings: list[str],
    work_dir: Path,
) -> Iterator[_Candidate]:
    binaries = _ffmpeg_binaries()
    assert binaries is not None
    ffmpeg = binaries[0]
    tmp_dir = work_dir / "_ffmpeg_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pattern = tmp_dir / "cand_%06d.jpg"
    select = f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval_s}),showinfo"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-i",
        str(info.source_path),
        "-vf",
        select,
        "-frames:v",
        str(max_frames),
        "-fps_mode",
        "vfr",
        "-q:v",
        "2",
        str(pattern),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
    except subprocess.CalledProcessError as exc:
        raise IngestError(
            f"{info.video_id}: ffmpeg extraction failed ({(exc.stderr or '').strip()[:200]})."
        ) from exc
    try:
        images = sorted(tmp_dir.glob("cand_*.jpg"))
        timestamps = _parse_ffmpeg_showinfo(completed.stderr)
        if len(images) != len(timestamps):
            raise IngestError(
                f"{info.video_id}: FFmpeg emitted {len(images)} image(s) but "
                f"{len(timestamps)} decoder timestamp(s); refusing guessed timestamps."
            )
        for jpg, pts_s in zip(images, timestamps, strict=True):
            with Image.open(jpg) as im:
                image = np.asarray(im.convert("RGB"))
            yield _Candidate(
                pts_time=pts_s,
                frame_idx=int(round(pts_s * info.fps)),
                image=image,
                timestamp_source="exact_pts",
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


_SHOWINFO_PTS_TIME_RE = re.compile(r"\bpts_time:(?P<pts>[^\s]+)")


def _parse_ffmpeg_showinfo(stderr: str) -> list[float]:
    """Parse decoder-derived PTS values from FFmpeg's ``showinfo`` filter."""
    raw_values = [match.group("pts") for match in _SHOWINFO_PTS_TIME_RE.finditer(stderr)]
    if not raw_values:
        raise IngestError(
            "FFmpeg did not report a usable frame timestamp; refusing CFR reconstruction."
        )
    timestamps: list[float] = []
    for raw in raw_values:
        try:
            pts = float(raw)
        except ValueError as exc:
            raise IngestError(f"FFmpeg reported invalid timestamp {raw!r}.") from exc
        if not math.isfinite(pts):
            raise IngestError(f"FFmpeg reported non-finite timestamp {raw!r}.")
        if pts < 0:
            raise IngestError(f"FFmpeg reported negative timestamp {pts}.")
        timestamps.append(pts)
    return timestamps


def _gray_thumb(image: np.ndarray) -> np.ndarray:
    im = Image.fromarray(image).convert("L").resize((32, 32), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32)


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def ingest_video(
    video_path: Path,
    out_root: Path,
    video_id: str | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_frames: int = DEFAULT_MAX_FRAMES,
    dedup_threshold: float = DEDUP_THRESHOLD,
    force: bool = False,
) -> IngestResult:
    """Extract keyframes from one video into the dataset at ``out_root``."""
    if interval_s <= 0:
        raise IngestError(f"interval_s must be > 0, got {interval_s}")
    if max_frames < 1:
        raise IngestError(f"max_frames must be >= 1, got {max_frames}")

    out_root = Path(out_root)
    info = probe_video(Path(video_path), video_id)

    final_frames_dir = out_root / "keyframes" / info.video_id
    if final_frames_dir.exists() and any(final_frames_dir.iterdir()) and not force:
        raise IngestError(
            f"{info.video_id}: keyframes already exist in {out_root}. "
            f"Re-run with --force to replace them."
        )

    staging_parent = out_root / ".hcmaic-staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f"{info.video_id}-", dir=staging_parent))
    try:
        result = _generate_video_outputs(
            info,
            staging_root,
            interval_s=interval_s,
            max_frames=max_frames,
            dedup_threshold=dedup_threshold,
        )
        from hcmaic.ingestion.validator import validate_dataset

        report = validate_dataset(staging_root, check_images=True)
        if not report.ok:
            messages = "; ".join(issue.message for issue in report.errors[:3])
            raise IngestError(f"{info.video_id}: staged output failed validation: {messages}")
        _commit_staged_video(staging_root, out_root, info.video_id)
        return result
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        with contextlib.suppress(OSError):
            staging_parent.rmdir()


def _generate_video_outputs(
    info: VideoInfo,
    out_root: Path,
    *,
    interval_s: float,
    max_frames: int,
    dedup_threshold: float,
) -> IngestResult:
    """Generate and validate one video's files without touching live data."""
    warnings: list[str] = []
    frames_dir = out_root / "keyframes" / info.video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    if info.backend == "ffmpeg":
        candidates = _iter_candidates_ffmpeg(info, interval_s, max_frames, warnings, out_root)
    else:
        candidates = _iter_candidates_opencv(info, interval_s, max_frames, warnings)

    kept_rows: list[dict[str, Any]] = []
    n_candidates = 0
    n_duplicates = 0
    last_thumb: np.ndarray | None = None
    duration = info.duration_s

    for cand in candidates:
        n_candidates += 1
        pts = cand.pts_time
        if pts < 0:
            warnings.append(
                f"{info.video_id}: dropped frame_idx={cand.frame_idx} with "
                f"negative timestamp {pts:.3f}s."
            )
            continue
        if duration > 0 and pts > duration + interval_s:
            warnings.append(
                f"{info.video_id}: dropped frame_idx={cand.frame_idx} with "
                f"timestamp {pts:.3f}s beyond declared duration {duration:.3f}s."
            )
            continue
        thumb = _gray_thumb(cand.image)
        if last_thumb is not None:
            diff = float(np.abs(thumb - last_thumb).mean())
            if diff < dedup_threshold:
                n_duplicates += 1
                continue
        last_thumb = thumb
        n = len(kept_rows) + 1
        image_path = frames_dir / f"{n:03d}.jpg"
        Image.fromarray(cand.image).save(image_path, quality=_JPEG_QUALITY)
        kept_rows.append(
            {
                "video_id": info.video_id,
                "n": n,
                "pts_time": round(pts, 3),
                "fps": round(info.fps, 6),
                "frame_idx": cand.frame_idx,
                "shot_id": None,
                "frame_id": f"{info.video_id}:{n:03d}",
                "shot_start": None,
                "shot_end": None,
                "width": info.width,
                "height": info.height,
                "timestamp_source": cand.timestamp_source,
                "ingestion_provider": info.backend,
                "sampling_policy": f"uniform-{interval_s:g}s",
            }
        )

    if not kept_rows:
        shutil.rmtree(frames_dir, ignore_errors=True)
        raise IngestError(
            f"{info.video_id}: no keyframes could be extracted "
            f"({n_candidates} candidates, {n_duplicates} duplicates). "
            f"The video may be empty or unreadable."
        )

    _append_mapping_rows(out_root, kept_rows)
    _write_media_info(out_root, info)

    return IngestResult(
        info=info,
        n_candidates=n_candidates,
        n_kept=len(kept_rows),
        n_duplicates=n_duplicates,
        warnings=warnings,
    )


def _commit_staged_video(staging_root: Path, out_root: Path, video_id: str) -> None:
    """Replace one live video only after its staged dataset is valid."""
    staged_frames = staging_root / "keyframes" / video_id
    staged_media = staging_root / MEDIA_INFO_DIR / f"{video_id}.json"
    staged_rows = _read_mapping_rows(staging_root)

    commit_root = staging_root / "_commit"
    backup_root = staging_root / "_backup"
    commit_root.mkdir()
    backup_root.mkdir()

    existing_rows = [row for row in _read_mapping_rows(out_root) if row.get("video_id") != video_id]
    _write_mapping_rows(commit_root, existing_rows + staged_rows)
    candidate_mapping = _mapping_path(commit_root)

    final_frames = out_root / "keyframes" / video_id
    final_mapping = _mapping_path(out_root)
    final_media = out_root / MEDIA_INFO_DIR / f"{video_id}.json"
    backup_frames = backup_root / "frames"
    backup_mapping = backup_root / SINGLE_FILE_NAME
    backup_media = backup_root / f"{video_id}.json"

    final_frames.parent.mkdir(parents=True, exist_ok=True)
    final_media.parent.mkdir(parents=True, exist_ok=True)
    had_frames = final_frames.exists()
    had_mapping = final_mapping.exists()
    had_media = final_media.exists()
    if had_mapping:
        shutil.copy2(final_mapping, backup_mapping)
    if had_media:
        shutil.copy2(final_media, backup_media)

    try:
        if had_frames:
            os.replace(final_frames, backup_frames)
        os.replace(staged_frames, final_frames)
        os.replace(candidate_mapping, final_mapping)
        os.replace(staged_media, final_media)
    except OSError as exc:
        shutil.rmtree(final_frames, ignore_errors=True)
        if backup_frames.exists():
            os.replace(backup_frames, final_frames)
        if backup_mapping.exists():
            os.replace(backup_mapping, final_mapping)
        elif not had_mapping:
            final_mapping.unlink(missing_ok=True)
        if backup_media.exists():
            os.replace(backup_media, final_media)
        elif not had_media:
            final_media.unlink(missing_ok=True)
        raise IngestError(
            f"{video_id}: atomic replacement failed; previous dataset restored: {exc}"
        ) from exc


def _mapping_path(out_root: Path) -> Path:
    return Path(out_root) / SINGLE_FILE_NAME


def _read_mapping_rows(out_root: Path) -> list[dict[str, str]]:
    path = _mapping_path(out_root)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_mapping_rows(out_root: Path, rows: list[dict[str, Any]]) -> None:
    path = _mapping_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MAPPING_COLUMNS})


def _append_mapping_rows(out_root: Path, new_rows: list[dict[str, Any]]) -> None:
    existing = [
        r for r in _read_mapping_rows(out_root) if r.get("video_id") != new_rows[0]["video_id"]
    ]
    _write_mapping_rows(out_root, existing + new_rows)


def _remove_mapping_rows(out_root: Path, video_id: str) -> None:
    existing = _read_mapping_rows(out_root)
    kept = [r for r in existing if r.get("video_id") != video_id]
    if len(kept) != len(existing):
        _write_mapping_rows(out_root, kept)


def _write_media_info(out_root: Path, info: VideoInfo) -> None:
    media_dir = Path(out_root) / MEDIA_INFO_DIR
    media_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": info.source_path.stem,
        "length": round(info.duration_s, 3),
        "width": info.width,
        "height": info.height,
        "fps": round(info.fps, 6),
        "ingest_backend": info.backend,
        "source_file": info.source_path.name,  # name only, never a local path
    }
    with open(media_dir / f"{info.video_id}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def collect_video_files(input_path: Path) -> list[Path]:
    """A single file, or every supported video directly inside a directory."""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(
            p
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise IngestError(
                f"No supported video files ({sorted(SUPPORTED_EXTENSIONS)}) found in {input_path}."
            )
        return files
    raise IngestError(f"Input path does not exist: {input_path}")


def ingest_dataset(
    input_path: Path,
    out_root: Path,
    video_id: str | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_frames: int = DEFAULT_MAX_FRAMES,
    force: bool = False,
) -> tuple[list[IngestResult], list[dict[str, str]]]:
    """Ingest one file or a directory of videos; returns (results, failures).

    Failures do not abort the batch; each failure records video file + reason.
    Writes ``ingest_report.json`` into the dataset root.
    """
    files = collect_video_files(input_path)
    if video_id is not None and len(files) > 1:
        raise IngestError(
            "--video-id can only be used when ingesting a single file, "
            f"but {len(files)} videos were found."
        )
    results: list[IngestResult] = []
    failures: list[dict[str, str]] = []
    for path in files:
        try:
            results.append(
                ingest_video(
                    path,
                    out_root,
                    video_id=video_id,
                    interval_s=interval_s,
                    max_frames=max_frames,
                    force=force,
                )
            )
        except IngestError as exc:
            failures.append({"file": path.name, "error": str(exc)})

    report = {
        "n_videos_ok": len(results),
        "n_videos_failed": len(failures),
        "interval_s": interval_s,
        "max_frames": max_frames,
        "videos": [
            {
                "video_id": r.info.video_id,
                "backend": r.info.backend,
                "duration_s": round(r.info.duration_s, 3),
                "n_candidates": r.n_candidates,
                "n_kept": r.n_kept,
                "n_duplicates": r.n_duplicates,
                "warnings": r.warnings,
            }
            for r in results
        ],
        "failures": failures,
    }
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "ingest_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return results, failures
