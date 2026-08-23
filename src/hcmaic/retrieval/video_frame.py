"""Bounded raw-video frame extraction for the local review inspector.

The decoder returns timestamps reported by the video backend.  The API can
join those timestamps to the canonical PTS timeline, or use the bounded exact
fallback below when that timeline is not available.  Neither path derives
``source_frame_idx`` from FPS.
"""

from __future__ import annotations

import base64
import math
import urllib.parse
from pathlib import Path
from typing import Any

MAX_DECODE_SEEK_FRAMES = 2_000
MAX_EXACT_DECODE_FRAMES = 200_000
MAX_REMOTE_SEEK_FRAMES = 512
DEFAULT_THUMBNAIL_WIDTH = 480
DEFAULT_JPEG_QUALITY = 72


class VideoFrameDecodeError(RuntimeError):
    """Raised when one raw-video frame cannot be decoded safely."""


def _capture_timestamp_ms(capture: Any, cv2: Any) -> int | None:
    value = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if not math.isfinite(value) or value < 0:
        return None
    return int(round(value))


def _capture_source_frame_idx(capture: Any, cv2: Any) -> int | None:
    """Read the zero-based presentation position reported after ``read()``."""

    value = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
    if not math.isfinite(value) or value < 1:
        return None
    rounded = round(value)
    if abs(value - rounded) > 0.01:
        return None
    # OpenCV reports the next frame position after a successful read on the
    # FFmpeg backend, so subtract one to identify the decoded frame itself.
    return int(rounded) - 1


def _encode_thumbnail(frame: Any, cv2: Any, *, thumbnail_width: int, jpeg_quality: int) -> str:
    height, width = frame.shape[:2]
    if width > thumbnail_width:
        target_height = max(1, round(height * thumbnail_width / width))
        frame = cv2.resize(frame, (thumbnail_width, target_height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise VideoFrameDecodeError("VIDEO_THUMBNAIL_ENCODE_FAILED")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def decode_video_frame(
    video_path: Path,
    target_timestamp_ms: int,
    *,
    thumbnail_width: int = DEFAULT_THUMBNAIL_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, Any]:
    """Decode one frame nearest to an explicit timestamp.

    The seek is performed once, then frames are read forward only until the
    first decoded timestamp at or after the target.  The neighboring decoded
    frame is retained so the nearest of the two can be returned without a
    second seek.  A bounded seek budget prevents a malformed stream from
    turning one UI click into an unbounded decode.
    """

    if not video_path.is_file():
        raise VideoFrameDecodeError("VIDEO_FILE_NOT_FOUND")
    if target_timestamp_ms < 0:
        raise ValueError("target_timestamp_ms must be non-negative")
    if thumbnail_width < 160 or thumbnail_width > 640:
        raise ValueError("thumbnail_width must be in [160, 640]")
    if jpeg_quality < 40 or jpeg_quality > 95:
        raise ValueError("jpeg_quality must be in [40, 95]")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional video extra
        raise VideoFrameDecodeError("VIDEO_DECODER_UNAVAILABLE") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoFrameDecodeError("VIDEO_DECODER_OPEN_FAILED")

    previous: tuple[int, Any] | None = None
    current: tuple[int, Any] | None = None
    try:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(target_timestamp_ms))
        for _ in range(MAX_DECODE_SEEK_FRAMES):
            ok, frame = capture.read()
            if not ok:
                break
            decoded_timestamp_ms = _capture_timestamp_ms(capture, cv2)
            if decoded_timestamp_ms is None:
                raise VideoFrameDecodeError("VIDEO_DECODE_TIMESTAMP_UNAVAILABLE")
            current = (decoded_timestamp_ms, frame.copy())
            if previous is not None and decoded_timestamp_ms < previous[0]:
                raise VideoFrameDecodeError("VIDEO_DECODE_TIMESTAMP_NON_MONOTONIC")
            if decoded_timestamp_ms >= target_timestamp_ms:
                break
            previous = current
    finally:
        capture.release()

    if current is None:
        raise VideoFrameDecodeError("VIDEO_DECODE_NO_FRAME")
    chosen = current
    if previous is not None and abs(previous[0] - target_timestamp_ms) < abs(
        current[0] - target_timestamp_ms
    ):
        chosen = previous
    return {
        "requested_timestamp_ms": int(target_timestamp_ms),
        "decoded_timestamp_ms": int(chosen[0]),
        "delta_ms": abs(int(chosen[0]) - int(target_timestamp_ms)),
        "image_data_url": _encode_thumbnail(
            chosen[1],
            cv2,
            thumbnail_width=thumbnail_width,
            jpeg_quality=jpeg_quality,
        ),
    }


def decode_exact_video_frame(
    video_path: Path,
    target_timestamp_ms: int,
    *,
    thumbnail_width: int = DEFAULT_THUMBNAIL_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, Any]:
    """Decode one frame and retain its presentation-order source index.

    This is the fail-closed fallback for videos without a materialized
    canonical PTS index.  Frames are read in presentation order from the
    beginning of the local/verified video; ``source_frame_idx`` is the count
    of successfully decoded frames starting at zero.  The target and decoded
    timestamps both come from the decoder, so no FPS-derived index is used.

    ``target_reached`` is false when the stream ends before the requested
    timestamp.  Callers must not promote that result as an exact extraction.
    """

    if not video_path.is_file():
        raise VideoFrameDecodeError("VIDEO_FILE_NOT_FOUND")
    if target_timestamp_ms < 0:
        raise ValueError("target_timestamp_ms must be non-negative")
    if thumbnail_width < 160 or thumbnail_width > 640:
        raise ValueError("thumbnail_width must be in [160, 640]")
    if jpeg_quality < 40 or jpeg_quality > 95:
        raise ValueError("jpeg_quality must be in [40, 95]")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional video extra
        raise VideoFrameDecodeError("VIDEO_DECODER_UNAVAILABLE") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoFrameDecodeError("VIDEO_DECODER_OPEN_FAILED")

    previous: tuple[int, int, Any] | None = None
    current: tuple[int, int, Any] | None = None
    target_reached = False
    try:
        for source_frame_idx in range(MAX_EXACT_DECODE_FRAMES):
            ok, frame = capture.read()
            if not ok:
                break
            decoded_timestamp_ms = _capture_timestamp_ms(capture, cv2)
            if decoded_timestamp_ms is None:
                raise VideoFrameDecodeError("VIDEO_DECODE_TIMESTAMP_UNAVAILABLE")
            if previous is not None and decoded_timestamp_ms < previous[0]:
                raise VideoFrameDecodeError("VIDEO_DECODE_TIMESTAMP_NON_MONOTONIC")
            current = (decoded_timestamp_ms, source_frame_idx, frame.copy())
            if decoded_timestamp_ms >= target_timestamp_ms:
                target_reached = True
                break
            previous = current
    finally:
        capture.release()

    if current is None:
        raise VideoFrameDecodeError("VIDEO_DECODE_NO_FRAME")
    chosen = current
    if previous is not None and abs(previous[0] - target_timestamp_ms) < abs(
        current[0] - target_timestamp_ms
    ):
        chosen = previous
    return {
        "requested_timestamp_ms": int(target_timestamp_ms),
        "decoded_timestamp_ms": int(chosen[0]),
        "delta_ms": abs(int(chosen[0]) - int(target_timestamp_ms)),
        "source_frame_idx": int(chosen[1]),
        "target_reached": target_reached,
        "image_data_url": _encode_thumbnail(
            chosen[2],
            cv2,
            thumbnail_width=thumbnail_width,
            jpeg_quality=jpeg_quality,
        ),
    }


def decode_exact_video_frame_url(
    video_url: str,
    target_timestamp_ms: int,
    *,
    thumbnail_width: int = DEFAULT_THUMBNAIL_WIDTH,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
) -> dict[str, Any]:
    """Decode one exact frame directly from an allowlisted HTTPS media URL.

    The URL must already have come from the immutable media manifest.  FFmpeg
    performs the network seek; the returned source index is the decoder's
    presentation position after ``read()`` rather than an FPS estimate.  A
    small forward-read budget handles a seek landing just before the target.
    """

    parsed = urllib.parse.urlsplit(str(video_url))
    if parsed.scheme != "https" or not parsed.hostname:
        raise VideoFrameDecodeError("VIDEO_REMOTE_URL_INVALID")
    if target_timestamp_ms < 0:
        raise ValueError("target_timestamp_ms must be non-negative")
    if thumbnail_width < 160 or thumbnail_width > 640:
        raise ValueError("thumbnail_width must be in [160, 640]")
    if jpeg_quality < 40 or jpeg_quality > 95:
        raise ValueError("jpeg_quality must be in [40, 95]")

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional video extra
        raise VideoFrameDecodeError("VIDEO_DECODER_UNAVAILABLE") from exc

    capture = cv2.VideoCapture(str(video_url))
    if not capture.isOpened():
        capture.release()
        raise VideoFrameDecodeError("VIDEO_REMOTE_DECODER_OPEN_FAILED")
    try:
        try:
            decoder_backend = str(capture.getBackendName() or "UNKNOWN").upper()
        except AttributeError:  # pragma: no cover - old optional OpenCV builds
            decoder_backend = "UNKNOWN"
        if decoder_backend not in {"FFMPEG", "UNKNOWN"}:
            raise VideoFrameDecodeError("VIDEO_REMOTE_FFMPEG_BACKEND_REQUIRED")
        if not capture.set(cv2.CAP_PROP_POS_MSEC, float(target_timestamp_ms)):
            raise VideoFrameDecodeError("VIDEO_REMOTE_SEEK_FAILED")

        previous: tuple[int, int, Any] | None = None
        current: tuple[int, int, Any] | None = None
        target_reached = False
        for _ in range(MAX_REMOTE_SEEK_FRAMES):
            ok, frame = capture.read()
            if not ok:
                break
            decoded_timestamp_ms = _capture_timestamp_ms(capture, cv2)
            source_frame_idx = _capture_source_frame_idx(capture, cv2)
            if decoded_timestamp_ms is None:
                raise VideoFrameDecodeError("VIDEO_REMOTE_TIMESTAMP_UNAVAILABLE")
            if source_frame_idx is None:
                raise VideoFrameDecodeError("VIDEO_REMOTE_SOURCE_INDEX_UNAVAILABLE")
            if previous is not None and (
                decoded_timestamp_ms < previous[0] or source_frame_idx < previous[1]
            ):
                raise VideoFrameDecodeError("VIDEO_REMOTE_POSITION_NON_MONOTONIC")
            current = (decoded_timestamp_ms, source_frame_idx, frame.copy())
            if decoded_timestamp_ms >= target_timestamp_ms:
                target_reached = True
                break
            previous = current
    finally:
        capture.release()

    if current is None:
        raise VideoFrameDecodeError("VIDEO_REMOTE_DECODE_NO_FRAME")
    chosen = current
    if previous is not None and abs(previous[0] - target_timestamp_ms) < abs(
        current[0] - target_timestamp_ms
    ):
        chosen = previous
    return {
        "requested_timestamp_ms": int(target_timestamp_ms),
        "decoded_timestamp_ms": int(chosen[0]),
        "delta_ms": abs(int(chosen[0]) - int(target_timestamp_ms)),
        "source_frame_idx": int(chosen[1]),
        "target_reached": target_reached,
        "decoder_backend": decoder_backend,
        "extraction_mode": "REMOTE_URL_FFMPEG",
        "image_data_url": _encode_thumbnail(
            chosen[2],
            cv2,
            thumbnail_width=thumbnail_width,
            jpeg_quality=jpeg_quality,
        ),
    }
