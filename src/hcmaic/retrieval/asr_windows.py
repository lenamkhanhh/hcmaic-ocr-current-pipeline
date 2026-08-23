"""Manifest-first ASR windows mapped to the canonical keyframe identity.

This module is the local P0-P4 contract for the raw-video ASR channel.  It is
deliberately independent from the legacy ``asr.jsonl`` artifact: a transcript
window can support many keyframes, and a keyframe can be supported by several
windows.  The only identity exported by this module is ``frame_uid`` (the
``video_id:source_frame_idx`` pair); FAISS row numbers are never accepted as
identity.

The implementation is dependency-light so it can be used to validate bundles
before a Kaggle run and in the local KIS runtime.  BGE-M3/Elasticsearch remain
replaceable index backends over the same canonical window/link artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hcmaic.retrieval.asr import ASRArtifactError
from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.channel_contract import ChannelContract, build_channel_evidence
from hcmaic.retrieval.ocr_bm25 import normalize_ocr_text

ASR_WINDOW_RECORDS_NAME = "asr_windows.jsonl"
ASR_FRAME_LINKS_NAME = "asr_frame_links.jsonl"
ASR_WINDOW_MANIFEST_NAME = "asr_window_manifest.json"
ASR_WINDOW_FORMAT = "hcmaic-asr-window-v2"


class ASRWindowArtifactError(ASRArtifactError):
    """Raised when an ASR-window bundle is unsafe or internally inconsistent."""


def _is_mock_provider(provider: str) -> bool:
    return "mock" in provider.casefold()


def stable_frame_uid(video_id: str, source_frame_idx: int) -> str:
    """Return the stable source identity used across every retrieval channel."""
    if not video_id.strip():
        raise ASRWindowArtifactError("video_id must not be blank")
    if source_frame_idx < 0:
        raise ASRWindowArtifactError("source_frame_idx must be non-negative")
    return f"{video_id}:{source_frame_idx}"


def _assert_stable_frame_uid(frame_uid: str, video_id: str, source_frame_idx: int) -> None:
    expected = stable_frame_uid(video_id, source_frame_idx)
    try:
        prefix, suffix = frame_uid.rsplit(":", 1)
        parsed_idx = int(suffix)
    except (AttributeError, ValueError) as exc:
        raise ASRWindowArtifactError(
            "frame_uid must use the stable video_id:source_frame_idx format"
        ) from exc
    if prefix != video_id or parsed_idx != source_frame_idx:
        raise ASRWindowArtifactError(
            f"frame_uid {frame_uid!r} does not match stable identity {expected!r}"
        )


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ASRWindowArtifactError(f"{field_name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ASRWindowArtifactError(f"{field_name} must be an integer") from exc
    if result < 0:
        raise ASRWindowArtifactError(f"{field_name} must be non-negative")
    return result


def _time_ms(value: Any, field_name: str, *, seconds: bool = False) -> int:
    if isinstance(value, bool):
        raise ASRWindowArtifactError(f"{field_name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ASRWindowArtifactError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ASRWindowArtifactError(f"{field_name} must be finite and non-negative")
    return int(round(parsed * 1000.0)) if seconds else int(round(parsed))


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for row in rows
    ]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _provider_fields(provider: str, revision: str) -> tuple[str, str]:
    if not provider.strip() or _is_mock_provider(provider):
        raise ASRWindowArtifactError("ASR provider must be a real non-mock provider")
    if not revision.strip():
        raise ASRWindowArtifactError("ASR revision must not be blank")
    return provider, revision


@dataclass(frozen=True)
class ASRWindowRecord:
    """One bounded transcript window with raw-video provenance."""

    window_uid: str
    video_id: str
    video_filename: str
    start_ms: int
    end_ms: int
    text: str
    provider: str
    revision: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.window_uid.strip() or not self.video_id.strip():
            raise ASRWindowArtifactError("window_uid and video_id must not be blank")
        if not self.video_filename.strip():
            raise ASRWindowArtifactError("video_filename must not be blank")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ASRWindowArtifactError("ASR window bounds are invalid")
        if not self.text.strip():
            raise ASRWindowArtifactError("ASR window text must not be blank")
        _provider_fields(self.provider, self.revision)
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ASRWindowArtifactError("ASR window confidence must be in [0, 1]")
        if not isinstance(self.metadata, Mapping):
            raise ASRWindowArtifactError("ASR window metadata must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_uid": self.window_uid,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "provider": self.provider,
            "revision": self.revision,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ASRFrameLink:
    """Many-to-many link between one ASR window and one canonical keyframe."""

    window_uid: str
    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int
    is_anchor: bool
    distance_to_midpoint_ms: int
    mapping_quality: str

    def __post_init__(self) -> None:
        if not self.window_uid.strip() or not self.frame_uid.strip():
            raise ASRWindowArtifactError("window_uid and frame_uid must not be blank")
        if not self.video_id.strip() or not self.video_filename.strip():
            raise ASRWindowArtifactError("frame link video identity must not be blank")
        if self.source_frame_idx < 0 or self.timestamp_ms < 0:
            raise ASRWindowArtifactError("frame link frame/timestamp must be non-negative")
        if self.distance_to_midpoint_ms < 0:
            raise ASRWindowArtifactError("distance_to_midpoint_ms must be non-negative")
        if not self.mapping_quality.strip():
            raise ASRWindowArtifactError("mapping_quality must not be blank")
        _assert_stable_frame_uid(self.frame_uid, self.video_id, self.source_frame_idx)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_uid": self.window_uid,
            "frame_uid": self.frame_uid,
            "video_id": self.video_id,
            "video_filename": self.video_filename,
            "source_frame_idx": self.source_frame_idx,
            "timestamp_ms": self.timestamp_ms,
            "is_anchor": self.is_anchor,
            "distance_to_midpoint_ms": self.distance_to_midpoint_ms,
            "mapping_quality": self.mapping_quality,
        }


@dataclass(frozen=True)
class ASRWindowArtifact:
    artifact_dir: Path
    windows: tuple[ASRWindowRecord, ...]
    links: tuple[ASRFrameLink, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _FrameRef:
    frame_uid: str
    video_id: str
    video_filename: str
    source_frame_idx: int
    timestamp_ms: int


def _coerce_frame(row: Mapping[str, Any]) -> _FrameRef:
    if "faiss_row" in row or "faiss_id" in row:
        raise ASRWindowArtifactError(
            "faiss_row/faiss_id cannot be used as frame identity; provide frame_uid"
        )
    try:
        frame_uid = str(row["frame_uid"])
        video_id = str(row["video_id"])
        video_filename = str(row["video_filename"])
        source_frame_idx = _as_int(row["source_frame_idx"], "source_frame_idx")
    except KeyError as exc:
        raise ASRWindowArtifactError(
            "keyframe rows require frame_uid, video_id, video_filename and source_frame_idx"
        ) from exc
    if "timestamp_ms" in row:
        timestamp_ms = _time_ms(row["timestamp_ms"], "timestamp_ms")
    elif "timestamp_s" in row:
        timestamp_ms = _time_ms(row["timestamp_s"], "timestamp_s", seconds=True)
    else:
        raise ASRWindowArtifactError("keyframe rows require timestamp_ms or timestamp_s")
    _assert_stable_frame_uid(frame_uid, video_id, source_frame_idx)
    return _FrameRef(
        frame_uid,
        video_id,
        video_filename,
        source_frame_idx,
        timestamp_ms,
    )


def _coerce_words(segment: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_words = segment.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        raise ASRWindowArtifactError(
            "P0 requires word-level timing for ASR segments longer than max_window_ms"
        )
    words: list[dict[str, Any]] = []
    for word in raw_words:
        if not isinstance(word, Mapping):
            raise ASRWindowArtifactError("word-level timing entries must be objects")
        if "start_ms" in word and "end_ms" in word:
            start_ms = _time_ms(word["start_ms"], "word.start_ms")
            end_ms = _time_ms(word["end_ms"], "word.end_ms")
        elif "start_s" in word and "end_s" in word:
            start_ms = _time_ms(word["start_s"], "word.start_s", seconds=True)
            end_ms = _time_ms(word["end_s"], "word.end_s", seconds=True)
        else:
            raise ASRWindowArtifactError(
                "P0 requires word-level timing entries with start_ms/end_ms"
            )
        text = str(word.get("text", "")).strip()
        if end_ms < start_ms or not text:
            raise ASRWindowArtifactError("word-level timing entry is invalid")
        words.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
    words.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["text"]))
    return words


def _segment_time(segment: Mapping[str, Any], name: str) -> int:
    if f"{name}_ms" in segment:
        return _time_ms(segment[f"{name}_ms"], f"segment.{name}_ms")
    if f"{name}_s" in segment:
        return _time_ms(segment[f"{name}_s"], f"segment.{name}_s", seconds=True)
    raise ASRWindowArtifactError(
        f"ASR segments require {name}_ms or {name}_s; legacy timestamp fields are not accepted"
    )


def _make_window(
    segment: Mapping[str, Any],
    *,
    segment_id: str,
    index: int,
    start_ms: int,
    end_ms: int,
    text: str,
    timing_status: str,
) -> ASRWindowRecord:
    provider, revision = _provider_fields(
        str(segment.get("provider", "")), str(segment.get("revision", ""))
    )
    confidence = segment.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
    metadata = dict(segment.get("metadata") or {})
    metadata.update(
        {
            "source_segment_ids": [segment_id],
            "timing_status": timing_status,
        }
    )
    video_id = str(segment.get("video_id", ""))
    return ASRWindowRecord(
        window_uid=f"{video_id}:{segment_id}:w{index:04d}",
        video_id=video_id,
        video_filename=str(segment.get("video_filename", "")),
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        provider=provider,
        revision=revision,
        confidence=confidence,
        metadata=metadata,
    )


def build_asr_windows(
    segments: Iterable[Mapping[str, Any]],
    *,
    target_window_ms: int = 30_000,
    max_window_ms: int = 40_000,
    overlap_ms: int = 5_000,
) -> tuple[ASRWindowRecord, ...]:
    """Build bounded windows without inventing timing for long ASR segments.

    Segments no longer than ``max_window_ms`` retain their source bounds.  A
    longer segment is split only from real word-level timestamps; a segment
    with synthetic/proportional timing fails closed so it is visible as the P0
    repair blocker rather than silently becoming false coverage.
    """
    if target_window_ms < 1 or max_window_ms < target_window_ms:
        raise ValueError("max_window_ms must be >= target_window_ms >= 1")
    if overlap_ms < 0 or overlap_ms >= target_window_ms:
        raise ValueError("overlap_ms must be in [0, target_window_ms)")
    output: list[ASRWindowRecord] = []
    seen_segment_ids: set[str] = set()
    for segment in segments:
        segment_id = str(segment.get("segment_id", "")).strip()
        if not segment_id:
            raise ASRWindowArtifactError("ASR segment_id must not be blank")
        if segment_id in seen_segment_ids:
            raise ASRWindowArtifactError(f"duplicate segment_id {segment_id!r}")
        seen_segment_ids.add(segment_id)
        start_ms = _segment_time(segment, "start")
        end_ms = _segment_time(segment, "end")
        if end_ms < start_ms:
            raise ASRWindowArtifactError("segment end_ms must be >= start_ms")
        text = str(segment.get("text", "")).strip()
        if not text:
            raise ASRWindowArtifactError("ASR segment text must not be blank")
        duration = end_ms - start_ms
        if duration <= max_window_ms:
            output.append(
                _make_window(
                    segment,
                    segment_id=segment_id,
                    index=0,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    timing_status="segment-bounds",
                )
            )
            continue

        words = _coerce_words(segment)
        if words[0]["start_ms"] < start_ms or words[-1]["end_ms"] > end_ms:
            raise ASRWindowArtifactError(
                f"word timing for segment {segment_id!r} falls outside segment bounds"
            )
        cursor = start_ms
        window_index = 0
        while cursor < end_ms:
            requested_end = min(cursor + target_window_ms, end_ms)
            selected = [
                word
                for word in words
                if word["end_ms"] > cursor and word["start_ms"] < requested_end
            ]
            if not selected:
                future = [word for word in words if word["start_ms"] >= cursor]
                if not future:
                    break
                cursor = future[0]["start_ms"]
                continue
            actual_start = selected[0]["start_ms"]
            actual_end = selected[-1]["end_ms"]
            if actual_end - actual_start > max_window_ms:
                raise ASRWindowArtifactError(
                    f"word timing cannot form a bounded window for segment {segment_id!r}"
                )
            output.append(
                _make_window(
                    segment,
                    segment_id=segment_id,
                    index=window_index,
                    start_ms=actual_start,
                    end_ms=actual_end,
                    text=" ".join(word["text"] for word in selected),
                    timing_status="word-aligned",
                )
            )
            window_index += 1
            if requested_end >= end_ms:
                break
            next_cursor = requested_end - overlap_ms
            if next_cursor <= cursor:
                next_cursor = cursor + 1
            cursor = next_cursor
    output.sort(key=lambda item: (item.video_id, item.start_ms, item.window_uid))
    if len({item.window_uid for item in output}) != len(output):
        raise ASRWindowArtifactError("generated ASR window_uid values are not unique")
    return tuple(output)


def map_windows_to_keyframes(
    windows: Iterable[ASRWindowRecord],
    keyframes: Iterable[Mapping[str, Any]],
    *,
    padding_ms: int = 1_500,
    good_tolerance_ms: int = 3_000,
    acceptable_tolerance_ms: int = 5_500,
) -> tuple[ASRFrameLink, ...]:
    """Map each transcript window to all interval-overlapping frames.

    Empty intervals fall back to the nearest frame and are labelled rather
    than discarded.  A gap above ``acceptable_tolerance_ms`` is retained as a
    ``COVERAGE_WARNING`` link, allowing the manifest to quantify the problem.
    A video with no keyframes has no link and is counted by the manifest gate.
    """
    if padding_ms < 0 or good_tolerance_ms < 0 or acceptable_tolerance_ms < good_tolerance_ms:
        raise ValueError("mapping tolerances must be non-negative and ordered")
    frame_by_video: dict[str, list[_FrameRef]] = defaultdict(list)
    seen_frame_uids: set[str] = set()
    for row in keyframes:
        frame = _coerce_frame(row)
        if frame.frame_uid in seen_frame_uids:
            raise ASRWindowArtifactError(f"duplicate keyframe frame_uid {frame.frame_uid!r}")
        seen_frame_uids.add(frame.frame_uid)
        frame_by_video[frame.video_id].append(frame)
    for frames in frame_by_video.values():
        frames.sort(key=lambda frame: (frame.timestamp_ms, frame.source_frame_idx))

    links: list[ASRFrameLink] = []
    for window in windows:
        frames = frame_by_video.get(window.video_id, [])
        if not frames:
            continue
        midpoint = (window.start_ms + window.end_ms) // 2
        selected = [
            frame
            for frame in frames
            if window.start_ms - padding_ms <= frame.timestamp_ms <= window.end_ms + padding_ms
        ]
        if selected:
            quality = "INTERVAL_OVERLAP"
        else:
            nearest = min(
                frames,
                key=lambda frame: (
                    abs(frame.timestamp_ms - midpoint),
                    frame.timestamp_ms,
                    frame.source_frame_idx,
                ),
            )
            selected = [nearest]
            distance = abs(nearest.timestamp_ms - midpoint)
            if distance <= good_tolerance_ms:
                quality = "NEAREST_GOOD"
            elif distance <= acceptable_tolerance_ms:
                quality = "NEAREST_ACCEPTABLE"
            else:
                quality = "COVERAGE_WARNING"
        anchor = min(
            selected,
            key=lambda frame: (
                abs(frame.timestamp_ms - midpoint),
                frame.timestamp_ms,
                frame.source_frame_idx,
            ),
        )
        for frame in selected:
            links.append(
                ASRFrameLink(
                    window_uid=window.window_uid,
                    frame_uid=frame.frame_uid,
                    video_id=frame.video_id,
                    video_filename=frame.video_filename,
                    source_frame_idx=frame.source_frame_idx,
                    timestamp_ms=frame.timestamp_ms,
                    is_anchor=frame.frame_uid == anchor.frame_uid,
                    distance_to_midpoint_ms=abs(frame.timestamp_ms - midpoint),
                    mapping_quality=quality,
                )
            )
    links.sort(
        key=lambda item: (
            item.window_uid,
            not item.is_anchor,
            item.timestamp_ms,
            item.frame_uid,
        )
    )
    unique: list[ASRFrameLink] = []
    seen_pairs: set[tuple[str, str]] = set()
    for link in links:
        pair = (link.window_uid, link.frame_uid)
        if pair not in seen_pairs:
            unique.append(link)
            seen_pairs.add(pair)
    return tuple(unique)


def _parse_jsonl(payload: bytes, row_type: type[Any], label: str) -> list[Any]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ASRWindowArtifactError(f"{label} is not valid UTF-8") from exc
    rows: list[Any] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("row is not an object")
            rows.append(row_type(**data))
        except (TypeError, ValueError, json.JSONDecodeError, ASRWindowArtifactError) as exc:
            raise ASRWindowArtifactError(f"invalid {label} row at line {line_number}") from exc
    return rows


def _validate_common_rows(
    windows: list[ASRWindowRecord], links: list[ASRFrameLink]
) -> tuple[str, str]:
    if not windows:
        raise ASRWindowArtifactError("ASR window artifact cannot be empty")
    providers = {window.provider for window in windows}
    revisions = {window.revision for window in windows}
    if len(providers) != 1 or len(revisions) != 1:
        raise ASRWindowArtifactError("ASR window artifact must contain one provider and revision")
    window_uids = [window.window_uid for window in windows]
    if len(window_uids) != len(set(window_uids)):
        raise ASRWindowArtifactError("ASR window artifact has duplicate window_uid values")
    link_pairs = [(link.window_uid, link.frame_uid) for link in links]
    if len(link_pairs) != len(set(link_pairs)):
        raise ASRWindowArtifactError("ASR window artifact has duplicate frame links")
    windows_by_uid = {window.window_uid: window for window in windows}
    for link in links:
        window = windows_by_uid.get(link.window_uid)
        if window is None:
            raise ASRWindowArtifactError(
                f"frame link references unknown window_uid {link.window_uid!r}"
            )
        if link.video_id != window.video_id or link.video_filename != window.video_filename:
            raise ASRWindowArtifactError("frame link video identity does not match its window")
    return next(iter(providers)), next(iter(revisions))


def write_asr_window_artifact(
    windows: Iterable[ASRWindowRecord],
    links: Iterable[ASRFrameLink],
    artifact_dir: Path,
    *,
    dataset_manifest_hash: str,
    keyframe_manifest_hash: str,
    mapping_config: Mapping[str, Any] | None = None,
    window_config: Mapping[str, Any] | None = None,
    quality_status: str = "UNVALIDATED_ON_HCMAIC",
) -> ASRWindowArtifact:
    """Write canonical windows, links and a complete engineering manifest."""
    if not dataset_manifest_hash.strip() or not keyframe_manifest_hash.strip():
        raise ASRWindowArtifactError("dataset/keyframe manifest hashes must not be blank")
    if quality_status not in {"UNVALIDATED_ON_HCMAIC", "VALIDATED_ON_HCMAIC"}:
        raise ASRWindowArtifactError("unsupported quality_status")
    windows_list = sorted(
        tuple(windows), key=lambda item: (item.video_id, item.start_ms, item.window_uid)
    )
    links_list = sorted(
        tuple(links),
        key=lambda item: (item.window_uid, not item.is_anchor, item.timestamp_ms, item.frame_uid),
    )
    provider, revision = _validate_common_rows(windows_list, links_list)
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise ASRWindowArtifactError(
            f"Artifact directory {artifact_dir} is not empty; use a new versioned path"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    windows_payload = _jsonl_bytes(window.to_dict() for window in windows_list)
    links_payload = _jsonl_bytes(link.to_dict() for link in links_list)
    (artifact_dir / ASR_WINDOW_RECORDS_NAME).write_bytes(windows_payload)
    (artifact_dir / ASR_FRAME_LINKS_NAME).write_bytes(links_payload)
    quality_counts = Counter(link.mapping_quality for link in links_list)
    linked_windows = {link.window_uid for link in links_list}
    manifest = {
        "format": ASR_WINDOW_FORMAT,
        "windows": ASR_WINDOW_RECORDS_NAME,
        "frame_links": ASR_FRAME_LINKS_NAME,
        "windows_sha256": _sha256(windows_payload),
        "frame_links_sha256": _sha256(links_payload),
        "n_windows": len(windows_list),
        "n_frame_links": len(links_list),
        "n_windows_with_links": len(linked_windows),
        "n_windows_without_links": len(windows_list) - len(linked_windows),
        "n_videos": len({window.video_id for window in windows_list}),
        "provider": provider,
        "revision": revision,
        "dataset_manifest_hash": dataset_manifest_hash,
        "keyframe_manifest_hash": keyframe_manifest_hash,
        "identity_field": "frame_uid",
        "forbidden_identity_fields": ["faiss_row", "faiss_id"],
        "mapping_quality_counts": dict(sorted(quality_counts.items())),
        "mapping_config": dict(mapping_config or {}),
        "mapping_config_declared": mapping_config is not None,
        "window_config": dict(window_config or {}),
        "window_config_declared": window_config is not None,
        "raw_video_source": True,
        "btc_artifacts_used": False,
        "artifact_status": "ENGINEERING_ARTIFACT_COMPLETE",
        "evidence_level": "REAL_PROVIDER_ARTIFACT",
        "quality_status": quality_status,
        "quality_claim": "UNVALIDATED_UNTIL_HCMAIC_QRELS",
    }
    _write_json(artifact_dir / ASR_WINDOW_MANIFEST_NAME, manifest)
    return load_asr_window_artifact(
        artifact_dir,
        dataset_manifest_hash=dataset_manifest_hash,
        keyframe_manifest_hash=keyframe_manifest_hash,
    )


def load_asr_window_artifact(
    artifact_dir: Path,
    *,
    dataset_manifest_hash: str | None = None,
    keyframe_manifest_hash: str | None = None,
) -> ASRWindowArtifact:
    """Load a window/link bundle only after hashes and identity checks pass."""
    artifact_dir = Path(artifact_dir).resolve()
    manifest_path = artifact_dir / ASR_WINDOW_MANIFEST_NAME
    windows_path = artifact_dir / ASR_WINDOW_RECORDS_NAME
    links_path = artifact_dir / ASR_FRAME_LINKS_NAME
    if not manifest_path.is_file() or not windows_path.is_file() or not links_path.is_file():
        raise ASRWindowArtifactError(f"ASR window artifact is unavailable in {artifact_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        windows_payload = windows_path.read_bytes()
        links_payload = links_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ASRWindowArtifactError(f"cannot read ASR window artifact in {artifact_dir}") from exc
    if not isinstance(manifest, dict):
        raise ASRWindowArtifactError("ASR window manifest must be a JSON object")
    if manifest.get("format") != ASR_WINDOW_FORMAT:
        raise ASRWindowArtifactError("unsupported ASR window artifact format")
    if manifest.get("identity_field") != "frame_uid":
        raise ASRWindowArtifactError("ASR window artifact must use frame_uid identity")
    if (
        manifest.get("raw_video_source") is not True
        or manifest.get("btc_artifacts_used") is not False
    ):
        raise ASRWindowArtifactError("ASR window artifact provenance is unsafe")
    if manifest.get("artifact_status") != "ENGINEERING_ARTIFACT_COMPLETE":
        raise ASRWindowArtifactError("ASR window artifact is not complete")
    if (
        dataset_manifest_hash is not None
        and manifest.get("dataset_manifest_hash") != dataset_manifest_hash
    ):
        raise ASRWindowArtifactError("ASR window dataset manifest hash mismatch")
    if (
        keyframe_manifest_hash is not None
        and manifest.get("keyframe_manifest_hash") != keyframe_manifest_hash
    ):
        raise ASRWindowArtifactError("ASR window keyframe manifest hash mismatch")
    if manifest.get("windows_sha256") != _sha256(windows_payload):
        raise ASRWindowArtifactError("ASR windows hash mismatch")
    if manifest.get("frame_links_sha256") != _sha256(links_payload):
        raise ASRWindowArtifactError("ASR frame links hash mismatch")
    windows = _parse_jsonl(windows_payload, ASRWindowRecord, "ASR window")
    links = _parse_jsonl(links_payload, ASRFrameLink, "ASR frame link")
    provider, revision = _validate_common_rows(windows, links)
    if len(windows) != int(manifest.get("n_windows", -1)):
        raise ASRWindowArtifactError("ASR manifest window count mismatch")
    if len(links) != int(manifest.get("n_frame_links", -1)):
        raise ASRWindowArtifactError("ASR manifest frame-link count mismatch")
    if provider != manifest.get("provider") or revision != manifest.get("revision"):
        raise ASRWindowArtifactError("ASR window provider/revision mismatch")
    if _jsonl_bytes(window.to_dict() for window in windows) != windows_payload:
        raise ASRWindowArtifactError("ASR windows are not canonical")
    if _jsonl_bytes(link.to_dict() for link in links) != links_payload:
        raise ASRWindowArtifactError("ASR frame links are not canonical")
    linked_windows = {link.window_uid for link in links}
    if len(linked_windows) != int(manifest.get("n_windows_with_links", -1)):
        raise ASRWindowArtifactError("ASR linked-window count mismatch")
    quality_counts = dict(sorted(Counter(link.mapping_quality for link in links).items()))
    if quality_counts != manifest.get("mapping_quality_counts", {}):
        raise ASRWindowArtifactError("ASR mapping quality counts mismatch")
    return ASRWindowArtifact(artifact_dir, tuple(windows), tuple(links), manifest)


@dataclass(frozen=True)
class _WindowDocument:
    window: ASRWindowRecord
    tokens: tuple[str, ...]


class ASRWindowRetrievalChannel:
    """Deterministic lexical ASR-window channel emitting frame-level hits."""

    def __init__(self, artifact: ASRWindowArtifact) -> None:
        self.artifact = artifact
        self._links_by_window: dict[str, tuple[ASRFrameLink, ...]] = defaultdict(tuple)
        link_groups: dict[str, list[ASRFrameLink]] = defaultdict(list)
        for link in artifact.links:
            link_groups[link.window_uid].append(link)
        self._links_by_window = {
            uid: tuple(
                sorted(
                    group,
                    key=lambda item: (not item.is_anchor, item.timestamp_ms, item.frame_uid),
                )
            )
            for uid, group in link_groups.items()
        }
        self._documents = tuple(
            _WindowDocument(window, tuple(normalize_ocr_text(window.text).split()))
            for window in artifact.windows
        )
        self._postings: dict[str, list[int]] = defaultdict(list)
        for document_idx, document in enumerate(self._documents):
            for token in set(document.tokens):
                self._postings[token].append(document_idx)
        self._df = Counter(token for document in self._documents for token in set(document.tokens))

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: Path,
        *,
        dataset_manifest_hash: str | None = None,
        keyframe_manifest_hash: str | None = None,
    ) -> ASRWindowRetrievalChannel:
        return cls(
            load_asr_window_artifact(
                artifact_dir,
                dataset_manifest_hash=dataset_manifest_hash,
                keyframe_manifest_hash=keyframe_manifest_hash,
            )
        )

    @property
    def provider(self) -> str:
        return str(self.artifact.manifest["provider"])

    @property
    def revision(self) -> str:
        return str(self.artifact.manifest["revision"])

    @property
    def execution_status(self) -> str:
        return "ENGINEERING_PROXY"

    @property
    def quality_status(self) -> str:
        return str(self.artifact.manifest.get("quality_status", "UNVALIDATED_ON_HCMAIC"))

    @property
    def dataset_manifest_hash(self) -> str | None:
        value = self.artifact.manifest.get("dataset_manifest_hash")
        return str(value) if isinstance(value, str) and value else None

    @property
    def artifact_hash(self) -> str | None:
        value = self.artifact.manifest.get("windows_sha256")
        return str(value) if isinstance(value, str) and value else None

    def channel_contract(self) -> ChannelContract:
        return ChannelContract(
            channel="asr",
            provider=self.provider,
            revision=self.revision,
            execution_status=self.execution_status,
            quality_status=self.quality_status,
            dataset_manifest_hash=self.dataset_manifest_hash,
            artifact_hash=self.artifact_hash,
            status="ready",
        )

    def search(self, text: str, top_k: int = 100) -> list[ChannelHit]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_tokens = normalize_ocr_text(text).split()
        if not query_tokens:
            return []
        candidates = sorted(
            {
                document_idx
                for token in query_tokens
                for document_idx in self._postings.get(token, [])
            }
        )
        n_windows = len(self._documents)
        best_by_frame: dict[str, tuple[float, _WindowDocument, ASRFrameLink, list[str]]] = {}
        for document_idx in candidates:
            document = self._documents[document_idx]
            matching_tokens = [token for token in query_tokens if token in document.tokens]
            if not matching_tokens:
                continue
            score = sum(
                math.log(1.0 + (n_windows + 1.0) / (self._df[token] + 1.0))
                for token in set(matching_tokens)
            )
            score *= document.window.confidence or 1.0
            links = self._links_by_window.get(document.window.window_uid, ())
            for link in links:
                link_score = score + (0.001 if link.is_anchor else 0.0)
                current = best_by_frame.get(link.frame_uid)
                candidate = (link_score, document, link, matching_tokens)
                if current is None or (link_score, link.is_anchor) > (
                    current[0],
                    current[2].is_anchor,
                ):
                    best_by_frame[link.frame_uid] = candidate
        ranked = sorted(
            best_by_frame.values(),
            key=lambda item: (
                -item[0],
                item[2].video_id,
                item[2].source_frame_idx,
                item[2].frame_uid,
            ),
        )[:top_k]
        return [
            ChannelHit(
                entity_id=link.frame_uid,
                video_id=link.video_id,
                timestamp_ms=link.timestamp_ms,
                modality="asr",
                score=float(score),
                rank=rank,
                provider=self.provider,
                evidence_text=document.window.text,
                frame_uid=link.frame_uid,
                video_filename=link.video_filename,
                source_frame_idx=link.source_frame_idx,
                evidence=build_channel_evidence(
                    channel="asr",
                    provider=self.provider,
                    revision=self.revision,
                    execution_status=self.execution_status,
                    quality_status=self.quality_status,
                    dataset_manifest_hash=self.dataset_manifest_hash,
                    artifact_hash=self.artifact_hash,
                    frame_uid=link.frame_uid,
                    video_id=link.video_id,
                    video_filename=link.video_filename,
                    source_frame_idx=link.source_frame_idx,
                    timestamp_ms=link.timestamp_ms,
                    score=float(score),
                    rank=rank,
                    channel_specific={
                        "asr_window_uid": document.window.window_uid,
                        "window_id": document.window.window_uid,
                        "window_uid": document.window.window_uid,
                        "start_ms": document.window.start_ms,
                        "end_ms": document.window.end_ms,
                        "text": document.window.text,
                        "text_raw": str(
                            document.window.metadata.get("text_raw", document.window.text)
                        ),
                        "confidence": document.window.confidence,
                        "matched_tokens": matching_tokens,
                        "mapping_quality": link.mapping_quality,
                        "is_anchor": link.is_anchor,
                        "support_frame_count": len(
                            self._links_by_window.get(document.window.window_uid, ())
                        ),
                    },
                    raw_provenance=self.channel_contract().to_raw_provenance(),
                ),
            )
            for rank, (score, document, link, matching_tokens) in enumerate(ranked, start=1)
        ]
