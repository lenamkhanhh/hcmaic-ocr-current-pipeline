"""Prepare source-frame review bundles for human range adjudication.

The review bundle is deliberately separate from official qrels.  It keeps the
stable source identity (``video_id:source_frame_idx``), source PTS timestamp,
proposal provenance, and an append-safe decision ledger.  Sampling is target
timestamp based: 3 FPS over an anchor-centred +/-20 second window, with the
anchor and proposed boundaries retained even when they do not land on a grid
instant.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REVIEW_SCHEMA_VERSION = 1
EVIDENCE_LEVEL = "HUMAN_REVIEW_DRAFT"
DEFAULT_FPS = 3.0
DEFAULT_WINDOW_BEFORE_S = 20.0
DEFAULT_WINDOW_AFTER_S = 20.0


class ReviewBundleError(RuntimeError):
    """Raised when a review bundle cannot be built safely."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows and fail with the source path in the error."""

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReviewBundleError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ReviewBundleError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(row)
    except FileNotFoundError as exc:
        raise ReviewBundleError(f"Missing JSONL input: {path}") from exc
    return rows


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ReviewBundleError(f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReviewBundleError(f"{field} must be an integer") from exc


def _float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ReviewBundleError(f"{field} must be numeric") from exc


def _normalise_pts(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, int]]:
    """Normalise PTS rows while preserving source frame indices."""

    normalised: list[dict[str, int]] = []
    seen: set[int] = set()
    for raw in rows:
        idx = _int(raw.get("source_frame_idx"), "source_frame_idx")
        if idx in seen:
            continue
        timestamp_ns = _int(raw.get("timestamp_ns"), "timestamp_ns")
        if idx < 0 or timestamp_ns < 0:
            raise ReviewBundleError("source frame index and timestamp must be non-negative")
        seen.add(idx)
        normalised.append(
            {"source_frame_idx": idx, "timestamp_ns": timestamp_ns, "pts": raw.get("pts", idx)}
        )
    normalised.sort(key=lambda row: (row["timestamp_ns"], row["source_frame_idx"]))
    if not normalised:
        raise ReviewBundleError("PTS table is empty")
    return normalised


def _nearest_index(values: Sequence[int], target: int) -> int:
    position = bisect.bisect_left(values, target)
    if position == 0:
        return 0
    if position == len(values):
        return len(values) - 1
    before = position - 1
    return position if abs(values[position] - target) < abs(values[before] - target) else before


def _row_for_frame_idx(rows: Sequence[Mapping[str, int]], frame_idx: int) -> Mapping[str, int]:
    by_idx = {int(row["source_frame_idx"]): row for row in rows}
    if frame_idx in by_idx:
        return by_idx[frame_idx]
    indices = sorted(by_idx)
    return by_idx[indices[_nearest_index(indices, frame_idx)]]


def _frame_payload(video_id: str, row: Mapping[str, int]) -> dict[str, Any]:
    idx = int(row["source_frame_idx"])
    timestamp_ns = int(row["timestamp_ns"])
    timestamp_ms = int(round(timestamp_ns / 1_000_000))
    return {
        "video_id": video_id,
        "frame_uid": f"{video_id}:{idx}",
        "source_frame_idx": idx,
        "timestamp_ns": timestamp_ns,
        "timestamp_ms": timestamp_ms,
        "timestamp_s": round(timestamp_ns / 1_000_000_000, 6),
        "pts": row.get("pts", idx),
    }


def build_sampling_plan(
    proposal: Mapping[str, Any],
    pts_rows: Iterable[Mapping[str, Any]],
    *,
    fps: float = DEFAULT_FPS,
    window_before_s: float = DEFAULT_WINDOW_BEFORE_S,
    window_after_s: float = DEFAULT_WINDOW_AFTER_S,
) -> list[dict[str, Any]]:
    """Return one row per selected source frame for a proposal.

    ``window_before_s`` and ``window_after_s`` are measured from the anchor;
    defaults therefore mean a total 40-second review window.  Target instants
    are mapped to the nearest decoded PTS row, then duplicate source frames are
    merged so source identity remains one-to-one.
    """

    fps = _float(fps, "fps")
    before = _float(window_before_s, "window_before_s")
    after = _float(window_after_s, "window_after_s")
    if fps <= 0 or before < 0 or after < 0:
        raise ValueError("fps must be > 0 and windows must be >= 0")

    rows = _normalise_pts(pts_rows)
    video_id = str(proposal.get("video_id") or "")
    if not video_id:
        raise ReviewBundleError("proposal is missing video_id")
    anchor_idx = _int(proposal.get("anchor"), "anchor")
    proposal_range = proposal.get("proposal") or {}
    left_idx = _int(proposal_range.get("left"), "proposal.left")
    right_idx = _int(proposal_range.get("right"), "proposal.right")
    if left_idx > right_idx:
        raise ReviewBundleError("proposal.left must be <= proposal.right")

    timestamps = [int(row["timestamp_ns"]) for row in rows]
    anchor_row = _row_for_frame_idx(rows, anchor_idx)
    anchor_ns = int(anchor_row["timestamp_ns"])
    first_ns = timestamps[0]
    last_ns = timestamps[-1]
    start_ns = max(first_ns, anchor_ns - int(round(before * 1_000_000_000)))
    end_ns = min(last_ns, anchor_ns + int(round(after * 1_000_000_000)))
    if start_ns > end_ns:
        start_ns, end_ns = end_ns, start_ns

    target_specs: list[tuple[int, str]] = []
    period_ns = 1_000_000_000 / fps
    target = float(start_ns)
    while target <= end_ns + 0.5:
        target_specs.append((int(round(target)), "grid"))
        target += period_ns
    if not target_specs or target_specs[-1][0] != end_ns:
        target_specs.append((end_ns, "window_end"))
    target_specs.extend(
        [
            (anchor_ns, "anchor"),
            (int(_row_for_frame_idx(rows, left_idx)["timestamp_ns"]), "proposal_left"),
            (int(_row_for_frame_idx(rows, right_idx)["timestamp_ns"]), "proposal_right"),
        ]
    )

    selected: dict[int, dict[str, Any]] = {}
    for target_ns, role in target_specs:
        row_position = _nearest_index(timestamps, target_ns)
        source_row = rows[row_position]
        idx = int(source_row["source_frame_idx"])
        entry = selected.setdefault(
            idx,
            {
                **_frame_payload(video_id, source_row),
                "target_timestamps_ms": [],
                "sample_roles": [],
                "is_anchor": False,
                "is_proposal_boundary": False,
                "proposal_boundaries": [],
            },
        )
        target_ms = int(round(target_ns / 1_000_000))
        if target_ms not in entry["target_timestamps_ms"]:
            entry["target_timestamps_ms"].append(target_ms)
        if role not in entry["sample_roles"]:
            entry["sample_roles"].append(role)
        if role == "anchor" or idx == int(anchor_row["source_frame_idx"]):
            entry["is_anchor"] = True
        if role in {"proposal_left", "proposal_right"}:
            entry["is_proposal_boundary"] = True
            boundary = "left" if role == "proposal_left" else "right"
            if boundary not in entry["proposal_boundaries"]:
                entry["proposal_boundaries"].append(boundary)

    ordered = sorted(
        selected.values(), key=lambda row: (row["timestamp_ns"], row["source_frame_idx"])
    )
    for row in ordered:
        row["target_timestamps_ms"].sort()
        row["sample_roles"].sort()
        row["proposal_boundaries"].sort()
        row["distance_to_nearest_target_ms"] = min(
            abs(row["timestamp_ms"] - target_ms) for target_ms in row["target_timestamps_ms"]
        )
    return ordered


def build_review_item(
    proposal: Mapping[str, Any],
    pts_rows: Iterable[Mapping[str, Any]],
    *,
    fps: float = DEFAULT_FPS,
    window_before_s: float = DEFAULT_WINDOW_BEFORE_S,
    window_after_s: float = DEFAULT_WINDOW_AFTER_S,
) -> dict[str, Any]:
    """Build a review item while retaining task/query/proposal provenance."""

    frames = build_sampling_plan(
        proposal,
        pts_rows,
        fps=fps,
        window_before_s=window_before_s,
        window_after_s=window_after_s,
    )
    proposal_range = proposal.get("proposal") or {}
    anchor_idx = _int(proposal.get("anchor"), "anchor")
    anchor_frame = next(
        (frame for frame in frames if frame["source_frame_idx"] == anchor_idx),
        next(frame for frame in frames if frame["is_anchor"]),
    )
    review_uid = str(proposal.get("range_review_uid") or proposal.get("query_uid") or "")
    if not review_uid:
        raise ReviewBundleError("proposal is missing range_review_uid/query_uid")
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_uid": review_uid,
        "range_review_uid": review_uid,
        "query_uid": proposal.get("query_uid"),
        "query": proposal.get("query"),
        "task": proposal.get("task"),
        "source": proposal.get("source"),
        "source_testcase_id": proposal.get("source_testcase_id"),
        "video_id": proposal.get("video_id"),
        "frame_count": proposal.get("frame_count"),
        "anchor": anchor_frame,
        "proposed_range": {
            "left": _int(proposal_range.get("left"), "proposal.left"),
            "right": _int(proposal_range.get("right"), "proposal.right"),
            "source_start_ms": proposal_range.get("source_start_ms"),
            "source_end_ms": proposal_range.get("source_end_ms"),
            "method": proposal_range.get("method"),
        },
        "sampling": {
            "fps": float(fps),
            "window_before_s": float(window_before_s),
            "window_after_s": float(window_after_s),
            "window_semantics": "anchor_centered",
        },
        "frames": frames,
    }


def validate_decision(item: Mapping[str, Any], body: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a teammate decision and return canonical range fields."""

    status = str(body.get("status") or "")
    if status not in {"accepted", "rejected", "edited"}:
        raise ValueError("status must be accepted, rejected, or edited")
    proposed = item.get("proposed_range") or {}
    left = body.get("left", proposed.get("left"))
    right = body.get("right", proposed.get("right"))
    try:
        left = int(left)
        right = int(right)
        anchor = int((item.get("anchor") or {}).get("source_frame_idx"))
    except (TypeError, ValueError) as exc:
        raise ValueError("decision range must contain integer left/right/anchor") from exc
    if left > right:
        raise ValueError("left <= right is required")
    if not left <= anchor <= right:
        raise ValueError("edited range must contain the anchor")
    frame_count = item.get("frame_count")
    if frame_count is not None and not 0 <= left < int(frame_count) and status != "rejected":
        raise ValueError("left is outside frame_count")
    if frame_count is not None and not 0 <= right < int(frame_count) and status != "rejected":
        raise ValueError("right is outside frame_count")
    return {"status": status, "left": left, "right": right, "anchor": anchor}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_status(output_dir: Path, payload: Mapping[str, Any]) -> None:
    """Persist a small liveness/status record before expensive work."""

    path = output_dir / "status.json"
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class DecisionStore:
    """Small atomic JSONL ledger keyed by ``review_uid``."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return load_jsonl(self.path)

    def get(self, review_uid: str) -> dict[str, Any] | None:
        return next((row for row in self.all() if row.get("review_uid") == review_uid), None)

    def upsert(self, record: Mapping[str, Any]) -> dict[str, Any]:
        review_uid = str(record.get("review_uid") or "")
        if not review_uid:
            raise ValueError("review_uid is required")
        canonical = {
            **dict(record),
            "schema_version": REVIEW_SCHEMA_VERSION,
            "review_uid": review_uid,
            "evidence_level": EVIDENCE_LEVEL,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        rows = [row for row in self.all() if row.get("review_uid") != review_uid]
        rows.append(canonical)
        rows.sort(key=lambda row: str(row.get("review_uid")))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return canonical


def _find_pts_path(pts_dir: Path, video_id: str, relative_path: str | None) -> Path:
    candidates = []
    if relative_path:
        rel = Path(relative_path)
        candidates.extend([pts_dir / rel, pts_dir / rel.name])
    candidates.extend([pts_dir / f"{video_id}.jsonl", pts_dir / "pts" / f"{video_id}.jsonl"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ReviewBundleError(f"Missing PTS table for {video_id} under {pts_dir}")


def _resolve_video_path(raw_root: Path, relative_path: str, video_id: str) -> Path:
    rel = Path(relative_path)
    candidates = [raw_root / rel, raw_root / rel.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(raw_root.rglob(f"{video_id}.*"))
    if len(matches) == 1 and matches[0].is_file():
        return matches[0]
    raise ReviewBundleError(
        f"Cannot resolve raw video for {video_id}; tried inventory path and unique filename search"
    )


def _materialize_images(
    items: list[dict[str, Any]],
    inventory_by_video: Mapping[str, Mapping[str, Any]],
    raw_root: Path,
    output_dir: Path,
    *,
    jpeg_quality: int = 90,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Decode each referenced source video once and write deduplicated JPEGs."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ReviewBundleError("materialization requires opencv-python-headless") from exc

    targets: dict[str, set[int]] = defaultdict(set)
    for item in items:
        for frame in item["frames"]:
            targets[str(item["video_id"])].add(int(frame["source_frame_idx"]))
    image_root = output_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    for video_id, frame_indices in sorted(targets.items()):
        inv = inventory_by_video.get(video_id)
        if not inv:
            raise ReviewBundleError(f"No video inventory row for {video_id}")
        video_path = _resolve_video_path(raw_root, str(inv.get("relative_path") or ""), video_id)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ReviewBundleError(f"OpenCV cannot open {video_path}")
        target_max = max(frame_indices)
        written: set[int] = set()
        try:
            for idx in range(target_max + 1):
                ok, image = capture.read()
                if not ok:
                    break
                if idx not in frame_indices:
                    if progress and idx and idx % 10_000 == 0:
                        progress(f"decoding {video_id}: source_frame_idx={idx}/{target_max}")
                    continue
                ok, encoded = cv2.imencode(
                    ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
                )
                if not ok:
                    raise ReviewBundleError(f"JPEG encoding failed for {video_id}:{idx}")
                relative = Path("images") / video_id / f"{idx}.jpg"
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(encoded.tobytes())
                written.add(idx)
                inventory.append(
                    {
                        "video_id": video_id,
                        "source_frame_idx": idx,
                        "frame_uid": f"{video_id}:{idx}",
                        "relative_path": relative.as_posix(),
                        "sha256": _sha256_file(destination),
                        "byte_size": destination.stat().st_size,
                    }
                )
                if progress and idx and idx % 10_000 == 0:
                    progress(f"decoding {video_id}: source_frame_idx={idx}/{target_max}")
        finally:
            capture.release()
        if written != frame_indices:
            missing = sorted(frame_indices - written)
            raise ReviewBundleError(
                f"Video ended before requested frames for {video_id}: {missing[:5]}"
            )
        for item in items:
            if item["video_id"] != video_id:
                continue
            for frame in item["frames"]:
                frame["image_relpath"] = (
                    Path("images") / video_id / f"{frame['source_frame_idx']}.jpg"
                ).as_posix()
        if progress:
            progress(f"materialized {video_id}: {len(written)} unique JPEGs")
    return sorted(inventory, key=lambda row: (row["video_id"], row["source_frame_idx"]))


def prepare_review_bundle(
    proposals_path: Path,
    inventory_path: Path,
    pts_dir: Path,
    output_dir: Path,
    *,
    fps: float = DEFAULT_FPS,
    window_before_s: float = DEFAULT_WINDOW_BEFORE_S,
    window_after_s: float = DEFAULT_WINDOW_AFTER_S,
    raw_root: Path | None = None,
    materialize: bool = False,
    jpeg_quality: int = 90,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Create JSONL review items and optionally materialize their JPEGs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase = "LOAD_INPUTS"
    items: list[dict[str, Any]] = []
    image_inventory: list[dict[str, Any]] = []
    started_at = datetime.now(UTC).isoformat()

    def write_items() -> None:
        items_path = output_dir / "review_items.jsonl"
        with items_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    def emit(message: str) -> None:
        nonlocal phase
        if message.startswith("planned"):
            phase = "PLAN"
        elif message.startswith("materialized") or message.startswith("decoding"):
            phase = "MATERIALIZE"
        _write_status(
            output_dir,
            {
                "status": "RUNNING",
                "quality_status": "UNVALIDATED",
                "evidence_level": EVIDENCE_LEVEL,
                "last_phase": phase,
                "message": message,
                "item_count": len(items),
                "started_at": started_at,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        if progress:
            progress(message)

    _write_status(
        output_dir,
        {
            "status": "RUNNING",
            "quality_status": "UNVALIDATED",
            "evidence_level": EVIDENCE_LEVEL,
            "last_phase": phase,
            "started_at": started_at,
        },
    )
    try:
        proposals = load_jsonl(Path(proposals_path))
        inventory_rows = load_jsonl(Path(inventory_path))
        inventory_by_video = {str(row.get("video_id")): row for row in inventory_rows}
        if len(inventory_by_video) != len(inventory_rows):
            raise ReviewBundleError("video_inventory contains duplicate video_id rows")
        pts_cache: dict[str, list[dict[str, Any]]] = {}
        pts_paths_by_video: dict[str, Path] = {}
        seen_uids: set[str] = set()
        for position, proposal in enumerate(proposals, start=1):
            uid = str(proposal.get("range_review_uid") or proposal.get("query_uid") or "")
            if uid in seen_uids:
                raise ReviewBundleError(f"duplicate range_review_uid: {uid}")
            seen_uids.add(uid)
            video_id = str(proposal.get("video_id") or "")
            inventory = inventory_by_video.get(video_id)
            if not inventory:
                raise ReviewBundleError(f"proposal references missing inventory video: {video_id}")
            pts_path = _find_pts_path(pts_dir, video_id, inventory.get("pts_table_relative_path"))
            pts_paths_by_video[video_id] = pts_path
            if video_id not in pts_cache:
                pts_cache[video_id] = load_jsonl(pts_path)
            items.append(
                build_review_item(
                    proposal,
                    pts_cache[video_id],
                    fps=fps,
                    window_before_s=window_before_s,
                    window_after_s=window_after_s,
                )
            )
            if position % 25 == 0:
                emit(f"planned {position}/{len(proposals)} review items")
        write_items()
        (output_dir / "review_decisions.jsonl").touch(exist_ok=True)

        if materialize:
            if raw_root is None:
                raise ReviewBundleError("raw_root is required when materialize=True")

            def materialize_progress(message: str) -> None:
                write_items()
                emit(message)

            image_inventory = _materialize_images(
                items,
                inventory_by_video,
                Path(raw_root),
                output_dir,
                jpeg_quality=jpeg_quality,
                progress=materialize_progress,
            )

        items_path = output_dir / "review_items.jsonl"
        write_items()
        decisions_path = output_dir / "review_decisions.jsonl"
        decisions_path.touch(exist_ok=True)
        if image_inventory:
            image_inventory_path = output_dir / "image_inventory.jsonl"
            with image_inventory_path.open("w", encoding="utf-8") as handle:
                for row in image_inventory:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        manifest = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "artifact_type": "hcmaic_groundtruth_range_review_bundle",
            "status": "ENGINEERING_ARTIFACT_COMPLETE",
            "quality_status": "UNVALIDATED",
            "evidence_level": EVIDENCE_LEVEL,
            "item_count": len(items),
            "unique_video_count": len({str(item["video_id"]) for item in items}),
            "frame_count": sum(len(item["frames"]) for item in items),
            "unique_materialized_frame_count": len(image_inventory),
            "sampling": {
                "fps": float(fps),
                "window_before_s": float(window_before_s),
                "window_after_s": float(window_after_s),
                "window_semantics": "anchor_centered",
            },
            "proposals_path": str(proposals_path),
            "inventory_path": str(inventory_path),
            "input_hashes": {
                "proposals_sha256": _sha256_file(Path(proposals_path)),
                "inventory_sha256": _sha256_file(Path(inventory_path)),
                "pts_tables_sha256": hashlib.sha256(
                    json.dumps(
                        [
                            {"video_id": video_id, "sha256": _sha256_file(path)}
                            for video_id, path in sorted(pts_paths_by_video.items())
                        ],
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "pts_table_count": len(pts_paths_by_video),
            },
            "files": {
                "review_items.jsonl": _sha256_file(items_path),
                "review_decisions.jsonl": _sha256_file(decisions_path),
            },
        }
        if image_inventory:
            manifest["files"]["image_inventory.jsonl"] = _sha256_file(
                output_dir / "image_inventory.jsonl"
            )
        manifest_path = output_dir / "review_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_status(
            output_dir,
            {
                "status": manifest["status"],
                "quality_status": manifest["quality_status"],
                "evidence_level": EVIDENCE_LEVEL,
                "last_phase": "DONE",
                "item_count": len(items),
                "frame_count": manifest["frame_count"],
                "unique_materialized_frame_count": manifest["unique_materialized_frame_count"],
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        return manifest
    except Exception as exc:
        failure = {
            "phase": phase,
            "status": "ENGINEERING_ARTIFACT_PARTIAL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        (output_dir / "failure_ledger.json").write_text(
            json.dumps([failure], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _write_status(
            output_dir,
            {
                **failure,
                "quality_status": "UNVALIDATED",
                "evidence_level": EVIDENCE_LEVEL,
                "item_count": len(items),
                "started_at": started_at,
            },
        )
        raise
