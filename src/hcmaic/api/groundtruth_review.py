"""FastAPI app for teammate review of proposed temporal ranges."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hcmaic.groundtruth.review import (
    EVIDENCE_LEVEL,
    REVIEW_SCHEMA_VERSION,
    DecisionStore,
    load_jsonl,
    validate_decision,
)

REVIEW_UI_DIR = Path(__file__).resolve().parent.parent / "ui" / "review_static"


class ReviewDecisionRequest(BaseModel):
    status: Literal["accepted", "rejected", "edited"]
    left: int | None = Field(default=None, ge=0)
    right: int | None = Field(default=None, ge=0)
    reviewer: str = Field(default="teammate", min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=2000)


def _load_items(review_root: Path) -> dict[str, dict[str, Any]]:
    path = review_root / "review_items.jsonl"
    rows = load_jsonl(path)
    items: dict[str, dict[str, Any]] = {}
    for row in rows:
        uid = str(row.get("review_uid") or "")
        if not uid or uid in items:
            raise ValueError(f"review_items.jsonl contains invalid/duplicate review_uid: {uid!r}")
        items[uid] = row
    return items


def _read_manifest(review_root: Path) -> dict[str, Any]:
    path = review_root / "review_manifest.json"
    if not path.is_file():
        return {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "evidence_level": EVIDENCE_LEVEL,
            "status": "ENGINEERING_ARTIFACT_COMPLETE",
            "quality_status": "UNVALIDATED",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid review_manifest.json: {path}") from exc
    return data if isinstance(data, dict) else {}


def _status_counts(items: dict[str, dict[str, Any]], store: DecisionStore) -> dict[str, int]:
    counts = {"pending": 0, "accepted": 0, "rejected": 0, "edited": 0}
    for uid in items:
        status = str((store.get(uid) or {}).get("status") or "pending")
        counts[status if status in counts else "pending"] += 1
    return counts


def create_groundtruth_review_app(
    review_root: Path,
    *,
    ui_dir: Path | None = None,
) -> FastAPI:
    """Create a standalone review app from a prepared bundle directory."""

    review_root = Path(review_root).resolve()
    if not review_root.is_dir():
        raise FileNotFoundError(review_root)
    items = _load_items(review_root)
    manifest = _read_manifest(review_root)
    store = DecisionStore(review_root / "review_decisions.jsonl")
    app = FastAPI(
        title="HCMAIC ground-truth range review",
        version="0.1.0",
        description="Human review draft; not official qrels and not retrieval-quality evidence.",
    )
    app.state.review_root = review_root
    app.state.items = items
    app.state.store = store
    app.state.manifest = manifest

    def with_runtime_fields(item: dict[str, Any]) -> dict[str, Any]:
        result = dict(item)
        decision = store.get(str(item["review_uid"]))
        result["decision"] = decision
        result["review_status"] = str((decision or {}).get("status") or "pending")
        frames = []
        for frame in item.get("frames", []):
            enriched = dict(frame)
            if frame.get("image_relpath"):
                enriched["image_url"] = (
                    f"/api/review/items/{item['review_uid']}/frames/"
                    f"{int(frame['source_frame_idx'])}/image"
                )
            else:
                enriched["image_url"] = None
            frames.append(enriched)
        result["frames"] = frames
        return result

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "item_count": len(items),
            "counts": _status_counts(items, store),
            "evidence_level": EVIDENCE_LEVEL,
            "quality_status": "UNVALIDATED",
            "sampling": manifest.get("sampling"),
        }

    @app.get("/api/review/items")
    def list_items(
        status: Literal["all", "pending", "accepted", "rejected", "edited"] = "all",
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        rows = []
        for uid, item in items.items():
            decision = store.get(uid)
            current_status = str((decision or {}).get("status") or "pending")
            if status != "all" and current_status != status:
                continue
            rows.append(
                {
                    "review_uid": uid,
                    "query_uid": item.get("query_uid"),
                    "query": item.get("query"),
                    "task": item.get("task"),
                    "source": item.get("source"),
                    "video_id": item.get("video_id"),
                    "anchor": item.get("anchor"),
                    "proposed_range": item.get("proposed_range"),
                    "review_status": current_status,
                }
            )
        rows.sort(key=lambda row: str(row["review_uid"]))
        return {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "evidence_level": EVIDENCE_LEVEL,
            "status": status,
            "total": len(rows),
            "counts": _status_counts(items, store),
            "items": rows[offset : offset + limit],
        }

    @app.get("/api/review/items/{review_uid}")
    def get_item(review_uid: str) -> dict[str, Any]:
        item = items.get(review_uid)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Unknown review_uid {review_uid!r}")
        return with_runtime_fields(item)

    @app.post("/api/review/items/{review_uid}/decision")
    def save_decision(review_uid: str, body: ReviewDecisionRequest) -> dict[str, Any]:
        item = items.get(review_uid)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Unknown review_uid {review_uid!r}")
        try:
            canonical = validate_decision(item, body.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = {
            **canonical,
            "review_uid": review_uid,
            "video_id": item.get("video_id"),
            "query_uid": item.get("query_uid"),
            "frame_uid": (item.get("anchor") or {}).get("frame_uid"),
            "reviewer": body.reviewer,
            "note": body.note,
            "reviewed_at": datetime.now(UTC).isoformat(),
        }
        return store.upsert(record)

    @app.get("/api/review/items/{review_uid}/frames/{source_frame_idx}/image")
    def get_frame_image(review_uid: str, source_frame_idx: int) -> FileResponse:
        item = items.get(review_uid)
        if item is None:
            raise HTTPException(status_code=404, detail=f"Unknown review_uid {review_uid!r}")
        frame = next(
            (
                frame
                for frame in item.get("frames", [])
                if int(frame.get("source_frame_idx", -1)) == source_frame_idx
            ),
            None,
        )
        if frame is None or not frame.get("image_relpath"):
            raise HTTPException(status_code=404, detail="Frame image is not materialized")
        image_path = (review_root / str(frame["image_relpath"])).resolve()
        if review_root not in image_path.parents or not image_path.is_file():
            raise HTTPException(status_code=404, detail="Frame image not found")
        media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(image_path, media_type=media_type)

    @app.get("/api/review/export")
    def export_decisions() -> dict[str, Any]:
        decisions = store.all()
        return {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "status": "HUMAN_REVIEW_DRAFT",
            "evidence_level": EVIDENCE_LEVEL,
            "quality_status": "UNVALIDATED",
            "item_count": len(items),
            "counts": _status_counts(items, store),
            "decisions": decisions,
        }

    static_dir = Path(ui_dir) if ui_dir else REVIEW_UI_DIR
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="review-ui")
    return app
