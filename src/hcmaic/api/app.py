"""FastAPI app serving search, frames, timeline, submission preview and UI.

Fail-fast: artifacts load inside create_app; a broken index refuses to
serve instead of reporting a healthy empty server (upstream bug class).
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from hcmaic.contracts.models import CanonicalSubmission, SearchRequest, SearchResult
from hcmaic.retrieval.feedback import FeedbackEvent
from hcmaic.retrieval.service import (
    RetrievalService,
    UnknownFrameError,
    UnknownVideoError,
    load_service,
)

UI_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"

_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


class ApiSearchRequest(BaseModel):
    """POST /search body; query_id is generated when omitted."""

    query_id: str | None = None
    session_id: str | None = None
    task_type: str = "kis"
    text: str = Field(min_length=1)
    filters: dict = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=500)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class SearchResponse(BaseModel):
    query_id: str
    task_type: str
    index_version: str
    latency_ms: float
    total_found: int
    results: list[SearchResult]


class SubmitPreviewRequest(BaseModel):
    query_id: str = Field(min_length=1)
    task_type: str = "kis"
    frame_id: str = Field(min_length=1)
    answer: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


def create_app(
    artifacts_dir: Path,
    dataset_root: Path | None = None,
    index_provider: str | None = None,
    service: RetrievalService | None = None,
) -> FastAPI:
    if service is None:
        service = load_service(
            artifacts_dir, dataset_root=dataset_root, index_provider=index_provider
        )

    app = FastAPI(
        title="HCMAIC keyframe search",
        version="0.1.0",
        description="Local keyframe retrieval MVP (HCMAIC 2026, Bảng A).",
    )
    app.state.service = service
    app.state.feedback_events = deque(maxlen=1000)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "index_size": service.index.size,
            "index_version": service.index_version,
            "embedding_provider": service.text_provider.name,
            "n_videos": len(service.video_ids()),
        }

    @app.get("/system/info")
    def system_info() -> dict:
        public_manifest = {
            key: value
            for key, value in service.artifacts.index_manifest.items()
            if key != "dataset_root"
        }
        config = public_manifest.get("config", {})
        return {
            "index_manifest": public_manifest,
            "dataset_manifest_hash": service.artifacts.dataset_manifest.get("dataset_hash"),
            "n_frames": len(service.artifacts.catalog),
            "video_ids": service.video_ids(),
            "runtime": {
                "embedding_provider": service.text_provider.name,
                "embedding_version": service.text_provider.version,
                "index_provider": public_manifest.get("index_provider", service.index.name),
                "fusion": config.get("fusion", {}).get("name", "single-stage"),
                "reranker": config.get("reranker", {}).get("name", "identity"),
            },
        }

    @app.post("/search", response_model=SearchResponse)
    def search(body: ApiSearchRequest) -> SearchResponse:
        request = SearchRequest(
            query_id=body.query_id or f"q-{uuid.uuid4().hex[:12]}",
            session_id=body.session_id,
            task_type=body.task_type,
            text=body.text,
            filters=body.filters,
            top_k=body.top_k,
        )
        start = time.perf_counter()
        try:
            results = service.search(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        return SearchResponse(
            query_id=request.query_id,
            task_type=request.task_type,
            index_version=service.index_version,
            latency_ms=round(latency_ms, 3),
            total_found=len(results),
            results=results,
        )

    @app.get("/frames/{frame_id}")
    def get_frame(frame_id: str, window: int = Query(default=5, ge=0, le=50)) -> dict:
        try:
            record = service.get_frame(frame_id)
            neighbors = service.neighbors(frame_id, window=window)
            shot_context = service.shot_context(frame_id)
        except UnknownFrameError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown frame_id {frame_id!r}. Use ids returned by /search.",
            ) from exc
        return {
            "frame": record.model_dump(),
            "neighbors": [
                {
                    **n.model_dump(),
                    "is_current": n.frame_id == frame_id,
                    "image_url": f"/frames/{n.frame_id}/image",
                }
                for n in neighbors
            ],
            "shot_context": {
                name: [
                    {
                        **item.model_dump(),
                        "image_url": f"/frames/{item.frame_id}/image",
                    }
                    for item in items
                ]
                for name, items in shot_context.items()
            },
            "image_url": f"/frames/{frame_id}/image",
        }

    @app.get("/frames/{frame_id}/image")
    def get_frame_image(frame_id: str) -> FileResponse:
        try:
            path = service.frame_image_path(frame_id)
        except UnknownFrameError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_id {frame_id!r}") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Image file for {frame_id} not found on disk.",
            )
        media_type = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media_type)

    @app.get("/videos/{video_id}/timeline")
    def get_timeline(video_id: str) -> dict:
        try:
            frames = service.timeline(video_id)
        except UnknownVideoError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown video_id {video_id!r}. Known: {service.video_ids()}",
            ) from exc
        return {
            "video_id": video_id,
            "n_frames": len(frames),
            "frames": [
                {**r.model_dump(), "image_url": f"/frames/{r.frame_id}/image"} for r in frames
            ],
        }

    @app.post("/submit/preview", response_model=CanonicalSubmission)
    def submit_preview(body: SubmitPreviewRequest) -> CanonicalSubmission:
        try:
            return service.submission_preview(
                query_id=body.query_id,
                task_type=body.task_type,
                frame_id=body.frame_id,
                answer=body.answer,
                confidence=body.confidence,
            )
        except UnknownFrameError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown frame_id {body.frame_id!r}"
            ) from exc

    @app.post("/feedback")
    def record_feedback(body: FeedbackEvent) -> dict:
        app.state.feedback_events.append(body.model_dump())
        return {
            "status": "recorded",
            "record_count": len(app.state.feedback_events),
            "event": body.model_dump(),
            "evidence_level": "SESSION_LOCAL",
        }

    if UI_DIR.is_dir():
        app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")

    return app
