"""FastAPI surface for the canonical raw-video-first KIS runtime."""

from __future__ import annotations

import base64
import binascii
import json
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from hcmaic.api.bfe_adapter import serialize_providers
from hcmaic.runtime.kis import KISRuntime
from hcmaic.skillpixel.retrieval import load_skillpixel_questions
from hcmaic.skillpixel.submission import (
    SubmissionValidationError,
    export_skillpixel_submission,
    validate_submission_csv,
)

UI_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"
_MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


class KISTextRequest(BaseModel):
    query_id: str | None = None
    text: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def validate_text(self) -> KISTextRequest:
        if not self.text.strip():
            raise ValueError("text must not be blank")
        return self


class KISImageRequest(BaseModel):
    query_id: str | None = None
    image_path: str | None = None
    image_base64: str | None = None
    top_k: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def validate_image(self) -> KISImageRequest:
        if bool(self.image_path) == bool(self.image_base64):
            raise ValueError("provide exactly one of image_path or image_base64")
        return self


class KISBatchItem(BaseModel):
    query_id: str = Field(min_length=1)
    task: str
    text: str | None = None
    image_path: str | None = None
    image_base64: str | None = None
    top_k: int = Field(default=100, ge=1, le=500)

    @model_validator(mode="after")
    def validate_task_payload(self) -> KISBatchItem:
        task = self.task.upper()
        if task not in {"TKIS", "VKIS"}:
            raise ValueError("task must be TKIS or VKIS")
        if task == "TKIS" and not (self.text or "").strip():
            raise ValueError("TKIS batch item needs text")
        if task == "VKIS" and bool(self.image_path) == bool(self.image_base64):
            raise ValueError("VKIS batch item needs exactly one image input")
        return self


class KISBatchRequest(BaseModel):
    queries: list[KISBatchItem] = Field(min_length=1, max_length=500)


class KISExportRequest(BaseModel):
    questions_path: str = Field(min_length=1)
    corpus_path: str = Field(min_length=1)
    output_path: str = Field(min_length=1)
    top_k: int = Field(default=100, ge=100, le=100)


def _query_payload(output: Any, latency_ms: float, runtime: KISRuntime) -> dict[str, Any]:
    return {
        "query_id": output.query.query_id,
        "task": output.query.task,
        "latency_ms": round(latency_ms, 3),
        "results": [result.to_dict() for result in output.results],
        "executed_channels": list(output.executed_channels),
        "unavailable_channels": dict(output.unavailable_channels),
        "channel_status": dict(runtime.channel_status),
        "channel_contracts": dict(runtime.channel_contracts),
        "candidate_count": output.candidate_count,
        "execution_status": "ENGINEERING_PROXY",
        "quality_status": "UNVALIDATED_ON_HCMAIC",
    }


@contextmanager
def _image_input(image_path: str | None, image_base64: str | None) -> Iterator[Path]:
    if image_path:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path
        return
    assert image_base64 is not None
    encoded = image_base64.split(",", 1)[-1]
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is invalid") from exc
    if not payload or len(payload) > 20 * 1024 * 1024:
        raise ValueError("image_base64 must be between 1 byte and 20 MiB")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        yield temporary
    finally:
        temporary.unlink(missing_ok=True)


def _make_query(item: KISBatchItem, image_path: Path | None = None) -> Any:
    from hcmaic.contracts.kis import KISQuery

    task = item.task.upper()
    return KISQuery(
        query_id=item.query_id,
        task=task,
        text=item.text,
        image_path=image_path or (Path(item.image_path) if item.image_path else None),
        top_k=item.top_k,
    )


def create_kis_app(runtime: KISRuntime, *, ui_dir: Path | None = UI_DIR) -> FastAPI:
    """Create an API/UI app backed by the actual KIS orchestrator."""
    app = FastAPI(
        title="HCMAIC KIS retrieval",
        version="0.2.0",
        description="Raw-video-first TKIS/VKIS retrieval with provenance-preserving results.",
    )
    app.state.kis_runtime = runtime
    app.state.feedback_events = deque(maxlen=1000)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/system/info")
    def system_info() -> dict[str, Any]:
        return {
            "index_manifest": dict(runtime.index.index_manifest),
            "dataset_manifest_hash": runtime.index.index_manifest.get("dataset_manifest_hash"),
            "n_frames": runtime.index.size,
            "video_ids": sorted({record.video_id for record in runtime.index.catalog}),
            "runtime": {
                "embedding_provider": runtime.provider.name,
                "embedding_version": runtime.provider.version,
                "index_provider": runtime.index.index_manifest.get("index_provider"),
                "fusion": runtime.orchestrator.fusion_method,
                "reranker": runtime.orchestrator.reranker,
                "channels": dict(runtime.channel_status),
                "channel_status": dict(runtime.channel_status),
                "channel_contracts": dict(runtime.channel_contracts),
                "quality_status": "UNVALIDATED_ON_HCMAIC",
            },
        }

    @app.get("/providers")
    def providers() -> dict[str, Any]:
        return serialize_providers(runtime)

    @app.post("/search/text")
    def search_text(body: KISTextRequest) -> dict[str, Any]:
        from hcmaic.contracts.kis import KISQuery

        query = KISQuery(
            query_id=body.query_id or f"q-{uuid.uuid4().hex[:12]}",
            task="TKIS",
            text=body.text,
            top_k=body.top_k,
            raw_text=body.text,
        )
        started = time.perf_counter()
        try:
            output = runtime.search(query)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _query_payload(output, (time.perf_counter() - started) * 1000.0, runtime)

    @app.post("/search/image")
    def search_image(body: KISImageRequest) -> dict[str, Any]:
        from hcmaic.contracts.kis import KISQuery

        query_id = body.query_id or f"q-{uuid.uuid4().hex[:12]}"
        started = time.perf_counter()
        try:
            with _image_input(body.image_path, body.image_base64) as path:
                output = runtime.search(
                    KISQuery(query_id=query_id, task="VKIS", image_path=path, top_k=body.top_k)
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _query_payload(output, (time.perf_counter() - started) * 1000.0, runtime)

    @app.post("/search/batch")
    def search_batch(body: KISBatchRequest) -> dict[str, Any]:
        queries: list[Any] = []
        temporary_paths: list[Path] = []
        try:
            for item in body.queries:
                if item.task.upper() == "VKIS" and item.image_base64:
                    encoded = item.image_base64.split(",", 1)[-1]
                    payload = base64.b64decode(encoded, validate=True)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                        handle.write(payload)
                        path = Path(handle.name)
                    temporary_paths.append(path)
                    queries.append(_make_query(item, path))
                else:
                    queries.append(_make_query(item))
            started = time.perf_counter()
            outputs = runtime.search_queries(queries)
            latency_ms = (time.perf_counter() - started) * 1000.0
        except (binascii.Error, FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)
        return {
            "results": {
                query_id: _query_payload(output, latency_ms / len(outputs), runtime)
                for query_id, output in outputs.items()
            },
            "query_order": [item.query_id for item in body.queries],
            "batch_latency_ms": round(latency_ms, 3),
        }

    @app.get("/frames/{frame_uid}/image")
    def frame_image(frame_uid: str) -> FileResponse:
        try:
            path = runtime.frame_image_path(frame_uid)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_uid {frame_uid!r}") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="frame image is missing")
        return FileResponse(path, media_type=_MEDIA_TYPES.get(path.suffix.lower()))

    @app.get("/frames/{frame_uid}")
    def frame_context(frame_uid: str, window: int = 5) -> dict[str, Any]:
        if window < 0 or window > 50:
            raise HTTPException(status_code=422, detail="window must be in [0, 50]")
        try:
            record = next(item for item in runtime.index.catalog if item.frame_id == frame_uid)
            timeline_rows = runtime.timeline(record.video_id)
        except StopIteration as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_uid {frame_uid!r}") from exc
        position = next(
            index for index, item in enumerate(timeline_rows) if item["frame_id"] == frame_uid
        )
        neighbors = timeline_rows[max(0, position - window) : position + window + 1]
        return {
            "frame": {**record.model_dump(), "image_url": f"/frames/{frame_uid}/image"},
            "neighbors": [
                {**item, "is_current": item["frame_id"] == frame_uid} for item in neighbors
            ],
            "image_url": f"/frames/{frame_uid}/image",
        }

    @app.get("/videos/{video_id}/timeline")
    def timeline(video_id: str) -> dict[str, Any]:
        try:
            frames = runtime.timeline(video_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown video_id {video_id!r}") from exc
        return {"video_id": video_id, "n_frames": len(frames), "frames": frames}

    @app.post("/submit/preview")
    def submit_preview(body: dict[str, Any]) -> dict[str, Any]:
        frame_uid = str(body.get("frame_id", "")).strip()
        try:
            record = next(item for item in runtime.index.catalog if item.frame_id == frame_uid)
        except StopIteration as exc:
            raise HTTPException(status_code=404, detail=f"Unknown frame_uid {frame_uid!r}") from exc
        source_frame_idx = (
            record.source_frame_idx if record.source_frame_idx is not None else record.frame_idx
        )
        return {
            "query_id": str(body.get("query_id", "manual")),
            "task_type": str(body.get("task_type", "TKIS")),
            "video_id": record.video_id,
            "video_filename": record.video_filename or f"{record.video_id}.mp4",
            "source_frame_idx": source_frame_idx,
            "timestamp_ms": record.timestamp_ms,
            "answer": body.get("answer"),
            "confidence": body.get("confidence"),
            "evidence_level": "SESSION_LOCAL",
            "quality_status": "UNVALIDATED_ON_HCMAIC",
        }

    @app.post("/feedback")
    def feedback(body: dict[str, Any]) -> dict[str, Any]:
        event = dict(body)
        app.state.feedback_events.append(event)
        return {
            "status": "recorded",
            "record_count": len(app.state.feedback_events),
            "event": event,
            "evidence_level": "SESSION_LOCAL",
        }

    @app.post("/exports/kis")
    def export_kis(body: KISExportRequest) -> dict[str, Any]:
        questions_path = Path(body.questions_path)
        corpus_path = Path(body.corpus_path)
        output_path = Path(body.output_path)
        questions = load_skillpixel_questions(questions_path)
        from hcmaic.contracts.kis import KISQuery

        queries: list[Any] = []
        for item in questions:
            image_path = Path(item.query_image)
            if item.task == "VKIS" and not image_path.is_absolute():
                image_path = questions_path.parent / image_path
            queries.append(
                KISQuery(
                    query_id=item.query_id,
                    task=item.task,
                    text=item.text or None,
                    image_path=image_path if item.task == "VKIS" else None,
                    top_k=body.top_k,
                )
            )
        outputs = runtime.search_queries(queries)
        temporary = output_path.with_suffix(output_path.suffix + ".kis-results.jsonl")
        try:
            temporary.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("w", encoding="utf-8") as handle:
                for item in questions:
                    handle.write(
                        json.dumps(
                            {
                                "query_id": item.query_id,
                                "answers": [
                                    result.to_dict() for result in outputs[item.query_id].results
                                ],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            try:
                stats = export_skillpixel_submission(
                    questions_path, temporary, corpus_path, output_path
                )
            except SubmissionValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors) from exc
        finally:
            temporary.unlink(missing_ok=True)
        validation = validate_submission_csv(output_path, questions_path, corpus_path)
        if not validation.ok:
            raise HTTPException(status_code=422, detail=list(validation.errors))
        return {
            "status": "validated",
            "output_path": str(stats.output_path),
            "n_queries": stats.n_queries,
            "answers_per_query": stats.answers_per_query,
            "quality_status": "UNVALIDATED_ON_HCMAIC",
        }

    if ui_dir is not None and Path(ui_dir).is_dir():
        app.mount("/", StaticFiles(directory=Path(ui_dir), html=True), name="ui")
    return app
