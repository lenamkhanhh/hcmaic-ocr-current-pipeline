"""RetrievalService: text -> ranked, fully-mapped keyframes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hcmaic.contracts.models import (
    CanonicalSubmission,
    FrameRecord,
    SearchRequest,
    SearchResult,
)
from hcmaic.embedding.base import EmbeddingProvider, get_provider
from hcmaic.indexing.artifacts import IndexArtifacts, load_index_artifacts
from hcmaic.indexing.base import SearchIndex
from hcmaic.indexing.numpy_index import ExactNumpyIndex


class UnknownFrameError(KeyError):
    pass


class UnknownVideoError(KeyError):
    pass


def _make_index(name: str) -> SearchIndex:
    if name in ("exact-numpy", "exact", "numpy"):
        return ExactNumpyIndex()
    if name == "faiss":
        from hcmaic.indexing.faiss_index import FaissIndex

        return FaissIndex()
    if name == "faiss-hnsw":
        from hcmaic.indexing.faiss_ann import FaissHNSWIndex

        return FaissHNSWIndex()
    raise ValueError(
        f"Unknown index provider {name!r}; expected exact-numpy, faiss, or faiss-hnsw."
    )


class RetrievalService:
    def __init__(
        self,
        artifacts: IndexArtifacts,
        text_provider: EmbeddingProvider | None = None,
        index: SearchIndex | None = None,
        dataset_root: Path | None = None,
    ) -> None:
        self.artifacts = artifacts
        emb_info = artifacts.index_manifest.get("embedding", {})
        provider_name = str(emb_info.get("provider", "mock"))
        self.text_provider = text_provider or get_provider(provider_name)
        if self.text_provider.dimension != artifacts.embeddings.shape[1]:
            raise ValueError(
                f"Text provider dimension {self.text_provider.dimension} != "
                f"index dimension {artifacts.embeddings.shape[1]}. The serving "
                f"provider must match the one used at build time "
                f"({emb_info.get('version')})."
            )
        expected_version = str(emb_info.get("version", ""))
        if not expected_version or self.text_provider.version != expected_version:
            raise ValueError(
                f"Text provider version {self.text_provider.version!r} != "
                f"index embedding version {expected_version!r}. Use the exact "
                "provider/model that built the index."
            )
        self.index = index or _make_index(
            str(artifacts.index_manifest.get("index_provider", "exact-numpy"))
        )
        self.index.build(artifacts.embeddings, artifacts.id_map)
        self.dataset_root = Path(
            dataset_root
            if dataset_root is not None
            else artifacts.index_manifest.get("dataset_root", ".")
        )

        self._by_frame_id: dict[str, FrameRecord] = {r.frame_id: r for r in artifacts.catalog}
        self._row_of: dict[str, int] = {fid: i for i, fid in enumerate(artifacts.id_map)}
        self._frames_of_video: dict[str, list[FrameRecord]] = {}
        for record in artifacts.catalog:
            self._frames_of_video.setdefault(record.video_id, []).append(record)
        for frames in self._frames_of_video.values():
            frames.sort(key=lambda r: (r.timestamp_ms, r.frame_id))

    # -- lookups ---------------------------------------------------------

    @property
    def index_version(self) -> str:
        return self.artifacts.index_version

    def video_ids(self) -> list[str]:
        return sorted(self._frames_of_video)

    def get_frame(self, frame_id: str) -> FrameRecord:
        record = self._by_frame_id.get(frame_id)
        if record is None:
            raise UnknownFrameError(frame_id)
        return record

    def frame_image_path(self, frame_id: str) -> Path:
        """Absolute image path, guaranteed inside the dataset root."""
        record = self.get_frame(frame_id)
        root = self.dataset_root.resolve()
        path = (root / record.image_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:  # defense in depth; validator also checks
            raise PermissionError(f"Image path for {frame_id} escapes the dataset root") from exc
        return path

    def timeline(self, video_id: str) -> list[FrameRecord]:
        frames = self._frames_of_video.get(video_id)
        if frames is None:
            raise UnknownVideoError(video_id)
        return frames

    def neighbors(self, frame_id: str, window: int = 5) -> list[FrameRecord]:
        """Frames of the same video around frame_id, ordered by timestamp."""
        record = self.get_frame(frame_id)
        frames = self.timeline(record.video_id)
        pos = next(i for i, r in enumerate(frames) if r.frame_id == frame_id)
        start = max(0, pos - window)
        return frames[start : pos + window + 1]

    def shot_context(self, frame_id: str) -> dict[str, list[FrameRecord]]:
        record = self.get_frame(frame_id)
        if not record.shot_id:
            return {"same_shot": [], "previous_shot": [], "next_shot": []}
        frames = self.timeline(record.video_id)
        shot_ids: list[str] = []
        for frame in frames:
            if frame.shot_id and frame.shot_id not in shot_ids:
                shot_ids.append(frame.shot_id)
        position = shot_ids.index(record.shot_id)
        previous = shot_ids[position - 1] if position > 0 else None
        following = shot_ids[position + 1] if position + 1 < len(shot_ids) else None
        return {
            "same_shot": [
                frame
                for frame in frames
                if frame.shot_id == record.shot_id and frame.frame_id != frame_id
            ],
            "previous_shot": [frame for frame in frames if previous and frame.shot_id == previous],
            "next_shot": [frame for frame in frames if following and frame.shot_id == following],
        }

    # -- search ----------------------------------------------------------

    def _video_filter_mask(self, video_ids: list[str]) -> np.ndarray:
        wanted = set(video_ids)
        mask = np.zeros(len(self.artifacts.id_map), dtype=bool)
        for vid in wanted:
            for record in self._frames_of_video.get(vid, []):
                mask[self._row_of[record.frame_id]] = True
        return mask

    def search(self, request: SearchRequest) -> list[SearchResult]:
        query_vec = self.text_provider.embed_texts([request.text])[0]

        mask: np.ndarray | None = None
        raw_filter = request.filters.get("video_ids")
        if raw_filter:
            if isinstance(raw_filter, str):
                raw_filter = [v.strip() for v in raw_filter.split(",") if v.strip()]
            if not isinstance(raw_filter, list):
                raise ValueError(
                    "filters.video_ids must be a list of video ids or a comma-separated string"
                )
            mask = self._video_filter_mask([str(v) for v in raw_filter])

        hits = self.index.search(query_vec, request.top_k, allowed_rows=mask)
        results: list[SearchResult] = []
        for rank, (frame_id, score) in enumerate(hits, start=1):
            record = self._by_frame_id[frame_id]
            results.append(
                SearchResult(
                    rank=rank,
                    final_score=score,
                    signal_scores={"visual": score},
                    video_id=record.video_id,
                    frame_id=record.frame_id,
                    frame_idx=record.frame_idx,
                    timestamp_ms=record.timestamp_ms,
                    image_url=f"/frames/{record.frame_id}/image",
                    evidence={
                        "keyframe_id": record.keyframe_id,
                        "image_path": record.image_path,
                        "pts": record.pts,
                        "title": record.metadata.get("title"),
                    },
                    index_version=self.index_version,
                )
            )
        return results

    # -- submission ------------------------------------------------------

    def submission_preview(
        self,
        query_id: str,
        task_type: str,
        frame_id: str,
        answer: str | None = None,
        confidence: float | None = None,
    ) -> CanonicalSubmission:
        record = self.get_frame(frame_id)
        return CanonicalSubmission(
            query_id=query_id,
            task_type=task_type,
            video_id=record.video_id,
            frame_id=record.frame_id,
            timestamp_ms=record.timestamp_ms,
            answer=answer,
            confidence=confidence,
            evidence={
                "keyframe_id": record.keyframe_id,
                "frame_idx": record.frame_idx,
                "image_path": record.image_path,
                "index_version": self.index_version,
            },
        )


def load_service(
    artifacts_dir: Path,
    dataset_root: Path | None = None,
    index_provider: str | None = None,
) -> RetrievalService:
    artifacts = load_index_artifacts(artifacts_dir)
    index = _make_index(index_provider) if index_provider else None
    return RetrievalService(artifacts, index=index, dataset_root=dataset_root)
