from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hcmaic.embedding.base import EmbeddingProvider
from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.channel_contract import ChannelContract
from hcmaic.retrieval.dual_visual import (
    DEFAULT_DUAL_SIGLIP2_MODEL,
    DualVisualService,
    emit_local_runtime_manifest,
    load_dual_visual_artifacts,
)


def test_dual_runtime_siglip_fallback_is_p14() -> None:
    assert DEFAULT_DUAL_SIGLIP2_MODEL == "google/siglip2-so400m-patch14-384"


class _FakeIndex:
    def __init__(self, scores: list[float], rows: list[int], dimension: int = 2) -> None:
        self.d = dimension
        self.ntotal = len(rows)
        self._scores = np.asarray(scores, dtype=np.float32)
        self._rows = np.asarray(rows, dtype=np.int64)

    def search(self, queries: np.ndarray, top_k: int):
        count = min(top_k, self.ntotal)
        return (
            np.tile(self._scores[:count], (len(queries), 1)),
            np.tile(self._rows[:count], (len(queries), 1)),
        )


class _CountingSearchIndex(_FakeIndex):
    def __init__(self, scores: list[float], rows: list[int], dimension: int = 2) -> None:
        super().__init__(scores, rows, dimension)
        self.search_calls = 0

    def search(self, queries: np.ndarray, top_k: int):
        self.search_calls += 1
        return super().search(queries, top_k)


class _FakeProvider(EmbeddingProvider):
    name = "fake"
    version = "fake-v1"

    @property
    def dimension(self) -> int:
        return 2

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(paths), 1))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class _CountingFakeProvider(_FakeProvider):
    def __init__(self) -> None:
        self.text_calls = 0

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.text_calls += 1
        return super().embed_texts(texts)


class _FakeObjectAdapter:
    provider = "rfdetr_coco"
    revision = "rfdetr-xlarge@" + ("r" * 64)
    execution_status = "ENGINEERING_PROXY"
    quality_status = "UNVALIDATED"
    dataset_manifest_hash = None
    artifact_hash = "o" * 64

    def __init__(self) -> None:
        self.queries: list[str] = []

    def channel_contract(self) -> ChannelContract:
        return ChannelContract(
            channel="object",
            provider=self.provider,
            revision=self.revision,
            execution_status=self.execution_status,
            quality_status=self.quality_status,
            artifact_hash=self.artifact_hash,
            status="ready",
            evidence={"index_version": "fake-object-v1"},
        )

    def object_aliases(self) -> dict[str, object]:
        return {
            "status": "ready",
            "version": "fake-object-alias-v1",
            "aliases": [{"alias": "people", "label": "person"}],
            "labels": ["person"],
        }

    def search(self, query: str, top_k: int = 100) -> list[ChannelHit]:
        self.queries.append(query)
        if query.strip().casefold() != "person":
            return []
        return [
            ChannelHit(
                entity_id="L21_V001:10031",
                video_id="L21_V001",
                timestamp_ms=334366,
                modality="object",
                score=0.92,
                rank=1,
                provider=self.provider,
                evidence_text="person",
                frame_uid="L21_V001:10031",
                video_filename="L21_V001.mp4",
                source_frame_idx=10031,
                evidence={
                    "label_raw": "person",
                    "max_confidence": 0.92,
                    "instance_count": 1,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                },
            )
        ][:top_k]


def test_default_multichannel_fusion_is_harmonic_without_changing_visual_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_artifact(root)
    artifacts = load_dual_visual_artifacts(
        root,
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.95, 0.7], [0, 1]),
        },
    )
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
        object_adapter=_FakeObjectAdapter(),
    )

    visual_only = service.search_text(
        "visual-only",
        "a slide",
        top_k=1,
        visual_indexes=("siglip2",),
    )
    combined = service.search_text(
        "visual-object",
        "a slide",
        top_k=1,
        visual_indexes=("siglip2",),
        object_query="person",
    )

    assert visual_only[0].fused_score == pytest.approx(0.9)
    assert combined[0].frame_uid == "L21_V001:10031"
    assert combined[0].channel_scores == {
        "siglip2": pytest.approx(0.9),
        "object": pytest.approx(0.92),
    }
    assert combined[0].fused_score == pytest.approx(1.0)
    assert service.health()["fusion"]["method"] == "harmonic"
    assert service.health()["fusion"]["artifact_method"] == "rrf"


def _write_artifact(root: Path, *, mismatch: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "feature_row": 0,
            "frame_uid": "L21_V001:10031",
            "keyframe_path": "data/processed/keyframes/L21/L21_V001/L21_V001_F000010031.jpg",
            "shot_id": "L21_V001_S000099",
            "source_frame_idx": 10031,
            "timestamp_ms": 334366,
            "video_id": "L21_V001",
        },
        {
            "feature_row": 1,
            "frame_uid": "L21_V001:10050",
            "keyframe_path": "data/processed/keyframes/L21/L21_V001/L21_V001_F000010050.jpg",
            "shot_id": "L21_V001_S000100",
            "source_frame_idx": 10050,
            "timestamp_ms": 335000,
            "video_id": "L21_V001",
        },
    ]
    (root / "frame_catalog.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    id_rows = [
        {"faiss_row": i, "feature_row": i, "frame_uid": row["frame_uid"]}
        for i, row in enumerate(rows)
    ]
    if mismatch:
        id_rows[1]["frame_uid"] = "L21_V001:WRONG"
    for name in ("siglip2", "qwen"):
        (root / f"{name}.index").write_bytes(b"stub-index")
        (root / f"{name}_id_map.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in id_rows), encoding="utf-8"
        )
        (root / f"{name}_index_manifest.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "identity_key": "frame_uid",
                    "index_type": "IndexFlatIP",
                    "dimension": 2,
                    "ntotal": 2,
                }
            ),
            encoding="utf-8",
        )
    (root / "dual_merge_manifest.json").write_text(
        json.dumps(
            {
                "status": "DUAL_MERGE_COMPLETE",
                "execution_status": "COMPLETE",
                "quality_status": "UNVALIDATED",
                "identity_key": "frame_uid",
                "row_count": 2,
                "unique_frame_uid": 2,
                "models": {
                    "siglip2": {"dimension": 2},
                    "qwen": {"dimension": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "fusion_contract.json").write_text(
        json.dumps({"method": "rrf", "rank_constant": 60, "weights": {"qwen": 1, "siglip2": 1}}),
        encoding="utf-8",
    )


def test_loader_rejects_cross_channel_identity_mismatch(tmp_path: Path):
    _write_artifact(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="frame_uid mapping mismatch"):
        load_dual_visual_artifacts(
            tmp_path,
            index_loader={
                "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
                "qwen": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            },
        )


def test_dual_search_fuses_by_frame_uid_and_preserves_evidence(tmp_path: Path):
    _write_artifact(tmp_path)
    artifacts = load_dual_visual_artifacts(
        tmp_path,
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.95, 0.7], [0, 1]),
        },
    )
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
    )
    results = service.search_text("q-1", "a slide", top_k=2)
    assert results[0].frame_uid == "L21_V001:10031"
    assert set(results[0].channel_scores) == {"siglip2", "qwen"}
    assert {item.channel for item in results[0].evidence} == {"qwen", "siglip2"}
    assert results[0].quality_status == "UNVALIDATED_ON_HCMAIC"


def test_internal_candidate_expansion_allows_large_pool_without_changing_public_limit(
    tmp_path: Path,
):
    _write_artifact(tmp_path)
    artifacts = load_dual_visual_artifacts(
        tmp_path,
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
        },
    )
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
        visual_indexes=("siglip2",),
    )

    with pytest.raises(ValueError, match="500"):
        service.search_text("public-bound", "a slide", top_k=501)

    expanded = service.search_text(
        "internal-expanded",
        "a slide",
        top_k=501,
        allow_large_top_k=True,
    )
    assert len(expanded) == 2


def test_object_channel_is_not_queried_or_fused_by_default(tmp_path: Path):
    _write_artifact(tmp_path)
    artifacts = load_dual_visual_artifacts(
        tmp_path,
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [1, 0]),
            "qwen": lambda _: _FakeIndex([0.95, 0.7], [1, 0]),
        },
    )
    adapter = _FakeObjectAdapter()
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
        object_adapter=adapter,
    )

    results = service.search_text("q-default", "a slide", top_k=2)

    assert adapter.queries == []
    assert all("object" not in result.channel_scores for result in results)
    assert service.channel_status()["object"]["status"] == "ready"


def test_object_channel_is_late_fused_by_frame_uid_when_explicitly_queried(tmp_path: Path):
    _write_artifact(tmp_path)
    artifacts = load_dual_visual_artifacts(
        tmp_path,
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.95, 0.7], [0, 1]),
        },
    )
    adapter = _FakeObjectAdapter()
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
        object_adapter=adapter,
    )

    results = service.search_text("q-object", "a slide", top_k=2, object_query="person")

    assert adapter.queries == ["person"]
    assert results[0].frame_uid == "L21_V001:10031"
    assert "object" in results[0].channel_scores
    object_evidence = next(item for item in results[0].evidence if item.channel == "object")
    assert object_evidence.metadata["label_raw"] == "person"
    assert object_evidence.metadata["max_confidence"] == pytest.approx(0.92)
    assert object_evidence.metadata["instance_count"] == 1


def test_object_only_search_returns_object_evidence_without_visual_embedding(tmp_path: Path):
    _write_artifact(tmp_path)
    artifacts = load_dual_visual_artifacts(
        tmp_path,
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [1, 0]),
            "qwen": lambda _: _FakeIndex([0.95, 0.7], [1, 0]),
        },
    )
    adapter = _FakeObjectAdapter()
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
        object_adapter=adapter,
    )

    results = service.search_object("q-object-only", "person", top_k=1)

    assert adapter.queries == ["person"]
    assert results[0].channel_scores == {"object": pytest.approx(0.92)}
    assert results[0].executed_channels == ("object",)
    assert results[0].evidence[0].channel == "object"
    assert results[0].evidence[0].metadata["label_raw"] == "person"


def test_video_catalog_views_are_indexed_for_repeated_media_requests(tmp_path: Path):
    _write_artifact(tmp_path / "artifact")
    artifacts = load_dual_visual_artifacts(
        tmp_path / "artifact",
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
        },
    )
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
    )

    assert service.has_video_id("L21_V001") is True
    assert service.has_video_id("missing") is False
    assert [row["frame_uid"] for row in service.timeline("L21_V001")] == [
        "L21_V001:10031",
        "L21_V001:10050",
    ]
    assert [row["frame_uid"] for row in service.timeline_window(
        "L21_V001", "L21_V001:10050", 1
    )] == [
        "L21_V001:10031",
        "L21_V001:10050",
    ]

    # Media endpoints must not rescan the mutable catalog after service startup.
    service._artifacts.catalog.clear()
    assert service.has_video_id("L21_V001") is True
    assert service._allowed_uids(["L21_V001"]) == {
        "L21_V001:10031",
        "L21_V001:10050",
    }
    assert len(service.timeline("L21_V001")) == 2
    assert (
        service.timeline_window("L21_V001", "L21_V001:10031", 1)[0]["frame_uid"]
        == "L21_V001:10031"
    )


def test_repeated_text_query_reuses_bounded_embedding_cache(tmp_path: Path):
    _write_artifact(tmp_path / "artifact")
    artifacts = load_dual_visual_artifacts(
        tmp_path / "artifact",
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
        },
    )
    siglip = _CountingFakeProvider()
    qwen = _CountingFakeProvider()
    service = DualVisualService(artifacts, siglip_provider=siglip, qwen_provider=qwen)

    service.search_text("query-1", "same query", top_k=1)
    service.search_text("query-2", "same query", top_k=1)

    assert siglip.text_calls == 1
    assert qwen.text_calls == 1


def test_repeated_exact_text_search_reuses_bounded_result_cache(tmp_path: Path):
    _write_artifact(tmp_path / "artifact")
    siglip_index = _CountingSearchIndex([0.9, 0.8], [0, 1])
    qwen_index = _CountingSearchIndex([0.9, 0.8], [0, 1])
    artifacts = load_dual_visual_artifacts(
        tmp_path / "artifact",
        index_loader={
            "siglip2": lambda _: siglip_index,
            "qwen": lambda _: qwen_index,
        },
    )
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
    )

    first = service.search_text("query-1", "same query", top_k=1)
    second = service.search_text("query-2", "same query", top_k=1)

    assert siglip_index.search_calls == 1
    assert qwen_index.search_calls == 1
    assert first[0].frame_uid == second[0].frame_uid
    assert second[0].query_id == "query-2"


def test_frame_image_path_stays_inside_configured_root(tmp_path: Path):
    _write_artifact(tmp_path / "artifact")
    artifacts = load_dual_visual_artifacts(
        tmp_path / "artifact",
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
        },
    )
    service = DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
        image_root=tmp_path / "images",
    )
    with pytest.raises(FileNotFoundError):
        service.frame_image_path("L21_V001:10031")

    service._artifacts.catalog[0]["keyframe_path"] = "../../outside.jpg"
    with pytest.raises(PermissionError):
        service.frame_image_path("L21_V001:10031")


def test_emit_manifest_explicitly_keeps_quality_unvalidated(tmp_path: Path):
    _write_artifact(tmp_path / "artifact")
    artifacts = load_dual_visual_artifacts(
        tmp_path / "artifact",
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
        },
    )
    out = emit_local_runtime_manifest(artifacts, tmp_path / "runtime.json")
    assert out["execution_status"] == "READY_LOCAL_PRECHECK"
    assert out["quality_status"] == "UNVALIDATED"
    assert json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))["row_count"] == 2
