from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from hcmaic.api.dual_app import create_dual_app
from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.channel_contract import ChannelContract
from hcmaic.retrieval.dual_visual import (
    DualVisualService,
    load_dual_visual_artifacts,
    load_dual_visual_service,
)
from test_dual_visual import _FakeIndex, _FakeProvider, _write_artifact


class _FakeOCRAdapter:
    provider = "deepsolo-parseq-elasticsearch"
    revision = "det-r1+rec-r1"
    execution_status = "ENGINEERING_PROXY"
    quality_status = "UNVALIDATED_ON_HCMAIC"
    dataset_manifest_hash = "dataset-hash"
    artifact_hash = "manifest-hash"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def channel_contract(self) -> ChannelContract:
        return ChannelContract(
            channel="ocr",
            provider=self.provider,
            revision=self.revision,
            execution_status=self.execution_status,
            quality_status=self.quality_status,
            dataset_manifest_hash=self.dataset_manifest_hash,
            artifact_hash=self.artifact_hash,
            status="ready",
            evidence={"index": "hcmaic_ocr_v1"},
        )

    def search(self, query: str, top_k: int = 100) -> list[ChannelHit]:
        self.queries.append(query)
        return [
            ChannelHit(
                entity_id="L21_V001:10031",
                video_id="L21_V001",
                timestamp_ms=334366,
                modality="ocr",
                score=4.2,
                rank=1,
                provider=self.provider,
                evidence_text="Đỗ",
                frame_uid="L21_V001:10031",
                video_filename="L21_V001.mp4",
                source_frame_idx=10031,
                evidence={
                    "crop_uid": "crop-ocr-1",
                    "matched_crops": [{"crop_uid": "crop-ocr-1", "text_nfc": "Đỗ"}],
                },
            )
        ][:top_k]


def _service(root: Path, ocr_adapter: _FakeOCRAdapter) -> DualVisualService:
    artifacts = load_dual_visual_artifacts(
        root,
        index_loader={
            "siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1]),
            "qwen": lambda _: _FakeIndex([0.95, 0.7], [0, 1]),
        },
    )
    return DualVisualService(
        artifacts,
        siglip_provider=_FakeProvider(),
        qwen_provider=_FakeProvider(),
        ocr_adapter=ocr_adapter,
    )


def test_staged_localhost_ui_executes_ocr_and_preserves_crop_evidence(tmp_path: Path):
    root = tmp_path / "artifact"
    _write_artifact(root)
    ocr = _FakeOCRAdapter()
    client = TestClient(create_dual_app(service=_service(root, ocr)))

    response = client.post(
        "/search/stages",
        json={
            "stages": [
                {
                    "stage_id": "S1",
                    "channels": {"ocr": "do"},
                    "top_k": 1,
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert ocr.queries == ["do"]
    assert body["executed_channels"] == ["ocr"]
    assert body["stage_status"]["S1"]["ocr"]["status"] == "ready"
    assert body["stage_results"]["S1"][0]["frame_uid"] == "L21_V001:10031"
    evidence = body["stage_results"]["S1"][0]["evidence"][0]
    assert evidence["channel"] == "ocr"
    assert evidence["metadata"]["crop_uid"] == "crop-ocr-1"


def test_staged_localhost_ui_fuses_visual_and_ocr_channels(tmp_path: Path):
    root = tmp_path / "artifact"
    _write_artifact(root)
    ocr = _FakeOCRAdapter()
    client = TestClient(create_dual_app(service=_service(root, ocr)))

    response = client.post(
        "/search/stages",
        json={
            "stages": [
                {
                    "stage_id": "S1",
                    "channels": {"text": "a slide", "ocr": "do"},
                    "top_k": 1,
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert ocr.queries == ["do"]
    assert body["executed_channels"] == ["siglip2", "ocr"]
    assert {item["channel"] for item in body["stage_results"]["S1"][0]["evidence"]} == {
        "siglip2",
        "ocr",
    }


def test_dual_loader_attaches_explicit_ocr_elasticsearch_config(tmp_path: Path):
    root = tmp_path / "artifact"
    _write_artifact(root)
    dual_manifest = json.loads((root / "dual_merge_manifest.json").read_text(encoding="utf-8"))
    dual_manifest["models"]["siglip2"]["revision"] = "fake-v1"
    (root / "dual_merge_manifest.json").write_text(json.dumps(dual_manifest), encoding="utf-8")
    index_manifest = json.loads((root / "siglip2_index_manifest.json").read_text(encoding="utf-8"))
    index_manifest["revision"] = "fake-v1"
    (root / "siglip2_index_manifest.json").write_text(json.dumps(index_manifest), encoding="utf-8")
    manifest = tmp_path / "ocr_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "hcmaic-dstext-parseq-ocr-merged-v1",
                "quality_status": "UNVALIDATED_ON_HCMAIC",
                "dataset_manifest_hash": "dataset-hash",
                "model_contract": {
                    "detector": {"model": "DeepSolo", "revision": "det-r1"},
                    "recognizer": {"model": "PARSeq", "revision": "rec-r1"},
                },
            }
        ),
        encoding="utf-8",
    )
    fake_client = SimpleNamespace(indices=SimpleNamespace(exists=lambda **_: True))

    service = load_dual_visual_service(
        root,
        visual_indexes=("siglip2",),
        index_loader={"siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1])},
        provider_loader={"siglip2": lambda **_: _FakeProvider()},
        ocr_es_url="http://127.0.0.1:9200",
        ocr_es_index="hcmaic_ocr_v1",
        ocr_es_manifest=manifest,
        ocr_es_client=fake_client,
        allow_engineering_proxy=True,
    )

    status = service.channel_status()["ocr"]
    assert status["status"] == "ready"
    assert status["index"] == "hcmaic_ocr_v1"
    assert status["quality_status"] == "UNVALIDATED_ON_HCMAIC"


def test_dual_loader_accepts_successful_ocr_snapshot_manifest(tmp_path: Path):
    root = tmp_path / "artifact"
    _write_artifact(root)
    dual_manifest = json.loads((root / "dual_merge_manifest.json").read_text(encoding="utf-8"))
    dual_manifest["models"]["siglip2"]["revision"] = "fake-v1"
    (root / "dual_merge_manifest.json").write_text(
        json.dumps(dual_manifest), encoding="utf-8"
    )
    index_manifest = json.loads(
        (root / "siglip2_index_manifest.json").read_text(encoding="utf-8")
    )
    index_manifest["revision"] = "fake-v1"
    (root / "siglip2_index_manifest.json").write_text(
        json.dumps(index_manifest), encoding="utf-8"
    )
    snapshot_manifest = tmp_path / "ocr_snapshot_manifest.json"
    snapshot_manifest.write_text(
        json.dumps(
            {
                "schema": "hcmaic-ocr-elasticsearch-snapshot-v1",
                "state": "SUCCESS",
                "snapshot": "hcmaic_ocr_v1_20260821",
                "indices": ["hcmaic_ocr_v1"],
                "shards": {"total": 1, "failed": 0, "successful": 1},
                "source_index": "hcmaic_ocr_v1",
                "source_doc_count": 4_320_073,
                "declared_row_count": 4_320_089,
                "count_delta": -16,
                "source_index_manifest_sha256": "a" * 64,
                "source_ocr_manifest_sha256": "b" * 64,
                "quality_status": "UNVALIDATED",
                "provenance_class": "ENGINEERING_PROXY",
            }
        ),
        encoding="utf-8",
    )
    fake_client = SimpleNamespace(indices=SimpleNamespace(exists=lambda **_: True))

    service = load_dual_visual_service(
        root,
        visual_indexes=("siglip2",),
        index_loader={"siglip2": lambda _: _FakeIndex([0.9, 0.8], [0, 1])},
        provider_loader={"siglip2": lambda **_: _FakeProvider()},
        ocr_es_url="http://127.0.0.1:9200",
        ocr_es_index="hcmaic_ocr_v1",
        ocr_es_manifest=snapshot_manifest,
        ocr_es_client=fake_client,
        allow_engineering_proxy=True,
    )

    assert service.channel_status()["ocr"]["status"] == "ready"
    assert service.ocr_adapter is not None
    assert service.ocr_adapter.manifest["format"] == (
        "hcmaic-dstext-parseq-ocr-merged-v1"
    )
    assert service.ocr_adapter.manifest["snapshot_provenance"]["count_delta"] == -16
    contract = service.ocr_adapter.channel_contract()
    assert contract.artifact_hash
    assert contract.evidence["snapshot_provenance"]["source_index_manifest_sha256"] == (
        "a" * 64
    )
