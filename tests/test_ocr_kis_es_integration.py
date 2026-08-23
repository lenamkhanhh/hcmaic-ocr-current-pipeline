"""KIS runtime wiring tests for the Elasticsearch OCR channel."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import hcmaic.runtime.kis as kis_runtime
from hcmaic.cli.main import build_parser
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.retrieval.ocr_elasticsearch import (
    ElasticsearchOCRChannel,
    ElasticsearchOCRError,
    load_ocr_manifest,
    validate_ocr_index,
)


class _Provider(EmbeddingProvider):
    name = "tiny-visual"
    version = "tiny-visual-v1"

    @property
    def dimension(self) -> int:
        return 3

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return l2_normalize(np.ones((len(paths), 3), dtype=np.float32))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.ones((len(texts), 3), dtype=np.float32))


def _index() -> SimpleNamespace:
    return SimpleNamespace(
        embeddings=np.zeros((1, 3), dtype=np.float32),
        provider_info={"provider": "tiny-visual", "version": "tiny-visual-v1"},
        index_manifest={"dataset_manifest_hash": "raw-hash"},
        size=1,
        dimension=3,
    )


def _write_manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": "hcmaic-dstext-parseq-ocr-merged-v1",
                "quality_status": "UNVALIDATED_ON_HCMAIC",
                "dataset_manifest_hash": "raw-hash",
                "model_contract": {
                    "detector": {"model": "DeepSolo", "revision": "det-r1"},
                    "recognizer": {"model": "PARSeq", "revision": "rec-r1"},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_kis_loader_attaches_elasticsearch_ocr_and_preserves_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _write_manifest(tmp_path / "ocr_manifest.json")
    fake_client = SimpleNamespace(
        indices=SimpleNamespace(exists=lambda **_: True),
    )
    calls: dict[str, object] = {}

    class _FakeChannel:
        provider = "deepsolo-parseq-elasticsearch"
        revision = "det-r1+rec-r1"
        execution_status = "ENGINEERING_PROXY"
        quality_status = "UNVALIDATED_ON_HCMAIC"
        dataset_manifest_hash = "raw-hash"
        artifact_hash = "merged-manifest-hash"

        def __init__(self, client, index_name, manifest_payload, **kwargs):
            calls["channel"] = (client, index_name, manifest_payload, kwargs)

    monkeypatch.setattr(kis_runtime, "load_skillpixel_index", lambda _: _index())
    monkeypatch.setattr(
        kis_runtime,
        "get_real_visual_provider",
        lambda **_: (_Provider(), {"requested_provider": "siglip2"}),
    )
    monkeypatch.setattr(kis_runtime, "make_elasticsearch_client", lambda *_args, **_: fake_client)
    monkeypatch.setattr(kis_runtime, "validate_ocr_index", lambda *_: None)
    monkeypatch.setattr(kis_runtime, "ElasticsearchOCRChannel", _FakeChannel)

    runtime = kis_runtime.load_kis_runtime(
        tmp_path / "index",
        provider="siglip2",
        ocr_es_url="https://es.example.test",
        ocr_es_index="hcmaic-ocr",
        ocr_es_manifest=manifest,
        allow_engineering_proxy=True,
    )

    assert runtime.orchestrator.optional_channels["ocr"].provider == (
        "deepsolo-parseq-elasticsearch"
    )
    assert runtime.channel_status["ocr"]["status"] == "ready"
    assert runtime.channel_status["ocr"]["backend"] == "elasticsearch"
    assert runtime.channel_status["ocr"]["index"] == "hcmaic-ocr"
    assert runtime.channel_contracts["ocr"]["dataset_manifest_hash"] == "raw-hash"
    assert runtime.channel_contracts["ocr"]["artifact_hash"] == "merged-manifest-hash"
    assert calls["channel"][1] == "hcmaic-ocr"


def test_kis_loader_keeps_es_ocr_disabled_without_engineering_proxy(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _write_manifest(tmp_path / "ocr_manifest.json")
    monkeypatch.setattr(kis_runtime, "load_skillpixel_index", lambda _: _index())
    monkeypatch.setattr(
        kis_runtime,
        "get_real_visual_provider",
        lambda **_: (_Provider(), {"requested_provider": "siglip2"}),
    )

    runtime = kis_runtime.load_kis_runtime(
        tmp_path / "index",
        provider="siglip2",
        ocr_es_url="https://es.example.test",
        ocr_es_index="hcmaic-ocr",
        ocr_es_manifest=manifest,
    )

    assert runtime.channel_status["ocr"]["status"] == "unavailable"
    assert runtime.channel_status["ocr"]["reason"] == "engineering_proxy_disabled_by_policy"
    assert "ocr" not in runtime.orchestrator.optional_channels


def test_kis_cli_exposes_es_ocr_configuration() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "search-kis",
            "--index",
            "index",
            "--task",
            "TKIS",
            "--query",
            "hello",
            "--ocr-es-url",
            "https://es.example.test",
            "--ocr-es-index",
            "hcmaic-ocr",
            "--ocr-es-manifest",
            "ocr_manifest.json",
            "--ocr-es-api-key-env",
            "HCMAIC_ES_KEY",
            "--ocr-es-exclude-low-conf",
            "--allow-engineering-proxy",
        ]
    )

    assert args.ocr_es_url == "https://es.example.test"
    assert args.ocr_es_index == "hcmaic-ocr"
    assert args.ocr_es_manifest == "ocr_manifest.json"
    assert args.ocr_es_api_key_env == "HCMAIC_ES_KEY"
    assert args.ocr_es_include_low_conf is False


def test_es_channel_adds_manifest_provenance_and_validates_index(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "ocr_manifest.json")
    manifest, manifest_hash = load_ocr_manifest(manifest_path)
    calls: list[dict[str, object]] = []

    class _Indices:
        def exists(self, **kwargs):
            calls.append(kwargs)
            return True

    class _Client:
        indices = _Indices()

        def search(self, **kwargs):
            return {
                "hits": {
                    "hits": [
                        {
                            "_score": 1.2,
                            "_source": {
                                "frame_uid": "V1:0",
                                "video_id": "V1",
                                "video_filename": "V1.mp4",
                                "source_frame_idx": 0,
                                "timestamp_ms": 0,
                                "text_nfc": "Đỗ",
                                "quality_status": "UNVALIDATED_ON_HCMAIC",
                            },
                            "inner_hits": {
                                "ocr_crops": {
                                    "hits": {
                                        "hits": [
                                            {
                                                "_source": {
                                                    "crop_uid": "crop-0",
                                                    "text_nfc": "Đỗ",
                                                    "rec_score": 0.9,
                                                }
                                            }
                                        ]
                                    }
                                }
                            },
                        }
                    ]
                }
            }

    client = _Client()
    validate_ocr_index(client, "hcmaic-ocr")
    channel = ElasticsearchOCRChannel(
        client,
        "hcmaic-ocr",
        manifest,
        manifest_sha256=manifest_hash,
    )
    hits = channel.search("do", top_k=1)

    assert calls == [{"index": "hcmaic-ocr"}]
    assert hits[0].provider == "deepsolo-parseq-elasticsearch"
    assert hits[0].evidence["raw_provenance"]["artifact_hash"] == manifest_hash
    assert hits[0].evidence["raw_provenance"]["index"] == "hcmaic-ocr"


def test_es_channel_wraps_backend_failures_without_exposing_error_payload() -> None:
    class _Client:
        def search(self, **kwargs):
            del kwargs
            raise RuntimeError("backend details")

    channel = ElasticsearchOCRChannel(
        _Client(),
        "hcmaic-ocr",
        {"model_contract": {}},
    )

    with pytest.raises(ElasticsearchOCRError, match="search failed") as error:
        channel.search("query")
    assert "backend details" not in str(error.value)
