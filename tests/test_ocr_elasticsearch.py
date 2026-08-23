"""Elasticsearch mapping, bulk and query contract tests without a live ES server."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from hcmaic.retrieval.ocr_elasticsearch import (
    build_ocr_index_mapping,
    bulk_index_ocr,
    ElasticsearchOCRError,
    fold_ocr_text,
    make_elasticsearch_client,
    ocr_row_to_document,
    search_ocr,
)


class _FakeIndices:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self.refreshed: list[str] = []

    def exists(self, *, index: str) -> bool:
        return False

    def create(self, *, index: str, settings: dict[str, Any], mappings: dict[str, Any]) -> None:
        self.created.append((index, {"settings": settings, "mappings": mappings}))

    def delete(self, *, index: str) -> None:
        self.deleted.append(index)

    def refresh(self, *, index: str) -> None:
        self.refreshed.append(index)


class _FakeClient:
    def __init__(self) -> None:
        self.indices = _FakeIndices()
        self.bulk_calls: list[list[Any]] = []

    def bulk(self, *, operations: list[Any], refresh: bool = False) -> dict[str, Any]:
        self.bulk_calls.append(operations)
        item_count = len(operations) // 2
        return {
            "errors": False,
            "items": [{"index": {"status": 201}} for _ in range(item_count)],
        }


def _row(crop_uid: str, text: str) -> dict[str, Any]:
    return {
        "crop_uid": crop_uid,
        "frame_uid": "V0:1",
        "video_id": "V0",
        "source_frame_idx": 1,
        "timestamp_ms": 40,
        "ocr_text_raw": text,
        "ocr_text_nfc": text,
        "ocr_text_folded": fold_ocr_text(text),
        "rec_score": 0.8,
        "det_score": 0.9,
        "confidence_status": "OK",
        "detector_model": "DeepSolo",
        "detector_revision": "d1",
        "recognizer_model": "PARSeq",
        "recognizer_revision": "r1",
        "bbox": [1, 2, 3, 4],
        "polygon": [[1, 2], [3, 4]],
        "source_shard_id": "shard_0000",
        "source_manifest_sha256": "manifest-hash",
        "quality_status": "UNVALIDATED_ON_HCMAIC",
        "execution_status": "ENGINEERING_ARTIFACT_COMPLETE",
    }


def test_mapping_and_text_folding_preserve_vietnamese_search_fields():
    mapping = build_ocr_index_mapping()
    assert mapping["settings"]["analysis"]["filter"]["hcmaic_char_ngrams"]["min_gram"] == 2
    props = mapping["mappings"]["properties"]
    assert props["crop_uid"]["type"] == "keyword"
    assert props["frame_uid"]["type"] == "keyword"
    assert props["text_nfc"]["fields"]["folded"]["analyzer"] == "hcmaic_folded"
    assert fold_ocr_text("Đỗ Sản phẩm") == "do san pham"


def test_bulk_index_is_idempotent_by_crop_uid_and_skips_empty(tmp_path: Path):
    rows = [_row("crop-1", "Đỗ"), _row("crop-empty", "")]
    (tmp_path / "ocr_lines.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (tmp_path / "ocr_manifest.json").write_text(
        json.dumps(
            {
                "format": "hcmaic-dstext-parseq-ocr-merged-v1",
                "quality_status": "UNVALIDATED_ON_HCMAIC",
            }
        ),
        encoding="utf-8",
    )
    client = _FakeClient()

    stats = bulk_index_ocr(
        client,
        tmp_path,
        index_name="hcmaic-ocr-test",
        batch_size=1,
        refresh=True,
    )

    assert stats["read_rows"] == 2
    assert stats["indexed_rows"] == 1
    assert stats["skipped_empty"] == 1
    assert client.indices.created[0][0] == "hcmaic-ocr-test"
    assert client.indices.refreshed == ["hcmaic-ocr-test"]
    assert client.bulk_calls[0][0] == {"index": {"_index": "hcmaic-ocr-test", "_id": "crop-1"}}


def test_bulk_dry_run_batches_without_needing_a_client(tmp_path: Path):
    row = _row("crop-dry", "Đỗ")
    (tmp_path / "ocr_lines.jsonl").write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (tmp_path / "ocr_manifest.json").write_text(
        json.dumps({"format": "hcmaic-dstext-parseq-ocr-merged-v1"}), encoding="utf-8"
    )

    stats = bulk_index_ocr(
        None,
        tmp_path,
        index_name="hcmaic-ocr-dry",
        batch_size=1,
        dry_run=True,
    )

    assert stats["dry_run"] is True
    assert stats["indexed_rows"] == 1
    assert stats["bulk_batches"] == 1


def test_search_uses_multifield_query_and_collapses_to_frame():
    client = _FakeClient()

    def search(*, index: str, body: dict[str, Any]) -> dict[str, Any]:
        client.last_search = (index, body)
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 4.5,
                        "_source": {
                            **_row("crop-1", "Đỗ"),
                            "video_filename": "V0.mp4",
                        },
                        "inner_hits": {
                            "ocr_crops": {"hits": {"hits": [{"_source": _row("crop-1", "Đỗ")}]}}
                        },
                    }
                ]
            }
        }

    client.search = search  # type: ignore[attr-defined]
    hits = search_ocr(client, "hcmaic-ocr-test", "do", top_k=5, video_ids=["V0"])

    body = client.last_search[1]
    assert body["query"]["bool"]["must"][0]["multi_match"]["query"] == "do"
    assert body["collapse"]["field"] == "frame_uid"
    assert body["query"]["bool"]["filter"] == [{"terms": {"video_id": ["V0"]}}]
    assert hits[0].frame_uid == "V0:1"
    assert hits[0].entity_id == "V0:1"
    assert hits[0].evidence["crop_uid"] == "crop-1"


def test_document_uses_crop_uid_as_es_id():
    doc = ocr_row_to_document(_row("crop-9", "Đỗ"), {"quality_status": "UNVALIDATED"})
    assert doc is not None
    assert doc["_id"] == "crop-9"
    assert doc["document"]["text_folded"] == "do"
    assert doc["document"]["frame_uid"] == "V0:1"


def test_anonymous_elasticsearch_is_opt_in_and_local_http_only():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ElasticsearchOCRError):
            make_elasticsearch_client("http://127.0.0.1:9200")
        client = make_elasticsearch_client(
            "http://127.0.0.1:9200", allow_anonymous_local=True
        )
        node_config = next(iter(client.transport.node_pool._all_nodes))
        assert node_config.host == "127.0.0.1"
