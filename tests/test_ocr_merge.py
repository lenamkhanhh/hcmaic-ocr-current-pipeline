"""Streaming merger tests for crop-level DeepSolo/PARSeq OCR artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmaic.ingestion.ocr_merge import OCRMergeError, merge_ocr_shards


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _row(frame_uid: str, video_id: str, source_frame_idx: int, crop_uid: str, text: str) -> dict:
    return {
        "crop_uid": crop_uid,
        "frame_uid": frame_uid,
        "video_id": video_id,
        "source_frame_idx": source_frame_idx,
        "timestamp_ms": source_frame_idx * 40,
        "shot_id": f"shot-{source_frame_idx}",
        "line_index": 0,
        "polygon": [[1, 2], [30, 2], [30, 12], [1, 12]],
        "bbox": [1, 2, 30, 12],
        "det_score": 0.91,
        "ocr_text_raw": text,
        "ocr_text_nfc": text,
        "ocr_text_folded": "do" if text == "Đỗ" else text.casefold(),
        "rec_score": 0.88,
        "confidence_status": "OK",
        "detector_model": "DeepSolo DS text official",
        "detector_revision": "det-rev-1",
        "recognizer_model": "PARSeq Vietnamese fine-tune",
        "recognizer_revision": "rec-rev-1",
        "ocr_candidates": [{"text": text, "score": 0.88}],
    }


def _make_shard(root: Path, shard_id: str, rows: list[dict], *, parquet: bool = False) -> Path:
    root.mkdir(parents=True)
    lines_path = root / ("ocr_lines.parquet" if parquet else "ocr_lines.jsonl")
    if parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), lines_path)
    else:
        lines_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    (root / "detection_status.jsonl").write_text(
        "".join(
            json.dumps(
                {"frame_uid": row["frame_uid"], "status": "OK", "line_count": 1},
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    ledger = {
        "status": "CLOSED",
        "failure_count": 1,
        "unresolved_count": 0,
        "counts_by_type": {"NO_TEXT": 1},
        "failures": [
            {
                "failure_type": "NO_TEXT",
                "frame_uid": rows[0]["frame_uid"],
                "crop_uid": None,
                "resolved": True,
            }
        ],
    }
    _write_json(root / "failure_ledger.json", ledger)
    manifest = {
        "status": "ENGINEERING_ARTIFACT_COMPLETE",
        "execution_status": "COMPLETE",
        "provenance_class": "ENGINEERING_PROXY",
        "quality_status": "UNVALIDATED",
        "target_shard_id": shard_id,
        "full_shard": True,
        "expected_frame_count": len({row["frame_uid"] for row in rows}),
        "selection_count": len({row["frame_uid"] for row in rows}),
        "crop_count": len(rows),
        "recognition_count": len(rows),
        "selection_sha256": f"selection-{shard_id}",
        "identity": (
            "frame_uid=video_id:source_frame_idx; detector-scoped crop_uid; faiss_row not identity"
        ),
        "detector": {
            "model": "DeepSolo DS text official",
            "revision": "det-rev-1",
            "weight_sha256": "det-weight-1",
        },
        "recognizer": {
            "model": "PARSeq Vietnamese fine-tune",
            "revision": "rec-rev-1",
            "checkpoint_sha256": "rec-weight-1",
        },
        "postflight": {"status": "POSTFLIGHT_GREEN"},
    }
    _write_json(root / "final_manifest.json", manifest)
    return root


def test_merge_streams_jsonl_and_parquet_with_provenance(tmp_path: Path):
    s0 = _make_shard(
        tmp_path / "s0",
        "shard_0000",
        [_row("V0:1", "V0", 1, "crop-0", "Đỗ")],
    )
    s1 = _make_shard(
        tmp_path / "s1",
        "shard_0001",
        [_row("V1:2", "V1", 2, "crop-1", "hello")],
        parquet=True,
    )

    summary = merge_ocr_shards([s0, s1], tmp_path / "merged", batch_size=1)

    assert summary["frame_count"] == 2
    assert summary["crop_count"] == 2
    assert summary["line_count"] == 2
    assert summary["quality_status"] == "UNVALIDATED_ON_HCMAIC"
    rows = [
        json.loads(line)
        for line in (tmp_path / "merged" / "ocr_lines.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["frame_uid"] == "V0:1"
    assert rows[0]["ocr_text_folded"] == "do"
    assert rows[0]["source_shard_id"] == "shard_0000"
    assert (tmp_path / "merged" / "ocr_lines.parquet").is_file()
    merged_ledger = json.loads(
        (tmp_path / "merged" / "failure_ledger.json").read_text(encoding="utf-8")
    )
    assert merged_ledger["failure_count"] == 2
    assert merged_ledger["unresolved_count"] == 0
    merged_manifest = json.loads(
        (tmp_path / "merged" / "ocr_manifest.json").read_text(encoding="utf-8")
    )
    assert merged_manifest["format"] == "hcmaic-dstext-parseq-ocr-merged-v1"
    assert merged_manifest["identity"].startswith("frame_uid=video_id:source_frame_idx")
    assert merged_manifest["source_shards"][0]["final_manifest_sha256"]


def test_merge_rejects_duplicate_crop_and_frame_identity(tmp_path: Path):
    first = _make_shard(
        tmp_path / "first",
        "shard_0000",
        [_row("V0:1", "V0", 1, "same-crop", "a")],
    )
    second = _make_shard(
        tmp_path / "second",
        "shard_0001",
        [_row("V0:1", "V0", 1, "other-crop", "b")],
    )

    with pytest.raises(OCRMergeError, match="duplicate frame_uid"):
        merge_ocr_shards([first, second], tmp_path / "merged")


def test_merge_rejects_frame_uid_mismatch(tmp_path: Path):
    bad = _row("V0:999", "V0", 1, "crop", "bad")
    shard = _make_shard(tmp_path / "bad", "shard_0000", [bad])

    with pytest.raises(OCRMergeError, match="frame_uid identity mismatch"):
        merge_ocr_shards([shard], tmp_path / "merged")


def test_merge_is_fail_closed_for_non_green_shard(tmp_path: Path):
    shard = _make_shard(
        tmp_path / "bad-status",
        "shard_0000",
        [_row("V0:1", "V0", 1, "crop", "text")],
    )
    manifest_path = shard / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["postflight"]["status"] = "POSTFLIGHT_INCOMPLETE"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(OCRMergeError, match="POSTFLIGHT_GREEN"):
        merge_ocr_shards([shard], tmp_path / "merged")


def test_merge_accepts_complete_partition_set_without_collapsing_target_shard_id(
    tmp_path: Path,
):
    first = _make_shard(
        tmp_path / "part0",
        "shard_0000",
        [_row("V0:1", "V0", 1, "crop-0", "Đỗ")],
    )
    second = _make_shard(
        tmp_path / "part1",
        "shard_0000",
        [_row("V0:2", "V0", 2, "crop-1", "x")],
    )
    for index, root in enumerate((first, second)):
        manifest_path = root / "final_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["full_shard"] = False
        manifest["partition"] = {
            "count": 2,
            "index": index,
            "global_frame_count": 2,
            "strategy": "sorted_frame_uid_round_robin",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    summary = merge_ocr_shards([first, second], tmp_path / "merged", batch_size=1)

    assert summary["frame_count"] == 2
    assert summary["line_count"] == 2
    merged_manifest = json.loads(
        (tmp_path / "merged" / "ocr_manifest.json").read_text(encoding="utf-8")
    )
    assert len(merged_manifest["source_shards"]) == 2
    assert {item["partition"]["index"] for item in merged_manifest["source_shards"]} == {0, 1}


def test_merge_requires_complete_partition_indices(tmp_path: Path):
    first = _make_shard(
        tmp_path / "part0",
        "shard_0000",
        [_row("V0:1", "V0", 1, "crop-0", "Đỗ")],
    )
    manifest_path = first / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["full_shard"] = False
    manifest["partition"] = {
        "count": 2,
        "index": 0,
        "global_frame_count": 2,
        "strategy": "sorted_frame_uid_round_robin",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(OCRMergeError, match="partition set"):
        merge_ocr_shards([first], tmp_path / "merged")
