import json

from tools import build_ocr_dstext_parseq_shard0005_full_notebook as builder
from tools import build_ocr_dstext_parseq_shard0001_smoke_notebook as smoke_builder


def test_shard0005_dry_notebook_is_full_shard_and_unarmed():
    path = builder.make_notebook(False)
    report = builder.validate_notebook(path, False)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    cell_ids = [cell.get("id") for cell in notebook["cells"]]

    assert report["expected_frame_count"] == 21_476
    assert report["enable_gpu"] is False
    assert report["machine_shape"] is None
    assert "TARGET_SHARD_ID = \"shard_0005\"" in code
    assert "MAX_FRAMES = None" in code
    assert "EXECUTE_PIPELINE = True" not in code
    assert cell_ids.index("selection-and-image-preflight") < cell_ids.index("runtime-and-source-preflight")
    assert "image_preflight.jsonl" in code
    assert "REVIEW_FRAME_COUNT = 12" in code
    assert "review_selection.json" in code
    assert "review_manifest.jsonl" in code
    assert "review_index.html" in code
    assert "review_frames_only" in code
    assert "review_crops" in code
    assert "buffered_handles_flush_128" in code
    assert "per_worker_zip_handle_cache" in code
    assert "MODEL_BOOTSTRAP_GREEN" in code
    assert "MODEL_INFERENCE_SMOKE_GREEN" in code
    assert "ocr_candidates" in code
    assert "inventory_sha256" in code
    assert "ocr_lines.parquet" in code
    assert "ENGINEERING_ARTIFACT_COMPLETE_REPORT_FAILED" in code


def test_bounded_smoke_builder_covers_all_failed_shards_with_postflight_fix():
    expected = {
        "shard_0000": "REPLACE_WITH_KAGGLE_OWNER",
        "shard_0001": "REPLACE_WITH_KAGGLE_OWNER",
        "shard_0002": "REPLACE_WITH_KAGGLE_OWNER",
    }
    for shard_id, owner in expected.items():
        path = smoke_builder.make_notebook(False, shard_id=shard_id)
        report = smoke_builder.validate_notebook(path, False, shard_id=shard_id)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))

        assert report["target_shard_id"] == shard_id
        assert report["smoke_frame_count"] == 12
        assert report["enable_gpu"] is False
        assert report["machine_shape"] is None
        assert metadata["id"].startswith(f"{owner}/hcmaic-ocr-dstext-s{shard_id[-4:]}-smoke-")
        assert f'TARGET_SHARD_ID = "{shard_id}"' in code
        assert "FULL_SHARD = False" in code
        assert 'RUN_MODE = "BOUNDED_SMOKE"' in code
        assert "def postflight_checkpoint(checkpoint_stage" in code
        assert '"artifact_stage"' in code
        assert "final_manifest.json" in code
        assert "failure_ledger" in code

