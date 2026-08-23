import json

from tools import build_ocr_dstext_parseq_shard0001_partition_notebooks as builder


def test_shard0001_partition_specs_are_balanced_and_assigned():
    assert builder.GLOBAL_SHARD_FRAME_COUNT == 24_935
    assert builder.PARTITION_COUNT == 3
    assert [builder.partition_expected_frame_count(index) for index in range(3)] == [8_312, 8_312, 8_311]
    assert [spec["owner"] for spec in builder.PARTITION_SPECS] == [
        "REPLACE_WITH_KAGGLE_OWNER",
        "REPLACE_WITH_KAGGLE_OWNER",
        "REPLACE_WITH_KAGGLE_OWNER",
    ]


def test_partition_notebooks_keep_shard1_identity_and_postflight_contract():
    for partition_index, spec in enumerate(builder.PARTITION_SPECS):
        path = builder.make_notebook(False, partition_index=partition_index)
        report = builder.validate_notebook(path, False, partition_index=partition_index)
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))

        assert report["partition_index"] == partition_index
        assert report["expected_frame_count"] == [8_312, 8_312, 8_311][partition_index]
        assert report["enable_gpu"] is False
        assert report["machine_shape"] is None
        assert metadata["id"] == f'{spec["owner"]}/{spec["name"]}'
        assert metadata["title"] == spec["name"]
        assert 'TARGET_SHARD_ID = "shard_0001"' in code
        assert "GLOBAL_SHARD_FRAME_COUNT = 24935" in code
        assert "PARTITION_COUNT = 3" in code
        assert f"PARTITION_INDEX = {partition_index}" in code
        assert 'RUN_MODE = "PARTITION_FULL"' in code
        assert "sorted_frame_uid_round_robin" in code
        assert "global_selection_sha256" in code
        assert "partition_selection_sha256" in code
        assert "def postflight_checkpoint(checkpoint_stage" in code
        assert '"artifact_stage"' in code
        assert "final_manifest.json" in code
        assert "failure_ledger" in code

    # Leave canonical deliverables armed for execution; the dry assertions
    # above must not overwrite a previously generated execute notebook.
    for partition_index in range(3):
        path = builder.make_notebook(True, partition_index=partition_index)
        report = builder.validate_notebook(path, True, partition_index=partition_index)
        assert report["enable_gpu"] is True
        assert report["machine_shape"] == "NvidiaTeslaT4"

