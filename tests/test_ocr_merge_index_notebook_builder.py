"""Static contract tests for the six-shard OCR merge/index notebook builder."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS.resolve()))

import build_ocr_merge_index_notebook as builder  # noqa: E402


def _code(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))
            sources.append("".join(cell.get("source", [])))
    return "\n".join(sources)


def test_smoke_builder_is_real_bounded_pipeline(tmp_path: Path):
    path = builder.build_notebook("SMOKE", tmp_path)
    metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    code = _code(path)

    assert metadata["id"] == f"{builder.OWNER}/{builder.SMOKE_NAME}"
    assert metadata["dataset_sources"] == [builder.TRANSFER_DATASET_SLUG]
    assert metadata["kernel_sources"] == []
    assert metadata["enable_gpu"] is False
    assert "MODE = 'SMOKE'" in code
    assert "EXECUTE_PIPELINE = True" in code
    assert "MAX_ROWS_PER_SOURCE = 512" in code
    for marker in (
        "SOURCE_PREFLIGHT_GREEN",
        "partition set",
        "frame_uid=video_id:source_frame_idx",
        "crop_uid",
        "failure_ledger.json",
        "ocr_lines.parquet",
        "ocr_manifest.json",
        "index_manifest.json",
        "HEARTBEAT_SECONDS",
    ):
        assert marker in code
    assert 'iter_artifact_rows(source["status_path"], MAX_STATUS_ROWS_PER_SOURCE)' not in code
    assert "batch_size=min(BATCH_SIZE, max(1, limit))" in code
    assert "KAGGLE_API_TOKEN" not in code
    assert "access_token" not in code


def test_full_builder_is_armed_only_for_merge_index(tmp_path: Path):
    path = builder.build_notebook("FULL", tmp_path)
    metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    code = _code(path)

    assert metadata["id"] == f"{builder.OWNER}/{builder.FULL_NAME}"
    assert metadata["enable_gpu"] is False
    assert "MODE = 'FULL'" in code
    assert "MAX_ROWS_PER_SOURCE = None" in code
    assert "ENGINEERING_ARTIFACT_COMPLETE" in code
    assert "quality_status" in code
