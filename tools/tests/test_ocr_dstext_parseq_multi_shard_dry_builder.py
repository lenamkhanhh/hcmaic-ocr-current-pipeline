import ast
import json
import os
import re
from pathlib import Path

import pytest

from tools import build_ocr_dstext_parseq_multi_shard_dry_notebooks as builder


@pytest.mark.parametrize("shard_id", sorted(builder.SPECS))
def test_multi_shard_dry_contract(shard_id):
    report = builder.build_one(shard_id)
    notebook_path = Path(report["path"])
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    metadata = json.loads((notebook_path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert report["shard_id"] == shard_id
    assert report["expected_frame_count"] == builder.SPECS[shard_id]["expected_frame_count"]
    assert report["enable_gpu"] is False
    assert report["machine_shape"] is None
    assert metadata["id"] == f"{builder.SPECS[shard_id]['owner']}/{builder.SPECS[shard_id]['name']}"
    assert f'TARGET_SHARD_ID = "{shard_id}"' in code
    assert f"EXPECTED_FRAME_COUNT = {builder.SPECS[shard_id]['expected_frame_count']}" in code
    assert "EXECUTE_PIPELINE = True" not in code
    assert f'OUT = Path("/kaggle/working/{builder.SPECS[shard_id]["name"]}")' in code
    assert "final_manifest.json" in code
    assert "failure_ledger" in code


@pytest.mark.parametrize("shard_id", sorted(builder.SPECS))
def test_multi_shard_execute_contract(shard_id):
    report = builder.build_one(shard_id, execute=True)
    notebook_path = Path(report["path"])
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    metadata = json.loads((notebook_path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))

    spec = builder.SPECS[shard_id]
    shard_label = shard_id.replace("shard_", "")

    assert report["shard_id"] == shard_id
    assert report["execute"] is True
    assert report["enable_gpu"] is True
    assert report["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["id"] == f'{spec["owner"]}/{spec["name"]}'
    assert 'EXECUTE_PIPELINE = True' in code
    assert f'OUT = Path("/kaggle/working/{spec["name"]}")' in code
    intro = "".join(notebook["cells"][0].get("source", []))
    assert f"keyframe shard {shard_label}" in intro
    assert f'{spec["expected_frame_count"]:,}' in intro
    assert "final_manifest.json" in code
    assert "failure_ledger" in code


def test_parallel_progress_checkpoint_is_atomic_and_read_is_retried():
    report = builder.build_one("shard_0004", execute=True)
    notebook_path = Path(report["path"])
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "os.replace(tmp_path, path)" in code
    assert "def read_progress_snapshot" in code
    assert "json.JSONDecodeError" in code
    assert "progress_read_retries" in code
    assert "read_progress_snapshot(progress, worker_id, process.poll())" in code


def _extract_function(source, function_name):
    tree = ast.parse(source)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    assert len(functions) == 1
    module = ast.Module(body=[functions[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, "<generated-notebook-helper>", "exec")


def _extract_functions(source, function_names):
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    }
    assert set(functions) == set(function_names)
    module = ast.Module(
        body=[functions[name] for name in function_names],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return compile(module, "<generated-notebook-helper>", "exec")


def test_progress_reader_recovers_from_partial_json():
    launch_source = builder.builder.FULL_LAUNCH
    namespace = {"json": json}

    class FakeTime:
        def __init__(self):
            self.sleeps = []

        def sleep(self, seconds):
            self.sleeps.append(seconds)

    fake_time = FakeTime()
    namespace["time"] = fake_time
    exec(_extract_function(launch_source, "read_progress_snapshot"), namespace)

    class PartialThenValidPath:
        def __init__(self):
            self.values = ["", '{"worker_id":0,', '{"worker_id":0,"phase":"OK"}']

        def read_text(self, encoding):
            assert encoding == "utf-8"
            return self.values.pop(0)

    result = namespace["read_progress_snapshot"](PartialThenValidPath(), 0, None)
    assert result == {"worker_id": 0, "phase": "OK"}
    assert fake_time.sleeps == [0.1, 0.1]


def test_worker_progress_write_replaces_only_complete_json(tmp_path, monkeypatch):
    worker_source = builder.builder.make_worker_source()
    namespace = {"json": json, "os": os}
    exec(_extract_function(worker_source, "write_json"), namespace)

    target = tmp_path / "progress.json"
    target.write_text('{"phase":"OLD"}\n', encoding="utf-8")
    payload = {"worker_id": 1, "phase": "INFERENCE_HEARTBEAT", "processed": 615}
    observed = {}
    real_replace = os.replace

    def checked_replace(source, destination):
        observed["tmp_payload"] = json.loads(Path(source).read_text(encoding="utf-8"))
        observed["target_before_replace"] = json.loads(target.read_text(encoding="utf-8"))
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", checked_replace)
    namespace["write_json"](target, payload)

    assert observed["tmp_payload"] == payload
    assert observed["target_before_replace"] == {"phase": "OLD"}
    assert json.loads(target.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("shard_id", sorted(builder.SPECS))
def test_execute_manifest_embeds_complete_code_provenance(shard_id):
    report = builder.build_one(shard_id, execute=True)
    notebook = json.loads(Path(report["path"]).read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell.get("source", [])) for cell in notebook["cells"]}
    config = cells["config"]
    postflight = cells["checkpoint-manifest-postflight"]

    values = {}
    for name in (
        "PIPELINE_CODE_SHA256",
        "NOTEBOOK_CONFIG_SHA256",
        "BUILDER_SOURCE_SHA256",
    ):
        match = re.search(rf'^{name} = "([0-9a-f]{{64}})"$', config, re.MULTILINE)
        assert match, f"missing 64-char {name} for {shard_id}"
        values[name] = match.group(1)

    assert len(set(values.values())) == 3
    assert '"code_revision": {' in postflight
    assert '"pipeline_code_sha256": PIPELINE_CODE_SHA256' in postflight
    assert '"notebook_config_sha256": NOTEBOOK_CONFIG_SHA256' in postflight
    assert '"builder_source_sha256": BUILDER_SOURCE_SHA256' in postflight
    assert '"builder_git_commit": BUILDER_GIT_COMMIT' in postflight
    assert '"builder_git_dirty": BUILDER_GIT_DIRTY' in postflight
    assert 'BUILDER_GIT_TRACKED = ' in config
    assert '"builder_git_tracked": BUILDER_GIT_TRACKED' in postflight


def test_generated_notebook_uses_ascii_source_separators_for_remote_hash_stability():
    report = builder.build_one("shard_0000", execute=True)
    notebook = json.loads(Path(report["path"]).read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "\ufffd" not in code
    assert "·" not in code


def test_postflight_streams_large_worker_outputs_without_global_row_lists():
    source = builder.builder.FULL_POSTFLIGHT

    assert "POSTFLIGHT_ROW_BATCH_SIZE" in source
    assert "POSTFLIGHT_HEARTBEAT_SECONDS" in source
    assert "def iter_jsonl" in source
    assert "def write_jsonl" in source
    assert "pyarrow.parquet" in source
    assert "ParquetWriter" in source
    assert "sqlite3" in source
    assert "postflight_state.json" in source
    assert "POSTFLIGHT_HEARTBEAT" in source
    assert "statuses = [row for" not in source
    assert "crops = [row for" not in source
    assert "lines = [row for" not in source
    assert "failures = [row for" not in source
    assert "pd.DataFrame(crops)" not in source
    assert "pd.DataFrame(lines)" not in source
    assert '"frame_status_count": len(status_uids)' in source
    assert '"crop_count": crop_count' in source
    assert '"recognition_count": recognition_count' in source
    assert '"failure_count": metrics["failure_count"]' in source
    assert "len(terminal_uids)" not in source
    assert "len(crops)" not in source
    assert "len(lines)" not in source
    assert "len(failures)" not in source


def test_postflight_heartbeat_keeps_checkpoint_and_artifact_stages_distinct():
    source = builder.builder.FULL_POSTFLIGHT
    namespace = {
        "POSTFLIGHT_HEARTBEAT_SECONDS": 240,
        "POSTFLIGHT_ROW_BATCH_SIZE": 50_000,
        "POSTFLIGHT_STATE_PATH": "postflight_state.json",
        "postflight_started": 0.0,
        "postflight_heartbeat_state": {"last": 0.0},
    }
    writes = []
    phases = []

    class FakeTime:
        def time(self):
            return 241.0

    namespace["time"] = FakeTime()
    namespace["write_json"] = lambda path, payload: writes.append((path, payload))
    namespace["phase"] = lambda name, **fields: phases.append((name, fields))
    exec(
        _extract_functions(
            source,
            ["postflight_checkpoint", "maybe_postflight_heartbeat"],
        ),
        namespace,
    )

    namespace["maybe_postflight_heartbeat"]("ocr_lines", 50_000)

    assert phases == [("POSTFLIGHT_HEARTBEAT", {
        "stage": "ocr_lines",
        "rows": 50_000,
        "elapsed_s": 241.0,
        "row_batch_size": 50_000,
    })]
    assert writes == [("postflight_state.json", {
        "status": "POSTFLIGHT_RUNNING",
        "stage": "HEARTBEAT",
        "artifact_stage": "ocr_lines",
        "timestamp_epoch": 241.0,
        "elapsed_s": 241.0,
        "row_batch_size": 50_000,
        "io_policy": "jsonl_stream_to_parquet_row_groups",
        "rows": 50_000,
    })]


def test_postflight_streams_large_artifact_through_heartbeat(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = builder.builder.FULL_POSTFLIGHT
    namespace = {
        "POSTFLIGHT_HEARTBEAT_SECONDS": 240,
        "POSTFLIGHT_ROW_BATCH_SIZE": 50_000,
        "POSTFLIGHT_STATE_PATH": tmp_path / "postflight_state.json",
        "postflight_started": 0.0,
        "postflight_heartbeat_state": {"last": 0.0},
        "pa": pa,
        "pq": pq,
        "json": json,
    }
    writes = []
    phases = []

    class FakeTime:
        def time(self):
            return 241.0

    namespace["time"] = FakeTime()
    namespace["write_json"] = lambda path, payload: writes.append((path, payload))
    namespace["phase"] = lambda name, **fields: phases.append((name, fields))
    exec(
        _extract_functions(
            source,
            [
                "iter_jsonl",
                "postflight_checkpoint",
                "maybe_postflight_heartbeat",
                "as_text",
                "as_int",
                "normalize_status",
                "stream_artifact",
            ],
        ),
        namespace,
    )

    source_path = tmp_path / "worker_status.jsonl"
    with source_path.open("w", encoding="utf-8") as handle:
        for index in range(100_001):
            handle.write(json.dumps({
                "frame_uid": f"video:{index}",
                "status": "OK",
                "line_count": 0,
                "crop_uids": [],
            }) + "\n")

    jsonl_path = tmp_path / "detection_status.jsonl"
    parquet_path = tmp_path / "detection_status.parquet"
    schema = pa.schema([
        pa.field("frame_uid", pa.string()),
        pa.field("status", pa.string()),
        pa.field("line_count", pa.int64()),
        pa.field("crop_uids", pa.list_(pa.string())),
    ])
    row_count = namespace["stream_artifact"](
        "detection_status",
        [source_path],
        jsonl_path,
        parquet_path,
        schema,
        namespace["normalize_status"],
    )

    assert row_count == 100_001
    assert sum(1 for _ in jsonl_path.open("r", encoding="utf-8")) == 100_001
    assert pq.ParquetFile(parquet_path).metadata.num_rows == 100_001
    assert any(name == "POSTFLIGHT_HEARTBEAT" for name, _ in phases)
    assert writes[-1][1]["stage"] == "ARTIFACT_GREEN"

