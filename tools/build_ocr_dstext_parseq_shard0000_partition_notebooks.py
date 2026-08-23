"""Build deterministic full-execution partitions for OCR shard 0000."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_ocr_dstext_parseq_shard0005_full_notebook as base


TARGET_SHARD_ID = "shard_0000"
TARGET_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0000-private"
GLOBAL_SHARD_FRAME_COUNT = 24_930
PARTITION_COUNT = 3
PARTITION_STRATEGY = "sorted_frame_uid_round_robin"
REVIEW_FRAME_COUNT = 12

PARTITION_SPECS = [
    {"index": 0, "owner": "REPLACE_WITH_KAGGLE_OWNER", "name": "hcmaic-ocr-dstext-parseq-s0000-part0-t4x2-20260820"},
    {"index": 1, "owner": "REPLACE_WITH_KAGGLE_OWNER", "name": "hcmaic-ocr-dstext-parseq-s0000-part1-t4x2-20260820"},
    {"index": 2, "owner": "REPLACE_WITH_KAGGLE_OWNER", "name": "hcmaic-ocr-dstext-parseq-s0000-part2-t4x2-20260820"},
]


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def partition_expected_frame_count(partition_index: int) -> int:
    if not 0 <= partition_index < PARTITION_COUNT:
        raise ValueError(f"invalid partition index: {partition_index}")
    base_count, remainder = divmod(GLOBAL_SHARD_FRAME_COUNT, PARTITION_COUNT)
    return base_count + int(partition_index < remainder)


def make_config(execute: bool, run_name: str, partition_index: int) -> str:
    expected = partition_expected_frame_count(partition_index)
    source = base.make_config(execute, run_name)
    source = replace_once(
        source,
        f"MAX_FRAMES = None\nEXPECTED_FRAME_COUNT = {expected}\nFULL_SHARD = True\nREVIEW_FRAME_COUNT = 12",
        f"MAX_FRAMES = None\nEXPECTED_FRAME_COUNT = {expected}\nFULL_SHARD = False\n"
        f"RUN_MODE = \"PARTITION_FULL\"\nPARTITION_INDEX = {partition_index}\n"
        f"PARTITION_COUNT = {PARTITION_COUNT}\nGLOBAL_SHARD_FRAME_COUNT = {GLOBAL_SHARD_FRAME_COUNT}\n"
        f"PARTITION_STRATEGY = \"{PARTITION_STRATEGY}\"\nREVIEW_FRAME_COUNT = {REVIEW_FRAME_COUNT}",
        "partition config",
    )
    source = replace_once(
        source,
        f'SELECTION_SEED = "hcmaic-ocr-{TARGET_SHARD_ID}-dstext-parseq-full-v1"',
        f'SELECTION_SEED = "hcmaic-ocr-{TARGET_SHARD_ID}-dstext-parseq-partition-v1"',
        "partition selection seed",
    )
    return source


def partition_selection_source() -> str:
    source = base.FULL_SELECTION
    old_count_gate = (
        "    if len(pool) != EXPECTED_FRAME_COUNT:\n"
        "        raise ValueError(f\"shard pool count changed: {len(pool)} != {EXPECTED_FRAME_COUNT}\")"
    )
    new_count_gate = (
        "    if len(pool) != GLOBAL_SHARD_FRAME_COUNT:\n"
        "        raise ValueError(f\"global shard pool count changed: {len(pool)} != {GLOBAL_SHARD_FRAME_COUNT}\")"
    )
    source = replace_once(source, old_count_gate, new_count_gate, "global shard count gate")
    old_identity = (
        "    # Full-shard identity is deterministic and does not use faiss/parquet row order.\n"
        "    selection = pool.sort_values([\"frame_uid\"]).reset_index(drop=True)\n"
        "    expected_uid = selection.apply(lambda row: f\"{row['video_id']}:{int(row['source_frame_idx'])}\", axis=1)\n"
        "    if not (selection[\"frame_uid\"].astype(str).values == expected_uid.values).all():\n"
        "        raise ValueError(\"frame_uid identity mismatch; expected video_id:source_frame_idx\")\n"
        "    if selection[\"frame_uid\"].astype(str).duplicated().any():\n"
        "        raise ValueError(\"duplicate canonical frame_uid after full-shard selection\")\n"
        "    selection_sha256 = hashlib.sha256(\"\\n\".join(selection[\"frame_uid\"].astype(str)).encode()).hexdigest()\n"
        "    inventory_sha256 = sha256_file(inventory_path)\n"
    )
    new_identity = (
        "    # Parent shard identity is deterministic and never uses faiss/parquet row order.\n"
        "    global_selection = pool.sort_values([\"frame_uid\"]).reset_index(drop=True)\n"
        "    expected_uid = global_selection.apply(\n"
        "        lambda row: f\"{row['video_id']}:{int(row['source_frame_idx'])}\", axis=1\n"
        "    )\n"
        "    if not (global_selection[\"frame_uid\"].astype(str).values == expected_uid.values).all():\n"
        "        raise ValueError(\"frame_uid identity mismatch; expected video_id:source_frame_idx\")\n"
        "    if global_selection[\"frame_uid\"].astype(str).duplicated().any():\n"
        "        raise ValueError(\"duplicate canonical frame_uid in parent shard\")\n"
        "    global_selection_sha256 = hashlib.sha256(\n"
        "        \"\\n\".join(global_selection[\"frame_uid\"].astype(str)).encode()\n"
        "    ).hexdigest()\n"
        "    selection = global_selection.iloc[PARTITION_INDEX::PARTITION_COUNT].reset_index(drop=True)\n"
        "    if len(selection) != EXPECTED_FRAME_COUNT:\n"
        "        raise ValueError(f\"partition count changed: {len(selection)} != {EXPECTED_FRAME_COUNT}\")\n"
        "    selection[\"partition_index\"] = PARTITION_INDEX\n"
        "    selection[\"partition_count\"] = PARTITION_COUNT\n"
        "    selection_sha256 = hashlib.sha256(\n"
        "        \"\\n\".join(selection[\"frame_uid\"].astype(str)).encode()\n"
        "    ).hexdigest()\n"
        "    inventory_sha256 = sha256_file(inventory_path)\n"
    )
    source = replace_once(source, old_identity, new_identity, "deterministic partition selection")
    old_manifest = '"selection_seed": SELECTION_SEED, "full_shard": True,'
    new_manifest = (
        '"selection_seed": SELECTION_SEED, "full_shard": FULL_SHARD, "run_mode": RUN_MODE,\n'
        '        "partition": {"index": PARTITION_INDEX, "count": PARTITION_COUNT,\n'
        '                      "global_frame_count": GLOBAL_SHARD_FRAME_COUNT,\n'
        '                      "global_selection_sha256": global_selection_sha256,\n'
        '                      "partition_selection_sha256": selection_sha256,\n'
        '                      "strategy": PARTITION_STRATEGY},'
    )
    return replace_once(source, old_manifest, new_manifest, "partition selection manifest")


def partition_postflight_source() -> str:
    source = base.FULL_POSTFLIGHT
    old_manifest = '"inventory_dataset_slug": INVENTORY_DATASET_SLUG, "full_shard": True,'
    new_manifest = (
        '"inventory_dataset_slug": INVENTORY_DATASET_SLUG, "full_shard": FULL_SHARD,\n'
        '        "run_mode": RUN_MODE,\n'
        '        "partition": {"index": PARTITION_INDEX, "count": PARTITION_COUNT,\n'
        '                      "global_frame_count": GLOBAL_SHARD_FRAME_COUNT,\n'
        '                      "strategy": PARTITION_STRATEGY},'
    )
    return replace_once(source, old_manifest, new_manifest, "partition final manifest")


def make_notebook(execute: bool, partition_index: int) -> Path:
    if not 0 <= partition_index < PARTITION_COUNT:
        raise ValueError(f"invalid partition index: {partition_index}")
    spec = PARTITION_SPECS[partition_index]
    run_name = spec["name"]
    expected = partition_expected_frame_count(partition_index)
    previous = {
        name: getattr(base, name)
        for name in (
            "TARGET_SHARD_ID", "TARGET_DATASET_SLUG", "EXPECTED_FRAME_COUNT",
            "KERNEL_OWNER", "DRY_NAME", "EXECUTE_NAME",
        )
    }
    base.TARGET_SHARD_ID = TARGET_SHARD_ID
    base.TARGET_DATASET_SLUG = TARGET_DATASET_SLUG
    base.EXPECTED_FRAME_COUNT = expected
    base.KERNEL_OWNER = spec["owner"]
    base.DRY_NAME = run_name
    base.EXECUTE_NAME = run_name
    try:
        config_base = make_config(execute, run_name, partition_index)
        selection_source = partition_selection_source()
        postflight_source = partition_postflight_source()
        worker_source = base.make_worker_source()
    finally:
        for name, value in previous.items():
            setattr(base, name, value)

    pipeline_sources = (
        selection_source, base.FULL_RUNTIME, base.pilot.ASSETS,
        worker_source, base.FULL_LAUNCH, postflight_source,
    )
    pipeline_code_sha256 = base.sha256_text("\n---HCMAIC-CELL---\n".join(pipeline_sources))
    notebook_config_sha256 = base.sha256_text(config_base)
    builder_paths = [
        Path(base.pilot.__file__).resolve(), Path(base.__file__).resolve(),
        Path(__file__).resolve(),
        base.TOOLS_ROOT / "build_ocr_dstext_parseq_multi_shard_dry_notebooks.py",
    ]
    builder_source_sha256 = base.sha256_source_bundle(builder_paths)
    builder_git_commit, builder_git_dirty, builder_git_tracked = base.git_source_revision(builder_paths)
    builder_source_files = [
        str(path.resolve().relative_to(base.ROOT)).replace("\\", "/")
        for path in builder_paths
    ]
    provenance_constants = "\n".join((
        f'PIPELINE_CODE_SHA256 = "{pipeline_code_sha256}"',
        f'NOTEBOOK_CONFIG_SHA256 = "{notebook_config_sha256}"',
        f'BUILDER_SOURCE_SHA256 = "{builder_source_sha256}"',
        f"BUILDER_GIT_COMMIT = {builder_git_commit!r}",
        f"BUILDER_GIT_DIRTY = {builder_git_dirty!r}",
        f"BUILDER_GIT_TRACKED = {builder_git_tracked!r}",
        f"BUILDER_SOURCE_FILES = {builder_source_files!r}",
    ))
    config = replace_once(
        config_base,
        "\n\nOUT.mkdir(parents=True, exist_ok=True)",
        f"\n\n{provenance_constants}\n\nOUT.mkdir(parents=True, exist_ok=True)",
        "partition config provenance",
    )
    config = replace_once(
        config,
        "      quality_status=QUALITY_STATUS)",
        "      quality_status=QUALITY_STATUS, pipeline_code_sha256=PIPELINE_CODE_SHA256,\n"
        "      notebook_config_sha256=NOTEBOOK_CONFIG_SHA256,\n"
        "      builder_source_sha256=BUILDER_SOURCE_SHA256)",
        "partition config phase revision",
    )
    shard_label = TARGET_SHARD_ID.replace("shard_", "")
    intro = (
        f"# {run_name}\n\n"
        f"Full OCR execution partition {partition_index + 1}/{PARTITION_COUNT} for "
        f"keyframe shard {shard_label}: official DSText DeepSolo detector → "
        "word crops → Vietnamese PARSeq.\n\n"
        f"- Selects exactly {expected:,} deterministic frames from the {GLOBAL_SHARD_FRAME_COUNT:,}-frame parent shard.\n"
        f"- Partition strategy: `{PARTITION_STRATEGY}`, joined by `frame_uid`; no faiss row identity.\n"
        "- Every selected image is resolved/decoded before model setup; first/middle/last probes and ZIP indexes are recorded.\n"
        "- Threshold `0.30`; one full-frame detector pass; no recall tiles.\n"
        "- Two concurrent GPU subprocesses when available; effective worker count is recorded.\n"
        "- Artifacts are `ENGINEERING_PROXY`; OCR/retrieval quality remains `UNVALIDATED`.\n"
    )
    cells = [
        base.pilot.cell(intro, "intro", "markdown"),
        base.pilot.cell(config, "config"),
        base.pilot.cell(selection_source, "selection-and-image-preflight"),
        base.pilot.cell(base.FULL_RUNTIME, "runtime-and-source-preflight"),
        base.pilot.cell(base.pilot.ASSETS, "model-asset-preflight"),
        base.pilot.cell("WORKER_SOURCE = " + repr(worker_source), "worker-source"),
        base.pilot.cell(base.FULL_LAUNCH, "parallel-inference"),
        base.pilot.cell(postflight_source, "checkpoint-manifest-postflight"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "hcmaic": {
                "status": "DRAFT_NOT_EXECUTED" if not execute else "EXECUTION_ARMED",
                "quality_status": "UNVALIDATED", "provenance_class": "ENGINEERING_PROXY",
                "identity": "frame_uid=video_id:source_frame_idx",
                "target_shard_id": TARGET_SHARD_ID, "expected_frame_count": expected,
                "global_shard_frame_count": GLOBAL_SHARD_FRAME_COUNT,
                "partition_index": partition_index, "partition_count": PARTITION_COUNT,
                "run_mode": "PARTITION_FULL",
            },
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out_dir = base.KERNEL_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": f'{spec["owner"]}/{run_name}', "title": run_name,
        "code_file": path.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": bool(execute), "enable_tpu": False,
        "enable_internet": True, "machine_shape": None if not execute else "NvidiaTeslaT4",
        "keywords": [],
        "dataset_sources": [base.INVENTORY_DATASET_SLUG, TARGET_DATASET_SLUG, base.WEIGHT_DATASET_SLUG],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def validate_notebook(path: Path, execute: bool, partition_index: int) -> dict:
    if not 0 <= partition_index < PARTITION_COUNT:
        raise ValueError(f"invalid partition index: {partition_index}")
    spec = PARTITION_SPECS[partition_index]
    expected = partition_expected_frame_count(partition_index)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = [
        "".join(item.get("source", []))
        for item in notebook["cells"]
        if item.get("cell_type") == "code"
    ]
    for source in sources:
        ast.parse(source)
    code = "\n".join(sources)
    required = (
        f'TARGET_SHARD_ID = "{TARGET_SHARD_ID}"', "MAX_FRAMES = None",
        f"EXPECTED_FRAME_COUNT = {expected}", "FULL_SHARD = False",
        'RUN_MODE = "PARTITION_FULL"', f"PARTITION_INDEX = {partition_index}",
        f"PARTITION_COUNT = {PARTITION_COUNT}", f"GLOBAL_SHARD_FRAME_COUNT = {GLOBAL_SHARD_FRAME_COUNT}",
        f'PARTITION_STRATEGY = "{PARTITION_STRATEGY}"', "REVIEW_FRAME_COUNT = 12",
        "review_selection.json", "review_manifest.jsonl", "review_index.html",
        "review_overlays", "review_frames_only", "review_crops",
        "DETECTION_SCORE_THRESHOLD = 0.30", "FULL_FRAME_ONLY = True", "RECALL_MODE = False",
        "TILE_PASS_COUNT = 0", "REQUESTED_GPU_WORKERS = 2", "CUDA_VISIBLE_DEVICES",
        "keyframe_inventory.parquet", "keyframe_inventory.jsonl", "IMAGE_PREFLIGHT_GREEN",
        "image_preflight.jsonl", "preflight_failures", "failure_ledger", "NO_TEXT",
        "READ_FAILED", "INFERENCE_FAILED", "PARSE_ERROR", "final_manifest.json",
        "detection_status.parquet", "crop_inventory.parquet", "ocr_lines.parquet",
        "MODEL_BOOTSTRAP_GREEN", "MODEL_INFERENCE_SMOKE_GREEN", "model_gates.json",
        "ocr_candidates", "candidate_policy", "inventory_sha256", "parseq_batch_size",
        "buffered_handles_flush_128", "per_worker_zip_handle_cache",
        "PIPELINE_CODE_SHA256", "NOTEBOOK_CONFIG_SHA256", "BUILDER_SOURCE_SHA256",
        "BUILDER_GIT_COMMIT", "BUILDER_GIT_DIRTY", "BUILDER_GIT_TRACKED",
        "frame_uid=video_id:source_frame_idx", "DEEPSOLO_WEIGHT_SHA256", "PARSEQ_WEIGHT_SHA256",
        "network fallback is intentionally disabled", "WEIGHT_DATASET_SLUG",
        "def postflight_checkpoint(checkpoint_stage", '"artifact_stage"',
        "global_selection_sha256", "partition_selection_sha256",
    )
    missing = [token for token in required if token not in code]
    forbidden = [token for token in (
        "__RECOGNIZER_MODEL__", "PP-OCR", "VietOCR", "access_token",
        "KAGGLE_API_TOKEN", 'faiss_row": str',
    ) if token in code]
    if missing or forbidden:
        raise ValueError({"missing": missing, "forbidden": forbidden})
    metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    expected_id = f'{spec["owner"]}/{spec["name"]}'
    if metadata["id"] != expected_id or metadata["title"] != spec["name"]:
        raise ValueError("kernel metadata id/title is not canonical")
    if metadata["enable_gpu"] != execute:
        raise ValueError("dry/execute GPU metadata mismatch")
    if not execute and ("EXECUTE_PIPELINE = True" in code or metadata.get("machine_shape") is not None):
        raise ValueError("dry notebook is armed or requests a machine")
    return {
        "path": str(path), "cells": len(notebook["cells"]), "execute": execute,
        "enable_gpu": metadata["enable_gpu"], "machine_shape": metadata.get("machine_shape"),
        "target_shard_id": TARGET_SHARD_ID, "partition_index": partition_index,
        "partition_count": PARTITION_COUNT, "expected_frame_count": expected,
        "source_chars": len(code),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-index", type=int, choices=range(PARTITION_COUNT), required=True)
    parser.add_argument("--execute", action="store_true", help="arm the full partition execution")
    args = parser.parse_args()
    path = make_notebook(args.execute, partition_index=args.partition_index)
    print(json.dumps(validate_notebook(path, args.execute, args.partition_index), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

