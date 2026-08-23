"""Build a bounded, fully executable shard-0001 OCR smoke notebook.

The smoke run uses the same detector, recognizer, runtime, worker and
postflight cells as the full-shard contract.  Only the deterministic selection
is bounded so that the complete manifest/postflight path can be validated
without paying the full shard cost.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import build_ocr_dstext_parseq_shard0005_full_notebook as base


SMOKE_SPECS = {
    "shard_0000": {
        "name": "hcmaic-ocr-dstext-s0000-smoke-t4x2-20260820",
        "owner": "REPLACE_WITH_KAGGLE_OWNER",
        "dataset_slug": "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0000-private",
    },
    "shard_0001": {
        "name": "hcmaic-ocr-dstext-s0001-smoke-t4x2-20260820",
        "owner": "REPLACE_WITH_KAGGLE_OWNER",
        "dataset_slug": "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0001-private",
    },
    "shard_0002": {
        "name": "hcmaic-ocr-dstext-s0002-smoke-t4x2-20260820",
        "owner": "REPLACE_WITH_KAGGLE_OWNER",
        "dataset_slug": "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0002-private",
    },
}
DEFAULT_SHARD_ID = "shard_0001"
SMOKE_NAME = SMOKE_SPECS[DEFAULT_SHARD_ID]["name"]
KERNEL_OWNER = SMOKE_SPECS[DEFAULT_SHARD_ID]["owner"]
TARGET_SHARD_ID = DEFAULT_SHARD_ID
TARGET_DATASET_SLUG = SMOKE_SPECS[DEFAULT_SHARD_ID]["dataset_slug"]
EXPECTED_FRAME_COUNT = 12


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def smoke_selection_source() -> str:
    source = base.FULL_SELECTION
    source = replace_once(
        source,
        '''    if len(pool) != EXPECTED_FRAME_COUNT:\n        raise ValueError(f"shard pool count changed: {len(pool)} != {EXPECTED_FRAME_COUNT}")''',
        '''    if FULL_SHARD and len(pool) != EXPECTED_FRAME_COUNT:\n        raise ValueError(f"shard pool count changed: {len(pool)} != {EXPECTED_FRAME_COUNT}")\n    if not FULL_SHARD and len(pool) < MAX_FRAMES:\n        raise ValueError(f"shard pool too small: {len(pool)} < {MAX_FRAMES}")''',
        "bounded shard pool gate",
    )
    source = replace_once(
        source,
        '    selection = pool.sort_values(["frame_uid"]).reset_index(drop=True)',
        '''    selection = pool.sort_values(["frame_uid"]).reset_index(drop=True)\n    if not FULL_SHARD:\n        selection = selection.head(MAX_FRAMES).reset_index(drop=True)''',
        "bounded deterministic selection",
    )
    source = replace_once(
        source,
        '"selection_seed": SELECTION_SEED, "full_shard": True,',
        '"selection_seed": SELECTION_SEED, "full_shard": FULL_SHARD, "run_mode": RUN_MODE, "smoke": not FULL_SHARD,',
        "selection manifest run mode",
    )
    return source


def smoke_postflight_source() -> str:
    source = base.FULL_POSTFLIGHT
    source = replace_once(
        source,
        '"inventory_dataset_slug": INVENTORY_DATASET_SLUG, "full_shard": True,',
        '"inventory_dataset_slug": INVENTORY_DATASET_SLUG, "full_shard": FULL_SHARD, "run_mode": RUN_MODE, "smoke": not FULL_SHARD,',
        "final manifest run mode",
    )
    return source


def make_notebook(execute: bool, shard_id: str = DEFAULT_SHARD_ID) -> Path:
    if shard_id not in SMOKE_SPECS:
        raise ValueError(f"unsupported smoke shard: {shard_id}")
    spec = SMOKE_SPECS[shard_id]
    smoke_name = spec["name"]
    kernel_owner = spec["owner"]
    target_dataset_slug = spec["dataset_slug"]
    target_shard_id = shard_id
    module_names = (
        "TARGET_SHARD_ID",
        "TARGET_DATASET_SLUG",
        "EXPECTED_FRAME_COUNT",
        "KERNEL_OWNER",
        "DRY_NAME",
        "EXECUTE_NAME",
    )
    previous = {name: getattr(base, name) for name in module_names}
    base.TARGET_SHARD_ID = target_shard_id
    base.TARGET_DATASET_SLUG = target_dataset_slug
    base.EXPECTED_FRAME_COUNT = EXPECTED_FRAME_COUNT
    base.KERNEL_OWNER = kernel_owner
    base.DRY_NAME = smoke_name
    base.EXECUTE_NAME = smoke_name
    module_previous = {
        "SMOKE_NAME": SMOKE_NAME,
        "KERNEL_OWNER": KERNEL_OWNER,
        "TARGET_SHARD_ID": TARGET_SHARD_ID,
        "TARGET_DATASET_SLUG": TARGET_DATASET_SLUG,
    }
    globals().update({
        "SMOKE_NAME": smoke_name,
        "KERNEL_OWNER": kernel_owner,
        "TARGET_SHARD_ID": target_shard_id,
        "TARGET_DATASET_SLUG": target_dataset_slug,
    })
    try:
        selection_source = smoke_selection_source()
        postflight_source = smoke_postflight_source()
        worker_source = base.make_worker_source()

        config_base = base.make_config(execute, smoke_name)
    finally:
        for name, value in previous.items():
            setattr(base, name, value)
        globals().update(module_previous)
    config_base = replace_once(
        config_base,
        f"MAX_FRAMES = None\nEXPECTED_FRAME_COUNT = {EXPECTED_FRAME_COUNT}\nFULL_SHARD = True\nREVIEW_FRAME_COUNT = 12",
        f"MAX_FRAMES = {EXPECTED_FRAME_COUNT}\nEXPECTED_FRAME_COUNT = {EXPECTED_FRAME_COUNT}\nFULL_SHARD = False\nRUN_MODE = \"BOUNDED_SMOKE\"\nREVIEW_FRAME_COUNT = {EXPECTED_FRAME_COUNT}",
        "bounded smoke config",
    )
    config_base = replace_once(
        config_base,
        f'SELECTION_SEED = "hcmaic-ocr-{target_shard_id}-dstext-parseq-full-v1"',
        f'SELECTION_SEED = "hcmaic-ocr-{target_shard_id}-dstext-parseq-smoke-v2"',
        "smoke selection seed",
    )

    pipeline_sources = (
        selection_source,
        base.FULL_RUNTIME,
        base.pilot.ASSETS,
        worker_source,
        base.FULL_LAUNCH,
        postflight_source,
    )
    pipeline_code_sha256 = base.sha256_text("\n---HCMAIC-CELL---\n".join(pipeline_sources))
    notebook_config_sha256 = base.sha256_text(config_base)
    builder_paths = [
        Path(base.pilot.__file__).resolve(),
        Path(base.__file__).resolve(),
        Path(__file__).resolve(),
        base.TOOLS_ROOT / "build_ocr_dstext_parseq_multi_shard_dry_notebooks.py",
    ]
    builder_source_sha256 = base.sha256_source_bundle(builder_paths)
    builder_git_commit, builder_git_dirty, builder_git_tracked = base.git_source_revision(builder_paths)
    builder_source_files = [
        str(path.resolve().relative_to(base.ROOT)).replace("\\", "/") for path in builder_paths
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
        "config code revision constants",
    )
    config = replace_once(
        config,
        "      quality_status=QUALITY_STATUS)",
        "      quality_status=QUALITY_STATUS, pipeline_code_sha256=PIPELINE_CODE_SHA256,\n"
        "      notebook_config_sha256=NOTEBOOK_CONFIG_SHA256,\n"
        "      builder_source_sha256=BUILDER_SOURCE_SHA256)",
        "config code revision phase",
    )

    cells = [
        base.pilot.cell(
            f"# {smoke_name}\n\n"
            f"Bounded full-pipeline OCR smoke for shard {target_shard_id[-4:]}: official DSText "
            "DeepSolo detector -> word crops -> Vietnamese PARSeq.\n\n"
            f"- Selects exactly {EXPECTED_FRAME_COUNT} deterministic frames from shard {target_shard_id[-4:]}; this is not a full-shard quality run.\n"
            "- Runs input/image preflight, runtime/model gates, two-worker inference, postflight, Parquet/JSONL artifacts and review HTML.\n"
            "- Threshold `0.30`; one full-frame detector pass; no recall tiles.\n"
            "- `frame_uid=video_id:source_frame_idx`; detector-scoped `crop_uid`; keyframe v1 remains immutable.\n"
            "- Artifacts are `ENGINEERING_PROXY`; OCR/retrieval quality remains `UNVALIDATED`.\n",
            "intro",
            "markdown",
        ),
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
                "quality_status": "UNVALIDATED",
                "provenance_class": "ENGINEERING_PROXY",
                "identity": "frame_uid=video_id:source_frame_idx",
                "target_shard_id": target_shard_id,
                "expected_frame_count": EXPECTED_FRAME_COUNT,
                "run_mode": "BOUNDED_SMOKE",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_dir = base.KERNEL_ROOT / smoke_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{smoke_name}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": f"{kernel_owner}/{smoke_name}",
        "title": smoke_name,
        "code_file": path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(execute),
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": None if not execute else "NvidiaTeslaT4",
        "keywords": [],
        "dataset_sources": [
            base.INVENTORY_DATASET_SLUG,
            target_dataset_slug,
            base.WEIGHT_DATASET_SLUG,
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def validate_notebook(path: Path, execute: bool, shard_id: str = DEFAULT_SHARD_ID) -> dict:
    if shard_id not in SMOKE_SPECS:
        raise ValueError(f"unsupported smoke shard: {shard_id}")
    spec = SMOKE_SPECS[shard_id]
    smoke_name = spec["name"]
    kernel_owner = spec["owner"]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for source in sources:
        ast.parse(source)
    code = "\n".join(sources)
    required = (
        f'TARGET_SHARD_ID = "{shard_id}"',
        f"MAX_FRAMES = {EXPECTED_FRAME_COUNT}",
        f"EXPECTED_FRAME_COUNT = {EXPECTED_FRAME_COUNT}",
        "FULL_SHARD = False",
        'RUN_MODE = "BOUNDED_SMOKE"',
        "DETECTION_SCORE_THRESHOLD = 0.30",
        "FULL_FRAME_ONLY = True",
        "RECALL_MODE = False",
        "TILE_PASS_COUNT = 0",
        "REQUESTED_GPU_WORKERS = 2",
        "CUDA_VISIBLE_DEVICES",
        "IMAGE_PREFLIGHT_GREEN",
        "failure_ledger",
        "NO_TEXT",
        "READ_FAILED",
        "INFERENCE_FAILED",
        "PARSE_ERROR",
        "final_manifest.json",
        "detection_status.parquet",
        "crop_inventory.parquet",
        "ocr_lines.parquet",
        "review_index.html",
        "PIPELINE_CODE_SHA256",
        "NOTEBOOK_CONFIG_SHA256",
        "BUILDER_SOURCE_SHA256",
        '"run_mode": RUN_MODE',
        "frame_uid=video_id:source_frame_idx",
        "DEEPSOLO_WEIGHT_SHA256",
        "PARSEQ_WEIGHT_SHA256",
        "network fallback is intentionally disabled",
        "WEIGHT_DATASET_SLUG",
    )
    missing = [token for token in required if token not in code]
    forbidden = [
        token for token in (
            "__RECOGNIZER_MODEL__",
            "PP-OCR",
            "VietOCR",
            "access_token",
            'faiss_row": str',
        )
        if token in code
    ]
    if missing or forbidden:
        raise ValueError({"missing": missing, "forbidden": forbidden})
    metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    if metadata["id"] != f"{kernel_owner}/{smoke_name}" or metadata["title"] != smoke_name:
        raise ValueError("kernel metadata id/title is not canonical")
    if metadata["enable_gpu"] != execute or metadata.get("machine_shape") != (
        "NvidiaTeslaT4" if execute else None
    ):
        raise ValueError("smoke GPU metadata mismatch")
    if execute and "EXECUTE_PIPELINE = False" in code:
        raise ValueError("execute smoke is not armed")
    return {
        "path": str(path),
        "cells": len(notebook["cells"]),
        "execute": execute,
        "enable_gpu": metadata["enable_gpu"],
        "machine_shape": metadata.get("machine_shape"),
        "target_shard_id": shard_id,
        "smoke_frame_count": EXPECTED_FRAME_COUNT,
        "source_chars": len(code),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="arm bounded smoke execution")
    parser.add_argument("--shard", choices=sorted(SMOKE_SPECS), default=DEFAULT_SHARD_ID)
    args = parser.parse_args()
    path = make_notebook(args.execute, shard_id=args.shard)
    print(json.dumps(validate_notebook(path, args.execute, shard_id=args.shard), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

