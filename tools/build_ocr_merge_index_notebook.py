"""Build the six-shard OCR merge/index notebooks.

The notebook consumes structured outputs from the 12 completed OCR kernels. It
does not download JPEGs or rerun detector/recognizer inference.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import textwrap
from pathlib import Path

OWNER = "REPLACE_WITH_KAGGLE_OWNER"
SMOKE_NAME = "hcmaic-ocr-merge-index-6shards-smoke-20260821"
FULL_NAME = "hcmaic-ocr-merge-index-6shards-full-20260821"
TRANSFER_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-merge-input-6shards-20260821-private"
OUT_ROOT = Path(__file__).resolve().parents[1] / "_deliverables" / "ocr_merge_index_notebooks"

SOURCE_KERNELS = [
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0000-part0-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0000-part1-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0000-part2-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0001-part0-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0001-part1-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0001-part2-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0002-part0-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0002-part1-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0002-part2-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0003-full-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0004-full-t4x2-20260820",
    "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-s0005-full-t4x2-20260820",
]

EXPECTED_LOGICAL_SHARDS = {
    "shard_0000": {"mode": "partition", "count": 3, "global_frame_count": 24930},
    "shard_0001": {"mode": "partition", "count": 3, "global_frame_count": 24935},
    "shard_0002": {"mode": "partition", "count": 3, "global_frame_count": 24926},
    "shard_0003": {"mode": "full", "expected_frame_count": 24948},
    "shard_0004": {"mode": "full", "expected_frame_count": 24906},
    "shard_0005": {"mode": "full", "expected_frame_count": 21476},
}

DETECTOR = {
    "model": "DeepSolo ResNet-50 DSText official",
    "revision": "dbadae995035246bad3376c7a44c015c69e9b313",
    "weight_sha256": "d48cd9212573b544d2a9503c65f2e4a75c80f598dbf81eb739f6dc63d3d27e0c",
}
RECOGNIZER = {
    "model": "PARSeq Vietnamese fine-tune",
    "revision": "76cc5f3cc6268457aac764653400fdff681f8271",
    "checkpoint_sha256": "8089b13c5ad115a96a608c6401eaab36b081393ea0a8323537b29a2dc80168f5",
}
MERGE_SOURCE_PATH = Path(__file__).resolve().parents[1] / "src" / "hcmaic" / "ingestion" / "ocr_merge.py"
MERGE_SOURCE = MERGE_SOURCE_PATH.read_text(encoding="utf-8")

CONFIG = r'''
from __future__ import annotations

import copy
import hashlib
import json
import platform
import shutil
import sys
import threading
import time
import types
from pathlib import Path

MODE = __MODE__
EXECUTE_PIPELINE = True
MAX_ROWS_PER_SOURCE = __MAX_ROWS__
MAX_STATUS_ROWS_PER_SOURCE = __MAX_ROWS__
BATCH_SIZE = __BATCH_SIZE__
HEARTBEAT_SECONDS = 240
INPUT_ROOT = Path("/kaggle/input")
RUN_ROOT = Path("/kaggle/working/hcmaic-ocr-merge-index-run")
OUTPUT = RUN_ROOT / "merged"
INDEX_NAME = "hcmaic_ocr_v1"
QUALITY_STATUS = "UNVALIDATED"
PROVENANCE_CLASS = "ENGINEERING_PROXY"
IDENTITY_CONTRACT = (
    "frame_uid=video_id:source_frame_idx; crop_uid is OCR line identity; "
    "faiss_row is not identity"
)
SOURCE_KERNELS = __SOURCE_KERNELS__
EXPECTED_LOGICAL_SHARDS = __EXPECTED_LOGICAL_SHARDS__
EXPECTED_DETECTOR = __DETECTOR__
EXPECTED_RECOGNIZER = __RECOGNIZER__
if not EXECUTE_PIPELINE:
    raise RuntimeError("This notebook is execution-enabled; use the builder for smoke/full")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
PHASE_LOG = RUN_ROOT / "phase_status.jsonl"


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def phase(name: str, **fields: object) -> None:
    row = {
        "phase": name,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": MODE,
        **fields,
    }
    with PHASE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    print("[PHASE] " + json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


phase(
    "CONFIG_READY",
    execute_pipeline=True,
    source_kernel_count=len(SOURCE_KERNELS),
    logical_shard_count=len(EXPECTED_LOGICAL_SHARDS),
    max_rows_per_source=MAX_ROWS_PER_SOURCE,
    batch_size=BATCH_SIZE,
    device="cpu",
    quality_status=QUALITY_STATUS,
)
'''

SOURCE_PREFLIGHT = r'''
def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object expected: {path}")
    return payload


def source_key(manifest: dict) -> str:
    target = str(manifest.get("target_shard_id", "")).strip()
    if manifest.get("full_shard") is True:
        return f"{target}#full"
    partition = manifest.get("partition")
    if not isinstance(partition, dict):
        raise RuntimeError(f"{target}: partition set contract is missing")
    return f"{target}#part{int(partition['index'])}/{int(partition['count'])}"


def find_artifact(root: Path, names: tuple[str, ...], label: str) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise RuntimeError(f"{label} missing under {root}")


def validate_source(root: Path, manifest_path: Path, manifest: dict) -> dict:
    target = str(manifest.get("target_shard_id", "")).strip()
    if target not in EXPECTED_LOGICAL_SHARDS:
        raise RuntimeError(f"unexpected target shard {target!r}")
    spec = EXPECTED_LOGICAL_SHARDS[target]
    if manifest.get("status") != "ENGINEERING_ARTIFACT_COMPLETE":
        raise RuntimeError(f"{target}: source status is not complete")
    if manifest.get("execution_status") not in ("COMPLETE", "COMPLETE_WITH_REPORT_FAILURE"):
        raise RuntimeError(f"{target}: source execution is not complete")
    if manifest.get("quality_status") not in ("UNVALIDATED", "UNVALIDATED_ON_HCMAIC"):
        raise RuntimeError(f"{target}: unsafe quality status")
    if manifest.get("provenance_class") != PROVENANCE_CLASS:
        raise RuntimeError(f"{target}: unsafe provenance class")
    identity = str(manifest.get("identity", ""))
    if "frame_uid=video_id:source_frame_idx" not in identity or "faiss_row" not in identity:
        raise RuntimeError(f"{target}: unsafe identity contract")
    for key, expected in EXPECTED_DETECTOR.items():
        if manifest.get("detector", {}).get(key) != expected:
            raise RuntimeError(f"{target}: detector {key} mismatch")
    for key, expected in EXPECTED_RECOGNIZER.items():
        if manifest.get("recognizer", {}).get(key) != expected:
            raise RuntimeError(f"{target}: recognizer {key} mismatch")

    key = source_key(manifest)
    if spec["mode"] == "full":
        if manifest.get("full_shard") is not True:
            raise RuntimeError(f"{target}: expected full-shard output")
        if int(manifest.get("expected_frame_count", -1)) != spec["expected_frame_count"]:
            raise RuntimeError(f"{target}: full frame count mismatch")
    else:
        partition = manifest.get("partition")
        if manifest.get("full_shard") is True or not isinstance(partition, dict):
            raise RuntimeError(f"{target}: expected partition output")
        if int(partition.get("count", -1)) != spec["count"]:
            raise RuntimeError(f"{target}: partition count mismatch")
        if int(partition.get("global_frame_count", -1)) != spec["global_frame_count"]:
            raise RuntimeError(f"{target}: global partition frame count mismatch")
        if not 0 <= int(partition.get("index", -1)) < spec["count"]:
            raise RuntimeError(f"{target}: partition index out of range")

    lines_path = find_artifact(
        root,
        ("ocr_lines.parquet", "ocr_lines.jsonl", "parseq_ocr_lines.parquet", "parseq_ocr_lines.jsonl"),
        "OCR line artifact",
    )
    status_path = find_artifact(
        root,
        (
            "detection_status.parquet",
            "detection_status.jsonl",
            "parseq_detection_status.parquet",
            "parseq_detection_status.jsonl",
        ),
        "detection status artifact",
    )
    ledger_path = root / "failure_ledger.json"
    if not ledger_path.is_file():
        raise RuntimeError(f"{target}: failure_ledger.json is missing")
    hashes = manifest.get("artifact_hashes", {})
    for required in ("failure_ledger.json", lines_path.name, status_path.name):
        if not isinstance(hashes, dict) or required not in hashes:
            raise RuntimeError(f"{target}: declared hash missing for {required}")
    postflight = manifest.get("postflight")
    postflight_status = postflight.get("status") if isinstance(postflight, dict) else None
    if postflight_status not in (None, "POSTFLIGHT_GREEN"):
        raise RuntimeError(f"{target}: source postflight is not green")
    return {
        "source_key": key,
        "root": root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "lines_path": lines_path,
        "status_path": status_path,
        "ledger_path": ledger_path,
        "postflight_status": postflight_status,
    }


found = {}
for manifest_path in sorted(INPUT_ROOT.rglob("final_manifest.json")):
    try:
        manifest = read_json(manifest_path)
        if str(manifest.get("target_shard_id", "")) not in EXPECTED_LOGICAL_SHARDS:
            continue
        source = validate_source(manifest_path.parent, manifest_path, manifest)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        continue
    if source["source_key"] in found:
        raise RuntimeError(f"duplicate source manifest for {source['source_key']}")
    found[source["source_key"]] = source

expected_keys = set()
for target, spec in EXPECTED_LOGICAL_SHARDS.items():
    if spec["mode"] == "full":
        expected_keys.add(f"{target}#full")
    else:
        expected_keys.update(f"{target}#part{i}/{spec['count']}" for i in range(spec["count"]))
if set(found) != expected_keys:
    raise RuntimeError(
        "partition set/source count mismatch: "
        f"expected={sorted(expected_keys)} found={sorted(found)}"
    )
SOURCES = [found[key] for key in sorted(found)]
LEGACY_MISSING_POSTFLIGHT = [
    source["source_key"] for source in SOURCES if source["postflight_status"] is None
]
SOURCE_CONTRACT = {
    "status": "SOURCE_PREFLIGHT_GREEN",
    "source_kernel_count": len(SOURCE_KERNELS),
    "source_count": len(SOURCES),
    "logical_shard_count": len(EXPECTED_LOGICAL_SHARDS),
    "source_keys": [source["source_key"] for source in SOURCES],
    "legacy_missing_postflight": LEGACY_MISSING_POSTFLIGHT,
    "identity": IDENTITY_CONTRACT,
    "model_contract": {"detector": EXPECTED_DETECTOR, "recognizer": EXPECTED_RECOGNIZER},
    "sources": [
        {
            "source_key": source["source_key"],
            "target_shard_id": source["manifest"]["target_shard_id"],
            "root": str(source["root"]),
            "final_manifest_sha256": source["manifest_sha256"],
            "selection_sha256": source["manifest"].get("selection_sha256"),
            "expected_frame_count": source["manifest"].get("expected_frame_count"),
            "crop_count": source["manifest"].get("crop_count"),
            "recognition_count": source["manifest"].get("recognition_count"),
            "failure_count": source["manifest"].get("failure_count"),
            "postflight_status": source["postflight_status"],
            "code_revision": source["manifest"].get("code_revision", {}),
        }
        for source in SOURCES
    ],
    "quality_status": QUALITY_STATUS,
    "provenance_class": PROVENANCE_CLASS,
}
write_json(RUN_ROOT / "source_contract.json", SOURCE_CONTRACT)
phase(
    "SOURCE_PREFLIGHT_GREEN",
    sources=len(SOURCES),
    logical_shards=len(EXPECTED_LOGICAL_SHARDS),
    legacy_missing_postflight=len(LEGACY_MISSING_POSTFLIGHT),
)
'''

MERGE_SETUP = r'''
# Execute the tested local merger in this isolated notebook namespace.  The
# only local dependency it needs is the two Unicode normalization functions.
ocr_text_module = types.ModuleType("hcmaic.retrieval.ocr_text")


def normalize_ocr_nfc(value: object) -> str:
    return "" if value is None else __import__("unicodedata").normalize("NFC", str(value)).strip()


def fold_ocr_text(value: object) -> str:
    import unicodedata
    text = normalize_ocr_nfc(value).translate(str.maketrans({"đ": "d", "Đ": "D"}))
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    return " ".join("".join(c for c in decomposed if not unicodedata.combining(c)).split())


ocr_text_module.normalize_ocr_nfc = normalize_ocr_nfc
ocr_text_module.fold_ocr_text = fold_ocr_text
sys.modules.setdefault("hcmaic", types.ModuleType("hcmaic"))
sys.modules.setdefault("hcmaic.retrieval", types.ModuleType("hcmaic.retrieval"))
sys.modules["hcmaic.retrieval.ocr_text"] = ocr_text_module
MERGE_NAMESPACE = {}
exec(__MERGE_SOURCE__, MERGE_NAMESPACE)
merge_ocr_shards = MERGE_NAMESPACE["merge_ocr_shards"]
iter_artifact_rows = MERGE_NAMESPACE["iter_artifact_rows"]
phase("MERGE_RUNTIME_GREEN", merger_format=MERGE_NAMESPACE["OCR_MERGED_FORMAT"])
'''

EXECUTE = r'''
def write_bounded_source_inputs() -> list[Path]:
    if MODE != "SMOKE":
        return [source["root"] for source in SOURCES]
    bounded_root = RUN_ROOT / "bounded_sources"
    if bounded_root.exists():
        shutil.rmtree(bounded_root)
    bounded_root.mkdir(parents=True)
    counts_by_target = {}
    bounded = []

    def take_rows(path: Path, limit: int) -> list[dict]:
        rows = []
        for index, row in enumerate(
            iter_artifact_rows(path, batch_size=min(BATCH_SIZE, max(1, limit)))
        ):
            rows.append(row)
            if index + 1 >= limit:
                break
        return rows

    for source in SOURCES:
        source_dir = bounded_root / source["source_key"].replace("#", "_").replace("/", "_")
        source_dir.mkdir()
        status_rows = take_rows(source["status_path"], MAX_STATUS_ROWS_PER_SOURCE)
        line_rows = take_rows(source["lines_path"], MAX_ROWS_PER_SOURCE)
        status_path = source_dir / "detection_status.jsonl"
        lines_path = source_dir / "ocr_lines.jsonl"
        status_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in status_rows),
            encoding="utf-8",
        )
        lines_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in line_rows),
            encoding="utf-8",
        )
        shutil.copy2(source["ledger_path"], source_dir / "failure_ledger.json")
        manifest = copy.deepcopy(source["manifest"])
        manifest["status"] = "ENGINEERING_ARTIFACT_COMPLETE"
        manifest["execution_status"] = "COMPLETE"
        manifest["quality_status"] = "UNVALIDATED"
        manifest["artifact_hashes"] = {}
        manifest["expected_frame_count"] = len(status_rows)
        manifest["frame_status_count"] = len(status_rows)
        manifest["selection_count"] = len(status_rows)
        manifest["crop_count"] = len(line_rows)
        manifest["recognition_count"] = len(line_rows)
        manifest["postflight"] = {"status": "POSTFLIGHT_GREEN", "derived_from": source["source_key"]}
        if isinstance(manifest.get("partition"), dict):
            target = manifest["target_shard_id"]
            counts_by_target[target] = counts_by_target.get(target, 0) + len(status_rows)
        bounded.append((source, source_dir, manifest))
    for source, source_dir, manifest in bounded:
        if isinstance(manifest.get("partition"), dict):
            manifest["partition"]["global_frame_count"] = counts_by_target[manifest["target_shard_id"]]
        write_json(source_dir / "final_manifest.json", manifest)
    phase(
        "BOUNDED_SOURCE_READY",
        sources=len(bounded),
        max_rows_per_source=MAX_ROWS_PER_SOURCE,
        max_status_rows_per_source=MAX_STATUS_ROWS_PER_SOURCE,
    )
    return [source_dir for _, source_dir, _ in bounded]


phase("MERGE_START", mode=MODE, require_postflight_green=False)
merge_roots = write_bounded_source_inputs()
done = threading.Event()


def heartbeat_loop() -> None:
    while not done.wait(HEARTBEAT_SECONDS):
        phase("MERGE_HEARTBEAT", mode=MODE, output=str(OUTPUT))


heartbeat_thread = threading.Thread(target=heartbeat_loop, name="ocr-merge-heartbeat", daemon=True)
heartbeat_thread.start()
try:
    SUMMARY = merge_ocr_shards(
        merge_roots,
        OUTPUT,
        batch_size=BATCH_SIZE,
        require_postflight_green=False,
        allow_report_failures=False,
    )
finally:
    done.set()
    heartbeat_thread.join(timeout=2)

if MODE == "SMOKE":
    SUMMARY["status"] = "ENGINEERING_ARTIFACT_SMOKE_COMPLETE"
    SUMMARY["execution_status"] = "COMPLETE_SMOKE"
write_json(OUTPUT / "runtime_manifest.json", {
    "status": "RUNTIME_GREEN",
    "mode": MODE,
    "device": "cpu",
    "batch_size": BATCH_SIZE,
    "heartbeat_seconds": HEARTBEAT_SECONDS,
    "source_kernel_count": len(SOURCE_KERNELS),
    "logical_shard_count": len(EXPECTED_LOGICAL_SHARDS),
    "quality_status": QUALITY_STATUS,
    "provenance_class": PROVENANCE_CLASS,
})
shutil.copy2(RUN_ROOT / "source_contract.json", OUTPUT / "source_contract.json")
phase("MERGE_COMPLETE", **SUMMARY)
'''

INDEX_POSTFLIGHT = r'''
preview_limit = 64 if MODE == "SMOKE" else 0
preview_count = 0
if preview_limit:
    with (OUTPUT / "ocr_es_bulk_preview.ndjson").open("w", encoding="utf-8") as handle:
        for row in iter_artifact_rows(OUTPUT / "ocr_lines.parquet", batch_size=BATCH_SIZE):
            document = {
                "_id": row.get("crop_uid"),
                "frame_uid": row.get("frame_uid"),
                "video_id": row.get("video_id"),
                "source_frame_idx": row.get("source_frame_idx"),
                "crop_uid": row.get("crop_uid"),
                "text_raw": row.get("ocr_text_raw", ""),
                "text_nfc": row.get("ocr_text_nfc", ""),
                "text_folded": row.get("ocr_text_folded", ""),
                "quality_status": QUALITY_STATUS,
                "provenance_class": PROVENANCE_CLASS,
            }
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")
            preview_count += 1
            if preview_count >= preview_limit:
                break

index_manifest = {
    "format": "hcmaic-dstext-parseq-ocr-es-index-v1",
    "status": "INDEX_EXPORT_READY",
    "index_name": INDEX_NAME,
    "document_id": "crop_uid",
    "collapse_key": "frame_uid",
    "text_fields": ["text_raw", "text_nfc", "text_folded"],
    "query_contract": "exact/match_phrase/fuzzy/n-gram then collapse to frame_uid",
    "source_artifact": "ocr_lines.parquet",
    "row_count": SUMMARY["line_count"],
    "preview_count": preview_count,
    "es_mutation": "localhost adapter after structured artifact transfer",
    "quality_status": QUALITY_STATUS,
    "provenance_class": PROVENANCE_CLASS,
}
write_json(OUTPUT / "index_manifest.json", index_manifest)

merged_manifest = read_json(OUTPUT / "ocr_manifest.json")
merged_manifest.update({
    "run_mode": MODE,
    "source_kernel_count": len(SOURCE_KERNELS),
    "logical_shard_count": len(EXPECTED_LOGICAL_SHARDS),
    "source_contract": "source_contract.json",
    "runtime_manifest": "runtime_manifest.json",
    "index_manifest": "index_manifest.json",
    "coverage": {
        "identity_gate": "FULL" if MODE == "FULL" else "BOUNDED_BATCH",
        "max_rows_per_source": MAX_ROWS_PER_SOURCE,
        "max_status_rows_per_source": MAX_STATUS_ROWS_PER_SOURCE,
        "source_kernel_count": len(SOURCE_KERNELS),
        "logical_shard_count": len(EXPECTED_LOGICAL_SHARDS),
    },
})
if MODE == "SMOKE":
    merged_manifest["status"] = "ENGINEERING_ARTIFACT_SMOKE_COMPLETE"
    merged_manifest["execution_status"] = "COMPLETE_SMOKE"
write_json(OUTPUT / "ocr_manifest.json", merged_manifest)
write_json(OUTPUT / "final_manifest.json", merged_manifest)
phase(
    "POSTFLIGHT_GREEN",
    status=merged_manifest["status"],
    execution_status=merged_manifest["execution_status"],
    frame_count=merged_manifest["counts"]["frame_count"],
    line_count=merged_manifest["counts"]["line_count"],
    failure_count=merged_manifest["counts"]["failure_count"],
    quality_status=QUALITY_STATUS,
    provenance_class=PROVENANCE_CLASS,
)
print("ENGINEERING_PROXY merge/index artifact ready; OCR/retrieval quality remains UNVALIDATED")
'''


def cell(source: str, cell_type: str, cell_id: str) -> dict:
    result = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.strip() + "\n",
        "id": cell_id,
    }
    if cell_type == "code":
        result["execution_count"] = None
        result["outputs"] = []
    return result


def render(template: str, mode: str) -> str:
    max_rows = "512" if mode == "SMOKE" else "None"
    batch_size = "4096" if mode == "SMOKE" else "50000"
    replacements = {
        "__MODE__": repr(mode),
        "__MAX_ROWS__": max_rows,
        "__BATCH_SIZE__": batch_size,
        "__SOURCE_KERNELS__": repr(SOURCE_KERNELS),
        "__EXPECTED_LOGICAL_SHARDS__": repr(EXPECTED_LOGICAL_SHARDS),
        "__DETECTOR__": repr(DETECTOR),
        "__RECOGNIZER__": repr(RECOGNIZER),
        "__MERGE_SOURCE__": repr(MERGE_SOURCE),
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def build_notebook(mode: str, root: Path | None = None) -> Path:
    mode = mode.upper()
    if mode not in {"SMOKE", "FULL"}:
        raise ValueError("mode must be SMOKE or FULL")
    slug = SMOKE_NAME if mode == "SMOKE" else FULL_NAME
    out_dir = (Path(root) if root is not None else OUT_ROOT) / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            cell(
                f"# HCMAIC OCR merge/index {mode.lower()}\n\n"
                "Consumes 12 private kernel outputs representing six logical shards. "
                "No source JPEGs are downloaded or rerun. "
                "Execution is ENGINEERING_PROXY only; quality remains UNVALIDATED.",
                "markdown",
                "intro",
            ),
            cell(render(CONFIG, mode), "code", "config-contract"),
            cell(SOURCE_PREFLIGHT, "code", "source-preflight"),
            cell(render(MERGE_SETUP, mode), "code", "merger-runtime"),
            cell(render(EXECUTE, mode), "code", "streaming-merge"),
            cell(INDEX_POSTFLIGHT, "code", "index-postflight"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "hcmaic": {
                "status": "EXECUTION_ARMED",
                "run_mode": mode,
                "quality_status": "UNVALIDATED",
                "provenance_class": "ENGINEERING_PROXY",
                "identity": "frame_uid=video_id:source_frame_idx",
                "source_kernel_count": len(SOURCE_KERNELS),
                "logical_shard_count": len(EXPECTED_LOGICAL_SHARDS),
                "contains_secrets": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path = out_dir / "main.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": f"{OWNER}/{slug}",
        "title": slug,
        "code_file": "main.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["hcmaic", "ocr", "merge", "elasticsearch"],
        # Cross-account kernel ACLs are not expressible in kernel-metadata.json.
        # The structured OCR transfer dataset is therefore the executable input;
        # SOURCE_KERNELS remains embedded in the notebook as provenance.
        "dataset_sources": [TRANSFER_DATASET_SLUG],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "None",
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        f"# {slug}\n\n"
        f"Mode: {mode}. Sources: 12 kernel outputs / 6 logical shards. "
        "The notebook writes a streaming crop-level OCR artifact and an "
        "Elasticsearch index contract; localhost ES mutation happens after "
        "structured output transfer. Quality remains UNVALIDATED.\n",
        encoding="utf-8",
    )
    validate_notebook(notebook_path, mode)
    return notebook_path


def validate_notebook(path: Path, mode: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = [
        "".join(item.get("source", []))
        for item in payload["cells"]
        if item.get("cell_type") == "code"
    ]
    for source in sources:
        ast.parse(source)
    code = "\n".join(sources)
    required = (
        f"MODE = '{mode}'",
        "EXECUTE_PIPELINE = True",
        "SOURCE_PREFLIGHT_GREEN",
        "partition set",
        "frame_uid=video_id:source_frame_idx",
        "crop_uid",
        "failure_ledger.json",
        "ocr_lines.parquet",
        "ocr_manifest.json",
        "index_manifest.json",
        "HEARTBEAT_SECONDS",
    )
    missing = [marker for marker in required if marker not in code]
    forbidden = [marker for marker in ("KAGGLE_API_TOKEN", "access_token", "__MODE__") if marker in code]
    if missing or forbidden:
        raise ValueError({"missing": missing, "forbidden": forbidden})
    metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    slug = SMOKE_NAME if mode == "SMOKE" else FULL_NAME
    if (
        metadata.get("id") != f"{OWNER}/{slug}"
        or metadata.get("dataset_sources") != [TRANSFER_DATASET_SLUG]
        or metadata.get("kernel_sources") != []
    ):
        raise ValueError("kernel metadata source contract mismatch")
    if metadata.get("enable_gpu") is not False or metadata.get("machine_shape") != "None":
        raise ValueError("merge/index notebook must be CPU-only")
    if mode == "SMOKE" and "MAX_ROWS_PER_SOURCE = 512" not in code:
        raise ValueError("smoke row bound missing")
    if mode == "FULL" and "MAX_ROWS_PER_SOURCE = None" not in code:
        raise ValueError("full row bound mismatch")
    return {
        "path": str(path),
        "mode": mode,
        "cells": len(payload["cells"]),
        "source_kernel_count": len(SOURCE_KERNELS),
        "logical_shard_count": len(EXPECTED_LOGICAL_SHARDS),
        "enable_gpu": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("SMOKE", "FULL"), default="SMOKE")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    path = build_notebook(args.mode, args.out_root)
    print(json.dumps(validate_notebook(path, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

