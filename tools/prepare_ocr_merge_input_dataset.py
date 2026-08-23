"""Validate and manifest the structured OCR transfer dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_ocr_merge_index_notebook import (
    EXPECTED_LOGICAL_SHARDS,
    RECOGNIZER,
    SOURCE_KERNELS,
    DETECTOR,
    TRANSFER_DATASET_SLUG,
)

REQUIRED_FILES = (
    "final_manifest.json",
    "failure_ledger.json",
    "ocr_lines.parquet",
    "detection_status.parquet",
)
SOURCE_KEYS = (
    "s0000_part0",
    "s0000_part1",
    "s0000_part2",
    "s0001_part0",
    "s0001_part1",
    "s0001_part2",
    "s0002_part0",
    "s0002_part1",
    "s0002_part2",
    "s0003",
    "s0004",
    "s0005",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(stage: Path) -> dict[str, object]:
    source_root = stage / "ocr_sources"
    if not source_root.is_dir():
        raise RuntimeError(f"missing source root: {source_root}")
    actual_keys = tuple(sorted(path.name for path in source_root.iterdir() if path.is_dir()))
    if actual_keys != tuple(sorted(SOURCE_KEYS)):
        raise RuntimeError(f"source key mismatch: {actual_keys}")

    forbidden = []
    source_entries = []
    for key in SOURCE_KEYS:
        root = source_root / key
        files = {path.name: path for path in root.iterdir() if path.is_file()}
        unexpected = sorted(set(files) - set(REQUIRED_FILES))
        missing = sorted(set(REQUIRED_FILES) - set(files))
        if missing or unexpected:
            raise RuntimeError(f"{key}: missing={missing}, unexpected={unexpected}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"}:
                forbidden.append(str(path.relative_to(stage)))

        manifest = json.loads(files["final_manifest.json"].read_text(encoding="utf-8"))
        # Source OCR notebooks store the model contract at the manifest root;
        # the merged artifact later nests it under model_contract.
        detector = manifest.get("detector") or {}
        recognizer = manifest.get("recognizer") or {}
        if detector.get("revision") != DETECTOR["revision"]:
            raise RuntimeError(f"{key}: detector revision mismatch")
        if recognizer.get("revision") != RECOGNIZER["revision"]:
            raise RuntimeError(f"{key}: recognizer revision mismatch")
        source_entries.append(
            {
                "source_key": key,
                "kernel": SOURCE_KERNELS[SOURCE_KEYS.index(key)],
                "target_shard_id": manifest.get("target_shard_id"),
                "full_shard": manifest.get("full_shard"),
                "partition": manifest.get("partition"),
                "status": manifest.get("status"),
                "quality_status": manifest.get("quality_status"),
                "manifest_sha256": sha256(files["final_manifest.json"]),
                "files": {
                    name: {"size": path.stat().st_size, "sha256": sha256(path)}
                    for name, path in sorted(files.items())
                },
            }
        )

    if forbidden:
        raise RuntimeError(f"forbidden media files in transfer staging: {forbidden[:5]}")
    payload = {
        "schema_version": "hcmaic-ocr-merge-input-v1",
        "dataset_slug": TRANSFER_DATASET_SLUG,
        "provenance_class": "ENGINEERING_PROXY",
        "quality_status": "UNVALIDATED",
        "identity_contract": "frame_uid=video_id:source_frame_idx; crop_uid is OCR line identity; faiss_row is not identity",
        "source_count": len(source_entries),
        "logical_shard_count": len(EXPECTED_LOGICAL_SHARDS),
        "contains_jpeg": False,
        "contains_secrets": False,
        "model_contract": {"detector": DETECTOR, "recognizer": RECOGNIZER},
        "sources": source_entries,
    }
    output = stage / "transfer_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(args.stage.resolve())
    print(json.dumps({"state": "TRANSFER_PREFLIGHT_GREEN", "source_count": payload["source_count"], "contains_jpeg": False}, indent=2))


if __name__ == "__main__":
    main()

