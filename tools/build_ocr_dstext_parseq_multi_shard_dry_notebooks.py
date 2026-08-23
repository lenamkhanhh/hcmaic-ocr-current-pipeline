"""Build dry or explicitly promoted DeepSolo DSText + Vietnamese PARSeq notebooks.

The shard0005 builder remains the source of truth for the execution contract.
This wrapper only supplies the shard count, input dataset, canonical notebook
slug, and Kaggle owner for each shard.  It remains dry by default; ``--execute``
is an explicit promotion step for a separately reviewed shard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_ocr_dstext_parseq_shard0005_full_notebook as builder  # noqa: E402


SPECS = {
    "shard_0000": {
        "expected_frame_count": 24_930,
        "owner": "REPLACE_WITH_KAGGLE_OWNER",
        "name": "hcmaic-ocr-dstext-parseq-s0000-full-t4x2-20260820",
    },
    "shard_0001": {
        "expected_frame_count": 24_935,
        "owner": "REPLACE_WITH_KAGGLE_OWNER",
        "name": "hcmaic-ocr-dstext-parseq-s0001-full-t4x2-20260820",
    },
    "shard_0002": {
        "expected_frame_count": 24_926,
        "owner": "REPLACE_WITH_KAGGLE_OWNER",
        "name": "hcmaic-ocr-dstext-parseq-s0002-full-t4x2-20260820",
    },
    "shard_0003": {
        "expected_frame_count": 24_948,
        "owner": "REPLACE_WITH_KAGGLE_OWNER",
        "name": "hcmaic-ocr-dstext-parseq-s0003-full-t4x2-20260820",
    },
    "shard_0004": {
        "expected_frame_count": 24_906,
        "owner": "REPLACE_WITH_KAGGLE_OWNER",
        "name": "hcmaic-ocr-dstext-parseq-s0004-full-t4x2-20260820",
    },
}


def build_one(shard_id: str, execute: bool = False) -> dict:
    if shard_id not in SPECS:
        raise ValueError(f"unsupported dry shard: {shard_id}")
    spec = SPECS[shard_id]
    global_names = (
        "TARGET_SHARD_ID", "TARGET_DATASET_SLUG", "EXPECTED_FRAME_COUNT",
        "KERNEL_OWNER", "DRY_NAME", "EXECUTE_NAME",
    )
    previous = {name: getattr(builder, name) for name in global_names}
    try:
        builder.TARGET_SHARD_ID = shard_id
        builder.TARGET_DATASET_SLUG = f"REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-{shard_id.replace('_', '-')}-private"
        builder.EXPECTED_FRAME_COUNT = spec["expected_frame_count"]
        builder.KERNEL_OWNER = spec["owner"]
        builder.DRY_NAME = spec["name"]
        builder.EXECUTE_NAME = spec["name"]
        path = builder.make_notebook(execute)
        report = builder.validate_notebook(path, execute)
        metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
        return {
            **report,
            "shard_id": shard_id,
            "owner": spec["owner"],
            "execute": execute,
            "kernel_id": metadata["id"],
            "target_dataset_slug": builder.TARGET_DATASET_SLUG,
            "out_path": f"/kaggle/working/{spec['name']}",
        }
    finally:
        for name, value in previous.items():
            setattr(builder, name, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        nargs="+",
        default=list(SPECS),
        choices=list(SPECS),
        help="shards to generate; defaults to 0000-0004",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="arm the separate full-execution kernel; dry is the default",
    )
    args = parser.parse_args()
    reports = [build_one(shard_id, execute=args.execute) for shard_id in args.shards]
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

