"""Build deterministic full-execution partitions for OCR shard 0001.

The shard-0000 partition builder contains the tested notebook assembly logic.
This module reuses that logic with shard-specific identity, count, and kernel
ownership settings while preserving shard-0001 provenance in generated cells.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_ocr_dstext_parseq_shard0000_partition_notebooks as template


TARGET_SHARD_ID = "shard_0001"
TARGET_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0001-private"
GLOBAL_SHARD_FRAME_COUNT = 24_935
PARTITION_COUNT = 3
PARTITION_STRATEGY = "sorted_frame_uid_round_robin"
REVIEW_FRAME_COUNT = 12

PARTITION_SPECS = [
    {"index": 0, "owner": "REPLACE_WITH_KAGGLE_OWNER", "name": "hcmaic-ocr-dstext-parseq-s0001-part0-t4x2-20260820"},
    {"index": 1, "owner": "REPLACE_WITH_KAGGLE_OWNER", "name": "hcmaic-ocr-dstext-parseq-s0001-part1-t4x2-20260820"},
    {"index": 2, "owner": "REPLACE_WITH_KAGGLE_OWNER", "name": "hcmaic-ocr-dstext-parseq-s0001-part2-t4x2-20260820"},
]


def partition_expected_frame_count(partition_index: int) -> int:
    if not 0 <= partition_index < PARTITION_COUNT:
        raise ValueError(f"invalid partition index: {partition_index}")
    base_count, remainder = divmod(GLOBAL_SHARD_FRAME_COUNT, PARTITION_COUNT)
    return base_count + int(partition_index < remainder)


def _with_shard_settings(callback):
    names = (
        "TARGET_SHARD_ID",
        "TARGET_DATASET_SLUG",
        "GLOBAL_SHARD_FRAME_COUNT",
        "PARTITION_COUNT",
        "PARTITION_STRATEGY",
        "REVIEW_FRAME_COUNT",
        "PARTITION_SPECS",
    )
    previous = {name: getattr(template, name) for name in names}
    previous_file = template.__file__
    template.TARGET_SHARD_ID = TARGET_SHARD_ID
    template.TARGET_DATASET_SLUG = TARGET_DATASET_SLUG
    template.GLOBAL_SHARD_FRAME_COUNT = GLOBAL_SHARD_FRAME_COUNT
    template.PARTITION_COUNT = PARTITION_COUNT
    template.PARTITION_STRATEGY = PARTITION_STRATEGY
    template.REVIEW_FRAME_COUNT = REVIEW_FRAME_COUNT
    template.PARTITION_SPECS = PARTITION_SPECS
    # Include this shard-specific builder in generated provenance.
    template.__file__ = str(Path(__file__).resolve())
    try:
        return callback()
    finally:
        for name, value in previous.items():
            setattr(template, name, value)
        template.__file__ = previous_file


def make_notebook(execute: bool, partition_index: int) -> Path:
    if not 0 <= partition_index < PARTITION_COUNT:
        raise ValueError(f"invalid partition index: {partition_index}")
    return _with_shard_settings(
        lambda: template.make_notebook(execute, partition_index=partition_index)
    )


def validate_notebook(path: Path, execute: bool, partition_index: int) -> dict:
    if not 0 <= partition_index < PARTITION_COUNT:
        raise ValueError(f"invalid partition index: {partition_index}")
    return _with_shard_settings(
        lambda: template.validate_notebook(path, execute, partition_index=partition_index)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-index", type=int, choices=range(PARTITION_COUNT), required=True)
    parser.add_argument("--execute", action="store_true", help="arm the full partition execution")
    args = parser.parse_args()
    path = make_notebook(args.execute, partition_index=args.partition_index)
    print(validate_notebook(path, args.execute, args.partition_index))


if __name__ == "__main__":
    main()

