"""Dataset manifest: content hashes for reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_NAME = "dataset_manifest.json"

_HASHED_SUFFIXES = (".csv", ".json", ".jpg", ".jpeg", ".png")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset_manifest(root: Path) -> dict[str, Any]:
    """Hash every dataset content file (sorted, relative POSIX paths)."""
    root = Path(root)
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _HASHED_SUFFIXES:
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "dataset_root": str(root),
        "n_files": len(files),
        "files": files,
        "dataset_hash": combined,
    }


def manifest_hash(manifest: dict[str, Any]) -> str:
    return str(manifest.get("dataset_hash", ""))


def write_manifest(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)


def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: manifest must be a JSON object")
    return data
