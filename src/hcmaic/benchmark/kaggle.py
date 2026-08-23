"""Metadata-only Kaggle packaging for the SkillPixel KIS benchmark."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PACKAGE_FORMAT = "hcmaic-skillpixel-kaggle-package-v1"
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024
_FORBIDDEN_SUFFIXES = {
    ".avi",
    ".bin",
    ".ckpt",
    ".faiss",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pth",
    ".pt",
    ".safetensors",
    ".webm",
}
_FORBIDDEN_NAMES = {
    ".env",
    "credentials.json",
    "kaggle.json",
    "service-account.json",
}
_MANIFEST_NAME = "package_manifest.json"
_CHECKSUMS_NAME = "checksums.sha256"


class KagglePackageError(RuntimeError):
    """Raised when a package would contain unsafe or oversized material."""


@dataclass(frozen=True)
class KagglePackageConfig:
    output_dir: Path
    raw_input: Path
    questions_path: Path
    corpus_path: Path
    index_dir: Path | None
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES

    def __post_init__(self) -> None:
        if self.max_file_bytes < 1:
            raise ValueError("max_file_bytes must be positive")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hash(path: Path) -> str | None:
    path = Path(path)
    if path.is_file():
        return _sha256_file(path)
    manifest = path / "dataset_manifest.json"
    return _sha256_file(manifest) if manifest.is_file() else None


def _dataset_hash(path: Path) -> str | None:
    manifest = Path(path) / "dataset_manifest.json"
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = payload.get("dataset_hash")
    return str(value) if value else None


def _write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _relative_files(package_dir: Path) -> list[Path]:
    return sorted(
        (
            path.relative_to(package_dir)
            for path in package_dir.rglob("*")
            if path.is_file() and path.name not in {_MANIFEST_NAME, _CHECKSUMS_NAME}
        ),
        key=lambda path: path.as_posix(),
    )


def _forbidden_reason(path: Path, max_file_bytes: int) -> str | None:
    name = path.name.casefold()
    if name in _FORBIDDEN_NAMES:
        return f"forbidden credential file: {path.name}"
    if path.suffix.casefold() in _FORBIDDEN_SUFFIXES:
        return f"forbidden raw/model/generated artifact: {path.name}"
    if path.stat().st_size > max_file_bytes:
        return f"file exceeds {max_file_bytes} bytes: {path.name}"
    return None


def _entries(package_dir: Path, max_file_bytes: int) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative in _relative_files(package_dir):
        path = package_dir / relative
        reason = _forbidden_reason(path, max_file_bytes)
        if reason is not None:
            raise KagglePackageError(reason)
        entries.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return entries


def build_kaggle_package(config: KagglePackageConfig) -> dict[str, Path]:
    """Write a small recipe/manifest package without copying data or weights."""
    raw_input = Path(config.raw_input).resolve()
    questions_path = Path(config.questions_path).resolve()
    corpus_path = Path(config.corpus_path).resolve()
    if not raw_input.exists():
        raise FileNotFoundError(raw_input)
    if not questions_path.is_file():
        raise FileNotFoundError(questions_path)
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    if config.index_dir is not None and not Path(config.index_dir).is_dir():
        raise FileNotFoundError(config.index_dir)

    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise KagglePackageError(f"package output must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    readme = """# SkillPixel KIS Kaggle package

This package is intentionally metadata-only. It contains a reproducible run
recipe and input hashes; it does **not** contain raw videos, model weights,
embedding arrays, FAISS indexes, credentials, or tokens.

Attach the raw SkillPixel videos and a locally approved model cache as separate
Kaggle inputs. Set these environment variables in the notebook or terminal:

```text
SKILLPIXEL_RAW_INPUT=/kaggle/input/skillpixel-videos/videos
SKILLPIXEL_QUESTIONS=/kaggle/input/skillpixel-data/questions.csv
SKILLPIXEL_CORPUS=/kaggle/input/skillpixel-data/corpus.csv
SKILLPIXEL_RUN=/kaggle/working/skillpixel-kis-run
```

Run `run_skillpixel_kis.py` after installing this repository and making the
real provider weights available through an explicit Kaggle input/cache. The
script keeps local-files-only provider selection by default and writes all
generated material to `/kaggle/working`.

No Kaggle API token is required by this package and no upload is performed.
"""
    _write_text(output_dir / "README.md", readme)
    runner = """from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


raw_input = required("SKILLPIXEL_RAW_INPUT")
questions = required("SKILLPIXEL_QUESTIONS")
corpus = required("SKILLPIXEL_CORPUS")
run_root = Path(os.environ.get("SKILLPIXEL_RUN", "/kaggle/working/skillpixel-kis-run"))
raw_root = run_root / "raw"
index_root = run_root / "visual" / "clip-v0"
benchmark_root = run_root / "benchmark"


def run(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "hcmaic.cli.main", *args], check=True)


run("ingest-raw", "--input", raw_input, "--output", str(raw_root), "--stride-frames", "10")
run(
    "build-skillpixel-index",
    "--input", str(raw_root),
    "--output", str(index_root),
    "--provider", "clip", "--strict-provider",
)
run(
    "benchmark-skillpixel",
    "--raw", str(raw_root),
    "--index", str(index_root),
    "--questions", questions,
    "--corpus", corpus,
    "--out", str(benchmark_root),
    "--providers", "clip,siglip2,jina-clip-v2",
    "--no-build-missing",
)
"""
    _write_text(output_dir / "run_skillpixel_kis.py", runner)

    manifest_path = output_dir / _MANIFEST_NAME
    manifest = {
        "format": PACKAGE_FORMAT,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "raw_video_source": True,
        "contains_raw_videos": False,
        "contains_model_weights": False,
        "contains_tokens": False,
        "contains_generated_index": False,
        "raw_input": str(raw_input),
        "raw_dataset_manifest_sha256": _input_hash(raw_input),
        "raw_dataset_hash": _dataset_hash(raw_input),
        "questions_path": str(questions_path),
        "questions_sha256": _sha256_file(questions_path),
        "corpus_path": str(corpus_path),
        "corpus_sha256": _sha256_file(corpus_path),
        "index_dir": str(Path(config.index_dir).resolve()) if config.index_dir else None,
        "index_manifest_sha256": (
            _sha256_file(Path(config.index_dir) / "index_manifest.json")
            if config.index_dir and (Path(config.index_dir) / "index_manifest.json").is_file()
            else None
        ),
        "max_file_bytes": config.max_file_bytes,
        "files": [],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    entries = _entries(output_dir, config.max_file_bytes)
    manifest["files"] = entries
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksums_path = output_dir / _CHECKSUMS_NAME
    _write_text(
        checksums_path,
        "\n".join(f"{entry['sha256']}  {entry['path']}" for entry in entries),
    )
    validate_kaggle_package(output_dir)
    return {
        "readme": output_dir / "README.md",
        "runner": output_dir / "run_skillpixel_kis.py",
        "manifest": manifest_path,
        "checksums": checksums_path,
    }


def validate_kaggle_package(
    package_dir: Path, *, max_file_bytes: int | None = None
) -> dict[str, Any]:
    """Fail closed if a package contains raw/model/credential material."""
    package_dir = Path(package_dir).resolve()
    manifest_path = package_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        raise KagglePackageError(f"missing {_MANIFEST_NAME}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KagglePackageError("invalid package manifest") from exc
    if manifest.get("format") != PACKAGE_FORMAT:
        raise KagglePackageError("unsupported package format")
    for field in (
        "contains_raw_videos",
        "contains_model_weights",
        "contains_tokens",
        "contains_generated_index",
    ):
        if manifest.get(field) is not False:
            raise KagglePackageError(f"unsafe manifest flag: {field}")
    limit = int(max_file_bytes or manifest.get("max_file_bytes", DEFAULT_MAX_FILE_BYTES))
    entries = _entries(package_dir, limit)
    expected = {str(entry["path"]): entry for entry in manifest.get("files", [])}
    actual = {str(entry["path"]): entry for entry in entries}
    if expected != actual:
        raise KagglePackageError("package file manifest/hash mismatch")
    return {
        "valid": True,
        "format": PACKAGE_FORMAT,
        "n_files": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "contains_raw_videos": False,
        "contains_model_weights": False,
        "contains_tokens": False,
        "contains_generated_index": False,
    }
