"""Versioned index artifacts: build, persist, load, and cross-validate.

Artifact directory layout:

    <artifacts>/
    ├── catalog.jsonl
    ├── dataset_manifest.json
    ├── embeddings.npy       # float32 [n, d], rows L2-normalized, catalog order
    ├── id_map.json          # frame_id per row
    └── index_manifest.json  # versions, hashes, config

Loading refuses to serve on any catalog/embeddings/id-map/manifest
disagreement (dimension, row count, id order) — the upstream 512-vs-1280
failure class is structurally rejected here.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.config import CompetitiveFoundationConfig, ProviderSpec, artifact_provenance
from hcmaic.contracts.models import FrameRecord
from hcmaic.embedding.base import EmbeddingProvider
from hcmaic.ingestion.catalog import CATALOG_NAME, load_catalog, write_catalog
from hcmaic.ingestion.manifest import (
    MANIFEST_NAME,
    build_dataset_manifest,
    manifest_hash,
    write_manifest,
)

INDEX_MANIFEST_NAME = "index_manifest.json"
EMBEDDINGS_NAME = "embeddings.npy"
ID_MAP_NAME = "id_map.json"

INDEX_FORMAT_VERSION = "hcmaic-index-v1"


class ArtifactError(RuntimeError):
    """Raised when artifacts are missing, corrupt, or inconsistent."""


def _code_version(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


@dataclass
class IndexArtifacts:
    catalog: list[FrameRecord]
    embeddings: np.ndarray
    id_map: list[str]
    index_manifest: dict[str, Any]
    dataset_manifest: dict[str, Any]
    artifacts_dir: Path

    @property
    def index_version(self) -> str:
        return str(self.index_manifest.get("index_version", "unknown"))


def build_index_artifacts(
    dataset_root: Path,
    catalog: list[FrameRecord],
    provider: EmbeddingProvider,
    out_dir: Path,
    index_provider: str = "exact-numpy",
    foundation_config: CompetitiveFoundationConfig | None = None,
) -> Path:
    """Embed the catalog and write the versioned artifact directory."""
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not catalog:
        raise ArtifactError(
            f"Catalog for {dataset_root} is empty — nothing to index. Run validate-data first."
        )

    image_paths = [dataset_root / r.image_path for r in catalog]
    embeddings = provider.embed_images(image_paths)
    if embeddings.shape != (len(catalog), provider.dimension):
        raise ArtifactError(
            f"Provider returned shape {embeddings.shape}, expected "
            f"({len(catalog)}, {provider.dimension})"
        )

    for record in catalog:
        record.embedding_version = provider.version

    dataset_manifest = build_dataset_manifest(dataset_root)
    ds_hash = manifest_hash(dataset_manifest)

    write_catalog(catalog, out_dir / CATALOG_NAME)
    write_manifest(dataset_manifest, out_dir / MANIFEST_NAME)
    np.save(out_dir / EMBEDDINGS_NAME, embeddings)
    with open(out_dir / ID_MAP_NAME, "w", encoding="utf-8") as f:
        json.dump([r.frame_id for r in catalog], f, indent=0)

    index_manifest = {
        "format": INDEX_FORMAT_VERSION,
        "index_version": f"{provider.version}+{ds_hash[:12]}",
        "dataset_manifest_hash": ds_hash,
        "dataset_root": str(dataset_root),
        "embedding": provider.info(),
        "index_provider": index_provider,
        "n_frames": len(catalog),
        "dimension": provider.dimension,
        "normalization": "l2",
        "code_version": _code_version(Path(__file__).resolve().parents[3]),
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    if foundation_config is None:
        foundation_config = CompetitiveFoundationConfig(
            dataset_adapter=ProviderSpec(name="local-fixture", version="v0"),
            ingestion_backend=ProviderSpec(name="raw-video", version="milestone-1"),
            shot_detector=ProviderSpec(name="uniform", version="fallback"),
            embedding_provider=ProviderSpec(
                name=provider.name,
                version=provider.version,
                params=(("dimension", provider.dimension),),
            ),
            index_provider=ProviderSpec(name=index_provider, version="v1"),
            fusion=ProviderSpec(name="single-stage", version="v1"),
            reranker=ProviderSpec(name="identity", version="v1"),
            benchmark_inputs=ProviderSpec(name="fixture", version="sample"),
            device="cpu",
            batch_size=1,
            seed=0,
        )
    index_manifest.update(
        artifact_provenance(
            foundation_config,
            code_version=str(index_manifest["code_version"]),
        )
    )
    write_manifest(index_manifest, out_dir / INDEX_MANIFEST_NAME)
    return out_dir


def load_index_artifacts(artifacts_dir: Path) -> IndexArtifacts:
    artifacts_dir = Path(artifacts_dir)
    required_files = (
        CATALOG_NAME,
        MANIFEST_NAME,
        EMBEDDINGS_NAME,
        ID_MAP_NAME,
        INDEX_MANIFEST_NAME,
    )
    for required in required_files:
        if not (artifacts_dir / required).is_file():
            raise ArtifactError(
                f"Missing artifact {required} in {artifacts_dir}. "
                f"Run: hcmaic build-index --input <dataset> --output {artifacts_dir}"
            )

    catalog = load_catalog(artifacts_dir / CATALOG_NAME)
    embeddings = np.load(artifacts_dir / EMBEDDINGS_NAME, allow_pickle=False)
    with open(artifacts_dir / ID_MAP_NAME, encoding="utf-8") as f:
        id_map = json.load(f)
    with open(artifacts_dir / INDEX_MANIFEST_NAME, encoding="utf-8") as f:
        index_manifest = json.load(f)
    with open(artifacts_dir / MANIFEST_NAME, encoding="utf-8") as f:
        dataset_manifest = json.load(f)

    n = len(catalog)
    if index_manifest.get("format") != INDEX_FORMAT_VERSION:
        raise ArtifactError(
            f"Unsupported index format {index_manifest.get('format')!r}; "
            f"expected {INDEX_FORMAT_VERSION!r}. Rebuild the index."
        )
    if embeddings.ndim != 2:
        raise ArtifactError(f"embeddings.npy must be 2-D, got {embeddings.shape}")
    if embeddings.shape[0] != n or len(id_map) != n:
        raise ArtifactError(
            f"Artifact row mismatch: catalog={n}, embeddings={embeddings.shape[0]}, "
            f"id_map={len(id_map)}. Rebuild the index."
        )
    if int(index_manifest.get("n_frames", -1)) != n:
        raise ArtifactError(
            f"Frame count mismatch: manifest says "
            f"{index_manifest.get('n_frames')}, catalog has {n}. Rebuild the index."
        )
    if int(index_manifest.get("dimension", -1)) != int(embeddings.shape[1]):
        raise ArtifactError(
            f"Dimension mismatch: manifest says {index_manifest.get('dimension')}, "
            f"embeddings.npy has {embeddings.shape[1]}. Rebuild the index."
        )
    embedding_manifest = index_manifest.get("embedding", {})
    if int(embedding_manifest.get("dimension", -1)) != int(embeddings.shape[1]):
        raise ArtifactError(
            f"Embedding dimension mismatch: provider manifest says "
            f"{embedding_manifest.get('dimension')}, embeddings.npy has "
            f"{embeddings.shape[1]}. Rebuild the index."
        )
    expected_dataset_hash = str(index_manifest.get("dataset_manifest_hash", ""))
    actual_dataset_hash = manifest_hash(dataset_manifest)
    if not expected_dataset_hash or expected_dataset_hash != actual_dataset_hash:
        raise ArtifactError(
            "Dataset manifest hash mismatch: index_manifest.json and "
            "dataset_manifest.json do not describe the same dataset. Rebuild the index."
        )
    if not np.isfinite(embeddings).all():
        raise ArtifactError("embeddings.npy contains non-finite values. Rebuild the index.")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
        raise ArtifactError("embeddings.npy rows are not L2-normalized. Rebuild the index.")
    catalog_ids = [r.frame_id for r in catalog]
    if catalog_ids != list(id_map):
        raise ArtifactError(
            "id_map.json order does not match catalog.jsonl order. Rebuild the "
            "index; do not hand-edit artifacts."
        )
    return IndexArtifacts(
        catalog=catalog,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        id_map=list(id_map),
        index_manifest=index_manifest,
        dataset_manifest=dataset_manifest,
        artifacts_dir=artifacts_dir,
    )
