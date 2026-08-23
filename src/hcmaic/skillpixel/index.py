"""Versioned NumPy-oracle and FAISS ``IndexFlatIP`` visual artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.contracts.models import FrameRecord
from hcmaic.embedding.base import EmbeddingProvider
from hcmaic.ingestion.catalog import build_catalog, load_catalog, write_catalog
from hcmaic.skillpixel.raw import RAW_MANIFEST_NAME, validate_raw_dataset

CATALOG_NAME = "catalog.jsonl"
EMBEDDINGS_NAME = "embeddings.npy"
ID_MAP_NAME = "id_map.json"
FAISS_NAME = "index.faiss"
INDEX_MANIFEST_NAME = "index_manifest.json"
PROVIDER_REPORT_NAME = "provider_report.json"
INDEX_FORMAT = "skillpixel-index-v1"


class SkillPixelIndexError(RuntimeError):
    """Raised when a SkillPixel index cannot be built or safely loaded."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _check_normalized(matrix: np.ndarray, *, label: str) -> None:
    if matrix.ndim != 2:
        raise SkillPixelIndexError(f"{label} must be 2-D, got {matrix.shape}")
    if matrix.dtype != np.float32:
        raise SkillPixelIndexError(f"{label} must be float32, got {matrix.dtype}")
    if not np.isfinite(matrix).all():
        raise SkillPixelIndexError(f"{label} contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4, rtol=1e-4):
        raise SkillPixelIndexError(f"{label} rows are not L2-normalized")


def _source_frame(record: FrameRecord) -> int:
    return record.source_frame_idx if record.source_frame_idx is not None else record.frame_idx


def _video_filename(record: FrameRecord) -> str:
    filename = record.video_filename or record.metadata.get("video_filename")
    if not filename:
        filename = f"{record.video_id}.mp4"
    return str(filename)


def _validate_catalog(catalog: list[FrameRecord]) -> None:
    seen_uid: set[str] = set()
    seen_source: set[tuple[str, int]] = set()
    for record in catalog:
        source_frame_idx = _source_frame(record)
        if record.frame_id in seen_uid:
            raise SkillPixelIndexError(f"duplicate frame_uid {record.frame_id}")
        seen_uid.add(record.frame_id)
        if record.frame_count is None or source_frame_idx >= record.frame_count:
            raise SkillPixelIndexError(
                f"source_frame_idx={source_frame_idx} out of range for {record.frame_id}"
            )
        if source_frame_idx < 0:
            raise SkillPixelIndexError(f"negative source_frame_idx for {record.frame_id}")
        key = (record.video_id, source_frame_idx)
        if key in seen_source:
            raise SkillPixelIndexError(f"duplicate (video_id, source_frame_idx)={key!r}")
        seen_source.add(key)


def _make_id_map(catalog: list[FrameRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, record in enumerate(catalog):
        source_frame_idx = _source_frame(record)
        keyframe_id: int | str
        try:
            keyframe_id = int(record.keyframe_id)
        except ValueError:
            keyframe_id = record.keyframe_id
        rows.append(
            {
                "faiss_row": row,
                "feature_row": row,
                "frame_uid": record.frame_id,
                "video_id": record.video_id,
                "video_filename": _video_filename(record),
                "keyframe_id": keyframe_id,
                "source_frame_idx": source_frame_idx,
                "timestamp_ms": record.timestamp_ms,
                "frame_count": record.frame_count,
                "image_path": record.image_path,
            }
        )
    return rows


@dataclass
class SkillPixelIndex:
    """Loaded visual index with both the production FAISS index and NumPy oracle."""

    artifact_dir: Path
    dataset_root: Path
    catalog: list[FrameRecord]
    embeddings: np.ndarray
    id_map: list[dict[str, Any]]
    dataset_manifest: dict[str, Any]
    index_manifest: dict[str, Any]
    faiss_index: Any

    @property
    def size(self) -> int:
        return len(self.id_map)

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def provider_info(self) -> dict[str, Any]:
        return dict(self.index_manifest.get("embedding", {}))

    def _query(self, query_vector: np.ndarray) -> np.ndarray:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape != (self.dimension,):
            raise ValueError(
                f"query dimension {query.shape[0]} != index dimension {self.dimension}"
            )
        _check_normalized(query.reshape(1, -1), label="query")
        return query

    def _ordered_hits(
        self, rows: np.ndarray, scores: np.ndarray
    ) -> list[tuple[dict[str, Any], float]]:
        pairs = [
            (self.id_map[int(row)], float(score))
            for row, score in zip(rows, scores, strict=True)
            if int(row) >= 0
        ]
        pairs.sort(
            key=lambda item: (
                -item[1],
                str(item[0]["video_id"]),
                int(item[0]["source_frame_idx"]),
                str(item[0]["frame_uid"]),
            )
        )
        return pairs

    def oracle_search(
        self, query_vector: np.ndarray, top_k: int
    ) -> list[tuple[dict[str, Any], float]]:
        if top_k < 1:
            return []
        query = self._query(query_vector)
        scores = self.embeddings @ query
        return self._ordered_hits(np.arange(self.size), scores)[:top_k]

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[dict[str, Any], float]]:
        if top_k < 1:
            return []
        query = self._query(query_vector)
        scores, rows = self.faiss_index.search(
            np.ascontiguousarray(query.reshape(1, -1)), self.size
        )
        return self._ordered_hits(rows[0], scores[0])[:top_k]


def build_skillpixel_index(
    dataset_root: Path,
    artifact_dir: Path,
    provider: EmbeddingProvider,
) -> SkillPixelIndex:
    """Build raw-video-derived embeddings and a persisted exact FAISS FlatIP index."""
    dataset_root = Path(dataset_root).resolve()
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise SkillPixelIndexError(
            f"Artifact directory {artifact_dir} is not empty; use a new versioned path"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - faiss extra is installed in CI
        raise SkillPixelIndexError(
            "FAISS is required for the production baseline; install with: uv sync --extra faiss"
        ) from exc

    stats = validate_raw_dataset(dataset_root)
    catalog = build_catalog(dataset_root)
    if len(catalog) != stats.n_frames:
        raise SkillPixelIndexError(
            f"catalog frame count {len(catalog)} != raw mapping count {stats.n_frames}"
        )
    _validate_catalog(catalog)

    image_paths = [dataset_root / record.image_path for record in catalog]
    try:
        embeddings = np.asarray(provider.embed_images(image_paths), dtype=np.float32)
    except Exception as exc:
        raise SkillPixelIndexError(f"provider {provider.name} failed embedding raw frames") from exc
    expected_shape = (len(catalog), provider.dimension)
    if embeddings.shape != expected_shape:
        raise SkillPixelIndexError(
            f"provider returned {embeddings.shape}; expected {expected_shape}"
        )
    _check_normalized(embeddings, label="embeddings")

    raw_manifest_path = dataset_root / RAW_MANIFEST_NAME
    dataset_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    if not dataset_manifest.get("dataset_hash"):
        raise SkillPixelIndexError("raw dataset manifest has no dataset_hash")

    for record in catalog:
        record.embedding_version = provider.version
    write_catalog(catalog, artifact_dir / CATALOG_NAME)
    np.save(artifact_dir / EMBEDDINGS_NAME, embeddings)
    id_map = _make_id_map(catalog)
    _write_json(artifact_dir / ID_MAP_NAME, id_map)
    _write_json(artifact_dir / "dataset_manifest.json", dataset_manifest)

    index = faiss.IndexFlatIP(provider.dimension)
    index.add(np.ascontiguousarray(embeddings))
    faiss.write_index(index, str(artifact_dir / FAISS_NAME))

    provider_info = provider.info()
    catalog_hash = _sha256_file(artifact_dir / CATALOG_NAME)
    id_map_hash = _sha256_file(artifact_dir / ID_MAP_NAME)
    index_manifest = {
        "format": INDEX_FORMAT,
        "index_version": (
            f"{provider.name}-{provider_info.get('model_revision', provider.version)}-"
            f"{dataset_manifest['dataset_hash'][:12]}"
        ),
        "dataset_root": str(dataset_root),
        "dataset_manifest_hash": dataset_manifest["dataset_hash"],
        "catalog_sha256": catalog_hash,
        "id_map_sha256": id_map_hash,
        "embedding": provider_info,
        "index_provider": "faiss-flat-ip",
        "index_parameters": {"metric": "inner_product", "exact": True},
        "n_frames": len(catalog),
        "dimension": provider.dimension,
        "dataset_id": str(dataset_manifest.get("dataset_id", "skillpixel-local")),
        "dataset_hash": dataset_manifest["dataset_hash"],
        "provider_id": provider.name,
        "model_revision": provider_info.get("model_revision", provider.version),
        "embedding_dimension": provider.dimension,
        "dtype": str(embeddings.dtype),
        "normalized": True,
        "preprocessing": provider_info.get("preprocess_hash", provider_info.get("preprocessing")),
        "index_type": "IndexFlatIP",
        "n_vectors": len(catalog),
        "mapping_sha256": id_map_hash,
        "code_sha": _code_version(),
        "fallback": None,
        "normalization": "l2",
        "raw_video_source": True,
        "btc_artifacts_used": False,
        "provider_execution": "validated-local",
        "code_version": _code_version(),
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    _write_json(artifact_dir / INDEX_MANIFEST_NAME, index_manifest)
    _write_json(
        artifact_dir / PROVIDER_REPORT_NAME,
        {
            "format": "hcmaic-provider-report-v1",
            "provider": provider_info,
            "provider_execution": "validated-local",
            "fallback": None,
            "raw_video_source": True,
            "btc_artifacts_used": False,
            "dataset_hash": dataset_manifest["dataset_hash"],
            "embedding_dimension": provider.dimension,
            "code_sha": index_manifest["code_sha"],
            "evidence_level": provider_info.get("evidence_level", "VALIDATED_LOCAL"),
        },
    )
    loaded = load_skillpixel_index(artifact_dir)
    if loaded.faiss_index.ntotal != len(catalog):
        raise SkillPixelIndexError("FAISS index row count mismatch immediately after build")
    return loaded


def load_skillpixel_index(artifact_dir: Path) -> SkillPixelIndex:
    """Load and fail closed on every catalog/vector/FAISS mapping mismatch."""
    artifact_dir = Path(artifact_dir).resolve()
    required = (
        CATALOG_NAME,
        EMBEDDINGS_NAME,
        ID_MAP_NAME,
        FAISS_NAME,
        "dataset_manifest.json",
        INDEX_MANIFEST_NAME,
        PROVIDER_REPORT_NAME,
    )
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        raise SkillPixelIndexError(f"Missing SkillPixel artifacts: {missing}")

    try:
        import faiss

        catalog = load_catalog(artifact_dir / CATALOG_NAME)
        embeddings = np.load(artifact_dir / EMBEDDINGS_NAME, allow_pickle=False)
        id_map = json.loads((artifact_dir / ID_MAP_NAME).read_text(encoding="utf-8"))
        dataset_manifest = json.loads(
            (artifact_dir / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        index_manifest = json.loads(
            (artifact_dir / INDEX_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        provider_report = json.loads(
            (artifact_dir / PROVIDER_REPORT_NAME).read_text(encoding="utf-8")
        )
        faiss_index = faiss.read_index(str(artifact_dir / FAISS_NAME))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SkillPixelIndexError(f"Cannot load SkillPixel artifacts in {artifact_dir}") from exc

    if index_manifest.get("format") != INDEX_FORMAT:
        raise SkillPixelIndexError(f"unsupported index format {index_manifest.get('format')!r}")
    if not isinstance(id_map, list):
        raise SkillPixelIndexError("id_map.json must be a list of row objects")
    if index_manifest.get("raw_video_source") is not True:
        raise SkillPixelIndexError("index is not marked as raw-video-derived")
    if index_manifest.get("btc_artifacts_used") is not False:
        raise SkillPixelIndexError("BTC artifacts are not allowed in the SkillPixel index")
    if index_manifest.get("provider_execution") != "validated-local":
        raise SkillPixelIndexError("index provider execution is not validated-local")
    if str(index_manifest.get("embedding", {}).get("provider", "")).lower() == "mock":
        raise SkillPixelIndexError("mock provider is not allowed in a production SkillPixel index")
    if provider_report.get("provider_execution") != "validated-local":
        raise SkillPixelIndexError("provider report execution is not validated-local")
    if provider_report.get("dataset_hash") != dataset_manifest.get("dataset_hash"):
        raise SkillPixelIndexError("provider report dataset hash mismatch")
    if provider_report.get("provider", {}).get("provider") != index_manifest.get("provider_id"):
        raise SkillPixelIndexError("provider report/provider id mismatch")
    if embeddings.ndim != 2 or len(catalog) != len(id_map) != embeddings.shape[0]:
        raise SkillPixelIndexError(
            f"row count mismatch catalog={len(catalog)} id_map={len(id_map)} "
            f"embeddings={embeddings.shape}"
        )
    if int(index_manifest.get("n_frames", -1)) != len(catalog):
        raise SkillPixelIndexError("index manifest n_frames mismatch")
    if int(index_manifest.get("dimension", -1)) != embeddings.shape[1]:
        raise SkillPixelIndexError("index manifest dimension mismatch")
    if int(faiss_index.ntotal) != len(catalog) or int(faiss_index.d) != embeddings.shape[1]:
        raise SkillPixelIndexError("FAISS index size/dimension mismatch")
    _check_normalized(np.asarray(embeddings, dtype=np.float32), label="embeddings.npy")

    if str(index_manifest.get("dataset_manifest_hash")) != str(
        dataset_manifest.get("dataset_hash")
    ):
        raise SkillPixelIndexError("dataset manifest hash mismatch")
    if _sha256_file(artifact_dir / CATALOG_NAME) != index_manifest.get("catalog_sha256"):
        raise SkillPixelIndexError("catalog hash mismatch")
    if _sha256_file(artifact_dir / ID_MAP_NAME) != index_manifest.get("id_map_sha256"):
        raise SkillPixelIndexError("id_map hash mismatch")

    _validate_catalog(catalog)
    for row, (record, mapping) in enumerate(zip(catalog, id_map, strict=True)):
        if not isinstance(mapping, dict):
            raise SkillPixelIndexError(f"id_map row {row} is not an object")
        if mapping.get("faiss_row") != row or mapping.get("feature_row") != row:
            raise SkillPixelIndexError(f"id_map row {row} has wrong feature/faiss row")
        if mapping.get("frame_uid") != record.frame_id:
            raise SkillPixelIndexError(f"id_map row {row} frame_uid mismatch")
        source_frame_idx = _source_frame(record)
        if mapping.get("source_frame_idx") != source_frame_idx:
            raise SkillPixelIndexError(
                f"id_map row {row} source_frame_idx mismatch: "
                f"{mapping.get('source_frame_idx')} != {source_frame_idx}"
            )
        if mapping.get("video_id") != record.video_id:
            raise SkillPixelIndexError(f"id_map row {row} video_id mismatch")
        if mapping.get("video_filename") != _video_filename(record):
            raise SkillPixelIndexError(f"id_map row {row} video_filename mismatch")
        if int(mapping.get("timestamp_ms", -1)) != record.timestamp_ms:
            raise SkillPixelIndexError(f"id_map row {row} timestamp_ms mismatch")
        if mapping.get("image_path") != record.image_path:
            raise SkillPixelIndexError(f"id_map row {row} image_path mismatch")

    dataset_root = Path(str(index_manifest.get("dataset_root", "")))
    if not dataset_root:
        raise SkillPixelIndexError("index manifest has no dataset_root")
    return SkillPixelIndex(
        artifact_dir=artifact_dir,
        dataset_root=dataset_root,
        catalog=catalog,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        id_map=id_map,
        dataset_manifest=dataset_manifest,
        index_manifest=index_manifest,
        faiss_index=faiss_index,
    )
