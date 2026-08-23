"""Local retrieval over the authoritative HCMAIC dual visual artifact.

The Kaggle merge contains two intentionally separate ``IndexFlatIP`` spaces:
SigLIP2 (1152-D) and Qwen3-VL-Embedding-2B (2048-D).  This module never
concatenates those vectors.  It validates both index/id-map pairs against the
same ``frame_uid`` catalog, embeds a query in each exact space, and fuses
available channel scores with min-max scaling and a harmonic mean.  The
artifact's historical fusion contract remains preserved as provenance.

The loader is deliberately fail-closed.  A successful Kaggle job is only
engineering evidence; ``quality_status`` remains ``UNVALIDATED`` until
approved qrels/benchmark evidence exists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from hcmaic.contracts.kis import Evidence, KISResult
from hcmaic.embedding.base import EmbeddingProvider, l2_normalize
from hcmaic.retrieval.asr_elasticsearch import (
    ASRElasticsearchAdapter,
    ASRElasticsearchConfig,
)
from hcmaic.retrieval.candidates import ChannelHit, FusedCandidate
from hcmaic.retrieval.channel_contract import ChannelContract, build_channel_evidence
from hcmaic.retrieval.fusion import harmonic_mean_fusion, reciprocal_rank_fusion
from hcmaic.retrieval.image_thumbnail import (
    DEFAULT_IMAGE_THUMBNAIL_QUALITY,
    DEFAULT_IMAGE_THUMBNAIL_WIDTH,
    build_image_thumbnail,
    thumbnail_cache_path,
)
from hcmaic.retrieval.media_resolver import RemoteMediaResolver
from hcmaic.retrieval.ocr_elasticsearch import (
    ElasticsearchOCRChannel,
    ElasticsearchOCRError,
    OCRElasticsearchConfig,
    load_ocr_manifest,
    make_elasticsearch_client,
    validate_ocr_index,
)
from hcmaic.retrieval.rfdetr_object_sidecar import (
    RfdetrObjectSidecarAdapter,
    RfdetrObjectSidecarArtifactError,
    RfdetrObjectSidecarUnavailableError,
)

EXPECTED_CHANNELS = ("siglip2", "qwen")
OBJECT_ENABLED = False
DEFAULT_RUNTIME_FUSION_METHOD = "harmonic"
DEFAULT_DUAL_SIGLIP2_MODEL = "google/siglip2-so400m-patch14-384"
SEARCH_RESULT_CACHE_LIMIT = 32
SEARCH_RESULT_CACHE_TTL_SECONDS = 10.0
LOGGER = logging.getLogger(__name__)
BASE_REQUIRED_FILES = (
    "dual_merge_manifest.json",
    "frame_catalog.jsonl",
    "fusion_contract.json",
)
INDEX_REQUIRED_FILES = {
    "siglip2": (
        "siglip2.index",
        "siglip2_id_map.jsonl",
        "siglip2_index_manifest.json",
    ),
    "qwen": (
        "qwen.index",
        "qwen_id_map.jsonl",
        "qwen_index_manifest.json",
    ),
}
REQUIRED_FILES = (
    *BASE_REQUIRED_FILES,
    *INDEX_REQUIRED_FILES["siglip2"],
    *INDEX_REQUIRED_FILES["qwen"],
)


class DualVisualArtifactError(ValueError):
    """Raised when the downloaded dual-index contract cannot be trusted."""


def _normalize_visual_indexes(visual_indexes: Any | None) -> tuple[str, ...]:
    if visual_indexes is None:
        return EXPECTED_CHANNELS
    if isinstance(visual_indexes, str):
        values = [item.strip() for item in visual_indexes.split(",")]
    else:
        values = [str(item).strip() for item in visual_indexes]
    selected = [value for value in values if value]
    if not selected:
        raise DualVisualArtifactError("visual_indexes must not be empty")
    if len(set(selected)) != len(selected):
        raise DualVisualArtifactError("visual_indexes must not contain duplicates")
    unknown = [value for value in selected if value not in EXPECTED_CHANNELS]
    if unknown:
        raise DualVisualArtifactError(
            f"visual_indexes contains unknown value(s): {sorted(set(unknown))}"
        )
    return tuple(name for name in EXPECTED_CHANNELS if name in selected)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DualVisualArtifactError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DualVisualArtifactError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DualVisualArtifactError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise DualVisualArtifactError(
                        f"JSONL row must be an object at {path}:{line_number}"
                    )
                rows.append(value)
    except OSError as exc:
        raise DualVisualArtifactError(f"cannot read JSONL artifact {path}: {exc}") from exc
    return rows


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    # The Kaggle merge notebook's `_canonical_hash` is JSON-based; preserve
    # that exact serialization rather than inventing a newline convention.
    payload = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _declared_hash(manifest: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = manifest.get(name)
        if isinstance(value, str) and value:
            return value.lower()
    hashes = manifest.get("hashes")
    if isinstance(hashes, Mapping):
        for name in names:
            value = hashes.get(name)
            if isinstance(value, str) and value:
                return value.lower()
    return None


def _validate_status(manifest: Mapping[str, Any], label: str) -> None:
    status = str(manifest.get("status", ""))
    if status and status not in {
        "COMPLETE",
        "DUAL_MERGE_COMPLETE",
        "INPUT_MANIFEST_GATE_GREEN",
        "ENGINEERING_ARTIFACT_COMPLETE",
        "INDEXED",
    }:
        raise DualVisualArtifactError(f"{label} is not complete: status={status!r}")


@dataclass(frozen=True)
class LoadedDualIndex:
    name: str
    index: Any
    id_map: tuple[str, ...]
    manifest: dict[str, Any]
    path: Path
    row_positions: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.row_positions is None:
            object.__setattr__(
                self,
                "row_positions",
                {uid: row_number for row_number, uid in enumerate(self.id_map)},
            )

    @property
    def dimension(self) -> int:
        return int(self.manifest["dimension"])

    @property
    def size(self) -> int:
        return int(self.manifest["ntotal"])

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int,
        allowed_uids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        vector = np.asarray(query, dtype=np.float32)
        if vector.shape != (1, self.dimension):
            raise ValueError(f"{self.name} query shape {vector.shape} != (1, {self.dimension})")
        if not np.isfinite(vector).all():
            raise ValueError(f"{self.name} query contains non-finite values")
        hits: list[tuple[str, float]] = []
        if allowed_uids is not None and hasattr(self.index, "reconstruct"):
            # Stage-2 refinement must score only the candidate pool.  IndexFlatIP
            # exposes reconstruct; this avoids a full-corpus FAISS search while
            # retaining exact inner-product semantics for the bounded pool. The
            # reverse map avoids rescanning all 146k ids for each stage query.
            row_positions = self.row_positions or {}
            candidate_rows = sorted(
                row_positions[uid] for uid in allowed_uids if uid in row_positions
            )
            if candidate_rows:
                candidate_vectors = np.asarray(
                    [self.index.reconstruct(row_number) for row_number in candidate_rows],
                    dtype=np.float32,
                )
                values = candidate_vectors @ vector[0]
                for row_number, score in zip(candidate_rows, values, strict=True):
                    value = float(score)
                    if math.isfinite(value):
                        hits.append((self.id_map[row_number], value))
        else:
            # Compatibility path for injected test indexes without reconstruct.
            # Production FAISS indexes use the bounded branch above.
            search_k = self.size if allowed_uids is not None else min(top_k, self.size)
            scores, rows = self.index.search(vector, search_k)
            if np.asarray(scores).shape != (1, search_k) or np.asarray(rows).shape != (1, search_k):
                raise DualVisualArtifactError(
                    f"{self.name} FAISS search returned unexpected shapes "
                    f"scores={np.asarray(scores).shape} rows={np.asarray(rows).shape}"
                )
            for score, row in zip(scores[0], rows[0], strict=True):
                row_number = int(row)
                if row_number < 0 or row_number >= len(self.id_map):
                    continue
                uid = self.id_map[row_number]
                if allowed_uids is not None and uid not in allowed_uids:
                    continue
                value = float(score)
                if not math.isfinite(value):
                    continue
                hits.append((uid, value))
        # FAISS is deterministic for IndexFlatIP, but sorting here makes ties
        # stable across CPU builds and keeps the identity key visible.
        hits.sort(key=lambda item: (-item[1], item[0]))
        return hits[:top_k]


@dataclass
class DualVisualArtifacts:
    root: Path
    merge_manifest: dict[str, Any]
    fusion_contract: dict[str, Any]
    catalog: list[dict[str, Any]]
    by_uid: dict[str, dict[str, Any]]
    indexes: dict[str, LoadedDualIndex]
    enabled_indexes: tuple[str, ...]

    @property
    def row_count(self) -> int:
        return len(self.catalog)

    @property
    def frame_uid_order(self) -> tuple[str, ...]:
        return tuple(str(row["frame_uid"]) for row in self.catalog)

    @property
    def index_version(self) -> str:
        return str(
            self.merge_manifest.get("merge_id")
            or self.merge_manifest.get("artifact_version")
            or self.merge_manifest.get("code_revision")
            or "dual-index-unversioned"
        )


def _load_faiss_index(path: Path) -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise DualVisualArtifactError(
            "FAISS is required to load the merged IndexFlatIP artifacts"
        ) from exc
    try:
        return faiss.read_index(str(path))
    except Exception as exc:  # pragma: no cover - FAISS error text varies
        raise DualVisualArtifactError(f"cannot load FAISS index {path}: {exc}") from exc


def _resolve_loader(
    index_loader: Mapping[str, Callable[[Path], Any]] | Callable[[Path], Any] | None,
    name: str,
) -> Callable[[Path], Any]:
    if index_loader is None:
        return _load_faiss_index
    if isinstance(index_loader, Mapping):
        loader = index_loader.get(name)
        if loader is None:
            raise DualVisualArtifactError(f"no test/index loader configured for {name}")
        return loader
    return index_loader


def load_dual_visual_artifacts(
    root: Path,
    *,
    enabled_indexes: Any | None = None,
    allow_empty_indexes: bool = False,
    index_loader: Mapping[str, Callable[[Path], Any]] | Callable[[Path], Any] | None = None,
) -> DualVisualArtifacts:
    """Validate and load the Kaggle dual-index output without loading models."""

    artifact_root = Path(root).expanduser().resolve()
    if not artifact_root.is_dir():
        raise DualVisualArtifactError(f"dual-index artifact directory not found: {artifact_root}")
    if allow_empty_indexes and enabled_indexes is not None:
        if isinstance(enabled_indexes, str):
            has_selection = bool(enabled_indexes.strip())
        else:
            has_selection = any(str(item).strip() for item in enabled_indexes)
        selected_indexes = _normalize_visual_indexes(enabled_indexes) if has_selection else ()
    else:
        selected_indexes = _normalize_visual_indexes(enabled_indexes)
    missing = [name for name in BASE_REQUIRED_FILES if not (artifact_root / name).is_file()]
    for name in selected_indexes:
        for filename in INDEX_REQUIRED_FILES[name]:
            if not (artifact_root / filename).is_file():
                missing.append(filename)
    if missing:
        raise DualVisualArtifactError(f"dual-index artifact missing files: {missing}")

    merge_manifest = _read_json(artifact_root / "dual_merge_manifest.json")
    _validate_status(merge_manifest, "dual_merge_manifest.json")
    if str(merge_manifest.get("identity_key")) != "frame_uid":
        raise DualVisualArtifactError("dual merge identity_key must be frame_uid")
    if str(merge_manifest.get("quality_status", "UNVALIDATED")).upper() not in {
        "UNVALIDATED",
        "UNVALIDATED_ON_HCMAIC",
    }:
        raise DualVisualArtifactError(
            "dual merge quality_status is not an approved local serving contract"
        )

    catalog = _read_jsonl(artifact_root / "frame_catalog.jsonl")
    expected_rows = int(merge_manifest.get("row_count", len(catalog)))
    if len(catalog) != expected_rows:
        raise DualVisualArtifactError(
            f"catalog row_count={len(catalog)} != manifest row_count={expected_rows}"
        )
    by_uid: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(catalog):
        required = (
            "feature_row",
            "frame_uid",
            "keyframe_path",
            "source_frame_idx",
            "timestamp_ms",
            "video_id",
        )
        missing_fields = [field for field in required if field not in row]
        if missing_fields:
            raise DualVisualArtifactError(
                f"catalog row {row_number} missing fields: {missing_fields}"
            )
        if int(row["feature_row"]) != row_number:
            raise DualVisualArtifactError(
                f"catalog feature_row {row['feature_row']} != physical row {row_number}"
            )
        uid = str(row["frame_uid"])
        if not uid or uid in by_uid:
            raise DualVisualArtifactError(f"catalog frame_uid is blank or duplicated: {uid!r}")
        if int(row["source_frame_idx"]) < 0 or int(row["timestamp_ms"]) < 0:
            raise DualVisualArtifactError(f"catalog row {row_number} has negative timing")
        path_value = str(row["keyframe_path"])
        if Path(path_value).is_absolute():
            raise DualVisualArtifactError(f"catalog keyframe_path must be relative: {path_value}")
        by_uid[uid] = row

    expected_catalog_hash = _declared_hash(merge_manifest, "catalog_sha256", "frame_catalog_sha256")
    if expected_catalog_hash:
        actual = _sha256(artifact_root / "frame_catalog.jsonl")
        if actual != expected_catalog_hash:
            raise DualVisualArtifactError(
                f"catalog sha256 mismatch: expected {expected_catalog_hash}, got {actual}"
            )
    expected_uid_hash = _declared_hash(merge_manifest, "frame_uid_order_hash")
    if expected_uid_hash:
        actual = _sha256_lines(list(by_uid))
        if actual != expected_uid_hash:
            raise DualVisualArtifactError(
                f"frame_uid order hash mismatch: expected {expected_uid_hash}, got {actual}"
            )

    fusion_contract = _read_json(artifact_root / "fusion_contract.json")
    if str(fusion_contract.get("method", "rrf")).lower() != "rrf":
        raise DualVisualArtifactError("local dual visual runtime currently supports only RRF")
    rank_constant = int(fusion_contract.get("rank_constant", 60))
    if rank_constant < 1:
        raise DualVisualArtifactError("fusion rank_constant must be >= 1")

    indexes: dict[str, LoadedDualIndex] = {}
    reference_ids: tuple[str, ...] | None = None
    for name in selected_indexes:
        id_rows = _read_jsonl(artifact_root / f"{name}_id_map.jsonl")
        if len(id_rows) != expected_rows:
            raise DualVisualArtifactError(
                f"{name} id map row_count={len(id_rows)} != catalog={expected_rows}"
            )
        ids: list[str] = []
        for physical_row, id_row in enumerate(id_rows):
            if int(id_row.get("faiss_row", -1)) != physical_row:
                raise DualVisualArtifactError(f"{name} faiss_row mismatch at {physical_row}")
            feature_row = int(id_row.get("feature_row", -1))
            if feature_row != physical_row:
                raise DualVisualArtifactError(
                    f"{name} feature_row mismatch at {physical_row}: {feature_row}"
                )
            uid = str(id_row.get("frame_uid", ""))
            if uid != str(catalog[feature_row]["frame_uid"]):
                raise DualVisualArtifactError(
                    f"frame_uid mapping mismatch for {name} at row {physical_row}: "
                    f"catalog={catalog[feature_row]['frame_uid']!r}, map={uid!r}"
                )
            ids.append(uid)
        id_tuple = tuple(ids)
        if reference_ids is None:
            reference_ids = id_tuple
        elif id_tuple != reference_ids:
            raise DualVisualArtifactError(f"cross-channel frame_uid mapping mismatch: {name}")

        index_manifest = _read_json(artifact_root / f"{name}_index_manifest.json")
        _validate_status(index_manifest, f"{name}_index_manifest.json")
        if str(index_manifest.get("identity_key", "frame_uid")) != "frame_uid":
            raise DualVisualArtifactError(f"{name} index identity_key must be frame_uid")
        dimension = int(index_manifest.get("dimension", 0))
        ntotal = int(index_manifest.get("ntotal", 0))
        if dimension < 1 or ntotal != expected_rows:
            raise DualVisualArtifactError(
                f"{name} index manifest dimension/ntotal invalid: {dimension}/{ntotal}"
            )
        expected_id_hash = _declared_hash(index_manifest, "id_map_sha256")
        if expected_id_hash:
            actual_id_hash = _sha256(artifact_root / f"{name}_id_map.jsonl")
            if actual_id_hash != expected_id_hash:
                raise DualVisualArtifactError(
                    f"{name} id-map sha256 mismatch: expected {expected_id_hash}, "
                    f"got {actual_id_hash}"
                )
        expected_index_uid_hash = _declared_hash(index_manifest, "frame_uid_order_hash")
        if expected_index_uid_hash:
            actual_index_uid_hash = _sha256_lines(ids)
            if actual_index_uid_hash != expected_index_uid_hash:
                raise DualVisualArtifactError(
                    f"{name} frame_uid order hash mismatch: expected "
                    f"{expected_index_uid_hash}, got {actual_index_uid_hash}"
                )
        index_path = artifact_root / f"{name}.index"
        if index_path.stat().st_size <= 0:
            raise DualVisualArtifactError(f"{name}.index is empty")
        expected_index_hash = _declared_hash(index_manifest, "sha256", "index_sha256")
        if expected_index_hash:
            actual_index_hash = _sha256(index_path)
            if actual_index_hash != expected_index_hash:
                raise DualVisualArtifactError(
                    f"{name}.index sha256 mismatch: expected {expected_index_hash}, "
                    f"got {actual_index_hash}"
                )
        index = _resolve_loader(index_loader, name)(index_path)
        actual_dimension = int(getattr(index, "d", dimension))
        actual_size = int(getattr(index, "ntotal", ntotal))
        if actual_dimension != dimension or actual_size != ntotal:
            raise DualVisualArtifactError(
                f"{name} loaded index shape {actual_dimension}/{actual_size} != manifest "
                f"{dimension}/{ntotal}"
            )
        indexes[name] = LoadedDualIndex(name, index, id_tuple, index_manifest, index_path)

    model_specs = merge_manifest.get("models", {})
    for name in selected_indexes:
        declared = model_specs.get(name, {}) if isinstance(model_specs, Mapping) else {}
        if int(declared.get("dimension", indexes[name].dimension)) != indexes[name].dimension:
            raise DualVisualArtifactError(f"{name} dimension disagrees with merged index")
        declared_revision = str(declared.get("revision") or declared.get("model_revision") or "")
        index_manifest = indexes[name].manifest
        index_revision = str(
            index_manifest.get("revision") or index_manifest.get("model_revision") or ""
        )
        if declared_revision and declared_revision != index_revision:
            raise DualVisualArtifactError(
                f"{name} revision disagrees with merged index: {declared_revision!r} != "
                f"{index_revision!r}"
            )
    return DualVisualArtifacts(
        artifact_root,
        merge_manifest,
        fusion_contract,
        catalog,
        by_uid,
        indexes,
        selected_indexes,
    )


def _provider_label(provider: EmbeddingProvider) -> str:
    return str(getattr(provider, "name", provider.__class__.__name__))


class DualVisualService:
    """Text/image local retrieval over both visual towers and the shared catalog."""

    def __init__(
        self,
        artifacts: DualVisualArtifacts,
        *,
        siglip_provider: EmbeddingProvider | None = None,
        qwen_provider: EmbeddingProvider | None = None,
        visual_indexes: Any | None = None,
        image_root: Path | None = None,
        video_root: Path | None = None,
        media_resolver: RemoteMediaResolver | None = None,
        thumbnail_cache_root: Path | None = None,
        candidate_multiplier: int = 5,
        max_per_video: int | None = None,
        object_adapter: Any | None = None,
        object_status: Mapping[str, Any] | None = None,
        asr_adapter: ASRElasticsearchAdapter | Any | None = None,
        ocr_adapter: Any | None = None,
        ocr_status: Mapping[str, Any] | None = None,
        allow_empty_visual: bool = False,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        if max_per_video is not None and max_per_video < 1:
            raise ValueError("max_per_video must be >= 1 or None")
        self._artifacts = artifacts
        timeline_rows: dict[str, list[dict[str, Any]]] = {}
        for catalog_row in artifacts.catalog:
            video_id = str(catalog_row["video_id"])
            timeline_rows.setdefault(video_id, []).append(catalog_row)
        self._timeline_by_video: dict[str, tuple[dict[str, Any], ...]] = {
            video_id: tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        int(row["timestamp_ms"]),
                        int(row["source_frame_idx"]),
                        str(row["frame_uid"]),
                    ),
                )
            )
            for video_id, rows in timeline_rows.items()
        }
        self._frame_uids_by_video = {
            video_id: frozenset(str(row["frame_uid"]) for row in rows)
            for video_id, rows in self._timeline_by_video.items()
        }
        self._timeline_position_by_frame_uid = {
            str(row["frame_uid"]): position
            for rows in self._timeline_by_video.values()
            for position, row in enumerate(rows)
        }
        self._video_id_order = tuple(sorted(self._timeline_by_video))
        self._video_id_set = frozenset(self._video_id_order)
        self._first_catalog_row_by_video = {
            video_id: rows[0] for video_id, rows in self._timeline_by_video.items()
        }
        selected_visual_indexes = (
            visual_indexes if visual_indexes is not None else artifacts.enabled_indexes
        )
        if allow_empty_visual:
            if visual_indexes is None:
                has_selection = bool(artifacts.enabled_indexes)
            elif isinstance(visual_indexes, str):
                has_selection = bool(visual_indexes.strip())
            else:
                has_selection = any(str(item).strip() for item in visual_indexes)
            self.enabled_indexes = (
                _normalize_visual_indexes(selected_visual_indexes) if has_selection else ()
            )
        else:
            self.enabled_indexes = _normalize_visual_indexes(selected_visual_indexes)
        missing_indexes = [name for name in self.enabled_indexes if name not in artifacts.indexes]
        if missing_indexes:
            raise ValueError(f"enabled visual indexes not loaded: {missing_indexes}")
        self.disabled_indexes = tuple(
            name for name in EXPECTED_CHANNELS if name not in self.enabled_indexes
        )
        self._providers: dict[str, EmbeddingProvider] = {}
        for name in self.enabled_indexes:
            provider = siglip_provider if name == "siglip2" else qwen_provider
            if provider is None:
                raise ValueError(f"{name} provider is required when {name} is enabled")
            if provider.dimension != artifacts.indexes[name].dimension:
                raise ValueError(
                    f"{name} provider dimension {provider.dimension} != index dimension "
                    f"{artifacts.indexes[name].dimension}"
                )
            self._providers[name] = provider
        self._model_specs = dict(
            artifacts.merge_manifest.get("models", {})
            if isinstance(artifacts.merge_manifest.get("models", {}), Mapping)
            else {}
        )
        self._visual_provider_name = "+".join(self.enabled_indexes) or "none"
        self._runtime_fusion_method = DEFAULT_RUNTIME_FUSION_METHOD
        if len(self.enabled_indexes) > 1:
            self._visual_provider_name = f"{self._visual_provider_name}-harmonic"
        self._query_vector_cache: dict[tuple[str, str], np.ndarray] = {}
        self._query_vector_cache_limit = 64
        self._search_result_cache: dict[
            tuple[Any, ...], tuple[float, tuple[KISResult, ...]]
        ] = {}
        self._search_result_cache_limit = SEARCH_RESULT_CACHE_LIMIT
        self._search_result_cache_ttl_seconds = SEARCH_RESULT_CACHE_TTL_SECONDS
        self.image_root = Path(image_root).expanduser().resolve() if image_root else None
        self.video_root = Path(video_root).expanduser().resolve() if video_root else None
        self.media_resolver = media_resolver
        self._thumbnail_cache_root = (
            Path(thumbnail_cache_root).expanduser().resolve()
            if thumbnail_cache_root is not None
            else None
        )
        self.candidate_multiplier = candidate_multiplier
        self.max_per_video = max_per_video
        self.object_adapter = object_adapter
        self._object_status = dict(object_status) if object_status is not None else None
        self.asr_adapter = asr_adapter
        self.ocr_adapter = ocr_adapter
        self._ocr_status = dict(ocr_status) if ocr_status is not None else None

    @property
    def artifacts(self) -> DualVisualArtifacts:
        return self._artifacts

    @property
    def runtime_fusion_contract(self) -> dict[str, Any]:
        artifact_contract = dict(self._artifacts.fusion_contract)
        return {
            **artifact_contract,
            "artifact_method": str(artifact_contract.get("method", "rrf")),
            "method": self._runtime_fusion_method,
            "runtime_contract": "harmonic-mean-minmax",
        }

    def _index_status(self, name: str) -> dict[str, Any]:
        spec = self._model_specs.get(name, {}) if isinstance(self._model_specs, Mapping) else {}
        if name in self._providers:
            provider = self._providers[name]
            return {
                "channel": name,
                "configured": True,
                "ready": True,
                "status": "ready",
                "reason": None,
                "provider": _provider_label(provider),
                "revision": str(getattr(provider, "version", "unknown")),
                "dimension": provider.dimension,
                "execution_status": "ENGINEERING_PROXY",
                "quality_status": "UNVALIDATED",
            }
        return {
            "channel": name,
            "configured": False,
            "ready": False,
            "status": "disabled_by_config",
            "reason": "disabled_by_config",
            "provider": str(spec.get("model_id") or name),
            "revision": str(spec.get("revision") or spec.get("model_revision") or "unknown"),
            "dimension": int(spec.get("dimension", 0)) if spec.get("dimension") else None,
            "execution_status": "DISABLED_BY_CONFIG",
            "quality_status": "UNVALIDATED",
        }

    def channel_status(self) -> dict[str, dict[str, Any]]:
        """Expose the visual-only boundary and optional-channel capability state."""

        if self.enabled_indexes:
            visual_status = {
                "channel": "visual",
                "configured": True,
                "ready": True,
                "status": "ready",
                "reason": None,
                "provider": self._visual_provider_name,
                "revision": self._artifacts.index_version,
                "boundary": "dual_visual_only_late_fusion_boundary",
            }
        else:
            visual_status = {
                "channel": "visual",
                "configured": False,
                "ready": False,
                "status": "disabled_by_config",
                "reason": "visual_disabled_for_asr_only_mode",
                "provider": "none",
                "revision": self._artifacts.index_version,
                "boundary": "explicit_asr_only_runtime",
            }
        statuses: dict[str, dict[str, Any]] = {"visual": visual_status}
        for name in EXPECTED_CHANNELS:
            statuses[name] = self._index_status(name)
        if self.ocr_adapter is not None:
            ocr_status = dict(self.ocr_adapter.channel_contract().to_status_dict())
            ocr_status.setdefault("boundary", "elasticsearch_crop_to_frame_uid_mapping")
            ocr_status.setdefault("index", getattr(self.ocr_adapter, "index_name", None))
            ocr_status.setdefault(
                "include_low_conf", getattr(self.ocr_adapter, "include_low_conf", True)
            )
        elif self._ocr_status is not None:
            ocr_status = dict(self._ocr_status)
        else:
            ocr_status = {
                "channel": "ocr",
                "configured": False,
                "ready": False,
                "status": "unavailable",
                "reason": "channel_not_attached_to_dual_visual_runtime",
                "provider": None,
                "revision": None,
                "boundary": "attach_via_kis_hybrid_orchestrator",
            }
        statuses["ocr"] = ocr_status
        if self.object_adapter is not None:
            object_status = dict(self.object_adapter.channel_contract().to_status_dict())
            object_status.setdefault("boundary", "independent_object_sidecar_late_fusion")
        elif self._object_status is not None:
            object_status = dict(self._object_status)
        else:
            object_status = {
                "channel": "object",
                "configured": False,
                "ready": False,
                "status": "unavailable",
                "reason": "channel_not_attached_to_dual_visual_runtime",
                "provider": None,
                "revision": None,
                "boundary": "attach_via_kis_hybrid_orchestrator",
            }
        statuses["object"] = object_status
        if self.asr_adapter is not None:
            if hasattr(self.asr_adapter, "status_dict"):
                asr_status = dict(self.asr_adapter.status_dict())
            else:
                asr_status = dict(self.asr_adapter.channel_contract().to_status_dict())
            asr_status.setdefault("boundary", "elasticsearch_segment_to_frame_uid_mapping")
        else:
            asr_status = {
                "channel": "asr",
                "configured": False,
                "ready": False,
                "status": "disabled_by_policy",
                "reason": "disabled_until_qrels_ablation_gain",
                "provider": None,
                "revision": None,
                "boundary": "attach_via_kis_hybrid_orchestrator",
            }
        statuses["asr"] = asr_status
        return statuses

    def channel_contracts(self) -> dict[str, dict[str, Any]]:
        visual_manifest = dict(self._artifacts.merge_manifest)
        visual_artifact_hash = (
            str(
                visual_manifest.get("catalog_sha256")
                or visual_manifest.get("frame_catalog_sha256")
                or visual_manifest.get("sha256")
                or ""
            )
            if visual_manifest.get("catalog_sha256")
            or visual_manifest.get("frame_catalog_sha256")
            or visual_manifest.get("sha256")
            else None
        )
        if self.enabled_indexes:
            visual_contract = ChannelContract(
                channel="visual",
                provider=self._visual_provider_name,
                revision=self._artifacts.index_version,
                execution_status="READY_LOCAL_PRECHECK",
                quality_status="UNVALIDATED",
                dataset_manifest_hash=(
                    str(visual_manifest.get("dataset_manifest_hash"))
                    if visual_manifest.get("dataset_manifest_hash")
                    else None
                ),
                artifact_hash=visual_artifact_hash,
                status="ready",
            )
        else:
            visual_contract = ChannelContract(
                channel="visual",
                provider="none",
                revision=self._artifacts.index_version,
                execution_status="DISABLED_BY_CONFIG",
                quality_status="UNVALIDATED",
                dataset_manifest_hash=(
                    str(visual_manifest.get("dataset_manifest_hash"))
                    if visual_manifest.get("dataset_manifest_hash")
                    else None
                ),
                artifact_hash=visual_artifact_hash,
                status="disabled_by_config",
                reason="visual_disabled_for_asr_only_mode",
                configured=False,
                ready=False,
            )
        unavailable = ChannelContract(
            channel="ocr",
            provider="unavailable",
            revision="unavailable",
            execution_status="ENGINEERING_PROXY",
            quality_status="UNVALIDATED",
            status="unavailable",
            configured=False,
            ready=False,
            reason="channel_not_attached_to_dual_visual_runtime",
        ).to_status_dict()
        if self.ocr_adapter is not None:
            ocr_contract = self.ocr_adapter.channel_contract().to_status_dict()
            ocr_contract.setdefault("index", getattr(self.ocr_adapter, "index_name", None))
        elif self._ocr_status is not None:
            ocr_contract = dict(self._ocr_status)
        else:
            ocr_contract = unavailable
        if self.object_adapter is not None:
            object_contract = self.object_adapter.channel_contract().to_status_dict()
        elif self._object_status is not None:
            object_contract = dict(self._object_status)
        else:
            object_contract = dict(unavailable)
            object_contract["channel"] = "object"
        if self.asr_adapter is not None:
            asr_contract = self.asr_adapter.channel_contract().to_status_dict()
        else:
            asr_contract = ChannelContract(
                channel="asr",
                provider="unavailable",
                revision="unavailable",
                execution_status="DISABLED_BY_POLICY",
                quality_status="UNVALIDATED",
                status="disabled_by_policy",
                configured=False,
                ready=False,
                reason="disabled_until_qrels_ablation_gain",
            ).to_status_dict()
        return {
            "visual": visual_contract.to_status_dict(),
            "ocr": ocr_contract,
            "object": object_contract,
            "asr": asr_contract,
        }

    def health(self) -> dict[str, Any]:
        video_ids = self.video_ids()
        channel_status = self.channel_status()
        return {
            "status": "ok",
            "execution_status": "READY_LOCAL_PRECHECK",
            "quality_status": "UNVALIDATED",
            "identity_key": "frame_uid",
            "configured_indexes": list(self.enabled_indexes),
            "enabled_indexes": list(self.enabled_indexes),
            "disabled_indexes": list(self.disabled_indexes),
            "row_count": self._artifacts.row_count,
            # Compatibility aliases consumed by the bundled static UI.  The
            # canonical fields above remain the source of truth.
            "index_size": self._artifacts.row_count,
            "n_videos": len(video_ids),
            "embedding_provider": self._visual_provider_name,
            "indexes": {
                name: {
                    "dimension": loaded.dimension,
                    "ntotal": loaded.size,
                    "index_type": loaded.manifest.get("index_type", "unknown"),
                }
                for name, loaded in self._artifacts.indexes.items()
            },
            "providers": {
                name: {
                    "name": str(status.get("provider") or name),
                    "version": str(status.get("revision") or "unknown"),
                    "dimension": status.get("dimension"),
                    "status": status.get("status"),
                    "reason": status.get("reason"),
                    "configured": bool(status.get("configured")),
                    "ready": bool(status.get("ready")),
                }
                for name, status in channel_status.items()
                if name in {"visual", *EXPECTED_CHANNELS}
            },
            "fusion": self.runtime_fusion_contract,
            "channel_status": channel_status,
            "channel_contracts": self.channel_contracts(),
            "media": (
                self.media_resolver.summary()
                if self.media_resolver is not None
                else {"status": "LOCAL_ROOTS_ONLY"}
            ),
        }

    def get_frame(self, frame_uid: str) -> dict[str, Any]:
        try:
            return self._artifacts.by_uid[frame_uid]
        except KeyError as exc:
            raise KeyError(frame_uid) from exc

    def video_ids(self) -> list[str]:
        return list(self._video_id_order)

    def has_video_id(self, video_id: str) -> bool:
        """Return indexed video membership without rescanning the catalog."""

        return str(video_id) in self._video_id_set

    def video_media_status(self, video_id: str) -> dict[str, Any]:
        """Return a non-sensitive availability contract for one source video.

        This is deliberately metadata-only: it never downloads a video.  The
        API/UI can therefore hide or disable a video control when the
        authoritative source-video inventory has not been attached yet.
        """

        base = {
            "video_id": video_id,
            "backend": None,
            "bytes": None,
            "range_capable": False,
            "provenance_status": None,
            "sha256_status": None,
            "source_path": None,
            "member_path": None,
            "canonical_source_path": None,
            "dataset_id": None,
            "revision": None,
            "media_info_id": None,
            "normalized_media_info_id": None,
            "range_probe_status": None,
            "range_probe_attempts": None,
            "source_manifest_id": None,
            "source_fingerprint": None,
            "source_fingerprint_semantics": None,
            "remote_content_fingerprint": None,
            "join_method": None,
        }
        if not self.has_video_id(video_id):
            return {
                **base,
                "available": False,
                "status": "UNKNOWN_VIDEO_ID",
                "stream_available": False,
                "stream_status": "VIDEO_NOT_FOUND",
            }

        if self.video_root is not None:
            row = self._first_catalog_row_by_video[video_id]
            filename = str(row.get("video_filename") or f"{video_id}.mp4")
            root = self.video_root.resolve()
            path = (root / filename).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                return {
                    **base,
                    "available": False,
                    "status": "UNAVAILABLE_UNSAFE_VIDEO_PATH",
                    "stream_available": False,
                    "stream_status": "UNAVAILABLE_UNSAFE_VIDEO_PATH",
                }
            if path.is_file():
                return {
                    **base,
                    "available": True,
                    "status": "AVAILABLE_LOCAL",
                    "stream_available": True,
                    "stream_status": "AVAILABLE_LOCAL",
                    "backend": "local",
                    "bytes": path.stat().st_size,
                    "provenance_status": "LOCAL_ONLY",
                }

        if self.media_resolver is not None:
            if video_id in self.media_resolver.manifest.videos:
                spec = self.media_resolver.manifest.videos[video_id]
                stream_available = self.media_resolver.can_stream_video(video_id)
                return {
                    **base,
                    "available": True,
                    "status": "AVAILABLE_REMOTE",
                    "stream_available": stream_available,
                    "backend": spec.backend,
                    "bytes": spec.bytes,
                    "range_capable": spec.range_capable,
                    "provenance_status": spec.provenance_status,
                    "sha256_status": spec.sha256_status,
                    "source_path": spec.source_path,
                    "member_path": spec.member_path,
                    "canonical_source_path": spec.canonical_source_path,
                    "dataset_id": spec.dataset_id,
                    "revision": spec.revision,
                    "media_info_id": spec.media_info_id,
                    "normalized_media_info_id": spec.normalized_media_info_id,
                    "range_probe_status": spec.range_probe_status,
                    "range_probe_attempts": spec.range_probe_attempts,
                    "source_manifest_id": spec.source_manifest_id,
                    "source_fingerprint": spec.source_fingerprint,
                    "source_fingerprint_semantics": spec.source_fingerprint_semantics,
                    "remote_content_fingerprint": spec.remote_content_fingerprint,
                    "join_method": spec.join_method,
                    "stream_status": (
                        "AVAILABLE_REMOTE_RANGE"
                        if stream_available
                        else "REMOTE_MEDIA_RANGE_UNSUPPORTED"
                    ),
                    "stream_reason": (
                        None
                        if stream_available
                        else "remote backend has no trusted byte-range contract"
                    ),
                }
            return {
                **base,
                "available": False,
                "status": "UNAVAILABLE_NO_VIDEO_MANIFEST_ENTRY",
                "stream_available": False,
                "stream_status": "UNAVAILABLE_NO_VIDEO_MANIFEST_ENTRY",
            }

        if self.video_root is not None:
            return {
                **base,
                "available": False,
                "status": "UNAVAILABLE_LOCAL_FILE_MISSING",
                "stream_available": False,
                "stream_status": "UNAVAILABLE_LOCAL_FILE_MISSING",
            }
        return {
            **base,
            "available": False,
            "status": "UNAVAILABLE_NO_VIDEO_ROOT_OR_MANIFEST",
            "stream_available": False,
            "stream_status": "UNAVAILABLE_NO_VIDEO_ROOT_OR_MANIFEST",
        }

    def frame_image_status(self, frame_uid: str) -> dict[str, Any]:
        """Return metadata-only keyframe availability for the local UI.

        This method deliberately does not resolve or download remote media.  It
        gives the UI a fail-closed reason before it tries the image endpoint,
        while keeping the actual file/hash verification in ``frame_image_path``.
        """

        row = self.get_frame(frame_uid)
        local_missing = False
        if self.image_root is not None:
            root = self.image_root.resolve()
            path = (root / str(row["keyframe_path"])).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                return {
                    "image_available": False,
                    "image_status": "UNAVAILABLE_UNSAFE_IMAGE_PATH",
                    "image_reason": "catalog keyframe_path escapes configured image_root",
                }
            if path.is_file():
                return {
                    "image_available": True,
                    "image_status": "AVAILABLE_LOCAL",
                    "image_reason": None,
                }
            local_missing = True

        if self.media_resolver is not None:
            if frame_uid in self.media_resolver.manifest.frames:
                media_status = self.media_resolver.media_status("frame", frame_uid)
                if not media_status["available"]:
                    return {
                        "image_available": False,
                        "image_status": media_status["status"],
                        "image_reason": media_status["reason"],
                    }
                return {
                    "image_available": True,
                    "image_status": "AVAILABLE_REMOTE",
                    "image_reason": (
                        "frame is declared in the allowlisted remote media manifest "
                        "and will resolve on demand"
                    ),
                }
            if local_missing:
                return {
                    "image_available": False,
                    "image_status": "UNAVAILABLE_LOCAL_FILE_MISSING",
                    "image_reason": (
                        "local keyframe file is missing and no remote manifest entry exists"
                    ),
                }
            return {
                "image_available": False,
                "image_status": "UNAVAILABLE_NO_IMAGE_MANIFEST_ENTRY",
                "image_reason": "frame_uid is not present in the configured remote media manifest",
            }

        if local_missing:
            return {
                "image_available": False,
                "image_status": "UNAVAILABLE_LOCAL_FILE_MISSING",
                "image_reason": "local keyframe file is missing",
            }
        return {
            "image_available": False,
            "image_status": "UNAVAILABLE_NO_IMAGE_ROOT_OR_MANIFEST",
            "image_reason": "image_root and remote media manifest are not configured",
        }

    def timeline(self, video_id: str) -> list[dict[str, Any]]:
        rows = self._timeline_by_video.get(str(video_id))
        if rows is None:
            raise KeyError(video_id)
        return list(rows)

    def first_keyframe(self, video_id: str) -> dict[str, Any]:
        """Return the deterministic first canonical catalog keyframe.

        ``_timeline_by_video`` is constructed in canonical order by timestamp,
        source frame index, then frame UID.  Returning only the first row keeps
        the direct-video UI path bounded; it must not materialize an entire
        video's timeline just to display one keyframe.
        """

        rows = self._timeline_by_video.get(str(video_id))
        if not rows:
            raise KeyError(video_id)
        return dict(rows[0])

    def timeline_window(
        self,
        video_id: str,
        frame_uid: str,
        window: int = 5,
    ) -> list[dict[str, Any]]:
        """Return a bounded timeline slice without copying/scanning the full video."""

        rows = self._timeline_by_video.get(str(video_id))
        if rows is None:
            raise KeyError(video_id)
        position = self._timeline_position_by_frame_uid.get(str(frame_uid))
        if position is None or str(rows[position]["video_id"]) != str(video_id):
            raise KeyError(frame_uid)
        radius = max(0, int(window))
        return list(rows[max(0, position - radius) : position + radius + 1])

    def resolve_frame_reference(
        self,
        video_id: str,
        *,
        frame_uid: str | None = None,
        source_frame_idx: int | None = None,
        timestamp_ms: int | None = None,
    ) -> dict[str, Any]:
        """Resolve one review reference without downloading media or writing events."""

        selectors = [frame_uid is not None, source_frame_idx is not None, timestamp_ms is not None]
        if sum(selectors) != 1:
            raise ValueError("provide exactly one of frame_uid, source_frame_idx, timestamp_ms")
        rows = self.timeline(video_id)
        selected_time = timestamp_ms
        if frame_uid is not None:
            if str(frame_uid).split(":", 1)[0] != video_id:
                raise ValueError("frame_uid belongs to another video")
            matches = [row for row in rows if str(row["frame_uid"]) == frame_uid]
            mapping_mode = "exact_frame_uid"
            mapping_method = "CATALOG_FRAME_UID"
            mapping_status = "RESOLVED_EXACT"
        elif source_frame_idx is not None:
            matches = [row for row in rows if int(row["source_frame_idx"]) == source_frame_idx]
            mapping_mode = "exact_source_frame_idx"
            mapping_method = "CATALOG_SOURCE_FRAME_IDX"
            mapping_status = "RESOLVED_EXACT"
        else:
            assert timestamp_ms is not None
            matches = [
                min(
                    rows,
                    key=lambda row: (
                        abs(int(row["timestamp_ms"]) - timestamp_ms),
                        int(row["timestamp_ms"]),
                        int(row["source_frame_idx"]),
                        str(row["frame_uid"]),
                    ),
                )
            ]
            mapping_mode = "nearest_catalog_timestamp"
            mapping_method = "CATALOG_NEAREST_TIMESTAMP"
            mapping_status = "RESOLVED_CATALOG_TIMESTAMP"
        if not matches:
            raise KeyError(frame_uid if frame_uid is not None else source_frame_idx)
        row = matches[0]
        resolved_time = int(row["timestamp_ms"])
        selected_time = resolved_time if selected_time is None else selected_time
        return {
            "video_id": video_id,
            "frame_uid": str(row["frame_uid"]),
            "resolved_frame_uid": str(row["frame_uid"]),
            "source_frame_idx": int(row["source_frame_idx"]),
            "selected_time_ms": int(selected_time),
            "resolved_timestamp_ms": resolved_time,
            "timestamp_ms": resolved_time,
            "delta_ms": abs(int(selected_time) - resolved_time),
            "mapping_mode": mapping_mode,
            "mapping_method": mapping_method,
            "mapping_status": mapping_status,
            "shot_id": row.get("shot_id"),
            "keyframe_path": row.get("keyframe_path"),
        }

    def frame_image_path(self, frame_uid: str) -> Path:
        row = self.get_frame(frame_uid)
        local_error: FileNotFoundError | None = None
        if self.image_root is not None:
            root = self.image_root.resolve()
            path = (root / str(row["keyframe_path"])).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise PermissionError(
                    "catalog keyframe_path escapes configured image_root"
                ) from exc
            if path.is_file():
                return path
            local_error = FileNotFoundError(path)
        if self.media_resolver is not None:
            return self.media_resolver.resolve_frame(frame_uid)
        if local_error is not None:
            raise local_error
        raise FileNotFoundError("image_root and remote media manifest are not configured")

    def frame_thumbnail_path(
        self,
        frame_uid: str,
        *,
        width: int = DEFAULT_IMAGE_THUMBNAIL_WIDTH,
        quality: int = DEFAULT_IMAGE_THUMBNAIL_QUALITY,
    ) -> Path:
        """Return a cached presentation thumbnail without changing the source image."""
        if self._thumbnail_cache_root is not None:
            cache_root = self._thumbnail_cache_root
        elif self.media_resolver is not None:
            cache_root = self.media_resolver.cache_root / "thumbnails"
        else:
            # Local-only tests/dev servers still need a writable cache, but it
            # must stay outside the immutable artifact/image root.
            cache_root = Path(tempfile.gettempdir()) / "hcmaic-thumbnails" / str(os.getpid())

        # Remote frame manifests are pinned to a dataset revision and member
        # path. Check that immutable derivative first so a browser refresh does
        # not download the full source JPEG merely to discover that the
        # thumbnail already exists.
        if self.media_resolver is not None and self.image_root is None:
            remote_cache_key = self.media_resolver.cache_identity("frame", frame_uid)
            cached_thumbnail = thumbnail_cache_path(
                cache_root,
                cache_key=remote_cache_key,
                width=width,
                quality=quality,
            )
            if cached_thumbnail.is_file():
                return cached_thumbnail

            # Preserve thumbnails created by the previous metadata-based key
            # when the full source is already warm in the media cache.
            cached_source = self.media_resolver.cached_media_path("frame", frame_uid)
            if cached_source.is_file():
                source_stat = cached_source.stat()
                legacy_key = (
                    f"{frame_uid}|{cached_source.resolve()}|"
                    f"{source_stat.st_size}|{source_stat.st_mtime_ns}"
                )
                legacy_thumbnail = thumbnail_cache_path(
                    cache_root,
                    cache_key=legacy_key,
                    width=width,
                    quality=quality,
                )
                if legacy_thumbnail.is_file():
                    return legacy_thumbnail

            source = self.frame_image_path(frame_uid)
            cache_key = remote_cache_key
        else:
            source = self.frame_image_path(frame_uid)
            source_stat = source.stat()
            cache_key = (
                f"{frame_uid}|{source.resolve()}|"
                f"{source_stat.st_size}|{source_stat.st_mtime_ns}"
            )
        return build_image_thumbnail(
            source,
            cache_root,
            cache_key=cache_key,
            width=width,
            quality=quality,
        )

    def video_path(self, video_id: str) -> Path:
        local_error: FileNotFoundError | None = None
        if self.video_root is not None:
            row = self._first_catalog_row_by_video.get(str(video_id))
            if row is None:
                raise KeyError(video_id)
            filename = str(row.get("video_filename") or f"{video_id}.mp4")
            root = self.video_root.resolve()
            path = (root / filename).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise PermissionError("video filename escapes configured video_root") from exc
            if path.is_file():
                return path
            local_error = FileNotFoundError(path)
        if self.media_resolver is not None:
            return self.media_resolver.resolve_video(video_id)
        if local_error is not None:
            raise local_error
        raise FileNotFoundError("video_root and remote media manifest are not configured")

    def local_video_path(self, video_id: str) -> Path | None:
        """Return an existing local video without invoking remote full-download."""

        if self.video_root is None:
            return None
        row = self._first_catalog_row_by_video.get(str(video_id))
        if row is None:
            raise KeyError(video_id)
        root = self.video_root.resolve()
        filename = str(row.get("video_filename") or f"{video_id}.mp4")
        path = (root / filename).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError("video filename escapes configured video_root") from exc
        return path if path.is_file() else None

    def remote_video_url(self, video_id: str) -> str | None:
        """Return the immutable manifest URL for one remote video, if present."""

        if self.media_resolver is None:
            return None
        return self.media_resolver.media_url("video", video_id)

    def stream_video_range(self, video_id: str, range_header: str | None):
        """Fetch a remote byte range; this path never calls ``resolve_video``."""

        if self.media_resolver is None:
            raise FileNotFoundError("remote media manifest is not configured")
        return self.media_resolver.stream_video_range(video_id, range_header)

    def _allowed_uids(self, video_ids: list[str] | None) -> set[str] | None:
        if not video_ids:
            return None
        wanted = set(video_ids)
        return {
            frame_uid
            for video_id in wanted
            for frame_uid in self._frame_uids_by_video.get(video_id, ())
        }

    def resolve_visual_indexes(self, visual_indexes: Any | None = None) -> tuple[str, ...]:
        """Resolve a request-level subset without loading new providers.

        The process-level configuration controls which indexes/providers are
        loaded. A request may select a non-empty subset of those loaded
        indexes, but it can never enable an index disabled at startup.
        """

        selected = (
            self.enabled_indexes
            if visual_indexes is None
            else _normalize_visual_indexes(visual_indexes)
        )
        unavailable = [name for name in selected if name not in self.enabled_indexes]
        if unavailable:
            raise ValueError(
                f"visual_indexes are not loaded in this process: {sorted(set(unavailable))}"
            )
        return selected

    def _search_vectors(
        self,
        query_vectors: Mapping[str, np.ndarray],
        *,
        query_id: str,
        task: str,
        top_k: int,
        video_ids: list[str] | None = None,
        visual_indexes: Any | None = None,
        candidate_frame_uids: set[str] | None = None,
        object_query: str | None = None,
        ocr_query: str | None = None,
        asr_query: str | None = None,
        asr_mode: str | None = None,
        allow_empty_visual: bool = False,
        fusion_method: str = "harmonic",
        allow_large_top_k: bool = False,
        image_path: Path | None = None,
    ) -> list[KISResult]:
        if top_k < 1 or (top_k > 500 and not allow_large_top_k):
            limit = "unbounded for internal candidate expansion" if allow_large_top_k else "500"
            raise ValueError(f"top_k must be in [1, {limit}]")
        fusion_method = str(fusion_method).strip().lower()
        if fusion_method not in {"rrf", "harmonic"}:
            raise ValueError("fusion_method must be one of: rrf, harmonic")
        active_indexes = () if allow_empty_visual else self.resolve_visual_indexes(visual_indexes)
        for name in active_indexes:
            if name not in query_vectors:
                continue
            vector = np.asarray(query_vectors[name], dtype=np.float32)
            if vector.shape != (1, self._artifacts.indexes[name].dimension):
                raise ValueError(f"{name} provider returned query shape {vector.shape}")
        cache_key = None if image_path is not None else self._search_result_cache_key(
            query_vectors,
            task=task,
            top_k=top_k,
            video_ids=video_ids,
            visual_indexes=active_indexes,
            candidate_frame_uids=candidate_frame_uids,
            object_query=object_query,
            ocr_query=ocr_query,
            asr_query=asr_query,
            asr_mode=asr_mode,
            allow_empty_visual=allow_empty_visual,
            fusion_method=fusion_method,
            allow_large_top_k=allow_large_top_k,
        )
        cached_results = self._cached_search_results(cache_key, query_id)
        if cached_results is not None:
            return cached_results
        allowed = self._allowed_uids(video_ids)
        if candidate_frame_uids is not None:
            candidate = {str(uid) for uid in candidate_frame_uids}
            allowed = candidate if allowed is None else allowed & candidate
        pool_size = min(self._artifacts.row_count, top_k * self.candidate_multiplier)
        channels: dict[str, list[ChannelHit]] = {}
        search_specs: list[tuple[str, str, np.ndarray]] = [
            (name, name, np.asarray(query_vectors[name], dtype=np.float32))
            for name in active_indexes
            if name in query_vectors
        ]
        if image_path is not None:
            if "siglip2" not in self._providers or "siglip2" not in self._artifacts.indexes:
                raise RuntimeError("SigLIP2 image provider/index is unavailable")
            search_specs.append(("image", "siglip2", self._query_image_vector(self._providers["siglip2"], image_path)))
        for channel_name, name, vector in search_specs:
            provider = self._providers[name]
            provider_info = dict(provider.info())
            if vector.shape != (1, self._artifacts.indexes[name].dimension):
                raise ValueError(f"{channel_name} provider returned query shape {vector.shape}")
            hits = self._artifacts.indexes[name].search(
                l2_normalize(vector), top_k=pool_size, allowed_uids=allowed
            )
            channel_hits: list[ChannelHit] = []
            for rank, (uid, score) in enumerate(hits, start=1):
                row = self._artifacts.by_uid[uid]
                channel_hits.append(
                    ChannelHit(
                        entity_id=uid,
                        video_id=str(row["video_id"]),
                        timestamp_ms=int(row["timestamp_ms"]),
                        modality=channel_name,
                        score=float(score),
                        rank=rank,
                        provider=_provider_label(provider),
                        frame_uid=uid,
                        video_filename=str(row.get("video_filename") or f"{row['video_id']}.mp4"),
                        source_frame_idx=int(row["source_frame_idx"]),
                        evidence=build_channel_evidence(
                            channel=channel_name,
                            provider=_provider_label(provider),
                            revision=str(getattr(provider, "version", "unknown")),
                            execution_status="ENGINEERING_PROXY",
                            quality_status="UNVALIDATED",
                            dataset_manifest_hash=(
                                str(self._artifacts.merge_manifest.get("dataset_manifest_hash"))
                                if self._artifacts.merge_manifest.get("dataset_manifest_hash")
                                else None
                            ),
                            artifact_hash=(
                                str(
                                    self._artifacts.merge_manifest.get("catalog_sha256")
                                    or self._artifacts.merge_manifest.get("frame_catalog_sha256")
                                    or self._artifacts.merge_manifest.get("sha256")
                                    or ""
                                )
                                if self._artifacts.merge_manifest.get("catalog_sha256")
                                or self._artifacts.merge_manifest.get("frame_catalog_sha256")
                                or self._artifacts.merge_manifest.get("sha256")
                                else None
                            ),
                            frame_uid=uid,
                            video_id=str(row["video_id"]),
                            video_filename=str(
                                row.get("video_filename") or f"{row['video_id']}.mp4"
                            ),
                            source_frame_idx=int(row["source_frame_idx"]),
                            timestamp_ms=int(row["timestamp_ms"]),
                            score=float(score),
                            rank=rank,
                            channel_specific={
                                "identity_key": "frame_uid",
                                "index_name": name,
                                "dimension": provider.dimension,
                                "model": provider_info.get("model_name") or provider_info.get("provider"),
                                "revision": provider_info.get("revision") or provider_info.get("version"),
                                "preprocessing": provider_info.get(
                                    "preprocessing", "provider-configured"
                                ),
                                "score_metric": "normalized_inner_product",
                            },
                            raw_provenance={
                                "merge_manifest": self._artifacts.index_version,
                                "index_manifest": dict(self._artifacts.indexes[name].manifest),
                            },
                        ),
                    )
                )
            channels.setdefault(channel_name, []).extend(channel_hits)

        unavailable_channels: dict[str, str] = {}
        ocr_hits: list[ChannelHit] = []
        if ocr_query is not None:
            if not isinstance(ocr_query, str) or not ocr_query.strip():
                raise ValueError("ocr_query must not be blank")
            if self.ocr_adapter is None:
                ocr_status = self.channel_status().get("ocr", {})
                unavailable_channels["ocr"] = str(
                    ocr_status.get("reason") or "ocr_channel_unavailable"
                )
            else:
                ocr_kwargs: dict[str, Any] = {"top_k": pool_size}
                if video_ids:
                    ocr_kwargs["video_ids"] = video_ids
                raw_ocr_hits = self.ocr_adapter.search(ocr_query, **ocr_kwargs)
                seen_ocr_uids: set[str] = set()
                for raw_hit in raw_ocr_hits:
                    identity = str(raw_hit.frame_uid or raw_hit.entity_id)
                    if raw_hit.frame_uid is not None and raw_hit.entity_id != identity:
                        raise ValueError(
                            "OCR channel entity_id/frame_uid mismatch: "
                            f"{raw_hit.entity_id!r} != {raw_hit.frame_uid!r}"
                        )
                    if identity not in self._artifacts.by_uid:
                        continue
                    if allowed is not None and identity not in allowed:
                        continue
                    if identity in seen_ocr_uids:
                        raise ValueError(f"duplicate frame_uid in OCR channel: {identity}")
                    seen_ocr_uids.add(identity)
                    row = self._artifacts.by_uid[identity]
                    score = float(raw_hit.score)
                    if not math.isfinite(score):
                        raise ValueError("OCR channel returned a non-finite score")
                    ocr_hits.append(
                        ChannelHit(
                            entity_id=identity,
                            video_id=str(row["video_id"]),
                            timestamp_ms=int(row["timestamp_ms"]),
                            modality="ocr",
                            score=score,
                            rank=int(raw_hit.rank),
                            provider=str(raw_hit.provider),
                            evidence_text=raw_hit.evidence_text,
                            frame_uid=identity,
                            video_filename=str(
                                row.get("video_filename") or f"{row['video_id']}.mp4"
                            ),
                            source_frame_idx=int(row["source_frame_idx"]),
                            evidence=dict(raw_hit.evidence),
                        )
                    )
                if ocr_hits:
                    channels["ocr"] = ocr_hits

        object_hits: list[ChannelHit] = []
        if object_query is not None:
            if not isinstance(object_query, str) or not object_query.strip():
                raise ValueError("object_query must not be blank")
            if self.object_adapter is None:
                object_status = self.channel_status().get("object", {})
                unavailable_channels["object"] = str(
                    object_status.get("reason") or "object_channel_unavailable"
                )
            else:
                raw_object_hits = self.object_adapter.search(object_query, top_k=pool_size)
                seen_object_uids: set[str] = set()
                for raw_hit in raw_object_hits:
                    identity = str(raw_hit.frame_uid or raw_hit.entity_id)
                    if raw_hit.frame_uid is not None and raw_hit.entity_id != identity:
                        raise ValueError(
                            "object channel entity_id/frame_uid mismatch: "
                            f"{raw_hit.entity_id!r} != {raw_hit.frame_uid!r}"
                        )
                    if identity not in self._artifacts.by_uid:
                        continue
                    if allowed is not None and identity not in allowed:
                        continue
                    if identity in seen_object_uids:
                        raise ValueError(f"duplicate frame_uid in object channel: {identity}")
                    seen_object_uids.add(identity)
                    row = self._artifacts.by_uid[identity]
                    score = float(raw_hit.score)
                    if not math.isfinite(score):
                        raise ValueError("object channel returned a non-finite score")
                    object_hits.append(
                        ChannelHit(
                            entity_id=identity,
                            video_id=str(row["video_id"]),
                            timestamp_ms=int(row["timestamp_ms"]),
                            modality="object",
                            score=score,
                            rank=len(object_hits) + 1,
                            provider=str(raw_hit.provider),
                            evidence_text=raw_hit.evidence_text,
                            frame_uid=identity,
                            video_filename=str(
                                row.get("video_filename") or f"{row['video_id']}.mp4"
                            ),
                            source_frame_idx=int(row["source_frame_idx"]),
                            evidence=dict(raw_hit.evidence),
                        )
                    )
                if object_hits:
                    channels["object"] = object_hits

        asr_hits: list[ChannelHit] = []
        if asr_query is not None:
            if not isinstance(asr_query, str) or not asr_query.strip():
                raise ValueError("asr_query must not be blank")
            if self.asr_adapter is None:
                asr_status = self.channel_status().get("asr", {})
                unavailable_channels["asr"] = str(
                    asr_status.get("reason") or "asr_channel_unavailable"
                )
            else:
                asr_kwargs: dict[str, Any] = {
                    # The Elasticsearch ASR adapter has a bounded top-k
                    # contract; visual/object/OCR candidates can still grow
                    # during adaptive bundle expansion without breaking ASR.
                    "top_k": min(pool_size, 500),
                    "allowed_frame_uids": allowed,
                    "video_ids": video_ids,
                }
                if asr_mode is not None:
                    asr_kwargs["mode"] = asr_mode
                raw_asr_hits = self.asr_adapter.search(asr_query, **asr_kwargs)
                seen_asr_uids: set[str] = set()
                for raw_hit in raw_asr_hits:
                    identity = str(raw_hit.frame_uid or raw_hit.entity_id)
                    if raw_hit.frame_uid is not None and raw_hit.entity_id != identity:
                        raise ValueError(
                            "ASR channel entity_id/frame_uid mismatch: "
                            f"{raw_hit.entity_id!r} != {raw_hit.frame_uid!r}"
                        )
                    if identity not in self._artifacts.by_uid:
                        continue
                    if allowed is not None and identity not in allowed:
                        continue
                    if identity in seen_asr_uids:
                        raise ValueError(f"duplicate frame_uid in ASR channel: {identity}")
                    seen_asr_uids.add(identity)
                    row = self._artifacts.by_uid[identity]
                    score = float(raw_hit.score)
                    if not math.isfinite(score):
                        raise ValueError("ASR channel returned a non-finite score")
                    asr_hits.append(
                        ChannelHit(
                            entity_id=identity,
                            video_id=str(row["video_id"]),
                            timestamp_ms=int(row["timestamp_ms"]),
                            modality="asr",
                            score=score,
                            rank=int(raw_hit.rank),
                            provider=str(raw_hit.provider),
                            evidence_text=raw_hit.evidence_text,
                            frame_uid=identity,
                            video_filename=str(
                                row.get("video_filename") or f"{row['video_id']}.mp4"
                            ),
                            source_frame_idx=int(row["source_frame_idx"]),
                            evidence=dict(raw_hit.evidence),
                        )
                    )
                if asr_hits:
                    channels["asr"] = asr_hits

        active_channels = [name for name, hits in channels.items() if hits]
        if len(active_channels) <= 1:
            only = active_channels[0] if active_channels else None
            fused: list[FusedCandidate] = []
            for hit in channels.get(only, []) if only is not None else []:
                fused.append(
                    FusedCandidate(
                        entity_id=hit.entity_id,
                        video_id=hit.video_id,
                        timestamp_ms=hit.timestamp_ms,
                        final_score=float(hit.score),
                        signal_scores={only: float(hit.score)},
                        normalized_scores={only: float(hit.score)},
                        evidence_texts={only: hit.evidence_text} if hit.evidence_text else {},
                        contributing_providers=[hit.provider],
                        explanation={"method": "direct"},
                        frame_uid=hit.frame_uid,
                        video_filename=hit.video_filename,
                        source_frame_idx=hit.source_frame_idx,
                        evidence={
                            only: {
                                "frame_uid": hit.frame_uid,
                                "video_filename": hit.video_filename,
                                "source_frame_idx": hit.source_frame_idx,
                                "timestamp_ms": hit.timestamp_ms,
                                "provider": hit.provider,
                                "score": hit.score,
                                "rank": hit.rank,
                                "text": hit.evidence_text,
                                "metadata": dict(hit.evidence),
                            }
                        },
                    )
                )
        elif fusion_method == "harmonic":
            fused = harmonic_mean_fusion(channels, top_k=pool_size)
        else:
            fused = reciprocal_rank_fusion(
                channels,
                rank_constant=int(self._artifacts.fusion_contract.get("rank_constant", 60)),
                top_k=pool_size,
            )
        if self.max_per_video is not None:
            selected: list[Any] = []
            counts: dict[str, int] = {}
            deferred: list[Any] = []
            for candidate in fused:
                if counts.get(candidate.video_id, 0) < self.max_per_video:
                    selected.append(candidate)
                    counts[candidate.video_id] = counts.get(candidate.video_id, 0) + 1
                else:
                    deferred.append(candidate)
            for candidate in deferred:
                if len(selected) >= top_k:
                    break
                selected.append(candidate)
            fused = selected
        results: list[KISResult] = []
        for rank, candidate in enumerate(fused[:top_k], start=1):
            row = self._artifacts.by_uid[candidate.entity_id]
            evidence: list[Evidence] = []
            for channel, payload in sorted(candidate.evidence.items()):
                evidence.append(
                    Evidence(
                        channel=channel,
                        frame_uid=candidate.entity_id,
                        video_id=str(row["video_id"]),
                        video_filename=str(row.get("video_filename") or f"{row['video_id']}.mp4"),
                        source_frame_idx=int(row["source_frame_idx"]),
                        timestamp_ms=int(row["timestamp_ms"]),
                        score=float(payload.get("score", 0.0)),
                        rank=int(payload.get("rank", rank)),
                        evidence_level="REAL_PROVIDER",
                        metadata={
                            "provider": payload.get("provider"),
                            "identity_key": "frame_uid",
                            **dict(payload.get("metadata") or {}),
                        },
                    )
                )
            results.append(
                KISResult(
                    query_id=query_id,
                    task=task,
                    rank=rank,
                    frame_uid=candidate.entity_id,
                    video_id=str(row["video_id"]),
                    video_filename=str(row.get("video_filename") or f"{row['video_id']}.mp4"),
                    source_frame_idx=int(row["source_frame_idx"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    channel_scores=dict(candidate.signal_scores),
                    fused_score=float(candidate.final_score),
                    evidence=tuple(evidence),
                    executed_channels=tuple(active_channels),
                    unavailable_channels=unavailable_channels,
                    evidence_level="REAL_PROVIDER",
                    quality_status="UNVALIDATED_ON_HCMAIC",
                )
            )
        self._store_search_results(cache_key, results)
        return results

    @staticmethod
    def _query_vector(provider: EmbeddingProvider, method: str, value: Any) -> np.ndarray:
        result = getattr(provider, method)([value])
        array = np.asarray(result, dtype=np.float32)
        if array.shape != (1, provider.dimension):
            raise ValueError(f"provider returned query shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("provider returned non-finite query embedding")
        return array

    def _cached_query_vector(
        self,
        provider_name: str,
        provider: EmbeddingProvider,
        value: str,
    ) -> np.ndarray:
        key = (provider_name, value)
        cached = self._query_vector_cache.pop(key, None)
        if cached is not None:
            self._query_vector_cache[key] = cached
            return cached

        vector = self._query_vector(provider, "embed_texts", value)
        self._query_vector_cache[key] = vector
        if len(self._query_vector_cache) > self._query_vector_cache_limit:
            self._query_vector_cache.pop(next(iter(self._query_vector_cache)))
        return vector

    @staticmethod
    def _cache_value(value: Any) -> str | None:
        if value is None:
            return None
        return str(value).strip()

    def _search_result_cache_key(
        self,
        query_vectors: Mapping[str, np.ndarray],
        *,
        task: str,
        top_k: int,
        video_ids: list[str] | None,
        visual_indexes: Any | None,
        candidate_frame_uids: set[str] | None,
        object_query: str | None,
        ocr_query: str | None,
        asr_query: str | None,
        asr_mode: str | None,
        allow_empty_visual: bool,
        fusion_method: str,
        allow_large_top_k: bool,
    ) -> tuple[Any, ...] | None:
        # Candidate-pool searches are intentionally uncached: their set is
        # request-specific and can be large.
        if candidate_frame_uids is not None:
            return None
        if visual_indexes is None:
            visual_key = tuple(self.enabled_indexes)
        elif isinstance(visual_indexes, str):
            visual_key = tuple(
                item.strip() for item in visual_indexes.split(",") if item.strip()
            )
        else:
            visual_key = tuple(
                str(item).strip() for item in visual_indexes if str(item).strip()
            )
        vector_key = tuple(
            (
                str(name),
                hashlib.sha256(
                    np.asarray(vector, dtype=np.float32).tobytes()
                ).hexdigest(),
            )
            for name, vector in sorted(query_vectors.items())
        )
        return (
            "search-result-v1",
            str(task).upper(),
            int(top_k),
            tuple(sorted({str(video_id) for video_id in (video_ids or [])})),
            visual_key,
            vector_key,
            self._cache_value(object_query),
            self._cache_value(ocr_query),
            self._cache_value(asr_query),
            self._cache_value(asr_mode),
            bool(allow_empty_visual),
            str(fusion_method).strip().lower(),
            bool(allow_large_top_k),
        )

    def _cached_search_results(
        self,
        key: tuple[Any, ...] | None,
        query_id: str,
    ) -> list[KISResult] | None:
        if key is None:
            return None
        entry = self._search_result_cache.pop(key, None)
        if entry is None:
            return None
        created_at, cached = entry
        if time.monotonic() - created_at > self._search_result_cache_ttl_seconds:
            return None
        self._search_result_cache[key] = entry
        return [replace(result, query_id=query_id) for result in cached]

    def _store_search_results(
        self,
        key: tuple[Any, ...] | None,
        results: list[KISResult],
    ) -> None:
        if key is None:
            return
        self._search_result_cache[key] = (time.monotonic(), tuple(results))
        while len(self._search_result_cache) > self._search_result_cache_limit:
            self._search_result_cache.pop(next(iter(self._search_result_cache)))

    def search_text(
        self,
        query_id: str,
        text: str,
        *,
        top_k: int = 100,
        video_ids: list[str] | None = None,
        visual_indexes: Any | None = None,
        candidate_frame_uids: set[str] | None = None,
        object_query: str | None = None,
        ocr_query: str | None = None,
        asr_query: str | None = None,
        asr_mode: str | None = None,
        fusion_method: str = "harmonic",
        allow_large_top_k: bool = False,
        image_path: Path | None = None,
    ) -> list[KISResult]:
        if not text.strip():
            raise ValueError("text must not be blank")
        active_indexes = self.resolve_visual_indexes(visual_indexes)
        return self._search_vectors(
            {
                name: self._cached_query_vector(name, provider, text)
                for name, provider in self._providers.items()
                if name in active_indexes
            },
            query_id=query_id,
            task="TKIS",
            top_k=top_k,
            video_ids=video_ids,
            visual_indexes=active_indexes,
            candidate_frame_uids=candidate_frame_uids,
            object_query=object_query,
            ocr_query=ocr_query,
            asr_query=asr_query,
            asr_mode=asr_mode,
            fusion_method=fusion_method,
            allow_large_top_k=allow_large_top_k,
            image_path=image_path,
        )

    def search_object(
        self,
        query_id: str,
        object_query: str,
        *,
        top_k: int = 100,
        video_ids: list[str] | None = None,
        candidate_frame_uids: set[str] | None = None,
        asr_query: str | None = None,
        asr_mode: str | None = None,
        fusion_method: str = "harmonic",
        allow_large_top_k: bool = False,
        visual_indexes: Any | None = None,
        image_path: Path | None = None,
    ) -> list[KISResult]:
        """Run the object sidecar as an independent, non-visual channel."""

        if self.object_adapter is None:
            raise RuntimeError("object adapter unavailable")
        active_visual_indexes = visual_indexes if visual_indexes is not None else (
            ("siglip2",) if image_path is not None else ()
        )
        return self._search_vectors(
            {},
            query_id=query_id,
            task="TKIS",
            top_k=top_k,
            video_ids=video_ids,
            visual_indexes=active_visual_indexes,
            candidate_frame_uids=candidate_frame_uids,
            object_query=object_query,
            asr_query=asr_query,
            asr_mode=asr_mode,
            allow_empty_visual=not active_visual_indexes,
            fusion_method=fusion_method,
            allow_large_top_k=allow_large_top_k,
            image_path=image_path,
        )

    def object_aliases(self) -> dict[str, Any]:
        """Expose the lightweight query-time object alias catalog."""

        if self.object_adapter is None:
            return {
                "status": "unavailable",
                "reason": "object adapter unavailable",
                "version": None,
                "aliases": [],
                "labels": [],
            }
        provider = getattr(self.object_adapter, "object_aliases", None)
        if not callable(provider):
            return {
                "status": "unavailable",
                "reason": "object adapter does not expose alias catalog",
                "version": None,
                "aliases": [],
                "labels": [],
            }
        return dict(provider())

    def search_ocr(
        self,
        query_id: str,
        ocr_query: str,
        *,
        top_k: int = 100,
        video_ids: list[str] | None = None,
        candidate_frame_uids: set[str] | None = None,
        object_query: str | None = None,
        asr_query: str | None = None,
        asr_mode: str | None = None,
        fusion_method: str = "harmonic",
        allow_large_top_k: bool = False,
        visual_indexes: Any | None = None,
        image_path: Path | None = None,
    ) -> list[KISResult]:
        """Run crop-level OCR independently, optionally with other channels."""

        if self.ocr_adapter is None:
            raise RuntimeError("OCR adapter unavailable")
        if not ocr_query.strip():
            raise ValueError("ocr_query must not be blank")
        active_visual_indexes = visual_indexes if visual_indexes is not None else (
            ("siglip2",) if image_path is not None else ()
        )
        return self._search_vectors(
            {},
            query_id=query_id,
            task="TKIS",
            top_k=top_k,
            video_ids=video_ids,
            visual_indexes=active_visual_indexes,
            candidate_frame_uids=candidate_frame_uids,
            object_query=object_query,
            ocr_query=ocr_query,
            asr_query=asr_query,
            asr_mode=asr_mode,
            allow_empty_visual=not active_visual_indexes,
            fusion_method=fusion_method,
            allow_large_top_k=allow_large_top_k,
            image_path=image_path,
        )

    def search_asr(
        self,
        query_id: str,
        asr_query: str,
        *,
        top_k: int = 100,
        video_ids: list[str] | None = None,
        candidate_frame_uids: set[str] | None = None,
        object_query: str | None = None,
        asr_mode: str | None = None,
        fusion_method: str = "harmonic",
        allow_large_top_k: bool = False,
        visual_indexes: Any | None = None,
        image_path: Path | None = None,
    ) -> list[KISResult]:
        """Run ASR independently, optionally refining an existing frame pool."""

        if self.asr_adapter is None:
            raise RuntimeError("ASR adapter unavailable")
        if not asr_query.strip():
            raise ValueError("asr_query must not be blank")
        active_visual_indexes = visual_indexes if visual_indexes is not None else (
            ("siglip2",) if image_path is not None else ()
        )
        return self._search_vectors(
            {},
            query_id=query_id,
            task="TKIS",
            top_k=top_k,
            video_ids=video_ids,
            visual_indexes=active_visual_indexes,
            candidate_frame_uids=candidate_frame_uids,
            object_query=object_query,
            asr_query=asr_query,
            asr_mode=asr_mode,
            allow_empty_visual=not active_visual_indexes,
            fusion_method=fusion_method,
            allow_large_top_k=allow_large_top_k,
            image_path=image_path,
        )

    def search_image(
        self,
        query_id: str,
        path: Path,
        *,
        top_k: int = 100,
        video_ids: list[str] | None = None,
        visual_indexes: Any | None = None,
        candidate_frame_uids: set[str] | None = None,
        allow_large_top_k: bool = False,
    ) -> list[KISResult]:
        query_path = Path(path).expanduser().resolve()
        if not query_path.is_file():
            raise FileNotFoundError(query_path)
        active_indexes = self.resolve_visual_indexes(visual_indexes or ("siglip2",))
        return self._search_vectors(
            {},
            query_id=query_id,
            task="VKIS",
            top_k=top_k,
            video_ids=video_ids,
            visual_indexes=active_indexes,
            candidate_frame_uids=candidate_frame_uids,
            allow_empty_visual=True,
            allow_large_top_k=allow_large_top_k,
            image_path=query_path,
        )

    @staticmethod
    def _query_image_vector(provider: EmbeddingProvider, path: Path) -> np.ndarray:
        """Encode one image with the provider's single-image query contract."""

        result = provider.embed_query_image(path)
        array = np.asarray(result, dtype=np.float32)
        if array.shape != (1, provider.dimension):
            raise ValueError(f"provider returned query shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("provider returned non-finite query embedding")
        return array


def emit_local_runtime_manifest(
    artifacts: DualVisualArtifacts,
    output_path: Path,
    *,
    include_file_hashes: bool = True,
) -> dict[str, Any]:
    """Write a portable local-serving manifest next to the downloaded artifact."""

    root = artifacts.root
    files = list(REQUIRED_FILES)
    payload: dict[str, Any] = {
        "status": "LOCAL_RUNTIME_MANIFEST",
        "execution_status": "READY_LOCAL_PRECHECK",
        "quality_status": "UNVALIDATED",
        "identity_key": "frame_uid",
        "row_count": artifacts.row_count,
        "frame_uid_order_hash": _sha256_lines(list(artifacts.frame_uid_order)),
        "artifact_root": str(root),
        "files": files,
        "models": artifacts.merge_manifest.get("models", {}),
        "fusion_contract": dict(artifacts.fusion_contract),
    }
    if include_file_hashes:
        payload["sha256"] = {name: _sha256(root / name) for name in files}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_dual_visual_service(
    root: Path,
    *,
    visual_indexes: Any | None = None,
    asr_only: bool = False,
    index_loader: Mapping[str, Callable[[Path], Any]] | Callable[[Path], Any] | None = None,
    provider_loader: Mapping[str, Callable[..., EmbeddingProvider]] | None = None,
    image_root: Path | None = None,
    video_root: Path | None = None,
    siglip_device: str | None = None,
    qwen_device: str | None = None,
    siglip_batch_size: int = 8,
    qwen_batch_size: int = 2,
    local_files_only: bool = True,
    siglip_model_path: Path | None = None,
    model_path: Path | None = None,
    media_manifest: Path | None = None,
    media_cache_root: Path | None = None,
    media_http_hosts: set[str] | None = None,
    object_sidecar: Path | None = None,
    object_enabled: bool = OBJECT_ENABLED,
    allow_engineering_proxy: bool = False,
    asr_config: ASRElasticsearchConfig | None = None,
    asr_client: Any | None = None,
    ocr_es_config: OCRElasticsearchConfig | None = None,
    ocr_es_url: str | None = None,
    ocr_es_index: str | None = None,
    ocr_es_manifest: Path | None = None,
    ocr_es_include_low_conf: bool = True,
    ocr_es_api_key_env: str = "ELASTIC_API_KEY",
    ocr_es_username_env: str | None = None,
    ocr_es_password_env: str | None = None,
    ocr_es_client: Any | None = None,
) -> DualVisualService:
    """Load the selected visual towers and the validated local indexes.

    Model construction is intentionally last and local-cache-only by default:
    a broken artifact fails before expensive model initialization, and a
    missing checkpoint cannot silently fall back to another model.
    """

    selected_indexes = () if asr_only else _normalize_visual_indexes(visual_indexes)
    artifacts = load_dual_visual_artifacts(
        root,
        enabled_indexes=selected_indexes,
        allow_empty_indexes=asr_only,
        index_loader=index_loader,
    )
    models = artifacts.merge_manifest.get("models", {})
    siglip_spec = models.get("siglip2", {}) if isinstance(models, Mapping) else {}
    qwen_spec = models.get("qwen", {}) if isinstance(models, Mapping) else {}
    providers: dict[str, EmbeddingProvider] = {}

    def _make_provider(name: str) -> EmbeddingProvider:
        if provider_loader is not None:
            factory = provider_loader.get(name)
            if factory is None:
                raise DualVisualArtifactError(f"no provider factory configured for {name}")
            spec = siglip_spec if name == "siglip2" else qwen_spec
            model_name = str(
                spec.get("model_id")
                or (
                    DEFAULT_DUAL_SIGLIP2_MODEL
                    if name == "siglip2"
                    else "Qwen/Qwen3-VL-Embedding-2B"
                )
            )
            revision = str(spec.get("revision") or spec.get("model_revision") or "")
            if not revision:
                raise DualVisualArtifactError(f"{name} revision must be pinned before serving")
            kwargs: dict[str, Any] = {
                "model_name": model_name,
                "device": siglip_device if name == "siglip2" else qwen_device,
                "local_files_only": local_files_only,
                "revision": revision,
                "batch_size": siglip_batch_size if name == "siglip2" else qwen_batch_size,
            }
            if name == "siglip2" and siglip_model_path is not None:
                kwargs["model_path"] = siglip_model_path
            if name == "qwen" and model_path is not None:
                kwargs["model_path"] = model_path
            return factory(**kwargs)
        if name == "siglip2":
            from hcmaic.embedding.siglip2 import RealSiglip2EmbeddingProvider

            model_name = str(siglip_spec.get("model_id", DEFAULT_DUAL_SIGLIP2_MODEL))
            revision = str(siglip_spec.get("revision") or siglip_spec.get("model_revision", ""))
            if not revision:
                raise DualVisualArtifactError("siglip2 revision must be pinned before serving")
            source = str(siglip_model_path or model_name)
            return RealSiglip2EmbeddingProvider(
                model_name=source,
                revision=revision,
                device=siglip_device,
                local_files_only=local_files_only,
                batch_size=siglip_batch_size,
            )
        from hcmaic.embedding.qwen3_vl import Qwen3VLEmbeddingProvider

        model_name = str(qwen_spec.get("model_id", "Qwen/Qwen3-VL-Embedding-2B"))
        revision = str(qwen_spec.get("revision") or qwen_spec.get("model_revision", ""))
        if not revision:
            raise DualVisualArtifactError("qwen revision must be pinned before serving")
        return Qwen3VLEmbeddingProvider(
            model_name=model_name,
            revision=revision,
            device=qwen_device,
            local_files_only=local_files_only,
            model_path=model_path,
            batch_size=qwen_batch_size,
        )

    for name in selected_indexes:
        providers[name] = _make_provider(name)
    media_resolver = None
    if media_manifest is not None:
        cache_root = media_cache_root or (root.parent / "media-cache")
        media_resolver = RemoteMediaResolver(
            media_manifest,
            cache_root,
            allowed_http_hosts=media_http_hosts,
        )
    object_adapter: RfdetrObjectSidecarAdapter | None = None
    object_status: dict[str, Any] | None = None
    if object_enabled:
        object_status = {
            "channel": "object",
            "configured": True,
            "ready": False,
            "status": "unavailable",
            "reason": "object_sidecar_not_configured",
            "provider": None,
            "revision": None,
            "execution_status": "UNAVAILABLE",
            "quality_status": "UNVALIDATED",
            "boundary": "independent_object_sidecar_late_fusion",
        }
        if object_sidecar is not None:
            try:
                object_adapter = RfdetrObjectSidecarAdapter.from_artifact(
                    object_sidecar,
                    allow_engineering_proxy=allow_engineering_proxy,
                    expected_frame_uids=set(artifacts.by_uid),
                )
                object_status = object_adapter.channel_contract().to_status_dict()
                object_status["boundary"] = "independent_object_sidecar_late_fusion"
            except (
                RfdetrObjectSidecarArtifactError,
                RfdetrObjectSidecarUnavailableError,
            ) as exc:
                LOGGER.warning("RF-DETR object sidecar unavailable for dual runtime: %s", exc)
                object_status["reason"] = "object_sidecar_load_failed"
                object_status["error"] = str(exc)
    configured_asr = asr_config
    if configured_asr is None and ASRElasticsearchConfig.env_is_declared():
        configured_asr = ASRElasticsearchConfig.from_env()
    asr_adapter = (
        ASRElasticsearchAdapter.from_config(configured_asr, client=asr_client)
        if configured_asr is not None
        else None
    )
    configured_ocr = ocr_es_config
    explicit_ocr = any(
        value is not None
        for value in (
            ocr_es_url,
            ocr_es_index,
            ocr_es_manifest,
        )
    )
    if configured_ocr is None and explicit_ocr:
        configured_ocr = OCRElasticsearchConfig(
            url=ocr_es_url,
            index=ocr_es_index or "hcmaic_ocr_v1",
            manifest_path=ocr_es_manifest,
            include_low_conf=ocr_es_include_low_conf,
            enabled=True,
            api_key_env=ocr_es_api_key_env,
            username_env=ocr_es_username_env,
            password_env=ocr_es_password_env,
        )
    elif configured_ocr is None and OCRElasticsearchConfig.env_is_declared():
        configured_ocr = OCRElasticsearchConfig.from_env()

    ocr_adapter: ElasticsearchOCRChannel | None = None
    ocr_status: dict[str, Any] | None = None
    if configured_ocr is not None:
        ocr_status = {
            "channel": "ocr",
            "configured": bool(configured_ocr.enabled),
            "ready": False,
            "status": "disabled_by_config" if not configured_ocr.enabled else "unavailable",
            "reason": (
                "ocr_es_disabled_by_config"
                if not configured_ocr.enabled
                else "ocr_es_config_incomplete"
            ),
            "provider": "deepsolo-parseq-elasticsearch",
            "revision": "unknown",
            "execution_status": (
                "DISABLED_BY_CONFIG" if not configured_ocr.enabled else "UNAVAILABLE"
            ),
            "quality_status": "UNVALIDATED",
            "index": configured_ocr.index,
            "include_low_conf": configured_ocr.include_low_conf,
            "boundary": "elasticsearch_crop_to_frame_uid_mapping",
        }
        if configured_ocr.enabled and not allow_engineering_proxy:
            ocr_status.update(
                {
                    "status": "disabled_by_policy",
                    "reason": "engineering_proxy_disabled_by_policy",
                    "execution_status": "DISABLED_BY_POLICY",
                }
            )
        elif configured_ocr.enabled:
            if not configured_ocr.url or configured_ocr.manifest_path is None:
                pass
            else:
                try:
                    manifest, manifest_hash = load_ocr_manifest(
                        configured_ocr.manifest_path,
                        allow_snapshot=True,
                    )
                    if manifest.get("format") != "hcmaic-dstext-parseq-ocr-merged-v1":
                        raise ElasticsearchOCRError(
                            "OCR manifest is not a merged DeepSolo/PARSeq artifact"
                        )
                    client = ocr_es_client or make_elasticsearch_client(
                        configured_ocr.url,
                        api_key_env=configured_ocr.api_key_env,
                        username_env=configured_ocr.username_env,
                        password_env=configured_ocr.password_env,
                        allow_anonymous_local=configured_ocr.allow_anonymous_local,
                    )
                    validate_ocr_index(client, configured_ocr.index)
                    ocr_adapter = ElasticsearchOCRChannel(
                        client,
                        configured_ocr.index,
                        manifest,
                        manifest_sha256=manifest_hash,
                        include_low_conf=configured_ocr.include_low_conf,
                    )
                    ocr_status = ocr_adapter.channel_contract().to_status_dict()
                    ocr_status.update(
                        {
                            "index": configured_ocr.index,
                            "include_low_conf": configured_ocr.include_low_conf,
                            "boundary": "elasticsearch_crop_to_frame_uid_mapping",
                        }
                    )
                except (ElasticsearchOCRError, OSError, ValueError):
                    LOGGER.warning(
                        "OCR Elasticsearch channel unavailable for dual runtime at index %s",
                        configured_ocr.index,
                    )
    return DualVisualService(
        artifacts,
        siglip_provider=providers.get("siglip2"),
        qwen_provider=providers.get("qwen"),
        visual_indexes=selected_indexes,
        image_root=image_root,
        video_root=video_root,
        media_resolver=media_resolver,
        object_adapter=object_adapter,
        object_status=object_status,
        asr_adapter=asr_adapter,
        ocr_adapter=ocr_adapter,
        ocr_status=ocr_status,
        allow_empty_visual=asr_only,
    )
