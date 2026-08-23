"""Elasticsearch adapter for the merged crop-level OCR artifact."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.channel_contract import ChannelContract
from hcmaic.retrieval.ocr_text import fold_ocr_text, normalize_ocr_nfc

OCR_INDEX_FORMAT = "hcmaic-ocr-v1"
OCR_ES_DEFAULT_INDEX = "hcmaic_ocr_v1"
OCR_MERGED_MANIFEST_FORMAT = "hcmaic-dstext-parseq-ocr-merged-v1"
OCR_ES_SNAPSHOT_SCHEMA = "hcmaic-ocr-elasticsearch-snapshot-v1"
_INDEX_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,254}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ElasticsearchOCRError(RuntimeError):
    """Raised when the optional Elasticsearch OCR adapter cannot operate."""


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off", "disabled"}


@dataclass(frozen=True)
class OCRElasticsearchConfig:
    """Explicit runtime configuration for the merged OCR Elasticsearch index."""

    url: str | None = None
    index: str = OCR_ES_DEFAULT_INDEX
    manifest_path: Path | None = None
    include_low_conf: bool = True
    enabled: bool = False
    api_key_env: str = "ELASTIC_API_KEY"
    username_env: str | None = None
    password_env: str | None = None
    allow_anonymous_local: bool = False

    def __post_init__(self) -> None:
        _validate_index_name(str(self.index))
        if self.url is not None:
            parsed = urlparse(str(self.url))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("OCR Elasticsearch URL must use http:// or https://")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    "OCR Elasticsearch URL must not contain credentials or query secrets"
                )
        if self.manifest_path is not None:
            object.__setattr__(self, "manifest_path", Path(self.manifest_path).expanduser())

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OCRElasticsearchConfig:
        values = dict(os.environ if env is None else env)
        declared = any(
            values.get(name)
            for name in (
                "HCMAIC_OCR_ES_URL",
                "HCMAIC_OCR_ES_INDEX",
                "HCMAIC_OCR_ES_MANIFEST",
            )
        )
        return cls(
            url=values.get("HCMAIC_OCR_ES_URL") or values.get("OCR_ES_URL"),
            index=values.get("HCMAIC_OCR_ES_INDEX", OCR_ES_DEFAULT_INDEX),
            manifest_path=(
                Path(values["HCMAIC_OCR_ES_MANIFEST"])
                if values.get("HCMAIC_OCR_ES_MANIFEST")
                else None
            ),
            include_low_conf=_env_bool(values, "HCMAIC_OCR_ES_INCLUDE_LOW_CONF", True),
            enabled=_env_bool(values, "HCMAIC_OCR_ES_ENABLED", declared),
            api_key_env=values.get("HCMAIC_OCR_ES_API_KEY_ENV", "ELASTIC_API_KEY"),
            username_env=values.get("HCMAIC_OCR_ES_USERNAME_ENV") or None,
            password_env=values.get("HCMAIC_OCR_ES_PASSWORD_ENV") or None,
            allow_anonymous_local=_env_bool(
                values, "HCMAIC_OCR_ES_ALLOW_ANONYMOUS_LOCAL", False
            ),
        )

    @staticmethod
    def env_is_declared(env: Mapping[str, str] | None = None) -> bool:
        values = os.environ if env is None else env
        return any(
            name in values
            for name in (
                "HCMAIC_OCR_ES_URL",
                "HCMAIC_OCR_ES_INDEX",
                "HCMAIC_OCR_ES_MANIFEST",
                "HCMAIC_OCR_ES_ENABLED",
                "OCR_ES_URL",
            )
        )


def _validate_index_name(index_name: str) -> str:
    if not _INDEX_NAME_RE.fullmatch(index_name) or ".." in index_name:
        raise ValueError(f"invalid Elasticsearch index name: {index_name!r}")
    return index_name


def build_ocr_index_mapping() -> dict[str, Any]:
    """Return the versioned mapping for exact, folded and n-gram OCR search."""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.max_ngram_diff": 13,
            "analysis": {
                "filter": {"hcmaic_char_ngrams": {"type": "ngram", "min_gram": 2, "max_gram": 15}},
                "analyzer": {
                    "hcmaic_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"],
                    },
                    "hcmaic_folded": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    },
                    "hcmaic_ngram": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding", "hcmaic_char_ngrams"],
                    },
                },
            },
        },
        "mappings": {
            "dynamic": False,
            "properties": {
                "crop_uid": {"type": "keyword"},
                "line_uid": {"type": "keyword"},
                "frame_uid": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "video_filename": {"type": "keyword", "ignore_above": 512},
                "shot_id": {"type": "keyword", "ignore_above": 512},
                "source_shard_id": {"type": "keyword"},
                "source_manifest_sha256": {"type": "keyword"},
                "quality_status": {"type": "keyword"},
                "execution_status": {"type": "keyword"},
                "confidence_status": {"type": "keyword"},
                "detector_model": {"type": "keyword", "ignore_above": 512},
                "detector_revision": {"type": "keyword", "ignore_above": 512},
                "recognizer_model": {"type": "keyword", "ignore_above": 512},
                "recognizer_revision": {"type": "keyword", "ignore_above": 512},
                "candidate_policy": {"type": "keyword", "ignore_above": 512},
                "source_frame_idx": {"type": "long"},
                "timestamp_ms": {"type": "long"},
                "line_index": {"type": "long"},
                "detector_line_index": {"type": "long"},
                "word_index": {"type": "long"},
                "det_score": {"type": "float"},
                "rec_score": {"type": "float"},
                "text_raw": {"type": "keyword", "index": False, "ignore_above": 4096},
                "text_nfc": {
                    "type": "text",
                    "analyzer": "hcmaic_text",
                    "fields": {
                        "folded": {"type": "text", "analyzer": "hcmaic_folded"},
                        "ngram": {"type": "text", "analyzer": "hcmaic_ngram"},
                    },
                },
                "text_folded": {
                    "type": "text",
                    "analyzer": "hcmaic_folded",
                    "fields": {"ngram": {"type": "text", "analyzer": "hcmaic_ngram"}},
                },
            },
        },
    }


def _int_or_none(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None


def _json_field(row: dict[str, Any], field: str) -> object:
    value = row.get(field)
    if value is not None:
        return value
    encoded = row.get(f"{field}_json")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        return json.loads(encoded)
    except json.JSONDecodeError:
        return None


def _expand_parquet_projection(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_row_json")
    if isinstance(raw, str) and raw:
        try:
            expanded = json.loads(raw)
        except json.JSONDecodeError:
            expanded = {}
        if isinstance(expanded, dict):
            result = {str(key): value for key, value in expanded.items()}
        else:
            result = {}
    else:
        result = {}
    for key, value in row.items():
        if not key.endswith("_json") and key != "raw_row_json":
            result.setdefault(key, value)
    for field in ("bbox", "polygon", "detector_polygon", "ocr_candidates"):
        if field not in result:
            result[field] = _json_field(row, field)
    return result


def iter_ocr_rows(artifact_dir: Path, *, batch_size: int = 50_000) -> Iterator[dict[str, Any]]:
    """Stream merged rows, preferring the lossless JSONL representation."""
    from hcmaic.ingestion.ocr_merge import iter_artifact_rows

    root = Path(artifact_dir)
    jsonl = root / "ocr_lines.jsonl"
    parquet = root / "ocr_lines.parquet"
    if jsonl.is_file():
        yield from iter_artifact_rows(jsonl, batch_size=batch_size)
        return
    if parquet.is_file():
        for row in iter_artifact_rows(parquet, batch_size=batch_size):
            yield _expand_parquet_projection(row)
        return
    raise ElasticsearchOCRError(f"merged OCR line artifact is unavailable in {root}")


def ocr_row_to_document(
    row: dict[str, Any], manifest: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Convert one canonical crop row to an ES bulk operation payload."""
    crop_uid = str(row.get("crop_uid", "")).strip()
    frame_uid = str(row.get("frame_uid", "")).strip()
    video_id = str(row.get("video_id", "")).strip()
    text_raw = normalize_ocr_nfc(row.get("ocr_text_raw", row.get("text_raw", "")))
    text_nfc = normalize_ocr_nfc(row.get("ocr_text_nfc", text_raw))
    if not crop_uid or not frame_uid or not video_id or not text_nfc:
        return None
    text_folded = normalize_ocr_nfc(row.get("ocr_text_folded", "")) or fold_ocr_text(text_nfc)
    source_manifest: dict[str, Any] = manifest if isinstance(manifest, dict) else {}
    model_contract_value = source_manifest.get("model_contract")
    model_contract: dict[str, Any] = (
        model_contract_value if isinstance(model_contract_value, dict) else {}
    )
    detector_value = model_contract.get("detector")
    detector: dict[str, Any] = detector_value if isinstance(detector_value, dict) else {}
    recognizer_value = model_contract.get("recognizer")
    recognizer: dict[str, Any] = recognizer_value if isinstance(recognizer_value, dict) else {}
    document: dict[str, Any] = {
        "crop_uid": crop_uid,
        "line_uid": str(row.get("line_uid") or crop_uid),
        "frame_uid": frame_uid,
        "video_id": video_id,
        "video_filename": row.get("video_filename"),
        "source_frame_idx": _int_or_none(row.get("source_frame_idx")),
        "timestamp_ms": _int_or_none(row.get("timestamp_ms")),
        "shot_id": row.get("shot_id"),
        "line_index": _int_or_none(row.get("line_index")),
        "detector_line_index": _int_or_none(row.get("detector_line_index")),
        "word_index": _int_or_none(row.get("word_index")),
        "det_score": _float_or_none(row.get("det_score")),
        "rec_score": _float_or_none(row.get("rec_score")),
        "text_raw": text_raw,
        "text_nfc": text_nfc,
        "text_folded": text_folded,
        "confidence_status": row.get("confidence_status"),
        "detector_model": row.get("detector_model") or detector.get("model"),
        "detector_revision": row.get("detector_revision") or detector.get("revision"),
        "recognizer_model": row.get("recognizer_model") or recognizer.get("model"),
        "recognizer_revision": row.get("recognizer_revision") or recognizer.get("revision"),
        "candidate_policy": row.get("candidate_policy"),
        "bbox": _json_field(row, "bbox"),
        "polygon": _json_field(row, "polygon"),
        "detector_polygon": _json_field(row, "detector_polygon"),
        "ocr_candidates": _json_field(row, "ocr_candidates"),
        "source_shard_id": row.get("source_shard_id"),
        "source_manifest_sha256": row.get("source_manifest_sha256"),
        "quality_status": row.get("quality_status")
        or source_manifest.get("quality_status", "UNVALIDATED_ON_HCMAIC"),
        "execution_status": row.get("execution_status") or "ENGINEERING_ARTIFACT_COMPLETE",
    }
    return {"_id": crop_uid, "document": document}


def _response_body(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    body = getattr(response, "body", None)
    if isinstance(body, dict):
        return body
    try:
        converted = dict(response)
    except (TypeError, ValueError) as exc:
        raise ElasticsearchOCRError("Elasticsearch response is not a JSON object") from exc
    return converted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_path(value: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "ocr_manifest.json"
    return path


def _normalize_snapshot_runtime_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a successful ES snapshot manifest for read-only runtime use.

    The snapshot is an aggregate-ledger artifact, not an OCR merge artifact.
    Only the dual runtime may opt into this compatibility normalization; index
    builders continue to require the original merged OCR manifest format.
    """

    if payload.get("schema") != OCR_ES_SNAPSHOT_SCHEMA:
        raise ElasticsearchOCRError("unsupported OCR runtime manifest schema")
    if payload.get("state") != "SUCCESS":
        raise ElasticsearchOCRError("OCR Elasticsearch snapshot is not successful")

    snapshot = payload.get("snapshot")
    source_index = payload.get("source_index")
    indices = payload.get("indices")
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise ElasticsearchOCRError("OCR snapshot manifest is missing snapshot")
    if not isinstance(source_index, str) or not source_index.strip():
        raise ElasticsearchOCRError("OCR snapshot manifest is missing source_index")
    if not isinstance(indices, list) or source_index not in indices:
        raise ElasticsearchOCRError("OCR snapshot manifest has inconsistent indices")

    shards = payload.get("shards")
    if not isinstance(shards, Mapping):
        raise ElasticsearchOCRError("OCR snapshot manifest is missing shard counts")
    total = shards.get("total")
    failed = shards.get("failed")
    successful = shards.get("successful")
    if (
        not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (total, failed, successful)
        )
        or total < 1
        or failed != 0
        or successful != total
    ):
        raise ElasticsearchOCRError("OCR snapshot manifest has invalid shard counts")

    source_doc_count = payload.get("source_doc_count")
    declared_row_count = payload.get("declared_row_count")
    count_delta = payload.get("count_delta")
    if (
        not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (source_doc_count, declared_row_count, count_delta)
        )
        or source_doc_count < 0
        or declared_row_count < 0
        or count_delta != source_doc_count - declared_row_count
    ):
        raise ElasticsearchOCRError("OCR snapshot manifest has inconsistent row counts")

    source_hashes: dict[str, str] = {}
    for field in ("source_index_manifest_sha256", "source_ocr_manifest_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ElasticsearchOCRError(f"OCR snapshot manifest has invalid {field}")
        source_hashes[field] = value

    quality_status = payload.get("quality_status", "UNVALIDATED")
    provenance_class = payload.get("provenance_class", "ENGINEERING_PROXY")
    if not isinstance(quality_status, str) or not quality_status.strip():
        raise ElasticsearchOCRError("OCR snapshot manifest has invalid quality_status")
    if not isinstance(provenance_class, str) or not provenance_class.strip():
        raise ElasticsearchOCRError("OCR snapshot manifest has invalid provenance_class")

    normalized = dict(payload)
    normalized["format"] = OCR_MERGED_MANIFEST_FORMAT
    normalized["snapshot_provenance"] = {
        "schema": OCR_ES_SNAPSHOT_SCHEMA,
        "snapshot": snapshot,
        "source_index": source_index,
        "indices": list(indices),
        "shards": {"total": total, "failed": failed, "successful": successful},
        "source_doc_count": source_doc_count,
        "declared_row_count": declared_row_count,
        "count_delta": count_delta,
        **source_hashes,
        "quality_status": quality_status,
        "provenance_class": provenance_class,
    }
    return normalized


def load_ocr_manifest(
    value: Path, *, allow_snapshot: bool = False
) -> tuple[dict[str, Any], str]:
    """Load an OCR manifest and return it with its file hash.

    ``allow_snapshot`` is intentionally opt-in and is reserved for the
    read-only dual runtime.  Indexing and KIS paths remain strict about the
    merged OCR artifact contract.
    """
    path = _manifest_path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ElasticsearchOCRError(f"cannot read OCR manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ElasticsearchOCRError(f"OCR manifest must be a JSON object: {path}")
    if allow_snapshot and payload.get("schema") == OCR_ES_SNAPSHOT_SCHEMA:
        payload = _normalize_snapshot_runtime_manifest(payload)
    return payload, _sha256_file(path)


def validate_ocr_index(client: Any, index_name: str) -> None:
    """Fail closed unless the configured Elasticsearch index is reachable."""
    index_name = _validate_index_name(index_name)
    try:
        exists = client.indices.exists(index=index_name)
    except Exception as exc:
        raise ElasticsearchOCRError("could not verify Elasticsearch OCR index") from exc
    if not bool(exists):
        raise ElasticsearchOCRError(f"Elasticsearch OCR index does not exist: {index_name}")


def ensure_ocr_index(client: Any, index_name: str, *, replace: bool = False) -> None:
    """Create one exact versioned index; deletion requires explicit ``replace``."""
    index_name = _validate_index_name(index_name)
    exists = bool(client.indices.exists(index=index_name))
    if exists and not replace:
        return
    if exists and replace:
        client.indices.delete(index=index_name)
    mapping = build_ocr_index_mapping()
    client.indices.create(
        index=index_name,
        settings=mapping["settings"],
        mappings=mapping["mappings"],
    )


def _load_manifest(artifact_dir: Path) -> dict[str, Any]:
    path = _manifest_path(artifact_dir)
    if not path.is_file():
        return {}
    return load_ocr_manifest(path)[0]


def bulk_index_ocr(
    client: Any,
    artifact_dir: Path,
    *,
    index_name: str,
    batch_size: int = 1_000,
    replace: bool = False,
    refresh: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stream OCR rows into Elasticsearch with ``crop_uid`` as idempotent ID."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    index_name = _validate_index_name(index_name)
    manifest = _load_manifest(Path(artifact_dir))
    if manifest.get("format") != "hcmaic-dstext-parseq-ocr-merged-v1":
        raise ElasticsearchOCRError(
            "index-ocr-es requires a merged hcmaic-dstext-parseq-ocr-merged-v1 artifact"
        )
    operations: list[Any] = []
    read_rows = indexed_rows = skipped_empty = batches = errors = 0
    error_examples: list[Any] = []
    if not dry_run:
        ensure_ocr_index(client, index_name, replace=replace)

    def flush() -> None:
        nonlocal batches, errors, operations
        if not operations:
            return
        if dry_run:
            batches += 1
            operations = []
            return
        response = _response_body(client.bulk(operations=operations, refresh=False))
        batches += 1
        if response.get("errors"):
            for item in response.get("items", []):
                action: Any = next(iter(item.values()), {}) if isinstance(item, dict) else {}
                if isinstance(action, dict) and int(action.get("status", 200)) >= 300:
                    errors += 1
                    if len(error_examples) < 5:
                        error_examples.append(action)
        operations = []

    for row in iter_ocr_rows(Path(artifact_dir), batch_size=batch_size):
        read_rows += 1
        payload = ocr_row_to_document(row, manifest)
        if payload is None:
            skipped_empty += 1
            continue
        indexed_rows += 1
        operations.extend(
            [
                {"index": {"_index": index_name, "_id": payload["_id"]}},
                payload["document"],
            ]
        )
        if len(operations) // 2 >= batch_size:
            flush()
    flush()
    if errors:
        raise ElasticsearchOCRError(
            f"Elasticsearch bulk indexing failed for {errors} document(s): {error_examples}"
        )
    if not dry_run and refresh:
        client.indices.refresh(index=index_name)
    return {
        "index": index_name,
        "artifact": str(Path(artifact_dir).resolve()),
        "format": manifest.get("format", "unknown"),
        "quality_status": manifest.get("quality_status", "UNVALIDATED_ON_HCMAIC"),
        "read_rows": read_rows,
        "indexed_rows": indexed_rows,
        "skipped_empty": skipped_empty,
        "bulk_batches": batches,
        "dry_run": dry_run,
    }


def build_ocr_search_body(
    query: str,
    *,
    top_k: int = 100,
    video_ids: Sequence[str] | None = None,
    include_low_conf: bool = True,
) -> dict[str, Any]:
    if not normalize_ocr_nfc(query):
        raise ValueError("OCR query must not be blank")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    filters: list[dict[str, Any]] = []
    if video_ids:
        filters.append({"terms": {"video_id": [str(value) for value in video_ids]}})
    must_not: list[dict[str, Any]] = []
    if not include_low_conf:
        must_not.append({"terms": {"confidence_status": ["LOW_CONF", "EMPTY"]}})
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "text_nfc^4",
                                "text_nfc.folded^3",
                                "text_nfc.ngram",
                                "text_folded^3",
                                "text_folded.ngram",
                            ],
                            "type": "best_fields",
                            "operator": "or",
                            "fuzziness": "AUTO",
                        }
                    }
                ],
                "filter": filters,
                "must_not": must_not,
            }
        },
        "collapse": {
            "field": "frame_uid",
            "inner_hits": {
                "name": "ocr_crops",
                "size": 5,
                "sort": [{"rec_score": "desc"}, {"_score": "desc"}],
            },
        },
        "sort": [{"_score": "desc"}, {"timestamp_ms": "asc"}],
    }


def _hit_source(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit.get("_source")
    return source if isinstance(source, dict) else {}


def _parse_inner_hits(hit: dict[str, Any]) -> list[dict[str, Any]]:
    inner = hit.get("inner_hits")
    if not isinstance(inner, dict):
        return []
    named = inner.get("ocr_crops")
    if not isinstance(named, dict):
        return []
    hits = named.get("hits")
    if not isinstance(hits, dict) or not isinstance(hits.get("hits"), list):
        return []
    return [_hit_source(item) for item in hits["hits"] if isinstance(item, dict)]


def search_ocr(
    client: Any,
    index_name: str,
    query: str,
    *,
    top_k: int = 100,
    video_ids: Sequence[str] | None = None,
    include_low_conf: bool = True,
) -> list[ChannelHit]:
    index_name = _validate_index_name(index_name)
    body = build_ocr_search_body(
        query, top_k=top_k, video_ids=video_ids, include_low_conf=include_low_conf
    )
    try:
        response = client.search(index=index_name, body=body)
    except TypeError:
        # A small compatibility path for clients/fakes that expose the newer
        # expanded keyword signature instead of the REST ``body`` argument.
        response = client.search(
            index=index_name,
            query=body["query"],
            collapse=body["collapse"],
            sort=body["sort"],
            size=body["size"],
        )
    payload = _response_body(response)
    hits_payload = payload.get("hits")
    raw_hits = hits_payload.get("hits", []) if isinstance(hits_payload, dict) else []
    result: list[ChannelHit] = []
    for rank, hit in enumerate(raw_hits, start=1):
        if not isinstance(hit, dict):
            continue
        source = _hit_source(hit)
        frame_uid = str(source.get("frame_uid", "")).strip()
        video_id = str(source.get("video_id", "")).strip()
        if not frame_uid or not video_id:
            continue
        matched = _parse_inner_hits(hit)
        primary = matched[0] if matched else source
        evidence = {
            "crop_uid": primary.get("crop_uid"),
            "matched_crops": [
                {
                    "crop_uid": item.get("crop_uid"),
                    "text_nfc": item.get("text_nfc"),
                    "rec_score": item.get("rec_score"),
                    "bbox": item.get("bbox"),
                    "polygon": item.get("polygon"),
                }
                for item in matched
            ],
            "bbox": primary.get("bbox"),
            "polygon": primary.get("polygon"),
            "det_score": primary.get("det_score"),
            "rec_score": primary.get("rec_score"),
            "confidence_status": primary.get("confidence_status"),
            "quality_status": source.get("quality_status"),
            "execution_status": source.get("execution_status"),
            "index": index_name,
        }
        result.append(
            ChannelHit(
                entity_id=frame_uid,
                video_id=video_id,
                timestamp_ms=_int_or_none(source.get("timestamp_ms")) or 0,
                modality="ocr",
                score=float(hit.get("_score") or 0.0),
                rank=rank,
                provider="deepsolo-parseq-elasticsearch",
                evidence_text=str(primary.get("text_nfc") or source.get("text_nfc") or ""),
                frame_uid=frame_uid,
                video_filename=source.get("video_filename"),
                source_frame_idx=_int_or_none(source.get("source_frame_idx")),
                evidence=evidence,
            )
        )
    return result


def make_elasticsearch_client(
    url: str,
    *,
    api_key_env: str = "ELASTIC_API_KEY",
    username_env: str | None = None,
    password_env: str | None = None,
    allow_anonymous_local: bool = False,
) -> Any:
    """Build an ES client from environment credentials without exposing them."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Elasticsearch URL must use http:// or https://")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Elasticsearch URL must not contain credentials or query secrets")
    try:
        from elasticsearch import Elasticsearch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ElasticsearchOCRError(
            "Elasticsearch client is not installed; install the system optional 'elastic' extra"
        ) from exc
    api_key = os.environ.get(api_key_env)
    if api_key:
        return Elasticsearch(url, api_key=api_key)
    if username_env and password_env:
        username = os.environ.get(username_env)
        password = os.environ.get(password_env)
        if username and password:
            return Elasticsearch(url, basic_auth=(username, password))
    if (
        allow_anonymous_local
        and parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    ):
        return Elasticsearch(url)
    raise ElasticsearchOCRError(
        f"no Elasticsearch credential found in {api_key_env}"
        + (f" or {username_env}/{password_env}" if username_env and password_env else "")
    )


class ElasticsearchOCRChannel:
    """Optional OCR channel; quality remains unvalidated until qrels pass."""

    provider = "deepsolo-parseq-elasticsearch"
    execution_status = "ENGINEERING_PROXY"

    def __init__(
        self,
        client: Any,
        index_name: str,
        manifest: dict[str, Any] | None = None,
        *,
        manifest_sha256: str | None = None,
        include_low_conf: bool = True,
    ) -> None:
        self.client = client
        self.index_name = _validate_index_name(index_name)
        self.manifest = manifest or {}
        self.include_low_conf = bool(include_low_conf)
        self.quality_status = str(self.manifest.get("quality_status", "UNVALIDATED_ON_HCMAIC"))
        model_contract = self.manifest.get("model_contract")
        model_contract = model_contract if isinstance(model_contract, dict) else {}
        detector = model_contract.get("detector")
        detector = detector if isinstance(detector, dict) else {}
        recognizer = model_contract.get("recognizer")
        recognizer = recognizer if isinstance(recognizer, dict) else {}
        self.revision = (
            f"{detector.get('revision', 'unknown')}+{recognizer.get('revision', 'unknown')}"
        )
        dataset_hash = self.manifest.get("dataset_manifest_hash")
        self.dataset_manifest_hash = str(dataset_hash) if dataset_hash else None
        self.artifact_hash = manifest_sha256

    def channel_contract(self) -> ChannelContract:
        evidence: dict[str, Any] = {
            "index": self.index_name,
            "format": self.manifest.get("format"),
            "include_low_conf": self.include_low_conf,
        }
        snapshot_provenance = self.manifest.get("snapshot_provenance")
        if isinstance(snapshot_provenance, Mapping):
            evidence["snapshot_provenance"] = dict(snapshot_provenance)
        return ChannelContract(
            channel="ocr",
            provider=self.provider,
            revision=self.revision,
            execution_status=self.execution_status,  # type: ignore[arg-type]
            quality_status=self.quality_status,  # type: ignore[arg-type]
            dataset_manifest_hash=self.dataset_manifest_hash,
            artifact_hash=self.artifact_hash,
            status="ready",
            evidence=evidence,
        )

    def search(
        self,
        query: str,
        top_k: int = 100,
        *,
        video_ids: Sequence[str] | None = None,
    ) -> list[ChannelHit]:
        try:
            hits = search_ocr(
                self.client,
                self.index_name,
                query,
                top_k=top_k,
                video_ids=video_ids,
                include_low_conf=self.include_low_conf,
            )
        except ElasticsearchOCRError:
            raise
        except Exception as exc:
            raise ElasticsearchOCRError(
                f"Elasticsearch OCR search failed for index {self.index_name}"
            ) from exc

        provenance = {
            "provider": self.provider,
            "revision": self.revision,
            "execution_status": self.execution_status,
            "quality_status": self.quality_status,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "artifact_hash": self.artifact_hash,
            "index": self.index_name,
        }
        enriched: list[ChannelHit] = []
        for hit in hits:
            evidence = dict(hit.evidence)
            evidence["raw_provenance"] = provenance
            enriched.append(replace(hit, provider=self.provider, evidence=evidence))
        return enriched
