"""Production-shaped Elasticsearch ASR channel for canonical frame retrieval.

The Elasticsearch index is a segment-level artifact.  This adapter deliberately
keeps that boundary explicit: it searches bounded transcript segments, expands
``segment_id`` through the versioned edge parquet, validates every target against
the canonical keyframe lookup, and only then emits ``ChannelHit`` instances keyed
by ``frame_uid``.  Retrieval evidence is engineering evidence; quality remains
``UNVALIDATED`` until qrels support a claim.
"""

from __future__ import annotations

import math
import os
import re
import socket
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hcmaic.retrieval.candidates import ChannelHit
from hcmaic.retrieval.channel_contract import ChannelContract

ASR_ES_PROVIDER = "aic26-transcripts-elasticsearch"
ASR_ES_MODES = ("pho", "whisper_v3", "rrf")
ASR_ES_MAX_SEGMENT_SIZE = 50
ASR_ES_DEFAULT_FUZZINESS = "AUTO:3,6"
ASR_ES_TIE_BREAKER = 0.15
ASR_ES_FUZZY_MINIMUM_SHOULD_MATCH = "60%"
ASR_ES_STATUS_VALUES = (
    "ready",
    "disabled_by_config",
    "disabled_by_policy",
    "unavailable",
    "schema_mismatch",
    "mapping_mismatch",
    "timeout",
)
_INDEX_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,254}$")
_MODE_ALIASES = {
    "pho": "pho",
    "phowhisper": "pho",
    "phowhisper_v1": "pho",
    "whisper": "whisper_v3",
    "whisperlarge": "whisper_v3",
    "whisper_large": "whisper_v3",
    "whisper-large": "whisper_v3",
    "whisper-v3": "whisper_v3",
    "whisper_v3": "whisper_v3",
    "rrf": "rrf",
}
_MODEL_FIELDS = {
    "pho": ("phowhisper_folded", "phowhisper_raw"),
    "whisper_v3": ("whisper_v3_folded", "whisper_v3_raw"),
}
_SOURCE_FIELDS = [
    "segment_id",
    "video_id",
    "start_ms",
    "end_ms",
    "phowhisper_raw",
    "whisper_v3_raw",
    "phowhisper_folded",
    "whisper_v3_folded",
    "mapped_keyframe_count",
    "anchor_frame_uid",
    "anchor_timestamp_ms",
]
_LOOKUP_REQUIRED_FIELDS = {"frame_uid", "video_id", "source_frame_idx", "timestamp_ms"}
_FUZZINESS_AUTO_RE = re.compile(r"^auto:(\d+),(\d+)$", re.IGNORECASE)
_EDGE_REQUIRED_FIELDS = {"segment_id", "frame_uid"}


class ASRElasticsearchError(RuntimeError):
    """Base error for the optional ASR Elasticsearch channel."""


class ASRElasticsearchUnavailableError(ASRElasticsearchError):
    """The configured Elasticsearch channel cannot currently be reached."""


class ASRElasticsearchTimeoutError(ASRElasticsearchError):
    """The bounded Elasticsearch request exceeded its timeout."""


class ASRESchemaMismatchError(ASRElasticsearchError):
    """The ES response or local mapping schema is not the declared contract."""


class ASRMappingMismatchError(ASRElasticsearchError):
    """A segment-to-canonical-frame mapping cannot be trusted."""


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().casefold()
    try:
        return _MODE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported ASR Elasticsearch mode: {mode!r}") from exc


def normalize_asr_query(query: str) -> str:
    """Apply the guide's deterministic Vietnamese query folding."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("ASR query must not be blank")
    decomposed = unicodedata.normalize("NFD", query).lower().replace("đ", "d")
    kept: list[str] = []
    for char in decomposed:
        if "\u0300" <= char <= "\u036f":
            continue
        if char.isalnum() or char.isspace() or char in ".,:/%+-":
            kept.append(char)
        else:
            kept.append(" ")
    normalized = " ".join("".join(kept).split())
    if not normalized:
        raise ValueError("ASR query is blank after normalization")
    return normalized


def _validate_index_name(index_name: str) -> str:
    if not _INDEX_NAME_RE.fullmatch(index_name) or ".." in index_name:
        raise ValueError(f"invalid Elasticsearch index name: {index_name!r}")
    return index_name


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().casefold() not in {"0", "false", "no", "off", "disabled"}


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _normalize_fuzziness(value: str | int | None) -> str | int | None:
    """Validate Elasticsearch match-query edit-distance syntax."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("ASR fuzziness must be AUTO, an integer in [0, 2], or null")
    if isinstance(value, int):
        if value not in {0, 1, 2}:
            raise ValueError("ASR fuzziness integer must be in [0, 2]")
        return value
    normalized = str(value).strip()
    if not normalized or normalized.casefold() in {"off", "none", "disabled"}:
        return None
    if normalized.casefold() == "auto":
        return "AUTO"
    auto_match = _FUZZINESS_AUTO_RE.fullmatch(normalized)
    if auto_match:
        low, high = (int(part) for part in auto_match.groups())
        if low > high:
            raise ValueError("ASR fuzziness AUTO thresholds must be ordered low,high")
        return f"AUTO:{low},{high}"
    if normalized.isdigit() and int(normalized) in {0, 1, 2}:
        return int(normalized)
    raise ValueError("ASR fuzziness must be AUTO, an integer in [0, 2], or null")


def _env_fuzziness(env: Mapping[str, str], name: str, default: str) -> str | int | None:
    value = env.get(name)
    return _normalize_fuzziness(default if value is None else value)


@dataclass(frozen=True)
class ASRElasticsearchConfig:
    """Explicit runtime configuration for the optional ASR channel."""

    url: str | None = None
    index: str = "aic26_transcripts_v1"
    edges_path: Path | None = None
    lookup_path: Path | None = None
    timeout_s: float = 2.0
    segment_top_n: int = ASR_ES_MAX_SEGMENT_SIZE
    mode: str = "rrf"
    fuzziness: str | int | None = ASR_ES_DEFAULT_FUZZINESS
    enabled: bool = False
    policy_enabled: bool = False
    rank_constant: int = 60
    api_key_env: str = "ELASTIC_API_KEY"
    username_env: str | None = None
    password_env: str | None = None
    dataset_manifest_hash: str | None = None
    artifact_hash: str | None = None

    def __post_init__(self) -> None:
        _validate_index_name(str(self.index))
        object.__setattr__(self, "mode", _normalize_mode(self.mode))
        object.__setattr__(self, "fuzziness", _normalize_fuzziness(self.fuzziness))
        if self.url is not None:
            parsed = urlparse(str(self.url))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("ASR Elasticsearch URL must use http:// or https://")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    "ASR Elasticsearch URL must not contain credentials or query secrets"
                )
        if self.timeout_s <= 0 or not math.isfinite(self.timeout_s):
            raise ValueError("timeout_s must be a finite positive number")
        if self.segment_top_n < 1 or self.segment_top_n > ASR_ES_MAX_SEGMENT_SIZE:
            raise ValueError(f"segment_top_n must be in [1, {ASR_ES_MAX_SEGMENT_SIZE}]")
        if self.rank_constant < 1:
            raise ValueError("rank_constant must be >= 1")
        if self.edges_path is not None:
            object.__setattr__(self, "edges_path", Path(self.edges_path).expanduser())
        if self.lookup_path is not None:
            object.__setattr__(self, "lookup_path", Path(self.lookup_path).expanduser())

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ASRElasticsearchConfig:
        values = dict(os.environ if env is None else env)
        return cls(
            url=values.get("HCMAIC_ASR_ES_URL") or values.get("ASR_ES_URL"),
            index=values.get("HCMAIC_ASR_ES_INDEX", "aic26_transcripts_v1"),
            edges_path=(
                Path(values["HCMAIC_ASR_ES_EDGES_PATH"])
                if values.get("HCMAIC_ASR_ES_EDGES_PATH")
                else None
            ),
            lookup_path=(
                Path(values["HCMAIC_ASR_ES_LOOKUP_PATH"])
                if values.get("HCMAIC_ASR_ES_LOOKUP_PATH")
                else None
            ),
            timeout_s=_env_float(values, "HCMAIC_ASR_ES_TIMEOUT_S", 2.0),
            segment_top_n=_env_int(values, "HCMAIC_ASR_ES_TOP_N", ASR_ES_MAX_SEGMENT_SIZE),
            mode=values.get("HCMAIC_ASR_ES_MODE", "rrf"),
            fuzziness=_env_fuzziness(values, "HCMAIC_ASR_ES_FUZZINESS", ASR_ES_DEFAULT_FUZZINESS),
            enabled=_env_bool(values, "HCMAIC_ASR_ES_ENABLED", False),
            policy_enabled=_env_bool(values, "HCMAIC_ASR_ES_POLICY_ENABLED", False),
            rank_constant=_env_int(values, "HCMAIC_ASR_ES_RANK_CONSTANT", 60),
            api_key_env=values.get("HCMAIC_ASR_ES_API_KEY_ENV", "ELASTIC_API_KEY"),
            username_env=values.get("HCMAIC_ASR_ES_USERNAME_ENV") or None,
            password_env=values.get("HCMAIC_ASR_ES_PASSWORD_ENV") or None,
        )

    @staticmethod
    def env_is_declared(env: Mapping[str, str] | None = None) -> bool:
        values = os.environ if env is None else env
        return any(
            name in values
            for name in (
                "HCMAIC_ASR_ES_URL",
                "HCMAIC_ASR_ES_INDEX",
                "HCMAIC_ASR_ES_EDGES_PATH",
                "HCMAIC_ASR_ES_LOOKUP_PATH",
                "HCMAIC_ASR_ES_ENABLED",
                "HCMAIC_ASR_ES_POLICY_ENABLED",
                "ASR_ES_URL",
            )
        )


@dataclass(frozen=True)
class ASRMappingIndex:
    edges_by_segment: dict[str, tuple[dict[str, Any], ...]]
    lookup_by_uid: dict[str, dict[str, Any]]
    edge_count: int
    frame_count: int
    unmapped_count: int


@dataclass(frozen=True)
class _SegmentHit:
    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    anchor_frame_uid: str | None
    anchor_timestamp_ms: int | None
    mapped_keyframe_count: int
    score: float
    rank: int
    source_models: tuple[str, ...]
    model_ranks: dict[str, int]
    raw_texts: dict[str, str]
    folded_texts: dict[str, str]


def _read_parquet_rows(path: Path, required_fields: set[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ASRMappingMismatchError(f"required ASR mapping file is unavailable: {path}")
    try:
        import pyarrow.parquet as parquet

        table = parquet.read_table(path)
    except Exception as exc:  # pragma: no cover - exact pyarrow errors vary by build
        raise ASRMappingMismatchError(f"cannot read ASR mapping parquet: {path}") from exc
    fields = set(table.column_names)
    missing = sorted(required_fields - fields)
    if missing:
        raise ASRESchemaMismatchError(
            f"ASR mapping schema is missing required field(s): {', '.join(missing)}"
        )
    return [dict(row) for row in table.to_pylist()]


def _coerce_nonnegative_int(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise ASRESchemaMismatchError(f"ASR field {field} is not an integer in {context}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ASRESchemaMismatchError(f"ASR field {field} is not an integer in {context}") from exc
    if result < 0:
        raise ASRMappingMismatchError(f"ASR field {field} is negative in {context}")
    return result


def _validate_frame_uid(frame_uid: str, *, context: str) -> tuple[str, int]:
    video_id, separator, source_idx = frame_uid.rpartition(":")
    if not separator or not video_id or not source_idx.isdigit():
        raise ASRMappingMismatchError(f"invalid canonical frame_uid {frame_uid!r} in {context}")
    return video_id, int(source_idx)


def load_asr_mapping(edges_path: Path, lookup_path: Path) -> ASRMappingIndex:
    """Load and validate only the segment-edge and canonical-frame metadata."""

    edge_rows = _read_parquet_rows(Path(edges_path), _EDGE_REQUIRED_FIELDS)
    lookup_rows = _read_parquet_rows(Path(lookup_path), _LOOKUP_REQUIRED_FIELDS)
    lookup_by_uid: dict[str, dict[str, Any]] = {}
    for row_number, raw in enumerate(lookup_rows, start=1):
        frame_uid = str(raw.get("frame_uid") or "").strip()
        if not frame_uid:
            raise ASRMappingMismatchError(f"lookup row {row_number} has blank frame_uid")
        video_id, source_idx = _validate_frame_uid(frame_uid, context=f"lookup row {row_number}")
        if frame_uid in lookup_by_uid:
            raise ASRMappingMismatchError(f"duplicate frame_uid in lookup: {frame_uid}")
        row = dict(raw)
        row["frame_uid"] = frame_uid
        row["video_id"] = str(raw.get("video_id") or "").strip()
        if not row["video_id"] or row["video_id"] != video_id:
            raise ASRMappingMismatchError(f"lookup video_id mismatch for {frame_uid}")
        row["source_frame_idx"] = _coerce_nonnegative_int(
            raw.get("source_frame_idx"), field="source_frame_idx", context=frame_uid
        )
        if row["source_frame_idx"] != source_idx:
            raise ASRMappingMismatchError(f"lookup source_frame_idx mismatch for {frame_uid}")
        row["timestamp_ms"] = _coerce_nonnegative_int(
            raw.get("timestamp_ms"), field="timestamp_ms", context=frame_uid
        )
        row.setdefault("video_filename", f"{row['video_id']}.mp4")
        lookup_by_uid[frame_uid] = row

    edges_by_segment: dict[str, list[dict[str, Any]]] = {}
    seen_edges: set[tuple[str, str]] = set()
    for row_number, raw in enumerate(edge_rows, start=1):
        segment_id = str(raw.get("segment_id") or "").strip()
        frame_uid = str(raw.get("frame_uid") or "").strip()
        if not segment_id or not frame_uid:
            raise ASRMappingMismatchError(f"edge row {row_number} has blank identity")
        _validate_frame_uid(frame_uid, context=f"edge row {row_number}")
        if frame_uid not in lookup_by_uid:
            raise ASRMappingMismatchError(
                f"edge frame_uid {frame_uid!r} is absent from canonical lookup"
            )
        edge_key = (segment_id, frame_uid)
        if edge_key in seen_edges:
            raise ASRMappingMismatchError(
                f"duplicate segment-to-frame edge: {segment_id!r} -> {frame_uid!r}"
            )
        seen_edges.add(edge_key)
        edge = dict(raw)
        edge["segment_id"] = segment_id
        edge["frame_uid"] = frame_uid
        if edge.get("overlap_ms") is not None:
            edge["overlap_ms"] = _coerce_nonnegative_int(
                edge["overlap_ms"], field="overlap_ms", context=f"edge row {row_number}"
            )
        edge["is_anchor"] = bool(edge.get("is_anchor", False))
        edges_by_segment.setdefault(segment_id, []).append(edge)

    return ASRMappingIndex(
        edges_by_segment={segment_id: tuple(rows) for segment_id, rows in edges_by_segment.items()},
        lookup_by_uid=lookup_by_uid,
        edge_count=len(edge_rows),
        frame_count=len(lookup_by_uid),
        unmapped_count=0,
    )


def _source_fields_for_mode(mode: str) -> list[str]:
    _normalize_mode(mode)
    return list(_SOURCE_FIELDS)


def _search_fields_for_mode(mode: str) -> list[str]:
    normalized = _normalize_mode(mode)
    if normalized == "rrf":
        return ["phowhisper_folded", "whisper_v3_folded"]
    return [_MODEL_FIELDS[normalized][0]]


def build_asr_search_body(
    query: str,
    *,
    mode: str = "pho",
    top_n: int = ASR_ES_MAX_SEGMENT_SIZE,
    video_ids: Sequence[str] | None = None,
    fuzziness: str | int | None = ASR_ES_DEFAULT_FUZZINESS,
) -> dict[str, Any]:
    """Build a bounded segment query without ES-side frame collapse."""

    normalized_query = normalize_asr_query(query)
    if top_n < 1 or top_n > ASR_ES_MAX_SEGMENT_SIZE:
        raise ValueError(f"top_n must be in [1, {ASR_ES_MAX_SEGMENT_SIZE}]")
    normalized = _normalize_mode(mode)
    normalized_fuzziness = _normalize_fuzziness(fuzziness)
    fields = _search_fields_for_mode(normalized)
    filters: list[dict[str, Any]] = []
    cleaned_videos = [str(value).strip() for value in (video_ids or ()) if str(value).strip()]
    if cleaned_videos:
        filters.append({"terms": {"video_id": cleaned_videos}})
    exact_clause: dict[str, Any] = {
        "query": normalized_query,
        "type": "best_fields",
        "fields": fields,
        "tie_breaker": ASR_ES_TIE_BREAKER,
        "operator": "and",
        "boost": 6,
        "_name": "exact",
    }
    fuzzy_clause: dict[str, Any] = {
        "query": normalized_query,
        "type": "best_fields",
        "fields": fields,
        "tie_breaker": ASR_ES_TIE_BREAKER,
        "prefix_length": 1,
        "max_expansions": 50,
        "fuzzy_transpositions": True,
        "minimum_should_match": ASR_ES_FUZZY_MINIMUM_SHOULD_MATCH,
        "boost": 1,
        "_name": "fuzzy",
    }
    if normalized_fuzziness is not None:
        fuzzy_clause["fuzziness"] = normalized_fuzziness
    return {
        "size": int(top_n),
        "track_total_hits": True,
        "_source": _source_fields_for_mode(normalized),
        "query": {
            "bool": {
                "minimum_should_match": 1,
                "should": [
                    {"multi_match": exact_clause},
                    {"multi_match": fuzzy_clause},
                ],
                "filter": filters,
            }
        },
        "sort": [{"_score": "desc"}, {"segment_id": "asc"}],
    }


def _response_body(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    body = getattr(response, "body", None)
    if isinstance(body, dict):
        return body
    try:
        converted = dict(response)
    except (TypeError, ValueError) as exc:
        raise ASRESchemaMismatchError("Elasticsearch response is not an object") from exc
    return converted


def _bounded_text(value: Any, limit: int = 2_000) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "…"


def _ordered_models(models: set[str]) -> tuple[str, ...]:
    return tuple(model for model in ("pho", "whisper_v3") if model in models)


def _parse_segments(response: Any, models: tuple[str, ...]) -> list[_SegmentHit]:
    payload = _response_body(response)
    hits_payload = payload.get("hits")
    if not isinstance(hits_payload, dict) or not isinstance(hits_payload.get("hits"), list):
        raise ASRESchemaMismatchError("Elasticsearch response is missing hits.hits")
    normalized_models = _ordered_models(set(models))
    if not normalized_models:
        raise ValueError("at least one ASR model is required to parse segments")
    parsed: list[_SegmentHit] = []
    seen_segment_ids: set[str] = set()
    for rank, raw_hit in enumerate(hits_payload["hits"], start=1):
        if not isinstance(raw_hit, dict) or not isinstance(raw_hit.get("_source"), dict):
            raise ASRESchemaMismatchError("Elasticsearch hit is missing _source")
        source = raw_hit["_source"]
        required = {
            "segment_id",
            "video_id",
            "start_ms",
            "end_ms",
            "anchor_frame_uid",
            "anchor_timestamp_ms",
            "mapped_keyframe_count",
        }
        for model in normalized_models:
            folded_field, raw_field = _MODEL_FIELDS[model]
            required.update((folded_field, raw_field))
        missing = sorted(field for field in required if field not in source)
        if missing:
            raise ASRESchemaMismatchError(
                f"Elasticsearch ASR document is missing field(s): {', '.join(missing)}"
            )
        segment_id = str(source.get("segment_id") or "").strip()
        video_id = str(source.get("video_id") or "").strip()
        if not segment_id or not video_id:
            raise ASRESchemaMismatchError("Elasticsearch ASR document has blank identity")
        if segment_id in seen_segment_ids:
            raise ASRESchemaMismatchError(f"duplicate segment_id in ES ranking: {segment_id}")
        seen_segment_ids.add(segment_id)
        try:
            score = float(raw_hit.get("_score"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ASRESchemaMismatchError(f"invalid ES score for segment {segment_id}") from exc
        if not math.isfinite(score):
            raise ASRESchemaMismatchError(f"non-finite ES score for segment {segment_id}")
        anchor = str(source.get("anchor_frame_uid") or "").strip() or None
        anchor_timestamp = None
        if source.get("anchor_timestamp_ms") is not None:
            anchor_timestamp = _coerce_nonnegative_int(
                source.get("anchor_timestamp_ms"),
                field="anchor_timestamp_ms",
                context=segment_id,
            )
        raw_texts: dict[str, str] = {}
        folded_texts: dict[str, str] = {}
        for candidate in ("pho", "whisper_v3"):
            folded_field, raw_field = _MODEL_FIELDS[candidate]
            if raw_field in source:
                raw_texts[candidate] = _bounded_text(source.get(raw_field))
            if folded_field in source:
                folded_texts[candidate] = _bounded_text(source.get(folded_field))
        parsed.append(
            _SegmentHit(
                segment_id=segment_id,
                video_id=video_id,
                start_ms=_coerce_nonnegative_int(
                    source.get("start_ms"), field="start_ms", context=segment_id
                ),
                end_ms=_coerce_nonnegative_int(
                    source.get("end_ms"), field="end_ms", context=segment_id
                ),
                anchor_frame_uid=anchor,
                anchor_timestamp_ms=anchor_timestamp,
                mapped_keyframe_count=_coerce_nonnegative_int(
                    source.get("mapped_keyframe_count"),
                    field="mapped_keyframe_count",
                    context=segment_id,
                ),
                score=score,
                rank=rank,
                source_models=normalized_models,
                model_ranks={model: rank for model in normalized_models},
                raw_texts=raw_texts,
                folded_texts=folded_texts,
            )
        )
    return parsed


def _parse_model_segments(response: Any, model: str) -> list[_SegmentHit]:
    normalized = _normalize_mode(model)
    if normalized == "rrf":
        raise ValueError("_parse_model_segments expects a single ASR model")
    return _parse_segments(response, (normalized,))


def _parse_combined_segments(response: Any) -> list[_SegmentHit]:
    return _parse_segments(response, ("pho", "whisper_v3"))


def _safe_edge_evidence(edge: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "segment_id",
        "frame_uid",
        "overlap_ms",
        "overlap_start_ms",
        "overlap_end_ms",
        "is_anchor",
        "edge_kind",
        "mapping_quality",
    )
    return {field: edge[field] for field in fields if field in edge}


def make_asr_elasticsearch_client(config: ASRElasticsearchConfig) -> Any:
    """Create an ES client without logging or exposing credential values."""

    if not config.url:
        raise ASRElasticsearchUnavailableError("ASR Elasticsearch URL is not configured")
    try:
        from elasticsearch import Elasticsearch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ASRElasticsearchUnavailableError(
            "Elasticsearch client is not installed for the ASR channel"
        ) from exc
    kwargs: dict[str, Any] = {"request_timeout": config.timeout_s}
    api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
    if api_key:
        kwargs["api_key"] = api_key
    elif config.username_env and config.password_env:
        username = os.environ.get(config.username_env)
        password = os.environ.get(config.password_env)
        if username and password:
            kwargs["basic_auth"] = (username, password)
    return Elasticsearch(config.url, **kwargs)


class ASRElasticsearchAdapter:
    """Bounded ASR segment search plus canonical frame expansion."""

    provider = ASR_ES_PROVIDER
    execution_status = "ENGINEERING_PROXY"
    quality_status = "UNVALIDATED"

    def __init__(
        self,
        *,
        config: ASRElasticsearchConfig,
        client: Any | None,
        mapping: ASRMappingIndex | None,
        status: str,
        reason: str | None,
    ) -> None:
        self.config = config
        self.client = client
        self.mapping = mapping
        self._status = status
        self._reason = reason
        self.revision = config.index
        self.dataset_manifest_hash = config.dataset_manifest_hash
        self.artifact_hash = config.artifact_hash

    @classmethod
    def from_config(
        cls,
        config: ASRElasticsearchConfig,
        *,
        client: Any | None = None,
        enabled: bool | None = None,
        policy_enabled: bool | None = None,
    ) -> ASRElasticsearchAdapter:
        effective = replace(
            config,
            enabled=config.enabled if enabled is None else bool(enabled),
            policy_enabled=(
                config.policy_enabled if policy_enabled is None else bool(policy_enabled)
            ),
        )
        if not effective.enabled:
            return cls(
                config=effective,
                client=None,
                mapping=None,
                status="disabled_by_config",
                reason="asr_es_disabled_by_config",
            )
        if not effective.policy_enabled:
            return cls(
                config=effective,
                client=None,
                mapping=None,
                status="disabled_by_policy",
                reason="asr_es_disabled_by_policy",
            )
        if not effective.url or effective.edges_path is None or effective.lookup_path is None:
            return cls(
                config=effective,
                client=None,
                mapping=None,
                status="unavailable",
                reason="asr_es_config_incomplete",
            )
        try:
            mapping = load_asr_mapping(effective.edges_path, effective.lookup_path)
        except ASRESchemaMismatchError:
            return cls(
                config=effective,
                client=None,
                mapping=None,
                status="schema_mismatch",
                reason="asr_es_mapping_schema_mismatch",
            )
        except ASRMappingMismatchError:
            return cls(
                config=effective,
                client=None,
                mapping=None,
                status="mapping_mismatch",
                reason="asr_es_mapping_mismatch",
            )
        if client is None:
            try:
                client = make_asr_elasticsearch_client(effective)
            except ASRElasticsearchError:
                return cls(
                    config=effective,
                    client=None,
                    mapping=mapping,
                    status="unavailable",
                    reason="asr_es_client_unavailable",
                )
        return cls(
            config=effective,
            client=client,
            mapping=mapping,
            status="ready",
            reason=None,
        )

    def status_dict(self) -> dict[str, Any]:
        mapping = self.mapping
        status = self._status
        execution_status = (
            "ENGINEERING_PROXY"
            if status == "ready"
            else "DISABLED_BY_CONFIG"
            if status == "disabled_by_config"
            else "DISABLED_BY_POLICY"
            if status == "disabled_by_policy"
            else "UNAVAILABLE"
        )
        return {
            "channel": "asr",
            "configured": bool(self.config.enabled),
            "ready": status == "ready",
            "status": status,
            "reason": self._reason,
            "provider": self.provider,
            "revision": self.revision,
            "execution_status": execution_status,
            "quality_status": self.quality_status,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "artifact_hash": self.artifact_hash,
            "mode": self.config.mode,
            "fuzziness": self.config.fuzziness,
            "index": self.config.index,
            "segment_top_n": self.config.segment_top_n,
            "timeout_s": self.config.timeout_s,
            "mapping": {
                "edge_count": mapping.edge_count if mapping else None,
                "frame_count": mapping.frame_count if mapping else None,
                "unmapped_count": mapping.unmapped_count if mapping else None,
            },
            "boundary": "elasticsearch_segment_to_frame_uid_mapping",
        }

    def channel_contract(self) -> ChannelContract:
        status = self._status
        return ChannelContract(
            channel="asr",
            provider=self.provider,
            revision=self.revision,
            execution_status=self.status_dict()["execution_status"],  # type: ignore[arg-type]
            quality_status=self.quality_status,  # type: ignore[arg-type]
            dataset_manifest_hash=self.dataset_manifest_hash,
            artifact_hash=self.artifact_hash,
            status=status,  # type: ignore[arg-type]
            reason=self._reason,
            configured=bool(self.config.enabled),
            ready=status == "ready",
            evidence={
                "index": self.config.index,
                "mode": self.config.mode,
                "fuzziness": self.config.fuzziness,
                "mapping": self.status_dict()["mapping"],
            },
        )

    def _set_failure(self, status: str, reason: str) -> None:
        if status not in ASR_ES_STATUS_VALUES:
            raise ValueError(f"unsupported ASR runtime status: {status}")
        self._status = status
        self._reason = reason

    def _ensure_ready(self) -> None:
        if self._status == "ready" and self.client is not None and self.mapping is not None:
            return
        raise ASRElasticsearchUnavailableError(
            self._reason or "ASR Elasticsearch channel unavailable"
        )

    @staticmethod
    def _is_timeout(exc: BaseException) -> bool:
        name = exc.__class__.__name__.casefold()
        return isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in name

    @staticmethod
    def _is_transport_failure(exc: BaseException) -> bool:
        name = exc.__class__.__name__.casefold()
        return isinstance(exc, (ConnectionError, OSError)) or any(
            token in name for token in ("connection", "transport", "network")
        )

    def _search_mode(
        self,
        query: str,
        mode: str,
        *,
        top_n: int,
        video_ids: Sequence[str] | None,
    ) -> list[_SegmentHit]:
        body = build_asr_search_body(
            query,
            mode=mode,
            top_n=top_n,
            video_ids=video_ids,
            fuzziness=self.config.fuzziness,
        )
        try:
            response = self.client.search(
                index=self.config.index,
                body=body,
                request_timeout=self.config.timeout_s,
            )
        except ASRElasticsearchError:
            raise
        except Exception as exc:
            if self._is_timeout(exc):
                self._set_failure("timeout", "asr_es_request_timeout")
                raise ASRElasticsearchTimeoutError("ASR Elasticsearch request timed out") from exc
            if self._is_transport_failure(exc):
                self._set_failure("unavailable", "asr_es_unavailable")
                raise ASRElasticsearchUnavailableError("ASR Elasticsearch is unavailable") from exc
            self._set_failure("unavailable", "asr_es_request_failed")
            raise ASRElasticsearchUnavailableError("ASR Elasticsearch request failed") from exc
        try:
            if mode == "rrf":
                return _parse_combined_segments(response)
            return _parse_model_segments(response, mode)
        except ASRESchemaMismatchError:
            self._set_failure("schema_mismatch", "asr_es_response_schema_mismatch")
            raise

    def search(
        self,
        query: str,
        top_k: int = 100,
        *,
        allowed_frame_uids: set[str] | None = None,
        video_ids: Sequence[str] | None = None,
        mode: str | None = None,
    ) -> list[ChannelHit]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("ASR query must not be blank")
        if top_k < 1 or top_k > 500:
            raise ValueError("top_k must be in [1, 500]")
        self._ensure_ready()
        if allowed_frame_uids is not None and not allowed_frame_uids:
            return []
        effective_mode = self.config.mode if mode is None else _normalize_mode(mode)
        normalized_query = normalize_asr_query(query)
        top_n = min(
            self.config.segment_top_n,
            max(top_k, top_k * 2),
            ASR_ES_MAX_SEGMENT_SIZE,
        )
        segments = self._search_mode(
            normalized_query,
            effective_mode,
            top_n=top_n,
            video_ids=video_ids,
        )
        allowed = None if allowed_frame_uids is None else {str(uid) for uid in allowed_frame_uids}
        allowed_videos = (
            None
            if video_ids is None
            else {str(video_id).strip() for video_id in video_ids if str(video_id).strip()}
        )
        aggregates: dict[str, list[tuple[_SegmentHit, dict[str, Any]]]] = {}
        for segment in segments:
            edges = self.mapping.edges_by_segment.get(segment.segment_id) if self.mapping else None
            if not edges:
                self._set_failure("mapping_mismatch", "asr_es_segment_mapping_missing")
                raise ASRMappingMismatchError(
                    f"segment_id {segment.segment_id!r} has no canonical mapping edges"
                )
            for edge in edges:
                frame_uid = str(edge["frame_uid"])
                lookup = self.mapping.lookup_by_uid.get(frame_uid) if self.mapping else None
                if lookup is None:
                    self._set_failure("mapping_mismatch", "asr_es_lookup_mismatch")
                    raise ASRMappingMismatchError(
                        f"edge frame_uid {frame_uid!r} is absent from canonical lookup"
                    )
                if allowed is not None and frame_uid not in allowed:
                    continue
                if allowed_videos is not None and str(lookup["video_id"]) not in allowed_videos:
                    continue
                aggregates.setdefault(frame_uid, []).append((segment, edge))

        frame_rows: list[tuple[str, float, list[tuple[_SegmentHit, dict[str, Any]]]]] = []
        for frame_uid, supports in aggregates.items():
            score = sum(float(segment.score) for segment, _ in supports)
            frame_rows.append((frame_uid, score, supports))
        frame_rows.sort(
            key=lambda item: (
                -item[1],
                str(self.mapping.lookup_by_uid[item[0]]["video_id"]),
                int(self.mapping.lookup_by_uid[item[0]]["source_frame_idx"]),
                item[0],
            )
        )

        results: list[ChannelHit] = []
        for rank, (frame_uid, frame_score, supports) in enumerate(frame_rows[:top_k], start=1):
            lookup = self.mapping.lookup_by_uid[frame_uid]
            ordered_supports = sorted(supports, key=lambda item: (item[0].rank, item[0].segment_id))
            primary, primary_edge = ordered_supports[0]
            source_models = _ordered_models(
                {model for segment, _ in ordered_supports for model in segment.source_models}
            )
            raw_texts: dict[str, str] = {}
            folded_texts: dict[str, str] = {}
            for segment, _ in ordered_supports:
                raw_texts.update(segment.raw_texts)
                folded_texts.update(segment.folded_texts)
            source_model = "+".join(source_models)
            evidence: dict[str, Any] = {
                "segment_id": primary.segment_id,
                "segment_start_ms": primary.start_ms,
                "segment_end_ms": primary.end_ms,
                "segment_rank": primary.rank,
                "segment_score": primary.score,
                "source_model": source_model,
                "source_models": list(source_models),
                "model_segment_ranks": dict(primary.model_ranks),
                "raw_transcript": primary.raw_texts.get(primary.source_models[0], ""),
                "folded_transcript": primary.folded_texts.get(primary.source_models[0], ""),
                "mapping_evidence": {
                    "support_count": len(ordered_supports),
                    "edge_count": len(ordered_supports),
                    "anchor_frame_uid": primary.anchor_frame_uid,
                    "anchor_timestamp_ms": primary.anchor_timestamp_ms,
                    "overlap_ms": primary_edge.get("overlap_ms"),
                    "is_anchor": bool(primary_edge.get("is_anchor", False)),
                    "supporting_edges": [
                        _safe_edge_evidence(edge) for _, edge in ordered_supports[:8]
                    ],
                },
                "supporting_segment_ids": [
                    segment.segment_id for segment, _ in ordered_supports[:8]
                ],
                "supporting_segments": [
                    {
                        "segment_id": segment.segment_id,
                        "source_model": "+".join(segment.source_models),
                        "segment_rank": segment.rank,
                        "segment_score": segment.score,
                        "start_ms": segment.start_ms,
                        "end_ms": segment.end_ms,
                        "raw_transcripts": dict(segment.raw_texts),
                        "folded_transcripts": dict(segment.folded_texts),
                        "mapping": _safe_edge_evidence(edge),
                    }
                    for segment, edge in ordered_supports[:8]
                ],
                "raw_transcripts": raw_texts,
                "folded_transcripts": folded_texts,
                "mode": effective_mode,
                "phowhisper_raw": raw_texts.get("pho"),
                "phowhisper_folded": folded_texts.get("pho"),
                "whisper_v3_raw": raw_texts.get("whisper_v3"),
                "whisper_v3_folded": folded_texts.get("whisper_v3"),
                "normalized_query": normalized_query,
                "search_fields": _search_fields_for_mode(effective_mode),
                "search_semantics": "best_fields_exact_plus_fuzzy",
                "frame_score_method": "sum_es_segment_scores",
                "frame_uid": frame_uid,
                "identity_key": "frame_uid",
                "index": self.config.index,
                "rank_constant": self.config.rank_constant,
            }
            results.append(
                ChannelHit(
                    entity_id=frame_uid,
                    video_id=str(lookup["video_id"]),
                    timestamp_ms=int(lookup["timestamp_ms"]),
                    modality="asr",
                    score=float(frame_score),
                    rank=rank,
                    provider=self.provider,
                    evidence_text=(
                        primary.raw_texts.get(primary.source_models[0])
                        or primary.folded_texts.get(primary.source_models[0])
                    ),
                    frame_uid=frame_uid,
                    video_filename=str(lookup.get("video_filename") or f"{lookup['video_id']}.mp4"),
                    source_frame_idx=int(lookup["source_frame_idx"]),
                    evidence=evidence,
                )
            )
        self._status = "ready"
        self._reason = None
        return results
