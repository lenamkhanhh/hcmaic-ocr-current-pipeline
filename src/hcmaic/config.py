"""Typed configuration and artifact provenance helpers."""

from __future__ import annotations

import dataclasses as _dc
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


@_dc.dataclass(frozen=True)
class ProviderSpec:
    name: str
    version: str | None = None
    params: tuple[tuple[str, Any], ...] | Mapping[str, Any] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.version is not None:
            payload["version"] = self.version
        if self.params:
            payload["params"] = (
                dict(self.params)
                if isinstance(self.params, Mapping)
                else {k: v for k, v in self.params}
            )
        return payload


@_dc.dataclass(frozen=True)
class CompetitiveFoundationConfig:
    dataset_adapter: ProviderSpec
    ingestion_backend: ProviderSpec
    shot_detector: ProviderSpec
    embedding_provider: ProviderSpec
    index_provider: ProviderSpec
    fusion: ProviderSpec
    reranker: ProviderSpec
    benchmark_inputs: ProviderSpec
    device: str
    batch_size: int
    seed: int
    sampling_policy: ProviderSpec | None = None
    modality_extractors: tuple[ProviderSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.device.strip():
            raise ValueError("device must not be blank")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.sampling_policy is None:
            object.__setattr__(
                self,
                "sampling_policy",
                ProviderSpec(name="uniform", version="fallback"),
            )
        assert self.sampling_policy is not None

    def to_dict(self) -> dict[str, Any]:
        sampling_policy = self.sampling_policy or ProviderSpec(name="uniform", version="fallback")
        return {
            "dataset_adapter": self.dataset_adapter.to_dict(),
            "ingestion_backend": self.ingestion_backend.to_dict(),
            "shot_detector": self.shot_detector.to_dict(),
            "sampling_policy": sampling_policy.to_dict(),
            "modality_extractors": [provider.to_dict() for provider in self.modality_extractors],
            "embedding_provider": self.embedding_provider.to_dict(),
            "index_provider": self.index_provider.to_dict(),
            "fusion": self.fusion.to_dict(),
            "reranker": self.reranker.to_dict(),
            "benchmark_inputs": self.benchmark_inputs.to_dict(),
            "device": self.device,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }


def _provider_spec(value: Any, *, default_name: str) -> ProviderSpec:
    if value is None:
        return ProviderSpec(name=default_name)
    if isinstance(value, str):
        return ProviderSpec(name=value)
    if not isinstance(value, Mapping):
        raise ValueError(f"provider spec for {default_name} must be a mapping")
    params = value.get("params", {})
    if params is not None and not isinstance(params, Mapping):
        raise ValueError(
            f"provider spec {value.get('name', default_name)} params must be a mapping"
        )
    return ProviderSpec(
        name=str(value.get("name", default_name)),
        version=str(value["version"]) if value.get("version") is not None else None,
        params=dict(params or {}),
    )


def load_config(path: Path) -> CompetitiveFoundationConfig:
    """Load and validate a small YAML foundation config without side effects."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency is locked
        raise ValueError("PyYAML is required to load foundation config files.") from exc

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"config file not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("config root must be a mapping")
    modalities = payload.get("modality_extractors", ())
    if not isinstance(modalities, (list, tuple)):
        raise ValueError("modality_extractors must be a list")
    return CompetitiveFoundationConfig(
        dataset_adapter=_provider_spec(
            payload.get("dataset_adapter"), default_name="local-fixture"
        ),
        ingestion_backend=_provider_spec(payload.get("ingestion_backend"), default_name="opencv"),
        shot_detector=_provider_spec(payload.get("shot_detector"), default_name="uniform"),
        sampling_policy=_provider_spec(payload.get("sampling_policy"), default_name="uniform"),
        modality_extractors=tuple(
            _provider_spec(item, default_name="disabled") for item in modalities
        ),
        embedding_provider=_provider_spec(payload.get("embedding_provider"), default_name="mock"),
        index_provider=_provider_spec(payload.get("index_provider"), default_name="exact-numpy"),
        fusion=_provider_spec(payload.get("fusion"), default_name="single-stage"),
        reranker=_provider_spec(payload.get("reranker"), default_name="identity"),
        benchmark_inputs=_provider_spec(
            payload.get("benchmark_inputs"), default_name="proxy-fixture"
        ),
        device=str(payload.get("device", "cpu")),
        batch_size=int(payload.get("batch_size", 1)),
        seed=int(payload.get("seed", 0)),
    )


def _canonicalize(value: Any) -> Any:
    if _dc.is_dataclass(value):
        return _canonicalize(_dc.asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return {
            str(k): _canonicalize(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    if isinstance(value, tuple):
        return [_canonicalize(v) for v in value]
    return value


def config_hash(config: CompetitiveFoundationConfig) -> str:
    payload = json.dumps(
        _canonicalize(config.to_dict()),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_provenance(
    config: CompetitiveFoundationConfig,
    *,
    code_version: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = {
        "config": config.to_dict(),
        "config_hash": config_hash(config),
        "code_version": code_version,
    }
    if extra:
        provenance["extra"] = _canonicalize(dict(extra))
    return provenance
