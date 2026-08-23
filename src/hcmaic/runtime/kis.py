"""Production KIS runtime composition over versioned raw-derived artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hcmaic.embedding.base import EmbeddingProvider
from hcmaic.embedding.factory import get_real_visual_provider
from hcmaic.retrieval.asr import ASRArtifactError, ASRRetrievalChannel, load_asr_artifact
from hcmaic.retrieval.asr_windows import (
    ASRWindowRetrievalChannel,
    load_asr_window_artifact,
)
from hcmaic.retrieval.channel_contract import ChannelContract
from hcmaic.retrieval.kis_orchestrator import KISHybridOrchestrator, KISHybridOutput
from hcmaic.retrieval.object_retrieval import (
    ObjectArtifactError,
    ObjectRetrievalChannel,
    load_object_artifact,
)
from hcmaic.retrieval.ocr_bm25 import (
    BM25OCRChannel,
    OCRArtifactError,
    load_ocr_artifact,
)
from hcmaic.retrieval.ocr_elasticsearch import (
    ElasticsearchOCRChannel,
    ElasticsearchOCRError,
    load_ocr_manifest,
    make_elasticsearch_client,
    validate_ocr_index,
)
from hcmaic.retrieval.rfdetr_object_sidecar import (
    RfdetrObjectSidecarAdapter,
    RfdetrObjectSidecarArtifactError,
    RfdetrObjectSidecarUnavailableError,
    load_rfdetr_object_sidecar,
)
from hcmaic.skillpixel.index import SkillPixelIndex, load_skillpixel_index
from hcmaic.skillpixel.retrieval import SkillPixelRetriever

OPTIONAL_CHANNELS = ("ocr", "object", "asr")


def _channel_status_entry(
    name: str,
    *,
    configured: bool,
    ready: bool,
    status: str,
    reason: str | None,
    provider: str | None = None,
    revision: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "channel": name,
        "configured": bool(configured),
        "ready": bool(ready),
        "status": status,
        "reason": reason,
        "provider": provider,
        "revision": revision,
    }
    payload.update(extra)
    return payload


def _channel_metadata(channel: Any) -> tuple[str | None, str | None]:
    provider = getattr(channel, "provider", None)
    revision = getattr(channel, "revision", None)
    return (
        str(provider) if provider is not None else None,
        str(revision) if revision is not None else None,
    )


def _normalize_channel_status(name: str, raw: Any) -> dict[str, Any]:
    """Normalize legacy string diagnostics into the structured status contract."""

    if isinstance(raw, Mapping):
        payload = dict(raw)
        status = str(payload.get("status") or "unavailable")
        if status.startswith("ready"):
            status = "ready"
        elif status.startswith("disabled"):
            status = "disabled_by_policy"
        elif status not in {"unavailable", "disabled_by_policy"}:
            status = "unavailable"
        payload["channel"] = name
        payload["status"] = status
        payload["ready"] = bool(payload.get("ready", status == "ready"))
        payload["configured"] = bool(payload.get("configured", payload["ready"]))
        payload.setdefault("reason", None)
        payload.setdefault("provider", None)
        payload.setdefault("revision", None)
        return payload
    value = str(raw or "")
    if value.startswith("ready"):
        status = "ready"
        reason = None
        configured = True
    elif value.startswith("disabled"):
        status = "disabled_by_policy"
        reason = "disabled_until_qrels_ablation_gain" if "qrels" in value else value
        configured = False
    elif value.startswith("unavailable:"):
        status = "unavailable"
        reason = value.split(":", 1)[1].strip() or "optional_channel_unavailable"
        configured = True
    else:
        status = "unavailable"
        reason = value or "optional_channel_not_configured"
        configured = False
    return _channel_status_entry(
        name,
        configured=configured,
        ready=status == "ready",
        status=status,
        reason=reason,
    )


def _build_channel_status(
    provider: EmbeddingProvider,
    optional_channels: Mapping[str, Any],
    *,
    asr_enabled: bool,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {
        "visual": _channel_status_entry(
            "visual",
            configured=True,
            ready=True,
            status="ready",
            reason=None,
            provider=str(provider.name),
            revision=str(provider.version),
        )
    }
    supplied = overrides or {}
    for name in OPTIONAL_CHANNELS:
        channel = optional_channels.get(name)
        channel_provider, channel_revision = _channel_metadata(channel)
        if name == "asr" and not asr_enabled:
            statuses[name] = _channel_status_entry(
                name,
                configured=channel is not None,
                ready=False,
                status="disabled_by_policy",
                reason="disabled_until_qrels_ablation_gain",
                provider=channel_provider,
                revision=channel_revision,
            )
        elif channel is not None:
            statuses[name] = _channel_status_entry(
                name,
                configured=True,
                ready=True,
                status="ready",
                reason=None,
                provider=channel_provider,
                revision=channel_revision,
            )
            supplied_status = supplied.get(name)
            if isinstance(supplied_status, Mapping):
                # Preserve loader-specific diagnostics such as an external
                # backend/index and manifest hash without allowing the
                # injected channel to claim a different readiness state.
                for key, value in supplied_status.items():
                    if key not in {
                        "channel",
                        "configured",
                        "ready",
                        "status",
                        "reason",
                        "provider",
                        "revision",
                    }:
                        statuses[name][str(key)] = value
        elif name in supplied:
            statuses[name] = _normalize_channel_status(name, supplied[name])
        else:
            statuses[name] = _channel_status_entry(
                name,
                configured=False,
                ready=False,
                status="unavailable",
                reason=f"{name}_artifact_not_configured",
            )
        if name in supplied and channel is None:
            statuses[name] = _normalize_channel_status(name, supplied[name])
    return statuses


def _build_channel_contracts(
    provider: EmbeddingProvider,
    optional_channels: Mapping[str, Any],
    channel_status: Mapping[str, Any],
    index: SkillPixelIndex,
) -> dict[str, dict[str, Any]]:
    visual_manifest = dict(index.index_manifest)
    visual_status = dict(channel_status.get("visual", {}))
    contracts: dict[str, dict[str, Any]] = {
        "visual": ChannelContract(
            channel="visual",
            provider=str(provider.name),
            revision=str(provider.version),
            execution_status="ENGINEERING_PROXY",
            quality_status="UNVALIDATED_ON_HCMAIC",
            dataset_manifest_hash=(
                str(visual_manifest.get("dataset_manifest_hash"))
                if visual_manifest.get("dataset_manifest_hash")
                else None
            ),
            artifact_hash=(
                str(
                    visual_manifest.get("index_sha256")
                    or visual_manifest.get("sha256")
                    or visual_manifest.get("catalog_sha256")
                    or visual_manifest.get("frame_catalog_sha256")
                )
                if visual_manifest.get("index_sha256")
                or visual_manifest.get("sha256")
                or visual_manifest.get("catalog_sha256")
                or visual_manifest.get("frame_catalog_sha256")
                else None
            ),
            status=str(visual_status.get("status", "ready")),
            reason=visual_status.get("reason"),
            configured=bool(visual_status.get("configured", True)),
            ready=bool(visual_status.get("ready", True)),
        ).to_status_dict()
    }

    for name in OPTIONAL_CHANNELS:
        raw_status = dict(channel_status.get(name, {}))
        channel = optional_channels.get(name)
        provider_name = getattr(channel, "provider", None) if channel is not None else None
        revision = getattr(channel, "revision", None) if channel is not None else None
        execution_status = (
            getattr(channel, "execution_status", None) if channel is not None else None
        )
        quality_status = getattr(channel, "quality_status", None) if channel is not None else None
        dataset_manifest_hash = (
            getattr(channel, "dataset_manifest_hash", None) if channel is not None else None
        )
        artifact_hash = getattr(channel, "artifact_hash", None) if channel is not None else None
        if provider_name is None:
            provider_name = raw_status.get("provider") or "unavailable"
        if revision is None:
            revision = raw_status.get("revision") or "unavailable"
        if execution_status is None:
            execution_status = (
                "DISABLED_BY_POLICY"
                if raw_status.get("status") == "disabled_by_policy"
                else "ENGINEERING_PROXY"
            )
        if quality_status is None:
            quality_status = "UNVALIDATED_ON_HCMAIC"
        contracts[name] = ChannelContract(
            channel=name,
            provider=str(provider_name),
            revision=str(revision),
            execution_status=str(execution_status),
            quality_status=str(quality_status),
            dataset_manifest_hash=(str(dataset_manifest_hash) if dataset_manifest_hash else None),
            artifact_hash=str(artifact_hash) if artifact_hash else None,
            status=str(raw_status.get("status", "unavailable")),
            reason=raw_status.get("reason"),
            configured=bool(raw_status.get("configured", False)),
            ready=bool(raw_status.get("ready", False)),
        ).to_status_dict()
    return contracts


@dataclass
class KISRuntime:
    """Loaded KIS graph and diagnostics used by CLI, API and rehearsal."""

    index: SkillPixelIndex
    provider: EmbeddingProvider
    retriever: SkillPixelRetriever
    orchestrator: KISHybridOrchestrator
    provider_selection: dict[str, Any]
    channel_status: dict[str, dict[str, Any]]
    channel_contracts: dict[str, dict[str, Any]]

    @classmethod
    def from_components(
        cls,
        index: SkillPixelIndex,
        provider: EmbeddingProvider,
        *,
        optional_channels: dict[str, Any] | None = None,
        provider_selection: dict[str, Any] | None = None,
        channel_status: dict[str, Any] | None = None,
        asr_enabled: bool = False,
        max_per_video: int | None = 5,
    ) -> KISRuntime:
        attached_channels = optional_channels or {}
        statuses = _build_channel_status(
            provider,
            attached_channels,
            asr_enabled=asr_enabled,
            overrides=channel_status,
        )
        contracts = _build_channel_contracts(provider, attached_channels, statuses, index)
        retriever = SkillPixelRetriever(index, provider)
        orchestrator = KISHybridOrchestrator(
            retriever,
            optional_channels=attached_channels,
            asr_enabled=asr_enabled,
            max_per_video=max_per_video,
            channel_status=statuses,
        )
        return cls(
            index=index,
            provider=provider,
            retriever=retriever,
            orchestrator=orchestrator,
            provider_selection=provider_selection or {},
            channel_status=statuses,
            channel_contracts=contracts,
        )

    def search(self, query: Any) -> KISHybridOutput:
        return self.orchestrator.search(query)

    def search_queries(self, queries: list[Any]) -> dict[str, KISHybridOutput]:
        return self.orchestrator.search_queries(queries)

    def frame_image_path(self, frame_uid: str) -> Path:
        for record in self.index.catalog:
            if record.frame_id == frame_uid:
                root = self.index.dataset_root.resolve()
                path = (root / record.image_path).resolve()
                try:
                    path.relative_to(root)
                except ValueError as exc:
                    raise PermissionError("KIS image path escapes dataset root") from exc
                return path
        raise KeyError(frame_uid)

    def timeline(self, video_id: str) -> list[dict[str, Any]]:
        frames = [record for record in self.index.catalog if record.video_id == video_id]
        if not frames:
            raise KeyError(video_id)
        frames.sort(
            key=lambda record: (
                record.source_frame_idx
                if record.source_frame_idx is not None
                else record.frame_idx,
                record.frame_id,
            )
        )
        return [
            {
                **record.model_dump(),
                "source_frame_idx": record.source_frame_idx
                if record.source_frame_idx is not None
                else record.frame_idx,
                "image_url": f"/frames/{record.frame_id}/image",
            }
            for record in frames
        ]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "kis_runtime": True,
            "index_size": self.index.size,
            "index_version": self.index.index_manifest.get("index_version"),
            "embedding_provider": self.provider.name,
            "embedding": self.provider.info(),
            "n_videos": len({record.video_id for record in self.index.catalog}),
            "channels": dict(self.channel_status),
            "channel_status": dict(self.channel_status),
            "channel_contracts": dict(self.channel_contracts),
            "execution_status": "ENGINEERING_PROXY",
            "quality_status": "UNVALIDATED_ON_HCMAIC",
        }


def load_kis_runtime(
    index_dir: Path,
    *,
    provider: str = "auto",
    device: str | None = None,
    local_files_only: bool = True,
    batch_size: int | None = None,
    ocr_artifact: Path | None = None,
    ocr_es_url: str | None = None,
    ocr_es_index: str | None = None,
    ocr_es_manifest: Path | None = None,
    ocr_es_api_key_env: str = "ELASTIC_API_KEY",
    ocr_es_username_env: str | None = None,
    ocr_es_password_env: str | None = None,
    ocr_es_include_low_conf: bool = True,
    object_artifact: Path | None = None,
    object_sidecar: Path | None = None,
    allow_engineering_proxy: bool = False,
    asr_artifact: Path | None = None,
    asr_window_artifact: Path | None = None,
    asr_keyframe_manifest_hash: str | None = None,
    asr_enabled: bool = False,
) -> KISRuntime:
    """Load the exact provider/index pair and optional local channel artifacts."""
    index = load_skillpixel_index(index_dir)
    expected_provider = str(index.provider_info.get("provider", ""))
    prefer = expected_provider if provider == "auto" else provider
    if prefer not in {"siglip2", "clip", "jina-clip-v2"}:
        raise ValueError(f"unsupported KIS visual provider {prefer!r}")
    visual_provider, selection = get_real_visual_provider(
        prefer=prefer,
        device=device,
        local_files_only=local_files_only,
        revision=index.provider_info.get("model_revision"),
        batch_size=batch_size,
    )
    if visual_provider.name != expected_provider:
        raise RuntimeError(
            f"loaded provider {visual_provider.name!r} does not match index provider "
            f"{expected_provider!r}; rebuild index with the cached real provider"
        )
    optional_channels: dict[str, Any] = {}
    dataset_hash = str(index.index_manifest.get("dataset_manifest_hash", "")).strip() or None
    channel_status = _build_channel_status(
        visual_provider,
        optional_channels,
        asr_enabled=asr_enabled,
    )

    def unavailable(name: str, reason: str, *, configured: bool = True) -> None:
        channel_status[name] = _channel_status_entry(
            name,
            configured=configured,
            ready=False,
            status="unavailable",
            reason=reason,
        )

    ocr_es_requested = any(
        value is not None
        for value in (
            ocr_es_url,
            ocr_es_index,
            ocr_es_manifest,
            ocr_es_username_env,
            ocr_es_password_env,
        )
    )
    if ocr_artifact is not None and ocr_es_requested:
        raise ValueError("configure either --ocr-artifact or Elasticsearch OCR, not both")

    if ocr_es_requested:
        if not ocr_es_url or not ocr_es_index:
            raise ValueError("Elasticsearch OCR requires both ocr_es_url and ocr_es_index")
        if not allow_engineering_proxy:
            unavailable("ocr", "engineering_proxy_disabled_by_policy")
        elif ocr_es_manifest is None:
            unavailable("ocr", "ocr_es_manifest_not_configured")
        else:
            try:
                manifest, manifest_sha256 = load_ocr_manifest(Path(ocr_es_manifest))
                if manifest.get("format") != "hcmaic-dstext-parseq-ocr-merged-v1":
                    raise ElasticsearchOCRError(
                        "Elasticsearch OCR runtime requires a merged OCR manifest"
                    )
                client = make_elasticsearch_client(
                    ocr_es_url,
                    api_key_env=ocr_es_api_key_env,
                    username_env=ocr_es_username_env,
                    password_env=ocr_es_password_env,
                )
                validate_ocr_index(client, ocr_es_index)
                optional_channels["ocr"] = ElasticsearchOCRChannel(
                    client,
                    ocr_es_index,
                    manifest,
                    manifest_sha256=manifest_sha256,
                    include_low_conf=ocr_es_include_low_conf,
                )
                provider_name, revision = _channel_metadata(optional_channels["ocr"])
                channel_status["ocr"] = _channel_status_entry(
                    "ocr",
                    configured=True,
                    ready=True,
                    status="ready",
                    reason=None,
                    provider=provider_name,
                    revision=revision,
                    backend="elasticsearch",
                    index=ocr_es_index,
                    manifest_sha256=manifest_sha256,
                    execution_status="ENGINEERING_PROXY",
                    quality_status=str(manifest.get("quality_status", "UNVALIDATED_ON_HCMAIC")),
                    include_low_conf=ocr_es_include_low_conf,
                )
            except (
                ElasticsearchOCRError,
                FileNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                unavailable("ocr", f"{type(exc).__name__}: {exc}")
    elif ocr_artifact is not None:
        if dataset_hash is None:
            unavailable("ocr", "index_dataset_manifest_hash_missing")
        else:
            try:
                optional_channels["ocr"] = BM25OCRChannel(
                    load_ocr_artifact(ocr_artifact, dataset_manifest_hash=dataset_hash)
                )
                provider_name, revision = _channel_metadata(optional_channels["ocr"])
                channel_status["ocr"] = _channel_status_entry(
                    "ocr",
                    configured=True,
                    ready=True,
                    status="ready",
                    reason=None,
                    provider=provider_name,
                    revision=revision,
                )
            except OCRArtifactError as exc:
                unavailable("ocr", f"{type(exc).__name__}: {exc}")
    if object_artifact is not None:
        if dataset_hash is None:
            unavailable("object", "index_dataset_manifest_hash_missing")
        else:
            try:
                optional_channels["object"] = ObjectRetrievalChannel(
                    load_object_artifact(object_artifact, dataset_manifest_hash=dataset_hash)
                )
                provider_name, revision = _channel_metadata(optional_channels["object"])
                channel_status["object"] = _channel_status_entry(
                    "object",
                    configured=True,
                    ready=True,
                    status="ready",
                    reason=None,
                    provider=provider_name,
                    revision=revision,
                )
            except ObjectArtifactError as exc:
                unavailable("object", f"{type(exc).__name__}: {exc}")
    elif object_sidecar is not None:
        if not allow_engineering_proxy:
            unavailable(
                "object",
                "engineering_proxy_disabled_by_policy",
                configured=True,
            )
        else:
            try:
                optional_channels["object"] = RfdetrObjectSidecarAdapter(
                    load_rfdetr_object_sidecar(
                        object_sidecar,
                        allow_engineering_proxy=True,
                    )
                )
                provider_name, revision = _channel_metadata(optional_channels["object"])
                channel_status["object"] = _channel_status_entry(
                    "object",
                    configured=True,
                    ready=True,
                    status="ready",
                    reason=None,
                    provider=provider_name,
                    revision=revision,
                )
            except (RfdetrObjectSidecarArtifactError, RfdetrObjectSidecarUnavailableError) as exc:
                unavailable("object", f"{type(exc).__name__}: {exc}")

    asr_configured = asr_artifact is not None or asr_window_artifact is not None
    if not asr_enabled:
        channel_status["asr"] = _channel_status_entry(
            "asr",
            configured=asr_configured,
            ready=False,
            status="disabled_by_policy",
            reason="disabled_until_qrels_ablation_gain",
        )
    elif asr_window_artifact is not None:
        if dataset_hash is None:
            unavailable("asr", "index_dataset_manifest_hash_missing")
        elif not asr_keyframe_manifest_hash:
            unavailable("asr", "keyframe_manifest_hash_missing")
        else:
            try:
                optional_channels["asr"] = ASRWindowRetrievalChannel(
                    load_asr_window_artifact(
                        asr_window_artifact,
                        dataset_manifest_hash=dataset_hash,
                        keyframe_manifest_hash=asr_keyframe_manifest_hash,
                    )
                )
                provider_name, revision = _channel_metadata(optional_channels["asr"])
                channel_status["asr"] = _channel_status_entry(
                    "asr",
                    configured=True,
                    ready=True,
                    status="ready",
                    reason=None,
                    provider=provider_name,
                    revision=revision,
                    variant="asr-window-v2",
                )
            except ASRArtifactError as exc:
                unavailable("asr", f"{type(exc).__name__}: {exc}")
    elif asr_artifact is not None:
        if dataset_hash is None:
            unavailable("asr", "index_dataset_manifest_hash_missing")
        else:
            try:
                optional_channels["asr"] = ASRRetrievalChannel(
                    load_asr_artifact(asr_artifact, dataset_manifest_hash=dataset_hash)
                )
                provider_name, revision = _channel_metadata(optional_channels["asr"])
                channel_status["asr"] = _channel_status_entry(
                    "asr",
                    configured=True,
                    ready=True,
                    status="ready",
                    reason=None,
                    provider=provider_name,
                    revision=revision,
                    variant="asr-v1",
                )
            except ASRArtifactError as exc:
                unavailable("asr", f"{type(exc).__name__}: {exc}")
    else:
        unavailable("asr", "asr_artifact_not_configured", configured=False)

    return KISRuntime.from_components(
        index,
        visual_provider,
        optional_channels=optional_channels,
        provider_selection=selection,
        channel_status=channel_status,
        asr_enabled=asr_enabled,
    )
