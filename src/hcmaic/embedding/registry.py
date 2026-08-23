"""Configuration-driven embedding provider registry and diagnostics."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderDescriptor:
    name: str
    model_revision: str
    dependencies: tuple[str, ...]
    lazy: bool
    install: str
    evidence_level: str


_DESCRIPTORS = (
    ProviderDescriptor(
        "mock",
        "mock-palette-v1",
        (),
        False,
        "built in",
        "VERIFIED",
    ),
    ProviderDescriptor(
        "clip",
        "openai/clip-vit-base-patch32",
        ("torch", "transformers"),
        True,
        "uv sync --extra clip",
        "INTERFACE_ONLY",
    ),
    ProviderDescriptor(
        "siglip2",
        "google/siglip2-base-patch16-224",
        ("torch", "transformers"),
        True,
        "uv sync --extra clip",
        "INTERFACE_ONLY",
    ),
    ProviderDescriptor(
        "jina-clip-v2",
        "jinaai/jina-clip-v2",
        ("torch", "transformers"),
        True,
        "uv sync --extra clip",
        "INTERFACE_ONLY",
    ),
)


def list_provider_descriptors() -> tuple[ProviderDescriptor, ...]:
    return _DESCRIPTORS


def get_provider_descriptor(name: str) -> ProviderDescriptor:
    for descriptor in _DESCRIPTORS:
        if descriptor.name == name:
            return descriptor
    names = ", ".join(item.name for item in _DESCRIPTORS)
    raise ValueError(f"Unknown embedding provider {name!r}; expected {names}.")


def provider_doctor(
    name: str, *, device: str = "cpu", revision: str | None = None
) -> dict[str, Any]:
    """Report capability without importing models or accessing the network."""
    descriptor = get_provider_descriptor(name)
    availability = {
        package: importlib.util.find_spec(package) is not None
        for package in descriptor.dependencies
    }
    dependencies_ok = all(availability.values())
    evidence = descriptor.evidence_level
    if name in {"siglip2", "jina-clip-v2"}:
        evidence = "INTERFACE_ONLY" if dependencies_ok else "BLOCKED"
    return {
        "provider": name,
        "model_revision": revision or descriptor.model_revision,
        "dependencies": availability,
        "dependencies_ok": dependencies_ok,
        "device": device,
        "dimension": None,
        "evidence_level": evidence,
        "install": descriptor.install,
        "action": (
            "Use an explicit local smoke command with approved model cache; "
            "ordinary tests and doctor do not fetch weights."
        ),
    }


def create_provider(name: str, **kwargs: Any):
    """Create a control provider or a lazy optional adapter."""
    if name in {"mock", "clip"}:
        from hcmaic.embedding.base import get_provider as _legacy_get_provider

        return _legacy_get_provider(name, **kwargs)
    if name in {"siglip2", "jina-clip-v2"}:
        from hcmaic.embedding.optional import DeferredOptionalEmbeddingProvider

        descriptor = get_provider_descriptor(name)
        return DeferredOptionalEmbeddingProvider(
            provider_name=name,
            model_revision=kwargs.pop("model_revision", descriptor.model_revision),
            device=kwargs.pop("device", "cpu"),
            **kwargs,
        )
    get_provider_descriptor(name)
    raise AssertionError("unreachable")
