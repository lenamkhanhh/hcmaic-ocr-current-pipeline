"""Allowlisted, manifest-backed resolution of cloud media for the local UI.

The retrieval indexes remain local.  This module only resolves the selected
JPEG/MP4 after a result is clicked.  A client never supplies a remote URL or a
Kaggle path; both are read from an immutable media manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

MediaKind = Literal["frame", "video"]
MediaFetcher = Callable[["MediaSpec", Path], None]
RangeFetcher = Callable[["MediaSpec", "VideoByteRange"], "RangeFetchResult"]
_KAGGLE_DATASET_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_KAGGLE_RANGE_ENDPOINT = "https://www.kaggle.com/api/v1/datasets/download"
_KAGGLE_RANGE_BACKEND = "kaggle_http_range"
_HUGGINGFACE_DATASET_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HUGGINGFACE_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HUGGINGFACE_HOST = "huggingface.co"
HUGGINGFACE_RANGE_BACKEND = "huggingface_http_range"


class MediaManifestError(ValueError):
    """The manifest is not a safe or internally consistent media contract."""


class MediaResolutionError(RuntimeError):
    """A declared media object could not be downloaded or verified."""


class MediaRangeRequestError(MediaResolutionError):
    """The client supplied a missing, invalid, or unsupported byte range."""

    def __init__(self, code: str, *, total: int | None = None) -> None:
        self.code = code
        self.total = total
        super().__init__(code)


class MediaRangeUnsupportedError(MediaResolutionError):
    """The remote backend cannot satisfy a byte-range request."""

    def __init__(self, code: str = "REMOTE_MEDIA_RANGE_UNSUPPORTED") -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class VideoByteRange:
    start: int
    end: int
    total: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class RangeFetchResult:
    status_code: int
    headers: Mapping[str, str]
    body: bytes
    remote_content_fingerprint: Mapping[str, str] | None = None


def parse_single_range(range_header: str | None, total: int) -> VideoByteRange:
    """Parse one RFC 7233 byte range without accepting multipart ranges."""

    if total < 1:
        raise ValueError("range total must be positive")
    if not range_header or not range_header.strip():
        raise MediaRangeRequestError("RANGE_HEADER_REQUIRED", total=total)
    value = range_header.strip()
    if not value.lower().startswith("bytes="):
        raise MediaRangeRequestError("RANGE_UNIT_UNSUPPORTED", total=total)
    spec = value[6:].strip()
    if "," in spec:
        raise MediaRangeRequestError("MULTI_RANGE_UNSUPPORTED", total=total)
    if spec.count("-") != 1:
        raise MediaRangeRequestError("RANGE_INVALID", total=total)
    start_text, end_text = (part.strip() for part in spec.split("-", 1))
    if not start_text and not end_text:
        raise MediaRangeRequestError("RANGE_INVALID", total=total)
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length < 1:
                raise ValueError
            start = max(0, total - suffix_length)
            end = total - 1
        else:
            start = int(start_text)
            if start < 0 or start >= total:
                raise ValueError
            end = total - 1 if not end_text else int(end_text)
            if end < start:
                raise ValueError
            end = min(end, total - 1)
    except ValueError as exc:
        raise MediaRangeRequestError("RANGE_NOT_SATISFIABLE", total=total) from exc
    return VideoByteRange(start, end, total)


def _safe_relative_path(value: Any, field: str) -> str:
    path = str(value or "").replace("\\", "/")
    if not path or path.startswith("/") or Path(path).drive:
        raise MediaManifestError(f"{field} must be a relative path")
    parts = [part for part in path.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise MediaManifestError(f"{field} must not contain traversal")
    return "/".join(parts)


def _sha256_value(value: Any) -> str | None:
    if value is None or value == "":
        return None
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise MediaManifestError("sha256 must be a 64-character hexadecimal digest")
    return digest


def _canonical_frame_uid(value: Any) -> str:
    frame_uid = str(value or "")
    try:
        video_id, source_idx = frame_uid.rsplit(":", 1)
        if not video_id or int(source_idx) < 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise MediaManifestError("frame_uid must be video_id:source_frame_idx") from exc
    return frame_uid


def _safe_http_url(value: Any, allowed_hosts: set[str]) -> str:
    url = str(value or "")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise MediaManifestError("http media URL must be an https URL without credentials")
    host = parsed.hostname.lower()
    if not allowed_hosts or host not in allowed_hosts:
        raise MediaManifestError(f"http media host is not allowlisted: {host}")
    if not parsed.path:
        raise MediaManifestError("http media URL must contain a path")
    # Keep the query in memory for signed URLs, but never write it to logs or
    # cache metadata.  The manifest itself is trusted/configured offline.
    return url


def _safe_huggingface_revision(value: Any) -> str:
    revision = str(value or "main").strip()
    if not _HUGGINGFACE_REVISION_RE.fullmatch(revision):
        raise MediaManifestError(
            "Hugging Face revision must be a simple branch, tag, or commit identifier"
        )
    return revision


def _huggingface_resolve_url(dataset: str, revision: str, path: str) -> str:
    return (
        f"https://{_HUGGINGFACE_HOST}/datasets/"
        f"{urllib.parse.quote(dataset, safe='/')}/resolve/"
        f"{urllib.parse.quote(revision, safe='')}/"
        f"{urllib.parse.quote(path, safe='/')}"
    )


def _is_huggingface_redirect_host(hostname: str | None) -> bool:
    host = str(hostname or "").lower().rstrip(".")
    return bool(
        host == _HUGGINGFACE_HOST
        or host == "www.huggingface.co"
        or host.endswith(".cdn.hf.co")
        or host in {"cdn-lfs.hf.co", "cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co"}
    )


@dataclass(frozen=True)
class MediaSpec:
    kind: MediaKind
    key: str
    backend: str
    path: str | None = None
    dataset: str | None = None
    revision: str | None = None
    url: str | None = None
    sha256: str | None = None
    bytes: int | None = None
    media_type: str | None = None
    range_capable: bool = False
    provenance_status: str = "ENGINEERING_PROXY"
    sha256_status: str | None = None
    source_path: str | None = None
    member_path: str | None = None
    canonical_source_path: str | None = None
    dataset_id: str | None = None
    media_info_id: str | None = None
    normalized_media_info_id: str | None = None
    range_probe_status: str | None = None
    range_probe_attempts: int | None = None
    source_manifest_id: str | None = None
    source_fingerprint: str | None = None
    source_fingerprint_semantics: str | None = None
    remote_content_fingerprint: dict[str, str] | None = None
    join_method: str | None = None

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        allowed_backends: set[str],
        allowed_http_hosts: set[str],
    ) -> MediaSpec:
        kind = str(row.get("kind", "")).lower()
        if kind not in {"frame", "video"}:
            raise MediaManifestError("media kind must be 'frame' or 'video'")
        key = (
            _canonical_frame_uid(row.get("frame_uid"))
            if kind == "frame"
            else str(row.get("video_id") or "").strip()
        )
        if not key:
            raise MediaManifestError(f"{kind} identity is required")
        backend = str(row.get("backend", "")).lower().strip()
        if backend not in allowed_backends:
            raise MediaManifestError(f"media backend is not allowed: {backend!r}")
        path = None
        dataset = None
        revision = None
        url = None
        if backend in {"kaggle", _KAGGLE_RANGE_BACKEND}:
            dataset = str(row.get("dataset") or "").strip()
            if not _KAGGLE_DATASET_RE.fullmatch(dataset):
                raise MediaManifestError("Kaggle dataset must be owner/slug")
            path = _safe_relative_path(row.get("path"), "Kaggle media path")
        elif backend == HUGGINGFACE_RANGE_BACKEND:
            dataset = str(row.get("dataset") or row.get("dataset_id") or "").strip()
            if not _HUGGINGFACE_DATASET_RE.fullmatch(dataset):
                raise MediaManifestError("Hugging Face dataset must be owner/name")
            path = _safe_relative_path(
                row.get("path") or row.get("member_path"),
                "Hugging Face media path",
            )
            revision = _safe_huggingface_revision(row.get("revision", "main"))
            url = _safe_http_url(
                _huggingface_resolve_url(dataset, revision, path),
                {_HUGGINGFACE_HOST},
            )
        elif backend == "http":
            url = _safe_http_url(row.get("url"), allowed_http_hosts)
        else:
            path = _safe_relative_path(row.get("path"), "media path")
        raw_bytes = row.get("bytes")
        try:
            size = None if raw_bytes is None else int(raw_bytes)
        except (TypeError, ValueError) as exc:
            raise MediaManifestError("media bytes must be an integer") from exc
        if size is not None and size < 1:
            raise MediaManifestError("media bytes must be positive")
        media_type = str(row.get("media_type") or "").strip().lower() or None
        if media_type is not None and "/" not in media_type:
            raise MediaManifestError("media_type must be a MIME type")
        raw_range_capable = row.get("range_capable", False)
        if type(raw_range_capable) is not bool:
            raise MediaManifestError("range_capable must be a boolean")
        digest = _sha256_value(row.get("sha256"))
        sha256_status = row.get("sha256_status")
        if sha256_status is not None and not isinstance(sha256_status, str):
            raise MediaManifestError("sha256_status must be a string")
        sha256_status = str(sha256_status or "").strip() or None
        optional_paths: dict[str, str | None] = {}
        for field in ("source_path", "member_path", "canonical_source_path"):
            value = row.get(field)
            optional_paths[field] = (
                None if value in {None, ""} else _safe_relative_path(value, field)
            )
        optional_text: dict[str, str | None] = {}
        for field in (
            "dataset_id",
            "media_info_id",
            "normalized_media_info_id",
            "source_manifest_id",
            "source_fingerprint_semantics",
            "join_method",
        ):
            value = row.get(field)
            if value is not None and not isinstance(value, str):
                raise MediaManifestError(f"{field} must be a string")
            optional_text[field] = str(value or "").strip() or None
        range_probe_status = row.get("range_probe_status")
        if range_probe_status is not None and not isinstance(range_probe_status, str):
            raise MediaManifestError("range_probe_status must be a string")
        range_probe_status = str(range_probe_status or "").strip() or None
        raw_probe_attempts = row.get("range_probe_attempts")
        if raw_probe_attempts is None:
            range_probe_attempts = None
        elif isinstance(raw_probe_attempts, bool):
            raise MediaManifestError("range_probe_attempts must be a non-negative integer")
        else:
            try:
                range_probe_attempts = int(raw_probe_attempts)
            except (TypeError, ValueError) as exc:
                raise MediaManifestError(
                    "range_probe_attempts must be a non-negative integer"
                ) from exc
            if range_probe_attempts < 0:
                raise MediaManifestError("range_probe_attempts must be a non-negative integer")
        source_fingerprint = _sha256_value(row.get("source_fingerprint"))
        remote_content_fingerprint = row.get("remote_content_fingerprint")
        if remote_content_fingerprint is None or remote_content_fingerprint == "":
            remote_content_fingerprint = None
        elif not isinstance(remote_content_fingerprint, dict):
            raise MediaManifestError("remote_content_fingerprint must be an object")
        else:
            remote_content_fingerprint = {
                str(key): str(value)
                for key, value in remote_content_fingerprint.items()
                if key in {"etag", "x-goog-hash"} and value not in {None, ""}
            }
            remote_content_fingerprint = remote_content_fingerprint or None
        raw_provenance = row.get("provenance_status")
        if raw_provenance is not None and not isinstance(raw_provenance, str):
            raise MediaManifestError("provenance_status must be a string")
        provenance_status = str(raw_provenance or "").strip().upper()
        if not provenance_status:
            provenance_status = (
                "ENGINEERING_PROXY_BYTES_ONLY"
                if kind == "video" and digest is None
                else "ENGINEERING_PROXY"
            )
        return cls(
            kind=kind,  # type: ignore[arg-type]
            key=key,
            backend=backend,
            path=path,
            dataset=dataset,
            revision=revision,
            url=url,
            sha256=digest,
            bytes=size,
            media_type=media_type,
            range_capable=raw_range_capable,
            provenance_status=provenance_status,
            sha256_status=sha256_status,
            source_path=optional_paths["source_path"],
            member_path=optional_paths["member_path"],
            canonical_source_path=optional_paths["canonical_source_path"],
            dataset_id=optional_text["dataset_id"],
            media_info_id=optional_text["media_info_id"],
            normalized_media_info_id=optional_text["normalized_media_info_id"],
            range_probe_status=range_probe_status,
            range_probe_attempts=range_probe_attempts,
            source_manifest_id=optional_text["source_manifest_id"],
            source_fingerprint=source_fingerprint,
            source_fingerprint_semantics=optional_text["source_fingerprint_semantics"],
            remote_content_fingerprint=remote_content_fingerprint,
            join_method=optional_text["join_method"],
        )

    @property
    def source_label(self) -> str:
        if self.backend in {"kaggle", _KAGGLE_RANGE_BACKEND}:
            return f"{self.dataset}:{self.path}"
        if self.backend == HUGGINGFACE_RANGE_BACKEND:
            return f"hf://{self.dataset}@{self.revision}:{self.path}"
        if self.backend == "http" and self.url:
            parsed = urllib.parse.urlsplit(self.url)
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return f"{self.backend}:{self.path}"

    @property
    def suffix(self) -> str:
        source = self.path or urllib.parse.urlsplit(self.url or "").path
        suffix = Path(source).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".mov"}:
            return suffix
        guessed = mimetypes.guess_extension(self.media_type or "")
        return guessed or (".jpg" if self.kind == "frame" else ".mp4")


class MediaManifest:
    """Read-only JSONL index keyed by canonical frame/video identity."""

    def __init__(
        self,
        path: Path,
        *,
        allowed_backends: set[str],
        allowed_http_hosts: set[str],
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise MediaManifestError(f"media manifest not found: {self.path}")
        self.frames: dict[str, MediaSpec] = {}
        self.videos: dict[str, MediaSpec] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise MediaManifestError(f"cannot read media manifest: {self.path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MediaManifestError(f"invalid JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise MediaManifestError(f"media row {line_number} must be an object")
            spec = MediaSpec.from_row(
                row,
                allowed_backends=allowed_backends,
                allowed_http_hosts=allowed_http_hosts,
            )
            target = self.frames if spec.kind == "frame" else self.videos
            if spec.key in target:
                raise MediaManifestError(f"duplicate {spec.kind} identity: {spec.key}")
            target[spec.key] = spec

    def get(self, kind: MediaKind, key: str) -> MediaSpec:
        table = self.frames if kind == "frame" else self.videos
        try:
            return table[key]
        except KeyError as exc:
            raise FileNotFoundError(f"no media manifest entry for {kind}:{key}") from exc

    def summary(self) -> dict[str, Any]:
        return {
            "manifest": str(self.path),
            "frame_count": len(self.frames),
            "video_count": len(self.videos),
            "range_capable_video_count": sum(spec.range_capable for spec in self.videos.values()),
            "provenance_statuses": dict(
                sorted(
                    Counter(
                        spec.provenance_status
                        for spec in [*self.frames.values(), *self.videos.values()]
                    ).items()
                )
            ),
            "backends": sorted(
                {spec.backend for spec in [*self.frames.values(), *self.videos.values()]}
            ),
        }


def write_kaggle_video_media_manifest(inventory_path: Path, output_path: Path) -> Path:
    """Adapt declared Kaggle video inventory rows to the on-demand resolver contract."""

    inventory = Path(inventory_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not inventory.is_file():
        raise MediaManifestError(f"canonical inventory not found: {inventory}")
    rows: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    for line_number, line in enumerate(inventory.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            source = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MediaManifestError(
                f"invalid canonical inventory JSON at line {line_number}"
            ) from exc
        if str(source.get("source_kind") or "").lower() != "kaggle":
            continue
        video_id = str(source.get("video_id") or "").strip()
        if not video_id or video_id in seen_video_ids:
            raise MediaManifestError(f"duplicate or missing Kaggle video_id at line {line_number}")
        dataset = str(
            source.get("dataset_id")
            or f"{source.get('dataset_owner')}/{source.get('dataset_slug')}"
        ).strip()
        member_path = _safe_relative_path(source.get("member_path"), "Kaggle video path")
        row = {
            "kind": "video",
            "video_id": video_id,
            "backend": "kaggle",
            "dataset": dataset,
            "path": member_path,
            "sha256": source.get("canonical_video_sha256"),
            "bytes": source.get("bytes"),
            "media_type": source.get("media_type") or "video/mp4",
            "range_capable": source.get("range_capable") is True,
            "provenance_status": source.get("provenance_status") or "ENGINEERING_PROXY",
            "source_manifest_id": source.get("source_manifest_id"),
        }
        MediaSpec.from_row(row, allowed_backends={"kaggle"}, allowed_http_hosts=set())
        rows.append(row)
        seen_video_ids.add(video_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    MediaManifest(output, allowed_backends={"kaggle"}, allowed_http_hosts=set())
    return output


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        raise MediaResolutionError("remote media redirects are disabled")


class _HuggingFaceRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow only redirects that remain within the Hugging Face delivery path."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != "https" or not _is_huggingface_redirect_host(parsed.hostname):
            raise MediaResolutionError("Hugging Face media redirect host is not allowlisted")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class RemoteMediaResolver:
    """Resolve manifest entries to verified local cache files."""

    def __init__(
        self,
        manifest_path: Path,
        cache_root: Path,
        *,
        allowed_backends: set[str] | None = None,
        allowed_http_hosts: set[str] | None = None,
        fetcher: MediaFetcher | None = None,
        range_fetcher: RangeFetcher | None = None,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
        max_range_bytes: int = 8 * 1024 * 1024,
        timeout_seconds: float = 60.0,
    ) -> None:
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if max_range_bytes < 1:
            raise ValueError("max_range_bytes must be positive")
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_range_bytes = max_range_bytes
        self.timeout_seconds = timeout_seconds
        self.fetcher = fetcher
        self.range_fetcher = range_fetcher
        self.manifest = MediaManifest(
            manifest_path,
            allowed_backends=allowed_backends
            or {"kaggle", "http", _KAGGLE_RANGE_BACKEND, HUGGINGFACE_RANGE_BACKEND},
            allowed_http_hosts={host.lower() for host in (allowed_http_hosts or set())},
        )
        # Manifest backends are immutable for the lifetime of this resolver.
        # Cache readiness once so result serialization does not rescan every
        # frame/video entry for each Top-K item.
        backends = {
            spec.backend
            for spec in [*self.manifest.frames.values(), *self.manifest.videos.values()]
        }
        self._backend_status_cache = {
            backend: self._backend_readiness(backend) for backend in sorted(backends)
        }
        self._locks: dict[Path, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @staticmethod
    def _backend_readiness(backend: str) -> dict[str, Any]:
        if backend == "kaggle":
            if importlib.util.find_spec("kagglehub") is None:
                return {
                    "backend": backend,
                    "ready": False,
                    "status": "UNAVAILABLE_DEPENDENCY_MISSING",
                    "reason": "optional dependency missing: kagglehub",
                }
            return {
                "backend": backend,
                "ready": True,
                "status": "READY",
                "reason": None,
                "dependency": "kagglehub",
            }
        if backend in {"http", _KAGGLE_RANGE_BACKEND, HUGGINGFACE_RANGE_BACKEND, "fixture"}:
            return {
                "backend": backend,
                "ready": True,
                "status": "READY",
                "reason": None,
            }
        return {
            "backend": backend,
            "ready": False,
            "status": "UNAVAILABLE_BACKEND_UNSUPPORTED",
            "reason": f"backend is not configured: {backend}",
        }

    def backend_status(self) -> dict[str, dict[str, Any]]:
        """Report dependency readiness without fetching remote media."""

        return {
            backend: dict(status) for backend, status in self._backend_status_cache.items()
        }

    def media_status(self, kind: MediaKind, key: str) -> dict[str, Any]:
        """Return fail-closed readiness for one manifest entry."""

        spec = self.manifest.get(kind, key)
        readiness = self._backend_status_cache[spec.backend]
        return {
            "kind": kind,
            "key": key,
            "backend": spec.backend,
            "bytes": spec.bytes,
            "media_type": spec.media_type,
            "range_capable": spec.range_capable,
            "available": bool(readiness["ready"]),
            "status": readiness["status"],
            "reason": readiness["reason"],
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
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.manifest.summary(),
            "cache_root": str(self.cache_root),
            "backend_status": self.backend_status(),
        }

    def resolve_frame(self, frame_uid: str) -> Path:
        return self.resolve("frame", frame_uid)

    def cache_identity(self, kind: MediaKind, key: str) -> str:
        """Return the immutable manifest identity used for local derivatives."""

        return self.manifest.get(kind, key).source_label

    def cached_media_path(self, kind: MediaKind, key: str) -> Path:
        """Return the verified-media cache location without downloading it."""

        return self._cache_paths(self.manifest.get(kind, key))[0]

    def media_url(self, kind: MediaKind, key: str) -> str | None:
        """Return a manifest-derived remote URL without fetching or logging it."""

        spec = self.manifest.get(kind, key)
        if not spec.range_capable or spec.backend not in {"http", HUGGINGFACE_RANGE_BACKEND}:
            return None
        return spec.url

    def resolve_video(self, video_id: str) -> Path:
        return self.resolve("video", video_id)

    def can_stream_video(self, video_id: str) -> bool:
        """Report whether the manifest/backend has a bounded range contract."""

        spec = self.manifest.get("video", video_id)
        return bool(
            spec.bytes
            and spec.range_capable
            and (
                spec.backend in {"http", _KAGGLE_RANGE_BACKEND}
                or spec.backend == HUGGINGFACE_RANGE_BACKEND
                or self.range_fetcher is not None
            )
        )

    def stream_video_range(self, video_id: str, range_header: str | None) -> RangeFetchResult:
        """Fetch exactly one bounded remote video range without touching the cache."""

        spec = self.manifest.get("video", video_id)
        if spec.bytes is None or not spec.range_capable:
            raise MediaRangeUnsupportedError(
                "REMOTE_MEDIA_RANGE_UNSUPPORTED: manifest has no trusted byte-range contract"
            )
        return self._fetch_validated_range(spec, range_header)

    def probe_video_range(self, video_id: str, range_header: str | None) -> RangeFetchResult:
        """Probe a declared Kaggle range backend without promoting its manifest row.

        This method is intentionally separate from ``stream_video_range``: a
        false ``range_capable`` value remains fail-closed for playback, while a
        bounded server-side probe can establish whether a versioned manifest
        may be generated with the capability enabled.
        """

        spec = self.manifest.get("video", video_id)
        if spec.backend != _KAGGLE_RANGE_BACKEND:
            raise MediaRangeUnsupportedError(
                "REMOTE_MEDIA_RANGE_UNSUPPORTED: probe backend is not Kaggle HTTP range"
            )
        if spec.bytes is None:
            raise MediaRangeUnsupportedError(
                "REMOTE_MEDIA_RANGE_UNSUPPORTED: manifest has no byte count"
            )
        return self._fetch_validated_range(spec, range_header)

    def _fetch_validated_range(self, spec: MediaSpec, range_header: str | None) -> RangeFetchResult:
        if spec.bytes is None:
            raise MediaRangeUnsupportedError(
                "REMOTE_MEDIA_RANGE_UNSUPPORTED: manifest has no byte count"
            )
        if range_header is None or not range_header.strip():
            byte_range = VideoByteRange(
                start=0,
                end=min(spec.bytes, self.max_range_bytes) - 1,
                total=spec.bytes,
            )
        else:
            byte_range = parse_single_range(range_header, spec.bytes)
            if byte_range.length > self.max_range_bytes and re.fullmatch(
                r"bytes=\s*\d+\s*-\s*", range_header.strip(), flags=re.IGNORECASE
            ):
                byte_range = VideoByteRange(
                    start=byte_range.start,
                    end=min(spec.bytes, byte_range.start + self.max_range_bytes) - 1,
                    total=spec.bytes,
                )
        if byte_range.length > self.max_range_bytes:
            raise MediaRangeRequestError("RANGE_TOO_LARGE", total=byte_range.total)
        if spec.backend == "http":
            fetched = self._fetch_http_range(spec, byte_range)
        elif spec.backend == _KAGGLE_RANGE_BACKEND:
            fetched = self._fetch_kaggle_http_range(spec, byte_range)
        elif spec.backend == HUGGINGFACE_RANGE_BACKEND:
            fetched = self._fetch_huggingface_http_range(spec, byte_range)
        elif self.range_fetcher is not None:
            fetched = self.range_fetcher(spec, byte_range)
        else:
            raise MediaRangeUnsupportedError()
        return self._validate_range_response(spec, byte_range, fetched)

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        wanted = name.lower()
        for key, value in headers.items():
            if str(key).lower() == wanted:
                return str(value)
        return None

    def _validate_range_response(
        self,
        spec: MediaSpec,
        byte_range: VideoByteRange,
        fetched: RangeFetchResult,
    ) -> RangeFetchResult:
        if fetched.status_code == 200:
            raise MediaRangeUnsupportedError("REMOTE_MEDIA_RANGE_UNSUPPORTED_HTTP_200")
        if fetched.status_code != 206:
            raise MediaResolutionError(f"REMOTE_MEDIA_RANGE_HTTP_STATUS_{fetched.status_code}")
        content_range = self._header(fetched.headers, "Content-Range")
        expected_range = f"bytes {byte_range.start}-{byte_range.end}/{byte_range.total}"
        if content_range != expected_range:
            raise MediaRangeUnsupportedError("REMOTE_MEDIA_RANGE_CONTENT_RANGE_MISMATCH")
        if len(fetched.body) != byte_range.length:
            raise MediaResolutionError("REMOTE_MEDIA_RANGE_LENGTH_MISMATCH")
        content_length = self._header(fetched.headers, "Content-Length")
        if content_length is not None and content_length != str(byte_range.length):
            raise MediaResolutionError("REMOTE_MEDIA_RANGE_LENGTH_MISMATCH")
        return RangeFetchResult(
            status_code=206,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": expected_range,
                "Content-Length": str(byte_range.length),
                "Content-Type": spec.media_type or "video/mp4",
            },
            body=fetched.body,
            remote_content_fingerprint=fetched.remote_content_fingerprint,
        )

    def resolve(self, kind: MediaKind, key: str) -> Path:
        spec = self.manifest.get(kind, key)
        target, metadata = self._cache_paths(spec)
        lock = self._lock_for(target)
        with lock:
            if self._cache_valid(spec, target, metadata):
                return target
            target.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
            try:
                self._download(spec, temporary)
                self._verify(spec, temporary)
                os.replace(temporary, target)
                self._write_metadata(spec, target, metadata)
            except MediaResolutionError:
                temporary.unlink(missing_ok=True)
                raise
            except (OSError, urllib.error.URLError) as exc:
                temporary.unlink(missing_ok=True)
                raise MediaResolutionError(f"media fetch failed for {spec.source_label}") from exc
            return target

    def _cache_paths(self, spec: MediaSpec) -> tuple[Path, Path]:
        identity = spec.sha256 or hashlib.sha256(spec.source_label.encode("utf-8")).hexdigest()
        target = self.cache_root / spec.kind / f"{identity}{spec.suffix}"
        return target, Path(f"{target}.json")

    def _lock_for(self, target: Path) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(target, threading.Lock())

    def _cache_valid(self, spec: MediaSpec, target: Path, metadata: Path) -> bool:
        if not target.is_file() or not metadata.is_file():
            return False
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            if payload.get("source") != spec.source_label:
                return False
            if int(payload.get("bytes", -1)) != target.stat().st_size:
                return False
            if spec.bytes is not None and target.stat().st_size != spec.bytes:
                return False
            return not spec.sha256 or payload.get("sha256") == spec.sha256
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _download(self, spec: MediaSpec, temporary: Path) -> None:
        if self.fetcher is not None:
            self.fetcher(spec, temporary)
            return
        if spec.backend == "kaggle":
            self._download_kaggle(spec, temporary)
            return
        if spec.backend == "http":
            self._download_http(spec, temporary)
            return
        if spec.backend == HUGGINGFACE_RANGE_BACKEND:
            self._download_huggingface(spec, temporary)
            return
        raise MediaResolutionError(f"no downloader configured for backend {spec.backend!r}")

    def _download_kaggle(self, spec: MediaSpec, temporary: Path) -> None:
        readiness = self._backend_readiness(spec.backend)
        if not readiness["ready"]:
            raise MediaResolutionError(f"{readiness['status']}: {readiness['reason']}")
        try:
            import kagglehub  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MediaResolutionError(
                "Kaggle media backend requires the optional 'kagglehub' package"
            ) from exc
        staging = Path(tempfile.mkdtemp(prefix="hcmaic-kaggle-media-"))
        try:
            downloaded = Path(
                kagglehub.dataset_download(
                    str(spec.dataset), path=str(spec.path), output_dir=str(staging)
                )
            )
            source = self._find_downloaded_file(downloaded, str(spec.path))
            self._copy_limited(source, temporary)
        except MediaResolutionError:
            raise
        except Exception as exc:  # noqa: BLE001 - third-party errors are normalized
            raise MediaResolutionError(
                f"Kaggle media fetch failed for {spec.source_label}"
            ) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    @staticmethod
    def _find_downloaded_file(downloaded: Path, remote_path: str) -> Path:
        if downloaded.is_file():
            return downloaded
        if not downloaded.is_dir():
            raise MediaResolutionError("Kaggle returned no local media file")
        direct = downloaded / Path(remote_path)
        if direct.is_file():
            return direct
        matches = list(downloaded.rglob(Path(remote_path).name))
        if len(matches) != 1:
            raise MediaResolutionError("Kaggle returned an ambiguous media path")
        return matches[0]

    @staticmethod
    def _remote_content_fingerprint(headers: Mapping[str, str]) -> dict[str, str]:
        fingerprint: dict[str, str] = {}
        for key, output_key in (("ETag", "etag"), ("X-Goog-Hash", "x-goog-hash")):
            value = RemoteMediaResolver._header(headers, key)
            if value:
                fingerprint[output_key] = value
        return fingerprint

    @staticmethod
    def _kaggle_range_url(spec: MediaSpec) -> str:
        if spec.backend != _KAGGLE_RANGE_BACKEND:
            raise MediaResolutionError("media spec is not a Kaggle range backend")
        if not spec.dataset or not spec.path:
            raise MediaRangeUnsupportedError("REMOTE_MEDIA_RANGE_UNSUPPORTED")
        owner, dataset = spec.dataset.split("/", 1)
        return (
            f"{_KAGGLE_RANGE_ENDPOINT}/{urllib.parse.quote(owner, safe='')}/"
            f"{urllib.parse.quote(dataset, safe='')}/{urllib.parse.quote(spec.path, safe='')}"
        )

    def _fetch_http_range(self, spec: MediaSpec, byte_range: VideoByteRange) -> RangeFetchResult:
        assert spec.url is not None
        return self._fetch_range_url(spec, byte_range, spec.url, follow_redirects=False)

    def _fetch_huggingface_http_range(
        self, spec: MediaSpec, byte_range: VideoByteRange
    ) -> RangeFetchResult:
        assert spec.url is not None
        return self._fetch_range_url(
            spec,
            byte_range,
            spec.url,
            follow_redirects=True,
            redirect_handler=_HuggingFaceRedirectHandler(),
        )

    def _fetch_kaggle_http_range(
        self, spec: MediaSpec, byte_range: VideoByteRange
    ) -> RangeFetchResult:
        return self._fetch_range_url(
            spec, byte_range, self._kaggle_range_url(spec), follow_redirects=True
        )

    def _fetch_range_url(
        self,
        spec: MediaSpec,
        byte_range: VideoByteRange,
        url: str,
        *,
        follow_redirects: bool,
        redirect_handler: urllib.request.HTTPRedirectHandler | None = None,
    ) -> RangeFetchResult:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": spec.media_type or "video/mp4",
                "Range": f"bytes={byte_range.start}-{byte_range.end}",
                "User-Agent": "hcmaic-local-media/1",
            },
        )
        if follow_redirects:
            opener = (
                urllib.request.build_opener(redirect_handler)
                if redirect_handler is not None
                else urllib.request.build_opener()
            )
        else:
            opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(request, timeout=self.timeout_seconds) as response:
                raw_status = getattr(response, "status", None)
                status_code = int(raw_status if raw_status is not None else response.getcode())
                headers = {str(key): str(value) for key, value in response.headers.items()}
                body = response.read(byte_range.length + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 200:
                raise MediaRangeUnsupportedError() from exc
            raise MediaResolutionError(
                f"REMOTE_MEDIA_UNAVAILABLE: range backend returned {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = str(getattr(exc, "reason", "")).lower()
            code = (
                "REMOTE_MEDIA_TIMEOUT"
                if "timed out" in reason
                else "REMOTE_MEDIA_REDIRECT_OR_TRANSPORT_ERROR"
            )
            raise MediaResolutionError(code) from exc
        except TimeoutError as exc:
            raise MediaResolutionError("REMOTE_MEDIA_TIMEOUT") from exc
        except OSError as exc:
            raise MediaResolutionError("REMOTE_MEDIA_REDIRECT_OR_TRANSPORT_ERROR") from exc
        return RangeFetchResult(
            status_code=status_code,
            headers=headers,
            body=body,
            remote_content_fingerprint=self._remote_content_fingerprint(headers),
        )

    def _download_huggingface(self, spec: MediaSpec, temporary: Path) -> None:
        self._download_http(
            spec,
            temporary,
            redirect_handler=_HuggingFaceRedirectHandler(),
        )

    def _download_http(
        self,
        spec: MediaSpec,
        temporary: Path,
        *,
        redirect_handler: urllib.request.HTTPRedirectHandler | None = None,
    ) -> None:
        assert spec.url is not None
        request = urllib.request.Request(
            spec.url,
            headers={
                "Accept": spec.media_type or "application/octet-stream",
                "User-Agent": "hcmaic-local-media/1",
            },
        )
        opener = (
            urllib.request.build_opener(redirect_handler)
            if redirect_handler is not None
            else urllib.request.build_opener(_NoRedirect)
        )
        try:
            with (
                opener.open(request, timeout=self.timeout_seconds) as response,
                temporary.open("wb") as output,
            ):
                declared = response.headers.get("Content-Length")
                if declared is not None:
                    try:
                        declared_bytes = int(declared)
                    except ValueError as exc:
                        raise MediaResolutionError(
                            "remote media returned an invalid content length"
                        ) from exc
                    if declared_bytes > self.max_bytes:
                        raise MediaResolutionError("remote media exceeds configured size limit")
                copied = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > self.max_bytes:
                        raise MediaResolutionError("remote media exceeds configured size limit")
                    output.write(chunk)
        except MediaResolutionError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            raise MediaResolutionError(f"HTTP media fetch failed for {spec.source_label}") from exc

    def _copy_limited(self, source: Path, target: Path) -> None:
        if not source.is_file():
            raise MediaResolutionError("remote media downloader returned no file")
        if source.stat().st_size > self.max_bytes:
            raise MediaResolutionError("remote media exceeds configured size limit")
        with source.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    def _verify(self, spec: MediaSpec, temporary: Path) -> None:
        if not temporary.is_file():
            raise MediaResolutionError(f"media downloader produced no file for {spec.source_label}")
        size = temporary.stat().st_size
        if size < 1 or size > self.max_bytes:
            raise MediaResolutionError("downloaded media has an invalid size")
        if spec.bytes is not None and size != spec.bytes:
            raise MediaResolutionError(
                f"media byte count mismatch for {spec.source_label}: "
                f"expected {spec.bytes}, got {size}"
            )
        if spec.sha256:
            digest = hashlib.sha256()
            with temporary.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != spec.sha256:
                raise MediaResolutionError(f"media sha256 mismatch for {spec.source_label}")

    @staticmethod
    def _write_metadata(spec: MediaSpec, target: Path, metadata: Path) -> None:
        payload = {
            "status": "CACHE_COMPLETE",
            "kind": spec.kind,
            "key": spec.key,
            "backend": spec.backend,
            "source": spec.source_label,
            "bytes": target.stat().st_size,
            "sha256": spec.sha256,
            "media_type": spec.media_type or mimetypes.guess_type(target.name)[0],
            "cached_at_unix": time.time(),
        }
        temporary = Path(f"{metadata}.{uuid.uuid4().hex}.part")
        try:
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, metadata)
        finally:
            temporary.unlink(missing_ok=True)
