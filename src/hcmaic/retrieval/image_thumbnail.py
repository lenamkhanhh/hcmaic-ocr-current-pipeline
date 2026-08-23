"""Bounded, cached thumbnails for dense retrieval result galleries.

The canonical keyframe endpoint remains lossless/original.  This module only
creates a presentation-sized JPEG copy and never mutates the source artifact.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

DEFAULT_IMAGE_THUMBNAIL_WIDTH = 320
DEFAULT_IMAGE_THUMBNAIL_QUALITY = 78
MIN_IMAGE_THUMBNAIL_WIDTH = 160
MAX_IMAGE_THUMBNAIL_WIDTH = 640
MIN_IMAGE_THUMBNAIL_QUALITY = 40
MAX_IMAGE_THUMBNAIL_QUALITY = 95


class ImageThumbnailError(RuntimeError):
    """Raised when a source image cannot be converted to a thumbnail."""


def validate_thumbnail_options(width: int, quality: int) -> None:
    if not MIN_IMAGE_THUMBNAIL_WIDTH <= width <= MAX_IMAGE_THUMBNAIL_WIDTH:
        raise ValueError(
            "thumbnail width must be in "
            f"[{MIN_IMAGE_THUMBNAIL_WIDTH}, {MAX_IMAGE_THUMBNAIL_WIDTH}]"
        )
    if not MIN_IMAGE_THUMBNAIL_QUALITY <= quality <= MAX_IMAGE_THUMBNAIL_QUALITY:
        raise ValueError(
            "thumbnail quality must be in "
            f"[{MIN_IMAGE_THUMBNAIL_QUALITY}, {MAX_IMAGE_THUMBNAIL_QUALITY}]"
        )


def thumbnail_cache_path(
    cache_root: Path,
    *,
    cache_key: str,
    width: int = DEFAULT_IMAGE_THUMBNAIL_WIDTH,
    quality: int = DEFAULT_IMAGE_THUMBNAIL_QUALITY,
) -> Path:
    """Return the deterministic cache path without touching the source image."""

    validate_thumbnail_options(width, quality)
    digest = hashlib.sha256(
        f"{cache_key}|width={width}|quality={quality}".encode()
    ).hexdigest()
    return Path(cache_root).expanduser().resolve() / f"{digest}.jpg"


def build_image_thumbnail(
    source: Path,
    cache_root: Path,
    *,
    cache_key: str,
    width: int = DEFAULT_IMAGE_THUMBNAIL_WIDTH,
    quality: int = DEFAULT_IMAGE_THUMBNAIL_QUALITY,
) -> Path:
    """Return an atomically cached JPEG thumbnail for ``source``.

    The cache key includes source metadata supplied by the caller.  Temporary
    files are written beside the destination and atomically replaced so a
    concurrent request cannot observe a partially encoded image.
    """

    validate_thumbnail_options(width, quality)
    if not source.is_file():
        raise FileNotFoundError(source)

    target = thumbnail_cache_path(
        cache_root,
        cache_key=cache_key,
        width=width,
        quality=quality,
    )
    target_root = target.parent
    target_root.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return target

    temporary: Path | None = None
    try:
        with Image.open(source) as opened:
            # JPEG decoders can use a reduced DCT resolution before the final
            # LANCZOS resize. This avoids fully materializing a multi-megapixel
            # source when the UI only needs a few-hundred-pixel card.
            if getattr(opened, "format", None) == "JPEG":
                opened.draft("RGB", (width * 2, width * 2))
            image = ImageOps.exif_transpose(opened)
            image.thumbnail((width, width), Image.Resampling.LANCZOS)
            if image.mode != "RGB":
                image = image.convert("RGB")

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{target.stem}.",
                suffix=".part",
                dir=target_root,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            image.save(
                temporary,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
        os.replace(temporary, target)
        temporary = None
        return target
    except UnidentifiedImageError as exc:
        raise ImageThumbnailError(f"IMAGE_THUMBNAIL_SOURCE_INVALID: {source}") from exc
    except (OSError, ValueError) as exc:
        raise ImageThumbnailError(f"IMAGE_THUMBNAIL_BUILD_FAILED: {source}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
