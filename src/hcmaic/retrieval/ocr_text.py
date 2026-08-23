"""Unicode-safe text normalization shared by OCR indexing adapters."""

from __future__ import annotations

import unicodedata

_DIACRITIC_TRANSLATION: dict[str, str | int | None] = {"đ": "d", "Đ": "D"}


def normalize_ocr_nfc(value: object) -> str:
    """Return a trimmed NFC string without inventing text for null values."""
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def fold_ocr_text(value: object) -> str:
    """Return a search-folded form while keeping token boundaries.

    ``unicodedata`` does not decompose Vietnamese ``đ``.  Translate it
    explicitly before removing combining marks so names such as ``Đỗ`` remain
    searchable as ``do`` without changing the raw/NFC evidence.
    """
    text = normalize_ocr_nfc(value).translate(str.maketrans(_DIACRITIC_TRANSLATION))
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    chars = [char for char in decomposed if not unicodedata.combining(char)]
    folded = "".join(chars)
    return " ".join(folded.split())
