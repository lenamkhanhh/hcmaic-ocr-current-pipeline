# HCMAIC OCR integration handoff

Shareable code snapshot for the HCMAIC OCR path, audited on 2026-08-23.
It contains the runtime adapter, crop-level Elasticsearch bridge, merger,
dual-serve wiring, notebook builders, contract tests, and the audit notes.

The Git tree intentionally does **not** contain keyframes, model weights,
logs, credentials, or the multi-gigabyte OCR runtime payload. The complete
current OCR handoff is published as split assets in release
`ocr-handoff-20260826-v1`; it contains no keyframes. The private Kaggle /
Hugging Face / Elasticsearch identifiers in the notebook builders remain
`REPLACE_WITH_*` placeholders.

See [`docs/OCR_RELEASE_HANDOFF_20260826.md`](docs/OCR_RELEASE_HANDOFF_20260826.md)
for the exact asset inventory, extraction commands, checksums, and known
evidence boundaries.

## Active architecture

```text
keyframe inventory
  -> dry-first DeepSolo DSText + Vietnamese PARSeq notebook
  -> final_manifest + detection_status + OCR lines + failure ledger
  -> disk-bounded crop/frame identity merger
  -> crop-level Elasticsearch index (crop_uid document id)
  -> collapse crop hits by frame_uid
  -> optional OCR channel in dual serve
  -> min-max per available channel + harmonic mean
  -> /search, /search/text, /search/stages
```

The durable join key is `frame_uid=video_id:source_frame_idx`. `faiss_row` is
never an identity key, and OCR remains a separate channel from visual FAISS.
Execution evidence is `ENGINEERING_PROXY`; retrieval quality is
`UNVALIDATED` until an approved qrels/CER/WER evaluation is available.

## Quick verification

Requires Python 3.11 and `uv`.

```powershell
uv sync --locked --extra elastic
uv run --extra elastic python -m pytest -q
uv run python -m compileall -q src/hcmaic/ingestion/ocr_merge.py src/hcmaic/retrieval/ocr_elasticsearch.py src/hcmaic/retrieval/ocr_text.py src/hcmaic/retrieval/fusion.py src/hcmaic/retrieval/dual_visual.py src/hcmaic/api/dual_app.py src/hcmaic/runtime/kis.py
```

The `elastic` extra is needed for the local Elasticsearch-client contract
test; no Elasticsearch server is contacted by this test suite.

No model, GPU, live Elasticsearch, Kaggle job, or full-corpus merge is needed
for these checks; the tests use synthetic fixtures and fake clients.

## Read in this order

1. `docs/OCR_INTEGRATION_AUDIT_20260823.md`
2. `docs/OCR_ARCHITECTURE.md`
3. `docs/OCR_ARTIFACT_CONTRACT.md`
4. `docs/OCR_NOTEBOOKS.md`
5. `src/hcmaic/retrieval/ocr_elasticsearch.py`
6. `src/hcmaic/ingestion/ocr_merge.py`
7. `src/hcmaic/retrieval/dual_visual.py` and `src/hcmaic/api/dual_app.py`

`docs/reference/legacy_channels_ocr.md` is preserved only as a clearly marked
legacy proposal. It describes a PP-OCR/VietOCR track and is not the active
DeepSolo/PARSeq runtime contract.

## Local configuration boundary

The runtime expects an already-built visual artifact bundle and, when OCR is
enabled, an Elasticsearch URL, index, and manifest path. Credentials are read
only from environment variables named by configuration; never put a token in
the URL, source tree, notebook, manifest, or command history.
