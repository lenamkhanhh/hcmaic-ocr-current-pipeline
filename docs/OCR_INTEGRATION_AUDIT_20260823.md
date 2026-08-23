# HCMAIC OCR integration audit — 2026-08-23

## Scope and evidence boundary

This is a read-only audit of the OCR path from notebook builders through
artifact manifests, merge/index contracts, adapters, dual serve, fusion, and
tests. It inspected the local source-of-truth runbook, manager handoff, OCR
failure gates, current OCR runtime research, the active nested `system` source,
the 12 local OCR source manifests, and the local Elasticsearch snapshot
manifest.

No model was loaded, no GPU or Kaggle job was started, no keyframe/corpus was
downloaded, and the 4.3M-row merge was not materialized. This handoff contains
code and redacted metadata only; private data and weights stay outside the
repository.

## Current active path

The active artifact/runtime path is:

1. The notebook builder emits a dry-first contract. Dry mode is CPU-free and
   does not read input, install packages, load models, or infer. Execute mode
   is an explicit promotion and records input preflight, model gates,
   checkpoint progress, failures, provenance, and hashes.
2. The detector is DeepSolo ResNet-50 DSText official, one full-frame pass,
   score threshold `0.30`, no recall tiles. The recognizer is a Vietnamese
   PARSeq fine-tune, top-1 output at the recorded `32x128` input contract.
3. Each source shard produces `final_manifest.json`, frame detection status,
   crop-level OCR rows, and `failure_ledger.json`. Raw text, NFC text, folded
   text, confidence, boxes/polygons, detector/recognizer provenance, and source
   manifest hash are carried forward.
4. `ocr_merge.py` streams the sources through disk-backed uniqueness tables.
   It validates source status, quality/provenance, identity, partition sets,
   detector/recognizer signatures, optional declared hashes, row counts, and
   failure-ledger consistency. It writes canonical JSONL/Parquet plus an
   aggregate manifest without mutating input artifacts.
5. `ocr_elasticsearch.py` indexes one document per `crop_uid`, keeps the
   immutable `frame_uid`, searches NFC/folded/ngram fields, then collapses
   results to one result per frame. The evidence returned to the UI keeps the
   matched crop, confidence, box, polygon, and provenance.
6. `dual_visual.py` attaches OCR only when explicitly configured and policy
   allows the engineering proxy. Query time does not call a detector or GPU.
   `dual_app.py` exposes the ordinary search and staged-search routes. OCR is
   a separate channel and never concatenates object/OCR data into visual FAISS
   vectors.

## Local artifact reconciliation

| Item | Observed value |
|---|---:|
| Source manifest files | 12 (9 partitions + 3 full shards) |
| Declared/status frame rows | 146,121 / 146,121 |
| OCR crop rows | 4,320,089 |
| Recognition rows | 4,320,089 |
| Failure events | 4,878, all `NO_TEXT` |
| Read/inference/parse failure events | 0 / 0 / 0 |
| Unresolved failures | 0 |
| Source quality | `UNVALIDATED` / `UNVALIDATED_ON_HCMAIC` |
| Source provenance | `ENGINEERING_PROXY` |
| Detector revision | `dbadae995035246bad3376c7a44c015c69e9b313` |
| Recognizer revision | `76cc5f3cc6268457aac764653400fdff681f8271` |
| Detector score threshold | `0.30` |

Nine partition manifests declare `POSTFLIGHT_GREEN`. The three full manifests
(`s0003`, `s0004`, `s0005`) have no `postflight` object. The merge function's
default is fail-closed (`require_postflight_green=True`), while the merge
notebook preflight records these three as legacy missing-postflight sources.
This is an unresolved contract mismatch; it is not silently waived here.

The Elasticsearch snapshot manifest is successful but not exact-complete:

- source index: `hcmaic_ocr_v1`
- snapshot: `hcmaic_ocr_v1_20260821`
- source documents: `4,320,073`
- declared source rows: `4,320,089`
- count delta: `-16`
- snapshot manifest SHA-256:
  `94ebc4b611d95aeb66c17c3140ea09a3534f888d74e1ae6b71bfa4ecb86ccf30`
- source index manifest SHA-256:
  `9af3e5a926247b4edc338e4575ec54879f58994e3ef7622a5564e83d8baaa9af`
- source OCR manifest SHA-256:
  `18f38ec1e405147b6bf32619ae67c7e7f02c1368c1015bfb2054476949a32923`

The snapshot manifest has no `model_contract`. The dual runtime can normalize
the snapshot schema for read-only serving, but the resulting OCR channel
revision becomes `unknown+unknown`; that is an evidence/provenance gap.

## Fusion and identity contract

`retrieval/fusion.py` implements the paper-aligned stage contract:

1. normalize each channel independently with min-max into `[0, 1]`;
2. fuse the scores available for that frame with the harmonic mean;
3. do not treat an unqueried or unavailable channel as score zero;
4. keep `frame_uid` as the candidate identity and preserve channel evidence.

RRF with rank constant `60` remains available as an alternate late-fusion
method. OCR is queried independently and only enters the fusion map when it
returns valid frame-level hits.

## Legacy/conflicting OCR track

The older `channels/ocr.md` proposal describes PP-OCRv6/VietOCR canaries and a
PP-OCR-based production plan. That document is preserved under
`docs/reference/legacy_channels_ocr.md` for history, but it does not match the
observed full artifact or current runtime, which are DeepSolo/PARSeq. A future
PP-OCR experiment must use a new versioned artifact/manifest and a separate
adapter; it must not overwrite or relabel the current data.

The older `ocr_bm25.py` frame-level adapter is also retained in source for
compatibility. It is not interchangeable with the crop-level merged OCR
artifact or the Elasticsearch adapter. The boundary must remain explicit.

## Tests and checks actually run

From the source `system` checkout:

```text
uv run python -m pytest -q tests/test_ocr_merge.py tests/test_ocr_elasticsearch.py tests/test_ocr_kis_es_integration.py tests/test_ocr_merge_index_notebook_builder.py tests/test_dual_ocr_integration.py --tb=short
=> PASS (all selected tests)

uv run python -m compileall -q src/hcmaic/ingestion/ocr_merge.py src/hcmaic/retrieval/ocr_elasticsearch.py src/hcmaic/retrieval/ocr_text.py src/hcmaic/retrieval/fusion.py src/hcmaic/retrieval/dual_visual.py src/hcmaic/api/dual_app.py src/hcmaic/runtime/kis.py
=> PASS
```

The scoped Ruff check was also run. It reports seven pre-existing style issues:
four line-length findings in `ocr_merge.py`/`dual_visual.py` and three import
ordering findings in OCR tests. They are recorded rather than changed in the
shared checkout because this task is an audit/package handoff.

## Remaining blockers before a stronger claim

1. Resolve or formally version the missing `postflight` field on the three full
   source manifests, then run the bounded-memory merge with declared-hash and
   identity validation.
2. Reconcile the Elasticsearch `-16` count delta against the source crop
   ledger and preserve the explanation in a new snapshot manifest.
3. Add the detector/recognizer `model_contract` to the snapshot compatibility
   manifest so the serving status does not report `unknown+unknown`.
4. Run a separate full read-only artifact hash/row/identity audit; this audit
   checked the metadata and contract paths but intentionally did not read all
   4.3M OCR rows or build a new index.
5. Run approved OCR quality evaluation (CER/WER, Vietnamese diacritics, and
   retrieval qrels). Until then, all results remain engineering evidence and
   retrieval quality remains `UNVALIDATED`.

