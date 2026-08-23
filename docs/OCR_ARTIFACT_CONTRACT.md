# OCR artifact contract

## Source shard

Each source root contains:

```text
final_manifest.json
detection_status.jsonl or detection_status.parquet
ocr_lines.jsonl or ocr_lines.parquet
failure_ledger.json
```

The source manifest must declare the target shard, execution/status and
quality state, `ENGINEERING_PROXY` provenance, detector and recognizer model
contracts, the identity string containing
`frame_uid=video_id:source_frame_idx`, expected/status/crop/recognition counts,
selection hash, and hashes for the materialized artifacts when available.

`failure_ledger.json` is closed and carries `failure_count`,
`unresolved_count`, `counts_by_type`, and a `failures` list. Resolved
`NO_TEXT` rows are execution history, not missing OCR data; unresolved rows
must block a green merge.

## Canonical OCR row

The merger preserves, where supplied:

```json
{
  "crop_uid": "detector-scoped-id",
  "frame_uid": "video_id:source_frame_idx",
  "video_id": "video_id",
  "source_frame_idx": 123,
  "timestamp_ms": 4920,
  "ocr_text_raw": "Đỗ",
  "ocr_text_nfc": "Đỗ",
  "ocr_text_folded": "do",
  "det_score": 0.91,
  "rec_score": 0.88,
  "confidence_status": "OK",
  "bbox": [x1, y1, x2, y2],
  "polygon": [[x, y], [x, y], [x, y], [x, y]],
  "detector_model": "...",
  "detector_revision": "...",
  "recognizer_model": "...",
  "recognizer_revision": "...",
  "source_shard_id": "shard_0000",
  "source_manifest_sha256": "...",
  "quality_status": "UNVALIDATED_ON_HCMAIC",
  "execution_status": "ENGINEERING_ARTIFACT_COMPLETE"
}
```

The raw string remains available for audit. NFC and folded text are explicit
search projections, not replacements for the raw label. Crop-level duplicate
labels/instances are retained; frame-level collapse happens only at retrieval
query time.

## Merged artifact

`ocr_manifest.json` uses
`hcmaic-dstext-parseq-ocr-merged-v1`. It records source shard manifest hashes,
model contract, counts, failure counts, identity policy, runtime/batch policy,
quality/provenance, and output artifact hashes. `ocr_merge.py` uses SQLite
tables to reject duplicate `frame_uid` and `crop_uid` without retaining all
identities in RAM.

## Elasticsearch snapshot compatibility

The local snapshot uses
`hcmaic-ocr-elasticsearch-snapshot-v1`. It is accepted only by the dual
read-only runtime through an explicit compatibility flag. Index builders and
the legacy KIS loader remain strict about the merged manifest format.

A snapshot is not a substitute for the source merge manifest: it must retain
source hashes, source/declared counts, count delta, quality/provenance, and a
model contract before it can be called provenance-complete.

