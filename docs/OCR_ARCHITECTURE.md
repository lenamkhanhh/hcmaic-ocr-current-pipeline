# OCR architecture and adapter boundaries

## Components

| Layer | File(s) | Contract |
|---|---|---|
| Notebook generation | `tools/build_ocr_dstext_parseq_*` | dry-first, explicit execution, image/model/failure gates |
| Merge ingestion | `src/hcmaic/ingestion/ocr_merge.py` | crop-level rows + frame status, disk-backed identity checks |
| Text normalization | `src/hcmaic/retrieval/ocr_text.py` | raw/NFC/folded are separate; no silent canonical label rewrite |
| Elasticsearch adapter | `src/hcmaic/retrieval/ocr_elasticsearch.py` | crop documents, query-time collapse to `frame_uid` |
| Legacy adapter | `src/hcmaic/retrieval/ocr_bm25.py` | older frame-level artifact only |
| Dual retrieval | `src/hcmaic/retrieval/dual_visual.py` | optional OCR channel, visual spaces remain separate |
| HTTP/stages | `src/hcmaic/api/dual_app.py` | `/search`, `/search/text`, `/search/stages` |
| KIS compatibility | `src/hcmaic/runtime/kis.py` | explicit engineering-proxy gate and strict merged manifest |
| Fusion | `src/hcmaic/retrieval/fusion.py` | min-max + harmonic mean, optional RRF(60) |

## Data flow

```text
final_manifest.json
      + detection_status.(jsonl|parquet)
      + ocr_lines.(jsonl|parquet)
      + failure_ledger.json
                |
                v
      merge_ocr_shards(...)
                |
                +-- ocr_manifest.json
                +-- frame_status.jsonl
                +-- ocr_lines.jsonl/parquet
                +-- failure_ledger.json/jsonl
                |
                v
      bulk_index_ocr(..., _id=crop_uid)
                |
                v
      search_ocr(..., collapse=frame_uid)
                |
                v
      ChannelHit(frame_uid, source_frame_idx, score, OCR evidence)
                |
                v
      DualVisualService -> harmonic_mean_fusion / reciprocal_rank_fusion
```

## Invariants

- `frame_uid` is exactly `video_id:source_frame_idx`.
- `crop_uid` identifies one OCR crop/line and is the Elasticsearch document id.
- `faiss_row` and local row positions are never join keys.
- OCR query code never loads detector/recognizer weights.
- OCR evidence is additive; visual indexes and vectors are not mutated.
- Missing/unqueried channels are omitted from fusion, not converted to zero.
- `ENGINEERING_PROXY` and `UNVALIDATED` are retained through API status and
  result provenance.

