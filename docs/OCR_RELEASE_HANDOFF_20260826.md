# OCR release handoff — 2026-08-26

Release: [`ocr-handoff-20260826-v1`](https://github.com/lenamkhanhh/hcmaic-ocr-current-pipeline/releases/tag/ocr-handoff-20260826-v1)

This release carries the current OCR execution artifacts and generated full
source notebooks. It contains **no keyframes**: no JPG, PNG, MP4, image ZIP,
or video payload is included. Model weights and credentials are also excluded.

## Download and extract

Download every release asset into one directory. The numbered `.7z.00N` files
are parts of one archive; start extraction from `.001` and keep all parts in
the same directory:

```powershell
7z x .\hcmaic-ocr-merged-20260821.7z.001 -o.\ocr-merged
7z x .\hcmaic-ocr-source-shards-20260821.7z.001 -o.\ocr-source-shards
7z x .\hcmaic-ocr-es-snapshot-20260821.7z.001 -o.\ocr-es-snapshot
7z x .\hcmaic-ocr-generated-notebooks-20260820.7z.001 -o.\ocr-notebooks
```

Verify the downloaded parts against the release asset
`ocr_release_assets.sha256` before extraction.

## Asset inventory

| Asset group | Parts | Contents |
| --- | ---: | --- |
| `hcmaic-ocr-merged-20260821.7z.00N` | 4 | Full merged Parquet plus frame status, manifests, and failure ledger |
| `hcmaic-ocr-source-shards-20260821.7z.001` | 1 | 12 source outputs: 9 partition sources plus full `s0003`, `s0004`, `s0005` |
| `hcmaic-ocr-es-snapshot-20260821.7z.00N` | 3 | Elasticsearch snapshot repository `hcmaic_ocr_v1_20260821` |
| `hcmaic-ocr-generated-notebooks-20260820.7z.001` | 1 | 12 full notebooks used by the current source manifests |

The merged artifact reports 146,121 frames, 4,320,089 OCR rows, 4,878
resolved `NO_TEXT` events, and zero unresolved failures. Its provenance is
`ENGINEERING_PROXY`; OCR/retrieval quality remains `UNVALIDATED`.

## Known boundaries

- The three full source manifests (`s0003`–`s0005`) omit the `postflight`
  object; this is retained as a handoff gap rather than silently repaired.
- The Elasticsearch snapshot has a `-16` count delta: 4,320,073 source
  documents versus 4,320,089 declared OCR rows.
- The snapshot manifest does not carry the detector/recognizer model contract;
  serving from the snapshot alone therefore has an evidence/provenance gap.
- The public release makes OCR text and metadata downloadable to anyone with
  access to this public repository.
