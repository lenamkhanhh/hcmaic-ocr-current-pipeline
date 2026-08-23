# Notebook and builder map

The shareable builders are intentionally source-only. They generate notebooks
into ignored `_deliverables/` when run locally; no generated notebook is
checked in.

## Inference builders

- `build_ocr_dstext_parseq_shard0002_pilot_notebook.py`: 100-frame dry/pilot
  contract used to validate the full-frame DeepSolo -> PARSeq flow.
- `build_ocr_dstext_parseq_shard0005_full_notebook.py`: source-of-truth full
  shard builder. It gates inventory selection, image resolution/decoding,
  model assets, progress checkpoints, failure ledger, review bundle, and final
  artifact hashes. `--execute` is an explicit promotion.
- `build_ocr_dstext_parseq_shard0000_partition_notebooks.py` and the shard
  0001/0002 wrappers: deterministic three-way partition builders.
- `build_ocr_dstext_parseq_multi_shard_dry_notebooks.py`: dry/execute wrapper
  for the logical shard specs.

## Merge/index builders

- `build_ocr_merge_index_notebook.py`: CPU-only smoke/full merge-index
  notebook. It preflights all 12 source contracts, streams bounded rows, and
  emits the merged artifact plus Elasticsearch index manifest.
- `prepare_ocr_merge_input_dataset.py`: validates the structured transfer
  layout and rejects media files in the merge input.

## Safe usage

1. Replace every `REPLACE_WITH_*` identifier locally; do not commit private
   slugs or credentials.
2. Run builder tests first. Dry mode must remain GPU-free and must not read
   input or load a model.
3. Review source preflight and all manifest contracts before `--execute`.
4. Keep source OCR, merge output, ES snapshot, and model weights in private
   versioned storage outside this code repository.

