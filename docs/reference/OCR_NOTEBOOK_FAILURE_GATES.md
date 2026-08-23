# OCR notebook failure gates (HCMAIC)

This note is an engineering checklist for future Kaggle OCR notebooks. It is
based on the OCR runs audited on 2026-08-17/18; it does not certify OCR or
retrieval quality.

## Failures already observed

| Failure | Impact | Required gate |
|---|---|---|
| Inventory paths were resolved without the `images/` payload prefix | All selected frames became `FileNotFoundError` after model setup | Resolve and decode deterministic first/middle/last samples; fail before model load |
| `PaddleOCR.predict(..., batch_size=...)` | Every frame failed at inference | Set batch size on the pipeline/module constructor; run one-call API smoke |
| CPU Paddle/PIR oneDNN regression | Model construction/inference failed with `ConvertPirAttribute2RuntimeAttribute` | Pin runtime; set `enable_mkldnn=False` only for the tested CPU contract |
| Paddle/PaddleX/LangChain and reader dependency drift | Import or model construction failed before OCR | Pin versions and preflight optional dependencies (`pypdfium2`, OpenCV contrib, pyclipper, compatibility imports) |
| VietOCR/Pillow bootstrap produced `PIL._util.is_directory` import errors, then 120/120 image decodes failed with `TypeError(...16 arguments (17 given))` | Replacing Pillow inside a live Papermill process mixed already-imported Python modules with newly replaced binary modules | Never uninstall/reinstall Pillow in-process; import/smoke the base Pillow first, install VietOCR with `--no-deps`, assert the Pillow version did not change, then decode a real input sample before model construction |
| Kaggle `code_file` could not import a local sibling helper | Notebook failed before pipeline start | Embed/package all helper code in the uploaded kernel |
| Kaggle kernel metadata title slug differed from the explicit kernel id | `kaggle kernels push` returned HTTP 400 before a usable draft existed | Keep metadata `title` slug-canonical (`title == id.split('/', 1)[1]`) and put the readable title in notebook Markdown |
| `kaggle kernels push` starts a kernel run as part of the push command | A draft upload can unexpectedly consume a runtime | Keep `EXECUTE_PIPELINE=False` in the pushed draft; inspect logs and require the run to be dry-only before asking for an execution run |
| PaddleX module constructors ignored the process-only MKLDNN flag | CPU detection entered the PIR/oneDNN path and failed on every frame | Apply `paddle.set_flags` after import and pass `enable_mkldnn=False` directly to detector/recognizer constructors; test one real canary |
| Generator left `__RECOGNIZER_MODEL__` in runtime/recognition code | Detection completed, then model construction failed with `UnknownModelError` | Replace placeholders in every generated cell and scan the final notebook before push |
| Final report error after inference | Raw artifacts could be mistaken for a quality result | Write manifests/checkpoints first; retain raw JSONL/crops and use `...REPORT_FAILED` status |
| PP-OCRv6 Vietnamese dictionary gap | High confidence but missing/wrong composed Vietnamese diacritics | Audit the actual recognizer charset; do not treat `lang=vi` metadata as proof |

## Required contracts

- Stable identity: `frame_uid=video_id:source_frame_idx`; never join on
  `faiss_row` or a local row number.
- Keep detector and recognizer outputs separate; preserve `polygon/bbox`,
  `frame_uid`, `shot_id`, `timestamp_ms`, raw text, NFC text, confidence and
  model contract for every line.
- Keep v1/keyframe inputs immutable. A/B recognizers must consume the same
  selection or the same hashed crop bundle.
- Use `EXECUTE_PIPELINE=False` in the draft. A Kaggle kernel push itself
  invokes a run; verify logs contain only `DRY_REVIEW_ONLY` before treating
  the upload as a draft, and never request an inference run automatically.
- A VietOCR notebook that also performs PP-OCR detection must bootstrap and
  verify the pinned Paddle/PaddleOCR/PaddleX detector contract too. Installing
  VietOCR alone is not a complete runtime contract.
- A green execution gate is not a retrieval-quality claim; use
  `quality_status=UNVALIDATED` until approved qrels/benchmark evidence exists.

## A/B OCR promotion gate

1. Same 120-frame selection and same detector/crops.
2. `latin_PP-OCRv5_mobile_rec` and VietOCR run in isolated notebooks when
   dependencies differ.
3. `selection_manifest`, `image_preflight`, `runtime_manifest`,
   `frame_status.jsonl`, `ocr_lines.jsonl`, `failure_ledger.json`, and a final
   manifest are all present and hash-linked.
4. Human-transcribe a fixed sample of Vietnamese lines and compare CER/WER,
   diacritic recall, empty-line rate, and failure count. Confidence is only an
   auxiliary signal.
5. Only then decide the full-corpus recognizer. Keep both raw outputs until
   the decision is recorded.
