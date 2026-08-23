# Upstream provenance

> File này giữ thông tin provenance kỹ thuật của upstream. Hướng dẫn team và
> report trạng thái hiện tại nằm tại `README.md` và `TRANG_THAI_HE_THONG.md`.

## Origin

- Repository: https://github.com/hhlearntocode/SoftSignalsRetrievalSystems-AIC2025
- Pinned commit: `e2c52124e691fc2c71d187d8f587fbe1bcddc38b` (branch head "clean", 2025)
- License: MIT (see `LICENSE`, Copyright (c) 2025 Hoang) — preserved verbatim.
- Cloned: 2026-07-26 into `system/`, full git history retained.
- Local working branch: `hcmaic-2026-foundation`.

## Baseline state as inspected (not assumed)

Verified by reading the pinned sources on 2026-07-26:

- `app.py` (1084 lines): FastAPI app with CLIP text/image search, FAISS,
  SQLite metadata, surrounding-frame browsing, translation, batch similarity.
  - Hard-coded absolute paths: `D:/keyframe_embeddings_clip.db`,
    `D:/keyframe_faiss_clip.index`, `D:/keyframe_faiss_map_clip.json`,
    `D:/keyframes` (StaticFiles mount at import time — the app cannot even
    import on a machine without that directory).
  - `EMBEDDING_DIM = 1280`, model `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`
    (~10 GB class model; unsafe default for a 4 GB VRAM laptop).
  - Reads table `keyframe_embeddings`.
  - `googletrans` imported and enabled by default; **not** listed in
    `requirements.txt` (confirmed missing dependency).
  - Startup swallows exceptions: server reports "running" without model/index.
  - Global mutable state (`clip_model`, `faiss_index`, caches) throughout.
- `migrate_embeddings.py` (298 lines): builds SQLite from BTC-style
  `clip-features-32/*.npy`, `map-keyframes/*.csv` (columns `n, pts_time, fps,
  frame_idx`), `media-info/*.json`, `keyframes/<video>/<n>.jpg`.
  - Creates table `keyframes` — **mismatch** with `app.py`'s
    `keyframe_embeddings`.
  - `EMBEDDING_DIM = 512` — **mismatch** with `app.py`'s 1280.
  - Writes `image_retrieval.db` — **mismatch** with the `D:/...` path served.
  - Aligns embeddings to images by sorted file position, not by keyframe id.
- `backend/*.py`: standalone embedding scripts (CLIP bigG, Jina, SigLIP2),
  each with its own constants; not importable as a package.
- `src/`: scene extraction notebooks/scripts + TransNetV2 weights (30 MB
  `.pth` committed to git).
- `static/`: plain HTML/CSS/JS operator UI (~5,300 lines) coupled to the
  upstream endpoint shapes (`/search/text`, `/frame/{id}`,
  `/frames/surrounding/{id}`, `/video/{id}/frames`, `/images/...`).
- No tests, no CI, no lockfile, no evaluation harness, no manifest.

Conclusion: the upstream serving path cannot run against the artifacts its own
migration script produces (table name, dimension, and path all disagree). Its
value is the endpoint/UI design, the BTC dataset conventions, and the
FAISS/SQLite flow — all of which are reused here behind tested contracts.

## Reused

- BTC dataset conventions: `keyframe_mapping.csv` columns
  (`n, pts_time, fps, frame_idx`), per-video `media-info/*.json` metadata,
  `keyframes/<video_id>/<nnn>.jpg` layout (parser in
  `src/hcmaic/ingestion/`).
- Endpoint design: health, text search, frame metadata, surrounding
  frames/timeline, safe image serving (re-specified as the mission API
  contract in `src/hcmaic/api/`).
- Retrieval flow: normalized embeddings + inner-product index + id map +
  SQLite/catalog metadata join (re-implemented with explicit row→frame
  mapping tests in `src/hcmaic/indexing/`).
- Operator UI concepts: query box + top-K grid + detail pane + surrounding
  frames/timeline + query history (rebuilt compactly in `src/hcmaic/ui/`).
- FAISS index shape (`IndexFlatIP` over L2-normalized vectors) as the optional
  accelerated index provider.

## Replaced / refactored (with reasons)

- All absolute paths → explicit `--input/--artifacts` CLI arguments and an
  `AppConfig` object. Reason: unrunnable on any other machine.
- `laion/CLIP-ViT-bigG-14` default → optional `openai/clip-vit-base-patch32`
  (ViT-B/32-class) provider, CPU-first. Reason: 4 GB VRAM target; tests must
  not need weights.
- `googletrans` external translation → removed. Reason: undeclared dependency,
  network call on the query path, non-deterministic.
- Monolithic `app.py` globals → `src/hcmaic/` package with typed contracts,
  provider interfaces, and dependency-injected FastAPI state. Reason:
  untestable global mutable state.
- Silent startup failure → fail-fast artifact loading with actionable errors.
- Unversioned artifacts → `dataset_manifest.json` + `index_manifest.json`
  with hashes and versions.
- `requirements.txt` (unpinned, incomplete) → `pyproject.toml` + `uv.lock`.
  The original file is preserved in git history at the pinned commit.
- Upstream root scripts (`app.py`, `migrate_embeddings.py`, `backend/`,
  `src/` scene tools, `static/`) are moved unmodified to
  `upstream_reference/` to keep the runnable tree unambiguous while
  preserving the files and history.

## Deviations from upstream behavior

- No translation, no LLM keyword parser, no Jina/SigLIP backends, no
  TransNetV2 execution (out of mission scope).
- Search results carry `signal_scores`, `index_version`, and full
  `FrameRecord` mapping instead of raw SQLite rows.
- Deterministic tie handling in ranking (upstream relied on FAISS ordering).
- Tests, evaluator, fixtures, and manifests are new mission-owned code.
