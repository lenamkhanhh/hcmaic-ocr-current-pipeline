"""Build the dry-first full-shard-0005 DSText + PARSeq OCR notebook.

This builder reuses the validated shard-0002 pilot cells, but tightens the
full-shard gates: all inventory rows are selected, every selected image is
resolved and decoded before model setup, ZIP members are inspected, and the
postflight hashes JSONL/parquet/status artifacts separately.  The generated
draft is deliberately unarmed and GPU-free; ``--execute`` is an explicit
promotion step for a later, separately reviewed kernel.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import build_ocr_dstext_parseq_shard0002_pilot_notebook as pilot  # noqa: E402


ROOT = pilot.ROOT
KERNEL_ROOT = pilot.KERNEL_ROOT
DRY_NAME = "hcmaic-ocr-dstext-parseq-s0005-full-t4x2-20260820"
EXECUTE_NAME = "hcmaic-ocr-dstext-parseq-s0005-full-20260819"
KERNEL_OWNER = "REPLACE_WITH_KAGGLE_OWNER"
TARGET_SHARD_ID = "shard_0005"
TARGET_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0005-private"
INVENTORY_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-input-6shards-20260816-private"
WEIGHT_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-weights-20260819-private"
EXPECTED_FRAME_COUNT = 21_476


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def sha256_source_bundle(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        digest.update(resolved.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_source_revision(paths: list[Path]) -> tuple[str | None, bool | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        relative_paths = [str(path.resolve().relative_to(ROOT)) for path in paths]
        tracked = all(
            subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", relative_path],
                capture_output=True,
                text=True,
            ).returncode == 0
            for relative_path in relative_paths
        )
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all", "--", *relative_paths],
            text=True,
        )
        return commit, bool(status.strip()) or not tracked, tracked
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None, None, None


def make_config(execute: bool, run_name: str) -> str:
    source = pilot.CONFIG
    replacements = (
        ('TARGET_SHARD_ID = "shard_0002"', f'TARGET_SHARD_ID = "{TARGET_SHARD_ID}"'),
        (
            'TARGET_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0002-private"',
            f'TARGET_DATASET_SLUG = "{TARGET_DATASET_SLUG}"',
        ),
        ('MAX_FRAMES = 100', 'MAX_FRAMES = None'),
        ('SELECTION_SEED = "hcmaic-ocr-s0002-dstext-parseq-v1"',
         f'SELECTION_SEED = "hcmaic-ocr-{TARGET_SHARD_ID}-dstext-parseq-full-v1"'),
        ('OUT = Path("/kaggle/working/hcmaic-ocr-dstext-parseq-s0002-pilot-20260819")',
         f'OUT = Path("/kaggle/working/{run_name}")'),
        ('MAX_FRAMES = None', f'MAX_FRAMES = None\nEXPECTED_FRAME_COUNT = {EXPECTED_FRAME_COUNT}\nFULL_SHARD = True\nREVIEW_FRAME_COUNT = 12'),
        (
            'target_shard_id=TARGET_SHARD_ID, max_frames=MAX_FRAMES,',
            'target_shard_id=TARGET_SHARD_ID, max_frames=MAX_FRAMES,\n'
            '      expected_frame_count=EXPECTED_FRAME_COUNT, full_shard=FULL_SHARD,\n'
            '      review_frame_count=REVIEW_FRAME_COUNT,',
        ),
    )
    for old, new in replacements:
        source = replace_once(source, old, new, "config")
    if execute:
        source = replace_once(source, "EXECUTE_PIPELINE = False", "EXECUTE_PIPELINE = True", "execute flag")
    return source


FULL_SELECTION = r'''
if not EXECUTE_PIPELINE:
    print("DRY_REVIEW_ONLY: inventory and images not read")
else:
    import html
    import pandas as pd
    import shutil
    from PIL import Image

    inventory_paths = sorted(INPUT_ROOT.glob("**/keyframe_inventory.parquet"))
    jsonl_paths = sorted(INPUT_ROOT.glob("**/keyframe_inventory.jsonl"))
    preferred_parquet = [path for path in inventory_paths if "ocr-input-6shards" in str(path)]
    preferred_jsonl = [path for path in jsonl_paths if "ocr-input-6shards" in str(path)]
    if preferred_parquet:
        inventory_path = preferred_parquet[0]
        inventory = pd.read_parquet(inventory_path)
    elif preferred_jsonl:
        inventory_path = preferred_jsonl[0]
        inventory = pd.read_json(inventory_path, lines=True)
    else:
        raise FileNotFoundError("keyframe_inventory.parquet/jsonl not found under /kaggle/input")

    required = {"frame_uid", "video_id", "source_frame_idx", "timestamp_ms", "shot_id",
                "image_shard_id", "image_member"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"inventory missing columns: {missing}")
    pool = inventory.loc[inventory["image_shard_id"].astype(str) == TARGET_SHARD_ID].copy()
    duplicate_mask = pool["frame_uid"].astype(str).duplicated(keep=False)
    if bool(duplicate_mask.any()):
        examples = pool.loc[duplicate_mask, "frame_uid"].astype(str).head(10).tolist()
        raise ValueError(f"duplicate frame_uid in full-shard inventory: {examples}")
    if pool["frame_uid"].isna().any() or pool["image_member"].isna().any():
        raise ValueError("full-shard inventory contains null frame_uid/image_member")
    if len(pool) != EXPECTED_FRAME_COUNT:
        raise ValueError(f"shard pool count changed: {len(pool)} != {EXPECTED_FRAME_COUNT}")

    # Full-shard identity is deterministic and does not use faiss/parquet row order.
    selection = pool.sort_values(["frame_uid"]).reset_index(drop=True)
    expected_uid = selection.apply(lambda row: f"{row['video_id']}:{int(row['source_frame_idx'])}", axis=1)
    if not (selection["frame_uid"].astype(str).values == expected_uid.values).all():
        raise ValueError("frame_uid identity mismatch; expected video_id:source_frame_idx")
    if selection["frame_uid"].astype(str).duplicated().any():
        raise ValueError("duplicate canonical frame_uid after full-shard selection")
    selection_sha256 = hashlib.sha256("\n".join(selection["frame_uid"].astype(str)).encode()).hexdigest()
    inventory_sha256 = sha256_file(inventory_path)
    inventory_bytes = inventory_path.stat().st_size

    owner, slug = TARGET_DATASET_SLUG.split("/", 1)
    dataset_roots = [INPUT_ROOT / "datasets" / owner / slug, INPUT_ROOT / slug,
                     INPUT_ROOT / owner / slug, INPUT_ROOT / TARGET_DATASET_SLUG]
    dataset_roots.extend(sorted(INPUT_ROOT.glob("**/" + slug)))
    dataset_roots = [root for index, root in enumerate(dataset_roots)
                     if root.is_dir() and root not in dataset_roots[:index]]
    if not dataset_roots:
        raise FileNotFoundError(f"dataset root unresolved: {TARGET_DATASET_SLUG}")

    # Inspect ZIP indexes before any model setup. Direct images remain the preferred
    # path; ZIP-only inputs are supported by carrying archive/member identity to workers.
    zip_indexes = []
    zip_checks = []
    zip_handles = {}
    for root in dataset_roots:
        for archive_path in (root / "images.zip", root / "images" / "images.zip"):
            if archive_path.is_file():
                with zipfile.ZipFile(archive_path) as archive:
                    names = set(archive.namelist())
                zip_indexes.append((str(archive_path), names))
                zip_checks.append({"archive": str(archive_path), "member_count": len(names)})

    def member_candidates(member):
        member = str(member).lstrip("/")
        short = member.removeprefix("images/")
        return member, short, (member, f"images/{member}", short, f"images/{short}")

    def read_image_bytes(member):
        member, short, archive_candidates = member_candidates(member)
        direct_candidates = []
        for root in dataset_roots:
            direct_candidates.extend((root / "images" / short, root / member, root / short))
        for path in direct_candidates:
            if path.is_file():
                return path.read_bytes(), str(path), None, None
        for archive_path, names in zip_indexes:
            for archive_member in archive_candidates:
                if archive_member in names:
                    archive = zip_handles.get(archive_path)
                    if archive is None:
                        archive = zipfile.ZipFile(archive_path)
                        zip_handles[archive_path] = archive
                    raw = archive.read(archive_member)
                    return raw, archive_path, archive_path, archive_member
        raise FileNotFoundError(f"unresolved image: {member}; roots={dataset_roots}")

    preflight_path = OUT / "image_preflight.jsonl"
    preflight_failures = []
    sample_indexes = sorted({0, len(selection) // 2, len(selection) - 1})
    sample_checks = []
    resolved_paths = []
    preflight_rows = []
    preflight_path.unlink(missing_ok=True)
    for index, row in selection.iterrows():
        frame_uid = str(row["frame_uid"])
        try:
            raw, resolved, archive_path, archive_member = read_image_bytes(row["image_member"])
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            check = {"frame_uid": frame_uid, "image_member": str(row["image_member"]),
                     "resolved_path": resolved, "resolved_archive_path": archive_path,
                     "resolved_archive_member": archive_member, "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest()}
            preflight_rows.append(check)
            resolved_paths.append({"resolved_image_path": resolved if archive_path is None else None,
                                   "resolved_archive_path": archive_path,
                                   "resolved_archive_member": archive_member,
                                   "input_bytes": len(raw), "input_sha256": check["sha256"]})
            if index in sample_indexes:
                sample_checks.append(check)
        except Exception as exc:
            failure = {"failure_type": "READ_FAILED", "phase": "image_preflight",
                       "frame_uid": frame_uid, "crop_uid": None, "error": repr(exc),
                       "resolved": False}
            preflight_failures.append(failure)
            preflight_rows.append({**failure, "image_member": str(row["image_member"])})

    preflight_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in preflight_rows),
        encoding="utf-8",
    )
    for archive in zip_handles.values():
        archive.close()

    if preflight_failures:
        write_json(OUT / "failure_ledger.json", {
            "failure_count": len(preflight_failures), "unresolved_count": len(preflight_failures),
            "failures": preflight_failures, "counts_by_type": {"READ_FAILED": len(preflight_failures)},
        })
        raise RuntimeError(f"image preflight failed for {len(preflight_failures)} frame(s)")
    if len(resolved_paths) != len(selection):
        raise RuntimeError("image preflight resolver coverage mismatch")

    selection = pd.concat([selection.reset_index(drop=True), pd.DataFrame(resolved_paths)], axis=1)
    if REVIEW_FRAME_COUNT < 1:
        raise ValueError("REVIEW_FRAME_COUNT must be positive")
    review_indexes = sorted({
        int(round(index * (len(selection) - 1) / max(1, REVIEW_FRAME_COUNT - 1)))
        for index in range(min(REVIEW_FRAME_COUNT, len(selection)))
    })
    review_uids = [str(selection.iloc[index]["frame_uid"]) for index in review_indexes]
    review_selection_sha256 = hashlib.sha256("\n".join(review_uids).encode()).hexdigest()
    selection["review_flag"] = selection["frame_uid"].astype(str).isin(set(review_uids))
    write_json(OUT / "review_selection.json", {
        "status": "REVIEW_SELECTION_GREEN", "requested_count": REVIEW_FRAME_COUNT,
        "selected_count": len(review_uids), "frame_uids": review_uids,
        "review_selection_sha256": review_selection_sha256,
    })
    selection.to_parquet(OUT / "selection.parquet", index=False)
    preflight_sha256 = sha256_file(preflight_path)
    write_json(OUT / "selection_manifest.json", {
        "status": "SELECTION_GREEN", "quality_status": QUALITY_STATUS,
        "target_shard_id": TARGET_SHARD_ID, "target_dataset_slug": TARGET_DATASET_SLUG,
        "inventory_dataset_slug": INVENTORY_DATASET_SLUG, "inventory_path": str(inventory_path),
        "inventory_bytes": inventory_bytes, "inventory_sha256": inventory_sha256,
        "selection_seed": SELECTION_SEED, "full_shard": True,
        "expected_frame_count": EXPECTED_FRAME_COUNT, "frame_count": len(selection),
        "selection_sha256": selection_sha256, "image_preflight_sha256": preflight_sha256,
        "review_selection_sha256": review_selection_sha256, "review_frame_count": len(review_uids),
        "identity": "frame_uid=video_id:source_frame_idx", "faiss_row_used_as_identity": False,
        "immutable_keyframe_v1": True,
        "code_revision": {
            "pipeline_code_sha256": PIPELINE_CODE_SHA256,
            "notebook_config_sha256": NOTEBOOK_CONFIG_SHA256,
            "builder_source_sha256": BUILDER_SOURCE_SHA256,
            "builder_git_commit": BUILDER_GIT_COMMIT,
            "builder_git_dirty": BUILDER_GIT_DIRTY,
            "builder_git_tracked": BUILDER_GIT_TRACKED,
            "builder_source_files": BUILDER_SOURCE_FILES,
        },
    })
    write_json(OUT / "image_preflight.json", {
        "status": "IMAGE_PREFLIGHT_GREEN", "expected_count": len(selection),
        "resolved_count": len(resolved_paths), "failed_count": 0,
        "sample_checks": sample_checks, "zip_checks": zip_checks,
        "resolver_order": "datasets/<owner>/<slug>/images before short mounts",
        "preflight_jsonl": "image_preflight.jsonl", "preflight_sha256": preflight_sha256,
        "review_selection_sha256": review_selection_sha256,
        "io_policy": "preflight_buffered_jsonl_with_zip_handle_cache",
    })
    phase("IMAGE_PREFLIGHT_GREEN", frames=len(selection), resolved=len(resolved_paths),
          decoded=len(sample_checks), preflight_sha256=preflight_sha256,
          io_policy="preflight_buffered_jsonl_with_zip_handle_cache")
'''


FULL_RUNTIME = replace_once(
    pilot.RUNTIME,
    '''        "source_revisions": actual_revisions, "compatibility_patch_count": len(applied),
        "torch_cuda_nccl_abi_mutated": False,''',
    '''        "source_revisions": actual_revisions, "compatibility_patch_count": len(applied),
        "torch_cuda_nccl_abi_mutated": False,
        "code_revision": {
            "pipeline_code_sha256": PIPELINE_CODE_SHA256,
            "notebook_config_sha256": NOTEBOOK_CONFIG_SHA256,
            "builder_source_sha256": BUILDER_SOURCE_SHA256,
            "builder_git_commit": BUILDER_GIT_COMMIT,
            "builder_git_dirty": BUILDER_GIT_DIRTY,
            "builder_git_tracked": BUILDER_GIT_TRACKED,
            "builder_source_files": BUILDER_SOURCE_FILES,
        },''',
    "runtime code revision provenance",
)


def make_worker_source() -> str:
    source = pilot.WORKER_SOURCE
    source = replace_once(
        source,
        "import hashlib, io, json, os, sys, time, typing, unicodedata",
        "import hashlib, io, json, os, sys, time, typing, unicodedata, zipfile",
        "worker imports",
    )
    source = replace_once(
        source,
        "from PIL import Image, ImageDraw",
        "from PIL import Image, ImageDraw, ImageFont",
        "worker review font import",
    )
    source = replace_once(
        source,
        '''def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\\n")''',
        '''_JSONL_HANDLES = {}
_JSONL_COUNTS = {}
_JSONL_FLUSH_EVERY = 128

def append_jsonl(path, row):
    handle = _JSONL_HANDLES.get(path)
    if handle is None:
        handle = path.open("a", encoding="utf-8", buffering=1024 * 1024)
        _JSONL_HANDLES[path] = handle
        _JSONL_COUNTS[path] = 0
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")) + "\\n")
    _JSONL_COUNTS[path] += 1
    if _JSONL_COUNTS[path] % _JSONL_FLUSH_EVERY == 0:
        handle.flush()

def flush_jsonl_handles(close=False):
    for path, handle in list(_JSONL_HANDLES.items()):
        handle.flush()
        if close:
            handle.close()
            _JSONL_HANDLES.pop(path, None)
            _JSONL_COUNTS.pop(path, None)''',
        "worker buffered JSONL I/O",
    )
    source = replace_once(
        source,
        '''def heartbeat(phase, **fields):
    write_json(OUT / "progress.json", {"worker_id": WORKER_ID, "phase": phase, "time": time.time(), **fields})''',
        '''def heartbeat(phase, **fields):
    flush_jsonl_handles()
    write_json(OUT / "progress.json", {"worker_id": WORKER_ID, "phase": phase, "time": time.time(), **fields})''',
        "worker heartbeat flush",
    )
    source = replace_once(
        source,
        '''    try:\n        path = Path(row["resolved_image_path"])\n        image = Image.open(path).convert("RGB")\n    except Exception as exc:''',
        '''    try:\n        image = load_image_from_row(row)\n    except Exception as exc:''',
        "worker image resolver",
    )
    source = replace_once(
        source,
        '''                    "bbox": [int(value) for value in word["bbox"]],
                    "det_score":''',
        '''                    "bbox": [int(value) for value in word["bbox"]],
                    "input_bytes": int(row.get("input_bytes", 0)),
                    "input_sha256": str(row.get("input_sha256", "")),
                    "det_score":''',
        "worker input provenance",
    )
    source = replace_once(
        source,
        'crop_dir = OUT / "line_crops"',
        'crop_dir = OUT / "review_crops"',
        "worker bounded crop directory",
    )
    source = replace_once(
        source,
        '''            word["crop"].save(saved, quality=95)
            meta = {"crop_uid": crop_uid,''',
        '''            if bool(row.get("review_flag", False)):
                word["crop"].save(saved, quality=95)
                crop_path_value = str(saved)
            else:
                crop_path_value = None
            meta = {"crop_uid": crop_uid,''',
        "worker bounded crop persistence",
    )
    source = replace_once(
        source,
        '"crop_path": str(saved), "worker_id": WORKER_ID}',
        '"crop_path": crop_path_value, "worker_id": WORKER_ID}',
        "worker crop path provenance",
    )
    source = replace_once(
        source,
        '''heartbeat("MODEL_GREEN", frames_assigned=len(assigned_rows), frames_pending=len(rows), frames_resumed=len(terminal_uids))''',
        '''_ZIP_HANDLES = {}

def load_image_from_row(row):
    archive_path = row.get("resolved_archive_path")
    archive_member = row.get("resolved_archive_member")
    if archive_path and str(archive_path) != "nan":
        archive = _ZIP_HANDLES.get(str(archive_path))
        if archive is None:
            archive = zipfile.ZipFile(str(archive_path))
            _ZIP_HANDLES[str(archive_path)] = archive
        raw = archive.read(str(archive_member))
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(Path(row["resolved_image_path"])).convert("RGB")

def close_zip_handles():
    for archive in _ZIP_HANDLES.values():
        archive.close()
    _ZIP_HANDLES.clear()

recognizer_tokenizer = getattr(recognizer, "tokenizer", None)
recognizer_charset = getattr(recognizer_tokenizer, "classes", None)
if recognizer_charset is None:
    recognizer_charset = getattr(recognizer_tokenizer, "charset", None)
recognizer_charset = ([str(item) for item in recognizer_charset]
                      if recognizer_charset is not None else [])
recognizer_charset_sha256 = (
    hashlib.sha256(json.dumps(recognizer_charset, ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8")).hexdigest()
    if recognizer_charset else None
)
recognizer_contract = {
    "model": "PARSeq Vietnamese fine-tune",
    "revision": "76cc5f3cc6268457aac764653400fdff681f8271",
    "checkpoint_sha256": hashlib.sha256((ASSET_ROOT / "best-parseq.ckpt").read_bytes()).hexdigest(),
    "img_size": list(img_size),
    "charset_length": len(recognizer_charset),
    "charset_sha256": recognizer_charset_sha256,
    "charset_source": "recognizer.tokenizer.classes_or_charset",
    "candidate_policy": "top1_only",
    "parseq_batch_size": BATCH_SIZE,
}
heartbeat("MODEL_BOOTSTRAP_GREEN", recognizer_contract=recognizer_contract,
          frames_assigned=len(assigned_rows), frames_pending=len(rows), frames_resumed=len(terminal_uids))
if not assigned_rows:
    raise RuntimeError("worker received no assigned frames; cannot run model smoke")
smoke_row = assigned_rows[0]
smoke_image = load_image_from_row(smoke_row)
smoke_original = np.asarray(smoke_image)
smoke_transformed = augment.get_transform(smoke_original).apply_image(smoke_original)
smoke_tensor = torch.as_tensor(smoke_transformed.astype("float32").transpose(2, 0, 1), device="cuda:0")
with torch.inference_mode():
    detector([{"image": smoke_tensor, "height": smoke_image.height, "width": smoke_image.width}])
    recognize([smoke_image])
model_inference_smoke_status = "MODEL_INFERENCE_SMOKE_GREEN"
write_json(OUT / "model_gates.json", {
    "model_bootstrap_status": "MODEL_BOOTSTRAP_GREEN",
    "model_inference_smoke_status": model_inference_smoke_status,
    "frame_uid": str(smoke_row["frame_uid"]),
    "recognizer_contract": recognizer_contract,
})
heartbeat(model_inference_smoke_status, frame_uid=str(smoke_row["frame_uid"]))
heartbeat("MODEL_GREEN", frames_assigned=len(assigned_rows), frames_pending=len(rows),
          frames_resumed=len(terminal_uids), recognizer_contract=recognizer_contract)''',
        "worker model bootstrap and smoke gates",
    )
    source = replace_once(
        source,
        '''            nfc = unicodedata.normalize("NFC", raw_text)
            status = "EMPTY" if not nfc.strip() else ("LOW_CONF" if rec_score < 0.35 else "OK")
            append_jsonl(ocr_path, {**meta, "ocr_text_raw": raw_text, "ocr_text_nfc": nfc,
                         "ocr_text_folded": fold_text(nfc), "rec_score": rec_score,
                         "confidence_status": status, "recognizer_model": "PARSeq Vietnamese fine-tune",
                         "recognizer_revision": "76cc5f3cc6268457aac764653400fdff681f8271"})''',
        '''            nfc = unicodedata.normalize("NFC", raw_text)
            status = "EMPTY" if not nfc.strip() else ("LOW_CONF" if rec_score < 0.35 else "OK")
            meta.update({"ocr_text_raw": raw_text, "ocr_text_nfc": nfc,
                         "ocr_text_folded": fold_text(nfc), "rec_score": rec_score,
                         "candidate_policy": "top1_only", "confidence_status": status,
                         "recognizer_model": "PARSeq Vietnamese fine-tune",
                         "recognizer_revision": "76cc5f3cc6268457aac764653400fdff681f8271"})
            append_jsonl(ocr_path, {**meta, "ocr_candidates": [{"rank": 1, "text_raw": raw_text,
                                                                  "text_nfc": nfc, "score": rec_score}]})''',
        "worker candidate provenance",
    )
    source = replace_once(
        source,
        "overlay_dir = OUT / \"overlays\"",
        "review_overlay_dir = OUT / \"review_overlays\"",
        "worker review overlay directory",
    )
    source = replace_once(
        source,
        "overlay_dir.mkdir(exist_ok=True)",
        "review_overlay_dir.mkdir(exist_ok=True)",
        "worker review overlay mkdir",
    )
    source = replace_once(
        source,
        '''    if frame_index < 10:
        overlay = image.copy(); draw = ImageDraw.Draw(overlay)
        for meta in frame_meta:
            draw.polygon([tuple(point) for point in meta["polygon"]], outline="red", width=2)
        overlay.save(overlay_dir / f"{frame_uid.replace(':', '_')}.jpg", quality=90)''',
        '''    if bool(row.get("review_flag", False)):
        overlay = image.copy(); draw = ImageDraw.Draw(overlay); font = ImageFont.load_default()
        for meta in frame_meta:
            points = [tuple(point) for point in meta["polygon"]]
            draw.polygon(points, outline="red", width=2)
            x0 = min(point[0] for point in points)
            y0 = min(point[1] for point in points)
            label = f"{meta.get('word_index', '?')}: {meta.get('ocr_text_nfc', '')}"
            draw.text((x0, max(0, y0 - 14)), label[:120], fill="yellow",
                      stroke_width=2, stroke_fill="black", font=font)
        if not frame_meta:
            draw.text((8, 8), "NO_TEXT", fill="yellow", stroke_width=2,
                      stroke_fill="black", font=font)
        overlay.save(review_overlay_dir / f"{frame_uid.replace(':', '_')}.jpg", quality=90)''',
        "worker bounded review overlay",
    )
    source = replace_once(
        source,
        '''        append_jsonl(status_path, {"frame_uid": frame_uid, "status": "NO_TEXT", "line_count": 0})
        frame_ok += 1''',
        '''        append_jsonl(status_path, {"frame_uid": frame_uid, "status": "NO_TEXT", "line_count": 0})
        if bool(row.get("review_flag", False)):
            overlay = image.copy(); draw = ImageDraw.Draw(overlay); font = ImageFont.load_default()
            draw.text((8, 8), "NO_TEXT", fill="yellow", stroke_width=2,
                      stroke_fill="black", font=font)
            overlay.save(review_overlay_dir / f"{frame_uid.replace(':', '_')}.jpg", quality=90)
        frame_ok += 1''',
        "worker no-text review overlay",
    )
    source = replace_once(
        source,
        '''    "detector_threshold": THRESHOLD, "full_frame_only": True, "tile_pass_count": 0,
})''',
        '''    "detector_threshold": THRESHOLD, "full_frame_only": True, "tile_pass_count": 0,
    "review_image_count": len(list(review_overlay_dir.glob("*.jpg"))),
    "review_crop_count": len(list(crop_dir.glob("*.jpg"))),
    "crop_persistence_policy": "review_frames_only",
    "jsonl_io_policy": "buffered_handles_flush_128",
    "zip_io_policy": "per_worker_zip_handle_cache",
    "recognizer_contract": recognizer_contract,
    "model_bootstrap_status": "MODEL_BOOTSTRAP_GREEN",
    "model_inference_smoke_status": model_inference_smoke_status,
    "parseq_batch_size": BATCH_SIZE,
})''',
        "worker manifest provenance and review count",
    )
    source = replace_once(
        source,
        '''write_json(OUT / "failure_ledger.json", {"failures": failures, ''',
        '''flush_jsonl_handles(close=True)
close_zip_handles()
write_json(OUT / "failure_ledger.json", {"failures": failures, ''',
        "worker final I/O flush",
    )
    return source


FULL_LAUNCH = pilot.LAUNCH
FULL_LAUNCH = replace_once(
    FULL_LAUNCH,
    '''    resolved_rows = []\n    for row in selection.to_dict("records"):\n        _, resolved = read_image_bytes(row["image_member"])\n        resolved_rows.append({**row, "resolved_image_path": str(resolved)})''',
    '''    # Selection already contains the immutable, fully preflighted resolver paths.\n    resolved_rows = selection.to_dict("records")''',
    "launch resolver reuse",
)


FULL_POSTFLIGHT = r'''
if not EXECUTE_PIPELINE:
    phase("DRY_REVIEW_COMPLETE", note="No input read, install, download, CUDA init, model load, inference, or artifact promotion")
    print("DRY REVIEW COMPLETE")
else:
    # Postflight must remain bounded even when a shard produces millions of
    # word crops.  Do not call load_jsonl() here: it materializes every row in
    # Python at once and was the cause of the s0001 kernel death after workers
    # had completed inference.  JSONL is merged line-by-line, Parquet is
    # written in fixed row-groups, and UID validation is disk-backed SQLite.
    import collections
    import html
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    import shutil
    import sqlite3
    import time

    POSTFLIGHT_ROW_BATCH_SIZE = 50_000
    POSTFLIGHT_HEARTBEAT_SECONDS = 240
    POSTFLIGHT_INDEX_PATH = OUT / "postflight_index.sqlite"
    POSTFLIGHT_STATE_PATH = OUT / "postflight_state.json"
    postflight_started = time.time()
    postflight_heartbeat_state = {"last": postflight_started}

    def iter_jsonl(path):
        if not path.is_file():
            raise FileNotFoundError(f"postflight input missing: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc

    def write_jsonl(path, rows):
        """Atomically write the bounded review manifest without partial output."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False,
                                            sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def postflight_checkpoint(checkpoint_stage, **fields):
        checkpoint_fields = dict(fields)
        artifact_stage = checkpoint_fields.pop("stage", None)
        payload = {
            "status": "POSTFLIGHT_RUNNING", "stage": checkpoint_stage,
            "timestamp_epoch": time.time(), "elapsed_s": round(time.time() - postflight_started, 1),
            "row_batch_size": POSTFLIGHT_ROW_BATCH_SIZE,
            "io_policy": "jsonl_stream_to_parquet_row_groups",
            **checkpoint_fields,
        }
        if artifact_stage is not None:
            payload["artifact_stage"] = artifact_stage
        write_json(POSTFLIGHT_STATE_PATH, payload)

    def maybe_postflight_heartbeat(stage, rows):
        now = time.time()
        if now - postflight_heartbeat_state["last"] < POSTFLIGHT_HEARTBEAT_SECONDS:
            return
        postflight_heartbeat_state["last"] = now
        fields = {"stage": stage, "rows": rows,
                  "elapsed_s": round(now - postflight_started, 1),
                  "row_batch_size": POSTFLIGHT_ROW_BATCH_SIZE}
        phase("POSTFLIGHT_HEARTBEAT", **fields)
        postflight_checkpoint("HEARTBEAT", **fields)

    def as_text(value):
        return None if value is None else str(value)

    def as_int(value):
        return None if value is None else int(value)

    def as_float(value):
        return None if value is None else float(value)

    def as_polygon(value):
        if value is None:
            return None
        return [[float(point[0]), float(point[1])] for point in value]

    def as_bbox(value):
        if value is None:
            return None
        return [int(point) for point in value]

    base_fields = [
        pa.field("crop_uid", pa.string()), pa.field("frame_uid", pa.string()),
        pa.field("video_id", pa.string()), pa.field("source_frame_idx", pa.int64()),
        pa.field("timestamp_ms", pa.int64()), pa.field("shot_id", pa.string()),
        pa.field("line_index", pa.int64()), pa.field("detector_line_index", pa.int64()),
        pa.field("word_index", pa.int64()), pa.field("polygon", pa.list_(pa.list_(pa.float64()))),
        pa.field("detector_polygon", pa.list_(pa.list_(pa.float64()))),
        pa.field("bbox", pa.list_(pa.int64())), pa.field("input_bytes", pa.int64()),
        pa.field("input_sha256", pa.string()), pa.field("det_score", pa.float64()),
        pa.field("detector_model", pa.string()), pa.field("detector_revision", pa.string()),
        pa.field("crop_path", pa.string()), pa.field("worker_id", pa.int64()),
    ]
    crop_schema = pa.schema(base_fields)
    candidate_schema = pa.struct([
        pa.field("rank", pa.int64()), pa.field("text_raw", pa.string()),
        pa.field("text_nfc", pa.string()), pa.field("score", pa.float64()),
    ])
    line_schema = pa.schema(base_fields + [
        pa.field("ocr_text_raw", pa.string()), pa.field("ocr_text_nfc", pa.string()),
        pa.field("ocr_text_folded", pa.string()), pa.field("rec_score", pa.float64()),
        pa.field("candidate_policy", pa.string()), pa.field("confidence_status", pa.string()),
        pa.field("recognizer_model", pa.string()), pa.field("recognizer_revision", pa.string()),
        pa.field("ocr_candidates", pa.list_(candidate_schema)),
    ])
    status_schema = pa.schema([
        pa.field("frame_uid", pa.string()), pa.field("status", pa.string()),
        pa.field("line_count", pa.int64()), pa.field("crop_uids", pa.list_(pa.string())),
    ])

    def normalize_crop(row):
        return {
            "crop_uid": as_text(row.get("crop_uid")), "frame_uid": as_text(row.get("frame_uid")),
            "video_id": as_text(row.get("video_id")), "source_frame_idx": as_int(row.get("source_frame_idx")),
            "timestamp_ms": as_int(row.get("timestamp_ms")), "shot_id": as_text(row.get("shot_id")),
            "line_index": as_int(row.get("line_index")),
            "detector_line_index": as_int(row.get("detector_line_index")),
            "word_index": as_int(row.get("word_index")), "polygon": as_polygon(row.get("polygon")),
            "detector_polygon": as_polygon(row.get("detector_polygon")),
            "bbox": as_bbox(row.get("bbox")), "input_bytes": as_int(row.get("input_bytes")),
            "input_sha256": as_text(row.get("input_sha256")), "det_score": as_float(row.get("det_score")),
            "detector_model": as_text(row.get("detector_model")),
            "detector_revision": as_text(row.get("detector_revision")),
            "crop_path": as_text(row.get("crop_path")), "worker_id": as_int(row.get("worker_id")),
        }

    def normalize_line(row):
        normalized = normalize_crop(row)
        candidates = []
        for candidate in row.get("ocr_candidates") or []:
            candidates.append({
                "rank": as_int(candidate.get("rank")), "text_raw": as_text(candidate.get("text_raw")),
                "text_nfc": as_text(candidate.get("text_nfc")), "score": as_float(candidate.get("score")),
            })
        normalized.update({
            "ocr_text_raw": as_text(row.get("ocr_text_raw")),
            "ocr_text_nfc": as_text(row.get("ocr_text_nfc")),
            "ocr_text_folded": as_text(row.get("ocr_text_folded")),
            "rec_score": as_float(row.get("rec_score")),
            "candidate_policy": as_text(row.get("candidate_policy")),
            "confidence_status": as_text(row.get("confidence_status")),
            "recognizer_model": as_text(row.get("recognizer_model")),
            "recognizer_revision": as_text(row.get("recognizer_revision")),
            "ocr_candidates": candidates,
        })
        return normalized

    def normalize_status(row):
        return {
            "frame_uid": as_text(row.get("frame_uid")), "status": as_text(row.get("status")),
            "line_count": as_int(row.get("line_count", 0)),
            "crop_uids": [as_text(value) for value in (row.get("crop_uids") or [])],
        }

    def normalize_failure(row):
        return {
            "failure_type": as_text(row.get("failure_type")), "phase": as_text(row.get("phase")),
            "frame_uid": as_text(row.get("frame_uid")), "crop_uid": as_text(row.get("crop_uid")),
            "error": as_text(row.get("error")), "resolved": bool(row.get("resolved", False)),
        }

    def stream_artifact(label, source_paths, jsonl_path, parquet_path, schema, normalizer, on_batch=None):
        if not source_paths:
            raise FileNotFoundError(f"no worker JSONL sources for {label}")
        jsonl_path.unlink(missing_ok=True)
        parquet_path.unlink(missing_ok=True)
        row_count = 0
        batch = []
        writer = pq.ParquetWriter(str(parquet_path), schema=schema, compression="snappy")
        try:
            with jsonl_path.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
                for source_path in source_paths:
                    for row in iter_jsonl(source_path):
                        normalized = normalizer(row)
                        output.write(json.dumps(normalized, ensure_ascii=False,
                                                sort_keys=True, separators=(",", ":")) + "\n")
                        batch.append(normalized)
                        row_count += 1
                        if len(batch) >= POSTFLIGHT_ROW_BATCH_SIZE:
                            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                            if on_batch is not None:
                                on_batch(batch)
                            batch.clear()
                            maybe_postflight_heartbeat(label, row_count)
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    if on_batch is not None:
                        on_batch(batch)
                    batch.clear()
                    maybe_postflight_heartbeat(label, row_count)
        finally:
            writer.close()
        postflight_checkpoint("ARTIFACT_GREEN", artifact=label, rows=row_count)
        phase("POSTFLIGHT_ARTIFACT_GREEN", artifact=label, rows=row_count,
              elapsed_s=round(time.time() - postflight_started, 1))
        return row_count

    worker_manifests = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted(OUT.glob("worker_*/worker_manifest.json"))]
    worker_gate_paths = sorted(OUT.glob("worker_*/model_gates.json"))
    worker_gates = [json.loads(path.read_text(encoding="utf-8")) for path in worker_gate_paths]
    expected_uids = set(selection["frame_uid"].astype(str))
    review_records = [row for row in selection.to_dict("records") if bool(row.get("review_flag", False))]
    review_uids = {str(row["frame_uid"]) for row in review_records}
    review_status_by_uid = {}
    review_lines = collections.defaultdict(list)
    review_failures = collections.defaultdict(list)
    metrics = {
        "status_rows": 0, "status_duplicates": 0, "crop_duplicates": 0,
        "line_duplicates": 0, "failure_count": 0, "unresolved_count": 0,
        "failure_counts": {kind: 0 for kind in ("NO_TEXT", "READ_FAILED", "INFERENCE_FAILED", "PARSE_ERROR")},
        "preflight_rows": 0, "preflight_unresolved": 0,
    }
    status_uids = set()
    index_path = POSTFLIGHT_INDEX_PATH
    index_path.unlink(missing_ok=True)
    index_db = sqlite3.connect(str(index_path))
    index_db.execute("PRAGMA journal_mode=OFF")
    index_db.execute("PRAGMA synchronous=OFF")
    index_db.execute("PRAGMA temp_store=FILE")
    index_db.execute("CREATE TABLE crop_uids (uid TEXT PRIMARY KEY)")
    index_db.execute("CREATE TABLE line_uids (uid TEXT PRIMARY KEY)")
    index_db.commit()

    def insert_uid_batch(table_name, values):
        before = index_db.total_changes
        index_db.executemany(
            f"INSERT OR IGNORE INTO {table_name}(uid) VALUES (?)",
            ((str(value),) for value in values),
        )
        index_db.commit()
        return len(values) - (index_db.total_changes - before)

    def on_status_batch(batch):
        for row in batch:
            uid = str(row["frame_uid"])
            metrics["status_rows"] += 1
            if uid in status_uids:
                metrics["status_duplicates"] += 1
            status_uids.add(uid)
            if uid in review_uids:
                review_status_by_uid[uid] = row

    def on_crop_batch(batch):
        metrics["crop_duplicates"] += insert_uid_batch(
            "crop_uids", [row["crop_uid"] for row in batch]
        )

    def on_line_batch(batch):
        metrics["line_duplicates"] += insert_uid_batch(
            "line_uids", [row["crop_uid"] for row in batch]
        )
        for row in batch:
            if str(row["frame_uid"]) in review_uids:
                review_lines[str(row["frame_uid"])].append(row)

    def stream_failure_ledger(source_paths):
        if not source_paths:
            raise FileNotFoundError("no worker failure ledgers")
        target = OUT / "failure_ledger.json"
        target.unlink(missing_ok=True)
        first = True
        with target.open("w", encoding="utf-8", buffering=1024 * 1024) as output:
            output.write('{"failures":[')
            for source_path in source_paths:
                for row in iter_jsonl(source_path):
                    normalized = normalize_failure(row)
                    if not first:
                        output.write(",")
                    output.write(json.dumps(normalized, ensure_ascii=False,
                                            sort_keys=True, separators=(",", ":")))
                    first = False
                    metrics["failure_count"] += 1
                    failure_type = normalized["failure_type"]
                    if failure_type in metrics["failure_counts"]:
                        metrics["failure_counts"][failure_type] += 1
                    if not normalized["resolved"]:
                        metrics["unresolved_count"] += 1
                    if normalized["frame_uid"] in review_uids:
                        review_failures[normalized["frame_uid"]].append(normalized)
                    maybe_postflight_heartbeat("failure_ledger", metrics["failure_count"])
            output.write('],"failure_count":%d,"unresolved_count":%d,"counts_by_type":%s}\n' % (
                metrics["failure_count"], metrics["unresolved_count"],
                json.dumps(metrics["failure_counts"], sort_keys=True, separators=(",", ":")),
            ))
        postflight_checkpoint("ARTIFACT_GREEN", artifact="failure_ledger", rows=metrics["failure_count"])
        phase("POSTFLIGHT_ARTIFACT_GREEN", artifact="failure_ledger",
              rows=metrics["failure_count"], elapsed_s=round(time.time() - postflight_started, 1))

    postflight_checkpoint("START", expected_frames=len(expected_uids), workers=len(worker_manifests))
    phase("POSTFLIGHT_START", expected_frames=len(expected_uids), workers=len(worker_manifests),
          row_batch_size=POSTFLIGHT_ROW_BATCH_SIZE, io_policy="jsonl_stream_to_parquet_row_groups")
    status_count = stream_artifact(
        "detection_status", sorted(OUT.glob("worker_*/detection_status.jsonl")),
        OUT / "detection_status.jsonl", OUT / "detection_status.parquet", status_schema,
        normalize_status, on_status_batch,
    )
    crop_count = stream_artifact(
        "crop_inventory", sorted(OUT.glob("worker_*/crop_inventory.jsonl")),
        OUT / "crop_inventory.jsonl", OUT / "crop_inventory.parquet", crop_schema,
        normalize_crop, on_crop_batch,
    )
    recognition_count = stream_artifact(
        "ocr_lines", sorted(OUT.glob("worker_*/ocr_lines.jsonl")),
        OUT / "ocr_lines.jsonl", OUT / "ocr_lines.parquet", line_schema,
        normalize_line, on_line_batch,
    )
    stream_failure_ledger(sorted(OUT.glob("worker_*/failure_ledger.jsonl")))

    preflight_path = OUT / "image_preflight.jsonl"
    preflight_uids = set()
    if not preflight_path.is_file():
        preflight_missing = True
    else:
        preflight_missing = False
        for row in iter_jsonl(preflight_path):
            metrics["preflight_rows"] += 1
            uid = as_text(row.get("frame_uid"))
            if bool(row.get("resolved", True)):
                preflight_uids.add(uid)
            else:
                metrics["preflight_unresolved"] += 1
            maybe_postflight_heartbeat("image_preflight", metrics["preflight_rows"])

    unique_crop_count = int(index_db.execute("SELECT COUNT(*) FROM crop_uids").fetchone()[0])
    unique_line_count = int(index_db.execute("SELECT COUNT(*) FROM line_uids").fetchone()[0])
    missing_lines = int(index_db.execute(
        "SELECT COUNT(*) FROM crop_uids c LEFT JOIN line_uids l ON c.uid = l.uid WHERE l.uid IS NULL"
    ).fetchone()[0])
    missing_crops = int(index_db.execute(
        "SELECT COUNT(*) FROM line_uids l LEFT JOIN crop_uids c ON l.uid = c.uid WHERE c.uid IS NULL"
    ).fetchone()[0])
    index_db.close()
    index_path.unlink(missing_ok=True)

    problems = []
    model_gates = {
        "status": "MODEL_GATES_GREEN" if worker_gates else "MODEL_GATES_MISSING",
        "workers": worker_gates,
    }
    if not worker_manifests:
        problems.append({"reason": "worker_manifest_missing"})
    if not worker_gates:
        problems.append({"reason": "model_gate_missing"})
    if len(worker_gates) != len(worker_manifests):
        problems.append({"reason": "model_gate_worker_coverage_mismatch",
                         "worker_manifests": len(worker_manifests), "model_gates": len(worker_gates)})
    if any(item.get("status") != "WORKER_COMPLETE" for item in worker_manifests):
        problems.append({"reason": "worker_not_complete"})
    if any(item.get("model_bootstrap_status") != "MODEL_BOOTSTRAP_GREEN"
           or item.get("model_inference_smoke_status") != "MODEL_INFERENCE_SMOKE_GREEN"
           for item in worker_gates + worker_manifests):
        problems.append({"reason": "model_gate_mismatch"})
    if status_uids != expected_uids or metrics["status_rows"] != len(status_uids):
        problems.append({"reason": "frame_uid_coverage_or_duplicate_mismatch",
                         "expected": len(expected_uids), "actual": len(status_uids),
                         "status_rows": metrics["status_rows"]})
    if metrics["status_duplicates"]:
        problems.append({"reason": "duplicate_frame_uid", "count": metrics["status_duplicates"]})
    if metrics["crop_duplicates"]:
        problems.append({"reason": "duplicate_crop_uid", "count": metrics["crop_duplicates"]})
    if metrics["line_duplicates"]:
        problems.append({"reason": "duplicate_recognition_crop_uid", "count": metrics["line_duplicates"]})
    if metrics["unresolved_count"]:
        problems.append({"reason": "unresolved_failures", "count": metrics["unresolved_count"]})
    if unique_crop_count != unique_line_count or missing_lines or missing_crops:
        problems.append({"reason": "recognition_crop_coverage_mismatch",
                         "crops": unique_crop_count, "recognized": unique_line_count,
                         "missing_lines": missing_lines, "missing_crops": missing_crops})
    if preflight_missing:
        problems.append({"reason": "image_preflight_missing"})
    elif preflight_uids != expected_uids or metrics["preflight_rows"] != len(expected_uids):
        problems.append({"reason": "image_preflight_coverage_mismatch",
                         "expected": len(expected_uids), "actual": metrics["preflight_rows"],
                         "resolved": len(preflight_uids)})
    if sum(int(item.get("frames_assigned", 0)) for item in worker_manifests) != len(expected_uids):
        problems.append({"reason": "worker_frame_assignment_mismatch"})

    # Bounded visual QA bundle: the full shard remains in JSONL/parquet, while
    # only the deterministic review subset gets copied with bbox/text overlays.
    review_root = OUT / "review_overlays"
    review_root.mkdir(exist_ok=True)
    review_rows = []
    for review_row in review_records:
        frame_uid = str(review_row["frame_uid"])
        safe_uid = frame_uid.replace(":", "_")
        candidates = sorted(OUT.glob(f"worker_*/review_overlays/{safe_uid}.jpg"))
        image_rel = None
        image_sha256 = None
        review_image_status = "MISSING"
        if candidates:
            target = review_root / f"{safe_uid}.jpg"
            shutil.copyfile(candidates[0], target)
            image_rel = f"review_overlays/{target.name}"
            image_sha256 = sha256_file(target)
            review_image_status = "OK"
        frame_status = review_status_by_uid.get(frame_uid, {})
        frame_failures = review_failures.get(frame_uid, [])
        frame_lines = review_lines.get(frame_uid, [])
        review_rows.append({
            "frame_uid": frame_uid,
            "video_id": str(review_row["video_id"]),
            "source_frame_idx": int(review_row["source_frame_idx"]),
            "timestamp_ms": int(review_row["timestamp_ms"]),
            "shot_id": str(review_row["shot_id"]),
            "review_image": image_rel,
            "review_image_sha256": image_sha256,
            "status": review_image_status,
            "frame_status": frame_status.get("status"),
            "line_count": len(frame_lines),
            "failure_types": sorted({str(item.get("failure_type")) for item in frame_failures}),
            "ocr_lines": [{
                "crop_uid": str(item.get("crop_uid")),
                "text_raw": str(item.get("ocr_text_raw", "")),
                "text_nfc": str(item.get("ocr_text_nfc", "")),
                "text_folded": str(item.get("ocr_text_folded", "")),
                "rec_score": item.get("rec_score"),
                "confidence_status": item.get("confidence_status"),
            } for item in frame_lines],
        })
    write_jsonl(OUT / "review_manifest.jsonl", review_rows)
    review_missing = [row["frame_uid"] for row in review_rows if row["status"] != "OK"]
    if len(review_rows) != len(review_uids):
        problems.append({"reason": "review_selection_coverage_mismatch",
                         "expected": len(review_uids), "actual": len(review_rows)})
    if review_missing:
        problems.append({"reason": "review_overlay_missing", "frame_uids": review_missing})

    cards = []
    for item in review_rows:
        image_html = (f'<img src="{html.escape(str(item["review_image"]))}" '
                      f'alt="{html.escape(item["frame_uid"])}">'
                      if item["review_image"] else "<p>Review image missing.</p>")
        line_items = []
        for text_item in item["ocr_lines"]:
            line_items.append(
                "<li><code>" + html.escape(text_item["text_nfc"]) + "</code> "
                + html.escape(str(text_item.get("confidence_status"))) + " "
                + html.escape(str(text_item.get("rec_score"))) + "</li>"
            )
        lines_html = "".join(line_items) or "<li>NO_TEXT / no recognized crop</li>"
        cards.append(
            "<article><h2>" + html.escape(item["frame_uid"]) + "</h2>"
            "<p>status=" + html.escape(str(item.get("frame_status")))
            + " | review=" + html.escape(item["status"])
            + " | lines=" + html.escape(str(item["line_count"])) + "</p>"
            + image_html + "<ul>" + lines_html + "</ul></article>"
        )
    review_html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>HCMAIC OCR review</title>"
        "<style>body{font-family:system-ui;background:#111827;color:#e5e7eb;margin:24px}"
        "article{border:1px solid #374151;border-radius:8px;padding:16px;margin:16px 0}"
        "img{display:block;max-width:100%;height:auto;background:#000}code{color:#fde68a}"
        "</style></head><body><h1>HCMAIC OCR bounded review</h1>"
        f"<p>ENGINEERING_PROXY | quality={html.escape(QUALITY_STATUS)} | "
        f"detector threshold={html.escape(str(DETECTION_SCORE_THRESHOLD))} | "
        f"selected={len(review_rows)} / {len(selection)}</p>"
        + "".join(cards) + "</body></html>"
    )
    (OUT / "review_index.html").write_text(review_html, encoding="utf-8")
    write_json(OUT / "model_gates.json", model_gates)
    required_artifacts = (
        "selection.parquet", "selection_manifest.json", "review_selection.json",
        "image_preflight.json", "image_preflight.jsonl", "detection_status.jsonl",
        "detection_status.parquet", "crop_inventory.jsonl", "crop_inventory.parquet",
        "ocr_lines.jsonl", "ocr_lines.parquet", "failure_ledger.json", "model_gates.json",
        "review_manifest.jsonl", "review_index.html", "phase_status.jsonl",
        "runtime_manifest.json", "model_asset_manifest.json",
    )
    missing_artifacts = [name for name in required_artifacts if not (OUT / name).is_file()]
    if missing_artifacts:
        problems.append({"reason": "artifact_missing", "artifacts": missing_artifacts})
    status = "ENGINEERING_ARTIFACT_COMPLETE" if not problems else "ENGINEERING_ARTIFACT_COMPLETE_REPORT_FAILED"
    postflight_status = "POSTFLIGHT_GREEN" if not problems else "POSTFLIGHT_INCOMPLETE"
    postflight_counts = {
        "status_rows": status_count, "crop_rows": crop_count, "recognition_rows": recognition_count,
        "unique_crop_uids": unique_crop_count, "unique_line_uids": unique_line_count,
        "failure_count": metrics["failure_count"], "unresolved_failure_count": metrics["unresolved_count"],
    }
    postflight_checkpoint("VALIDATED", status=status, **postflight_counts)
    phase("POSTFLIGHT_GREEN" if not problems else "POSTFLIGHT_INCOMPLETE",
           status=status, frames=len(status_uids), crops=crop_count, recognized=recognition_count,
           failures=metrics["failure_count"], quality_status=QUALITY_STATUS,
           io_policy="jsonl_stream_to_parquet_row_groups",
           row_batch_size=POSTFLIGHT_ROW_BATCH_SIZE)
    artifact_hashes = {name: sha256_file(OUT / name) for name in required_artifacts}
    artifact_hashes.update({
        f"review_overlays/{path.name}": sha256_file(path)
        for path in sorted(review_root.glob("*.jpg"))
    })
    artifact_hashes.update({
        path.relative_to(OUT).as_posix(): sha256_file(path)
        for path in sorted(OUT.glob("worker_*/review_crops/*.jpg"))
    })
    manifest = {
        "status": status, "execution_status": "COMPLETE" if not problems else "COMPLETE_WITH_REPORT_FAILURE",
        "provenance_class": PROVENANCE_CLASS, "quality_status": QUALITY_STATUS,
        "human_review_required": True, "problems": problems,
        "code_revision": {
            "pipeline_code_sha256": PIPELINE_CODE_SHA256,
            "notebook_config_sha256": NOTEBOOK_CONFIG_SHA256,
            "builder_source_sha256": BUILDER_SOURCE_SHA256,
            "builder_git_commit": BUILDER_GIT_COMMIT,
            "builder_git_dirty": BUILDER_GIT_DIRTY,
            "builder_git_tracked": BUILDER_GIT_TRACKED,
            "builder_source_files": BUILDER_SOURCE_FILES,
            "definition": "pipeline hash covers ordered executable cells excluding generated config provenance literals",
        },
        "target_shard_id": TARGET_SHARD_ID, "target_dataset_slug": TARGET_DATASET_SLUG,
        "inventory_dataset_slug": INVENTORY_DATASET_SLUG, "full_shard": True,
        "expected_frame_count": EXPECTED_FRAME_COUNT, "selection_sha256": selection_sha256,
        "selection_count": len(selection), "frame_status_count": len(status_uids),
        "crop_count": crop_count, "recognition_count": recognition_count,
        "failure_count": metrics["failure_count"],
        "unresolved_failure_count": metrics["unresolved_count"],
        "inventory": {"path": str(inventory_path), "bytes": inventory_bytes,
                      "sha256": inventory_sha256},
        "runtime": {"requested_gpu_workers": REQUESTED_GPU_WORKERS,
                    "effective_gpu_workers": effective_gpu_workers,
                    "parseq_batch_size": PARSEQ_BATCH_SIZE,
                    "heartbeat_seconds": HEARTBEAT_SECONDS,
                    "full_frame_only": FULL_FRAME_ONLY, "recall_mode": RECALL_MODE,
                    "tile_pass_count": TILE_PASS_COUNT,
                    "jsonl_io_policy": "buffered_handles_flush_128",
                    "zip_io_policy": "per_worker_zip_handle_cache",
                    "crop_persistence_policy": "review_frames_only",
                    "postflight_io_policy": "jsonl_stream_to_parquet_row_groups",
                    "postflight_row_batch_size": POSTFLIGHT_ROW_BATCH_SIZE,
                    "postflight_index_validation": "sqlite_disk_backed"},
        "postflight": {"status": postflight_status,
                        "io_policy": "jsonl_stream_to_parquet_row_groups",
                        "row_batch_size": POSTFLIGHT_ROW_BATCH_SIZE,
                        "peak_rows_in_memory": POSTFLIGHT_ROW_BATCH_SIZE,
                        "counts": postflight_counts,
                        "checkpoint": "postflight_state.json",
                        "index_validation": "sqlite_disk_backed"},
        "model_gates": model_gates,
        "review": {"requested_count": REVIEW_FRAME_COUNT, "selected_count": len(review_rows),
                    "selection_sha256": review_selection_sha256,
                    "manifest": "review_manifest.jsonl", "html": "review_index.html",
                    "overlay_count": sum(item["status"] == "OK" for item in review_rows),
                    "crop_count": sum(item.get("review_crop_count", 0) for item in worker_manifests),
                    "crop_persistence_policy": "review_frames_only"},
        "detector": {"model": DEEPSOLO_MODEL_LABEL, "revision": DEEPSOLO_REVISION,
                     "weight_sha256": DEEPSOLO_WEIGHT_SHA256, "score_threshold": DETECTION_SCORE_THRESHOLD,
                     "full_frame_only": FULL_FRAME_ONLY, "recall_mode": RECALL_MODE,
                     "tile_pass_count": TILE_PASS_COUNT, "passes_per_frame": PASSES_PER_FRAME},
        "recognizer": {"model": "PARSeq Vietnamese fine-tune", "revision": PARSEQ_REVISION,
                       "checkpoint_sha256": PARSEQ_WEIGHT_SHA256,
                       "raw_and_normalized_preserved": True, "candidate_policy": "top1_only",
                       "parseq_batch_size": PARSEQ_BATCH_SIZE,
                       "worker_contracts": [item.get("recognizer_contract") for item in worker_manifests]},
        "parallelism": {"requested_gpu_workers": REQUESTED_GPU_WORKERS,
                        "effective_gpu_workers": effective_gpu_workers, "worker_manifests": worker_manifests},
        "identity": "frame_uid=video_id:source_frame_idx; detector-scoped crop_uid; faiss_row not identity",
        "artifact_hashes": artifact_hashes,
    }
    write_json(OUT / "final_manifest.json", manifest)
    postflight_checkpoint("COMPLETE", status=status, **postflight_counts)
'''


def make_notebook(execute: bool) -> Path:
    run_name = EXECUTE_NAME if execute else DRY_NAME
    out_dir = KERNEL_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_source = make_worker_source()
    config_base = make_config(execute, run_name)
    pipeline_sources = (
        FULL_SELECTION,
        FULL_RUNTIME,
        pilot.ASSETS,
        worker_source,
        FULL_LAUNCH,
        FULL_POSTFLIGHT,
    )
    pipeline_code_sha256 = sha256_text("\n---HCMAIC-CELL---\n".join(pipeline_sources))
    notebook_config_sha256 = sha256_text(config_base)
    builder_paths = [
        Path(pilot.__file__).resolve(),
        Path(__file__).resolve(),
        TOOLS_ROOT / "build_ocr_dstext_parseq_multi_shard_dry_notebooks.py",
    ]
    builder_source_sha256 = sha256_source_bundle(builder_paths)
    builder_git_commit, builder_git_dirty, builder_git_tracked = git_source_revision(builder_paths)
    builder_source_files = [str(path.resolve().relative_to(ROOT)).replace("\\", "/") for path in builder_paths]
    provenance_constants = "\n".join((
        f'PIPELINE_CODE_SHA256 = "{pipeline_code_sha256}"',
        f'NOTEBOOK_CONFIG_SHA256 = "{notebook_config_sha256}"',
        f'BUILDER_SOURCE_SHA256 = "{builder_source_sha256}"',
        f'BUILDER_GIT_COMMIT = {builder_git_commit!r}',
        f'BUILDER_GIT_DIRTY = {builder_git_dirty!r}',
        f'BUILDER_GIT_TRACKED = {builder_git_tracked!r}',
        f'BUILDER_SOURCE_FILES = {builder_source_files!r}',
    ))
    config = replace_once(
        config_base,
        "\n\nOUT.mkdir(parents=True, exist_ok=True)",
        f"\n\n{provenance_constants}\n\nOUT.mkdir(parents=True, exist_ok=True)",
        "config code revision constants",
    )
    config = replace_once(
        config,
        "      quality_status=QUALITY_STATUS)",
        "      quality_status=QUALITY_STATUS, pipeline_code_sha256=PIPELINE_CODE_SHA256,\n"
        "      notebook_config_sha256=NOTEBOOK_CONFIG_SHA256,\n"
        "      builder_source_sha256=BUILDER_SOURCE_SHA256)",
        "config code revision phase",
    )
    shard_label = TARGET_SHARD_ID.replace("shard_", "")
    cells = [
        pilot.cell(
            f"# {run_name}\n\n"
            f"Full OCR execution contract for all frames in keyframe shard {shard_label}: "
            "official DSText DeepSolo detector → word crops → Vietnamese PARSeq.\n\n"
            "- Dry draft is CPU-free and does not read input, install packages, load models, or infer.\n"
            f"- Full execution selects exactly the inventory rows for shard {shard_label} and gates the expected count {EXPECTED_FRAME_COUNT:,}.\n"
            "- Every selected image is resolved/decoded before model setup; first/middle/last probes and ZIP indexes are recorded.\n"
            "- Threshold `0.30`; one full-frame detector pass; no recall tiles.\n"
            "- Two concurrent GPU subprocesses when available; effective worker count is recorded, never assumed.\n"
            "- `frame_uid=video_id:source_frame_idx`; detector-scoped `crop_uid`; keyframe v1 remains immutable.\n"
            "- Artifacts are `ENGINEERING_PROXY`; OCR/retrieval quality remains `UNVALIDATED`.\n"
            "- Set `EXECUTE_PIPELINE=True` only after the dry kernel and cells 1–4 are reviewed.\n",
            "intro", "markdown"),
        pilot.cell(config, "config"),
        # Input preflight intentionally precedes runtime/model setup.
        pilot.cell(FULL_SELECTION, "selection-and-image-preflight"),
        pilot.cell(FULL_RUNTIME, "runtime-and-source-preflight"),
        pilot.cell(pilot.ASSETS, "model-asset-preflight"),
        pilot.cell("WORKER_SOURCE = " + repr(worker_source), "worker-source"),
        pilot.cell(FULL_LAUNCH, "parallel-inference"),
        pilot.cell(FULL_POSTFLIGHT, "checkpoint-manifest-postflight"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "hcmaic": {"status": "DRAFT_NOT_EXECUTED" if not execute else "EXECUTION_ARMED",
                       "quality_status": "UNVALIDATED", "identity": "frame_uid=video_id:source_frame_idx",
                       "target_shard_id": TARGET_SHARD_ID, "expected_frame_count": EXPECTED_FRAME_COUNT},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = out_dir / f"{run_name}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": f"{KERNEL_OWNER}/{run_name}", "title": run_name, "code_file": path.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": bool(execute), "enable_tpu": False, "enable_internet": True,
        # Kaggle's accelerator enum is NvidiaTeslaT4.  The requested worker
        # count is recorded in the notebook/runtime contract separately; do
        # not invent a GpuT4x2 machine_shape value that Kaggle silently ignores.
        "machine_shape": None if not execute else "NvidiaTeslaT4",
        "keywords": [], "dataset_sources": [INVENTORY_DATASET_SLUG, TARGET_DATASET_SLUG, WEIGHT_DATASET_SLUG],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def validate_notebook(path: Path, execute: bool) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(item.get("source", [])) for item in notebook["cells"] if item.get("cell_type") == "code"]
    for source in sources:
        ast.parse(source)
    for source in (FULL_SELECTION, FULL_RUNTIME, make_worker_source(), FULL_LAUNCH, FULL_POSTFLIGHT):
        ast.parse(source)
    code = "\n".join(sources)
    required = (
        f'TARGET_SHARD_ID = "{TARGET_SHARD_ID}"', 'MAX_FRAMES = None',
        f'EXPECTED_FRAME_COUNT = {EXPECTED_FRAME_COUNT}', 'FULL_SHARD = True',
        'REVIEW_FRAME_COUNT = 12', 'review_selection.json', 'review_manifest.jsonl',
        'review_index.html', 'review_overlays', 'review_frames_only', 'review_crops',
        'DETECTION_SCORE_THRESHOLD = 0.30', 'FULL_FRAME_ONLY = True', 'RECALL_MODE = False',
        'TILE_PASS_COUNT = 0', 'REQUESTED_GPU_WORKERS = 2', 'CUDA_VISIBLE_DEVICES',
        'keyframe_inventory.parquet', 'keyframe_inventory.jsonl', 'IMAGE_PREFLIGHT_GREEN',
        'image_preflight.jsonl', 'preflight_failures', 'failure_ledger',
        'NO_TEXT', 'READ_FAILED', 'INFERENCE_FAILED', 'PARSE_ERROR',
        'ENGINEERING_ARTIFACT_COMPLETE_REPORT_FAILED', 'final_manifest.json',
        'detection_status.parquet', 'crop_inventory.parquet', 'ocr_lines.parquet',
        'MODEL_BOOTSTRAP_GREEN', 'MODEL_INFERENCE_SMOKE_GREEN', 'model_gates.json',
        'ocr_candidates', 'candidate_policy', 'inventory_sha256', 'parseq_batch_size',
        'buffered_handles_flush_128', 'per_worker_zip_handle_cache',
        'PIPELINE_CODE_SHA256', 'NOTEBOOK_CONFIG_SHA256', 'BUILDER_SOURCE_SHA256',
        'BUILDER_GIT_COMMIT', 'BUILDER_GIT_DIRTY', 'BUILDER_GIT_TRACKED', '"code_revision": {',
        'frame_uid=video_id:source_frame_idx', 'DEEPSOLO_WEIGHT_SHA256', 'PARSEQ_WEIGHT_SHA256',
        'network fallback is intentionally disabled', 'WEIGHT_DATASET_SLUG',
    )
    missing = [token for token in required if token not in code]
    forbidden = [token for token in (
        "__RECOGNIZER_MODEL__", "PP-OCR", "VietOCR", "access_token", "KAGGLE_API_TOKEN",
        'faiss_row": str',
    ) if token in code]
    if missing or forbidden:
        raise ValueError({"missing": missing, "forbidden": forbidden})
    metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    expected_id = f"{KERNEL_OWNER}/{path.parent.name}"
    if metadata["id"] != expected_id or metadata["title"] != path.parent.name:
        raise ValueError("kernel metadata id/title is not canonical")
    if metadata["enable_gpu"] != execute:
        raise ValueError("dry/execute GPU metadata mismatch")
    if not execute and ("EXECUTE_PIPELINE = True" in code or metadata.get("machine_shape") is not None):
        raise ValueError("dry notebook is armed or requests a machine")
    return {"path": str(path), "cells": len(notebook["cells"]), "execute": execute,
            "enable_gpu": metadata["enable_gpu"], "machine_shape": metadata.get("machine_shape"),
            "expected_frame_count": EXPECTED_FRAME_COUNT, "source_chars": len(code)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="arm the separate full-execution kernel")
    args = parser.parse_args()
    path = make_notebook(args.execute)
    print(json.dumps(validate_notebook(path, args.execute), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

