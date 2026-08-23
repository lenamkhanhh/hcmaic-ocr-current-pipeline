"""Build a dry-first two-GPU DSText official + PARSeq VN OCR pilot.

The generated Kaggle notebook selects exactly 100 deterministic frames from
keyframe shard 0002.  Execution is disabled by default; the dry draft performs
no input reads, installs, downloads, model loads, or inference.
"""

from __future__ import annotations

import argparse
import ast
import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = ROOT / "_deliverables" / "_kaggle-kernels"
OUT_NAME = "hcmaic-ocr-dstext-parseq-s0002-pilot-20260819"
KERNEL_ID = f"REPLACE_WITH_KAGGLE_OWNER/{OUT_NAME}"
TARGET_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0002-private"
INVENTORY_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-input-6shards-20260816-private"
WEIGHT_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-weights-20260819-private"


def cell(source: str, cell_id: str, cell_type: str = "code") -> dict:
    payload = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": [line + "\n" for line in textwrap.dedent(source).strip("\n").splitlines()],
    }
    if cell_type == "code":
        payload.update({"execution_count": None, "outputs": []})
    return payload


CONFIG = r'''
from pathlib import Path
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import zipfile

EXECUTE_PIPELINE = False
TARGET_SHARD_ID = "shard_0002"
TARGET_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-keyframes-shard-0002-private"
INVENTORY_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-input-6shards-20260816-private"
WEIGHT_DATASET_SLUG = "REPLACE_WITH_KAGGLE_OWNER/hcmaic-ocr-dstext-parseq-weights-20260819-private"
MAX_FRAMES = 100
SELECTION_SEED = "hcmaic-ocr-s0002-dstext-parseq-v1"
DETECTION_SCORE_THRESHOLD = 0.30
FULL_FRAME_ONLY = True
RECALL_MODE = False
TILE_PASS_COUNT = 0
PASSES_PER_FRAME = 1
REQUESTED_GPU_WORKERS = 2
PARSEQ_BATCH_SIZE = 32
HEARTBEAT_SECONDS = 240
OUT = Path("/kaggle/working/hcmaic-ocr-dstext-parseq-s0002-pilot-20260819")
INPUT_ROOT = Path("/kaggle/input")
SOURCE_ROOT = Path("/kaggle/working/sources")
ASSET_ROOT = Path("/kaggle/working/model-assets")
QUALITY_STATUS = "UNVALIDATED"
PROVENANCE_CLASS = "ENGINEERING_PROXY"

DEEPSOLO_REPO = "https://github.com/ViTAE-Transformer/DeepSolo.git"
DEEPSOLO_REVISION = "dbadae995035246bad3376c7a44c015c69e9b313"
GOMATCHING_REPO = "https://github.com/Hxyz-123/GoMatching.git"
GOMATCHING_REVISION = "3f7f5dd4f962a03f2f61ccc6e753e3c001460264"
DEEPSOLO_WEIGHT_URL = "https://example.invalid/DEEPSOLO_WEIGHT_URL"
DEEPSOLO_WEIGHT_FILE = "DSText_res50_300queries_finetune.pth"
DEEPSOLO_WEIGHT_BYTES = 176_152_379
DEEPSOLO_WEIGHT_SHA256 = "d48cd9212573b544d2a9503c65f2e4a75c80f598dbf81eb739f6dc63d3d27e0c"
DEEPSOLO_MODEL_LABEL = "DeepSolo ResNet-50 DSText official"

PARSEQ_REPO = "https://github.com/lynguyenminh/vietnamese-scenetext-detection-recognition.git"
PARSEQ_REVISION = "76cc5f3cc6268457aac764653400fdff681f8271"
PARSEQ_DRIVE_ID = "REPLACE_WITH_PARSEQ_MODEL_ID"
PARSEQ_WEIGHT_FILE = "best-parseq.ckpt"
PARSEQ_WEIGHT_BYTES = 287_344_161
PARSEQ_WEIGHT_SHA256 = "8089b13c5ad115a96a608c6401eaab36b081393ea0a8323537b29a2dc80168f5"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase(name, **fields):
    row = {"phase": name, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
    append_jsonl(OUT / "phase_status.jsonl", row)
    print("[PHASE] " + json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


OUT.mkdir(parents=True, exist_ok=True)
phase("CONFIG_READY" if EXECUTE_PIPELINE else "DRY_REVIEW_CONFIG",
      target_shard_id=TARGET_SHARD_ID, max_frames=MAX_FRAMES,
      detection_score_threshold=DETECTION_SCORE_THRESHOLD,
      full_frame_only=FULL_FRAME_ONLY, recall_mode=RECALL_MODE,
      tile_pass_count=TILE_PASS_COUNT, requested_gpu_workers=REQUESTED_GPU_WORKERS,
      quality_status=QUALITY_STATUS)
'''


RUNTIME = r'''
if not EXECUTE_PIPELINE:
    print("DRY_REVIEW_ONLY: no package install, source clone, CUDA initialization, or model load")
else:
    import importlib.metadata as importlib_metadata
    import importlib.util
    import torch

    protected_names = ("torch", "torchvision", "numpy", "scipy", "Pillow", "nvidia-nccl-cu12")
    def version_or_none(name):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None

    protected_before = {name: version_or_none(name) for name in protected_names}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; this execution notebook requires at least one GPU")
    gpu_count = int(torch.cuda.device_count())
    effective_gpu_workers = min(REQUESTED_GPU_WORKERS, gpu_count)
    if effective_gpu_workers < 1:
        raise RuntimeError(f"No usable GPU worker; device_count={gpu_count}")
    if gpu_count < REQUESTED_GPU_WORKERS:
        phase("DUAL_GPU_UNAVAILABLE_SINGLE_GPU_FALLBACK", requested=REQUESTED_GPU_WORKERS, available=gpu_count)

    optional = {
        "termcolor": "termcolor==2.4.0", "yacs": "yacs==0.1.8",
        "fvcore": "fvcore==0.1.5.post20221221", "iopath": "iopath==0.1.10",
        "portalocker": "portalocker==2.10.1", "pycocotools": "pycocotools==2.0.7",
        "omegaconf": "omegaconf==2.3.0", "pytorch-lightning": "pytorch-lightning==2.5.3",
        "timm": "timm==0.6.13", "gdown": "gdown==5.2.0",
    }
    for package_name, requirement in optional.items():
        if version_or_none(package_name) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", requirement])

    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    repos = (
        ("DeepSolo", DEEPSOLO_REPO, DEEPSOLO_REVISION),
        ("GoMatching", GOMATCHING_REPO, GOMATCHING_REVISION),
        ("parseq-vn", PARSEQ_REPO, PARSEQ_REVISION),
    )
    actual_revisions = {}
    for name, url, revision in repos:
        destination = SOURCE_ROOT / name
        if not destination.is_dir():
            subprocess.check_call(["git", "clone", url, str(destination)])
        subprocess.check_call(["git", "-C", str(destination), "checkout", "--detach", revision])
        actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"source revision mismatch for {name}: {actual} != {revision}")
        actual_revisions[name] = actual

    deepsolo_plus = SOURCE_ROOT / "DeepSolo" / "DeepSolo++"
    detectron2_root = deepsolo_plus / "detectron2"
    gomatching_root = SOURCE_ROOT / "GoMatching"
    adet_cuda = gomatching_root / "third_party" / "adet" / "layers" / "csrc" / "DeformAttn" / "ms_deform_attn_cuda.cu"
    source = adet_cuda.read_text(encoding="utf-8")
    replacements = {
        'AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda"':
            'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda"',
        'AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda"':
            'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda"',
    }
    applied = []
    for old, new in replacements.items():
        if old in source:
            source = source.replace(old, new)
            applied.append(old)
        elif new not in source:
            raise RuntimeError("Unexpected AdelaiDet source; compatibility patch gate failed")
    adet_cuda.write_text(source, encoding="utf-8")
    os.environ["MAX_JOBS"] = "1"

    def build_extension(root, label):
        result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=str(root),
                                text=True, capture_output=True)
        print(label + "_STDOUT\n" + result.stdout[-12000:], flush=True)
        print(label + "_STDERR\n" + result.stderr[-12000:], flush=True)
        if result.returncode:
            raise RuntimeError(f"{label} build failed with exit code {result.returncode}")

    build_extension(detectron2_root, "DETECTRON2")
    build_extension(gomatching_root / "third_party", "ADET")
    protected_after = {name: version_or_none(name) for name in protected_names}
    changed = {name: {"before": protected_before[name], "after": protected_after[name]}
               for name in protected_names if protected_before[name] != protected_after[name]}
    if changed:
        raise RuntimeError(f"protected runtime changed: {changed}")
    write_json(OUT / "runtime_manifest.json", {
        "status": "RUNTIME_GREEN", "quality_status": QUALITY_STATUS,
        "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu_count": gpu_count, "requested_gpu_workers": REQUESTED_GPU_WORKERS,
        "effective_gpu_workers": effective_gpu_workers, "protected_before": protected_before,
        "protected_after": protected_after, "protected_changed": changed,
        "source_revisions": actual_revisions, "compatibility_patch_count": len(applied),
        "torch_cuda_nccl_abi_mutated": False,
    })
    phase("RUNTIME_GREEN", gpu_count=gpu_count, effective_gpu_workers=effective_gpu_workers)
'''


SELECTION = r'''
if not EXECUTE_PIPELINE:
    print("DRY_REVIEW_ONLY: inventory and images not read")
else:
    import pandas as pd
    from PIL import Image

    inventory_paths = sorted(INPUT_ROOT.glob("**/keyframe_inventory.parquet"))
    if not inventory_paths:
        raise FileNotFoundError("keyframe_inventory.parquet not found under /kaggle/input")
    preferred = [path for path in inventory_paths if "ocr-input-6shards" in str(path)]
    inventory_path = preferred[0] if preferred else inventory_paths[0]
    inventory = pd.read_parquet(inventory_path)
    required = {"frame_uid", "video_id", "source_frame_idx", "timestamp_ms", "shot_id",
                "image_shard_id", "source_dataset_slug", "image_member"}
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"inventory missing columns: {missing}")
    pool = inventory.loc[inventory["image_shard_id"].astype(str) == TARGET_SHARD_ID].copy()
    pool = pool.drop_duplicates("frame_uid")
    if len(pool) < MAX_FRAMES:
        raise ValueError(f"shard pool too small: {len(pool)} < {MAX_FRAMES}")
    pool["selection_rank"] = pool["frame_uid"].astype(str).map(
        lambda uid: hashlib.sha256(f"{SELECTION_SEED}|{uid}".encode()).hexdigest())
    selection = pool.sort_values(["selection_rank", "frame_uid"]).head(MAX_FRAMES).reset_index(drop=True)
    expected_uid = selection.apply(lambda row: f"{row['video_id']}:{int(row['source_frame_idx'])}", axis=1)
    if not (selection["frame_uid"].astype(str).values == expected_uid.values).all():
        raise ValueError("frame_uid identity mismatch; expected video_id:source_frame_idx")
    selection_sha256 = hashlib.sha256("\n".join(selection["frame_uid"].astype(str)).encode()).hexdigest()
    selection.to_parquet(OUT / "selection.parquet", index=False)

    owner, slug = TARGET_DATASET_SLUG.split("/", 1)
    dataset_roots = [INPUT_ROOT / "datasets" / owner / slug, INPUT_ROOT / slug,
                     INPUT_ROOT / owner / slug, INPUT_ROOT / TARGET_DATASET_SLUG]
    dataset_roots.extend(sorted(INPUT_ROOT.glob("**/" + slug)))
    dataset_roots = [root for index, root in enumerate(dataset_roots)
                     if root.is_dir() and root not in dataset_roots[:index]]
    if not dataset_roots:
        raise FileNotFoundError(f"dataset root unresolved: {TARGET_DATASET_SLUG}")

    def read_image_bytes(member):
        member = str(member).lstrip("/")
        short = member.removeprefix("images/")
        # Required ordering: long Kaggle dataset mount /images first, then short mounts.
        candidates = []
        for root in dataset_roots:
            candidates.extend((root / "images" / short, root / member, root / short))
        for path in candidates:
            if path.is_file():
                return path.read_bytes(), path
        raise FileNotFoundError(f"unresolved image: {member}; roots={dataset_roots}")

    checks = []
    for index in sorted({0, len(selection) // 2, len(selection) - 1}):
        row = selection.iloc[index]
        raw, resolved = read_image_bytes(row["image_member"])
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        checks.append({"frame_uid": str(row["frame_uid"]), "image_member": str(row["image_member"]),
                       "resolved_path": str(resolved), "bytes": len(raw),
                       "sha256": hashlib.sha256(raw).hexdigest()})
    write_json(OUT / "selection_manifest.json", {
        "status": "SELECTION_GREEN", "quality_status": QUALITY_STATUS,
        "target_shard_id": TARGET_SHARD_ID, "target_dataset_slug": TARGET_DATASET_SLUG,
        "inventory_path": str(inventory_path), "selection_seed": SELECTION_SEED,
        "frame_count": len(selection), "selection_sha256": selection_sha256,
        "identity": "frame_uid=video_id:source_frame_idx", "faiss_row_used_as_identity": False,
        "immutable_keyframe_v1": True,
    })
    write_json(OUT / "image_preflight.json", {
        "status": "IMAGE_PREFLIGHT_GREEN", "checks": checks,
        "resolver_order": "datasets/<owner>/<slug>/images before short mounts",
    })
    phase("IMAGE_PREFLIGHT_GREEN", frames=len(selection), decoded=len(checks), selection_sha256=selection_sha256)
'''


ASSETS = r'''
if not EXECUTE_PIPELINE:
    print("DRY_REVIEW_ONLY: weights not searched, downloaded, copied, or loaded")
else:
    import shutil

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    dstext_path = ASSET_ROOT / DEEPSOLO_WEIGHT_FILE
    parseq_path = ASSET_ROOT / PARSEQ_WEIGHT_FILE

    def verify(path, expected_bytes, expected_sha256):
        return path.is_file() and path.stat().st_size == expected_bytes and sha256_file(path) == expected_sha256

    def find_exact_mount(expected_bytes, expected_sha256, suffixes):
        for suffix in suffixes:
            for path in sorted(INPUT_ROOT.glob("**/" + suffix)):
                if verify(path, expected_bytes, expected_sha256):
                    return path
        return None

    dstext_source = find_exact_mount(DEEPSOLO_WEIGHT_BYTES, DEEPSOLO_WEIGHT_SHA256,
                                     (DEEPSOLO_WEIGHT_FILE, "*.pth"))
    if not dstext_source:
        raise FileNotFoundError(
            f"exact DSText weight not mounted from {WEIGHT_DATASET_SLUG}; "
            "network fallback is intentionally disabled"
        )
    shutil.copyfile(dstext_source, dstext_path)
    dstext_provenance = "HASH_GATED_PRIVATE_KAGGLE_BUNDLE"
    if not verify(dstext_path, DEEPSOLO_WEIGHT_BYTES, DEEPSOLO_WEIGHT_SHA256):
        raise RuntimeError("DSText official weight hash/size gate failed")

    parseq_source = find_exact_mount(PARSEQ_WEIGHT_BYTES, PARSEQ_WEIGHT_SHA256,
                                     (PARSEQ_WEIGHT_FILE, "*.ckpt"))
    if not parseq_source:
        raise FileNotFoundError(
            f"exact PARSeq VN checkpoint not mounted from {WEIGHT_DATASET_SLUG}; "
            "network fallback is intentionally disabled"
        )
    shutil.copyfile(parseq_source, parseq_path)
    parseq_provenance = "HASH_GATED_PRIVATE_KAGGLE_BUNDLE"
    if not verify(parseq_path, PARSEQ_WEIGHT_BYTES, PARSEQ_WEIGHT_SHA256):
        raise RuntimeError("PARSeq VN weight hash/size gate failed")

    write_json(OUT / "model_asset_manifest.json", {
        "status": "MODEL_ASSETS_GREEN", "quality_status": QUALITY_STATUS,
        "detector": {"model": DEEPSOLO_MODEL_LABEL, "path": str(dstext_path),
                     "bytes": dstext_path.stat().st_size, "sha256": sha256_file(dstext_path),
                     "source": dstext_provenance, "repo_revision": DEEPSOLO_REVISION},
        "recognizer": {"model": "PARSeq Vietnamese fine-tune", "path": str(parseq_path),
                       "bytes": parseq_path.stat().st_size, "sha256": sha256_file(parseq_path),
                       "source": parseq_provenance, "repo_revision": PARSEQ_REVISION},
    })
    phase("MODEL_ASSETS_GREEN", detector_sha256=DEEPSOLO_WEIGHT_SHA256,
          recognizer_sha256=PARSEQ_WEIGHT_SHA256)
'''


WORKER_SOURCE = r'''
import hashlib, io, json, os, sys, time, typing, unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

OUT = Path(os.environ["HCMAIC_WORKER_OUT"])
ASSIGNMENT = Path(os.environ["HCMAIC_ASSIGNMENT"])
SOURCE_ROOT = Path(os.environ["HCMAIC_SOURCE_ROOT"])
ASSET_ROOT = Path(os.environ["HCMAIC_ASSET_ROOT"])
THRESHOLD = float(os.environ["HCMAIC_THRESHOLD"])
BATCH_SIZE = int(os.environ["HCMAIC_PARSEQ_BATCH"])
WORKER_ID = int(os.environ["HCMAIC_WORKER_ID"])
OUT.mkdir(parents=True, exist_ok=True)

def write_json(path, payload):
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)

def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def load_jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def rewrite_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

def fold_text(text):
    nfc = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in unicodedata.normalize("NFD", nfc.lower()) if unicodedata.category(ch) != "Mn")

def heartbeat(phase, **fields):
    write_json(OUT / "progress.json", {"worker_id": WORKER_ID, "phase": phase, "time": time.time(), **fields})

for path in (SOURCE_ROOT / "DeepSolo" / "DeepSolo++" / "detectron2",
             SOURCE_ROOT / "GoMatching", SOURCE_ROOT / "GoMatching" / "third_party"):
    sys.path.insert(0, str(path))
import adet  # noqa: F401
import gomatching  # noqa: F401
from adet.config import add_deepsolo_cfg
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from gomatching.config import add_gom_config

cfg = get_cfg()
add_deepsolo_cfg(cfg)
add_gom_config(cfg)
cfg.merge_from_file(str(SOURCE_ROOT / "GoMatching" / "configs" / "GoMatching_DSText.yaml"))
cfg.defrost()
cfg.MODEL.META_ARCHITECTURE = "TransformerPureDetector"
cfg.MODEL.WEIGHTS = ""
cfg.MODEL.DEVICE = "cuda:0"
cfg.MODEL.TRANSFORMER.NUM_QUERIES = 300
cfg.MODEL.TRANSFORMER.NUM_POINTS = 25
cfg.MODEL.TRANSFORMER.VOC_SIZE = 37
cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST = THRESHOLD
cfg.INPUT.MIN_SIZE_TEST = 1280
cfg.INPUT.MAX_SIZE_TEST = 3000
cfg.DATASETS.TEST = ()
cfg.freeze()
detector = build_model(cfg)
payload = torch.load(ASSET_ROOT / "DSText_res50_300queries_finetune.pth", map_location="cpu", weights_only=True)
state = payload.get("model", payload) if isinstance(payload, dict) else payload
incompatible = detector.load_state_dict(state, strict=False)
if incompatible.missing_keys or incompatible.unexpected_keys:
    raise RuntimeError(f"DSText checkpoint incompatible: missing={incompatible.missing_keys[:8]}, unexpected={incompatible.unexpected_keys[:8]}")
detector.to("cuda:0").eval()
augment = T.ResizeShortestEdge([1280, 1280], 3000)

parseq_root = SOURCE_ROOT / "parseq-vn"
sys.path.insert(0, str(parseq_root))
import pytorch_lightning.utilities.types as pl_types
for alias in ("EPOCH_OUTPUT", "STEP_OUTPUT"):
    if not hasattr(pl_types, alias):
        setattr(pl_types, alias, typing.Any)
from src.parseq.strhub.models.utils import load_from_checkpoint
try:
    recognizer = load_from_checkpoint(str(ASSET_ROOT / "best-parseq.ckpt"), weights_only=False)
except TypeError as exc:
    if "weights_only" not in str(exc):
        raise
    recognizer = load_from_checkpoint(str(ASSET_ROOT / "best-parseq.ckpt"))
recognizer.eval().to("cuda:0")
img_size = tuple(int(value) for value in getattr(recognizer, "hparams", {}).get("img_size", (32, 128)))

assigned_rows = pd.read_parquet(ASSIGNMENT).to_dict("records")
failure_path = OUT / "failure_ledger.jsonl"
status_path = OUT / "detection_status.jsonl"
crop_path = OUT / "crop_inventory.jsonl"
ocr_path = OUT / "ocr_lines.jsonl"
crop_dir = OUT / "line_crops"
overlay_dir = OUT / "overlays"
crop_dir.mkdir(exist_ok=True)
overlay_dir.mkdir(exist_ok=True)

# Resume contract: retain terminal frames, but remove every partial artifact for
# non-terminal frame_uids before retrying them.  Deterministic crop_uids then
# cannot accumulate duplicate rows after a worker interruption.
existing_status = load_jsonl(status_path)
terminal_uids = {str(row["frame_uid"]) for row in existing_status if row.get("status") in {"OK", "NO_TEXT"}}
rows = [row for row in assigned_rows if str(row["frame_uid"]) not in terminal_uids]
pending_uids = {str(row["frame_uid"]) for row in rows}
rewrite_jsonl(status_path, [row for row in existing_status if str(row.get("frame_uid")) not in pending_uids])
rewrite_jsonl(crop_path, [row for row in load_jsonl(crop_path) if str(row.get("frame_uid")) not in pending_uids])
rewrite_jsonl(ocr_path, [row for row in load_jsonl(ocr_path) if str(row.get("frame_uid")) not in pending_uids])
rewrite_jsonl(failure_path, [row for row in load_jsonl(failure_path) if str(row.get("frame_uid")) not in pending_uids])

def boundary_to_polygon(boundary):
    points = np.asarray(boundary, dtype=np.float32).reshape(-1, 4)
    return np.concatenate([points[:, :2], points[::-1, 2:4]], axis=0)

def ordered_quad(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    return np.asarray([points[np.argmin(sums)], points[np.argmin(diffs)],
                       points[np.argmax(sums)], points[np.argmax(diffs)]], dtype=np.float32)

def contiguous_runs(values, threshold):
    active = np.asarray(values) >= threshold
    runs, start = [], None
    for index, is_active in enumerate(active.tolist() + [False]):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            runs.append((start, index)); start = None
    return runs

def merge_runs(runs, max_gap):
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= max_gap:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged

def word_crops_from_polygon(image, polygon):
    import cv2
    points = np.asarray(polygon, dtype=np.float32)
    if points.size == 0 or not np.isfinite(points).all() or len(points) < 4:
        raise ValueError("invalid DeepSolo polygon")
    source_quad = ordered_quad(cv2.boxPoints(cv2.minAreaRect(points)).astype(np.float32))
    width = max(float(np.linalg.norm(source_quad[1] - source_quad[0])),
                float(np.linalg.norm(source_quad[2] - source_quad[3])))
    height = max(float(np.linalg.norm(source_quad[3] - source_quad[0])),
                 float(np.linalg.norm(source_quad[2] - source_quad[1])))
    out_width, out_height = max(16, min(4096, int(round(width)))), max(12, min(1024, int(round(height))))
    target_quad = np.asarray([[0, 0], [out_width - 1, 0], [out_width - 1, out_height - 1],
                              [0, out_height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_quad, target_quad)
    inverse = np.linalg.inv(transform).astype(np.float32)
    warped = cv2.warpPerspective(np.asarray(image.convert("RGB")), transform, (out_width, out_height),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rectified = Image.fromarray(warped).convert("RGB")
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if foreground_ratio < 0.002 or foreground_ratio > 0.65:
        block_size = max(3, min(31, (min(out_height, out_width) // 2) * 2 + 1))
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    line_runs = merge_runs(contiguous_runs((mask > 0).sum(axis=1),
                                           max(1, int(round(out_width * 0.012)))),
                           max(1, int(round(out_height * 0.08))))
    if not line_runs:
        line_runs = [(0, out_height)]
    results = []
    for line_start, line_end in line_runs:
        line_height = line_end - line_start
        joined = cv2.dilate(mask[line_start:line_end],
                            np.ones((1, max(2, int(round(line_height * 0.20)))), dtype=np.uint8))
        word_runs = merge_runs(contiguous_runs((joined > 0).sum(axis=0),
                                               max(1, int(round(line_height * 0.08)))),
                               max(1, int(round(line_height * 0.12)))) or [(0, out_width)]
        pad_x, pad_y = max(1, int(round(line_height * 0.10))), max(1, int(round(line_height * 0.16)))
        for word_index, (word_start, word_end) in enumerate(word_runs):
            x0, x1 = max(0, word_start - pad_x), min(out_width, word_end + pad_x)
            y0, y1 = max(0, line_start - pad_y), min(out_height, line_end + pad_y)
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            rect_points = np.asarray([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
            source_points = cv2.perspectiveTransform(rect_points, inverse)[0]
            source_points[:, 0] = np.clip(source_points[:, 0], 0, image.width)
            source_points[:, 1] = np.clip(source_points[:, 1], 0, image.height)
            x_min, y_min = np.floor(source_points.min(axis=0)).astype(int)
            x_max, y_max = np.ceil(source_points.max(axis=0)).astype(int)
            results.append({"word_index": word_index,
                            "crop": rectified.crop((x0, y0, x1, y1)).convert("RGB"),
                            "polygon": [[float(x), float(y)] for x, y in source_points.tolist()],
                            "bbox": [max(0, int(x_min)), max(0, int(y_min)),
                                     min(image.width, int(x_max)), min(image.height, int(y_max))]})
    return results or [{"word_index": 0, "crop": rectified,
                        "polygon": [[float(x), float(y)] for x, y in points.tolist()],
                        "bbox": [max(0, int(points[:, 0].min())), max(0, int(points[:, 1].min())),
                                 min(image.width, int(points[:, 0].max())), min(image.height, int(points[:, 1].max()))]}]

def recognize(crops):
    tensors = []
    for image in crops:
        resized = image.resize((img_size[1], img_size[0]), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append((tensor - 0.5) / 0.5)
    batch = torch.stack(tensors).to("cuda:0")
    with torch.inference_mode():
        logits = recognizer(batch)
        probabilities = logits.softmax(-1)
        labels, scores = recognizer.tokenizer.decode(probabilities)
    return [(str(label), float(score.mean().item()) if hasattr(score, "mean") else float(score))
            for label, score in zip(labels, scores)]

heartbeat("MODEL_GREEN", frames_assigned=len(assigned_rows), frames_pending=len(rows), frames_resumed=len(terminal_uids))
frame_ok = len(terminal_uids)
crop_count = len(load_jsonl(crop_path))
recognition_count = len(load_jsonl(ocr_path))
failures = load_jsonl(failure_path)
started = time.time()
for frame_index, row in enumerate(rows):
    frame_uid = str(row["frame_uid"])
    try:
        path = Path(row["resolved_image_path"])
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        failure = {"failure_type": "READ_FAILED", "phase": "image_read", "frame_uid": frame_uid,
                   "crop_uid": None, "error": repr(exc), "resolved": False}
        append_jsonl(failure_path, failure); failures.append(failure)
        append_jsonl(status_path, {"frame_uid": frame_uid, "status": "READ_FAILED", "line_count": 0})
        continue
    try:
        original = np.asarray(image)
        transformed = augment.get_transform(original).apply_image(original)
        tensor = torch.as_tensor(transformed.astype("float32").transpose(2, 0, 1), device="cuda:0")
        with torch.inference_mode():
            output = detector([{"image": tensor, "height": image.height, "width": image.width}])[0]
        instances = output["instances"].to("cpu")
        boundaries = instances.bd.numpy() if instances.has("bd") else np.empty((0, 0, 4), dtype=np.float32)
        scores = instances.scores.numpy().tolist() if instances.has("scores") else []
    except Exception as exc:
        failure = {"failure_type": "INFERENCE_FAILED", "phase": "detection", "frame_uid": frame_uid,
                   "crop_uid": None, "error": repr(exc), "resolved": False}
        append_jsonl(failure_path, failure); failures.append(failure)
        append_jsonl(status_path, {"frame_uid": frame_uid, "status": "INFERENCE_FAILED", "line_count": 0})
        continue
    if len(boundaries) == 0:
        failure = {"failure_type": "NO_TEXT", "phase": "detection", "frame_uid": frame_uid,
                   "crop_uid": None, "error": None, "resolved": True}
        append_jsonl(failure_path, failure); failures.append(failure)
        append_jsonl(status_path, {"frame_uid": frame_uid, "status": "NO_TEXT", "line_count": 0})
        frame_ok += 1
        continue

    frame_crops, frame_meta, frame_had_failure = [], [], False
    for detector_line_index, boundary in enumerate(boundaries):
        detector_polygon = boundary_to_polygon(boundary)
        try:
            words = word_crops_from_polygon(image, detector_polygon)
        except Exception as exc:
            failure = {"failure_type": "PARSE_ERROR", "phase": "word_crop", "frame_uid": frame_uid,
                       "crop_uid": None, "error": repr(exc), "resolved": False}
            append_jsonl(failure_path, failure); failures.append(failure)
            frame_had_failure = True
            continue
        for word in words:
            rounded = [[round(float(x), 2), round(float(y), 2)] for x, y in word["polygon"]]
            crop_uid = hashlib.sha256((frame_uid + "|dstext-official-word|" + json.dumps(rounded)).encode()).hexdigest()[:24]
            saved = crop_dir / f"{crop_uid}.jpg"
            word["crop"].save(saved, quality=95)
            meta = {"crop_uid": crop_uid, "frame_uid": frame_uid, "video_id": str(row["video_id"]),
                    "source_frame_idx": int(row["source_frame_idx"]), "timestamp_ms": int(row["timestamp_ms"]),
                    "shot_id": str(row["shot_id"]), "line_index": len(frame_meta),
                    "detector_line_index": detector_line_index, "word_index": int(word["word_index"]),
                    "polygon": rounded, "detector_polygon": [[round(float(x), 2), round(float(y), 2)]
                                                               for x, y in detector_polygon],
                    "bbox": [int(value) for value in word["bbox"]],
                    "det_score": float(scores[detector_line_index]) if detector_line_index < len(scores) else None,
                    "detector_model": "DeepSolo ResNet-50 DSText official",
                    "detector_revision": "dbadae995035246bad3376c7a44c015c69e9b313",
                    "crop_path": str(saved), "worker_id": WORKER_ID}
            append_jsonl(crop_path, meta)
            frame_crops.append(word["crop"]); frame_meta.append(meta); crop_count += 1
    for start in range(0, len(frame_crops), BATCH_SIZE):
        batch_crops = frame_crops[start:start + BATCH_SIZE]
        batch_meta = frame_meta[start:start + BATCH_SIZE]
        try:
            decoded = recognize(batch_crops)
        except Exception as exc:
            for meta in batch_meta:
                failure = {"failure_type": "INFERENCE_FAILED", "phase": "recognition",
                           "frame_uid": frame_uid, "crop_uid": meta["crop_uid"],
                           "error": repr(exc), "resolved": False}
                append_jsonl(failure_path, failure); failures.append(failure)
                frame_had_failure = True
            continue
        for meta, (raw_text, rec_score) in zip(batch_meta, decoded):
            nfc = unicodedata.normalize("NFC", raw_text)
            status = "EMPTY" if not nfc.strip() else ("LOW_CONF" if rec_score < 0.35 else "OK")
            append_jsonl(ocr_path, {**meta, "ocr_text_raw": raw_text, "ocr_text_nfc": nfc,
                         "ocr_text_folded": fold_text(nfc), "rec_score": rec_score,
                         "confidence_status": status, "recognizer_model": "PARSeq Vietnamese fine-tune",
                         "recognizer_revision": "76cc5f3cc6268457aac764653400fdff681f8271"})
            recognition_count += 1
    append_jsonl(status_path, {"frame_uid": frame_uid, "status": "INFERENCE_FAILED" if frame_had_failure else "OK",
                               "line_count": len(frame_meta),
                               "crop_uids": [item["crop_uid"] for item in frame_meta]})
    frame_ok += 1
    if frame_index < 10:
        overlay = image.copy(); draw = ImageDraw.Draw(overlay)
        for meta in frame_meta:
            draw.polygon([tuple(point) for point in meta["polygon"]], outline="red", width=2)
        overlay.save(overlay_dir / f"{frame_uid.replace(':', '_')}.jpg", quality=90)
    if (frame_index + 1) % 5 == 0 or frame_index + 1 == len(rows):
        heartbeat("INFERENCE_HEARTBEAT", processed=frame_index + 1, frames=len(rows),
                  crops=crop_count, recognized=recognition_count, failures=len(failures),
                  elapsed_s=round(time.time() - started, 1))

write_json(OUT / "failure_ledger.json", {"failures": failures, "failure_count": len(failures),
           "unresolved_count": sum(not item["resolved"] for item in failures),
           "counts_by_type": {kind: sum(item["failure_type"] == kind for item in failures)
                              for kind in ("NO_TEXT", "READ_FAILED", "INFERENCE_FAILED", "PARSE_ERROR")}})
write_json(OUT / "worker_manifest.json", {
    "status": "WORKER_COMPLETE", "worker_id": WORKER_ID, "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "frames_assigned": len(assigned_rows), "frames_pending_at_start": len(rows),
    "frames_resumed": len(terminal_uids), "frames_terminal": len({row["frame_uid"] for row in
        [json.loads(line) for line in status_path.read_text(encoding="utf-8").splitlines() if line.strip()]}),
    "crop_count": crop_count, "recognition_count": recognition_count,
    "failure_count": len(failures), "unresolved_failure_count": sum(not item["resolved"] for item in failures),
    "detector_threshold": THRESHOLD, "full_frame_only": True, "tile_pass_count": 0,
})
heartbeat("WORKER_COMPLETE", frames=len(rows), crops=crop_count, recognized=recognition_count)
'''


LAUNCH = r'''
if not EXECUTE_PIPELINE:
    print("DRY_REVIEW_ONLY: worker script not written and no GPU subprocess launched")
else:
    import pandas as pd

    worker_script = OUT / "ocr_worker.py"
    worker_script.write_text(WORKER_SOURCE, encoding="utf-8")
    assignment_dir = OUT / "assignments"
    assignment_dir.mkdir(exist_ok=True)
    resolved_rows = []
    for row in selection.to_dict("records"):
        _, resolved = read_image_bytes(row["image_member"])
        resolved_rows.append({**row, "resolved_image_path": str(resolved)})
    assignments = [[] for _ in range(effective_gpu_workers)]
    for index, row in enumerate(resolved_rows):
        assignments[index % effective_gpu_workers].append(row)

    processes = []
    for worker_id, rows in enumerate(assignments):
        assignment = assignment_dir / f"worker_{worker_id:02d}.parquet"
        pd.DataFrame(rows).to_parquet(assignment, index=False)
        worker_out = OUT / f"worker_{worker_id:02d}"
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(worker_id), "HCMAIC_WORKER_ID": str(worker_id),
            "HCMAIC_WORKER_OUT": str(worker_out), "HCMAIC_ASSIGNMENT": str(assignment),
            "HCMAIC_SOURCE_ROOT": str(SOURCE_ROOT), "HCMAIC_ASSET_ROOT": str(ASSET_ROOT),
            "HCMAIC_THRESHOLD": str(DETECTION_SCORE_THRESHOLD), "HCMAIC_PARSEQ_BATCH": str(PARSEQ_BATCH_SIZE),
        })
        log_handle = (OUT / f"worker_{worker_id:02d}.log").open("w", encoding="utf-8")
        process = subprocess.Popen([sys.executable, str(worker_script)], env=env,
                                   stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        processes.append((worker_id, process, log_handle, worker_out))
    phase("PARALLEL_INFERENCE_START", workers=len(processes), assignments=[len(rows) for rows in assignments])

    def read_progress_snapshot(path, worker_id, returncode):
        for attempt in range(5):
            try:
                raw = path.read_text(encoding="utf-8")
                if not raw.strip():
                    raise json.JSONDecodeError("empty progress", raw, 0)
                return json.loads(raw)
            except (json.JSONDecodeError, OSError):
                if attempt < 4:
                    time.sleep(0.1)
        return {"worker_id": worker_id, "phase": "PROGRESS_READ_RETRY",
                "returncode": returncode, "progress_read_retries": 5}

    while any(process.poll() is None for _, process, _, _ in processes):
        time.sleep(min(HEARTBEAT_SECONDS, 20))
        snapshots = []
        for worker_id, process, _, worker_out in processes:
            progress = worker_out / "progress.json"
            snapshots.append(read_progress_snapshot(progress, worker_id, process.poll()) if progress.is_file()
                             else {"worker_id": worker_id, "phase": "STARTING", "returncode": process.poll()})
        if int(time.time()) % HEARTBEAT_SECONDS < 20:
            phase("PARALLEL_HEARTBEAT", workers=snapshots)
    failures = []
    for worker_id, process, log_handle, worker_out in processes:
        log_handle.close()
        if process.returncode != 0:
            failures.append({"worker_id": worker_id, "returncode": process.returncode,
                             "log": str(OUT / f"worker_{worker_id:02d}.log")})
        if not (worker_out / "worker_manifest.json").is_file():
            failures.append({"worker_id": worker_id, "reason": "worker_manifest_missing"})
    if failures:
        write_json(OUT / "parallel_failure.json", {"failures": failures})
        raise RuntimeError(f"parallel worker failure: {failures}")
    phase("PARALLEL_INFERENCE_COMPLETE", workers=len(processes))
'''


POSTFLIGHT = r'''
if not EXECUTE_PIPELINE:
    phase("DRY_REVIEW_COMPLETE", note="No input read, install, download, CUDA init, model load, inference, or artifact promotion")
    print("DRY REVIEW COMPLETE")
else:
    import pandas as pd

    worker_manifests = [json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted(OUT.glob("worker_*/worker_manifest.json"))]
    statuses = [row for path in sorted(OUT.glob("worker_*/detection_status.jsonl")) for row in load_jsonl(path)]
    crops = [row for path in sorted(OUT.glob("worker_*/crop_inventory.jsonl")) for row in load_jsonl(path)]
    lines = [row for path in sorted(OUT.glob("worker_*/ocr_lines.jsonl")) for row in load_jsonl(path)]
    failures = [row for path in sorted(OUT.glob("worker_*/failure_ledger.jsonl")) for row in load_jsonl(path)]
    expected_uids = set(selection["frame_uid"].astype(str))
    terminal_uids = {str(row["frame_uid"]) for row in statuses}
    crop_uids = [str(row["crop_uid"]) for row in crops]
    line_uids = [str(row["crop_uid"]) for row in lines]
    problems = []
    if terminal_uids != expected_uids:
        problems.append({"reason": "frame_uid_coverage_mismatch", "expected": len(expected_uids), "actual": len(terminal_uids)})
    if len(crop_uids) != len(set(crop_uids)):
        problems.append({"reason": "duplicate_crop_uid"})
    unresolved = [item for item in failures if not item.get("resolved", False)]
    if unresolved:
        problems.append({"reason": "unresolved_failures", "count": len(unresolved)})
    if set(line_uids) != set(crop_uids):
        problems.append({"reason": "recognition_crop_coverage_mismatch", "crops": len(set(crop_uids)),
                         "recognized": len(set(line_uids))})

    def write_jsonl(path, rows):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_jsonl(OUT / "detection_status.jsonl", sorted(statuses, key=lambda row: str(row["frame_uid"])))
    write_jsonl(OUT / "crop_inventory.jsonl", sorted(crops, key=lambda row: str(row["crop_uid"])))
    write_jsonl(OUT / "ocr_lines.jsonl", sorted(lines, key=lambda row: str(row["crop_uid"])))
    write_json(OUT / "failure_ledger.json", {
        "failure_count": len(failures), "unresolved_count": len(unresolved), "failures": failures,
        "counts_by_type": {kind: sum(item.get("failure_type") == kind for item in failures)
                           for kind in ("NO_TEXT", "READ_FAILED", "INFERENCE_FAILED", "PARSE_ERROR")},
    })
    pd.DataFrame(statuses).to_parquet(OUT / "detection_status.parquet", index=False)
    pd.DataFrame(crops).to_parquet(OUT / "crop_inventory.parquet", index=False)
    pd.DataFrame(lines).to_parquet(OUT / "ocr_lines.parquet", index=False)
    status = "ENGINEERING_ARTIFACT_COMPLETE" if not problems else "ENGINEERING_ARTIFACT_INCOMPLETE"
    manifest = {
        "status": status, "execution_status": "COMPLETE" if not problems else "INCOMPLETE",
        "provenance_class": PROVENANCE_CLASS, "quality_status": QUALITY_STATUS,
        "human_review_required": True, "problems": problems,
        "target_shard_id": TARGET_SHARD_ID, "selection_sha256": selection_sha256,
        "selection_count": len(selection), "frame_status_count": len(terminal_uids),
        "crop_count": len(crops), "recognition_count": len(lines),
        "failure_count": len(failures), "unresolved_failure_count": len(unresolved),
        "detector": {"model": DEEPSOLO_MODEL_LABEL, "revision": DEEPSOLO_REVISION,
                     "weight_sha256": DEEPSOLO_WEIGHT_SHA256, "score_threshold": DETECTION_SCORE_THRESHOLD,
                     "full_frame_only": FULL_FRAME_ONLY, "recall_mode": RECALL_MODE,
                     "tile_pass_count": TILE_PASS_COUNT, "passes_per_frame": PASSES_PER_FRAME},
        "recognizer": {"model": "PARSeq Vietnamese fine-tune", "revision": PARSEQ_REVISION,
                       "checkpoint_sha256": PARSEQ_WEIGHT_SHA256, "raw_and_normalized_preserved": True},
        "parallelism": {"requested_gpu_workers": REQUESTED_GPU_WORKERS,
                        "effective_gpu_workers": effective_gpu_workers, "worker_manifests": worker_manifests},
        "identity": "frame_uid=video_id:source_frame_idx; detector-scoped crop_uid; faiss_row not identity",
        "artifact_hashes": {name: sha256_file(OUT / name) for name in
                            ("selection.parquet", "detection_status.jsonl", "crop_inventory.jsonl",
                             "ocr_lines.jsonl", "failure_ledger.json", "runtime_manifest.json",
                             "model_asset_manifest.json")},
    }
    write_json(OUT / "final_manifest.json", manifest)
    phase("POSTFLIGHT_GREEN" if not problems else "POSTFLIGHT_INCOMPLETE",
          status=status, frames=len(terminal_uids), crops=len(crops), recognized=len(lines),
          failures=len(failures), quality_status=QUALITY_STATUS)
'''


def make_notebook(execute: bool) -> Path:
    out_dir = KERNEL_ROOT / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    config = CONFIG.replace("EXECUTE_PIPELINE = False", "EXECUTE_PIPELINE = True", 1) if execute else CONFIG
    cells = [
        cell(
            f"# {OUT_NAME}\n\n"
            "Bounded 100-frame OCR pilot on shard 0002: official DSText DeepSolo detector → word crops → Vietnamese PARSeq.\n\n"
            "- Threshold `0.30`; exactly one full-frame detector pass; no recall tiles.\n"
            "- Two concurrent GPU subprocesses when Kaggle exposes two GPUs; explicit one-GPU fallback is manifested.\n"
            "- `frame_uid=video_id:source_frame_idx`; detector-scoped `crop_uid`; keyframe v1 remains immutable.\n"
            "- Execution artifacts are `ENGINEERING_PROXY`; OCR/retrieval quality remains `UNVALIDATED`.\n"
            "- Set `EXECUTE_PIPELINE=True` only after dry cells 1–4 are reviewed.\n",
            "intro", "markdown"),
        cell(config, "config"),
        cell(RUNTIME, "runtime-and-source-preflight"),
        cell(SELECTION, "selection-and-image-preflight"),
        cell(ASSETS, "model-asset-preflight"),
        cell("WORKER_SOURCE = " + repr(WORKER_SOURCE), "worker-source"),
        cell(LAUNCH, "parallel-inference"),
        cell(POSTFLIGHT, "checkpoint-manifest-postflight"),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "hcmaic": {"status": "DRAFT_NOT_EXECUTED" if not execute else "EXECUTION_ARMED",
                       "quality_status": "UNVALIDATED", "identity": "frame_uid=video_id:source_frame_idx"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = out_dir / f"{OUT_NAME}.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": KERNEL_ID, "title": OUT_NAME, "code_file": path.name,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": bool(execute), "enable_tpu": False, "enable_internet": True,
        "keywords": [],
        "dataset_sources": [INVENTORY_DATASET_SLUG, TARGET_DATASET_SLUG, WEIGHT_DATASET_SLUG],
        "kernel_sources": [], "competition_sources": [],
        "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def validate_notebook(path: Path, execute: bool) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(item.get("source", [])) for item in notebook["cells"] if item.get("cell_type") == "code"]
    for source in sources:
        ast.parse(source)
    ast.parse(WORKER_SOURCE)
    code = "\n".join(sources)
    required = [
        'DETECTION_SCORE_THRESHOLD = 0.30', 'FULL_FRAME_ONLY = True', 'RECALL_MODE = False',
        'TILE_PASS_COUNT = 0', 'REQUESTED_GPU_WORKERS = 2', 'CUDA_VISIBLE_DEVICES',
        'IMAGE_PREFLIGHT_GREEN', 'failure_ledger', 'NO_TEXT', 'READ_FAILED',
        'INFERENCE_FAILED', 'PARSE_ERROR', 'final_manifest.json',
        'frame_uid=video_id:source_frame_idx', 'DEEPSOLO_WEIGHT_SHA256', 'PARSEQ_WEIGHT_SHA256',
        'network fallback is intentionally disabled', 'WEIGHT_DATASET_SLUG',
    ]
    missing = [token for token in required if token not in code]
    forbidden = [token for token in ("__RECOGNIZER_MODEL__", "PP-OCR", "VietOCR", "faiss_row\": str") if token in code]
    if missing or forbidden:
        raise ValueError({"missing": missing, "forbidden": forbidden})
    metadata = json.loads((path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    if metadata["enable_gpu"] != execute:
        raise ValueError("dry/execute GPU metadata mismatch")
    if not execute and "EXECUTE_PIPELINE = True" in code:
        raise ValueError("dry notebook is armed")
    return {"path": str(path), "cells": len(notebook["cells"]), "execute": execute,
            "enable_gpu": metadata["enable_gpu"], "worker_source_chars": len(WORKER_SOURCE)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="arm execution and GPU metadata")
    args = parser.parse_args()
    path = make_notebook(args.execute)
    print(json.dumps(validate_notebook(path, args.execute), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

