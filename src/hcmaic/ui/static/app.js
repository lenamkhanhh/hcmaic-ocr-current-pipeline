"use strict";

/**
 * HCMAIC KIS Retrieval — Matcha Dark Console Controller
 * Strictly UI-only presentation layer.
 * Features:
 * - 3-Pane Layout (Left Composer, Center Results Matrix, Right Stable Queue Rail)
 * - 5 Horizontal Channel Tabs for Stage 1, Stage 2, and dynamic Trake stages
 * - Dense Results Grid (Authoritative Backend Ranking)
 * - Optional / Lazy Top Context Timeline Strip (±15 frames on select)
 * - Temporal Media Inspector Modal (Replay, Prev, Next, Extract, Queue (Q))
 * - Zero Backend Mutation & Preserved API Contracts
 */

const $ = (id) => document.getElementById(id);
const HISTORY_KEY = "hcmaic.queryHistory.v1";
const MAX_HISTORY = 20;
const FIXED_VISUAL_INDEXES = Object.freeze(["siglip2"]);
const STAGE_CHANNELS = ["text", "ocr", "asr", "image", "object"];
const STAGE_CHANNEL_LABELS = Object.freeze({
  text: "Text",
  ocr: "OCR",
  asr: "ASR",
  image: "Image",
  object: "Object",
});
const DEFAULT_CHANNEL_ENABLED = false;
const STAGED_MIN_STAGES = 2;
const STAGED_MAX_STAGES = 5;
const TRAKE_MIN_STAGES = 2;
const TRAKE_MAX_STAGES = 5;
const TRAKE_MAX_DELTA_MS = 60_000;
const ALL_HITS_DEFAULT_MIN_GAP_MS = 3_000;
const GALLERY_THUMBNAIL_WIDTH = 320;
const GALLERY_THUMBNAIL_QUALITY = 78;
const GALLERY_EAGER_THUMBNAIL_LIMIT = 6;
const GALLERY_EAGER_VIDEO_LIMIT = 6;
const GALLERY_EAGER_MAX_IMAGES = 18;
const MAX_QUERY_IMAGE_BYTES = 10 * 1024 * 1024;
const MAX_QUERY_IMAGE_URL_LENGTH = 2048;
const QUERY_IMAGE_TYPES = Object.freeze(["image/jpeg", "image/png"]);
const BUNDLE_BORDER_PALETTE = Object.freeze([
  "#6f8f8d",
  "#9b7e6f",
  "#7e8fa6",
  "#9b8f63",
  "#8b7699",
  "#6f9485",
]);
const PREVIEW_SAMPLE_MODE = typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("preview") === "sample";

function thumbnailUrl(frameUid, width = GALLERY_THUMBNAIL_WIDTH, quality = GALLERY_THUMBNAIL_QUALITY) {
  return `/frames/${encodeURIComponent(frameUid)}/thumbnail?width=${width}&quality=${quality}`;
}

// Application State
let lastQueryId = null;
let selectedFrameId = null;
let selectedResult = null;
let lastResolvedPosition = null;
let lastExactExtraction = null;
let lastResultIds = [];
let queryRevision = 0;
let kisMode = false;
let currentTimeline = [];
let currentTemporal = null;
let lastSearchData = null;
let directVideoState = null;
let directVideoRequestSequence = 0;
let currentViewMode = "grouped"; // "grouped" | "flat"
let currentStageFilter = "all";  // "all" | "S1" … "S5"
let previewQueueItems = [];
let currentQueueItems = [];
let currentQueuePreview = false;
let activeSearchController = null;
let activeSearchRequestSequence = 0;
const CONTEXT_WINDOW_FRAMES = 15;
const DETAIL_TIMELINE_STEP_MS = 1000;
let topNeighborRequestId = 0;
let detailTimelineRequestId = 0;
let detailTimelineTimer = null;
let lastDetailTimelineRequestMs = null;
let inspectorStageItems = [];
let inspectorActiveStageId = null;
let activeInspectorTask = "KIS";
let activeSubmissionTask = "KIS";
let inspectorStageDrafts = new Map();
let inspectorQaDrafts = new Map();
let inspectorContextKey = null;
let inspectorQueryId = null;
const queueTemporalModeById = new Map();
let inspectorMarkerDrag = null;
let suppressInspectorMarkerClick = false;
let inspectorRangeMarkers = { L: null, R: null };
let detailMediaGeneration = 0;
const deferredImageObservers = new WeakMap();
const sessionId = "ui-" + Math.random().toString(36).slice(2);

const FALLBACK_OBJECT_ALIAS_ENTRIES = Object.freeze([
  ["person", "person"],
  ["people", "person"],
  ["persons", "person"],
  ["human", "person"],
  ["adult", "person"],
  ["adults", "person"],
  ["individual", "person"],
  ["individuals", "person"],
  ["man", "person"],
  ["woman", "person"],
  ["child", "person"],
  ["car", "car"],
  ["cars", "car"],
  ["automobile", "car"],
  ["auto", "car"],
  ["autos", "car"],
  ["sedan", "car"],
  ["sedans", "car"],
  ["motorcycle", "motorcycle"],
  ["motorbike", "motorcycle"],
  ["bicycle", "bicycle"],
  ["bike", "bicycle"],
  ["cycle", "bicycle"],
  ["cycles", "bicycle"],
  ["pushbike", "bicycle"],
  ["pushbikes", "bicycle"],
  ["airplane", "airplane"],
  ["plane", "airplane"],
  ["planes", "airplane"],
  ["aeroplane", "airplane"],
  ["aeroplanes", "airplane"],
  ["aircraft", "airplane"],
  ["máy bay", "airplane"],
  ["apple", "apple"],
  ["táo", "apple"],
  ["banana", "banana"],
  ["chuối", "banana"],
  ["backpack", "backpack"],
  ["rucksack", "backpack"],
  ["knapsack", "backpack"],
  ["schoolbag", "backpack"],
  ["bus", "bus"],
  ["truck", "truck"],
  ["tv", "tv"],
  ["television", "tv"],
  ["telly", "tv"],
  ["tivi", "tv"],
  ["dog", "dog"],
  ["puppy", "dog"],
  ["puppies", "dog"],
  ["cat", "cat"],
  ["kitty", "cat"],
  ["kitties", "cat"],
  ["feline", "cat"],
  ["bird", "bird"],
  ["boat", "boat"],
  ["thuyền", "boat"],
  ["rowboat", "boat"],
  ["sailboat", "boat"],
  ["speedboat", "boat"],
  ["book", "book"],
  ["sách", "book"],
  ["flask", "bottle"],
  ["mug", "cup"],
  ["mugs", "cup"],
  ["cup", "cup"],
  ["bottle", "bottle"],
  ["chair", "chair"],
  ["armchair", "chair"],
  ["clock", "clock"],
  ["đồng hồ", "clock"],
  ["couch", "couch"],
  ["sofa", "couch"],
  ["settee", "couch"],
  ["settees", "couch"],
  ["divan", "couch"],
  ["cow", "cow"],
  ["bò", "cow"],
  ["dining table", "dining table"],
  ["bàn ăn", "dining table"],
  ["table", "dining table"],
  ["dinner table", "dining table"],
  ["fork", "fork"],
  ["nĩa", "fork"],
  ["handbag", "handbag"],
  ["túi xách", "handbag"],
  ["purse", "handbag"],
  ["purses", "handbag"],
  ["backpack", "backpack"],
  ["umbrella", "umbrella"],
  ["cell phone", "cell phone"],
  ["smartphone", "cell phone"],
  ["laptop", "laptop"],
  ["notebook computer", "laptop"],
  ["portable computer", "laptop"],
  ["keyboard", "keyboard"],
  ["bàn phím", "keyboard"],
  ["knife", "knife"],
  ["dao", "knife"],
  ["microwave", "microwave"],
  ["lò vi sóng", "microwave"],
  ["microwave oven", "microwave"],
  ["mouse", "mouse"],
  ["chuột", "mouse"],
  ["orange", "orange"],
  ["cam", "orange"],
  ["potted plant", "potted plant"],
  ["plant", "potted plant"],
  ["plants", "potted plant"],
  ["houseplant", "potted plant"],
  ["houseplants", "potted plant"],
  ["indoor plant", "potted plant"],
  ["cây cảnh", "potted plant"],
  ["cây trong chậu", "potted plant"],
  ["refrigerator", "refrigerator"],
  ["tủ lạnh", "refrigerator"],
  ["remote control", "remote"],
  ["điều khiển", "remote"],
  ["remote controls", "remote"],
  ["scissors", "scissors"],
  ["kéo", "scissors"],
  ["shears", "scissors"],
  ["spoon", "spoon"],
  ["thìa", "spoon"],
  ["teaspoon", "spoon"],
  ["tablespoon", "spoon"],
  ["stop sign", "stop sign"],
  ["biển báo dừng", "stop sign"],
  ["suitcase", "suitcase"],
  ["vali", "suitcase"],
  ["luggage", "suitcase"],
  ["baggage", "suitcase"],
  ["tie", "tie"],
  ["cà vạt", "tie"],
  ["necktie", "tie"],
  ["neck tie", "tie"],
  ["toilet", "toilet"],
  ["bồn cầu", "toilet"],
  ["traffic light", "traffic light"],
  ["đèn giao thông", "traffic light"],
  ["stoplight", "traffic light"],
  ["stop light", "traffic light"],
  ["train", "train"],
  ["tàu hỏa", "train"],
  ["railway train", "train"],
  ["subway train", "train"],
  ["vase", "vase"],
  ["bình hoa", "vase"],
  ["flower vase", "vase"],
  ["wine glass", "wine glass"],
  ["ly rượu vang", "wine glass"],
  ["goblet", "wine glass"],
  ["zebra", "zebra"],
  ["ngựa vằn", "zebra"],
  ["parasol", "umbrella"],
  ["ball", "sports ball"],
  ["balls", "sports ball"],
  ["teddy", "teddy bear"],
  ["stuffed bear", "teddy bear"],
  ["plush bear", "teddy bear"],
  ["tennis racquet", "tennis racket"],
  ["racquet", "tennis racket"],
]);
let objectAliasCatalog = {
  status: "fallback",
  version: "ui-object-alias-fallback-v3",
  aliases: FALLBACK_OBJECT_ALIAS_ENTRIES.map(([alias, label]) => ({ alias, label })),
  labels: [...new Set(FALLBACK_OBJECT_ALIAS_ENTRIES.map(([, label]) => label))],
};
let objectAliasLookup = new Map(FALLBACK_OBJECT_ALIAS_ENTRIES);

/* ==========================================================================
   Utility & API Helpers
   ========================================================================== */

function setStatus(text, isError = false, spinning = false) {
  const el = $("status");
  if (!el) return;
  el.className = isError ? "status-message error" : "status-message muted";
  el.innerHTML = spinning ? '<span class="spinner">&#9696;</span> ' + escapeHtml(text) : escapeHtml(text);
}

function escapeHtml(str) {
  if (typeof str !== "string") return String(str ?? "");
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeXml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/*
 * Deterministic UI-only fixture. It is opt-in via ?preview=sample and is
 * intentionally labelled as synthetic; it never participates in a query or
 * replaces backend ranking. The preview cards use a pinned, public HF video
 * source so the inspector can exercise the real media element while the
 * backend/media bridge is offline.
 */
const PREVIEW_VIDEO_SPECS = [
  ["L29_V002", "bicycle / street", "#9d8ec2"],
  ["L29_V015", "rider / garden", "#8cb372"],
  ["L29_V023", "person / market", "#c29d79"],
  ["L30_V096", "vehicle / road", "#7895a8"],
  ["L27_V004", "person / trees", "#a77e9a"],
  ["L28_V011", "street / traffic", "#9a9b6e"],
  ["L23_V004", "bike / plaza", "#b27e68"],
  ["L30_V028", "outdoor / crowd", "#718e8c"],
];

// Preview-only media mapping. Production rows still receive video_url from
// the local manifest-backed API and must not expose a remote delivery URL.
const PREVIEW_HF_DATASET = "NHANGIOI/AIC2026";
const PREVIEW_HF_REVISION = "120220a0237d7051aadf69b94fbf34336cd5ea77";
const PREVIEW_HF_VIDEO_FOLDERS = Object.freeze({
  L29_V002: "Videos_L29_a",
  L29_V015: "Videos_L29_a",
  L29_V023: "Videos_L29_a",
  L30_V096: "Videos_L30_a",
  L27_V004: "Videos_L27_a",
  L28_V011: "Videos_L28_a",
  L23_V004: "Videos_L23_a",
  L30_V028: "Videos_L30_a",
});

function previewVideoUrl(videoId) {
  const folder = PREVIEW_HF_VIDEO_FOLDERS[videoId];
  if (!folder) return "";
  const datasetPath = PREVIEW_HF_DATASET.split("/").map(encodeURIComponent).join("/");
  return [
    "https://huggingface.co/datasets",
    datasetPath,
    "resolve",
    PREVIEW_HF_REVISION,
    "raw_video",
    encodeURIComponent(folder),
    `${encodeURIComponent(videoId)}.mp4`,
  ].join("/");
}

function previewFrameImage(label, accent, _stage, variant) {
  const safeLabel = escapeXml(label);
  const safeVariant = escapeXml(variant);
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 320" role="img" aria-label="${safeLabel}">
      <defs>
        <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stop-color="#19241c"/>
          <stop offset="1" stop-color="${accent}" stop-opacity=".72"/>
        </linearGradient>
      </defs>
      <rect width="640" height="320" fill="url(#bg)"/>
      <path d="M0 236 C110 188 170 258 270 208 S470 196 640 230 V320 H0Z" fill="#0b100d" opacity=".72"/>
      <path d="M0 84 C110 48 210 102 318 68 S520 48 640 90" fill="none" stroke="#e3ebe5" stroke-opacity=".32" stroke-width="5"/>
      <circle cx="512" cy="88" r="42" fill="#e3ebe5" opacity=".18"/>
      <text x="24" y="286" fill="#e3ebe5" font-family="monospace" font-size="18" font-weight="600">PREVIEW SAMPLE</text>
      <text x="24" y="307" fill="#e3ebe5" opacity=".8" font-family="monospace" font-size="14">${safeVariant}</text>
    </svg>`;
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

function buildPreviewSampleData() {
  const results = [];
  const stageResults = { S1: [], S2: [] };
  let finalRank = 1;

  for (let videoIndex = 0; videoIndex < PREVIEW_VIDEO_SPECS.length; videoIndex += 1) {
    const [videoId, scene, accent] = PREVIEW_VIDEO_SPECS[videoIndex];
    const baseFrame = 5600 + videoIndex * 1731;
    const baseScore = 0.941 - videoIndex * 0.031;
    const previewVideo = previewVideoUrl(videoId);

    for (const [stageId, frameOffset, scoreOffset] of [["S1", 0, 0], ["S2", 141, -0.006]]) {
      const sourceFrameIdx = baseFrame + frameOffset;
      const frameUid = `${videoId}:${sourceFrameIdx}`;
      const score = baseScore + scoreOffset;
      const result = {
        frame_id: frameUid,
        frame_uid: frameUid,
        image_url: previewFrameImage(`${videoId} ${scene}`, accent, stageId, `${videoId}:${sourceFrameIdx}`),
        video_id: videoId,
        frame_idx: sourceFrameIdx,
        source_frame_idx: sourceFrameIdx,
        timestamp_ms: 1200 + videoIndex * 4300 + frameOffset * 40,
        shot_id: `${videoId}-shot-01`,
        rank: videoIndex + 1,
        rank_in_stage: videoIndex + 1,
        final_rank: finalRank,
        final_score: score,
        signal_scores: { siglip2: score },
        image_available: true,
        image_status: "PREVIEW_SAMPLE",
        video_url: previewVideo || null,
        video_stream_available: Boolean(previewVideo),
        video_stream_status: previewVideo ? "AVAILABLE_REMOTE_RANGE" : "PREVIEW_ONLY",
        video_backend: previewVideo ? "huggingface_http_range" : "preview_only",
        video_revision: previewVideo ? PREVIEW_HF_REVISION : null,
        video_range_capable: Boolean(previewVideo),
        video_provenance_status: "ENGINEERING_PROXY",
        stage_id: stageId,
      };
      results.push(result);
      stageResults[stageId].push(result);
      finalRank += 1;
    }
  }

  return {
    query_id: "preview-sample-contact-sheet",
    query: "a man riding a bicycle",
    results,
    fused_results: results,
    stage_results: stageResults,
    total_found: results.length,
    latency_ms: 0,
    quality_status: "UNVALIDATED",
    enabled_indexes: ["siglip2"],
  };
}

function loadPreviewSample() {
  if (!PREVIEW_SAMPLE_MODE) return;
  const data = buildPreviewSampleData();
  lastSearchData = data;
  lastQueryId = data.query_id;
  lastResultIds = data.results.map((item) => item.frame_id);
  currentViewMode = "grouped";

  const grouped = $("viewGroupedBtn");
  const flat = $("viewFlatBtn");
  grouped?.classList.add("active");
  flat?.classList.remove("active");

  populateVideoFilter([...new Set(data.results.map((item) => item.video_id))]);
  if ($("s1Text") && !$("s1Text").value) $("s1Text").value = "a man riding a bicycle";
  if ($("s2Text") && !$("s2Text").value) $("s2Text").value = "person and bicycle outdoors";

  renderResultsView();
  renderChannelStatus({
    visual: { status: "preview_only", reason: "Synthetic UI fixture; no query sent." },
    siglip2: { status: "preview_only" },
    ocr: { status: "unavailable" },
    object: { status: "unavailable" },
  });
  updateStatusSummary(data);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON response */ }
  if (!res.ok) {
    const detail = formatApiErrorDetail(body, res.statusText);
    throw new Error(`${res.status}: ${detail}`);
  }
  return body;
}

function formatApiErrorDetail(body, fallback = "Request failed") {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  return detail != null ? JSON.stringify(detail) : fallback;
}

function beginSearchRequest() {
  activeSearchController?.abort();
  const controller = typeof AbortController === "function" ? new AbortController() : null;
  const request = {
    id: ++activeSearchRequestSequence,
    controller,
    signal: controller?.signal,
  };
  activeSearchController = controller;
  return request;
}

function isCurrentSearchRequest(requestId) {
  return requestId === activeSearchRequestSequence;
}

function endSearchRequest(requestId) {
  if (isCurrentSearchRequest(requestId)) activeSearchController = null;
}

function isAbortError(error) {
  return error?.name === "AbortError";
}

function normalizeObjectAlias(value) {
  return String(value ?? "").trim().toLocaleLowerCase().replace(/\s+/g, " ");
}

function setObjectAliasCatalog(data) {
  const aliases = Array.isArray(data?.aliases)
    ? data.aliases.filter((item) => item && typeof item.alias === "string" && typeof item.label === "string")
    : [];
  const labels = Array.isArray(data?.labels)
    ? data.labels.filter((label) => typeof label === "string")
    : [];
  if (!aliases.length && !labels.length) return;

  objectAliasCatalog = {
    status: data?.status || "ready",
    version: data?.version || "unknown",
    aliases,
    labels,
  };
  objectAliasLookup = new Map();
  for (const item of aliases) {
    const alias = normalizeObjectAlias(item.alias);
    const label = normalizeObjectAlias(item.label);
    if (alias && label) objectAliasLookup.set(alias, label);
  }
  for (const label of labels) {
    const normalized = normalizeObjectAlias(label);
    if (normalized && !objectAliasLookup.has(normalized)) {
      objectAliasLookup.set(normalized, normalized);
    }
  }
  updateObjectAliasOptions();
  document.querySelectorAll(".object-query-row").forEach((row) => updateObjectQueryRow(row));
}

function resolveObjectAlias(value) {
  const normalized = normalizeObjectAlias(value);
  return objectAliasLookup.get(normalized) || "";
}

function updateObjectAliasOptions() {
  const datalist = $("objectAliasOptions");
  if (!datalist) return;
  datalist.innerHTML = "";
  const seen = new Set();
  for (const item of objectAliasCatalog.aliases || []) {
    const alias = normalizeObjectAlias(item.alias);
    if (!alias || seen.has(alias)) continue;
    seen.add(alias);
    const option = document.createElement("option");
    option.value = item.alias;
    option.label = `→ ${item.label}`;
    datalist.appendChild(option);
  }
  for (const label of objectAliasCatalog.labels || []) {
    const normalized = normalizeObjectAlias(label);
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    const option = document.createElement("option");
    option.value = label;
    option.label = "COCO raw label";
    datalist.appendChild(option);
  }
}

async function loadObjectAliases() {
  updateObjectAliasOptions();
  if (PREVIEW_SAMPLE_MODE) return;
  try {
    const data = await api("/object/aliases");
    if (data?.status === "ready") setObjectAliasCatalog(data);
  } catch (err) {
    // The fallback catalog keeps the builder usable while the backend is offline.
    console.warn("Object alias catalog unavailable; using UI fallback:", err);
  }
}

function stageBlockFor(stageId, scope = document) {
  const stage = String(stageId).toLowerCase().replace(/^trake-/, "");
  if (scope.matches?.(".stage-block")) {
    const scopeStage = String(scope.dataset.stage || "").toLowerCase().replace(/^trake-/, "");
    if (scopeStage === stage || scope.classList.contains(`${stage}-block`)) return scope;
  }
  return scope.querySelector?.(`.stage-block[data-stage="${stage}"]`) ||
    scope.querySelector?.(`.${stage}-block`) || null;
}

function domStageId(value) {
  const match = String(value || "").match(/s(\d+)/i);
  return match ? `S${match[1]}` : String(value || "").toUpperCase();
}

function stageIdForBlock(block) {
  if (!block) return "";
  const classStage = [...(block.classList || [])]
    .find((className) => /^s\d+-block$/i.test(className));
  return domStageId(block.dataset.stage || classStage);
}

function stageBlocksFor(root) {
  if (!root) return [];
  return [...root.querySelectorAll(".stage-block")]
    .filter((block) => block.parentElement === root)
    .sort((left, right) => stageNumber(stageIdForBlock(left)) - stageNumber(stageIdForBlock(right)));
}

function stageIdsFromBlocks(root) {
  return stageBlocksFor(root)
    .map(stageIdForBlock)
    .filter((stageId) => /^S[1-5]$/.test(stageId));
}

function objectBuilder(stageId, scope = document) {
  const block = stageBlockFor(stageId, scope);
  return block?.querySelector(".object-query-builder") ||
    document.querySelector(`#${stageId.toLowerCase()}ObjectBuilder`);
}

function buildObjectQuery(stageId, scope = document) {
  const builder = objectBuilder(stageId, scope);
  if (!builder) return "";
  const clauses = [];
  for (const row of builder.querySelectorAll(".object-query-row")) {
    const countInput = row.querySelector(".object-count");
    const aliasInput = row.querySelector(".object-alias");
    const alias = aliasInput ? aliasInput.value.trim() : "";
    const count = countInput ? Number.parseInt(countInput.value, 10) : NaN;
    if (alias && Number.isInteger(count) && count >= 1) {
      clauses.push(`${count} ${alias}`);
    }
  }
  const query = clauses.join(" + ");
  const hidden = builder.querySelector('input[type="hidden"]') || $(`${stageId.toLowerCase()}Object`);
  if (hidden) hidden.value = query;
  const preview = builder.querySelector(".object-query-preview");
  if (preview) preview.textContent = `Query: ${query || "—"}`;
  return query;
}

function updateObjectQueryRow(row) {
  if (!row) return;
  const aliasInput = row.querySelector(".object-alias");
  const resolution = row.querySelector(".object-alias-resolution");
  const rawAlias = aliasInput ? aliasInput.value.trim() : "";
  const resolved = resolveObjectAlias(rawAlias);
  if (resolution) {
    resolution.classList.toggle("is-mapped", Boolean(rawAlias && resolved));
    resolution.classList.toggle("is-unmapped", Boolean(rawAlias && !resolved));
    resolution.textContent = !rawAlias
      ? "type an object alias"
      : resolved
        ? `→ ${resolved}`
        : "unmapped alias (raw label will be tried)";
  }
  const builder = row.closest(".object-query-builder");
  if (builder?.dataset.stage) {
    buildObjectQuery(domStageId(builder.dataset.stage), builder.closest(".stage-block") || document);
  }
}

function refreshObjectQueryRows(stageId, scope = document) {
  const builder = objectBuilder(stageId, scope);
  if (!builder) return;
  const rows = [...builder.querySelectorAll(".object-query-row")];
  rows.forEach((row, index) => {
    row.dataset.rowIndex = String(index);
    const remove = row.querySelector(".object-remove-row");
    if (remove) remove.hidden = rows.length <= 1;
  });
  buildObjectQuery(stageId, builder.closest(".stage-block") || scope);
}

function wireObjectQueryRow(row, stageId, scope = document) {
  if (!row || row.dataset.wired === "true") return;
  row.dataset.wired = "true";
  row.querySelector(".object-count")?.addEventListener("input", () => updateObjectQueryRow(row));
  row.querySelector(".object-alias")?.addEventListener("input", () => updateObjectQueryRow(row));
  row.querySelector(".object-add-row")?.addEventListener("click", () => addObjectQueryRow(stageId, row, scope));
  row.querySelector(".object-remove-row")?.addEventListener("click", () => removeObjectQueryRow(stageId, row, scope));
  updateObjectQueryRow(row);
}

function addObjectQueryRow(stageId, sourceRow, scope = sourceRow?.closest(".stage-block") || document) {
  const builder = objectBuilder(stageId, scope);
  const rowsContainer = builder?.querySelector(".object-query-rows");
  if (!rowsContainer) return;
  const rows = [...rowsContainer.querySelectorAll(".object-query-row")];
  const template = sourceRow || rows[rows.length - 1];
  if (!template) return;
  const row = template.cloneNode(true);
  row.dataset.wired = "false";
  row.querySelector(".object-count").value = "1";
  row.querySelector(".object-alias").value = "";
  const resolution = row.querySelector(".object-alias-resolution");
  if (resolution) resolution.textContent = "type an object alias";
  if (sourceRow && sourceRow.parentElement === rowsContainer) {
    sourceRow.after(row);
  } else {
    rowsContainer.appendChild(row);
  }
  wireObjectQueryRow(row, stageId, scope);
  refreshObjectQueryRows(stageId, scope);
  row.querySelector(".object-alias")?.focus();
}

function removeObjectQueryRow(stageId, row, scope = row?.closest(".stage-block") || document) {
  const builder = objectBuilder(stageId, scope);
  const rows = builder ? [...builder.querySelectorAll(".object-query-row")] : [];
  if (rows.length <= 1) {
    row.querySelector(".object-count").value = "1";
    row.querySelector(".object-alias").value = "";
    updateObjectQueryRow(row);
    return;
  }
  row.remove();
  refreshObjectQueryRows(stageId, scope);
}

function setupObjectQueryBuilders() {
  updateObjectAliasOptions();
  for (const builder of document.querySelectorAll(".object-query-builder")) {
    const stageId = domStageId(builder.dataset.stage);
    if (!stageId) continue;
    const rowsContainer = builder.querySelector(".object-query-rows");
    setupHorizontalWheelScroll(rowsContainer);
    const scope = builder.closest(".stage-block") || document;
    for (const row of builder.querySelectorAll(".object-query-row")) {
      wireObjectQueryRow(row, stageId, scope);
    }
    refreshObjectQueryRows(stageId, scope);
  }
}

function trakeStageIds() {
  return stageIdsFromBlocks($("trakeStages"));
}

function stagedStageIds() {
  return stageIdsFromBlocks($("stagedStages"));
}

function setSelectedStageChannel(channel, tabButtons, panels) {
  tabButtons.forEach((button) => {
    const selected = button.dataset.channel === channel;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
  panels.forEach((panel) => panel.classList.toggle("active", panel.dataset.channel === channel));
}

function createStageBlock(stageNumber, { trake = false } = {}) {
  const template = document.querySelector("#stagedSearchSection .s1-block");
  if (!template) return null;
  const stageId = `S${stageNumber}`;
  const stage = stageId.toLowerCase();
  const block = template.cloneNode(true);
  if (trake) block.classList.add("trake-stage-block");
  block.classList.remove("s1-block");
  block.classList.add(`${stage}-block`);
  block.dataset.stage = stage;

  for (const node of [block, ...block.querySelectorAll("*")]) {
    for (const attr of ["id", "class", "data-stage", "aria-label", "title"]) {
      if (!node.hasAttribute?.(attr)) continue;
      const value = node.getAttribute(attr);
      if (value) node.setAttribute(attr, value.replace(/s1/gi, stage));
    }
  }
  const badge = block.querySelector(".stage-badge");
  if (badge) {
    badge.textContent = `Stage ${stageNumber}`;
    for (const className of [...badge.classList]) {
      if (/^s\d+-badge$/i.test(className) || className === "trake-stage-badge" || className === "stage-generic-badge") {
        badge.classList.remove(className);
      }
    }
    badge.classList.add(stageNumber === 1 ? "s1-badge" : stageNumber === 2 ? "s2-badge" : "stage-generic-badge");
    if (trake) badge.classList.add("trake-stage-badge");
  }
  const tabs = block.querySelector(".stage-channel-tabs");
  if (tabs) {
    tabs.dataset.stage = stage;
    delete tabs.dataset.tabsReady;
  }
  const panels = block.querySelector(".channel-input-panels");
  if (panels) panels.dataset.stage = stage;
  const picker = block.querySelector(".asr-mode-picker");
  if (picker) {
    picker.dataset.stage = stage;
    delete picker.dataset.asrReady;
  }
  const builder = block.querySelector(".object-query-builder");
  if (builder) builder.dataset.stage = stage;
  for (const row of block.querySelectorAll(".object-query-rows")) row.dataset.stage = stage;
  for (const input of block.querySelectorAll(".object-count, .object-alias")) {
    input.value = input.classList.contains("object-count") ? "1" : "";
  }
  for (const input of block.querySelectorAll("textarea.input-composer")) {
    input.value = "";
  }
  for (const input of block.querySelectorAll("textarea.image-query-url-input")) {
    input.value = "";
    delete input.dataset.imagePasteReady;
  }
  for (const input of block.querySelectorAll("input[type=file]")) {
    input.value = "";
    delete input.dataset.imageReady;
    delete input.dataset.imagePasteReady;
  }
  for (const target of block.querySelectorAll("[data-image-paste-target]")) {
    delete target.dataset.imagePasteReady;
  }
  for (const preview of block.querySelectorAll(".image-query-preview")) {
    preview.hidden = true;
    preview.removeAttribute("src");
    delete preview.dataset.objectUrl;
    preview.onload = null;
    preview.onerror = null;
  }
  for (const status of block.querySelectorAll(".image-query-status")) {
    status.textContent = "";
  }
  for (const clear of block.querySelectorAll(".image-query-clear")) {
    clear.disabled = true;
  }
  for (const row of block.querySelectorAll(".object-query-row")) {
    row.dataset.wired = "false";
  }
  setSelectedStageChannel(
    "text",
    block.querySelectorAll(".channel-tab-btn"),
    block.querySelectorAll(".channel-panel"),
  );
  return block;
}

function createTrakeStageBlock(stageNumber) {
  return createStageBlock(stageNumber, { trake: true });
}

function renderStagedStageComposer(count) {
  const root = $("stagedStages");
  if (!root) return;
  const safeCount = Math.max(STAGED_MIN_STAGES, Math.min(STAGED_MAX_STAGES, count));
  let currentCount = stageBlocksFor(root).length;
  while (currentCount < safeCount) {
    const stageNumber = currentCount + 1;
    const block = createStageBlock(stageNumber);
    if (block) root.appendChild(block);
    currentCount += 1;
  }
  while (currentCount > safeCount) {
    root.lastElementChild?.remove();
    currentCount -= 1;
  }
  const countLabel = $("stagedStageCount");
  if (countLabel) countLabel.textContent = `${safeCount} stages`;
  setupChannelTabs();
  setupImageInputs();
  setupAsrModeToggles();
  setupObjectQueryBuilders();
  root.querySelectorAll(".stage-channel-tabs").forEach(setupHorizontalWheelScroll);
}

function setupStagedComposer() {
  if (!$('stagedStages')) return;
  renderStagedStageComposer(STAGED_MIN_STAGES);
  $("addStagedStage")?.addEventListener("click", () => {
    const next = Math.min(STAGED_MAX_STAGES, stagedStageIds().length + 1);
    renderStagedStageComposer(next);
  });
  $("removeStagedStage")?.addEventListener("click", () => {
    const next = Math.max(STAGED_MIN_STAGES, stagedStageIds().length - 1);
    renderStagedStageComposer(next);
  });
}

function renderTrakeStageComposer(count) {
  const root = $("trakeStages");
  if (!root) return;
  const safeCount = Math.max(TRAKE_MIN_STAGES, Math.min(TRAKE_MAX_STAGES, count));
  let currentCount = stageBlocksFor(root).length;
  while (currentCount < safeCount) {
    const stageNumber = currentCount + 1;
    const block = createTrakeStageBlock(stageNumber);
    if (block) root.appendChild(block);
    currentCount += 1;
  }
  while (currentCount > safeCount) {
    root.lastElementChild?.remove();
    currentCount -= 1;
  }
  const countLabel = $("trakeStageCount");
  if (countLabel) countLabel.textContent = `${safeCount} stages`;
  setupChannelTabs();
  setupImageInputs();
  setupAsrModeToggles();
  setupObjectQueryBuilders();
  root.querySelectorAll(".stage-channel-tabs").forEach(setupHorizontalWheelScroll);
}

function setupTrakeComposer() {
  if (!$('trakeStages')) return;
  renderTrakeStageComposer(TRAKE_MIN_STAGES);
  $("addTrakeStage")?.addEventListener("click", () => {
    const next = Math.min(TRAKE_MAX_STAGES, trakeStageIds().length + 1);
    renderTrakeStageComposer(next);
  });
  $("removeTrakeStage")?.addEventListener("click", () => {
    const next = Math.max(TRAKE_MIN_STAGES, trakeStageIds().length - 1);
    renderTrakeStageComposer(next);
  });
}

/* ==========================================================================
   System Info & Runtime Channels
   ========================================================================== */

async function loadSystemInfo() {
  if (PREVIEW_SAMPLE_MODE) return;

  try {
    const [health, info, providers] = await Promise.all([
      api("/health"),
      api("/system/info").catch(() => ({})),
      api("/system/providers").catch(() => null),
    ]);
    kisMode = health.kis_runtime === true;
    const videoIds = Array.isArray(info.video_ids) ? info.video_ids : [];

    renderChannelStatus(
      providers?.channel_status || health.channel_status || health.channels || info.runtime?.channel_status || {}
    );
    populateVideoFilter(videoIds);
  } catch (err) {
    console.warn("System info unavailable (backend offline):", err);
  }
}

function populateVideoFilter(videoIds) {
  const videoFilter = $("videoFilter");
  const options = $("videoFilterOptions");
  if (!videoFilter || !options) return;

  const seen = new Set();
  videoFilterCatalog = [];
  for (const rawVideoId of Array.isArray(videoIds) ? videoIds : []) {
    const value = String(rawVideoId ?? "").trim();
    if (!value || seen.has(value)) continue;
    seen.add(value);
    videoFilterCatalog.push(value);
  }

  // A refreshed catalog must never leave a committed filter pointing at an
  // option that no longer exists. This is local state only; no search is sent.
  const normalizedValue = normalizeVideoFilterValue(videoFilter.value);
  if (videoFilter.value !== normalizedValue) videoFilter.value = normalizedValue;
  if (normalizedValue && !videoFilterCatalog.includes(normalizedValue)) {
    videoFilter.value = "";
  }
  closeVideoFilterList();
  renderVideoFilterOptions();
}

let videoFilterTypeaheadBuffer = "";
let videoFilterTypeaheadTimer = null;
let videoFilterCatalog = [];
let videoFilterActiveIndex = -1;
let videoFilterIsOpen = false;

function normalizeVideoFilterToken(value) {
  const raw = String(value ?? "").trim().toUpperCase();
  const structured = raw.match(/^L?\s*(\d{1,2})\s*[_\-\s]?\s*V?\s*(\d{1,3})$/);
  if (structured) {
    return `L${structured[1].padStart(2, "0")}_V${structured[2].padStart(3, "0")}`;
  }
  return raw.replace(/[^A-Z0-9]/g, "");
}

function normalizeVideoFilterValue(value) {
  const raw = String(value ?? "").trim();
  return !raw || raw.toUpperCase() === "ALL" ? "" : raw;
}

function videoFilterOptionElements() {
  const list = $("videoFilterOptions");
  return list ? [...list.children] : [];
}

function findVideoFilterMatch(value) {
  const raw = String(value ?? "");
  const query = normalizeVideoFilterToken(raw);
  const typedDigits = raw.replace(/\D/g, "");
  if (!query && !typedDigits) return null;

  for (let index = 0; index < videoFilterCatalog.length; index += 1) {
    const optionValue = videoFilterCatalog[index];
    const normalized = normalizeVideoFilterToken(optionValue);
    const optionDigits = optionValue.replace(/\D/g, "");
    if (
      normalized.startsWith(query) ||
      normalized.includes(query) ||
      (typedDigits.length > 0 && optionDigits.startsWith(typedDigits))
    ) {
      return { index, value: optionValue };
    }
  }
  return null;
}

function setVideoFilterHint(text = "") {
  const hint = $("videoFilterHint");
  if (!hint) return;
  hint.textContent = text;
  hint.hidden = !text;
}

function updateVideoFilterHighlight(index, { scroll = true } = {}) {
  const input = $("videoFilterInput");
  const options = videoFilterOptionElements();
  videoFilterActiveIndex = Number.isInteger(index) && index >= 0 && index < options.length
    ? index
    : -1;

  options.forEach((option, optionIndex) => {
    const active = optionIndex === videoFilterActiveIndex;
    option.classList.toggle("is-active", active);
    option.setAttribute("aria-selected", String(active));
  });

  if (!input) return;
  const activeOption = options[videoFilterActiveIndex];
  input.setAttribute("aria-activedescendant", activeOption?.id || "");
  if (activeOption && scroll) activeOption.scrollIntoView?.({ block: "nearest" });
}

function renderVideoFilterOptions() {
  const list = $("videoFilterOptions");
  if (!list) return;
  list.replaceChildren();

  const entries = [{ value: "", label: "All videos" }, ...videoFilterCatalog.map((value) => ({
    value,
    label: value,
  }))];
  entries.forEach((entry, index) => {
    const option = document.createElement("button");
    option.type = "button";
    option.id = `video-filter-option-${index}`;
    option.className = "video-filter-option";
    option.dataset.videoValue = entry.value;
    option.setAttribute("role", "option");
    option.setAttribute("aria-selected", "false");
    option.textContent = entry.label;
    option.addEventListener("mousedown", (event) => event.preventDefault());
    option.addEventListener("click", () => commitVideoFilter(entry.value));
    list.appendChild(option);
  });

  const committedValue = $("videoFilter")?.value || "";
  const selectedIndex = entries.findIndex((entry) => entry.value === committedValue);
  const match = videoFilterTypeaheadBuffer
    ? findVideoFilterMatch(videoFilterTypeaheadBuffer)
    : null;
  updateVideoFilterHighlight(match ? match.index + 1 : selectedIndex, { scroll: false });
}

function openVideoFilterList({ preserveQuery = false } = {}) {
  const combo = $("videoFilterCombo");
  const input = $("videoFilterInput");
  const list = $("videoFilterOptions");
  const toggle = $("videoFilterToggle");
  if (!combo || !input || !list) return;

  if (!videoFilterIsOpen) {
    videoFilterIsOpen = true;
    videoFilterTypeaheadBuffer = preserveQuery ? input.value : "";
    if (!preserveQuery) input.value = "";
    setVideoFilterHint("");
    list.hidden = false;
    combo.classList.add("is-open");
    input.setAttribute("aria-expanded", "true");
    toggle?.setAttribute("aria-expanded", "true");
    renderVideoFilterOptions();
  }
  input.focus();
}

function closeVideoFilterList() {
  const combo = $("videoFilterCombo");
  const input = $("videoFilterInput");
  const list = $("videoFilterOptions");
  const toggle = $("videoFilterToggle");
  videoFilterIsOpen = false;
  videoFilterTypeaheadBuffer = "";
  if (videoFilterTypeaheadTimer) clearTimeout(videoFilterTypeaheadTimer);
  videoFilterTypeaheadTimer = null;
  if (list) list.hidden = true;
  combo?.classList.remove("is-open");
  input?.setAttribute("aria-expanded", "false");
  toggle?.setAttribute("aria-expanded", "false");
  input?.setAttribute("aria-activedescendant", "");
  if (input) {
    input.value = "";
    input.placeholder = $("videoFilter")?.value || "All videos";
  }
  setVideoFilterHint("");
  videoFilterActiveIndex = -1;
}

function commitVideoFilter(value) {
  const videoFilter = $("videoFilter");
  if (!videoFilter) return false;
  const nextValue = normalizeVideoFilterValue(value);
  const changed = videoFilter.value !== nextValue;
  videoFilter.value = nextValue;
  closeVideoFilterList();
  if (changed) {
    videoFilter.dispatchEvent(new Event("change", { bubbles: true }));
  }
  return changed;
}

function isDirectVideoResult(item = selectedResult) {
  return item?.direct_video === true || directVideoState?.mode === "direct_video";
}

function clearDirectVideoState({ clearResults = false } = {}) {
  const wasDirect = directVideoState?.mode === "direct_video" || selectedResult?.direct_video === true;
  directVideoState = null;
  if (wasDirect || clearResults) {
    selectedResult = null;
    selectedFrameId = null;
    lastResolvedPosition = null;
    lastExactExtraction = null;
    clearTopNeighborStrip();
  }
  if (clearResults) lastSearchData = null;
}

function normalizeDirectVideoFrame(data, videoId) {
  const raw = data?.frame || data?.keyframe || data || {};
  const normalizedVideoId = String(raw.video_id || data?.video_id || videoId || "").trim();
  const frameUid = String(raw.frame_uid || raw.frame_id || data?.frame_uid || "").trim();
  const sourceFrameIdx = Number(raw.source_frame_idx ?? data?.source_frame_idx);
  const timestampMs = Number(raw.timestamp_ms ?? data?.timestamp_ms);
  if (!normalizedVideoId || normalizedVideoId !== videoId) {
    throw new Error("Direct video response has an unexpected video_id.");
  }
  if (!/^.+:\d+$/.test(frameUid) || frameUid !== `${videoId}:${sourceFrameIdx}`) {
    throw new Error("Direct video response failed canonical frame_uid validation.");
  }
  if (!Number.isSafeInteger(sourceFrameIdx) || sourceFrameIdx < 0 ||
      !Number.isFinite(timestampMs) || timestampMs < 0) {
    throw new Error("Direct video response has invalid canonical frame metadata.");
  }

  return {
    direct_video: true,
    direct_mode: "first_keyframe",
    video_id: videoId,
    frame_id: frameUid,
    frame_uid: frameUid,
    source_frame_idx: sourceFrameIdx,
    frame_idx: sourceFrameIdx,
    timestamp_ms: timestampMs,
    shot_id: raw.shot_id,
    keyframe_path: raw.keyframe_path,
    image_url: raw.image_url || data?.image_url || `/frames/${encodeURIComponent(frameUid)}/image`,
    thumbnail_url: raw.thumbnail_url || data?.thumbnail_url || thumbnailUrl(frameUid),
    image_available: raw.image_available ?? data?.image_available,
    image_status: raw.image_status ?? data?.image_status,
    image_reason: raw.image_reason ?? data?.image_reason,
    video_url: raw.video_url || data?.video_url || `/videos/${encodeURIComponent(videoId)}/stream`,
    video_available: raw.video_available ?? data?.video_available,
    video_stream_available: raw.video_stream_available ?? data?.video_stream_available,
    video_status: raw.video_status ?? data?.video_status,
    video_stream_status: raw.video_stream_status ?? data?.video_stream_status,
    video_stream_reason: raw.video_stream_reason ?? data?.video_stream_reason,
    video_backend: raw.video_backend ?? data?.video_backend,
    video_provenance_status: raw.video_provenance_status ?? data?.video_provenance_status,
  };
}

async function enterDirectVideoMode(videoId) {
  const requestId = ++directVideoRequestSequence;
  clearDirectVideoState({ clearResults: true });
  hideDetail();
  directVideoState = {
    mode: "direct_video",
    status: "loading",
    video_id: videoId,
    frame: null,
  };
  renderResultsView();

  try {
    const data = await api(`/videos/${encodeURIComponent(videoId)}/first-keyframe`);
    if (requestId !== directVideoRequestSequence || $("videoFilter")?.value !== videoId) return;
    const frame = normalizeDirectVideoFrame(data, videoId);
    directVideoState = {
      mode: "direct_video",
      status: "ready",
      video_id: videoId,
      frame,
    };
    selectedResult = frame;
    selectedFrameId = frame.frame_uid;
    renderResultsView();
  } catch (err) {
    if (requestId !== directVideoRequestSequence || $("videoFilter")?.value !== videoId) return;
    directVideoState = {
      mode: "direct_video",
      status: "error",
      video_id: videoId,
      frame: null,
      error: err?.message || "First canonical keyframe unavailable.",
    };
    renderResultsView();
  }
}

async function handleVideoFilterChange() {
  const videoFilter = $("videoFilter");
  const videoId = normalizeVideoFilterValue(videoFilter?.value);
  if (videoFilter && videoFilter.value !== videoId) videoFilter.value = videoId;
  ++directVideoRequestSequence;
  if (!videoId) {
    clearDirectVideoState({ clearResults: true });
    hideDetail();
    renderResultsView();
    return;
  }

  const { activeStages } = collectActiveStagePayloads();
  if (activeStages.length) {
    // A real stage query keeps the existing filter-only semantics.  Selecting
    // a video must not silently switch an active retrieval into direct mode.
    const wasDirect = directVideoState?.mode === "direct_video" || selectedResult?.direct_video === true;
    clearDirectVideoState();
    if (wasDirect) hideDetail();
    renderResultsView();
    return;
  }

  await enterDirectVideoMode(videoId);
}

function setupVideoFilterTypeahead() {
  const combo = $("videoFilterCombo");
  const input = $("videoFilterInput");
  const list = $("videoFilterOptions");
  const toggle = $("videoFilterToggle");
  if (!combo || !input || !list || combo.dataset.typeaheadReady === "true") return;
  combo.dataset.typeaheadReady = "true";

  input.addEventListener("input", () => {
    if (!videoFilterIsOpen) openVideoFilterList({ preserveQuery: true });
    const rawInput = String(input.value ?? "").trim();
    if (!rawInput || rawInput.toUpperCase() === "ALL") {
      const changed = commitVideoFilter("");
      if (!changed) void handleVideoFilterChange();
      return;
    }
    videoFilterTypeaheadBuffer = input.value;
    const match = findVideoFilterMatch(videoFilterTypeaheadBuffer);
    updateVideoFilterHighlight(match ? match.index + 1 : -1);
    setVideoFilterHint(
      videoFilterTypeaheadBuffer.trim() && !match
        ? "No matching video"
        : "",
    );
    if (videoFilterTypeaheadTimer) clearTimeout(videoFilterTypeaheadTimer);
    videoFilterTypeaheadTimer = setTimeout(() => {
      videoFilterTypeaheadBuffer = input.value;
    }, 800);
  });

  input.addEventListener("click", () => openVideoFilterList());
  input.addEventListener("focus", () => openVideoFilterList());
  toggle?.addEventListener("click", () => openVideoFilterList());

  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      commitVideoFilter("");
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const options = videoFilterOptionElements();
      const activeOption = options[videoFilterActiveIndex];
      if (activeOption) commitVideoFilter(activeOption.dataset.videoValue || "");
      else closeVideoFilterList();
      return;
    }
    const options = videoFilterOptionElements();
    if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      if (!videoFilterIsOpen) openVideoFilterList();
      if (!options.length) return;
      let next = videoFilterActiveIndex;
      if (event.key === "ArrowDown") next = next < 0 ? 0 : Math.min(next + 1, options.length - 1);
      if (event.key === "ArrowUp") next = next < 0 ? options.length - 1 : Math.max(next - 1, 0);
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = options.length - 1;
      updateVideoFilterHighlight(next);
    }
  });

  document.addEventListener("pointerdown", (event) => {
    if (videoFilterIsOpen && !combo.contains(event.target)) closeVideoFilterList();
  });
}

function selectedVisualIndexes() {
  return [...FIXED_VISUAL_INDEXES];
}

const CHANNEL_STATUS_NAMES = ["text", "ocr", "asr", "image", "object"];

function renderChannelStatus(statuses) {
  const root = $("channelStatus");
  if (!root) return;
  root.innerHTML = "";
  const rawSource = statuses && typeof statuses === "object" ? statuses : {};
  const source = Object.fromEntries(
    Object.entries(rawSource)
      .filter(([name]) => !["trake", "qa"].includes(String(name).toLowerCase()))
  );
  root.hidden = false;
  for (const name of CHANNEL_STATUS_NAMES) {
    const raw = source[name];
    const rawStatus = typeof raw === "string"
      ? raw
      : raw?.status || (raw?.available ? "ready" : "unavailable");
    const status = String(rawStatus || "unavailable");
    const ready = raw?.available === true || status === "ready" || status === "available";
    const item = document.createElement("span");
    item.className = `channel-status ${ready ? "channel-ready" : "channel-unavailable"}`;
    item.textContent = name;
    item.dataset.status = status;
    item.setAttribute("aria-label", `${name}: ${ready ? "ready" : "unavailable"}`);
    const reason = typeof raw === "object" ? raw?.reason : "";
    item.title = [status, reason].filter(Boolean).join(": ");
    root.appendChild(item);
  }
}

/* ==========================================================================
   Search Normalization & API Execution
   ========================================================================== */

function normalizeKisResponse(data) {
  if (!data.results || !data.results.length || !data.results[0].frame_uid) return data;
  return {
    ...data,
    total_found: data.results.length,
    executed_channels: data.executed_channels || [],
    unavailable_channels: data.unavailable_channels || {},
    channel_status: data.channel_status || {},
    results: data.results.map((result) => ({
      frame_id: result.frame_uid,
      frame_uid: result.frame_uid,
      image_url: `/frames/${encodeURIComponent(result.frame_uid)}/image`,
      thumbnail_url: result.thumbnail_url || thumbnailUrl(result.frame_uid),
      video_id: result.video_id,
      frame_idx: result.source_frame_idx,
      source_frame_idx: result.source_frame_idx,
      timestamp_ms: result.timestamp_ms,
      shot_id: result.shot_id,
      keyframe_path: result.keyframe_path,
      rank: result.rank,
      rank_in_stage: result.rank,
      final_score: result.rerank_score ?? result.fused_score ?? result.score ?? 0,
      signal_scores: result.channel_scores || {},
      evidence: Array.isArray(result.evidence) ? result.evidence : [],
      image_available: result.image_available,
      image_status: result.image_status,
      image_reason: result.image_reason,
      video_url: result.video_url,
      video_stream_available: result.video_stream_available,
      video_stream_status: result.video_stream_status,
      video_backend: result.video_backend,
      video_bytes: result.video_bytes,
      video_range_capable: result.video_range_capable,
      video_provenance_status: result.video_provenance_status,
      stage_id: "S1",
    })),
  };
}

function normalizeStageResult(result, defaultStage = "S1") {
  const frameUid = result.frame_uid || result.frame_id;
  return {
    frame_id: frameUid,
    frame_uid: frameUid,
    image_url: result.image_url || `/frames/${encodeURIComponent(frameUid)}/image`,
    thumbnail_url: result.thumbnail_url || thumbnailUrl(frameUid),
    video_id: result.video_id,
    frame_idx: result.source_frame_idx,
    source_frame_idx: result.source_frame_idx,
    timestamp_ms: result.timestamp_ms,
    shot_id: result.shot_id,
    keyframe_path: result.keyframe_path,
    rank: result.rank_in_stage ?? result.rank,
    rank_in_stage: result.rank_in_stage ?? result.rank,
    final_rank: result.final_rank,
    final_score: result.fusion_score ?? result.fused_score ?? result.score ?? 0,
    signal_scores: result.channel_scores || {},
    evidence: Array.isArray(result.evidence) ? result.evidence : [],
    image_available: result.image_available,
    image_status: result.image_status,
    image_reason: result.image_reason,
    video_url: result.video_url,
    video_stream_available: result.video_stream_available,
    video_stream_status: result.video_stream_status,
    video_backend: result.video_backend,
    video_bytes: result.video_bytes,
    video_range_capable: result.video_range_capable,
    video_provenance_status: result.video_provenance_status,
    stage_id: result.stage_id || defaultStage,
    bundle_id: result.bundle_id || null,
    bundle_rank: result.bundle_rank ?? null,
    bundle_score: result.bundle_score ?? null,
    ...(Object.prototype.hasOwnProperty.call(result, "bundle_temporal_enabled") ||
      Object.prototype.hasOwnProperty.call(result, "temporal_enabled")
      ? { bundle_temporal_enabled: result.bundle_temporal_enabled ?? result.temporal_enabled ?? null }
      : {}),
    qa_answer: result.qa_answer || null,
    track_id: result.track_id || null,
    track_rank: result.track_rank ?? null,
    track_score: result.track_score ?? null,
    event_step: result.event_step ?? null,
    selection_kind: result.selection_kind || null,
    delta_from_previous_ms: result.delta_from_previous_ms ?? null,
  };
}

function normalizeStageResponse(data) {
  const stageResults = {};
  const stageIds = stageIdsFromData(data);
  for (const stageId of stageIds) {
    stageResults[stageId] = (data.stage_results?.[stageId] || [])
      .map((r) => normalizeStageResult(r, stageId));
  }
  const fused = [];
  const seen = new Set();
  const responseItems = data.results || data.fused_results || [];
  for (const raw of responseItems) {
    const item = normalizeStageResult(raw, raw.stage_id || "S1");
    const frameId = frameUidOf(item);
    const identity = `${item.stage_id || ""}\u0000${item.bundle_id || ""}\u0000${frameId}`;
    if (!frameId || seen.has(identity)) continue;
    seen.add(identity);
    fused.push(item);
  }
  // Compatibility fallback for older backends that did not return a merged
  // union. New backends provide the score/video/stage ordering in `results`.
  for (const stageId of stageIds) {
    for (const item of stageResults[stageId] || []) {
      const frameId = frameUidOf(item);
      const identity = `${item.stage_id || stageId}\u0000${item.bundle_id || ""}\u0000${frameId}`;
      if (!frameId || seen.has(identity)) continue;
      seen.add(identity);
      fused.push(item);
    }
  }
  fused.forEach((item, index) => { item.final_rank = index + 1; });
  return {
    ...data,
    stage_ids: stageIds,
    stage_results: stageResults,
    fused_results: fused,
    results: fused,
    total_found: fused.length,
  };
}

function stageIdsFromData(data) {
  const declared = Array.isArray(data?.stage_ids) ? data.stage_ids : [];
  const keys = Object.keys(data?.stage_results || {});
  return [...new Set([...declared, ...keys])]
    .filter((stageId) => /^S[1-5]$/.test(String(stageId)))
    .sort((a, b) => stageNumber(a) - stageNumber(b));
}

function stageNumber(stageId) {
  const match = String(stageId || "S1").match(/^S(\d+)$/i);
  return match ? Number(match[1]) : Number.MAX_SAFE_INTEGER;
}

function resultScore(item) {
  const score = Number(item?.final_score ?? item?.fusion_score ?? item?.score ?? 0);
  return Number.isFinite(score) ? score : 0;
}

function allHitsSpacingMs() {
  const value = Number($("allHitsSpacing")?.value ?? ALL_HITS_DEFAULT_MIN_GAP_MS);
  return Number.isFinite(value) && value >= 0 ? Math.trunc(value) : ALL_HITS_DEFAULT_MIN_GAP_MS;
}

function compareAllHits(a, b) {
  const scoreDelta = resultScore(b) - resultScore(a);
  if (Math.abs(scoreDelta) > 1e-12) return scoreDelta;
  const stageDelta = stageNumber(a?.stage_id) - stageNumber(b?.stage_id);
  if (stageDelta) return stageDelta;
  const rankDelta = Number(a?.all_hits_rank ?? a?.rank_in_stage ?? a?.rank ?? Number.MAX_SAFE_INTEGER) -
    Number(b?.all_hits_rank ?? b?.rank_in_stage ?? b?.rank ?? Number.MAX_SAFE_INTEGER);
  if (rankDelta) return rankDelta;
  return frameUidOf(a).localeCompare(frameUidOf(b));
}

function diversifyAllHits(items, minGapMs = ALL_HITS_DEFAULT_MIN_GAP_MS) {
  const gapMs = Math.max(0, Math.trunc(Number(minGapMs) || 0));
  const ordered = [...(items || [])].sort(compareAllHits);
  const bundleGroups = new Map();
  const unbundled = [];
  for (const item of ordered) {
    const bundleId = String(item?.bundle_id || "").trim();
    if (!bundleId) {
      unbundled.push(item);
      continue;
    }
    if (!bundleGroups.has(bundleId)) bundleGroups.set(bundleId, []);
    bundleGroups.get(bundleId).push(item);
  }
  const selectedBundleItems = [];
  const selectedBundlePositions = new Map();
  for (const group of bundleGroups.values()) {
    const videoId = String(group[0]?.video_id || "");
    const positions = group
      .map((item) => [String(item?.stage_id || "S1"), Number(item?.timestamp_ms)])
      .filter(([, timestamp]) => Number.isFinite(timestamp));
    const previous = selectedBundlePositions.get(videoId) || [];
    const duplicate = gapMs > 0 && positions.some(([stageId, timestamp]) =>
      previous.some(([previousStage, previousTimestamp]) =>
        stageId === previousStage && Math.abs(timestamp - previousTimestamp) < gapMs
      )
    );
    if (duplicate) continue;
    selectedBundleItems.push(...group);
    selectedBundlePositions.set(videoId, [...previous, ...positions]);
  }

  const selected = [];
  const timestamps = new Map();
  const exactSeen = new Set();

  for (const item of [...selectedBundleItems, ...unbundled]) {
    const frameId = frameUidOf(item);
    if (!frameId) continue;
    const stageId = item.stage_id || "S1";
    const bundleId = String(item?.bundle_id || "").trim();
    const exactKey = `${bundleId}\u0000${stageId}\u0000${frameId}`;
    if (exactSeen.has(exactKey)) continue;
    exactSeen.add(exactKey);

    const videoId = item.video_id || "";
    const rawTimestamp = Number(item.timestamp_ms);
    const timestamp = Number.isFinite(rawTimestamp) ? rawTimestamp : null;
    const positionKey = `${videoId}\u0000${stageId}`;
    const previous = timestamps.get(positionKey) || [];
    if (gapMs > 0 && timestamp !== null && previous.some((value) => Math.abs(timestamp - value) < gapMs)) {
      continue;
    }

    selected.push(item);
    if (timestamp !== null) timestamps.set(positionKey, [...previous, timestamp]);
  }

  return selected.map((item, index) => ({
    ...item,
    all_hits_rank: index + 1,
    final_rank: index + 1,
  }));
}

function compareStageCandidates(a, b) {
  const stageDelta = stageNumber(a?.stage_id) - stageNumber(b?.stage_id);
  if (stageDelta) return stageDelta;
  const scoreDelta = resultScore(b) - resultScore(a);
  if (Math.abs(scoreDelta) > 1e-12) return scoreDelta;
  const rankDelta = Number(a?.rank_in_stage ?? a?.rank ?? Number.MAX_SAFE_INTEGER) -
    Number(b?.rank_in_stage ?? b?.rank ?? Number.MAX_SAFE_INTEGER);
  if (rankDelta) return rankDelta;
  return frameUidOf(a).localeCompare(frameUidOf(b));
}

function stageResultItems(data) {
  if (Array.isArray(data?.all_hits)) {
    return data.all_hits.filter((item) => Boolean(frameUidOf(item)));
  }
  const source = Array.isArray(data?.results) && data.results.length
    ? data.results
    : stageIdsFromData(data).flatMap((stageId) =>
      (data?.stage_results?.[stageId] || []).map((item) => ({ ...item, stage_id: item.stage_id || stageId }))
    );
  const seen = new Set();
  return source.filter((item) => {
    const frameId = frameUidOf(item);
    const identity = `${item.stage_id || ""}\u0000${item.bundle_id || ""}\u0000${frameId}`;
    if (!frameId || seen.has(identity)) return false;
    seen.add(identity);
    return true;
  });
}

function normalizeTrakeResponse(data) {
  const stageIds = Array.isArray(data.stage_ids) && data.stage_ids.length
    ? data.stage_ids
    : Object.keys(data.stage_results || {});
  const tracks = (data.tracks || []).map((track) => ({
    ...track,
    stages: (track.stages || []).map((item, index) => normalizeStageResult(
      {
        ...item,
        stage_id: item.stage_id || stageIds[index] || `S${index + 1}`,
        track_id: track.track_id,
        track_rank: track.track_rank,
        track_score: track.score,
      },
      stageIds[index] || "S1",
    )),
  }));
  const results = tracks.flatMap((track) => track.stages);
  return {
    ...data,
    mode: "trake",
    stage_ids: stageIds,
    tracks,
    results,
    total_found: tracks.length,
  };
}

function hasTemporalModeField(item) {
  return Boolean(item) && (
    Object.prototype.hasOwnProperty.call(item, "bundle_temporal_enabled") ||
    Object.prototype.hasOwnProperty.call(item, "temporal_enabled")
  );
}

function temporalModeValue(item) {
  if (!hasTemporalModeField(item)) return null;
  const raw = Object.prototype.hasOwnProperty.call(item, "bundle_temporal_enabled")
    ? item.bundle_temporal_enabled
    : item.temporal_enabled;
  return raw == null ? null : Boolean(raw);
}

function temporalModeValueOr(item, fallback) {
  return hasTemporalModeField(item)
    ? temporalModeValue(item)
    : fallback === undefined
    ? null
    : fallback == null
    ? null
    : Boolean(fallback);
}

function normalizeBundleResponse(data) {
  const stageIds = Array.isArray(data.stage_ids) && data.stage_ids.length
    ? data.stage_ids
    : Object.keys(data.stage_results || {});
  const responseTemporalMode = temporalModeValue(data);
  const normalizeAllHitItems = (items) => (Array.isArray(items) ? items : []).map((item) => ({
    ...normalizeStageResult(item, item.stage_id || "S1"),
    all_hits_rank: item.all_hits_rank ?? null,
    // Keep bundle identity at the UI boundary.  Without this, a complete
    // backend bundle is flattened into unrelated cards and cannot be queued
    // or exported as one Trake chain.
    bundle_id: item.bundle_id ?? null,
    bundle_rank: item.bundle_rank ?? null,
    bundle_score: item.bundle_score ?? null,
    ...(hasTemporalModeField(item) || hasTemporalModeField(data)
      ? { bundle_temporal_enabled: temporalModeValueOr(item, responseTemporalMode) }
      : {}),
  }));
  const allHitsRaw = normalizeAllHitItems(
    data.all_hits_raw || data.all_hits || stageIds.flatMap((stageId) => (
      data.stage_results?.[stageId] || []
    ).map((item) => ({ ...item, stage_id: item.stage_id || stageId }))),
  );
  const allHitsEligibleRaw = normalizeAllHitItems(
    data.all_hits_eligible_raw || data.all_hits || allHitsRaw,
  );
  const allHits = normalizeAllHitItems(data.all_hits || allHitsEligibleRaw);
  const bundles = (data.bundles || []).map((bundle, index) => {
    const bundleScore = Number(bundle.bundle_score ?? bundle.score ?? 0);
    const normalizedStages = (bundle.stages || []).map((item, stageIndex) => {
      const normalized = normalizeStageResult(
        {
          ...item,
          stage_id: item.stage_id || stageIds[stageIndex] || `S${stageIndex + 1}`,
        },
        stageIds[stageIndex] || "S1",
      );
      normalized.bundle_id = bundle.bundle_id || null;
      normalized.bundle_rank = bundle.bundle_rank ?? index + 1;
      normalized.bundle_score = Number.isFinite(bundleScore) ? bundleScore : 0;
      if (hasTemporalModeField(item) || hasTemporalModeField(bundle) || hasTemporalModeField(data)) {
        normalized.bundle_temporal_enabled = temporalModeValueOr(
          item,
          temporalModeValueOr(bundle, responseTemporalMode),
        );
      }
      return normalized;
    });
    return {
      ...bundle,
      bundle_rank: bundle.bundle_rank ?? index + 1,
      bundle_score: Number.isFinite(bundleScore) ? bundleScore : 0,
      score: Number.isFinite(bundleScore) ? bundleScore : 0,
      stages: normalizedStages,
    };
  });
  const results = bundles.flatMap((bundle) => bundle.stages);
  return {
    ...data,
    mode: data.mode === "all_hits" ? "all_hits" : "bundle",
    stage_ids: stageIds,
    active_stage_ids: data.active_stage_ids || stageIds,
    bundles,
    all_hits_raw: allHitsRaw,
    all_hits_eligible_raw: allHitsEligibleRaw,
    all_hits: allHits,
    results: data.mode === "all_hits" ? allHits : results,
    fused_results: data.mode === "all_hits" ? allHits : results,
    total_found: data.mode === "all_hits" ? allHits.length : bundles.length,
  };
}

function renderAsrEvidence(result) {
  const root = $("asrEvidence");
  if (!root) return;
  const entries = (Array.isArray(result?.evidence) ? result.evidence : [])
    .filter((item) => item && item.channel === "asr")
    .slice(0, 2);
  if (!entries.length) {
    root.hidden = true;
    root.textContent = "";
    return;
  }
  const lines = entries.map((item) => {
    const metadata = item.metadata || {};
    const segmentId = metadata.segment_id || "—";
    const transcript = metadata.phowhisper_raw || metadata.whisper_v3_raw || metadata.raw_transcript || "—";
    const model = metadata.source_model || "—";
    const interval = metadata.segment_start_ms != null && metadata.segment_end_ms != null
      ? `${metadata.segment_start_ms}–${metadata.segment_end_ms} ms`
      : "interval unavailable";
    return `ASR · ${model} · segment_id=${segmentId} · ${interval}\n${String(transcript).slice(0, 240)}`;
  });
  root.hidden = false;
  root.textContent = lines.join("\n\n");
}

function channelToggleButton(stageId, channel, scope = document) {
  const stage = String(stageId).toLowerCase();
  return scope.querySelector(
    `.stage-channel-tabs[data-stage="${stage}"] [data-channel-toggle="${channel}"]`,
  );
}

function defaultChannelEnabled(stageId, channel) {
  // Every configured stage is ready for the primary text query. Optional
  // channels remain opt-in so adding S3-S5 does not silently invoke a model.
  return channel === "text";
}

function isChannelEnabled(stageId, channel, scope = document) {
  const toggle = channelToggleButton(stageId, channel, scope);
  return toggle
    ? toggle.getAttribute("aria-pressed") === "true"
    : defaultChannelEnabled(stageId, channel);
}

function setChannelToggleState(stageId, channel, enabled, scope = document) {
  const toggle = channelToggleButton(stageId, channel, scope);
  if (!toggle) return;

  const label = STAGE_CHANNEL_LABELS[channel] || channel;
  toggle.setAttribute("aria-pressed", String(enabled));
  toggle.setAttribute("aria-label", `${label} channel ${enabled ? "ON" : "OFF"}`);
  toggle.classList.toggle("is-enabled", enabled);
  toggle.textContent = enabled ? "ON" : "OFF";
  toggle.title = `${enabled ? "Disable" : "Enable"} ${label} channel`;

  const stage = String(stageId).toLowerCase();
  const panel = scope.querySelector(
    `.channel-input-panels[data-stage="${stage}"] .channel-panel[data-channel="${channel}"]`,
  );
  panel?.classList.toggle("channel-disabled", !enabled);
  setStageInputAvailability(stageId, channel, enabled, scope);
}

function setStageInputAvailability(stageId, channel, enabled, scope = document) {
  const stage = String(stageId).toLowerCase();
  const panel = scope.querySelector(
    `.channel-input-panels[data-stage="${stage}"] .channel-panel[data-channel="${channel}"]`,
  );
  if (!panel) return;

  const inputs = panel.querySelectorAll("textarea, input:not([type=hidden]), select");
  inputs.forEach((input) => {
    // Keep an existing draft in the DOM, but make an OFF channel impossible
    // to edit or accidentally submit through keyboard/paste/drop events.
    input.disabled = !enabled;
    if ("readOnly" in input) input.readOnly = !enabled;
    input.setAttribute("aria-disabled", String(!enabled));
    input.setAttribute("data-channel-enabled", String(enabled));
    input.dataset.channelEnabled = String(enabled);
  });
  if (channel === "object") {
    panel.querySelectorAll(".object-query-rows, .object-query-preview").forEach((node) => {
      node.classList.toggle("channel-disabled", !enabled);
    });
  }
}

function stageImageInput(stageId, scope = document) {
  const stage = String(stageId).toLowerCase();
  return scope.querySelector?.(`#${stage}Image`) || $(`${stage}Image`);
}

function stageImageUrlInput(stageId, scope = document) {
  const stage = String(stageId).toLowerCase();
  return scope.querySelector?.(`[data-image-url-input="${stage}"]`) || $(`${stage}ImageUrl`);
}

function stageImageFile(stageId, scope = document) {
  return stageImageInput(stageId, scope)?.files?.[0] || null;
}

function stageImageUrl(stageId, scope = document) {
  return String(stageImageUrlInput(stageId, scope)?.value || "").trim();
}

function isPrivateImageHost(hostname) {
  const host = String(hostname || "").toLowerCase().replace(/^\[|\]$/g, "");
  if (
    !host ||
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host.endsWith(".local")
  ) return true;
  if (host.includes(":") && (
    host === "::1" || host.startsWith("fc") || host.startsWith("fd") || host.startsWith("fe80:")
  )) return true;

  const octets = host.split(".");
  if (octets.length !== 4 || octets.some((part) => !/^\d{1,3}$/.test(part))) return false;
  const [first, second] = octets.map((part) => Number(part));
  if (octets.some((part) => Number(part) > 255)) return true;
  return first === 0 || first === 10 || first === 127 ||
    (first === 169 && second === 254) ||
    (first === 172 && second >= 16 && second <= 31) ||
    (first === 192 && second === 168);
}

function validateImageQueryUrl(rawValue) {
  const value = String(rawValue || "").trim();
  if (!value) return { valid: true, value: "", message: "" };
  if (value.length > MAX_QUERY_IMAGE_URL_LENGTH) {
    return { valid: false, value, message: "Image URL is too long." };
  }

  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    return { valid: false, value, message: "Enter a valid HTTP or HTTPS image URL." };
  }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    return { valid: false, value, message: "Image URL must use HTTP or HTTPS." };
  }
  if (!parsed.hostname || parsed.username || parsed.password || isPrivateImageHost(parsed.hostname)) {
    return { valid: false, value, message: "Use a public HTTP(S) image URL without credentials." };
  }
  return { valid: true, value: parsed.href, message: "" };
}

function setImageQueryStatus(stageId, message, isError = false, scope = document) {
  const stage = String(stageId).toLowerCase();
  const status = scope.querySelector?.(`#${stage}ImageStatus`) || $(`${stage}ImageStatus`);
  if (!status) return;
  status.textContent = message || "";
  status.classList.toggle("is-error", Boolean(isError));
}

function stageImagePreview(stageId, scope = document) {
  return scope.querySelector?.(`[data-image-preview="${String(stageId).toLowerCase()}"]`);
}

function stageImageClearButton(stageId, scope = document) {
  return scope.querySelector?.(`[data-image-clear="${String(stageId).toLowerCase()}"]`);
}

function revokeStageImagePreview(preview) {
  if (!preview) return;
  if (preview.dataset.objectUrl && typeof URL.revokeObjectURL === "function") {
    URL.revokeObjectURL(preview.dataset.objectUrl);
  }
  delete preview.dataset.objectUrl;
  preview.onload = null;
  preview.onerror = null;
  preview.removeAttribute("src");
  preview.hidden = true;
}

function clearStageImageFile(stageId, scope = document) {
  const input = stageImageInput(stageId, scope);
  if (!input) return;
  input.value = "";
  if (input) {
    try {
      if (typeof DataTransfer === "function") {
        const transfer = new DataTransfer();
        input.files = transfer.files;
      }
    } catch {
      // Some browsers expose input.files as read-only; clearing value is the
      // compatible fallback for the normal file picker path.
    }
  }
}

function clearStageImage(stageId, scope = document) {
  clearStageImageFile(stageId, scope);
  const urlInput = stageImageUrlInput(stageId, scope);
  const preview = stageImagePreview(stageId, scope);
  const clear = stageImageClearButton(stageId, scope);
  if (urlInput) urlInput.value = "";
  revokeStageImagePreview(preview);
  if (clear) clear.disabled = true;
  setImageQueryStatus(stageId, "", false, scope);
}

function assignStageImageFile(input, file) {
  if (!input) return false;
  try {
    if (typeof DataTransfer === "function") {
      const transfer = new DataTransfer();
      if (file) transfer.items.add(file);
      input.files = transfer.files;
      return true;
    }
  } catch {
    // Fall through to the testable/non-browser input fallback below.
  }
  try {
    Object.defineProperty(input, "files", {
      configurable: true,
      value: file ? [file] : [],
    });
    return true;
  } catch {
    return false;
  }
}

function pastedStageImageFile(stageId, item) {
  const blob = item?.getAsFile?.();
  if (!blob) return null;
  const type = String(blob.type || item.type || "").toLowerCase();
  const extension = type === "image/jpeg" ? "jpg" : type === "image/png" ? "png" : "img";
  const name = String(blob.name || "").trim() || `pasted-${String(stageId).toLowerCase()}.${extension}`;
  if (typeof File === "function") return new File([blob], name, { type });
  return blob;
}

function applyStageImageFile(stageId, input, file, scope, source = "file") {
  if (!isChannelEnabled(stageId, "image", scope)) {
    setImageQueryStatus(stageId, "Enable Image channel before choosing or pasting an image.", true, scope);
    return false;
  }
  const type = String(file?.type || "").toLowerCase();
  if (!QUERY_IMAGE_TYPES.includes(type)) {
    clearStageImage(stageId, scope);
    setImageQueryStatus(
      stageId,
      source === "paste" ? "Pasted image must be JPEG or PNG." : "Choose a JPEG or PNG image.",
      true,
      scope,
    );
    return false;
  }
  if (!Number.isFinite(Number(file?.size)) || file.size <= 0 || file.size > MAX_QUERY_IMAGE_BYTES) {
    clearStageImage(stageId, scope);
    setImageQueryStatus(stageId, "Image must be between 1 byte and 10 MB.", true, scope);
    return false;
  }
  if (source === "paste" || source === "drop") {
    if (!assignStageImageFile(input, file)) {
      setImageQueryStatus(stageId, "Could not attach the pasted image; use the file picker.", true, scope);
      return false;
    }
  }

  const urlInput = stageImageUrlInput(stageId, scope);
  const preview = stageImagePreview(stageId, scope);
  const clear = stageImageClearButton(stageId, scope);
  if (urlInput) urlInput.value = "";
  revokeStageImagePreview(preview);
  if (preview && typeof URL.createObjectURL === "function") {
    const objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    preview.dataset.objectUrl = objectUrl;
    preview.alt = String(file.name || "Selected image");
    preview.hidden = false;
  }
  if (clear) clear.disabled = false;
  const label = source === "paste" ? `Pasted image ${file.name || "ready"}`
    : source === "drop" ? `Dropped image ${file.name || "ready"}`
      : file.name || "Image";
  setImageQueryStatus(stageId, `${label} ready for the next search.`, false, scope);
  return true;
}

function applyStageImageUrl(stageId, input, rawUrl, scope) {
  if (!isChannelEnabled(stageId, "image", scope)) {
    setImageQueryStatus(stageId, "Enable Image channel before typing or pasting an image URL.", true, scope);
    return false;
  }
  const validation = validateImageQueryUrl(rawUrl);
  const preview = stageImagePreview(stageId, scope);
  const clear = stageImageClearButton(stageId, scope);
  if (!validation.valid) {
    clearStageImageFile(stageId, scope);
    revokeStageImagePreview(preview);
    if (clear) clear.disabled = true;
    setImageQueryStatus(stageId, validation.message, true, scope);
    return false;
  }

  clearStageImageFile(stageId, scope);
  revokeStageImagePreview(preview);
  if (preview && validation.value) {
    // Keep preview rendering browser-local. The validated URL is serialized
    // separately as image_url metadata; it is never copied into text.
    preview.referrerPolicy = "no-referrer";
    preview.alt = "Image URL preview";
    preview.onload = () => {
      preview.hidden = false;
      setImageQueryStatus(
        stageId,
        "Image URL ready for the next search.",
        false,
        scope,
      );
    };
    preview.onerror = () => {
      preview.hidden = true;
      setImageQueryStatus(stageId, "The URL did not resolve to a displayable image.", true, scope);
    };
    preview.src = validation.value;
    preview.hidden = false;
  }
  if (clear) clear.disabled = !validation.value;
  if (validation.value) {
    setImageQueryStatus(
      stageId,
      "Image URL ready for the next search.",
      false,
      scope,
    );
  }
  return Boolean(validation.value);
}

function clipboardImageUrl(event) {
  const clipboard = event?.clipboardData;
  const uriList = String(clipboard?.getData?.("text/uri-list") || "")
    .split(/\r?\n/)
    .find((line) => line && !line.trim().startsWith("#"));
  return String(uriList || clipboard?.getData?.("text/plain") || "").trim();
}

function handleStageImagePaste(stageId, input, urlInput, scope, event) {
  if (event?.defaultPrevented) return;
  if (!isChannelEnabled(stageId, "image", scope)) {
    event?.preventDefault?.();
    setImageQueryStatus(stageId, "Enable Image channel before pasting an image.", true, scope);
    return;
  }
  const items = [...(event?.clipboardData?.items || [])];
  const item = items.find((candidate) => candidate?.kind === "file");
  const file = pastedStageImageFile(stageId, item);
  if (file) {
    event?.preventDefault?.();
    applyStageImageFile(stageId, input, file, scope, "paste");
    return;
  }

  const text = clipboardImageUrl(event);
  if (text) {
    event?.preventDefault?.();
    if (urlInput) urlInput.value = text;
    applyStageImageUrl(stageId, urlInput, text, scope);
    return;
  }

  event?.preventDefault?.();
  setImageQueryStatus(stageId, "Paste a JPEG or PNG image or an HTTP(S) image URL.", true, scope);
}

function handleStageImageDrop(stageId, input, urlInput, scope, event) {
  event?.preventDefault?.();
  event?.currentTarget?.classList?.remove("is-dragging");
  if (!isChannelEnabled(stageId, "image", scope)) {
    setImageQueryStatus(stageId, "Enable Image channel before dropping an image.", true, scope);
    return;
  }
  const files = [...(event?.dataTransfer?.files || [])];
  if (files.length) {
    applyStageImageFile(stageId, input, files[0], scope, "drop");
    return;
  }
  const text = String(
    event?.dataTransfer?.getData?.("text/uri-list") ||
    event?.dataTransfer?.getData?.("text/plain") ||
    "",
  ).trim();
  if (text) {
    if (urlInput) urlInput.value = text;
    applyStageImageUrl(stageId, urlInput, text, scope);
    return;
  }
  setImageQueryStatus(stageId, "Drop a JPEG or PNG image or an HTTP(S) image URL.", true, scope);
}

function setupImageInputs() {
  for (const input of document.querySelectorAll("input[type=file][id$='Image']")) {
    if (input.dataset.imageReady === "true") continue;
    const stageId = domStageId(input.id.replace(/Image$/, ""));
    const stage = stageId.toLowerCase();
    const scope = input.closest(".stage-block") || document;
    const urlInput = stageImageUrlInput(stageId, scope);
    const dropzone = scope.querySelector?.(`[data-image-dropzone="${stage}"]`) ||
      scope.querySelector?.(`[data-image-paste-target="${stage}"]`) ||
      input.closest?.(".image-query-picker");
    const clear = scope.querySelector?.(`[data-image-clear="${stage}"]`);
    input.addEventListener("change", () => {
      const file = input.files?.[0] || null;
      if (!file) {
        clearStageImage(stageId, scope);
        return;
      }
      applyStageImageFile(stageId, input, file, scope);
    });
    urlInput?.addEventListener("input", () => {
      const value = String(urlInput.value || "").trim();
      if (!value) {
        if (!stageImageFile(stageId, scope)) {
          revokeStageImagePreview(stageImagePreview(stageId, scope));
          if (clear) clear.disabled = true;
          setImageQueryStatus(stageId, "", false, scope);
        }
        return;
      }
      applyStageImageUrl(stageId, urlInput, value, scope);
    });

    const pasteHandler = (event) => handleStageImagePaste(stageId, input, urlInput, scope, event);
    for (const target of new Set([dropzone, urlInput, input])) {
      if (!target || target.dataset.imagePasteReady === "true") continue;
      target.addEventListener("paste", pasteHandler);
      target.dataset.imagePasteReady = "true";
    }
    dropzone?.addEventListener("dragover", (event) => {
      event.preventDefault();
      if (isChannelEnabled(stageId, "image", scope)) {
        dropzone.classList.add("is-dragging");
      }
    });
    dropzone?.addEventListener("dragleave", () => dropzone.classList.remove("is-dragging"));
    dropzone?.addEventListener("drop", (event) => {
      handleStageImageDrop(stageId, input, urlInput, scope, event);
    });
    clear?.addEventListener("click", () => clearStageImage(stageId, scope));
    input.dataset.imageReady = "true";
  }
}

function stageAsrMode(stageId, scope = document) {
  const stage = String(stageId).toLowerCase();
  const active = scope.querySelector(
    `.asr-mode-picker[data-stage="${stage}"] .asr-mode-btn.active`,
  );
  return active?.dataset.asrMode || "rrf";
}

function setStageAsrMode(stageId, mode, scope = document) {
  const stage = String(stageId).toLowerCase();
  const picker = scope.querySelector(`.asr-mode-picker[data-stage="${stage}"]`);
  if (!picker) return;
  const buttons = picker.querySelectorAll(".asr-mode-btn");
  buttons.forEach((button) => {
    const selected = button.dataset.asrMode === mode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function setupAsrModeToggles() {
  for (const picker of document.querySelectorAll(".asr-mode-picker")) {
    if (picker.dataset.asrReady === "true") continue;
    const stage = domStageId(picker.dataset.stage);
    const scope = picker.closest(".stage-block") || document;
    const buttons = picker.querySelectorAll(".asr-mode-btn");
    const initial = [...buttons].find((button) => button.classList.contains("active"));
    setStageAsrMode(stage, initial?.dataset.asrMode || "rrf", scope);
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const mode = button.dataset.asrMode || "rrf";
        setStageAsrMode(stage, mode, scope);
        const label = button.querySelector(".asr-mode-label")?.textContent || mode;
        setStatus(`ASR mode ${label} selected for the next search.`, false);
      });
    });
    picker.dataset.asrReady = "true";
  }
}

function stagePayload(stageId, scope = document) {
  const prefix = stageId.toLowerCase();
  const payload = {};
  for (const channel of STAGE_CHANNELS) {
    if (channel === "image") {
      const enabled = isChannelEnabled(stageId, channel, scope);
      const file = enabled ? stageImageFile(stageId, scope) : null;
      const rawUrl = enabled && !file ? stageImageUrl(stageId, scope) : "";
      const url = validateImageQueryUrl(rawUrl);
      // Local multipart bytes take precedence. Otherwise pass only a URL that
      // has passed the client-side HTTP(S)/public-host validation as image
      // metadata; never promote it to the text channel.
      payload[channel] = file ? { file_key: `${prefix}_image` } : "";
      if (!file && url.valid && url.value) {
        payload.image_url = url.value;
      }
      continue;
    }
    if (channel === "object") {
      const value = buildObjectQuery(stageId, scope);
      payload[channel] = isChannelEnabled(stageId, channel, scope) ? value : "";
      continue;
    }
    const input = scope.querySelector?.(`#${prefix + channel[0].toUpperCase() + channel.slice(1)}`) ||
      $(prefix + channel[0].toUpperCase() + channel.slice(1));
    const val = input ? input.value.trim() : "";
    const enabled = isChannelEnabled(stageId, channel, scope);
    payload[channel] = enabled ? val : "";
  }
  return payload;
}

function hasNonWhitespaceInput(value) {
  return typeof value === "string" && value.replace(/\s/g, "") !== "";
}

function stageHasRawInput(stageId, scope = document) {
  const block = stageBlockFor(stageId, scope);
  if (!block) return false;

  for (const channel of STAGE_CHANNELS) {
    if (channel === "image") {
      if (stageImageFile(stageId, scope) || stageImageUrl(stageId, scope)) return true;
      continue;
    }
    if (channel === "object") {
      if (hasNonWhitespaceInput(buildObjectQuery(stageId, block))) return true;
      continue;
    }
    const prefix = String(stageId).toLowerCase();
    const input = block.querySelector(
      `#${prefix + channel[0].toUpperCase() + channel.slice(1)}`,
    );
    if (hasNonWhitespaceInput(input?.value || "")) return true;
  }
  return false;
}

function collectActiveStagePayloads(root = $("stagedStages") || document) {
  const payloads = stagedStageIds().map((stageId) => {
    const stageBlock = stageBlockFor(stageId, root) || root;
    const imageEnabled = isChannelEnabled(stageId, "image", stageBlock);
    const imageFile = stageImageFile(stageId, stageBlock);
    const rawImageUrl = imageEnabled && !imageFile ? stageImageUrl(stageId, stageBlock) : "";
    const imageUrlValidation = validateImageQueryUrl(rawImageUrl);
    const imageUrl = imageUrlValidation.valid ? imageUrlValidation.value : "";
    return {
      stageId,
      channels: stagePayload(stageId, stageBlock),
      asrMode: stageAsrMode(stageId, stageBlock),
      imageFile,
      imageUrl,
      imageUrlError: imageUrlValidation.valid ? "" : imageUrlValidation.message,
      imageEnabled,
      hasRawInput: stageHasRawInput(stageId, stageBlock),
    };
  });
  const invalidImageUrl = payloads.find(
    (stage) => stage.imageEnabled && !stage.imageFile && stage.imageUrlError,
  );
  const missingImage = payloads.find(
    (stage) => stage.imageEnabled && !stage.imageFile && !stage.imageUrl && !stage.imageUrlError,
  );
  return {
    payloads,
    activeStages: payloads.filter((stage) => isStageActive(stage.channels)),
    // Draft text in an OFF channel is deliberately retained for the user but
    // omitted from the request. Whitespace-only stages are also inert.
    // error: null when all enabled channels have valid image input.
    error: invalidImageUrl
      ? `${invalidImageUrl.stageId} ${invalidImageUrl.imageUrlError}`
      : missingImage
        ? `${missingImage.stageId} Image is enabled; choose a JPEG/PNG or enter a valid HTTP(S) image URL before searching.`
        : null,
  };
}

function isTemporalSearchEnabled() {
  return $("temporalSearchToggle")?.getAttribute("aria-pressed") === "true";
}

function setTemporalSearchEnabled(enabled) {
  const toggle = $("temporalSearchToggle");
  if (!toggle) return;
  toggle.setAttribute("aria-pressed", String(enabled));
  toggle.classList.toggle("active", enabled);
  toggle.textContent = enabled ? "Trake: ≤ 60s" : "Trake: OFF";
  toggle.title = enabled
    ? "ON: same-video ordered stages with maximum 60 second adjacent gap"
    : "OFF: ordered same-video bundle without a maximum time gap";
}

function setupTemporalSearchToggle() {
  const toggle = $("temporalSearchToggle");
  if (!toggle || toggle.dataset.ready === "true") return;
  setTemporalSearchEnabled(false);
  toggle.addEventListener("click", () => {
    setTemporalSearchEnabled(!isTemporalSearchEnabled());
  });
  toggle.dataset.ready = "true";
}

function isStageActive(channels) {
  return Object.values(channels).some((val) => {
    if (val && typeof val === "object") {
      return hasNonWhitespaceInput(val.file_key || val.image_url || "");
    }
    return hasNonWhitespaceInput(String(val || ""));
  });
}

function buildStageMultipartForm(metadata, activeStages) {
  const form = new FormData();
  form.append("metadata", JSON.stringify(metadata));
  for (const stage of activeStages) {
    if (stage.imageFile && stage.channels.image?.file_key) {
      form.append(stage.channels.image.file_key, stage.imageFile, stage.imageFile.name);
    }
  }
  return form;
}

async function runUnifiedSearch() {
  const collected = collectActiveStagePayloads();
  const { activeStages } = collected;
  if (collected.error) {
    setStatus(collected.error, true);
    return;
  }
  if (!activeStages.length) {
    setStatus("No active stage query.", false);
    return;
  }

  clearDirectVideoState();

  const stageIds = activeStages.map(({ stageId }) => stageId);
  const temporalEnabled = isTemporalSearchEnabled();
  const queryId = "ui-bundle-" + Date.now();
  const request = beginSearchRequest();
  setStatus(
    currentViewMode === "flat"
      ? `Searching ${stageIds.join(" → ")} for raw All Hits…`
      : temporalEnabled
      ? `Searching ${stageIds.join(" → ")} and linking Δ ≤ 60s bundles…`
        : `Searching ${stageIds.join(" → ")} and grouping ordered same-video bundles…`,
    false,
    true,
  );
  hideDetail();

  const topK = parseInt($("topK")?.value, 10) || 500;
  const allHitsMinGapMs = allHitsSpacingMs();
  const selectedVideoId = normalizeVideoFilterValue($("videoFilter")?.value);
  const stages = activeStages.map(({ stageId, channels, asrMode }) => ({
    stage_id: stageId,
    channels,
    asr_mode: asrMode,
    top_k: topK,
  }));

  try {
    const metadata = {
      query_id: queryId,
      view_mode: currentViewMode === "flat" ? "all_hits" : "grouped",
      all_hits_min_gap_ms: allHitsMinGapMs,
      stages,
      temporal_enabled: temporalEnabled,
      max_delta_ms: TRAKE_MAX_DELTA_MS,
      top_k: topK,
    };
    if (selectedVideoId) metadata.video_ids = [selectedVideoId];
    const hasImages = activeStages.some((stage) => Boolean(stage.imageFile));
    const rawData = await api("/search/bundles", {
      method: "POST",
      ...(hasImages ? {} : { headers: { "Content-Type": "application/json" } }),
      body: hasImages ? buildStageMultipartForm(metadata, activeStages) : JSON.stringify(metadata),
      signal: request.signal,
    });
    if (!isCurrentSearchRequest(request.id)) return;
    const data = normalizeBundleResponse(rawData);
    data.all_hits = diversifyAllHits(
      data.all_hits_eligible_raw || data.all_hits || data.all_hits_raw || [],
      allHitsMinGapMs,
    );
    lastSearchData = data;
    lastQueryId = data.query_id || queryId;
    lastResultIds = (currentViewMode === "flat" ? data.all_hits : data.results)
      .map((r) => frameUidOf(r));
    queryRevision += 1;

    const historyText = activeStages
      .map(({ channels }) => channels.text || channels.ocr || channels.asr || channels.object)
      .find(Boolean);
    if (historyText) pushHistory(historyText);

    renderResultsView();
    renderChannelStatus(data.channel_status || {});
    updateStatusSummary(data);
    await refreshQueue();
  } catch (err) {
    if (isAbortError(err) || !isCurrentSearchRequest(request.id)) return;
    console.warn("Unified bundle search execution failed:", err);
    const msg = (err.message && (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")))
      ? "Backend offline — unable to execute search."
      : "Search failed — " + err.message;
    setStatus(msg, true);
  } finally {
    endSearchRequest(request.id);
  }
}

async function runTrakeSearch() {
  const stageIds = trakeStageIds();
  const configuredStages = stageIds.map((stageId) => ({
    stage_id: stageId,
    channels: stagePayload(stageId, stageBlockFor(stageId, $("trakeStages") || document) || document),
    asr_mode: stageAsrMode(stageId, stageBlockFor(stageId, $("trakeStages") || document) || document),
    top_k: Math.min(500, Math.max(20, parseInt($("topK")?.value || "20", 10) * 3)),
  }));
  const stages = configuredStages.filter((stage) => isStageActive(stage.channels));
  if (stages.length < 2) {
    setStatus("No complete Trake query yet.", false);
    return;
  }

  clearDirectVideoState();

  const queryId = "ui-trake-" + Date.now();
  const request = beginSearchRequest();
  setStatus(`Searching ${stageIds.join(" → ")} independently and linking tracks (Δ ≤ 60s)…`, false, true);
  hideDetail();

  try {
    const selectedVideoId = normalizeVideoFilterValue($("videoFilter")?.value);
    const body = {
      query_id: queryId,
      stages,
      max_delta_ms: TRAKE_MAX_DELTA_MS,
      top_k: parseInt($("topK")?.value || "20", 10),
    };
    if (selectedVideoId) body.video_ids = [selectedVideoId];
    const rawData = await api("/search/trake", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: request.signal,
    });
    if (!isCurrentSearchRequest(request.id)) return;
    const data = normalizeTrakeResponse(rawData);
    lastSearchData = data;
    lastQueryId = data.query_id || queryId;
    lastResultIds = data.results.map((result) => frameUidOf(result));
    queryRevision += 1;

    const historyText = stages
      .flatMap((stage) => [stage.channels.text, stage.channels.ocr, stage.channels.asr, stage.channels.object])
      .find((value) => value && value.trim());
    if (historyText) pushHistory(historyText);

    renderResultsView();
    renderChannelStatus(data.stage_status?.S1 || data.channel_status || {});
    updateStatusSummary(data);
    await refreshQueue();
  } catch (err) {
    if (isAbortError(err) || !isCurrentSearchRequest(request.id)) return;
    console.warn("Trake search execution failed:", err);
    const msg = (err.message && (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")))
      ? "Backend offline — unable to execute Trake search."
      : "Trake search failed — " + err.message;
    setStatus(msg, true);
  } finally {
    endSearchRequest(request.id);
  }
}

async function runSearch(text) {
  const query = (text ?? $("query").value).trim();
  if (!query && $("taskType").value !== "VKIS") {
    setStatus("Enter a search query first.", true);
    return;
  }
  if (text) $("query").value = query;

  clearDirectVideoState();

  setStatus("Searching…", false, true);
  hideDetail();

  const taskType = $("taskType").value;
  const topK = parseInt($("topK").value, 10);
  const videoId = normalizeVideoFilterValue($("videoFilter").value);
  const visualIndexes = selectedVisualIndexes();
  if (!visualIndexes.length && ["VKIS", "TKIS", "kis"].includes(taskType)) {
    setStatus("SigLIP2 visual search is unavailable.", true);
    return;
  }

  const request = beginSearchRequest();
  try {
    let rawData = null;
    if (taskType === "VKIS") {
      const file = $("queryImage").files[0];
      if (!file) throw new Error("Choose a query image for VKIS first.");
      const params = new URLSearchParams({
        top_k: String(topK),
        query_id: "ui-" + Date.now(),
        visual_indexes: visualIndexes.join(","),
      });
      rawData = await api(`/search/image?${params.toString()}`, {
        method: "POST",
        headers: { "Content-Type": file.type },
        body: file,
        signal: request.signal,
      });
    } else if (kisMode && ["TKIS", "kis"].includes(taskType)) {
      const body = {
        query_id: "ui-" + Date.now(),
        text: query,
        top_k: topK,
        visual_indexes: visualIndexes,
      };
      if (videoId) body.video_ids = [videoId];
      rawData = await api("/search/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: request.signal,
      });
    } else {
      const body = {
        text: query,
        task_type: taskType,
        top_k: topK,
        visual_indexes: visualIndexes,
      };
      if (videoId) body.video_ids = [videoId];
      rawData = await api("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: request.signal,
      });
    }

    if (!isCurrentSearchRequest(request.id)) return;
    const data = normalizeKisResponse(rawData);
    lastSearchData = data;
    lastQueryId = data.query_id || "ui-" + Date.now();
    lastResultIds = (data.results || []).map((r) => r.frame_id);
    queryRevision += 1;
    if (query) pushHistory(query);

    renderResultsView();
    if (data.channel_status) renderChannelStatus(data.channel_status);
    updateStatusSummary(data);
    await refreshQueue();
  } catch (err) {
    if (isAbortError(err) || !isCurrentSearchRequest(request.id)) return;
    console.warn("Search execution failed:", err);
    const msg = (err.message && (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")))
      ? "Backend offline — unable to execute search."
      : "Search failed — " + err.message;
    setStatus(msg, true);
  } finally {
    endSearchRequest(request.id);
  }
}

function updateStatusSummary(data) {
  const resultStats = $("resultStats");
  const statVideoCount = $("statVideoCount");
  const statHitCount = $("statHitCount");
  const statLatency = $("statLatency");

  const results = currentViewMode === "flat" && Array.isArray(data.all_hits)
    ? diversifyAllHits(data.all_hits, allHitsSpacingMs())
    : (data.results || []);
  const uniqueVideos = new Set(results.map((r) => r.video_id).filter(Boolean));

  if (resultStats) resultStats.hidden = false;
  if (statVideoCount) statVideoCount.textContent = `${uniqueVideos.size} videos`;
  if (statHitCount) statHitCount.textContent = `${results.length} hits`;
  if (statLatency) statLatency.textContent = `${data.latency_ms ?? 0} ms`;

  setStatus(
    `Query ${data.query_id || lastQueryId}: found ${results.length} hit(s) across ${uniqueVideos.size} video(s)` +
    (results.length === 0 ? " — no matches; adjust query parameters or clear filters." : "")
  );
}

/* ==========================================================================
   Authoritative Results Rendering & Video Grouping
   ========================================================================== */

function frameUidOf(item) {
  return item?.frame_uid || item?.frame_id || "";
}

function normalizedQueryId(value) {
  const queryId = String(value ?? "").trim();
  return queryId || null;
}

function queryIdForResult(result = null, override = null) {
  return normalizedQueryId(
    override || result?.query_id || (result === selectedResult ? inspectorQueryId : null) || lastQueryId,
  ) || "manual";
}

function currentInspectorQueryId() {
  return normalizedQueryId(selectedResult?.query_id || inspectorQueryId) || "manual";
}

function visibleResultsForData(data) {
  const source = data?.mode === "bundle" && currentViewMode === "grouped"
    ? (Array.isArray(data.results) ? data.results : [])
    : stageResultItems(data);
  const selectedVideoId = normalizeVideoFilterValue($("videoFilter")?.value);
  const visible = source.filter((item) => {
    if (selectedVideoId && item.video_id !== selectedVideoId) return false;
    if (currentStageFilter !== "all" && item.stage_id !== currentStageFilter) return false;
    return Boolean(frameUidOf(item));
  });
  const ordered = currentViewMode === "flat"
    ? diversifyAllHits(visible, allHitsSpacingMs())
    : visible.sort(compareStageCandidates);
  return ordered;
}

function updateSelectedCardStyles() {
  for (const card of document.querySelectorAll(".result-card")) {
    card.classList.toggle("selected", card.dataset.frameId === selectedFrameId);
  }
}

function selectResult(result, { inspect = false, loadContext = true } = {}) {
  const frameId = frameUidOf(result);
  if (!frameId) return;

  selectedResult = { ...result, query_id: queryIdForResult(result) };
  selectedFrameId = frameId;
  lastResolvedPosition = null;
  lastExactExtraction = null;
  updateSelectedCardStyles();
  // The inspector fetches the same context and renders its own neighbor
  // strip. Avoid racing a duplicate /frames request against video startup.
  if (inspect) void openDetail(frameId);
  else if (loadContext) void loadTopNeighbors(frameId, CONTEXT_WINDOW_FRAMES);
}

function selectPrimaryResult(data) {
  const visible = visibleResultsForData(data);
  if (!visible.length) {
    selectedResult = null;
    selectedFrameId = null;
    clearTopNeighborStrip();
    return;
  }

  const current = visible.find((item) => frameUidOf(item) === selectedFrameId);
  if (current) {
    selectedResult = { ...current, query_id: queryIdForResult(current, lastQueryId) };
    updateSelectedCardStyles();
    return;
  }
  selectResult(visible[0]);
}

function renderDirectVideoView(container) {
  const state = directVideoState;
  container.classList.add("direct-video-results");
  container.classList.remove("bundle-results", "trake-results");

  if (!state || state.status === "loading") {
    container.innerHTML = `
      <div class="empty-state direct-video-loading">
        <div class="empty-icon">◌</div>
        <h3>Loading video</h3>
        <p class="muted">Resolving the first canonical keyframe…</p>
      </div>`;
    return;
  }
  if (state.status !== "ready" || !state.frame) {
    container.innerHTML = `
      <div class="empty-state direct-video-error">
        <div class="empty-icon">—</div>
        <h3>Video unavailable</h3>
        <p class="muted">${escapeHtml(state.error || "First canonical keyframe unavailable.")}</p>
      </div>`;
    return;
  }

  const frame = state.frame;
  const card = document.createElement("article");
  card.className = "direct-video-card";
  card.dataset.videoId = frame.video_id;
  card.dataset.frameId = frame.frame_uid;
  card.setAttribute("aria-label", `Direct video ${frame.video_id} · ${frame.frame_uid}`);
  const imageMarkup = frame.image_available === false
    ? `<div class="direct-video-image-placeholder">Keyframe unavailable</div>`
    : `<img src="${escapeHtml(frame.thumbnail_url || thumbnailUrl(frame.frame_uid))}" alt="${escapeHtml(frame.frame_uid)}" loading="eager">`;
  card.innerHTML = `
    <button type="button" class="direct-video-preview" title="Open video inspector">
      ${imageMarkup}
      <span class="direct-video-open-hint">Open inspector</span>
    </button>
    <div class="direct-video-copy">
      <span class="direct-video-label">DIRECT VIDEO · FIRST KEYFRAME</span>
      <strong>${escapeHtml(frame.video_id)}</strong>
      <span title="${escapeHtml(frame.frame_uid)}">${escapeHtml(frame.frame_uid)}</span>
      <span>src ${escapeHtml(String(frame.source_frame_idx))} · ${escapeHtml(formatVideoTime(frame.timestamp_ms / 1000))}</span>
    </div>`;
  card.querySelector(".direct-video-preview")?.addEventListener("click", () => {
    selectResult(frame, { inspect: true, loadContext: false });
  });
  container.appendChild(card);
}

function buildVideoGroupMap(data) {
  const groupMap = new Map();
  const selectedVideoId = normalizeVideoFilterValue($("videoFilter")?.value);

  function addItem(item) {
    const vid = item.video_id || "UNKNOWN_VIDEO";
    if (selectedVideoId && vid !== selectedVideoId) return;

    if (!groupMap.has(vid)) {
      groupMap.set(vid, {
        video_id: vid,
        best_score: -Infinity,
        items: [],
      });
    }
    const grp = groupMap.get(vid);
    const score = resultScore(item);
    if (score > grp.best_score) grp.best_score = score;
    grp.items.push(item);
  }

  // Grouping changes presentation only. The backend identity remains the
  // canonical frame_uid and each stage owner is retained on the item.
  for (const item of stageResultItems(data)) addItem(item);

  for (const grp of groupMap.values()) {
    // Within a video keep stage order (S1 before S2 …) and score descending.
    grp.items.sort(compareStageCandidates);
  }

  return new Map([...groupMap.entries()].sort(([, left], [, right]) => {
    const scoreDelta = right.best_score - left.best_score;
    return Math.abs(scoreDelta) > 1e-12
      ? scoreDelta
      : left.video_id.localeCompare(right.video_id);
  }));
}

function renderResultsView() {
  const container = $("results");
  if (!container) return;
  disconnectDeferredImageObserver(container);
  container.innerHTML = "";
  container.classList.add("gallery-first");

  if (directVideoState?.mode === "direct_video") {
    renderDirectVideoView(container);
    return;
  }

  if (!lastSearchData || (!lastSearchData.results?.length && !lastSearchData.stage_results)) {
    clearTopNeighborStrip();
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🍵</div>
        <h3>No search results</h3>
        <p class="muted">Try adjusting your query terms or clearing video filters.</p>
      </div>`;
    return;
  }

  selectPrimaryResult(lastSearchData);

  if (
    currentViewMode === "grouped" &&
    Array.isArray(lastSearchData.bundles) &&
    lastSearchData.bundles.length
  ) {
    renderBundleMatrix(container, lastSearchData);
    setupDeferredImages(container, { eagerLimit: eagerImageLimitForGallery(container) });
    return;
  }

  if (currentViewMode === "grouped") {
    renderGroupedView(container, lastSearchData);
  } else {
    renderFlatView(container, lastSearchData);
  }
  setupDeferredImages(container, { eagerLimit: eagerImageLimitForGallery(container) });
}

function disconnectDeferredImageObserver(root) {
  const state = deferredImageObservers.get(root);
  if (!state) return;
  const observer = state.observer || state;
  observer.disconnect?.();
  state.cleanup?.();
  deferredImageObservers.delete(root);
}

function eagerImageLimitForGallery(root) {
  const galleryVideoCards = [...root.querySelectorAll(".gallery-video-card")];
  if (!galleryVideoCards.length) return GALLERY_EAGER_THUMBNAIL_LIMIT;
  const groupedCount = galleryVideoCards
    .slice(0, GALLERY_EAGER_VIDEO_LIMIT)
    .reduce((count, card) => count + card.querySelectorAll("img[data-src]").length, 0);
  return Math.min(GALLERY_EAGER_MAX_IMAGES, groupedCount);
}

function thumbnailFallbackFor(image) {
  const parent = image?.parentElement || image?.closest?.(".card-image-box, .timeline-thumb, .neighbor-thumb");
  if (!parent) return null;
  let fallback = parent.querySelector?.(".thumbnail-fallback");
  if (!fallback) {
    fallback = document.createElement("div");
    fallback.className = "card-image-placeholder thumbnail-fallback";
    const label = document.createElement("span");
    label.textContent = "image unavailable";
    const reason = document.createElement("span");
    reason.className = "muted-sm";
    reason.textContent = "REMOTE_MEDIA_UNAVAILABLE";
    const retry = document.createElement(parent.tagName === "BUTTON" ? "span" : "button");
    if (retry.tagName === "SPAN") {
      retry.setAttribute("role", "button");
      retry.tabIndex = 0;
    } else {
      retry.type = "button";
    }
    retry.className = "thumbnail-retry-btn";
    retry.textContent = "Retry thumbnail";
    const retryHandler = (event) => {
      event.stopPropagation();
      retryThumbnail(image);
    };
    retry.addEventListener("click", retryHandler);
    retry.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") retryHandler(event);
    });
    fallback.append(label, reason, retry);
    parent.appendChild(fallback);
  }
  return fallback;
}

function showThumbnailFallback(image) {
  if (!image) return;
  image.dataset.thumbnailState = "error";
  image.hidden = true;
  const fallback = thumbnailFallbackFor(image);
  if (fallback) fallback.hidden = false;
}

function retryThumbnail(image) {
  const source = image?.dataset.thumbnailSource || image?.getAttribute?.("src");
  if (!image || !source) return;
  image.dataset.thumbnailRetry = String(Number(image.dataset.thumbnailRetry || 0) + 1);
  image.setAttribute("data-thumbnail-retry", image.dataset.thumbnailRetry);
  image.dataset.thumbnailState = "retrying";
  image.hidden = false;
  const fallback = image.parentElement?.querySelector?.(".thumbnail-fallback");
  fallback?.remove();
  image.removeAttribute("src");
  const assign = () => {
    image.src = source;
  };
  if (typeof window.requestAnimationFrame === "function") window.requestAnimationFrame(assign);
  else setTimeout(assign, 0);
}

function setupThumbnailImage(image, source = image?.dataset?.src || "") {
  if (!image || !source) return;
  image.dataset.thumbnailSource = source;
  if (image.dataset.thumbnailLifecycle === "true") return;
  image.dataset.thumbnailLifecycle = "true";
  image.dataset.thumbnailRetry = image.dataset.thumbnailRetry || "0";
  image.setAttribute("data-thumbnail-retry", image.dataset.thumbnailRetry);
  image.addEventListener("load", () => {
    image.dataset.thumbnailState = "loaded";
    image.hidden = false;
    image.parentElement?.querySelector?.(".thumbnail-fallback")?.remove();
  });
  image.addEventListener("error", () => showThumbnailFallback(image));
}

function loadVisibleDeferredImages(root, images, loadImage) {
  const viewportWidth = Number(window.innerWidth) || 1280;
  const viewportHeight = Number(window.innerHeight) || 900;
  let loaded = 0;
  for (const image of images) {
    if (!image?.dataset?.src) continue;
    const rect = image.getBoundingClientRect?.();
    if (!rect) continue;
    const visible = rect.bottom >= -160 && rect.top <= viewportHeight + 160 &&
      rect.right >= -160 && rect.left <= viewportWidth + 160;
    if (visible) {
      loadImage(image);
      loaded += 1;
    }
  }
  return loaded;
}

/**
 * Defer gallery/timeline thumbnails until they are close to the viewport.
 * A geometry pass covers scroll containers where IntersectionObserver does
 * not receive a useful viewport transition; all fallbacks keep the 320px
 * thumbnail URL and never switch to full-resolution media.
 */
function setupDeferredImages(root = document, { eagerLimit = 0 } = {}) {
  disconnectDeferredImageObserver(root);
  const images = [...root.querySelectorAll("img[data-src]")];
  if (!images.length) return;
  images.forEach((image) => setupThumbnailImage(image, image.dataset.src));

  const loadImage = (image) => {
    const source = image.dataset.src || image.dataset.thumbnailSource;
    if (!source || image.dataset.deferredLoaded === "true") return;
    image.dataset.thumbnailSource = source;
    image.dataset.deferredLoaded = "true";
    image.dataset.thumbnailState = "loading";
    image.loading = "eager";
    image.src = source;
    image.removeAttribute("data-src");
  };

  const eagerCount = Math.min(
    images.length,
    Math.max(0, Math.trunc(Number(eagerLimit) || 0)),
  );
  images.slice(0, eagerCount).forEach((image) => {
    image.fetchPriority = "high";
    loadImage(image);
  });
  const deferredImages = images.slice(eagerCount);
  if (!deferredImages.length) return;

  loadVisibleDeferredImages(root, deferredImages, loadImage);
  if (!deferredImages.some((image) => image.dataset.src)) return;

  if (!("IntersectionObserver" in window)) {
    deferredImages.forEach(loadImage);
    return;
  }

  let scheduled = false;
  const scheduleVisibleLoad = () => {
    if (scheduled) return;
    scheduled = true;
    const run = () => {
      scheduled = false;
      loadVisibleDeferredImages(root, deferredImages, loadImage);
    };
    if (typeof window.requestAnimationFrame === "function") window.requestAnimationFrame(run);
    else setTimeout(run, 0);
  };
  const scrollHost = root.closest?.(".main-content") || root;
  scrollHost.addEventListener?.("scroll", scheduleVisibleLoad, { passive: true });
  window.addEventListener?.("resize", scheduleVisibleLoad, { passive: true });
  const cleanup = () => {
    scrollHost.removeEventListener?.("scroll", scheduleVisibleLoad);
    window.removeEventListener?.("resize", scheduleVisibleLoad);
  };

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      loadImage(entry.target);
      observer.unobserve(entry.target);
    }
    if (!deferredImages.some((image) => image.dataset.src)) {
      observer.disconnect();
      cleanup();
      deferredImageObservers.delete(root);
    }
  }, { root: null, rootMargin: "160px 0px", threshold: 0.01 });
  deferredImageObservers.set(root, { observer, cleanup });
  deferredImages.forEach((image) => observer.observe(image));
}

function renderBundleMatrix(container, data) {
  const selectedVideoId = normalizeVideoFilterValue($("videoFilter")?.value);
  const bundles = (Array.isArray(data.bundles) ? data.bundles : []).filter((bundle) => {
    if (!selectedVideoId) return true;
    const bundleVideoId = String(
      bundle.video_id || bundle.stages?.find((item) => item?.video_id)?.video_id || "",
    ).trim();
    return bundleVideoId === selectedVideoId;
  });
  const stageIds = Array.isArray(data.stage_ids) && data.stage_ids.length
    ? data.stage_ids
    : [...new Set(bundles.flatMap((bundle) => (bundle.stages || []).map((item) => item.stage_id)))];
  if (!bundles.length || !stageIds.length) {
    const constraint = data.temporal_enabled ? " and Δ ≤ 60s" : "";
    container.innerHTML = `<div class="empty-state"><p class="muted">No complete ordered same-video bundles match the selected stages${constraint}.</p></div>`;
    return;
  }

  renderBundleContactSheet(container, data, bundles, stageIds);
}

function renderBundleContactSheet(container, data, bundles, stageIds) {
  container.classList.add("bundle-results");
  container.classList.remove("trake-results");

  const visibleStageIds = currentStageFilter === "all"
    ? stageIds
    : stageIds.filter((stageId) => stageId === currentStageFilter);
  const stageOrder = new Map(stageIds.map((stageId, index) => [stageId, index]));
  const matrix = document.createElement("div");
  matrix.className = "bundle-contact-sheet-grid";

  for (const [bundleIndex, bundle] of bundles.entries()) {
    const items = (Array.isArray(bundle.stages) ? bundle.stages : [])
      .filter((item) => visibleStageIds.includes(item.stage_id))
      .sort((left, right) => {
        const leftOrder = stageOrder.get(left.stage_id) ?? Number.MAX_SAFE_INTEGER;
        const rightOrder = stageOrder.get(right.stage_id) ?? Number.MAX_SAFE_INTEGER;
        return leftOrder - rightOrder;
    });
    if (!items.length) continue;

    const bundleScore = Number(bundle.bundle_score ?? bundle.score ?? 0);
    const bundleId = bundle.bundle_id || `${bundle.video_id || "video"}-${bundleIndex + 1}`;
    const bundleCard = document.createElement("article");
    bundleCard.className = "bundle-video-card";
    const bundleColorIndex = bundleIndex % BUNDLE_BORDER_PALETTE.length;
    bundleCard.style.setProperty("--bundle-accent", BUNDLE_BORDER_PALETTE[bundleColorIndex]);
    bundleCard.dataset.bundleId = String(bundleId);
    bundleCard.dataset.videoId = bundle.video_id || "";
    bundleCard.dataset.bundleRank = String(bundle.bundle_rank ?? bundleIndex + 1);
    bundleCard.dataset.bundleColor = String(bundleColorIndex);
    bundleCard.dataset.stageCount = String(items.length);
    bundleCard.setAttribute(
      "aria-label",
      `Bundle ${bundleIndex + 1} · ${bundle.video_id || "video"} · ${items.length} stages`,
    );
    bundleCard.title = `${bundle.video_id || "video"} · ${items.length}-stage bundle`;
    const stageStack = document.createElement("div");
    stageStack.className = "bundle-stage-stack";

    for (const [itemIndex, item] of items.entries()) {
      const stageId = item.stage_id || stageIds[itemIndex] || "S1";
      const card = createResultCard(item, stageId, { minimal: true });
      card.classList.add("bundle-contact-card");
      card.dataset.videoId = bundle.video_id || item.video_id || "";
      card.dataset.bundleId = String(bundleId);
      card.dataset.bundleGroupStart = itemIndex === 0 ? "true" : "false";
      if (itemIndex === 0) {
        card.classList.add("bundle-group-start");
        const scoreChip = document.createElement("span");
        scoreChip.className = "bundle-score-chip";
        scoreChip.textContent = `B${bundleIndex + 1} · ${bundleScore.toFixed(3)}`;
        scoreChip.title = `Bundle ${bundleIndex + 1} · harmonic score ${bundleScore.toFixed(4)}`;
        card.querySelector(".card-image-box")?.appendChild(scoreChip);
      }
      card.setAttribute(
        "aria-label",
        `${bundle.video_id || item.video_id || "video"} · ${stageId} · ${frameUidOf(item)}`,
      );
      stageStack.appendChild(card);
    }
    bundleCard.appendChild(stageStack);
    matrix.appendChild(bundleCard);
  }

  if (!matrix.children.length) {
    container.innerHTML = `<div class="empty-state"><p class="muted">No results match the current stage filter.</p></div>`;
    return;
  }
  container.appendChild(matrix);
}

function renderGroupedView(container, data) {
  const groupMap = buildVideoGroupMap(data);
  if (groupMap.size === 0) {
    container.innerHTML = `<div class="empty-state"><p class="muted">No video groups match the current filters.</p></div>`;
    return;
  }

  // Render every frame into one contact sheet. Group ordering is still encoded
  // by the sequence of cards (video -> S1..Sn -> score desc), but no video gets
  // its own variable-height CSS column, so S1/S2/S3 candidates stay contiguous
  // from left to right and top to bottom.
  const matrix = document.createElement("div");
  matrix.className = "video-contact-sheet-grid";

  for (const grp of groupMap.values()) {
    const visibleItems = grp.items.filter((item) =>
      currentStageFilter === "all" || item.stage_id === currentStageFilter
    );
    if (!visibleItems.length) continue;

    for (const [index, item] of visibleItems.entries()) {
      const card = createResultCard(item, item.stage_id || "S1", { minimal: true });
      card.dataset.videoId = grp.video_id;
      card.dataset.videoGroupStart = index === 0 ? "true" : "false";
      card.setAttribute(
        "aria-label",
        `${grp.video_id} · ${item.stage_id || "S1"} · ${frameUidOf(item)}`,
      );
      matrix.appendChild(card);
    }
  }

  if (!matrix.children.length) {
    container.innerHTML = `<div class="empty-state"><p class="muted">No results match the current stage filter.</p></div>`;
    return;
  }
  container.appendChild(matrix);
}

function renderFlatView(container, data) {
  const grid = document.createElement("div");
  grid.className = "results-grid";

  const items = visibleResultsForData(data);
  const selectedVideoId = normalizeVideoFilterValue($("videoFilter")?.value);

  for (const r of items) {
    if (selectedVideoId && r.video_id !== selectedVideoId) continue;

    grid.appendChild(createResultCard(r, r.stage_id || "S1", { minimal: true }));
  }

  if (grid.children.length === 0) {
    container.innerHTML = `<div class="empty-state"><p class="muted">No results match the current filters.</p></div>`;
  } else {
    container.appendChild(grid);
  }
}

/**
 * Creates individual candidate card with absolute translucent identity footer
 * Format strictly: video_id : source_frame_idx
 */
function createResultCard(r, stageId, { minimal = false } = {}) {
  const card = document.createElement("div");
  const frameId = frameUidOf(r);
  card.className = "result-card" + (minimal ? " gallery-card" : "") +
    (frameId === selectedFrameId ? " selected" : "");
  card.dataset.frameId = frameId;
  card.tabIndex = 0;

  const stageBadgeClass = stageBadgeClassFor(stageId);
  const rankLabel = stageId ? `${stageId} #${r.rank_in_stage ?? r.rank ?? 1}` : `#${r.rank ?? 1}`;
  const scoreVal = Number(r.final_score || 0).toFixed(4);
  const footerIdentity = `${escapeHtml(r.video_id)} : ${escapeHtml(r.source_frame_idx ?? r.frame_idx ?? "")}`;
  const galleryIdentity = escapeHtml(frameId);

  const imageMarkup = r.image_available === false
    ? `<div class="card-image-placeholder"><span>image unavailable</span><span class="muted-sm">${escapeHtml(r.image_status || "IMAGE_UNAVAILABLE")}</span></div>`
    : `<img loading="lazy" decoding="async" data-src="${escapeHtml(r.thumbnail_url || thumbnailUrl(frameId))}" alt="${escapeHtml(frameId)}">`;

  const actionMarkup = `
      <div class="card-actions-overlay">
        <button type="button" class="card-action-btn act-inspect card-action-icon" aria-label="Inspect keyframe" title="Inspect keyframe">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><circle cx="10.8" cy="10.8" r="6.3"></circle><path d="m16 16 4.5 4.5"></path></svg>
        </button>
        <button type="button" class="card-action-btn act-queue card-action-icon" aria-label="Add frame to queue" title="Add frame to queue">
          <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M4 4.5h16v14H4z"></path><path d="M8 8h8M8 12h4M12 10v7M8.5 13.5h7"></path></svg>
        </button>
      </div>`;

  card.innerHTML = minimal ? `
    <div class="card-image-box">
      ${imageMarkup}
      <div class="card-badges-top">
        <span class="badge ${stageBadgeClass}">${escapeHtml(stageId || "S1")}</span>
      </div>
      ${actionMarkup}
      <div class="card-identity-footer gallery-identity-footer">
        <span class="identity-text" title="${galleryIdentity}">${galleryIdentity}</span>
      </div>
    </div>` : `
    <div class="card-image-box">
      ${imageMarkup}
      <div class="card-badges-top">
        <span class="badge ${stageBadgeClass}">${escapeHtml(stageId || "S1")}</span>
        <span class="badge badge-rank">${escapeHtml(rankLabel)}</span>
      </div>
      <div class="card-score-top">${scoreVal}</div>

      ${actionMarkup}

      <div class="card-identity-footer">
        <span class="identity-text" title="${escapeHtml(r.frame_id)}">${footerIdentity}</span>
        <span class="identity-meta">${((r.timestamp_ms || 0) / 1000).toFixed(1)}s</span>
      </div>
    </div>`;
  card.setAttribute(
    "aria-label",
    `${stageId || "S1"} result ${r.video_id || "video"} ${r.source_frame_idx ?? r.frame_idx ?? ""}. Click to inspect.`,
  );
  card.title = "Click to inspect keyframe details";

  const cardImg = card.querySelector("img");
  if (cardImg) {
    setupThumbnailImage(cardImg, cardImg.dataset.src || thumbnailUrl(frameId));
  }

  const inspectBtn = card.querySelector(".act-inspect");
  const queueBtn = card.querySelector(".act-queue");

  inspectBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      selectResult(r, { inspect: true });
    });

  queueBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    addResultToQueue(r, stageId);
  });

  card.addEventListener("click", () => {
    selectResult(r, { inspect: true });
  });

  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      selectResult(r, { inspect: true });
    } else if (e.key === " " || e.key.toLowerCase() === "q") {
      e.preventDefault();
      e.stopPropagation();
      addResultToQueue(r, stageId);
    }
  });

  return card;
}

function stageBadgeClassFor(stageId) {
  const stage = String(stageId || "S1").toUpperCase();
  if (stage === "S1") return "s1-badge";
  if (stage === "S2") return "s2-badge";
  if (stage === "S3") return "s3-badge";
  if (stage === "S4") return "s4-badge";
  if (stage === "S5") return "s5-badge";
  return "stage-generic-badge";
}

/* ==========================================================================
   Lazy Top Context Timeline Strip (±15 Frames)
   ========================================================================== */

function clearTopNeighborStrip() {
  topNeighborRequestId += 1;
  const strip = $("topNeighborStrip");
  const container = $("topNeighborsContainer");
  const label = $("topTimelineFrameId");
  if (strip) strip.hidden = true;
  if (container) {
    disconnectDeferredImageObserver(container);
    container.innerHTML = "";
  }
  if (label) label.textContent = "";
}

function timelineResult(item) {
  const frameId = frameUidOf(item);
  if (!frameId) return null;
  const timestamp = Number(item.timestamp_ms);
  const videoId = item.video_id || selectedResult?.video_id || frameId.split(":")[0];
  return {
    ...item,
    frame_id: frameId,
    frame_uid: frameId,
    image_url: item.image_url || `/frames/${encodeURIComponent(frameId)}/image`,
    thumbnail_url: item.thumbnail_url || selectedResult?.thumbnail_url || thumbnailUrl(frameId),
    video_id: videoId,
    video_url: item.video_url || selectedResult?.video_url ||
      (videoId ? `/videos/${encodeURIComponent(videoId)}/stream` : null),
    video_stream_available: item.video_stream_available ?? selectedResult?.video_stream_available,
    frame_idx: item.frame_idx ?? item.source_frame_idx,
    source_frame_idx: item.source_frame_idx ?? item.frame_idx,
    timestamp_ms: Number.isFinite(timestamp) ? timestamp : 0,
    direct_video: item.direct_video ?? isDirectVideoResult(item),
    stage_id: item.stage_id || (isDirectVideoResult(item) ? null : selectedResult?.stage_id || "S1"),
  };
}

function renderTimelineThumb(container, item, activeUid, kind, onSelect) {
  const result = timelineResult(item);
  if (!result) return;

  const isTop = kind === "top";
  const thumb = document.createElement("button");
  thumb.type = "button";
  thumb.className = isTop ? "timeline-thumb" : "neighbor-thumb";
  if (frameUidOf(result) === activeUid) thumb.classList.add(isTop ? "active" : "current");
  thumb.title = frameUidOf(result);

  if (result.image_available === false) {
    thumb.innerHTML = `<div class="card-image-placeholder"><span>unavailable</span></div>`;
  } else {
    thumb.innerHTML = `<img loading="lazy" data-src="${escapeHtml(result.thumbnail_url || thumbnailUrl(frameUidOf(result)))}" alt="${escapeHtml(frameUidOf(result))}">`;
    const image = thumb.querySelector("img");
    setupThumbnailImage(image, result.thumbnail_url || thumbnailUrl(frameUidOf(result)));
  }

  const time = document.createElement("span");
  time.textContent = `${(Number(result.timestamp_ms || 0) / 1000).toFixed(1)}s`;
  thumb.appendChild(time);
  thumb.addEventListener("click", () => onSelect(result));
  container.appendChild(thumb);
}

function renderTopNeighborStrip(neighbors, activeUid) {
  const strip = $("topNeighborStrip");
  const container = $("topNeighborsContainer");
  const label = $("topTimelineFrameId");
  const valid = (neighbors || []).map(timelineResult).filter(Boolean);
  if (!strip || !container || !valid.length) {
    clearTopNeighborStrip();
    return;
  }

  strip.hidden = false;
  if (label) label.textContent = activeUid || "—";
  container.innerHTML = "";
  for (const item of valid) {
    renderTimelineThumb(container, item, activeUid, "top", (result) => selectResult(result));
  }
  setupDeferredImages(container);
}

function previewNeighborItems(frameId) {
  const videoId = frameId.split(":")[0];
  return (lastSearchData?.results || [])
    .filter((item) => item.video_id === videoId)
    .sort((a, b) => Number(a.timestamp_ms || 0) - Number(b.timestamp_ms || 0))
    .map((item) => ({ ...item, is_current: frameUidOf(item) === frameId }));
}

async function loadTopNeighbors(frameId, windowSize = CONTEXT_WINDOW_FRAMES) {
  if (!frameId) return;
  const requestId = ++topNeighborRequestId;

  if (PREVIEW_SAMPLE_MODE) {
    renderTopNeighborStrip(previewNeighborItems(frameId), frameId);
    return;
  }

  try {
    const data = await api(`/frames/${encodeURIComponent(frameId)}?window=${windowSize}`);
    if (requestId !== topNeighborRequestId || selectedFrameId !== frameId) return;
    renderTopNeighborStrip(data.neighbors || [], frameId);
  } catch {
    if (requestId === topNeighborRequestId) clearTopNeighborStrip();
  }
}

function renderDetailNeighborStrip(neighbors, activeUid) {
  const strip = $("neighbors");
  const label = $("detailTimelineFrameId");
  if (!strip) return;

  const valid = (neighbors || []).map(timelineResult).filter(Boolean);
  const neighborBox = strip.closest(".inspector-neighbor-strip");
  if (neighborBox) neighborBox.hidden = !valid.length;
  disconnectDeferredImageObserver(strip);
  strip.innerHTML = "";
  if (label) label.textContent = activeUid || "—";
  if (!valid.length) {
    const empty = document.createElement("span");
    empty.className = "muted-sm timeline-empty-state";
    empty.textContent = "Nearby keyframes unavailable.";
    strip.appendChild(empty);
    return;
  }

  for (const item of valid) {
    renderTimelineThumb(strip, item, activeUid, "detail", (result) => {
      selectResult(result, { inspect: true });
    });
  }
  setupDeferredImages(strip);
}

function stageNumber(stageId) {
  const match = String(stageId || "").match(/(\d+)/);
  return match ? Number.parseInt(match[1], 10) : Number.MAX_SAFE_INTEGER;
}

function mergeInspectorStageDraft(item) {
  const stageId = String(item?.stage_id || "").toUpperCase();
  const draft = inspectorStageDrafts.get(stageId);
  if (!draft) return item;
  const hasFrameOverride = Object.prototype.hasOwnProperty.call(draft, "frame_uid") ||
    Object.prototype.hasOwnProperty.call(draft, "frame_id");
  const frameUid = hasFrameOverride
    ? (draft.frame_uid ?? draft.frame_id ?? null)
    : item.frame_uid;
  const sourceFrameIdx = Object.prototype.hasOwnProperty.call(draft, "source_frame_idx")
    ? draft.source_frame_idx
    : item.source_frame_idx;
  return {
    ...item,
    ...draft,
    stage_id: stageId,
    frame_id: frameUid,
    frame_uid: frameUid,
    source_frame_idx: sourceFrameIdx,
    timestamp_ms: draft.timestamp_ms ?? item.timestamp_ms,
    qa_answer: draft.qa_answer ?? item.qa_answer ?? "",
  };
}

function inspectorStageTimestampSeconds(item) {
  const timestampMs = Number(item?.timestamp_ms);
  return Number.isFinite(timestampMs) ? Math.max(0, timestampMs / 1000) : 0;
}

function inspectorCurrentVideoId() {
  return selectedResult?.video_id || selectedFrameId?.split(":")[0] || null;
}

function inspectorContextKeyFor(frameId = selectedFrameId, result = selectedResult) {
  const videoId = String(result?.video_id || frameId?.split(":")[0] || "").trim();
  const bundleId = String(result?.bundle_id || "").trim();
  const trackId = String(result?.track_id || result?.chain_id || "").trim();
  const queryId = String(result?.query_id || inspectorQueryId || lastQueryId || "").trim();
  // The selected/extracted frame is mutable within one inspector context.
  // Including it here clears per-stage drafts after exact Extract. Bundle/track
  // identity remains the boundary that should reset the inspector state.
  return [queryId, videoId, bundleId, trackId].join("\u0001");
}

function ensureInspectorContext() {
  const nextKey = inspectorContextKeyFor();
  if (inspectorContextKey === nextKey) return;
  inspectorContextKey = nextKey;
  inspectorQueryId = normalizedQueryId(selectedResult?.query_id || lastQueryId);
  inspectorStageDrafts = new Map();
  inspectorQaDrafts = new Map();
}

function preserveInspectorContext() {
  inspectorContextKey = inspectorContextKeyFor();
  inspectorQueryId = normalizedQueryId(selectedResult?.query_id || lastQueryId);
}

function inspectorStageTimelineDuration(items = inspectorStageItems) {
  const videoDuration = Number($("detailVideo")?.duration);
  if (Number.isFinite(videoDuration) && videoDuration > 0) return videoDuration;
  return (items || []).reduce(
    (max, item) => Math.max(max, inspectorStageTimestampSeconds(item)),
    0,
  );
}

function inspectorIsTrake() {
  return activeInspectorTask === "TRAKE" ||
    lastSearchData?.mode === "trake" ||
    Boolean(
      selectedResult?.track_id ||
      selectedResult?.bundle_id ||
      (selectedResult && hasTemporalModeField(selectedResult)),
    );
}

function inspectorStageDelta(items, index) {
  if (index === 0) {
    return { visible: inspectorIsTrake(), valid: true, label: "anchor" };
  }

  const currentMs = Number(items[index]?.timestamp_ms);
  const previousMs = Number(items[index - 1]?.timestamp_ms);
  if (!Number.isFinite(currentMs) || !Number.isFinite(previousMs)) {
    return { visible: true, valid: false, label: "Δ unknown" };
  }
  const deltaMs = currentMs - previousMs;
  const strictlyForward = deltaMs > 0;
  if (!strictlyForward) {
    return {
      visible: true,
      valid: false,
      deltaMs,
      label: `Δ ${formatVideoTime(Math.abs(deltaMs) / 1000)} · out of order`,
    };
  }
  if (!inspectorIsTrake()) return { visible: false, valid: true, deltaMs, label: "" };
  const hasPersistedTemporalMode = Boolean(
    selectedResult && (
      Object.prototype.hasOwnProperty.call(selectedResult, "bundle_temporal_enabled") ||
      Object.prototype.hasOwnProperty.call(selectedResult, "temporal_enabled")
    ),
  );
  const temporalEnabled = hasPersistedTemporalMode
    ? temporalModeEnabled(selectedResult)
    : lastSearchData?.mode === "trake";
  if (!temporalEnabled) {
    return {
      visible: true,
      valid: true,
      deltaMs,
      label: `Δ ${formatVideoTime(deltaMs / 1000)} · valid`,
    };
  }
  const valid = deltaMs <= TRAKE_MAX_DELTA_MS;
  return {
    visible: true,
    valid,
    deltaMs,
    label: `Δ ${formatVideoTime(deltaMs / 1000)} · ${valid ? "valid" : "> 60s"}`,
  };
}

function inspectorStageTimelineItems(includeDrafts = true) {
  if (isDirectVideoResult()) return [];
  const videoId = selectedResult?.video_id || selectedFrameId?.split(":")[0];
  if (!videoId) return [];

  // A flat All Hits card belongs to one complete bundle.  Do not rebuild its
  // inspector from every stage hit in the video, otherwise a high-scoring S1
  // from bundle A can be paired with S2 from bundle B.
  const selectedBundleId = String(selectedResult?.bundle_id || selectedResult?.track_id || "").trim();
  const byStage = new Map();
  const consider = (raw, fallbackStage = null) => {
    if (!raw || (raw.video_id && raw.video_id !== videoId)) return;
    const rawBundleId = String(raw.bundle_id || raw.track_id || "").trim();
    if (selectedBundleId && rawBundleId !== selectedBundleId) return;
    const frameId = frameUidOf(raw);
    const timestampMs = Number(raw.timestamp_ms);
    const stageId = String(raw.stage_id || fallbackStage || "").toUpperCase();
    const allowPending = includeDrafts && Boolean(raw.stage_draft_pending || raw.stage_frame_pending);
    if ((!frameId && !allowPending) || !stageId || !Number.isFinite(timestampMs)) return;

    const item = {
      ...raw,
      video_id: videoId,
      frame_id: frameId,
      frame_uid: frameId,
      stage_id: stageId,
      timestamp_ms: timestampMs,
      source_frame_idx: raw.source_frame_idx ?? raw.frame_idx,
    };
    const current = byStage.get(stageId);
    const isSelected = frameId === selectedFrameId;
    const currentScore = Number(current?.final_score ?? current?.score ?? -Infinity);
    const itemScore = Number(item.final_score ?? item.score ?? -Infinity);
    if (!current || isSelected || itemScore > currentScore) byStage.set(stageId, item);
  };

  const queueGroupItems = Array.isArray(selectedResult?.queue_group_items)
    ? selectedResult.queue_group_items
    : [];
  if (!queueGroupItems.length) {
    for (const item of lastSearchData?.results || []) consider(item);
    for (const [stageId, items] of Object.entries(lastSearchData?.stage_results || {})) {
      for (const item of Array.isArray(items) ? items : []) consider(item, stageId);
    }
    for (const bundle of lastSearchData?.bundles || []) {
      if (bundle.video_id !== videoId) continue;
      for (const item of bundle.stages || []) consider(item);
    }
  }
  for (const item of queueGroupItems) {
    consider(item);
  }
  consider(selectedResult, selectedResult?.stage_id || "S1");

  const sorted = [...byStage.values()].sort((left, right) => {
    const stageDelta = stageNumber(left.stage_id) - stageNumber(right.stage_id);
    return stageDelta || left.timestamp_ms - right.timestamp_ms;
  });
  return includeDrafts ? sorted.map(mergeInspectorStageDraft) : sorted;
}

function inspectorActiveStageItem() {
  return inspectorStageItems.find(
    (item) => String(item.stage_id).toUpperCase() === String(inspectorActiveStageId || "").toUpperCase(),
  ) || null;
}

function syncInspectorStageForm() {
  const item = inspectorActiveStageItem();
  const answer = $("answer");
  const stageId = String(item?.stage_id || inspectorActiveStageId || "").toUpperCase();
  const qaValue = inspectorQaDrafts.has(stageId)
    ? inspectorQaDrafts.get(stageId)
    : item?.qa_answer || "";
  if (answer && document.activeElement !== answer) answer.value = qaValue;
}

function updateInspectorStageAnswer(value) {
  const stageId = String(inspectorActiveStageId || "").toUpperCase();
  if (!stageId) return;
  inspectorQaDrafts.set(stageId, String(value || "").slice(0, 100));
  const status = $("qaApplyStatus");
  if (status) status.textContent = "Unsaved answer · click Apply answer";
}

async function applyInspectorStageAnswer() {
  const stageId = String(inspectorActiveStageId || "").toUpperCase();
  const item = inspectorStageItems.find((candidate) => candidate.stage_id === stageId);
  const status = $("qaApplyStatus");
  if (!stageId || !item) {
    if (status) status.textContent = "Select a stage before applying an answer.";
    return;
  }

  const value = String($("answer")?.value || "").trim().slice(0, 100);
  const current = inspectorStageDrafts.get(stageId) || {};
  inspectorStageDrafts.set(stageId, {
    ...current,
    qa_answer: value,
    qa_answer_applied: true,
  });
  item.qa_answer = value;
  inspectorQaDrafts.delete(stageId);

  const queueItem = queueItemForInspectorStage(stageId, item.frame_uid);
  if (queueItem) {
    if (currentQueuePreview) {
      previewQueueItems = previewQueueItems.map((candidate) =>
        candidate.queue_item_id === queueItem.queue_item_id
          ? { ...candidate, qa_answer: value }
          : candidate,
      );
      renderQueueItems(previewQueueItems, true);
    } else {
      try {
        await api(`/review/queue/${encodeURIComponent(queueItem.queue_item_id)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ qa_answer: value }),
        });
        await refreshQueue();
      } catch (err) {
        if (status) status.textContent = `Queue answer update failed — ${err.message}`;
        return;
      }
    }
  }
  renderInspectorStageSequence();
  setInspectorActiveStage(stageId, { syncForm: true });
  if (status) status.textContent = value ? `Applied to ${stageId}.` : `Cleared answer on ${stageId}.`;
  setStatus(value ? `QA answer applied to ${stageId}.` : `QA answer cleared on ${stageId}.`);
}

function setInspectorActiveStage(stageId, { syncForm = false } = {}) {
  inspectorActiveStageId = stageId ? String(stageId).toUpperCase() : null;
  document.querySelectorAll(".inspector-stage-marker").forEach((marker) => {
    marker.classList.toggle("active", marker.dataset.stageId === inspectorActiveStageId);
  });
  document.querySelectorAll(".inspector-stage-sequence-item").forEach((item) => {
    const active = item.dataset.stageId === inspectorActiveStageId;
    item.classList.toggle("active", active);
    item.setAttribute("aria-current", active ? "true" : "false");
  });
  if (syncForm) syncInspectorStageForm();
}

function seekToInspectorStage(item) {
  if (!item) return;
  const targetSeconds = Math.max(0, Number(item.timestamp_ms || 0) / 1000);
  const video = $("detailVideo");
  inspectorActiveStageId = item.stage_id;
  if (video && Number.isFinite(targetSeconds)) {
    video.currentTime = targetSeconds;
    if (video.readyState >= 1) syncVideoSeekBar();
  }
  setInspectorActiveStage(item.stage_id, { syncForm: true });
  updatePlayerPosition();
}

function renderInspectorStageTimeline() {
  const markerRoot = $("stageTimelineMarkers");
  const sequenceRoot = $("inspectorStageSequence");
  if (!markerRoot || !sequenceRoot) return;

  ensureInspectorContext();
  const videoId = inspectorCurrentVideoId();
  const baseItems = inspectorStageTimelineItems(false);
  for (const item of baseItems) {
    const stageId = String(item.stage_id).toUpperCase();
    if (!inspectorStageDrafts.has(stageId)) {
      inspectorStageDrafts.set(stageId, { qa_answer: item.qa_answer || "" });
    }
  }
  inspectorStageItems = inspectorStageTimelineItems(true);
  inspectorActiveStageId = selectedResult?.stage_id
    ? String(selectedResult.stage_id).toUpperCase()
    : inspectorActiveStageId;
  markerRoot.innerHTML = "";

  const duration = inspectorStageTimelineDuration(inspectorStageItems);
  const endLabel = $("stageTimelineEnd");
  if (endLabel) endLabel.textContent = duration > 0 ? formatVideoTime(duration) : "—";
  renderInspectorRangeTimelineMarkers();
  if ($("inspectorVideoId")) $("inspectorVideoId").textContent = videoId || "—";
  if ($("inspectorStageCount")) {
    $("inspectorStageCount").textContent = `${inspectorStageItems.length} stage${inspectorStageItems.length === 1 ? "" : "s"}`;
  }

  if (!inspectorStageItems.length) {
    renderInspectorStageSequence();
    setInspectorActiveStage(null);
    return;
  }

  inspectorStageItems.forEach((item, index) => {
    const stageId = item.stage_id;
    const timestampSeconds = inspectorStageTimestampSeconds(item);
    const position = duration > 0
      ? Math.max(0, Math.min(100, (timestampSeconds / duration) * 100))
      : 0;
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = `inspector-stage-marker ${stageBadgeClassFor(stageId)}${item.stage_draft_pending ? " stage-marker-draft" : ""}`;
    marker.dataset.stageId = stageId;
    marker.style.left = `${position}%`;
    marker.title = `${stageId} · Drag to move · ${formatVideoTime(timestampSeconds)} · ${frameUidOf(item) || "pending extract"}`;
    marker.setAttribute("aria-label", `Drag ${stageId} to a new video time; current ${formatVideoTime(timestampSeconds)}`);
    marker.innerHTML = `<span class="stage-marker-dot"></span><span class="stage-marker-label">${escapeHtml(stageId)}</span>`;
    marker.addEventListener("pointerdown", (event) => beginStageMarkerDrag(event, stageId));
    marker.addEventListener("click", (event) => {
      if (suppressInspectorMarkerClick) {
        suppressInspectorMarkerClick = false;
        event.preventDefault();
        return;
      }
      seekToInspectorStage(item);
    });
    markerRoot.appendChild(marker);
  });

  renderInspectorStageSequence();
  setInspectorActiveStage(inspectorActiveStageId);
  syncInspectorStageTimeline();
}

function renderInspectorStageSequence() {
  const sequenceRoot = $("inspectorStageSequence");
  if (!sequenceRoot) return;
  sequenceRoot.innerHTML = "";

  if (!inspectorStageItems.length) {
    const empty = document.createElement("span");
    empty.className = "inspector-stage-sequence-empty";
    empty.textContent = "No stage markers in this result.";
    sequenceRoot.appendChild(empty);
    return;
  }

  inspectorStageItems.forEach((item, index) => {
    const stageId = String(item.stage_id).toUpperCase();
    const timestampSeconds = inspectorStageTimestampSeconds(item);
    const frameUid = frameUidOf(item);
    const delta = inspectorStageDelta(inspectorStageItems, index);

    const row = document.createElement("div");
    row.className = "inspector-stage-sequence-row";
    row.dataset.stageId = stageId;

    const sequenceItem = document.createElement("button");
    sequenceItem.type = "button";
    sequenceItem.className = "inspector-stage-sequence-item";
    sequenceItem.dataset.stageId = stageId;
    sequenceItem.setAttribute("data-stage-seek", stageId);
    sequenceItem.setAttribute("aria-current", "false");
    const deltaMarkup = delta.visible
      ? `<span class="inspector-sequence-delta ${delta.valid ? "stage-delta-valid" : "stage-delta-invalid"}">${escapeHtml(delta.label)}</span>`
      : "";
    const draftMarkup = item.stage_draft_pending
      ? '<span class="inspector-sequence-draft">pending Extract</span>'
      : "";
    const answerMarkup = item.qa_answer
      ? `<span class="inspector-sequence-answer" title="${escapeHtml(item.qa_answer)}">QA: ${escapeHtml(item.qa_answer)}</span>`
      : "";
    sequenceItem.innerHTML = `
      <span class="inspector-sequence-stage ${stageBadgeClassFor(stageId)}">${escapeHtml(stageId)}</span>
      <span class="inspector-sequence-copy">
        <strong>${escapeHtml(formatVideoTime(timestampSeconds))}</strong>
        <span title="${escapeHtml(frameUid || "pending extract")}">${escapeHtml(frameUid || "pending extract")}</span>
        ${deltaMarkup}${draftMarkup}${answerMarkup}
      </span>
      <span class="inspector-sequence-arrow" aria-hidden="true">›</span>`;
    sequenceItem.addEventListener("click", () => seekToInspectorStage(item));

    const editButton = document.createElement("button");
    editButton.type = "button";
    editButton.className = "inspector-stage-edit";
    editButton.setAttribute("data-stage-edit", stageId);
    editButton.setAttribute("aria-label", `Edit frame ID for ${stageId}`);
    editButton.title = `Edit ${stageId} frame ID manually`;
    editButton.textContent = "✎";
    editButton.addEventListener("click", () => openStageFrameEditor(stageId));

    const markerButton = document.createElement("button");
    markerButton.type = "button";
    markerButton.className = "inspector-stage-mark";
    markerButton.setAttribute("data-stage-mark", stageId);
    markerButton.setAttribute("aria-label", `Mark ${stageId} at the current stopped playhead`);
    markerButton.title = `Mark ${stageId} at current playhead (no seek)`;
    markerButton.textContent = "⌖";
    markerButton.addEventListener("click", (event) => {
      event.stopPropagation();
      markCurrentStageAtPlayhead(stageId);
    });

    const stageCard = document.createElement("div");
    stageCard.className = "inspector-stage-card";
    stageCard.append(sequenceItem, editButton);
    row.append(stageCard, markerButton);
    sequenceRoot.appendChild(row);
  });
}

function inspectorTimelineSecondsFromPointer(event) {
  const track = $("stageTimelineMarkers");
  if (!track) return 0;
  const rect = track.getBoundingClientRect();
  if (!rect.width) return 0;
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  return ratio * inspectorStageTimelineDuration(inspectorStageItems);
}

function updateInspectorMarkerDrag(stageId, seconds) {
  const item = inspectorStageItems.find((candidate) => candidate.stage_id === stageId);
  const track = $("stageTimelineMarkers");
  const video = $("detailVideo");
  if (!item || !track) return;
  const duration = inspectorStageTimelineDuration(inspectorStageItems);
  item.timestamp_ms = Math.round(Math.max(0, seconds) * 1000);
  item.stage_draft_pending = true;
  const marker = track.querySelector(`[data-stage-id="${CSS.escape(stageId)}"]`);
  if (marker) {
    marker.style.left = `${duration > 0 ? Math.max(0, Math.min(100, (seconds / duration) * 100)) : 0}%`;
    marker.classList.add("stage-marker-draft");
  }
  if (video && Number.isFinite(seconds)) {
    video.currentTime = Math.max(0, seconds);
    if (video.readyState >= 1) syncVideoSeekBar();
  }
  setInspectorActiveStage(stageId);
  syncInspectorStageTimeline();
}

function beginStageMarkerDrag(event, stageId) {
  if (event.button !== undefined && event.button !== 0) return;
  const marker = event.currentTarget;
  event.preventDefault();
  try {
    marker.setPointerCapture?.(event.pointerId);
  } catch {
    // Pointer capture is best-effort; synthetic events and detached markers may reject it.
  }
  setInspectorActiveStage(stageId, { syncForm: true });
  inspectorMarkerDrag = {
    marker,
    pointerId: event.pointerId,
    stageId,
    moved: false,
    seconds: inspectorStageTimestampSeconds(inspectorActiveStageItem()),
  };

  const move = (moveEvent) => {
    if (!inspectorMarkerDrag || moveEvent.pointerId !== inspectorMarkerDrag.pointerId) return;
    inspectorMarkerDrag.moved = true;
    inspectorMarkerDrag.seconds = inspectorTimelineSecondsFromPointer(moveEvent);
    updateInspectorMarkerDrag(stageId, inspectorMarkerDrag.seconds);
  };
  const end = (endEvent) => {
    if (!inspectorMarkerDrag || endEvent.pointerId !== inspectorMarkerDrag.pointerId) return;
    const drag = inspectorMarkerDrag;
    document.removeEventListener("pointermove", move);
    document.removeEventListener("pointerup", end);
    try {
      marker.releasePointerCapture?.(endEvent.pointerId);
    } catch {
      // The marker may already have been detached after a rerender.
    }
    inspectorMarkerDrag = null;
    if (!drag.moved) {
      seekToInspectorStage(inspectorActiveStageItem());
      return;
    }
    suppressInspectorMarkerClick = true;
    const current = inspectorStageDrafts.get(stageId) || {};
    inspectorStageDrafts.set(stageId, {
      ...current,
      timestamp_ms: Math.round(drag.seconds * 1000),
      stage_draft_pending: true,
    });
    renderInspectorStageTimeline();
    setInspectorActiveStage(stageId, { syncForm: true });
  };
  document.addEventListener("pointermove", move);
  document.addEventListener("pointerup", end, { once: false });
}

function updateInspectorSelectionHeader(item) {
  const frameUid = frameUidOf(item);
  if (!frameUid) return;
  selectedFrameId = frameUid;
  selectedResult = { ...(selectedResult || {}), ...item, frame_id: frameUid, frame_uid: frameUid };
  $("detailTitle").textContent = frameUid;
  const stageTag = $("detailStageTag");
  if (stageTag && isDirectVideoResult(item)) {
    stageTag.textContent = "";
    stageTag.hidden = true;
  } else if (stageTag) {
    stageTag.textContent = item.stage_id || "S1";
    stageTag.className = `inspector-badge ${stageBadgeClassFor(item.stage_id)}`;
    stageTag.hidden = false;
  }
  const scoreTag = $("detailScoreTag");
  if (scoreTag && isDirectVideoResult(item)) {
    scoreTag.textContent = "";
    scoreTag.hidden = true;
  } else if (scoreTag && Number.isFinite(Number(item.final_score))) {
    scoreTag.textContent = `Score: ${Number(item.final_score).toFixed(4)}`;
    scoreTag.hidden = false;
  }
}

function commitInspectorStageFrame(stageId, resolved, source = "manual") {
  const normalizedStage = String(stageId || "").toUpperCase();
  const item = inspectorStageItems.find((candidate) => candidate.stage_id === normalizedStage);
  const frameUid = resolved?.frame_uid || resolved?.resolved_frame_uid;
  if (!item || !frameUid || resolved.video_id !== inspectorCurrentVideoId()) return false;
  const timestampMs = Number(resolved.resolved_timestamp_ms ?? resolved.timestamp_ms ?? resolved.selected_time_ms);
  if (!Number.isFinite(timestampMs) || timestampMs < 0) return false;
  const currentDraft = inspectorStageDrafts.get(normalizedStage) || {};
  const next = {
    ...item,
    ...resolved,
    stage_id: normalizedStage,
    frame_id: frameUid,
    frame_uid: frameUid,
    source_frame_idx: resolved.source_frame_idx,
    timestamp_ms: timestampMs,
    qa_answer: currentDraft.qa_answer ?? item.qa_answer ?? "",
    stage_frame_source: source,
    stage_frame_verified: source !== "canonical_apply",
    stage_frame_pending: false,
    stage_draft_pending: false,
  };
  inspectorStageDrafts.set(normalizedStage, next);
  if (normalizedStage === inspectorActiveStageId) {
    updateInspectorSelectionHeader(next);
    const video = $("detailVideo");
    if (video && video.readyState >= 1) video.currentTime = timestampMs / 1000;
    lastResolvedPosition = { ...resolved, frame_uid: frameUid, resolved_timestamp_ms: timestampMs };
    if (source === "manual") {
      lastExactExtraction = {
        ...resolved,
        frame_uid: frameUid,
        requested_timestamp_ms: timestampMs,
        resolved_timestamp_ms: timestampMs,
        raw_verification: "MANUAL_CANONICAL",
        stage_id: normalizedStage,
      };
    }
  }
  renderInspectorStageTimeline();
  setInspectorActiveStage(normalizedStage, { syncForm: true });
  return true;
}

function commitCanonicalInspectorSelection(resolved) {
  const frameUid = resolved?.frame_uid || resolved?.resolved_frame_uid;
  const videoId = String(resolved?.video_id || "").trim();
  const timestampMs = Number(resolved?.resolved_timestamp_ms ?? resolved?.timestamp_ms);
  const stageId = String(
    inspectorActiveStageId || selectedResult?.stage_id || "S1",
  ).toUpperCase();
  const item = inspectorStageItems.find((candidate) => candidate.stage_id === stageId) || selectedResult;
  if (
    !item ||
    !frameUid ||
    videoId !== inspectorCurrentVideoId() ||
    frameUid !== `${videoId}:${resolved.source_frame_idx}` ||
    !Number.isFinite(timestampMs) ||
    timestampMs < 0
  ) return false;

  const authoritativeQueryId = normalizedQueryId(
    item.query_id || selectedResult?.query_id || inspectorQueryId,
  );
  const currentDraft = inspectorStageDrafts.get(stageId) || {};
  const next = {
    ...item,
    ...resolved,
    ...(authoritativeQueryId ? { query_id: authoritativeQueryId } : {}),
    stage_id: stageId,
    frame_id: frameUid,
    frame_uid: frameUid,
    source_frame_idx: Number(resolved.source_frame_idx),
    timestamp_ms: timestampMs,
    qa_answer: currentDraft.qa_answer ?? item.qa_answer ?? "",
    stage_frame_source: "manual_seek",
    stage_frame_verified: false,
    stage_frame_pending: false,
    stage_draft_pending: false,
  };
  inspectorStageDrafts.set(stageId, next);
  inspectorStageItems = inspectorStageItems.some((candidate) => candidate.stage_id === stageId)
    ? inspectorStageItems.map((candidate) => candidate.stage_id === stageId ? next : candidate)
    : [next];
  selectedFrameId = frameUid;
  selectedResult = {
    ...(selectedResult || {}),
    ...next,
    frame_id: frameUid,
    frame_uid: frameUid,
    video_id: videoId,
    ...(authoritativeQueryId ? { query_id: authoritativeQueryId } : {}),
  };
  inspectorActiveStageId = stageId;
  lastResolvedPosition = {
    ...resolved,
    frame_uid: frameUid,
    requested_timestamp_ms: timestampMs,
    resolved_timestamp_ms: timestampMs,
  };
  lastExactExtraction = null;
  clearExactExtractionStatus();
  preserveInspectorContext();
  updateSelectedCardStyles();
  updateInspectorSelectionHeader(next);
  updateDetailResolvedMetadata(resolved, timestampMs);
  renderInspectorStageTimeline();
  setInspectorActiveStage(stageId, { syncForm: true });
  return true;
}

async function markCurrentStageAtPlayhead(stageId = inspectorActiveStageId) {
  const normalizedStage = String(stageId || "").toUpperCase();
  const item = inspectorStageItems.find((candidate) => candidate.stage_id === normalizedStage);
  if (!item) {
    setStatus(`Cannot mark ${normalizedStage || "stage"}: stage is not in this result.`, true);
    return false;
  }
  const timestampMs = videoTimeMs();
  const videoId = inspectorCurrentVideoId();
  const mediaGeneration = detailMediaGeneration;
  let resolved = null;
  let resolveError = null;
  if (!PREVIEW_SAMPLE_MODE && videoId) {
    try {
      resolved = await api(
        `/v1/videos/${encodeURIComponent(videoId)}/resolve?timestamp_ms=${timestampMs}`,
      );
    } catch (err) {
      resolveError = err instanceof Error ? err.message : String(err);
    }
  }
  if (mediaGeneration !== detailMediaGeneration || selectedFrameId?.split(":")[0] !== videoId) {
    return false;
  }
  const resolvedFrameUid = resolved?.frame_uid || resolved?.resolved_frame_uid || null;
  const resolvedSourceFrameIdx = resolved?.source_frame_idx;
  const resolvedTimestampMs = Number(resolved?.resolved_timestamp_ms ?? resolved?.timestamp_ms);
  const hasCanonicalResolution = Boolean(
    resolvedFrameUid &&
    resolved?.video_id === videoId &&
    Number.isInteger(Number(resolvedSourceFrameIdx)) &&
    Number.isFinite(resolvedTimestampMs) &&
    resolvedTimestampMs >= 0,
  );
  lastExactExtraction = null;
  clearExactExtractionStatus();
  if (hasCanonicalResolution) {
    lastResolvedPosition = {
      ...resolved,
      requested_timestamp_ms: timestampMs,
      resolved_timestamp_ms: resolvedTimestampMs,
    };
    updateDetailResolvedMetadata(resolved, timestampMs);
  } else {
    lastResolvedPosition = null;
  }
  const currentDraft = inspectorStageDrafts.get(normalizedStage) || {};
  inspectorStageDrafts.set(normalizedStage, {
    ...item,
    ...currentDraft,
    stage_id: normalizedStage,
    frame_id: hasCanonicalResolution ? resolvedFrameUid : null,
    frame_uid: hasCanonicalResolution ? resolvedFrameUid : null,
    source_frame_idx: hasCanonicalResolution ? Number(resolvedSourceFrameIdx) : null,
    timestamp_ms: timestampMs,
    requested_timestamp_ms: timestampMs,
    resolved_timestamp_ms: hasCanonicalResolution ? resolvedTimestampMs : null,
    mapping_status: hasCanonicalResolution
      ? (resolved.mapping_status || "RESOLVED_CANONICAL")
      : "TIMESTAMP_ONLY",
    stage_frame_source: "playhead",
    stage_frame_verified: false,
    stage_frame_pending: true,
    stage_draft_pending: true,
  });
  inspectorActiveStageId = normalizedStage;
  renderInspectorStageTimeline();
  setInspectorActiveStage(normalizedStage, { syncForm: true });
  // Marking is annotation-only: never seek the stopped video or invent an
  // exact source_frame_idx. Extract remains the sole raw-decode authority.
  const resolutionNote = hasCanonicalResolution
    ? `mapped to ${resolvedFrameUid} · src #${resolvedSourceFrameIdx}`
    : `canonical mapping unavailable${resolveError ? ` (${resolveError})` : ""}`;
  setStatus(`Marked ${normalizedStage} at ${formatVideoTime(timestampMs / 1000)} · ${resolutionNote}. Extract to verify the source frame.`, !hasCanonicalResolution);
  return true;
}

function rangeMarkerLabel(marker) {
  if (!marker) return "—";
  const index = marker.source_frame_idx == null ? "?" : `#${marker.source_frame_idx}`;
  const frameUid = marker.frame_uid ? ` ${marker.frame_uid}` : "";
  return `${formatVideoTime(marker.timestamp_ms / 1000)} ${index}${frameUid}`;
}

function renderRangeMarkerStatus() {
  const status = $("rangeMarkerStatus");
  if (!status) return;
  const left = inspectorRangeMarkers.L;
  const right = inspectorRangeMarkers.R;
  const invalid = left && right && left.timestamp_ms > right.timestamp_ms;
  status.textContent = `L ${rangeMarkerLabel(left)} · R ${rangeMarkerLabel(right)}${invalid ? " · invalid order" : ""}`;
  status.classList.toggle("is-invalid", Boolean(invalid));
  renderInspectorRangeTimelineMarkers();
}

function renderInspectorRangeTimelineMarkers() {
  const track = $("stageTimelineMarkers");
  if (!track) return;

  track.querySelectorAll(".inspector-range-marker").forEach((marker) => marker.remove());
  const duration = inspectorStageTimelineDuration(inspectorStageItems);
  if (!(duration > 0)) return;

  for (const side of ["L", "R"]) {
    const marker = inspectorRangeMarkers[side];
    const timestampMs = Number(marker?.timestamp_ms);
    if (!Number.isFinite(timestampMs)) continue;

    const element = document.createElement("span");
    element.className = `inspector-range-marker range-marker-${side}`;
    element.dataset.markerSide = side;
    element.style.left = `${Math.max(0, Math.min(100, (timestampMs / 1000 / duration) * 100))}%`;
    element.title = `${side} · ${rangeMarkerLabel(marker)}`;
    element.setAttribute("aria-label", `${side} marker at ${formatVideoTime(timestampMs / 1000)}`);
    element.innerHTML = `<span class="range-marker-line"></span><span class="range-marker-label">${side}</span>`;
    track.appendChild(element);
  }
}

function setInspectorRangeMarker(side) {
  const normalizedSide = String(side || "").toUpperCase();
  if (!(normalizedSide in inspectorRangeMarkers) || !selectedFrameId) return false;
  const timestampMs = videoTimeMs();
  const active = inspectorActiveStageItem();
  const resolved = lastResolvedPosition &&
    Number(lastResolvedPosition.requested_timestamp_ms ?? lastResolvedPosition.selected_time_ms) === timestampMs
    ? lastResolvedPosition
    : null;
  inspectorRangeMarkers[normalizedSide] = {
    video_id: inspectorCurrentVideoId(),
    timestamp_ms: timestampMs,
    frame_uid: resolved?.frame_uid || (active && Number(active.timestamp_ms) === timestampMs ? frameUidOf(active) : null),
    source_frame_idx: resolved?.source_frame_idx ?? (active && Number(active.timestamp_ms) === timestampMs ? active.source_frame_idx : null),
    mapping_status: resolved?.mapping_status || "TIMESTAMP_ONLY",
  };
  renderRangeMarkerStatus();
  setStatus(`Marker ${normalizedSide} set at ${formatVideoTime(timestampMs / 1000)}.`);
  return true;
}

function clearInspectorRangeMarkers() {
  inspectorRangeMarkers = { L: null, R: null };
  renderRangeMarkerStatus();
  setStatus("L/R markers cleared.");
}

function clearInspectorRangeMarker(side) {
  const normalizedSide = String(side || "").toUpperCase();
  if (!(normalizedSide in inspectorRangeMarkers)) return false;
  inspectorRangeMarkers[normalizedSide] = null;
  renderRangeMarkerStatus();
  setStatus(`Marker ${normalizedSide} cleared.`);
  return true;
}

async function applySourceFrameIdx() {
  if (!selectedFrameId) return;
  const input = $("sourceFrameIdxInput");
  const raw = String(input?.value || "").trim();
  if (!/^\d+$/.test(raw)) {
    setStatus("Enter a non-negative canonical source_frame_idx.", true);
    return;
  }
  const sourceFrameIdx = Number.parseInt(raw, 10);
  const videoId = inspectorCurrentVideoId();
  const button = $("applySourceFrameIdx");
  const mediaGeneration = detailMediaGeneration;
  const frameIdAtStart = selectedFrameId;
  if (!videoId || !Number.isSafeInteger(sourceFrameIdx)) return;
  if (button) button.disabled = true;
  try {
    const resolved = await api(
      `/v1/videos/${encodeURIComponent(videoId)}/resolve?source_frame_idx=${sourceFrameIdx}`,
    );
    if (
      resolved.video_id !== videoId ||
      Number(resolved.source_frame_idx) !== sourceFrameIdx ||
      resolved.frame_uid !== `${videoId}:${sourceFrameIdx}`
    ) {
      throw new Error("Canonical frame identity validation failed.");
    }
    if (!inspectorRequestIsCurrent(mediaGeneration, frameIdAtStart)) return;
    const timestampMs = Number(resolved.resolved_timestamp_ms ?? resolved.timestamp_ms);
    if (!Number.isFinite(timestampMs) || timestampMs < 0) {
      throw new Error("Resolved frame has no valid timestamp.");
    }
    if (!commitCanonicalInspectorSelection(resolved)) {
      throw new Error("Canonical frame could not be committed to the active inspector stage.");
    }
    const video = $("detailVideo");
    const seekEvents = ["loadedmetadata", "loadeddata", "durationchange"];
    let seekCompleted = false;
    const cleanupSeekListeners = () => {
      if (!video) return;
      seekEvents.forEach((eventName) => video.removeEventListener(eventName, seekResolvedVideo));
      if (video.onloadedmetadata === seekResolvedVideo) video.onloadedmetadata = null;
    };
    const seekResolvedVideo = () => {
      if (seekCompleted) return;
      if (!video || !inspectorRequestMatches(mediaGeneration, resolved.frame_uid)) {
        cleanupSeekListeners();
        return;
      }
      seekCompleted = true;
      video.currentTime = timestampMs / 1000;
      syncVideoSeekBar();
      cleanupSeekListeners();
    };
    if (video) {
      if (video.readyState >= 1) seekResolvedVideo();
      else {
        video.onloadedmetadata = seekResolvedVideo;
        seekEvents.forEach((eventName) => video.addEventListener(eventName, seekResolvedVideo));
      }
    }
    updatePlayerPosition();
    await refreshDetailTimeline(timestampMs);
    if (!inspectorRequestIsCurrent(mediaGeneration, resolved.frame_uid)) return;
    setStatus(`Applied ${resolved.frame_uid} → ${formatVideoTime(timestampMs / 1000)}.`);
  } catch (err) {
    setStatus(`Source index seek blocked — ${err.message}`, true);
  } finally {
    if (button) button.disabled = false;
  }
}

function openStageFrameEditor(stageId) {
  const item = inspectorStageItems.find((candidate) => candidate.stage_id === stageId);
  const panel = $("stageFrameEditPanel");
  const input = $("stageFrameUidInput");
  if (!item || !panel || !input) return;
  setInspectorActiveStage(stageId, { syncForm: true });
  $("stageFrameEditStage").textContent = stageId;
  $("stageFrameEditStatus").textContent = "";
  input.value = frameUidOf(item) || `${inspectorCurrentVideoId() || "video"}:`;
  panel.hidden = false;
  input.focus();
  input.select();
}

function closeStageFrameEditor() {
  const panel = $("stageFrameEditPanel");
  if (panel) panel.hidden = true;
  const status = $("stageFrameEditStatus");
  if (status) status.textContent = "";
}

function parseManualFrameId(value) {
  const raw = String(value || "").trim();
  const currentVideoId = inspectorCurrentVideoId();
  if (/^\d+$/.test(raw)) {
    return { videoId: currentVideoId, sourceFrameIdx: Number.parseInt(raw, 10) };
  }
  const separator = raw.lastIndexOf(":");
  if (separator <= 0) throw new Error("Use frame ID video_id:source_frame_idx.");
  const videoId = raw.slice(0, separator).trim();
  const sourceFrameIdx = Number.parseInt(raw.slice(separator + 1), 10);
  if (!videoId || !Number.isInteger(sourceFrameIdx) || sourceFrameIdx < 0) {
    throw new Error("Frame ID must contain a non-negative source_frame_idx.");
  }
  return { videoId, sourceFrameIdx };
}

async function applyManualStageFrame() {
  const stageId = inspectorActiveStageId;
  const status = $("stageFrameEditStatus");
  const button = $("stageFrameApplyBtn");
  const mediaGeneration = detailMediaGeneration;
  const frameIdAtStart = selectedFrameId;
  if (!stageId) return;
  try {
    const { videoId, sourceFrameIdx } = parseManualFrameId($("stageFrameUidInput")?.value);
    if (!videoId || videoId !== inspectorCurrentVideoId()) {
      throw new Error("Frame ID must belong to the selected video.");
    }
    if (button) button.disabled = true;
    if (status) status.textContent = "Validating canonical frame…";
    const resolved = await api(
      `/v1/videos/${encodeURIComponent(videoId)}/resolve?source_frame_idx=${sourceFrameIdx}`,
    );
    if (
      resolved.video_id !== videoId ||
      Number(resolved.source_frame_idx) !== sourceFrameIdx ||
      resolved.frame_uid !== `${videoId}:${sourceFrameIdx}`
    ) {
      throw new Error("Canonical frame identity validation failed.");
    }
    if (!inspectorRequestIsCurrent(mediaGeneration, frameIdAtStart)) return;
    if (!commitInspectorStageFrame(stageId, resolved, "manual")) {
      throw new Error("Frame could not be applied to this stage.");
    }
    if (status) status.textContent = `Saved ${resolved.frame_uid}.`;
    setStatus(`Updated ${stageId} to ${resolved.frame_uid}.`);
    closeStageFrameEditor();
  } catch (err) {
    if (status) status.textContent = `Not saved — ${err.message}`;
    setStatus(`Manual frame edit blocked — ${err.message}`, true);
  } finally {
    if (button) button.disabled = false;
  }
}

function syncInspectorStageTimeline() {
  const video = $("detailVideo");
  if (!video || !inspectorStageItems.length) return;
  const timestampSeconds = Number.isFinite(video.currentTime) ? video.currentTime : 0;
  const timeLabel = $("stageTimelinePlayheadTime");
  if (timeLabel) timeLabel.textContent = formatVideoTime(timestampSeconds);

  const nearest = inspectorStageItems.reduce((best, item) => {
    if (!best) return item;
    const bestDelta = Math.abs(Number(best.timestamp_ms || 0) / 1000 - timestampSeconds);
    const itemDelta = Math.abs(Number(item.timestamp_ms || 0) / 1000 - timestampSeconds);
    return itemDelta < bestDelta ? item : best;
  }, null);
  if (nearest) setInspectorActiveStage(nearest.stage_id);
}

function setupHorizontalWheelScroll(target) {
  const container = typeof target === "string" ? $(target) : target;
  if (!container || container.dataset.wheelScrollReady === "true") return;

  container.dataset.wheelScrollReady = "true";
  container.addEventListener("wheel", (event) => {
    const maxScrollLeft = container.scrollWidth - container.clientWidth;
    if (maxScrollLeft <= 0) return;

    const delta = Math.abs(event.deltaX) > Math.abs(event.deltaY)
      ? event.deltaX
      : event.deltaY;
    if (!delta) return;

    event.preventDefault();
    container.scrollLeft = Math.max(
      0,
      Math.min(maxScrollLeft, container.scrollLeft + delta),
    );
  }, { passive: false });
}

function updatePreviewDetailTimeline(timestampMs) {
  if (!PREVIEW_SAMPLE_MODE || !selectedFrameId) return;
  const candidates = previewNeighborItems(selectedFrameId);
  if (!candidates.length) return;
  const active = candidates.reduce((best, item) => {
    const bestDelta = Math.abs(Number(best.timestamp_ms || 0) - timestampMs);
    const itemDelta = Math.abs(Number(item.timestamp_ms || 0) - timestampMs);
    return itemDelta < bestDelta ? item : best;
  });
  const activeUid = frameUidOf(active);
  if ($("detailTimelineFrameId")?.textContent === activeUid && $("neighbors")?.children.length) return;
  renderDetailNeighborStrip(candidates, activeUid);
}

function scheduleDetailTimelineRefresh() {
  if (!selectedFrameId || $("detail")?.hidden) return;
  const timestampMs = videoTimeMs();
  if (PREVIEW_SAMPLE_MODE) {
    updatePreviewDetailTimeline(timestampMs);
    return;
  }

  if (lastDetailTimelineRequestMs != null &&
      Math.abs(timestampMs - lastDetailTimelineRequestMs) < DETAIL_TIMELINE_STEP_MS) {
    return;
  }
  if (detailTimelineTimer) clearTimeout(detailTimelineTimer);
  detailTimelineTimer = setTimeout(() => {
    detailTimelineTimer = null;
    void refreshDetailTimeline(timestampMs);
  }, 250);
}

async function refreshDetailTimeline(timestampMs) {
  const requestId = detailTimelineRequestId;
  const videoId = selectedResult?.video_id || selectedFrameId?.split(":")[0];
  const mediaGeneration = detailMediaGeneration;
  const frameIdAtStart = selectedFrameId;
  if (!videoId) return;
  lastDetailTimelineRequestMs = timestampMs;

  try {
    const resolved = await api(
      `/v1/videos/${encodeURIComponent(videoId)}/resolve?timestamp_ms=${timestampMs}`,
    );
    const resolvedUid = resolved.frame_uid || resolved.frame_id;
    if (!resolvedUid) return;
    const data = await api(`/frames/${encodeURIComponent(resolvedUid)}?window=${CONTEXT_WINDOW_FRAMES}`);
    if (requestId !== detailTimelineRequestId ||
        !inspectorRequestIsCurrent(mediaGeneration, frameIdAtStart)) return;
    lastResolvedPosition = { ...resolved, requested_timestamp_ms: timestampMs };
    updateDetailResolvedMetadata(resolved, timestampMs);
    currentTimeline = data.neighbors || [];
    renderDetailNeighborStrip(currentTimeline, resolvedUid);
    updatePlayerPosition();
  } catch {
    // Keep the last valid context strip when nearest-frame resolution is unavailable.
  }
}

/* ==========================================================================
   Temporal Media Inspector / Lightbox Modal
   ========================================================================== */

function stopDetailMedia() {
  detailMediaGeneration += 1;
  const detailVideo = $("detailVideo");
  if (detailVideo) {
    detailVideo.pause();
    detailVideo.onloadedmetadata = null;
    detailVideo.onerror = null;
    detailVideo.removeAttribute("src");
    detailVideo.load();
  }
  const previewFrame = $("previewFrame");
  if (previewFrame) {
    previewFrame.removeAttribute("src");
    previewFrame.hidden = true;
  }
  resetVideoSeekBar();
}

function primeDetailVideo(frameId) {
  const result = selectedResult;
  const detailVideo = $("detailVideo");
  const videoUrl = result?.video_url;
  if (!detailVideo || !videoUrl || result?.video_stream_available !== true) return null;

  const targetSeconds = Number(result.timestamp_ms || 0) / 1000;
  const mediaGeneration = detailMediaGeneration;
  detailVideo.pause();
  detailVideo.removeAttribute("src");
  detailVideo.load();
  detailVideo.onloadedmetadata = () => {
    if (mediaGeneration === detailMediaGeneration && selectedFrameId === frameId) {
      detailVideo.currentTime = targetSeconds;
    }
  };
  $("previewFrame")?.removeAttribute("src");
  if ($("previewFrame")) $("previewFrame").hidden = true;
  if ($("videoBox")) $("videoBox").hidden = false;
  $("videoStatus").textContent = "Starting bounded video stream…";
  // Start this request before the context fetch hides the metadata round-trip.
  detailVideo.src = videoUrl;
  return { video_url: videoUrl, target_seconds: targetSeconds };
}

async function openDetail(frameId) {
  if (!frameId) return;
  stopDetailMedia();
  if ($("sourceFrameIdxInput")) $("sourceFrameIdxInput").value = "";
  const openGeneration = detailMediaGeneration;
  inspectorRangeMarkers = { L: null, R: null };
  renderRangeMarkerStatus();
  detailTimelineRequestId += 1;
  if (detailTimelineTimer) clearTimeout(detailTimelineTimer);
  detailTimelineTimer = null;
  lastDetailTimelineRequestMs = null;

  if (PREVIEW_SAMPLE_MODE) {
    const sample = (lastSearchData?.results || []).find((item) =>
      (item.frame_uid || item.frame_id) === frameId
    ) || previewQueueItems.find((item) =>
      (item.frame_uid || item.frame_id) === frameId
    ) || selectedResult;
    if (!sample) return;

    selectedResult = sample;
    selectedFrameId = frameId;
    lastResolvedPosition = null;
    lastExactExtraction = null;
    updateSelectedCardStyles();

    const stageTag = $("detailStageTag");
    const scoreTag = $("detailScoreTag");
    $("detailTitle").textContent = sample.frame_uid || sample.frame_id || frameId;
    if (isDirectVideoResult(sample)) {
      stageTag.textContent = "";
      stageTag.hidden = true;
      scoreTag.textContent = "";
      scoreTag.hidden = true;
    } else {
      stageTag.textContent = sample.stage_id || "S1";
      stageTag.className = `inspector-badge ${stageBadgeClassFor(sample.stage_id)}`;
      stageTag.hidden = false;
      scoreTag.textContent = `Score: ${Number(sample.final_score || 0).toFixed(4)}`;
      scoreTag.hidden = false;
    }

    const videoBox = $("videoBox");
    const detailVideo = $("detailVideo");
    const previewFrame = $("previewFrame");
    const keyframeFallbackBox = $("keyframeFallbackBox");
    const videoStatus = $("videoStatus");
    const mediaGeneration = detailMediaGeneration;
    detailVideo.pause();
    detailVideo.removeAttribute("src");
    detailVideo.load();
    resetVideoSeekBar();
    detailVideo.onloadedmetadata = null;
    detailVideo.onerror = null;
    previewFrame.hidden = true;
    previewFrame.removeAttribute("src");
    $("detailImage").src = sample.image_url || "";
    $("imageStatus").hidden = false;
    $("imageStatus").textContent = "PREVIEW SAMPLE · synthetic keyframe image; video uses HF pinned preview source.";

    if (sample.video_url && sample.video_stream_available) {
      videoBox.hidden = false;
      keyframeFallbackBox.hidden = true;
      const targetSeconds = Number(sample.timestamp_ms || 0) / 1000;
      detailVideo.onloadedmetadata = () => {
        if (mediaGeneration === detailMediaGeneration) detailVideo.currentTime = targetSeconds;
      };
      detailVideo.onerror = () => {
        if (mediaGeneration !== detailMediaGeneration) return;
        detailVideo.removeAttribute("src");
        detailVideo.load();
        videoBox.hidden = true;
        keyframeFallbackBox.hidden = false;
        videoStatus.textContent =
          "HF pinned preview source unavailable in this browser; showing the synthetic keyframe fixture.";
      };
      detailVideo.src = sample.video_url;
      detailVideo.load();
      detailVideo.currentTime = targetSeconds;
      videoStatus.textContent =
        `HF pinned preview source @ ${sample.video_revision}; timestamp-aligned proxy, exact source frame not claimed.`;
    } else {
      videoBox.hidden = true;
      keyframeFallbackBox.hidden = false;
      videoStatus.textContent = "Preview video source is not declared for this sample; showing the fixture image.";
    }
    $("playerPosition").textContent =
      `video: ${sample.video_id || "—"} · time: ${sample.timestamp_ms || 0} ms · source_frame_idx: ${sample.source_frame_idx ?? "—"}`;
    $("detailMeta").innerHTML = "";
    for (const [key, value] of Object.entries({
      "Video ID": sample.video_id,
      "Source Frame Idx": sample.source_frame_idx,
      "Frame UID": sample.frame_uid || frameId,
      "Timestamp": `${((sample.timestamp_ms || 0) / 1000).toFixed(3)} s`,
      "Shot ID": sample.shot_id || "none",
      "Video Source": sample.video_backend === "huggingface_http_range"
        ? `Hugging Face @ ${sample.video_revision}`
        : "PREVIEW_ONLY",
      "Provenance": "ENGINEERING_PROXY / PREVIEW_ONLY",
    })) {
      const term = document.createElement("dt");
      const desc = document.createElement("dd");
      term.textContent = key;
      desc.textContent = String(value ?? "—");
      $("detailMeta").append(term, desc);
    }
    currentTimeline = previewNeighborItems(frameId);
    lastDetailTimelineRequestMs = Number(sample.timestamp_ms || 0);
    renderDetailNeighborStrip(currentTimeline, frameId);
    renderInspectorStageTimeline();
    const selectedTask = String(sample.submission_task || "").toUpperCase();
    setInspectorTask(selectedTask === "TRAKE" ? "TRAKE" : selectedTask === "QA" ? "QA" : "KIS");
    renderAsrEvidence(sample);
    $("submissionPreview").textContent =
      "Preview sample only — no submission or backend request was made.";
    $("detail").hidden = false;
    setStatus(`Preview sample: ${frameId}`);
    return;
  }

  const primedVideo = primeDetailVideo(frameId);
  const mediaGeneration = openGeneration;
  try {
    setStatus(`Loading frame ${frameId}…`, false, true);
    const data = await api(`/frames/${encodeURIComponent(frameId)}?window=${CONTEXT_WINDOW_FRAMES}`);
    if (openGeneration !== detailMediaGeneration) return;
    selectedFrameId = frameId;
    lastResolvedPosition = null;
    lastExactExtraction = null;
    updateSelectedCardStyles();

    const f = data.frame || {};
    const metadata = f.metadata || {};

    $("detailTitle").textContent = f.frame_uid || f.frame_id || frameId;
    const stageTag = $("detailStageTag");
    const scoreTag = $("detailScoreTag");

    if (selectedResult && !isDirectVideoResult(selectedResult)) {
      stageTag.textContent = selectedResult.stage_id || "S1";
      stageTag.className = `inspector-badge ${stageBadgeClassFor(selectedResult.stage_id)}`;
      stageTag.hidden = false;
      scoreTag.textContent = `Score: ${Number(selectedResult.final_score || 0).toFixed(4)}`;
      scoreTag.hidden = false;
    } else {
      stageTag.textContent = "";
      scoreTag.textContent = "";
      stageTag.hidden = true;
      scoreTag.hidden = true;
    }

    // Keyframe Image Section
    const image = $("detailImage");
    const imageStatusEl = $("imageStatus");
    const imageAvailable = data.image_available ?? f.image_available;
    const imageStatus = data.image_status || f.image_status || "IMAGE_STATUS_UNKNOWN";
    const imageReason = data.image_reason || f.image_reason;

    if (imageAvailable === false) {
      image.removeAttribute("src");
      image.alt = "Keyframe unavailable";
      if (imageStatusEl) {
        imageStatusEl.textContent = `Keyframe unavailable (${imageStatus}): ${imageReason || "reason not declared"}.`;
      }
    } else {
      image.src = data.image_url || f.image_url || `/frames/${encodeURIComponent(frameId)}/image`;
      image.alt = f.frame_uid || f.frame_id || frameId;
      image.onerror = () => {
        if (!inspectorRequestIsCurrent(mediaGeneration, frameId)) return;
        image.removeAttribute("src");
        if (imageStatusEl) {
          imageStatusEl.textContent = "Keyframe unavailable (REMOTE_MEDIA_UNAVAILABLE): remote fetch failed.";
        }
      };
      if (imageStatusEl) {
        imageStatusEl.textContent = imageStatus === "AVAILABLE_REMOTE"
          ? "Keyframe available on demand (manifest & hash gated)."
          : "Keyframe available locally.";
      }
    }

    // Video Section
    const videoBox = $("videoBox");
    const detailVideo = $("detailVideo");
    const previewFrame = $("previewFrame");
    const videoStatus = $("videoStatus");
    const videoAvailable = data.video_stream_available === true ||
      (data.video_stream_available == null && data.video_available === true);
    const videoUrl = data.video_url || `/videos/${encodeURIComponent(f.video_id)}/stream`;
    const reusePrimedVideo = videoAvailable && primedVideo?.video_url === videoUrl;

    currentTimeline = data.neighbors || [];
    lastDetailTimelineRequestMs = Number(f.timestamp_ms || 0);
    if (!reusePrimedVideo) {
      detailVideo.pause();
      detailVideo.removeAttribute("src");
      detailVideo.load();
    }
    resetVideoSeekBar();
    previewFrame.hidden = true;
    previewFrame.removeAttribute("src");
    videoBox.hidden = false;

    if (videoAvailable) {
      if (!reusePrimedVideo) detailVideo.src = videoUrl;
      const targetSeconds = Number(f.timestamp_ms || 0) / 1000;
      detailVideo.onloadedmetadata = () => {
        if (mediaGeneration === detailMediaGeneration) detailVideo.currentTime = targetSeconds;
      };
      if (detailVideo.readyState >= 1) detailVideo.currentTime = targetSeconds;
      videoStatus.textContent =
        "ENGINEERING_PROXY source video; seek is timestamp-aligned and is not an exact source frame export. Quality: UNVALIDATED.";
    } else {
      videoStatus.textContent =
        `Source video streaming unavailable (${data.video_stream_status || data.video_status || "NO_VIDEO_MANIFEST"})` +
        `${data.video_stream_reason ? `: ${data.video_stream_reason}` : "."} ` +
        "Preview timestamp only; exact frame export blocked.";

      try {
        const availability = await api(`/v1/videos/${encodeURIComponent(f.video_id)}/availability`);
        if (!inspectorRequestMatches(mediaGeneration, frameId)) return;
        if (availability.external_id) {
          previewFrame.src = `https://www.youtube.com/embed/${encodeURIComponent(availability.external_id)}?start=${Math.floor(Number(f.timestamp_ms || 0) / 1000)}&enablejsapi=1`;
          previewFrame.hidden = false;
          videoStatus.textContent += ` YouTube preview at ${f.timestamp_ms} ms; proxy only.`;
        }
      } catch {
        /* Advisory fallback */
      }
    }

    // Metadata Grid
    const metaEl = $("detailMeta");
    if (metaEl) {
      metaEl.innerHTML = "";
      const rows = {
        "Video ID": f.video_id,
        "Source Frame Idx": f.source_frame_idx ?? f.frame_idx ?? "—",
        "Frame UID": f.frame_uid || f.frame_id,
        "Timestamp": `${((f.timestamp_ms || 0) / 1000).toFixed(3)} s (${f.timestamp_ms || 0} ms)`,
        "Timestamp Source": metadata.timestamp_source || "legacy mapping",
        "Shot ID": f.shot_id || "none",
        "Video Stream": data.video_stream_status || data.video_status || "unknown",
        "Provenance": data.video_provenance_status || "—",
        "Title": metadata.title || "—",
      };
      for (const [k, v] of Object.entries(rows)) {
        const term = document.createElement("dt");
        const desc = document.createElement("dd");
        term.textContent = k;
        desc.textContent = String(v);
        metaEl.append(term, desc);
      }
    }
    renderAsrEvidence(selectedResult);

    // Neighbor Timeline Strip in Inspector: anchored to the selected keyframe.
    renderDetailNeighborStrip(currentTimeline, frameId);
    renderInspectorStageTimeline();
    const selectedTask = String(selectedResult?.submission_task || "").toUpperCase();
    setInspectorTask(selectedTask === "TRAKE" ? "TRAKE" : selectedTask === "QA" ? "QA" : "KIS");

    $("submissionPreview").textContent =
      "No preview yet. Canonical preview only — nothing is submitted anywhere.";
    $("detail").hidden = false;
    updatePlayerPosition();
    setStatus("");
  } catch (err) {
    setStatus("Frame load failed — " + err.message, true);
  }
}

function hideDetail() {
  stopDetailMedia();
  const detailModal = $("detail");
  if (detailModal) detailModal.hidden = true;
  if (detailTimelineTimer) clearTimeout(detailTimelineTimer);
  detailTimelineTimer = null;
  detailTimelineRequestId += 1;
  lastDetailTimelineRequestMs = null;
  inspectorStageItems = [];
  inspectorActiveStageId = null;
  $("stageTimelineMarkers")?.replaceChildren();
  $("inspectorStageSequence")?.replaceChildren();
  selectedFrameId = null;
  lastResolvedPosition = null;
  lastExactExtraction = null;
  inspectorRangeMarkers = { L: null, R: null };
  renderRangeMarkerStatus();
  const asrEvidence = $("asrEvidence");
  if (asrEvidence) {
    asrEvidence.hidden = true;
    asrEvidence.textContent = "";
  }
  for (const card of document.querySelectorAll(".result-card")) {
    card.classList.remove("selected");
  }
}

function formatVideoTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function resetVideoSeekBar() {
  const seekBar = $("videoSeekBar");
  const duration = $("videoDuration");
  if (seekBar) {
    seekBar.disabled = true;
    seekBar.max = "0";
    seekBar.value = "0";
  }
  if (duration) duration.textContent = "0:00 / —";
}

function syncVideoSeekBar() {
  const video = $("detailVideo");
  const seekBar = $("videoSeekBar");
  const duration = $("videoDuration");
  if (!video || !seekBar) return;

  const total = Number(video.duration);
  const current = Number(video.currentTime);
  const hasDuration = Number.isFinite(total) && total > 0;
  seekBar.disabled = !hasDuration;
  seekBar.max = hasDuration ? String(total) : "0";
  seekBar.value = hasDuration
    ? String(Math.min(Math.max(Number.isFinite(current) ? current : 0, 0), total))
    : "0";
  if (duration) {
    duration.textContent = `${formatVideoTime(Number.isFinite(current) ? current : 0)} / ${formatVideoTime(total)}`;
  }
}

function videoTimeMs() {
  const video = $("detailVideo");
  return Math.max(0, Math.round((Number.isFinite(video.currentTime) ? video.currentTime : 0) * 1000));
}

function inspectorRequestIsCurrent(mediaGeneration, frameId = selectedFrameId) {
  return inspectorRequestMatches(mediaGeneration, frameId) &&
    !$("detail")?.hidden;
}

function inspectorRequestMatches(mediaGeneration, frameId = selectedFrameId) {
  return mediaGeneration === detailMediaGeneration && selectedFrameId === frameId;
}

function updateDetailResolvedMetadata(resolved, requestedTimestampMs = null) {
  if (!resolved) return;
  const frameUid = resolved.frame_uid || resolved.resolved_frame_uid;
  const sourceFrameIdx = resolved.source_frame_idx;
  const resolvedTimestampMs = Number(resolved.resolved_timestamp_ms ?? resolved.timestamp_ms);
  const meta = $("detailMeta");
  if (!meta) return;
  const values = new Map([
    ["Source Frame Idx", sourceFrameIdx ?? "—"],
    ["Frame UID", frameUid || "—"],
    ["Timestamp", Number.isFinite(resolvedTimestampMs)
      ? `${(resolvedTimestampMs / 1000).toFixed(3)} s (${resolvedTimestampMs} ms)`
      : "—"],
    ["Timestamp Source", resolved.mapping_method || resolved.mapping_status || "—"],
  ]);
  for (const [label, value] of values) {
    const term = [...meta.querySelectorAll("dt")].find((candidate) => candidate.textContent.trim() === label);
    if (term?.nextElementSibling) term.nextElementSibling.textContent = String(value);
  }
  const requested = Number(requestedTimestampMs);
  const statusLabel = "Mapping Status";
  let statusTerm = [...meta.querySelectorAll("dt")].find((candidate) => candidate.textContent.trim() === statusLabel);
  if (!statusTerm) {
    statusTerm = document.createElement("dt");
    statusTerm.textContent = statusLabel;
    const statusValue = document.createElement("dd");
    meta.append(statusTerm, statusValue);
  }
  if (statusTerm.nextElementSibling) {
    const delta = Number.isFinite(requested) && Number.isFinite(resolvedTimestampMs)
      ? ` · Δ ${Math.abs(resolvedTimestampMs - requested)} ms`
      : "";
    statusTerm.nextElementSibling.textContent = `${resolved.mapping_status || "RESOLVED"}${delta}`;
  }
}

function clearExactExtractionStatus() {
  const imageStatus = $("imageStatus");
  if (!imageStatus || imageStatus.dataset.exactExtraction !== "true") return;
  imageStatus.hidden = true;
  imageStatus.textContent = "";
  delete imageStatus.dataset.exactExtraction;
}

function normalizeQaAnswer(value) {
  const answer = String(value ?? "").trim().slice(0, 100);
  return answer || null;
}

function getQueueQaAnswer(queryId = null, queueItems = currentQueueItems) {
  const answers = [...new Set(
    queueItems
      .filter((item) => !queryId || item.query_id === queryId)
      .filter((item) => {
        const task = String(item.submission_task || "").trim().toUpperCase();
        return !task || task === "QA";
      })
      .map((item) => normalizeQaAnswer(item.qa_answer || item.answer))
      .filter(Boolean),
  )];
  if (!answers.length) {
    throw new Error("QA answer is empty; open a queued candidate and apply the answer there.");
  }
  if (answers.length > 1) {
    throw new Error("Queue contains multiple QA answers; keep one answer across the queue.");
  }
  return answers[0];
}

function queueItemsForAic26(task, queryId) {
  const normalizedTask = String(task || "KIS").trim().toUpperCase();
  return currentQueueItems.filter((item) => {
    if (String(item.query_id || "") !== String(queryId || "")) return false;
    const itemTask = String(item.submission_task || "").trim().toUpperCase();
    return normalizedTask === "TRAKE"
      ? itemTask === "TRAKE"
      : !itemTask || itemTask === normalizedTask;
  });
}

function aic26QueueTaskLabel(item) {
  const task = String(item?.submission_task || "").trim().toUpperCase();
  return task || "UNSPECIFIED";
}

function aic26QueueItemsForQuery(queryId) {
  return currentQueueItems.filter((item) => String(item?.query_id || "") === String(queryId || ""));
}

function aic26QueueTaskMismatch(task, queryId, queueItems) {
  const queuedTasks = [...new Set(queueItems.map(aic26QueueTaskLabel))].sort();
  const requestedTask = String(task || "KIS").trim().toUpperCase();
  return new Error(
    `Queue task mismatch for query ${queryId}: selected ${requestedTask}, ` +
    `queued ${queuedTasks.join(", ")}. Select the queued task or clear incompatible items.`,
  );
}

function syncAuthoritativeSubmissionTaskFromQueue(items) {
  const queueItems = Array.isArray(items) ? items : [];
  const queuedTasks = [...new Set(
    queueItems
      .map((item) => String(item?.submission_task || "").trim().toUpperCase())
      .filter(Boolean),
  )];
  if (queuedTasks.length !== 1 || queueItems.some((item) => !String(item?.submission_task || "").trim())) return;
  const queuedTask = queuedTasks[0];
  if (["KIS", "QA", "TRAKE"].includes(queuedTask) && queuedTask !== activeSubmissionTask) {
    setAuthoritativeSubmissionTask(queuedTask);
  }
}

function trakeStageIdsForMetadata(result) {
  const targetVideoId = String(result?.video_id || selectedResult?.video_id || "");
  const targetChainId = String(
    result?.track_id || result?.chain_id || result?.bundle_id || "",
  );
  const candidates = [
    ...(Array.isArray(result?.queue_group_items) ? result.queue_group_items : []),
    ...(Array.isArray(result?.stages) ? result.stages : []),
    ...(Array.isArray(selectedResult?.queue_group_items) ? selectedResult.queue_group_items : []),
    ...(Array.isArray(selectedResult?.stages) ? selectedResult.stages : []),
    ...(Array.isArray(inspectorStageItems) ? inspectorStageItems : []),
    result,
  ];
  const stageIds = new Set();
  for (const item of candidates) {
    if (!item) continue;
    const itemVideoId = String(item.video_id || "");
    const itemChainId = String(item.track_id || item.chain_id || item.bundle_id || "");
    if (targetVideoId && itemVideoId && itemVideoId !== targetVideoId) continue;
    if (targetChainId && itemChainId && itemChainId !== targetChainId) continue;
    const candidateStageId = item.stage_id;
    if (candidateStageId) stageIds.add(domStageId(candidateStageId));
  }
  return [...stageIds].sort((left, right) => {
    const leftNumber = stageNumber(left) || Number.MAX_SAFE_INTEGER;
    const rightNumber = stageNumber(right) || Number.MAX_SAFE_INTEGER;
    return leftNumber - rightNumber || left.localeCompare(right);
  });
}

function trakeQueueMetadata(result, stageId, queryId, videoId) {
  const normalizedStageId = domStageId(stageId || result?.stage_id || "S1");
  const stageIndex = Math.max(1, stageNumber(normalizedStageId) || 1);
  const explicitEventStep = Number(result?.event_step);
  const explicitSelectionKind = String(result?.selection_kind || "").trim();
  const explicitMetadataIsValid = Number.isInteger(explicitEventStep) && explicitEventStep >= 0 && (
    !explicitSelectionKind || explicitSelectionKind === `E${explicitEventStep + 1}`
  );
  const stageIds = trakeStageIdsForMetadata(result);
  const contiguousStageIndex = stageIds.indexOf(normalizedStageId);
  const hasAuthoritativeStageOrder = stageIds.length > 0 && contiguousStageIndex >= 0;
  const eventStep = hasAuthoritativeStageOrder
    ? contiguousStageIndex
    : explicitMetadataIsValid
      ? explicitEventStep
      : stageIndex - 1;
  const authoritativeSelectionKind = `E${eventStep + 1}`;
  const selectionKind = hasAuthoritativeStageOrder
    ? authoritativeSelectionKind
    : explicitMetadataIsValid && explicitSelectionKind
      ? explicitSelectionKind
      : authoritativeSelectionKind;
  return {
    stage_id: normalizedStageId,
    chain_id: result?.track_id || result?.chain_id || result?.bundle_id || `${queryId}:${videoId}`,
    event_step: eventStep,
    selection_kind: selectionKind,
  };
}

function temporalModeEnabled(item) {
  return temporalModeValue(item) === true;
}

function validateAic26Queue(task, queryId) {
  const normalizedTask = String(task || "KIS").trim().toUpperCase();
  const scopedQueueItems = aic26QueueItemsForQuery(queryId);
  if (!scopedQueueItems.length) {
    throw new Error(`No queue items for query ${queryId}.`);
  }
  const incompatibleItems = scopedQueueItems.filter((item) => {
    const itemTask = String(item?.submission_task || "").trim().toUpperCase();
    return itemTask && itemTask !== normalizedTask;
  });
  if (incompatibleItems.length || (
    normalizedTask === "TRAKE" && scopedQueueItems.some((item) => !String(item?.submission_task || "").trim())
  )) {
    throw aic26QueueTaskMismatch(normalizedTask, queryId, scopedQueueItems);
  }
  const queueItems = queueItemsForAic26(normalizedTask, queryId);
  if (!queueItems.length) {
    throw aic26QueueTaskMismatch(normalizedTask, queryId, scopedQueueItems);
  }
  if (normalizedTask !== "TRAKE") return queueItems;

  const groups = new Map();
  for (const item of queueItems) {
    const chainId = String(item.chain_id || "").trim();
    if (!chainId) {
      throw new Error("TRAKE queue item is missing chain_id metadata.");
    }
    const eventStep = Number(item.event_step);
    if (!Number.isInteger(eventStep) || eventStep < 0) {
      throw new Error(`TRAKE chain ${chainId} has an invalid event_step.`);
    }
    const stageId = domStageId(item.stage_id);
    const expectedSelectionKind = `E${eventStep + 1}`;
    if (String(item.selection_kind || "").toUpperCase() !== expectedSelectionKind) {
      throw new Error(`TRAKE chain ${chainId} has a stage_id/event_step/selection_kind mismatch.`);
    }
    const videoId = String(item.video_id || "").trim();
    const sourceFrameIdx = Number(item.source_frame_idx);
    const timestampMs = Number(item.timestamp_ms);
    if (item.stage_draft_pending || item.stage_frame_pending) {
      throw new Error(`TRAKE chain ${chainId} contains a pending stage; Extract it before export.`);
    }
    if (
      !videoId ||
      !Number.isInteger(sourceFrameIdx) ||
      sourceFrameIdx < 0 ||
      !Number.isInteger(timestampMs) ||
      timestampMs < 0
    ) {
      throw new Error(`TRAKE chain ${chainId} has an invalid video/source_frame_idx.`);
    }
    const expectedFrameUid = `${videoId}:${sourceFrameIdx}`;
    if (item.frame_uid && String(item.frame_uid) !== expectedFrameUid) {
      throw new Error(`TRAKE chain ${chainId} has a frame_uid/source_frame_idx identity mismatch.`);
    }
    const group = groups.get(chainId) || [];
    group.push({ item, eventStep, videoId, sourceFrameIdx, timestampMs });
    groups.set(chainId, group);
  }

  let eventCount = null;
  for (const [chainId, members] of groups) {
    members.sort((left, right) => left.eventStep - right.eventStep);
    const expectedSteps = members.map((_member, index) => index);
    const actualSteps = members.map((member) => member.eventStep);
    if (actualSteps.some((step, index) => step !== expectedSteps[index])) {
      throw new Error(`TRAKE chain ${chainId} must contain contiguous E1..EN event_step values.`);
    }
    const physicalStageNumbers = members.map((member) => stageNumber(domStageId(member.item.stage_id)));
    if (physicalStageNumbers.some((stage, index) =>
      !Number.isInteger(stage) ||
      stage < 1 ||
      stage > STAGED_MAX_STAGES ||
      (index > 0 && stage <= physicalStageNumbers[index - 1])
    )) {
      throw new Error(`TRAKE chain ${chainId} physical stage_id values must be strictly ordered.`);
    }
    const videos = new Set(members.map((member) => member.videoId));
    if (videos.size !== 1) {
      throw new Error(`TRAKE chain ${chainId} spans multiple videos.`);
    }
    const sourceIndexes = members.map((member) => member.sourceFrameIdx);
    if (sourceIndexes.some((value, index) => index > 0 && value <= sourceIndexes[index - 1])) {
      throw new Error(`TRAKE chain ${chainId} requires strictly increasing source_frame_idx.`);
    }
    const timestamps = members.map((member) => member.timestampMs);
    const temporalEnabled = members.some((member) => temporalModeEnabled(member.item));
    if (timestamps.some((value, index) => index > 0 && (
      value <= timestamps[index - 1] ||
      (temporalEnabled && value - timestamps[index - 1] > TRAKE_MAX_DELTA_MS)
    ))) {
      throw new Error(
        temporalEnabled
          ? `TRAKE chain ${chainId} must keep each timestamp gap within 60s and in order.`
          : `TRAKE chain ${chainId} timestamps must be strictly increasing.`,
      );
    }
    const currentEventCount = members.length;
    if (eventCount == null) {
      eventCount = currentEventCount;
    } else if (eventCount !== currentEventCount) {
      throw new Error("TRAKE chains must have the same event count.");
    }
  }
  return queueItems;
}

function queueItemForInspectorStage(stageId, frameUid) {
  const normalizedStage = String(stageId || "").toUpperCase();
  const queryId = currentInspectorQueryId();
  const scopedItems = currentQueueItems.filter((item) =>
    String(item.query_id || "") === queryId,
  );
  const exact = scopedItems.find((item) =>
    String(item.frame_uid || "") === String(frameUid || "") &&
    String(item.stage_id || "").toUpperCase() === normalizedStage,
  );
  return exact || scopedItems.find((item) => String(item.frame_uid || "") === String(frameUid || "")) || null;
}

function markInspectorStagePendingAtPlayhead(timestampMs) {
  ensureInspectorContext();
  const item = inspectorActiveStageItem();
  const stageId = String(item?.stage_id || "").toUpperCase();
  if (!item || !stageId || item.stage_draft_pending) return false;
  const currentDraft = inspectorStageDrafts.get(stageId) || {};
  const pending = {
    ...item,
    ...currentDraft,
    stage_id: stageId,
    frame_id: null,
    frame_uid: null,
    source_frame_idx: null,
    timestamp_ms: timestampMs,
    requested_timestamp_ms: timestampMs,
    resolved_timestamp_ms: null,
    mapping_status: "PENDING_EXACT_EXTRACT",
    stage_frame_source: "playhead",
    stage_frame_verified: false,
    stage_frame_pending: true,
    stage_draft_pending: true,
  };
  inspectorStageDrafts.set(stageId, pending);
  inspectorStageItems = inspectorStageItems.map((candidate) =>
    String(candidate.stage_id || "").toUpperCase() === stageId
      ? pending
      : candidate,
  );
  lastResolvedPosition = null;
  renderInspectorStageTimeline();
  return true;
}

function updatePlayerPosition() {
  syncVideoSeekBar();
  syncInspectorStageTimeline();
  if (!selectedFrameId) return;
  const timestamp = videoTimeMs();
  const exactPosition = lastExactExtraction?.requested_timestamp_ms === timestamp &&
    ["PASS", "MANUAL_CANONICAL"].includes(lastExactExtraction.raw_verification)
    ? lastExactExtraction
    : null;
  if (lastExactExtraction && !exactPosition) {
    lastExactExtraction = null;
    clearExactExtractionStatus();
  }
  // Seeking is navigation only. It must not erase an already resolved or
  // extracted stage. Stage mutation is explicit through Mark/Extract/Apply or
  // an intentional timeline-marker drag.
  const resolvedPosition = exactPosition || (
    lastResolvedPosition &&
    Number(lastResolvedPosition.requested_timestamp_ms ?? lastResolvedPosition.selected_time_ms) === timestamp
      ? lastResolvedPosition
      : null
  );
  const pendingStage = Boolean(inspectorActiveStageItem()?.stage_draft_pending);
  const resolvedIdx = pendingStage
    ? "pending exact Extract"
    : resolvedPosition?.source_frame_idx ?? selectedResult?.source_frame_idx ?? selectedResult?.frame_idx ?? "—";
  const posEl = $("playerPosition");
  if (posEl) {
    posEl.textContent =
      `video: ${selectedFrameId.split(":")[0]} · time: ${timestamp} ms · source_frame_idx: ${resolvedIdx}` +
      (exactPosition
        ? ` · exact: ${exactPosition.raw_verification}`
        : resolvedPosition
          ? ` · resolved: ${resolvedPosition.mapping_status}`
          : pendingStage
            ? " · exact: pending Extract"
            : "");
  }
  scheduleDetailTimelineRefresh();
}

function seekVideo(delta) {
  const video = $("detailVideo");
  video.currentTime = Math.max(0, (video.currentTime || 0) + delta);
}

function replayVideo() {
  const video = $("detailVideo");
  const targetSeconds = Number(selectedResult?.timestamp_ms || 0) / 1000;
  video.currentTime = targetSeconds;
  video.play().catch(() => {});
}

function inspectorNavigationVideos() {
  if (!lastSearchData || isDirectVideoResult()) return [];

  const source = stageResultItems(lastSearchData)
    .filter((item) => currentStageFilter === "all" || item.stage_id === currentStageFilter)
    .filter((item) => Boolean(frameUidOf(item)) && String(item.video_id || "").trim());
  const groups = new Map();

  source.forEach((item, order) => {
    const videoId = String(item.video_id).trim();
    if (!groups.has(videoId)) {
      groups.set(videoId, {
        video_id: videoId,
        first_order: order,
        best_score: resultScore(item),
        items: [],
      });
    }
    const group = groups.get(videoId);
    group.items.push(item);
    group.best_score = Math.max(group.best_score, resultScore(item));
  });

  return [...groups.values()]
    .sort((left, right) => {
      const scoreDelta = right.best_score - left.best_score;
      return Math.abs(scoreDelta) > 1e-12
        ? scoreDelta
        : left.first_order - right.first_order;
    })
    .map((group) => [...group.items].sort(compareStageCandidates)[0])
    .filter(Boolean);
}

function selectAdjacentVideo(direction) {
  const candidates = inspectorNavigationVideos();
  const currentVideoId = String(selectedResult?.video_id || selectedFrameId?.split(":")[0] || "");
  const currentIndex = candidates.findIndex((item) => String(item.video_id) === currentVideoId);
  if (currentIndex < 0) return false;

  const target = candidates[currentIndex + Number(direction || 0)];
  if (!target) return false;
  selectResult(target, { inspect: true });
  return true;
}

function selectNeighbor(direction) {
  if (!selectedFrameId || !currentTimeline.length) return;
  const activeUid = lastResolvedPosition?.frame_uid || selectedFrameId;
  const index = currentTimeline.findIndex((item) => frameUidOf(item) === activeUid);
  const target = currentTimeline[index + direction];
  if (target) selectResult(timelineResult(target), { inspect: true });
}

/* ===========================================================================
   Exact Source Frame & Review Queue Integration
   ========================================================================== */

async function extractSourceFrame() {
  if (!selectedFrameId) return;
  if (PREVIEW_SAMPLE_MODE) {
    setStatus("Raw video decode is unavailable in preview sample mode.", true);
    return;
  }

  const videoId = selectedFrameId.split(":")[0];
  const stageId = String(inspectorActiveStageId || selectedResult?.stage_id || "S1").toUpperCase();
  const requestedTimestampMs = videoTimeMs();
  const button = $("btnExtractSourceFrame");
  const mediaGeneration = detailMediaGeneration;
  const frameIdAtStart = selectedFrameId;
  if (button) button.disabled = true;
  lastResolvedPosition = null;
  lastExactExtraction = null;
  clearExactExtractionStatus();

  try {
    // The raw endpoint is authoritative: with canonical PTS it joins the
    // decoded timestamp, otherwise it returns the on-demand decoder's own
    // presentation-order source_frame_idx.
    setStatus(`Decoding exact source frame at ${requestedTimestampMs} ms…`, false, true);
    const decoded = await api(
      `/v1/videos/${encodeURIComponent(videoId)}/raw-frame?timestamp_ms=${requestedTimestampMs}`,
    );
    if (!inspectorRequestIsCurrent(mediaGeneration, frameIdAtStart)) return;
    if (decoded.raw_verification !== "PASS") {
      throw new Error(`EXACT_SOURCE_FRAME_VERIFICATION_FAILED (${decoded.frame_uid || "—"})`);
    }
    const resolved = decoded;
    const verifiedPosition = {
      ...resolved,
      requested_timestamp_ms: requestedTimestampMs,
      raw_verification: "PASS",
      decoded_timestamp_ms: decoded.decoded_timestamp_ms,
      raw_decode_delta_ms: decoded.delta_ms,
      raw_mapping_delta_ms: decoded.decoded_mapping_delta_ms,
      raw_image_data_url: decoded.image_data_url,
    };
    if (inspectorStageItems.length && !commitInspectorStageFrame(stageId, resolved, "extract")) {
      throw new Error(`Could not apply exact frame to ${stageId}`);
    }
    lastExactExtraction = verifiedPosition;
    lastResolvedPosition = verifiedPosition;
    updateDetailResolvedMetadata(resolved, requestedTimestampMs);
    updatePlayerPosition();
    const detailImage = $("detailImage");
    if (detailImage && decoded.image_data_url) {
      detailImage.src = decoded.image_data_url;
      detailImage.alt = resolved.frame_uid;
    }
    const imageStatus = $("imageStatus");
    if (imageStatus) {
      imageStatus.hidden = false;
      imageStatus.dataset.exactExtraction = "true";
      imageStatus.textContent =
        `Exact raw frame · ${resolved.frame_uid} · src #${resolved.source_frame_idx} · ` +
        `decoded ${decoded.decoded_timestamp_ms} ms · Δ ${decoded.delta_ms} ms.`;
    }
    setStatus(
      `Extracted exact ${resolved.frame_uid} · src #${resolved.source_frame_idx} · raw Δ ${decoded.delta_ms} ms`,
    );
  } catch (err) {
    lastResolvedPosition = null;
    lastExactExtraction = null;
    clearExactExtractionStatus();
    setStatus(`Exact extraction blocked — ${err.message}`, true);
  } finally {
    if (button) button.disabled = false;
  }
}

function queueItemIdentity(item) {
  const task = String(item?.submission_task || "KIS").trim().toUpperCase();
  return [
    item?.query_id || "",
    task,
    item?.bundle_id || "",
    item?.chain_id || "",
    item?.event_step ?? "",
    domStageId(item?.stage_id || "S1"),
    item?.frame_uid || item?.frame_id || "",
  ].join("\u0000");
}

async function addResultToQueue(result, stageId, queryIdOverride = null) {
  const requestedTask = activeSubmissionTask;
  const isTrakeCell = requestedTask === "TRAKE" || activeInspectorTask === "TRAKE" ||
    result?.submission_task === "TRAKE" || Boolean(result?.track_id);
  const submissionTask = isTrakeCell ? "TRAKE" : requestedTask;
  const queryId = queryIdForResult(result, queryIdOverride);
  const trakeMetadata = isTrakeCell
    ? trakeQueueMetadata(result, stageId, queryId, result.video_id)
    : null;
  const body = {
    query_id: queryId,
    stage_id: trakeMetadata?.stage_id || domStageId(stageId || result.stage_id || "S1"),
    video_id: result.video_id,
    bundle_id: result.bundle_id || null,
    frame_uid: result.frame_uid || result.frame_id,
    source_frame_idx: result.source_frame_idx ?? result.frame_idx,
    timestamp_ms: result.timestamp_ms,
    shot_id: result.shot_id || null,
    rank: result.rank_in_stage || result.rank || result.track_rank || 1,
    scores: result.signal_scores || {},
    selection_reason: "selected_result",
    ...(result?.stage_frame_source === "manual_seek"
      ? { origin: "manual_seek" }
      : { origin: "search_result" }),
    submission_task: submissionTask,
    chain_id: trakeMetadata?.chain_id || null,
    event_step: trakeMetadata?.event_step ?? null,
    selection_kind: trakeMetadata?.selection_kind || null,
    qa_answer: normalizeQaAnswer(result?.qa_answer),
    ...(hasTemporalModeField(result)
      ? { bundle_temporal_enabled: temporalModeValue(result) }
      : {}),
  };

  if (PREVIEW_SAMPLE_MODE) {
    const uid = body.frame_uid;
    const duplicate = previewQueueItems.some((item) => queueItemIdentity(item) === queueItemIdentity(body));
    if (!duplicate) {
      previewQueueItems.push({
        ...result,
        ...body,
        queue_item_id: `preview-${encodeURIComponent(queueItemIdentity(body))}`,
        keyframe_id: uid,
        thumbnail_url: result.thumbnail_url || thumbnailUrl(uid),
      });
    }
    $("queueStatus").textContent = duplicate
      ? "Already in preview queue (deduped)."
      : `Added ${uid} to local preview queue.`;
    renderQueueItems(previewQueueItems, true);
    return true;
  }

  try {
    const response = await api("/review/queue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("queueStatus").textContent = response.duplicate ? "Already in queue (deduped)." : "Added to queue.";
    if (response.item) {
      const responseItem = { ...response.item };
      const hasTemporalMode = result && (
        Object.prototype.hasOwnProperty.call(result, "bundle_temporal_enabled") ||
        Object.prototype.hasOwnProperty.call(result, "temporal_enabled")
      );
      if (hasTemporalMode) {
        responseItem.bundle_temporal_enabled = temporalModeValue(result);
        if (responseItem.queue_item_id) {
          queueTemporalModeById.set(responseItem.queue_item_id, responseItem.bundle_temporal_enabled);
        }
      }
      const responseKey = responseItem.queue_item_id || [
        responseItem.query_id,
        responseItem.stage_id,
        responseItem.bundle_id,
        responseItem.frame_uid,
      ].join("|");
      const alreadyRendered = currentQueueItems.some((item) => {
        const itemKey = item.queue_item_id || [
          item.query_id,
          item.stage_id,
          item.bundle_id,
          item.frame_uid,
        ].join("|");
        return itemKey === responseKey;
      });
      if (!alreadyRendered) renderQueueItems([...currentQueueItems, responseItem]);
    }
    await refreshQueue();
    return true;
  } catch (err) {
    $("queueStatus").textContent = `Queue add failed (${body.stage_id}) — ${err.message}`;
    return false;
  }
}

async function queueSelectedInspectorStage(activeStage, stageId) {
  if (!activeStage || activeStage.stage_draft_pending || !frameUidOf(activeStage)) return false;
  const frameUid = frameUidOf(activeStage);
  const queued = await addResultToQueue(
    { ...activeStage, frame_id: frameUid, frame_uid: frameUid },
    stageId,
    currentInspectorQueryId(),
  );
  if (queued) {
    setStatus(`Queued selected ${stageId} frame ${frameUid}. Use Extract after seeking to queue a new position.`);
  }
  return queued;
}

async function addCurrentPositionToQueue() {
  if (!selectedFrameId) return;
  const activeStage = inspectorActiveStageItem();
  const fallbackStage = activeStage || selectedResult;
  const videoId = String(
    selectedResult?.video_id || activeStage?.video_id || selectedFrameId.split(":")[0] || "",
  ).trim();
  const bundleId = String(
    selectedResult?.bundle_id || activeStage?.bundle_id || "",
  ).trim();
  const sourceStages = Array.isArray(inspectorStageItems) && inspectorStageItems.length
    ? inspectorStageItems
    : (fallbackStage ? [fallbackStage] : []);
  const stageItems = [...new Map(
    sourceStages
      .filter((item) => {
        if (!item || (item.video_id && String(item.video_id) !== videoId)) return false;
        if (!bundleId) return true;
        return !item.bundle_id || String(item.bundle_id) === bundleId;
      })
      .map((item) => [String(item.stage_id || "S1").toUpperCase(), item]),
  ).values()].sort(compareStageCandidates);
  if (!stageItems.length) return;

  const pendingStage = stageItems.find((item) =>
    item.stage_draft_pending || item.stage_frame_pending ||
    !frameUidOf(item) || item.source_frame_idx == null || item.timestamp_ms == null,
  );
  if (pendingStage) {
    const stageId = String(pendingStage.stage_id || "S1").toUpperCase();
    const message = `Cannot queue ${stageId}: Extract/Mark its exact source frame first.`;
    $("queueStatus").textContent = message;
    setStatus(message, true);
    return;
  }

  let queuedCount = 0;
  for (const item of stageItems) {
    const stageId = String(item.stage_id || "S1").toUpperCase();
    const queued = await addResultToQueue(
      { ...item, frame_id: frameUidOf(item), frame_uid: frameUidOf(item) },
      stageId,
      currentInspectorQueryId(),
    );
    if (!queued) return;
    queuedCount += 1;
  }

  const mode = PREVIEW_SAMPLE_MODE ? "preview queue" : "review queue";
  const status = `Queued ${queuedCount} stage${queuedCount === 1 ? "" : "s"} in ${mode}.`;
  $("queueStatus").textContent = status;
  setStatus(status);
}

function queueGroupKey(item, fallbackIndex = 0) {
  const authoritativeGroupId = String(item?.queue_group_id || "").trim();
  if (authoritativeGroupId) return authoritativeGroupId;
  const queryId = String(item?.query_id || lastQueryId || "manual");
  const videoId = String(item?.video_id || "");
  const chainId = String(item?.chain_id || item?.bundle_id || "").trim();
  if (chainId || String(item?.submission_task || "").toUpperCase() === "TRAKE") {
    return `${queryId}\u0000${chainId || `${queryId}:${videoId}`}\u0000${videoId}`;
  }
  return `${queryId}\u0000item\u0000${item?.queue_item_id || item?.frame_uid || fallbackIndex}`;
}

function queueGroupAuthoritativeId(group) {
  return String(group?.items?.[0]?.queue_group_id || group?.key || "").trim();
}

function groupQueueItems(items) {
  const groups = new Map();
  (Array.isArray(items) ? items : []).forEach((rawItem, index) => {
    const item = { ...rawItem };
    const key = queueGroupKey(item, index);
    let group = groups.get(key);
    if (!group) {
      group = { key, items: [], firstPosition: Number.MAX_SAFE_INTEGER };
      groups.set(key, group);
    }
    group.items.push(item);
    const position = Number(item.queue_position);
    if (Number.isFinite(position)) {
      group.firstPosition = Math.min(group.firstPosition, position);
    } else if (group.firstPosition === Number.MAX_SAFE_INTEGER) {
      group.firstPosition = index;
    }
  });
  return [...groups.values()]
    .sort((left, right) => left.firstPosition - right.firstPosition || left.key.localeCompare(right.key))
    .map((group, index) => ({
      ...group,
      groupIndex: index,
      items: [...group.items].sort((left, right) => {
        const stepLeft = Number(left.event_step);
        const stepRight = Number(right.event_step);
        if (Number.isFinite(stepLeft) && Number.isFinite(stepRight) && stepLeft !== stepRight) {
          return stepLeft - stepRight;
        }
        return (Number(left.queue_position) || 0) - (Number(right.queue_position) || 0);
      }),
    }));
}

function queueGroupTooltip(group) {
  const first = group.items[0] || {};
  const chain = first.chain_id || first.bundle_id || "—";
  const lines = [
    `video_id: ${first.video_id || "—"}`,
    `chain_or_bundle_id: ${chain}`,
    ...group.items.map((item, index) => {
    const stage = item.selection_kind || item.stage_id || `S${index + 1}`;
    const frame = item.frame_uid || `${item.video_id || "—"}:${item.source_frame_idx ?? "—"}`;
    const answer = normalizeQaAnswer(item.qa_answer || item.answer);
    return `${stage}: ${frame}${answer ? ` · QA ${answer}` : ""}`;
  }),
  ];
  return lines.join("\n");
}

function renderQueueItems(items, preview = false) {
  const queueItems = (Array.isArray(items) ? items : []).map((item) => {
    const rememberedMode = item?.queue_item_id
      ? queueTemporalModeById.get(item.queue_item_id)
      : undefined;
    return rememberedMode === undefined
      ? { ...item }
      : { ...item, bundle_temporal_enabled: rememberedMode };
  });
  currentQueueItems = queueItems;
  currentQueuePreview = preview;
  const groups = groupQueueItems(currentQueueItems);
  const count = queueItems.length;
  const list = $("queueItems");
  if ($("queueBadge")) $("queueBadge").textContent = String(count);
  if ($("queueCountIndicator")) $("queueCountIndicator").textContent = String(count);
  if (!list) return;

  disconnectDeferredImageObserver(list);
  list.innerHTML = "";
  if (!queueItems.length) {
    $("queueStatus").textContent = preview
      ? "Preview queue empty. Select a sample frame, then press Q to add it."
      : "Empty. Select a frame, then press Q to add it to the review queue.";
    return;
  }

  $("queueStatus").textContent = preview
    ? `${count} preview item(s) · ${groups.length} bundle(s) · local only`
    : `${count} item(s) · ${groups.length} bundle(s) in review queue`;

  for (const group of groups) {
    const item = group.items[0];
    const uid = item.frame_uid || item.queue_item_id;
    const thumbUrl = item.thumbnail_url || thumbnailUrl(uid, GALLERY_THUMBNAIL_WIDTH, GALLERY_THUMBNAIL_QUALITY);
    const frameLine = group.items.map((member, index) => {
      const stage = member.selection_kind || member.stage_id || `S${index + 1}`;
      return `${stage} ${member.source_frame_idx ?? "—"}`;
    }).join(" · ");
    const qaLine = group.items
      .map((member, index) => {
        const answer = normalizeQaAnswer(member.qa_answer || member.answer);
        return answer ? `${member.selection_kind || member.stage_id || `S${index + 1}`}: ${answer}` : "";
      })
      .filter(Boolean)
      .join(" · ");
    const stageStrip = group.items.map((member, index) =>
      `<span class="queue-item-stage-chip">${escapeHtml(member.selection_kind || member.stage_id || `S${index + 1}`)}</span>`
    ).join("");
    const fullTitle = queueGroupTooltip(group);
    const li = document.createElement("li");
    li.className = "queue-item-card queue-item-group";
    li.draggable = true;
    li.dataset.queueGroupId = group.key;
    li.dataset.queueItemId = item.queue_item_id || uid;
    li.title = fullTitle;
    li.innerHTML = `
      <div class="queue-item-top">
        <img class="queue-item-thumb" loading="lazy" data-src="${escapeHtml(thumbUrl)}" alt="${escapeHtml(uid || "")}">
        <div class="queue-item-info">
          <span class="queue-item-video">${escapeHtml(item.video_id || "—")}</span>
          <span class="queue-item-stage-strip" title="${escapeHtml(fullTitle)}">${stageStrip}</span>
          <span class="queue-item-row-2" title="${escapeHtml(frameLine)}">${escapeHtml(frameLine)}</span>
          ${qaLine ? `<span class="queue-item-row-3" title="${escapeHtml(qaLine)}">QA ${escapeHtml(qaLine)}</span>` : ""}
        </div>
        <button type="button" class="queue-item-remove" title="Remove this bundle from queue">✕</button>
      </div>`;

    const thumbImg = li.querySelector(".queue-item-thumb");
    if (thumbImg) {
      thumbImg.addEventListener("error", () => {
        const placeholder = document.createElement("span");
        placeholder.className = "queue-item-thumb queue-item-thumb-placeholder";
        placeholder.textContent = "N/A";
        placeholder.title = "Thumbnail unavailable";
        thumbImg.replaceWith(placeholder);
      }, { once: true });
    }

    li.addEventListener("click", (event) => {
      if (event.target.closest(".queue-item-remove")) return;
      selectedResult = {
        query_id: item.query_id || null,
        frame_id: uid,
        frame_uid: uid,
        image_url: item.image_url || `/frames/${encodeURIComponent(uid)}/image`,
        thumbnail_url: item.thumbnail_url || thumbnailUrl(uid),
        video_id: item.video_id,
        source_frame_idx: item.source_frame_idx,
        timestamp_ms: item.timestamp_ms,
        stage_id: item.stage_id || "S1",
        final_score: item.scores?.fused_score ?? item.final_score ?? 0,
        signal_scores: item.scores || item.signal_scores || {},
        shot_id: item.shot_id || null,
        qa_answer: item.qa_answer || "",
        submission_task: item.submission_task || null,
        chain_id: item.chain_id || null,
        bundle_id: item.bundle_id || item.chain_id || null,
        ...(hasTemporalModeField(item)
          ? { bundle_temporal_enabled: temporalModeValue(item) }
          : {}),
        event_step: item.event_step,
        selection_kind: item.selection_kind,
        queue_group_items: group.items.map((member) => ({
          ...member,
          frame_id: member.frame_uid || member.queue_item_id,
          final_score: member.scores?.fused_score ?? member.final_score ?? 0,
          signal_scores: member.scores || member.signal_scores || {},
        })),
      };
      openDetail(uid);
    });

    li.addEventListener("dragstart", (event) => {
      li.classList.add("queue-item-dragging");
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", group.key);
    });
    li.addEventListener("dragend", () => li.classList.remove("queue-item-dragging"));
    li.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    });
    li.addEventListener("drop", (event) => {
      event.preventDefault();
      const fromId = event.dataTransfer.getData("text/plain");
      if (fromId && fromId !== group.key) reorderQueueItems(fromId, group.key);
    });

    li.querySelector(".queue-item-remove")?.addEventListener("click", async (event) => {
      event.stopPropagation();
      const groupIds = new Set(group.items.map((member) => member.frame_uid || member.queue_item_id));
      if (preview) {
        previewQueueItems = previewQueueItems.filter((queued) => !groupIds.has(queued.frame_uid || queued.queue_item_id));
        renderQueueItems(previewQueueItems, true);
        return;
      }
      try {
        await Promise.all(group.items.map((member) =>
          api(`/review/queue/${encodeURIComponent(member.queue_item_id)}`, { method: "DELETE" })
        ));
        await refreshQueue();
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        $("queueStatus").textContent = "Queue bundle remove failed — " + message;
      }
    });

    list.appendChild(li);
  }
  setupDeferredImages(list);
}

async function reorderQueueItems(fromId, toId) {
  const groups = groupQueueItems(currentQueueItems);
  const fromIndex = groups.findIndex((group) => group.key === fromId);
  const toIndex = groups.findIndex((group) => group.key === toId);
  if (fromIndex < 0 || toIndex < 0 || fromIndex === toIndex) return;
  const [moved] = groups.splice(fromIndex, 1);
  const insertionIndex = fromIndex < toIndex ? toIndex - 1 : toIndex;
  groups.splice(insertionIndex, 0, moved);
  const reorderedItems = groups.flatMap((group) => group.items).map((item, index) => ({
    ...item,
    queue_position: index,
  }));
  if (currentQueuePreview) {
    aic26PreviewCache = null;
    renderQueueItems(reorderedItems, true);
    $("queueStatus").textContent = "Preview queue reordered locally.";
    return;
  }

  const queryIds = new Set();
  let missingQueryId = false;
  for (const item of reorderedItems) {
    const queryId = String(item?.query_id || "").trim();
    if (!queryId) {
      missingQueryId = true;
      continue;
    }
    queryIds.add(queryId);
  }
  if (missingQueryId || queryIds.size !== 1) {
    $("queueStatus").textContent = missingQueryId
      ? "Queue reorder blocked — every item needs an authoritative query_id."
      : "Queue reorder blocked — reorder one query at a time; refresh the queue and retry.";
    return;
  }
  const queryId = [...queryIds][0];
  const orderedGroupIds = groups.map(queueGroupAuthoritativeId);
  if (orderedGroupIds.some((groupId) => !groupId)) {
    $("queueStatus").textContent = "Queue reorder blocked — a group is missing its authoritative group id.";
    return;
  }
  try {
    const response = await api("/review/queue/reorder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query_id: queryId, ordered_group_ids: orderedGroupIds }),
    });
    if (!Array.isArray(response?.items)) {
      throw new Error("authoritative reorder response is missing items");
    }
    aic26PreviewCache = null;
    renderQueueItems(response.items, false);
    $("queueStatus").textContent = "Queue reordered and saved.";
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    $("queueStatus").textContent = `Queue reorder failed — ${message}. Order was not changed.`;
  }
}

async function refreshQueue() {
  if (PREVIEW_SAMPLE_MODE) {
    renderQueueItems(previewQueueItems, true);
    return;
  }

  try {
    const data = await api("/review/queue");
    renderQueueItems(data.items || []);
  } catch (err) {
    console.warn("Review queue unavailable (backend offline):", err);
    const message = err instanceof Error ? err.message : String(err);
    if ($("queueBadge")) $("queueBadge").textContent = String(currentQueueItems.length);
    if ($("queueCountIndicator")) $("queueCountIndicator").textContent = String(currentQueueItems.length);
    if ($("queueStatus")) {
      $("queueStatus").textContent = "Queue unavailable — " + message;
    }
  }
}

function buildAic26Request() {
  syncAuthoritativeSubmissionTaskFromQueue(currentQueueItems);
  const task = activeSubmissionTask;
  const queueQueryIds = [...new Set(
    currentQueueItems.map((item) => item.query_id).filter(Boolean)
  )];
  const queryId = queueQueryIds[0] || lastQueryId || "manual";
  if (queueQueryIds.length > 1) {
    throw new Error("Queue contains multiple query_id values; export one query at a time.");
  }
  const validQueueItems = validateAic26Queue(task, queryId);
  const request = {
    query_id: queryId,
    task,
    filename: $("aic26Filename").value.trim() || null,
    target_rows: 100,
    delta: 3,
  };
  if (task === "QA") {
    request.answer = getQueueQaAnswer(queryId, validQueueItems);
  }
  if (task === "TRAKE") {
    request.event_count = Math.max(
      ...validQueueItems.map((item) => Number(item.event_step) + 1),
    );
  }
  return request;
}

function splitAic26CsvRecords(csvText) {
  const text = String(csvText ?? "");
  if (!text) return [];

  const records = [];
  let recordStart = 0;
  let insideQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (insideQuotes && text[index + 1] === '"') {
        index += 1;
      } else {
        insideQuotes = !insideQuotes;
      }
    } else if (character === "\n" && !insideQuotes) {
      const recordEnd = index > recordStart && text[index - 1] === "\r" ? index - 1 : index;
      if (recordEnd > recordStart) records.push(text.slice(recordStart, recordEnd));
      recordStart = index + 1;
    }
  }

  const trailingRecord = text.slice(recordStart).replace(/\r$/, "");
  if (trailingRecord) records.push(trailingRecord);
  return records;
}

let aic26PreviewOpen = false;
let aic26PreviewCache = null;

function aic26PreviewCacheKey(request) {
  const queueSnapshot = currentQueueItems.map((item) => ({
    queue_item_id: item.queue_item_id || null,
    queue_position: item.queue_position ?? null,
    query_id: item.query_id || null,
    submission_task: item.submission_task || null,
    frame_uid: item.frame_uid || null,
    source_frame_idx: item.source_frame_idx ?? null,
    timestamp_ms: item.timestamp_ms ?? null,
    chain_id: item.chain_id || null,
    stage_id: item.stage_id || null,
    event_step: item.event_step ?? null,
    selection_kind: item.selection_kind || null,
    qa_answer: normalizeQaAnswer(item.qa_answer || item.answer),
  }));
  return JSON.stringify({ request, queue: queueSnapshot });
}

function openAic26Preview() {
  const previewDialog = $("aic26PreviewDialog");
  if (!previewDialog) return;
  // Re-centre on every fresh open so a dialog dragged partly off-screen can
  // always be recovered by pressing Preview again.
  previewDialog.style.left = "";
  previewDialog.style.top = "";
  previewDialog.style.transform = "";
  previewDialog.hidden = false;
  previewDialog.setAttribute("aria-hidden", "false");
  previewDialog.classList.toggle("is-open", true);
  aic26PreviewOpen = true;
}

function closeAic26Preview() {
  const previewDialog = $("aic26PreviewDialog");
  if (!previewDialog) return;
  previewDialog.hidden = true;
  previewDialog.setAttribute("aria-hidden", "true");
  previewDialog.classList.toggle("is-open", false);
  aic26PreviewOpen = false;
}

function toggleAic26Preview() {
  const previewDialog = $("aic26PreviewDialog");
  const preview = $("aic26Preview");
  if (!previewDialog || !aic26PreviewOpen || !preview?.textContent) return false;
  closeAic26Preview();
  return true;
}

let aic26PreviewDragState = null;

function setupAic26PreviewDrag() {
  const previewDialog = $("aic26PreviewDialog");
  const dragHandle = $("aic26PreviewDragHandle");
  if (!previewDialog || !dragHandle) return;

  const finishDrag = (event) => {
    if (!aic26PreviewDragState) return;
    if (event && dragHandle.hasPointerCapture?.(event.pointerId)) {
      dragHandle.releasePointerCapture(event.pointerId);
    }
    aic26PreviewDragState = null;
    previewDialog.classList.remove("is-dragging");
  };

  dragHandle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest("button")) return;
    const rect = previewDialog.getBoundingClientRect();
    previewDialog.style.left = `${rect.left}px`;
    previewDialog.style.top = `${rect.top}px`;
    previewDialog.style.transform = "none";
    aic26PreviewDragState = {
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
      pointerId: event.pointerId,
    };
    previewDialog.classList.add("is-dragging");
    dragHandle.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  });

  dragHandle.addEventListener("pointermove", (event) => {
    if (!aic26PreviewDragState || event.pointerId !== aic26PreviewDragState.pointerId) return;
    const rect = previewDialog.getBoundingClientRect();
    const allowPartial = true;
    const minLeft = allowPartial ? Math.min(8, 120 - rect.width) : 8;
    const minTop = allowPartial ? Math.min(8, 80 - rect.height) : 8;
    const maxLeft = allowPartial ? window.innerWidth - 120 : Math.max(8, window.innerWidth - rect.width - 8);
    const maxTop = allowPartial ? window.innerHeight - 80 : Math.max(8, window.innerHeight - rect.height - 8);
    const left = Math.min(maxLeft, Math.max(minLeft, event.clientX - aic26PreviewDragState.offsetX));
    const top = Math.min(maxTop, Math.max(minTop, event.clientY - aic26PreviewDragState.offsetY));
    previewDialog.style.left = `${left}px`;
    previewDialog.style.top = `${top}px`;
  });

  dragHandle.addEventListener("pointerup", finishDrag);
  dragHandle.addEventListener("pointercancel", finishDrag);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !previewDialog.hidden) closeAic26Preview();
  });
}

function showAic26CsvPreview(csvText) {
  const preview = $("aic26Preview");
  const previewDialog = $("aic26PreviewDialog");
  const records = splitAic26CsvRecords(csvText);
  const visibleRecords = records;
  if (preview) {
    preview.textContent = visibleRecords.length ? `${visibleRecords.join("\r\n")}\r\n` : "";
    preview.dataset.totalRows = String(records.length);
    preview.dataset.previewRows = String(visibleRecords.length);
  }
  if ($("aic26PreviewDialogMeta")) {
    $("aic26PreviewDialogMeta").textContent = `${records.length} row${records.length === 1 ? "" : "s"} · headerless CSV · scroll to review`;
  }
  if (records.length && previewDialog) openAic26Preview();
  if (!records.length && previewDialog && !previewDialog.hidden) closeAic26Preview();
  return { totalRows: records.length, previewRows: visibleRecords.length };
}

async function previewAic26Submission() {
  if (toggleAic26Preview()) return;
  try {
    if (PREVIEW_SAMPLE_MODE) throw new Error("Preview sample mode has no submission backend.");
    const request = buildAic26Request();
    const requestKey = aic26PreviewCacheKey(request);
    if (aic26PreviewCache?.key === requestKey) {
      showAic26CsvPreview(aic26PreviewCache.csv);
      return;
    }
    const data = await api("/v1/submissions/aic26/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    aic26PreviewCache = { key: requestKey, csv: data.csv };
    showAic26CsvPreview(data.csv);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    $("queueStatus").textContent = "CSV preview failed — " + message;
    setStatus("CSV preview failed — " + message, true);
    console.error("CSV preview failed —", message);
  }
}

async function downloadAic26Submission() {
  try {
    if (PREVIEW_SAMPLE_MODE) throw new Error("Preview sample mode has no submission backend.");
    const request = buildAic26Request();
    const response = await fetch("/v1/submissions/aic26/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const payload = await response.json();
        detail = formatApiErrorDetail(payload, detail);
      } catch { /* response may not be JSON */ }
      throw new Error(`${response.status}: ${detail}`);
    }
    const blob = await response.blob();
    const csvText = await blob.text();
    showAic26CsvPreview(csvText);
    const disposition = response.headers.get("content-disposition") || "";
    const filenameMatch = disposition.match(/filename="([^"]+)"/i);
    const filename = filenameMatch ? filenameMatch[1] : "aic26-submission.csv";
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    $("queueStatus").textContent = "CSV download failed — " + message;
    setStatus("CSV download failed — " + message, true);
    console.error("CSV download failed —", message);
  }
}

/* ==========================================================================
   Temporal Selection, Feedback & QA Submission
   ========================================================================== */

function setAuthoritativeSubmissionTask(task) {
  const normalized = String(task || "KIS").toUpperCase();
  const supported = ["KIS", "QA", "TRAKE"].includes(normalized) ? normalized : "KIS";
  activeSubmissionTask = supported;
  activeInspectorTask = supported;
  const submissionTask = $("submissionTask");
  if (submissionTask) submissionTask.value = supported;
  const stateTask = supported === "TRAKE" ? "TRAKE" : "KIS";
  const selectionTask = $("selectionTask");
  if (selectionTask) selectionTask.value = stateTask;

  document.querySelectorAll("[data-inspector-task]").forEach((tab) => {
    const active = tab.dataset.inspectorTask === supported;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  const qaPanel = $("inspectorQaPanel");
  if (qaPanel) qaPanel.hidden = supported !== "QA";
  renderInspectorStageSequence();
  syncInspectorStageForm();
}

function setInspectorTask(task) {
  setAuthoritativeSubmissionTask(task);
}

function getTemporalSelection() {
  const task = $("selectionTask")?.value || (activeInspectorTask === "TRAKE" ? "TRAKE" : "KIS");
  if (task === "KIS") return { task, eventStep: 0, selectionKind: "KIS" };
  const activeStageNumber = stageNumber(inspectorActiveStageId);
  const validStage = /^S\d+$/i.test(String(inspectorActiveStageId || "")) && activeStageNumber >= 1;
  const eventNumber = validStage ? activeStageNumber : 1;
  return { task, eventStep: eventNumber - 1, selectionKind: `E${eventNumber}` };
}

async function recordFeedback(kind) {
  if (!selectedFrameId) return;
  const body = {
    session_id: sessionId,
    query_revision: Math.max(1, queryRevision),
    positive_ids: kind === "positive" ? [selectedFrameId] : [],
    negative_ids: kind === "negative" ? [selectedFrameId] : [],
    prior_result_ids: lastResultIds,
  };
  try {
    const data = await api("/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("feedbackStatus").textContent = `${kind} recorded locally (#${data.record_count})`;
  } catch (err) {
    $("feedbackStatus").textContent = "Feedback failed: " + err.message;
  }
}

/* ==========================================================================
   Query History
   ========================================================================== */

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function pushHistory(query) {
  if (!query || !query.trim()) return;
  let history = getHistory().filter((q) => q !== query);
  history.unshift(query);
  history = history.slice(0, MAX_HISTORY);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  const list = $("history");
  if (!list) return;
  list.innerHTML = "";
  for (const q of getHistory()) {
    const li = document.createElement("li");
    li.textContent = q;
    li.title = "Run this query again";
    li.addEventListener("click", () => {
      if ($("s1Text")) $("s1Text").value = q;
      runUnifiedSearch();
    });
    list.appendChild(li);
  }
}

/* ==========================================================================
   Sidebar Tabs & 5 Channel Tabs Switcher
   ========================================================================== */

function setupSidebarTabs() {
  const tabStaged = $("tabStaged");
  const tabTrake = $("tabTrake");
  const secStaged = $("stagedSearchSection");
  const secTrake = $("trakeSearchSection");
  if (!tabStaged || !tabTrake || !secStaged || !secTrake) return;

  tabStaged.addEventListener("click", () => {
    tabStaged.classList.add("active");
    tabTrake.classList.remove("active");
    secStaged.classList.add("active");
    secTrake.classList.remove("active");
  });

  tabTrake.addEventListener("click", () => {
    tabTrake.classList.add("active");
    tabStaged.classList.remove("active");
    secTrake.classList.add("active");
    secStaged.classList.remove("active");
  });
}

function setStagePanelCollapsed(collapsed) {
  const stagePanel = $("stagePanel");
  const workspace = document.querySelector(".workspace-layout");
  const expandBtn = $("stagePanelExpandBtn");
  if (!stagePanel || !workspace) return;

  stagePanel.classList.toggle("sidebar-collapsed", collapsed);
  workspace.classList.toggle("stage-collapsed-layout", collapsed);
  if (expandBtn) expandBtn.hidden = !collapsed;
  $("stagePanelCollapseBtn")?.setAttribute("aria-expanded", String(!collapsed));
}

function setQueueCollapsed(collapsed) {
  const queueBox = $("queueBox");
  const workspace = document.querySelector(".workspace-layout");
  if (!queueBox || !workspace) return;

  queueBox.classList.toggle("queue-collapsed", collapsed);
  workspace.classList.toggle("queue-collapsed-layout", collapsed);
  $("queueToggleBtn")?.classList.toggle("active", !collapsed);
  $("queueCollapseBtn")?.setAttribute("aria-expanded", String(!collapsed));
}

function setupSidebarCollapse() {
  $("stagePanelCollapseBtn")?.addEventListener("click", () => setStagePanelCollapsed(true));
  $("stagePanelExpandBtn")?.addEventListener("click", () => setStagePanelCollapsed(false));
  $("queueCollapseBtn")?.addEventListener("click", () => setQueueCollapsed(true));
  $("queueToggleBtn")?.addEventListener("click", () => {
    const queueBox = $("queueBox");
    setQueueCollapsed(!queueBox?.classList.contains("queue-collapsed"));
  });
}

function setupChannelTabs() {
  for (const tabStrip of document.querySelectorAll(".stage-channel-tabs")) {
    if (tabStrip.dataset.tabsReady === "true") continue;
    const stage = String(tabStrip.dataset.stage || "").toLowerCase();
    const scope = tabStrip.closest(".stage-block") || document;
    const panelsContainer = scope.querySelector(`.channel-input-panels[data-stage="${stage}"]`);
    if (!panelsContainer) continue;

    const tabBtns = tabStrip.querySelectorAll(".channel-tab-btn");
    const toggleBtns = tabStrip.querySelectorAll(".channel-toggle-btn");
    const panels = panelsContainer.querySelectorAll(".channel-panel");
    const initialChannel = [...tabBtns].find((button) => button.classList.contains("active"))?.dataset.channel || "text";
    setSelectedStageChannel(initialChannel, tabBtns, panels);

    toggleBtns.forEach((toggle) => {
      const channel = toggle.dataset.channelToggle;
      if (!channel) return;
      const enabled = defaultChannelEnabled(stage.toUpperCase(), channel);
      setChannelToggleState(stage.toUpperCase(), channel, enabled, scope);
      toggle.addEventListener("click", (event) => {
        event.stopPropagation();
        const nextEnabled = toggle.getAttribute("aria-pressed") !== "true";
        setChannelToggleState(stage.toUpperCase(), channel, nextEnabled, scope);
        const label = STAGE_CHANNEL_LABELS[channel] || channel;
        setStatus(
          `${label} channel ${nextEnabled ? "enabled" : "disabled"} for the next search.`,
          false,
        );
      });
    });

    tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetChannel = btn.dataset.channel;
        setSelectedStageChannel(targetChannel, tabBtns, panels);
      });
    });
    tabStrip.dataset.tabsReady = "true";
  }
}

function setupViewControls() {
  const btnGrouped = $("viewGroupedBtn");
  const btnFlat = $("viewFlatBtn");
  const stageFilter = $("stageFilter");
  const videoFilter = $("videoFilter");
  const allHitsSpacing = $("allHitsSpacing");
  setupVideoFilterTypeahead();

  btnGrouped.addEventListener("click", () => {
    currentViewMode = "grouped";
    btnGrouped.classList.add("active");
    btnFlat.classList.remove("active");
    refreshUnifiedViewMode();
  });

  btnFlat.addEventListener("click", () => {
    currentViewMode = "flat";
    btnFlat.classList.add("active");
    btnGrouped.classList.remove("active");
    refreshUnifiedViewMode();
  });

  stageFilter.addEventListener("change", (e) => {
    currentStageFilter = e.target.value;
    renderResultsView();
  });

  videoFilter.addEventListener("change", () => {
    void handleVideoFilterChange();
  });

  allHitsSpacing?.addEventListener("change", () => {
    if (lastSearchData?.all_hits_eligible_raw || lastSearchData?.all_hits_raw) {
      lastSearchData.all_hits = diversifyAllHits(
        lastSearchData.all_hits_eligible_raw || lastSearchData.all_hits_raw,
        allHitsSpacingMs(),
      );
    }
    renderResultsView();
    if (lastSearchData) updateStatusSummary(lastSearchData);
  });
}

function refreshUnifiedViewMode() {
  // A grouped response contains raw per-stage diagnostics, not an All Hits
  // result set. Re-rendering it as flat cards would split complete bundles.
  if (lastSearchData?.mode === "bundle" || lastSearchData?.mode === "all_hits") {
    void runUnifiedSearch();
    return;
  }
  renderResultsView();
}

/* ==========================================================================
   Scoped Keyboard Shortcuts
   ========================================================================== */

function setupKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    if (clearSelectedTextOnDelete(e)) return;
    // Crucial safeguard: Never trigger shortcuts if typing inside an input, textarea, or select
    if (e.target.matches("input, textarea, select")) return;
    if (e.defaultPrevented) return;

    if (e.key === "Escape") {
      hideDetail();
    } else if (e.key.toLowerCase() === "q" || e.key === " ") {
      if (!$("detail").hidden) {
        e.preventDefault();
        addCurrentPositionToQueue();
      } else if (selectedResult) {
        e.preventDefault();
        addResultToQueue(selectedResult, selectedResult.stage_id || "S1");
      }
    }
  });
}

function clearSelectedTextOnDelete(event) {
  const target = event?.target;
  if (!target?.matches?.("input, textarea")) return false;
  if (event.key !== "Backspace" && event.key !== "Delete") return false;

  const start = target.selectionStart;
  const end = target.selectionEnd;
  if (!Number.isInteger(start) || !Number.isInteger(end)) return false;
  if (start !== 0 || end !== target.value.length || start === end) return false;

  event.preventDefault();
  event.stopPropagation();
  target.value = "";
  const inputEvent = typeof InputEvent === "function"
    ? new InputEvent("input", {
      bubbles: true,
      inputType: "deleteContentBackward",
      data: null,
    })
    : new Event("input", { bubbles: true });
  target.dispatchEvent(inputEvent);
  return true;
}

/* ==========================================================================
   Event Wiring & Bootstrap
   ========================================================================== */

function init() {
  // Search actions
  $("runSearchBtn")?.addEventListener("click", runUnifiedSearch);

  // Inspector actions
  $("closeDetail").addEventListener("click", hideDetail);
  $("btnReplay").addEventListener("click", replayVideo);
  $("btnExtractSourceFrame").addEventListener("click", extractSourceFrame);
  $("btnQueueModal").addEventListener("click", addCurrentPositionToQueue);
  $("btnMarkCurrentStage")?.addEventListener("click", () => markCurrentStageAtPlayhead());
  $("applySourceFrameIdx")?.addEventListener("click", applySourceFrameIdx);
  $("sourceFrameIdxInput")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") applySourceFrameIdx();
  });
  $("btnSetMarkerL")?.addEventListener("click", () => setInspectorRangeMarker("L"));
  $("btnSetMarkerR")?.addEventListener("click", () => setInspectorRangeMarker("R"));
  $("btnClearMarkerL")?.addEventListener("click", () => clearInspectorRangeMarker("L"));
  $("btnClearMarkerR")?.addEventListener("click", () => clearInspectorRangeMarker("R"));
  $("seekBack5").addEventListener("click", () => seekVideo(-5));
  $("seekBack1").addEventListener("click", () => seekVideo(-1));
  $("seekForward1").addEventListener("click", () => seekVideo(1));
  $("seekForward5").addEventListener("click", () => seekVideo(5));
  $("previousFrame").addEventListener("click", () => selectAdjacentVideo(-1));
  $("nextFrame").addEventListener("click", () => selectAdjacentVideo(1));
  $("applyQaAnswer")?.addEventListener("click", applyInspectorStageAnswer);
  $("stageFrameApplyBtn")?.addEventListener("click", applyManualStageFrame);
  $("stageFrameCancelBtn")?.addEventListener("click", closeStageFrameEditor);
  $("answer")?.addEventListener("input", (event) => updateInspectorStageAnswer(event.target.value));
  $("positiveBtn").addEventListener("click", () => recordFeedback("positive"));
  $("negativeBtn").addEventListener("click", () => recordFeedback("negative"));
  document.querySelectorAll("[data-inspector-task]").forEach((tab) => {
    tab.addEventListener("click", () => setInspectorTask(tab.dataset.inspectorTask));
  });
  $("submissionTask")?.addEventListener("change", (event) => {
    setAuthoritativeSubmissionTask(event.target.value);
  });
  setAuthoritativeSubmissionTask($("submissionTask")?.value || "KIS");

  $("detailVideo").addEventListener("pause", updatePlayerPosition);
  $("detailVideo").addEventListener("timeupdate", updatePlayerPosition);
  $("detailVideo").addEventListener("loadedmetadata", () => {
    syncVideoSeekBar();
    renderInspectorStageTimeline();
  });
  $("detailVideo").addEventListener("durationchange", () => {
    syncVideoSeekBar();
    renderInspectorStageTimeline();
  });
  $("detailVideo").addEventListener("seeked", () => {
    syncVideoSeekBar();
    updatePlayerPosition();
  });
  const videoSeekBar = $("videoSeekBar");
  videoSeekBar.addEventListener("input", () => {
    const video = $("detailVideo");
    const targetSeconds = Number(videoSeekBar.value);
    if (!Number.isFinite(targetSeconds)) return;
    video.currentTime = targetSeconds;
    updatePlayerPosition();
  });

  // Close inspector when clicking outside modal card
  $("detail").addEventListener("click", (e) => {
    if (e.target === $("detail")) hideDetail();
  });

  // Queue actions
  $("refreshQueue").addEventListener("click", refreshQueue);
  $("previewAic26").addEventListener("click", previewAic26Submission);
  $("downloadAic26").addEventListener("click", downloadAic26Submission);
  $("closeAic26Preview")?.addEventListener("click", closeAic26Preview);
  $("closeAic26PreviewFooter")?.addEventListener("click", closeAic26Preview);
  setupAic26PreviewDrag();

  // History
  $("clearHistory").addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
  });

  // Setup unified search, views & shortcuts
  setupSidebarCollapse();
  setupStagedComposer();
  setupTemporalSearchToggle();
  setupHorizontalWheelScroll("topNeighborsContainer");
  setupHorizontalWheelScroll("neighbors");
  document.querySelectorAll(".stage-channel-tabs").forEach(setupHorizontalWheelScroll);
  setupChannelTabs();
  setupImageInputs();
  setupAsrModeToggles();
  setupObjectQueryBuilders();
  setupViewControls();
  setupKeyboardShortcuts();
  renderHistory();
  loadSystemInfo();
  loadObjectAliases();
  loadPreviewSample();
  refreshQueue();
}

// Start application on DOM ready
document.addEventListener("DOMContentLoaded", init);
