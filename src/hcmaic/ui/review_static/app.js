/**
 * HCMAIC Ground-Truth Range Review Dashboard
 * Refined minimal "premium utility" interface for adjudicating 142 temporal range proposals.
 *
 * Invariants preserved:
 * - frame_uid = video_id:source_frame_idx
 * - source_frame_idx, timestamp_s, timestamp_ms, pts
 * - sample_roles: grid, anchor, proposal_left, proposal_right
 * - evidence_level: HUMAN_REVIEW_DRAFT
 * - quality_status: UNVALIDATED
 */

(() => {
  'use strict';

  const state = {
    items: [],           // Filtered proposals list
    rawItems: [],        // Unfiltered proposals list from server
    counts: { pending: 0, accepted: 0, edited: 0, rejected: 0 },
    total: 0,
    selected: null,      // Full detail payload of currently selected item
    currentFilter: 'pending',
    searchQuery: '',
    reviewerName: localStorage.getItem('hcmaic_reviewer') || 'teammate'
  };

  const lightboxState = {
    isOpen: false,
    currentFrame: null,
    scale: 1.0,
    fitScale: 1.0,
    minScale: 0.1,
    maxScale: 8.0,
    translateX: 0,
    translateY: 0,
    isDragging: false,
    startX: 0,
    startY: 0
  };

  const $ = (id) => document.getElementById(id);

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[char]));
  }

  function formatTime(timestamp_s, timestamp_ms) {
    if (timestamp_s !== undefined && timestamp_s !== null) {
      const s = Number(timestamp_s).toFixed(3);
      const ms = timestamp_ms !== undefined && timestamp_ms !== null ? ` (${Math.round(timestamp_ms)}ms)` : '';
      return `${s}s${ms}`;
    }
    return '--';
  }

  function showToast(message, duration = 2800) {
    const toast = $('toast');
    if (!toast) return;
    toast.textContent = message;
    toast.classList.remove('hidden');
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => {
      toast.classList.add('hidden');
    }, duration);
  }

  // =========================================================================
  // Health & Metric Counters
  // =========================================================================

  function updateProgressAndCounts(counts, total) {
    state.counts = counts || state.counts;
    const c = state.counts;
    const totalItems = total || (c.pending + c.accepted + c.edited + c.rejected) || 0;
    const reviewed = (c.accepted || 0) + (c.edited || 0) + (c.rejected || 0);
    const remaining = c.pending !== undefined ? c.pending : Math.max(0, totalItems - reviewed);

    $('reviewedCount').textContent = reviewed;
    $('totalCount').textContent = totalItems;
    $('remainingBadge').textContent = `${remaining} remaining`;

    const pct = totalItems > 0 ? Math.round((reviewed / totalItems) * 100) : 0;
    $('topProgressBar').style.width = `${pct}%`;

    $('cntPending').textContent = c.pending || 0;
    $('cntAll').textContent = totalItems;
    $('cntAccepted').textContent = c.accepted || 0;
    $('cntEdited').textContent = c.edited || 0;
    $('cntRejected').textContent = c.rejected || 0;
  }

  async function loadHealth() {
    try {
      const response = await fetch('/health');
      if (!response.ok) return;
      const data = await response.json();
      if (data.evidence_level) $('evidenceTag').textContent = data.evidence_level;
      if (data.quality_status) $('qualityTag').textContent = data.quality_status;
      if (data.counts) updateProgressAndCounts(data.counts, data.item_count);
    } catch (err) {
      console.warn('Failed to load /health:', err);
    }
  }

  // =========================================================================
  // Queue & Items Listing
  // =========================================================================

  function filterItems() {
    let filtered = [...state.rawItems];
    const q = state.searchQuery.trim().toLowerCase();
    if (q) {
      filtered = filtered.filter((item) => {
        const vid = String(item.video_id || '').toLowerCase();
        const ruid = String(item.review_uid || '').toLowerCase();
        const quid = String(item.query_uid || '').toLowerCase();
        const query = String(item.query || '').toLowerCase();
        return vid.includes(q) || ruid.includes(q) || quid.includes(q) || query.includes(q);
      });
    }
    state.items = filtered;
    renderQueue();
  }

  function renderQueue() {
    const listEl = $('queueList');
    if (!state.items || state.items.length === 0) {
      listEl.innerHTML = `<div class="queue-empty-msg">No proposals found for current filter.</div>`;
      return;
    }

    listEl.innerHTML = state.items.map((item) => {
      const active = state.selected && state.selected.review_uid === item.review_uid ? ' active' : '';
      const anchor = item.anchor || {};
      const range = item.proposed_range || {};
      const status = item.review_status || 'pending';

      return `
        <button class="queue-item-card${active}" data-uid="${esc(item.review_uid)}" role="option" aria-selected="${active ? 'true' : 'false'}">
          <div class="queue-card-top">
            <span class="queue-card-title">${esc(item.video_id)} · ${esc(item.review_uid)}</span>
            <span class="status-pill ${esc(status)}">${esc(status)}</span>
          </div>
          <div class="queue-card-query">${esc(item.query || '(No query text)')}</div>
          <div class="queue-card-meta">
            <span>anchor idx <b>${esc(anchor.source_frame_idx ?? '--')}</b></span>
            <span class="queue-range-tag">[${esc(range.left ?? '?')}, ${esc(range.right ?? '?')}]</span>
          </div>
        </button>
      `;
    }).join('');

    listEl.querySelectorAll('.queue-item-card').forEach((btn) => {
      btn.addEventListener('click', () => loadDetail(btn.dataset.uid));
    });
  }

  async function loadList() {
    const status = state.currentFilter;
    const url = `/api/review/items?status=${encodeURIComponent(status)}&limit=200`;
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      state.rawItems = data.items || [];
      updateProgressAndCounts(data.counts, data.total);
      filterItems();

      // If nothing selected, select first item if available
      if (!state.selected && state.items.length > 0) {
        loadDetail(state.items[0].review_uid);
      }
    } catch (err) {
      console.error('Failed to load items list:', err);
      $('queueList').innerHTML = `<div class="queue-empty-msg">Error loading proposals list.</div>`;
    }
  }

  // =========================================================================
  // Detail View & Frame Rendering
  // =========================================================================

  function lookupFrameTimestamp(frameIdx) {
    if (!state.selected || !state.selected.frames) return null;
    const idx = Number(frameIdx);
    if (isNaN(idx)) return null;

    // Exact match in sampled frames
    const match = state.selected.frames.find((f) => Number(f.source_frame_idx) === idx);
    if (match) {
      return { timestamp_s: match.timestamp_s, timestamp_ms: match.timestamp_ms, exact: true };
    }

    // Estimate using anchor frame
    const anchor = state.selected.anchor;
    if (anchor && anchor.timestamp_s !== undefined) {
      const diffFrames = idx - Number(anchor.source_frame_idx);
      const estSec = anchor.timestamp_s + (diffFrames / 30.0);
      return { timestamp_s: Math.max(0, estSec), timestamp_ms: Math.max(0, estSec * 1000), exact: false };
    }
    return null;
  }

  function validateRange() {
    if (!state.selected) return { valid: false, message: 'No proposal selected' };

    const leftVal = $('leftInput').value.trim();
    const rightVal = $('rightInput').value.trim();
    const alertEl = $('rangeAlert');
    const alertTextEl = $('rangeAlertText');
    const saveBtn = $('saveEditBtn');

    if (leftVal === '' || rightVal === '') {
      alertEl.classList.remove('hidden');
      alertTextEl.textContent = 'Left and right boundary indices cannot be empty.';
      if (saveBtn) saveBtn.disabled = true;
      return { valid: false };
    }

    const left = Number(leftVal);
    const right = Number(rightVal);
    const anchorIdx = Number((state.selected.anchor || {}).source_frame_idx);

    if (isNaN(left) || isNaN(right) || !Number.isInteger(left) || !Number.isInteger(right) || left < 0 || right < 0) {
      alertEl.classList.remove('hidden');
      alertTextEl.textContent = 'Indices must be non-negative integers.';
      if (saveBtn) saveBtn.disabled = true;
      return { valid: false };
    }

    if (left > right) {
      alertEl.classList.remove('hidden');
      alertTextEl.textContent = `Invalid Range: Left boundary (${left}) must be <= Right boundary (${right}).`;
      if (saveBtn) saveBtn.disabled = true;
      return { valid: false };
    }

    if (!isNaN(anchorIdx) && (anchorIdx < left || anchorIdx > right)) {
      alertEl.classList.remove('hidden');
      alertTextEl.textContent = `Invalid Range: Anchor frame (${anchorIdx}) must remain inside [left: ${left}, right: ${right}].`;
      if (saveBtn) saveBtn.disabled = true;
      return { valid: false };
    }

    const frameCount = state.selected.frame_count;
    if (frameCount !== undefined && frameCount !== null && (left >= frameCount || right >= frameCount)) {
      alertEl.classList.remove('hidden');
      alertTextEl.textContent = `Warning: Indices exceed total video frame count (${frameCount}).`;
      if (saveBtn) saveBtn.disabled = true;
      return { valid: false };
    }

    // Range is valid
    alertEl.classList.add('hidden');
    if (saveBtn) saveBtn.disabled = false;

    // Update time feedback
    const leftTime = lookupFrameTimestamp(left);
    $('leftTimeDisplay').textContent = leftTime ? formatTime(leftTime.timestamp_s, leftTime.timestamp_ms) : '--';

    const rightTime = lookupFrameTimestamp(right);
    $('rightTimeDisplay').textContent = rightTime ? formatTime(rightTime.timestamp_s, rightTime.timestamp_ms) : '--';

    const diffFrames = right - left + 1;
    $('rangeSpanFrames').textContent = diffFrames;

    if (leftTime && rightTime) {
      const diffSec = Math.abs(rightTime.timestamp_s - leftTime.timestamp_s).toFixed(2);
      $('rangeSpanTime').textContent = `~${diffSec}s`;
    } else {
      $('rangeSpanTime').textContent = '--';
    }

    // Update in-range highlights across frame cards
    updateFrameRangeHighlights(left, right);

    return { valid: true, left, right };
  }

  function updateFrameRangeHighlights(left, right) {
    document.querySelectorAll('.sequence-frame-card').forEach((card) => {
      const idx = Number(card.dataset.idx);
      if (idx >= left && idx <= right) {
        card.classList.add('in-range');
      } else {
        card.classList.remove('in-range');
      }
    });
  }

  function isFrameOutsideWindow(frame, item) {
    const anchor = item.anchor;
    if (!anchor || anchor.timestamp_s === undefined) return false;
    const sampling = item.sampling || {};
    const beforeS = sampling.window_before_s !== undefined ? Number(sampling.window_before_s) : 20.0;
    const afterS = sampling.window_after_s !== undefined ? Number(sampling.window_after_s) : 20.0;

    const anchorTime = Number(anchor.timestamp_s);
    const frameTime = Number(frame.timestamp_s);
    if (isNaN(anchorTime) || isNaN(frameTime)) return false;

    // Small tolerance of 0.15s for 3 FPS discretization
    return frameTime < (anchorTime - beforeS - 0.15) || frameTime > (anchorTime + afterS + 0.15);
  }

  function renderKeyStrip(item) {
    const stripEl = $('keyFramesStrip');
    if (!stripEl) return;

    const anchor = item.anchor || {};
    const proposed = item.proposed_range || {};
    const frames = item.frames || [];

    const leftFrame = frames.find((f) => Number(f.source_frame_idx) === Number(proposed.left)) || {
      source_frame_idx: proposed.left,
      frame_uid: `${item.video_id}:${proposed.left}`,
      sample_roles: ['proposal_left'],
      image_url: null
    };

    const anchorFrame = frames.find((f) => f.is_anchor || Number(f.source_frame_idx) === Number(anchor.source_frame_idx)) || anchor;

    const rightFrame = frames.find((f) => Number(f.source_frame_idx) === Number(proposed.right)) || {
      source_frame_idx: proposed.right,
      frame_uid: `${item.video_id}:${proposed.right}`,
      sample_roles: ['proposal_right'],
      image_url: null
    };

    function renderKeyCard(title, frame, roleTagClass, isAnchor = false) {
      const hasImage = Boolean(frame.image_url);
      const mediaHtml = hasImage
        ? `<img loading="lazy" decoding="async" src="${esc(frame.image_url)}" alt="${esc(frame.frame_uid)}">`
        : `<div class="no-image-placeholder">
             <span>JPEG chưa materialize</span>
           </div>`;

      return `
        <div class="key-frame-card${isAnchor ? ' anchor-card' : ''}" data-idx="${esc(frame.source_frame_idx)}">
          <div class="key-card-head">
            <span class="key-card-caption">${title}</span>
            <span class="seq-role-tag ${roleTagClass}">idx ${esc(frame.source_frame_idx ?? '--')}</span>
          </div>
          <div class="key-media-box${hasImage ? ' has-image' : ''}" title="${hasImage ? 'Click to inspect in lightbox' : 'No JPEG'}">
            ${mediaHtml}
          </div>
          <div class="key-card-foot">
            <span class="key-foot-uid">${esc(frame.frame_uid || '--')}</span>
            <span class="key-foot-time">${formatTime(frame.timestamp_s, frame.timestamp_ms)}</span>
          </div>
        </div>
      `;
    }

    stripEl.innerHTML = `
      ${renderKeyCard('Left Boundary', leftFrame, 'tag-left')}
      ${renderKeyCard('Anchor Frame', anchorFrame, 'tag-anchor', true)}
      ${renderKeyCard('Right Boundary', rightFrame, 'tag-right')}
    `;

    // Lightbox open trigger
    stripEl.querySelectorAll('.key-frame-card').forEach((card) => {
      const media = card.querySelector('.key-media-box.has-image');
      if (media) {
        const idx = Number(card.dataset.idx);
        const frameObj = frames.find((f) => Number(f.source_frame_idx) === idx) ||
          (idx === Number(anchor.source_frame_idx) ? anchor : null);
        if (frameObj && frameObj.image_url) {
          media.addEventListener('click', () => openLightbox(frameObj));
        }
      }
    });
  }

  function renderFramesGrid(item) {
    const gridEl = $('framesGrid');
    const frames = item.frames || [];
    $('frameCountBadge').textContent = `${frames.length} sampled frames (3 FPS)`;

    const proposed = item.proposed_range || {};
    const curLeft = Number($('leftInput').value || proposed.left);
    const curRight = Number($('rightInput').value || proposed.right);

    gridEl.innerHTML = frames.map((frame) => {
      const idx = Number(frame.source_frame_idx);
      const isAnchor = Boolean(frame.is_anchor);
      const isBoundary = Boolean(frame.is_proposal_boundary);
      const inRange = idx >= curLeft && idx <= curRight;
      const hasImage = Boolean(frame.image_url);
      const outsideWindow = isFrameOutsideWindow(frame, item);

      let cardClasses = 'sequence-frame-card';
      if (isAnchor) cardClasses += ' anchor';
      if (isBoundary) cardClasses += ' boundary';
      if (inRange) cardClasses += ' in-range';

      // Badges
      let roleBadgesHtml = '';
      if (isAnchor) {
        roleBadgesHtml += `<span class="seq-role-tag tag-anchor">Anchor</span>`;
      }
      if (frame.proposal_boundaries && frame.proposal_boundaries.includes('left')) {
        roleBadgesHtml += `<span class="seq-role-tag tag-left">Prop Left</span>`;
      }
      if (frame.proposal_boundaries && frame.proposal_boundaries.includes('right')) {
        roleBadgesHtml += `<span class="seq-role-tag tag-right">Prop Right</span>`;
      }
      if (outsideWindow) {
        roleBadgesHtml += `<span class="seq-role-tag tag-outwindow">Outside ±20s</span>`;
      }
      if (!isAnchor && !isBoundary && !outsideWindow) {
        roleBadgesHtml += `<span class="seq-role-tag tag-grid">3 FPS</span>`;
      }

      const mediaHtml = hasImage
        ? `<img loading="lazy" decoding="async" src="${esc(frame.image_url)}" alt="${esc(frame.frame_uid)}">`
        : `<div class="no-image-placeholder">
             <span>No JPEG</span>
           </div>`;

      return `
        <figure class="${cardClasses}" data-idx="${idx}">
          <div class="seq-card-topbar">
            <div class="seq-top-badges">${roleBadgesHtml}</div>
            <div class="seq-top-actions">
              <button class="seq-mini-btn set-left-btn" data-idx="${idx}" title="Set as Left boundary">◀ L</button>
              <button class="seq-mini-btn set-right-btn" data-idx="${idx}" title="Set as Right boundary">R ▶</button>
            </div>
          </div>
          <div class="frame-thumb-box${hasImage ? ' has-image' : ''}" title="${hasImage ? 'Click to inspect in lightbox' : 'No JPEG materialized'}">
            ${mediaHtml}
          </div>
          <figcaption class="seq-card-bottombar">
            <div class="seq-bot-left">
              <button class="seq-icon-btn zoom-card-btn" data-idx="${idx}" title="Inspect in Lightbox">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              </button>
              <span class="seq-bot-label" title="${esc(frame.frame_uid)}">${esc(frame.frame_uid)}</span>
            </div>
            <div class="seq-bot-time">
              <span>idx <b>${idx}</b></span> · <span>${formatTime(frame.timestamp_s, frame.timestamp_ms)}</span>
            </div>
          </figcaption>
        </figure>
      `;
    }).join('');

    // Attach click listeners to open lightbox
    gridEl.querySelectorAll('.sequence-frame-card').forEach((card) => {
      const idx = Number(card.dataset.idx);
      const frameObj = frames.find((f) => Number(f.source_frame_idx) === idx);
      const media = card.querySelector('.frame-thumb-box.has-image');
      const zoomBtn = card.querySelector('.zoom-card-btn');

      if (frameObj && frameObj.image_url) {
        if (media) {
          media.addEventListener('click', (e) => {
            if (e.target.closest('.seq-card-topbar') || e.target.closest('.seq-card-bottombar')) return;
            openLightbox(frameObj);
          });
        }
        if (zoomBtn) {
          zoomBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openLightbox(frameObj);
          });
        }
      }
    });

    // Quick action buttons (Set Left / Set Right)
    gridEl.querySelectorAll('.set-left-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        $('leftInput').value = btn.dataset.idx;
        validateRange();
        showToast(`Set Left boundary to idx ${btn.dataset.idx}`);
      });
    });

    gridEl.querySelectorAll('.set-right-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        $('rightInput').value = btn.dataset.idx;
        validateRange();
        showToast(`Set Right boundary to idx ${btn.dataset.idx}`);
      });
    });
  }

  function renderDetail(item) {
    $('emptyState').classList.add('hidden');
    $('reviewArticle').classList.remove('hidden');

    // Header & Meta
    $('proposalVideoTag').textContent = item.video_id || '--';
    $('proposalTitle').textContent = item.review_uid || '--';

    const status = item.review_status || 'pending';
    const statusBadge = $('proposalStatusBadge');
    statusBadge.textContent = status;
    statusBadge.className = `status-pill ${status}`;

    $('proposalTaskBadge').textContent = `Task: ${item.task || 'kis'}`;
    $('proposalSourceBadge').textContent = `Source: ${item.source || 'ai_raw'}`;

    $('queryUidLabel').textContent = item.query_uid || '';
    $('queryText').textContent = item.query || '(No query text available)';

    $('metaVideoId').textContent = item.video_id || '--';

    const anchor = item.anchor || {};
    $('metaAnchor').textContent = `idx ${anchor.source_frame_idx ?? '--'} · ${formatTime(anchor.timestamp_s, anchor.timestamp_ms)}`;

    const proposed = item.proposed_range || {};
    $('metaProposedRange').textContent = `[${proposed.left ?? '?'}, ${proposed.right ?? '?'}]`;

    const decision = item.decision || {};
    if (decision.reviewer || decision.reviewed_at) {
      $('reviewedByChip').classList.remove('hidden');
      $('metaReviewedBy').textContent = `${decision.reviewer || 'teammate'} (${decision.status || status})`;
    } else {
      $('reviewedByChip').classList.add('hidden');
    }

    // Adjudication Controls Values
    $('leftInput').value = decision.left ?? proposed.left ?? '';
    $('rightInput').value = decision.right ?? proposed.right ?? '';

    $('anchorIdxDisplay').textContent = anchor.source_frame_idx ?? '--';
    $('anchorTimeDisplay').textContent = formatTime(anchor.timestamp_s, anchor.timestamp_ms);

    $('reviewerInput').value = state.reviewerName;
    $('noteInput').value = decision.note || '';

    // Validate inputs & durations
    validateRange();

    // Render Key Strip & Sequence Grid
    renderKeyStrip(item);
    renderFramesGrid(item);
  }

  async function loadDetail(uid) {
    if (!uid) return;
    try {
      const response = await fetch(`/api/review/items/${encodeURIComponent(uid)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.selected = await response.json();
      renderQueue();
      renderDetail(state.selected);
    } catch (err) {
      console.error('Failed to load item detail:', err);
      showToast('Error loading proposal detail.');
    }
  }

  // =========================================================================
  // Fullscreen Image Lightbox Viewer (Precise Fit vs 100% Actual Scale)
  // =========================================================================

  function calculateFitScale(img, viewport) {
    const nw = img.naturalWidth || 640;
    const nh = img.naturalHeight || 360;
    const vw = Math.max(100, viewport.clientWidth - 40);
    const vh = Math.max(100, viewport.clientHeight - 40);
    return Math.min(vw / nw, vh / nh, 1.0);
  }

  function applyLightboxTransform() {
    const wrapper = $('lightboxImageWrapper');
    const viewport = $('lightboxViewport');
    if (!wrapper || !viewport) return;

    wrapper.style.transform = `translate(${lightboxState.translateX}px, ${lightboxState.translateY}px) scale(${lightboxState.scale})`;

    const isZoomed = lightboxState.scale > (lightboxState.fitScale * 1.01);
    if (isZoomed) {
      viewport.classList.add('can-pan');
    } else {
      viewport.classList.remove('can-pan', 'is-dragging');
      lightboxState.isDragging = false;
    }

    // Display formatted zoom level
    const pct = Math.round(lightboxState.scale * 100);
    $('zoomLevelDisplay').textContent = `${pct}%`;
  }

  function setLightboxZoom(newScale, focalX = 0, focalY = 0) {
    const clampedScale = Math.min(Math.max(newScale, lightboxState.minScale), lightboxState.maxScale);
    if (focalX !== 0 || focalY !== 0) {
      const factor = clampedScale / lightboxState.scale;
      lightboxState.translateX = focalX - (focalX - lightboxState.translateX) * factor;
      lightboxState.translateY = focalY - (focalY - lightboxState.translateY) * factor;
    }
    lightboxState.scale = clampedScale;
    applyLightboxTransform();
  }

  function resetLightboxToFit() {
    const img = $('lightboxImage');
    const viewport = $('lightboxViewport');
    lightboxState.fitScale = calculateFitScale(img, viewport);
    lightboxState.scale = lightboxState.fitScale;
    lightboxState.translateX = 0;
    lightboxState.translateY = 0;
    applyLightboxTransform();
  }

  function setLightboxToActualSize() {
    // 100% represents actual 1:1 image pixels (scale = 1.0)
    lightboxState.scale = 1.0;
    lightboxState.translateX = 0;
    lightboxState.translateY = 0;
    applyLightboxTransform();
  }

  function openLightbox(frame) {
    if (!frame || !frame.image_url) {
      showToast('Image is not materialized for this frame.');
      return;
    }

    lightboxState.currentFrame = frame;
    lightboxState.isOpen = true;

    // Set metadata
    $('lightboxFrameUid').textContent = frame.frame_uid || '--';
    $('lightboxFrameIdx').textContent = frame.source_frame_idx ?? '--';
    $('lightboxTimeDisplay').textContent = formatTime(frame.timestamp_s, frame.timestamp_ms);

    // Build role tags
    let tagsHtml = '';
    if (frame.is_anchor) {
      tagsHtml += `<span class="seq-role-tag tag-anchor">Anchor</span>`;
    }
    if (frame.proposal_boundaries && frame.proposal_boundaries.includes('left')) {
      tagsHtml += `<span class="seq-role-tag tag-left">Prop Left</span>`;
    }
    if (frame.proposal_boundaries && frame.proposal_boundaries.includes('right')) {
      tagsHtml += `<span class="seq-role-tag tag-right">Prop Right</span>`;
    }
    if (state.selected && isFrameOutsideWindow(frame, state.selected)) {
      tagsHtml += `<span class="seq-role-tag tag-outwindow">Outside ±20s</span>`;
    }
    if (!frame.is_anchor && (!frame.proposal_boundaries || frame.proposal_boundaries.length === 0)) {
      tagsHtml += `<span class="seq-role-tag tag-grid">3 FPS Grid</span>`;
    }
    $('lightboxRoleBadges').innerHTML = tagsHtml;

    // Load image
    const imgEl = $('lightboxImage');
    imgEl.onload = () => {
      resetLightboxToFit();
    };
    imgEl.src = frame.image_url;
    imgEl.alt = frame.frame_uid || 'Frame view';

    // Update Nav buttons
    updateLightboxNavState();

    const dialog = $('lightboxModal');
    if (dialog && !dialog.open) {
      dialog.showModal();
    }
  }

  function closeLightbox() {
    const dialog = $('lightboxModal');
    if (dialog && dialog.open) {
      dialog.close();
    }
    lightboxState.isOpen = false;
    lightboxState.currentFrame = null;
  }

  function updateLightboxNavState() {
    if (!state.selected || !state.selected.frames || !lightboxState.currentFrame) return;
    const materializedFrames = state.selected.frames.filter((f) => f.image_url);
    const curIdx = materializedFrames.findIndex(
      (f) => Number(f.source_frame_idx) === Number(lightboxState.currentFrame.source_frame_idx)
    );

    $('lightboxPrevFrameBtn').disabled = curIdx <= 0;
    $('lightboxNextFrameBtn').disabled = curIdx === -1 || curIdx >= materializedFrames.length - 1;
  }

  function navigateLightboxFrame(direction) {
    if (!state.selected || !state.selected.frames || !lightboxState.currentFrame) return;
    const materializedFrames = state.selected.frames.filter((f) => f.image_url);
    const curIdx = materializedFrames.findIndex(
      (f) => Number(f.source_frame_idx) === Number(lightboxState.currentFrame.source_frame_idx)
    );
    if (curIdx === -1) return;

    const targetIdx = curIdx + direction;
    if (targetIdx >= 0 && targetIdx < materializedFrames.length) {
      openLightbox(materializedFrames[targetIdx]);
    }
  }

  function setupLightboxEvents() {
    const dialog = $('lightboxModal');
    const viewport = $('lightboxViewport');

    // Close button
    $('closeLightboxBtn').addEventListener('click', closeLightbox);
    dialog.addEventListener('cancel', () => {
      lightboxState.isOpen = false;
      lightboxState.currentFrame = null;
    });

    // Zoom Buttons
    $('zoomInBtn').addEventListener('click', () => setLightboxZoom(lightboxState.scale * 1.25));
    $('zoomOutBtn').addEventListener('click', () => setLightboxZoom(lightboxState.scale / 1.25));
    $('zoomFitBtn').addEventListener('click', resetLightboxToFit);
    $('zoomActualBtn').addEventListener('click', setLightboxToActualSize);

    // Prev / Next Frame buttons
    $('lightboxPrevFrameBtn').addEventListener('click', () => navigateLightboxFrame(-1));
    $('lightboxNextFrameBtn').addEventListener('click', () => navigateLightboxFrame(1));

    // Set Left / Set Right from Lightbox
    $('lightboxSetLeftBtn').addEventListener('click', () => {
      if (!lightboxState.currentFrame) return;
      $('leftInput').value = lightboxState.currentFrame.source_frame_idx;
      validateRange();
      showToast(`Set Left boundary to idx ${lightboxState.currentFrame.source_frame_idx}`);
    });

    $('lightboxSetRightBtn').addEventListener('click', () => {
      if (!lightboxState.currentFrame) return;
      $('rightInput').value = lightboxState.currentFrame.source_frame_idx;
      validateRange();
      showToast(`Set Right boundary to idx ${lightboxState.currentFrame.source_frame_idx}`);
    });

    // Mouse Wheel Zoom centered on cursor focal point
    viewport.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const focalX = e.clientX - (rect.left + rect.width / 2);
      const focalY = e.clientY - (rect.top + rect.height / 2);
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      setLightboxZoom(lightboxState.scale * factor, focalX, focalY);
    }, { passive: false });

    // Drag to pan ONLY when zoomed in
    viewport.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      if (lightboxState.scale <= (lightboxState.fitScale * 1.01)) return; // Don't drag when at Fit scale
      lightboxState.isDragging = true;
      lightboxState.startX = e.clientX - lightboxState.translateX;
      lightboxState.startY = e.clientY - lightboxState.translateY;
      viewport.classList.add('is-dragging');
    });

    window.addEventListener('mousemove', (e) => {
      if (!lightboxState.isDragging) return;
      lightboxState.translateX = e.clientX - lightboxState.startX;
      lightboxState.translateY = e.clientY - lightboxState.startY;
      applyLightboxTransform();
    });

    window.addEventListener('mouseup', () => {
      if (lightboxState.isDragging) {
        lightboxState.isDragging = false;
        viewport.classList.remove('is-dragging');
      }
    });

    // Double-click to toggle Fit / 2x zoom
    viewport.addEventListener('dblclick', (e) => {
      if (e.target === $('lightboxSetLeftBtn') || e.target === $('lightboxSetRightBtn')) return;
      if (lightboxState.scale <= (lightboxState.fitScale * 1.05)) {
        const rect = viewport.getBoundingClientRect();
        const focalX = e.clientX - (rect.left + rect.width / 2);
        const focalY = e.clientY - (rect.top + rect.height / 2);
        setLightboxZoom(2.0, focalX, focalY);
      } else {
        resetLightboxToFit();
      }
    });

    // Window resize recalculates Fit if currently at Fit
    window.addEventListener('resize', () => {
      if (lightboxState.isOpen && Math.abs(lightboxState.scale - lightboxState.fitScale) < 0.05) {
        resetLightboxToFit();
      }
    });
  }

  // =========================================================================
  // Decision Submission & Persistence
  // =========================================================================

  async function submitDecision(status) {
    if (!state.selected) return;

    let left = Number($('leftInput').value);
    let right = Number($('rightInput').value);
    const proposed = state.selected.proposed_range || {};

    if (status === 'accepted') {
      left = Number(proposed.left);
      right = Number(proposed.right);
      $('leftInput').value = left;
      $('rightInput').value = right;
    } else if (status === 'rejected') {
      left = Number(proposed.left);
      right = Number(proposed.right);
    }

    const reviewer = $('reviewerInput').value.trim() || 'teammate';
    const note = $('noteInput').value.trim() || null;

    state.reviewerName = reviewer;
    localStorage.setItem('hcmaic_reviewer', reviewer);

    const validation = validateRange();
    if (status === 'edited' && !validation.valid) {
      showToast('Please fix range validation error before saving.');
      return;
    }

    const payload = { status, left, right, reviewer, note };

    try {
      const response = await fetch(`/api/review/items/${encodeURIComponent(state.selected.review_uid)}/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      if (!response.ok) {
        showToast(`Decision error: ${data.detail || 'Failed to save'}`);
        return;
      }

      state.selected.decision = data;
      state.selected.review_status = data.status;
      showToast(`Decision saved: ${status.toUpperCase()}`);

      await loadList();
      renderDetail(state.selected);

      // Auto-advance to next pending item
      if (state.currentFilter === 'pending') {
        const nextPending = state.items.find((i) => i.review_uid !== state.selected.review_uid && i.review_status === 'pending');
        if (nextPending) {
          loadDetail(nextPending.review_uid);
        }
      }
    } catch (err) {
      console.error('Failed to save decision:', err);
      showToast('Network error saving decision.');
    }
  }

  // =========================================================================
  // Export Decisions
  // =========================================================================

  async function exportDecisions() {
    try {
      const response = await fetch('/api/review/export');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `review_decisions_export_${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      showToast('Exported review decisions JSON.');
    } catch (err) {
      console.error('Export failed:', err);
      showToast('Failed to export decisions.');
    }
  }

  // =========================================================================
  // Keyboard Shortcuts & Navigation
  // =========================================================================

  function navigateQueue(direction) {
    if (!state.items || state.items.length === 0) return;
    const currentIndex = state.selected
      ? state.items.findIndex((i) => i.review_uid === state.selected.review_uid)
      : -1;

    let targetIndex = currentIndex + direction;
    if (targetIndex < 0) targetIndex = 0;
    if (targetIndex >= state.items.length) targetIndex = state.items.length - 1;

    if (targetIndex !== currentIndex && state.items[targetIndex]) {
      loadDetail(state.items[targetIndex].review_uid);
    }
  }

  function setupEventListeners() {
    // Search
    const searchInput = $('searchInput');
    const clearBtn = $('clearSearchBtn');

    searchInput.addEventListener('input', () => {
      state.searchQuery = searchInput.value;
      if (state.searchQuery) {
        clearBtn.classList.remove('hidden');
      } else {
        clearBtn.classList.add('hidden');
      }
      filterItems();
    });

    clearBtn.addEventListener('click', () => {
      searchInput.value = '';
      state.searchQuery = '';
      clearBtn.classList.add('hidden');
      filterItems();
      searchInput.focus();
    });

    // Segmented Filter Tabs
    document.querySelectorAll('.filter-btn').forEach((tab) => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.filter-btn').forEach((t) => t.classList.remove('active'));
        tab.classList.add('active');
        state.currentFilter = tab.dataset.status;
        loadList();
      });
    });

    // Inputs Validation
    $('leftInput').addEventListener('input', validateRange);
    $('rightInput').addEventListener('input', validateRange);

    // Reset to Proposed
    $('resetRangeBtn').addEventListener('click', () => {
      if (!state.selected) return;
      const proposed = state.selected.proposed_range || {};
      $('leftInput').value = proposed.left ?? '';
      $('rightInput').value = proposed.right ?? '';
      validateRange();
      showToast('Reset to original proposal range.');
    });

    // Decision Buttons
    $('acceptBtn').addEventListener('click', () => submitDecision('accepted'));
    $('saveEditBtn').addEventListener('click', () => submitDecision('edited'));
    $('rejectBtn').addEventListener('click', () => submitDecision('rejected'));

    // Export Button
    $('exportBtn').addEventListener('click', exportDecisions);

    // Shortcuts Modal
    const modal = $('shortcutsModal');
    $('shortcutsBtn').addEventListener('click', () => modal.showModal());
    $('closeShortcutsBtn').addEventListener('click', () => modal.close());
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.close();
    });

    // Setup Lightbox Event Listeners
    setupLightboxEvents();

    // Global Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      // If Lightbox is open: handle lightbox specific shortcuts
      if (lightboxState.isOpen) {
        if (e.key === 'Escape') {
          e.preventDefault();
          closeLightbox();
          return;
        }
        if (e.key === '+' || e.key === '=') {
          e.preventDefault();
          setLightboxZoom(lightboxState.scale * 1.25);
          return;
        }
        if (e.key === '-' || e.key === '_') {
          e.preventDefault();
          setLightboxZoom(lightboxState.scale / 1.25);
          return;
        }
        if (e.key === '0') {
          e.preventDefault();
          resetLightboxToFit();
          return;
        }
        if (e.key === 'ArrowLeft') {
          e.preventDefault();
          navigateLightboxFrame(-1);
          return;
        }
        if (e.key === 'ArrowRight') {
          e.preventDefault();
          navigateLightboxFrame(1);
          return;
        }
        return;
      }

      // Ignore shortcut keys if focused on text input or textarea
      const tag = (document.activeElement && document.activeElement.tagName) || '';
      const isInput = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

      if (e.key === '?' && !isInput) {
        e.preventDefault();
        modal.open ? modal.close() : modal.showModal();
        return;
      }

      if (modal.open) return;

      if (isInput) return;

      if (e.key === 'a' || e.key === 'A') {
        e.preventDefault();
        submitDecision('accepted');
      } else if (e.key === 'e' || e.key === 'E') {
        e.preventDefault();
        submitDecision('edited');
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        submitDecision('rejected');
      } else if (e.key === 'ArrowUp' || e.key === 'k' || e.key === 'K') {
        e.preventDefault();
        navigateQueue(-1);
      } else if (e.key === 'ArrowDown' || e.key === 'j' || e.key === 'J') {
        e.preventDefault();
        navigateQueue(1);
      }
    });
  }

  // =========================================================================
  // App Initialization
  // =========================================================================

  setupEventListeners();
  loadHealth().catch(console.error);
  loadList().catch(console.error);
})();
