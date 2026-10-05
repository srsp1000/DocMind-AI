/* ── State ── */
let sessionId = null;
let currentOutput = "";
let currentChoice = "Viva Questions";
let currentDifficulty = "Medium";
let currentBackend = "Groq (Cloud)";
let currentSearchMode = "hybrid";
let currentFramework = "native";
let lightboxImageUrl = "";
let lightboxSourceUrl = "";
let _pageCount = 0;

/* ── Backend toggle ── */
function setBackend(btn) {
  document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentBackend = btn.dataset.value;
  const badge = document.getElementById("modelBadge");
  if (currentBackend === "Groq (Cloud)") {
    badge.textContent = "llama-3.1-8b · Cloud";
    badge.style.cssText = "background:rgba(167,139,250,0.1);color:#a78bfa;border-color:rgba(167,139,250,0.25)";
  } else {
    badge.textContent = "mistral · Local";
    badge.style.cssText = "background:rgba(16,185,129,0.1);color:#10b981;border-color:rgba(16,185,129,0.25)";
  }
}

/* ── Difficulty ── */
function setDiff(btn, val) {
  btn.closest('.diff-row').querySelectorAll(".diff-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentDifficulty = val;
}

/* ── Search Mode ── */
function setSearchMode(btn, mode) {
  document.querySelectorAll("#searchModeRow .diff-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentSearchMode = mode;
}

/* ── Framework ── */
function setFramework(btn, fw) {
  document.querySelectorAll("#frameworkRow .diff-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentFramework = fw;
}

/* ── Drop zone ── */
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function removeFile() {
  sessionId = null; fileInput.value = "";
  document.getElementById("filePreview").classList.add("hidden");
  document.getElementById("ragStatus").classList.add("hidden");
  document.getElementById("dropZone").classList.remove("hidden");
  document.getElementById("genBtn").disabled = true;
  document.getElementById("genHint").classList.remove("hidden");
}

async function handleFile(file) {
  if (!file.name.toLowerCase().endsWith(".pdf")) { alert("Only PDF files are supported."); return; }
  dropZone.classList.add("hidden");
  const prog = document.getElementById("uploadProgress");
  const fill = document.getElementById("progressFill");
  const label = document.getElementById("progressLabel");
  prog.classList.remove("hidden");
  let pct = 0;
  const interval = setInterval(() => { pct = Math.min(pct + Math.random() * 15, 85); fill.style.width = pct + "%"; }, 200);
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    clearInterval(interval);
    fill.style.width = "100%";
    label.textContent = "Index built!";
    setTimeout(() => {
      prog.classList.add("hidden"); fill.style.width = "0%";
      sessionId = data.session_id;
      document.getElementById("fileName").textContent = data.file_name;
      document.getElementById("fileMeta").textContent = `✓ Indexed · ${data.chunk_count} chunks · ${data.page_count || '?'} pages · FAISS ready`;
      _pageCount = data.page_count || 0;
      document.getElementById("filePreview").classList.remove("hidden");
      document.getElementById("ragStatus").classList.remove("hidden");
      document.getElementById("ragStatusText").textContent = `RAG ACTIVE · ${data.chunk_count} vectors · ${data.page_count || '?'} pages`;
      document.getElementById("genBtn").disabled = false;
      document.getElementById("genHint").classList.add("hidden");
      document.getElementById("fcBtn").style.display = "block";
    }, 600);
  } catch (err) {
    clearInterval(interval); prog.classList.add("hidden");
    dropZone.classList.remove("hidden"); fill.style.width = "0%";
    alert("Upload error: " + err.message);
  }
}

/* ── Generate ── */
async function generate() {
  if (!sessionId) return;
  currentChoice = document.getElementById("outputType").value;
  const btn = document.getElementById("genBtn");
  const btnText = document.getElementById("genBtnText");
  btn.disabled = true;
  btnText.innerHTML = '<div class="spinner"></div> Generating…';
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("choice", currentChoice);
  form.append("difficulty", currentDifficulty);
  form.append("backend", currentBackend);
  form.append("retriever_mode", currentSearchMode);
  form.append("framework", currentFramework);
  try {
    const res = await fetch("/generate", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Generation failed");
    currentOutput = data.output;
    showOutput(currentOutput);
    renderCitations(data.citations || [], document.getElementById("outputCitations"));
    document.getElementById("chatCard").classList.remove("hidden");
    document.getElementById("chatMessages").innerHTML = '<div class="chat-empty">Ask anything about the generated content</div>';
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    btn.disabled = false;
    btnText.innerHTML = '<span class="gen-icon">⚡</span> Generate Content';
  }
}

function renderFormattedMarkdown(text) {
  if (!text) return "";
  let raw = text;

  // 1. Process Math Block & Inline LaTeX equations
  raw = raw.replace(/\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$/g, (match, p1, p2) => {
    const eq = (p1 || p2 || "").trim();
    if (typeof katex !== "undefined") {
      try {
        return `<div class="katex-math-block">${katex.renderToString(eq, { displayMode: true, throwOnError: false })}</div>`;
      } catch (e) {}
    }
    const cleanEq = eq
      .replace(/\\upsilon/g, "υ").replace(/\\phi/g, "φ").replace(/\\theta/g, "θ").replace(/\\alpha/g, "α").replace(/\\beta/g, "β")
      .replace(/\\sum/g, "∑").replace(/\\in/g, "∈").replace(/\\neq/g, "≠").replace(/\\leq/g, "≤").replace(/\\geq/g, "≥")
      .replace(/\\times/g, "×").replace(/\\cases/g, "").replace(/\\begin\{.*?\}/g, "").replace(/\\end\{.*?\}/g, "")
      .replace(/\\text\{(.*?)\}/g, "$1").replace(/_\{?(.*?)\}?/g, "<sub>$1</sub>").replace(/\^\{?(.*?)\}?/g, "<sup>$1</sup>");
    return `<div class="katex-math-block"><code>${cleanEq}</code></div>`;
  });

  raw = raw.replace(/\\\(([\s\S]*?)\\\)|\$([^\$\n]+)\$/g, (match, p1, p2) => {
    const eq = (p1 || p2 || "").trim();
    if (typeof katex !== "undefined") {
      try {
        return `<span class="katex-math">${katex.renderToString(eq, { displayMode: false, throwOnError: false })}</span>`;
      } catch (e) {}
    }
    const cleanEq = eq
      .replace(/\\upsilon/g, "υ").replace(/\\phi/g, "φ").replace(/\\theta/g, "θ").replace(/\\alpha/g, "α").replace(/\\beta/g, "β")
      .replace(/\\in/g, "∈").replace(/\\neq/g, "≠").replace(/_\{?(.*?)\}?/g, "<sub>$1</sub>").replace(/\^\{?(.*?)\}?/g, "<sup>$1</sup>");
    return `<span class="katex-math">${cleanEq}</span>`;
  });

  // 2. Format Source tags like [Source: Page X/Y] or *Source: Page X* into citation pills
  raw = raw.replace(/\[?\*?Source:\s*Page\s*([0-9\/\–,\s]+)\*?\]?/gi, '<span class="output-source-pill">📌 Source: Page $1</span>');

  // 3. Use marked.js if available
  if (typeof marked !== "undefined" && typeof marked.parse === "function") {
    try {
      marked.setOptions({ breaks: true, gfm: true });
      return marked.parse(raw);
    } catch (e) {
      console.warn("marked.parse error fallback:", e);
    }
  }

  // Fallback Markdown Parsing
  let html = raw
    .replace(/^#### (.*$)/gim, '<h4 class="out-h4">$1</h4>')
    .replace(/^### (.*$)/gim, '<h3 class="out-h3">$1</h3>')
    .replace(/^## (.*$)/gim, '<h2 class="out-h2">$1</h2>')
    .replace(/^# (.*$)/gim, '<h1 class="out-h1">$1</h1>')
    .replace(/^---$/gim, '<hr class="out-hr">')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/^\s*[-*]\s+(.*$)/gim, '<li class="out-li">$1</li>')
    .replace(/^\s*(\d+)\.\s+(.*$)/gim, '<li class="out-li-num"><span class="li-num">$1.</span> $2</li>')
    .replace(/\n\n/g, '<br><br>');

  return html;
}

function showOutput(text) {
  document.getElementById("outputEmpty").classList.add("hidden");
  const box = document.getElementById("outputBox");
  box.innerHTML = renderFormattedMarkdown(text);
  box.classList.remove("hidden");
  document.getElementById("copyBtn").classList.remove("hidden");
  document.getElementById("dlBtn").classList.remove("hidden");
}

function copyOutput() {
  navigator.clipboard.writeText(currentOutput).then(() => {
    const btn = document.getElementById("copyBtn");
    btn.textContent = "✓ Copied";
    setTimeout(() => btn.textContent = "⎘ Copy", 2000);
  });
}

async function downloadPDF() {
  const form = new FormData();
  form.append("text", currentOutput);
  form.append("output_type", currentChoice);
  const res = await fetch("/download", { method: "POST", body: form });
  if (!res.ok) { alert("Download failed"); return; }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "study_material.pdf";
  a.click(); URL.revokeObjectURL(url);
}

/* ══════════════════════════════════════════
   CHAT + ADVANCED VISUAL SYSTEM
══════════════════════════════════════════ */

async function sendChat() {
  const input = document.getElementById("chatInput");
  const question = input.value.trim();
  if (!question || !sessionId) return;
  input.value = "";

  const msgs = document.getElementById("chatMessages");
  const empty = msgs.querySelector(".chat-empty");
  if (empty) empty.remove();

  // User bubble
  appendMsg(msgs, "user", question);

  // Skeleton visual placeholder + thinking bubble
  const thinkId = "think-" + Date.now();
  const skelId  = "skel-"  + Date.now();
  msgs.innerHTML += buildThinkingBubble(thinkId);
  msgs.innerHTML += buildSkeletonVisual(skelId);
  msgs.scrollTop = msgs.scrollHeight;

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("question", question);
  form.append("backend", currentBackend);
  form.append("generated_output", currentOutput);
  form.append("retriever_mode", currentSearchMode);
  form.append("framework", currentFramework);

  try {
    const res = await fetch("/chat", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Chat failed");

    // Remove thinking + skeleton
    document.getElementById(thinkId)?.remove();
    document.getElementById(skelId)?.remove();

    // AI text bubble
    appendMsg(msgs, "ai", data.content);

    // Visuals
    if (data.visuals && data.visuals.length > 0) {
      for (const v of data.visuals) {
        const card = buildVisualCard(v);
        msgs.innerHTML += card;
      }
    } else if (data.visual_hint && data.visual_hint !== "none") {
      // Show "no visual found" note
      msgs.innerHTML += `<div style="font-size:0.68rem;color:#334155;padding:4px 0 8px">No diagram found for: <em>${escHtml(data.visual_hint)}</em></div>`;
    }

    // Citation pills in chat
    if (data.citations && data.citations.length > 0) {
      let cHtml = '<div class="chat-citations">';
      for (const c of data.citations) {
        const pages = (c.pages || []).join(', ');
        cHtml += `<span class="cite-pill" title="${escAttr(c.text_preview)}">📄 Page ${pages} · ${(c.score * 100).toFixed(0)}%</span>`;
      }
      cHtml += '</div>';
      msgs.innerHTML += cHtml;
    }

    msgs.scrollTop = msgs.scrollHeight;

  } catch (err) {
    document.getElementById(thinkId)?.remove();
    document.getElementById(skelId)?.remove();
    appendMsg(msgs, "ai", "Error: " + err.message);
  }
}

function appendMsg(container, role, text) {
  if (role === "user") {
    container.innerHTML += `<div class="msg-label msg-label-r">You</div><div class="msg-user">${escHtml(text)}</div>`;
  } else {
    container.innerHTML += `<div class="msg-label">AI</div><div class="msg-ai">${renderFormattedMarkdown(text)}</div>`;
  }
  container.scrollTop = container.scrollHeight;
}

function buildThinkingBubble(id) {
  return `<div class="msg-label">AI</div>
  <div class="msg-ai" id="${id}" style="display:flex;align-items:center;gap:8px;color:#475569">
    <div class="spinner" style="width:12px;height:12px;border-width:1.5px"></div>
    <em>Thinking…</em>
  </div>`;
}

function buildSkeletonVisual(id) {
  return `<div class="visual-skeleton" id="${id}">
    <div class="skel-bar skel-header"></div>
    <div class="skel-bar skel-img"></div>
    <div class="skel-bar skel-footer"></div>
  </div>`;
}

function buildVisualCard(v) {
  const imgSrc = v.data_uri || v.image_url || "";
  if (!imgSrc) return "";

  // Badge
  const badgeClass = v.type === "local" ? "badge-local" : (v.provider === "Wikipedia" || v.provider === "Wikimedia Commons") ? "badge-wiki" : "badge-unsplash";
  const badgeLabel = v.type === "local" ? "📐 Local" : v.provider === "Wikimedia Commons" ? "🖼 Commons" : v.provider === "Wikipedia" ? "📖 Wikipedia" : "📷 Unsplash";

  // Caption + source
  const caption = escHtml(v.caption || "");
  const sourceHtml = v.source_url
    ? `<a href="${v.source_url}" target="_blank" rel="noopener">View source ↗</a>`
    : "";

  // Encode for onclick
  const safeUrl = imgSrc.startsWith("data:") ? "" : imgSrc;
  const safeSrc = escAttr(imgSrc);
  const safeCaption = escAttr(v.caption || "");
  const safeSource = escAttr(v.source_url || "");
  const safeTitle = escAttr(v.title || "Visual");

  return `
  <div class="visual-card">
    <div class="visual-header">
      <span class="visual-title">${escHtml(v.title || "Visual")}</span>
      <span class="visual-badge ${badgeClass}">${badgeLabel}</span>
    </div>
    <div class="visual-img-wrap" onclick="openLightbox('${safeSrc}','${safeCaption}','${safeSource}','${safeTitle}')">
      <img src="${safeSrc}" alt="${safeTitle}" loading="lazy" onerror="this.closest('.visual-card').style.display='none'">
      <div class="visual-zoom-hint">🔍 Click to expand</div>
    </div>
    <div class="visual-footer">
      <div class="visual-caption">${caption} ${sourceHtml}</div>
      ${safeUrl ? `<button class="visual-dl" onclick="downloadImage('${safeUrl}','${safeTitle}')">⬇</button>` : ""}
    </div>
  </div>`;
}

/* ── Lightbox ── */
function openLightbox(src, caption, sourceUrl, title) {
  lightboxImageUrl = src;
  lightboxSourceUrl = sourceUrl;
  document.getElementById("lightboxImg").src = src;
  document.getElementById("lightboxImg").alt = title;
  document.getElementById("lightboxCaption").innerHTML =
    caption + (sourceUrl ? ` <a href="${sourceUrl}" target="_blank" rel="noopener">View source ↗</a>` : "");
  const srcBtn = document.getElementById("lightboxSource");
  if (sourceUrl) { srcBtn.href = sourceUrl; srcBtn.style.display = "inline-flex"; }
  else { srcBtn.style.display = "none"; }
  document.getElementById("lightbox").classList.add("open");
  document.body.style.overflow = "hidden";
}

function closeLightbox(e) {
  if (e && e.target !== document.getElementById("lightbox") && !e.target.classList.contains("lightbox-close")) return;
  document.getElementById("lightbox").classList.remove("open");
  document.body.style.overflow = "";
}

document.addEventListener("keydown", e => { if (e.key === "Escape") { document.getElementById("lightbox").classList.remove("open"); document.body.style.overflow = ""; } });

function downloadLightboxImage() {
  if (!lightboxImageUrl) return;
  downloadImage(lightboxImageUrl, "visual");
}

function downloadImage(url, name) {
  if (url.startsWith("data:")) {
    const a = document.createElement("a");
    a.href = url; a.download = (name || "diagram") + ".svg";
    a.click(); return;
  }
  fetch(url).then(r => r.blob()).then(blob => {
    const ext = url.split(".").pop().split("?")[0] || "png";
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (name || "visual") + "." + ext;
    a.click(); URL.revokeObjectURL(a.href);
  }).catch(() => window.open(url, "_blank"));
}

/* ── Helpers ── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/\n/g, "<br>");
}

function escAttr(str) {
  return String(str).replace(/'/g, "&#39;").replace(/"/g, "&quot;");
}


/* ══════════════════════════════════════════
   FLASHCARD SYSTEM
══════════════════════════════════════════ */
let fcCards = [];
let fcIndex = 0;
let fcCount = 10;
let fcDifficulty = "Medium";
let fcRatings = {}; // index → true(known) / false(review)
let fcFlipped = false;

function openFcModal() {
  const overlay = document.getElementById("fcOverlay") || document.getElementById("fcModal");
  if (overlay) overlay.classList.remove("hidden");
}
function openFlashcardModal() {
  openFcModal();
}
function closeFcModal() {
  const overlay = document.getElementById("fcOverlay") || document.getElementById("fcModal");
  if (overlay) overlay.classList.add("hidden");
}

function setFcCount(btn, n) {
  fcCount = n;
  document.querySelectorAll(".fc-count-btn, .count-btn").forEach(b => {
    b.classList.remove("fc-count-active", "active");
  });
  if (btn) btn.classList.add("fc-count-active", "active");
}
function setCount(n) {
  setFcCount(null, n);
}

function setFcDiff(btn, val) {
  fcDifficulty = val;
  const modal = document.getElementById("fcOverlay") || document.getElementById("fcModal");
  if (modal) {
    modal.querySelectorAll(".diff-btn").forEach(b => b.classList.remove("active"));
  }
  if (btn) btn.classList.add("active");
}

async function generateFlashcards() {
  if (!sessionId) return;
  const btn = document.getElementById("fcGenBtn") || document.getElementById("modalGenBtn");
  const txt = document.getElementById("fcGenText") || document.getElementById("modalGenText");
  if (btn) btn.disabled = true;
  if (txt) txt.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:6px"></div> Generating…';

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("count", fcCount);
  form.append("difficulty", fcDifficulty);
  form.append("backend", currentBackend);

  try {
    const res = await fetch("/flashcards", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to generate flashcards");

    fcCards = data.cards;
    fcIndex = 0;
    fcRatings = {};
    fcFlipped = false;

    closeFcModal();
    renderFlashcards();
    const sec = document.getElementById("fcSection");
    if (sec) {
      sec.classList.remove("hidden");
      sec.scrollIntoView({ behavior: "smooth", block: "start" });
    }

  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    if (btn) btn.disabled = false;
    if (txt) txt.textContent = "⚡ Generate Flashcards";
  }
}

function renderFlashcards() {
  if (!fcCards.length) return;

  // Show section, hide complete screen
  document.getElementById("fcSection").classList.remove("hidden");
  const viewer = document.getElementById("fcViewer") || document.querySelector(".fc-viewer");

  // Update card content
  const card = fcCards[fcIndex];
  document.getElementById("fcQuestion").textContent = card.question;
  document.getElementById("fcAnswer").textContent = card.answer;

  // Reset flip
  fcFlipped = false;
  document.getElementById("fcCard").classList.remove("flipped");

  // Progress
  const total = fcCards.length;
  const done = Object.keys(fcRatings).length;
  const known = Object.values(fcRatings).filter(Boolean).length;
  const unknown = done - known;
  const remaining = total - done;

  document.getElementById("fcProgressFill").style.width = `${(done / total) * 100}%`;
  document.getElementById("fcProgressLabel").textContent = `${done} / ${total}`;
  document.getElementById("fcKnown").textContent = known;
  document.getElementById("fcUnknown").textContent = unknown;
  document.getElementById("fcRemaining").textContent = remaining;

  // Nav buttons
  document.querySelector(".fc-prev").disabled = fcIndex === 0;
  document.querySelector(".fc-next").disabled = fcIndex === fcCards.length - 1;

  // Thumbnail strip
  const strip = document.getElementById("fcStrip");
  strip.innerHTML = fcCards.map((_, i) => {
    let cls = "fc-thumb";
    if (i === fcIndex) cls += " active";
    else if (fcRatings[i] === true)  cls += " known";
    else if (fcRatings[i] === false) cls += " unknown";
    return `<div class="${cls}" onclick="goToCard(${i})">${i + 1}</div>`;
  }).join("");

  // Check if all rated → show completion
  if (done === total) {
    showCompletion();
  }
}

function flipCard() {
  fcFlipped = !fcFlipped;
  document.getElementById("fcCard").classList.toggle("flipped", fcFlipped);
}

function fcNav(dir) {
  const next = fcIndex + dir;
  if (next >= 0 && next < fcCards.length) {
    fcIndex = next;
    renderFlashcards();
  }
}

function goToCard(i) {
  fcIndex = i;
  renderFlashcards();
}

function rateCard(known) {
  fcRatings[fcIndex] = known;
  // Auto-advance to next unrated card
  const nextUnrated = fcCards.findIndex((_, i) => i > fcIndex && fcRatings[i] === undefined);
  if (nextUnrated !== -1) {
    fcIndex = nextUnrated;
  } else {
    // Try from beginning
    const firstUnrated = fcCards.findIndex((_, i) => fcRatings[i] === undefined);
    if (firstUnrated !== -1) fcIndex = firstUnrated;
  }
  renderFlashcards();
}

function showCompletion() {
  const known = Object.values(fcRatings).filter(Boolean).length;
  const total = fcCards.length;
  const pct = Math.round((known / total) * 100);
  const emoji = pct >= 80 ? "🎉" : pct >= 50 ? "👍" : "📚";

  const viewer = document.querySelector(".fc-viewer");
  const rating = document.getElementById("fcRating");
  viewer.innerHTML = `
    <div class="fc-complete">
      <div class="fc-complete-icon">${emoji}</div>
      <div class="fc-complete-title">Session Complete!</div>
      <div class="fc-complete-sub">You got <strong style="color:#34d399">${known}/${total}</strong> cards correct (${pct}%)</div>
      <button class="fc-restart-btn" onclick="restartFlashcards()">🔄 Restart</button>
    </div>`;
  rating.style.display = "none";
}

function restartFlashcards() {
  fcIndex = 0;
  fcRatings = {};
  fcFlipped = false;
  document.getElementById("fcRating").style.display = "";
  // Re-render viewer
  document.querySelector(".fc-viewer").innerHTML = `
    <button class="fc-nav fc-prev" onclick="fcNav(-1)">‹</button>
    <div class="fc-card-wrap" id="fcCardWrap">
      <div class="fc-card" id="fcCard" onclick="flipCard()">
        <div class="fc-card-inner" id="fcCardInner">
          <div class="fc-face fc-front">
            <div class="fc-face-label">Question</div>
            <div class="fc-face-text" id="fcQuestion">—</div>
            <div class="fc-tap-hint">Tap to reveal answer</div>
          </div>
          <div class="fc-face fc-back">
            <div class="fc-face-label">Answer</div>
            <div class="fc-face-text" id="fcAnswer">—</div>
          </div>
        </div>
      </div>
    </div>
    <button class="fc-nav fc-next" onclick="fcNav(1)">›</button>`;
  renderFlashcards();
}

function closeFlashcards() {
  document.getElementById("fcSection").classList.add("hidden");
}

/* ── Keyboard navigation ── */
document.addEventListener("keydown", e => {
  if (document.getElementById("fcSection").classList.contains("hidden")) return;
  if (e.key === "ArrowRight") fcNav(1);
  if (e.key === "ArrowLeft")  fcNav(-1);
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); flipCard(); }
  if (e.key === "1") rateCard(false);
  if (e.key === "2") rateCard(true);
});

/* ── Export CSV ── */
function exportCSV() {
  if (!fcCards.length) return;
  const rows = [["Question", "Answer"], ...fcCards.map(c => [
    `"${c.question.replace(/"/g, '""')}"`,
    `"${c.answer.replace(/"/g, '""')}"`
  ])];
  const csv = rows.map(r => r.join(",")).join("\n");
  downloadText(csv, "flashcards.csv", "text/csv");
}

/* ── Export Anki (tab-separated) ── */
function exportAnki() {
  if (!fcCards.length) return;
  const tsv = fcCards.map(c => `${c.question}\t${c.answer}`).join("\n");
  downloadText(tsv, "flashcards_anki.txt", "text/plain");
}

function downloadText(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  a.click(); URL.revokeObjectURL(url);
}


/* ══════════════════════════════════════════
   FLASHCARD SYSTEM v2
══════════════════════════════════════════ */
let _fcCards = [];
let _fcIdx   = 0;
let _fcCount = 10;
let _fcDiff  = "Medium";
let _fcRatings = {};
let _fcIsFlipped = false;

function openFcModal() {
  document.getElementById("fcOverlay").classList.remove("hidden");
  document.getElementById("fcSettings").style.display = "";
  document.getElementById("fcViewer").style.display   = "none";
  document.getElementById("fcDone").style.display     = "none";
}
function closeFcModal() {
  document.getElementById("fcOverlay").classList.add("hidden");
}

function setFcCount(btn, n) {
  _fcCount = n;
  document.querySelectorAll(".fc-count-btn").forEach(b => b.classList.remove("fc-count-active"));
  btn.classList.add("fc-count-active");
}
function setFcDiff(btn, val) {
  _fcDiff = val;
  document.querySelectorAll("#fcOverlay .diff-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}

async function generateFlashcards() {
  if (!sessionId) return;
  const btn = document.getElementById("fcGenBtn");
  const txt = document.getElementById("fcGenText");
  btn.disabled = true;
  txt.innerHTML = '<div class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:6px"></div>Generating…';

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("count", _fcCount);
  form.append("difficulty", _fcDiff);
  form.append("backend", currentBackend);
  form.append("retriever_mode", currentSearchMode);
  form.append("framework", currentFramework);

  try {
    const res  = await fetch("/flashcards", { method:"POST", body:form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed");

    _fcCards   = data.cards;
    _fcIdx     = 0;
    _fcRatings = {};
    _fcIsFlipped = false;

    document.getElementById("fcSettings").style.display = "none";
    document.getElementById("fcViewer").style.display   = "";
    _fcRender();

  } catch(err) {
    alert("Flashcard error: " + err.message);
  } finally {
    btn.disabled = false;
    txt.textContent = "⚡ Generate Flashcards";
  }
}

function _fcRender() {
  const total   = _fcCards.length;
  const done    = Object.keys(_fcRatings).length;
  const known   = Object.values(_fcRatings).filter(Boolean).length;
  const review  = done - known;
  const left    = total - done;

  // Card text
  const card = _fcCards[_fcIdx];
  document.getElementById("fcQ").textContent = card.question;
  document.getElementById("fcA").textContent = card.answer;

  // Reset flip
  _fcIsFlipped = false;
  document.getElementById("fcFlip").classList.remove("flipped");

  // Counter
  document.getElementById("fcCounter").textContent = `${_fcIdx + 1} / ${total}`;

  // Progress
  document.getElementById("fcProgFill").style.width = `${(done/total)*100}%`;
  document.getElementById("fcStatKnown").textContent  = `${known} ✓`;
  document.getElementById("fcStatReview").textContent = `${review} ✗`;
  document.getElementById("fcStatLeft").textContent   = `${left} left`;

  // Nav arrows
  document.getElementById("fcPrev").disabled = _fcIdx === 0;
  document.getElementById("fcNext").disabled = _fcIdx === total - 1;

  // Dots
  const dots = document.getElementById("fcDots");
  dots.innerHTML = _fcCards.map((_, i) => {
    let cls = "fc-dot";
    if (i === _fcIdx) cls += " cur";
    else if (_fcRatings[i] === true)  cls += " d-known";
    else if (_fcRatings[i] === false) cls += " d-rev";
    return `<div class="${cls}" onclick="fcGoto(${i})">${i+1}</div>`;
  }).join("");

  // All done?
  if (done === total) _fcShowDone();
}

function fcFlipCard() {
  _fcIsFlipped = !_fcIsFlipped;
  document.getElementById("fcFlip").classList.toggle("flipped", _fcIsFlipped);
}

function fcNav(dir) {
  const n = _fcIdx + dir;
  if (n >= 0 && n < _fcCards.length) { _fcIdx = n; _fcRender(); }
}
function fcGoto(i) { _fcIdx = i; _fcRender(); }

function fcRate(known) {
  _fcRatings[_fcIdx] = known;
  // Auto-advance to next unrated
  const next = _fcCards.findIndex((_, i) => i > _fcIdx && _fcRatings[i] === undefined);
  if (next !== -1) { _fcIdx = next; _fcRender(); return; }
  const first = _fcCards.findIndex((_, i) => _fcRatings[i] === undefined);
  if (first !== -1) { _fcIdx = first; _fcRender(); return; }
  _fcRender(); // triggers done
}

function _fcShowDone() {
  const total = _fcCards.length;
  const known = Object.values(_fcRatings).filter(Boolean).length;
  const pct   = Math.round((known/total)*100);
  document.getElementById("fcDoneEmoji").textContent = pct >= 80 ? "🎉" : pct >= 50 ? "👍" : "📚";
  document.getElementById("fcDoneSub").innerHTML =
    `You got <strong style="color:#34d399">${known}/${total}</strong> correct (${pct}%)`;
  document.getElementById("fcViewer").style.display = "none";
  document.getElementById("fcDone").style.display   = "";
}

function fcRestart() {
  _fcIdx = 0; _fcRatings = {}; _fcIsFlipped = false;
  document.getElementById("fcDone").style.display   = "none";
  document.getElementById("fcViewer").style.display = "";
  _fcRender();
}

// Keyboard shortcuts (only when overlay open)
document.addEventListener("keydown", e => {
  if (document.getElementById("fcOverlay").classList.contains("hidden")) return;
  if (e.key === "ArrowRight") fcNav(1);
  if (e.key === "ArrowLeft")  fcNav(-1);
  if (e.key === " " || e.key === "Enter") { e.preventDefault(); fcFlipCard(); }
  if (e.key === "1") fcRate(false);
  if (e.key === "2") fcRate(true);
  if (e.key === "Escape") closeFcModal();
});

// Export CSV
function exportCSV() {
  if (!_fcCards.length) return;
  const rows = [["Question","Answer"], ..._fcCards.map(c =>
    [`"${c.question.replace(/"/g,'""')}"`, `"${c.answer.replace(/"/g,'""')}"`]
  )];
  _dlText(rows.map(r=>r.join(",")).join("\n"), "flashcards.csv", "text/csv");
}

// Export Anki (tab-separated)
function exportAnki() {
  if (!_fcCards.length) return;
  _dlText(_fcCards.map(c=>`${c.question}\t${c.answer}`).join("\n"), "flashcards_anki.txt", "text/plain");
}

function _dlText(content, name, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([content],{type:mime}));
  a.download = name; a.click();
}


/* ══════════════════════════════════════════
   CITATION RENDERING
══════════════════════════════════════════ */

function renderCitations(citations, container) {
  if (!container) return;
  if (!citations || citations.length === 0) {
    container.classList.add("hidden");
    return;
  }
  let html = '<div class="citations-header" onclick="this.parentElement.classList.toggle(\'expanded\')">📎 Source Citations <span class="citations-toggle">▾</span></div>';
  html += '<div class="citations-list">';
  for (const c of citations) {
    const pages = (c.pages || []).join(", ");
    const score = (c.score * 100).toFixed(0);
    html += `
      <div class="citation-card">
        <div class="citation-meta">
          <span class="cite-pill">📄 Page ${escHtml(pages)}</span>
          <span class="cite-pill cite-chunk">Chunk #${c.chunk_id}</span>
          <span class="cite-pill cite-score">${score}% match</span>
        </div>
        <div class="citation-preview">${escHtml(c.text_preview)}</div>
      </div>`;
  }
  html += '</div>';
  container.innerHTML = html;
  container.classList.remove("hidden");
}


/* ══════════════════════════════════════════
   METRICS DASHBOARD
══════════════════════════════════════════ */

async function openMetrics() {
  const overlay = document.getElementById("metricsOverlay");
  overlay.classList.remove("hidden");
  const body = document.getElementById("metricsBody");
  body.innerHTML = '<div class="metrics-loading"><div class="spinner" style="width:20px;height:20px;border-width:2px"></div> Loading analytics…</div>';

  if (!sessionId) {
    body.innerHTML = '<div class="metrics-empty">Upload a PDF first to see analytics</div>';
    return;
  }

  try {
    const res = await fetch(`/metrics/${sessionId}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Failed to fetch metrics");

    let html = '<div class="metrics-grid">';

    // Metric cards
    html += metricCard("📄 Pages", data.page_count || 0, "");
    html += metricCard("🧩 Chunks", data.chunk_count || 0, "");
    html += metricCard("⚡ Ingestion", (data.ingestion_ms || 0).toFixed(0), "ms");
    html += metricCard("🔢 Queries", data.total_queries || 0, "");
    html += '</div>';

    // Summary stats
    const summary = data.summary || {};
    if (summary.total_ms) {
      html += '<div class="metrics-section-title">Latency Distribution</div>';
      html += '<div class="metrics-grid">';
      html += metricCard("Mean", summary.total_ms.mean.toFixed(0), "ms");
      html += metricCard("Median", summary.total_ms.median.toFixed(0), "ms");
      html += metricCard("P95", summary.total_ms.p95.toFixed(0), "ms");
      html += metricCard("Max", summary.total_ms.max.toFixed(0), "ms");
      html += '</div>';
    }

    if (summary.mean_similarity) {
      html += '<div class="metrics-section-title">Similarity Scores</div>';
      html += '<div class="metrics-grid">';
      html += metricCard("Mean", summary.mean_similarity.mean.toFixed(3), "");
      html += metricCard("Max", summary.mean_similarity.max.toFixed(3), "");
      html += metricCard("Min", summary.mean_similarity.min.toFixed(3), "");
      html += '</div>';
    }

    // Retriever mode breakdown
    if (summary.retriever_modes) {
      html += '<div class="metrics-section-title">Retriever Modes Used</div>';
      html += '<div class="metrics-tags">';
      for (const [mode, count] of Object.entries(summary.retriever_modes)) {
        html += `<span class="cite-pill">${mode}: ${count}</span>`;
      }
      html += '</div>';
    }

    // Recent queries table
    const recent = data.recent || [];
    if (recent.length > 0) {
      html += '<div class="metrics-section-title">Recent Queries</div>';
      html += '<div class="metrics-table-wrap"><table class="metrics-table">';
      html += '<tr><th>Op</th><th>Retrieval</th><th>LLM</th><th>Total</th><th>Sim</th><th>Mode</th></tr>';
      for (const r of recent.slice(-10)) {
        html += `<tr>
          <td>${r.operation}</td>
          <td>${r.retrieval_ms.toFixed(0)}ms</td>
          <td>${r.llm_ms.toFixed(0)}ms</td>
          <td>${r.total_ms.toFixed(0)}ms</td>
          <td>${r.mean_similarity.toFixed(3)}</td>
          <td>${r.retriever_mode}</td>
        </tr>`;
      }
      html += '</table></div>';
    }

    body.innerHTML = html;

  } catch (err) {
    body.innerHTML = `<div class="metrics-empty">Error: ${escHtml(err.message)}</div>`;
  }
}

function closeMetrics() {
  document.getElementById("metricsOverlay").classList.add("hidden");
}

function metricCard(label, value, unit) {
  return `<div class="metric-card">
    <div class="metric-value">${value}<span class="metric-unit">${unit}</span></div>
    <div class="metric-label">${label}</div>
  </div>`;
}
