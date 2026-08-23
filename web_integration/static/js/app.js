/* app.js — Resume Ranking System frontend */

"use strict";

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".panel").forEach(p => p.classList.add("hidden"));
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.remove("hidden");
  });
});

// ── JD type toggle (text / file) ──────────────────────────────────────────────
function setupToggle(radioName, textId, fileWrapId, fileNameId) {
  const radios = document.querySelectorAll(`input[name="${radioName}"]`);
  const textEl = document.getElementById(textId);
  const fileWrap = document.getElementById(`${fileWrapId}-wrap`);
  const labels = document.querySelectorAll(`[data-for="${textId}"], [data-for="${fileWrapId}"]`);

  radios.forEach(r => {
    r.addEventListener("change", () => {
      const isFile = r.value === "file";
      textEl.classList.toggle("hidden", isFile);
      fileWrap.classList.toggle("hidden", !isFile);
      labels.forEach(l => l.classList.remove("active"));
      r.parentElement.classList.add("active");
    });
  });

  // File name display
  const fileInput = document.getElementById(fileWrapId);
  if (fileInput) {
    fileInput.addEventListener("change", () => {
      const nameEl = document.getElementById(`${fileWrapId}-name`);
      if (nameEl) nameEl.textContent = fileInput.files[0]?.name || "No file chosen";
    });
  }
}

setupToggle("jd-type-s", "jd-text-s", "jd-file-s", "jd-file-s-name");
setupToggle("jd-type-l", "jd-text-l", "jd-file-l", "jd-file-l-name");

// ── Resume file chips ─────────────────────────────────────────────────────────
const resumeInput = document.getElementById("resume-files");
const chipsEl     = document.getElementById("resume-chips");

resumeInput?.addEventListener("change", () => {
  chipsEl.innerHTML = "";
  const files = Array.from(resumeInput.files);
  if (files.length > 50) {
    showAlert(chipsEl, "Maximum 50 resume files allowed.", "warn");
    return;
  }
  files.forEach(f => {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.textContent = f.name;
    chip.title = f.name;
    chipsEl.appendChild(chip);
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function showAlert(container, message, type = "error") {
  const div = document.createElement("div");
  div.className = `alert alert-${type}`;
  div.textContent = message;
  container.prepend(div);
}

function pct(val) {
  return `${(val * 100).toFixed(1)}%`;
}

function setBusy(btn, busy) {
  btn.disabled = busy;
  btn.querySelector(".btn-text").classList.toggle("hidden", busy);
  btn.querySelector(".btn-spinner").classList.toggle("hidden", !busy);
}

function rankBadgeClass(rank) {
  if (rank === 1) return "top1";
  if (rank === 2) return "top2";
  if (rank === 3) return "top3";
  return "";
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(container, data) {
  container.innerHTML = "";
  container.classList.remove("hidden");

  // Header
  const isLive   = data.mode === "live";
  const headerEl = document.createElement("div");
  headerEl.className = "results-header";
  headerEl.innerHTML = `
    <div>
      <div class="results-title">Top ${data.results.length} Ranked Resumes</div>
      <div class="results-meta">
        ${isLive
          ? `${data.successfully_extracted} of ${data.total_uploaded} resumes processed`
          : `Searched ${1429} stored resumes`}
      </div>
    </div>
    <div class="results-pills">
      <span class="pill pill-hybrid">Shortlist</span>
    </div>
  `;
  container.appendChild(headerEl);

  // Warn on failed extractions
  if (isLive && data.failed_extractions?.length) {
    const warnDiv = document.createElement("div");
    warnDiv.className = "alert alert-warn";
    const names = data.failed_extractions.map(f => f.filename).join(", ");
    warnDiv.textContent = `⚠ ${data.failed_extractions.length} file(s) could not be extracted: ${names}`;
    container.appendChild(warnDiv);
  }

  if (!data.results.length) {
    const noneEl = document.createElement("div");
    noneEl.className = "alert alert-info";
    noneEl.textContent = "No results returned.";
    container.appendChild(noneEl);
    return;
  }

  // Cards
  const list = document.createElement("div");
  list.className = "result-list";

  data.results.forEach((r, idx) => {
    const delay = idx * 40;
    const label = isLive
      ? (r.filename || `Resume ${r.resume_id}`)
      : `Resume #${r.resume_id}`;

    const card = document.createElement("div");
    card.className = "result-card";
    card.style.animationDelay = `${delay}ms`;

    card.innerHTML = `
      <div class="rank-badge ${rankBadgeClass(r.rank)}">#${r.rank}</div>
      <div class="card-body">
        <div class="card-title">
          ${escHtml(label)}
          ${r.category ? `<span class="category-tag">${escHtml(r.category)}</span>` : ""}
        </div>
        ${r.preview ? `<div class="card-preview">${escHtml(r.preview)}</div>` : ""}
        <div class="score-bars">
          ${scoreRow("Match Score",   r.hybrid_score, "bar-hybrid")}
        </div>
      </div>
    `;
    list.appendChild(card);
  });

  container.appendChild(list);

  // Animate bars after paint
  requestAnimationFrame(() => {
    container.querySelectorAll(".bar-fill").forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  });
}

function scoreRow(label, val, cls) {
  const w = `${Math.min(val * 100, 100).toFixed(1)}%`;
  return `
    <div class="score-row">
      <span class="score-label">${label}</span>
      <div class="bar-track">
        <div class="bar-fill ${cls}" style="width:0" data-width="${w}"></div>
      </div>
      <span class="score-val">${pct(val)}</span>
    </div>
  `;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Form submit: Mode 1 — Stored ──────────────────────────────────────────────
document.getElementById("form-stored").addEventListener("submit", async e => {
  e.preventDefault();
  const btn       = document.getElementById("btn-stored");
  const resultsEl = document.getElementById("results-stored");
  resultsEl.innerHTML = "";
  resultsEl.classList.add("hidden");
  setBusy(btn, true);

  const form = e.target;
  const jdType = form.querySelector("input[name='jd-type-s']:checked")?.value || "text";
  const fd = new FormData();

  if (jdType === "text") {
    const txt = form.querySelector("#jd-text-s").value.trim();
    if (!txt) { showAlert(resultsEl, "Please enter a job description.", "error"); resultsEl.classList.remove("hidden"); setBusy(btn, false); return; }
    fd.append("jd_text", txt);
  } else {
    const file = form.querySelector("#jd-file-s").files[0];
    if (!file) { showAlert(resultsEl, "Please choose a JD PDF file.", "error"); resultsEl.classList.remove("hidden"); setBusy(btn, false); return; }
    fd.append("jd_file", file);
  }
  fd.append("top_k", form.querySelector("input[name='top_k']").value);

  try {
    const resp = await fetch("/api/rank/stored", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) { showAlert(resultsEl, data.error || "Server error.", "error"); resultsEl.classList.remove("hidden"); }
    else renderResults(resultsEl, data);
  } catch (err) {
    showAlert(resultsEl, `Network error: ${err.message}`, "error");
    resultsEl.classList.remove("hidden");
  } finally {
    setBusy(btn, false);
  }
});

// ── Form submit: Mode 2 — Live ────────────────────────────────────────────────
document.getElementById("form-live").addEventListener("submit", async e => {
  e.preventDefault();
  const btn       = document.getElementById("btn-live");
  const resultsEl = document.getElementById("results-live");
  resultsEl.innerHTML = "";
  resultsEl.classList.add("hidden");
  setBusy(btn, true);

  const form = e.target;
  const jdType = form.querySelector("input[name='jd-type-l']:checked")?.value || "text";
  const fd = new FormData();

  if (jdType === "text") {
    const txt = form.querySelector("#jd-text-l").value.trim();
    if (!txt) { showAlert(resultsEl, "Please enter a job description.", "error"); resultsEl.classList.remove("hidden"); setBusy(btn, false); return; }
    fd.append("jd_text", txt);
  } else {
    const file = form.querySelector("#jd-file-l").files[0];
    if (!file) { showAlert(resultsEl, "Please choose a JD PDF file.", "error"); resultsEl.classList.remove("hidden"); setBusy(btn, false); return; }
    fd.append("jd_file", file);
  }

  const resumeFiles = document.getElementById("resume-files").files;
  if (!resumeFiles.length) {
    showAlert(resultsEl, "Please upload at least one resume PDF.", "error");
    resultsEl.classList.remove("hidden");
    setBusy(btn, false);
    return;
  }
  Array.from(resumeFiles).forEach(f => fd.append("resume_files", f));
  fd.append("top_k", form.querySelector("input[name='top_k']").value);

  try {
    const resp = await fetch("/api/rank/live", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) { showAlert(resultsEl, data.error || "Server error.", "error"); resultsEl.classList.remove("hidden"); }
    else renderResults(resultsEl, data);
  } catch (err) {
    showAlert(resultsEl, `Network error: ${err.message}`, "error");
    resultsEl.classList.remove("hidden");
  } finally {
    setBusy(btn, false);
  }
});
