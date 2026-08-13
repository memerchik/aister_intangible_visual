"use strict";

const MAX_BYTES = 12 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 2 * 1000 * 1000;
const ACCEPTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const LABEL_TONES = {
  "01_opishnyan_ceramics": "tone-opishnyan",
  "02_ornek": "tone-ornek",
  "03_bubnivka_ceramics": "tone-bubnivka",
  "04_petrykivka_painting": "tone-petrykivka",
  "05_kosiv_ceramics": "tone-kosiv",
};

const ui = {
  uploadView: document.querySelector("#uploadView"),
  loadingView: document.querySelector("#loadingView"),
  resultView: document.querySelector("#resultView"),
  fileInput: document.querySelector("#fileInput"),
  dropZone: document.querySelector("#dropZone"),
  selectedFile: document.querySelector("#selectedFile"),
  selectedPreview: document.querySelector("#selectedPreview"),
  selectedName: document.querySelector("#selectedName"),
  selectedMeta: document.querySelector("#selectedMeta"),
  removeFile: document.querySelector("#removeFile"),
  analyzeButton: document.querySelector("#analyzeButton"),
  uploadError: document.querySelector("#uploadError"),
  resultImage: document.querySelector("#resultImage"),
  motifOverlay: document.querySelector("#motifOverlay"),
  motifToggle: document.querySelector("#motifToggle"),
  matchTitle: document.querySelector("#matchTitle"),
  matchKind: document.querySelector("#matchKind"),
  topScore: document.querySelector("#topScore"),
  matchExplanation: document.querySelector("#matchExplanation"),
  heritageSource: document.querySelector("#heritageSource"),
  resultSeal: document.querySelector("#resultSeal"),
  processingTime: document.querySelector("#processingTime"),
  reviewNotice: document.querySelector("#reviewNotice"),
  reviewTitle: document.querySelector("#reviewTitle"),
  reviewReason: document.querySelector("#reviewReason"),
  rankingList: document.querySelector("#rankingList"),
  confirmTop: document.querySelector("#confirmTop"),
  chooseAnother: document.querySelector("#chooseAnother"),
  feedbackPanel: document.querySelector("#feedbackPanel"),
  closeFeedback: document.querySelector("#closeFeedback"),
  finalLabel: document.querySelector("#finalLabel"),
  humanSelectedName: document.querySelector("#humanSelectedName"),
  copySummary: document.querySelector("#copySummary"),
  contributionDetails: document.querySelector("#contributionDetails"),
  contributionNote: document.querySelector("#contributionNote"),
  contributionConsent: document.querySelector("#contributionConsent"),
  contributeButton: document.querySelector("#contributeButton"),
  contributionStatus: document.querySelector("#contributionStatus"),
  newAnalysisTop: document.querySelector("#newAnalysisTop"),
};

const state = {
  file: null,
  dataUrl: "",
  result: null,
  motifVisible: true,
  contributionsEnabled: true,
  predictionController: null,
  predictionSequence: 0,
};

async function loadPublicConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) return;
    const config = await response.json();
    state.contributionsEnabled = Boolean(config.contributions_enabled);
    ui.contributionDetails.hidden = !state.contributionsEnabled;
  } catch (_error) {
    // Local/offline previews retain the documented contribution controls.
  }
}

function showView(view) {
  ui.uploadView.hidden = view !== "upload";
  ui.loadingView.hidden = view !== "loading";
  ui.resultView.hidden = view !== "result";
  document.querySelector(".tradition-guide").hidden = view === "loading";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function scoreText(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function setError(message) {
  ui.uploadError.textContent = message;
  ui.uploadError.hidden = !message;
  if (message && !ui.uploadView.hidden) ui.uploadError.focus();
}

async function readApiResponse(response) {
  const responseText = await response.text();
  let payload = {};
  if (responseText) {
    try {
      payload = JSON.parse(responseText);
    } catch (_error) {
      payload = {};
    }
  }
  if (response.status === 429) {
    const retryAfter = response.headers.get("Retry-After");
    const waitText = retryAfter ? ` Try again in about ${retryAfter} seconds.` : " Try again shortly.";
    throw new Error(
      (typeof payload.error === "string"
        ? payload.error
        : "Another image is already being analyzed.") + waitText
    );
  }
  if (!response.ok) {
    throw new Error(
      typeof payload.error === "string"
        ? payload.error
        : `The analysis service returned error ${response.status}.`
    );
  }
  return payload;
}

function clearFile() {
  state.file = null;
  state.dataUrl = "";
  ui.fileInput.value = "";
  ui.selectedPreview.removeAttribute("src");
  ui.selectedFile.hidden = true;
  ui.dropZone.hidden = false;
  ui.analyzeButton.disabled = true;
  setError("");
}

function normalizedUpload(file, sourceUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.addEventListener("load", () => {
      const pixels = image.naturalWidth * image.naturalHeight;
      if (pixels <= MAX_IMAGE_PIXELS) {
        resolve({ dataUrl: sourceUrl, resized: false });
        return;
      }
      const scale = Math.sqrt(MAX_IMAGE_PIXELS / pixels);
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.floor(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.floor(image.naturalHeight * scale));
      const context = canvas.getContext("2d");
      if (!context) {
        reject(new Error("This browser could not prepare the image."));
        return;
      }
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      resolve({ dataUrl: canvas.toDataURL("image/jpeg", 0.9), resized: true });
    });
    image.addEventListener("error", () => reject(new Error("The selected image could not be decoded.")));
    image.src = sourceUrl;
  });
}

function selectFile(file) {
  setError("");
  if (!file) return;
  if (!ACCEPTED_TYPES.has(file.type)) {
    setError("Choose a JPEG, PNG, or WebP image.");
    return;
  }
  if (file.size > MAX_BYTES) {
    setError("This image is larger than the 12 MB limit.");
    return;
  }
  const reader = new FileReader();
  reader.addEventListener("load", async () => {
    try {
      const prepared = await normalizedUpload(file, String(reader.result));
      state.file = file;
      state.dataUrl = prepared.dataUrl;
      ui.selectedPreview.src = state.dataUrl;
      ui.selectedName.textContent = file.name;
      const resizeNote = prepared.resized ? " · optimized for analysis" : "";
      ui.selectedMeta.textContent = `${formatBytes(file.size)} · ${file.type.replace("image/", "").toUpperCase()}${resizeNote}`;
      ui.dropZone.hidden = true;
      ui.selectedFile.hidden = false;
      ui.analyzeButton.disabled = false;
    } catch (error) {
      setError(error instanceof Error ? error.message : "The selected image could not be prepared.");
    }
  });
  reader.addEventListener("error", () => setError("The selected image could not be read."));
  reader.readAsDataURL(file);
}

async function analyze() {
  if (!state.file || !state.dataUrl) return;
  if (state.predictionController) state.predictionController.abort();
  const controller = new AbortController();
  const sequence = ++state.predictionSequence;
  state.predictionController = controller;
  ui.analyzeButton.disabled = true;
  showView("loading");
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: state.dataUrl, file_name: state.file.name }),
      signal: controller.signal,
    });
    const payload = await readApiResponse(response);
    if (sequence !== state.predictionSequence) return;
    state.result = payload;
    renderResult(payload);
    showView("result");
  } catch (error) {
    if (sequence !== state.predictionSequence) return;
    showView("upload");
    const message = error instanceof DOMException && error.name === "AbortError"
      ? "The previous analysis was cancelled. You can try another image."
      : (error instanceof Error ? error.message : "The analysis could not be completed.");
    setError(message);
  } finally {
    if (sequence === state.predictionSequence) {
      state.predictionController = null;
      ui.analyzeButton.disabled = !state.file;
    }
  }
}

function renderMotifRegions(regions) {
  ui.motifOverlay.replaceChildren();
  const namespace = "http://www.w3.org/2000/svg";
  regions.forEach((region) => {
    const rectangle = document.createElementNS(namespace, "rect");
    rectangle.setAttribute("x", String(Number(region.x) * 1000));
    rectangle.setAttribute("y", String(Number(region.y) * 1000));
    rectangle.setAttribute("width", String(Number(region.width) * 1000));
    rectangle.setAttribute("height", String(Number(region.height) * 1000));
    rectangle.setAttribute("rx", "5");
    rectangle.setAttribute("aria-label", `Motif region ${region.rank}`);
    ui.motifOverlay.append(rectangle);
  });
}

function renderRanking(ranking) {
  ui.rankingList.replaceChildren();
  ranking.forEach((item) => {
    const row = document.createElement("li");
    row.className = `ranking-item ${LABEL_TONES[item.label]}`;

    const rank = document.createElement("span");
    rank.textContent = String(item.rank).padStart(2, "0");

    const name = document.createElement("div");
    name.className = "ranking-name";
    const strong = document.createElement("strong");
    strong.textContent = item.name;
    const small = document.createElement("small");
    small.textContent = item.kind;
    name.append(strong, small);

    const barWrap = document.createElement("div");
    barWrap.className = "ranking-bar";
    const bar = document.createElement("progress");
    bar.max = 1;
    bar.value = Number(item.score);
    bar.setAttribute("aria-label", `${item.name} ranking score ${scoreText(item.score)}`);
    barWrap.append(bar);

    const score = document.createElement("span");
    score.className = "ranking-score";
    score.textContent = scoreText(item.score);
    row.append(rank, name, barWrap, score);
    ui.rankingList.append(row);
  });
}

function populateFinalChoices(ranking, selectedLabel) {
  ui.finalLabel.replaceChildren();
  ranking.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.label;
    option.textContent = item.name;
    option.selected = item.label === selectedLabel;
    ui.finalLabel.append(option);
  });
  updateHumanChoice();
}

function renderResult(result) {
  const top = result.top_match;
  ui.resultImage.src = state.dataUrl;
  ui.matchTitle.textContent = top.name;
  ui.matchKind.textContent = top.kind;
  ui.topScore.textContent = scoreText(top.score);
  ui.matchExplanation.textContent = top.look_for;
  ui.processingTime.textContent = `${(Number(result.processing_ms) / 1000).toFixed(1)}s`;
  ui.resultSeal.className = `result-seal ${LABEL_TONES[top.label]}`;
  ui.heritageSource.hidden = !top.source_url;
  if (top.source_url) ui.heritageSource.href = top.source_url;
  ui.reviewNotice.classList.toggle("is-boundary", Boolean(result.review_recommended));
  ui.reviewTitle.textContent = result.review_recommended ? "Compare this ceramic boundary" : "Treat this as an assisted match";
  ui.reviewReason.textContent = result.review_reason;
  state.motifVisible = true;
  ui.motifOverlay.removeAttribute("hidden");
  ui.motifToggle.setAttribute("aria-pressed", "true");
  ui.motifToggle.lastChild.textContent = " Motif focus on";
  renderMotifRegions(result.motif_regions || []);
  renderRanking(result.ranking);
  populateFinalChoices(result.ranking, top.label);
  ui.feedbackPanel.hidden = true;
  ui.contributionConsent.checked = false;
  ui.contributeButton.disabled = true;
  ui.contributionNote.value = "";
  ui.contributionStatus.textContent = "";
}

function updateHumanChoice() {
  if (!state.result) return;
  const selected = state.result.ranking.find((item) => item.label === ui.finalLabel.value);
  ui.humanSelectedName.textContent = selected ? selected.name : "";
}

function openFeedback(useTopMatch) {
  if (!state.result) return;
  if (useTopMatch) ui.finalLabel.value = state.result.top_match.label;
  updateHumanChoice();
  ui.feedbackPanel.hidden = false;
  ui.feedbackPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  window.setTimeout(() => ui.finalLabel.focus(), 450);
}

function resultSummary() {
  if (!state.result) return "";
  const selected = state.result.ranking.find((item) => item.label === ui.finalLabel.value);
  const ranking = state.result.ranking
    .map((item) => `${item.rank}. ${item.name} — ${scoreText(item.score)}`)
    .join("\n");
  return [
    "AISTER Ornament Lens · assisted preview v0.5",
    `Human-selected result: ${selected ? selected.name : "Not selected"}`,
    `Model best visual match: ${state.result.top_match.name}`,
    "",
    "Model ranking (uncalibrated scores):",
    ranking,
    "",
    "This is an assisted visual suggestion, not an official cultural attribution.",
  ].join("\n");
}

async function copySummary() {
  try {
    await navigator.clipboard.writeText(resultSummary());
    ui.copySummary.lastChild.textContent = " Copied";
    window.setTimeout(() => { ui.copySummary.lastChild.textContent = " Copy result summary"; }, 1800);
  } catch (_error) {
    ui.contributionStatus.textContent = "Copying is unavailable in this browser. Select and copy the result manually.";
  }
}

async function contribute() {
  if (!state.result || !ui.contributionConsent.checked) return;
  ui.contributeButton.disabled = true;
  ui.contributionStatus.textContent = "Saving to the expert-review queue…";
  try {
    const response = await fetch("/api/contribute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prediction_id: state.result.prediction_id,
        final_label: ui.finalLabel.value,
        note: ui.contributionNote.value,
        consent: true,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "The contribution could not be saved.");
    ui.contributionStatus.textContent = `${payload.message} Reference: ${payload.contribution_id}`;
    ui.contributionConsent.disabled = true;
    ui.contributionNote.disabled = true;
  } catch (error) {
    ui.contributionStatus.textContent = error instanceof Error ? error.message : "The contribution could not be saved.";
    ui.contributeButton.disabled = !ui.contributionConsent.checked;
  }
}

function resetApplication() {
  if (state.predictionController) state.predictionController.abort();
  state.predictionController = null;
  state.predictionSequence += 1;
  state.result = null;
  clearFile();
  ui.feedbackPanel.hidden = true;
  ui.contributionConsent.disabled = false;
  ui.contributionNote.disabled = false;
  showView("upload");
  ui.dropZone.focus();
}

ui.fileInput.addEventListener("change", () => selectFile(ui.fileInput.files[0]));
ui.removeFile.addEventListener("click", clearFile);
ui.analyzeButton.addEventListener("click", analyze);
ui.newAnalysisTop.addEventListener("click", resetApplication);
ui.confirmTop.addEventListener("click", () => openFeedback(true));
ui.chooseAnother.addEventListener("click", () => openFeedback(false));
ui.closeFeedback.addEventListener("click", () => { ui.feedbackPanel.hidden = true; });
ui.finalLabel.addEventListener("change", updateHumanChoice);
ui.copySummary.addEventListener("click", copySummary);
ui.contributionConsent.addEventListener("change", () => {
  ui.contributeButton.disabled = !ui.contributionConsent.checked;
});
ui.contributeButton.addEventListener("click", contribute);
ui.motifToggle.addEventListener("click", () => {
  state.motifVisible = !state.motifVisible;
  ui.motifOverlay.toggleAttribute("hidden", !state.motifVisible);
  ui.motifToggle.setAttribute("aria-pressed", String(state.motifVisible));
  ui.motifToggle.lastChild.textContent = state.motifVisible ? " Motif focus on" : " Motif focus off";
});

["dragenter", "dragover"].forEach((eventName) => {
  ui.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    ui.dropZone.classList.add("is-dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  ui.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    ui.dropZone.classList.remove("is-dragging");
  });
});
ui.dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
ui.dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    ui.fileInput.click();
  }
});
document.addEventListener("paste", (event) => {
  if (!ui.uploadView.hidden) {
    const imageItem = Array.from(event.clipboardData?.items || []).find((item) => item.type.startsWith("image/"));
    if (imageItem) selectFile(imageItem.getAsFile());
  }
});

loadPublicConfig();
