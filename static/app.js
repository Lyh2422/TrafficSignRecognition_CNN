const $ = (selector) => document.querySelector(selector);

const state = { file: null, previewUrl: null };
const elements = {
  fileInput: $("#fileInput"), dropZone: $("#dropZone"), previewImage: $("#previewImage"),
  fileName: $("#fileName"), predictButton: $("#predictButton"), error: $("#errorMessage"),
  emptyResult: $("#emptyResult"), resultContent: $("#resultContent"), resultState: $("#resultState"),
  systemStatus: $("#systemStatus"), modeBadge: $("#modeBadge"), historyBody: $("#historyBody"),
  historyEmpty: $("#historyEmpty"), clearHistory: $("#clearHistory"),
};

function setError(message = "") { elements.error.textContent = message; }

function chooseFile(file) {
  setError();
  if (!file || !file.type.startsWith("image/")) {
    setError("请选择有效的图片文件。");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    setError("图片不能超过 8 MB。");
    return;
  }
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.file = file;
  state.previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = state.previewUrl;
  elements.fileName.textContent = file.name;
  elements.dropZone.classList.add("has-image");
  elements.predictButton.disabled = false;
  elements.resultState.textContent = "图像已就绪";
}

elements.fileInput.addEventListener("change", (event) => chooseFile(event.target.files[0]));
["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault(); elements.dropZone.classList.add("is-dragging");
}));
["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => {
  event.preventDefault(); elements.dropZone.classList.remove("is-dragging");
}));
elements.dropZone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

elements.predictButton.addEventListener("click", async () => {
  if (!state.file) return;
  setError();
  elements.predictButton.disabled = true;
  elements.predictButton.querySelector("span").textContent = "识别中";
  elements.dropZone.classList.add("scanning");
  elements.resultState.textContent = "正在分析图像";
  const body = new FormData();
  body.append("file", state.file);
  try {
    const response = await fetch("/api/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "识别请求失败。");
    renderResult(data);
    await loadHistory();
  } catch (error) {
    setError(error.message);
    elements.resultState.textContent = "识别失败";
  } finally {
    elements.predictButton.disabled = false;
    elements.predictButton.querySelector("span").textContent = "重新识别";
    elements.dropZone.classList.remove("scanning");
  }
});

function renderResult(data) {
  const confidence = data.prediction.confidence * 100;
  elements.emptyResult.hidden = true;
  elements.resultContent.hidden = false;
  elements.resultState.textContent = `记录 #${data.record_id}`;
  $("#classCode").textContent = String(data.prediction.id).padStart(2, "0");
  $("#className").textContent = data.prediction.name_zh;
  $("#classNameEn").textContent = data.prediction.name_en;
  $("#confidenceValue").textContent = `${confidence.toFixed(1)}%`;
  $("#confidenceRing").style.setProperty("--progress", `${confidence * 3.6}deg`);
  $("#latencyValue").textContent = `${data.inference_ms.toFixed(2)} ms`;
  $("#imageSize").textContent = `${data.image.width} × ${data.image.height}`;
  $("#resultMode").textContent = data.mode === "cnn" ? "CNN 模型" : "演示模式";

  $("#topList").innerHTML = data.top3.map((item, index) => `
    <div class="top-item">
      <span>0${index + 1}</span><span>${escapeHtml(item.name_zh)}</span>
      <span class="bar"><i style="width:${Math.max(2, item.confidence * 100)}%"></i></span>
      <strong>${(item.confidence * 100).toFixed(1)}%</strong>
    </div>`).join("");
  const warning = $("#modeWarning");
  warning.hidden = !data.warning;
  warning.textContent = data.warning ? `注意：${data.warning}` : "";
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    elements.systemStatus.textContent = data.mode === "cnn" ? "模型已就绪" : "服务运行正常";
    elements.modeBadge.textContent = data.mode === "cnn" ? "CNN ONLINE" : "DEMO MODE";
    elements.modeBadge.title = data.model_message || "";
  } catch {
    elements.systemStatus.textContent = "后端未连接";
    elements.modeBadge.textContent = "OFFLINE";
  }
}

async function loadHistory() {
  const response = await fetch("/api/history?limit=12");
  const data = await response.json();
  elements.historyEmpty.hidden = data.items.length > 0;
  elements.historyBody.innerHTML = data.items.map((item) => `
    <tr>
      <td title="${escapeHtml(item.filename)}">${escapeHtml(shorten(item.filename, 24))}</td>
      <td>${escapeHtml(item.name_zh)} <small>#${String(item.class_id).padStart(2, "0")}</small></td>
      <td>${(item.confidence * 100).toFixed(1)}%</td>
      <td>${item.inference_ms.toFixed(2)} ms</td>
      <td><span class="mode-tag ${item.mode}">${item.mode === "cnn" ? "CNN" : "DEMO"}</span></td>
      <td>${escapeHtml(item.created_at)}</td>
    </tr>`).join("");
}

elements.clearHistory.addEventListener("click", async () => {
  if (!confirm("确定清空本机上的全部识别记录吗？")) return;
  await fetch("/api/history", { method: "DELETE" });
  await loadHistory();
});

function shorten(value, length) { return value.length > length ? `${value.slice(0, length - 1)}…` : value; }
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

loadHealth();
loadHistory().catch(() => {});
