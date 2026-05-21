/**
 * Model Manager — vanilla JS SPA
 * All fetch calls use relative paths (proxy-safe).
 */

/* ================================================================
   Globals
   ================================================================ */
let autoRefreshTimer = null;
let autoRefreshEnabled = false;
let chatMessages = [];          // [{role, content}]
let activeModel = "";           // set from Models tab "Use" button

/* ================================================================
   Tab routing
   ================================================================ */
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + target).classList.add("active");
    });
  });

  document.querySelectorAll(".sub-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const group = btn.dataset.group;
      const target = btn.dataset.subtab;
      document.querySelectorAll(`.sub-tab-btn[data-group="${group}"]`)
        .forEach(b => b.classList.remove("active"));
      document.querySelectorAll(`.sub-panel[data-group="${group}"]`)
        .forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.querySelector(`.sub-panel[data-group="${group}"][data-subtab="${target}"]`)
        .classList.add("active");
    });
  });
}

/* ================================================================
   Utility helpers
   ================================================================ */
async function apiFetch(path, opts = {}) {
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`HTTP ${resp.status}: ${text}`);
  }
  return resp.json();
}

function fmt(n) { return Number(n).toLocaleString(); }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function setStatus(elId, msg, cls = "") {
  const el = document.getElementById(elId);
  if (!el) return;
  el.textContent = msg;
  el.className = "status-msg " + cls;
}

function makeBadge(type, label) {
  const span = document.createElement("span");
  span.className = `badge badge-${type}`;
  span.textContent = label;
  return span;
}

/* ================================================================
   Dashboard
   ================================================================ */
async function loadDashboard() {
  try {
    const [statusData, gpuData] = await Promise.all([
      apiFetch("/api/status"),
      apiFetch("/api/gpu"),
    ]);
    renderContainerCards(statusData);
    renderHeaderVram(gpuData.gpus);
  } catch (err) {
    console.error("Dashboard load error:", err);
  }
}

function renderContainerCards(data) {
  const NAMES = {
    "local-ollama":    "Ollama",
    "local-nim-llama": "NIM Llama",
    "local-hf":        "HuggingFace TGI",
  };

  const row = document.getElementById("container-cards");
  row.innerHTML = "";

  for (const [name, info] of Object.entries(data.containers)) {
    const card = document.createElement("div");
    card.className = "card";

    const label = NAMES[name] || name;
    const badgeType = info.running ? "running" : "stopped";
    const badgeLabel = info.running ? "Running" : "Stopped";
    const modelText = info.models.length
      ? info.models.slice(0, 2).join(", ")
      : (info.running ? "No models loaded" : "—");

    card.innerHTML = `
      <div class="card-title">${label}</div>
      <div class="mb-8" id="badge-${name.replace(/-/g,"_")}"></div>
      <div class="text-muted text-sm">${modelText}</div>
    `;
    const badgeEl = card.querySelector(`#badge-${name.replace(/-/g,"_")}`);
    badgeEl.appendChild(makeBadge(badgeType, badgeLabel));
    row.appendChild(card);
  }

  // Chain server card
  const chainCard = document.createElement("div");
  chainCard.className = "card";
  const chainHealthy = data.chain_server.healthy;
  chainCard.innerHTML = `
    <div class="card-title">Chain Server</div>
    <div id="chain-badge"></div>
    <div class="text-muted text-sm mt-12">localhost:8000</div>
  `;
  chainCard.querySelector("#chain-badge").appendChild(
    makeBadge(chainHealthy ? "healthy" : "unhealthy", chainHealthy ? "Healthy" : "Unreachable")
  );
  row.appendChild(chainCard);
}

function renderHeaderVram(gpus) {
  const bar    = document.getElementById("header-vram-bar");
  const label  = document.getElementById("header-vram-label");
  if (!gpus || !gpus.length) {
    if (label) label.textContent = "No GPU";
    return;
  }
  // Sum across GPUs
  const total = gpus.reduce((s, g) => s + g.total_mb, 0);
  const used  = gpus.reduce((s, g) => s + g.used_mb,  0);
  const pct   = total > 0 ? Math.round((used / total) * 100) : 0;

  if (bar) {
    bar.style.width = pct + "%";
    bar.classList.toggle("high", pct > 85);
  }
  if (label) {
    label.textContent = `VRAM ${fmt(used)}/${fmt(total)} MB (${pct}%)`;
  }
}

/* Auto-refresh */
function toggleAutoRefresh() {
  const btn = document.getElementById("refresh-toggle");
  autoRefreshEnabled = !autoRefreshEnabled;
  if (autoRefreshEnabled) {
    btn.classList.add("active");
    btn.textContent = "⏸ Auto-refresh ON";
    autoRefreshTimer = setInterval(loadDashboard, 10000);
  } else {
    btn.classList.remove("active");
    btn.textContent = "▶ Auto-refresh";
    clearInterval(autoRefreshTimer);
  }
}

/* ================================================================
   Models — Ollama
   ================================================================ */
async function loadOllamaModels() {
  setStatus("ollama-status", "Loading…");
  try {
    const data = await apiFetch("/api/ollama/models");
    renderOllamaTable(data.models);
    setStatus("ollama-status", data.error || "");
  } catch (err) {
    setStatus("ollama-status", "Error: " + err.message, "text-danger");
  }
}

function renderOllamaTable(models) {
  const tbody = document.querySelector("#ollama-table tbody");
  tbody.innerHTML = "";
  if (!models.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="text-muted">No models found</td></tr>';
    return;
  }
  for (const m of models) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${m.name}</td>
      <td>${m.size}</td>
      <td>${m.modified}</td>
      <td>
        <div class="flex gap-8">
          <button class="btn btn-secondary" onclick="useModel('${m.name}')">▶ Use</button>
          <button class="btn btn-danger"    onclick="deleteOllamaModel('${m.name}')">🗑</button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  }
}

function useModel(name) {
  activeModel = name;
  document.getElementById("chat-model").value = name;
  // Switch to chat tab
  document.querySelector('.tab-btn[data-tab="chat"]').click();
}

async function pullOllamaModel() {
  const input = document.getElementById("pull-input");
  const model = input.value.trim();
  if (!model) return;
  setStatus("ollama-status", `Pulling ${model}…`, "text-accent");
  try {
    await apiFetch("/api/ollama/pull", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model}),
    });
    setStatus("ollama-status", `Pull started for ${model}. Refresh in a moment.`, "text-success");
    input.value = "";
  } catch (err) {
    setStatus("ollama-status", "Pull error: " + err.message, "text-danger");
  }
}

async function deleteOllamaModel(name) {
  if (!confirm(`Delete model "${name}"?`)) return;
  setStatus("ollama-status", `Deleting ${name}…`);
  try {
    const data = await apiFetch("/api/ollama/delete", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model: name}),
    });
    setStatus("ollama-status", data.status === "deleted" ? `Deleted ${name}` : data.output);
    loadOllamaModels();
  } catch (err) {
    setStatus("ollama-status", "Delete error: " + err.message, "text-danger");
  }
}

/* ================================================================
   Models — HuggingFace
   ================================================================ */
async function loadHFCache() {
  setStatus("hf-status", "Loading…");
  try {
    const data = await apiFetch("/api/hf/cache");
    renderHFList(data.models);
    setStatus("hf-status", data.error || (data.models.length === 0 ? "Cache is empty or local-hf is not running." : ""));
  } catch (err) {
    setStatus("hf-status", "Error: " + err.message, "text-danger");
  }
}

function renderHFList(models) {
  const ul = document.getElementById("hf-list");
  ul.innerHTML = "";
  if (!models.length) {
    ul.innerHTML = '<li class="text-muted">No cached models found.</li>';
    return;
  }
  for (const m of models) {
    const li = document.createElement("li");
    li.className = "mb-8";
    li.textContent = m;
    ul.appendChild(li);
  }
}

/* ================================================================
   Chat
   ================================================================ */
function initChat() {
  const textarea = document.getElementById("chat-textarea");
  textarea.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendChat();
    }
  });

  document.getElementById("params-toggle").addEventListener("click", () => {
    const panel = document.getElementById("params-panel");
    panel.classList.toggle("open");
    document.getElementById("params-toggle").textContent =
      panel.classList.contains("open") ? "▲ Parameters" : "▼ Parameters";
  });

  document.getElementById("temp-slider").addEventListener("input", e => {
    document.getElementById("temp-val").textContent = e.target.value;
  });
  document.getElementById("tokens-slider").addEventListener("input", e => {
    document.getElementById("tokens-val").textContent = e.target.value;
  });
}

async function fetchChatModels() {
  const host  = document.getElementById("chat-host").value.trim() || "local-ollama";
  const port  = document.getElementById("chat-port").value.trim() || "8000";
  setStatus("chat-status", "Fetching models…");
  try {
    const resp = await fetch(`/api/chat`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        host, port: parseInt(port),
        model: "__list__",
        messages: [],
      }),
    });
    // Actually call /v1/models via dedicated path
    const data = await apiFetch(`/api/status`);
    // Get models from container matching host name
    const containers = data.containers || {};
    const container = containers[host] || {};
    const models = container.models || [];
    const select = document.getElementById("chat-model");
    select.innerHTML = "";
    if (models.length) {
      models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m; opt.textContent = m;
        select.appendChild(opt);
      });
      setStatus("chat-status", `${models.length} model(s) loaded`);
    } else {
      select.innerHTML = '<option value="">-- no models --</option>';
      setStatus("chat-status", "No models found for " + host);
    }
  } catch (err) {
    setStatus("chat-status", "Error: " + err.message, "text-danger");
  }
}

function clearChat() {
  chatMessages = [];
  document.getElementById("messages").innerHTML = "";
  setStatus("chat-status", "Chat cleared.");
}

function addMessageBubble(role, content) {
  const div = document.createElement("div");
  div.className = `msg msg-${role}`;
  div.textContent = content;
  document.getElementById("messages").appendChild(div);
  document.getElementById("messages").scrollTop = 99999;
  return div;
}

async function sendChat() {
  const textarea = document.getElementById("chat-textarea");
  const content  = textarea.value.trim();
  if (!content) return;

  const host  = document.getElementById("chat-host").value.trim() || "local-ollama";
  const port  = parseInt(document.getElementById("chat-port").value.trim() || "8000");
  const model = document.getElementById("chat-model").value.trim();
  if (!model) { setStatus("chat-status", "Please select a model.", "text-danger"); return; }

  const temperature = parseFloat(document.getElementById("temp-slider").value);
  const max_tokens  = parseInt(document.getElementById("tokens-slider").value);

  textarea.value = "";
  chatMessages.push({role: "user", content});
  addMessageBubble("user", content);

  // Streaming assistant bubble
  const assistantDiv = document.createElement("div");
  assistantDiv.className = "msg msg-assistant streaming";
  document.getElementById("messages").appendChild(assistantDiv);
  document.getElementById("messages").scrollTop = 99999;

  let fullText = "";

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({host, port, model, messages: chatMessages, temperature, max_tokens}),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, {stream: true});
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") break;

        try {
          const parsed = JSON.parse(payload);
          if (parsed.error) {
            assistantDiv.textContent = "Error: " + parsed.error;
            assistantDiv.classList.remove("streaming");
            return;
          }
          const delta = parsed.choices?.[0]?.delta?.content || "";
          fullText += delta;
          assistantDiv.textContent = fullText;
          document.getElementById("messages").scrollTop = 99999;
        } catch (_) {}
      }
    }
  } catch (err) {
    assistantDiv.textContent = "Stream error: " + err.message;
  }

  assistantDiv.classList.remove("streaming");
  chatMessages.push({role: "assistant", content: fullText});
}

/* ================================================================
   System — GPU + logs
   ================================================================ */
async function loadGPUInfo() {
  try {
    const data = await apiFetch("/api/gpu");
    renderGPUTable(data.gpus);
    renderAdvisor(data.gpus);
  } catch (err) {
    document.getElementById("gpu-table-body").innerHTML =
      `<tr><td colspan="6" class="text-danger">Error: ${err.message}</td></tr>`;
  }
}

function renderGPUTable(gpus) {
  const tbody = document.getElementById("gpu-table-body");
  tbody.innerHTML = "";
  if (!gpus || !gpus.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="text-muted">No GPU info available</td></tr>';
    return;
  }
  for (const g of gpus) {
    const pct = g.total_mb > 0 ? Math.round((g.used_mb / g.total_mb) * 100) : 0;
    const barCls = pct > 85 ? "high" : "";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${g.name}</td>
      <td>${fmt(g.total_mb)} MB</td>
      <td>
        <div class="flex items-center gap-8">
          <div class="vram-bar-wrap" style="flex:1">
            <div class="vram-bar ${barCls}" style="width:${pct}%"></div>
          </div>
          <span>${fmt(g.used_mb)} MB (${pct}%)</span>
        </div>
      </td>
      <td>${fmt(g.free_mb)} MB</td>
      <td>${g.utilization}%</td>
    `;
    tbody.appendChild(tr);
  }
}

const MODEL_SIZES = [
  {label: "4B FP16",  gb: 8},
  {label: "4B Q8",    gb: 4},
  {label: "4B Q4",    gb: 2.5},
  {label: "7B FP16",  gb: 14},
  {label: "7B Q8",    gb: 7.5},
  {label: "7B Q4",    gb: 4},
  {label: "13B FP16", gb: 26},
  {label: "13B Q8",   gb: 13},
  {label: "13B Q4",   gb: 7},
  {label: "70B Q4",   gb: 37},
];

function renderAdvisor(gpus) {
  const tbody = document.getElementById("advisor-body");
  tbody.innerHTML = "";
  if (!gpus || !gpus.length) return;
  const freeGb = gpus.reduce((s, g) => s + g.free_mb, 0) / 1024;

  for (const m of MODEL_SIZES) {
    const fits = Math.floor(freeGb / m.gb);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${m.label}</td>
      <td>~${m.gb} GB</td>
      <td class="${fits > 0 ? "fits" : "no-fit"}">${fits > 0 ? `${fits}x` : "No"}</td>
    `;
    tbody.appendChild(tr);
  }
}

async function loadLogs() {
  const container = document.getElementById("log-container-select").value;
  const output    = document.getElementById("log-output");
  output.textContent = "Loading…";
  try {
    const data = await apiFetch("/api/logs/" + container);
    output.textContent = data.logs || "(empty)";
    output.scrollTop = output.scrollHeight;
  } catch (err) {
    output.textContent = "Error: " + err.message;
  }
}

/* ================================================================
   Launch / Stop actions
   ================================================================ */
async function launchModel() {
  const profile = prompt("Enter launch profile (e.g. ollama, nim-llama, hf):", "ollama");
  if (!profile) return;
  const hfModel = profile === "hf" ? prompt("HuggingFace model ID:", "") : null;
  try {
    const data = await apiFetch("/api/launch", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({profile, hf_model: hfModel || undefined}),
    });
    alert(data.status + "\n" + data.output);
    loadDashboard();
  } catch (err) {
    alert("Launch error: " + err.message);
  }
}

async function stopAll() {
  if (!confirm("Stop all model containers?")) return;
  try {
    const data = await apiFetch("/api/stop", {method: "POST"});
    alert(data.status + "\n" + data.output);
    loadDashboard();
  } catch (err) {
    alert("Stop error: " + err.message);
  }
}

/* ================================================================
   Bootstrap
   ================================================================ */
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initChat();

  // Initial loads
  loadDashboard();
  loadGPUInfo();

  // Wire up buttons
  document.getElementById("refresh-toggle").addEventListener("click", toggleAutoRefresh);
  document.getElementById("refresh-now").addEventListener("click", loadDashboard);
  document.getElementById("launch-btn").addEventListener("click", launchModel);
  document.getElementById("stop-btn").addEventListener("click", stopAll);

  // Models tab
  document.getElementById("ollama-refresh").addEventListener("click", loadOllamaModels);
  document.getElementById("pull-btn").addEventListener("click", pullOllamaModel);
  document.getElementById("hf-refresh").addEventListener("click", loadHFCache);

  // Chat tab
  document.getElementById("fetch-models-btn").addEventListener("click", fetchChatModels);
  document.getElementById("new-chat-btn").addEventListener("click", clearChat);
  document.getElementById("send-btn").addEventListener("click", sendChat);

  // System tab
  document.getElementById("gpu-refresh").addEventListener("click", loadGPUInfo);
  document.getElementById("log-refresh").addEventListener("click", loadLogs);

  // Load models when switching to Models tab
  document.querySelector('.tab-btn[data-tab="models"]').addEventListener("click", () => {
    loadOllamaModels();
  });

  // Load system info when switching to System tab
  document.querySelector('.tab-btn[data-tab="system"]').addEventListener("click", () => {
    loadGPUInfo();
  });

  // Populate chat model if activeModel already set
  if (activeModel) {
    document.getElementById("chat-model").value = activeModel;
  }
});
