"use strict";

/* ---------------------------------- state --------------------------------- */
const state = {
  sessionId: null,
  difficulty: 2,
  configured: true, // LLM 是否已配置（API Key）
  busy: false,
  resumePending: false, // 面试开始后是否仍在等待提交简历（提交前禁止作答）
};

const $ = (id) => document.getElementById(id);

/* --------------------------------- helpers -------------------------------- */
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.add("hidden"), 3200);
}

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `请求失败 (${res.status})`);
  return data;
}

function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(s) {
  return esc(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

/** Minimal markdown -> HTML (headings / tables / lists / paragraphs). */
function mdToHtml(md) {
  const lines = String(md || "").split(/\r?\n/);
  let html = "", inTable = false, tableBuf = [], inList = null, para = [];

  const flushPara = () => {
    if (para.length) { html += `<p>${inline(para.join(" "))}</p>`; para = []; }
  };
  const closeTable = () => {
    if (!inTable) return;
    const rows = tableBuf.filter((_, i) => i !== 1); // skip |---| separator
    html += "<table><thead><tr>" + rows[0].map((c) => `<th>${inline(c)}</th>`).join("") + "</tr></thead><tbody>" +
      rows.slice(1).map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>").join("") +
      "</tbody></table>";
    inTable = false; tableBuf = [];
  };
  const closeList = () => {
    if (inList) { html += `</${inList}>`; inList = null; }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushPara(); closeTable(); closeList(); continue; }
    if (/^-{3,}$/.test(line)) { flushPara(); html += "<hr>"; continue; }
    if (/^```/.test(line)) continue;

    // heading
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) { flushPara(); closeTable(); closeList(); html += `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`; continue; }

    // table
    if (line.startsWith("|")) {
      flushPara(); closeList();
      const cells = line.split("|").slice(1, -1).map((c) => c.trim());
      if (/^:?-{2,}:?$/.test(cells.join(""))) { inTable = true; continue; }
      if (!inTable) inTable = true;
      tableBuf.push(cells);
      continue;
    }
    if (inTable) closeTable();

    // lists
    const li = line.match(/^[-*]\s+(.*)$/) || line.match(/^\d+[.、]\s+(.*)$/);
    if (li) {
      flushPara();
      const tag = inList || "ul";
      if (!inList) { html += "<ul>"; inList = "ul"; }
      html += `<li>${inline(li[1])}</li>`;
      continue;
    }
    closeList();
    para.push(line);
  }
  flushPara(); closeTable(); closeList();
  return html;
}

/* -------------------------------- screens --------------------------------- */
function showScreen(name) {
  ["setup", "chat", "report"].forEach((s) => $("screen-" + s).classList.toggle("hidden", s !== name));
}

function setDifficulty(tier, signal = 0) {
  state.difficulty = tier;
  document.querySelectorAll(".seg").forEach((seg) => {
    seg.classList.toggle("active", Number(seg.dataset.tier) === tier);
  });
  const names = { 1: "L1 · 概念基础", 2: "L2 · 应用场景", 3: "L3 · 深度设计" };
  const arrow = signal > 0 ? " ↑ 难度上调" : signal < 0 ? " ↓ 难度下调" : "";
  $("diff-label").textContent = (names[tier] || "") + arrow;
}

function addMessage(speaker, content, opts = {}) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${speaker}`;
  const avatar = speaker === "interviewer" ? "面" : "我";
  wrap.innerHTML =
    `<div class="avatar">${avatar}</div>` +
    `<div><div class="bubble"></div><div class="meta-line"></div></div>`;
  wrap.querySelector(".bubble").textContent = content;
  const meta = wrap.querySelector(".meta-line");
  if (speaker === "interviewer" && opts.tier) meta.innerHTML = `<span class="chip-tier">难度 ${opts.tier}</span>`;
  if (speaker === "candidate" && opts.quality) meta.innerHTML = `答题质量 ${"★".repeat(Math.max(1, Math.min(5, opts.quality)))}<span>${opts.judgment ? " · " + opts.judgment : ""}</span>`;
  if (opts.tier) setDifficulty(opts.difficulty ?? state.difficulty);
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrap;
}

function setBusy(busy) {
  state.busy = busy;
  $("typing").classList.toggle("hidden", !busy);
  const locked = busy || state.resumePending;
  $("btn-send").disabled = locked;
  $("btn-end").disabled = busy;
  $("answer-input").disabled = locked;
}

/* --------------------------------- setup ---------------------------------- */
async function refreshStatus({ prompt = false } = {}) {
  try {
    const h = await api("/api/health");
    state.configured = !!h.configured;
    $("badge-model").textContent = state.configured ? `模型: ${h.model}` : "模型: 未配置";
    $("badge-unconfigured").classList.toggle("hidden", state.configured);
    $("setup-need-key").classList.toggle("hidden", state.configured);
    if (!state.configured) {
      $("btn-start").disabled = true;
      if (prompt) openLlmSettings(); // 弹窗提醒填写 API Key
    } else {
      $("btn-start").disabled = $("jd-input").value.trim().length < 20;
    }
  } catch (_) { /* ignore */ }
}

/* ----------------------------- LLM settings ------------------------------- */
let llmCfg = null;

function setStatus(text, ok) {
  const el = $("cfg-status");
  el.textContent = text;
  el.className = "status" + (ok === undefined ? "" : ok ? " ok" : " err");
}

function syncKeyUi() {
  const hasKey = !!(llmCfg && llmCfg.api_key_set);
  $("cfg-clear-row").classList.toggle("hidden", !hasKey);
  const hint = $("cfg-key-hint");
  if ($("cfg-clear").checked) hint.textContent = "保存后将清除 Key，服务回到未配置状态。";
  else if (hasKey) hint.textContent = "当前已配置 Key（sk-••••••）；留空表示保持不变，勾选下方可清除。";
  else hint.textContent = "必填。保存后立即生效，无需重启。";
}

async function openLlmSettings() {
  try {
    llmCfg = await api("/api/llm/config");
    $("cfg-key").type = "password";
    $("cfg-key-toggle").textContent = "显示";
    $("cfg-key").value = "";
    $("cfg-clear").checked = false;
    $("cfg-base").value = llmCfg.base_url || "";
    $("cfg-model").value = llmCfg.model || "";
    $("cfg-base").placeholder = llmCfg.default_base_url;
    $("cfg-model").placeholder = llmCfg.default_model;
    syncKeyUi();
    if (llmCfg.configured) {
      setStatus(`当前：已连接模型 ${llmCfg.active_model}`);
    } else {
      setStatus("⚠ 尚未配置 LLM。请填写 API Key 后保存，才能开始面试。", false);
    }
  } catch (e) {
    toast("读取配置失败: " + e.message);
  }
  $("llm-modal").classList.remove("hidden");
}

function closeLlmSettings() {
  $("llm-modal").classList.add("hidden");
}

function formCredentials() {
  return {
    api_key: $("cfg-key").value.trim(),
    base_url: $("cfg-base").value.trim(),
    model: $("cfg-model").value.trim(),
  };
}

async function testLlm() {
  const btn = $("btn-llm-test");
  btn.disabled = true;
  setStatus("正在测试连接…");
  try {
    const r = await api("/api/llm/test", "POST", formCredentials());
    setStatus((r.ok ? "✅ " : "❌ ") + r.message, r.ok);
  } catch (e) {
    setStatus("❌ 请求失败：" + e.message, false);
  } finally {
    btn.disabled = false;
  }
}

async function refreshModels() {
  const btn = $("btn-models-refresh");
  btn.disabled = true;
  btn.textContent = "拉取中…";
  setStatus("正在拉取模型列表…");
  try {
    const cred = formCredentials();
    const r = await api("/api/llm/models", "POST", {
      api_key: cred.api_key,
      base_url: cred.base_url,
    });
    if (!r.ok) {
      setStatus("❌ " + (r.message || "拉取失败"), false);
      return;
    }
    const list = r.models || [];
    const dl = $("model-suggest");
    dl.innerHTML = list.map((m) => `<option value="${esc(m)}"></option>`).join("");
    const cur = $("cfg-model").value.trim();
    if (!cur && list.length) $("cfg-model").value = list[0];
    setStatus(`✅ 拉取到 ${list.length} 个可用模型，点击输入框可从下拉选择`, true);
  } catch (e) {
    setStatus("❌ 请求失败：" + e.message, false);
  } finally {
    btn.disabled = false;
    btn.textContent = "🔄 拉取当前账号可用模型";
  }
}

async function saveLlm() {
  const btn = $("btn-llm-save");
  btn.disabled = true;
  setStatus("正在保存…");
  try {
    // API Key：勾选“清除” → 置空；有输入 → 更新；留空 → 保持不变
    let api_key = null;
    if ($("cfg-clear").checked) api_key = "";
    else {
      const v = $("cfg-key").value.trim();
      api_key = v ? v : null;
    }
    llmCfg = await api("/api/llm/config", "POST", {
      api_key,
      base_url: $("cfg-base").value.trim(),
      model: $("cfg-model").value.trim(),
      persist: $("cfg-persist").checked,
    });
    await refreshStatus();
    if (llmCfg.configured) {
      setStatus(`✅ 已保存 · 模型 ${llmCfg.active_model}`, true);
      toast("已应用新的 LLM 配置");
    } else {
      setStatus("已清除 Key —— 请填写 API Key 后才能开始面试。", false);
      toast("LLM 配置已清除");
    }
    setTimeout(closeLlmSettings, 900);
  } catch (e) {
    setStatus("❌ 保存失败：" + e.message, false);
  } finally {
    btn.disabled = false;
  }
}

$("btn-settings").addEventListener("click", () => openLlmSettings());
$("btn-llm-close").addEventListener("click", closeLlmSettings);
$("llm-modal").addEventListener("click", (ev) => {
  if (ev.target === $("llm-modal")) closeLlmSettings();
});
$("cfg-key").addEventListener("input", () => {
  if ($("cfg-key").value.trim()) $("cfg-clear").checked = false;
  syncKeyUi();
});
$("cfg-clear").addEventListener("change", syncKeyUi);
$("cfg-key-toggle").addEventListener("click", () => {
  const el = $("cfg-key");
  const show = el.type === "password";
  el.type = show ? "text" : "password";
  $("cfg-key-toggle").textContent = show ? "隐藏" : "显示";
});
document.querySelectorAll(".presets button").forEach((b) =>
  b.addEventListener("click", () => {
    $("cfg-base").value = b.dataset.base;
    $("cfg-model").value = b.dataset.model;
  })
);
$("btn-llm-test").addEventListener("click", testLlm);
$("btn-models-refresh").addEventListener("click", refreshModels);
$("btn-llm-save").addEventListener("click", saveLlm);
document.addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape") return;
  if (!$("llm-modal").classList.contains("hidden")) closeLlmSettings();
  else if (!$("dlg-mask").classList.contains("hidden")) closeDialog();
  else setSidebarOpen(false);
});

/* ------------------------- sidebar: drawer (narrow) ------------------------- */
function setSidebarOpen(open) {
  document.body.classList.toggle("sidebar-open", !!open);
}
$("btn-sidebar-toggle").addEventListener("click", () => {
  setSidebarOpen(!document.body.classList.contains("sidebar-open"));
});
$("side-scrim").addEventListener("click", () => setSidebarOpen(false));

/* ------------------------------ confirm dialog ------------------------------ */
/* 页内对话框（取代 prompt/confirm：在受限环境里原生弹窗可能被静默拦截） */
let dlgAction = null;

function openDialog({ title, body, okText = "确定", danger = false, onOk }) {
  $("dlg-title").textContent = title;
  const wrap = $("dlg-body");
  wrap.innerHTML = "";
  if (typeof body === "string") wrap.textContent = body;
  else if (body) wrap.appendChild(body);
  const ok = $("dlg-ok");
  ok.textContent = okText;
  ok.disabled = false;
  ok.classList.toggle("danger", !!danger);
  dlgAction = onOk || null;
  $("dlg-mask").classList.remove("hidden");
  const first = wrap.querySelector("input, textarea");
  if (first) {
    first.focus();
    if (first.select) first.select();
  }
}

function closeDialog() {
  $("dlg-mask").classList.add("hidden");
  dlgAction = null;
}

$("dlg-close").addEventListener("click", closeDialog);
$("dlg-cancel").addEventListener("click", closeDialog);
$("dlg-mask").addEventListener("click", (ev) => {
  if (ev.target === $("dlg-mask")) closeDialog();
});
$("dlg-ok").addEventListener("click", async () => {
  const fn = dlgAction;
  if (!fn) { closeDialog(); return; }
  const ok = $("dlg-ok");
  ok.disabled = true;
  let keep = false;
  try { keep = (await fn()) === false; }
  catch (_) { keep = true; }
  ok.disabled = false;
  if (!keep) closeDialog();
});

/* --------------------------- sidebar: actions ------------------------------- */
function openRenameDialog(s) {
  const input = document.createElement("input");
  input.type = "text";
  input.maxLength = 120;
  input.value = sessionTitle(s);
  input.placeholder = "输入会话名称…";
  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); $("dlg-ok").click(); }
  });
  openDialog({
    title: "重命名会话",
    body: input,
    okText: "保存",
    onOk: async () => {
      const name = input.value.trim();
      if (!name) { toast("名称不能为空"); return false; }
      try {
        await api(`/api/sessions/${s.id}/rename`, "POST", { title: name });
        toast("已重命名");
        loadSessions();
        return true;
      } catch (e) { toast(e.message); return false; }
    },
  });
}

function confirmDeleteSession(id) {
  openDialog({
    title: "删除会话",
    body: "确定删除该面试会话？会话消息与评估报告将一并删除，且无法恢复。",
    okText: "删除",
    danger: true,
    onOk: async () => {
      try {
        await api(`/api/sessions/${id}`, "DELETE");
        toast("已删除");
        if (state.sessionId === id) resetToSetup();
        loadSessions();
        return true;
      } catch (e) { toast(e.message); return false; }
    },
  });
}

function confirmEndInterview() {
  openDialog({
    title: "结束面试",
    body: "确定结束本次面试并生成评估报告吗？结束后将无法继续作答。",
    okText: "结束并出报告",
    onOk: async () => {
      try {
        const r = await api(`/api/sessions/${state.sessionId}/end`, "POST", { reason: "user_requested" });
        toast("报告生成完毕");
        showReport(r.report_md || "");
        loadSessions();
        return true;
      } catch (e) { toast(e.message); return false; }
    },
  });
}

$("btn-sample").addEventListener("click", async () => {
  try {
    const res = await fetch("/static/sample_jd.txt");
    $("jd-input").value = await res.text();
    $("btn-start").disabled = false;
  } catch (_) {
    toast("示例 JD 加载失败");
  }
});

$("jd-input").addEventListener("input", () => {
  $("btn-start").disabled = $("jd-input").value.trim().length < 20;
  $("setup-error").classList.add("hidden");
});

$("btn-start").addEventListener("click", startInterview);

async function startInterview() {
  const jd = $("jd-input").value.trim();
  if (jd.length < 20) return;
  if (!state.configured) {
    $("setup-need-key").classList.remove("hidden");
    openLlmSettings();
    return;
  }
  $("btn-start").disabled = true;
  $("setup-error").classList.add("hidden");
  $("setup-need-key").classList.add("hidden");
  try {
    const s = await api("/api/sessions", "POST", { jd });
    state.sessionId = s.session_id;
    state.difficulty = s.difficulty;
    // interviewer card
    $("iv-role").textContent = `${s.role_title || "岗位"} · ${s.style}风格面试官`;
    $("iv-company").textContent = s.company ? `代表 ${s.company}` : "公司背景未指明 · 基于 JD 面试";
    const chips = $("iv-focus");
    chips.innerHTML = (s.focus_points || []).map((f) => `<span class="chip">${esc(f)}</span>`).join("");
    // reset messages
    $("messages").innerHTML = "";
    setDifficulty(s.difficulty);
    addMessage("interviewer", s.opening, { tier: s.tier, difficulty: s.difficulty });
    showScreen("chat");
    showResumePanel(true); // 生成面试官后，先上传简历，第一个问题基于简历
    loadSessions(); // 侧栏刷新：新会话置顶高亮
  } catch (e) {
    $("setup-error").textContent = e.message;
    $("setup-error").classList.remove("hidden");
  } finally {
    $("btn-start").disabled = false;
  }
}

/* ------------------------------- resume ---------------------------------- */
function showResumePanel(show) {
  $("resume-panel").classList.toggle("hidden", !show);
  $("resume-error").classList.add("hidden");
  state.resumePending = show;
  if (show) resetResumeFile();
  setBusy(false); // 同步发送按钮锁定状态（resumePending 为真时禁止作答）
}

function resetResumeFile() {
  const input = $("resume-file");
  input.value = "";
  $("resume-file-name").textContent = "";
  $("resume-file-name").classList.add("hidden");
  $("resume-file-hint").classList.remove("hidden");
  $("resume-drop").classList.remove("has-file");
  $("btn-resume-submit").disabled = true;
}

function setResumeFile(file) {
  if (!file) { resetResumeFile(); return; }
  $("resume-file-name").textContent = `${file.name}（${(file.size / 1024).toFixed(0)} KB）`;
  $("resume-file-name").classList.remove("hidden");
  $("resume-file-hint").classList.add("hidden");
  $("resume-drop").classList.add("has-file");
  $("btn-resume-submit").disabled = false;
  $("resume-error").classList.add("hidden");
}

$("resume-file").addEventListener("change", () => {
  setResumeFile($("resume-file").files[0] || null);
});

// 拖拽选择简历文件
const resumeDrop = $("resume-drop");
["dragover", "dragenter"].forEach((evt) =>
  resumeDrop.addEventListener(evt, (e) => { e.preventDefault(); resumeDrop.classList.add("drag"); })
);
["dragleave", "drop"].forEach((evt) =>
  resumeDrop.addEventListener(evt, (e) => { e.preventDefault(); resumeDrop.classList.remove("drag"); })
);
resumeDrop.addEventListener("drop", (e) => {
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (!f) return;
  const dt = new DataTransfer(); // 同步进 input，保证提交时能取到文件
  dt.items.add(f);
  $("resume-file").files = dt.files;
  setResumeFile(f);
});

$("btn-resume-submit").addEventListener("click", async () => {
  if (state.busy || !state.sessionId) return;
  const file = $("resume-file").files && $("resume-file").files[0];
  $("resume-error").classList.add("hidden");
  if (!file) {
    $("resume-error").textContent = "请先选择简历文件";
    $("resume-error").classList.remove("hidden");
    return;
  }
  const btn = $("btn-resume-submit");
  btn.disabled = true;
  btn.textContent = "上传中…";
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(`/api/sessions/${state.sessionId}/resume`, { method: "POST", body: fd });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `请求失败 (${res.status})`);
    addMessage("interviewer", data.reply, { tier: data.tier, difficulty: data.difficulty });
    showResumePanel(false);
    $("answer-input").focus();
    toast("简历已收到，面试官将基于简历提问");
  } catch (e) {
    $("resume-error").textContent = e.message;
    $("resume-error").classList.remove("hidden");
    btn.disabled = false;
    btn.textContent = "提交简历";
  }
});

/* --------------------------------- chat ----------------------------------- */
$("answer-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  if (state.busy || !state.sessionId) return;
  const text = $("answer-input").value.trim();
  if (!text) return;
  $("answer-input").value = "";
  submitAnswer(text);
});

async function submitAnswer(text) {
  setBusy(true);
  addMessage("candidate", text);
  try {
    const r = await api(`/api/sessions/${state.sessionId}/answer`, "POST", { answer: text });
    addMessage("interviewer", r.reply, {
      tier: r.tier, difficulty: r.difficulty,
      quality: r.quality, judgment: r.judgment,
    });
    if (r.signal) setDifficulty(r.difficulty, r.signal);
    if (r.finished) {
      toast("面试已结束，正在生成评估报告…");
      setTimeout(() => { showReport(r.report_md || ""); loadSessions(); }, 400);
    }
  } catch (e) {
    toast(e.message);
    if (/已结束/.test(e.message)) resetToSetup();
  } finally {
    setBusy(false);
  }
}

$("btn-end").addEventListener("click", () => {
  if (state.busy || !state.sessionId) return;
  confirmEndInterview();
});

$("answer-input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    $("answer-form").dispatchEvent(new Event("submit"));
  }
});

/* --------------------------------- report --------------------------------- */
function showReport(md) {
  $("report-content").innerHTML = mdToHtml(md);
  showScreen("report");
  window.scrollTo({ top: 0 });
}

function resetToSetup() {
  state.sessionId = null;
  $("messages").innerHTML = "";
  showResumePanel(false);
  showScreen("setup");
  $("jd-input").focus();
  setSidebarOpen(false);
  highlightSidebar();
}

$("btn-report-reset").addEventListener("click", resetToSetup);

/* ---------------------------- sidebar: sessions ---------------------------- */
let sideSessions = [];
let sideSearchTimer = null;

const TIER_NAMES = { 1: "L1 · 概念基础", 2: "L2 · 应用场景", 3: "L3 · 深度设计" };

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getMonth() + 1}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function sessionTitle(s) {
  return (s.title && s.title.trim()) || s.role_title || "未命名面试";
}

async function loadSessions() {
  const q = encodeURIComponent($("side-search").value.trim());
  try {
    const r = await api(`/api/sessions?limit=100${q ? "&q=" + q : ""}`);
    sideSessions = r.sessions || [];
    renderSidebar();
  } catch (_) {
    $("side-list").innerHTML = `<p class="side-empty">加载失败</p>`;
  }
}

function renderSidebar() {
  const box = $("side-list");
  if (!sideSessions.length) {
    box.innerHTML = `<p class="side-empty">${$("side-search").value.trim() ? "没有匹配的会话" : "还没有面试记录\n点「＋ 新面试」开始第一场"}</p>`;
    return;
  }
  box.innerHTML = sideSessions.map((s) => {
    const active = s.id === state.sessionId ? " active" : "";
    const badge = s.status === "finished"
      ? `<span class="side-badge">已结束</span>`
      : `<span class="side-badge badge-live">进行中</span>`;
    const meta = [s.company || "", fmtTime(s.created_at), `${s.message_count || 0} 条消息`].filter(Boolean).join(" · ");
    return (
      `<div class="side-item${active}" data-id="${esc(s.id)}">` +
        `<div class="side-item-top"><span class="side-item-name" title="${esc(sessionTitle(s))}">${esc(sessionTitle(s))}</span>${badge}</div>` +
        `<div class="side-item-sub">${esc(meta)}</div>` +
        `<div class="side-item-actions">` +
          `<button class="side-act" data-act="open" title="打开会话">打开</button>` +
          `<button class="side-act" data-act="rename" title="重命名">重命名</button>` +
          `<button class="side-act danger" data-act="delete" title="删除会话">删除</button>` +
        `</div>` +
      `</div>`
    );
  }).join("");
}

function highlightSidebar() {
  renderSidebar(); // 仅刷新高亮，不重新拉取
}

async function openSession(id) {
  try {
    const d = await api(`/api/sessions/${id}`);
    state.sessionId = d.id || id;
    state.difficulty = d.difficulty || 2;
    setSidebarOpen(false); // 窄屏下选中后收起抽屉
    renderSidebar();

    if (d.status === "finished") {
      // 已结束：直接回看报告（内含完整对话附录）
      const html = d.report_html || mdToHtml(d.report_md || "");
      $("report-content").innerHTML = html;
      showScreen("report");
      window.scrollTo({ top: 0 });
      return;
    }

    // 进行中：续面 —— 重建面试官卡片 + 回放消息
    const p = d.profile || {};
    $("iv-role").textContent = `${p.role_title || "岗位"} · ${p.style || ""}风格面试官`;
    $("iv-company").textContent = p.company ? `代表 ${p.company}` : "公司背景未指明 · 基于 JD 面试";
    $("iv-focus").innerHTML = (p.focus_points || []).map((f) => `<span class="chip">${esc(f)}</span>`).join("");
    $("messages").innerHTML = "";
    setDifficulty(d.difficulty || 2);

    let lastCandidate = null;
    for (const m of d.messages || []) {
      if (m.speaker === "candidate") {
        lastCandidate = addMessage("candidate", m.content);
      } else if (m.speaker === "interviewer") {
        addMessage("interviewer", m.content, {
          tier: TIER_NAMES[m.difficulty] || null,
          difficulty: m.difficulty || state.difficulty,
        });
        if (m.quality != null && lastCandidate) {
          const meta = lastCandidate.querySelector(".meta-line");
          if (meta) meta.textContent = `答题质量 ${'★'.repeat(Math.max(1, Math.min(5, m.quality)))}`;
        }
        lastCandidate = null;
      }
    }
    // 未提交过简历且尚无作答记录 → 续面仍需先交简历
    const hasResume = !!d.resume;
    const hasAnswer = (d.messages || []).some((m) => m.speaker === "candidate");
    showResumePanel(!hasResume && !hasAnswer);
    showScreen("chat");
    setBusy(false);
    if (!state.resumePending) $("answer-input").focus();
    toast(hasResume || hasAnswer ? "已续面：继续回答即可" : "已续面：请先提交简历");
  } catch (e) {
    toast(e.message);
    loadSessions();
  }
}

$("side-search").addEventListener("input", () => {
  clearTimeout(sideSearchTimer);
  sideSearchTimer = setTimeout(loadSessions, 300);
});

$("side-list").addEventListener("click", (ev) => {
  const item = ev.target.closest(".side-item");
  if (!item) return;
  const id = item.dataset.id;
  const act = ev.target.closest(".side-act")?.dataset.act;
  if (act === "open" || !act) {
    openSession(id);
  } else if (act === "rename") {
    const s = sideSessions.find((x) => x.id === id);
    if (s) openRenameDialog(s);
  } else if (act === "delete") {
    confirmDeleteSession(id);
  }
});

$("btn-new-session").addEventListener("click", resetToSetup);

/* ----------------------------------- init ---------------------------------- */
refreshStatus({ prompt: true });
loadSessions();
