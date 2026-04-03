/* RAG Knowledge Base — Frontend Application Logic */

const API_BASE = "";
const messageHistory = [];

/* ===== Initialization ===== */

function init() {
  checkHealth();
  loadDocuments();

  document.getElementById("upload-form").addEventListener("submit", handleUpload);
  document.getElementById("chat-form").addEventListener("submit", handleQuery);
  document.getElementById("clear-chat").addEventListener("click", clearChat);
}

document.addEventListener("DOMContentLoaded", init);

/* ===== Health Check ===== */

async function checkHealth() {
  const el = document.getElementById("health-status");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("unhealthy");
    const data = await res.json();
    if (data.status === "healthy") {
      el.textContent = "Connected";
      el.className = "connected";
    } else {
      throw new Error("unhealthy");
    }
  } catch (err) {
    el.textContent = "Disconnected";
    el.className = "disconnected";
    disableForms(true);
  }
}

function disableForms(disabled) {
  const uploadBtn = document.querySelector("#upload-form button[type='submit']");
  const chatInput = document.getElementById("chat-input");
  const chatBtn = document.querySelector("#chat-form button[type='submit']");
  if (uploadBtn) uploadBtn.disabled = disabled;
  if (chatInput) chatInput.disabled = disabled;
  if (chatBtn) chatBtn.disabled = disabled;
}

/* ===== Document Management ===== */

async function loadDocuments() {
  const container = document.getElementById("document-list");
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) {
      const detail = await extractErrorDetail(res);
      showError(container, detail);
      return;
    }
    const data = await res.json();
    if (!data.documents || data.documents.length === 0) {
      container.innerHTML = '<p class="empty-state">No documents uploaded yet.</p>';
    } else {
      renderDocumentList(data.documents);
    }
  } catch (err) {
    if (err instanceof TypeError) {
      showError(container, "Backend is unreachable");
    } else {
      showError(container, err.message);
    }
  }
}

function renderDocumentList(documents) {
  const container = document.getElementById("document-list");
  container.innerHTML = "";
  documents.forEach((doc) => {
    const card = document.createElement("div");
    card.className = "document-card";
    card.dataset.documentId = doc.document_id;

    const info = document.createElement("div");
    info.className = "doc-info";

    const title = document.createElement("strong");
    title.textContent = doc.filename;
    info.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "doc-meta";
    meta.textContent =
      `Source: ${doc.source} · Type: ${doc.content_type} · Size: ${formatFileSize(doc.file_size)} · Uploaded: ${formatDate(doc.uploaded_at)}`;
    info.appendChild(meta);

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "delete-btn";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", () => handleDelete(doc.document_id));

    card.appendChild(info);
    card.appendChild(deleteBtn);
    container.appendChild(card);
  });
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr) {
  try {
    return new Date(dateStr).toLocaleDateString();
  } catch {
    return dateStr;
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const form = event.target;
  const submitBtn = form.querySelector("button[type='submit']");
  const statusEl = document.getElementById("upload-status");
  statusEl.innerHTML = "";

  const formData = new FormData();
  const fileInput = document.getElementById("file-input");
  const sourceInput = document.getElementById("source-input");
  const chunkSizeInput = document.getElementById("chunk-size-input");
  const chunkOverlapInput = document.getElementById("chunk-overlap-input");

  formData.append("file", fileInput.files[0]);
  formData.append("source", sourceInput.value);
  formData.append("chunk_size", chunkSizeInput.value);
  formData.append("chunk_overlap", chunkOverlapInput.value);

  setLoading(submitBtn, true);
  try {
    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const detail = await extractErrorDetail(res);
      showError(statusEl, detail);
      return;
    }
    const data = await res.json();
    statusEl.innerHTML = "";
    const msg = document.createElement("p");
    msg.textContent = `Uploaded "${data.filename}" (ID: ${data.document_id}, ${data.chunk_count} chunks)`;
    statusEl.appendChild(msg);
    form.reset();
    await loadDocuments();
  } catch (err) {
    if (err instanceof TypeError) {
      showError(statusEl, "Backend is unreachable");
    } else {
      showError(statusEl, err.message);
    }
  } finally {
    setLoading(submitBtn, false);
  }
}

async function handleDelete(documentId) {
  const card = document.querySelector(`.document-card[data-document-id="${documentId}"]`);
  const deleteBtn = card ? card.querySelector(".delete-btn") : null;
  if (deleteBtn) setLoading(deleteBtn, true);

  try {
    const res = await fetch(`${API_BASE}/documents/${documentId}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const detail = await extractErrorDetail(res);
      const container = document.getElementById("document-list");
      showError(container, detail);
      return;
    }
    await loadDocuments();
  } catch (err) {
    const container = document.getElementById("document-list");
    if (err instanceof TypeError) {
      showError(container, "Backend is unreachable");
    } else {
      showError(container, err.message);
    }
  } finally {
    if (deleteBtn) setLoading(deleteBtn, false);
  }
}

function setLoading(button, loading) {
  if (!button) return;
  button.disabled = loading;
  if (loading) {
    button.dataset.originalText = button.textContent;
    button.textContent = "Loading...";
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
  }
}

/* ===== Chat Interface ===== */

async function handleQuery(event) {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;

  const sendBtn = document.querySelector("#chat-form button[type='submit']");

  messageHistory.push({ role: "user", content: question });
  appendMessage("user", question);
  input.value = "";

  setLoading(sendBtn, true);
  try {
    // Build chat history from recent user/assistant messages (exclude errors)
    const history = messageHistory
      .filter((m) => m.role === "user" || m.role === "assistant")
      .map((m) => ({
        role: m.role,
        content: m.role === "user" ? m.content : (m.content || ""),
      }));
    // Don't include the current question (already in history from the push above),
    // but the backend expects history WITHOUT the current question
    const pastHistory = history.slice(0, -1);

    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5, chat_history: pastHistory }),
    });
    if (!res.ok) {
      const detail = await extractErrorDetail(res);
      messageHistory.push({ role: "error", content: detail });
      appendMessage("error", detail);
      return;
    }
    const data = await res.json();
    const displayText = data.answer || data.message || "No answer returned.";
    const assistantPayload = {
      answer: displayText,
      confidence: data.confidence,
      sources: data.sources,
      message: data.message,
    };
    messageHistory.push({
      role: "assistant",
      content: displayText,
      confidence: data.confidence,
      sources: data.sources,
    });
    appendMessage("assistant", assistantPayload);
  } catch (err) {
    const errorMsg = err instanceof TypeError ? "Backend is unreachable" : err.message;
    messageHistory.push({ role: "error", content: errorMsg });
    appendMessage("error", errorMsg);
  } finally {
    setLoading(sendBtn, false);
  }
}

function appendMessage(role, content) {
  const thread = document.getElementById("message-thread");
  const el = document.createElement("div");

  if (role === "user") {
    el.className = "user-message";
    el.textContent = content;
  } else if (role === "assistant") {
    el.className = "assistant-message";
    const answer = document.createElement("p");
    answer.textContent = content.answer || content.message || content.content || "";
    el.appendChild(answer);

    const conf = content.confidence ?? content.confidence;
    if (conf !== undefined && conf !== null) {
      const confEl = document.createElement("span");
      confEl.className = "confidence";
      confEl.textContent = `Confidence: ${(conf * 100).toFixed(0)}%`;
      el.appendChild(confEl);
    }

    const sources = content.sources;
    if (sources && sources.length > 0) {
      el.appendChild(renderSources(sources));
    }
  } else if (role === "error") {
    el.className = "error-message";
    el.textContent = content;
  }

  thread.appendChild(el);
  thread.scrollTop = thread.scrollHeight;
}

function renderSources(sources) {
  const container = document.createElement("div");
  container.className = "source-cards";
  sources.forEach((src) => {
    const card = document.createElement("div");
    card.className = "source-card";

    const label = document.createElement("span");
    label.className = "source-label";
    label.textContent = src.source;

    const badge = document.createElement("span");
    badge.className = "score-badge";
    badge.textContent = src.score.toFixed(2);

    const snippet = document.createElement("span");
    snippet.className = "snippet";
    snippet.textContent = src.text_snippet;

    card.appendChild(label);
    card.appendChild(badge);
    card.appendChild(snippet);
    container.appendChild(card);
  });
  return container;
}

function clearChat() {
  messageHistory.length = 0;
  document.getElementById("message-thread").innerHTML = "";
}

/* ===== Error Handling Utilities ===== */

function showError(container, message) {
  const el = document.createElement("div");
  el.className = "error";
  el.textContent = message;
  container.appendChild(el);
}

async function extractErrorDetail(response) {
  try {
    const data = await response.json();
    return data.detail || `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}
