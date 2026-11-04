(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const csrf = document.querySelector('meta[name="csrf-token"]').content;
  const scheme = $("app").dataset.scheme;

  const state = { data: null, skipped: [], previous: null, busy: false, mine: false };

  function showError(message) {
    const el = $("error");
    el.textContent = message || "";
    el.hidden = !message;
  }

  async function api(path, options = {}) {
    const headers = { Accept: "application/json", "X-CSRF-Token": csrf };
    if (options.body) headers["Content-Type"] = "application/json";
    const res = await fetch(path, { credentials: "same-origin", ...options, headers });
    if (res.status === 401) { window.location.href = "/auth/login?next=/label"; throw new Error("auth"); }
    if (res.status === 429) throw new Error("Zu viele Anfragen – kurz warten.");
    if (!res.ok) throw new Error("Fehler (" + res.status + ")");
    return res.json();
  }

  function render(data, opts = {}) {
    state.data = data;
    state.mine = Boolean(opts.mine);
    const { image, labels, suggestion, progress } = data;
    $("progress").max = Math.max(progress.total, 1);
    $("progress").value = progress.done;
    $("progress-text").textContent = progress.done + " / " + progress.total;

    const hasImage = Boolean(image);
    $("photo").hidden = !hasImage;
    $("buttons").hidden = !hasImage;
    $("skip").hidden = !hasImage;
    $("empty").hidden = hasImage;
    $("back").disabled = !state.previous || state.mine;
    const confirm = $("confirm");
    confirm.hidden = true;
    if (!hasImage) return;

    $("photo").src = image.url;
    const wrap = $("buttons");
    wrap.replaceChildren();
    labels.forEach((label, index) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn" + (suggestion && suggestion.label === label.key ? " suggested" : "");
      btn.dataset.label = label.key;
      btn.textContent = (index + 1) + " · " + label.display;
      btn.addEventListener("click", () => submit(label.key));
      wrap.appendChild(btn);
    });
    if (suggestion) {
      confirm.hidden = false;
      confirm.dataset.label = suggestion.label;
      const prefix = state.mine ? "Dein Label: " : (suggestion.zero_shot ? "Vorschlag (Zero-Shot): " : "Vorschlag: ");
      const tail = state.mine ? " – ändern oder bestätigen" : " – bestätigen oder korrigieren";
      confirm.textContent = prefix + suggestion.display + tail;
    }
  }

  async function load() {
    showError("");
    const qs = new URLSearchParams({ scheme, skipped: state.skipped.join(",") });
    render(await api("/api/next?" + qs.toString()));
  }

  function preload(data) {
    if (data.image) { const img = new Image(); img.src = data.image.url; }
  }

  async function submit(label) {
    if (state.busy || !state.data || !state.data.image) return;
    state.busy = true;
    showError("");
    const current = state.data.image;
    try {
      const resp = await api("/api/label", {
        method: "POST",
        body: JSON.stringify({ sha256: current.sha256, scheme, label, skipped: state.skipped }),
      });
      state.previous = current;
      render(resp.next);
      preload(resp.next);
    } catch (err) {
      if (err.message !== "auth") showError(err.message);
    } finally {
      state.busy = false;
    }
  }

  async function skip() {
    if (state.busy || !state.data || !state.data.image) return;
    state.busy = true;
    state.skipped.push(state.data.image.sha256);
    try { await load(); } catch (err) { if (err.message !== "auth") showError(err.message); }
    finally { state.busy = false; }
  }

  async function back() {
    if (state.busy || !state.previous) return;
    state.busy = true;
    try {
      const prev = state.previous;
      const mine = await api("/api/my-label/" + prev.sha256 + "?scheme=" + scheme);
      const labels = state.data.labels;
      const entry = labels.find((l) => l.key === mine.label);
      render({
        scheme, labels, image: prev, progress: state.data.progress,
        suggestion: entry ? { label: entry.key, display: entry.display, confidence: 1, model_version: "mine", zero_shot: false } : null,
      }, { mine: true });
    } catch (err) {
      if (err.message !== "auth") showError(err.message);
    } finally {
      state.busy = false;
    }
  }

  $("confirm").addEventListener("click", () => submit($("confirm").dataset.label));
  $("skip").addEventListener("click", skip);
  $("back").addEventListener("click", back);

  document.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey || !state.data) return;
    if (event.key === "Enter") {
      const label = $("confirm").hidden ? null : $("confirm").dataset.label;
      if (label) { event.preventDefault(); submit(label); }
    } else if (event.key.toLowerCase() === "s") {
      skip();
    } else if (/^[1-9]$/.test(event.key)) {
      const label = state.data.labels[Number(event.key) - 1];
      if (label) submit(label.key);
    }
  });

  load().catch((err) => { if (err.message !== "auth") showError(err.message); });
})();
