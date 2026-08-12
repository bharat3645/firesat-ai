/* FireSat-AI dashboard — vanilla JS, no build step.
 * Talks to the FastAPI backend under /api/*. Set window.FIRESAT_API_BASE
 * before this script loads if the API is not served from the same origin
 * as this page (CORS is open on the backend, so cross-origin works too).
 */
(function () {
  "use strict";

  const API_BASE = window.FIRESAT_API_BASE || "";
  const RISK_COLORS = ["#2fae66", "#e6b333", "#d1495b"]; // No Risk, Moderate, High
  const RISK_NAMES = ["No Risk", "Moderate", "High"];

  const state = {
    map: null,
    regionLayers: {}, // region_id -> L.rectangle
    regions: [],
    activeRegionId: null,
    activeHorizon: "horizon_1m",
    predictionCache: {}, // region_id -> latest RiskPredictionOut
  };

  async function fetchJSON(path) {
    const res = await fetch(API_BASE + path);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const err = new Error(body.detail || `${res.status} ${res.statusText}`);
      err.status = res.status;
      throw err;
    }
    return res.json();
  }

  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else e.setAttribute(k, v);
    });
    (children || []).forEach((c) => e.appendChild(c));
    return e;
  }

  // ---------------------------------------------------------------- health
  async function checkHealth() {
    const badge = document.getElementById("health-badge");
    try {
      const health = await fetchJSON("/api/health");
      if (health.ready) {
        badge.textContent = `backend ready · regions: ${health.regions_loaded.join(", ")}`;
        badge.className = "health-badge ok";
      } else {
        badge.textContent = `backend not ready: ${health.detail}`;
        badge.className = "health-badge bad";
      }
      return health.ready;
    } catch (e) {
      badge.textContent = `backend unreachable: ${e.message}`;
      badge.className = "health-badge bad";
      return false;
    }
  }

  // ------------------------------------------------------------------ map
  function initMap() {
    state.map = L.map("map", { zoomControl: true }).setView([62.5, -149.0], 5);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 12,
    }).addTo(state.map);
  }

  function riskColorForRegion(regionId) {
    const pred = state.predictionCache[regionId];
    if (!pred) return "#5b6b8c";
    const h = pred.horizons[state.activeHorizon];
    if (!h) return "#5b6b8c";
    return RISK_COLORS[h.risk_class_id];
  }

  function recolorMap() {
    Object.keys(state.regionLayers).forEach((id) => {
      const layer = state.regionLayers[id];
      layer.setStyle({
        fillColor: riskColorForRegion(id),
        color: id === state.activeRegionId ? "#ffffff" : riskColorForRegion(id),
      });
    });
  }

  async function loadRegions() {
    state.regions = await fetchJSON("/api/regions");
    const picker = document.getElementById("region-picker");
    picker.innerHTML = "";

    state.regions.forEach((region) => {
      const bounds = [
        [region.bbox[1], region.bbox[0]],
        [region.bbox[3], region.bbox[2]],
      ];
      const layer = L.rectangle(bounds, {
        weight: 2,
        color: "#5b6b8c",
        fillColor: "#5b6b8c",
        fillOpacity: 0.35,
      }).addTo(state.map);
      layer.bindTooltip(region.name);
      layer.on("click", () => selectRegion(region.id));
      state.regionLayers[region.id] = layer;

      const btn = el("button", { text: region.name });
      btn.addEventListener("click", () => selectRegion(region.id));
      picker.appendChild(btn);
      region._btn = btn;
    });

    if (state.regions.length) {
      const group = L.featureGroup(Object.values(state.regionLayers));
      state.map.fitBounds(group.getBounds().pad(0.6));
      selectRegion(state.regions[0].id);
    }
  }

  // -------------------------------------------------------------- horizon
  function initHorizonToggle() {
    const container = document.getElementById("horizon-toggle");
    container.querySelectorAll("button[data-horizon]").forEach((btn) => {
      btn.addEventListener("click", () => {
        container.querySelectorAll("button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeHorizon = btn.dataset.horizon;
        recolorMap();
        if (state.activeRegionId) renderHorizonBars(state.predictionCache[state.activeRegionId]);
        renderRiskSummary(state.predictionCache[state.activeRegionId]);
      });
    });
  }

  // --------------------------------------------------------- region select
  async function selectRegion(regionId) {
    state.activeRegionId = regionId;
    state.regions.forEach((r) => r._btn.classList.toggle("active", r.id === regionId));

    document.getElementById("risk-summary-body").innerHTML =
      '<p class="empty-state">Loading prediction&hellip;</p>';

    try {
      const [prediction, history, fireHistory] = await Promise.all([
        fetchJSON(`/api/risk/${regionId}`),
        fetchJSON(`/api/risk/${regionId}/history?months=24`),
        fetchJSON(`/api/risk/${regionId}/fire-history`),
      ]);
      state.predictionCache[regionId] = prediction;
      recolorMap();
      renderRiskSummary(prediction);
      renderHorizonBars(prediction);
      renderAttention(prediction);
      renderTrend(history);
      renderFireHistory(fireHistory);
    } catch (e) {
      document.getElementById("risk-summary-body").innerHTML =
        `<p class="empty-state">Could not load prediction: ${e.message}<br/>` +
        `Run <code>scripts/generate_demo_data.py</code> and <code>scripts/train_demo.py</code> ` +
        `on the backend, then reload.</p>`;
    }
  }

  // ----------------------------------------------------------- rendering
  function renderRiskSummary(prediction) {
    const body = document.getElementById("risk-summary-body");
    if (!prediction) {
      body.innerHTML = '<p class="empty-state">No prediction available.</p>';
      return;
    }
    const h = prediction.horizons[state.activeHorizon];
    body.innerHTML = "";
    body.appendChild(
      el("p", {}, [
        el("span", { class: `risk-pill risk-${h.risk_class_id}`, text: h.risk_class }),
      ])
    );
    body.appendChild(
      el("p", { class: "card-sub", text: `As of ${prediction.as_of} · ${h.months}-month horizon` })
    );
    RISK_NAMES.forEach((name, i) => {
      const pct = (h.probabilities[name] * 100).toFixed(1);
      body.appendChild(
        el("div", { class: "prob-row" }, [
          el("span", { class: "label", text: name }),
          el("span", { class: "bar-track" }, [
            el("span", {
              class: "bar-fill",
              style: `width:${pct}%;background:${RISK_COLORS[i]}`,
            }),
          ]),
          el("span", { class: "value", text: `${pct}%` }),
        ])
      );
    });
  }

  function renderHorizonBars(prediction) {
    const container = document.getElementById("horizon-bars");
    container.innerHTML = "";
    if (!prediction) return;
    Object.entries(prediction.horizons).forEach(([key, h]) => {
      const block = el("div", { class: "horizon-block" }, [
        el("h4", {}, [
          el("span", { class: `risk-pill risk-${h.risk_class_id}`, text: `${h.months}mo: ${h.risk_class}` }),
        ]),
      ]);
      RISK_NAMES.forEach((name, i) => {
        const pct = (h.probabilities[name] * 100).toFixed(1);
        block.appendChild(
          el("div", { class: "prob-row" }, [
            el("span", { class: "label", text: name }),
            el("span", { class: "bar-track" }, [
              el("span", { class: "bar-fill", style: `width:${pct}%;background:${RISK_COLORS[i]}` }),
            ]),
            el("span", { class: "value", text: `${pct}%` }),
          ])
        );
      });
      container.appendChild(block);
    });
  }

  function renderAttention(prediction) {
    const channelDiv = document.getElementById("channel-attention");
    channelDiv.innerHTML = "";
    const entries = Object.entries(prediction.channel_attention).sort((a, b) => b[1] - a[1]);
    const maxVal = Math.max(...entries.map(([, v]) => v), 1e-6);
    entries.forEach(([name, val]) => {
      const pct = ((val / maxVal) * 100).toFixed(1);
      channelDiv.appendChild(
        el("div", { class: "attn-row" }, [
          el("span", { class: "label", text: name }),
          el("span", { class: "bar-track" }, [el("span", { class: "bar-fill", style: `width:${pct}%` })]),
          el("span", { class: "value", text: val.toFixed(3) }),
        ])
      );
    });

    const temporalDiv = document.getElementById("temporal-attention");
    temporalDiv.innerHTML = "";
    const top = prediction.temporal_attention.slice(0, 6);
    const maxT = Math.max(...top.map((t) => t.weight), 1e-6);
    top.forEach((t) => {
      const pct = ((t.weight / maxT) * 100).toFixed(1);
      temporalDiv.appendChild(
        el("div", { class: "attn-row" }, [
          el("span", { class: "label", text: t.time }),
          el("span", { class: "bar-track" }, [el("span", { class: "bar-fill", style: `width:${pct}%` })]),
          el("span", { class: "value", text: t.weight.toFixed(3) }),
        ])
      );
    });
  }

  function renderTrend(history) {
    const container = document.getElementById("trend-chart");
    container.innerHTML = "";
    if (!history.length) {
      container.innerHTML = '<p class="empty-state">No history available.</p>';
      return;
    }
    const w = 340;
    const h = 90;
    const pad = 6;
    const stepX = (w - 2 * pad) / Math.max(history.length - 1, 1);
    const yFor = (classId) => h - pad - (classId / 2) * (h - 2 * pad);

    const points = history.map((p, i) => {
      const cls = p.horizons[state.activeHorizon].risk_class_id;
      return [pad + i * stepX, yFor(cls), cls];
    });

    const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);

    const pathEl = document.createElementNS(svgNS, "path");
    pathEl.setAttribute("d", path);
    pathEl.setAttribute("fill", "none");
    pathEl.setAttribute("stroke", "#4f9dde");
    pathEl.setAttribute("stroke-width", "1.5");
    svg.appendChild(pathEl);

    points.forEach(([x, y, cls], i) => {
      const c = document.createElementNS(svgNS, "circle");
      c.setAttribute("cx", x.toFixed(1));
      c.setAttribute("cy", y.toFixed(1));
      c.setAttribute("r", "2.4");
      c.setAttribute("fill", RISK_COLORS[cls]);
      const title = document.createElementNS(svgNS, "title");
      title.textContent = `${history[i].as_of}: ${RISK_NAMES[cls]}`;
      c.appendChild(title);
      svg.appendChild(c);
    });

    container.appendChild(svg);
    const caption = el("p", {
      class: "card-sub",
      text: `${history[0].as_of} → ${history[history.length - 1].as_of}`,
    });
    container.appendChild(caption);
  }

  function renderFireHistory(fireHistory) {
    document.getElementById("fire-history-sub").textContent =
      `${fireHistory.n_fire_events} recorded ignition(s) · ${Math.round(fireHistory.total_acres_burned).toLocaleString()} acres total`;
    const container = document.getElementById("fire-history");
    container.innerHTML = "";
    if (!fireHistory.ignitions.length) {
      container.innerHTML = '<p class="empty-state">No fire events recorded in this dataset.</p>';
      return;
    }
    fireHistory.ignitions
      .slice()
      .reverse()
      .slice(0, 12)
      .forEach((ev) => {
        container.appendChild(
          el("div", { class: "fire-event" }, [
            el("span", { text: ev.time }),
            el("span", { text: `${Math.round(ev.acres).toLocaleString()} ac` }),
          ])
        );
      });
  }

  // ------------------------------------------------------------------ boot
  async function main() {
    initMap();
    initHorizonToggle();
    const ready = await checkHealth();
    if (ready) {
      await loadRegions();
    } else {
      document.getElementById("risk-summary-body").innerHTML =
        '<p class="empty-state">Backend is not ready yet. Run the demo data + training scripts, then reload this page.</p>';
    }
  }

  document.addEventListener("DOMContentLoaded", main);
})();
