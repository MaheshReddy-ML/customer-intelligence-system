const $ = (selector) => document.querySelector(selector);

const form = $("#predict-form");
const healthPill = $("#health-pill");
const clusterId = $("#cluster-id");
const clusterName = $("#cluster-name");
const summary = $("#summary");
const businessAction = $("#business-action");
const modelStatus = $("#model-status");
const probabilityBars = $("#probability-bars");
const segmentList = $("#segment-list");
const kpiRail = $("#kpi-rail");
const saveState = $("#save-state");
const toast = $("#toast");
const confidenceCanvas = $("#confidence-canvas");

const featureKeys = [
  "recency",
  "frequency",
  "monetary_value",
  "total_quantity",
  "avg_unit_price",
];

const segmentPalette = ["#20d6b2", "#79a9ff", "#ff725c", "#f8c95c", "#b58cff"];
let dashboardCache = { kmeans: [], segments: [] };
let predictionTimer = null;
let toastTimer = null;

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 3200);
}

async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeout || 9000);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function setHealth(state, label) {
  healthPill.classList.remove("ready", "offline");
  if (state) healthPill.classList.add(state);
  healthPill.querySelector("strong").textContent = label;
  $("#board-status").textContent = label;
}

function bindInputs() {
  featureKeys.forEach((key) => {
    const range = form.elements[key];
    const number = form.elements[`${key}_number`];

    range.addEventListener("input", () => {
      number.value = range.value;
      queuePrediction();
    });

    number.addEventListener("input", () => {
      const max = Number(number.max);
      const value = Math.min(max, Number(number.value || 0));
      range.value = value;
      queuePrediction();
    });
  });
}

function payloadFromForm() {
  return Object.fromEntries(
    featureKeys.map((key) => [key, Number(form.elements[key].value)])
  );
}

function queuePrediction() {
  saveState.textContent = "Updating";
  clearTimeout(predictionTimer);
  predictionTimer = setTimeout(() => {
    predict().catch(() => {
      saveState.textContent = "Retry needed";
      showToast("Prediction service did not respond. Check the FastAPI server.");
    });
  }, 420);
}

function bestProbability(probabilities) {
  const values = probabilities?.[0] || [];
  return values.length ? Math.max(...values) : 0;
}

function drawConfidence(probabilities) {
  const canvas = confidenceCanvas;
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const values = probabilities?.[0] || [0, 0, 0];
  const max = Math.max(...values, 0.01);

  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  const centerX = rect.width / 2;
  const centerY = rect.height / 2;
  const radius = Math.min(rect.width, rect.height) * 0.36;

  context.lineWidth = 1;
  for (let ring = 1; ring <= 4; ring += 1) {
    context.strokeStyle = `rgba(255,255,255,${0.07 + ring * 0.025})`;
    context.beginPath();
    context.arc(centerX, centerY, (radius / 4) * ring, 0, Math.PI * 2);
    context.stroke();
  }

  values.forEach((value, index) => {
    const start = -Math.PI / 2 + (Math.PI * 2 * index) / values.length;
    const end = start + Math.PI * 2 / values.length - 0.08;
    const outer = radius * (0.4 + (value / max) * 0.72);
    const gradient = context.createLinearGradient(centerX - outer, centerY, centerX + outer, centerY);
    gradient.addColorStop(0, segmentPalette[index % segmentPalette.length]);
    gradient.addColorStop(1, segmentPalette[(index + 1) % segmentPalette.length]);
    context.strokeStyle = gradient;
    context.lineWidth = 20;
    context.lineCap = "round";
    context.beginPath();
    context.arc(centerX, centerY, outer, start, end);
    context.stroke();
  });
}

function renderPrediction(result) {
  const confidence = bestProbability(result.probabilities);

  clusterId.textContent = result.cluster_id;
  clusterName.textContent = result.cluster_name;
  summary.textContent = result.summary;
  businessAction.textContent = result.business_action;
  modelStatus.textContent = result.model_status;
  $("#confidence-score").textContent = `${Math.round(confidence * 100)}%`;
  saveState.textContent = "Synced";

  probabilityBars.innerHTML = "";
  const probabilities = result.probabilities?.[0] || [];
  probabilities.forEach((value, index) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span>Cluster ${index}</span>
      <div class="prob-track"><div class="prob-fill"></div></div>
      <strong>${Math.round(value * 100)}%</strong>
    `;
    probabilityBars.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".prob-fill").style.width = `${Math.max(value * 100, 1.5)}%`;
    });
  });

  drawConfidence(result.probabilities);
}

async function predict() {
  saveState.textContent = "Running";
  const result = await fetchJson("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadFromForm()),
    timeout: 20000,
  });
  renderPrediction(result);
}

function renderKpis(segments, metrics, artifacts) {
  const best = metrics.length
    ? metrics.reduce((winner, item) =>
        Number(item.SilhouetteScore) > Number(winner.SilhouetteScore) ? item : winner
      )
    : null;
  const premium = segments.find((item) => item.ClusterName.includes("Premium")) || segments[0];
  const dormant = segments.find((item) => item.ClusterName.includes("Dormant")) || segments.at(-1);

  $("#board-segment-count").textContent = `${segments.length || "--"} segments`;
  $("#best-k").textContent = best ? best.K : "--";
  $("#best-score").textContent = best ? Number(best.SilhouetteScore).toFixed(3) : "--";
  $("#artifact-state").textContent = artifacts?.embedding_tsne ? "Ready" : "Missing";

  const cards = [
    {
      label: "Segment families",
      value: segments.length || "--",
      detail: "Labeled customer groups available to the API.",
      accent: "var(--teal)",
    },
    {
      label: "Best silhouette",
      value: best ? Number(best.SilhouetteScore).toFixed(3) : "--",
      detail: best ? `Strongest evaluation at K=${best.K}.` : "Metrics unavailable.",
      accent: "var(--coral)",
    },
    {
      label: "Premium frequency",
      value: premium ? Number(premium.Frequency).toFixed(1) : "--",
      detail: premium ? premium.ClusterName : "Segment unavailable.",
      accent: "var(--blue)",
    },
    {
      label: "Dormant recency",
      value: dormant ? Number(dormant.Recency).toFixed(0) : "--",
      detail: dormant ? dormant.ClusterName : "Segment unavailable.",
      accent: "var(--gold)",
    },
  ];

  kpiRail.innerHTML = "";
  cards.forEach((card) => {
    const item = document.createElement("article");
    item.className = "metric-card";
    item.style.setProperty("--accent", card.accent);
    item.innerHTML = `
      <span>${card.label}</span>
      <strong>${card.value}</strong>
      <p>${card.detail}</p>
    `;
    kpiRail.appendChild(item);
  });
}

function renderSegments(segments) {
  segmentList.innerHTML = "";
  segments.forEach((segment, index) => {
    const item = document.createElement("article");
    item.className = "segment-item";
    item.style.animationDelay = `${index * 80}ms`;
    item.style.setProperty("--accent", segmentPalette[index % segmentPalette.length]);
    item.innerHTML = `
      <h3>Cluster ${segment.Cluster}: ${segment.ClusterName}</h3>
      <p>${segment.Summary}</p>
      <div class="stat-row">
        <span>Recency ${Number(segment.Recency).toFixed(1)}</span>
        <span>Frequency ${Number(segment.Frequency).toFixed(1)}</span>
        <span>Value ${Number(segment.MonetaryValue).toFixed(1)}</span>
        <span>Quantity ${Number(segment.TotalQuantity).toFixed(1)}</span>
      </div>
    `;
    segmentList.appendChild(item);
  });
}

function drawMetrics(metrics) {
  const canvas = $("#metrics-chart");
  const context = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;

  if (!metrics.length || rect.width === 0) return;

  canvas.width = rect.width * ratio;
  canvas.height = rect.height * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);

  const padding = { top: 30, right: 34, bottom: 44, left: 52 };
  const width = rect.width - padding.left - padding.right;
  const height = rect.height - padding.top - padding.bottom;
  const maxInertia = Math.max(...metrics.map((item) => Number(item.Inertia)));
  const maxSilhouette = Math.max(...metrics.map((item) => Number(item.SilhouetteScore)));

  context.strokeStyle = "rgba(255,255,255,.11)";
  context.lineWidth = 1;
  context.font = "12px Inter, system-ui";
  context.fillStyle = "rgba(219,229,244,.7)";
  context.textAlign = "right";

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (height / 4) * i;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(rect.width - padding.right, y);
    context.stroke();
    context.fillText(`${Math.round(maxInertia - (maxInertia / 4) * i)}`, padding.left - 10, y + 4);
  }

  metrics.forEach((item, index) => {
    const x = padding.left + (width / (metrics.length - 1)) * index;
    const barHeight = (Number(item.Inertia) / maxInertia) * height;
    const barWidth = Math.max(13, width / metrics.length / 2.4);
    const y = padding.top + height - barHeight;
    const gradient = context.createLinearGradient(0, y, 0, padding.top + height);
    gradient.addColorStop(0, "#20d6b2");
    gradient.addColorStop(0.62, "#79a9ff");
    gradient.addColorStop(1, "rgba(121,169,255,.18)");

    context.fillStyle = gradient;
    context.fillRect(x - barWidth / 2, y, barWidth, barHeight);
    context.fillStyle = "rgba(247,249,252,.82)";
    context.textAlign = "center";
    context.fillText(`K${item.K}`, x, rect.height - 16);
  });

  context.beginPath();
  metrics.forEach((item, index) => {
    const x = padding.left + (width / (metrics.length - 1)) * index;
    const y = padding.top + height - (Number(item.SilhouetteScore) / maxSilhouette) * height;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = "#ff725c";
  context.lineWidth = 3;
  context.stroke();

  metrics.forEach((item, index) => {
    const x = padding.left + (width / (metrics.length - 1)) * index;
    const y = padding.top + height - (Number(item.SilhouetteScore) / maxSilhouette) * height;
    context.fillStyle = "#f8c95c";
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fill();
  });
}

async function loadDashboard() {
  const [health, dashboard] = await Promise.all([
    fetchJson("/api/health"),
    fetchJson("/api/dashboard"),
  ]);

  dashboardCache = dashboard;
  setHealth("ready", health.status);
  renderKpis(dashboard.segments, dashboard.kmeans, dashboard.artifacts);
  renderSegments(dashboard.segments);
  drawMetrics(dashboard.kmeans);
  await predict();
}

async function loadExample(segment = 1) {
  saveState.textContent = "Loading preset";
  const example = await fetchJson(`/api/example?segment=${segment}`);
  featureKeys.forEach((key) => {
    form.elements[key].value = example[key];
    form.elements[`${key}_number`].value = example[key];
  });
  await predict();
}

function animateSignals() {
  const canvas = $("#signal-canvas");
  const context = canvas.getContext("2d");
  let width = 0;
  let height = 0;
  let tick = 0;
  const lanes = Array.from({ length: 18 }, (_, index) => ({
    y: 0.08 + index * 0.052,
    speed: 0.002 + Math.random() * 0.004,
    offset: Math.random(),
    hue: index % 3,
  }));

  function resize() {
    const ratio = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function frame() {
    tick += 1;
    context.clearRect(0, 0, width, height);
    lanes.forEach((lane) => {
      const color = lane.hue === 0 ? "32,214,178" : lane.hue === 1 ? "121,169,255" : "255,114,92";
      const y = lane.y * height + Math.sin(tick * 0.01 + lane.offset * 4) * 16;
      const x = ((lane.offset + tick * lane.speed) % 1.4) * width - width * 0.2;
      const gradient = context.createLinearGradient(x - 180, y, x + 220, y);
      gradient.addColorStop(0, `rgba(${color},0)`);
      gradient.addColorStop(0.5, `rgba(${color},0.18)`);
      gradient.addColorStop(1, `rgba(${color},0)`);
      context.strokeStyle = gradient;
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(x - 180, y);
      context.lineTo(x + 220, y);
      context.stroke();
    });
    requestAnimationFrame(frame);
  }

  resize();
  window.addEventListener("resize", () => {
    resize();
    drawMetrics(dashboardCache.kmeans || []);
    drawConfidence([[0.33, 0.33, 0.34]]);
  });
  frame();
}

bindInputs();
animateSignals();

form.addEventListener("submit", (event) => {
  event.preventDefault();
  predict().catch(() => showToast("Prediction failed. The API may still be starting."));
});

$("#preset-bar").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-segment]");
  if (!button) return;
  loadExample(button.dataset.segment).catch(() => showToast("Could not load that preset."));
});

loadDashboard().catch(() => {
  setHealth("offline", "Offline");
  showToast("FastAPI is offline. Start it with uvicorn src.api:app --reload.");
});
