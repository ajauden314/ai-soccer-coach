
import {
  ObjectDetector,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const MODEL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite";
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const CFG = {
  minScore: 0.35,
  acquireScore: 0.66,
  strongScore: 0.88,
  trackScore: 0.45,
  minCoverage: 0.52,
  edgeThreshold: 14,
  maxEdges: 5500,
  scanWidth: 320,
  minRadius: 5,
  maxRadius: 46,
  nearTorso: 0.8,
  kickSpeed: 2.2,
  cooldown: 1200,
  lostTime: 850,
  maxTrail: 16,
  predictionWindow: 0.3,
  maxCandidates: 6
};

const video = document.getElementById("tactiq-camera-video");
const stage = document.getElementById("tactiq-camera-stage");

if (!video || !stage) {
  throw new Error("TACTIQ: Camera video or camera stage was not found.");
}

const S = {
  detector: null,
  loading: false,
  failed: false,

  ball: null,
  track: null,
  pending: null,
  trail: [],
  lastSeen: 0,

  lastVT: -1,
  lastDetect: 0,
  minInterval: 33,

  nearInfo: null,
  lastKick: 0,
  kicks: [],

  seen: 0,
  tried: 0,

  auto: true,
  voice: true,

  heightCm: null,
  heightUnit: "cm",

  previousBallSpeed: 0,
  previousBallTime: 0
};

// --------------------------------------------------
// STORAGE
// --------------------------------------------------

try {
  S.heightCm = parseFloat(localStorage.getItem("tactiq_height_cm")) || null;
  S.heightUnit = localStorage.getItem("tactiq_height_unit") === "ftin"
    ? "ftin"
    : "cm";
} catch {}

// --------------------------------------------------
// CALIBRATION
// --------------------------------------------------

const M_PER_TORSO = 0.288;
const MPS_TO_MPH = 2.236936;
const MPS_TO_KMH = 3.6;

const toMps = t =>
  S.heightCm ? t * (S.heightCm / 100) * M_PER_TORSO : null;

const toMph = t => {
  const v = toMps(t);
  return v == null ? null : v * MPS_TO_MPH;
};

const toKmh = t => {
  const v = toMps(t);
  return v == null ? null : v * MPS_TO_KMH;
};

const speedLong = t => {
  const mph = toMph(t);
  const kmh = toKmh(t);

  return mph == null
    ? `${t.toFixed(1)} torso-lengths/s (uncalibrated)`
    : `${mph.toFixed(1)} mph (${kmh.toFixed(1)} km/h)`;
};

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

const clamp = (v, min, max) =>
  Math.max(min, Math.min(max, v));

// --------------------------------------------------
// UI
// --------------------------------------------------

const overlay = document.createElement("canvas");
overlay.className =
  "absolute inset-0 w-full h-full pointer-events-none";
overlay.style.transform = "scaleX(-1)";
stage.appendChild(overlay);

const octx = overlay.getContext("2d");

const panel = document.createElement("div");

panel.className =
  "mt-3 rounded-xl border border-emerald-900/40 bg-emerald-500/5 p-3 space-y-2";

panel.innerHTML = `
<div class="flex items-center justify-between">
  <span class="text-[10px] font-bold text-emerald-500">
    BALL &amp; KICK ANALYSIS
  </span>
  <span id="tq-ball-status"
    class="text-[9px] font-mono px-2 py-0.5 rounded-md border">
    IDLE
  </span>
</div>

<div class="grid grid-cols-4 gap-2 text-center">

  <div class="rounded-lg bg-[#111c16] p-2">
    <div class="text-[9px] text-slate-400">Kicks</div>
    <div id="tq-kicks"
      class="text-base font-mono font-bold text-[#00f59b]">0</div>
  </div>

  <div class="rounded-lg bg-[#111c16] p-2">
    <div class="text-[9px] text-slate-400">MPH</div>
    <div id="tq-mph"
      class="text-base font-mono font-bold text-[#00f59b]">--</div>
  </div>

  <div class="rounded-lg bg-[#111c16] p-2">
    <div class="text-[9px] text-slate-400">KM/H</div>
    <div id="tq-kmh"
      class="text-base font-mono font-bold text-[#00f59b]">--</div>
  </div>

  <div class="rounded-lg bg-[#111c16] p-2">
    <div class="text-[9px] text-slate-400">Ball seen</div>
    <div id="tq-rate"
      class="text-base font-mono font-bold text-[#00f59b]">--%</div>
  </div>

</div>

<div class="flex flex-wrap items-center gap-2">

  <label class="text-[10px] text-slate-400">Height</label>

  <input id="tq-height" type="number"
    min="3" max="230" placeholder="170"
    class="w-16 text-[11px] rounded-md px-2 py-1">

  <input id="tq-height-in" type="number"
    min="0" max="11" placeholder="in"
    class="hidden w-14 text-[11px] rounded-md px-2 py-1">

  <select id="tq-height-unit"
    class="text-[11px] rounded-md px-2 py-1">
    <option value="cm">cm</option>
    <option value="ftin">ft / in</option>
  </select>

  <span id="tq-cal-status"
    class="text-[9px] text-amber-500">UNCALIBRATED</span>

</div>

<p id="tq-last"
  class="text-[11px] leading-relaxed text-slate-400">
  Start the camera and keep your body and the ball in frame.
</p>

<div class="flex flex-wrap gap-2">

  <label class="text-[10px] inline-flex items-center gap-1">
    <input id="tq-auto" type="checkbox" checked>
    Post feedback to chat
  </label>

  <label class="text-[10px] inline-flex items-center gap-1">
    <input id="tq-voice" type="checkbox" checked>
    Speak cue
  </label>

  <button id="tq-report" type="button"
    class="text-[10px] rounded-lg px-3 py-1.5 bg-emerald-500/10">
    Session report
  </button>

  <button id="tq-reset" type="button"
    class="text-[10px] rounded-lg px-3 py-1.5 bg-slate-100 dark:bg-[#16241d]">
    Reset
  </button>

</div>

<p class="text-[9px] text-slate-400">
  Shape-based and AI-assisted ball tracking. Speed is estimated
  from 2D camera motion and calibrated using your height.
  Camera angle, perspective, and occlusion affect accuracy.
</p>
`;

const slot = document.getElementById("tactiq-ball-panel-slot");

if (slot) {
  slot.appendChild(panel);
} else {
  const observation = document.getElementById("tactiq-live-observation");

  if (observation?.parentElement) {
    observation.parentElement.insertAdjacentElement("afterend", panel);
  }
}

const $ = id => document.getElementById(id);

$("tq-auto").onchange = e => S.auto = e.target.checked;
$("tq-voice").onchange = e => S.voice = e.target.checked;
$("tq-reset").onclick = resetSession;
$("tq-report").onclick = () => postToChat(sessionReport());

// --------------------------------------------------
// HEIGHT CALIBRATION
// --------------------------------------------------

function readHeightCm() {
  const a = parseFloat($("tq-height").value);

  let cm;

  if (S.heightUnit === "ftin") {
    const inches = parseFloat($("tq-height-in").value) || 0;

    if (!(a >= 3 && a <= 7) || inches < 0 || inches > 11) {
      return null;
    }

    cm = (a * 12 + inches) * 2.54;
  } else {
    cm = a;
  }

  return cm >= 100 && cm <= 230 ? cm : null;
}

function saveHeight() {
  try {
    if (S.heightCm) {
      localStorage.setItem("tactiq_height_cm", String(S.heightCm));
    } else {
      localStorage.removeItem("tactiq_height_cm");
    }
  } catch {}
}

function writeHeightInputs() {
  if (!S.heightCm) {
    $("tq-height").value = "";
    $("tq-height-in").value = "";
    return;
  }

  if (S.heightUnit === "ftin") {
    const total = Math.round(S.heightCm / 2.54);

    $("tq-height").value = Math.floor(total / 12);
    $("tq-height-in").value = total % 12;
  } else {
    $("tq-height").value = Math.round(S.heightCm);
    $("tq-height-in").value = "";
  }
}

function refreshCalibrationUI() {
  const calibrated = !!S.heightCm;

  $("tq-cal-status").textContent =
    calibrated ? "CALIBRATED" : "UNCALIBRATED";

  $("tq-cal-status").className =
    calibrated
      ? "text-[9px] text-emerald-500"
      : "text-[9px] text-amber-500";

  const last = S.kicks[S.kicks.length - 1];

  setSpeedTiles(last ? last.speed : null);
}

function applyHeightUnitUI() {
  const ftin = S.heightUnit === "ftin";

  $("tq-height-unit").value = S.heightUnit;
  $("tq-height-in").classList.toggle("hidden", !ftin);

  $("tq-height").placeholder = ftin ? "ft" : "170";
  $("tq-height").min = ftin ? "3" : "100";
  $("tq-height").max = ftin ? "7" : "230";
}

function onHeightEdited() {
  S.heightCm = readHeightCm();
  saveHeight();
  refreshCalibrationUI();
}

$("tq-height").oninput = onHeightEdited;
$("tq-height-in").oninput = onHeightEdited;

$("tq-height-unit").onchange = e => {
  S.heightUnit = e.target.value === "ftin" ? "ftin" : "cm";

  try {
    localStorage.setItem("tactiq_height_unit", S.heightUnit);
  } catch {}

  applyHeightUnitUI();
  writeHeightInputs();
  refreshCalibrationUI();
};

applyHeightUnitUI();
writeHeightInputs();
refreshCalibrationUI();

function setSpeedTiles(t) {
  const mph = t == null ? null : toMph(t);
  const kmh = t == null ? null : toKmh(t);

  $("tq-mph").textContent = mph == null ? "--" : mph.toFixed(1);
  $("tq-kmh").textContent = kmh == null ? "--" : kmh.toFixed(1);
}

function setStatus(value, kind = "neutral") {
  const el = $("tq-ball-status");

  el.textContent = value;

  const colors = {
    ok: "emerald",
    warn: "amber",
    err: "rose",
    neutral: "slate"
  };

  const c = colors[kind] || colors.neutral;

  el.className =
    `text-[9px] font-mono px-2 py-0.5 rounded-md border border-${c}-500/25 bg-${c}-500/10 text-${c}-500`;
}

// --------------------------------------------------
// CANVAS POSITIONING
// --------------------------------------------------

function fit() {
  const r = overlay.getBoundingClientRect();
  const d = Math.max(1, window.devicePixelRatio || 1);

  if (
    overlay.width !== Math.round(r.width * d) ||
    overlay.height !== Math.round(r.height * d)
  ) {
    overlay.width = Math.round(r.width * d);
    overlay.height = Math.round(r.height * d);

    octx.setTransform(d, 0, 0, d, 0, 0);
  }
}

function cover() {
  const w = overlay.clientWidth;
  const h = overlay.clientHeight;

  const sa = video.videoWidth / video.videoHeight;
  const ta = w / h;

  if (sa > ta) {
    const v = ta / sa;

    return {
      w,
      h,
      fx: x => (x - (1 - v) / 2) / v,
      fy: y => y
    };
  }

  const v = sa / ta;

  return {
    w,
    h,
    fx: x => x,
    fy: y => (y - (1 - v) / 2) / v
  };
}

// --------------------------------------------------
// PLAYER BODY TRACKING
// --------------------------------------------------

function bodyInfo() {
  const lm = window.tactiqLandmarks;

  if (!lm || !video.videoWidth) return null;

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const px = p => ({
    x: p.x * vw,
    y: p.y * vh,
    v: p.visibility ?? 1
  });

  const get = i => lm[i] ? px(lm[i]) : null;

  const ls = get(11);
  const rs = get(12);
  const lh = get(23);
  const rh = get(24);

  if (!ls || !rs || !lh || !rh) return null;

  const la = get(27);
  const ra = get(28);

  if (!la || !ra) return null;

  const sm = {
    x: (ls.x + rs.x) / 2,
    y: (ls.y + rs.y) / 2
  };

  const hm = {
    x: (lh.x + rh.x) / 2,
    y: (lh.y + rh.y) / 2
  };

  const torso = dist(sm, hm);

  if (torso < 10) return null;

  const parts = [];

  const add = (i, radius) => {
    const p = get(i);

    if (p && p.v >= 0.4) {
      parts.push({ p, r: radius });
    }
  };

  add(0, 0.45);
  add(11, 0.2);
  add(12, 0.2);
  add(13, 0.15);
  add(14, 0.15);
  add(23, 0.2);
  add(24, 0.2);
  add(25, 0.14);
  add(26, 0.14);

  return { torso, la, ra, parts };
}

function inAvoid(x, y, body) {
  if (!body) return false;

  return body.parts.some(({ p, r }) =>
    Math.hypot(x - p.x, y - p.y) < r * body.torso
  );
}

function avoidList(body, scale, ox = 0, oy = 0) {
  if (!body) return null;

  return body.parts.map(({ p, r }) => ({
    x: (p.x - ox) * scale,
    y: (p.y - oy) * scale,
    r: r * body.torso * scale
  }));
}

// --------------------------------------------------
// OPTIMIZED SHAPE DETECTOR
// --------------------------------------------------

let GX, GY, MG, ACC;

function houghCircles(gray, W, H, rMin, rMax, avoid) {
  const RN = rMax - rMin + 1;
  const n = W * H;
  const need = n * RN;

  if (!ACC || ACC.length < need) {
    ACC = new Uint16Array(need);
  }

  if (!GX || GX.length < n) {
    GX = new Float32Array(n);
    GY = new Float32Array(n);
    MG = new Float32Array(n);
  }

  ACC.fill(0, 0, need);

  GX.fill(0, 0, n);
  GY.fill(0, 0, n);
  MG.fill(0, 0, n);

  const hist = new Uint32Array(256);

  // Sobel edge detection.
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;

      const gx =
        -gray[i - W - 1] + gray[i - W + 1]
        - 2 * gray[i - 1] + 2 * gray[i + 1]
        - gray[i + W - 1] + gray[i + W + 1];

      const gy =
        -gray[i - W - 1] - 2 * gray[i - W] - gray[i - W + 1]
        + gray[i + W - 1] + 2 * gray[i + W] + gray[i + W + 1];

      GX[i] = gx;
      GY[i] = gy;

      const mag = Math.hypot(gx, gy) / 4;

      MG[i] = mag;
      hist[Math.min(255, mag | 0)]++;
    }
  }

  // Adaptive edge threshold.
  let threshold = CFG.edgeThreshold;
  let total = 0;

  for (let b = 255; b >= 0; b--) {
    total += hist[b];

    if (total > CFG.maxEdges) {
      threshold = Math.max(threshold, b + 1);
      break;
    }
  }

  // Circle voting.
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;

      if (MG[i] < threshold) continue;

      const mag = Math.hypot(GX[i], GY[i]);

      if (!mag) continue;

      const ux = GX[i] / mag;
      const uy = GY[i] / mag;

      for (let r = rMin; r <= rMax; r++) {
        for (const sign of [-1, 1]) {
          const cx = Math.round(x + sign * ux * r);
          const cy = Math.round(y + sign * uy * r);

          if (
            cx < 1 || cy < 1 ||
            cx >= W - 1 || cy >= H - 1
          ) continue;

          const idx = (cy * W + cx) * RN + r - rMin;

          if (ACC[idx] < 65535) ACC[idx]++;
        }
      }
    }
  }

  const voteAt = (cx, cy, k) => {
    if (cx < 0 || cy < 0 || cx >= W || cy >= H) return 0;

    const base = (cy * W + cx) * RN;

    return ACC[base + k]
      + (k > 0 ? ACC[base + k - 1] : 0)
      + (k < RN - 1 ? ACC[base + k + 1] : 0);
  };

  // Find local peaks rather than sorting every possible circle.
  const candidates = [];

  for (let cy = 2; cy < H - 2; cy++) {
    for (let cx = 2; cx < W - 2; cx++) {
      const base = (cy * W + cx) * RN;

      for (let k = 0; k < RN; k++) {
        if (!ACC[base + k]) continue;

        const r = rMin + k;
        const votes = voteAt(cx, cy, k);

        if (votes < 0.28 * 2 * Math.PI * r) continue;

        if (avoid && avoid.some(a =>
          Math.hypot(cx - a.x, cy - a.y) < a.r
        )) continue;

        // Require a local maximum.
        let maximum = true;

        for (let dy = -1; dy <= 1 && maximum; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (!dx && !dy) continue;

            if (voteAt(cx + dx, cy + dy, k) > votes) {
              maximum = false;
              break;
            }
          }
        }

        if (!maximum) continue;

        const q = votes / (2 * Math.PI * r);

        candidates.push({
          cx,
          cy,
          k,
          r,
          q
        });
      }
    }
  }

  candidates.sort((a, b) => b.q - a.q);

  const result = [];

  for (const c of candidates.slice(0, 50)) {
    if (result.some(o =>
      Math.hypot(o.x - c.cx, o.y - c.cy) <
      0.6 * Math.max(o.r, c.r)
    )) continue;

    // Ring coverage verification.
    let hits = 0;
    let samples = 0;

    for (let j = 0; j < 24; j++) {
      const angle = j * Math.PI * 2 / 24;

      const ux = Math.cos(angle);
      const uy = Math.sin(angle);

      let found = false;

      for (let d = -2; d <= 2; d++) {
        const x = Math.round(c.cx + (c.r + d) * ux);
        const y = Math.round(c.cy + (c.r + d) * uy);

        if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) {
          continue;
        }

        const i = y * W + x;

        if (MG[i] < threshold) continue;

        const mag = Math.hypot(GX[i], GY[i]) || 1;

        const radial =
          Math.abs(GX[i] * ux + GY[i] * uy) / mag;

        if (radial > 0.72) {
          found = true;
          break;
        }
      }

      samples++;

      if (found) hits++;
    }

    const coverage = samples ? hits / samples : 0;

    if (coverage < CFG.minCoverage) continue;

    const vote = Math.min(1, c.q);

    const score = 0.4 * vote + 0.6 * coverage;

    result.push({
      x: c.cx,
      y: c.cy,
      r: c.r,
      score,
      coverage
    });

    if (result.length >= CFG.maxCandidates) break;
  }

  return result.sort((a, b) => b.score - a.score);
}

// --------------------------------------------------
// IMAGE PROCESSING
// --------------------------------------------------

const scan = document.createElement("canvas");
const sctx = scan.getContext("2d", {
  willReadFrequently: true
});

let G1 = null;
let G2 = null;

function grabGray(sx, sy, sw, sh, W, H) {
  if (scan.width !== W || scan.height !== H) {
    scan.width = W;
    scan.height = H;
  }

  sctx.drawImage(video, sx, sy, sw, sh, 0, 0, W, H);

  const data = sctx.getImageData(0, 0, W, H).data;
  const n = W * H;

  if (!G1 || G1.length < n) {
    G1 = new Float32Array(n);
    G2 = new Float32Array(n);
  }

  for (let i = 0, p = 0; i < n; i++, p += 4) {
    G1[i] =
      0.299 * data[p] +
      0.587 * data[p + 1] +
      0.114 * data[p + 2];
  }

  G2.set(G1.subarray(0, n));

  // Light blur reduces sensor noise.
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;

      G2[i] = (
        G1[i - W - 1] + G1[i - W] + G1[i - W + 1] +
        G1[i - 1] + G1[i] + G1[i + 1] +
        G1[i + W - 1] + G1[i + W] + G1[i + W + 1]
      ) / 9;
    }
  }

  return G2;
}

// --------------------------------------------------
// GLOBAL SEARCH
// --------------------------------------------------

function searchGlobal(body) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const W = CFG.scanWidth;
  const H = Math.max(60, Math.round(vh * W / vw));

  const scale = W / vw;

  const gray = grabGray(0, 0, vw, vh, W, H);

  let rMin = CFG.minRadius;
  let rMax = CFG.maxRadius;

  if (body) {
    const torso = body.torso * scale;

    rMin = Math.max(4, Math.round(torso * 0.12));
    rMax = Math.min(
      CFG.maxRadius,
      Math.round(torso * 0.45)
    );
  }

  const circles = houghCircles(
    gray,
    W,
    H,
    rMin,
    Math.max(rMin + 2, rMax),
    avoidList(body, scale)
  );

  return circles.map(c => ({
    x: c.x / scale,
    y: c.y / scale,
    r: c.r / scale,
    score: c.score,
    coverage: c.coverage
  }));
}

// --------------------------------------------------
// TRACKED REGION SEARCH
// --------------------------------------------------

function searchRoi(track, prediction, gate, body) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const size = gate + track.r * 1.5;

  const sx = Math.max(0, prediction.x - size);
  const sy = Math.max(0, prediction.y - size);

  const ex = Math.min(vw, prediction.x + size);
  const ey = Math.min(vh, prediction.y + size);

  const sw = ex - sx;
  const sh = ey - sy;

  if (sw < 24 || sh < 24) return [];

  let scale = clamp(16 / track.r, 0.4, 1.6);

  scale = Math.min(scale, 300 / sw, 300 / sh);

  const W = Math.max(16, Math.round(sw * scale));
  const H = Math.max(16, Math.round(sh * scale));

  const gray = grabGray(sx, sy, sw, sh, W, H);

  const rMin = Math.max(3, Math.round(track.r * 0.65 * scale));
  const rMax = Math.max(
    rMin + 2,
    Math.round(track.r * 1.5 * scale)
  );

  return houghCircles(
    gray,
    W,
    H,
    rMin,
    rMax,
    avoidList(body, scale, sx, sy)
  ).map(c => ({
    x: sx + c.x / scale,
    y: sy + c.y / scale,
    r: c.r / scale,
    score: c.score,
    coverage: c.coverage
  }));
}

// --------------------------------------------------
// OPTIONAL MEDIAPIPE OBJECT DETECTOR
// --------------------------------------------------

async function ensureDetector() {
  if (S.detector || S.loading || S.failed) return;

  S.loading = true;

  try {
    const vision = await FilesetResolver.forVisionTasks(WASM);

    const options = {
      runningMode: "VIDEO",
      scoreThreshold: CFG.minScore,
      categoryAllowlist: ["sports ball"],
      maxResults: 3
    };

    try {
      S.detector = await ObjectDetector.createFromOptions(
        vision,
        {
          ...options,
          baseOptions: {
            modelAssetPath: MODEL,
            delegate: "GPU"
          }
        }
      );
    } catch {
      S.detector = await ObjectDetector.createFromOptions(
        vision,
        {
          ...options,
          baseOptions: {
            modelAssetPath: MODEL,
            delegate: "CPU"
          }
        }
      );
    }
  } catch (error) {
    console.warn("MediaPipe ball detector unavailable.", error);
    S.failed = true;
  }

  S.loading = false;
}

function runDetector(now, body) {
  if (!S.detector) return null;

  try {
    const result = S.detector.detectForVideo(video, now);

    let best = null;

    for (const detection of result.detections || []) {
      const bb = detection.boundingBox;
      const score = detection.categories?.[0]?.score ?? 0;

      if (!bb || score < CFG.minScore) continue;

      const ratio = bb.width / bb.height;

      if (ratio < 0.6 || ratio > 1.67) continue;

      const candidate = {
        x: bb.originX + bb.width / 2,
        y: bb.originY + bb.height / 2,
        r: (bb.width + bb.height) / 4,
        score
      };

      if (inAvoid(candidate.x, candidate.y, body)) continue;

      if (!best || score > best.score) {
        best = candidate;
      }
    }

    return best;
  } catch (error) {
    console.warn("MediaPipe frame detection failed.", error);
    return null;
  }
}

// --------------------------------------------------
// CANDIDATE SELECTION
// --------------------------------------------------

function pickBest(candidates, reference, gate) {
  let best = null;

  for (const c of candidates) {
    const d = reference ? dist(c, reference) : 0;

    if (reference && d > gate) continue;

    const adjusted =
      c.score - (reference ? 0.15 * d / gate : 0);

    if (!best || adjusted > best.adjusted) {
      best = { ...c, adjusted };
    }
  }

  return best;
}

// --------------------------------------------------
// BALL TRACKING
// --------------------------------------------------

function registerHit(c, now, restart = false) {
  S.seen++;

  let tr = S.track;

  if (tr && !restart) {
    const dt = Math.max(0.016, (now - tr.t) / 1000);

    const rawVx = (c.x - tr.x) / dt;
    const rawVy = (c.y - tr.y) / dt;

    // Smooth velocity without allowing a single noisy frame
    // to completely change the prediction.
    tr.vx = tr.vx * 0.55 + rawVx * 0.45;
    tr.vy = tr.vy * 0.55 + rawVy * 0.45;

    const px = tr.x + tr.vx * dt;
    const py = tr.y + tr.vy * dt;

    tr.x = 0.8 * c.x + 0.2 * px;
    tr.y = 0.8 * c.y + 0.2 * py;

    tr.r = 0.65 * tr.r + 0.35 * c.r;

    tr.hits++;
    tr.miss = 0;
    tr.t = now;
    tr.score = c.score;
  } else {
    S.track = tr = {
      x: c.x,
      y: c.y,
      r: c.r,
      vx: 0,
      vy: 0,
      t: now,
      hits: 2,
      miss: 0,
      score: c.score
    };

    S.trail = [];
    S.nearInfo = null;
  }

  S.pending = null;

  S.ball = {
    x: tr.x,
    y: tr.y,
    r: tr.r,
    score: c.score,
    t: now,
    stale: false
  };

  S.lastSeen = now;

  S.trail.push(S.ball);

  if (S.trail.length > CFG.maxTrail) {
    S.trail.shift();
  }

  onBall(S.ball, now);
}

function registerMiss(now) {
  if (S.pending && now - S.pending.t > 700) {
    S.pending = null;
  }

  if (!S.track) return;

  S.track.miss++;

  if (S.ball) S.ball.stale = true;

  if (now - S.track.t > CFG.lostTime) {
    S.track = null;
    S.ball = null;
    S.trail = [];
    S.nearInfo = null;
  }
}

// --------------------------------------------------
// MAIN DETECTION
// --------------------------------------------------

function detectBall(now) {
  S.tried++;

  if (!video.videoWidth || !video.videoHeight) return;

  const body = bodyInfo();
  const tr = S.track;

  let prediction = null;
  let gate = 0;

  if (tr) {
    const dt = Math.min(
      CFG.predictionWindow,
      Math.max(0, (now - tr.t) / 1000)
    );

    prediction = {
      x: tr.x + tr.vx * dt,
      y: tr.y + tr.vy * dt
    };

    gate =
      tr.r * 3 +
      Math.hypot(tr.vx, tr.vy) * dt * 0.7 +
      tr.r * 0.8 * tr.miss + 12;
  }

  // First search around the predicted ball location.
  if (tr) {
    const roi = searchRoi(tr, prediction, gate, body);

    let hit = pickBest(roi, prediction, gate);

    if (hit && hit.score >= CFG.trackScore) {
      registerHit(hit, now);
      return;
    }
  }

  // Search the whole camera image when necessary.
  const global = searchGlobal(body);

  let aiBall = null;

  // Run the heavier model less frequently.
  if (
    S.detector &&
    (!tr || tr.miss > 0 || S.tried % 5 === 0)
  ) {
    aiBall = runDetector(now, body);
  }

  // Fuse shape detection and AI detection.
  if (aiBall) {
    let matched = false;

    for (const c of global) {
      if (dist(c, aiBall) < Math.max(c.r, aiBall.r) * 1.2) {
        c.score = Math.min(1, c.score + 0.2);
        matched = true;
      }
    }

    if (!matched && aiBall.score >= 0.5) {
      global.push({
        ...aiBall,
        score: aiBall.score * 0.9
      });
    }
  }

  // Recover a lost ball using the predicted trajectory.
  if (tr) {
    const nearby = pickBest(global, prediction, gate * 2);

    if (nearby && nearby.score >= CFG.trackScore) {
      registerHit(nearby, now);
      return;
    }
  }

  // Require repeated evidence before acquiring a new ball.
  let acquisition = null;

  for (const c of global) {
    if (
      c.score >= CFG.acquireScore &&
      (!acquisition || c.score > acquisition.score)
    ) {
      acquisition = c;
    }
  }

  if (acquisition) {
    const p = S.pending;

    const confirmed =
      acquisition.score >= CFG.strongScore ||
      (
        p &&
        now - p.t < 700 &&
        dist(acquisition, p) <
          Math.max(p.r * 2.5, 24)
      );

    if (confirmed) {
      registerHit(acquisition, now, !!tr);
      return;
    }

    S.pending = {
      ...acquisition,
      t: now
    };
  }

  registerMiss(now);
}

// --------------------------------------------------
// BALL SPEED
// --------------------------------------------------

function ballSpeed(torso) {
  const trail = S.trail;

  if (trail.length < 2 || !torso) return 0;

  const b = trail[trail.length - 1];

  let a = trail[0];

  for (let i = trail.length - 2; i >= 0; i--) {
    if (b.t - trail[i].t >= 70) {
      a = trail[i];
      break;
    }
  }

  const dt = (b.t - a.t) / 1000;

  if (dt <= 0) return 0;

  return dist(a, b) / dt / torso;
}

// --------------------------------------------------
// KICK ANALYSIS
// --------------------------------------------------

function onBall(ball, now) {
  const body = bodyInfo();

  if (!body) return;

  const dl = dist(ball, body.la);
  const dr = dist(ball, body.ra);

  const near = Math.min(dl, dr) / body.torso;

  if (near < CFG.nearTorso) {
    S.nearInfo = {
      t: now,
      foot: dl <= dr ? "left" : "right",
      snap: {
        ...(window.tactiqPoseMeasurements || {})
      },
      torso: body.torso
    };
  }

  const speed = ballSpeed(body.torso);

  const first = S.trail[0];
  const last = S.trail[S.trail.length - 1];

  // Require movement after a recent foot-proximity event.
  if (
    speed > CFG.kickSpeed &&
    S.nearInfo &&
    now - S.nearInfo.t < 500 &&
    now - S.lastKick > CFG.cooldown
  ) {
    // Reject an implausible one-frame speed spike.
    const previous = S.previousBallSpeed;

    if (
      previous > 0 &&
      speed > previous * 4 &&
      now - S.previousBallTime < 100
    ) {
      return;
    }

    S.lastKick = now;

    const dx = last.x - first.x;
    const dy = last.y - first.y;

    const len = Math.hypot(dx, dy) || 1;

    registerKick({
      ...S.nearInfo,
      speed,
      rise: -dy / len,
      time: Date.now()
    });
  }

  S.previousBallSpeed = speed;
  S.previousBallTime = now;
}

// --------------------------------------------------
// COACHING ANALYSIS
// --------------------------------------------------

const fmt = v => v == null ? null : Math.round(v);

function analyzeKick(k) {
  const s = k.snap || {};
  const left = k.foot === "left";

  const kickKnee = fmt(left ? s.leftKnee : s.rightKnee);
  const plantKnee = fmt(left ? s.rightKnee : s.leftKnee);
  const lean = fmt(s.torsoLean);

  const notes = [];

  let cue = "Plant. Strike. Follow.";

  if (s.confidence != null && s.confidence < 60) {
    notes.push(
      "Body landmarks were partly visible. Improve lighting and camera positioning for more reliable measurements."
    );
  }

  if (plantKnee != null) {
    if (plantKnee > 170) {
      notes.push(
        `Your plant-leg knee read about ${plantKnee}°. A slight bend can help with balance.`
      );

      cue = "Soft plant knee.";
    } else if (plantKnee < 110) {
      notes.push(
        `Your plant-leg knee read about ${plantKnee}°. Compare your balance across several attempts.`
      );
    } else {
      notes.push(
        `Plant-leg knee read about ${plantKnee}°.`
      );
    }
  }

  if (kickKnee != null) {
    notes.push(
      `Kicking-leg knee read about ${kickKnee}° near the detected contact period.`
    );
  }

  if (lean != null && Math.abs(lean) > 20) {
    notes.push(
      `Your torso was angled approximately ${Math.abs(lean)}° from vertical. If the ball is traveling too high, try keeping your chest over the ball.`
    );

    cue = "Chest over ball.";
  }

  if (k.rise > 0.6) {
    notes.push(
      "The ball traveled upward in the camera view, suggesting a lofted strike."
    );
  } else if (k.rise < 0.15) {
    notes.push(
      "The ball followed a relatively flat path in the camera view."
    );
  }

  if (!notes.length) {
    notes.push(
      "Kick detected. More visible body measurements are needed for detailed technique feedback."
    );
  }

  return {
    notes,
    cue,
    kickKnee,
    plantKnee,
    lean
  };
}

// --------------------------------------------------
// CHAT INTEGRATION
// --------------------------------------------------

function postToChat(md) {
  try {
    const conv = getActiveConversation();

    if (conv) {
      conv.messages.push({
        role: "model",
        text: md
      });

      conv.updatedAt = Date.now();

      saveConversations();
    }

    appendMessageBubble("model", md, true);
  } catch (error) {
    console.error("TACTIQ chat integration failed:", error);
  }
}

// --------------------------------------------------
// REGISTER KICK
// --------------------------------------------------

function registerKick(k) {
  const result = analyzeKick(k);

  k.result = result;

  S.kicks.push(k);

  const count = S.kicks.length;

  $("tq-kicks").textContent = count;

  setSpeedTiles(k.speed);

  $("tq-last").textContent =
    `Kick #${count} (${k.foot} foot). ${result.notes.join(" ")}`;

  const message = `## Kick #${count} — ${k.foot} foot

**Ball speed:** ${speedLong(k.speed)}

**Plant knee:** ${result.plantKnee ?? "n/a"}°
**Kicking knee:** ${result.kickKnee ?? "n/a"}°
**Torso lean:** ${result.lean == null ? "n/a" : Math.abs(result.lean)}°

${result.notes.map(n => "- " + n).join("\n")}

**Cue for the next rep:** ${result.cue}

*Camera-based 2D estimates. These are practice observations, not a technique grade or injury assessment.*`;

  if (S.auto) {
    postToChat(message);
  }

  if (S.voice && "speechSynthesis" in window) {
    speechSynthesis.cancel();

    speechSynthesis.speak(
      new SpeechSynthesisUtterance(result.cue)
    );
  }
}

// --------------------------------------------------
// SESSION REPORT
// --------------------------------------------------

function sessionReport() {
  const kicks = S.kicks;
  const n = kicks.length;

  if (!n) {
    return "## Session report\n\nNo kicks detected yet. Make sure your ball and full body are visible.";
  }

  const avg = arr =>
    arr.reduce((a, b) => a + b, 0) / arr.length;

  const left = kicks.filter(k => k.foot === "left").length;
  const right = n - left;

  const speeds = kicks.map(k => k.speed);

  const knees = kicks
    .map(k => k.result.plantKnee)
    .filter(v => v != null);

  const half = Math.floor(n / 2);

  let trend = "Record at least four kicks to see a speed trend.";

  if (n >= 4) {
    const first = avg(speeds.slice(0, half));
    const second = avg(speeds.slice(half));

    if (second > first * 1.1) {
      trend = "Later kicks were faster than earlier kicks.";
    } else if (second < first * 0.9) {
      trend = "Later kicks were slower than earlier kicks.";
    } else {
      trend = "Ball speed remained relatively consistent.";
    }
  }

  const detectionRate = S.tried
    ? Math.round(S.seen / S.tried * 100)
    : 0;

  return `## TACTIQ session report

| Metric | Result |
|---|---:|
| Kicks detected | ${n} |
| Left-foot kicks | ${left} |
| Right-foot kicks | ${right} |
| Average speed | ${speedLong(avg(speeds))} |
| Fastest kick | ${speedLong(Math.max(...speeds))} |
| Average plant knee | ${knees.length ? Math.round(avg(knees)) + "°" : "n/a"} |
| Detection rate | ${detectionRate}% |

### Session observations

${trend}

${left && right && Math.min(left, right) / n < 0.25
  ? "Consider including additional practice with your less-used foot."
  : "Both-foot practice can help develop consistency."}

*Camera-based estimates. Results depend on camera angle, lighting, and visibility.*`;
}

// --------------------------------------------------
// RESET
// --------------------------------------------------

function resetSession() {
  Object.assign(S, {
    ball: null,
    trail: [],
    track: null,
    pending: null,
    nearInfo: null,
    kicks: [],
    seen: 0,
    tried: 0,
    lastKick: 0,
    previousBallSpeed: 0,
    previousBallTime: 0
  });

  $("tq-kicks").textContent = "0";
  $("tq-rate").textContent = "--%";
  $("tq-last").textContent =
    "Session reset. Strike the ball to begin.";

  setSpeedTiles(null);
}

// --------------------------------------------------
// DRAW BALL TRACKING
// --------------------------------------------------

function draw() {
  const c = cover();

  octx.clearRect(0, 0, c.w, c.h);

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const point = p => ({
    x: c.fx(p.x / vw) * c.w,
    y: c.fy(p.y / vh) * c.h
  });

  // Ball movement trail.
  if (S.trail.length > 1) {
    octx.lineWidth = 2;
    octx.lineCap = "round";

    for (let i = 1; i < S.trail.length; i++) {
      const a = point(S.trail[i - 1]);
      const b = point(S.trail[i]);

      octx.strokeStyle =
        `rgba(255,212,0,${i / S.trail.length})`;

      octx.beginPath();
      octx.moveTo(a.x, a.y);
      octx.lineTo(b.x, b.y);
      octx.stroke();
    }
  }

  if (!S.ball) return;

  const p = point(S.ball);

  const radius = Math.max(
    10,
    (S.ball.r / vh) * c.h * 1.1
  );

  octx.save();

  octx.shadowColor = "rgba(255,212,0,0.95)";
  octx.shadowBlur = 14;

  octx.strokeStyle = "#ffd400";
  octx.lineWidth = 4;

  if (S.ball.stale) {
    octx.globalAlpha = 0.5;
    octx.setLineDash([6, 5]);
  }

  octx.beginPath();
  octx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  octx.stroke();

  octx.restore();

  // Ball label.
  const body = bodyInfo();

  const mph =
    body && S.heightCm
      ? toMph(ballSpeed(body.torso))
      : null;

  const label =
    mph != null && mph >= 1
      ? `BALL ${Math.round(mph)} mph`
      : "BALL";

  octx.save();

  // Canvas is mirrored by CSS, so reverse the label.
  octx.translate(
    p.x,
    Math.max(12, p.y - radius - 8)
  );

  octx.scale(-1, 1);

  octx.font = "bold 11px monospace";
  octx.textAlign = "center";
  octx.textBaseline = "middle";

  const width = octx.measureText(label).width + 10;

  octx.fillStyle = "rgba(0,0,0,0.75)";
  octx.fillRect(-width / 2, -9, width, 18);

  octx.fillStyle = "#ffd400";
  octx.fillText(label, 0, 0);

  octx.restore();
}

// --------------------------------------------------
// MAIN LOOP
// --------------------------------------------------

function loop() {
  requestAnimationFrame(loop);

  fit();

  const active =
    !!video.srcObject &&
    video.readyState >= 2;

  if (!active) {
    octx.clearRect(
      0,
      0,
      overlay.clientWidth,
      overlay.clientHeight
    );

    if (S.ball || S.track) {
      S.ball = null;
      S.trail = [];
      S.track = null;
      S.pending = null;
      S.nearInfo = null;
    }

    setStatus("IDLE");
    return;
  }

  if (!S.detector && !S.loading && !S.failed) {
    ensureDetector();
  }

  if (video.currentTime === S.lastVT) return;

  S.lastVT = video.currentTime;

  const now = performance.now();

  if (now - S.lastDetect >= S.minInterval) {
    S.lastDetect = now;

    try {
      detectBall(now);
    } catch (error) {
      console.error("TACTIQ detection error:", error);
      setStatus("DETECTION ERROR", "err");
    }

    const elapsed = performance.now() - now;

    // Adapt processing frequency to the device.
    S.minInterval = clamp(elapsed * 1.5, 33, 150);

    $("tq-rate").textContent =
      S.tried
        ? Math.round(S.seen / S.tried * 100) + "%"
        : "--%";

    const live = S.ball && !S.ball.stale;

    setStatus(
      live
        ? "BALL TRACKED"
        : S.pending
          ? "CONFIRMING BALL"
          : "SEARCHING",
      live ? "ok" : "warn"
    );
  }

  draw();
}

// --------------------------------------------------
// START
// --------------------------------------------------

requestAnimationFrame(loop);

console.log("TACTIQ enhanced ball tracking initialized.");
