// TACTIQ add-on: browser-side ball detection + kick/rep analysis + automatic feedback.
// Needs the existing page (video #tactiq-camera-video, window.tactiqPoseMeasurements,
// window.tactiqLandmarks, and the chat helpers appendMessageBubble / getActiveConversation).
//
// Ball detection works two ways:
//   1) SHAPE DETECTOR (always on): finds round outlines in the video using edge detection and a
//      circle Hough transform. It looks only at brightness edges, never at colour, so a ball of
//      ANY colour or pattern is found as long as its outline is round and reasonably visible.
//   2) MediaPipe "sports ball" object detector (optional extra): used when it finds a ball.
// Whichever ball is found gets a YELLOW outline on the camera view.
import { ObjectDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const MODEL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite";
const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MIN_SCORE = 0.3;      // ball confidence threshold (object detector)
const NEAR_TORSOS = 0.8;    // ball counts as "at the foot" within this many torso lengths
const KICK_SPEED = 2.2;     // torso-lengths/second needed to count as a strike
const COOLDOWN_MS = 1200;
const LOST_MS = 700;

// ---------- shape (round object) detector settings ----------
const SCAN_W = 200;         // video is downscaled to this width for shape detection
const R_MIN = 4;            // smallest ball radius searched (scan pixels)
const R_MAX = 38;           // largest ball radius searched (scan pixels)
const EDGE_MIN = 14;        // minimum edge strength (0-255 brightness steps)
const SHAPE_MIN = 0.45;     // how complete the circle outline must be (0-1)

const video = document.getElementById("tactiq-camera-video");
const stage = document.getElementById("tactiq-camera-stage");
const S = { detector: null, loading: false, failed: false, frame: 0, lastVT: -1, ball: null, trail: [], lastSeen: 0,
  nearInfo: null, lastKick: 0, kicks: [], seen: 0, tried: 0, auto: true, voice: true, heightUnit: "cm" };

try { S.heightCm = parseFloat(localStorage.getItem("tactiq_height_cm")) || null; } catch { S.heightCm = null; }
try { S.heightUnit = localStorage.getItem("tactiq_height_unit") === "ftin" ? "ftin" : "cm"; } catch { S.heightUnit = "cm"; }

// ---------- speed calibration ----------
// Shoulder-to-hip distance is roughly 28.8% of standing height, so the player's height turns
// "torso-lengths per second" into approximate metres per second, then miles/hour and km/hour.
const M_PER_TORSO = 0.288;
const MPS_TO_MPH = 2.236936;
const MPS_TO_KMH = 3.6;
const toMps = (t) => (S.heightCm ? t * (S.heightCm / 100) * M_PER_TORSO : null);
const toMph = (t) => { const m = toMps(t); return m == null ? null : m * MPS_TO_MPH; };
const toKmh = (t) => { const m = toMps(t); return m == null ? null : m * MPS_TO_KMH; };
const speedShort = (t) => { const v = toMph(t); return v == null ? t.toFixed(1) : Math.round(v) + " mph"; };
const speedLong = (t) => {
  const mph = toMph(t), kmh = toKmh(t);
  return mph == null ? `${t.toFixed(1)} torso-lengths/s (relative; enter your height to calibrate)`
                     : `${mph.toFixed(1)} mph (${kmh.toFixed(1)} km/h)`;
};

// ---------- UI ----------
const overlay = document.createElement("canvas");
overlay.className = "absolute inset-0 w-full h-full pointer-events-none";
overlay.style.transform = "scaleX(-1)";
stage.appendChild(overlay);
const octx = overlay.getContext("2d");

const panel = document.createElement("div");
panel.className = "mt-3 rounded-xl border border-emerald-900/40 bg-emerald-500/5 p-3 space-y-2";
panel.innerHTML = `
  <div class="flex items-center justify-between">
    <span class="text-[10px] font-bold text-emerald-500">BALL &amp; KICK ANALYSIS</span>
    <span id="tq-ball-status" class="text-[9px] font-mono px-2 py-0.5 rounded-md border border-slate-500/25 bg-slate-500/10 text-slate-500">IDLE</span>
  </div>
  <div class="grid grid-cols-4 gap-2 text-center">
    <div class="rounded-lg border border-emerald-900/30 bg-[#111c16] p-2"><div class="text-[9px] text-slate-400">Kicks</div><div id="tq-kicks" class="text-base font-mono font-bold text-[#00f59b]">0</div></div>
    <div class="rounded-lg border border-emerald-900/30 bg-[#111c16] p-2"><div class="text-[9px] text-slate-400">Last speed</div><div class="flex items-baseline justify-center gap-1"><span id="tq-mph" class="text-base font-mono font-bold text-[#00f59b]">--</span><span class="text-[9px] text-slate-400">mph</span></div></div>
    <div class="rounded-lg border border-emerald-900/30 bg-[#111c16] p-2"><div class="text-[9px] text-slate-400">Last speed</div><div class="flex items-baseline justify-center gap-1"><span id="tq-kmh" class="text-base font-mono font-bold text-[#00f59b]">--</span><span class="text-[9px] text-slate-400">km/h</span></div></div>
    <div class="rounded-lg border border-emerald-900/30 bg-[#111c16] p-2"><div class="text-[9px] text-slate-400">Ball seen</div><div id="tq-rate" class="text-base font-mono font-bold text-[#00f59b]">--%</div></div>
  </div>
  <div class="flex flex-wrap items-center gap-2">
    <label for="tq-height" class="text-[10px] text-slate-500 dark:text-slate-400 shrink-0">Your height</label>
    <input id="tq-height" type="number" min="100" max="230" step="1" placeholder="170" class="w-16 text-[11px] rounded-md px-2 py-1 bg-slate-50 dark:bg-[#0b130f] border border-slate-200 dark:border-emerald-900/70 text-slate-800 dark:text-slate-100 focus:outline-none focus:border-emerald-500">
    <input id="tq-height-in" type="number" min="0" max="11" step="1" placeholder="in" class="hidden w-14 text-[11px] rounded-md px-2 py-1 bg-slate-50 dark:bg-[#0b130f] border border-slate-200 dark:border-emerald-900/70 text-slate-800 dark:text-slate-100 focus:outline-none focus:border-emerald-500">
    <select id="tq-height-unit" class="text-[11px] rounded-md px-1.5 py-1 bg-slate-50 dark:bg-[#0b130f] border border-slate-200 dark:border-emerald-900/70 text-slate-800 dark:text-slate-100 focus:outline-none focus:border-emerald-500">
      <option value="cm">cm</option>
      <option value="ftin">ft / in</option>
    </select>
    <span id="tq-cal-status" class="text-[9px] font-mono text-amber-500">UNCALIBRATED</span>
  </div>
  <p id="tq-last" class="text-[11px] leading-relaxed text-slate-600 dark:text-slate-300">Start the camera, keep your full body and the ball in frame, then strike the ball. A yellow ring shows when the ball is detected, and each kick is analyzed automatically.</p>
  <div class="flex flex-wrap gap-2">
    <label class="text-[10px] text-slate-500 dark:text-slate-400 inline-flex items-center gap-1"><input id="tq-auto" type="checkbox" checked> Post feedback to chat</label>
    <label class="text-[10px] text-slate-500 dark:text-slate-400 inline-flex items-center gap-1"><input id="tq-voice" type="checkbox" checked> Speak cue</label>
    <button id="tq-report" type="button" class="text-[10px] font-semibold rounded-lg px-3 py-1.5 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/25">Session report</button>
    <button id="tq-reset" type="button" class="text-[10px] font-semibold rounded-lg px-3 py-1.5 bg-slate-100 dark:bg-[#16241d] border border-slate-200 dark:border-emerald-950 text-slate-600 dark:text-slate-300">Reset</button>
  </div>
  <p class="text-[9px] text-slate-400 leading-relaxed">The ball is found by its round outline, not its colour, so any colour of ball works. It can still miss a ball that blends into the background or confuse other round objects. Enter your height to convert speed to approximate mph and km/h. Speeds are 2D estimates measured across the camera view, so kicks toward or away from the camera read low.</p>`;
const slot = document.getElementById("tactiq-ball-panel-slot");
if (slot) slot.appendChild(panel);
else document.getElementById("tactiq-live-observation").parentElement.insertAdjacentElement("afterend", panel);
const $ = (id) => document.getElementById(id);
$("tq-auto").onchange = (e) => (S.auto = e.target.checked);
$("tq-voice").onchange = (e) => (S.voice = e.target.checked);
$("tq-reset").onclick = resetSession;

function setSpeedTiles(t) {
  const mph = t == null ? null : toMph(t), kmh = t == null ? null : toKmh(t);
  $("tq-mph").textContent = mph == null ? "--" : mph.toFixed(1);
  $("tq-kmh").textContent = kmh == null ? "--" : kmh.toFixed(1);
}
function refreshCalibrationUI() {
  const cal = !!S.heightCm;
  $("tq-cal-status").textContent = cal ? "CALIBRATED" : "UNCALIBRATED";
  $("tq-cal-status").className = "text-[9px] font-mono " + (cal ? "text-emerald-500" : "text-amber-500");
  const last = S.kicks[S.kicks.length - 1];
  setSpeedTiles(last ? last.speed : null);
}

// ---------- height input (cm or ft/in) ----------
function readHeightCm() {
  const a = parseFloat($("tq-height").value);
  let cm;
  if (S.heightUnit === "ftin") {
    const inches = parseFloat($("tq-height-in").value) || 0;
    if (!(a >= 0)) return null;
    cm = (a * 12 + inches) * 2.54;
  } else {
    cm = a;
  }
  return cm >= 100 && cm <= 230 ? cm : null;
}
function applyHeightUnitUI() {
  const ftin = S.heightUnit === "ftin";
  $("tq-height-unit").value = S.heightUnit;
  $("tq-height-in").classList.toggle("hidden", !ftin);
  $("tq-height").placeholder = ftin ? "ft" : "170";
  $("tq-height").min = ftin ? "3" : "100";
  $("tq-height").max = ftin ? "7" : "230";
  $("tq-height").step = "1";
}
function writeHeightInputs() {
  if (!S.heightCm) { $("tq-height").value = ""; $("tq-height-in").value = ""; return; }
  if (S.heightUnit === "ftin") {
    const total = Math.round(S.heightCm / 2.54);
    $("tq-height").value = Math.floor(total / 12);
    $("tq-height-in").value = total % 12;
  } else {
    $("tq-height").value = Math.round(S.heightCm);
    $("tq-height-in").value = "";
  }
}
function saveHeight() {
  try { S.heightCm ? localStorage.setItem("tactiq_height_cm", String(S.heightCm)) : localStorage.removeItem("tactiq_height_cm"); } catch {}
}
function onHeightEdited() {
  S.heightCm = readHeightCm();
  saveHeight();
  refreshCalibrationUI();
}
$("tq-height").oninput = onHeightEdited;
$("tq-height-in").oninput = onHeightEdited;
$("tq-height-unit").onchange = (e) => {
  S.heightUnit = e.target.value === "ftin" ? "ftin" : "cm";
  try { localStorage.setItem("tactiq_height_unit", S.heightUnit); } catch {}
  applyHeightUnitUI();
  writeHeightInputs();
  refreshCalibrationUI();
};
applyHeightUnitUI();
writeHeightInputs();
refreshCalibrationUI();
$("tq-report").onclick = () => postToChat(sessionReport());

function setStatus(text, kind) {
  const el = $("tq-ball-status");
  el.textContent = text;
  const c = { ok: "emerald", warn: "amber", err: "rose" }[kind] || "slate";
  el.className = `text-[9px] font-mono px-2 py-0.5 rounded-md border border-${c}-500/25 bg-${c}-500/10 text-${c}-500`;
}

// ---------- helpers ----------
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function cover() {
  const w = overlay.clientWidth, h = overlay.clientHeight;
  const sa = (video.videoWidth || w) / (video.videoHeight || h), ta = w / h;
  if (sa > ta) { const v = ta / sa; return { w, h, fx: (x) => (x - (1 - v) / 2) / v, fy: (y) => y }; }
  const v = sa / ta; return { w, h, fx: (x) => x, fy: (y) => (y - (1 - v) / 2) / v };
}
function fit() {
  const r = overlay.getBoundingClientRect(), d = Math.max(1, devicePixelRatio || 1);
  if (overlay.width !== Math.round(r.width * d) || overlay.height !== Math.round(r.height * d)) {
    overlay.width = Math.round(r.width * d); overlay.height = Math.round(r.height * d);
    octx.setTransform(d, 0, 0, d, 0, 0);
  }
}
// Pose landmarks in video pixels, plus torso length (shoulder midpoint to hip midpoint).
function bodyInfo() {
  const lm = window.tactiqLandmarks;
  if (!lm) return null;
  const vw = video.videoWidth, vh = video.videoHeight;
  const px = (p) => ({ x: p.x * vw, y: p.y * vh, v: p.visibility ?? 1 });
  const [ls, rs, lh, rh, la, ra] = [11, 12, 23, 24, 27, 28].map((i) => px(lm[i]));
  const nose = lm[0] ? px(lm[0]) : null;
  const sm = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 }, hm = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 };
  const torso = dist(sm, hm);
  return torso > 10 ? { torso, la, ra, nose } : null;
}

// ---------- shape-based (colour-independent) ball detector ----------
// <hough>
let H_ACC = null;
// gray: Float32Array (W*H) of brightness values 0-255.
// Finds the most complete circle outline with radius between rMin and rMax (pixels).
// avoid: optional {x,y,r} circle to ignore (e.g. the player's head). prev: optional last ball {x,y,r}.
function houghBall(gray, W, H, rMin, rMax, avoid, prev) {
  const RN = rMax - rMin + 1;
  const need = W * H * RN;
  if (!H_ACC || H_ACC.length !== need) H_ACC = new Uint16Array(need); else H_ACC.fill(0);
  const acc = H_ACC;
  const gxA = new Float32Array(W * H), gyA = new Float32Array(W * H);
  let sum = 0, cnt = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const gx = -gray[i - W - 1] + gray[i - W + 1] - 2 * gray[i - 1] + 2 * gray[i + 1] - gray[i + W - 1] + gray[i + W + 1];
      const gy = -gray[i - W - 1] - 2 * gray[i - W] - gray[i - W + 1] + gray[i + W - 1] + 2 * gray[i + W] + gray[i + W + 1];
      gxA[i] = gx; gyA[i] = gy;
      sum += Math.hypot(gx, gy) / 4; cnt++;
    }
  }
  const thr = Math.max(EDGE_MIN, (sum / Math.max(1, cnt)) * 2);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const hyp = Math.hypot(gxA[i], gyA[i]);
      if (hyp / 4 < thr) continue;
      const ux = gxA[i] / hyp, uy = gyA[i] / hyp;
      for (let r = rMin; r <= rMax; r++) {
        for (let s = -1; s <= 1; s += 2) {
          const cx = Math.round(x + s * ux * r), cy = Math.round(y + s * uy * r);
          if (cx < 0 || cy < 0 || cx >= W || cy >= H) continue;
          acc[(cy * W + cx) * RN + (r - rMin)]++;
        }
      }
    }
  }
  const s3 = (cx, cy, k) => {
    if (cx < 0 || cy < 0 || cx >= W || cy >= H) return 0;
    const b = (cy * W + cx) * RN;
    return acc[b + k] + (k > 0 ? acc[b + k - 1] : 0) + (k < RN - 1 ? acc[b + k + 1] : 0);
  };
  const cands = [];
  for (let cy = 0; cy < H; cy++) {
    for (let cx = 0; cx < W; cx++) {
      const base = (cy * W + cx) * RN;
      for (let k = 0; k < RN; k++) {
        if (acc[base + k] === 0) continue;
        const r = rMin + k, v = s3(cx, cy, k);
        if (v < 0.25 * 2 * Math.PI * r) continue;
        if (avoid && Math.hypot(cx - avoid.x, cy - avoid.y) < avoid.r) continue;
        cands.push({ cx, cy, k, r, q: v / (2 * Math.PI * r) });
      }
    }
  }
  if (!cands.length) return null;
  cands.sort((a, b) => b.q - a.q);
  let best = null;
  for (const c of cands.slice(0, 16)) {
    let tot = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) tot += s3(c.cx + dx, c.cy + dy, c.k);
    let score = Math.min(1.2, tot / (2 * Math.PI * c.r));
    if (prev && Math.hypot(c.cx - prev.x, c.cy - prev.y) < Math.max(6, prev.r * 3)) score *= 1.25;
    if (!best || score > best.score) best = { x: c.cx, y: c.cy, r: c.r, score };
  }
  return best && best.score >= SHAPE_MIN ? best : null;
}
// </hough>

const scan = document.createElement("canvas");
const sctx = scan.getContext("2d", { willReadFrequently: true });
let G1 = null, G2 = null;

function detectShape() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;
  const W = SCAN_W, H = Math.max(40, Math.round((vh * W) / vw)), scale = W / vw;
  if (scan.width !== W || scan.height !== H) { scan.width = W; scan.height = H; }
  if (!G1 || G1.length !== W * H) { G1 = new Float32Array(W * H); G2 = new Float32Array(W * H); }
  sctx.drawImage(video, 0, 0, W, H);
  const d = sctx.getImageData(0, 0, W, H).data;
  for (let i = 0, p = 0; i < W * H; i++, p += 4) G1[i] = 0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2];
  // light 3x3 blur to suppress noise before edge detection
  G2.set(G1);
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    G2[i] = (G1[i - W - 1] + G1[i - W] + G1[i - W + 1] + G1[i - 1] + G1[i] + G1[i + 1] + G1[i + W - 1] + G1[i + W] + G1[i + W + 1]) / 9;
  }
  let rMin = R_MIN, rMax = R_MAX, avoid = null;
  const body = bodyInfo();
  if (body) {
    // A ball is much smaller than the player's torso; the head is round too, so ignore it.
    const tsc = body.torso * scale;
    rMin = Math.max(3, Math.round(tsc * 0.08));
    rMax = Math.min(R_MAX, Math.round(tsc * 0.5));
    if (rMax < rMin + 2) { rMin = R_MIN; rMax = R_MAX; }
    if (body.nose) avoid = { x: body.nose.x * scale, y: body.nose.y * scale, r: tsc * 0.45 };
  }
  const prev = S.ball ? { x: S.ball.x * scale, y: S.ball.y * scale, r: S.ball.r * scale } : null;
  const hit = houghBall(G2, W, H, rMin, rMax, avoid, prev);
  return hit ? { x: hit.x / scale, y: hit.y / scale, r: hit.r / scale, score: Math.min(1, hit.score), shape: true } : null;
}

// ---------- detection loop ----------
async function ensureDetector() {
  if (S.detector || S.loading || S.failed) return;
  S.loading = true;
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM);
    const opts = { runningMode: "VIDEO", scoreThreshold: MIN_SCORE, categoryAllowlist: ["sports ball"], maxResults: 3 };
    try { S.detector = await ObjectDetector.createFromOptions(vision, { ...opts, baseOptions: { modelAssetPath: MODEL, delegate: "GPU" } }); }
    catch { S.detector = await ObjectDetector.createFromOptions(vision, { ...opts, baseOptions: { modelAssetPath: MODEL, delegate: "CPU" } }); }
  } catch (e) { console.warn("Optional ball object model failed; using shape detector only.", e); S.failed = true; }
  S.loading = false;
}

function detectBall(now) {
  S.tried++;
  let best = null;
  if (S.detector) {
    const res = S.detector.detectForVideo(video, now);
    for (const d of res.detections || []) {
      const bb = d.boundingBox, sc = d.categories?.[0]?.score ?? 0;
      if (!bb || sc < MIN_SCORE) continue;
      // A ball is roughly square; reject very elongated boxes.
      const ar = bb.width / bb.height; if (ar < 0.5 || ar > 2) continue;
      if (!best || sc > best.score) best = { x: bb.originX + bb.width / 2, y: bb.originY + bb.height / 2, r: (bb.width + bb.height) / 4, score: sc };
    }
  }
  // Colour-independent fallback: any round object found by its outline.
  if (!best) best = detectShape();
  if (best) {
    S.seen++;
    if (S.ball) { best.x = S.ball.x * 0.3 + best.x * 0.7; best.y = S.ball.y * 0.3 + best.y * 0.7; best.r = S.ball.r * 0.5 + best.r * 0.5; } // light smoothing
    best.t = now; S.ball = best; S.lastSeen = now; S.trail.push(best);
    if (S.trail.length > 12) S.trail.shift();
    onBall(best, now);
  } else if (now - S.lastSeen > LOST_MS) { S.ball = null; S.trail = []; S.nearInfo = null; }
}

function ballSpeed(torso) {
  const t = S.trail; if (t.length < 2) return 0;
  const b = t[t.length - 1]; let a = t[0];
  for (let i = t.length - 2; i >= 0; i--) { if (b.t - t[i].t >= 70) { a = t[i]; break; } }
  const dt = (b.t - a.t) / 1000; if (dt <= 0) return 0;
  return dist(a, b) / dt / torso;
}

function onBall(ball, now) {
  const body = bodyInfo(); if (!body) return;
  const dl = dist(ball, body.la), dr = dist(ball, body.ra);
  const near = Math.min(dl, dr) / body.torso;
  if (near < NEAR_TORSOS) S.nearInfo = { t: now, foot: dl <= dr ? "left" : "right", snap: { ...window.tactiqPoseMeasurements }, torso: body.torso };
  const speed = ballSpeed(body.torso);
  const a = S.trail[0], b = S.trail[S.trail.length - 1];
  if (speed > KICK_SPEED && S.nearInfo && now - S.nearInfo.t < 500 && now - S.lastKick > COOLDOWN_MS) {
    S.lastKick = now;
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
    registerKick({ ...S.nearInfo, speed, rise: -dy / len, time: Date.now() });
  }
}

// ---------- feedback ----------
const fmt = (v) => (v == null ? null : Math.round(v));
function analyzeKick(k) {
  const s = k.snap || {}, left = k.foot === "left";
  const kickKnee = fmt(left ? s.leftKnee : s.rightKnee), plantKnee = fmt(left ? s.rightKnee : s.leftKnee);
  const lean = fmt(s.torsoLean), vis = s.confidence;
  const notes = []; let cue = "Plant. Strike. Follow.";
  if (vis != null && vis < 60) notes.push("Body landmarks were only partly visible, so the joint readings for this kick are low-confidence. Improve lighting and keep your whole body in frame.");
  if (plantKnee != null) {
    if (plantKnee > 170) { notes.push(`Your plant-leg knee read about ${plantKnee}° (almost straight). A slight bend usually helps balance and control.`); cue = "Soft plant knee."; }
    else if (plantKnee < 110) { notes.push(`Your plant-leg knee read about ${plantKnee}° (deeply bent). If the kick felt unstable, try a slightly taller plant.`); }
    else notes.push(`Plant-leg knee read about ${plantKnee}°, a moderate bend.`);
  }
  if (kickKnee != null) notes.push(`Kicking-leg knee read about ${kickKnee}° at the closest frame to contact; compare this across several kicks, since a single frame is noisy.`);
  if (lean != null && Math.abs(lean) > 20) { notes.push(`Your torso was angled about ${Math.abs(lean)}° from vertical. If the ball is flying higher than you want, work on keeping your chest over the ball.`); cue = "Chest over ball."; }
  if (k.rise > 0.6) notes.push("The ball left on a rising path in the camera view, which suggests a lofted strike. If you wanted a low drive, strike the middle of the ball and stay over it.");
  else if (k.rise < 0.15) notes.push("The ball left on a fairly flat path in the camera view.");
  if (!notes.length) notes.push("Kick detected, but there were no body measurements for this frame.");
  return { notes, cue, kickKnee, plantKnee, lean };
}

function registerKick(k) {
  const r = analyzeKick(k); k.result = r; S.kicks.push(k);
  $("tq-kicks").textContent = S.kicks.length;
  setSpeedTiles(k.speed);
  $("tq-last").textContent = `Kick #${S.kicks.length} (${k.foot} foot). ${r.notes.join(" ")}`;
  const md = `## Kick #${S.kicks.length} — ${k.foot} foot

**Ball speed:** ${speedLong(k.speed)}  
**Plant knee:** ${r.plantKnee ?? "n/a"}° · **Kicking knee:** ${r.kickKnee ?? "n/a"}° · **Torso lean:** ${r.lean == null ? "n/a" : Math.abs(r.lean)}°

${r.notes.map((n) => "- " + n).join("\n")}

**Cue for the next rep:** ${r.cue}

*2D camera estimates, not a technique grade or injury assessment.*`;
  if (S.auto) postToChat(md);
  if (S.voice && "speechSynthesis" in window) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(r.cue)); }
}

function sessionReport() {
  const n = S.kicks.length;
  if (!n) return "## Session report\n\nNo kicks have been detected yet. Make sure the ball and your whole body are in frame, then strike the ball.";
  const avg = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const l = S.kicks.filter((k) => k.foot === "left").length, rt = n - l;
  const sp = S.kicks.map((k) => k.speed), pk = S.kicks.map((k) => k.result.plantKnee).filter((v) => v != null);
  const half = Math.floor(n / 2);
  const trend = n >= 4 ? (avg(sp.slice(half)) > avg(sp.slice(0, half)) * 1.1 ? "Later kicks were faster than earlier ones." : avg(sp.slice(half)) < avg(sp.slice(0, half)) * 0.9 ? "Later kicks were slower, which can mean fatigue; consider a rest." : "Ball speed stayed fairly consistent.") : "Record at least 4 kicks to see a trend.";
  return `## Session report

| Metric | Value |
|---|---:|
| Kicks detected | ${n} (L ${l} / R ${rt}) |
| Average ball speed | ${speedLong(avg(sp))} |
| Fastest | ${speedLong(Math.max(...sp))} |
| Average plant knee | ${pk.length ? Math.round(avg(pk)) + "°" : "n/a"} |
| Ball detected in frames | ${S.tried ? Math.round((S.seen / S.tried) * 100) : 0}% |

${trend}${l && rt && Math.min(l, rt) / n < 0.25 ? " You used one foot far more than the other; add reps with your weaker foot." : ""}

These are camera-based estimates and should be treated as practice guidance only.`;
}

function postToChat(md) {
  try {
    const conv = getActiveConversation();
    if (conv) { conv.messages.push({ role: "model", text: md }); conv.updatedAt = Date.now(); saveConversations(); }
    appendMessageBubble("model", md, true);
  } catch (e) { console.error("Could not post to chat", e); }
}

function resetSession() {
  Object.assign(S, { ball: null, trail: [], nearInfo: null, kicks: [], seen: 0, tried: 0, lastKick: 0 });
  $("tq-kicks").textContent = "0"; setSpeedTiles(null); $("tq-rate").textContent = "--%";
  $("tq-last").textContent = "Session reset. Strike the ball to begin.";
}

// ---------- main loop ----------
function loop() {
  requestAnimationFrame(loop);
  fit();
  const active = !!video.srcObject && video.readyState >= 2;
  if (!active) { octx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight); if (S.ball) { S.ball = null; S.trail = []; } setStatus("IDLE", "neutral"); return; }
  // The optional object model loads in the background; the shape detector works immediately.
  if (!S.detector && !S.loading && !S.failed) ensureDetector();
  if (video.currentTime === S.lastVT) return;
  S.lastVT = video.currentTime;
  if (++S.frame % 2 === 0) {
    try { detectBall(performance.now()); } catch (e) { console.error("Ball detection error", e); }
    $("tq-rate").textContent = S.tried ? Math.round((S.seen / S.tried) * 100) + "%" : "--%";
    setStatus(S.ball ? "BALL TRACKED" : "SEARCHING FOR BALL", S.ball ? "ok" : "warn");
  }
  draw();
}

function draw() {
  const c = cover(); octx.clearRect(0, 0, c.w, c.h);
  const vw = video.videoWidth, vh = video.videoHeight;
  const pt = (p) => ({ x: c.fx(p.x / vw) * c.w, y: c.fy(p.y / vh) * c.h });
  if (S.trail.length > 1) {
    octx.lineWidth = 2; octx.lineCap = "round";
    for (let i = 1; i < S.trail.length; i++) {
      const a = pt(S.trail[i - 1]), b = pt(S.trail[i]);
      octx.strokeStyle = `rgba(255,212,0,${i / S.trail.length})`;
      octx.beginPath(); octx.moveTo(a.x, a.y); octx.lineTo(b.x, b.y); octx.stroke();
    }
  }
  if (S.ball) {
    const p = pt(S.ball), r = Math.max(10, (S.ball.r / vh) * c.h * 1.1);
    // Bright yellow outline with a soft glow so the player can see the ball is detected.
    octx.save();
    octx.shadowColor = "rgba(255,212,0,0.95)"; octx.shadowBlur = 14;
    octx.strokeStyle = "#ffd400"; octx.lineWidth = 4;
    octx.beginPath(); octx.arc(p.x, p.y, r, 0, Math.PI * 2); octx.stroke();
    octx.restore();
    octx.strokeStyle = "rgba(0,0,0,0.55)"; octx.lineWidth = 1;
    octx.beginPath(); octx.arc(p.x, p.y, r + 2.5, 0, Math.PI * 2); octx.stroke();
    // Label (the canvas is mirrored by CSS, so flip the text back so it reads correctly).
    const body = bodyInfo();
    const mph = body && S.heightCm ? toMph(ballSpeed(body.torso)) : null;
    const label = mph != null && mph >= 1 ? `BALL ${Math.round(mph)} mph` : "BALL";
    octx.save();
    octx.translate(p.x, Math.max(12, p.y - r - 8)); octx.scale(-1, 1);
    octx.font = "bold 11px 'JetBrains Mono', monospace"; octx.textAlign = "center"; octx.textBaseline = "middle";
    const tw = octx.measureText(label).width + 10;
    octx.fillStyle = "rgba(0,0,0,0.7)"; octx.fillRect(-tw / 2, -9, tw, 18);
    octx.fillStyle = "#ffd400"; octx.fillText(label, 0, 0);
    octx.restore();
  }
}

requestAnimationFrame(loop);
