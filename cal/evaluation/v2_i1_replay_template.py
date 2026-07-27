"""Self-contained HTML template for the deterministic I1 replay."""

from __future__ import annotations

import json
from typing import Any


_PAYLOAD_MARKER = "__I1_REPLAY_PAYLOAD__"

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>I1 V4 Calibration Replay</title>
  <style>
    :root {
      --bg: #f4f1e9;
      --panel: rgba(255,255,255,.78);
      --ink: #17221f;
      --muted: #66716c;
      --line: #c9cec7;
      --accent: #126d5c;
      --accent-soft: #d8ede7;
      --truth: #d05842;
      --a: #3379c5;
      --b: #9b55b7;
      --wall: #5f6865;
      --seen: #f0bb4f;
      --shadow: 0 14px 36px rgba(28,42,37,.10);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111816;
        --panel: rgba(27,37,34,.92);
        --ink: #eef4f0;
        --muted: #a6b2ad;
        --line: #42504b;
        --accent: #72d7c2;
        --accent-soft: #203e37;
        --truth: #ff8a71;
        --a: #75b3ff;
        --b: #d497ef;
        --wall: #8d9995;
        --seen: #ffd071;
        --shadow: 0 14px 36px rgba(0,0,0,.26);
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 12% -10%, color-mix(in srgb, var(--accent) 12%, transparent), transparent 36rem),
        var(--bg);
      color: var(--ink);
      font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { width: min(1380px, calc(100% - 32px)); margin: 28px auto 56px; }
    header { display: grid; grid-template-columns: 1fr auto; gap: 20px; align-items: end; margin-bottom: 18px; }
    h1 { margin: 0 0 5px; font-size: clamp(25px, 4vw, 42px); letter-spacing: -.035em; }
    header p { margin: 0; color: var(--muted); max-width: 760px; }
    .badge {
      border: 1px solid var(--accent);
      color: var(--accent);
      border-radius: 999px;
      padding: 7px 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .notice {
      padding: 11px 14px;
      border-left: 4px solid var(--truth);
      background: color-mix(in srgb, var(--truth) 8%, var(--panel));
      border-radius: 8px;
      margin-bottom: 16px;
    }
    .controlbar, .card, .metrics, details {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }
    .controlbar {
      display: grid;
      grid-template-columns: auto auto minmax(180px, 1fr) minmax(220px, .8fr) minmax(220px, .9fr);
      gap: 10px;
      align-items: center;
      padding: 13px;
      border-radius: 12px;
      position: sticky;
      top: 8px;
      z-index: 4;
    }
    button, select, input { font: inherit; }
    button, select {
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
    }
    button { padding: 0 13px; cursor: pointer; font-weight: 700; }
    button:hover { border-color: var(--accent); color: var(--accent); }
    button:focus-visible, select:focus-visible, input:focus-visible {
      outline: 3px solid color-mix(in srgb, var(--accent) 34%, transparent);
      outline-offset: 2px;
    }
    select { padding: 0 10px; width: 100%; }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    .stepbox { display: grid; grid-template-columns: minmax(100px, 1fr) 72px; gap: 8px; align-items: center; }
    output { color: var(--accent); font-variant-numeric: tabular-nums; font-weight: 800; text-align: right; }
    .condition-note { margin: 12px 2px 15px; color: var(--muted); min-height: 24px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 16px;
    }
    .metric { padding: 13px 15px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric span { display: block; color: var(--muted); font-size: 12px; }
    .metric strong { display: block; font-size: 25px; font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
    .boards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    .card { border-radius: 12px; overflow: hidden; }
    .card-head { padding: 12px 14px 8px; }
    .card h2 { margin: 0; font-size: 17px; }
    .card p { margin: 2px 0 0; color: var(--muted); font-size: 12px; min-height: 38px; }
    .canvas-wrap { padding: 0 10px 10px; }
    canvas {
      display: block;
      width: 100%;
      aspect-ratio: 1;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: color-mix(in srgb, var(--panel) 85%, transparent);
    }
    .readout {
      display: grid;
      grid-template-columns: 1.2fr 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }
    .readout section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 13px 15px;
    }
    .readout h3 { font-size: 13px; margin: 0 0 7px; color: var(--muted); }
    .readout p { margin: 2px 0; }
    code { color: var(--accent); }
    .posterior, .hypotheses { display: flex; height: 10px; gap: 2px; margin-top: 9px; }
    .posterior i, .hypotheses i { display: block; min-width: 2px; border-radius: 2px; background: var(--accent); }
    .hypotheses i { background: var(--a); }
    details { border-radius: 10px; margin-top: 14px; padding: 10px 14px; }
    summary { cursor: pointer; font-weight: 700; }
    .meta { display: grid; grid-template-columns: minmax(150px,.35fr) 1fr; gap: 4px 14px; margin-top: 10px; font-size: 12px; }
    .meta dt { color: var(--muted); }
    .meta dd { margin: 0; word-break: break-all; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 14px; color: var(--muted); font-size: 12px; }
    .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
    @media (max-width: 900px) {
      header { grid-template-columns: 1fr; align-items: start; }
      .controlbar { grid-template-columns: auto auto 1fr; position: static; }
      .controlbar select { grid-column: span 3; }
      .boards { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .metric:nth-child(2) { border-right: 0; }
      .metric:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
      .readout { grid-template-columns: 1fr; }
    }
    @media (max-width: 520px) {
      main { width: min(100% - 18px, 1380px); margin-top: 14px; }
      .controlbar { grid-template-columns: 1fr 1fr; }
      .stepbox, .controlbar select { grid-column: span 2; }
    }
  </style>
</head>
<body>
<main id="replay">
  <header>
    <div>
      <h1>I1 V4 Calibration Replay</h1>
      <p>把同一条真实轨迹交给四种系统条件，逐帧观察“世界发生了什么、系统看到了什么、系统相信什么”。</p>
    </div>
    <div class="badge" id="seedBadge">Calibration seed</div>
  </header>

  <div class="notice"><strong>边界：</strong>左侧真值只供人类解释和离线评分；学习器每步只收到局部二值占据栅格与动作编号。</div>

  <div class="controlbar" aria-label="回放控制">
    <button id="play" type="button" aria-label="播放或暂停">▶ 播放</button>
    <button id="prev" type="button" aria-label="上一步">← 上一步</button>
    <div class="stepbox">
      <input id="step" type="range" min="0" value="0" aria-label="时间步">
      <output id="stepOut" for="step">0 / 200</output>
    </div>
    <select id="condition" aria-label="实验条件"></select>
    <select id="event" aria-label="跳转到关键事件"></select>
  </div>

  <p class="condition-note" id="conditionNote"></p>

  <section class="metrics" aria-label="当前条件的整段指标">
    <div class="metric"><span>自我识别 F1</span><strong id="mSelf">—</strong></div>
    <div class="metric"><span>身份一致性</span><strong id="mIdentity">—</strong></div>
    <div class="metric"><span>可见身份覆盖</span><strong id="mCoverage">—</strong></div>
    <div class="metric"><span>遮挡目标概率</span><strong id="mHidden">—</strong></div>
  </section>

  <section class="boards">
    <article class="card">
      <div class="card-head"><h2>1 · 世界真值</h2><p>评估器知道的真实位置；不是学习器输入。</p></div>
      <div class="canvas-wrap"><canvas id="truth" width="330" height="330"></canvas></div>
    </article>
    <article class="card">
      <div class="card-head"><h2>2 · 观测与轨迹身份</h2><p>亮区是系统判定可见的区域；编号是统一实体图的轨迹 ID。</p></div>
      <div class="canvas-wrap"><canvas id="observation" width="330" height="330"></canvas></div>
    </article>
    <article class="card">
      <div class="card-head"><h2>3 · 系统的占据信念</h2><p>颜色越深，系统越相信该格有东西；圆环标出当前 self。</p></div>
      <div class="canvas-wrap"><canvas id="belief" width="330" height="330"></canvas></div>
    </article>
  </section>

  <div class="legend">
    <span><i style="background:var(--truth)"></i>self 真值</span>
    <span><i style="background:var(--a)"></i>目标 A</span>
    <span><i style="background:var(--b)"></i>目标 B</span>
    <span><i style="background:var(--seen)"></i>当前感知到占据</span>
    <span><i style="background:var(--wall)"></i>静态遮挡物</span>
  </div>

  <div class="readout">
    <section><h3>当前动作</h3><p id="actionText">—</p><p id="suppliedText">—</p></section>
    <section><h3>自我身份后验</h3><p id="selfText">—</p><div class="posterior" id="posterior"></div></section>
    <section><h3>全局关联假设</h3><p id="hypothesisText">—</p><div class="hypotheses" id="hypotheses"></div></section>
  </div>

  <details>
    <summary>复现信息与证据边界</summary>
    <dl class="meta" id="metadata"></dl>
  </details>
</main>

<script id="replay-data" type="application/json">__I1_REPLAY_PAYLOAD__</script>
<script>
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("replay-data").textContent);
  const size = data.arena.size;
  const logical = 330;
  const cell = logical / size;
  const root = document.getElementById("replay");
  const stepInput = document.getElementById("step");
  const stepOut = document.getElementById("stepOut");
  const playButton = document.getElementById("play");
  const conditionSelect = document.getElementById("condition");
  const eventSelect = document.getElementById("event");
  let conditionName = data.conditionOrder[0];
  let step = 0;
  let timer = null;

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const ctxFor = (id) => {
    const canvas = document.getElementById(id);
    const ratio = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = logical * ratio;
    canvas.height = logical * ratio;
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return context;
  };
  const xy = (index) => [index % size, Math.floor(index / size)];
  const fillCell = (context, x, y, color, inset = 1) => {
    context.fillStyle = color;
    context.fillRect(x * cell + inset, y * cell + inset, cell - inset * 2, cell - inset * 2);
  };
  const grid = (context) => {
    context.strokeStyle = css("--line");
    context.lineWidth = .65;
    for (let i = 0; i <= size; i += 1) {
      const p = i * cell;
      context.beginPath(); context.moveTo(p, 0); context.lineTo(p, logical); context.stroke();
      context.beginPath(); context.moveTo(0, p); context.lineTo(logical, p); context.stroke();
    }
  };
  const dot = (context, position, color, label, outlined = false) => {
    const x = (position[0] + .5) * cell;
    const y = (position[1] + .5) * cell;
    context.beginPath();
    context.arc(x, y, cell * .29, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
    if (outlined) {
      context.strokeStyle = css("--ink");
      context.lineWidth = 3;
      context.stroke();
    }
    context.fillStyle = "#fff";
    context.font = `700 ${Math.max(10, cell * .34)}px ui-sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(label, x, y);
  };
  const clear = (context) => {
    context.clearRect(0, 0, logical, logical);
    context.fillStyle = css("--panel");
    context.fillRect(0, 0, logical, logical);
  };

  function drawTruth(frame) {
    const context = ctxFor("truth");
    clear(context);
    const visible = new Set(frame.agentVisible);
    for (let index = 0; index < size * size; index += 1) {
      if (!visible.has(index)) {
        const [x, y] = xy(index);
        fillCell(context, x, y, "rgba(80,90,87,.10)", 0);
      }
    }
    frame.truthStatic.forEach((index) => {
      const [x, y] = xy(index);
      fillCell(context, x, y, css("--wall"), 2);
    });
    dot(context, frame.truth.self, css("--truth"), "S", !frame.truthVisible.self);
    dot(context, frame.truth.a, css("--a"), "A", !frame.truthVisible.a);
    dot(context, frame.truth.b, css("--b"), "B", !frame.truthVisible.b);
    grid(context);
  }

  function drawObservation(frame) {
    const context = ctxFor("observation");
    clear(context);
    const visible = new Set(frame.agentVisible);
    for (let index = 0; index < size * size; index += 1) {
      const [x, y] = xy(index);
      fillCell(
        context, x, y,
        visible.has(index) ? "rgba(67,159,137,.11)" : "rgba(60,69,66,.18)",
        0
      );
    }
    frame.learnedStatic.forEach((index) => {
      const [x, y] = xy(index);
      fillCell(context, x, y, css("--wall"), 3);
    });
    frame.sensed.forEach((index) => {
      const [x, y] = xy(index);
      fillCell(context, x, y, css("--seen"), 5);
    });
    frame.tracks.forEach((track) => {
      dot(context, [track.x, track.y], track.isSelf ? css("--truth") : css("--accent"), String(track.id), track.isSelf);
    });
    grid(context);
  }

  function drawBelief(frame) {
    const context = ctxFor("belief");
    clear(context);
    frame.belief.forEach((value, index) => {
      const [x, y] = xy(index);
      const alpha = .03 + .91 * value / 100;
      fillCell(context, x, y, `rgba(18,109,92,${alpha})`, 1);
    });
    frame.tracks.forEach((track) => {
      const x = (track.x + .5) * cell;
      const y = (track.y + .5) * cell;
      context.beginPath();
      context.arc(x, y, cell * .34, 0, Math.PI * 2);
      context.strokeStyle = track.isSelf ? css("--truth") : css("--ink");
      context.lineWidth = track.isSelf ? 4 : 1.5;
      context.stroke();
      context.fillStyle = css("--ink");
      context.font = `700 ${Math.max(9, cell * .3)}px ui-monospace`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(String(track.id), x, y);
    });
    grid(context);
  }

  function bar(container, values, className) {
    container.replaceChildren();
    values.forEach((item) => {
      const part = document.createElement("i");
      part.className = className;
      part.style.flexGrow = String(Math.max(.002, item.value));
      part.title = item.label;
      part.style.opacity = String(.35 + .65 * item.value);
      container.appendChild(part);
    });
  }

  function render() {
    const condition = data.conditions[conditionName];
    const frame = condition.frames[step];
    root.dataset.condition = conditionName;
    root.dataset.step = String(step);
    stepInput.value = String(step);
    stepOut.value = `${step} / ${data.steps}`;
    document.getElementById("conditionNote").textContent = condition.description;
    document.getElementById("actionText").textContent = `世界实际执行：${frame.actionName}（${frame.action}）`;
    document.getElementById("suppliedText").textContent = `交给系统：${frame.suppliedActionName}（${frame.suppliedAction}）`;
    document.getElementById("selfText").textContent =
      frame.selfId === null ? "当前没有达到阈值的 self ID" : `当前 self = 轨迹 ${frame.selfId}`;
    document.getElementById("hypothesisText").textContent =
      `${frame.hypothesisWeights.length} 个并行假设；条宽表示权重`;

    const posterior = Object.entries(frame.selfPosterior).map(([identity, value]) => ({
      value,
      label: `轨迹 ${identity}: ${Number(value).toFixed(3)}`
    }));
    bar(document.getElementById("posterior"), posterior, "posterior-part");
    bar(
      document.getElementById("hypotheses"),
      frame.hypothesisWeights.map((value, index) => ({
        value,
        label: `假设 ${index + 1}: ${Number(value).toFixed(3)}`
      })),
      "hypothesis-part"
    );
    drawTruth(frame);
    drawObservation(frame);
    drawBelief(frame);
  }

  function setCondition(name) {
    conditionName = name;
    const condition = data.conditions[name];
    const metric = condition.metrics;
    document.getElementById("mSelf").textContent = metric.self_f1.toFixed(3);
    document.getElementById("mIdentity").textContent = metric.identity_consistency.toFixed(3);
    document.getElementById("mCoverage").textContent = metric.visible_identity_coverage.toFixed(3);
    document.getElementById("mHidden").textContent = metric.distractor_hidden_probability.toFixed(3);
    eventSelect.replaceChildren();
    condition.events.forEach((event, index) => {
      const option = document.createElement("option");
      option.value = String(event.step);
      option.textContent = event.label;
      if (index === 0) option.selected = true;
      eventSelect.appendChild(option);
    });
    render();
  }

  function stop() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
    playButton.textContent = "▶ 播放";
  }

  function togglePlay() {
    if (timer !== null) { stop(); return; }
    playButton.textContent = "Ⅱ 暂停";
    timer = window.setInterval(() => {
      if (step >= data.steps) {
        stop();
        return;
      }
      step += 1;
      render();
    }, 170);
  }

  data.conditionOrder.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = data.conditions[name].label;
    conditionSelect.appendChild(option);
  });
  stepInput.max = String(data.steps);
  stepInput.addEventListener("input", () => { step = Number(stepInput.value); render(); });
  conditionSelect.addEventListener("change", () => { stop(); setCondition(conditionSelect.value); });
  eventSelect.addEventListener("change", () => { stop(); step = Number(eventSelect.value); render(); });
  playButton.addEventListener("click", togglePlay);
  document.getElementById("prev").addEventListener("click", () => {
    stop();
    step = Math.max(0, step - 1);
    render();
  });
  window.addEventListener("resize", render);
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input,select,button")) return;
    if (event.key === "ArrowRight") { stop(); step = Math.min(data.steps, step + 1); render(); }
    if (event.key === "ArrowLeft") { stop(); step = Math.max(0, step - 1); render(); }
    if (event.key === " ") { event.preventDefault(); togglePlay(); }
  });

  document.getElementById("seedBadge").textContent = `Calibration seed ${data.seed}`;
  const metadata = [
    ["用途", "presentation-only（演示，不是新的实验结论）"],
    ["学习器输入", data.learnerInput.join(" + ")],
    ["真值进入学习器", String(data.evaluatorTruthUsedForLearning)],
    ["协议", `${data.protocol.path} · ${data.protocol.sha256}`],
    ["正式证据", `${data.formalEvidence.resultPath} · ${data.formalEvidence.resultSha256}`],
    ["正式实现 commit", data.formalEvidence.implementationCommit],
    ["动作序列 SHA-256", data.actionScheduleSha256],
    ["四条件数据 SHA-256", data.conditionDataSha256],
    ...Object.entries(data.sourceFiles).map(([name, digest]) => [`源码 ${name}`, digest])
  ];
  const metadataRoot = document.getElementById("metadata");
  metadata.forEach(([term, description]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = term;
    dd.textContent = description;
    metadataRoot.append(dt, dd);
  });
  setCondition(conditionName);
  window.__I1_REPLAY_READY__ = true;
})();
</script>
</body>
</html>
"""


def render_replay_html(payload: dict[str, Any]) -> str:
    """Render payload as deterministic UTF-8 standalone HTML."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).replace("</", "<\\/")
    return _HTML_TEMPLATE.replace(_PAYLOAD_MARKER, serialized)
