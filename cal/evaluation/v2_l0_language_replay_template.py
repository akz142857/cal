"""Self-contained HTML template for the deterministic L0 replay."""

from __future__ import annotations

import json
from typing import Any


_PAYLOAD_MARKER = "__L0_REPLAY_PAYLOAD__"

_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>L0 Development Language Replay</title>
  <style>
    :root {
      --bg:#f3f1e9;--panel:rgba(255,255,255,.82);--ink:#18221f;
      --muted:#69736f;--line:#cbd0ca;--accent:#126d5c;--soft:#dceee8;
      --truth:#ce5b46;--a:#397ec5;--b:#9a58b5;--seen:#efb84c;
      --wall:#606a66;--good:#18794e;--bad:#bc3f35;--idle:#89938f;
      --query-self:#db4f87;--query-permanence:#df8d12;--query-identity:#7556d8;
      --shadow:0 13px 34px rgba(29,42,38,.10);
    }
    @media (prefers-color-scheme:dark) {
      :root {
        --bg:#101715;--panel:rgba(26,36,33,.94);--ink:#edf4f0;
        --muted:#a7b2ae;--line:#43504c;--accent:#72d7c2;--soft:#203e37;
        --truth:#ff8d76;--a:#79b6ff;--b:#d59aef;--seen:#ffd071;
        --wall:#8f9b97;--good:#65d49b;--bad:#ff8a80;--idle:#7e8c87;
        --query-self:#ff87b6;--query-permanence:#ffc45f;--query-identity:#ad98ff;
        --shadow:0 13px 34px rgba(0,0,0,.25);
      }
    }
    *{box-sizing:border-box}
    body{
      margin:0;color:var(--ink);
      background:
        radial-gradient(circle at 12% -8%,color-mix(in srgb,var(--accent) 13%,transparent),transparent 38rem),
        var(--bg);
      font:14px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    }
    main{width:min(1500px,calc(100% - 30px));margin:24px auto 50px}
    header{display:flex;gap:20px;justify-content:space-between;align-items:end;margin-bottom:15px}
    h1{margin:0 0 4px;font-size:clamp(26px,4vw,43px);letter-spacing:-.04em}
    header p{margin:0;color:var(--muted);max-width:820px}
    .badge{border:1px solid var(--accent);color:var(--accent);border-radius:99px;padding:7px 12px;font-weight:800;white-space:nowrap}
    .notice{padding:11px 14px;border-left:4px solid var(--truth);background:color-mix(in srgb,var(--truth) 8%,var(--panel));border-radius:8px;margin-bottom:14px}
    .panel,.controls,.metrics,details{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow);backdrop-filter:blur(12px)}
    .controls{
      display:grid;grid-template-columns:auto auto minmax(170px,1fr) minmax(220px,.75fr) minmax(220px,.9fr) auto;
      gap:9px;align-items:center;padding:12px;border-radius:12px;position:sticky;top:7px;z-index:5
    }
    button,select,input{font:inherit}
    button,select{min-height:38px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--ink)}
    button{padding:0 12px;cursor:pointer;font-weight:750}
    button:hover{border-color:var(--accent);color:var(--accent)}
    button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid color-mix(in srgb,var(--accent) 32%,transparent);outline-offset:2px}
    select{padding:0 9px;width:100%}
    input[type=range]{width:100%;accent-color:var(--accent)}
    .stepbox{display:grid;grid-template-columns:minmax(100px,1fr) 76px;gap:8px;align-items:center}
    output{color:var(--accent);font-weight:850;text-align:right;font-variant-numeric:tabular-nums}
    .active-only{display:flex;gap:6px;align-items:center;white-space:nowrap;color:var(--muted)}
    .active-only input{accent-color:var(--accent)}
    .representation-note{margin:10px 2px 13px;color:var(--muted);min-height:22px}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);border-radius:12px;overflow:hidden;margin-bottom:14px}
    .metric{padding:11px 14px;border-right:1px solid var(--line)}
    .metric:last-child{border-right:0}
    .metric span{display:block;color:var(--muted);font-size:12px}
    .metric strong{display:block;font-size:23px;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
    .workspace{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(430px,.95fr);gap:14px;align-items:start}
    .panel{border-radius:12px;overflow:hidden}
    .panel-head{padding:13px 15px;border-bottom:1px solid var(--line)}
    .panel-head h2{font-size:18px;margin:0}
    .panel-head p{margin:2px 0 0;color:var(--muted);font-size:12px}
    .boards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;padding:11px}
    .board h3{margin:0 0 6px;font-size:12px;color:var(--muted)}
    canvas{display:block;width:100%;aspect-ratio:1;border:1px solid var(--line);border-radius:8px;background:var(--panel)}
    .frame-facts{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;padding:0 11px 11px}
    .fact{padding:8px 10px;border:1px solid var(--line);border-radius:8px;min-width:0}
    .fact span{display:block;color:var(--muted);font-size:11px}
    .fact strong{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .legend{display:flex;flex-wrap:wrap;gap:11px;padding:0 12px 12px;color:var(--muted);font-size:11px}
    .legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
    .language-head{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:start}
    .score{font-size:12px;color:var(--muted);text-align:right}
    .score strong{display:block;color:var(--accent);font-size:22px}
    .warmup{margin:12px;padding:18px;border:1px dashed var(--line);border-radius:9px;color:var(--muted)}
    .language-list{padding:9px;display:grid;gap:7px}
    .language-row{display:grid;grid-template-columns:minmax(190px,1fr) minmax(120px,.7fr) 78px;gap:10px;align-items:center;padding:8px 9px;border:1px solid var(--line);border-radius:9px;background:color-mix(in srgb,var(--panel) 92%,transparent)}
    .language-row.inactive{opacity:.42}
    .language-row.hidden{display:none}
    .sentence{min-width:0}
    .sentence b{display:block;font-size:13px;font-weight:760}
    .sentence small{color:var(--muted)}
    .probability{min-width:0}
    .bar{height:8px;border-radius:99px;background:color-mix(in srgb,var(--idle) 23%,transparent);overflow:hidden;margin-bottom:4px}
    .bar i{display:block;height:100%;background:var(--accent);border-radius:inherit}
    .probability small{display:flex;justify-content:space-between;color:var(--muted);font-variant-numeric:tabular-nums}
    .verdict{border-radius:7px;padding:5px 6px;text-align:center;font-size:11px;font-weight:850;border:1px solid var(--line)}
    .verdict.good{color:var(--good);border-color:var(--good);background:color-mix(in srgb,var(--good) 8%,transparent)}
    .verdict.bad{color:var(--bad);border-color:var(--bad);background:color-mix(in srgb,var(--bad) 8%,transparent)}
    .verdict.idle{color:var(--idle)}
    .explain{padding:11px 13px;margin:0 9px 9px;border-radius:9px;background:var(--soft);font-size:12px}
    details{border-radius:10px;margin-top:14px;padding:10px 14px}
    summary{cursor:pointer;font-weight:750}
    .meta{display:grid;grid-template-columns:minmax(150px,.32fr) 1fr;gap:4px 13px;margin-top:10px;font-size:12px}
    .meta dt{color:var(--muted)}
    .meta dd{margin:0;word-break:break-all;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    @media(max-width:1050px){
      .controls{grid-template-columns:auto auto 1fr}
      .controls select,.active-only{grid-column:span 3}
      .workspace{grid-template-columns:1fr}
    }
    @media(max-width:720px){
      main{width:min(100% - 16px,1500px);margin-top:12px}
      header{display:block}.badge{display:inline-block;margin-top:10px}
      .metrics{grid-template-columns:repeat(2,1fr)}
      .metric:nth-child(2){border-right:0}.metric:nth-child(-n+2){border-bottom:1px solid var(--line)}
      .boards{grid-template-columns:1fr}.board h3{font-size:14px}
      .language-row{grid-template-columns:1fr}.workspace{display:block}.language-panel{margin-top:12px}
    }
  </style>
</head>
<body>
<main id="replay">
  <header>
    <div>
      <h1>L0 · 语言从哪里来</h1>
      <p>沿着同一段世界轨迹，观察 I1 如何形成实体信念，以及一个独立线性读出器如何把这些状态翻译成受控中文命题。</p>
    </div>
    <div class="badge" id="seedBadge">Development seed</div>
  </header>

  <div class="notice"><strong>最重要的边界：</strong>真实答案只在更新完成后用于训练读出器和页面对照；它从未进入 I1。这个页面是可重复的 development 演示，不是 V8 holdout 证据。</div>

  <div class="controls" aria-label="回放控制">
    <button id="play" type="button">▶ 播放</button>
    <button id="prev" type="button">← 上一步</button>
    <div class="stepbox">
      <input id="step" type="range" min="0" value="0" aria-label="时间步">
      <output id="stepOut" for="step">0 / 200</output>
    </div>
    <select id="representation" aria-label="语言读出来源"></select>
    <select id="event" aria-label="跳转到关键事件"></select>
    <label class="active-only"><input id="activeOnly" type="checkbox"> 只看当前有效命题</label>
  </div>
  <p class="representation-note" id="representationNote"></p>

  <section class="metrics" aria-label="冻结 development 整体指标">
    <div class="metric"><span>四类宏平均平衡准确率</span><strong id="mMacro">—</strong></div>
    <div class="metric"><span>自我</span><strong id="mSelf">—</strong></div>
    <div class="metric"><span>遮挡后仍存在</span><strong id="mPermanence">—</strong></div>
    <div class="metric"><span>重现后身份</span><strong id="mIdentity">—</strong></div>
  </section>

  <div class="workspace">
    <section class="panel">
      <div class="panel-head">
        <h2>这一步，学习器经历了什么</h2>
        <p>真值供你理解；中间和右侧才接近系统实际收到和形成的内容。</p>
      </div>
      <div class="boards">
        <div class="board"><h3>1 · 世界真值（评分器）</h3><canvas id="truth" width="260" height="260"></canvas></div>
        <div class="board"><h3>2 · 局部观测与轨迹</h3><canvas id="observation" width="260" height="260"></canvas></div>
        <div class="board"><h3>3 · I1 占据信念</h3><canvas id="belief" width="260" height="260"></canvas></div>
      </div>
      <div class="frame-facts">
        <div class="fact"><span>执行动作</span><strong id="actionText">—</strong></div>
        <div class="fact"><span>I1 当前认为的 self</span><strong id="selfText">—</strong></div>
        <div class="fact"><span>并行关联假设</span><strong id="hypothesisText">—</strong></div>
      </div>
      <div class="legend">
        <span><i style="background:var(--truth)"></i>self 真值</span>
        <span><i style="background:var(--a)"></i>目标 A</span>
        <span><i style="background:var(--b)"></i>目标 B</span>
        <span><i style="background:var(--seen)"></i>当前传感器占据</span>
        <span><i style="background:var(--wall)"></i>静态遮挡物</span>
        <span><i style="background:var(--query-self)"></i>语言查询标记</span>
      </div>
    </section>

    <section class="panel language-panel">
      <div class="panel-head language-head">
        <div><h2>这一步，L0 能说出什么</h2><p id="languageSubhead">第 12 步之后开始读出。</p></div>
        <div class="score">当前有效命题<strong id="frameScore">—</strong></div>
      </div>
      <div class="warmup" id="warmup">I1 正在积累最初的轨迹和自我证据。到 warmup 结束后，语言读出器才开始接受查询。</div>
      <div class="language-list" id="languageList"></div>
      <p class="explain" id="explain">概率表示读出器认为句子为真的程度；绿色/红色只表示它和离线真实答案是否一致。</p>
    </section>
  </div>

  <details>
    <summary>复现信息、证据级别和安全边界</summary>
    <dl class="meta" id="metadata"></dl>
  </details>
</main>

<script id="replay-data" type="application/json">__L0_REPLAY_PAYLOAD__</script>
<script>
(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("replay-data").textContent);
  const size = data.arena.size;
  const logical = 260;
  const cell = logical / size;
  const root = document.getElementById("replay");
  const stepInput = document.getElementById("step");
  const stepOut = document.getElementById("stepOut");
  const playButton = document.getElementById("play");
  const representationSelect = document.getElementById("representation");
  const eventSelect = document.getElementById("event");
  const activeOnly = document.getElementById("activeOnly");
  let representationName = data.representationOrder[0];
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
    context.lineWidth = .6;
    for (let i = 0; i <= size; i += 1) {
      const p = i * cell;
      context.beginPath(); context.moveTo(p, 0); context.lineTo(p, logical); context.stroke();
      context.beginPath(); context.moveTo(0, p); context.lineTo(logical, p); context.stroke();
    }
  };
  const clear = (context) => {
    context.clearRect(0, 0, logical, logical);
    context.fillStyle = css("--panel");
    context.fillRect(0, 0, logical, logical);
  };
  const dot = (context, position, color, label, outlined = false) => {
    const x = (position[0] + .5) * cell;
    const y = (position[1] + .5) * cell;
    context.beginPath(); context.arc(x, y, cell * .3, 0, Math.PI * 2);
    context.fillStyle = color; context.fill();
    if (outlined) {
      context.strokeStyle = css("--ink"); context.lineWidth = 3; context.stroke();
    }
    context.fillStyle = "#fff";
    context.font = `700 ${Math.max(9, cell * .35)}px ui-sans-serif`;
    context.textAlign = "center"; context.textBaseline = "middle";
    context.fillText(label, x, y);
  };

  function drawTruth(frame) {
    const context = ctxFor("truth"); clear(context);
    const visible = new Set(frame.agentVisible);
    for (let index = 0; index < size * size; index += 1) {
      if (!visible.has(index)) {
        const [x, y] = xy(index); fillCell(context, x, y, "rgba(80,90,87,.10)", 0);
      }
    }
    frame.truthStatic.forEach((index) => {
      const [x, y] = xy(index); fillCell(context, x, y, css("--wall"), 2);
    });
    dot(context, frame.truth.self, css("--truth"), "S", !frame.truthVisible.self);
    dot(context, frame.truth.a, css("--a"), "A", !frame.truthVisible.a);
    dot(context, frame.truth.b, css("--b"), "B", !frame.truthVisible.b);
    grid(context);
  }

  function drawObservation(frame) {
    const context = ctxFor("observation"); clear(context);
    const visible = new Set(frame.agentVisible);
    for (let index = 0; index < size * size; index += 1) {
      const [x, y] = xy(index);
      fillCell(context, x, y, visible.has(index) ? "rgba(67,159,137,.11)" : "rgba(60,69,66,.18)", 0);
    }
    frame.learnedStatic.forEach((index) => {
      const [x, y] = xy(index); fillCell(context, x, y, css("--wall"), 3);
    });
    frame.sensed.forEach((index) => {
      const [x, y] = xy(index); fillCell(context, x, y, css("--seen"), 5);
    });
    frame.tracks.forEach((track) => {
      dot(context, [track.x, track.y], track.isSelf ? css("--truth") : css("--accent"), String(track.id), track.isSelf);
    });
    grid(context);
  }

  function drawBelief(frame) {
    const context = ctxFor("belief"); clear(context);
    frame.belief.forEach((value, index) => {
      const [x, y] = xy(index);
      fillCell(context, x, y, `rgba(18,109,92,${.03 + .91 * value / 100})`, 1);
    });
    frame.tracks.forEach((track) => {
      const x = (track.x + .5) * cell;
      const y = (track.y + .5) * cell;
      context.beginPath(); context.arc(x, y, cell * .34, 0, Math.PI * 2);
      context.strokeStyle = track.isSelf ? css("--truth") : css("--ink");
      context.lineWidth = track.isSelf ? 4 : 1.5; context.stroke();
      context.fillStyle = css("--ink");
      context.font = `700 ${Math.max(8, cell * .3)}px ui-monospace`;
      context.textAlign = "center"; context.textBaseline = "middle";
      context.fillText(String(track.id), x, y);
    });
    grid(context);
  }

  function drawQueryMarkers(languageFrame) {
    if (!languageFrame.ready) return;
    const markers = languageFrame.items.filter((item) => item.active && item.queryPosition !== null);
    const colors = {
      self: css("--query-self"),
      permanence: css("--query-permanence"),
      identity: css("--query-identity")
    };
    const prefixes = {self: "我", permanence: "存", identity: "同"};
    ["truth", "observation", "belief"].forEach((canvasId) => {
      const context = ctxFor(canvasId);
      markers.forEach((item) => {
        const [x, y] = item.queryPosition;
        const centerX = (x + .5) * cell;
        const centerY = (y + .5) * cell;
        context.save();
        context.beginPath();
        context.arc(centerX, centerY, cell * .43, 0, Math.PI * 2);
        context.strokeStyle = colors[item.group];
        context.lineWidth = 2.5;
        context.setLineDash([3, 2]);
        context.stroke();
        context.setLineDash([]);
        context.fillStyle = colors[item.group];
        context.font = `800 ${Math.max(8, cell * .28)}px ui-sans-serif`;
        context.textAlign = "center";
        context.textBaseline = "bottom";
        context.fillText(`${prefixes[item.group]}${item.index % 2 + 1}`, centerX, centerY - cell * .31);
        context.restore();
      });
    });
  }

  function renderLanguage(languageFrame) {
    const list = document.getElementById("languageList");
    const warmup = document.getElementById("warmup");
    if (!languageFrame.ready) {
      warmup.hidden = false;
      list.replaceChildren();
      document.getElementById("frameScore").textContent = "等待";
      document.getElementById("languageSubhead").textContent = `warmup：第 ${data.warmup} 步开始读出`;
      return;
    }
    warmup.hidden = true;
    list.replaceChildren();
    languageFrame.items.forEach((item) => {
      const row = document.createElement("div");
      row.className = `language-row${item.active ? "" : " inactive"}${activeOnly.checked && !item.active ? " hidden" : ""}`;
      const sentence = document.createElement("div");
      sentence.className = "sentence";
      const title = document.createElement("b");
      title.textContent = item.sentence;
      const group = document.createElement("small");
      group.textContent = `${item.groupLabel} · ${item.active ? "本步有效" : "本步未注册评分"}`;
      sentence.append(title, group);

      const probability = document.createElement("div");
      probability.className = "probability";
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("i");
      fill.style.width = `${item.probabilityTrue * 100}%`;
      bar.appendChild(fill);
      const values = document.createElement("small");
      const p = document.createElement("span");
      p.textContent = `真 ${Math.round(item.probabilityTrue * 100)}%`;
      const prediction = document.createElement("span");
      prediction.textContent = item.predictedTrue ? "判断：真" : "判断：假";
      values.append(p, prediction);
      probability.append(bar, values);

      const verdict = document.createElement("div");
      verdict.className = `verdict ${item.active ? (item.correct ? "good" : "bad") : "idle"}`;
      verdict.textContent = item.active
        ? `${item.correct ? "✓" : "✕"} 真值：${item.truthTrue ? "真" : "假"}`
        : "未评分";
      row.append(sentence, probability, verdict);
      list.appendChild(row);
    });
    document.getElementById("frameScore").textContent =
      languageFrame.activeCount ? `${languageFrame.correctCount} / ${languageFrame.activeCount}` : "无查询";
    document.getElementById("languageSubhead").textContent =
      "同一组受控中文句子；只改变当前状态和读出来源。";
  }

  function render() {
    const frame = data.visualFrames[step];
    const languageFrame = data.languageConditions[representationName].frames[step];
    root.dataset.step = String(step);
    root.dataset.representation = representationName;
    stepInput.value = String(step);
    stepOut.value = `${step} / ${data.steps}`;
    document.getElementById("actionText").textContent = `${frame.actionName}（${frame.action}）`;
    document.getElementById("selfText").textContent =
      frame.selfId === null ? "尚未稳定识别" : `轨迹 ${frame.selfId}`;
    document.getElementById("hypothesisText").textContent =
      `${frame.hypothesisWeights.length} 个`;
    drawTruth(frame); drawObservation(frame); drawBelief(frame);
    drawQueryMarkers(languageFrame);
    renderLanguage(languageFrame);
  }

  function setRepresentation(name) {
    representationName = name;
    const condition = data.languageConditions[name];
    const aggregate = data.aggregateMetrics[name];
    document.getElementById("representationNote").textContent = condition.description;
    document.getElementById("mMacro").textContent = aggregate.macro_balanced_accuracy.toFixed(3);
    document.getElementById("mSelf").textContent = aggregate.self_balanced_accuracy.toFixed(3);
    document.getElementById("mPermanence").textContent = aggregate.permanence_balanced_accuracy.toFixed(3);
    document.getElementById("mIdentity").textContent = aggregate.identity_balanced_accuracy.toFixed(3);
    render();
  }

  function stop() {
    if (timer !== null) window.clearInterval(timer);
    timer = null; playButton.textContent = "▶ 播放";
  }
  function togglePlay() {
    if (timer !== null) { stop(); return; }
    playButton.textContent = "Ⅱ 暂停";
    timer = window.setInterval(() => {
      if (step >= data.steps) { stop(); return; }
      step += 1; render();
    }, 190);
  }

  data.representationOrder.forEach((name) => {
    const option = document.createElement("option");
    option.value = name; option.textContent = data.languageConditions[name].label;
    representationSelect.appendChild(option);
  });
  data.events.forEach((event) => {
    const option = document.createElement("option");
    option.value = String(event.step); option.textContent = event.label;
    eventSelect.appendChild(option);
  });
  stepInput.max = String(data.steps);
  stepInput.addEventListener("input", () => { stop(); step = Number(stepInput.value); render(); });
  representationSelect.addEventListener("change", () => setRepresentation(representationSelect.value));
  eventSelect.addEventListener("change", () => { stop(); step = Number(eventSelect.value); render(); });
  activeOnly.addEventListener("change", render);
  playButton.addEventListener("click", togglePlay);
  document.getElementById("prev").addEventListener("click", () => {
    stop(); step = Math.max(0, step - 1); render();
  });
  window.addEventListener("resize", render);
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input,select,button")) return;
    if (event.key === "ArrowRight") { stop(); step = Math.min(data.steps, step + 1); render(); }
    if (event.key === "ArrowLeft") { stop(); step = Math.max(0, step - 1); render(); }
    if (event.key === " ") { event.preventDefault(); togglePlay(); }
  });

  document.getElementById("seedBadge").textContent = `Development-validation seed ${data.seed}`;
  const metadata = [
    ["用途", `${data.evidenceLevel} · presentation-only`],
    ["holdout seeds 被访问", String(data.holdoutSeedsAccessed)],
    ["I1 输入", data.learnerInput.join(" + ")],
    ["真值进入 I1", String(data.evaluatorTruthUsedForI1)],
    ["真值用于读出器训练", String(data.evaluatorTruthUsedForReadoutTraining)],
    ["语言梯度进入 I1", String(data.languageGradientsReachI1)],
    ["协议", `${data.protocol.path} · ${data.protocol.sha256}`],
    ["冻结 development 证据", `${data.formalEvidence.resultPath} · ${data.formalEvidence.resultSha256}`],
    ["冻结 gates", `${data.formalEvidence.gateCount} / ${data.formalEvidence.gateCount} · ${data.formalEvidence.allGatesPassed}`],
    ["V8 source-lock", `${data.sourceLock.tag} · ${data.sourceLock.targetCommit}`],
    ["动作序列 SHA-256", data.actionScheduleSha256],
    ["回放数据 SHA-256", data.replayDataSha256],
    ...Object.entries(data.sourceFiles).map(([name, digest]) => [`源码 ${name}`, digest])
  ];
  const metadataRoot = document.getElementById("metadata");
  metadata.forEach(([term, description]) => {
    const dt = document.createElement("dt"); const dd = document.createElement("dd");
    dt.textContent = term; dd.textContent = description; metadataRoot.append(dt, dd);
  });
  setRepresentation(representationName);
  window.__L0_REPLAY_READY__ = true;
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
