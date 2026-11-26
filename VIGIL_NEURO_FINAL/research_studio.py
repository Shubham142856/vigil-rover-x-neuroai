"""G10 Research Studio: deterministic replay verification and interactive HTML studio generator."""
import argparse
import base64
import csv
import io
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import struct
import zlib

import numpy as np

from baseline_policy import decide


def _png_bytes(rgb_uint8: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as a valid PNG without external dependencies."""
    h, w, _ = rgb_uint8.shape
    raw = b"".join(b"\x00" + rgb_uint8[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def verify_replay_bundle(run_dir: Path) -> dict:
    """Verify that replay.jsonl, frames.npz, and summary.json describe one identical run."""
    records = [json.loads(line) for line in (run_dir / "replay.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "frames.npz") as frames:
        n_frames = int(frames["rgb"].shape[0])
        cap = frames["capture_s"]
        deliv = frames["delivery_s"]
        if n_frames != int(summary["delivered_frames"]):
            raise AssertionError(f"Frame count mismatch in {run_dir}")
        for idx, rec in enumerate(records):
            s = rec["sensors"]
            fi = s["frame_index"]
            if fi is not None:
                if abs(float(s["capture_s"]) - float(cap[fi])) > 1e-9:
                    raise AssertionError(f"Capture timestamp mismatch at step {idx}")
                if abs(float(s["delivery_s"]) - float(deliv[fi])) > 1e-9:
                    raise AssertionError(f"Delivery timestamp mismatch at step {idx}")
            if rec["action"] is not None:
                expected_action = decide(s)[0]
                if rec["action"] != expected_action:
                    raise AssertionError(f"Decision mismatch at step {idx}: {rec['action']} != {expected_action}")
        if records[-1]["state"] != summary["final"]:
            raise AssertionError(f"Final evaluator state mismatch in {run_dir}")
        encoded_frames = [base64.b64encode(_png_bytes(frames["rgb"][i])).decode("ascii") for i in range(n_frames)]
    return {
        "name": run_dir.name,
        "steps": len(records),
        "delivered_frames": n_frames,
        "summary": summary,
        "records": records,
        "frames_b64": encoded_frames,
        "verified": True,
    }


def build_studio(runs_dir: Path, gates_csv: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    replay_names = ("my-first-drive", "rover-freeze", "rover-wall")
    replays = {}
    verification = {}
    for name in replay_names:
        rdir = runs_dir / name
        if (rdir / "replay.jsonl").exists():
            bundle = verify_replay_bundle(rdir)
            verification[name] = {
                "verified": True,
                "steps": bundle["steps"],
                "delivered_frames": bundle["delivered_frames"],
                "scene": bundle["summary"]["scene"],
                "fault": bundle["summary"]["fault"],
                "final": bundle["summary"]["final"],
            }
            replays[name] = bundle

    def read_json_if(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    with gates_csv.open(encoding="utf-8") as f:
        gates = list(csv.DictReader(f))

    payload = {
        "replays": replays,
        "gates": gates,
        "vision_benchmark": read_json_if(runs_dir / "vision-benchmark/summary.json"),
        "camera_escape": read_json_if(runs_dir / "camera-escape-zero-fp/summary.json"),
        "compass_benchmark": read_json_if(runs_dir / "compass-benchmark/summary.json"),
        "closed_loop": read_json_if(runs_dir / "closed-loop-eval/summary.json"),
        "scientific_validation": read_json_if(runs_dir / "scientific-validation/validation_report.json"),
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>VIGIL-ROVER X NeuroAI — Research Studio</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background:#0b111e; color:#e2e8f0; margin:0; padding:20px; }}
  h1,h2,h3 {{ margin:0 0 8px 0; color:#f8fafc; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap:16px; margin-top:16px; }}
  .card {{ background:#141e33; border:1px solid #243352; border-radius:10px; padding:16px; }}
  .controls {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }}
  button, select {{ background:#2563eb; color:white; border:none; padding:6px 12px; border-radius:6px; cursor:pointer; font-weight:600; }}
  button.sec {{ background:#334155; }}
  canvas {{ background:#090d16; border:1px solid #334155; border-radius:8px; width:100%; height:auto; }}
  img.cam {{ image-rendering:pixelated; width:256px; height:192px; border:1px solid #334155; border-radius:6px; background:#000; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  td,th {{ padding:6px 8px; border-bottom:1px solid #243352; text-align:left; }}
  .pill {{ padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }}
  .met {{ background:#065f46; color:#a7f3d0; }}
  .plan {{ background:#334155; color:#cbd5e1; }}
  .warn {{ color:#fbbf24; font-weight:600; }}
  .mono {{ font-family: ui-monospace, monospace; font-size:12px; color:#94a3b8; }}
</style>
</head>
<body>
<h1>VIGIL-ROVER X NeuroAI — Reproducible Research Studio</h1>
<p class="mono">Driven strictly by recorded PyBullet replay.jsonl, frames.npz, FAFB v783 connectome audits, and paired benchmarks.</p>

<div class="card">
  <div class="controls">
    <label>Recorded Run:
      <select id="runSel"></select>
    </label>
    <button id="playBtn">Play</button>
    <button id="stepBtn" class="sec">Step +1</button>
    <button id="resetBtn" class="sec">Reset</button>
    <label><input type="checkbox" id="oracleChk"/> Evaluator Oracle State</label>
    <span id="tickInfo" class="mono"></span>
  </div>
  <div class="grid">
    <div>
      <h3>Delivered Rover Camera (96x128 @ 10 Hz)</h3>
      <img id="camImg" class="cam" alt="Rover camera frame"/>
      <div id="sensorPanel" class="mono" style="margin-top:8px;"></div>
    </div>
    <div>
      <h3>Recorded Trajectory &amp; Compass Heading</h3>
      <canvas id="trajCanvas" width="360" height="240"></canvas>
      <div id="oraclePanel" class="mono" style="margin-top:8px;"></div>
    </div>
  </div>
</div>

<div class="grid">
  <div class="card">
    <h3>Acceptance Gates</h3>
    <table id="gatesTbl"><thead><tr><th>Gate</th><th>Workstream</th><th>Status</th></tr></thead><tbody></tbody></table>
  </div>
  <div class="card">
    <h3>Closed-Loop Recovery Ablation (G08)</h3>
    <table id="clTbl"><thead><tr><th>Policy</th><th>Collisions</th><th>False Stops</th><th>Safe Success</th><th>Abstentions</th></tr></thead><tbody></tbody></table>
  </div>
</div>

<script>
const DATA = {json.dumps(payload)};
const runSel = document.getElementById('runSel');
const camImg = document.getElementById('camImg');
const sensorPanel = document.getElementById('sensorPanel');
const oraclePanel = document.getElementById('oraclePanel');
const tickInfo = document.getElementById('tickInfo');
const oracleChk = document.getElementById('oracleChk');
const canvas = document.getElementById('trajCanvas');
const ctx = canvas.getContext('2d');

Object.keys(DATA.replays).forEach(k => {{
  const o = document.createElement('option');
  o.value = k; o.textContent = k + ' (' + DATA.replays[k].summary.scene + ' / ' + DATA.replays[k].summary.fault + ')';
  runSel.appendChild(o);
}});

let stepIdx = 0, timer = null;
function currentRun() {{ return DATA.replays[runSel.value]; }}

function render() {{
  const run = currentRun();
  if (!run) return;
  const rec = run.records[stepIdx];
  const s = rec.sensors;
  tickInfo.textContent = `Step ${{stepIdx}} / ${{run.records.length - 1}} | t=${{s.time_s.toFixed(2)}}s | action=${{rec.action || 'END'}}`;
  if (s.frame_index !== null && run.frames_b64[s.frame_index]) {{
    camImg.src = 'data:image/png;base64,' + run.frames_b64[s.frame_index];
  }}
  sensorPanel.innerHTML =
    `frame_valid=${{s.frame_valid}} | age=${{s.frame_age_s !== null ? s.frame_age_s.toFixed(2)+'s' : 'N/A'}}<br/>` +
    `whisker_L=${{s.whisker_left}} | whisker_R=${{s.whisker_right}}<br/>` +
    `obstacle_center=${{s.obstacle_center.toFixed(3)}} | looming=${{s.looming_center.toFixed(3)}}`;
  if (oracleChk.checked) {{
    const st = rec.state;
    oraclePanel.innerHTML = `[EVALUATOR] x=${{st.x_m.toFixed(3)}}m, y=${{st.y_m.toFixed(3)}}m, yaw=${{(st.yaw_rad*180/Math.PI).toFixed(1)}}&deg;, speed=${{st.speed_m_s.toFixed(3)}}m/s, body_contacts=${{st.body_contact_events}}`;
  }} else {{
    oraclePanel.textContent = 'Evaluator oracle hidden (sensor-only view).';
  }}

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#334155'; ctx.beginPath(); ctx.moveTo(20, 120); ctx.lineTo(340, 120); ctx.stroke();
  ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i <= stepIdx; i++) {{
    const st = run.records[i].state;
    const px = 30 + st.x_m * 180;
    const py = 120 - st.y_m * 180;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }}
  ctx.stroke();
  const cur = rec.state;
  ctx.fillStyle = cur.body_contact ? '#ef4444' : '#22c55e';
  ctx.beginPath(); ctx.arc(30 + cur.x_m * 180, 120 - cur.y_m * 180, 5, 0, Math.PI*2); ctx.fill();
}}

document.getElementById('playBtn').onclick = () => {{
  if (timer) {{ clearInterval(timer); timer = null; document.getElementById('playBtn').textContent = 'Play'; return; }}
  document.getElementById('playBtn').textContent = 'Pause';
  timer = setInterval(() => {{
    const run = currentRun();
    if (stepIdx < run.records.length - 1) {{ stepIdx++; render(); }}
    else {{ clearInterval(timer); timer = null; document.getElementById('playBtn').textContent = 'Play'; }}
  }}, 120);
}};
document.getElementById('stepBtn').onclick = () => {{ const r = currentRun(); if (stepIdx < r.records.length - 1) {{ stepIdx++; render(); }} }};
document.getElementById('resetBtn').onclick = () => {{ stepIdx = 0; render(); }};
runSel.onchange = () => {{ stepIdx = 0; render(); }};
oracleChk.onchange = render;

const gt = document.querySelector('#gatesTbl tbody');
DATA.gates.forEach(g => {{
  const tr = document.createElement('tr');
  const cls = g.Status === 'Met' ? 'met' : 'plan';
  tr.innerHTML = `<td>${{g.Milestone}}</td><td>${{g.Workstream}}</td><td><span class="pill ${{cls}}">${{g.Status}}</span></td>`;
  gt.appendChild(tr);
}});

if (DATA.closed_loop && DATA.closed_loop.policies) {{
  const ct = document.querySelector('#clTbl tbody');
  Object.entries(DATA.closed_loop.policies).forEach(([name, m]) => {{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${{name}}</td><td>${{m.total_body_collisions}}</td><td>${{m.false_emergency_stops}}</td><td>${{(m.safe_success_rate*100).toFixed(1)}}%</td><td>${{m.abstentions}}</td>`;
    ct.appendChild(tr);
  }});
}}
render();
</script>
</body>
</html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    manifest = {"replays_verified": verification, "studio_html": str(out_dir / "index.html")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--gates", type=Path, default=Path("VIGIL_NEURO_FINAL/Implementation_Acceptance_Gates.csv"))
    p.add_argument("--out", type=Path, default=Path("runs/research-studio"))
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    rep = build_studio(a.runs, a.gates, a.out)
    print(json.dumps(rep, indent=2))
    if a.serve:
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(a.out), **kwargs)
        ThreadingHTTPServer(("127.0.0.1", a.port), handler).serve_forever()
