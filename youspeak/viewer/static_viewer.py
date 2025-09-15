from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import json
import webbrowser

from ..parsers.subtitles import parse_srt_bytes, parse_vtt_bytes


INDEX_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <link rel=\"stylesheet\" href=\"viewer.css\" />
</head>
<body>
  <div class=\"topbar\">
    <button id=\"play\">Play</button>
    <button id=\"pause\">Pause</button>
    <label>Speed <select id=\"speed\"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select>x</label>
    <button id=\"zoomIn\">Zoom +</button>
    <button id=\"zoomOut\">Zoom -</button>
    <input id=\"search\" placeholder=\"search text\" />
  </div>
  <div id=\"controls\"></div>
  <input id=\"slider\" type=\"range\" min=\"0\" max=\"1000\" step=\"1\" value=\"0\" />
  <div id=\"container\">
    <svg id=\"timeline\"></svg>
  </div>
  <script>window.__DATA__ = {data_json};</script>
  <script src=\"viewer.js\"></script>
</body>
</html>"""

VIEWER_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; }
.topbar { display: flex; gap: 8px; align-items: center; padding: 8px; border-bottom: 1px solid #ddd; }
#slider { width: 100%; }
#container { position: relative; height: 70vh; overflow: auto; border-top: 1px solid #eee; }
#timeline { height: 100%; background: #fafafa; }
.ctrlRow { display: grid; grid-template-columns: 1fr auto auto auto; align-items: center; gap: 8px; padding: 6px 8px; border-bottom: 1px solid #f0f0f0; }
.ctrlRow input { width: 80px; }
.laneLabel { font-size: 12px; fill: #444; }
.cue { fill: #4a90e2; opacity: .6; cursor: pointer; }
.cue.highlight { fill: #e24a5a; }
.playhead { stroke: #222; stroke-width: 1; }
"""

VIEWER_JS = """
(function(){
const svg = document.getElementById('timeline');
const container = document.getElementById('container');
const controls = document.getElementById('controls');
const slider = document.getElementById('slider');
const btnPlay = document.getElementById('play');
const btnPause = document.getElementById('pause');
const selSpeed = document.getElementById('speed');
const btnZIn = document.getElementById('zoomIn');
const btnZOut = document.getElementById('zoomOut');
const inputSearch = document.getElementById('search');

let data = null;
let pxPerSec = 50;
let xOffset = 0; // deprecated (using scroll instead)
let playing = false;
let playStartMs = 0;
let playStartT = 0;
let currentT = 0;
let transforms = [];

function boot(){
  if (window.__DATA__) { data = window.__DATA__; init(); }
  else {
    fetch('data.json').then(r=>r.json()).then(d=>{ data=d; init(); }).catch(()=>{
      console.error('Failed to load data.json and no inline data provided.');
    });
  }
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();

function init(){
  const lanes = data.tracks.length;
  if (!transforms.length) transforms = data.tracks.map(()=>({a:1,b:0}));
  const laneHeight = 22, laneGap = 8, topPad = 20, leftPad = 100, rightPad=10, bottomPad=20;
  // compute global min start and max end after transforms
  let tMin = 0, tMax = data.duration_seconds;
  data.tracks.forEach((tr, idx)=>{
    const {a,b} = transforms[idx]||{a:1,b:0};
    tr.cues.forEach(c=>{
      const s = a*c.s + b;
      const e = a*c.e + b;
      if (s < tMin) tMin = s;
      if (e > tMax) tMax = e;
    });
  });
  const totalWidth = leftPad + ((tMax - tMin)*pxPerSec) + rightPad;
  const height = topPad + lanes*(laneHeight+laneGap) + bottomPad;
  svg.setAttribute('viewBox', `0 0 ${totalWidth} ${height}`);
  svg.style.width = `${totalWidth}px`;
  svg.innerHTML = '';
  // controls UI
  controls.innerHTML = '';
  data.tracks.forEach((tr, idx)=>{
    const row = document.createElement('div');
    row.className = 'ctrlRow';
    const name = document.createElement('div'); name.textContent = tr.name;
    const aInp = document.createElement('input'); aInp.type='text'; aInp.value=String(transforms[idx].a);
    const bInp = document.createElement('input'); bInp.type='text'; bInp.value=String(transforms[idx].b);
    const resetBtn = document.createElement('button'); resetBtn.textContent='Reset';
    aInp.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ const v=parseFloat((aInp.value||'').replace(',','.')); if(!isNaN(v)){ transforms[idx].a=v; reflow(); } } });
    bInp.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ const v=parseFloat((bInp.value||'').replace(',','.')); if(!isNaN(v)){ transforms[idx].b=v; reflow(); } } });
    resetBtn.addEventListener('click', ()=>{ transforms[idx]={a:1,b:0}; aInp.value='1'; bInp.value='0'; reflow(); });
    row.appendChild(name); row.appendChild(aInp); row.appendChild(bInp); row.appendChild(resetBtn);
    controls.appendChild(row);
  });
  data.tracks.forEach((tr, idx)=>{
    const y = topPad + idx*(laneHeight+laneGap) + laneHeight*0.75;
    const text = document.createElementNS('http://www.w3.org/2000/svg','text');
    text.setAttribute('x', 8);
    text.setAttribute('y', y);
    text.setAttribute('class','laneLabel');
    text.textContent = tr.name;
    svg.appendChild(text);
  });
  data.tracks.forEach((tr, idx)=>{
    const yTop = topPad + idx*(laneHeight+laneGap);
    const {a,b} = transforms[idx]||{a:1,b:0};
    tr.cues.forEach(c=>{
      const s = a*c.s + b; const e = a*c.e + b; const ds = Math.max(0, s - tMin); const de = Math.max(0, e - tMin);
      const x = leftPad + (ds)*pxPerSec;
      const w = Math.max(1, (de-ds)*pxPerSec);
      const r = document.createElementNS('http://www.w3.org/2000/svg','rect');
      r.setAttribute('x', x);
      r.setAttribute('y', yTop);
      r.setAttribute('width', w);
      r.setAttribute('height', laneHeight);
      r.setAttribute('class','cue');
      r.addEventListener('mouseenter', ()=>{ r.classList.add('highlight'); });
      r.addEventListener('mouseleave', ()=>{ r.classList.remove('highlight'); });
      r.addEventListener('click', ()=>{ setTime(c.s); });
      const title = document.createElementNS('http://www.w3.org/2000/svg','title');
      title.textContent = c.t || '';
      r.appendChild(title);
      svg.appendChild(r);
    });
  });
  const ph = document.createElementNS('http://www.w3.org/2000/svg','line');
  ph.setAttribute('x1', leftPad);
  ph.setAttribute('x2', leftPad);
  ph.setAttribute('y1', 0);
  ph.setAttribute('y2', height);
  ph.setAttribute('class','playhead');
  svg.appendChild(ph);

  function redrawPlayhead(){
    const x = leftPad + ((currentT - tMin)*pxPerSec);
    ph.setAttribute('x1', x);
    ph.setAttribute('x2', x);
  }

  function setTime(t){
    currentT = Math.max(0, Math.min(t, data.duration_seconds));
    slider.value = String(Math.round((currentT/data.duration_seconds)*1000));
    redrawPlayhead();
  }

  slider.addEventListener('input', ()=>{
    const frac = Number(slider.value)/1000;
    setTime(frac*data.duration_seconds);
  });

  btnPlay.addEventListener('click', ()=>{
    if (playing) return;
    playing = true;
    playStartMs = performance.now();
    playStartT = currentT;
    requestAnimationFrame(tick);
  });
  btnPause.addEventListener('click', ()=>{ playing=false; });
  selSpeed.addEventListener('change', ()=>{});
  btnZIn.addEventListener('click', ()=>{ pxPerSec = Math.min(pxPerSec*1.25, 500); reflow(); });
  btnZOut.addEventListener('click', ()=>{ pxPerSec = Math.max(pxPerSec/1.25, 5); reflow(); });
  inputSearch.addEventListener('input', ()=>{
    const q = inputSearch.value.toLowerCase().trim();
    const rects = svg.querySelectorAll('rect.cue');
    rects.forEach(r=>{
      const title = r.getAttribute('title')||'';
      if (!q) r.classList.remove('highlight');
      else if (title.toLowerCase().includes(q)) r.classList.add('highlight');
      else r.classList.remove('highlight');
    });
  });

  function tick(ts){
    if (!playing) return;
    const speed = Number(selSpeed.value)||1;
    const dt = (ts - playStartMs)/1000 * speed;
    setTime(playStartT + dt);
    if (currentT >= data.duration_seconds) { playing=false; return; }
    const viewW = container.clientWidth;
    const margin=80;
    const x = leftPad + ((currentT - tMin)*pxPerSec);
    const rightEdge = container.scrollLeft + viewW;
    if (x > rightEdge - margin) {
      container.scrollLeft = Math.max(0, x - (viewW - margin));
    }
    requestAnimationFrame(tick);
  }

  setTime(0);
  function reflow(){
    const t = currentT; init(); setTime(t);
  }
}
})();
"""


def generate_static_preview(paths: Sequence[Path], out_dir: Path, *, title: str = "Subtitle Preview", open_browser: bool = False) -> dict:
	"""Generate a static subtitle viewer in out_dir for the given files.

	Returns a small dict with output path and files.
	"""
	paths = list(paths)
	if not paths:
		raise ValueError("No subtitle paths provided")
	# Build data.json structure
	tracks = []
	max_end = 0.0
	for p in paths:
		data = p.read_bytes()
		ext = p.suffix.lower().lstrip(".")
		if ext == "srt":
			segs = parse_srt_bytes(data)
		elif ext == "vtt":
			segs = parse_vtt_bytes(data)
		else:
			raise ValueError(f"Unsupported extension: {ext}")
		cues = []
		for s in segs:
			if s.end_seconds > s.start_seconds:
				cues.append({"s": float(s.start_seconds), "e": float(s.end_seconds), "t": s.text or ""})
				if s.end_seconds > max_end:
					max_end = float(s.end_seconds)
		tracks.append({"name": p.name, "cues": cues})

	out_dir.mkdir(parents=True, exist_ok=True)
	data_obj = {"title": title, "duration_seconds": float(max_end), "tracks": tracks}
	(out_dir / "data.json").write_text(json.dumps(data_obj, ensure_ascii=False, indent=2), encoding="utf-8")
	# Also inline the JSON in index.html so local file:// loads work without CORS
	(out_dir / "index.html").write_text(INDEX_HTML_TEMPLATE.format(title=title, data_json=json.dumps(data_obj)), encoding="utf-8")
	(out_dir / "viewer.css").write_text(VIEWER_CSS, encoding="utf-8")
	(out_dir / "viewer.js").write_text(VIEWER_JS, encoding="utf-8")

	if open_browser:
		webbrowser.open((out_dir / "index.html").absolute().as_uri())

	return {"out": str(out_dir), "files": [str(p) for p in paths]}


