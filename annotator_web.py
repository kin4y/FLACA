import os
import io
import uuid
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from skimage.color import rgb2lab, lab2rgb
from ColorClassifier_manual import ColorSegmenter, AnalysisConfig
from fpdf import FPDF
import base64

app = FastAPI(title="FLACA Precision Annotator")

# In-memory session storage
sessions = {}

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FLACA Web Workspace</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #121212; color: #e0e0e0; font-family: system-ui, sans-serif; overflow: hidden; height: 100vh; display: flex; flex-direction: column; }
        #top-nav { background: #1e1e1e; padding: 10px 20px; border-bottom: 1px solid #333; flex-shrink: 0; z-index: 100; }
        #main-container { display: flex; flex: 1; overflow: hidden; }
        #sidebar { width: 420px; background: #1e1e1e; border-right: 1px solid #333; padding: 20px; overflow-y: auto; flex-shrink: 0; }
        #workspace { flex: 1; position: relative; background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; }
        
        canvas { cursor: none; image-rendering: pixelated; outline: none; }
        
        .slider-group { margin-bottom: 15px; }
        .slider-group label { display: block; font-size: 0.8rem; margin-bottom: 3px; color: #888; text-transform: uppercase; }
        .slider-group span { color: #3498db; font-weight: bold; float: right; font-family: monospace; }
        input[type="range"] { width: 100%; height: 6px; background: #333; border-radius: 5px; outline: none; }
        
        #preview-pane { position: absolute; bottom: 20px; right: 20px; display: flex; flex-direction: column; gap: 15px; pointer-events: auto; }
        .mini-preview { width: 350px; background: rgba(25,25,25,0.95); border: 1px solid #444; border-radius: 8px; padding: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.8); }
        .mini-preview canvas { width: 100%; height: auto; border-radius: 4px; border: 1px solid #333; cursor: pointer; }
        .mini-preview h6 { font-size: 0.7rem; margin: 0 0 10px 0; color: #aaa; display: flex; justify-content: space-between; align-items: center; }
        
        .adj-panel { background: #1a1a1a; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 20px; }
        .adj-panel h6 { color: #3498db; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        
        .layer-item { background: #262626; border-radius: 8px; margin-bottom: 12px; border: 1px solid #333; overflow: hidden; }
        .layer-header { padding: 10px 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; background: #2d2d2d; }
        .layer-header:hover { background: #353535; }
        .sub-list { padding: 10px 15px; background: #1e1e1e; border-top: 1px solid #333; }
        .sub-item { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 0.8rem; }
        .sub-name-input { background: none; border: none; border-bottom: 1px solid #444; color: #fff; width: 150px; }
        
        .swatch { width: 34px; height: 34px; border-radius: 6px; border: 1px solid #555; cursor: pointer; transition: transform 0.1s; }
        .swatch:hover { transform: scale(1.1); border-color: #fff; }
        
        .btn-xs { padding: 2px 5px; font-size: 0.65rem; }
    </style>
</head>
<body>
    <div id="top-nav" class="d-flex justify-content-between align-items-center">
        <h5 class="m-0" style="color: #3498db; font-weight: 800;">🏺 FLACA <span style="color: #fff; font-weight: 300;">Engine</span></h5>
        <div class="d-flex align-items-center gap-2">
            <span id="status-msg" class="text-secondary me-3" style="font-size: 0.8rem;">Ready</span>
            <input type="file" id="file-input" class="d-none" accept="image/*">
            <input type="file" id="project-input" class="d-none" accept=".json">
            <button class="btn btn-sm btn-outline-light px-3" onclick="document.getElementById('file-input').click()">UPLOAD</button>
            <button class="btn btn-sm btn-outline-info ms-2 px-3" onclick="exportProject()">SAVE PROJECT</button>
            <button class="btn btn-sm btn-outline-warning ms-2 px-3" onclick="document.getElementById('project-input').click()">LOAD PROJECT</button>
            <button class="btn btn-sm btn-primary ms-2 px-3" onclick="previewReport()">GENERATE REPORT</button>
        </div>
    </div>

    <div id="main-container">
        <div id="sidebar">
            <!-- IMAGE ADJUSTMENTS SUBMENU -->
            <div class="adj-panel">
                <h6>✨ Global Image Adjustments <button class="btn btn-xs btn-outline-danger float-end" onclick="resetAdjustments()">RESET</button></h6>
                <div class="slider-group">
                    <label>Brightness <span id="v_bri">1.0</span></label>
                    <input type="range" class="adj-slider" id="s_bri" min="0" max="3" step="0.1" value="1.0">
                </div>
                <div class="slider-group">
                    <label>Contrast <span id="v_con">1.0</span></label>
                    <input type="range" class="adj-slider" id="s_con" min="0" max="3" step="0.1" value="1.0">
                </div>
                <div class="slider-group">
                    <label>Saturation <span id="v_sat">1.0</span></label>
                    <input type="range" class="adj-slider" id="s_sat" min="0" max="3" step="0.1" value="1.0">
                </div>
                <div class="slider-group">
                    <label>Hue Shift <span id="v_hue">0</span></label>
                    <input type="range" class="adj-slider" id="s_hue" min="-180" max="180" step="1" value="0">
                </div>
                <div class="slider-group">
                    <label>Gamma <span id="v_gam">1.0</span></label>
                    <input type="range" class="adj-slider" id="s_gam" min="0.1" max="3" step="0.1" value="1.0">
                </div>
                <div class="slider-group">
                    <label>Sharpness <span id="v_sha">1.0</span></label>
                    <input type="range" class="adj-slider" id="s_sha" min="0" max="5" step="0.1" value="1.0">
                </div>
                <div class="slider-group">
                    <label>Blur <span id="v_blu">0</span></label>
                    <input type="range" class="adj-slider" id="s_blu" min="0" max="10" step="1" value="0">
                </div>
                <div class="slider-group">
                    <label>Temperature <span id="v_tem">0</span></label>
                    <input type="range" class="adj-slider" id="s_tem" min="-100" max="100" step="1" value="0">
                </div>
                <div class="slider-group">
                    <label>Tint (G-M) <span id="v_tin">0</span></label>
                    <input type="range" class="adj-slider" id="s_tin" min="-100" max="100" step="1" value="0">
                </div>
                <div class="slider-group">
                    <label>Denoise <span id="v_den">0</span></label>
                    <input type="range" class="adj-slider" id="s_den" min="0" max="20" step="1" value="0">
                </div>
            </div>

            <div class="slider-group">
                <label>Target Color</label>
                <div class="d-flex gap-3 align-items-center bg-dark p-2 rounded border border-secondary">
                    <input type="color" id="color-picker" class="form-control form-control-color p-0 border-0" value="#3498db" style="width: 50px; height: 30px; background: none;">
                    <span id="hex-label" style="float:none; font-family: monospace;">#3498db</span>
                </div>
            </div>

            <div class="form-check form-switch mb-3 p-0" style="padding-left: 2.5em !important;">
                <input class="form-check-input" type="checkbox" id="pipette-toggle">
                <label class="form-check-label text-info fw-bold">🧪 PIPETTE MODE</label>
            </div>

            <div class="slider-group">
                <label>Pipette Brush Size <span id="brush-val">15</span></label>
                <input type="range" id="brush-slider" min="1" max="100" value="15">
            </div>

            <hr style="border-color: #444;">

            <div class="slider-group">
                <label>Selection Threshold <span id="thr-val">10</span></label>
                <input type="range" id="thr-slider" min="0" max="25" value="10">
            </div>

            <div class="adj-panel">
                <h6>⚖️ Similarity Weights</h6>
                <div class="slider-group"><label>L* <span id="wl-val">1.0</span></label><input type="range" id="wl-slider" min="0" max="5" step="0.1" value="1.0"></div>
                <div class="slider-group"><label>a* <span id="wa-val">1.0</span></label><input type="range" id="wa-slider" min="0" max="5" step="0.1" value="1.0"></div>
                <div class="slider-group"><label>b* <span id="wb-val">1.0</span></label><input type="range" id="wb-slider" min="0" max="5" step="0.1" value="1.0"></div>
            </div>

            <div class="form-check form-switch mb-3 p-0" style="padding-left: 2.5em !important;">
                <input class="form-check-input" type="checkbox" id="light-toggle">
                <label class="form-check-label fw-bold text-warning">💡 LIGHT ON (HIDE MASK)</label>
            </div>

            <div class="mb-3">
                <label class="form-label text-secondary fw-bold" style="font-size: 0.75rem;">ANNOTATION CATEGORY</label>
                <input type="text" id="layer-name" class="form-control form-control-sm bg-dark text-white border-secondary" placeholder="e.g. Moss">
                <button class="btn btn-outline-primary btn-sm w-100 mt-2" onclick="saveLayer()">SAVE & MERGE</button>
            </div>

            <div id="layer-list"></div>
            
            <hr style="border-color: #444;">
            <div class="adj-panel">
                <h6>🧬 FLACA Clustering</h6>
                <div class="slider-group"><label>K Clusters <span id="k-val-disp">5</span></label><input type="range" id="k-val" min="2" max="20" value="5" oninput="document.getElementById('k-val-disp').innerText=this.value"></div>
                <div class="slider-group"><label>L* Threshold <span id="l-val">10</span></label><input type="range" id="l-thresh" min="0" max="50" value="10" oninput="document.getElementById('l-val').innerText=this.value"></div>
                <div class="slider-group"><label>Chroma Threshold <span id="c-val">6</span></label><input type="range" id="c-thresh" min="0" max="25" value="6" oninput="document.getElementById('c-val').innerText=this.value"></div>
                <div class="slider-group"><label>ab_step <span id="ab-val">1.0</span></label><input type="range" id="ab-step" min="0.1" max="10" step="0.1" value="1.0" oninput="document.getElementById('ab-val').innerText=this.value"></div>
                <button class="btn btn-outline-info btn-sm w-100 mb-2" onclick="runClustering()">RUN ANALYSIS</button>
                <div id="cluster-results" class="d-flex flex-wrap gap-2"></div>
            </div>
        </div>

        <div id="workspace">
            <canvas id="main-canvas"></canvas>
            <div id="preview-pane">
                <div class="mini-preview"><h6>HEATMAP <button class="btn btn-xs btn-outline-info" onclick="toggleFullscreen('heat-canvas')">⛶</button></h6><canvas id="heat-canvas"></canvas></div>
                <div class="mini-preview"><h6>ISOLATION <button class="btn btn-xs btn-outline-info" onclick="toggleFullscreen('mask-canvas')">⛶</button></h6><canvas id="mask-canvas"></canvas></div>
            </div>
        </div>
    </div>

    <!-- Modals -->
    <div class="modal fade" id="previewModal" tabindex="-1"><div class="modal-dialog modal-xl"><div class="modal-content bg-dark text-white border-secondary"><div class="modal-header border-secondary bg-black"><h5 class="modal-title">📄 Analysis Preview</h5><button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button></div><div id="report-preview-body" class="modal-body bg-black" style="max-height: 75vh; overflow-y: auto; padding: 40px;"></div><div class="modal-footer border-secondary bg-black"><button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button><button type="button" class="btn btn-success" onclick="downloadPDF()">Export PDF</button></div></div></div></div>
    <div class="modal fade" id="fsModal" tabindex="-1"><div class="modal-dialog modal-fullscreen"><div class="modal-content bg-black"><div class="modal-header border-0 bg-dark"><button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button></div><div class="modal-body d-flex align-items-center justify-content-center p-0"><canvas id="fs-canvas" style="max-width: 95%; max-height: 95%; object-fit: contain;"></canvas></div></div></div></div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let originalImg = new Image(), sid = null;
        let zoom = 1.0, panX = 0, panY = 0, mouseX = 0, mouseY = 0, isDragging = false, lastMouseX = 0, lastMouseY = 0;
        const mainCanvas = document.getElementById('main-canvas'), ctx = mainCanvas.getContext('2d');
        const heatCanvas = document.getElementById('heat-canvas'), maskCanvas = document.getElementById('mask-canvas');

        function resizeCanvas() { mainCanvas.width = mainCanvas.parentElement.clientWidth; mainCanvas.height = mainCanvas.parentElement.clientHeight; }
        window.addEventListener('resize', resizeCanvas); resizeCanvas();
        function loop() { draw(); requestAnimationFrame(loop); } requestAnimationFrame(loop);

        document.getElementById('file-input').onchange = async (e) => {
            const formData = new FormData(); formData.append('file', e.target.files[0]); setStatus("Uploading...");
            const res = await fetch('/upload', { method: 'POST', body: formData }); const data = await res.json(); sid = data.sid; originalImg.src = 'data:image/jpeg;base64,' + data.image;
            originalImg.onload = () => { zoom = Math.min(mainCanvas.width/originalImg.width, mainCanvas.height/originalImg.height)*0.9; panX = mainCanvas.width/2; panY = mainCanvas.height/2; updateMask(); setStatus("Ready"); };
        };

        mainCanvas.onwheel = (e) => {
            if (!sid) return; e.preventDefault(); const rect = mainCanvas.getBoundingClientRect(); const mx = e.clientX-rect.left, my = e.clientY-rect.top;
            const delta = e.deltaY < 0 ? 1.15 : 0.85; const newZoom = Math.min(100, Math.max(0.1, zoom * delta));
            panX = mx - (mx - panX) * (newZoom / zoom); panY = my - (my - panY) * (newZoom / zoom); zoom = newZoom;
        };
        mainCanvas.onmousedown = (e) => { if (document.getElementById('pipette-toggle').checked) sampleColor(e); else { isDragging = true; lastMouseX = e.clientX; lastMouseY = e.clientY; } };
        window.onmousemove = (e) => { const rect = mainCanvas.getBoundingClientRect(); mouseX = e.clientX-rect.left; mouseY = e.clientY-rect.top; if (isDragging) { panX += (e.clientX-lastMouseX); panY += (e.clientY-lastMouseY); lastMouseX = e.clientX; lastMouseY = e.clientY; } };
        window.onmouseup = () => isDragging = false;

        function draw() {
            ctx.fillStyle = '#000'; ctx.fillRect(0, 0, mainCanvas.width, mainCanvas.height);
            if (!originalImg.src) return;
            const w = originalImg.width * zoom, h = originalImg.height * zoom;
            ctx.drawImage(originalImg, panX - w/2, panY - h/2, w, h);
            if (mouseX >= 0 && mouseX <= mainCanvas.width && mouseY >= 0 && mouseY <= mainCanvas.height) {
                const bSize = parseInt(document.getElementById('brush-slider').value); const cursorR = (bSize * zoom) / 2;
                ctx.save(); ctx.beginPath(); ctx.arc(mouseX, mouseY, cursorR, 0, Math.PI * 2); ctx.strokeStyle = 'white'; ctx.lineWidth = 2; ctx.stroke();
                ctx.beginPath(); ctx.arc(mouseX, mouseY, Math.max(0, cursorR - 1.5), 0, Math.PI * 2); ctx.strokeStyle = 'black'; ctx.lineWidth = 1; ctx.stroke();
                ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.fillRect(mouseX-6, mouseY, 12, 1); ctx.fillRect(mouseX, mouseY-6, 1, 12); ctx.restore();
            }
        }

        async function updateMask() {
            if (!sid) return;
            const adjs = {}; document.querySelectorAll('.adj-slider').forEach(s => adjs[s.id.split('_')[1]] = parseFloat(s.value));
            const params = { 
                sid, hex: document.getElementById('color-picker').value, 
                thr: document.getElementById('thr-slider').value, light: document.getElementById('light-toggle').checked,
                wl: document.getElementById('wl-slider').value, wa: document.getElementById('wa-slider').value, wb: document.getElementById('wb-slider').value,
                adjs: adjs
            };
            const res = await fetch('/process', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(params) });
            const data = await res.json(); const visImg = new Image(); visImg.src = 'data:image/jpeg;base64,' + data.vis; visImg.onload = () => { originalImg = visImg; };
            updateMini(heatCanvas, data.heat); updateMini(maskCanvas, data.mask);
        }

        function updateMini(canvas, b64) { const img = new Image(); img.src = 'data:image/jpeg;base64,' + b64; img.onload = () => { canvas.width = img.width; canvas.height = img.height; canvas.getContext('2d').drawImage(img, 0, 0); }; }

        async function sampleColor(e) {
            const w = originalImg.width * zoom, h = originalImg.height * zoom; const ix = (mouseX - (panX - w/2)) / zoom, iy = (mouseY - (panY - h/2)) / zoom;
            const res = await fetch('/sample', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ sid, x: Math.round(ix), y: Math.round(iy), brush: document.getElementById('brush-slider').value }) });
            const data = await res.json(); document.getElementById('color-picker').value = data.hex; document.getElementById('hex-label').innerText = data.hex; updateMask();
        }

        async function saveLayer() {
            const name = document.getElementById('layer-name').value; if (!name) return alert("Enter name");
            const res = await fetch('/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ sid, name }) });
            const data = await res.json(); renderLayers(data.layers);
        }

        function renderLayers(layers) {
            const list = document.getElementById('layer-list'); list.innerHTML = '<h6 class="text-secondary mt-3">SAVED CATEGORIES</h6>';
            for (const [name, data] of Object.entries(layers)) {
                const item = document.createElement('div'); item.className = 'layer-item';
                item.innerHTML = `<div class="layer-header" onclick="this.nextElementSibling.classList.toggle('d-none')"><span>${name}</span><span class="badge bg-primary">${data.total_pct.toFixed(2)}%</span></div>
                    <div class="sub-list d-none">${data.subs.map((s, i) => `<div class="sub-item"><input type="checkbox" ${s.active ? 'checked' : ''} onchange="toggleSub('${name}', ${i})"><div style="background:${s.hex}; width:12px; height:12px;"></div><input class="sub-name-input" value="${s.label || s.hex}" onchange="renameSub('${name}', ${i}, this.value)"><span class="ms-auto">${s.pct.toFixed(2)}%</span></div>`).join('')}<img src="data:image/png;base64,${data.mask}" class="mt-2 w-100 rounded"></div>`;
                list.appendChild(item);
            }
        }

        async function toggleSub(category, index) { const res = await fetch('/toggle_sublayer', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ sid, category, index }) }); const data = await res.json(); renderLayers(data.layers); updateMask(); }
        async function renameSub(category, index, newName) { await fetch('/rename_sublayer', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ sid, category, index, name: newName }) }); }
        async function runClustering() {
            const res = await fetch('/cluster', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ sid, k: document.getElementById('k-val').value, l: document.getElementById('l-thresh').value, c: document.getElementById('c-thresh').value, ab: document.getElementById('ab-step').value }) });
            const data = await res.json(); renderClusters(data.clusters);
        }
        function renderClusters(clusters) {
            const resDiv = document.getElementById('cluster-results'); resDiv.innerHTML = '';
            clusters.forEach(c => { const btn = document.createElement('div'); btn.className = 'swatch'; btn.style.backgroundColor = c.hex; btn.onclick = () => { document.getElementById('color-picker').value = c.hex; updateMask(); }; resDiv.appendChild(btn); });
        }

        async function previewReport() {
            const res = await fetch('/report_data?sid=' + sid); const data = await res.json();
            const body = document.getElementById('report-preview-body');
            body.innerHTML = `<div class="row mb-5"><div class="col-md-6 text-center"><h6>Original</h6><img src="data:image/jpeg;base64,${data.original}" class="w-100 rounded"></div><div class="col-md-6 text-center"><h6>Adjusted</h6><img src="data:image/jpeg;base64,${data.adjusted}" class="w-100 rounded"></div></div>`;
            data.layers.forEach(layer => { body.innerHTML += `<div class="mb-5 p-4 border border-secondary rounded bg-dark shadow"><div class="d-flex justify-content-between mb-3"><h4 class="m-0 text-info fw-bold">${layer.name}</h4><span class="badge bg-primary fs-5">${layer.total_pct.toFixed(2)}% Area</span></div><div class="row"><div class="col-md-7"><img src="data:image/jpeg;base64,${layer.image}" class="w-100 rounded border border-secondary" style="background:white;"></div><div class="col-md-5"><ul class="list-unstyled">${layer.subs.map(s => `<li><div class="d-flex align-items-center gap-2 mb-2"><div style="background:${s.hex}; width:15px; height:15px;"></div><span><b>${s.label || s.hex}</b>: ${s.pct.toFixed(2)}%</span></div></li>`).join('')}</ul></div></div></div>`; });
            new bootstrap.Modal('#previewModal').show();
        }

        async function exportProject() { const res = await fetch('/export_project?sid=' + sid); const data = await res.json(); const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], {type:'application/json'})); a.download = `flaca_project.json`; a.click(); }
        function downloadPDF() { window.open('/report?sid=' + sid, '_blank'); }
        function setStatus(msg) { document.getElementById('status-msg').innerText = msg; }
        function resetAdjustments() { document.querySelectorAll('.adj-slider').forEach(s => { s.value = 1.0; s.dispatchEvent(new Event('input')); }); updateMask(); }

        document.querySelectorAll('input[type="range"]').forEach(s => { s.oninput = (e) => { const valElem = document.getElementById(e.target.id.replace('s_', 'v_').replace('-slider', '-val')); if (valElem) valElem.innerText = e.target.value; if (!e.target.id.includes('brush') && !e.target.id.includes('k-val')) updateMask(); }; });
        document.getElementById('color-picker').oninput = updateMask; document.getElementById('light-toggle').onchange = updateMask;
    </script>
</body>
</html>
"""

# --- BACKEND LOGIC ---

@app.get("/", response_class=HTMLResponse)
async def index(): return HTML_TEMPLATE

def apply_adjustments(arr, adjs):
    img = Image.fromarray(arr)
    img = ImageEnhance.Brightness(img).enhance(adjs.get('bri', 1.0))
    img = ImageEnhance.Contrast(img).enhance(adjs.get('con', 1.0))
    img = ImageEnhance.Color(img).enhance(adjs.get('sat', 1.0))
    
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if adjs.get('hue', 0) != 0:
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + adjs['hue']) % 180
        cv_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if adjs.get('gam', 1.0) != 1.0:
        table = np.array([((i / 255.0) ** (1.0/adjs['gam'])) * 255 for i in np.arange(0, 256)]).astype("uint8")
        cv_img = cv2.LUT(cv_img, table)
    if adjs.get('blu', 0) > 0:
        k = int(adjs['blu']) * 2 + 1; cv_img = cv2.GaussianBlur(cv_img, (k, k), 0)
    if adjs.get('tem', 0) != 0 or adjs.get('tin', 0) != 0:
        b, g, r = cv2.split(cv_img.astype(np.float32)); r += adjs.get('tem', 0); b -= adjs.get('tem', 0); g += adjs.get('tin', 0)
        cv_img = cv2.merge([np.clip(b,0,255), np.clip(g,0,255), np.clip(r,0,255)]).astype(np.uint8)
    if adjs.get('den', 0) > 0:
        cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, adjs['den'], adjs['den'], 7, 21)
        
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    sid = str(uuid.uuid4()); contents = await file.read(); img = Image.open(io.BytesIO(contents)).convert('RGB')
    if max(img.size) > 1000: img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    arr = np.array(img)
    sessions[sid] = {"raw_orig": arr, "raw_adj": arr, "lab": rgb2lab(arr/255.0), "layers": {}, "adjs": {}, "clusters": [], "last_mask": None, "last_hex": "#3498db"}
    _, b = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return {"sid": sid, "image": base64.b64encode(b).decode()}

@app.post("/process")
async def process(data: dict):
    sid = data['sid']; sess = sessions.get(sid); adjs = data.get('adjs', {}); sess['adjs'] = adjs
    sess['raw_adj'] = apply_adjustments(sess['raw_orig'], adjs); sess['lab'] = rgb2lab(sess['raw_adj'] / 255.0)
    h = data['hex'].lstrip('#'); t_rgb = np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])
    t_lab = rgb2lab(t_rgb.reshape(1, 1, 3) / 255.0).flatten()
    wl, wa, wb = float(data.get('wl', 1)), float(data.get('wa', 1)), float(data.get('wb', 1))
    diff = sess['lab'] - t_lab
    dist = np.sqrt((wl * (diff[:,:,0]**2)) + (wa * (diff[:,:,1]**2)) + (wb * (diff[:,:,2]**2)))
    mask = dist <= float(data['thr']); sess.update({"last_mask": mask, "last_hex": data['hex'], "last_tol": data['thr'], "last_weights": (wl, wa, wb)})
    if data['light']: out = sess['raw_adj'].copy()
    else:
        out = (sess['raw_adj'] * (0.3 + 0.7 * (np.clip(1.0 - dist/100, 0, 1)**2))[:, :, np.newaxis]).astype(np.uint8)
        if np.any(mask):
            kernel = np.ones((3,3), np.uint8); edge = mask ^ cv2.erode(mask.astype(np.uint8), kernel).astype(bool); out[edge] = [255, 255, 255]
    p_stat = np.clip(1.0 - dist/100, 0, 1)**3
    heat = (np.full_like(sess['raw_adj'], 255) * (1.0 - p_stat)[:,:,np.newaxis] + t_rgb * p_stat[:,:,np.newaxis]).astype(np.uint8)
    iso = np.where(mask[:,:,None], sess['raw_adj'], 255).astype(np.uint8)
    def to_b64(img): return base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]).decode()
    return {"vis": to_b64(out), "heat": to_b64(heat), "mask": to_b64(iso)}

@app.post("/save")
async def save_layer(data: dict):
    sess = sessions.get(data['sid']); name = data['name']; mask = sess['last_mask']
    sub = {"mask": mask, "hex": sess['last_hex'], "pct": (np.sum(mask)/mask.size)*100, "thr": sess['last_tol'], "active": True, "label": f"Selection {len(sess['layers'].get(name, {'subs':[]})['subs'])+1}", "params": {"weights": sess['last_weights']}}
    if name in sess['layers']: sess['layers'][name]['subs'].append(sub)
    else: sess['layers'][name] = {"subs": [sub]}
    ui_layers = {}
    for n, d in sess['layers'].items():
        comp = np.zeros(sess['raw_adj'].shape[:2], dtype=bool)
        for s in d['subs']: 
            if s.get('active', True): comp = np.logical_or(comp, s['mask'])
        ui_layers[n] = {"total_pct": (np.sum(comp)/comp.size)*100, "mask": base64.b64encode(cv2.imencode(".png", (comp*255).astype(np.uint8))[1]).decode(), "subs": [{"hex": s['hex'], "pct": s['pct'], "thr": s['thr'], "active": s['active'], "label": s['label'], "params": s['params']} for s in d['subs']]}
    return {"layers": ui_layers}

@app.post("/toggle_sublayer")
async def toggle_sub(data: dict):
    sess = sessions.get(data['sid']); cat = data['category']; idx = data['index']
    sess['layers'][cat]['subs'][idx]['active'] = not sess['layers'][cat]['subs'][idx]['active']
    ui_layers = {}
    for n, d in sess['layers'].items():
        comp = np.zeros(sess['raw_adj'].shape[:2], dtype=bool)
        for s in d['subs']: 
            if s.get('active', True): comp = np.logical_or(comp, s['mask'])
        ui_layers[n] = {"total_pct": (np.sum(comp)/comp.size)*100, "mask": base64.b64encode(cv2.imencode(".png", (comp*255).astype(np.uint8))[1]).decode(), "subs": [{"hex": s['hex'], "pct": s['pct'], "thr": s['thr'], "active": s['active'], "label": s['label'], "params": s['params']} for s in d['subs']]}
    return {"layers": ui_layers}

@app.post("/rename_sublayer")
async def rename_sub(data: dict):
    sess = sessions.get(data['sid']); sess['layers'][data['category']]['subs'][data['index']]['label'] = data['name']
    return {"status": "ok"}

@app.post("/sample")
async def sample(data: dict):
    sess = sessions.get(data['sid']); x, y, b = data['x'], data['y'], int(data['brush']); h, w = sess['raw_adj'].shape[:2]; bh = b // 2
    reg = sess['raw_adj'][max(0, y-bh):min(h, y+bh+1), max(0, x-bh):min(w, x+bh+1)]
    if reg.size == 0: return {"hex": sess['last_hex']}
    return {"hex": '#%02x%02x%02x' % tuple(np.mean(reg, axis=(0,1)).astype(np.uint8))}

@app.post("/cluster")
async def cluster(data: dict):
    sess = sessions.get(data['sid']); cfg = AnalysisConfig(n_clusters=int(data['k']), L_thresh=float(data['l']), C_thresh=float(data['c']), ab_step=float(data['ab'])); seg = ColorSegmenter(config=cfg, raw_rgb=sess['raw_adj'].reshape(-1, 3)); seg.run_clustering()
    results = [{"label": v['label'], "percent": v['percent'], "hex": '#%02x%02x%02x' % tuple(v['rgb'])} for k, v in seg.clusters.items()]
    sess['clusters'] = results; return {"clusters": results}

@app.get("/report_data")
async def report_data(sid: str):
    sess = sessions.get(sid)
    if not sess: return {"error": "Invalid session"}
    to_b64 = lambda img: base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1]).decode()
    layers = []
    for name, data in sess['layers'].items():
        comp = np.zeros(sess['raw_adj'].shape[:2], dtype=bool)
        for s in data['subs']: 
            if s.get('active', True): comp = np.logical_or(comp, s['mask'])
        iso = np.full_like(sess['raw_adj'], 255); iso[comp] = sess['raw_adj'][comp]
        
        # Filter subs to only include serializable metadata
        filtered_subs = [
            {
                "hex": s['hex'], 
                "pct": s['pct'], 
                "thr": s['thr'], 
                "label": s['label'], 
                "params": s['params']
            } for s in data['subs'] if s['active']
        ]
        
        layers.append({
            "name": name, 
            "total_pct": (np.sum(comp)/comp.size)*100, 
            "image": to_b64(iso), 
            "subs": filtered_subs
        })
    return {"original": to_b64(sess['raw_orig']), "adjusted": to_b64(sess['raw_adj']), "adjs": sess['adjs'], "layers": layers}

@app.get("/report")
async def report(sid: str):
    sess = sessions.get(sid); pdf = FPDF(); pdf.add_page(); pdf.set_font("helvetica", "B", 20); pdf.set_text_color(52, 152, 219); pdf.cell(0, 15, "FLACA Absolute Reproductivity Report", ln=1, align="C"); pdf.ln(10)
    for name, data in sess['layers'].items():
        pdf.add_page(); pdf.set_font("helvetica", "B", 16); pdf.set_text_color(0); pdf.cell(0, 12, f"Category: {name}", ln=1)
        comp = np.zeros(sess['raw_adj'].shape[:2], dtype=bool)
        for s in data['subs']: 
            if s.get('active', True): comp = np.logical_or(comp, s['mask'])
        iso = np.full_like(sess['raw_adj'], 255); iso[comp] = sess['raw_adj'][comp]
        p = f"temp_{uuid.uuid4()}.jpg"; Image.fromarray(iso).save(p); pdf.image(p, x=10, w=140); os.remove(p)
        for s in data['subs']:
            if not s['active']: continue
            pdf.set_font("helvetica", "B", 10); pdf.cell(0, 6, f"- {s['label'] or s['hex']}: {s['pct']:.2f}% Area", ln=1)
            pdf.set_font("helvetica", "", 8); pdf.set_text_color(120); pdf.cell(0, 5, f"  Weights(L:{s['params']['weights'][0]}, a:{s['params']['weights'][1]}, b:{s['params']['weights'][2]}) | Thr: {s['thr']}", ln=1); pdf.set_text_color(0)
    pdf.add_page(); pdf.set_font("helvetica", "B", 14); pdf.cell(0, 10, "Data Audit Trail", ln=1); pdf.ln(5)
    p1 = f"t1_{sid}.jpg"; Image.fromarray(sess['raw_orig']).save(p1); pdf.image(p1, x=10, w=180); os.remove(p1); pdf.ln(5)
    p2 = f"t2_{sid}.jpg"; Image.fromarray(sess['raw_adj']).save(p2); pdf.image(p2, x=10, w=180); os.remove(p2)
    pdf.set_font("helvetica", "", 9); [pdf.cell(0, 6, f"  {k}: {v}", ln=1) for k, v in sess['adjs'].items()]
    buf = io.BytesIO(); pdf.output(buf); buf.seek(0); return StreamingResponse(buf, media_type="application/pdf", headers={"Content-Disposition": "attachment;filename=report.pdf"})

@app.get("/export_project")
async def export_project(sid: str):
    sess = sessions.get(sid); layers_export = {}
    for name, data in sess['layers'].items():
        subs_export = []
        for s in data['subs']:
            _, b = cv2.imencode(".png", (s['mask']*255).astype(np.uint8))
            subs_export.append({"mask_png": base64.b64encode(b).decode(), "hex": s['hex'], "pct": s['pct'], "thr": s['thr'], "active": s['active'], "label": s['label'], "params": s['params']})
        layers_export[name] = {"subs": subs_export}
    _, img_b = cv2.imencode(".jpg", cv2.cvtColor(sess['raw_orig'], cv2.COLOR_RGB2BGR))
    return {"version": 3.0, "image_jpg": base64.b64encode(img_b).decode(), "layers": layers_export, "adjs": sess['adjs'], "clusters": sess['clusters']}

@app.post("/import_project")
async def import_project(data: dict):
    session_id = str(uuid.uuid4()); arr = np.array(Image.open(io.BytesIO(base64.b64decode(data['image_jpg']))).convert('RGB')); lab = rgb2lab(arr/255.0); layers = {}
    for name, l_data in data['layers'].items():
        subs = []
        for s in l_data['subs']:
            mask_arr = cv2.imdecode(np.frombuffer(base64.b64decode(s['mask_png']), np.uint8), cv2.IMREAD_GRAYSCALE)
            subs.append({"mask": (mask_arr > 127), "hex": s['hex'], "pct": s['pct'], "thr": s['thr'], "active": s.get('active', True), "label": s.get('label', ''), "params": s['params']})
        layers[name] = {"subs": subs}
    sessions[session_id] = {"raw_orig": arr, "raw_adj": arr, "lab": rgb2lab(arr/255.0), "layers": layers, "adjs": data.get('adjs', {}), "clusters": data.get('clusters', []), "last_mask": None, "last_hex": "#3498db"}
    ui_layers = {}
    for n, d in layers.items():
        comp = np.zeros(arr.shape[:2], dtype=bool)
        for s in d['subs']: 
            if s.get('active', True): comp = np.logical_or(comp, s['mask'])
        ui_layers[n] = {"total_pct": (np.sum(comp)/comp.size)*100, "mask": base64.b64encode(cv2.imencode(".png", (comp*255).astype(np.uint8))[1]).decode(), "subs": [{"hex": s['hex'], "pct": s['pct'], "thr": s['thr'], "active": s['active'], "label": s['label'], "params": s['params']} for s in d['subs']]}
    return {"sid": session_id, "image": base64.b64encode(cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))[1]).decode(), "layers": ui_layers, "clusters": sessions[session_id]["clusters"], "adjs": data.get('adjs', {})}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
