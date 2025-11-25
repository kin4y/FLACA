#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn
import numpy as np
import cv2
import io
import json
from typing import List, Tuple

app = FastAPI(title="Polygon Cutter Web UI")

# reuse processing function from the previous script but simplified
def cut_polygons_from_image_bytes(image_bytes: bytes, polygons, background=None, export_alpha=True):
    # decode image bytes to BGR
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_list = []
    for poly in polygons:
        if len(poly) >= 3:
            pts_list.append(np.array(poly, np.int32).reshape((-1,1,2)))
    if pts_list:
        cv2.fillPoly(mask, pts_list, 255)
    b, g, r = cv2.split(img)
    alpha = mask.copy()
    if export_alpha:
        out = cv2.merge((b,g,r,alpha))
        ext = ".png"
        is_success, buffer = cv2.imencode(ext, out)
        return buffer.tobytes(), "image/png"
    else:
        if background is None:
            background = (255,255,255)
        bg_img = np.zeros_like(img)
        bg_img[:] = background
        mask_bool = mask.astype(bool)
        out = bg_img.copy()
        out[mask_bool] = img[mask_bool]
        is_success, buffer = cv2.imencode(".png", out)
        return buffer.tobytes(), "image/png"

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Polygon Cutter</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 12px; }
    #canvas { border:1px solid #333; max-width: 90vw; max-height: 70vh; }
    .toolbar { margin: 8px 0; }
    .swatch { display:inline-block; width:22px; height:16px; border:1px solid #111; margin-right:6px; cursor:pointer; }
    .active { outline: 3px solid yellow; }
    .button { padding:6px 10px; margin-right:8px; cursor:pointer; }
  </style>
</head>
<body>
  <h2>Polygon Cutter (web)</h2>
  <div class="toolbar">
    <input id="file" type="file" accept="image/*">
    <button id="complete" class="button">Complete Polygon</button>
    <button id="newpoly" class="button">New Polygon</button>
    <button id="undo" class="button">Undo Last</button>
    <button id="delete" class="button">Delete Poly Under Mouse</button>
    <button id="export_alpha" class="button">Export PNG (alpha)</button>
    <button id="export_flat" class="button">Export PNG (background)</button>
  </div>
  <div>
    Background: 
    <span class="swatch" data-color="255,255,255" style="background:rgb(255,255,255)"></span>
    <span class="swatch" data-color="0,0,0" style="background:rgb(0,0,0)"></span>
    <span class="swatch" data-color="127,127,127" style="background:rgb(127,127,127)"></span>
    <span class="swatch" data-color="255,0,0" style="background:rgb(255,0,0)"></span>
    <span class="swatch" data-color="0,255,0" style="background:rgb(0,255,0)"></span>
    <span class="swatch" data-color="0,0,255" style="background:rgb(0,0,255)"></span>
  </div>
  <canvas id="canvas"></canvas>
  <script>
    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    let img = new Image();
    let polygons = []; // array of arrays of [x,y]
    let current = [];
    let mouse = {x:0,y:0};
    let selectedBg = "255,255,255";
    let imgBytes = null; // ArrayBuffer for upload
    const swatches = document.querySelectorAll('.swatch');
    swatches.forEach(s => {
      s.addEventListener('click', () => {
        swatches.forEach(x => x.classList.remove('active'));
        s.classList.add('active');
        selectedBg = s.dataset.color;
      });
    });
    swatches[0].classList.add('active');

    document.getElementById('file').addEventListener('change', async (ev) => {
      const f = ev.target.files[0];
      if(!f) return;
      const r = new FileReader();
      r.onload = function(e) {
        img.src = e.target.result;
        img.onload = function() {
          canvas.width = img.width;
          canvas.height = img.height;
          draw();
        }
      }
      imgBytes = await f.arrayBuffer();
      r.readAsDataURL(f);
    });

    canvas.addEventListener('mousemove', (ev) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = Math.round((ev.clientX - rect.left) * (canvas.width / rect.width));
      mouse.y = Math.round((ev.clientY - rect.top) * (canvas.height / rect.height));
      draw();
    });

    canvas.addEventListener('click', (ev) => {
      current.push([mouse.x, mouse.y]);
      draw();
    });

    document.getElementById('complete').addEventListener('click', () => {
      if(current.length >= 3) {
        polygons.push(current.slice());
        current = [];
        draw();
      } else {
        alert('Need at least 3 points to complete polygon');
      }
    });
    document.getElementById('newpoly').addEventListener('click', () => { current = []; draw(); });
    document.getElementById('undo').addEventListener('click', () => { polygons.pop(); draw(); });
    document.getElementById('delete').addEventListener('click', () => {
      // find polygon under mouse by point-in-polygon
      let idx = polyIndexUnderPoint(mouse);
      if(idx >= 0) { polygons.splice(idx,1); draw(); } else { alert('No polygon under pointer'); }
    });

    function polyIndexUnderPoint(pt) {
      for(let i=0;i<polygons.length;i++){
        const poly = polygons[i];
        if(pointInPoly(pt, poly)) return i;
      }
      return -1;
    }

    function pointInPoly(pt, poly) {
      // ray casting
      let x = pt.x, y = pt.y;
      let inside = false;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        let xi = poly[i][0], yi = poly[i][1];
        let xj = poly[j][0], yj = poly[j][1];
        let intersect = ((yi > y) != (yj > y)) &&
                        (x < (xj - xi) * (y - yi) / (yj - yi + 0.0) + xi);
        if (intersect) inside = !inside;
      }
      return inside;
    }

    function draw() {
      if(!img.src) {
        ctx.clearRect(0,0,canvas.width, canvas.height);
        return;
      }
      ctx.drawImage(img, 0, 0);
      // draw completed polygons
      ctx.lineWidth = 2;
      for(let poly of polygons) {
        if(poly.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(poly[0][0], poly[0][1]);
          for(let i=1;i<poly.length;i++) ctx.lineTo(poly[i][0], poly[i][1]);
          ctx.closePath();
          ctx.fillStyle = 'rgba(0,200,0,0.15)';
          ctx.fill();
          ctx.strokeStyle = 'rgb(0,200,0)';
          ctx.stroke();
        }
      }
      // draw current
      if(current.length>0) {
        ctx.beginPath();
        ctx.moveTo(current[0][0], current[0][1]);
        for(let i=1;i<current.length;i++) ctx.lineTo(current[i][0], current[i][1]);
        ctx.strokeStyle = 'rgb(200,0,0)';
        ctx.stroke();
        for(let p of current) {
          ctx.beginPath();
          ctx.arc(p[0], p[1], 4, 0, Math.PI*2);
          ctx.fillStyle = 'rgb(200,0,0)';
          ctx.fill();
        }
      }
      // mouse coords
      ctx.fillStyle = 'white';
      ctx.fillRect(5,5,120,18);
      ctx.fillStyle = 'black';
      ctx.fillText('Mouse: '+mouse.x+','+mouse.y, 8, 18);
    }

    async function doExport(alpha) {
      if(!imgBytes) { alert('Upload an image first'); return; }
      // ensure current poly isn't lost
      let payload = {
        polygons: polygons.concat(current.length>=3 ? [current] : []),
        background: selectedBg,
        alpha: alpha
      };
      const form = new FormData();
      const blob = new Blob([imgBytes]);
      form.append('image', blob, 'image.png');
      form.append('meta', JSON.stringify(payload));
      const resp = await fetch('/process', { method:'POST', body: form });
      if(!resp.ok) {
        alert('Processing failed: ' + resp.statusText);
        return;
      }
      const buf = await resp.arrayBuffer();
      const url = URL.createObjectURL(new Blob([buf], { type: 'image/png' }));
      const a = document.createElement('a');
      a.href = url;
      a.download = alpha ? 'cut_alpha.png' : 'cut_flat.png';
      a.click();
      URL.revokeObjectURL(url);
    }

    document.getElementById('export_alpha').addEventListener('click', () => doExport(true));
    document.getElementById('export_flat').addEventListener('click', () => doExport(false));
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

@app.post("/process")
async def process(image: UploadFile = File(...), meta: str = Form(...)):
    """
    meta is JSON with keys:
    - polygons: list of polygons, each list of [x,y]
    - background: "R,G,B" string
    - alpha: boolean
    """
    meta_obj = json.loads(meta)
    polygons = meta_obj.get("polygons", [])
    background_str = meta_obj.get("background", "255,255,255")
    alpha = bool(meta_obj.get("alpha", True))
    # convert background string to BGR tuple
    try:
        r,g,b = [int(x) for x in background_str.split(",")]
        bg = (b,g,r)
    except Exception:
        bg = (255,255,255)
    image_bytes = await image.read()
    out_bytes, mime = cut_polygons_from_image_bytes(image_bytes, polygons, background=bg, export_alpha=alpha)
    return StreamingResponse(io.BytesIO(out_bytes), media_type=mime)

if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
