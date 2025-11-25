from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse # NEW IMPORT
import uvicorn
import tempfile
import shutil
import os
import uuid
import matplotlib
import numpy as np
import cv2
import io
import json

# Set Matplotlib backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import custom modules
from ColorClassifier_manual import k_color_analysis, visualize_color_cluster
# from stain_detection import detect_stains # REMOVED: Stain Detection Import

app = FastAPI(title="Archaeological Image Analysis Suite")

# ---------- Configuration ---------- #
STATIC_DIR = "static_results"
TEMP_DIR = "temp_uploads"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app.mount("/static_results", StaticFiles(directory=STATIC_DIR), name="static_results")
app.mount("/temp_uploads", StaticFiles(directory=TEMP_DIR), name="temp_uploads")

active_bundles = {}

# ---------- Defaults ---------- #
color_defaults = {
    "k": 5, "L_thresh": 30.0, "C_thresh": 0.1, "ab_step": 1.0, "point_size": 12,
    "size_mode": "sqrt", "top_n_chroma": "", "top_n_achro": "",
    "pie_show_labels": "True", "show_input": "True", "show_plots_initial": "False",
    "show_plots_final": "True", "random_seed": 42, "shrink_img": 1.0,
}

# stain_defaults = { # REMOVED: Stain Defaults
#     "dark_thresh": 120, "sat_thresh": 60, "diff_thresh": 25,
# }

# ---------- Helpers ---------- #
def save_all_open_figures(prefix="plot"):
    urls = []
    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        path = os.path.join(STATIC_DIR, f"{prefix}_{i}_{uuid.uuid4().hex}.png")
        fig.savefig(path, bbox_inches="tight", dpi=100)
        urls.append(f"/static_results/{os.path.basename(path)}")
    plt.close("all")
    return urls

def cut_polygons_from_image_bytes(image_bytes: bytes, polygons, background=None, export_alpha=True, crop_to_poly=False):
    """
    Processes the image with polygon masking, background replacement, and optional cropping.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    h, w = img.shape[:2]
    
    # 1. Prepare Masks & Points
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_list = []
    all_points = []
    
    for poly in polygons:
        if len(poly) >= 3:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            pts_list.append(pts)
            all_points.append(pts)
    
    if pts_list:
        cv2.fillPoly(mask, pts_list, 255)

    # 2. CROP LOGIC
    if crop_to_poly and all_points:
        combined_pts = np.concatenate(all_points)
        x, y, w_rect, h_rect = cv2.boundingRect(combined_pts)
        
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w_rect = min(w - x, w_rect + 2*padding)
        h_rect = min(h - y, h_rect + 2*padding)
        
        img = img[y:y+h_rect, x:x+w_rect]
        mask = mask[y:y+h_rect, x:x+w_rect]
    
    # 3. Prepare Background
    if background is None:
        background = (255, 255, 255) # White default (BGR)
    
    bg_img = np.zeros_like(img)
    bg_img[:] = background
    
    # 4. Combine
    mask_bool = mask.astype(bool)
    combined_img = bg_img.copy()
    combined_img[mask_bool] = img[mask_bool]
    
    # 5. Handle Export
    if export_alpha:
        b, g, r = cv2.split(combined_img)
        alpha = mask.copy()
        out = cv2.merge((b, g, r, alpha))
    else:
        out = combined_img
    
    is_success, buffer = cv2.imencode(".png", out)
    return buffer.tobytes(), "image/png"


def modern_style():
    """Returns the CSS for the modern DARK MODE UI."""
    return """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Dark Mode Palette (Zinc) */
            --bg-body: #09090b;       /* Zinc 950 */
            --bg-panel: #18181b;      /* Zinc 900 */
            --bg-element: #27272a;    /* Zinc 800 */
            
            --text-main: #f4f4f5;     /* Zinc 100 */
            --text-sub: #a1a1aa;      /* Zinc 400 */
            
            --border: #3f3f46;        /* Zinc 700 */
            
            --primary: #0d9488;       /* Teal 600 */
            --primary-hover: #0f766e; /* Teal 700 */
            --accent: #ea580c;        /* Orange 600 */
            
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
            
            --radius: 0.5rem;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }

        /* Header */
        header {
            background: var(--bg-panel);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-sm);
        }
        header h1 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        header h1 span { color: var(--primary); }
        
        /* Nav */
        nav {
            background: var(--bg-panel);
            padding: 0.5rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
            justify-content: center;
        }
        nav a {
            color: var(--text-sub);
            text-decoration: none;
            padding: 0.5rem 1rem;
            border-radius: 0.375rem;
            font-weight: 500;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        nav a:hover {
            background-color: var(--bg-element);
            color: var(--text-main);
        }

        /* Layout */
        main {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1.5rem;
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 2rem;
            align-items: start;
        }
        main.full-width {
            grid-template-columns: 1fr;
            max-width: 1000px;
        }

        /* Panels */
        .panel {
            background: var(--bg-panel);
            border-radius: var(--radius);
            box-shadow: var(--shadow-md);
            padding: 1.5rem;
            border: 1px solid var(--border);
        }

        /* Sidebar Inputs */
        label {
            display: block;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            color: var(--text-sub);
            margin-bottom: 0.4rem;
            margin-top: 1.2rem;
        }
        label:first-of-type { margin-top: 0; }

        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 0.6rem;
            background-color: var(--bg-body);
            border: 1px solid var(--border);
            color: var(--text-main);
            border-radius: 0.375rem;
            font-family: inherit;
            box-sizing: border-box;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: 2px solid var(--primary);
            border-color: transparent;
        }
        input[type="file"] {
            font-size: 0.875rem;
            color: var(--text-sub);
            margin-top: 0.25rem;
        }

        /* Buttons */
        .btn, input[type="submit"], button {
            display: inline-flex;
            justify-content: center;
            align-items: center;
            padding: 0.6rem 1.2rem;
            border: none;
            border-radius: 0.375rem;
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            text-decoration: none;
            width: 100%;
            box-sizing: border-box;
            margin-top: 1rem;
        }
        .btn:hover, input[type="submit"]:hover, button:hover {
            background-color: var(--primary-hover);
        }
        .btn-secondary {
            background-color: var(--bg-element);
            color: var(--text-main);
            border: 1px solid var(--border);
        }
        .btn-secondary:hover {
            background-color: var(--border);
        }
        .btn-danger {
            background-color: #7f1d1d; /* Dark red */
            color: #fecaca;
        }
        .btn-danger:hover { background-color: #991b1b; }

        /* Toolbars (Horizontal) */
        .toolbar {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        .toolbar .btn {
            width: auto;
            margin-top: 0;
            font-size: 0.8rem;
            padding: 0.4rem 0.8rem;
        }

        /* Swatches Grid */
        .swatch-container { 
            display: grid; 
            grid-template-columns: repeat(6, 1fr); 
            gap: 6px; 
            margin-top: 0.5rem; 
        }
        .swatch {
            width: 100%;
            aspect-ratio: 1;
            border-radius: 4px;
            cursor: pointer;
            border: 2px solid var(--border);
            transition: transform 0.1s;
        }
        .swatch:hover { transform: scale(1.1); z-index:2; border-color:white; }
        .swatch.active {
            border-color: white;
            box-shadow: 0 0 0 2px var(--primary);
            transform: scale(1.1);
            z-index:2;
        }

        /* Canvas */
        #canvas-container {
            width: 100%;
            height: 65vh;
            background-color: #000000;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            position: relative;
            cursor: crosshair;
        }
        
        /* Info Boxes */
        .info-box {
            background-color: rgba(13, 148, 136, 0.1); /* Teal tint */
            border-left: 3px solid var(--primary);
            padding: 1rem;
            border-radius: 4px;
            font-size: 0.9rem;
            color: #ccfbf1; /* Teal 100 */
            margin: 1rem 0;
        }
        .info-box strong { color: white; }
        .info-box a { color: var(--primary); }
        
        /* Landing Page Grid */
        .landing-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 3rem;
        }

        /* Gallery & Lightbox */
        /* Small gallery items for cluster inspections */
        .gallery-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); 
            gap: 10px; 
            margin-top: 15px; 
        }
        .gallery-item { 
            position: relative; 
            cursor: pointer; 
            border: 1px solid var(--border); 
            border-radius: 4px; 
            overflow: hidden; 
            transition: all 0.2s; 
            background: var(--bg-element); 
        }
        .gallery-item:hover { border-color: var(--primary); transform: translateY(-2px); box-shadow: var(--shadow-sm); }
        .gallery-item img { width: 100%; height: 100px; object-fit: cover; display: block; }
        .gallery-item span { 
            position: absolute; 
            bottom: 0; left: 0; right: 0; 
            background: rgba(0,0,0,0.7); 
            color: white; 
            font-size: 0.7rem; 
            padding: 2px 5px; 
            text-align: center;
        }
        
        /* Large Primary Plot Items */
        .gallery-item-large {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            position: relative;
            cursor: pointer;
            transition: all 0.2s;
            background: #000; /* Plots are on black background */
        }
        .gallery-item-large:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        .gallery-item-large img {
            width: 100%;
            height: auto; 
            max-height: 400px; /* Max size for core plots */
            object-fit: contain;
            display: block;
        }
        .gallery-item-large span {
            position: absolute;
            top: 0; left: 0; 
            background: rgba(0,0,0,0.8);
            color: var(--text-main);
            font-size: 0.8rem;
            padding: 4px 8px;
            border-bottom-right-radius: 4px;
        }


        #lightbox { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.95); z-index: 9999; display: flex; justify-content: center; align-items: center; visibility: hidden; opacity: 0; transition: opacity 0.2s; }
        #lightbox.active { visibility: visible; opacity: 1; }
        #lightbox-canvas { max-width: 95%; max-height: 95%; cursor: grab; }
        #lightbox-canvas:active { cursor: grabbing; }
        #lightbox-close { position: absolute; top: 20px; right: 30px; color: white; font-size: 2rem; cursor: pointer; z-index: 10000; }
        #lightbox-info { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); color: #aaa; font-size: 0.9rem; pointer-events: none; }
        
        /* Utilities */
        .hidden { display: none; }
        hr { border: 0; border-top: 1px solid var(--border); margin: 2rem 0; }
        footer {
            text-align: center;
            padding: 2rem;
            color: var(--text-sub);
            font-size: 0.8rem;
            margin-top: auto;
            border-top: 1px solid var(--border);
        }
    </style>
    <script>
        let lb_img = new Image();
        let lb_scale = 1, lb_originX = 0, lb_originY = 0;
        let lb_isPanning = false, lb_startX = 0, lb_startY = 0;
        
        function openLightbox(url) {
            const lb = document.getElementById('lightbox');
            const cvs = document.getElementById('lightbox-canvas');
            const ctx = cvs.getContext('2d');
            
            lb.classList.add('active');
            lb_img.src = url;
            lb_img.onload = () => {
                cvs.width = window.innerWidth;
                cvs.height = window.innerHeight;
                // Fit image
                lb_scale = Math.min((cvs.width-100)/lb_img.width, (cvs.height-100)/lb_img.height);
                lb_originX = (cvs.width - lb_img.width * lb_scale) / 2;
                lb_originY = (cvs.height - lb_img.height * lb_scale) / 2;
                drawLightbox();
            }
            
            // Event Listeners for Zoom/Pan
            cvs.onwheel = (e) => {
                e.preventDefault();
                const rect = cvs.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                const worldX = (mx - lb_originX) / lb_scale;
                const worldY = (my - lb_originY) / lb_scale;
                
                const factor = e.deltaY < 0 ? 1.1 : 0.9;
                lb_scale *= factor;
                
                lb_originX = mx - worldX * lb_scale;
                lb_originY = my - worldY * lb_scale;
                drawLightbox();
            };
            
            cvs.onmousedown = (e) => {
                lb_isPanning = true;
                lb_startX = e.clientX - lb_originX;
                lb_startY = e.clientY - lb_originY;
                cvs.style.cursor = 'grabbing';
            };
            window.onmouseup = () => { lb_isPanning = false; document.getElementById('lightbox-canvas').style.cursor = 'grab'; };
            cvs.onmousemove = (e) => {
                if(!lb_isPanning) return;
                lb_originX = e.clientX - lb_startX;
                lb_originY = e.clientY - lb_startY;
                drawLightbox();
            };
            cvs.oncontextmenu = e => e.preventDefault();
            
            // ESC key to close
            document.onkeydown = (e) => {
                if (e.key === 'Escape' && lb.classList.contains('active')) {
                    closeLightbox();
                }
            };
        }
        
        function drawLightbox() {
            const cvs = document.getElementById('lightbox-canvas');
            const ctx = cvs.getContext('2d');
            ctx.clearRect(0,0,cvs.width, cvs.height);
            ctx.save();
            ctx.translate(lb_originX, lb_originY);
            ctx.scale(lb_scale, lb_scale);
            ctx.drawImage(lb_img, 0, 0);
            ctx.restore();
        }
        
        function closeLightbox() {
            document.getElementById('lightbox').classList.remove('active');
            window.onmouseup = null;
            document.onkeydown = null; 
        }
    </script>
    """

def page_layout(main, sidebar=None, show_nav=True):
    nav_html = """<nav><a href="/">🏠 Home</a><a href="/polygon_cutter">✂️ Polygon Cutter</a><a href="/color_analysis">🎨 Color Analysis</a></nav>""" if show_nav else "" # MODIFIED: Removed Stain Detection link
    content = f"""<main><div class="panel sidebar">{sidebar}</div><div class="panel analysis">{main}</div></main>""" if sidebar else f"""<main class="full-width"><div class="panel">{main}</div></main>"""
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Archaeological Analysis</title>
        {modern_style()}
    </head>
    <body>
        <header><h1><span>🏺</span> Archaeological Analysis</h1></header>
        {nav_html}
        {content}
        <div id="lightbox">
            <div id="lightbox-close" onclick="closeLightbox()">&times;</div>
            <canvas id="lightbox-canvas"></canvas>
            <div id="lightbox-info">Scroll to Zoom • Drag to Pan • ESC to Close</div>
        </div>
        <footer>Built for El Tajín, Veracruz</footer>
    </body>
    </html>
    """

# ---------- Routes ---------- #

@app.get("/", response_class=HTMLResponse)
async def home():
    main = """
    <div style="text-align:center; max-width:700px; margin:0 auto;">
        <h2 style="font-size:2rem; color:var(--text-main); margin-bottom:0.5rem;">Select a Tool</h2>
        <p style="color:var(--text-sub);">Advanced digital tools for archaeological conservation and surface analysis.</p>
    </div>
    
    <div class="landing-grid">
        <a href="/polygon_cutter" style="background:var(--bg-panel); border-radius:var(--radius); padding:2rem; text-align:center; border:1px solid var(--border); text-decoration:none; color:inherit; display:block;">
            <div style="font-size:3rem; margin-bottom:1rem;">✂️</div><h2>Polygon Cutter</h2><p style="color:var(--text-sub)">Mask artifacts & remove backgrounds.</p>
        </a>
        <a href="/color_analysis" style="background:var(--bg-panel); border-radius:var(--radius); padding:2rem; text-align:center; border:1px solid var(--border); text-decoration:none; color:inherit; display:block;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎨</div><h2>Color Analysis</h2><p style="color:var(--text-sub)">Automated clustering (FLACA).</p>
        </a>
        
    </div>
    """ # MODIFIED: Removed Stain Detection card
    return page_layout(main, sidebar=None, show_nav=True)

# ========== POLYGON CUTTER ROUTES ========== #

@app.get("/polygon_cutter", response_class=HTMLResponse)
async def polygon_cutter_page():
    colors = [
        ("255,255,255","#FFF"), ("0,0,0","#000"), ("128,128,128","#808080"), ("255,0,0","#F00"), ("0,255,0","#0F0"), ("0,0,255","#00F"),
        ("255,255,0","#FF0"), ("0,255,255","#0FF"), ("255,0,255","#F0F"), ("255,165,0","#FFA500"), ("128,0,128","#800080"), ("0,128,0","#008000"),
        ("255,192,203","#FFC0CB"), ("0,128,128","#008080"), ("165,42,42","#A52A2A"), ("245,245,220","#F5F5DC"), ("128,0,0","#800000"), ("0,0,128","#000080"),
        ("255,215,0","#FFD700"), ("192,192,192","#C0C0C0"), ("75,0,130","#4B0082"), ("250,128,114","#FA8072"), ("64,224,208","#40E0D0"), ("107,142,35","#6B8E23")
    ]
    swatches = "".join([f'<div class="swatch {"active" if i==0 else ""}" data-color="{c[0]}" style="background:{c[1]}"></div>' for i, c in enumerate(colors)])
    
    sidebar = f"""
    <h3>Control Panel</h3>
    <label>1. Upload Image</label><input id="file" type="file" accept="image/*">
    <label>2. Background Fill</label><div class="swatch-container">{swatches}</div>
    <label>3. Settings</label><div style="background:var(--bg-element); padding:8px; margin-top:5px;"><input type="checkbox" id="cropCheckbox" style="margin:0;"> <span style="font-size:0.85rem;">Crop to Selection</span></div>
    <hr>
    <button id="analyze_color" class="btn">🎨 Analyze Colors</button>
    <button id="export_alpha" class="btn btn-secondary">⬇️ Transparent PNG</button>
    <button id="export_flat" class="btn btn-secondary">⬇️ Flat PNG</button>
    <div class="info-box" style="margin-top:2rem;"><strong>Hotkeys:</strong> C=Complete, Z=Undo, Wheel=Zoom</div>
    """
    main = """
    <div class="toolbar">
        <button id="complete" class="btn" style="width:auto">Complete (C)</button>
        <button id="newpoly" class="btn btn-secondary" style="width:auto">New (N)</button>
        <button id="undo" class="btn btn-secondary" style="width:auto">Undo (Z)</button>
        <button id="clearall" class="btn btn-danger" style="margin-left:auto; width:auto;">Clear</button>
    </div>
    <div id="canvas-container"><canvas id="canvas"></canvas></div>
    <div style="margin-top:5px; font-size:0.8rem; color:var(--text-sub); display:flex; justify-content:space-between;">
        <span id="statusText">Load image to start</span><span id="zoomText">Zoom: 100%</span>
    </div>
    <script>
    let container=document.getElementById('canvas-container'), canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d');
    let img=new Image(), polygons=[], current=[], scale=1, originX=0, originY=0, isPanning=false, startPanX=0, startPanY=0, selectedBg="255,255,255", imgBytes=null;
    document.querySelectorAll('.swatch').forEach(s=>{s.onclick=()=>{document.querySelectorAll('.swatch').forEach(x=>x.classList.remove('active'));s.classList.add('active');selectedBg=s.dataset.color;}});
    function resizeCanvas(){canvas.width=container.clientWidth;canvas.height=container.clientHeight;draw();}
    window.onresize=resizeCanvas; resizeCanvas();
    document.getElementById('file').onchange=async(ev)=>{const f=ev.target.files[0];if(!f)return;const r=new FileReader();r.onload=(e)=>{img.src=e.target.result;img.onload=()=>{scale=Math.min((canvas.width-40)/img.width,(canvas.height-40)/img.height);originX=(canvas.width-img.width*scale)/2;originY=(canvas.height-img.height*scale)/2;draw();document.getElementById('statusText').innerText="Active";}};imgBytes=await f.arrayBuffer();r.readAsDataURL(f);};
    function toWorld(sx,sy){return{x:(sx-originX)/scale,y:(sy-originY)/scale};}
    container.onwheel=(e)=>{e.preventDefault();const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,ws=toWorld(mx,my),zf=e.deltaY<0?1.1:0.9;scale*=zf;originX=mx-ws.x*scale;originY=my-ws.y*scale;draw();}
    container.onmousedown=(e)=>{if(e.button===1||e.button===2){isPanning=true;startPanX=e.clientX-originX;startPanY=e.clientY-originY;container.style.cursor='grabbing';e.preventDefault();}};
    window.onmouseup=()=>{isPanning=false;container.style.cursor='crosshair';};
    container.onmousemove=(e)=>{if(isPanning){originX=e.clientX-startPanX;originY=e.clientY-startPanY;draw();}};
    container.oncontextmenu=e=>e.preventDefault();
    container.onclick=(ev)=>{if(isPanning||!img.src)return;const r=canvas.getBoundingClientRect(),pt=toWorld(ev.clientX-r.left,ev.clientY-r.top);current.push([pt.x,pt.y]);draw();};
    document.getElementById('complete').onclick=()=>{if(current.length>=3){polygons.push(current.slice());current=[];draw();}else alert('3+ points needed');};
    document.getElementById('newpoly').onclick=()=>{current=[];draw();};
    document.getElementById('undo').onclick=()=>{current.pop();draw();};
    document.getElementById('clearall').onclick=()=>{polygons=[];current=[];draw();};
    document.onkeydown=(e)=>{if(e.key==='c'||e.key==='Enter')document.getElementById('complete').click();if(e.key==='z')document.getElementById('undo').click();};
    function draw(){
        ctx.clearRect(0,0,canvas.width,canvas.height);document.getElementById('zoomText').innerText=`Zoom: ${(scale*100).toFixed(0)}%`;
        if(!img.src){ctx.fillStyle='#555';ctx.textAlign='center';ctx.fillText("Upload Image",canvas.width/2,canvas.height/2);return;}
        ctx.save();ctx.translate(originX,originY);ctx.scale(scale,scale);ctx.drawImage(img,0,0);
        const lw=2/scale,rad=3/scale;
        for(let p of polygons){if(p.length<2)continue;ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);for(let i=1;i<p.length;i++)ctx.lineTo(p[i][0],p[i][1]);ctx.closePath();ctx.fillStyle='rgba(13,148,136,0.3)';ctx.fill();ctx.strokeStyle='#0d9488';ctx.lineWidth=lw;ctx.stroke();}
        if(current.length>0){ctx.beginPath();ctx.moveTo(current[0][0],current[0][1]);for(let i=1;i<current.length;i++)ctx.lineTo(current[i][0],current[i][1]);ctx.strokeStyle='#ea580c';ctx.lineWidth=lw;ctx.stroke();for(let p of current){ctx.beginPath();ctx.arc(p[0],p[1],rad,0,Math.PI*2);ctx.fillStyle='#ea580c';ctx.fill();}}
        ctx.restore();
    }
    async function processImage(mode){
        if(!imgBytes){alert('Upload image');return;}
        const crop=document.getElementById('cropCheckbox').checked;
        const form=new FormData(); form.append('image',new Blob([imgBytes]),'img.png');
        form.append('meta',JSON.stringify({polygons:polygons.concat(current.length>=3?[current]:[]),background:selectedBg,alpha:mode==='alpha',crop:crop}));
        const btn=document.getElementById(mode==='analyze'?'analyze_color':'export_'+mode); btn.innerText="Working..."; btn.disabled=true;
        try{
            if(mode==='analyze'){
                const r=await fetch('/api/process_cut_and_store',{method:'POST',body:form});
                if(r.ok) window.location.href=`/color_analysis?preprocessed=${(await r.json()).filename}`;
            }else{
                const r=await fetch('/api/process_cut_download',{method:'POST',body:form});
                if(!r.ok) throw new Error('Fail');
                const u=URL.createObjectURL(await r.blob()); const a=document.createElement('a'); a.href=u; a.download=`cut_${mode}.png`; a.click();
            }
        }catch(e){alert(e);}finally{btn.innerText=mode==='analyze'?'🎨 Analyze Colors':(mode==='alpha'?'⬇️ Transparent PNG':'⬇️ Flat PNG'); btn.disabled=false;}
    }
    document.getElementById('export_alpha').onclick=()=>processImage('alpha');
    document.getElementById('export_flat').onclick=()=>processImage('flat');
    document.getElementById('analyze_color').onclick=()=>processImage('analyze');
    </script>
    """
    return page_layout(main, sidebar)


@app.post("/api/process_cut_download")
async def process_cut_download(image: UploadFile = File(...), meta: str = Form(...)):
    m = json.loads(meta); alpha = m.get("alpha", True); crop = m.get("crop", False)
    try: r,g,b = [int(x) for x in m.get("background","255,255,255").split(",")]; bg=(b,g,r)
    except: bg=(255,255,255)
    b_out, mime = cut_polygons_from_image_bytes(await image.read(), m.get("polygons",[]), background=bg, export_alpha=alpha, crop_to_poly=crop)
    return StreamingResponse(io.BytesIO(b_out), media_type=mime)

@app.post("/api/process_cut_and_store")
async def process_cut_and_store(image: UploadFile = File(...), meta: str = Form(...)):
    m = json.loads(meta); crop = m.get("crop", False)
    try: r,g,b = [int(x) for x in m.get("background","255,255,255").split(",")]; bg=(b,g,r)
    except: bg=(255,255,255)
    b_out, _ = cut_polygons_from_image_bytes(await image.read(), m.get("polygons",[]), background=bg, export_alpha=True, crop_to_poly=crop)
    fname = f"cut_{uuid.uuid4().hex}.png"; 
    with open(os.path.join(TEMP_DIR, fname), "wb") as f: f.write(b_out)
    return JSONResponse({"filename": fname})


# ========== COLOR ANALYSIS ROUTES ========== #

@app.get("/color_analysis", response_class=HTMLResponse)
async def color_analysis_page(preprocessed: str = None):
    return page_layout("""<div class="info-box"><h4>🎨 FLACA</h4><p>Upload image to start.</p></div>""", sidebar=color_form(color_defaults, preprocessed_file=preprocessed))

def color_form(p, preprocessed_file=None, reuse_path=None):
    if preprocessed_file: inp = f'<div class="info-box">✅ Ready: {preprocessed_file}<input type="hidden" name="preprocessed_file" value="{preprocessed_file}"><a href="/color_analysis" style="color:#f87171">Cancel</a></div>'
    elif reuse_path: inp = f'<div class="info-box">🔄 Re-using Image<input type="hidden" name="reuse_path" value="{reuse_path}"><div style="margin-top:5px; border-top:1px solid #444"><label>Or New:</label><input type="file" name="ref_image"></div></div>'
    else: inp = '<label>Upload Image</label><input type="file" name="ref_image" required>'
    return f"""
    <h3>Settings</h3>
    <form action="/color_analyze" enctype="multipart/form-data" method="post">
        {inp}
        <label>K (Colors)</label><input type="number" name="k" value="{p['k']}" min="1">
        <button type="button" onclick="toggleAdv()" class="btn btn-secondary" style="font-size:0.8rem">Advanced ⚙️</button>
        <div id="adv" class="hidden" style="margin-top:10px; padding:10px; border:1px solid var(--border);">
          <label>L_thresh</label><input type="number" step="0.1" name="L_thresh" value="{p['L_thresh']}">
          <label>C_thresh</label><input type="number" step="0.01" name="C_thresh" value="{p['C_thresh']}">
          <label>ab_step</label><input type="number" step="0.1" name="ab_step" value="{p['ab_step']}">
          <label>point_size</label><input type="number" name="point_size" value="{p['point_size']}">
          <label>shrink_img</label><input type="number" step="0.01" name="shrink_img" value="{p['shrink_img']}">
          <label>Show Plots</label><select name="show_plots_final"><option value="True">Yes</option><option value="False">No</option></select>
        </div>
        <input type="submit" value="Run Analysis">
        <button formaction="/color_restore_defaults" formmethod="post" class="btn btn-secondary">Reset</button>
    </form>
    <script>function toggleAdv(){{ document.getElementById('adv').classList.toggle('hidden'); }}</script>
    """

@app.post("/color_restore_defaults", response_class=HTMLResponse)
async def color_restore_defaults(): return page_layout("Defaults restored.", sidebar=color_form(color_defaults))

@app.post("/color_analyze", response_class=HTMLResponse)
async def color_analyze(
    ref_image: UploadFile = File(None),
    preprocessed_file: str = Form(None),
    reuse_path: str = Form(None),
    k: int = Form(5),
    L_thresh: float = Form(30.0),
    C_thresh: float = Form(0.1),
    ab_step: float = Form(1.0),
    point_size: int = Form(12),
    random_seed: int = Form(42),
    shrink_img: float = Form(1),
    show_plots_final: str = Form("True"),
):
    params = locals().copy()
    
    # FIX: Changed loop variable from 'k' to 'key' to avoid overwriting the 'k' parameter
    for key in ['ref_image', 'preprocessed_file', 'reuse_path']: 
        params.pop(key, None)
    
    src = None
    if ref_image and ref_image.filename:
        src = os.path.join(tempfile.mkdtemp(), ref_image.filename)
        with open(src, "wb") as f: shutil.copyfileobj(ref_image.file, f)
    elif preprocessed_file and os.path.exists(os.path.join(TEMP_DIR, preprocessed_file)): 
        src = os.path.join(TEMP_DIR, preprocessed_file)
    elif reuse_path and os.path.exists(reuse_path): 
        src = reuse_path
    
    if not src: return page_layout("<h3>❌ Error</h3><p>No valid image provided.</p>")
    
    try:
        bundle, img = k_color_analysis(
            src, k=k, L_thresh=L_thresh, C_thresh=C_thresh, 
            ab_step=ab_step, point_size=point_size, size_mode="sqrt", 
            pie_show_labels=True, show_input=True, show_plots_initial=False, 
            show_plots_final=True, # We always want initial plots for display
            random_seed=random_seed, shrink_img=shrink_img
        )
    except Exception as e: 
        return page_layout(f"<h3>❌ Analysis Error</h3><p>{e}</p>")

    urls = save_all_open_figures("analysis")
    bid = uuid.uuid4().hex
    
    active_bundles[bid] = {
        "bundle": bundle, "img": img, "params": params, 
        "source_path": src, "analysis": urls, "visuals": [], "type": "color"
    }
    
    # Redirect to the new dedicated results page
    return RedirectResponse(url=f"/color_results/{bid}", status_code=303)


def render_color_results_page(bundle_id: str, sess: dict):
    # 1. Get initial plots (first 3 from analysis)
    # The order is: 0: Input Image, 1: Pie Chart, 2: Palette Plot, 3+: Final Plots (if requested)
    initial_plots = sess["analysis"]
    
    primary_plots_data = [
        {"url": initial_plots[0], "label": "Input Image"},
        {"url": initial_plots[1], "label": "Color Palette"},
        {"url": initial_plots[2], "label": "Pie Chart"},
    ]
    
    # 2. Build Primary Plots HTML
    primary_plots_html = ""
    for item in primary_plots_data:
        primary_plots_html += f"""
        <div class="gallery-item-large" onclick="openLightbox('{item["url"]}')">
            <img src="{item["url"]}">
            <span>{item["label"]}</span>
        </div>
        """
        
    # 3. Build Inspection Gallery HTML (Reverse order to show newest first)
    gallery_html = ""
    if sess["visuals"]:
        for item in sess["visuals"][::-1]:
            gallery_html += f"""
            <div class="gallery-item" onclick="openLightbox('{item["url"]}')">
                <img src="{item["url"]}">
                <span>{item["label"]}</span>
            </div>
            """
    
    main = f"""
        <h3 style="margin-top:0">Analysis Results (k={sess['params']['k']})</h3>
        
        <h4>🖼️ Core Visualization</h4>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:15px;">
            {primary_plots_html}
        </div>

        <div class="panel" style="margin-top:20px; border-color:var(--primary);">
            <h4 style="margin-top:0;">🔍 Inspect Cluster</h4>
            <form action="/color_visualize" method="post" style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
                <input type="hidden" name="bundle_id" value="{bundle_id}">
                <input name="cluster" placeholder="Cluster ID (e.g. 0)" required style="width:120px; flex-grow:0;">
                <select name="visualization" style="width:150px; flex-grow:0;">
                    <option value="highlight">Highlight</option>
                    <option value="mask">Binary Mask</option>
                    <option value="overlay">Color Overlay</option>
                </select>
                <button type="submit" class="btn" style="margin:0; width:auto; flex-grow:1;">Visualize</button>
            </form>
        </div>
        
        {f'<h4 style="margin-top:20px;">🗂️ Cluster Inspection Gallery ({len(sess["visuals"])} plots)</h4><div class="gallery-grid">' + gallery_html + '</div>' if gallery_html else '<div class="info-box" style="margin-top:20px;">Use the form above to generate and view cluster-specific masks here.</div>'}
        
        <form action="/color_restart" method="post" style="margin-top:20px;"><input type="hidden" name="bundle_id" value="{bundle_id}"><button class="btn btn-secondary">🔄 Restart with New Settings</button></form>
    """
    return page_layout(main, sidebar=color_form(sess['params'], reuse_path=sess['source_path']))


@app.get("/color_results/{bundle_id}", response_class=HTMLResponse)
async def color_results_page(bundle_id: str):
    sess = active_bundles.get(bundle_id)
    if not sess: return page_layout("<h3>Session Expired</h3><p>The analysis session was not found or has expired.</p>")
    return render_color_results_page(bundle_id, sess)


@app.post("/color_visualize", response_class=HTMLResponse)
async def color_visualize(bundle_id:str=Form(...), cluster:str=Form(...), visualization:str=Form("highlight")):
    sess = active_bundles.get(bundle_id)
    if not sess: return page_layout("Session Expired")
    
    visualize_color_cluster(sess["img"], cluster=cluster, bundle=sess["bundle"], visualization=visualization)
    new_urls = save_all_open_figures(f"vis_{cluster}")
    
    # Add new visualization plots to history
    for u in new_urls:
        sess["visuals"].append({"url": u, "label": f"ID {cluster} ({visualization})"})
    
    # Redirect back to the unified results page to display the updated gallery
    return RedirectResponse(url=f"/color_results/{bundle_id}", status_code=303)

@app.post("/color_restart", response_class=HTMLResponse)
async def color_restart(bundle_id: str = Form(...)):
    sess = active_bundles.get(bundle_id)
    if sess:
        # Clear visuals but keep source path and parameters
        sess['visuals'] = [] 
        return page_layout("<h3>Restarting...</h3>", sidebar=color_form(sess['params'], reuse_path=sess.get('source_path')))
    return page_layout("<h3>Session Expired</h3>")


# ========== STAIN DETECTION ROUTES ========== # # REMOVED: All Stain Detection Code

# @app.get("/stain_detection", response_class=HTMLResponse)
# async def stain_detection_page():
#     # ... UI for stain detection removed
#     pass

# @app.post("/stain_analyze", response_class=HTMLResponse)
# async def stain_analyze(image_before: UploadFile = File(...), image_after: UploadFile = File(...), dark_thresh: int = Form(120), sat_thresh: int = Form(60), diff_thresh: int = Form(25)):
#     # ... logic for stain detection removed
#     pass


if __name__ == "__main__":
    print("🏺 Starting Archaeological Image Analysis Suite...")
    uvicorn.run(app, host="0.0.0.0", port=8000)