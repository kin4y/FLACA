from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import numpy as np
import cv2
import io
import json
import os
import uuid

from config import TEMP_DIR
from ui_templates import page_layout

router = APIRouter()

def cut_polygons_from_image_bytes(image_bytes: bytes, polygons, background=None, export_alpha=True, crop_to_poly=False):
    """
    Processes the image with polygon masking, background replacement, and optional cropping.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    
    h, w = img.shape[:2]
    
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
    
    if background is None:
        background = (255, 255, 255) 
    
    bg_img = np.zeros_like(img)
    bg_img[:] = background
    
    mask_bool = mask.astype(bool)
    combined_img = bg_img.copy()
    combined_img[mask_bool] = img[mask_bool]
    
    if export_alpha:
        b, g, r = cv2.split(combined_img)
        alpha = mask.copy()
        out = cv2.merge((b, g, r, alpha))
    else:
        out = combined_img
    
    is_success, buffer = cv2.imencode(".png", out)
    return buffer.tobytes(), "image/png"

@router.get("/polygon_cutter", response_class=HTMLResponse)
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

@router.post("/api/process_cut_download")
async def process_cut_download(image: UploadFile = File(...), meta: str = Form(...)):
    m = json.loads(meta); alpha = m.get("alpha", True); crop = m.get("crop", False)
    try: r,g,b = [int(x) for x in m.get("background","255,255,255").split(",")]; bg=(b,g,r)
    except: bg=(255,255,255)
    b_out, mime = cut_polygons_from_image_bytes(await image.read(), m.get("polygons",[]), background=bg, export_alpha=alpha, crop_to_poly=crop)
    return StreamingResponse(io.BytesIO(b_out), media_type=mime)

@router.post("/api/process_cut_and_store")
async def process_cut_and_store(image: UploadFile = File(...), meta: str = Form(...)):
    m = json.loads(meta); crop = m.get("crop", False)
    try: r,g,b = [int(x) for x in m.get("background","255,255,255").split(",")]; bg=(b,g,r)
    except: bg=(255,255,255)
    b_out, _ = cut_polygons_from_image_bytes(await image.read(), m.get("polygons",[]), background=bg, export_alpha=True, crop_to_poly=crop)
    fname = f"cut_{uuid.uuid4().hex}.png"
    with open(os.path.join(TEMP_DIR, fname), "wb") as f: f.write(b_out)
    return JSONResponse({"filename": fname})