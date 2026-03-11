from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
import tempfile
import shutil
import os
import uuid
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- CRITICAL FIX FOR COMPATIBILITY ---
plt.show = lambda: None 

from config import TEMP_DIR, STATIC_DIR, active_bundles, color_defaults
from ui_templates import page_layout
from ColorClassifier_manual import k_color_analysis, visualize_color_cluster

router = APIRouter()

def save_all_open_figures(prefix="plot"):
    urls = []
    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        path = os.path.join(STATIC_DIR, f"{prefix}_{i}_{uuid.uuid4().hex}.png")
        fig.savefig(path, bbox_inches="tight", dpi=100)
        urls.append(f"/static_results/{os.path.basename(path)}")
    plt.close("all")
    return urls

def color_form(p, preprocessed_file=None, reuse_path=None):
    if preprocessed_file: inp = f'<div class="info-box">✅ Ready: {preprocessed_file}<input type="hidden" name="preprocessed_file" value="{preprocessed_file}"><a href="/color_analysis" style="color:#f87171">Cancel</a></div>'
    elif reuse_path: inp = f'<div class="info-box">🔄 Re-using Image<input type="hidden" name="reuse_path" value="{reuse_path}"><div style="margin-top:5px; border-top:1px solid #444"><label>Or New:</label><input type="file" name="ref_image"></div></div>'
    else: inp = '<label>Upload Image</label><input type="file" name="ref_image" required>'
    
    return f"""
    <h3>Settings</h3>
    <form action="/color_analyze" enctype="multipart/form-data" method="post">
        {inp}
        <label>K (Colors)</label><input type="number" name="k" value="{p['k']}" min="1">
        
        <label>Feature Weights (0.0 - 5.0)</label>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:5px; background:var(--bg-element); padding:5px; border-radius:4px;">
            <div><span style="font-size:0.8rem">L* (Light)</span><input type="number" step="0.1" name="weight_L" value="{p['weight_L']}"></div>
            <div><span style="font-size:0.8rem">C (Chroma)</span><input type="number" step="0.1" name="weight_C" value="{p['weight_C']}"></div>
            <div><span style="font-size:0.8rem">a* (Red/Grn)</span><input type="number" step="0.1" name="weight_a" value="{p['weight_a']}"></div>
            <div><span style="font-size:0.8rem">b* (Blu/Yel)</span><input type="number" step="0.1" name="weight_b" value="{p['weight_b']}"></div>
            <div><span style="font-size:0.8rem">Hue (Angle)</span><input type="number" step="0.1" name="weight_h" value="{p['weight_h']}"></div>
        </div>

        <button type="button" onclick="toggleAdv()" class="btn btn-secondary" style="font-size:0.8rem; margin-top:10px;">Advanced ⚙️</button>
        <div id="adv" class="hidden" style="margin-top:10px; padding:10px; border:1px solid var(--border);">
          <label>L_thresh</label><input type="number" step="0.1" name="L_thresh" value="{p['L_thresh']}">
          <label>C_thresh</label><input type="number" step="0.01" name="C_thresh" value="{p['C_thresh']}">
          <label>ab_step</label><input type="number" step="0.1" name="ab_step" value="{p['ab_step']}">
          <label>point_size</label><input type="number" name="point_size" value="{p['point_size']}">
          <label>shrink_img</label><input type="number" step="0.01" name="shrink_img" value="{p['shrink_img']}">
          <label>Show Plots</label><select name="show_plots_final"><option value="True">Yes</option><option value="False">No</option></select>
        </div>
        <input type="submit" value="Run Analysis" style="margin-top:15px;">
        <button formaction="/color_restore_defaults" formmethod="post" class="btn btn-secondary">Reset</button>
    </form>
    <script>function toggleAdv(){{ document.getElementById('adv').classList.toggle('hidden'); }}</script>
    """

@router.get("/color_analysis", response_class=HTMLResponse)
async def color_analysis_page(preprocessed: str = None):
    return page_layout("""<div class="info-box"><h4>🎨 FLACA</h4><p>Upload image to start.</p></div>""", sidebar=color_form(color_defaults, preprocessed_file=preprocessed))

@router.post("/color_restore_defaults", response_class=HTMLResponse)
async def color_restore_defaults(): 
    return page_layout("Defaults restored.", sidebar=color_form(color_defaults))

@router.post("/color_analyze", response_class=HTMLResponse)
async def color_analyze(
    ref_image: UploadFile = File(None), preprocessed_file: str = Form(None), reuse_path: str = Form(None),
    k: int = Form(5), L_thresh: float = Form(30.0), C_thresh: float = Form(0.1), ab_step: float = Form(1.0),
    point_size: int = Form(12), random_seed: int = Form(42), shrink_img: float = Form(1), show_plots_final: str = Form("True"),
    weight_L: float = Form(0.0), weight_a: float = Form(1.0), weight_b: float = Form(1.0), weight_C: float = Form(0.0), weight_h: float = Form(0.0)
):
    params = locals().copy()
    for key in ['ref_image', 'preprocessed_file', 'reuse_path']: params.pop(key, None)
    
    src = None
    if ref_image and ref_image.filename:
        src = os.path.join(TEMP_DIR, ref_image.filename)
        with open(src, "wb") as f: shutil.copyfileobj(ref_image.file, f)
    elif preprocessed_file and os.path.exists(os.path.join(TEMP_DIR, preprocessed_file)): src = os.path.join(TEMP_DIR, preprocessed_file)
    elif reuse_path and os.path.exists(reuse_path): src = reuse_path
    
    if not src: return page_layout("<h3>❌ Error</h3><p>No valid image provided.</p>")

    cluster_features = {'L': weight_L, 'a': weight_a, 'b': weight_b, 'C': weight_C, 'h': weight_h}

    try:
        bundle, img = k_color_analysis(
            ref_image=src, k=k, L_thresh=L_thresh, C_thresh=C_thresh, ab_step=ab_step, point_size=point_size, 
            show_input=False, show_plots_initial=False, show_plots_final=True,
            random_seed=random_seed, shrink_img=shrink_img, cluster_features=cluster_features
        )
    except Exception as e: 
        import traceback; traceback.print_exc()
        return page_layout(f"<h3>❌ Analysis Error</h3><p>{e}</p>")

    urls = save_all_open_figures("analysis")
    
    # --- PRE-GENERATE MASKS FOR CLICKABLE BUCKETS ---
    cluster_mask_urls = {}
    for cluster_id, cluster_data in bundle.clusters.items():
        # Generate the highlight mask for this specific cluster
        visualize_color_cluster(img, cluster=cluster_data['label'], bundle=bundle, visualization='highlight')
        # Grab the URL of the newly generated plot
        mask_urls = save_all_open_figures(f"mask_{cluster_id}_{uuid.uuid4().hex}")
        if mask_urls: cluster_mask_urls[cluster_id] = mask_urls[-1]

    bid = uuid.uuid4().hex
    active_bundles[bid] = {"bundle": bundle, "img": img, "params": params, "source_path": src, "analysis": urls, "masks": cluster_mask_urls, "visuals": []}
    
    return RedirectResponse(url=f"/color_results/{bid}", status_code=303)


def render_color_results_page(bundle_id: str, sess: dict):
    is_3d_source = False
    area_m2 = None
    
    # Check if this image came from our 3D Mesh pipeline (looks for the JSON sidecar)
    if os.path.exists(sess["source_path"] + ".json"):
        is_3d_source = True
        with open(sess["source_path"] + ".json", "r") as f:
            meta = json.load(f)
            area_m2 = meta.get("area_m2")
            sess["obj_path"] = meta.get("obj_path")

    # The 3D Export header (only shows if area_m2 is found)
    export_3d_html = ""
    if is_3d_source:
        export_3d_html = f"""
        <div class="panel" style="border-color:var(--accent); background:rgba(234, 88, 12, 0.1); margin-bottom: 20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h4 style="margin:0; color:var(--accent);">☁️ 3D Mesh Workflow Detected</h4>
                    <p style="font-size:0.9rem; margin:5px 0 0 0; color:var(--text-sub);">
                        Total Calculated Surface Area: <strong>{area_m2:.2f} m²</strong>
                    </p>
                </div>
                <form action="/api/export_mesh_result" method="post" style="margin:0;">
                    <input type="hidden" name="bundle_id" value="{bundle_id}">
                    <button class="btn" style="background-color:var(--accent); width:auto; margin:0;">
                        📐 Wrap Texture & View 3D
                    </button>
                </form>
            </div>
        </div>
        """

    # --- THIS IS THE CLICKABLE BUCKET DASHBOARD ---
    dashboard_html = """
    <style>
        .bucket-btn { display: flex; align-items: center; justify-content: space-between; padding: 12px; background: var(--bg-element); border: 2px solid transparent; border-radius: 6px; cursor: pointer; margin-bottom: 8px; transition: 0.2s; color: var(--text-main); text-align: left; }
        .bucket-btn:hover { border-color: var(--border); }
        .bucket-btn.active { border-color: var(--primary); background: rgba(13, 148, 136, 0.1); }
        .bucket-color { width: 20px; height: 20px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.2); flex-shrink: 0; }
    </style>
    <script>
        function showMask(url, btnId) {
            document.getElementById('interactive-mask').src = url;
            document.querySelectorAll('.bucket-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(btnId).classList.add('active');
        }
    </script>
    <div style="display: grid; grid-template-columns: 300px 1fr; gap: 20px; margin-top: 20px;">
        <div id="bucket-list">
            <h4 style="margin-top:0">Color Buckets</h4>
    """
    
    first_url = None
    # Generate a button for each cluster
    for k, cluster_data in sess["bundle"].clusters.items():
        pct = cluster_data['percent']
        # Dynamically calculate the square meters if we have the 3D area
        sqm_text = f"<br><span style='font-size:0.8rem; color:var(--text-sub)'>~ {(pct / 100.0) * area_m2:.4f} m²</span>" if area_m2 else ""
        color_hex = '#%02x%02x%02x' % tuple(cluster_data['rgb'])
        mask_url = sess["masks"].get(k, "")
        
        if not first_url: first_url = mask_url
        active_class = "active" if k == 0 else ""
        
        dashboard_html += f"""
        <button id="btn-cluster-{k}" class="bucket-btn {active_class}" onclick="showMask('{mask_url}', 'btn-cluster-{k}')">
            <div style="display:flex; align-items:center; gap:10px;">
                <div class="bucket-color" style="background-color: {color_hex};"></div>
                <div><strong>{cluster_data['label']}</strong>: {pct:.1f}% {sqm_text}</div>
            </div>
        </button>
        """
        
    dashboard_html += f"""
        </div>
        <div style="background: #000; border-radius: 8px; border: 1px solid var(--border); overflow: hidden; display: flex; align-items: center; justify-content: center; min-height: 400px;">
            <img id="interactive-mask" src="{first_url}" style="max-width: 100%; max-height: 60vh; cursor: pointer;" onclick="openLightbox(this.src)">
        </div>
    </div>
    """

    # Primary analysis plots (pie chart, panels) from the original matplotlib output
    initial_plots = sess["analysis"]
    primary_plots_html = ""
    for i, plot in enumerate(initial_plots):
        primary_plots_html += f'<div class="gallery-item-large" onclick="openLightbox(\'{plot}\')"><img src="{plot}"></div>'

    main = f"""
        <h3 style="margin-top:0">Analysis Dashboard</h3>
        {export_3d_html}
        {dashboard_html}
        <hr>
        <h4>Core Distributions</h4>
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:15px;">{primary_plots_html}</div>
        
        <form action="/color_restart" method="post" style="margin-top:20px;"><input type="hidden" name="bundle_id" value="{bundle_id}"><button class="btn btn-secondary">🔄 Restart with New Settings</button></form>
    """
    
    return page_layout(main, sidebar=color_form(sess['params'], reuse_path=sess['source_path']))

@router.get("/color_results/{bundle_id}", response_class=HTMLResponse)
async def color_results_page(bundle_id: str):
    sess = active_bundles.get(bundle_id)
    if not sess: return page_layout("<h3>Session Expired</h3><p>The analysis session was not found or has expired.</p>")
    return render_color_results_page(bundle_id, sess)

@router.post("/color_visualize", response_class=HTMLResponse)
async def color_visualize(bundle_id:str=Form(...), cluster:str=Form(...), visualization:str=Form("highlight")):
    sess = active_bundles.get(bundle_id)
    if not sess: return page_layout("Session Expired")
    
    try:
        visualize_color_cluster(sess["img"], cluster=cluster, bundle=sess["bundle"], visualization=visualization)
        new_urls = save_all_open_figures(f"vis_{cluster}")
    except Exception as e:
        return page_layout(f"<h3>Visual Error</h3><p>{e}</p>")
    
    for u in new_urls:
        sess["visuals"].append({"url": u, "label": f"{cluster} ({visualization})"})
    
    return RedirectResponse(url=f"/color_results/{bundle_id}", status_code=303)

@router.post("/color_restart", response_class=HTMLResponse)
async def color_restart(bundle_id: str = Form(...)):
    sess = active_bundles.get(bundle_id)
    if sess:
        sess['visuals'] = [] 
        return page_layout("<h3>Restarting...</h3>", sidebar=color_form(sess['params'], reuse_path=sess.get('source_path')))
    return page_layout("<h3>Session Expired</h3>")