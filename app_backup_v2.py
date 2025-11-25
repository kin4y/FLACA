from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import tempfile
import shutil
import os
import uuid
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ColorClassifier_manual import k_color_analysis, visualize_color_cluster
from stain_detection import detect_stains

app = FastAPI(title="Archaeological Image Analysis Suite")

STATIC_DIR = "static_results"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static_results", StaticFiles(directory=STATIC_DIR), name="static_results")

active_bundles = {}

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


color_defaults = {
    "k": 5, "L_thresh": 30.0, "C_thresh": 0.1, "ab_step": 1.0, "point_size": 12,
    "size_mode": "sqrt", "top_n_chroma": "", "top_n_achro": "",
    "pie_show_labels": "True", "show_input": "True", "show_plots_initial": "False",
    "show_plots_final": "True", "random_seed": 42, "shrink_img": 0.1,
}

stain_defaults = {
    "dark_thresh": 120,
    "sat_thresh": 60,
    "diff_thresh": 25,
}


def maya_theme_style():
    return """
    <style>
        body {
            font-family: 'Poppins', 'Segoe UI', sans-serif;
            margin: 0;
            background: linear-gradient(180deg,#fdf7ec 0%, #fdf3e6 100%);
            color: #2c2b2b;
        }

        header {
            position: relative;
            background: linear-gradient(90deg,#c1440e,#d98e32,#14957a);
            color: #fff;
            text-align: center;
            padding: 25px 15px;
            border-bottom: 6px solid #a1250d;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }

        header h1 {
            margin: 0;
            font-size: 2em;
            letter-spacing: 0.05em;
            text-shadow: 1px 2px 3px rgba(0,0,0,0.4);
        }

        .dog {
            position: absolute;
            top: 5px;
            right: 25px;
            width: 85px;
            height: 85px;
            background: url('https://media.tenor.com/8ZOBp0fZcMsAAAAi/dog-wag.gif') center/contain no-repeat;
        }

        nav {
            background: #f9edda;
            padding: 12px;
            text-align: center;
            border-bottom: 2px solid #c1440e;
        }

        nav a {
            color: #14957a;
            text-decoration: none;
            padding: 8px 16px;
            margin: 0 8px;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.3s;
            display: inline-block;
        }

        nav a:hover {
            background: #14957a;
            color: white;
        }

        main {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            padding: 25px;
        }

        .full-width {
            width: 100%;
        }

        .panel {
            background-color: #fff6e8;
            border-radius: 15px;
            border: 2px solid #c1440e;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .analysis {
            width: 63%;
            background: #fff9f1;
        }

        .sidebar {
            width: 32%;
            background: #f9f1e7;
            overflow-y: auto;
            max-height: 85vh;
        }

        .landing-options {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 40px;
        }

        .option-card {
            background: white;
            border: 3px solid #14957a;
            border-radius: 15px;
            padding: 30px;
            width: 300px;
            text-align: center;
            cursor: pointer;
            transition: 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .option-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            border-color: #c1440e;
        }

        .option-card h2 {
            color: #c1440e;
            margin-bottom: 15px;
        }

        .option-card p {
            color: #555;
            line-height: 1.6;
        }

        img {
            border-radius: 8px;
            border: 2px solid #14957a;
            margin: 8px;
            max-width: 100%;
        }

        label {
            display: inline-block;
            margin-top: 6px;
            color: #3e3d3d;
            font-weight: 500;
        }

        input[type=submit], button {
            background: linear-gradient(45deg,#14957a,#d98e32);
            color: white;
            border: none;
            padding: 8px 14px;
            border-radius: 8px;
            margin-top: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: 0.3s;
        }

        input[type=submit]:hover, button:hover {
            opacity: 0.9;
            transform: scale(1.04);
        }

        select, input, textarea {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 5px;
            margin-bottom: 10px;
            width: 95%;
        }

        .advanced {
            margin-top: 10px;
            background: #e1f2e6;
            border: 1px solid #14957a;
            padding: 10px;
            border-radius: 10px;
        }

        .info-box {
            background: #fff9e6;
            border-left: 4px solid #d98e32;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }

        .instruction-box {
            background: #e8f4f8;
            border: 2px solid #14957a;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
        }

        .instruction-box h4 {
            margin-top: 0;
            color: #14957a;
        }

        .instruction-box ol {
            margin: 10px 0;
            padding-left: 20px;
        }

        .instruction-box li {
            margin: 8px 0;
            line-height: 1.5;
        }

        footer {
            text-align: center;
            color: #333;
            font-weight: 500;
            padding: 15px;
            background: #f9edda;
            border-top: 3px solid #c1440e;
        }

        h3,h4 { color: #a1250d; }
    </style>

    <script>
        function toggleAdvanced(){
          var adv=document.getElementById('adv');
          adv.style.display=(adv.style.display==='none' ? 'block' : 'none');
        }
    </script>
    """


def color_form_html(params):
    p = params
    return f"""
    <form action="/color_analyze" enctype="multipart/form-data" method="post">
        <label>Upload Image</label><br>
        <input type="file" name="ref_image" accept="image/*" required><br><br>

        <label>Number of Colors (k):</label><br>
        <input type="number" name="k" value="{p['k']}" min="1" required><br><br>

        <button type="button" onclick="toggleAdvanced()">Show/Hide Advanced ⚙️</button>
        <div id="adv" class="advanced" style="display:none;">
          <label>L_thresh:</label><input type="number" step="0.1" name="L_thresh" value="{p['L_thresh']}"><br>
          <label>C_thresh:</label><input type="number" step="0.01" name="C_thresh" value="{p['C_thresh']}"><br>
          <label>ab_step:</label><input type="number" step="0.1" name="ab_step" value="{p['ab_step']}"><br>
          <label>point_size:</label><input type="number" name="point_size" value="{p['point_size']}"><br>
          <label>size_mode:</label>
          <select name="size_mode">
             <option {"selected" if p["size_mode"]=="sqrt" else ""}>sqrt</option>
             <option {"selected" if p["size_mode"]=="linear" else ""}>linear</option>
          </select><br>
          <label>top_n_chroma:</label><input type="text" name="top_n_chroma" value="{p['top_n_chroma']}"><br>
          <label>top_n_achro:</label><input type="text" name="top_n_achro" value="{p['top_n_achro']}"><br>
          <label>pie_show_labels:</label>
          <select name="pie_show_labels"><option>True</option><option {"selected" if p["pie_show_labels"]=="False" else ""}>False</option></select><br>
          <label>show_input:</label>
          <select name="show_input"><option>True</option><option {"selected" if p["show_input"]=="False" else ""}>False</option></select><br>
          <label>show_plots_initial:</label>
          <select name="show_plots_initial"><option>False</option><option {"selected" if p["show_plots_initial"]=="True" else ""}>True</option></select><br>
          <label>show_plots_final:</label>
          <select name="show_plots_final"><option>True</option><option {"selected" if p["show_plots_final"]=="False" else ""}>False</option></select><br>
          <label>random_seed:</label><input type="number" name="random_seed" value="{p['random_seed']}"><br>
          <label>shrink_img:</label><input type="number" step="0.01" name="shrink_img" value="{p['shrink_img']}"><br>
        </div>
        <br>
        <input type="submit" value="Run Analysis">
        <button formaction="/color_restore_defaults" formmethod="post">Restore Defaults</button>
    </form>
    """


def stain_form_html(params):
    p = params
    return f"""
    <form action="/stain_analyze" enctype="multipart/form-data" method="post">
        <div class="instruction-box">
            <h4>📍 ROI Point Selection Instructions</h4>
            <p><strong>When the image windows appear, you will select 4 points to define the Region of Interest (ROI):</strong></p>
            <ol>
                <li><strong>Top-Left corner</strong> – Click the upper-left point of your analysis area</li>
                <li><strong>Top-Right corner</strong> – Click the upper-right point</li>
                <li><strong>Bottom-Right corner</strong> – Click the lower-right point</li>
                <li><strong>Bottom-Left corner</strong> – Click the lower-left point</li>
            </ol>
            <p>⚠️ <strong>Important:</strong> Follow this clockwise order starting from top-left. The points define a quadrilateral that will be perspective-corrected for analysis.</p>
            <p>🖱️ <strong>Tip:</strong> Click carefully on each corner. Red circles will appear as you click to confirm your selection.</p>
        </div>

        <label><strong>Upload "Before" Image (Reference)</strong></label><br>
        <input type="file" name="image_before" accept="image/*" required><br><br>

        <label><strong>Upload "After" Image (Target)</strong></label><br>
        <input type="file" name="image_after" accept="image/*" required><br><br>

        <button type="button" onclick="toggleAdvanced()">Show/Hide Advanced Parameters ⚙️</button>
        <div id="adv" class="advanced" style="display:none;">
          <label>Dark Threshold (0-255):</label>
          <input type="number" name="dark_thresh" value="{p['dark_thresh']}" min="0" max="255"><br>
          <small>Lower values detect lighter stains, higher values only very dark stains. Default: 120</small><br><br>

          <label>Saturation Threshold (0-255):</label>
          <input type="number" name="sat_thresh" value="{p['sat_thresh']}" min="0" max="255"><br>
          <small>Filters colored spots. Lower values = more strict (only grayscale). Default: 60</small><br><br>

          <label>Difference Threshold (0-255):</label>
          <input type="number" name="diff_thresh" value="{p['diff_thresh']}" min="0" max="255"><br>
          <small>Minimum brightness difference to count as stain. Higher = fewer detections. Default: 25</small><br><br>
        </div>
        <br>
        <input type="submit" value="Run Stain Detection">
        <button formaction="/stain_restore_defaults" formmethod="post">Restore Defaults</button>
    </form>
    """


def page_layout(main, sidebar=None, show_nav=True):
    nav_html = """
    <nav>
        <a href="/">🏠 Home</a>
        <a href="/color_analysis">🎨 Color Analysis</a>
        <a href="/stain_detection">🔍 Stain Detection</a>
    </nav>
    """ if show_nav else ""
    
    if sidebar:
        content = f"""
        <main>
            <div class="panel analysis">{main}</div>
            <div class="panel sidebar">{sidebar}</div>
        </main>
        """
    else:
        content = f"""
        <main>
            <div class="panel full-width">{main}</div>
        </main>
        """
    
    return f"""
    <html><head>{maya_theme_style()}</head>
    <body>
      <header>
        <h1>🏺 Archaeological Image Analysis Suite</h1>
        <div class="dog"></div>
      </header>
      {nav_html}
      {content}
      <footer>Inspired by Totonaco & Maya heritage · El Tajín, Veracruz · Crafted with 🧡</footer>
    </body></html>
    """


# ---------- Routes ---------- #

@app.get("/", response_class=HTMLResponse)
async def home():
    main = """
    <h2 style="text-align: center; color: #14957a;">Welcome to the Archaeological Image Analysis Suite</h2>
    <p style="text-align: center; max-width: 700px; margin: 20px auto; line-height: 1.8;">
        This tool was developed for archaeological research at <strong>El Tajín, Veracruz, Mexico</strong>, 
        focusing on the analysis and documentation of surface conditions on historical artifacts and structures.
        Choose an analysis method below:
    </p>
    
    <div class="landing-options">
        <a href="/color_analysis" style="text-decoration: none;">
            <div class="option-card">
                <h2>🎨 Color Analysis</h2>
                <p><strong>FLACA</strong> – Fast Lightweight Automated Color Analyzer</p>
                <p>Automatically cluster and analyze colors in archaeological photographs using k-means clustering in Lab color space.</p>
                <p style="margin-top: 15px; font-size: 0.9em; color: #14957a;"><strong>→ Start Color Analysis</strong></p>
            </div>
        </a>
        
        <a href="/stain_detection" style="text-decoration: none;">
            <div class="option-card">
                <h2>🔍 Stain Detection</h2>
                <p>Compare before/after images to detect and quantify surface stains and discolorations.</p>
                <p>Specially optimized for dark stains on archaeological surfaces.</p>
                <p style="margin-top: 15px; font-size: 0.9em; color: #14957a;"><strong>→ Start Stain Detection</strong></p>
            </div>
        </a>
    </div>
    
    <div class="info-box" style="max-width: 800px; margin: 40px auto;">
        <h4>ℹ️ About This Tool</h4>
        <p>This open-source prototypical software is designed for exploratory and educational purposes in archaeological conservation. 
        For formal scientific analysis or publication, please verify results with qualified conservation or imaging professionals.</p>
    </div>
    """
    return page_layout(main, sidebar=None, show_nav=False)


# ========== COLOR ANALYSIS ROUTES ========== #

@app.get("/color_analysis", response_class=HTMLResponse)
async def color_analysis_page():
    main = "<h3>🎨 FLACA – Color Analysis</h3>" + color_form_html(color_defaults)
    return page_layout(main, "<i>No visualizations yet. Upload an image to start.</i>")


@app.post("/color_restore_defaults", response_class=HTMLResponse)
async def color_restore_defaults():
    main = "<h3>🎨 FLACA – Color Analysis</h3>" + color_form_html(color_defaults)
    return page_layout(main, "<i>Defaults restored.</i>")


@app.post("/color_analyze", response_class=HTMLResponse)
async def color_analyze(
    ref_image: UploadFile = File(...),
    k: int = Form(5),
    L_thresh: float = Form(30.0),
    C_thresh: float = Form(0.1),
    ab_step: float = Form(1.0),
    point_size: int = Form(12),
    size_mode: str = Form("sqrt"),
    top_n_chroma: str = Form(""),
    top_n_achro: str = Form(""),
    pie_show_labels: str = Form("True"),
    show_input: str = Form("True"),
    show_plots_initial: str = Form("False"),
    show_plots_final: str = Form("True"),
    random_seed: int = Form(42),
    shrink_img: float = Form(0.1),
):
    params = locals().copy()
    params.pop("ref_image")

    top_n_chroma = None if not top_n_chroma.strip() else int(top_n_chroma)
    top_n_achro = None if not top_n_achro.strip() else int(top_n_achro)

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, ref_image.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(ref_image.file, f)

    bundle, img = k_color_analysis(
        tmp_path, k=k, L_thresh=L_thresh, C_thresh=C_thresh,
        ab_step=ab_step, point_size=point_size, size_mode=size_mode,
        top_n_chroma=top_n_chroma, top_n_achro=top_n_achro,
        pie_show_labels=(pie_show_labels.lower() == "true"),
        show_input=(show_input.lower() == "true"),
        show_plots_initial=(show_plots_initial.lower() == "true"),
        show_plots_final=(show_plots_final.lower() == "true"),
        random_seed=random_seed, shrink_img=shrink_img,
    )

    analysis_urls = save_all_open_figures(prefix="analysis")
    bundle_id = uuid.uuid4().hex
    active_bundles[bundle_id] = {
        "bundle": bundle, 
        "img": img, 
        "params": params, 
        "analysis": analysis_urls, 
        "visuals": [],
        "type": "color"
    }

    main = f"""
      <h3>✅ Color Analysis Complete</h3>
      <p><strong>Image:</strong> {ref_image.filename} | <strong>K =</strong> {k} colors</p>
      <div class="info-box">
        <strong>Results:</strong> The color clustering analysis has been completed. 
        Scroll down to see the visualizations, then optionally add specific cluster visualizations below.
      </div>
      {''.join(f'<img src="{u}" style="max-width: 95%;">' for u in analysis_urls)}
      <hr>
      <h4>🔍 Add Cluster Visualization</h4>
      <form action="/color_visualize" method="post">
         <input type="hidden" name="bundle_id" value="{bundle_id}">
         <label>Cluster Name/ID:</label><input name="cluster" required placeholder="e.g., 0, 1, red, blue">
         <label>Visualization Type:</label>
         <select name="visualization">
            <option>highlight</option>
            <option>mask</option>
            <option>overlay</option>
         </select>
         <input type="submit" value="Add Visualization">
      </form>
      <form action="/color_restart" method="post">
         <input type="hidden" name="bundle_id" value="{bundle_id}">
         <input type="submit" value="🔄 New Analysis (keep params)">
      </form>
    """
    return page_layout(main, "<i>No cluster visualizations yet. Use the form to add specific cluster views.</i>")


@app.post("/color_visualize", response_class=HTMLResponse)
async def color_visualize(bundle_id: str = Form(...), cluster: str = Form(...), visualization: str = Form("highlight")):
    sess = active_bundles.get(bundle_id)
    if not sess or sess.get("type") != "color": 
        return page_layout("<h3>❌ Session expired or invalid</h3>", None)

    visualize_color_cluster(sess["img"], cluster=cluster, bundle=sess["bundle"], visualization=visualization)
    vis_urls = save_all_open_figures(prefix=f"vis_{cluster}")
    sess["visuals"].extend(vis_urls)

    main = f"""
      <h3>✅ Color Analysis & Cluster Visualizations</h3>
      <div class="info-box">
        <strong>Main Analysis Results:</strong>
      </div>
      {''.join(f'<img src="{u}" style="max-width: 95%;">' for u in sess['analysis'])}
      <hr>
      <h4>🔍 Add More Cluster Visualizations</h4>
      <form action="/color_visualize" method="post">
         <input type="hidden" name="bundle_id" value="{bundle_id}">
         <label>Cluster Name/ID:</label><input name="cluster" required placeholder="e.g., 0, 1, red, blue">
         <label>Visualization Type:</label>
         <select name="visualization">
            <option>highlight</option>
            <option>mask</option>
            <option>overlay</option>
         </select>
         <input type="submit" value="Add Visualization">
      </form>
      <form action="/color_restart" method="post">
         <input type="hidden" name="bundle_id" value="{bundle_id}">
         <input type="submit" value="🔄 New Analysis (keep params)">
      </form>
    """
    sidebar = f"""
      <h3>🎨 Cluster Visualizations</h3>
      {''.join(f'<img src="{u}" style="max-width: 100%;">' for u in sess['visuals']) if sess['visuals'] else '<i>No visualizations yet.</i>'}
    """
    return page_layout(main, sidebar)


@app.post("/color_restart", response_class=HTMLResponse)
async def color_restart(bundle_id: str = Form(...)):
    sess = active_bundles.get(bundle_id)
    params = sess["params"] if sess and sess.get("type") == "color" else color_defaults
    main = "<h3>🔄 Restart Color Analysis</h3>" + color_form_html(params)
    return page_layout(main, "<i>Previous parameters loaded. Upload new image to analyze.</i>")


# ========== STAIN DETECTION ROUTES ========== #

@app.get("/stain_detection", response_class=HTMLResponse)
async def stain_detection_page():
    main = "<h3>🔍 Stain Detection Analysis</h3>" + stain_form_html(stain_defaults)
    sidebar = """
    <h4>📖 How It Works</h4>
    <p><strong>Step 1:</strong> Upload two images (before and after)</p>
    <p><strong>Step 2:</strong> Images will be automatically aligned using feature detection</p>
    <p><strong>Step 3:</strong> Interactive windows will appear - select 4 ROI points on each image</p>
    <p><strong>Step 4:</strong> The algorithm detects dark/gray stains by comparing brightness and saturation</p>
    <p><strong>Step 5:</strong> View results showing original images, detected stains, and evaluation mask</p>
    
    <h4 style="margin-top: 20px;">🎯 Best Practices</h4>
    <ul style="font-size: 0.9em; line-height: 1.6;">
        <li>Use similar lighting conditions</li>
        <li>Ensure images have overlapping areas</li>
        <li>Select ROI points carefully in clockwise order</li>
        <li>Adjust thresholds if detection is too sensitive or not sensitive enough</li>
    </ul>
    """
    return page_layout(main, sidebar)


@app.post("/stain_restore_defaults", response_class=HTMLResponse)
async def stain_restore_defaults():
    main = "<h3>🔍 Stain Detection Analysis</h3>" + stain_form_html(stain_defaults)
    return page_layout(main, "<i>Defaults restored.</i>")


@app.post("/stain_analyze", response_class=HTMLResponse)
async def stain_analyze(
    image_before: UploadFile = File(...),
    image_after: UploadFile = File(...),
    dark_thresh: int = Form(120),
    sat_thresh: int = Form(60),
    diff_thresh: int = Form(25),
):
    params = {
        "dark_thresh": dark_thresh,
        "sat_thresh": sat_thresh,
        "diff_thresh": diff_thresh,
    }

    # Save uploaded files temporarily
    tmp_dir = tempfile.mkdtemp()
    path_before = os.path.join(tmp_dir, image_before.filename)
    path_after = os.path.join(tmp_dir, image_after.filename)
    
    with open(path_before, "wb") as f:
        shutil.copyfileobj(image_before.file, f)
    
    with open(path_after, "wb") as f:
        shutil.copyfileobj(image_after.file, f)

    try:
        # Run stain detection (this will open OpenCV windows for ROI selection)
        ratio = detect_stains(
            pathA=path_before,
            pathB=path_after,
            dark_thresh=dark_thresh,
            sat_thresh=sat_thresh,
            diff_thresh=diff_thresh,
            plot=True
        )
        
        # Save all generated plots
        result_urls = save_all_open_figures(prefix="stain_result")
        
        bundle_id = uuid.uuid4().hex
        active_bundles[bundle_id] = {
            "params": params,
            "results": result_urls,
            "ratio": ratio,
            "type": "stain"
        }

        main = f"""
          <h3>✅ Stain Detection Complete</h3>
          <div class="info-box">
            <p><strong>Before Image:</strong> {image_before.filename}</p>
            <p><strong>After Image:</strong> {image_after.filename}</p>
            <p><strong>Stain Ratio:</strong> {ratio:.4f} ({ratio*100:.2f}% of ROI area)</p>
          </div>
          
          <h4>📊 Analysis Results</h4>
          <p>The images below show:</p>
          <ul>
            <li><strong>Original B:</strong> The "after" image in the analysis frame</li>
            <li><strong>Detected Black Stains:</strong> Stains highlighted in red overlay</li>
            <li><strong>Original A:</strong> The "before" (reference) image</li>
            <li><strong>Stain Mask:</strong> Binary mask showing ROI (white) and detected stains (red)</li>
          </ul>
          
          {''.join(f'<img src="{u}" style="max-width: 95%; margin: 10px 0;">' for u in result_urls)}
          
          <hr>
          <form action="/stain_restart" method="post">
             <input type="hidden" name="bundle_id" value="{bundle_id}">
             <input type="submit" value="🔄 New Analysis (keep params)">
          </form>
        """
        
        sidebar = f"""
          <h4>📈 Detection Statistics</h4>
          <p><strong>Stain Ratio:</strong> {ratio:.4f}</p>
          <p><strong>Percentage:</strong> {ratio*100:.2f}%</p>
          
          <h4>⚙️ Parameters Used</h4>
          <p><strong>Dark Threshold:</strong> {dark_thresh}</p>
          <p><strong>Saturation Threshold:</strong> {sat_thresh}</p>
          <p><strong>Difference Threshold:</strong> {diff_thresh}</p>
          
          <div class="info-box" style="margin-top: 20px;">
            <strong>💡 Interpretation:</strong><br>
            The stain ratio indicates what fraction of the Region of Interest (ROI) 
            has been identified as containing dark/gray stains compared to the reference image.
          </div>
        """
        
        return page_layout(main, sidebar)
        
    except Exception as e:
        main = f"""
          <h3>❌ Error During Stain Detection</h3>
          <div class="info-box" style="border-color: #c1440e;">
            <p><strong>Error:</strong> {str(e)}</p>
            <p>Please ensure:</p>
            <ul>
              <li>Both images are valid and can be loaded</li>
              <li>Images have sufficient overlapping features for alignment</li>
              <li>You selected all 4 ROI points correctly in the OpenCV windows</li>
            </ul>
          </div>
          <form action="/stain_detection" method="get">
             <input type="submit" value="← Back to Stain Detection">
          </form>
        """
        return page_layout(main, "<i>Analysis failed. See error details.</i>")
    finally:
        # Cleanup temporary files
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/stain_restart", response_class=HTMLResponse)
async def stain_restart(bundle_id: str = Form(...)):
    sess = active_bundles.get(bundle_id)
    params = sess["params"] if sess and sess.get("type") == "stain" else stain_defaults
    main = "<h3>🔄 Restart Stain Detection</h3>" + stain_form_html(params)
    return page_layout(main, "<i>Previous parameters loaded. Upload new images to analyze.</i>")


# ========== SERVER STARTUP ========== #

if __name__ == "__main__":
    print("🏺 Starting Archaeological Image Analysis Suite...")
    print("📍 Access the application at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)