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

app = FastAPI(title="FLACA - Fast Lightweight Automated Color Analyzer")

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
        fig.savefig(path, bbox_inches="tight")
        urls.append(f"/static_results/{os.path.basename(path)}")
    plt.close("all")
    return urls


defaults = {
    "k": 5, "L_thresh": 30.0, "C_thresh": 0.1, "ab_step": 1.0, "point_size": 12,
    "size_mode": "sqrt", "top_n_chroma": "", "top_n_achro": "",
    "pie_show_labels": "True", "show_input": "True", "show_plots_initial": "False",
    "show_plots_final": "True", "random_seed": 42, "shrink_img": 0.1,
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

        main {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            padding: 25px;
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
        }

        img {
            border-radius: 8px;
            border: 2px solid #14957a;
            margin: 8px;
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


def form_html(params):
    p = params
    return f"""
    <form action="/analyze" enctype="multipart/form-data" method="post">
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
        <button formaction="/restore_defaults" formmethod="post">Restore Defaults</button>
    </form>
    """


def page_layout(main, sidebar):
    return f"""
    <html><head>{maya_theme_style()}</head>
    <body>
      <header>
        <h1>🐶 FLACA · Fast Lightweight Automated Color Analyzer</h1>
        <div class="dog"></div>
      </header>
      <main>
        <div class="panel analysis">{main}</div>
        <div class="panel sidebar">{sidebar}</div>
      </main>
      <footer>Color wisdom inspired by Totonaco & Maya palettes — crafted with 🧡 by FLACA</footer>
    </body></html>
    """


# ---------- Routes ---------- #

@app.get("/", response_class=HTMLResponse)
async def home():
    main = "<h3>Start a New Analysis</h3>" + form_html(defaults)
    return page_layout(main, "<i>No visualizations yet.</i>")


@app.post("/restore_defaults", response_class=HTMLResponse)
async def restore_defaults():
    main = "<h3>Start a New Analysis</h3>" + form_html(defaults)
    return page_layout(main, "<i>Defaults restored.</i>")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
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
    active_bundles[bundle_id] = {"bundle": bundle, "img": img, "params": params, "analysis": analysis_urls, "visuals": []}

    main = f"""
      <h3>Analysis Complete ✅</h3>
      <p>Image: {ref_image.filename} | K = {k}</p>
      {''.join(f'<img src="{u}" width="400">' for u in analysis_urls)}
      <hr><h4>Add Visualization</h4>
      <form action="/visualize" method="post">
         <input type="hidden" name="bundle_id" value="{bundle_id}">
         <label>Cluster:</label><input name="cluster" required>
         <label>Visualization:</label>
         <select name="visualization"><option>highlight</option><option>mask</option><option>overlay</option></select>
         <input type="submit" value="Add Visualization">
      </form>
      <form action="/restart" method="post">
         <input type="hidden" name="bundle_id" value="{bundle_id}">
         <input type="submit" value="Restart (keep params)">
      </form>
    """
    return page_layout(main, "<i>No visualizations yet.</i>")


@app.post("/visualize", response_class=HTMLResponse)
async def visualize(bundle_id: str = Form(...), cluster: str = Form(...), visualization: str = Form("highlight")):
    sess = active_bundles.get(bundle_id)
    if not sess: return "<h3>Session expired</h3>"

    visualize_color_cluster(sess["img"], cluster=cluster, bundle=sess["bundle"], visualization=visualization)
    vis_urls = save_all_open_figures(prefix=f"vis_{cluster}")
    sess["visuals"].extend(vis_urls)

    main = f"""
      <h3>Analysis & Visuals</h3>
      {''.join(f'<img src="{u}" width="400">' for u in sess['analysis'])}
      <hr><h4>Add More Visualization</h4>
      <form action="/visualize" method="post">
         <input type="hidden" name="bundle_id" value="{bundle_id}">
         <label>Cluster:</label><input name="cluster" required>
         <label>Visualization:</label>
         <select name="visualization"><option>highlight</option><option>mask</option><option>overlay</option></select>
         <input type="submit" value="Add Visualization">
      </form>
      <form action="/restart" method="post">
         <input type="hidden" name="bundle_id" value="{bundle_id}">
         <input type="submit" value="Restart (keep params)">
      </form>
    """
    sidebar = f"<h3>Visualizations</h3>{''.join(f'<img src=\"{u}\" width=\"220\">' for u in sess['visuals'])}"
    return page_layout(main, sidebar)


@app.post("/restart", response_class=HTMLResponse)
async def restart(bundle_id: str = Form(...)):
    sess = active_bundles.get(bundle_id)
    params = sess["params"] if sess else defaults
    main = "<h3>Restart Analysis</h3>" + form_html(params)
    return page_layout(main, "<i>Parameters loaded.</i>")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)