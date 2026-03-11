from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from routers import mesh_3d
from config import STATIC_DIR, TEMP_DIR
from ui_templates import page_layout

# Import the routers
from routers import polygon, color

app = FastAPI(title="Archaeological Image Analysis Suite")

# Mount Static Directories
app.mount("/static_results", StaticFiles(directory=STATIC_DIR), name="static_results")
app.mount("/temp_uploads", StaticFiles(directory=TEMP_DIR), name="temp_uploads")

# Include Routers
app.include_router(polygon.router)
app.include_router(color.router)
app.include_router(mesh_3d.router)

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
    """
    return page_layout(main, sidebar=None, show_nav=True)

if __name__ == "__main__":
    print("🏺 Starting Archaeological Image Analysis Suite...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)