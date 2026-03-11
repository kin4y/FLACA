from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
import os
import uuid
import json
import shutil
import numpy as np
import cv2
import trimesh
from PIL import Image

from config import TEMP_DIR, active_bundles
from ui_templates import page_layout

router = APIRouter()

@router.get("/mesh_upload", response_class=HTMLResponse)
async def mesh_upload_page():
    main = """
    <div class="info-box">
        <h4>☁️ 3D Mesh Analysis</h4>
        <p>Upload a textured <strong>.obj</strong> model and its accompanying texture map. We will use the 3D UV coordinates to mathematically mask the exact artifact surface, guaranteeing zero background contamination.</p>
    </div>
    <form action="/api/process_mesh" method="post" enctype="multipart/form-data" style="margin-top:20px; background:var(--bg-element); padding:20px; border-radius:8px;">
        <label>1. Select 3D Model (.obj)</label>
        <input type="file" name="obj_file" accept=".obj" required style="margin-bottom:15px;">
        
        <label>2. Select Texture Map (.png / .jpg)</label>
        <input type="file" name="tex_file" accept=".png,.jpg,.jpeg" required style="margin-bottom:15px;">
        
        <button type="submit" class="btn" onclick="this.innerText='Calculating UV Masks & 3D Area...'">🚀 Upload & Analyze</button>
    </form>
    """
    return page_layout(main, sidebar=None)

@router.post("/api/process_mesh")
async def process_mesh(obj_file: UploadFile = File(...), tex_file: UploadFile = File(...)):
    base_id = uuid.uuid4().hex
    obj_filename = f"model_{base_id}.obj"
    tex_filename = f"texture_{base_id}.png"
    
    obj_path = os.path.join(TEMP_DIR, obj_filename)
    tex_path = os.path.join(TEMP_DIR, tex_filename)
    
    with open(obj_path, "wb") as f: shutil.copyfileobj(obj_file.file, f)
    with open(tex_path, "wb") as f: shutil.copyfileobj(tex_file.file, f)
        
    try:
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        total_area = float(mesh.area)
        
        tex_img = Image.open(tex_path).convert("RGBA")
        tex_arr = np.array(tex_img)
        h, w = tex_arr.shape[:2]
        
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            raise ValueError("The uploaded OBJ does not contain UV mapping coordinates.")
            
        uvs = mesh.visual.uv
        faces = mesh.faces
        
        pixel_uvs = np.zeros_like(uvs)
        pixel_uvs[:, 0] = uvs[:, 0] * w
        pixel_uvs[:, 1] = (1.0 - uvs[:, 1]) * h
        pixel_uvs = pixel_uvs.astype(np.int32)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        triangles = pixel_uvs[faces]
        cv2.fillPoly(mask, triangles, 255)
        
        tex_arr[:, :, 3] = mask
        
        masked_tex_filename = f"masked_tex_{base_id}.png"
        masked_tex_path = os.path.join(TEMP_DIR, masked_tex_filename)
        Image.fromarray(tex_arr).save(masked_tex_path)

        meta = {"area_m2": total_area, "obj_path": obj_path}
        with open(masked_tex_path + ".json", "w") as f:
            json.dump(meta, f)
            
        return RedirectResponse(url=f"/color_analysis?preprocessed={masked_tex_filename}", status_code=303)
        
    except Exception as e:
        import traceback; traceback.print_exc()
        return page_layout(f"<h3>❌ Mesh Processing Error</h3><p>{str(e)}</p>")

@router.post("/api/export_mesh_result")
async def export_mesh_result(bundle_id: str = Form(...)):
    sess = active_bundles.get(bundle_id)
    if not sess: return JSONResponse({"error": "Session expired"}, 400)
    
    bundle = sess["bundle"]
    h, w = bundle.original_shape
    
    valid_mask_flat = bundle.valid_mask
    pixel_cluster_ids = bundle.cluster_labels[bundle.pixel_to_bin_idx]
    
    full_cluster_ids = np.full(valid_mask_flat.shape, -1, dtype=np.int32)
    full_cluster_ids[valid_mask_flat] = pixel_cluster_ids
    mask_2d = full_cluster_ids.reshape(h, w)
    
    # 1. Generate "All Colors" Texture
    clustered_img_all = np.zeros((h, w, 4), dtype=np.uint8)
    
    cluster_info = {}
    
    for k in range(bundle.cfg.n_clusters):
        if k in bundle.clusters:
            color_rgb = bundle.clusters[k]['rgb']
            
            # Add to the "All" map
            clustered_img_all[mask_2d == k] = [*color_rgb, 255]
            
            # 2. Generate Individual Transparent Textures for EACH Bucket
            single_img = np.zeros((h, w, 4), dtype=np.uint8)
            single_img[mask_2d == k] = [*color_rgb, 255]
            
            single_filename = f"tex_{bundle_id}_c{k}.png"
            cv2.imwrite(os.path.join(TEMP_DIR, single_filename), cv2.cvtColor(single_img, cv2.COLOR_RGBA2BGRA))
            
            cluster_info[k] = {
                "label": bundle.clusters[k]['label'],
                "percent": bundle.clusters[k]['percent'],
                "color": '#%02x%02x%02x' % tuple(color_rgb),
                "url": f"/temp_uploads/{single_filename}"
            }
            
    tex_all_filename = f"tex_{bundle_id}_all.png"
    cv2.imwrite(os.path.join(TEMP_DIR, tex_all_filename), cv2.cvtColor(clustered_img_all, cv2.COLOR_RGBA2BGRA))
    
    # Save texture references to session
    sess["mesh_textures"] = {
        "all": f"/temp_uploads/{tex_all_filename}",
        "original": f"/temp_uploads/{os.path.basename(sess['source_path'])}",
        "clusters": cluster_info
    }
    
    return RedirectResponse(url=f"/view_mesh/{bundle_id}", status_code=303)

@router.get("/view_mesh/{bundle_id}", response_class=HTMLResponse)
async def view_mesh(bundle_id: str):
    sess = active_bundles.get(bundle_id)
    if not sess or "mesh_textures" not in sess: 
        return HTMLResponse("Session Expired or 3D data not found. Please recalculate.")
    
    obj_url = f"/temp_uploads/{os.path.basename(sess['obj_path'])}"
    textures = sess["mesh_textures"]
    
    # Extract Area if available
    area_m2 = None
    if os.path.exists(sess["source_path"] + ".json"):
        with open(sess["source_path"] + ".json", "r") as f:
            meta = json.load(f)
            area_m2 = meta.get("area_m2")

    # Generate UI Buttons for Buckets
    ui_buttons = f"""
        <button id="btn-orig" class="btn btn-secondary" style="width:100%; margin-bottom:10px; text-align:left;" onclick="swapTexture('original', 'btn-orig')">📸 Original Masked Texture</button>
        <button id="btn-all" class="btn" style="width:100%; margin-bottom:10px; text-align:left;" onclick="swapTexture('all', 'btn-all')">🎨 All Color Clusters</button>
        <hr style="border-color:#3f3f46; margin:15px 0;">
    """
    
    js_texture_loads = f"""
        textures['original'] = textureLoader.load('{textures["original"]}');
        textures['original'].encoding = THREE.sRGBEncoding;
        textures['all'] = textureLoader.load('{textures["all"]}');
        textures['all'].encoding = THREE.sRGBEncoding;
    """
    
    for k, info in textures["clusters"].items():
        sqm_text = f"~ {(info['percent'] / 100.0) * area_m2:.4f} m²" if area_m2 else ""
        ui_buttons += f"""
        <button id="btn-c{k}" class="btn btn-secondary" style="width:100%; margin-bottom:5px; display:flex; align-items:center; gap:10px; text-align:left; background:#18181b;" onclick="swapTexture('c{k}', 'btn-c{k}')">
            <div style="width:15px; height:15px; border-radius:3px; background:{info['color']}; flex-shrink:0;"></div>
            <div style="font-size:0.8rem; line-height:1.2;">
                <strong>{info['label']}</strong>: {info['percent']:.1f}%<br>
                <span style="color:#a1a1aa;">{sqm_text}</span>
            </div>
        </button>
        """
        js_texture_loads += f"""
        textures['c{k}'] = textureLoader.load('{info["url"]}');
        textures['c{k}'].encoding = THREE.sRGBEncoding;
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive 3D Surface Viewer</title>
        <style>
            body {{ margin: 0; background-color: #09090b; color: white; font-family: 'Inter', sans-serif; overflow: hidden; }} 
            #ui-panel {{ position: absolute; top: 20px; left: 20px; width: 300px; background: rgba(24, 24, 27, 0.85); backdrop-filter: blur(10px); padding: 20px; border-radius: 12px; border: 1px solid #3f3f46; max-height: 90vh; overflow-y: auto; box-shadow: 0 4px 15px rgba(0,0,0,0.5); z-index: 10; }}
            .btn {{ padding: 10px 15px; border: none; border-radius: 6px; cursor: pointer; color: white; transition: 0.2s; font-family: inherit; }}
            .btn-secondary {{ background: #27272a; border: 1px solid #3f3f46; }}
            .btn.active-btn {{ background: #0d9488 !important; border-color: #0f766e !important; }}
        </style>
        <script src="https://unpkg.com/three@0.126.0/build/three.min.js"></script>
        <script src="https://unpkg.com/three@0.126.0/examples/js/loaders/OBJLoader.js"></script>
        <script src="https://unpkg.com/three@0.126.0/examples/js/controls/OrbitControls.js"></script>
    </head>
    <body>
        <div id="ui-panel">
            <h3 style="margin: 0 0 5px 0; color: #f4f4f5;">3D Inspector</h3>
            {ui_buttons}
            <div style="margin-top: 20px; text-align:center;">
                <a href="/color_results/{bundle_id}" style="color: #0d9488; text-decoration: none; font-size: 0.9rem;">← Back to Dashboard</a>
            </div>
        </div>
        
        <script>
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x09090b);
            const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 10000);
            const renderer = new THREE.WebGLRenderer({{antialias: true}});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.outputEncoding = THREE.sRGBEncoding;
            document.body.appendChild(renderer.domElement);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            // --- EVEN LIGHTING SETUP ---
            scene.add(new THREE.AmbientLight(0xffffff, 0.8));
            const sun = new THREE.DirectionalLight(0xffffff, 0.6);
            sun.position.set(1, 1, 1);
            scene.add(sun);
            const fill = new THREE.DirectionalLight(0xffffff, 0.4);
            fill.position.set(-1, 0, 1);
            scene.add(fill);
            const back = new THREE.DirectionalLight(0xffffff, 0.3);
            back.position.set(0, 0, -1);
            scene.add(back);
            // ---------------------------

            const textures = {{}};
            const textureLoader = new THREE.TextureLoader();
            {js_texture_loads}

            let activeMesh = null;
            const objLoader = new THREE.OBJLoader();
            objLoader.load('{obj_url}', function(object) {{
                object.traverse(function(child) {{
                    if (child.isMesh) {{
                        activeMesh = child;
                        child.material.map = textures['all'];
                        child.material.transparent = true;
                        child.material.alphaTest = 0.05;
                        child.material.side = THREE.DoubleSide;
                    }}
                }});
                
                const box = new THREE.Box3().setFromObject(object);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                object.position.sub(center);
                scene.add(object);
                
                const maxDim = Math.max(size.x, size.y, size.z);
                const cameraZ = Math.abs(maxDim / 2 / Math.tan(camera.fov * Math.PI / 360)) * 1.5;
                camera.position.set(0, 0, cameraZ);
                controls.update();
            }});

            function swapTexture(key, btnId) {{
                if (activeMesh && textures[key]) {{
                    activeMesh.material.map = textures[key];
                    activeMesh.material.needsUpdate = true;
                    document.querySelectorAll('.btn, .active-btn').forEach(b => {{
                        b.className = 'btn btn-secondary';
                    }});
                    document.getElementById(btnId).className = 'btn active-btn';
                }}
            }}
            document.getElementById('btn-all').className = 'btn active-btn';

            function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
            animate();
            window.addEventListener('resize', () => {{ 
                camera.aspect = window.innerWidth / window.innerHeight; 
                camera.updateProjectionMatrix(); 
                renderer.setSize(window.innerWidth, window.innerHeight); 
            }});
        </script>
    </body>
    </html>
    """
    return html