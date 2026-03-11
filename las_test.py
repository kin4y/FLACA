import json
import trimesh
import numpy as np
import os
import uuid
import cv2
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import queue
import threading
import uvicorn
from PIL import Image
from ColorClassifier_manual import ColorSegmenter, AnalysisConfig

app = FastAPI()
TEMP_DIR = "surface_percent_data"
os.makedirs(TEMP_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=TEMP_DIR), name="static")

# Progress tracking
progress_queues = {}
progress_lock = threading.Lock()

def report_progress(session_id, percent, message):
    """Report progress for a session"""
    with progress_lock:
        if session_id in progress_queues:
            try:
                progress_queues[session_id].put_nowait({'percent': percent, 'message': message})
            except:
                pass

def pack_rectangles_square(rectangles):
    """Pack rectangles into a roughly square canvas."""
    total_area = sum(w * h for w, h in [(r[0], r[1]) for r in rectangles])
    canvas_side = int(np.ceil(np.sqrt(total_area * 1.2)))
    
    rectangles = sorted(rectangles, key=lambda r: r[1], reverse=True)
    
    current_shelf = {'y': 0, 'height': 0, 'x': 0, 'remaining_width': canvas_side}
    packed = []
    max_x = 0
    max_y = 0
    
    for width, height, face_idx in rectangles:
        if width <= current_shelf['remaining_width']:
            x = current_shelf['x']
            y = current_shelf['y']
            packed.append((x, y, width, height, face_idx))
            
            current_shelf['x'] += width
            current_shelf['remaining_width'] -= width
            current_shelf['height'] = max(current_shelf['height'], height)
            
            max_x = max(max_x, x + width)
            max_y = max(max_y, y + height)
        else:
            new_y = current_shelf['y'] + current_shelf['height']
            
            if new_y + height > canvas_side:
                canvas_side = new_y + height
            
            current_shelf = {
                'y': new_y,
                'height': height,
                'x': width,
                'remaining_width': canvas_side - width
            }
            packed.append((0, new_y, width, height, face_idx))
            
            max_x = max(max_x, width)
            max_y = max(max_y, new_y + height)
    
    return packed, max_x, max_y

def get_face_uv_bounds(face_uv, texture_width, texture_height):
    """Get the pixel bounding box for a face in UV space."""
    u_min = face_uv[:, 0].min()
    u_max = face_uv[:, 0].max()
    v_min = face_uv[:, 1].min()
    v_max = face_uv[:, 1].max()
    
    x_min = int(np.floor(u_min * texture_width))
    x_max = int(np.ceil(u_max * texture_width))
    y_min = int(np.floor((1.0 - v_max) * texture_height))
    y_max = int(np.ceil((1.0 - v_min) * texture_height))
    
    x_min = max(0, x_min)
    x_max = min(texture_width, x_max)
    y_min = max(0, y_min)
    y_max = min(texture_height, y_max)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return x_min, y_min, width, height

def point_in_triangle_2d(p, v0, v1, v2):
    """Check if point p is inside triangle using barycentric coordinates."""
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    
    dot00 = np.dot(v0v1, v0v1)
    dot01 = np.dot(v0v1, v0v2)
    dot02 = np.dot(v0v1, v0p)
    dot11 = np.dot(v0v2, v0v2)
    dot12 = np.dot(v0v2, v0p)
    
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def extract_face_pixels_exact(face_uv, texture_img):
    """Extract exact pixels with pixel-perfect sharp edges."""
    tex_h, tex_w = texture_img.shape[:2]
    x_min, y_min, width, height = get_face_uv_bounds(face_uv, tex_w, tex_h)
    
    if width <= 0 or height <= 0:
        return np.array([]).reshape(0, 3), 0, 0
    
    bbox_region = texture_img[y_min:y_min+height, x_min:x_min+width].copy()
    
    v0_px = np.array([(face_uv[0, 0] * tex_w - x_min), ((1.0 - face_uv[0, 1]) * tex_h - y_min)])
    v1_px = np.array([(face_uv[1, 0] * tex_w - x_min), ((1.0 - face_uv[1, 1]) * tex_h - y_min)])
    v2_px = np.array([(face_uv[2, 0] * tex_w - x_min), ((1.0 - face_uv[2, 1]) * tex_h - y_min)])
    
    pixels = []
    for y in range(height):
        for x in range(width):
            p = np.array([float(x) + 0.5, float(y) + 0.5])
            if point_in_triangle_2d(p, v0_px, v1_px, v2_px):
                pixels.append(bbox_region[y, x])
    
    if len(pixels) == 0:
        return np.array([]).reshape(0, 3), width, height
    
    return np.array(pixels, dtype=np.uint8), width, height

def create_masked_atlas_for_clustering(atlas, valid_mask):
    """Create masked atlas with random padding from valid pixels."""
    valid_pixels = atlas[valid_mask]
    n_valid = len(valid_pixels)
    print(f"Valid pixels extracted: {n_valid}")
    
    side = int(np.ceil(np.sqrt(n_valid)))
    needed = side * side
    masked_valid_mask = np.ones(n_valid, dtype=bool)
    
    if n_valid < needed:
        padding_count = needed - n_valid
        print(f"Adding {padding_count} padding pixels (random samples from valid pixels)")
        random_indices = np.random.choice(n_valid, size=padding_count, replace=True)
        padding = valid_pixels[random_indices]
        valid_pixels = np.vstack([valid_pixels, padding])
        padding_mask = np.zeros(padding_count, dtype=bool)
        masked_valid_mask = np.concatenate([masked_valid_mask, padding_mask])
    
    masked_atlas = valid_pixels.reshape(side, side, 3)
    masked_valid_mask = masked_valid_mask.reshape(side, side)
    
    print(f"Masked atlas: {side}x{side}")
    print(f"Real face pixels: {masked_valid_mask.sum()} ({masked_valid_mask.sum()/needed*100:.2f}%)")
    print(f"Random padding: {(~masked_valid_mask).sum()} ({(~masked_valid_mask).sum()/needed*100:.2f}%)")
    
    return masked_atlas, masked_valid_mask, side

def create_flattened_atlas_exact(mesh, texture_path):
    """Create flattened atlas using exact pixel extraction."""
    img_bgr = cv2.imread(texture_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tex_h, tex_w, _ = img_rgb.shape
    
    face_areas = mesh.area_faces
    face_uvs = mesh.visual.uv[mesh.faces]
    
    print(f"Total faces: {len(face_areas)}")
    print(f"Total mesh area: {face_areas.sum():.4f} m²")
    print(f"Texture resolution: {tex_w}x{tex_h}")
    
    face_pixel_data = []
    rectangles = []
    
    for face_idx in range(len(face_areas)):
        face_uv = face_uvs[face_idx]
        pixels, est_width, est_height = extract_face_pixels_exact(face_uv, img_rgb)
        
        if len(pixels) == 0:
            face_pixel_data.append(None)
            continue
        
        n_pixels = len(pixels)
        side = int(np.ceil(np.sqrt(n_pixels)))
        face_pixel_data.append(pixels)
        rectangles.append((side, side, face_idx))
    
    print(f"Faces with pixels: {len([f for f in face_pixel_data if f is not None])}")
    
    packed, canvas_width, canvas_height = pack_rectangles_square(rectangles)
    
    print(f"Atlas size: {canvas_width} x {canvas_height}")
    aspect_ratio = max(canvas_width, canvas_height) / min(canvas_width, canvas_height)
    print(f"Aspect ratio: {aspect_ratio:.2f}:1")
    
    atlas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    valid_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
    face_regions = {}
    
    for x, y, width, height, face_idx in packed:
        pixels = face_pixel_data[face_idx]
        if pixels is None:
            continue
        
        n_pixels = len(pixels)
        needed = width * height
        
        if n_pixels < needed:
            padding = np.zeros((needed - n_pixels, 3), dtype=np.uint8)
            pixels = np.vstack([pixels, padding])
        elif n_pixels > needed:
            pixels = pixels[:needed]
        
        face_rect = pixels.reshape(height, width, 3)
        atlas[y:y+height, x:x+width] = face_rect
        
        n_valid = len(face_pixel_data[face_idx])
        valid_flat = np.zeros(needed, dtype=bool)
        valid_flat[:n_valid] = True
        valid_rect = valid_flat.reshape(height, width)
        valid_mask[y:y+height, x:x+width] = valid_rect
        
        face_regions[face_idx] = (x, y, width, height)
    
    print(f"Valid pixels in atlas: {valid_mask.sum()}")
    print(f"Background pixels in atlas: {(~valid_mask).sum()}")
    print(f"Total atlas pixels: {canvas_width * canvas_height}")
    
    background_stats = {
        'total_pixels': canvas_width * canvas_height,
        'valid_pixels': int(valid_mask.sum()),
        'background_pixels': int((~valid_mask).sum()),
        'valid_percentage': float(valid_mask.sum() / (canvas_width * canvas_height) * 100),
        'background_percentage': float((~valid_mask).sum() / (canvas_width * canvas_height) * 100)
    }
    
    return atlas, face_regions, valid_mask, background_stats

def calculate_3d_percentages(obj_path, texture_path, k=5, cluster_features=None, 
                            L_thresh=10.0, C_thresh=6.0, ab_step=1.0, random_seed=42,
                            run_stability_analysis=True, n_stability_runs=20, session_id=None):
    """Compute precise 3D surface area percentage using exact pixel extraction."""
    if cluster_features is None:
        cluster_features = {'L': 0, 'a': 1, 'b': 1, 'C': 1, 'h': 1}
    
    np.random.seed(random_seed)
    
    report_progress(session_id, 5, "Loading 3D mesh...")
    mesh = trimesh.load(obj_path, process=False)
    face_areas = mesh.area_faces
    
    report_progress(session_id, 10, "Extracting pixels from UV map...")
    atlas, face_regions, valid_mask, background_stats = create_flattened_atlas_exact(mesh, texture_path)
    
    report_progress(session_id, 25, "Saving atlas images...")
    atlas_path = os.path.join(TEMP_DIR, f"atlas_{uuid.uuid4().hex}.png")
    Image.fromarray(atlas).save(atlas_path)
    
    report_progress(session_id, 30, "Creating masked atlas with random padding...")
    masked_atlas, masked_valid_mask, masked_side = create_masked_atlas_for_clustering(atlas, valid_mask)
    masked_atlas_path = os.path.join(TEMP_DIR, f"masked_atlas_{uuid.uuid4().hex}.png")
    Image.fromarray(masked_atlas).save(masked_atlas_path)
    
    padding_pixels = (~masked_valid_mask).sum()
    
    report_progress(session_id, 40, "Running color clustering...")
    cfg = AnalysisConfig(
        L_thresh=L_thresh,
        C_thresh=C_thresh,
        ab_step=ab_step,
        shrink_img=1.0,
        n_clusters=k,
        random_state=random_seed,
        cluster_features=cluster_features
    )
    
    segmenter = ColorSegmenter(masked_atlas_path, cfg)
    segmenter.run_clustering()
    
    masked_pixel_clusters = segmenter.cluster_labels[segmenter.pixel_to_bin_idx]
    masked_cluster_map = masked_pixel_clusters.reshape(masked_side, masked_side)
    
    atlas_cluster_map = np.zeros(atlas.shape[:2], dtype=np.uint8)
    valid_pixel_clusters = masked_cluster_map[masked_valid_mask]
    atlas_cluster_map[valid_mask] = valid_pixel_clusters
    
    # Run stability analysis if requested
    report_progress(session_id, 60, "Calculating surface areas...")
    stability_stats = None
    if run_stability_analysis and padding_pixels > 0:
        report_progress(session_id, 65, f"Running stability analysis (0/{n_stability_runs})...")
        
        all_results = []
        for run in range(n_stability_runs):
            report_progress(session_id, 65 + int(25 * run / n_stability_runs), 
                          f"Stability run {run+1}/{n_stability_runs}...")
            
            masked_atlas_temp, masked_valid_mask_temp, masked_side_temp = create_masked_atlas_for_clustering(atlas, valid_mask)
            temp_path = os.path.join(TEMP_DIR, f"temp_masked_{run}.png")
            Image.fromarray(masked_atlas_temp).save(temp_path)
            
            segmenter_temp = ColorSegmenter(temp_path, cfg)
            segmenter_temp.run_clustering()
            
            masked_pixel_clusters_temp = segmenter_temp.cluster_labels[segmenter_temp.pixel_to_bin_idx]
            masked_cluster_map_temp = masked_pixel_clusters_temp.reshape(masked_side_temp, masked_side_temp)
            atlas_cluster_map_temp = np.zeros(atlas.shape[:2], dtype=np.uint8)
            valid_pixel_clusters_temp = masked_cluster_map_temp[masked_valid_mask_temp]
            atlas_cluster_map_temp[valid_mask] = valid_pixel_clusters_temp
            
            face_cluster_areas_temp = np.zeros(k)
            for face_idx, (x, y, width, height) in face_regions.items():
                face_area = face_areas[face_idx]
                face_region_mask = valid_mask[y:y+height, x:x+width]
                face_region_clusters = atlas_cluster_map_temp[y:y+height, x:x+width]
                valid_clusters = face_region_clusters[face_region_mask]
                
                if len(valid_clusters) > 0:
                    unique, counts = np.unique(valid_clusters, return_counts=True)
                    dominant_cluster = unique[np.argmax(counts)]
                    face_cluster_areas_temp[dominant_cluster] += face_area
            
            total_area_temp = face_cluster_areas_temp.sum()
            percentages_temp = (face_cluster_areas_temp / total_area_temp) * 100
            all_results.append(percentages_temp)
            
            os.remove(temp_path)
        
        all_results = np.array(all_results)
        mean_percentages = all_results.mean(axis=0)
        std_percentages = all_results.std(axis=0)
        percentile_5 = np.percentile(all_results, 5, axis=0)
        percentile_95 = np.percentile(all_results, 95, axis=0)
        max_variation = np.max(percentile_95 - percentile_5)
        
        stability_stats = {
            'mean': mean_percentages.tolist(),
            'std': std_percentages.tolist(),
            'percentile_5': percentile_5.tolist(),
            'percentile_95': percentile_95.tolist(),
            'max_variation': float(max_variation),
            'n_runs': n_stability_runs
        }
    
    # Create visualizations
    report_progress(session_id, 92, "Creating visualizations...")
    atlas_cluster_viz = np.zeros_like(atlas)
    
    for face_idx, (x, y, width, height) in face_regions.items():
        face_region_mask = valid_mask[y:y+height, x:x+width]
        face_region_clusters = atlas_cluster_map[y:y+height, x:x+width]
        
        valid_clusters = face_region_clusters[face_region_mask]
        if len(valid_clusters) > 0:
            unique, counts = np.unique(valid_clusters, return_counts=True)
            dominant_cluster = unique[np.argmax(counts)]
            cluster_color = segmenter.clusters[dominant_cluster]['rgb']
            atlas_cluster_viz[y:y+height, x:x+width] = cluster_color
    
    cluster_atlases = {}
    for cluster_id in range(k):
        cluster_atlas = np.zeros_like(atlas)
        
        for face_idx, (x, y, width, height) in face_regions.items():
            face_region_mask = valid_mask[y:y+height, x:x+width]
            face_region_clusters = atlas_cluster_map[y:y+height, x:x+width]
            
            valid_clusters = face_region_clusters[face_region_mask]
            if len(valid_clusters) > 0:
                unique, counts = np.unique(valid_clusters, return_counts=True)
                dominant_cluster = unique[np.argmax(counts)]
                
                if dominant_cluster == cluster_id:
                    cluster_atlas[y:y+height, x:x+width] = atlas[y:y+height, x:x+width]
                else:
                    cluster_atlas[y:y+height, x:x+width] = (atlas[y:y+height, x:x+width] * 0.2).astype(np.uint8)
        
        cluster_atlases[cluster_id] = cluster_atlas
    
    cluster_viz_filename = f"atlas_clusters_{uuid.uuid4().hex}.png"
    Image.fromarray(atlas_cluster_viz).save(os.path.join(TEMP_DIR, cluster_viz_filename))
    
    cluster_atlas_files = {}
    for cluster_id, cluster_atlas in cluster_atlases.items():
        filename = f"atlas_cluster_{cluster_id}_{uuid.uuid4().hex}.png"
        Image.fromarray(cluster_atlas).save(os.path.join(TEMP_DIR, filename))
        cluster_atlas_files[cluster_id] = filename
    
    # Calculate face areas per cluster
    face_cluster_areas = np.zeros(k)
    for face_idx, (x, y, width, height) in face_regions.items():
        face_area = face_areas[face_idx]
        face_region_mask = valid_mask[y:y+height, x:x+width]
        face_region_clusters = atlas_cluster_map[y:y+height, x:x+width]
        valid_clusters = face_region_clusters[face_region_mask]
        
        if len(valid_clusters) > 0:
            unique, counts = np.unique(valid_clusters, return_counts=True)
            dominant_cluster = unique[np.argmax(counts)]
            face_cluster_areas[dominant_cluster] += face_area
    
    total_area = face_cluster_areas.sum()
    results = {}
    for cluster_id in range(k):
        percentage = (face_cluster_areas[cluster_id] / total_area) * 100
        results[cluster_id] = percentage
    
    # Create texture mask
    report_progress(session_id, 97, "Creating 3D viewer mask...")
    img_bgr = cv2.imread(texture_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tex_h, tex_w, _ = img_rgb.shape
    
    from skimage.color import rgb2lab
    lab_flat = rgb2lab(img_rgb.astype(float) / 255.0).reshape(-1, 3)
    L = lab_flat[:, 0]
    a = lab_flat[:, 1]
    b = lab_flat[:, 2]
    C = np.hypot(a, b)
    
    is_achro = (L < cfg.L_thresh) | (C < cfg.C_thresh)
    a_bin = np.round(a / cfg.ab_step).astype(np.int32)
    b_bin = np.round(b / cfg.ab_step).astype(np.int32)
    group_id = is_achro.astype(np.int32)
    
    texture_keys = np.column_stack([group_id, a_bin, b_bin])
    
    bin_to_cluster = {}
    for bin_idx, key in enumerate(segmenter.unique_keys):
        key_tuple = tuple(key)
        bin_to_cluster[key_tuple] = segmenter.cluster_labels[bin_idx]
    
    pixel_clusters = np.zeros(len(lab_flat), dtype=np.uint8)
    for i in range(len(lab_flat)):
        key_tuple = tuple(texture_keys[i])
        if key_tuple in bin_to_cluster:
            pixel_clusters[i] = bin_to_cluster[key_tuple]
        else:
            min_dist = np.inf
            closest_cluster = 0
            for cluster_id in range(k):
                cluster_mask = segmenter.cluster_labels == cluster_id
                if cluster_mask.any():
                    cluster_bins = np.where(cluster_mask)[0]
                    cluster_lab = segmenter.bin_lab[cluster_bins].mean(axis=0)
                    dist = np.linalg.norm(lab_flat[i] - cluster_lab)
                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = cluster_id
            pixel_clusters[i] = closest_cluster
    
    texture_mask = pixel_clusters.reshape(tex_h, tex_w)
    texture_mask_filename = f"texture_mask_{uuid.uuid4().hex}.png"
    cv2.imwrite(os.path.join(TEMP_DIR, texture_mask_filename), texture_mask)
    
    report_progress(session_id, 100, "Complete!")
    
    return {
        "percentages": results,
        "atlas_file": os.path.basename(atlas_path),
        "masked_atlas_file": os.path.basename(masked_atlas_path),
        "atlas_cluster_viz": cluster_viz_filename,
        "cluster_atlas_files": {int(i): cluster_atlas_files[i] for i in range(k)},
        "texture_mask_file": texture_mask_filename,
        "k": k,
        "tex_file": os.path.basename(texture_path),
        "obj_file": os.path.basename(obj_path),
        "total_area_m2": float(total_area),
        "cluster_areas_m2": {int(i): float(face_cluster_areas[i]) for i in range(k)},
        "atlas_coverage": float(valid_mask.sum() / (atlas.shape[0] * atlas.shape[1]) * 100),
        "cluster_colors": {int(i): segmenter.clusters[i]['rgb'].tolist() for i in range(k)},
        "background_stats": {
            **background_stats,
            'masked_atlas_pixels': int(masked_side * masked_side),
            'masked_atlas_real_pixels': int(masked_valid_mask.sum()),
            'masked_atlas_padding': int((~masked_valid_mask).sum()),
            'masked_atlas_padding_percentage': float((~masked_valid_mask).sum() / (masked_side * masked_side) * 100)
        },
        "stability_stats": stability_stats
    }

@app.get("/progress/{session_id}")
async def progress_stream(session_id: str):
    """Server-Sent Events endpoint for progress updates"""
    async def event_generator():
        q = queue.Queue()
        with progress_lock:
            progress_queues[session_id] = q
        
        try:
            while True:
                try:
                    progress = q.get(timeout=30)
                    yield f"data: {json.dumps(progress)}\n\n"
                    
                    if progress['percent'] >= 100:
                        break
                except queue.Empty:
                    yield f"data: {json.dumps({'percent': -1, 'message': 'Processing...'})}\n\n"
        finally:
            with progress_lock:
                if session_id in progress_queues:
                    del progress_queues[session_id]
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <body style="font-family:sans-serif; background:#111; color:white; display:flex; justify-content:center; align-items:center; min-height:100vh; margin:0; padding:20px;">
        <div style="background:#222; padding:30px; border-radius:12px; border:1px solid #444; max-width:600px;">
            <h2>High-Precision 3D Surface Area Analyzer</h2>
            <p style="color:#888;">Exact pixel extraction from UV map - no sampling</p>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="files" multiple required><br><br>
                
                <label>Clusters (k):</label>
                <input type="number" name="k" value="5" min="2" max="20" style="width:60px;"><br><br>
                
                <fieldset style="border:1px solid #444; padding:10px; margin:10px 0;">
                    <legend>Cluster Features</legend>
                    <label><input type="checkbox" name="feat_L" value="1"> Lightness (L)</label><br>
                    <label><input type="checkbox" name="feat_a" value="1" checked> Red-Green (a)</label><br>
                    <label><input type="checkbox" name="feat_b" value="1" checked> Blue-Yellow (b)</label><br>
                    <label><input type="checkbox" name="feat_C" value="1" checked> Chroma (C)</label><br>
                    <label><input type="checkbox" name="feat_h" value="1" checked> Hue (h)</label><br>
                </fieldset>
                
                <label>L Threshold:</label>
                <input type="number" name="L_thresh" value="10" step="0.1" style="width:80px;"><br><br>
                
                <label>C Threshold:</label>
                <input type="number" name="C_thresh" value="6" step="0.1" style="width:80px;"><br><br>
                
                <label>AB Step:</label>
                <input type="number" name="ab_step" value="1" step="0.1" style="width:80px;"><br><br>
                
                <label>Random Seed:</label>
                <input type="number" name="random_seed" value="42" style="width:80px;"><br><br>
                
                <label><input type="checkbox" name="run_stability" checked> Run Stability Analysis (20 runs)</label><br><br>
                
                <button type="submit" style="padding:10px 20px; background:#10b981; color:white; border:none; cursor:pointer; border-radius:5px;" onclick="startAnalysis(event)">Analyze Surface Area</button>
            </form>
        </div>
        <div id="loading" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); z-index:9999; justify-content:center; align-items:center; flex-direction:column;">
            <div style="background:#222; padding:40px; border-radius:12px; border:1px solid #444; text-align:center; min-width:500px;">
                <h3 style="color:#10b981; margin-top:0;">Processing Analysis</h3>
                <div id="progress-text" style="color:#ccc; margin:20px 0; font-size:15px; min-height:24px;">Starting...</div>
                <div style="width:100%; height:24px; background:#333; border-radius:12px; overflow:hidden; margin:20px 0;">
                    <div id="progress-bar" style="width:0%; height:100%; background:linear-gradient(90deg, #10b981, #059669); transition:width 0.5s; display:flex; align-items:center; justify-content:center; color:white; font-size:12px; font-weight:bold;"></div>
                </div>
                <div id="progress-percent" style="color:#888; font-size:13px;">0%</div>
            </div>
        </div>
        <script>
            let sessionId = null;
            let eventSource = null;
            
            function startAnalysis(e) {
                e.preventDefault();
                sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                document.getElementById('loading').style.display = 'flex';
                
                eventSource = new EventSource('/progress/' + sessionId);
                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.percent >= 0) {
                        document.getElementById('progress-bar').style.width = data.percent + '%';
                        document.getElementById('progress-bar').textContent = Math.round(data.percent) + '%';
                        document.getElementById('progress-percent').textContent = Math.round(data.percent) + '%';
                        document.getElementById('progress-text').textContent = data.message;
                    }
                    if (data.percent >= 100) {
                        eventSource.close();
                    }
                };
                
                const form = e.target.closest('form');
                const formData = new FormData(form);
                formData.append('session_id', sessionId);
                
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                }).then(response => response.text())
                  .then(html => {
                      document.open();
                      document.write(html);
                      document.close();
                  });
                
                return false;
            }
        </script>
    </body>
    """

@app.post("/upload")
async def handle_upload(
    files: list[UploadFile] = File(...), 
    k: int = Form(5),
    feat_L: int = Form(0),
    feat_a: int = Form(1),
    feat_b: int = Form(1),
    feat_C: int = Form(1),
    feat_h: int = Form(1),
    L_thresh: float = Form(10.0),
    C_thresh: float = Form(6.0),
    ab_step: float = Form(1.0),
    random_seed: int = Form(42),
    run_stability: int = Form(1),
    session_id: str = Form(None)
):
    obj_name, tex_name, tex_path = None, None, None
    for file in files:
        path = os.path.join(TEMP_DIR, file.filename)
        with open(path, "wb") as f: 
            f.write(await file.read())
        if file.filename.lower().endswith('.obj'): 
            obj_name = file.filename
        if file.filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            tex_name = file.filename
            tex_path = path
    
    cluster_features = {
        'L': feat_L,
        'a': feat_a,
        'b': feat_b,
        'C': feat_C,
        'h': feat_h
    }
    
    data = calculate_3d_percentages(
        os.path.join(TEMP_DIR, obj_name), 
        tex_path, 
        k=k,
        cluster_features=cluster_features,
        L_thresh=L_thresh,
        C_thresh=C_thresh,
        ab_step=ab_step,
        random_seed=random_seed,
        run_stability_analysis=bool(run_stability),
        n_stability_runs=20,
        session_id=session_id
    )
    
    btns = "".join([
        f'<button onclick="setMask({i})" class="btn" id="btn-{i}" style="border-left: 4px solid rgb({data["cluster_colors"][i][0]},{data["cluster_colors"][i][1]},{data["cluster_colors"][i][2]});">Cluster {i}</button>' 
        for i in range(k)
    ])
    rows = "".join([
        f'<div class="row">'
        f'<span>Cluster {i}</span>'
        f'<div><b>{data["percentages"][i]:.2f}%</b><br>'
        f'<small style="color:#888">{data["cluster_areas_m2"][i]:.4f} m²</small>'
        + (f'<br><small style="color:#666">±{(data["stability_stats"]["percentile_95"][i] - data["stability_stats"]["percentile_5"][i])/2:.2f}% (90% CI)</small>' if data.get("stability_stats") else '')
        + '</div>'
        f'</div>' 
        for i in range(k)
    ])

    return HTMLResponse(f"""
    <style>
        body {{ margin:0; display:flex; background:#000; color:#eee; font-family:sans-serif; height:100vh; overflow:hidden; }}
        #sidebar {{ width:340px; background:#151515; padding:20px; border-right:1px solid #333; overflow-y:auto; }}
        #main {{ flex-grow:1; display:flex; flex-direction:column; }}
        #viewer {{ flex-grow:1; position:relative; }}
        #atlas-viewer {{ height:300px; background:#0a0a0a; border-top:1px solid #333; padding:10px; overflow:auto; }}
        .btn {{ width:100%; margin:5px 0; padding:12px; cursor:pointer; background:#222; color:#fff; border:1px solid #444; border-radius:4px; text-align:left; }}
        .btn.active {{ background:#059669; }}
        .row {{ display:flex; justify-content:space-between; padding:12px 0; border-bottom:1px solid #333; }}
        .row > div {{ text-align:right; }}
        small {{ font-size:11px; }}
        .info {{ background:#1a1a1a; padding:10px; margin:10px 0; border-radius:4px; font-size:12px; }}
        #atlas-display {{ max-height:280px; width:auto; border:1px solid #444; border-radius:4px; }}
    </style>
    
    <div id="sidebar">
        <h3>3D Surface Analysis</h3>
        <p style="color:#888; font-size:13px;">Total Surface: {data["total_area_m2"]:.4f} m²</p>
        <div class="info">
            <strong>Background Exclusion:</strong><br>
            Original atlas: {data["background_stats"]["total_pixels"]:,} px<br>
            Valid: {data["background_stats"]["valid_pixels"]:,} ({data["background_stats"]["valid_percentage"]:.1f}%)<br>
            Masked atlas: {data["background_stats"]["masked_atlas_pixels"]:,} px<br>
            Padding: {data["background_stats"]["masked_atlas_padding"]:,} ({data["background_stats"]["masked_atlas_padding_percentage"]:.1f}%)<br>
            <span style="color:#10b981;">✓ Random padding used</span>
        </div>
        {'<div class="info"><strong>Stability (95%ile):</strong><br>Max: ' + f'{data["stability_stats"]["max_variation"]:.2f}%<br>Runs: {data["stability_stats"]["n_runs"]}</div>' if data.get("stability_stats") else ''}
        <div class="info">
            <strong>Config:</strong><br>
            L:{cluster_features['L']} a:{cluster_features['a']} b:{cluster_features['b']}<br>
            C:{cluster_features['C']} h:{cluster_features['h']}
        </div>
        {btns}
        <button onclick="setMask(-1)" class="btn active" id="btn-all">All Clusters</button>
        <div style="margin-top:20px;">
            <h4>Surface Distribution</h4>
            {rows}
        </div>
        {'<div style="margin-top:15px;"><h4>Stability Analysis</h4>' + ''.join([f'<div class="row"><span>C{i}</span><div><small style="color:#888;">Mean:{data["stability_stats"]["mean"][i]:.1f}%<br>5-95%:{data["stability_stats"]["percentile_5"][i]:.1f}-{data["stability_stats"]["percentile_95"][i]:.1f}%</small></div></div>' for i in range(k)]) + '</div>' if data.get("stability_stats") else ''}
        <div style="margin-top:15px;">
            <a href="/static/{data['atlas_file']}" target="_blank" style="color:#10b981; font-size:11px;">📄 Atlas</a> | 
            <a href="/static/{data['masked_atlas_file']}" target="_blank" style="color:#10b981; font-size:11px;">🎯 Masked</a>
        </div>
    </div>
    <div id="main">
        <div id="viewer"></div>
        <div id="atlas-viewer">
            <div style="padding:10px; color:#888; font-size:12px;">Atlas: <span id="atlas-label">All Clusters</span></div>
            <img id="atlas-display" src="/static/{data['atlas_cluster_viz']}" alt="Atlas">
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/OBJLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        let scene, camera, renderer, customMat, model;
        const clusterAtlasFiles = {json.dumps(data['cluster_atlas_files'])};
        const allClustersAtlas = '/static/{data["atlas_cluster_viz"]}';

        async function init() {{
            scene = new THREE.Scene();
            const viewerDiv = document.getElementById('viewer');
            camera = new THREE.PerspectiveCamera(45, viewerDiv.clientWidth/viewerDiv.clientHeight, 0.001, 1000);
            camera.position.set(0.5, 0.5, 0.5);
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(viewerDiv.clientWidth, viewerDiv.clientHeight);
            viewerDiv.appendChild(renderer.domElement);
            new THREE.OrbitControls(camera, renderer.domElement);

            const tex = new THREE.TextureLoader().load('/static/{data["tex_file"]}');
            const mask = new THREE.TextureLoader().load('/static/{data["texture_mask_file"]}');
            mask.magFilter = THREE.NearestFilter;
            mask.minFilter = THREE.NearestFilter;

            customMat = new THREE.ShaderMaterial({{
                uniforms: {{ tex: {{value: tex}}, mask: {{value: mask}}, activeID: {{value: -1.0}} }},
                vertexShader: `varying vec2 vUv; void main() {{ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }}`,
                fragmentShader: `varying vec2 vUv; uniform sampler2D tex; uniform sampler2D mask; uniform float activeID;
                    void main() {{
                        vec4 color = texture2D(tex, vUv);
                        float id = texture2D(mask, vUv).r * 255.0;
                        if (activeID >= 0.0 && abs(id - activeID) > 0.5) {{
                            gl_FragColor = vec4(color.rgb * 0.15, 1.0);
                        }} else {{ gl_FragColor = color; }}
                    }}`,
                side: THREE.DoubleSide
            }});

            new THREE.OBJLoader().load('/static/{data["obj_file"]}', (obj) => {{
                obj.traverse(c => {{ if(c.isMesh) c.material = customMat; }});
                model = obj; scene.add(obj);
                const box = new THREE.Box3().setFromObject(obj);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3()).length();
                camera.position.copy(center).add(new THREE.Vector3(size, size, size).multiplyScalar(0.5));
                camera.lookAt(center);
            }});
            animate();
        }}

        function setMask(id) {{
            customMat.uniforms.activeID.value = parseFloat(id);
            document.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
            const atlasDisplay = document.getElementById('atlas-display');
            const atlasLabel = document.getElementById('atlas-label');
            if(id === -1) {{
                document.getElementById('btn-all').classList.add('active');
                atlasDisplay.src = allClustersAtlas;
                atlasLabel.textContent = 'All Clusters';
            }} else {{
                document.getElementById('btn-' + id).classList.add('active');
                atlasDisplay.src = '/static/' + clusterAtlasFiles[id];
                atlasLabel.textContent = 'Cluster ' + id;
            }}
        }}

        function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); }}
        
        window.addEventListener('resize', () => {{
            const viewerDiv = document.getElementById('viewer');
            camera.aspect = viewerDiv.clientWidth / viewerDiv.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(viewerDiv.clientWidth, viewerDiv.clientHeight);
        }});
        init();
    </script>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)