import streamlit as st
import cv2
import numpy as np
import io
import os
import base64
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from ColorClassifier_manual import ColorSegmenter, AnalysisConfig
import plotly.express as px
from fpdf import FPDF

# Set up the web page layout
st.set_page_config(layout="wide", page_title="Interactive Color Annotator")
st.title("Interactive Color Map Annotator")

# Initialize session state
if 'saved_layers' not in st.session_state or not isinstance(st.session_state.saved_layers, dict):
    st.session_state.saved_layers = {} 
if 'target_hex' not in st.session_state:
    st.session_state.target_hex = "#3498db"
if 'cluster_data' not in st.session_state:
    st.session_state.cluster_data = []

# SAFE STATE UPDATE
if 'pending_target_hex' in st.session_state:
    st.session_state.target_hex = st.session_state.pop('pending_target_hex')

# --- DATA CACHING ---
@st.cache_data
def get_processed_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    MAX_SIZE = 1000
    if max(img.size) > MAX_SIZE:
        img.thumbnail((MAX_SIZE, MAX_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img)
    lab = rgb2lab(arr / 255.0)
    return arr, lab

# Sidebar
with st.sidebar:
    st.header("1. Artifact Upload")
    uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        st.header("2. Pipette Settings")
        pipette_active = st.toggle("🧪 Activate Pipette Mode", value=False)
        brush_size = st.slider("Brush Size", 1, 50, 15)
        st.color_picker("Search Color", key="target_hex")
        
        st.header("3. View Settings")
        light_on = st.toggle("💡 Light On", value=False)
        
        st.header("5. Annotation")
        layer_name = st.text_input("Category Name")
        if st.button("Save/Merge Layer") and layer_name:
            st.session_state.trigger_save = True

        st.divider()
        st.header("6. Clustering")
        k_val = st.number_input("K Clusters", 1, 20, 5)
        if st.button("Run Analysis"):
            st.session_state.run_analysis_trigger = True
        
        if st.session_state.cluster_data:
            for i, c in enumerate(st.session_state.cluster_data):
                col1, col2 = st.columns([1, 3])
                with col1: st.markdown(f'<div style="background:{c["hex"]}; height:25px; border-radius:2px;"></div>', unsafe_allow_html=True)
                with col2:
                    if st.button(f"Use {c['label']} ({c['percent']:.1f}%)", key=f"use_{i}"):
                        st.session_state.pending_target_hex = c["hex"]
                        st.rerun()

# Main Logic
if uploaded_file:
    # Load and cache original data
    img_array, img_lab = get_processed_image(uploaded_file.getvalue())
    h_img, w_img = img_array.shape[:2]
    
    # 1. Similarity Math
    h_hex = st.session_state.target_hex.lstrip('#')
    t_rgb = np.array([int(h_hex[i:i+2], 16) for i in (0, 2, 4)])
    t_lab = rgb2lab(t_rgb.reshape(1, 1, 3) / 255.0).flatten()
    dist = np.sqrt(np.sum((img_lab - t_lab)**2, axis=2))

    # --- FRAGMENT FOR LIVE INTERACTION ---
    @st.fragment
    def interactive_workspace():
        st.subheader("4. Live Similarity Controls")
        sim_range = st.slider("Search Window (Lab)", 0, 250, (0, 80))
        r_min, r_max = sim_range
        tol = st.slider("Selection Cutoff", float(r_min), float(r_max), float(r_min + (r_max-r_min)*0.4))
        
        # Fast Visualization
        mask = dist <= tol
        base_dim = 0.3
        prox = np.clip(1.0 - (dist - r_min) / max(1e-5, r_max - r_min), 0, 1)
        img_vis = (img_array * (base_dim + (1.0-base_dim)*(prox**2))[:, :, np.newaxis]).astype(np.uint8)
        
        if not light_on and np.any(mask):
            img_vis[mask] = (img_vis[mask] * 0.7 + t_rgb * 0.3).astype(np.uint8)
            kernel = np.ones((3,3), np.uint8)
            edge = mask ^ cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            img_vis[edge] = [255, 255, 255]

        main_disp = img_array if light_on else img_vis
        
        # PLOTLY WORKSPACE
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Image(z=main_disp))
        
        if pipette_active:
            # Dense grid for reliable click capture
            grid_x, grid_y = np.meshgrid(np.linspace(0, w_img-1, 40), np.linspace(0, h_img-1, 40))
            fig.add_trace(go.Scatter(
                x=grid_x.flatten(), y=grid_y.flatten(),
                mode='markers', marker=dict(opacity=0, size=25),
                showlegend=False, hoverinfo='skip'
            ))

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), height=750,
            dragmode='pan' if not pipette_active else 'select',
            clickmode='event+select',
            hovermode=False, xaxis_visible=False, yaxis_visible=False,
            modebar_remove=['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraselayer', 'lasso2d']
        )
        
        # Cursor Styling
        cursor_px = max(12, (brush_size * (1100 / w_img)))
        r = cursor_px / 2
        if pipette_active:
            svg = f"""<svg width='{cursor_px}' height='{cursor_px}' viewBox='0 0 {cursor_px} {cursor_px}' xmlns='http://www.w3.org/2000/svg'><circle cx='{r}' cy='{r}' r='{max(1, r-1)}' fill='none' stroke='white' stroke-width='2'/><circle cx='{r}' cy='{r}' r='{max(0.5, r-2)}' fill='none' stroke='black' stroke-width='1'/></svg>"""
            svg_b64 = base64.b64encode(svg.encode()).decode()
            st.markdown(f"<style>.main-svg, .nsewdrag, .drag, iframe {{ cursor: url('data:image/svg+xml;base64,{svg_b64}') {r} {r}, crosshair !important; }}</style>", unsafe_allow_html=True)

        st.subheader("Interactive Selection (Wheel: Zoom | Drag: Pan | Click: Sample)")
        event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", config={'scrollZoom': True})
        
        # Click Capture
        if pipette_active and event and "selection" in event:
            pts = event["selection"].get("points", [])
            if pts:
                sx, sy = int(pts[0].get("x")), int(pts[0].get("y"))
                bh = brush_size // 2
                reg = img_array[max(0, sy-bh):min(h_img, sy+bh+1), max(0, sx-bh):min(w_img, sx+bh+1)]
                if reg.size > 0:
                    new_h = '#%02x%02x%02x' % tuple(np.mean(reg, axis=(0,1)).astype(np.uint8))
                    st.session_state.pending_target_hex = new_h
                    st.toast(f"Captured: {new_h}", icon="🧪")
                    st.rerun()
        
        st.session_state.last_tolerance = tol
        st.session_state.last_mask = mask

    interactive_workspace()

    # --- OTHER DISPLAYS ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Proximity Heatmap")
        s_prox = np.clip(1.0 - (dist / 100.0), 0, 1)
        i_v2 = (np.full_like(img_array, 255) * (1-(s_prox**3)[:,:,np.newaxis]) + t_rgb * (s_prox**3)[:,:,np.newaxis]).astype(np.uint8)
        st.image(i_v2, use_container_width=True)
    with col2:
        st.subheader("Isolated Selection")
        if 'last_mask' in st.session_state:
            i_v3 = np.full_like(img_array, 255); i_v3[st.session_state.last_mask] = img_array[st.session_state.last_mask]
            st.image(i_v3, use_container_width=True)

    # Save logic
    if getattr(st.session_state, 'trigger_save', False):
        m = st.session_state.last_mask
        n_s = {"hex": st.session_state.target_hex, "percent": (np.sum(m)/(h_img*w_img))*100, "params": {"thr": round(st.session_state.last_tolerance, 1)}}
        if layer_name in st.session_state.saved_layers:
            st.session_state.saved_layers[layer_name]['mask'] = np.logical_or(st.session_state.saved_layers[layer_name]['mask'], m)
            st.session_state.saved_layers[layer_name]['sub'].append(n_s)
        else:
            st.session_state.saved_layers[layer_name] = {"mask": m, "sub": [n_s]}
        st.sidebar.success(f"Saved {layer_name}")

    if st.session_state.saved_layers:
        st.divider()
        cols = st.columns(3)
        for i, (name, data) in enumerate(st.session_state.saved_layers.items()):
            with cols[i%3]:
                st.write(f"### {name}")
                for s in data['sub']:
                    st.markdown(f'<div style="display:flex; align-items:center; gap:8px;"><div style="background:{s["hex"]}; width:12px; height:12px;"></div><b>{s["hex"]}</b>: {s["percent"]:.2f}%</div>', unsafe_allow_html=True)
                st.image((data['mask']*255).astype(np.uint8), use_container_width=True)

        if st.button("📊 Generate PDF Report"):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("helvetica", "B", 16); pdf.cell(0, 10, "FLACA Analysis Report", ln=1, align="C"); pdf.ln(10)
            for name, data in st.session_state.saved_layers.items():
                pdf.set_font("helvetica", "B", 12); pdf.cell(0, 10, f"Layer: {name}", ln=1)
                for s in data['sub']: pdf.set_font("helvetica", "", 10); pdf.cell(0, 6, f"- {s['hex']}: {s['percent']:.2f}%", ln=1)
            st.download_button("Download Report", pdf.output(), "analysis.pdf")

    if getattr(st.session_state, 'run_analysis_trigger', False):
        with st.spinner("Processing with FLACA..."):
            cfg = AnalysisConfig(n_clusters=int(k_val))
            seg = ColorSegmenter(config=cfg, raw_rgb=img_array.reshape(-1, 3))
            seg.run_clustering()
            st.session_state.cluster_data = [{"label": v['label'], "percent": v['percent'], "hex": '#%02x%02x%02x' % tuple(v['rgb'])} for k, v in seg.clusters.items()]
            st.rerun()
else:
    st.info("Upload image to start.")
