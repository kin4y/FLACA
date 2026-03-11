import os

STATIC_DIR = "static_results"
TEMP_DIR = "temp_uploads"

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Global session state memory
active_bundles = {}

# Defaults
color_defaults = {
    "k": 5, "L_thresh": 30.0, "C_thresh": 0.1, "ab_step": 1.0, "point_size": 12,
    "size_mode": "sqrt", "top_n_chroma": "", "top_n_achro": "",
    "pie_show_labels": "True", "show_input": "True", "show_plots_initial": "False",
    "show_plots_final": "True", "random_seed": 42, "shrink_img": 1.0,
    "weight_L": 0.0, "weight_a": 1.0, "weight_b": 1.0, "weight_C": 0.0, "weight_h": 0.0
}