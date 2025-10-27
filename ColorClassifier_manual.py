#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Imports
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from skimage.color import lab2rgb  # local import to limit global deps
try:
    from sklearn.cluster import KMeans
    _HAS_SK = True
except Exception:
    _HAS_SK = False
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from joblib import Parallel, delayed


# In[2]:


# =======================
# IO and color conversion
# =======================

def load_rgb_pixels(image_path, shrink_img=1.0):
    """
    Load an image and return all pixels as an (N, 3) uint8 RGB array (no dedup).

    Parameters
    ----------
    image_path : str
        Path to the image file.
    shrink_img : float, optional
        Scale factor to resize the image while preserving aspect ratio.
        Values < 1.0 shrink the image, > 1.0 enlarge it. Default is 1.0 (no scaling).

    Returns
    -------
    np.ndarray
        Flattened RGB pixel array of shape (N, 3), dtype uint8.
    """
    img = Image.open(image_path).convert('RGB')

    # Resize while preserving aspect ratio
    if shrink_img != 1.0:
        w, h = img.size
        new_size = (int(w * shrink_img), int(h * shrink_img))
        img = img.resize(new_size, Image.LANCZOS)

    rgb = np.array(img, dtype=np.uint8).reshape(-1, 3)
    return rgb, img

def rgb_to_lab(rgb_u8):
    """Convert uint8 RGB to CIE Lab."""
    return rgb2lab(rgb_u8.astype(float) / 255.0)

def lab_to_rgb_u8(lab):
    """Convert Lab to sRGB uint8."""
    rgb = lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)

# ==========================================
# Threshold split and quantization helpers
# ==========================================

def split_masks(lab, L_thresh=10.0, C_thresh=6.0):
    """
    Return (chroma_mask, achro_mask) based on:
      achromatic: (L < L_thresh) or (C < C_thresh)
      chromatic: complement
    """
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    C = np.hypot(a, b)
    achro_mask = (L < L_thresh) | (C < C_thresh)
    chroma_mask = ~achro_mask
    return chroma_mask, achro_mask

def quantize_ab(a, b, ab_step):
    """
    Quantize a*, b* into integer bins using rounding by ab_step.
    Returns (a_bin, b_bin) as int32 arrays.
    """
    a_bin = np.round(a / ab_step).astype(np.int32)
    b_bin = np.round(b / ab_step).astype(np.int32)
    return a_bin, b_bin

# ==========================================
# Stacking by (a,b) only, pixel-weighted bins
# ==========================================

def lab_stack_by_ab_pixel_weighted(lab, ab_step=1.0):
    """
    Group Lab pixels by (a,b) only, using quantization of a*, b* by ab_step.
    Pixel-weighted (every pixel contributes).

    For each (a,b) bin ("stack"), compute:
      - count: number of pixels in the bin
      - L_rep: median(L*) among pixels in the bin
      - a_rep, b_rep: mean(a*), mean(b*)
      - C_rep: median chroma among pixels in the bin

    Returns a dict with arrays (one row per bin):
      L_rep, a_rep, b_rep, C_rep, counts, bins_unique, inv
    """
    if lab.size == 0:
        return {
            "L_rep": np.zeros((0,), dtype=float),
            "a_rep": np.zeros((0,), dtype=float),
            "b_rep": np.zeros((0,), dtype=float),
            "C_rep": np.zeros((0,), dtype=float),
            "counts": np.zeros((0,), dtype=int),
            "bins_unique": np.zeros((0, 2), dtype=np.int32),
            "inv": np.zeros((0,), dtype=np.int32),
        }

    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    C = np.hypot(a, b)

    a_bin, b_bin = quantize_ab(a, b, ab_step)
    bins = np.stack([a_bin, b_bin], axis=1)

    bins_unique, inv = np.unique(bins, axis=0, return_inverse=True)
    n_bins = bins_unique.shape[0]

    counts = np.bincount(inv, minlength=n_bins)

    L_rep = np.zeros(n_bins, dtype=float)
    a_rep = np.zeros(n_bins, dtype=float)
    b_rep = np.zeros(n_bins, dtype=float)
    C_rep = np.zeros(n_bins, dtype=float)

    for k in range(n_bins):
        idx = np.where(inv == k)[0]
        L_rep[k] = np.median(L[idx])
        a_rep[k] = np.mean(a[idx])
        b_rep[k] = np.mean(b[idx])
        C_rep[k] = np.median(C[idx])

    return {
        "L_rep": L_rep,
        "a_rep": a_rep,
        "b_rep": b_rep,
        "C_rep": C_rep,
        "counts": counts,
        "bins_unique": bins_unique,  # shape (n_bins, 2) integer (a_bin, b_bin)
        "inv": inv,                  # per-pixel mapping to bin index (subset-relative)
    }


# In[3]:


# ==========================================
# Group build (chroma/achro)
# ==========================================

def build_group_from_mask(lab, mask, ab_step, group_name):
    """
    Stack a masked subset of Lab pixels by (a,b), compute representatives and coordinates
    appropriate for that group's panel.

    Returns a dict:
      {
        'group': 'chroma'|'achro',
        'counts': (K,),
        'bins_unique': (K,2) integer (a_bin,b_bin),
        'lab_rep': (K,3),
        'rgb_rep': (K,3) uint8,
        'coords_panel': (K,2) -> chroma: (a*,b*), achro: (L*, C),
        'C_rep': (K,),
        'a_rep': (K,),
        'b_rep': (K,),
        'L_rep': (K,),
      }
    """
    # guard for empty mask
    if mask is None or lab.size == 0 or np.count_nonzero(mask) == 0:
        return {
            'group': group_name,
            'counts': np.zeros((0,), dtype=int),
            'bins_unique': np.zeros((0, 2), dtype=np.int32),
            'lab_rep': np.zeros((0, 3), dtype=float),
            'rgb_rep': np.zeros((0, 3), dtype=np.uint8),
            'coords_panel': np.zeros((0, 2), dtype=float),
            'C_rep': np.zeros((0,), dtype=float),
            'a_rep': np.zeros((0,), dtype=float),
            'b_rep': np.zeros((0,), dtype=float),
            'L_rep': np.zeros((0,), dtype=float),
        }

    subset = lab[mask]

    # Use previously defined function
    res = lab_stack_by_ab_pixel_weighted(subset, ab_step=ab_step)

    # Representatives and display colors
    lab_rep = np.column_stack([res["L_rep"], res["a_rep"], res["b_rep"]])
    rgb_rep = lab_to_rgb_u8(lab_rep)

    # Panel coordinates
    if group_name == 'chroma':
        coords_panel = np.column_stack([res["a_rep"], res["b_rep"]])  # (a, b)
    elif group_name == 'achro':
        coords_panel = np.column_stack([res["L_rep"], res["C_rep"]])  # (L, C)
    else:
        raise ValueError("group_name must be 'chroma' or 'achro'")

    return {
        'group': group_name,
        'counts': res["counts"],
        'bins_unique': res["bins_unique"],   # quantized (a_bin,b_bin)
        'lab_rep': lab_rep,
        'rgb_rep': rgb_rep,
        'coords_panel': coords_panel,
        'C_rep': res["C_rep"],
        'a_rep': res["a_rep"],
        'b_rep': res["b_rep"],
        'L_rep': res["L_rep"],
    }

# ==========================================
# Panel-consistent ordering
# ==========================================

def chroma_panel_order(a_rep, b_rep):
    """
    Order for chromatic bins consistent with (a,b) panel:
    sort by hue angle, then by chroma radius (ascending).
    Returns index array.
    """
    theta = np.arctan2(b_rep, a_rep)  # [-pi, pi]
    theta = (theta + 2*np.pi) % (2*np.pi)  # [0, 2pi)
    r = np.hypot(a_rep, b_rep)
    order = np.lexsort((r, theta))  # primary theta, secondary r
    return order

def achro_panel_order(L_rep, C_rep):
    """
    Order for achromatic bins consistent with (L,C) panel:
    sort by L* (ascending), then C (ascending).
    Returns index array.
    """
    order = np.lexsort((C_rep, L_rep))
    return order

# ==========================================
# Bin naming
# ==========================================

def generate_bin_names(n):
    """
    Generate bin names: A..Z, then A1..Z1, then A2..Z2, etc.
    Example for n=30: ['A','B',...,'Z','A1','B1','C1','D1']
    """
    names = []
    for i in range(n):
        letter = chr(ord('A') + (i % 26))
        cycle = i // 26
        name = letter if cycle == 0 else f"{letter}{cycle}"
        names.append(name)
    return names


# In[4]:


# ==========================================
# Combined bins for single pie (chroma first, then achro)
# ==========================================

def prepare_combined_bins(
    chroma_group,
    achro_group,
    total_pixels,
    top_n_chroma=None,
    top_n_achro=None
):
    """
    Prepare a single combined sequence of bins:
      - all chromatic bins in panel order (angle then radius),
      - followed by all achromatic bins in panel order (L then C).

    Optionally truncate each group with top-N before concatenating.

    Inputs:
      chroma_group, achro_group: dicts from build_group_from_mask(...)
      total_pixels: total pixels in the original image (for percentages)
      top_n_chroma, top_n_achro: optional ints

    Returns dict:
      {
        'total_pixels': int,
        'counts': (M,) int,
        'rgb': (M,3) uint8,
        'names': list[str] length M (A..Z, A1..Z1,...),
        'group_flags': (M,) ndarray of 'chroma' or 'achro',
        'keys': (M,3) int32 array [group_id, a_bin, b_bin], where group_id: 0=chroma, 1=achro,
        'index_maps': {
            'chroma': kept_indices_in_chroma_group,  # indices into chroma_group arrays
            'achro': kept_indices_in_achro_group,    # indices into achro_group arrays
        },
        'percents': (M,) float (percent of total_pixels)
      }
    """
    # Order each group in panel-consistent order
    idx_ch = chroma_panel_order(chroma_group['a_rep'], chroma_group['b_rep']) if chroma_group['counts'].size else np.array([], dtype=int)
    idx_ac = achro_panel_order(achro_group['L_rep'], achro_group['C_rep']) if achro_group['counts'].size else np.array([], dtype=int)

    # Apply top-N truncation
    if top_n_chroma is not None:
        idx_ch = idx_ch[:min(top_n_chroma, idx_ch.size)]
    if top_n_achro is not None:
        idx_ac = idx_ac[:min(top_n_achro, idx_ac.size)]

    # Reorder arrays per group
    ch_counts = chroma_group['counts'][idx_ch]
    ch_rgb    = chroma_group['rgb_rep'][idx_ch]
    ch_bins   = chroma_group['bins_unique'][idx_ch]  # (a_bin,b_bin)

    ac_counts = achro_group['counts'][idx_ac]
    ac_rgb    = achro_group['rgb_rep'][idx_ac]
    ac_bins   = achro_group['bins_unique'][idx_ac]   # (a_bin,b_bin)

    # Concatenate chroma then achro
    counts = np.concatenate([ch_counts, ac_counts], axis=0) if ch_counts.size or ac_counts.size else np.zeros((0,), dtype=int)
    rgb    = np.concatenate([ch_rgb, ac_rgb], axis=0) if ch_rgb.size or ac_rgb.size else np.zeros((0, 3), dtype=np.uint8)

    # Group flags
    flags_ch = np.array(['chroma'] * ch_counts.size, dtype='<U6')
    flags_ac = np.array(['achro']  * ac_counts.size, dtype='<U6')
    group_flags = np.concatenate([flags_ch, flags_ac], axis=0) if flags_ch.size or flags_ac.size else np.zeros((0,), dtype='<U6')

    # Keys: [group_id, a_bin, b_bin] with group_id 0 for chroma, 1 for achro
    keys_ch = np.column_stack([np.zeros(ch_bins.shape[0], dtype=np.int32), ch_bins]).astype(np.int32) if ch_bins.size else np.zeros((0, 3), dtype=np.int32)
    keys_ac = np.column_stack([np.ones(ac_bins.shape[0],  dtype=np.int32), ac_bins]).astype(np.int32) if ac_bins.size else np.zeros((0, 3), dtype=np.int32)
    keys = np.vstack([keys_ch, keys_ac]) if keys_ch.size or keys_ac.size else np.zeros((0, 3), dtype=np.int32)

    # Bin names A..Z, A1..Z1, ...
    names = generate_bin_names(len(counts))

    # Percentages
    percents = (counts.astype(float) / float(total_pixels) * 100.0) if total_pixels > 0 else np.zeros_like(counts, dtype=float)

    return {
        'total_pixels': int(total_pixels),
        'counts': counts,
        'rgb': rgb,
        'names': names,
        'group_flags': group_flags,
        'keys': keys,
        'index_maps': {
            'chroma': idx_ch,
            'achro': idx_ac,
        },
        'percents': percents,
    }

# ==========================================
# Name lookup helper
# ==========================================

def build_name_to_index(names):
    """
    Build a dictionary mapping bin name -> combined index.
    """
    return {name: i for i, name in enumerate(names)}


# In[5]:


# ==========================
# Image preview helper
# ==========================

def show_input_image(image_path, title="Input image"):
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(6, 6), dpi=120)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================
# Two-panel scatter (chroma vs achro)
# ==========================

def plot_two_panels(
    chroma_group,
    achro_group,
    point_size=12,
    size_mode='sqrt',
    chroma_title="Chromatic (a*, b*)",
    achro_title="Achromatic (L*, C*)"
):
    """
    Plot two panels from group dicts produced by build_group_from_mask:
      - left: chromatic points at (a*, b*)
      - right: achromatic points at (L*, C*)
    Marker size encodes stack size (pixel count).
    """
    chroma_coords = chroma_group['coords_panel']
    chroma_rgb = chroma_group['rgb_rep']
    chroma_counts = chroma_group['counts']

    achro_coords = achro_group['coords_panel']
    achro_rgb = achro_group['rgb_rep']
    achro_counts = achro_group['counts']

    if size_mode == 'sqrt':
        sizes_ch = point_size * np.sqrt(np.maximum(chroma_counts, 1)) if chroma_counts.size else None
        sizes_ac = point_size * np.sqrt(np.maximum(achro_counts, 1)) if achro_counts.size else None
    else:
        sizes_ch = point_size * np.maximum(chroma_counts, 1) if chroma_counts.size else None
        sizes_ac = point_size * np.maximum(achro_counts, 1) if achro_counts.size else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Chromatic panel: (a, b)
    if chroma_coords.size:
        axes[0].scatter(
            chroma_coords[:, 0], chroma_coords[:, 1],
            c=chroma_rgb / 255.0, s=sizes_ch, marker='s', edgecolors='none'
        )
    axes[0].set_title(chroma_title)
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    # Achromatic panel: (L, C)
    if achro_coords.size:
        axes[1].scatter(
            achro_coords[:, 0], achro_coords[:, 1],
            c=achro_rgb / 255.0, s=sizes_ac, marker='s', edgecolors='none'
        )
    axes[1].set_title(achro_title + " — axes: L* (x), C (y)")
    axes[1].set_aspect('auto')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================
# Single combined pie (chroma first, achro next) with bin names
# ==========================

def plot_single_combined_pie(
    combined_bins,
    title="Pixel share per stack — Combined (Chroma → Achro)",
    show_labels=True,
    min_label_percent=0.2,
    label_offset=1.15,
    line_width=0.8,
    font_size=9
):
    """
    Draw a single pie using the combined bins dict returned by prepare_combined_bins.
    Uses combined order: chroma first (panel order), then achro (panel order).
    Labels show bin name and percentage; tiny slices can be skipped with min_label_percent.
    """
    counts = combined_bins['counts']
    if counts is None or len(counts) == 0:
        print("No bins to plot for pie chart.")
        return

    total_pixels = combined_bins['total_pixels']
    colors = (combined_bins['rgb'] / 255.0)
    perc = combined_bins['percents']
    names = combined_bins['names']
    flags = combined_bins['group_flags']  # 'chroma' or 'achro'

    # Optional: subtle wedge edge to separate slices
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(
        perc,
        colors=colors,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(linewidth=0.5, edgecolor='white')
    )
    ax.set_title(title)

    if show_labels:
        for w, p, nm, gf in zip(wedges, perc, names, flags):
            if p < min_label_percent:
                continue
            ang = 0.5 * (w.theta1 + w.theta2)
            ang_rad = np.deg2rad(ang)
            x = np.cos(ang_rad)
            y = np.sin(ang_rad)
            x_text = label_offset * x
            y_text = label_offset * y
            ha = 'left' if x >= 0 else 'right'
            label = f"{nm} {p:.2f}%"
            # color-coding group in label (optional): append marker
            # label += " (C)" if gf == 'chroma' else " (A)"
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(x_text, y_text),
                ha=ha,
                va='center',
                fontsize=font_size,
                textcoords='data',
                arrowprops=dict(
                    arrowstyle='-',
                    lw=line_width,
                    color='0.3',
                    shrinkA=0,
                    shrinkB=0,
                    relpos=(0.5, 0.5)
                )
            )

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


# In[6]:


# ==========================
# Orchestration helpers
# ==========================

def compute_lab_from_image_path(image_path, shrink_img):
    """
    Load image -> RGB uint8 -> Lab float.
    Returns (rgb_pixels_uint8, lab_float) where both are flattened to (N,3).
    """
    rgb_pixels, img = load_rgb_pixels(image_path, shrink_img)
    lab = rgb_to_lab(rgb_pixels)
    return rgb_pixels, lab, img

def assemble_groups(lab, L_thresh=10.0, C_thresh=6.0, ab_step=1.0):
    """
    Split Lab pixels into chroma/achro using thresholds and build each group.

    Returns:
      {
        'chroma': group_dict,
        'achro': group_dict,
        'masks': {'chroma': chroma_mask, 'achro': achro_mask},
        'totals': {'pixels': int},
      }
    """
    chroma_mask, achro_mask = split_masks(lab, L_thresh=L_thresh, C_thresh=C_thresh)

    chroma_group = build_group_from_mask(lab, chroma_mask, ab_step=ab_step, group_name='chroma')
    achro_group  = build_group_from_mask(lab, achro_mask,  ab_step=ab_step, group_name='achro')

    return {
        'chroma': chroma_group,
        'achro': achro_group,
        'masks': {'chroma': chroma_mask, 'achro': achro_mask},
        'totals': {'pixels': int(lab.shape[0])},
    }

def summarize_groups(groups_bundle):
    """
    Print basic stats for chroma/achro groups.
    """
    total_pixels = groups_bundle['totals']['pixels']
    chroma = groups_bundle['chroma']
    achro  = groups_bundle['achro']

    n_chroma_bins = int(chroma['counts'].size)
    n_achro_bins  = int(achro['counts'].size)

    n_chroma_pixels = int(chroma['counts'].sum()) if n_chroma_bins else 0
    n_achro_pixels  = int(achro['counts'].sum())  if n_achro_bins else 0

    print(f"Total pixels: {total_pixels}")
    print(f"Chromatic pixels: {n_chroma_pixels} ({(n_chroma_pixels/total_pixels*100.0):.2f}%) -> bins: {n_chroma_bins}")
    if n_chroma_bins:
        print(f"  Chromatic bin size median: {int(np.median(chroma['counts']))}, max: {int(np.max(chroma['counts']))}")
    print(f"Achromatic pixels: {n_achro_pixels} ({(n_achro_pixels/total_pixels*100.0):.2f}%) -> bins: {n_achro_bins}")
    if n_achro_bins:
        print(f"  Achromatic bin size median: {int(np.median(achro['counts']))}, max: {int(np.max(achro['counts']))}")

# ==========================
# Main entry: split pipeline
# ==========================

def run_color_analysis_split(
    image_path,
    L_thresh=10.0,          # very dark -> achromatic panel
    C_thresh=6.0,           # low chroma -> achromatic panel
    ab_step=1.0,            # (a,b) quantization for stacking
    point_size=12,
    size_mode='sqrt',
    top_n_chroma=None,      # keep only first N bins in chroma panel order (None = all)
    top_n_achro=None,       # keep only first N bins in achro panel order (None = all)
    pie_show_labels=True,
    show_input=True,
    show_plots=False, 
    shrink_img = 1.0
):
    """
    Orchestrates the full pipeline in smaller steps:
      - Load and convert to Lab
      - Split into chroma/achro groups
      - Summarize stats
      - Plot two-panel scatter
      - Prepare combined bins (chroma first, then achro) and plot a single pie
      - Return a bundle with groups and combined bins
    """
    # Load & convert
    rgb_pixels, lab, img = compute_lab_from_image_path(image_path, shrink_img)
    total_pixels = rgb_pixels.shape[0]
    if show_input:
        show_input_image(image_path)

    # Build groups
    gb = assemble_groups(lab, L_thresh=L_thresh, C_thresh=C_thresh, ab_step=ab_step)

    # Stats
    summarize_groups(gb)
    if show_plots is True:
        # Plots: panels
        plot_two_panels(
            gb['chroma'],
            gb['achro'],
            point_size=point_size,
            size_mode=size_mode,
            chroma_title=f"Chromatic (a*, b*) — ab_step={ab_step}",
            achro_title=f"Achromatic (L* vs C) — L<{L_thresh} or C<{C_thresh}",
        )

    # Combined bins for single pie
    combined = prepare_combined_bins(
        gb['chroma'],
        gb['achro'],
        total_pixels=total_pixels,
        top_n_chroma=top_n_chroma,
        top_n_achro=top_n_achro
    )

    # Name lookup map
    name_to_index = build_name_to_index(combined['names'])
    if show_plots is True:
        # Plot single combined pie
        plot_single_combined_pie(
            combined,
            title="Pixel share per stack — Combined (Chroma → Achro)",
            show_labels=pie_show_labels,
            min_label_percent=0.2,
            label_offset=1.15,
            line_width=0.8,
            font_size=10
        )

    # Return bundle
    return {
        'image_path': image_path,
        'total_pixels': total_pixels,
        'groups': gb,                     # {'chroma':..., 'achro':..., 'masks':..., 'totals':...}
        'combined_bins': combined,        # combined dict with counts, rgb, names, flags, keys, percents
        'name_to_index': name_to_index,   # dict name -> combined index
        'params': {
            'L_thresh': L_thresh,
            'C_thresh': C_thresh,
            'ab_step': ab_step,
            'point_size': point_size,
            'size_mode': size_mode,
            'top_n_chroma': top_n_chroma,
            'top_n_achro': top_n_achro,
        }
    }


# In[7]:


# ==========================================
# Bin spec lookup
# ==========================================

def get_bin_spec(combined_bins, name_to_index, bin_name):
    """
    Resolve a bin name into a spec:
      {
        'name': str,
        'group': 'chroma'|'achro',
        'group_id': 0|1,
        'a_bin': int,
        'b_bin': int,
        'rgb': (3,) uint8,
        'index': int (combined index)
      }
    """
    if bin_name not in name_to_index:
        raise KeyError(f"Unknown bin name: {bin_name}")
    idx = name_to_index[bin_name]
    group_flag = combined_bins['group_flags'][idx]
    group_id = 0 if group_flag == 'chroma' else 1
    a_bin = int(combined_bins['keys'][idx, 1])
    b_bin = int(combined_bins['keys'][idx, 2])
    rgb = combined_bins['rgb'][idx]
    return {
        'name': bin_name,
        'group': group_flag,
        'group_id': group_id,
        'a_bin': a_bin,
        'b_bin': b_bin,
        'rgb': rgb,
        'index': int(idx),
    }

# ==========================================
# Classification of any image to a chosen bin
# ==========================================

def classify_image_pixels_to_bin(
    img=None,
    bin_spec=None,
    L_thresh=10.0,
    C_thresh=6.0,
    ab_step=1.0,
    lab=None,
    precomputed_lab=False
):
    """
    Classify pixels into the selected bin (by name/spec).

    Now supports precomputed Lab input to skip repeated
    image→Lab conversions for performance.
    """

    if precomputed_lab and lab is not None:
        # lab is (N,3) array
        H = W = None  # shape known to caller
        lab_flat = lab
    else:
        # Original path: build from img directly
        W, H = img.size
        rgb = np.array(img, dtype=np.uint8).reshape(-1, 3)
        lab_flat = rgb_to_lab(rgb)

    # Extract channels
    L = lab_flat[:, 0]
    a = lab_flat[:, 1]
    b = lab_flat[:, 2]
    C = np.hypot(a, b)

    # Threshold masks
    achro_mask = (L < L_thresh) | (C < C_thresh)
    chroma_mask = ~achro_mask

    # Quantize a*, b*
    a_bin = np.round(a / ab_step).astype(np.int32)
    b_bin = np.round(b / ab_step).astype(np.int32)

    # Group mask
    if bin_spec['group'] == 'chroma':
        group_mask = chroma_mask
    else:
        group_mask = achro_mask

    # Bin equality
    same_bin = (a_bin == bin_spec['a_bin']) & (b_bin == bin_spec['b_bin'])

    # Combine
    flat_mask = group_mask & same_bin
    count = int(flat_mask.sum())

    # Reshape if we have image dims
    if H is not None and W is not None:
        mask = flat_mask.reshape(H, W)
    else:
        mask = flat_mask  # caller will reshape
    return {
        'mask': mask,
        'count': count,
        'size': (H, W) if (H and W) else (None, None),
        'bin_spec': bin_spec,
    }

# ==========================================
# Visualization helpers for debug masks
# ==========================================

def visualize_bin_mask(img, mask, mode='highlight', alpha=0.65, title=None):
    """
    Visualize pixels belonging to the chosen bin.
    Modes:
      - 'mask': show binary mask
      - 'highlight': show original image with selected pixels highlighted and others dimmed
    """
    arr = np.array(img, dtype=np.uint8)
    H, W = mask.shape

    if mode == 'mask':
        plt.figure(figsize=(6, 6), dpi=120)
        plt.imshow(mask, cmap='gray')
        plt.title(title or "Bin mask")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return

    if mode == 'highlight':
        # Dim background
        dim = (arr.astype(np.float32) * 0.15).astype(np.uint8)
        out = dim.copy()
        # Keep original colors where mask is True
        out[mask] = arr[mask]
        plt.figure(figsize=(6, 6), dpi=120)
        plt.imshow(out)
        plt.title(title or "Highlighted bin pixels")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return

    raise ValueError("mode must be 'mask' or 'highlight'")

# ==========================================
# High-level debug function
# ==========================================

def debug_classify_pixels_by_bin(
    img,
    combined_bins_bundle,
    bin_name,
    L_thresh=None,
    C_thresh=None,
    ab_step=None,
    visualize='highlight'
):
    """
    Debug helper to classify any image's pixels by a bin chosen by name (e.g. 'A','B','A1',...).
    combined_bins_bundle is the returned dict from run_color_analysis_split(...):
      {
        'combined_bins': ...,
        'name_to_index': ...,
        'params': {'L_thresh', 'C_thresh', 'ab_step', ...}
      }

    Parameters:
      - img: image to classify (can be different from the one used to define bins)
      - bin_name: string bin name, e.g. 'A', 'Z', 'B1'
      - L_thresh, C_thresh, ab_step: override thresholds/step; defaults to those from bundle
      - visualize: 'highlight' or 'mask' or None

    Returns:
      {
        'mask': (H,W) bool,
        'count': int,
        'size': (H,W),
        'bin_spec': {...},
      }
    """
    combined_bins = combined_bins_bundle['combined_bins']
    name_to_index = combined_bins_bundle['name_to_index']
    params = combined_bins_bundle['params']

    # Use provided overrides or fall back to original analysis params
    L_t = L_thresh if L_thresh is not None else params['L_thresh']
    C_t = C_thresh if C_thresh is not None else params['C_thresh']
    step = ab_step if ab_step is not None else params['ab_step']

    # Resolve bin
    bin_spec = get_bin_spec(combined_bins, name_to_index, bin_name)

    # Classify pixels
    result = classify_image_pixels_to_bin(
        img,
        bin_spec,
        L_thresh=L_t,
        C_thresh=C_t,
        ab_step=step
    )

    H, W = result['size']
    print(f"Bin: {bin_spec['name']} [{bin_spec['group']}], a_bin={bin_spec['a_bin']}, b_bin={bin_spec['b_bin']}")
    print(f"Image: {image_path} — size: {W}x{H}")
    print(f"Matched pixels: {result['count']}")

    # Visualize
    if visualize in ('highlight', 'mask'):
        title = f"Bin {bin_spec['name']} ({bin_spec['group']}): {result['count']} px"
        visualize_bin_mask(img, result['mask'], mode=visualize, title=title)

    return result


# In[29]:


# K-means over existing bins (based solely on the provided bundle)
# - Cluster on Lab representatives in the existing combined order
# - Provide structures to aggregate/plot clusters
# - Do NOT rewrite earlier code; this augments the bundle and provides new plots

def _get_lab_features_in_combined_order(bundle):
    """
    Build feature matrix X (M,3) in the combined order from bundle.
    Uses Lab representatives from groups and the index maps in combined_bins.
    """
    gb = bundle['groups']
    cb = bundle['combined_bins']
    idx_ch = cb['index_maps']['chroma']
    idx_ac = cb['index_maps']['achro']

    X_ch = gb['chroma']['lab_rep'][idx_ch] if idx_ch.size else np.zeros((0, 3), dtype=float)
    X_ac = gb['achro']['lab_rep'][idx_ac] if idx_ac.size else np.zeros((0, 3), dtype=float)

    X = np.vstack([X_ch, X_ac]) if (X_ch.size or X_ac.size) else np.zeros((0, 3), dtype=float)
    return X


def kmeans_bins_from_bundle(bundle, n_clusters=4, random_state=0, normalize=True):
    """
    Run KMeans on the bin representatives (Lab). Returns:
      {
        'assignments': (M,) int cluster ids in [0..n_clusters-1] following combined order,
        'centers': (K,3) float (Lab centers),
        'n_clusters': int,
      }
    """
    X = _get_lab_features_in_combined_order(bundle)
    if X.size == 0:
        raise ValueError("No bins available for clustering.")

    X_proc = X.copy()
    if normalize:
        mu = X_proc.mean(axis=0, keepdims=True)
        sd = X_proc.std(axis=0, keepdims=True) + 1e-8
        X_proc = (X_proc - mu) / sd

    if _HAS_SK:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_proc)
        centers = km.cluster_centers_
        if normalize:
            centers = centers * sd + mu
    else:
        # Simple fallback k-means
        rng = np.random.default_rng(random_state)
        M = X_proc.shape[0]
        if n_clusters > M:
            raise ValueError("n_clusters cannot exceed the number of bins.")
        init_idx = rng.choice(M, size=n_clusters, replace=False)
        centers = X_proc[init_idx]
        for _ in range(25):
            d2 = ((X_proc[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            new_centers = np.array([
                X_proc[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                for k in range(n_clusters)
            ])
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        if normalize:
            centers = centers * sd + mu

    return {
        'assignments': labels.astype(int),
        'centers': centers,
        'n_clusters': int(n_clusters),
    }


def _weighted_mean_rgb_uint8(rgb_u8, weights):
    """
    Weighted mean in sRGB space (uint8 in, uint8 out).
    """
    if rgb_u8.size == 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    w = weights.astype(np.float64)
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        return rgb_u8.mean(axis=0).astype(np.uint8)
    mean = (rgb_u8.astype(np.float64) * w[:, None]).sum(axis=0) / s
    mean = np.clip(np.round(mean), 0, 255).astype(np.uint8)
    return mean


def build_cluster_structures(bundle, clustering, cluster_prefix="C"):
    """
    Build cluster aggregation strictly from bundle + clustering result.

    Returns:
      {
        'clusters': {
            k: {
                'id': k,
                'label': f'{cluster_prefix}{k+1}',
                'member_indices': np.array([...], int),   # indices into combined order
                'member_names': [list of original bin names],
                'member_flags': [list of 'chroma'|'achro'],
                'member_keys': (Nk,3) int32 [group_id, a_bin, b_bin],
                'counts': (Nk,) int per member,
                'total_count': int sum of counts,
                'rgb_agg': (3,) uint8 weighted by counts,
            },
            ...
        },
        'combined_bins_clustered': {
            'total_pixels': same as original,
            'counts': (K,) int per cluster,
            'rgb': (K,3) uint8 aggregated,
            'names': [K] cluster labels (e.g., 'C1','C2',...),
            'percents': (K,) float,
            'member_name_lists': list of lists of member bin names (for legend),
        }
      }
    """
    cb = bundle['combined_bins']
    counts = cb['counts']
    rgb = cb['rgb']
    names = cb['names']
    flags = cb['group_flags']
    keys = cb['keys']  # per-bin [group_id, a_bin, b_bin]
    total_pixels = cb['total_pixels']

    labels = clustering['assignments']  # shape (M,)
    K = int(clustering['n_clusters'])

    clusters = {}
    agg_counts = []
    agg_rgb = []
    agg_names = []
    member_name_lists = []

    for k in range(K):
        member_idx = np.where(labels == k)[0]
        member_names = [names[i] for i in member_idx]
        member_flags = [flags[i] for i in member_idx]
        member_keys = keys[member_idx] if member_idx.size else np.zeros((0, 3), dtype=np.int32)
        member_counts = counts[member_idx] if member_idx.size else np.zeros((0,), dtype=int)

        total_count = int(member_counts.sum())
        rgb_agg = _weighted_mean_rgb_uint8(rgb[member_idx], member_counts) if member_idx.size else np.array([0, 0, 0], dtype=np.uint8)
        label = f"{cluster_prefix}{k+1}"

        clusters[k] = {
            'id': k,
            'label': label,
            'member_indices': member_idx.astype(int),
            'member_names': member_names,
            'member_flags': member_flags,
            'member_keys': member_keys,
            'counts': member_counts.astype(int),
            'total_count': total_count,
            'rgb_agg': rgb_agg,
        }

        agg_counts.append(total_count)
        agg_rgb.append(rgb_agg)
        agg_names.append(label)
        member_name_lists.append(member_names)

    agg_counts = np.array(agg_counts, dtype=int)
    agg_rgb = np.array(agg_rgb, dtype=np.uint8)
    percents = (agg_counts.astype(float) / float(total_pixels)) * 100.0 if total_pixels > 0 else np.zeros_like(agg_counts, dtype=float)

    combined_bins_clustered = {
        'total_pixels': total_pixels,
        'counts': agg_counts,
        'rgb': agg_rgb,
        'names': agg_names,
        'percents': percents,
        'member_name_lists': member_name_lists,
    }

    return {
        'clusters': clusters,
        'combined_bins_clustered': combined_bins_clustered,
    }


def plot_cluster_highlight_panels(
    bundle,
    clustering,
    point_size=12,
    size_mode='sqrt',
    annotate_names=True,
    cmap_name='tab10',
    fill_clusters=False,   # set True for translucent fill instead of outline
    alpha_fill=0.15        # transparency for filled polygons
):
    """
    Show both panels (chroma and achro) with ALL clusters highlighted:
      - Base points use original bin colors.
      - Each cluster is outlined (or softly filled) as a convex hull.
      - Optionally annotate member bins with their original bin names and cluster labels.
      - Legend maps cluster label → color.

    Parameters
    ----------
    bundle : dict
        Original analysis bundle.
    clustering : dict
        Result from kmeans_bins_from_bundle(...).
    point_size : int
        Base size for scatter tiles.
    size_mode : str
        Either 'sqrt' (default) or 'linear' for scaling by counts.
    annotate_names : bool
        If True, annotate member squares with name and cluster label.
    cmap_name : str
        Matplotlib colormap name for distinct cluster colors.
    fill_clusters : bool
        If True, fill cluster hulls with translucent color.
    alpha_fill : float
        Opacity for cluster fills (used if fill_clusters=True).
    """

    gb = bundle['groups']
    cb = bundle['combined_bins']
    names = cb['names']
    idx_ch = cb['index_maps']['chroma']
    idx_ac = cb['index_maps']['achro']

    # --- Base data for panels ---
    chroma_coords = gb['chroma']['coords_panel']
    chroma_rgb = gb['chroma']['rgb_rep']
    chroma_counts = gb['chroma']['counts']

    achro_coords = gb['achro']['coords_panel']
    achro_rgb = gb['achro']['rgb_rep']
    achro_counts = gb['achro']['counts']

    # --- Point sizes ---
    if size_mode == 'sqrt':
        sizes_ch = point_size * np.sqrt(np.maximum(chroma_counts, 1)) if chroma_counts.size else None
        sizes_ac = point_size * np.sqrt(np.maximum(achro_counts, 1)) if achro_counts.size else None
    else:
        sizes_ch = point_size * np.maximum(chroma_counts, 1) if chroma_counts.size else None
        sizes_ac = point_size * np.maximum(achro_counts, 1) if achro_counts.size else None

    # --- Cluster info ---
    assignments = clustering['assignments']
    n_clusters = int(clustering['n_clusters'])

    cmap = plt.get_cmap(cmap_name, n_clusters)
    cluster_colors = [cmap(i) for i in range(n_clusters)]
    cluster_labels = [f"C{i+1}" for i in range(n_clusters)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # --- Draw base panels ---
    if chroma_coords.size:
        axes[0].scatter(
            chroma_coords[:, 0], chroma_coords[:, 1],
            c=chroma_rgb / 255.0, s=sizes_ch, marker='s',
            edgecolors='none', alpha=0.85
        )
    axes[0].set_title("Chromatic (a*, b*) — All clusters")
    axes[0].set_aspect('equal')
    axes[0].axis('off')

    if achro_coords.size:
        axes[1].scatter(
            achro_coords[:, 0], achro_coords[:, 1],
            c=achro_rgb / 255.0, s=sizes_ac, marker='s',
            edgecolors='none', alpha=0.85
        )
    axes[1].set_title("Achromatic (L*, C) — All clusters")
    axes[1].set_aspect('auto')
    axes[1].axis('off')

    # --- Overlay cluster hulls ---
    for k in range(n_clusters):
        color = cluster_colors[k]
        label = cluster_labels[k]

        member_combined_idx = np.where(assignments == k)[0]
        if member_combined_idx.size == 0:
            continue

        # Split into chroma/achro slices
        ch_members = member_combined_idx[member_combined_idx < idx_ch.size]
        ac_members = member_combined_idx[member_combined_idx >= idx_ch.size] - idx_ch.size

        ch_bin_idx = idx_ch[ch_members] if ch_members.size else np.array([], dtype=int)
        ac_bin_idx = idx_ac[ac_members] if ac_members.size else np.array([], dtype=int)

        # --- Chromatic convex hull ---
        if ch_bin_idx.size > 2:
            pts = chroma_coords[ch_bin_idx]
            hull = ConvexHull(pts)
            polygon = Polygon(
                pts[hull.vertices],
                closed=True,
                fill=fill_clusters,
                facecolor=color if fill_clusters else 'none',
                alpha=alpha_fill if fill_clusters else 1.0,
                edgecolor=color,
                linewidth=2
            )
            axes[0].add_patch(polygon)
        elif ch_bin_idx.size > 0:
            axes[0].scatter(chroma_coords[ch_bin_idx, 0],
                            chroma_coords[ch_bin_idx, 1],
                            facecolors='none', edgecolors=[color],
                            s=point_size * 20, linewidths=1.8, marker='s')

        # --- Achromatic convex hull ---
        if ac_bin_idx.size > 2:
            pts = achro_coords[ac_bin_idx]
            hull = ConvexHull(pts)
            polygon = Polygon(
                pts[hull.vertices],
                closed=True,
                fill=fill_clusters,
                facecolor=color if fill_clusters else 'none',
                alpha=alpha_fill if fill_clusters else 1.0,
                edgecolor=color,
                linewidth=2
            )
            axes[1].add_patch(polygon)
        elif ac_bin_idx.size > 0:
            axes[1].scatter(achro_coords[ac_bin_idx, 0],
                            achro_coords[ac_bin_idx, 1],
                            facecolors='none', edgecolors=[color],
                            s=point_size * 20, linewidths=1.8, marker='s')

        # --- Optional annotations ---
        if annotate_names:
            for local_i, comb_i in zip(ch_bin_idx, ch_members):
                nm = names[int(comb_i)]
                x, y = chroma_coords[local_i]
                axes[0].text(x, y, f"{nm}({label})", color=color,
                             fontsize=8, ha='center', va='center')
            for local_i, comb_i in zip(ac_bin_idx, ac_members + idx_ch.size):
                nm = names[int(comb_i)]
                x, y = achro_coords[local_i]
                axes[1].text(x, y, f"{nm}({label})", color=color,
                             fontsize=8, ha='center', va='center')

    # --- Build matching legend ---
    handles = [
        Line2D(
            [0], [0],
            marker='s', color='none', markerfacecolor='none',
            markeredgecolor=cluster_colors[i],
            markeredgewidth=2, markersize=10,
            label=cluster_labels[i]
        )
        for i in range(n_clusters)
    ]

    axes[1].legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=False,
        title="Clusters"
    )

    plt.tight_layout()
    plt.show()



def plot_clusters_combined_pie(bundle, cluster_struct, show_labels=True, min_label_percent=0.2, font_size=9):
    """
    Plot a cluster-aggregated pie (each slice is a cluster).
    Adds a legend listing which original bin names belong to each cluster.
    """
    cbk = cluster_struct['combined_bins_clustered']
    counts = cbk['counts']
    if counts.size == 0:
        print("No clusters to plot.")
        return

    colors = cbk['rgb'] / 255.0
    perc = cbk['percents']
    names = cbk['names']
    member_name_lists = cbk['member_name_lists']

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(
        perc,
        colors=colors,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(linewidth=0.5, edgecolor='white')
    )
    ax.set_title("Cluster-aggregated pixel share")

    # Annotate big-enough slices
    if show_labels:
        for w, p, nm in zip(wedges, perc, names):
            if p < min_label_percent:
                continue
            ang = 0.5 * (w.theta1 + w.theta2)
            ang_rad = np.deg2rad(ang)
            x, y = np.cos(ang_rad), np.sin(ang_rad)
            ax.annotate(
                f"{nm} {p:.2f}%",
                xy=(x, y),
                xytext=(1.15 * x, 1.15 * y),
                ha='left' if x >= 0 else 'right',
                va='center',
                fontsize=font_size,
                textcoords='data',
                arrowprops=dict(arrowstyle='-', lw=0.8, color='0.3', shrinkA=0, shrinkB=0)
            )

    # ✅ Build proper legend handles for all clusters
    handles = [
        Patch(facecolor=colors[i], edgecolor='none', label=f"{names[i]}")
        for i in range(len(names))
    ]

    ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=font_size,
        frameon=False,
        title="Clusters"
    )

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def update_bundle_with_clustering(bundle, cluster_struct):
    """
    Return a shallow-copied bundle augmented with clustering info, without
    altering existing fields used by debug_classify...().
    """
    new_bundle = dict(bundle)
    new_bundle['clustering'] = {
        'assignments': cluster_struct['clusters'],  # detailed per-cluster dicts
        'combined_bins_clustered': cluster_struct['combined_bins_clustered']
    }
    return new_bundle

import numpy as np

def build_cluster_label_maps(updated_bundle):
    """
    Build maps:
      - cluster_label -> list of member combined indices
      - cluster_label -> list of member original bin names
    Requires updated_bundle produced by run_kmeans_and_plot_from_bundle(...)
    """
    if 'clustering' not in updated_bundle:
        raise ValueError("No clustering found in bundle. Run run_kmeans_and_plot_from_bundle(...) first.")
    clusters = updated_bundle['clustering']['assignments']  # per-cluster detail dicts
    cb = updated_bundle['combined_bins']

    label_to_indices = {}
    label_to_names = {}

    for k, info in clusters.items():
        label = info['label']  # e.g., 'C1'
        idxs = info['member_indices']  # combined-bin indices
        names = [cb['names'][i] for i in idxs]
        label_to_indices[label] = idxs
        label_to_names[label] = names

    return label_to_indices, label_to_names


def fast_classify_pixels_by_cluster(
    img,
    updated_bundle,
    cluster_label,
    visualize="highlight",
    L_thresh=None,
    C_thresh=None,
    ab_step=None,
    n_jobs=-1,
):
    """
    Fast replacement for debug_classify_pixels_by_cluster.

    Converts the image to Lab once, computes all member-bin masks
    in parallel, aggregates them, and visualizes.
    """

    # Resolve cluster membership
    label_to_indices, label_to_names = build_cluster_label_maps(updated_bundle)
    if cluster_label not in label_to_indices:
        raise KeyError(f"Unknown cluster label: {cluster_label}")

    member_names = label_to_names[cluster_label]
    combined_bins = updated_bundle["combined_bins"]
    name_to_index = updated_bundle["name_to_index"]

    # Parameters
    params = updated_bundle["params"]
    L_t = L_thresh if L_thresh is not None else params["L_thresh"]
    C_t = C_thresh if C_thresh is not None else params["C_thresh"]
    step = ab_step if ab_step is not None else params["ab_step"]

    # Convert entire image to Lab ONCE
    W, H = img.size
    rgb = np.array(img, dtype=np.uint8).reshape(-1, 3)
    lab_flat = rgb_to_lab(rgb)

    # Prepare per-bin masks in parallel
    def mask_for_one_bin(name):
        bin_spec = get_bin_spec(combined_bins, name_to_index, name)
        res = classify_image_pixels_to_bin(
            lab=lab_flat,
            precomputed_lab=True,
            bin_spec=bin_spec,
            L_thresh=L_t,
            C_thresh=C_t,
            ab_step=step,
        )
        return res["mask"].astype(bool).ravel()

    # Parallel or sequential mask collection
    if len(member_names) > 1:
        masks_flat = Parallel(n_jobs=n_jobs)(
            delayed(mask_for_one_bin)(nm) for nm in member_names
        )
    else:
        masks_flat = [mask_for_one_bin(member_names[0])]

    masks_flat = np.stack(masks_flat, axis=0)

    # Aggregate with logical OR (vectorized)
    agg_mask_flat = np.any(masks_flat, axis=0)
    agg_mask = agg_mask_flat.reshape(H, W)
    total_count = int(np.count_nonzero(agg_mask))
    percent = np.round(total_count / updated_bundle.get('total_pixels') * 100, 2)

    # Optional visualization
    if visualize in ("highlight", "mask"):
        title = f"Cluster {cluster_label} — {total_count} px / {percent} %"
        visualize_bin_mask(img, agg_mask, mode=visualize, title=title)

    return {
        "mask": agg_mask,
        "count": total_count,
        "percent": percent,
        "size": (H, W),
        "cluster_label": cluster_label,
        "member_bins": member_names,
    }

# ---------------------------
# Example driver (using your bundle)
# ---------------------------

def run_kmeans_and_plot_from_bundle(bundle, n_clusters=4, show_plots=True, random_state=0):
    """
    - Run k-means on existing bins
    - Build aggregation
    - Plot the first cluster highlighted on panels
    - Plot cluster-aggregated pie with legend
    - Return updated bundle that includes clustering (non-breaking for debug function)
    """
    clustering = kmeans_bins_from_bundle(bundle, n_clusters=n_clusters, random_state=random_state, normalize=True)
    cluster_struct = build_cluster_structures(bundle, clustering, cluster_prefix="C")

    if show_plots is True:
        # Plot: highlight first cluster (Cluster 1) over both panels
        plot_cluster_highlight_panels(bundle, clustering, point_size=bundle['params'].get('point_size', 12), annotate_names=False)

        # Plot: cluster-aggregated pie with legend listing member bin names
        plot_clusters_combined_pie(bundle, cluster_struct, show_labels=True, min_label_percent=0.2, font_size=10)

    # Update bundle (non-destructive)
    updated_bundle = update_bundle_with_clustering(bundle, cluster_struct)
    return updated_bundle


# In[30]:


def k_color_analysis(
    ref_image, 
    k,
    L_thresh=30.0,
    C_thresh=0.1,
    ab_step=1.0,
    point_size=12,
    size_mode='sqrt',
    top_n_chroma=None,
    top_n_achro=None,
    pie_show_labels=True,
    show_input=True,
    show_plots_initial=False,
    show_plots_final=True,
    random_seed = 42,
    shrink_img = 0.1
):

    rgb_pixels, lab, img = compute_lab_from_image_path(ref_image, shrink_img)

    bundle = run_color_analysis_split(
        ref_image,
        L_thresh=L_thresh,
        C_thresh=C_thresh,
        ab_step=ab_step,
        point_size=point_size,
        size_mode=size_mode,
        top_n_chroma=top_n_chroma,
        top_n_achro=top_n_achro,
        pie_show_labels=pie_show_labels,
        show_input=show_input,
        show_plots= show_plots_initial,
        shrink_img=shrink_img
    )
    updated_bundle= run_kmeans_and_plot_from_bundle(bundle, n_clusters=k, show_plots=show_plots_final,random_state=random_seed)

    return updated_bundle, img


# In[31]:


def visualize_color_cluster(image,cluster: str, bundle, visualization = 'highlight'):
    _ = fast_classify_pixels_by_cluster(
        img=image,
        updated_bundle=bundle,
        cluster_label=cluster,
        visualize=visualization
    )


# In[ ]:




