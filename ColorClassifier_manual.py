#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any

# ==========================================
# Configuration & Data Structures
# ==========================================

@dataclass
class AnalysisConfig:
    """Configuration for color segmentation."""
    L_thresh: float = 10.0
    C_thresh: float = 6.0
    ab_step: float = 1.0
    shrink_img: float = 1.0
    n_clusters: int = 4
    random_state: int = 42
    # Feature weights for clustering: 1.0 = full influence, 0.0 = ignored
    cluster_features: Dict[str, float] = field(default_factory=lambda: {'a': 1.0, 'b': 1.0})

# ==========================================
# Core Processing Engine
# ==========================================

class ColorSegmenter:
    """
    Handles image loading, Lab conversion, quantization (binning), 
    and clustering logic. Now supports Alpha-channel masking!
    """
    def __init__(self, config: AnalysisConfig, image_path: Optional[str] = None, raw_rgb: Optional[np.ndarray] = None):
        self.cfg = config
        
        # 1. Load Data
        if raw_rgb is not None:
            self.image_path = "3D_Point_Cloud"
            self.full_rgb_flat = raw_rgb
            self.original_shape = (raw_rgb.shape[0], 1)
            self.is_3d = True
            self.valid_mask = np.ones(raw_rgb.shape[0], dtype=bool) # All points valid
        elif image_path is not None:
            self.image_path = image_path
            self.full_rgb_flat, self.original_shape, self.valid_mask = self._load_image_with_mask()
            self.is_3d = False
        else:
            raise ValueError("Must provide either image_path or raw_rgb.")
            
        # --- THE MASKING MAGIC ---
        # Extract ONLY the pixels that are not transparent
        self.rgb_flat = self.full_rgb_flat[self.valid_mask]
        self.total_valid_pixels = self.rgb_flat.shape[0]
        
        if self.total_valid_pixels == 0:
            raise ValueError("The provided image/mask contains no valid visible pixels to analyze.")

        # 2. Convert ONLY valid pixels to LAB
        self.lab_flat = rgb2lab(self.rgb_flat.astype(float) / 255.0)
        self.L = self.lab_flat[:, 0]
        self.a = self.lab_flat[:, 1]
        self.b = self.lab_flat[:, 2]
        self.C = np.hypot(self.a, self.b)
        
        # 3. Determine Chroma/Achro Masks (only on valid pixels)
        self.is_achro = (self.L < self.cfg.L_thresh) | (self.C < self.cfg.C_thresh)
        self.is_chroma = ~self.is_achro
        
        # 4. Quantize (Binning)
        self._quantize_bins()
        
        # 5. Placeholders for clustering results
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.clusters: Dict[int, Dict[str, Any]] = {}
        
    def _load_image_with_mask(self) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
        """Loads image and explicitly extracts the Alpha channel to create a boolean mask."""
        # Convert to RGBA to guarantee an alpha channel exists
        img = Image.open(self.image_path).convert('RGBA')
        
        if self.cfg.shrink_img != 1.0:
            w, h = img.size
            new_size = (max(1, int(w * self.cfg.shrink_img)), max(1, int(h * self.cfg.shrink_img)))
            img = img.resize(new_size, Image.LANCZOS)
        
        arr = np.array(img, dtype=np.uint8)
        H, W = arr.shape[:2]
        
        # Split RGB and Alpha
        rgb = arr[:, :, :3].reshape(-1, 3)
        alpha = arr[:, :, 3].reshape(-1)
        
        # True if pixel is visible (alpha > 0), False if transparent
        valid_mask = alpha > 0 
        
        return rgb, (H, W), valid_mask

    def _quantize_bins(self) -> None:
        a_bin = np.round(self.a / max(self.cfg.ab_step, 0.001)).astype(np.int32)
        b_bin = np.round(self.b / max(self.cfg.ab_step, 0.001)).astype(np.int32)
        
        group_id = self.is_achro.astype(np.int32)
        raw_keys = np.column_stack([group_id, a_bin, b_bin])
        
        self.unique_keys, self.pixel_to_bin_idx = np.unique(raw_keys, axis=0, return_inverse=True)
        self.n_bins = self.unique_keys.shape[0]
        
        self.bin_counts = np.bincount(self.pixel_to_bin_idx, minlength=self.n_bins)
        
        L_sum = np.bincount(self.pixel_to_bin_idx, weights=self.L)
        a_sum = np.bincount(self.pixel_to_bin_idx, weights=self.a)
        b_sum = np.bincount(self.pixel_to_bin_idx, weights=self.b)
        
        safe_counts = np.maximum(self.bin_counts, 1)
        
        self.bin_L = L_sum / safe_counts
        self.bin_a = a_sum / safe_counts
        self.bin_b = b_sum / safe_counts
        self.bin_C = np.hypot(self.bin_a, self.bin_b)
        
        self.bin_lab = np.column_stack([self.bin_L, self.bin_a, self.bin_b])
        self.bin_rgb = self._lab_to_rgb_u8(self.bin_lab)
        
        self.bin_is_achro = self.unique_keys[:, 0] == 1
        self.bin_names = self._generate_names(self.n_bins)

    def _generate_names(self, n: int) -> np.ndarray:
        names = []
        for i in range(n):
            letter = chr(ord('A') + (i % 26))
            cycle = i // 26
            name = letter if cycle == 0 else f"{letter}{cycle}"
            names.append(name)
        return np.array(names)

    def _lab_to_rgb_u8(self, lab: np.ndarray) -> np.ndarray:
        rgb = lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    def run_clustering(self) -> None:
        features = []
        
        w_L = self.cfg.cluster_features.get('L', 0.0)
        if w_L > 0: features.append(self.bin_L * w_L)
            
        w_a = self.cfg.cluster_features.get('a', 0.0)
        if w_a > 0: features.append(self.bin_a * w_a)
            
        w_b = self.cfg.cluster_features.get('b', 0.0)
        if w_b > 0: features.append(self.bin_b * w_b)
            
        w_C = self.cfg.cluster_features.get('C', 0.0)
        if w_C > 0: features.append(self.bin_C * w_C)
            
        w_h = self.cfg.cluster_features.get('h', 0.0)
        if w_h > 0:
            theta = np.arctan2(self.bin_b, self.bin_a)
            features.append(theta * w_h * 10.0)

        if not features:
            raise ValueError("No features selected for clustering.")
            
        X = np.column_stack(features)
        
        # --- PREVENT CRASH IF n_samples < k ---
        n_samples = X.shape[0]
        if n_samples < self.cfg.n_clusters:
            print(f"⚠️ Warning: Found only {n_samples} unique color bins. Reducing k to {n_samples}.")
            self.cfg.n_clusters = max(1, n_samples)
            
        mu = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_norm = (X - mu) / std
        
        kmeans = KMeans(n_clusters=self.cfg.n_clusters, random_state=self.cfg.random_state, n_init=10)
        self.cluster_labels = kmeans.fit_predict(X_norm, sample_weight=self.bin_counts)
        self.cluster_centers = kmeans.cluster_centers_ 
        
        self.clusters = {}
        for k in range(self.cfg.n_clusters):
            mask = (self.cluster_labels == k)
            count = self.bin_counts[mask].sum()
            
            if count > 0:
                members_rgb = self.bin_rgb[mask].astype(float)
                members_w = self.bin_counts[mask][:, None]
                avg_rgb = np.sum(members_rgb * members_w, axis=0) / count
            else:
                avg_rgb = np.array([0,0,0])
                
            self.clusters[k] = {
                'label': f"C{k+1}",
                'count': count,
                # Percentage is now accurately based ONLY on the valid archaeological surface!
                'percent': (count / self.total_valid_pixels) * 100, 
                'rgb': avg_rgb.astype(np.uint8),
                'member_bins': self.bin_names[mask].tolist(),
                'member_indices': np.where(mask)[0]
            }

    def get_cluster_mask(self, cluster_id_or_label: Union[int, str]) -> Dict[str, Any]:
        target_id = -1
        if isinstance(cluster_id_or_label, str):
            for k, v in self.clusters.items():
                if v['label'] == cluster_id_or_label:
                    target_id = k
                    break
        else:
            target_id = cluster_id_or_label

        if target_id not in self.clusters:
            raise KeyError(f"Cluster {cluster_id_or_label} not found.")

        # Reconstruct the boolean mask for ONLY the valid pixels
        valid_pixel_cluster_ids = self.cluster_labels[self.pixel_to_bin_idx]
        valid_mask_flat = (valid_pixel_cluster_ids == target_id)
        
        # Map the valid pixels BACK to the full original image size
        full_mask_flat = np.zeros(self.full_rgb_flat.shape[0], dtype=bool)
        full_mask_flat[self.valid_mask] = valid_mask_flat
        
        mask = full_mask_flat if self.is_3d else full_mask_flat.reshape(self.original_shape)
        
        return {
            'mask': mask,
            'count': self.clusters[target_id]['count'],
            'percent': self.clusters[target_id]['percent'],
            'rgb': self.clusters[target_id]['rgb']
        }

# ==========================================
# Visualization Engine
# ==========================================

class ColorVisualizer:
    def __init__(self, segmenter: ColorSegmenter):
        self.seg = segmenter

    def plot_panels(self, highlight_clusters: bool = True, title_suffix: str = "") -> None:
        s = self.seg
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        idx_chr = np.where(~s.bin_is_achro)[0]
        idx_ach = np.where(s.bin_is_achro)[0]
        sizes = 12 * np.sqrt(np.maximum(s.bin_counts, 1))

        ax = axes[0]
        if len(idx_chr) > 0:
            ax.scatter(s.bin_a[idx_chr], s.bin_b[idx_chr], c=s.bin_rgb[idx_chr]/255.0, s=sizes[idx_chr], marker='s', edgecolors='none', alpha=0.9)
            if highlight_clusters and s.cluster_labels is not None:
                self._draw_hulls(ax, idx_chr, x_data=s.bin_a, y_data=s.bin_b)
        ax.set_title(f"Chromatic (a*, b*) {title_suffix}")
        ax.set_aspect('equal')
        ax.axis('off')

        ax = axes[1]
        if len(idx_ach) > 0:
            ax.scatter(s.bin_L[idx_ach], s.bin_C[idx_ach], c=s.bin_rgb[idx_ach]/255.0, s=sizes[idx_ach], marker='s', edgecolors='none', alpha=0.9)
            if highlight_clusters and s.cluster_labels is not None:
                self._draw_hulls(ax, idx_ach, x_data=s.bin_L, y_data=s.bin_C)
        ax.set_title(f"Achromatic (L*, C) {title_suffix}")
        ax.axis('off')
        
        if highlight_clusters and s.clusters:
            self._add_cluster_legend(ax)

        plt.tight_layout()
        plt.show()

    def _draw_hulls(self, ax: plt.Axes, subset_indices: np.ndarray, x_data: np.ndarray, y_data: np.ndarray) -> None:
        cmap = plt.get_cmap('tab10', self.seg.cfg.n_clusters)
        for k in range(self.seg.cfg.n_clusters):
            cluster_mask = (self.seg.cluster_labels == k)
            relevant_indices = np.intersect1d(np.where(cluster_mask)[0], subset_indices)
            if len(relevant_indices) < 3: continue
            pts = np.column_stack([x_data[relevant_indices], y_data[relevant_indices]])
            try:
                hull = ConvexHull(pts)
                color = cmap(k)
                poly = Polygon(pts[hull.vertices], closed=True, edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
                ax.add_patch(poly)
            except Exception: pass

    def _add_cluster_legend(self, ax: plt.Axes) -> None:
        cmap = plt.get_cmap('tab10', self.seg.cfg.n_clusters)
        handles = [Patch(facecolor=cmap(i), edgecolor='none', label=f"C{i+1}") for i in range(self.seg.cfg.n_clusters)]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, title="Clusters")

    def plot_pie(self, by: str = 'cluster', min_percent: float = 0.5) -> None:
        if by == 'cluster':
            data = [self.seg.clusters[k] for k in range(self.seg.cfg.n_clusters)]
            data.sort(key=lambda x: x['count'], reverse=True)
            counts = [x['count'] for x in data]
            colors = [x['rgb']/255.0 for x in data]
            labels = [x['label'] for x in data]
            percents = [x['percent'] for x in data]
            title = "Cluster Distribution"
        else:
            idx = np.argsort(self.seg.bin_counts)[::-1]
            counts = self.seg.bin_counts[idx]
            colors = self.seg.bin_rgb[idx] / 255.0
            labels = self.seg.bin_names[idx]
            percents = (counts / self.seg.total_valid_pixels) * 100
            title = "Bin Distribution"

        fig, ax = plt.subplots(figsize=(8, 8))
        if len(counts) == 0 or sum(counts) == 0: return 
            
        wedges, _ = ax.pie(counts, colors=colors, startangle=90, counterclock=False, wedgeprops=dict(width=1, edgecolor='white', linewidth=0.5))
        
        for w, p, l in zip(wedges, percents, labels):
            if p < min_percent: continue
            ang = (w.theta2 + w.theta1) / 2.0
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            ha = "left" if x > 0 else "right"
            ax.annotate(f"{l}\n{p:.1f}%", xy=(x, y), xytext=(1.15*x, 1.15*y), ha=ha, va="center", arrowprops=dict(arrowstyle="-", color="0.5"))
            
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def visualize_mask(self, mask: np.ndarray, title: str = "Mask") -> None:
        if self.seg.is_3d: return
        img = self.seg.full_rgb_flat.reshape(self.seg.original_shape + (3,))
        dimmed = (img * 0.15).astype(np.uint8)
        out = dimmed.copy()
        out[mask] = img[mask]
        
        plt.figure(figsize=(6, 6))
        plt.imshow(out)
        plt.title(title)
        plt.axis('off')
        plt.show()

# ==========================================
# Legacy API Wrapper
# ==========================================

def k_color_analysis(
    ref_image: Optional[str] = None, 
    raw_rgb: Optional[np.ndarray] = None,
    k: int = 5,
    L_thresh: float = 30.0,
    C_thresh: float = 0.1,
    ab_step: float = 1.0,
    cluster_features: Dict[str, float] = None,
    point_size: int = 12,
    size_mode: str = 'sqrt',
    top_n_chroma: Optional[int] = None,
    top_n_achro: Optional[int] = None,
    pie_show_labels: bool = True,
    show_input: bool = True,
    show_plots_initial: bool = False,
    show_plots_final: bool = True,
    random_seed: int = 42,
    shrink_img: float = 1.0
) -> Tuple[ColorSegmenter, Any]:
    
    if cluster_features is None:
        cluster_features = {'a': 1.0, 'b': 1.0}
        
    cfg = AnalysisConfig(
        L_thresh=L_thresh, C_thresh=C_thresh, ab_step=ab_step,
        shrink_img=shrink_img, n_clusters=k, random_state=random_seed,
        cluster_features=cluster_features
    )
    
    # Init correctly mapping arguments
    seg = ColorSegmenter(config=cfg, image_path=ref_image, raw_rgb=raw_rgb)
    viz = ColorVisualizer(seg)
    
    if show_input and not seg.is_3d:
        plt.figure(figsize=(6,6))
        plt.imshow(Image.open(ref_image))
        plt.title("Input Image")
        plt.axis('off')
        plt.show()

    if show_plots_initial:
        viz.plot_panels(highlight_clusters=False, title_suffix="(Bins)")
        viz.plot_pie(by='bin')

    seg.run_clustering()

    if show_plots_final:
        viz.plot_panels(highlight_clusters=True, title_suffix="(Clustered)")
        viz.plot_pie(by='cluster')
    
    return_img = None
    if not seg.is_3d:
        return_img = Image.open(ref_image)
        if shrink_img != 1.0:
            w, h = return_img.size
            return_img = return_img.resize((max(1, int(w*shrink_img)), max(1, int(h*shrink_img))))
            
    return seg, return_img

def visualize_color_cluster(image: np.ndarray, cluster: str, bundle: Any, visualization: str = 'highlight') -> None:
    if not isinstance(bundle, ColorSegmenter):
        print("Error: Bundle is not a ColorSegmenter instance.")
        return

    res = bundle.get_cluster_mask(cluster)
    viz = ColorVisualizer(bundle)
    
    if visualization == 'highlight':
        viz.visualize_mask(res['mask'], title=f"Cluster {cluster}")
    elif visualization == 'mask':
        if not bundle.is_3d:
            plt.imshow(res['mask'], cmap='gray')
            plt.show()