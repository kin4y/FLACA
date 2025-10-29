#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# This program processes and compares two user-provided images — representing conditions before and after — in order to detect and analyze the presence of surface stains. The current implementation is optimized for identifying dark or black stains, developed specifically in the context of analyzing discolorations on black plastic surfaces at the archaeological site of El Tajín, Veracruz, Mexico.
# 
# To operate the program, users should follow the provided instructions carefully. Fields or code cells labeled with the prefix “DEV” (denoting development) contain the core Python code necessary for program execution. Modifying these sections may lead to runtime errors or loss of functionality.
# 
# ### Disclaimer:
# This software is an open-source, prototypical tool designed for exploratory and educational purposes. It should not be used for formal scientific analysis or publication without verification and oversight by qualified conservation or imaging professionals.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, color
from IPython.display import Markdown, display
import imutils
from typing import Any, Dict, Tuple


# In[30]:


# DEV - Optional Color Adjustment with internal ROI mask
def match_colors(src_lab: np.ndarray,
                 ref_lab: np.ndarray,
                 roi_points: np.ndarray) -> np.ndarray:
    """
    Match the color distribution of a source image to a reference image
    within a polygonal ROI.

    Parameters
    ----------
    src_lab : np.ndarray
        Source image in Lab color space (float64).
    ref_lab : np.ndarray
        Reference image in Lab color space (float64).
    roi_points : np.ndarray
        Polygon vertices defining the region of interest in (x, y) format.

    Returns
    -------
    matched_rgb : np.ndarray
        RGB image (uint8) with colors adjusted inside the ROI and preserved outside.
    """

    # --- Create ROI mask ---
    pts = np.array(roi_points, dtype=np.int32).reshape((-1, 2))
    mask = np.zeros(src_lab.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    roi_mask = mask > 0

    # --- Apply histogram matching only inside ROI ---
    matched_lab = src_lab.copy()
    matched_roi = exposure.match_histograms(
        src_lab[roi_mask],
        ref_lab[roi_mask],
        channel_axis=-1
    )
    matched_lab[roi_mask] = matched_roi

    # --- Convert back to RGB ---
    matched_rgb = (np.clip(color.lab2rgb(matched_lab), 0, 1) * 255).astype(np.uint8)
    return matched_rgb


def increase_contrast(src_lab: np.ndarray,
                      roi_points: np.ndarray,
                      clipLimit: float = 3.0,
                      tileGridSize: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Enhance local contrast within a polygonal ROI using CLAHE on the L-channel.

    Parameters
    ----------
    src_lab : np.ndarray
        Source image in Lab color space (float64).
    roi_points : np.ndarray
        Polygon vertices defining the region of interest in (x, y) format.
    clipLimit : float, optional
        CLAHE clip limit (default 3.0).
    tileGridSize : tuple[int,int], optional
        CLAHE tile grid size (default (8,8)).

    Returns
    -------
    enhanced_rgb : np.ndarray
        RGB image (uint8) with enhanced contrast inside the ROI.
    """

    # --- Create ROI mask ---
    pts = np.array(roi_points, dtype=np.int32).reshape((-1, 2))
    mask = np.zeros(src_lab.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # --- Convert Lab to RGB (uint8) for OpenCV ---
    rgb_uint8 = (np.clip(color.lab2rgb(src_lab), 0, 1) * 255).astype(np.uint8)
    lab_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)

    # --- Split channels ---
    l_channel, a_channel, b_channel = cv2.split(lab_uint8)

    # --- Apply CLAHE on L-channel ---
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    l_clahe = clahe.apply(l_channel)

    # --- Replace only inside ROI ---
    l_new = l_channel.copy()
    l_new[mask > 0] = l_clahe[mask > 0]

    # --- Merge channels and convert back to RGB ---
    lab_new = cv2.merge((l_new, a_channel, b_channel))
    enhanced_rgb = cv2.cvtColor(lab_new, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


# In[31]:


# DEV - Auto image Alignment 
# CODE TAKEN FROM https://github.com/itsabk/PixCompare with adjustments by Yanik Wettstein

def align_images(im1, im2, detector, auto_crop=False):
    """
    Aligns two images using feature detection and homography.

    Args:
        im1 (numpy.ndarray): The first image to be aligned.
        im2 (numpy.ndarray): The second image to which the first image is aligned.
        detector: The feature detection method to use.
        auto_crop (bool, optional): Whether to automatically crop the black borders introduced during alignment. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - aligned_image (numpy.ndarray): The aligned version of the first image.
            - homography (numpy.ndarray): The homography matrix used for the alignment.
            - success (bool): A boolean indicating whether the alignment was successful.
            - crop_coords (tuple): A tuple containing the cropping coordinates (x_min, x_max, y_min, y_max) if auto_crop is True, otherwise None.
    """
    im1_gray, im2_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = detector.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = detector.detectAndCompute(im2_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2 if descriptors1.dtype == np.float32 else cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        h, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        height, width = im2.shape[:2]
        im1_aligned = cv2.warpPerspective(im1, h, (width, height))

        if auto_crop:
            mask = cv2.cvtColor(im1_aligned, cv2.COLOR_BGR2GRAY) > 0
            coords = np.argwhere(mask)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            return im1_aligned, h, True, (x_min, x_max, y_min, y_max)
        else:
            return im1_aligned, h, True, None
    else:
        print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
        return im1, None, False, None


def compare_images(
    image1,
    image2,
    method='AKAZE',
    align=False,
    auto_crop=False,
    sensitivity_threshold=45,
    blur_value=(21, 21)
):
    """
    Compare two already-loaded RGB images, optionally align them using feature-based methods,
    and highlight differences based on a sensitivity threshold.

    This function is designed for visual comparison and alignment tasks, such as detecting 
    changes or differences between two archaeological site photos, restoration stages, or 
    temporal image sets.

    Parameters
    ----------
    image1 : np.ndarray
        First input image (already loaded in RGB format).
    image2 : np.ndarray
        Second input image (already loaded in RGB format).
    method : str, optional
        Feature detection algorithm to use for image alignment.
        Supported values include:
        'SIFT', 'BRISK', 'AKAZE', 'KAZE', 'BRIEF', 'FREAK', 
        'LATCH', 'LUCID', 'DAISY', 'ORB'.
        Default is 'AKAZE'.
    align : bool, optional
        Whether to align `image1` to `image2` before comparing. 
        Alignment uses feature matching and homography estimation. 
        Default is False.
    auto_crop : bool, optional
        If True, automatically crop black borders introduced by alignment.
        Default is False.
    sensitivity_threshold : int, optional
        Threshold value controlling how strongly differences are highlighted. 
        Lower values increase sensitivity. Default is 45.
    blur_value : tuple(int, int), optional
        Kernel size for Gaussian blurring applied before thresholding to 
        reduce noise. Default is (21, 21).

    Returns
    -------
    tuple or None
        If `align` is True and alignment succeeds:
            (aligned_image1, image2, homography_matrix)
        Otherwise:
            None (no return, can be extended to visualization or difference output).

    Raises
    ------
    ValueError
        If an invalid feature detection `method` is specified.
    Exception
        Propagates errors from image alignment or OpenCV operations.

    Examples
    --------
    >>> img1 = load_image("before_restoration.jpg")
    >>> img2 = load_image("after_restoration.jpg")
    >>> aligned1, aligned2, H = compare_images(img1, img2, align=True)
    >>> print(H)
    """

    try:
        # --- Validate input images ---
        if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
            raise TypeError("Both image1 and image2 must be NumPy arrays (RGB images).")

        # --- Initialize feature detection methods ---
        method_dict = {
            'SIFT': cv2.SIFT_create(),
            'BRISK': cv2.BRISK_create(),
            'AKAZE': cv2.AKAZE_create(),
            'KAZE': cv2.KAZE_create(),
            'BRIEF': cv2.xfeatures2d.BriefDescriptorExtractor_create(),
            'FREAK': cv2.xfeatures2d.FREAK_create(),
            'LATCH': cv2.xfeatures2d.LATCH_create(),
            'LUCID': cv2.xfeatures2d.LUCID_create(),
            'DAISY': cv2.xfeatures2d.DAISY_create(),
            'ORB': cv2.ORB_create(5000)
        }

        detector = method_dict.get(method)
        if detector is None:
            raise ValueError(
                "Invalid feature detection method. Choose from: "
                "SIFT, BRISK, AKAZE, KAZE, BRIEF, FREAK, LATCH, LUCID, DAISY, ORB."
            )

        # --- Optional image alignment ---
        if align:
            image1_aligned, H, alignment_success, crop_coords = align_images(
                image1, image2, detector=detector, auto_crop=auto_crop
            )
            if alignment_success:
                return image1_aligned, image2, H

    except Exception as e:
        print(f"⚠️ Error during comparison: {str(e)}")
        raise


# In[32]:


# DEV - AUXILIARY FUNCTIONS
# ==========================================

def get_points_cv(img: np.ndarray, n_points: int = 4, window_name: str = "Select points") -> np.ndarray:
    """
    Interactive point selection using OpenCV.

    Opens a window showing the image and allows the user to click on `n_points` locations.
    The function waits until all points are selected and returns them as an array.

    Parameters
    ----------
    img : np.ndarray
        Input image (RGB or BGR).
    n_points : int, optional
        Number of points to select (default is 4).
    window_name : str, optional
        Name of the OpenCV window (default is "Select points").

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) containing the selected points as (x, y) coordinates.
    """
    global clicked_points
    clicked_points = []
    clone = img.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, param)

    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, click_event, clone)
    print(f"Please click {n_points} points on the image '{window_name}'.")

    while len(clicked_points) < n_points:
        cv2.waitKey(100)

    return np.array(clicked_points, dtype=np.float32)


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and convert it to RGB.

    Parameters
    ----------
    path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        The loaded RGB image.

    Raises
    ------
    FileNotFoundError
        If the image could not be loaded.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"❌ Could not load {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def warp_image(
    image: np.ndarray,
    src_points: np.ndarray,
    master_size: tuple[int, int],
    dst_points: np.ndarray | None = None,
    fill_value: int | float = 0,
    return_float: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Warp an image into a rectangular master frame using a perspective transform.

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB/BGR).
    src_points : np.ndarray
        Four source points in the image (top-left, top-right, bottom-right, bottom-left).
    master_size : tuple[int, int]
        Size (width, height) of the output master frame.
    dst_points : np.ndarray, optional
        Four destination points in the master frame. If None, defaults to a rectangle
        inset by 10% margin.
    fill_value : int or float, optional
        Value used for pixels outside the image boundaries (default 0).
    return_float : bool, optional
        If True, returns the warped image as float32; otherwise as uint8.

    Returns
    -------
    warped : np.ndarray
        Warped image.
    M : np.ndarray
        Homography matrix from source to destination points.
    M_inv : np.ndarray
        Inverse homography matrix.
    dst_points : np.ndarray
        Destination points used for the warp.
    """
    if dst_points is None:
        margin = min(master_size[0], master_size[1]) * 0.1
        dst_points = np.array([
            [margin, margin],
            [master_size[0] - margin, margin],
            [master_size[0] - margin, master_size[1] - margin],
            [margin, master_size[1] - margin]
        ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points)
    warped = cv2.warpPerspective(
        image.astype(np.float32),
        M,
        master_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_value
    )

    if not return_float:
        warped = np.clip(warped, 0, 255).astype(np.uint8)

    return warped, M, np.linalg.inv(M), dst_points


def color_post_process(A_roi: np.ndarray, B_roi: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform color alignment and contrast enhancement on two images.

    Steps:
    1. Match the colors of B_roi to A_roi within the ROI.
    2. Apply local contrast enhancement (CLAHE) to both images.

    Parameters
    ----------
    A_roi : np.ndarray
        Reference image ROI.
    B_roi : np.ndarray
        Target image ROI to be matched and enhanced.
    roi : np.ndarray
        Polygon defining the region of interest.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the processed A_roi and B_roi.
    """
    B_aligned = match_colors(B_roi, A_roi, roi)
    A_contrast = increase_contrast(A_roi, roi)
    B_contrast = increase_contrast(B_aligned, roi)
    return A_roi, B_aligned


def transform_points(x_points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a homography matrix to a set of 2D points.

    Parameters
    ----------
    x_points : np.ndarray
        Input points, shape (N, 2).
    H : np.ndarray
        3x3 homography matrix.

    Returns
    -------
    np.ndarray
        Transformed points, shape (N, 2).
    """
    pts = np.asarray(x_points, dtype=np.float64).reshape(-1, 1, 2)
    y_pts = cv2.perspectiveTransform(pts, H)
    return np.round(y_pts).astype(np.float32).reshape(-1, 2)


def compute_master_size(roi_raw: np.ndarray) -> tuple[int, int]:
    """
    Compute the width and height of a master rectangle from a quadrilateral ROI.

    The function preserves perspective ratios by averaging opposing sides.

    Parameters
    ----------
    roi_raw : np.ndarray
        Four points of the ROI in order: top-left, top-right, bottom-right, bottom-left.

    Returns
    -------
    tuple[int, int]
        Width and height of the master rectangle.
    """
    pts = np.array(roi_raw, dtype=np.float32)

    width_top = np.linalg.norm(pts[1] - pts[0])
    width_bottom = np.linalg.norm(pts[2] - pts[3])
    height_left = np.linalg.norm(pts[3] - pts[0])
    height_right = np.linalg.norm(pts[2] - pts[1])

    avg_width = (width_top + width_bottom) * 0.5
    avg_height = (height_left + height_right) * 0.5

    return int(round(avg_width)), int(round(avg_height))


def create_evaluation_mask(B_image: np.ndarray,
                           roi_points: np.ndarray,
                           stain_mask: np.ndarray) -> np.ndarray:
    """
    Create an RGB visualization mask showing:
    - ROI in white
    - Detected stains in red on top of ROI
    - Everything else black

    Parameters
    ----------
    B_image : np.ndarray
        Input image (H,W,3), used to get the shape.
    roi_points : np.ndarray
        Polygon vertices defining ROI in (x, y) format.
    stain_mask : np.ndarray
        Binary mask of detected stains (same HxW as B_image).

    Returns
    -------
    mask_rgb : np.ndarray
        RGB mask: black outside ROI, white inside ROI, red for stain pixels.
    """
    # --- Create ROI mask ---
    pts = np.array(roi_points, dtype=np.int32).reshape((-1, 2))
    roi_mask = np.zeros(B_image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [pts], 255)

    # --- Create RGB mask ---
    mask_rgb = np.zeros((*B_image.shape[:2], 3), dtype=np.uint8)  # black everywhere
    mask_rgb[roi_mask > 0] = [255, 255, 255]  # ROI in white
    mask_rgb[stain_mask > 0] = [255, 0, 0]    # Stains in red on top

    return mask_rgb


# In[33]:


# DEV - GEOMETRICAL IMAGE PROCESSING (full-frame)
# ==========================================
def process_images_fullframe(
    A_orig: np.ndarray,
    B_orig: np.ndarray,
    auto_align: bool = False,
    method: str = "AKAZE"
) -> Dict[str, Any]:
    """
    Process two full-frame images: optionally align them, select or compute ROI,
    and warp both images into a common master frame.

    Parameters
    ----------
    A_orig : np.ndarray
        Reference image (RGB) to which B will be aligned.
    B_orig : np.ndarray
        Target image (RGB) to be aligned to A.
    auto_align : bool, optional
        If True, performs automatic alignment using feature matching.
        If False, user manually selects reference points. Default is False.
    method : str, optional
        Feature detection method for auto-alignment (e.g., 'AKAZE', 'ORB').
        Default is "AKAZE".

    Returns
    -------
    dict
        Dictionary containing:
        - 'A_aligned_framed': np.ndarray, A warped to master frame
        - 'B_aligned_framed': np.ndarray, B warped to master frame
        - 'MA': np.ndarray, homography matrix of A
        - 'MB': np.ndarray, homography matrix of B
        - 'invMA': np.ndarray, inverse of A's homography
        - 'invMB': np.ndarray, inverse of B's homography
        - 'A_orig': np.ndarray, original A
        - 'B_orig': np.ndarray, original B
        - 'roi': np.ndarray, destination ROI in master frame
        - 'ptsA': np.ndarray, ROI points in original A
        - 'ptsB': np.ndarray, ROI points in original B
    """
    hA, wA = A_orig.shape[:2]
    hB, wB = B_orig.shape[:2]

    # Initialize variables
    A_aligned = A_orig.copy()
    B_aligned = B_orig.copy()
    roi_raw = None
    master_size = None

    if auto_align:
        # -------------------------------
        # Automatic alignment
        # -------------------------------
        A_aligned, B_aligned, MA_1 = compare_images(
            A_orig, B_orig,
            method=method,
            align=True,
            auto_crop=False,
            sensitivity_threshold=40,
            blur_value=(7, 7)
        )
        print("Images automatically aligned.")
        invMA_1 = np.linalg.inv(MA_1)

        # Select ROI on overlay of aligned images
        overlay = cv2.addWeighted(A_aligned, 0.5, B_aligned, 0.5, 0)
        roi_raw = get_points_cv(overlay, 4, "Select 4 points (ROI) on Overlay")

        # Compute master frame size based on ROI
        master_size = compute_master_size(roi_raw)

        # Warp both images into master frame
        A_aligned_framed, MA_2, invMA, _ = warp_image(A_aligned, roi_raw, master_size)
        B_aligned_framed, MB, invMB, roi = warp_image(B_aligned, roi_raw, master_size)

        # Total homography for A
        MA = MA_2 @ MA_1
        invMA = np.linalg.inv(MA)

        # Original points in unaligned images
        ptsB = roi_raw
        ptsA = transform_points(roi_raw, invMA_1)

    else:
        # -------------------------------
        # Manual point selection
        # -------------------------------
        print("Manual selection of reference points.")

        # Step 1: Select 4 points on A
        ptsA = get_points_cv(A_orig, 4, "Select 4 points on Image A (TL, TR, BR, BL)")

        # Step 2: Select 4 points on B
        ptsB = get_points_cv(B_orig, 4, "Select 4 points on Image B (TL, TR, BR, BL)")

        # Step 3: Determine master frame size from A's ROI
        master_size = compute_master_size(ptsA)

        # Warp images into master frame
        A_aligned_framed, MA, invMA, roi = warp_image(A_orig, ptsA, master_size)
        B_aligned_framed, MB_1, invMB_1, _ = warp_image(B_orig, ptsB, master_size)
        MB = MB_1
        invMB = np.linalg.inv(MB)

    # Close any OpenCV windows
    cv2.destroyAllWindows()

    # Return all relevant data
    return {
        'A_aligned_framed': A_aligned_framed,
        'B_aligned_framed': B_aligned_framed,
        'MA': MA,
        'MB': MB,
        'invMA': invMA,
        'invMB': invMB,
        'A_orig': A_orig,
        'B_orig': B_orig,
        'roi': roi,
        'ptsA': ptsA,
        'ptsB': ptsB
    }


# In[34]:


def detect_black_stains(A_full: np.ndarray,
                                          B_full: np.ndarray,
                                          roi_points: np.ndarray,
                                          dark_thresh: int = 70,
                                          sat_thresh: int = 60,
                                          diff_thresh: int = 25,
                                          fill_value: int | float = 255):
    """
    Detect gray/black stains in B compared to A **only inside ROI**, ignoring colored dark areas and fill regions (NaN).

    Pixels outside the ROI are ignored from all computations. Produces a mask, a visualization overlay,
    and the ratio of stain pixels restricted to the ROI.

    Parameters
    ----------
    A_full : np.ndarray
        Reference RGB image.
    B_full : np.ndarray
        Target RGB image to detect stains.
    roi_points : np.ndarray
        Polygon vertices defining the ROI (x, y) in full-frame coordinates.
    dark_thresh : int
        Value threshold for dark pixels in V channel (default 70).
    sat_thresh : int
        Value threshold for low saturation (grayish) pixels in S channel (default 60).
    diff_thresh : int
        Minimum difference in brightness compared to A to consider as stain (default 25).
    fill_value : int or float
        Value to replace NaNs in the images before processing (default 255).

    Returns
    -------
    stain_mask : np.ndarray
        Binary mask of detected grayish/black stains (uint8, 0/255).
    vis : np.ndarray
        RGB visualization image overlaying detected stains in red.
    ratio_stain : float
        Fraction of ROI pixels detected as stains.
    roi_mask : np.ndarray
        Binary mask of the ROI (uint8, 0/1).
    """

    # --- Ensure polygon ---
    pts = np.array(roi_points, dtype=np.int32).reshape((-1, 2))

    # --- ROI mask ---
    roi_mask = np.zeros(B_full.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [pts], 1)  # 1 inside ROI

    # --- Valid pixels (not NaN) ---
    valid_mask = ~np.isnan(B_full).any(axis=2) if B_full.ndim == 3 else ~np.isnan(B_full)
    process_mask = (roi_mask > 0) & valid_mask

    # --- Safe copies with fill_value ---
    A_safe = np.nan_to_num(A_full, nan=fill_value).astype(np.uint8)
    B_safe = np.nan_to_num(B_full, nan=fill_value).astype(np.uint8)

    # --- Convert to HSV ---
    hsvA = cv2.cvtColor(A_safe, cv2.COLOR_RGB2HSV)
    hsvB = cv2.cvtColor(B_safe, cv2.COLOR_RGB2HSV)
    _, S_B, V_B = cv2.split(hsvB)
    _, _, V_A = cv2.split(hsvA)

    # --- Grayish dark mask ---
    dark_mask = (V_B < dark_thresh).astype(np.uint8)
    gray_mask = (S_B < sat_thresh).astype(np.uint8)
    neutral_dark = cv2.bitwise_and(dark_mask, gray_mask)

    # --- Compare brightness to reference ---
    brightness_diff = cv2.subtract(V_A, V_B)
    _, darker_mask = cv2.threshold(brightness_diff, diff_thresh, 1, cv2.THRESH_BINARY)

    # --- Combine masks restricted to ROI+valid ---
    stain_mask = cv2.bitwise_and(neutral_dark, darker_mask)
    stain_mask[~process_mask] = 0

    # --- Morphological cleanup ---
    kernel = np.ones((1, 1), np.uint8)
    stain_mask = cv2.morphologyEx(stain_mask * 255, cv2.MORPH_OPEN, kernel)
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_DILATE, kernel)

    # --- Visualization overlay ---
    vis = B_safe.copy()
    overlay = vis.copy()
    overlay[stain_mask > 0] = [255, 0, 0]  # red overlay
    vis = cv2.addWeighted(overlay, 0.5, vis, 0.5, 0)

    # --- ROI-restricted ratio ---
    ratio_stain = np.count_nonzero(stain_mask) / np.count_nonzero(roi_mask)

    return stain_mask, vis, ratio_stain, roi_mask


# In[38]:


def detect_stains(pathA,
                  pathB,
                  dark_thresh = 120, # Threshold for detecting black stains
                  sat_thresh = 120, # Threshold for saturation loss
                  diff_thresh = 34, # Threshold for the difference in color in Lab-Space
                  max_height = 1000, # max image height, reduces the images height, preserves ratio
                  fill_value=[255,0,255], # Background fill value in RGB space
                  plot = True):

    picture_before = load_image(pathA)
    picture_after = load_image(pathB)
    geometry = process_images_fullframe(picture_before, picture_after, auto_align=True, method = "AKAZE")
    # DEV - Create Evaluation Mask
    stain_mask, B_stained, ratio,_ = detect_black_stains(
        geometry.get("A_aligned_framed"),
        geometry.get("B_aligned_framed"),
        geometry.get("roi"),
        dark_thresh = 120,
        sat_thresh = 60,
        diff_thresh = 25
    )

    mask_rgb = create_evaluation_mask(
        geometry.get("B_aligned_framed"),
        geometry.get("roi"),
        stain_mask
    )

    mask_image,_,_,_=  warp_image(mask_rgb, 
                                   geometry.get("roi"), 
                                   (geometry.get("B_orig").shape[1],
                                    geometry.get("B_orig").shape[0]), 
                                   dst_points = geometry.get("ptsB"))



    print(f"Red ratio inside ROI: {ratio:.4f}")


    if plot is True: 
        B_image,_,_,_ = warp_image(geometry.get("B_aligned_framed"), 
                                       geometry.get("roi"), 
                                       (geometry.get("B_orig").shape[1],
                                        geometry.get("B_orig").shape[0]), 
                                       dst_points = geometry.get("ptsB"))

        B_stains_image,_,_,_ =  warp_image(B_stained, 
                                       geometry.get("roi"), 
                                       (geometry.get("B_orig").shape[1],
                                        geometry.get("B_orig").shape[0]), 
                                       dst_points = geometry.get("ptsB"))

        A_image,_,_,_=  warp_image(geometry.get("A_aligned_framed"), 
                                       geometry.get("roi"), 
                                       (geometry.get("A_orig").shape[1],
                                        geometry.get("A_orig").shape[0]), 
                                       dst_points = geometry.get("ptsA"))


        # List of images to display: (image array, title, optional colormap)
        images = [
            (B_image, "Original B", None),
            (B_stains_image, "Detected Black Stains", None),
            (A_image, "Original A", None),
            (mask_image, "Stain Mask", "gray")
        ]

        # Use constrained_layout=True for a tight layout
        # A smaller, more typical figsize (e.g., (10, 8)) is usually better than (50, 20)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), constrained_layout=True)
        
        for ax, (img, title, cmap) in zip(axes.flat, images):
            # Handle grayscale vs RGB correctly
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                ax.imshow(img, cmap=cmap if cmap else "gray")
            else:
                # For RGB, ensure proper range and handle NaN
                img_disp = np.nan_to_num(img, nan=0.0)  # replace NaN with 0 for display
                if img_disp.dtype != np.uint8:
                    img_disp = np.clip(img_disp, 0, 255).astype(np.uint8)
                ax.imshow(img_disp)
        
            ax.set_title(title, fontsize=12) # Adjusted font size for smaller figure
            ax.axis("off")
            ax.set_aspect("equal")  # Keep true aspect ratio - this is essential!
        
        # With constrained_layout=True, you typically don't need plt.subplots_adjust or plt.tight_layout()
        plt.show()


