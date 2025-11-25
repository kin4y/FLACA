#!/usr/bin/env python3
"""
Interactive polygon cutter with OpenCV.

Controls (mouse / keyboard):
 - Left click: add point to current polygon
 - Right click: remove last point from current polygon
 - Enter or 'c': complete current polygon (if >= 3 points) and start a new one
 - 'n': start a new empty polygon (abandons current points if any)
 - 'z': undo last completed polygon
 - 'x': delete polygon under mouse (click mouse to position, then press 'x')
 - 'b': cycle through background swatches
 - 't': toggle showing filled masks overlay
 - 'a': export PNG with alpha channel (transparent outside)
 - 'f': export flattened PNG with chosen background color
 - 'q' or ESC: quit
 - 'h': print this help
"""

import cv2
import numpy as np
import os
from tkinter import Tk, filedialog

# -----------------------
# Helper: pixel-perfect mask + export
# -----------------------
def cut_polygons_from_image(image_bgr, polygons, background=None, export_alpha=True):
    """
    image_bgr: HxWx3 uint8 BGR
    polygons: list of polygons, each is Nx2 int (x,y)
    background: None or (B,G,R) tuple for flattened export
    export_alpha: if True produce image with alpha (HxWx4 PNG); if False, return flattened BGR
    returns: result image (uint8). If export_alpha True -> HxWx4 BGRA, else HxWx3 BGR
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_list = []
    for poly in polygons:
        if len(poly) >= 3:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            pts_list.append(pts)
    if pts_list:
        cv2.fillPoly(mask, pts_list, 255)  # pixel-perfect binary mask, no anti-alias
    # create BGRA
    b, g, r = cv2.split(image_bgr)
    alpha = mask.copy()
    bgra = cv2.merge((b, g, r, alpha))
    if export_alpha:
        return bgra
    else:
        # flatten onto background color
        if background is None:
            background = (255, 255, 255)  # white default (B,G,R)
        bg_img = np.zeros_like(image_bgr, dtype=np.uint8)
        bg_img[:, :] = background
        # where mask==255 take original, else background
        mask_bool = mask.astype(bool)
        out = bg_img.copy()
        out[mask_bool] = image_bgr[mask_bool]
        return out

# -----------------------
# Interactive OpenCV UI
# -----------------------
class PolygonEditor:
    def __init__(self, img):
        self.img_orig = img.copy()
        self.h, self.w = img.shape[:2]
        self.polygons = []  # list of lists of (x,y)
        self.current = []   # current building polygon
        self.active_bg_idx = 0
        self.bg_swatches = [
            (255,255,255), # white
            (0,0,0),       # black
            (127,127,127), # gray
            (0,0,255),     # red (BGR)
            (0,255,0),     # green
            (255,0,0),     # blue
            (0,255,255),   # yellow
            (255,0,255),   # magenta
        ]
        self.show_mask_overlay = True
        self.mouse_pos = (0,0)
        self.window_name = "Polygon Cutter - press 'h' for help"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            # add point
            self.current.append((x,y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # remove last point
            if self.current:
                self.current.pop()

    def draw(self):
        disp = self.img_orig.copy()
        # draw completed polygons
        for poly in self.polygons:
            if len(poly) >= 3:
                pts = np.array(poly, np.int32).reshape((-1,1,2))
                cv2.polylines(disp, [pts], isClosed=True, color=(0,255,0), thickness=2)
                cv2.fillPoly(disp, [pts], color=(0,255,0,))  # visual fill (green)
            else:
                for p in poly:
                    cv2.circle(disp, p, 3, (0,255,0), -1)
        # draw current polygon
        if self.current:
            pts_current = np.array(self.current, np.int32).reshape((-1,1,2))
            cv2.polylines(disp, [pts_current], isClosed=False, color=(0,0,255), thickness=2)
            for p in self.current:
                cv2.circle(disp, p, 4, (0,0,255), -1)
        # mask overlay preview
        if self.show_mask_overlay:
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            pts_list = []
            for poly in self.polygons:
                if len(poly) >= 3:
                    pts_list.append(np.array(poly, np.int32).reshape((-1,1,2)))
            if pts_list:
                cv2.fillPoly(mask, pts_list, 255)
                overlay = disp.copy()
                overlay[mask==255] = (overlay[mask==255] * 0.2 + np.array([0, 200, 0]) * 0.8).astype(np.uint8)
                alpha = 0.35
                disp = cv2.addWeighted(overlay, alpha, disp, 1-alpha, 0)
        # draw background swatches
        sw_w = 30
        sw_h = 20
        margin = 8
        for i,c in enumerate(self.bg_swatches):
            bx = disp.shape[1] - (i+1)*(sw_w + margin)
            by = margin
            cv2.rectangle(disp, (bx,by), (bx+sw_w, by+sw_h), c, -1)
            if i == self.active_bg_idx:
                cv2.rectangle(disp, (bx-2,by-2), (bx+sw_w+2, by+sw_h+2), (0,255,255), 2)
        # instructions text
        info = [
            "LMB: add point  RMB: remove last point  Enter/c: complete poly",
            "n: new poly  z: undo last poly  x: delete poly under mouse",
            "b: cycle background  t: toggle mask preview  a: export PNG with alpha",
            "f: export flattened PNG with background  q/Esc: quit  h: help"
        ]
        y0 = disp.shape[0] - 80
        for i,line in enumerate(info):
            cv2.putText(disp, line, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        # draw mouse pos
        cv2.putText(disp, f"Mouse: {self.mouse_pos}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, disp)

    def polygon_index_under_point(self, pt):
        # return index of polygon that contains point, or -1
        x, y = pt
        for i, poly in enumerate(self.polygons):
            if len(poly) >= 3:
                mask = np.zeros((self.h, self.w), dtype=np.uint8)
                pts = np.array(poly, np.int32).reshape((-1,1,2))
                cv2.fillPoly(mask, [pts], 255)
                if mask[y, x] == 255:
                    return i
        return -1

    def export_alpha(self, outpath):
        out = cut_polygons_from_image(self.img_orig, self.polygons, export_alpha=True)
        # ensure PNG
        cv2.imwrite(outpath, out)
        print(f"Saved with alpha: {outpath}")

    def export_flat(self, outpath):
        bg = self.bg_swatches[self.active_bg_idx]
        out = cut_polygons_from_image(self.img_orig, self.polygons, background=bg, export_alpha=False)
        cv2.imwrite(outpath, out)
        print(f"Saved flattened: {outpath}")

    def run(self):
        print(__doc__)
        while True:
            self.draw()
            key = cv2.waitKey(50) & 0xFF
            if key == 255:  # no key
                continue
            if key in (27, ord('q')):  # ESC or q
                break
            elif key == ord('h'):
                print(__doc__)
            elif key == ord('\r') or key == ord('c'):
                # complete current polygon
                if len(self.current) >= 3:
                    self.polygons.append(self.current.copy())
                    self.current = []
                    print("Polygon completed. Total polygons:", len(self.polygons))
                else:
                    print("Need at least 3 points to complete a polygon.")
            elif key == ord('n'):
                self.current = []
                print("Started new polygon.")
            elif key == ord('z'):
                if self.polygons:
                    self.polygons.pop()
                    print("Undid last polygon. remaining:", len(self.polygons))
            elif key == ord('x'):
                idx = self.polygon_index_under_point(self.mouse_pos)
                if idx >= 0:
                    self.polygons.pop(idx)
                    print(f"Deleted polygon {idx}.")
                else:
                    print("No polygon under mouse to delete.")
            elif key == ord('b'):
                self.active_bg_idx = (self.active_bg_idx + 1) % len(self.bg_swatches)
                print("Background color set to", self.bg_swatches[self.active_bg_idx])
            elif key == ord('t'):
                self.show_mask_overlay = not self.show_mask_overlay
                print("Mask preview:", self.show_mask_overlay)
            elif key == ord('a'):
                # Save with alpha - choose filename via dialog
                fname = self.pick_filename(suffix="_alpha.png")
                if fname:
                    self.export_alpha(fname)
            elif key == ord('f'):
                fname = self.pick_filename(suffix="_flat.png")
                if fname:
                    self.export_flat(fname)

        cv2.destroyAllWindows()

    def pick_filename(self, suffix="_out.png"):
        # simple filedialog for saving
        root = Tk()
        root.withdraw()
        save_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=("cut"+suffix),
                                                 filetypes=[("PNG image","*.png")])
        root.destroy()
        return save_path

# -----------------------
# Entry: ask user to pick an image
# -----------------------
def pick_image_with_dialog():
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Select image to open", filetypes=[
        ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")])
    root.destroy()
    return path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive polygon cutter (OpenCV)")
    parser.add_argument("--image", "-i", help="Path to image to open", default=None)
    args = parser.parse_args()

    img_path = args.image
    if not img_path:
        img_path = pick_image_with_dialog()
    if not img_path or not os.path.exists(img_path):
        print("No image selected or file not found.")
        return
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Failed to load image.")
        return
    editor = PolygonEditor(img)
    editor.run()

if __name__ == "__main__":
    main()
