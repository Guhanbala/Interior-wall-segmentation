import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import torch
import numpy as np

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ================= SHADOW REMOVAL =================
def remove_shadows(img):
    """Gentle shadow removal to help wall detection"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply gentle CLAHE to L channel to reduce shadow impact
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return result


# ================= PREPROCESS FOR SEGFORMER =================
def preprocess_for_segformer(img):
    # First remove shadows
    img = remove_shadows(img)
    
    # Gentle bilateral filter to smooth while preserving edges
    img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    
    # Very subtle sharpening
    blur = cv2.GaussianBlur(img, (0, 0), 0.5)
    img = cv2.addWeighted(img, 1.1, blur, -0.1, 0)

    return img


# ================= CONFIG =================
SAM3_CHECKPOINT = "sam3.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DISPLAY_W, DISPLAY_H = 900, 600

WALL_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (255, 200, 100), (200, 100, 255)
]

# ================= LOAD MODEL =================
print("Loading SAM 3...")
model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT)
model.to(DEVICE)
processor = Sam3Processor(model)
print("SAM 3 loaded!")

# ================= GLOBALS =================
image_np = None
wall_masks = []
tk_img = None

# ================= DETECT WALLS =================
def detect_all_walls():
    global wall_masks, image_np

    if image_np is None:
        return

    print("Detecting walls...")
    pil_image = Image.fromarray(image_np)
    state = processor.set_image(pil_image)
    
    results_wall = processor.set_text_prompt(state=state, prompt="interior painted wall")
    masks_wall = results_wall.get("masks", None)
    scores_wall = results_wall.get("scores", [])

    if masks_wall is None or masks_wall.numel() == 0:
        messagebox.showinfo("No Walls", "No wall masks detected.")
        return

    orig_h, orig_w = image_np.shape[:2]
    wall_masks = []

    for i in range(masks_wall.shape[0]):
        low_res_mask = masks_wall[i].squeeze().cpu().numpy()
        score = float(scores_wall[i]) if i < len(scores_wall) else 0.0

        full_mask = cv2.resize((low_res_mask > 0.28).astype(np.uint8), (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST) * 255

        # Morphological operations to clean up
        kernel = np.ones((9, 9), np.uint8)
        full_mask = cv2.dilate(full_mask, kernel, iterations=1)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Filter out very small masks (increased threshold)
        if cv2.countNonZero(full_mask) < 15000:
            continue
        if score < 0.38:
            continue

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_canvas = np.zeros_like(full_mask)

        for cnt in contours:
            # Filter small contours more aggressively
            if cv2.contourArea(cnt) < 15000:
                continue

            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            rectified = snap_to_axis(approx, tolerance=18)

            cv2.drawContours(clean_canvas, [rectified], -1, 255, -1)

        clean_canvas = cv2.morphologyEx(clean_canvas, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

        # Final size check - only keep substantial walls
        if cv2.countNonZero(clean_canvas) > 15000:
            wall_masks.append({
                "mask": clean_canvas,
                "color": WALL_COLORS[i % len(WALL_COLORS)],
                "number": i + 1,
                "score": score
            })

    draw_overlays()
    messagebox.showinfo("Done", f"Detected {len(wall_masks)} walls. Click anywhere near a wall to select it.")

# ================= SNAP TO AXIS =================
def snap_to_axis(contour, tolerance=18):
    if len(contour) < 2:
        return contour
    
    pts = contour[:, 0, :].astype(float)
    
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        
        if dx < tolerance:
            avg_x = (p1[0] + p2[0]) / 2
            pts[i][0] = avg_x
            pts[(i + 1) % len(pts)][0] = avg_x
            
        if dy < tolerance:
            avg_y = (p1[1] + p2[1]) / 2
            pts[i][1] = avg_y
            pts[(i + 1) % len(pts)][1] = avg_y
            
    return pts.astype(np.int32).reshape(-1, 1, 2)

# ================= DRAW OVERLAYS =================
def draw_overlays():
    global tk_img

    if image_np is None or not wall_masks:
        return

    display_img = cv2.resize(image_np, (DISPLAY_W, DISPLAY_H))
    overlay_pil = Image.fromarray(display_img).convert("RGBA")
    draw = ImageDraw.Draw(overlay_pil)

    for wall in wall_masks:
        mask_resized = cv2.resize(wall["mask"], (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 127).astype(np.uint8)

        color_rgba = wall["color"] + (85,)
        colored_layer = np.zeros((DISPLAY_H, DISPLAY_W, 4), dtype=np.uint8)
        colored_layer[mask_resized == 1] = color_rgba

        overlay_pil = Image.alpha_composite(overlay_pil, Image.fromarray(colored_layer, "RGBA"))

        coords = np.where(mask_resized == 1)
        if len(coords[0]) > 0:
            cy = int(np.mean(coords[0]))
            cx = int(np.mean(coords[1]))
            draw.text((cx - 12, cy - 20), str(wall['number']), fill="white", font_size=28)

    tk_img = ImageTk.PhotoImage(overlay_pil)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

# ================= CLICK ANYWHERE TO SELECT NEAREST WALL =================
def on_canvas_click(event):
    if not wall_masks or image_np is None:
        return

    orig_h, orig_w = image_np.shape[:2]
    click_x = int(event.x * orig_w / DISPLAY_W)
    click_y = int(event.y * orig_h / DISPLAY_H)

    selected = None
    min_dist = float("inf")

    # First, check if click is directly ON any mask
    for wall in wall_masks:
        mask = wall["mask"]
        if click_y < mask.shape[0] and click_x < mask.shape[1] and mask[click_y, click_x] > 127:
            selected = wall
            highlight_wall(selected)
            return  # Immediately select this wall and exit

    # If not clicked on a mask, find the nearest wall by center distance
    for wall in wall_masks:
        mask = wall["mask"]
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        dist = np.hypot(click_x - cx, click_y - cy)

        if dist < min_dist:
            min_dist = dist
            selected = wall

    # Select the nearest wall (no distance limit)
    if selected:
        highlight_wall(selected)
    else:
        messagebox.showinfo("No Walls", "No walls detected to select.")

# ================= HIGHLIGHT SELECTED WALL =================
def highlight_wall(wall):
    mask = (wall["mask"] // 255).astype(np.uint8)
    highlight = image_np.copy()
    highlight[mask == 1] = [70, 255, 70]

    blended = cv2.addWeighted(image_np, 0.6, highlight, 0.4, 0)
    resized = cv2.resize(blended, (DISPLAY_W, DISPLAY_H))

    tk_img_new = ImageTk.PhotoImage(Image.fromarray(resized))
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=tk_img_new)
    canvas.image = tk_img_new

    messagebox.showinfo("Selected", f"Wall {wall['number']} selected!")

# ================= LOAD IMAGE =================
def load_image():
    global image_np, wall_masks

    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return

    img = Image.open(path).convert("RGB")
    image_np = preprocess_for_segformer(np.array(img))

    wall_masks = []

    canvas.delete("all")
    canvas.create_text(DISPLAY_W//2, DISPLAY_H//2,
                       text="Detecting walls...", fill="white", font=("Arial", 16))
    root.update()

    detect_all_walls()

# ================= UI =================
root = tk.Tk()
root.title("SAM 3 Wall App - Shadow Corrected")
root.configure(bg="#333333")

tk.Button(root, text="Load Photo", command=load_image, font=("Arial", 14), padx=20, pady=10).pack(pady=15)

canvas = tk.Canvas(root, width=DISPLAY_W, height=DISPLAY_H, bg="gray20", highlightthickness=0)
canvas.pack(padx=10, pady=5)

canvas.bind("<Button-1>", on_canvas_click)

tk.Label(root,
         text="Load Photo → Auto-detect walls → Click anywhere near a wall to select & highlight it",
         fg="white", bg="#333333", font=("Arial", 10)).pack(pady=10)

root.mainloop()
