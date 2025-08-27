# streamlit_app.py
import io
import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# =========================
# Data structures
# =========================
@dataclass
class Point:
    x: float
    y: float

@dataclass
class Rect:
    minX: float; minY: float; maxX: float; maxY: float
    @staticmethod
    def from_points(a: Tuple[float, float], b: Tuple[float, float]) -> "Rect":
        (x1, y1), (x2, y2) = a, b
        return Rect(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

# =========================
# Utilities
# =========================
def parse_csv_points(uploaded) -> List[Point]:
    text = uploaded.getvalue().decode("utf-8-sig")
    rows = list(csv.reader(text.splitlines()))
    pts: List[Point] = []
    if not rows:
        return pts

    header = [h.strip().lower() for h in rows[0]]
    data = rows[1:]
    ax = ay = None
    hx, hy = {"x", "cx", "lx", "left"}, {"y", "cy", "ly", "top"}

    if len(header) >= 2 and (set(header) & hx or set(header) & hy):
        for i, h in enumerate(header):
            if ax is None and h in hx: ax = i
            if ay is None and h in hy: ay = i
        if ax is None or ay is None: ax, ay = 0, 1
    else:
        data = rows
        ax, ay = 0, 1

    for r in data:
        if len(r) < 2: continue
        try:
            pts.append(Point(float(r[ax]), float(r[ay])))
        except:  # skip bad rows
            pass
    return pts

def draw_points_on_image(base_img: Image.Image, points: List[Point], normalized: bool=False) -> Image.Image:
    img = base_img.copy().convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    radius = max(2, int(0.006 * max(w, h)))
    for p in points:
        x = p.x * w if normalized else p.x
        y = p.y * h if normalized else p.y
        bbox = [(x - radius, y - radius), (x + radius, y + radius)]
        draw.ellipse(bbox, fill=(0, 0, 0, 210))
    return img

def add_grid_overlay(img: Image.Image, step_hint: int=20) -> Image.Image:
    w, h = img.size
    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    step = max(50, min(w, h) // step_hint)
    for x in range(0, w, step): g.line([(x, 0), (x, h)], fill=(0, 0, 0, 40), width=1)
    for y in range(0, h, step): g.line([(0, y), (w, y)], fill=(0, 0, 0, 40), width=1)
    return Image.alpha_composite(img, grid)

def count_in_rect(points: List[Point], rect: Rect, img_w: int, img_h: int, normalized: bool=False) -> int:
    c = 0
    for p in points:
        x = p.x * img_w if normalized else p.x
        y = p.y * img_h if normalized else p.y
        if rect.minX <= x <= rect.maxX and rect.minY <= y <= rect.maxY:
            c += 1
    return c

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Image Region Click Counter", layout="wide")
st.title("Image Region Click Counter")

with st.sidebar:
    st.header("Inputs")
    img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    csv_file = st.file_uploader("Upload CSV of points", type=["csv"])
    normalized = st.checkbox("Points are normalized (0..1)", value=False)
    show_grid = st.checkbox("Show grid overlay", value=True)
    stroke_width = st.slider("Rectangle stroke width", 2, 12, 4)
    st.caption("CSV headers like `x,y`, `cx,cy`, or `left,top` are detected; otherwise first two columns are used.")

if img_file is None or csv_file is None:
    st.info("Upload an image and a CSV to begin.")
    st.stop()

# Load image & points
base_img = Image.open(img_file).convert("RGBA")
w, h = base_img.size
points = parse_csv_points(csv_file)

# Downscale very large images (perf)
if w * h > 12_000_000:  # ~12 MP
    scale = (12_000_000 / (w * h)) ** 0.5
    base_img = base_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    w, h = base_img.size

# Render overlay
bg_img = draw_points_on_image(base_img, points, normalized=normalized)
if show_grid:
    bg_img = add_grid_overlay(bg_img)

# ---- Important: pass a PIL.Image (RGB), not a NumPy array ----
bg_img_rgb = bg_img.convert("RGB")

# Canvas size preserving aspect ratio
MAX_W, MAX_H = 1200, 900
scale = min(MAX_W / w, MAX_H / h, 1.0)
canvas_w = max(1, int(w * scale))
canvas_h = max(1, int(h * scale))

st.subheader("Draw a rectangle over the image")
canvas_result = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_image=bg_img_rgb,     # <— PIL Image (truthy, no ambiguous bool)
    background_color=None,
    update_streamlit=True,
    width=canvas_w,
    height=canvas_h,
    drawing_mode="rect",
    key=f"canvas_{w}x{h}",
)

# Collect rectangles
rects: List[Rect] = []
if canvas_result.json_data is not None:
    for obj in canvas_result.json_data.get("objects", []):
        if obj.get("type") == "rect":
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            w_obj = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
            h_obj = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
            rects.append(Rect.from_points((left, top), (left + w_obj, top + h_obj)))

if not rects:
    st.warning("Draw a rectangle to see selection details and stats.")
    st.stop()

current_rect = rects[-1]

# Map back to original image pixels
scale_x = w / canvas_w
scale_y = h / canvas_h
mapped_rect = Rect(
    minX=current_rect.minX * scale_x,
    minY=current_rect.minY * scale_y,
    maxX=current_rect.maxX * scale_x,
    maxY=current_rect.maxY * scale_y,
)

# Stats
count = count_in_rect(points, mapped_rect, w, h, normalized=normalized)
rect_w = mapped_rect.maxX - mapped_rect.minX
rect_h = mapped_rect.maxY - mapped_rect.minY
pct = (count / max(1, len(points))) * 100
area_pct = (rect_w * rect_h) / (w * h) * 100

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"**Rect (px):** ({mapped_rect.minX:.0f}, {mapped_rect.minY:.0f}) → "
        f"({mapped_rect.maxX:.0f}, {mapped_rect.maxY:.0f})"
    )
with col2:
    st.markdown(f"**Clicks in region:** **{count}** / {len(points)}")

st.caption(
    f"Width×Height: {rect_w:.0f}×{rect_h:.0f} px · "
    f"Area: {area_pct:.2f}% of image · "
    f"Hit rate: {pct:.2f}%"
)

# =========================
# Download selection & stats as CSV
# =========================
def build_stats_csv_bytes() -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "image_width_px","image_height_px","points_normalized",
        "rect_minX_px","rect_minY_px","rect_maxX_px","rect_maxY_px",
        "rect_width_px","rect_height_px",
        "area_pct_of_image","clicks_in_region","total_clicks","hit_rate_pct"
    ])
    writer.writerow([
        w, h, normalized,
        int(mapped_rect.minX), int(mapped_rect.minY), int(mapped_rect.maxX), int(mapped_rect.maxY),
        int(rect_w), int(rect_h),
        f"{area_pct:.4f}", count, len(points), f"{pct:.4f}"
    ])
    return output.getvalue().encode("utf-8")

st.download_button(
    label="⬇️ Download selection stats (CSV)",
    data=build_stats_csv_bytes(),
    file_name="selection_stats.csv",
    mime="text/csv",
)
