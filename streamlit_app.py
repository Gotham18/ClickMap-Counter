import io
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------- Data structures ----------
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

# ---------- CSV parsing ----------
def parse_csv_points(file) -> List[Point]:
    """Parses a CSV with at least two columns for X/Y. Tries header-based detection, else uses first two columns."""
    text = file.getvalue().decode("utf-8-sig")
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
            if ax is None and h in hx:
                ax = i
            if ay is None and h in hy:
                ay = i
        if ax is None or ay is None:
            ax, ay = 0, 1
    else:
        data = rows
        ax, ay = 0, 1

    for r in data:
        if len(r) < 2:
            continue
        try:
            pts.append(Point(float(r[ax]), float(r[ay])))
        except:
            pass
    return pts

# ---------- Drawing helpers ----------
def draw_points_on_image(base_img: Image.Image, points: List[Point], normalized: bool=False) -> Image.Image:
    """Returns a copy of base_img with points drawn as small circles."""
    img = base_img.copy().convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    radius = max(2, int(0.006 * max(w, h)))  # scale dot size with image size

    for p in points:
        x = p.x * w if normalized else p.x
        y = p.y * h if normalized else p.y
        # small filled circle
        bbox = [(x - radius, y - radius), (x + radius, y + radius)]
        draw.ellipse(bbox, fill=(0, 0, 0, 200))  # dark dot

    return img

def count_in_rect(points: List[Point], rect: Rect, img_w: int, img_h: int, normalized: bool=False) -> int:
    """Counts points within the rectangle (inclusive)."""
    c = 0
    for p in points:
        x = p.x * img_w if normalized else p.x
        y = p.y * img_h if normalized else p.y
        if rect.minX <= x <= rect.maxX and rect.minY <= y <= rect.maxY:
            c += 1
    return c

# ---------- UI ----------
st.set_page_config(page_title="Click Counter (Streamlit)", layout="wide")
st.title("Image Region Click Counter")

with st.sidebar:
    st.header("Inputs")
    img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    csv_file = st.file_uploader("Upload CSV of points", type=["csv"])
    normalized = st.checkbox("Points are normalized (0..1)", value=False)
    show_grid = st.checkbox("Show light grid overlay", value=True)
    stroke_width = st.slider("Rectangle stroke width", 2, 12, 4)
    instructions = st.expander("CSV format tips")
    with instructions:
        st.markdown(
            "- Works with headers like `x,y`, `cx,cy`, `left,top`, etc., or just the first two columns.\n"
            "- If normalized is on, points are treated as fractions of width/height."
        )

if img_file is None or csv_file is None:
    st.info("Upload an image and a CSV in the sidebar to begin.")
    st.stop()

# Load image & points
base_img = Image.open(img_file).convert("RGBA")
points = parse_csv_points(csv_file)
w, h = base_img.size

# Pre-render points on the image so users can draw a selection over them
bg_img = draw_points_on_image(base_img, points, normalized=normalized)

# Optional subtle grid overlay
if show_grid:
    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid)
    step = max(50, min(w, h) // 20)
    for x in range(0, w, step):
        gdraw.line([(x, 0), (x, h)], fill=(0, 0, 0, 40), width=1)
    for y in range(0, h, step):
        gdraw.line([(0, y), (w, y)], fill=(0, 0, 0, 40), width=1)
    bg_img = Image.alpha_composite(bg_img, grid)

st.subheader("Draw a rectangle over the image")
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_image=bg_img,
    update_streamlit=True,
    height=min(900, int(h * min(1.0, 1200 / max(1, w)))),  # scale down super large images for UX
    width=min(1200, w),
    drawing_mode="rect",
    key="canvas",
)

# Interpret latest rectangle (if any)
rect_display = st.empty()
metrics = st.empty()

rects = []
if canvas_result.json_data is not None:
    for obj in canvas_result.json_data.get("objects", []):
        if obj.get("type") == "rect":
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            width_obj = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
            height_obj = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
            r = Rect.from_points((left, top), (left + width_obj, top + height_obj))
            rects.append(r)

if not rects:
    st.warning("Draw a rectangle to see the selection details and count.")
    st.stop()

current_rect = rects[-1]
count = count_in_rect(points, current_rect, w, h, normalized=normalized)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        f"**Rect (px):** ({current_rect.minX:.0f}, {current_rect.minY:.0f}) → "
        f"({current_rect.maxX:.0f}, {current_rect.maxY:.0f})"
    )
with col2:
    st.markdown(f"**Clicks in region:** **{count}** / {len(points)}")

# Also show percent and area
rect_w = current_rect.maxX - current_rect.minX
rect_h = current_rect.maxY - current_rect.minY
pct = (count / max(1, len(points))) * 100
area_pct = (rect_w * rect_h) / (w * h) * 100

st.caption(
    f"Width×Height: {rect_w:.0f}×{rect_h:.0f} px · "
    f"Area: {area_pct:.2f}% of image · "
    f"Hit rate: {pct:.2f}%"
)
