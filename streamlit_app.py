# streamlit_app.py
import io
import csv
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw
import numpy as np
import streamlit as st
from streamlit_cropper import st_cropper  # <-- fallback component

# ---------------- Data structures ----------------
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

# ---------------- Utilities ----------------
def parse_csv_points(uploaded) -> List[Point]:
    """Parse CSV with X/Y columns; supports headers x,y / cx,cy / left,top; else uses first two cols."""
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
        except:
            pass
    return pts

def draw_points_on_image(base_img: Image.Image, points: List[Point], normalized: bool=False) -> Image.Image:
    """Return base_img with points drawn as small dots."""
    img = base_img.copy().convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    radius = max(2, int(0.006 * max(w, h)))
    for p in points:
        x = p.x * w if normalized else p.x
        y = p.y * h if normalized else p.y
        draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=(0, 0, 0, 210))
    return img

def add_grid_overlay(img: Image.Image, step_hint: int=20) -> Image.Image:
    """Light grid overlay for alignment."""
    w, h = img.size
    grid = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g = ImageDraw.Draw(grid)
    step = max(50, min(w, h) // step_hint)
    for x in range(0, w, step):
        g.line([(x, 0), (x, h)], fill=(0, 0, 0, 40), width=1)
    for y in range(0, h, step):
        g.line([(0, y), (w, y)], fill=(0, 0, 0, 40), width=1)
    return Image.alpha_composite(img, grid)

def count_in_rect(points: List[Point], rect: Rect, img_w: int, img_h: int, normalized: bool=False) -> int:
    """Count points inside rect (inclusive)."""
    c = 0
    for p in points:
        x = p.x * img_w if normalized else p.x
        y = p.y * img_h if normalized else p.y
        if rect.minX <= x <= rect.maxX and rect.minY <= y <= rect.maxY:
            c += 1
    return c

# ---------------- UI ----------------
st.set_page_config(page_title="Image Region Click Counter", layout="wide")
st.title("Image Region Click Counter")

with st.sidebar:
    st.header("Inputs")
    img_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])
    csv_file = st.file_uploader("Upload CSV of points", type=["csv"])
    normalized = st.checkbox("Points are normalized (0..1)", value=False)
    show_grid = st.checkbox("Show grid overlay", value=True)
    st.caption("Headers like `x,y`, `cx,cy`, or `left,top` are detected; otherwise first two columns are used.")

if img_file is None or csv_file is None:
    st.info("Upload an image and a CSV to begin.")
    st.stop()

# Load image & points
base_img = Image.open(img_file).convert("RGBA")
img_w, img_h = base_img.size
points = parse_csv_points(csv_file)

# Downscale very large images (perf)
if img_w * img_h > 12_000_000:
    scale = (12_000_000 / (img_w * img_h)) ** 0.5
    base_img = base_img.resize((max(1, int(img_w * scale)), max(1, int(img_h * scale))), Image.LANCZOS)
    img_w, img_h = base_img.size

# Build background with dots (+ optional grid) and convert to RGB for cropper
bg_img = draw_points_on_image(base_img, points, normalized=normalized)
if show_grid:
    bg_img = add_grid_overlay(bg_img)
bg_rgb = bg_img.convert("RGB")

st.subheader("Draw a rectangle")
# st_cropper shows the image and lets you draw/move/resize a rectangle.
# It returns the cropped PIL image AND the bbox (in displayed image coordinates).
crop, bbox = st_cropper(
    bg_rgb,
    realtime_update=True,
    box_color='#00aaff',
    aspect_ratio=None,         # free-form rectangle
    return_type='both',
    key="cropper"
)

# bbox contains left, top, width, height in the displayed image space (same as bg_rgb).
left = float(bbox['left'])
top = float(bbox['top'])
width = float(bbox['width'])
height = float(bbox['height'])
rect = Rect.from_points((left, top), (left+width, top+height))

# Map directly to original bg_rgb pixels (cropper didn’t scale; it shows bg_rgb as provided)
w, h = bg_rgb.size
mapped_rect = Rect(
    minX=rect.minX,
    minY=rect.minY,
    maxX=rect.maxX,
    maxY=rect.maxY
)

# Compute stats
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

# ---------------- Download CSV (selection + stats) ----------------
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
