"""
OCT Segment Labeling Tool
──────────────────────────
Gradio app for manual annotation of OCT skin boundaries (AEJ, SC, EDJ).
Uses existing detection algorithms as initial predictions, then allows
the user to click-edit individual points on each boundary.

Output:
  labels/<folder_name>/<image_stem>.json   — coordinate data
  labels/<folder_name>/<image_stem>_overlay.png — visualisation
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from scipy.ndimage import gaussian_filter1d

from oct_algo import (
    load_image_state,
    detect_aej,
    detect_sc,
    detect_edj,
)

# ───────────────────────── paths ─────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "dataset"
LABELS_DIR = ROOT_DIR / "labels"

# ───────────────────────── label config ──────────────────
LABEL_NAMES = ["AEJ", "SC", "EDJ"]
LABEL_COLORS = {
    "AEJ": (255, 77, 77),   # 🔴
    "SC":  (255, 255, 0),    # 🟡
    "EDJ": (57, 255, 20),    # 🟢
}
COLOR_EMOJIS = {"AEJ": "🔴", "SC": "🟡", "EDJ": "🟢"}

APP_TITLE = "# 🏷️ OCT Segment Labeling Tool"

# ───────── default auto-detect params (good defaults) ────
DEFAULT_PARAMS = dict(
    search_top=20, search_bottom=280,
    gamma_aej=3.5, gamma_sc=3.5, gamma_edj=3.5,
    sigma=3.0, gauss_ksize=41, artifact_strength=0.2,
    dark_search_depth=18, edj_offset=30, edj_window=50,
    edj_method="3", fuzzy_sigma=10.0, edj_search_bottom=280,
    preproc_method="2", tv_weight=0.1, tex_beta_aej=0.0,
    thresh_scale=1.0, use_clahe=True, coh_thresh=0.3,
    tex_win=9, tex_beta_sc=0.0, tex_beta_edj=1.0,
)


# ───────────────────────── helpers ───────────────────────


def _get_folder_choices():
    """Scan DATA_DIR for subfolders containing .pgm or .txt image files."""
    folders = set()
    for ext in ("*.pgm", "*.txt"):
        folders |= {p.parent for p in DATA_DIR.rglob(ext)}
    # Also include DATA_DIR itself if it directly contains images
    for ext in ("*.pgm", "*.txt"):
        if list(DATA_DIR.glob(ext)):
            folders.add(DATA_DIR)
            break
    return sorted([(str(f.relative_to(ROOT_DIR)), str(f)) for f in sorted(folders)],
                  key=lambda x: x[0])


def _get_image_list(folder):
    """Return list of (display_name, full_path) for images in folder."""
    if not folder:
        return []
    p = Path(folder)
    import re

    def _numeric_key(f):
        """Extract number after 'o_' for numeric sorting (e.g. o_123.pgm → 123)."""
        m = re.search(r'o_(\d+)', f.stem)
        return int(m.group(1)) if m else float('inf')

    files = sorted(list(p.glob("*.pgm")) + list(p.glob("*.txt")),
                   key=_numeric_key)
    return [(f.name, str(f)) for f in files]


def _labels_path(folder, image_name):
    """Path to the JSON label file for a given image."""
    folder_name = Path(folder).relative_to(ROOT_DIR) if Path(folder).is_relative_to(ROOT_DIR) else Path(folder).name
    stem = Path(image_name).stem
    return LABELS_DIR / str(folder_name) / f"{stem}.json"


def _overlay_path(folder, image_name):
    """Path to the overlay PNG for a given image."""
    folder_name = Path(folder).relative_to(ROOT_DIR) if Path(folder).is_relative_to(ROOT_DIR) else Path(folder).name
    stem = Path(image_name).stem
    return LABELS_DIR / str(folder_name) / f"{stem}_overlay.png"


def _load_image_rgb(path):
    """Load a .pgm or .txt image, rotate 90° CW, return RGB numpy array."""
    if path.endswith('.txt'):
        img_gray = np.loadtxt(path).astype(np.uint8)
    else:
        img_gray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_gray is None:
        raise ValueError(f"Cannot read: {path}")
    img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)


# ──────────── auto-detect boundaries ─────────────────────

def _auto_detect(path):
    """Run existing pipeline with default params and return boundary arrays."""
    try:
        state = load_image_state(path)
        p = DEFAULT_PARAMS
        res_aej = detect_aej(
            state, p["search_top"], p["search_bottom"], p["gamma_aej"],
            p["sigma"], p["gauss_ksize"], p["artifact_strength"],
            p["preproc_method"], p["tv_weight"], p["tex_beta_aej"],
            p["thresh_scale"], p["use_clahe"], p["coh_thresh"], p["tex_win"],
        )
        if res_aej:
            state.update(res_aej)
        res_sc = detect_sc(
            state, state.get("AEJ"), state.get("w"), p["gamma_sc"],
            p["gauss_ksize"], p["dark_search_depth"], p["sigma"],
            p["preproc_method"], p["tex_beta_sc"],
        )
        if res_sc:
            state.update(res_sc)
        res_edj = detect_edj(
            state, state.get("w"), state.get("SC_boundary"), state.get("AEJ"),
            p["gamma_edj"], p["gauss_ksize"], p["edj_offset"], p["edj_window"],
            p["edj_search_bottom"], p["sigma"], p["edj_method"],
            p["fuzzy_sigma"], p["preproc_method"], p["tex_beta_edj"],
        )
        if res_edj:
            state.update(res_edj)
        # boundaries are 1-D arrays of length W (row index per column)
        AEJ = state.get("AEJ")
        SC = state.get("SC_boundary")
        EDJ = state.get("EDJ")
        W = state["W"]
        labels = {}
        if AEJ is not None:
            labels["AEJ"] = [[int(x), int(AEJ[x])] for x in range(W)]
        if SC is not None:
            labels["SC"] = [[int(x), int(SC[x])] for x in range(W)]
        if EDJ is not None:
            labels["EDJ"] = [[int(x), int(EDJ[x])] for x in range(W)]
        return labels
    except Exception as e:
        print(f"Auto-detect failed for {path}: {e}")
        return {name: [] for name in LABEL_NAMES}


# ──────────── rendering ─────────────────────────────────

def _render_overlay(img_rgb, labels, active_label=None, zoom=1.0, pan_x=0, pan_y=0, edit_radius=5):
    """Draw label polylines on image, apply zoom/pan crop, return display image."""
    canvas = img_rgb.copy()
    H, W = canvas.shape[:2]

    for name in LABEL_NAMES:
        pts = labels.get(name, [])
        if len(pts) < 2:
            # Draw individual dots
            for pt in pts:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, LABEL_COLORS[name], -1, cv2.LINE_AA)
            continue
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(canvas, [pts_arr], False, LABEL_COLORS[name], 2, cv2.LINE_AA)
        # Draw dots at each point for the active label
        if name == active_label:
            for pt in pts:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, LABEL_COLORS[name], -1, cv2.LINE_AA)

    # Apply zoom/pan
    if zoom > 1.0:
        crop_w = int(W / zoom)
        crop_h = int(H / zoom)
        cx = int(np.clip(pan_x, crop_w // 2, W - crop_w // 2))
        cy = int(np.clip(pan_y, crop_h // 2, H - crop_h // 2))
        x1 = cx - crop_w // 2
        y1 = cy - crop_h // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        canvas = canvas[y1:y2, x1:x2]
        canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_NEAREST)

    return canvas


def _render_from_state(state):
    """Convenience: extract fields from state dict and render."""
    if not state or state.get("img_raw") is None:
        return None
    return _render_overlay(
        state["img_raw"], state["labels"],
        active_label=state.get("active_label", "AEJ"),
        zoom=state.get("zoom", 1.0),
        pan_x=state.get("pan_x", 0), pan_y=state.get("pan_y", 0),
    )


# ──────────── state management ───────────────────────────

def _make_empty_state():
    return {
        "folder": None,
        "images": [],
        "idx": 0,
        "img_raw": None,
        "img_name": None,
        "img_path": None,
        "labels": {n: [] for n in LABEL_NAMES},
        "undo_stack": [],
        "redo_stack": [],
        "active_label": "AEJ",
        "zoom": 1.0,
        "pan_x": 0,
        "pan_y": 0,
    }


def _load_labels_from_disk(folder, image_name):
    """Load previously saved labels from JSON, or return None."""
    jp = _labels_path(folder, image_name)
    if jp.exists():
        try:
            with open(jp) as f:
                data = json.load(f)
            return data.get("labels", {n: [] for n in LABEL_NAMES})
        except Exception:
            pass
    return None


def _save_labels_to_disk(state):
    """Save current labels to JSON + overlay PNG."""
    if not state or state.get("img_raw") is None:
        return "⚠️ Không có dữ liệu."
    folder = state["folder"]
    img_name = state["img_name"]
    jp = _labels_path(folder, img_name)
    jp.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    data = {
        "image": img_name,
        "labels": state["labels"],
    }
    with open(jp, "w") as f:
        json.dump(data, f, indent=2)

    # Overlay PNG
    overlay = _render_overlay(state["img_raw"], state["labels"])
    op = _overlay_path(folder, img_name)
    cv2.imwrite(str(op), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return f"✅ Đã lưu: {jp.name} + {op.name}"


def _get_status_text(state):
    """Build status text showing labeled/unlabeled images."""
    if not state or not state.get("folder") or not state.get("images"):
        return "📋 Chưa chọn folder."
    folder = state["folder"]
    images = state["images"]
    total = len(images)
    labeled = 0
    lines = []
    for name, path in images:
        jp = _labels_path(folder, name)
        if jp.exists():
            labeled += 1
            lines.append(f"✅ {name}")
        else:
            lines.append(f"❌ {name}")
    header = f"**📋 Tiến độ: {labeled}/{total} ảnh đã gán nhãn**\n\n"
    return header + " · ".join(lines)


def _get_points_table(labels, active_label):
    """Create a summary of current label point counts."""
    parts = []
    for name in LABEL_NAMES:
        pts = labels.get(name, [])
        marker = "**→**" if name == active_label else " "
        parts.append(f"{marker} {COLOR_EMOJIS[name]} {name}: {len(pts)} điểm")
    return " &nbsp;|&nbsp; ".join(parts)


# ──────────── event handlers ─────────────────────────────

def on_select_folder(folder, state):
    """User selected a new folder."""
    if not folder:
        state = _make_empty_state()
        return (state, gr.update(choices=[], value=None),
                None, APP_TITLE + "\n**Chưa chọn folder**", "", "")

    images = _get_image_list(folder)
    state = _make_empty_state()
    state["folder"] = folder
    state["images"] = images

    if images:
        state["idx"] = 0
        return _load_and_render(state, 0)

    return (state, gr.update(choices=images, value=None),
            None, APP_TITLE + "\n**Folder trống — không có ảnh .pgm/.txt**", "", "")


def _load_and_render(state, idx):
    """Load image at index, auto-detect or load saved labels, render canvas."""
    images = state["images"]
    if not images or idx < 0 or idx >= len(images):
        return (state, gr.update(), None, APP_TITLE, "", "")

    state["idx"] = idx
    name, path = images[idx]
    state["img_name"] = name
    state["img_path"] = path
    state["img_raw"] = _load_image_rgb(path)
    state["undo_stack"] = []
    state["redo_stack"] = []
    state["zoom"] = 1.0
    state["pan_x"] = state["img_raw"].shape[1] // 2
    state["pan_y"] = state["img_raw"].shape[0] // 2

    # Load saved labels or auto-detect
    saved = _load_labels_from_disk(state["folder"], name)
    if saved:
        state["labels"] = saved
        info_msg = "📂 Loaded saved labels"
    else:
        state["labels"] = _auto_detect(path)
        info_msg = "🤖 Auto-detected boundaries"

    H, W = state["img_raw"].shape[:2]
    header = (
        f"{APP_TITLE}\n"
        f"**Image:** `{name}` &nbsp;·&nbsp; {H}×{W} px &nbsp;·&nbsp; "
        f"[{idx+1}/{len(images)}] &nbsp;·&nbsp; {info_msg}\n"
        f"**Boundaries:** {_get_points_table(state['labels'], state['active_label'])}"
    )

    canvas = _render_from_state(state)
    status = _get_status_text(state)

    return (
        state,
        gr.update(choices=images, value=path),
        canvas,
        header,
        _get_points_table(state["labels"], state["active_label"]),
        status,
    )


def on_select_image(path, state):
    """User selected image from dropdown."""
    if not path or not state or not state.get("images"):
        return state, gr.update(), None, APP_TITLE, "", ""

    # Auto-save removed

    # Find index
    for i, (name, p) in enumerate(state["images"]):
        if p == path:
            return _load_and_render(state, i)

    return state, gr.update(), None, APP_TITLE, "", ""


def on_prev(state):
    """Go to previous image."""
    if not state or not state.get("images"):
        return state, gr.update(), None, APP_TITLE, "", ""

    idx = max(0, state.get("idx", 0) - 1)
    return _load_and_render(state, idx)


def on_next(state):
    """Go to next image."""
    if not state or not state.get("images"):
        return state, gr.update(), None, APP_TITLE, "", ""

    idx = min(len(state["images"]) - 1, state.get("idx", 0) + 1)
    return _load_and_render(state, idx)


def on_click_image(state, tool, eraser_size, evt: gr.SelectData):
    """User clicked on the canvas image — add/move/erase point for active label."""
    if not state or state.get("img_raw") is None:
        return state, None, ""

    x_click, y_click = evt.index  # (col, row) from Gradio — in original image coordinates
    # Gradio's frontend automatically converts display coordinates to the
    # natural image pixel coordinates using naturalWidth/naturalHeight.

    H, W = state["img_raw"].shape[:2]

    # Convert zoom/pan coordinates back to original
    zoom = state.get("zoom", 1.0)
    if zoom > 1.0:
        crop_w = int(W / zoom)
        crop_h = int(H / zoom)
        cx = int(np.clip(state.get("pan_x", W // 2), crop_w // 2, W - crop_w // 2))
        cy = int(np.clip(state.get("pan_y", H // 2), crop_h // 2, H - crop_h // 2))
        # Map coords back to original image coords within the crop
        x_orig = int((x_click / W) * crop_w + cx - crop_w // 2)
        y_orig = int((y_click / H) * crop_h + cy - crop_h // 2)
    else:
        x_orig, y_orig = x_click, y_click

    active = state.get("active_label", "AEJ")
    pts = state["labels"].get(active, [])

    # Save undo state
    import copy
    state["undo_stack"].append(copy.deepcopy(state["labels"]))
    state["redo_stack"] = []

    if tool == "Tẩy":
        new_pts = []
        for pt in pts:
            d = ((pt[0] - x_orig) ** 2 + (pt[1] - y_orig) ** 2) ** 0.5
            if d > eraser_size:
                new_pts.append(pt)
        state["labels"][active] = new_pts
    else:
        # Pen mode: Soft brush for dense areas, move for nearby points, insert for sparse gaps
        import numpy as np
        brush_sigma = max(5, int(eraser_size))
        
        # Check if clicking to move an exact sparse point
        closest_idx = None
        closest_dist = float("inf")
        for i, pt in enumerate(pts):
            d = ((pt[0] - x_orig) ** 2 + (pt[1] - y_orig) ** 2) ** 0.5
            if d < closest_dist:
                closest_dist = d
                closest_idx = i

        if closest_dist < 8 and closest_idx is not None:
            # Move exact point
            pts[closest_idx] = [x_orig, y_orig]
            state["labels"][active] = pts
        else:
            # Determine if we are on a dense segment or a sparse gap
            closest_x_dist = min([abs(pt[0] - x_orig) for pt in pts]) if pts else float("inf")
            
            if closest_x_dist > 10:
                # Sparse gap -> insert point
                inserted = False
                for i, pt in enumerate(pts):
                    if pt[0] > x_orig:
                        pts.insert(i, [x_orig, y_orig])
                        inserted = True
                        break
                if not inserted:
                    pts.append([x_orig, y_orig])
                state["labels"][active] = pts
            else:
                # Dense segment -> apply soft brush (Gaussian pull)
                new_pts = []
                for pt in pts:
                    dx = pt[0] - x_orig
                    if abs(dx) < 3 * brush_sigma:
                        w = np.exp(-(dx**2) / (2.0 * (brush_sigma**2)))
                        new_y = int(pt[1] * (1 - w) + y_orig * w)
                        new_pts.append([pt[0], new_y])
                    else:
                        new_pts.append(pt)
                
                # Make sure the exact clicked X is present to anchor the peak
                inserted = False
                for i, pt in enumerate(new_pts):
                    if pt[0] == x_orig:
                        pt[1] = y_orig
                        inserted = True
                        break
                    elif pt[0] > x_orig:
                        new_pts.insert(i, [x_orig, y_orig])
                        inserted = True
                        break
                if not inserted:
                    new_pts.append([x_orig, y_orig])
                    
                state["labels"][active] = new_pts

    canvas = _render_from_state(state)
    points_info = _get_points_table(state["labels"], active)
    return state, canvas, points_info


def on_undo(state):
    """Undo last point action."""
    if not state or not state.get("undo_stack"):
        return state, _render_from_state(state) if state else None, ""
    import copy
    state["redo_stack"].append(copy.deepcopy(state["labels"]))
    state["labels"] = state["undo_stack"].pop()
    canvas = _render_from_state(state)
    return state, canvas, _get_points_table(state["labels"], state.get("active_label", "AEJ"))


def on_redo(state):
    """Redo last undone action."""
    if not state or not state.get("redo_stack"):
        return state, _render_from_state(state) if state else None, ""
    import copy
    state["undo_stack"].append(copy.deepcopy(state["labels"]))
    state["labels"] = state["redo_stack"].pop()
    canvas = _render_from_state(state)
    return state, canvas, _get_points_table(state["labels"], state.get("active_label", "AEJ"))


def on_clear_label(state):
    """Clear all points for the active label."""
    if not state:
        return state, None, ""
    import copy
    state["undo_stack"].append(copy.deepcopy(state["labels"]))
    state["redo_stack"] = []
    active = state.get("active_label", "AEJ")
    state["labels"][active] = []
    canvas = _render_from_state(state)
    return state, canvas, _get_points_table(state["labels"], active)


def on_clear_all(state):
    """Clear all labels for the current image."""
    if not state:
        return state, None, ""
    import copy
    state["undo_stack"].append(copy.deepcopy(state["labels"]))
    state["redo_stack"] = []
    state["labels"] = {n: [] for n in LABEL_NAMES}
    canvas = _render_from_state(state)
    return state, canvas, _get_points_table(state["labels"], state.get("active_label", "AEJ"))


def on_redetect(state):
    """Re-run auto-detection for current image, replacing current labels."""
    if not state or not state.get("img_path"):
        return state, None, ""
    import copy
    state["undo_stack"].append(copy.deepcopy(state["labels"]))
    state["redo_stack"] = []
    state["labels"] = _auto_detect(state["img_path"])
    canvas = _render_from_state(state)
    gr.Info("🤖 Đã chạy lại auto-detect!")
    return state, canvas, _get_points_table(state["labels"], state.get("active_label", "AEJ"))


def on_change_label(label_name, state):
    """Switch active label."""
    if not state:
        return state, None, ""
    state["active_label"] = label_name
    canvas = _render_from_state(state)
    return state, canvas, _get_points_table(state["labels"], label_name)


def on_zoom_change(zoom_val, state):
    """Update zoom level."""
    if not state:
        return state, None
    state["zoom"] = zoom_val
    return state, _render_from_state(state)


def on_pan(direction, state):
    """Pan the view in a direction."""
    if not state or state.get("img_raw") is None:
        return state, None
    step = 50
    if direction == "←":
        state["pan_x"] = state.get("pan_x", 0) - step
    elif direction == "→":
        state["pan_x"] = state.get("pan_x", 0) + step
    elif direction == "↑":
        state["pan_y"] = state.get("pan_y", 0) - step
    elif direction == "↓":
        state["pan_y"] = state.get("pan_y", 0) + step
    return state, _render_from_state(state)


def on_save(state):
    """Save labels + overlay for current image."""
    if not state or state.get("img_raw") is None:
        gr.Info("⚠️ Không có ảnh để lưu!")
        return _get_status_text(state)
    msg = _save_labels_to_disk(state)
    gr.Info(msg)
    return _get_status_text(state)


def on_delete_point(state):
    """Delete the last point of the active label."""
    if not state:
        return state, None, ""
    import copy
    active = state.get("active_label", "AEJ")
    pts = state["labels"].get(active, [])
    if not pts:
        return state, _render_from_state(state), _get_points_table(state["labels"], active)
    state["undo_stack"].append(copy.deepcopy(state["labels"]))
    state["redo_stack"] = []
    pts.pop()
    state["labels"][active] = pts
    canvas = _render_from_state(state)
    return state, canvas, _get_points_table(state["labels"], active)


# ──────────── Gradio UI ──────────────────────────────────

_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="emerald",
    neutral_hue="slate",
)
_CSS = """
.gradio-container { max-width: 1600px !important; }
.label-btn-aej { background: linear-gradient(135deg, #ff4d4d, #cc0000) !important; color: white !important; }
.label-btn-sc  { background: linear-gradient(135deg, #ffee00, #ccaa00) !important; color: black !important; }
.label-btn-edj { background: linear-gradient(135deg, #39ff14, #22cc00) !important; color: black !important; }
.nav-btn { min-width: 80px !important; font-size: 1.1em !important; }
.action-btn { min-width: 60px !important; }

/* ── Image canvas alignment fix ──
   Make the image fill the container width so that the displayed image
   and the interaction frame (container) are exactly the same size.
   Height auto-adjusts to maintain aspect ratio. */
#oct-canvas .image-container {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
#oct-canvas .image-container img {
    width: 100% !important;
    height: auto !important;
    object-fit: fill !important;
    display: block !important;
}
"""

with gr.Blocks(title="OCT Segment Labeling Tool") as demo:

    app_state = gr.State(_make_empty_state())

    md_header = gr.Markdown(value=APP_TITLE + "\n**Chọn folder để bắt đầu gán nhãn!**")

    with gr.Row():
        # ──── LEFT: Canvas ────
        with gr.Column(scale=3):
            img_canvas = gr.Image(
                label="🖼️ Canvas — Click để đặt điểm",
                type="numpy",
                interactive=False,
                buttons=[],
                elem_id="oct-canvas",
            )
            md_points = gr.Markdown(value="")

            with gr.Row():
                sl_zoom = gr.Slider(
                    1.0, 5.0, value=1.0, step=0.25,
                    label="🔍 Zoom", scale=3,
                )
                btn_pan_l = gr.Button("←", size="sm", elem_classes=["action-btn"], scale=1)
                btn_pan_u = gr.Button("↑", size="sm", elem_classes=["action-btn"], scale=1)
                btn_pan_d = gr.Button("↓", size="sm", elem_classes=["action-btn"], scale=1)
                btn_pan_r = gr.Button("→", size="sm", elem_classes=["action-btn"], scale=1)
                btn_zoom_reset = gr.Button("⟲ Reset", size="sm", elem_classes=["action-btn"], scale=1)

        # ──── RIGHT: Controls ────
        with gr.Column(scale=1, min_width=340):

            # --- Folder & Image ---
            with gr.Accordion("📂 Chọn Dữ Liệu", open=True):
                dd_folder = gr.Dropdown(
                    choices=_get_folder_choices(),
                    value=None,
                    label="Thư mục",
                    interactive=True,
                )
                dd_image = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Ảnh",
                    interactive=True,
                )
                with gr.Row():
                    btn_prev = gr.Button("◀ Prev", variant="secondary", size="sm",
                                         elem_classes=["nav-btn"])
                    btn_next = gr.Button("Next ▶", variant="secondary", size="sm",
                                         elem_classes=["nav-btn"])

            # --- Label selector ---
            with gr.Accordion("🏷️ Chọn Nhãn", open=True):
                rd_label = gr.Radio(
                    choices=LABEL_NAMES,
                    value="AEJ",
                    label="Nhãn đang gán",
                    info="Click trên ảnh sẽ thêm điểm cho nhãn này",
                )

            # --- Tool selector ---
            with gr.Accordion("🛠️ Công cụ", open=True):
                rd_tool = gr.Radio(
                    choices=["Bút vẽ", "Tẩy"],
                    value="Bút vẽ",
                    label="Công cụ hiện tại",
                    info="Bút vẽ: Uốn mượt nét vẽ/Thêm điểm. Tẩy: Xóa vùng.",
                )
                sl_eraser = gr.Slider(
                    minimum=5, maximum=100, value=20, step=1,
                    label="Kích thước cọ / tẩy (px)",
                    visible=True
                )

            # --- Actions ---
            with gr.Accordion("⚡ Thao Tác", open=True):
                with gr.Row():
                    btn_undo = gr.Button("↩ Undo", size="sm", elem_classes=["action-btn"])
                    btn_redo = gr.Button("↪ Redo", size="sm", elem_classes=["action-btn"])
                    btn_del = gr.Button("⌫ Xóa điểm cuối", size="sm", elem_classes=["action-btn"])
                with gr.Row():
                    btn_clear_label = gr.Button("🗑 Xóa nhãn hiện tại", size="sm", variant="stop")
                    btn_clear_all = gr.Button("🗑 Xóa tất cả", size="sm", variant="stop")
                btn_redetect = gr.Button("🤖 Auto-detect lại", size="sm", variant="secondary")

            # --- Save ---
            btn_save = gr.Button("💾 Lưu Nhãn", variant="primary", size="lg")

            # --- Status ---
            with gr.Accordion("📋 Trạng Thái", open=False):
                md_status = gr.Markdown(value="📋 Chưa chọn folder.")

    # ──────────── common output list ─────────────────────
    main_outputs = [app_state, dd_image, img_canvas, md_header, md_points, md_status]
    canvas_outputs = [app_state, img_canvas, md_points]

    # ──────────── wiring ─────────────────────────────────

    dd_folder.change(on_select_folder, inputs=[dd_folder, app_state], outputs=main_outputs)
    dd_image.change(on_select_image, inputs=[dd_image, app_state], outputs=main_outputs)

    btn_prev.click(on_prev, inputs=[app_state], outputs=main_outputs)
    btn_next.click(on_next, inputs=[app_state], outputs=main_outputs)

    img_canvas.select(on_click_image, inputs=[app_state, rd_tool, sl_eraser], outputs=canvas_outputs)

    rd_label.change(on_change_label, inputs=[rd_label, app_state], outputs=canvas_outputs)
    
    btn_undo.click(on_undo, inputs=[app_state], outputs=canvas_outputs)
    btn_redo.click(on_redo, inputs=[app_state], outputs=canvas_outputs)
    btn_del.click(on_delete_point, inputs=[app_state], outputs=canvas_outputs)
    btn_clear_label.click(on_clear_label, inputs=[app_state], outputs=canvas_outputs)
    btn_clear_all.click(on_clear_all, inputs=[app_state], outputs=canvas_outputs)
    btn_redetect.click(on_redetect, inputs=[app_state], outputs=canvas_outputs)

    sl_zoom.change(on_zoom_change, inputs=[sl_zoom, app_state], outputs=[app_state, img_canvas])

    btn_pan_l.click(lambda s: on_pan("←", s), inputs=[app_state], outputs=[app_state, img_canvas])
    btn_pan_r.click(lambda s: on_pan("→", s), inputs=[app_state], outputs=[app_state, img_canvas])
    btn_pan_u.click(lambda s: on_pan("↑", s), inputs=[app_state], outputs=[app_state, img_canvas])
    btn_pan_d.click(lambda s: on_pan("↓", s), inputs=[app_state], outputs=[app_state, img_canvas])

    def reset_zoom(state):
        if state:
            state["zoom"] = 1.0
            if state.get("img_raw") is not None:
                state["pan_x"] = state["img_raw"].shape[1] // 2
                state["pan_y"] = state["img_raw"].shape[0] // 2
        return state, _render_from_state(state) if state else None, gr.update(value=1.0)

    btn_zoom_reset.click(reset_zoom, inputs=[app_state], outputs=[app_state, img_canvas, sl_zoom])

    btn_save.click(on_save, inputs=[app_state], outputs=[md_status])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        theme=_THEME,
        css=_CSS,
    )
