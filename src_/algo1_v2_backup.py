import os
import glob
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import gradio as gr
from datetime import datetime

OUTPUT_DIR = "results"
DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Mutable image state — updated by load_image()
# ---------------------------------------------------------------------------
state = {}


def load_image(path):
    """Load, rotate, preprocess, and store everything in `state`."""
    img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"Cannot read image: {path}")

    img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

    img_proc = cv2.medianBlur(img_gray, 3)
    img_proc = cv2.fastNlMeansDenoising(
        img_proc, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_proc)

    H, W = img_clahe.shape
    state.update(
        path=path,
        name=os.path.basename(path),
        img_ori=img_gray,
        img_ori_rgb=cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB),
        img_clahe=img_clahe,
        img_clahe_rgb=cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB),
        img_base=img_clahe.astype(np.float32) / 255.0,
        H=H,
        W=W,
    )
    print(f"Loaded: {path} | {H}×{W} px")


# Load default image at startup
_default_candidates = sorted(glob.glob(os.path.join(DATA_DIR, "*.pgm")))
if not _default_candidates:
    _default_candidates = sorted(glob.glob(os.path.join("../", DATA_DIR, "*.pgm")))
if _default_candidates:
    load_image(_default_candidates[0])
else:
    raise FileNotFoundError(f"No .pgm files found in {DATA_DIR}/")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def prepare_work(artifact_strength):
    w = state["img_base"].copy()
    w[w < w.mean()] = 0

    if artifact_strength > 0:
        row_median = np.median(w, axis=1, keepdims=True)
        w = w - artifact_strength * row_median
        w = np.clip(w, 0, None)
        wmax = w.max()
        if wmax > 0:
            w = w / wmax

    return w


def _build_layer_img(mask, color, boundary_pts_list, boundary_colors):
    img = state["img_ori_rgb"].copy().astype(np.float32)
    ov = np.zeros_like(img)
    ov[mask] = color
    img[mask] = img[mask] * 0.55 + ov[mask] * 0.45
    img = np.clip(img, 0, 255).astype(np.uint8)
    # for pts, c in zip(boundary_pts_list, boundary_colors):
    #     cv2.polylines(img, [pts], False, c, 2, cv2.LINE_AA)
    return img


def compute_and_draw(search_top, search_bottom, gamma_aej, edj_offset, edj_window,
                     gamma_edj, sigma, artifact_strength, gauss_ksize,
                     dark_search_depth):
    H, W = state["H"], state["W"]
    w = prepare_work(artifact_strength)

    # AEJ detection
    ks = int(gauss_ksize) | 1
    aej_blurred = cv2.GaussianBlur(w, (ks, ks), 0)
    aej_img = np.power(aej_blurred, gamma_aej)
    grad_map = np.abs(np.gradient(aej_img, axis=0))

    s_top = int(min(max(search_top, 0), H - 2))
    s_bot = int(min(max(search_bottom, s_top + 1), H - 1))
    AEJ = np.argmax(grad_map[s_top:s_bot], axis=0) + s_top
    AEJ = gaussian_filter1d(AEJ.astype(np.float32), sigma=sigma).astype(int)

    # EDJ detection
    edj_win = int(edj_window)
    offsets = np.arange(edj_win).reshape(-1, 1)
    starts = (AEJ + int(edj_offset)).reshape(1, -1)
    row_idx = starts + offsets
    valid = row_idx < H
    row_idx = np.clip(row_idx, 0, H - 1)
    col_idx = np.broadcast_to(np.arange(W).reshape(1, -1), row_idx.shape)
    rois = w[row_idx, col_idx]
    rois = np.where(valid, rois, 0)
    rois = np.power(rois, gamma_edj)
    rois = gaussian_filter1d(rois, sigma=1.4, axis=0)
    g = np.abs(np.gradient(rois, axis=0))
    EDJ = starts.ravel() + np.argmax(g, axis=0)
    EDJ = np.clip(EDJ, 0, H - 1)
    EDJ = gaussian_filter1d(EDJ.astype(np.float32), sigma=sigma).astype(int)

    # SC boundary via per-column minimum gradient below AEJ
    depth = int(dark_search_depth)
    row_off = np.arange(depth).reshape(-1, 1)
    starts_sc = (AEJ + 2).reshape(1, -1)
    search_r = starts_sc + row_off
    ends_sc = EDJ.reshape(1, -1)
    valid_sc = (search_r < H) & (search_r < ends_sc)
    search_r_clip = np.clip(search_r, 0, H - 1)
    c_idx = np.broadcast_to(np.arange(W).reshape(1, -1), search_r.shape)
    sg = grad_map[search_r_clip, c_idx]
    sg = np.where(valid_sc, sg, np.inf)
    SC_boundary = starts_sc.ravel() + np.argmin(sg, axis=0)
    SC_boundary = np.clip(SC_boundary, 0, H - 1)
    SC_boundary = gaussian_filter1d(SC_boundary.astype(np.float32), sigma=sigma).astype(int)

    # Polyline points
    xs = np.arange(W)
    pts_aej = np.column_stack([xs, AEJ]).astype(np.int32)
    pts_sc = np.column_stack([xs, SC_boundary]).astype(np.int32)
    pts_edj = np.column_stack([xs, EDJ]).astype(np.int32)
    all_pts = [pts_aej, pts_sc, pts_edj]
    all_colors = [(255, 77, 77), (255, 255, 0), (57, 255, 20)]

    # Result overlay
    result = state["img_ori_rgb"].copy()
    for pts, c in zip(all_pts, all_colors):
        cv2.polylines(result, [pts], False, c, 1, cv2.LINE_AA)

    # Gradient map
    grad_norm = (grad_map / (grad_map.max() + 1e-8) * 255).astype(np.uint8)
    grad_color = cv2.applyColorMap(grad_norm, cv2.COLORMAP_HOT)
    grad_color = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB)
    cv2.line(grad_color, (0, s_top), (W - 1, s_top), (0, 255, 255), 1)
    cv2.line(grad_color, (0, s_bot), (W - 1, s_bot), (0, 255, 255), 1)
    cv2.polylines(grad_color, [pts_sc], False, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.polylines(grad_color, [pts_edj], False, (255, 0, 0), 1, cv2.LINE_AA)

    # Per-column region masks
    rows = np.arange(H).reshape(-1, 1)
    aej_r = AEJ.reshape(1, -1)
    sc_r = SC_boundary.reshape(1, -1)
    edj_r = EDJ.reshape(1, -1)

    mask_sc = (rows >= aej_r) & (rows < sc_r)
    mask_epi = (rows >= sc_r) & (rows < edj_r)
    mask_dermis = (rows >= edj_r) & (rows < np.minimum(edj_r + 50, H))

    img_sc = _build_layer_img(mask_sc, [0, 200, 255], all_pts, all_colors)
    img_epi = _build_layer_img(mask_epi, [255, 100, 255], all_pts, all_colors)
    img_dermis = _build_layer_img(mask_dermis, [50, 255, 100], all_pts, all_colors)

    return result, state["img_clahe_rgb"], grad_color, img_sc, img_epi, img_dermis


def save_all(search_top, search_bottom, gamma_aej, edj_offset, edj_window,
             gamma_edj, sigma, artifact_strength, gauss_ksize,
             dark_search_depth):
    result, clahe, grad, sc, epi, dermis = compute_and_draw(
        search_top, search_bottom, gamma_aej, edj_offset, edj_window,
        gamma_edj, sigma, artifact_strength, gauss_ksize,
        dark_search_depth)

    img_stem = os.path.splitext(state["name"])[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(OUTPUT_DIR, f"{img_stem}_{stamp}")
    os.makedirs(folder, exist_ok=True)

    names = ["boundaries", "clahe", "gradient", "layer_sc", "layer_epidermis", "layer_dermis"]
    images = [result, clahe, grad, sc, epi, dermis]
    for name, img in zip(names, images):
        cv2.imwrite(os.path.join(folder, f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return f"✅ Đã lưu {len(images)} ảnh vào: {folder}/"


# ---------------------------------------------------------------------------
# Image loading handlers
# ---------------------------------------------------------------------------
def _get_data_choices():
    pgms = sorted(glob.glob(os.path.join(DATA_DIR, "*.pgm")))
    if not pgms:
        pgms = sorted(glob.glob(os.path.join("../", DATA_DIR, "*.pgm")))
    return pgms


def on_select_data_image(path):
    """Load an image from the data/ dropdown."""
    if not path:
        return [gr.update()] * 8
    load_image(path)
    return _after_load()


def on_upload_image(file_obj):
    """Load a user-uploaded image file."""
    if file_obj is None:
        return [gr.update()] * 8
    load_image(file_obj)
    return _after_load()


def _after_load():
    """Return updated header + slider ranges + trigger a re-draw."""
    H, W = state["H"], state["W"]
    header = (
        f"# 🔬 OCT Skin Boundary Segmentation\n"
        f"**Image:** `{state['name']}` &nbsp;·&nbsp; {H}×{W} px\n"
        f"**Boundaries:** 🔴 AEJ &nbsp;·&nbsp; 🟡 SC (Dark Band) &nbsp;·&nbsp; 🟢 EDJ"
    )
    return [
        gr.update(value=header),                           # md_header
        gr.update(maximum=H - 2, value=min(100, H - 2)),  # sl_top
        gr.update(maximum=H - 1, value=min(280, H - 1)),  # sl_bot
        gr.update(maximum=H - 2),                          # sl_eoff
        gr.update(maximum=H - 1),                          # sl_ewin
        gr.update(),                                       # (placeholder)
        gr.update(),
        gr.update(),
    ]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="OCT Boundary Segmentation",
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css=".gradio-container { max-width: 1500px !important; }",
) as demo:

    H0, W0 = state["H"], state["W"]

    md_header = gr.Markdown(
        f"# 🔬 OCT Skin Boundary Segmentation\n"
        f"**Image:** `{state['name']}` &nbsp;·&nbsp; {H0}×{W0} px\n"
        f"**Boundaries:** 🔴 AEJ &nbsp;·&nbsp; 🟡 SC (Dark Band) &nbsp;·&nbsp; 🟢 EDJ"
    )

    with gr.Row():
        with gr.Column(scale=3):
            img_result = gr.Image(label="Kết quả AEJ, SC & EDJ", type="numpy", height=400)
            with gr.Row():
                img_cl = gr.Image(label="Sau CLAHE", type="numpy", height=240)
                img_grd = gr.Image(label="Gradient Map", type="numpy", height=240)
            with gr.Row():
                img_sc_out = gr.Image(label="Layer: Stratum Corneum", type="numpy", height=240)
                img_epi_out = gr.Image(label="Layer: Epidermis", type="numpy", height=240)
                img_der_out = gr.Image(label="Layer: Dermis", type="numpy", height=240)

        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📂 Load Image")
            dd_data = gr.Dropdown(
                choices=_get_data_choices(),
                value=state["path"],
                label="Chọn ảnh từ data/",
                interactive=True,
            )
            file_upload = gr.File(
                label="Hoặc upload ảnh mới (.pgm / .png / .jpg)",
                file_types=[".pgm", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
                type="filepath",
            )

            gr.Markdown("### 🧹 Artifact Removal")
            sl_art = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                               label="Horizontal Artifact Strength",
                               info="Row-wise median subtraction (0=off, 1=full)")

            gr.Markdown("### ⚙️ AEJ")
            sl_top = gr.Slider(0, H0 - 2, value=100, step=1, label="Search Top")
            sl_bot = gr.Slider(1, H0 - 1, value=280, step=1, label="Search Bottom")
            sl_gaej = gr.Slider(0.5, 5.0, value=3.5, step=0.05, label="Gamma AEJ")
            sl_gks = gr.Slider(3, 60, value=41, step=2, label="Gauss Kernel Size",
                               info="Kích thước bộ lọc Gauss cho AEJ (số lẻ)")

            gr.Markdown("### 🌑 Dark Band / SC Boundary")
            sl_dsd = gr.Slider(5, 80, value=7, step=1, label="Dark Search Depth",
                               info="Chiều sâu tìm min-gradient dưới AEJ (pixels)")

            gr.Markdown("### ⚙️ EDJ")
            sl_eoff = gr.Slider(0, 100, value=35, step=1, label="EDJ Offset")
            sl_ewin = gr.Slider(10, 200, value=10, step=1, label="EDJ Window")
            sl_gedj = gr.Slider(0.5, 20.0, value=7.0, step=0.1, label="Gamma EDJ")

            gr.Markdown("### ⚙️ Smoothing")
            sl_sig = gr.Slider(1.0, 30.0, value=3, step=0.5, label="Sigma")

            btn = gr.Button("💾 Lưu Toàn Bộ", variant="primary", size="lg")
            status = gr.Textbox(label="", interactive=False, max_lines=1)

    # --- Wiring ---
    slider_inputs = [sl_top, sl_bot, sl_gaej, sl_eoff, sl_ewin, sl_gedj,
                     sl_sig, sl_art, sl_gks, sl_dsd]
    draw_outputs = [img_result, img_cl, img_grd, img_sc_out, img_epi_out, img_der_out]

    load_outputs = [md_header, sl_top, sl_bot, sl_eoff, sl_ewin,
                    status, status, status]

    dd_data.change(
        on_select_data_image, inputs=[dd_data], outputs=load_outputs
    ).then(
        compute_and_draw, inputs=slider_inputs, outputs=draw_outputs
    )

    file_upload.change(
        on_upload_image, inputs=[file_upload], outputs=load_outputs
    ).then(
        compute_and_draw, inputs=slider_inputs, outputs=draw_outputs
    )

    for sl in slider_inputs:
        sl.change(compute_and_draw, inputs=slider_inputs, outputs=draw_outputs,
                  show_progress="hidden")

    btn.click(save_all, inputs=slider_inputs, outputs=[status])
    demo.load(compute_and_draw, inputs=slider_inputs, outputs=draw_outputs)

demo.launch(server_name="0.0.0.0", server_port=7860)
