import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

def load_image_state(path):
    """Load, rotate, preprocess, and return image state dictionary."""
    if path.endswith('.txt'):
        img_gray = np.loadtxt(path)
        img_gray = img_gray.astype(np.uint8)
    else:
        img_gray = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img_gray is None:
        raise ValueError(f"Cannot read image: {path}")

    img_gray = cv2.rotate(img_gray, cv2.ROTATE_90_CLOCKWISE)

    img_median = cv2.medianBlur(img_gray, 3)
    img_nlm = cv2.fastNlMeansDenoising(
        img_median, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_nlm)

    H, W = img_clahe.shape
    state = {
        "path": path,
        "name": os.path.basename(path),
        "img_ori": img_gray,
        "img_ori_rgb": cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB),
        "img_median_rgb": cv2.cvtColor(img_median, cv2.COLOR_GRAY2RGB),
        "img_nlm": img_nlm,
        "img_nlm_rgb": cv2.cvtColor(img_nlm, cv2.COLOR_GRAY2RGB),
        "img_clahe": img_clahe,
        "img_clahe_rgb": cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB),
        "img_base": img_clahe.astype(np.float32) / 255.0,
        "H": H,
        "W": W,
    }
    print(f"Loaded: {path} | {H}×{W} px")
    return state


def prepare_work(state, artifact_strength, use_clahe):
    if use_clahe:
        w = state["img_clahe"].astype(np.float32) / 255.0
    else:
        w = state["img_nlm"].astype(np.float32) / 255.0
        
    w[w < w.mean()] = 0

    if artifact_strength > 0:
        row_median = np.median(w, axis=1, keepdims=True)
        w = w - artifact_strength * row_median
        w = np.clip(w, 0, None)
        wmax = w.max()
        if wmax > 0:
            w = w / wmax

    return w


def _build_layer_img(state, mask, color, boundary_pts_list, boundary_colors):
    img = state["img_ori_rgb"].copy().astype(np.float32)
    ov = np.zeros_like(img)
    ov[mask] = color
    img[mask] = img[mask] * 0.55 + ov[mask] * 0.45
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def detect_edj_graph(w, SC_boundary, edj_offset, search_bottom, H, W, sigma, fuzzy_sigma,
                     external_cost=None):
    """
    EDJ detection via graph shortest-path (DP column sweep).
    Based on: 'Semi-automated localization of DEJ in OCT images' (graph + fuzzy smoothing).
    `external_cost`: cost map dựng sẵn (chế độ phân rã); None thì tự dựng từ log-attenuation.
    Returns: (EDJ, viz_cost_map, viz_path_on_cost, viz_final)
    """
    edj_off = int(edj_offset)
    search_bottom = int(search_bottom)

    # Per-column ROI row ranges [r_start, r_end)
    r_starts = np.clip(SC_boundary + edj_off, 0, H - 1)
    r_ends = np.full(W, search_bottom)
    r_ends = np.maximum(r_starts + 1, r_ends) # ensure at least 1px height
    r_ends = np.clip(r_ends, 0, H)

    # Build attenuation cost map (full image, then ROI mask applied)
    if external_cost is not None:
        atten_norm = external_cost / (external_cost.max() + 1e-8)
    else:
        log_w = np.log1p(w * 255.0)
        atten = np.abs(np.gradient(log_w, axis=0))
        atten_norm = atten / (atten.max() + 1e-8)
    # Low attenuation at boundary = low cost -> invert
    cost_map = 1.0 - atten_norm

    # DP shortest path column-by-column (3-connected: up/same/down)
    INF = 1e9
    dp = np.full((H, W), INF, dtype=np.float64)
    parent = np.zeros((H, W), dtype=np.int32)

    # Initialise left column within each column's ROI
    for r in range(r_starts[0], r_ends[0]):
        dp[r, 0] = cost_map[r, 0]

    for c in range(1, W):
        rs = r_starts[c]
        re = r_ends[c]
        for r in range(rs, re):
            best_cost = INF
            best_parent = r
            for dr in (-1, 0, 1):
                pr = r + dr
                if rs <= pr < re and dp[pr, c - 1] < best_cost:
                    best_cost = dp[pr, c - 1]
                    best_parent = pr
            dp[r, c] = best_cost + cost_map[r, c]
            parent[r, c] = best_parent

    # Traceback: find best row at last column
    rs_last, re_last = r_starts[W - 1], r_ends[W - 1]
    best_r = rs_last + np.argmin(dp[rs_last:re_last, W - 1])
    path = np.zeros(W, dtype=np.int32)
    path[W - 1] = best_r
    for c in range(W - 2, -1, -1):
        path[c] = parent[path[c + 1], c + 1]

    # Fuzzy smoothing (Gaussian-weighted, equivalent to fuzzy membership avg)
    EDJ = gaussian_filter1d(path.astype(np.float32), sigma=fuzzy_sigma).astype(int)
    EDJ = np.clip(EDJ, 0, H - 1)

    # Visualization images
    xs = np.arange(W)
    pts_top = np.column_stack([xs, r_starts]).astype(np.int32)
    pts_bot = np.column_stack([xs, np.clip(r_ends - 1, 0, H - 1)]).astype(np.int32)

    cost_u8 = (cost_map / (cost_map.max() + 1e-8) * 255).astype(np.uint8)
    viz_cost = cv2.applyColorMap(cost_u8, cv2.COLORMAP_VIRIDIS)
    viz_cost = cv2.cvtColor(viz_cost, cv2.COLOR_BGR2RGB)
    cv2.polylines(viz_cost, [pts_top], False, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(viz_cost, [pts_bot], False, (255, 255, 255), 1, cv2.LINE_AA)

    viz_path = viz_cost.copy()
    pts_path = np.column_stack([xs, path]).astype(np.int32)
    cv2.polylines(viz_path, [pts_path], False, (255, 80, 0), 1, cv2.LINE_AA)

    return EDJ, viz_cost, viz_path


def get_decomposition(state, tv_weight, thresh_scale, coh_thresh, tex_win):
    """Phân rã log-domain có cache — chỉ chạy lại khi tham số phân rã đổi."""
    from oct_denoise import decompose_oct
    from oct_cost import texture_energy, texture_transition_cost

    key = (state["path"], float(tv_weight), float(thresh_scale), float(coh_thresh), int(tex_win))
    if state.get("decomp_key") == key:
        return state["decomp"]

    decomp = decompose_oct(state["img_ori"], tv_weight, thresh_scale, coh_thresh)
    decomp["G_texture"] = texture_transition_cost(texture_energy(decomp["Ui"], tex_win), tex_win)

    state["decomp_key"] = key
    state["decomp"] = decomp
    return decomp


def _to_rgb_norm(x):
    x_u8 = np.clip((x - x.min()) / (x.max() - x.min() + 1e-8) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(x_u8, cv2.COLOR_GRAY2RGB)


def stage_cost_map(state, gamma, beta):
    """Cost map hai kênh cho một bước dò biên, tái dùng G_texture đã cache."""
    from oct_cost import build_cost_map
    decomp = state["decomp"]
    G_total, _ = build_cost_map(decomp["Us_log"], decomp["G_texture"], gamma, beta)
    return G_total


def detect_aej(state, search_top, search_bottom, gamma_aej, sigma, gauss_ksize, artifact_strength,
               preproc_method="2", tv_weight=0.1, tex_beta=0.0, thresh_scale=1.0, use_clahe=True,
               coh_thresh=0.3, tex_win=9):
    if not state:
        return None
    H, W = state["H"], state["W"]

    if "1" in preproc_method:
        # 1. Truyền thống: NLM + Median -> Artifact Suppression -> Gaussian Blur -> Gamma
        w = prepare_work(state, artifact_strength, use_clahe)
        ks = int(gauss_ksize) | 1
        aej_blurred = cv2.GaussianBlur(w, (ks, ks), 0)
        aej_img = np.power(aej_blurred, gamma_aej)
        grad_map = np.abs(np.gradient(aej_img, axis=0))
        residual = state["img_ori"].astype(np.float32) / 255.0 - aej_img

        state["ui_img2"] = state["img_median_rgb"]
        state["ui_img3"] = state["img_nlm_rgb"]
        if use_clahe:
            state["ui_img4"] = state["img_clahe_rgb"]
        else:
            state["ui_img4"] = state["img_nlm_rgb"] # Same as img3 if CLAHE is off
    else:
        # 2. Phân rã log-domain -> cost map hai kênh (không tổng hợp lại thành ảnh)
        decomp = get_decomposition(state, tv_weight, thresh_scale, coh_thresh, tex_win)

        Us_u8 = np.clip(decomp["Us_lin"] * 255.0, 0, 255).astype(np.uint8)
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            Us_u8 = clahe.apply(Us_u8)
        w = Us_u8.astype(np.float32) / 255.0
        aej_blurred = w
        aej_img = w

        grad_map = stage_cost_map(state, gamma_aej, tex_beta)
        residual = decomp["Ur"]

        state["ui_img2"] = cv2.cvtColor(Us_u8, cv2.COLOR_GRAY2RGB)
        state["ui_img3"] = _to_rgb_norm(decomp["Ui"])
        state["ui_img4"] = _to_rgb_norm(decomp["Ur"])

    state["w"] = w  # Lưu lại w (float) cho SC, EDJ dùng chung

    # --- EVALUATION ---
    try:
        from oct_eval import evaluate_denoising
        I_ori_f = state["img_ori"].astype(np.float32) / 255.0
        grad_ori = np.abs(np.gradient(I_ori_f, axis=0))
        w_index = W // 2 # Lấy cột giữa ảnh
        res_img, fig_hist, fig_prof = evaluate_denoising(residual, grad_ori, grad_map, w_index)
        state["eval_residual"] = res_img
        state["eval_hist"] = fig_hist
        state["eval_profile"] = fig_prof
    except Exception as e:
        print("Evaluation error:", e)

    s_top = int(min(max(search_top, 0), H - 2))
    s_bot = int(min(max(search_bottom, s_top + 1), H - 1))
    AEJ = np.argmax(grad_map[s_top:s_bot], axis=0) + s_top
    AEJ = gaussian_filter1d(AEJ.astype(np.float32), sigma=sigma).astype(int)

    if "1" in preproc_method:
        aej_step1 = cv2.cvtColor(np.clip(aej_blurred * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        aej_step2 = cv2.cvtColor(np.clip(aej_img * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        decomp = state["decomp"]
        aej_step1 = _to_rgb_norm(np.abs(np.gradient(decomp["Us_log"], axis=0)))
        aej_step2 = _to_rgb_norm(decomp["G_texture"])

    grad_norm = (grad_map / (grad_map.max() + 1e-8) * 255).astype(np.uint8)
    aej_step3 = cv2.applyColorMap(grad_norm, cv2.COLORMAP_HOT)
    aej_step3 = cv2.cvtColor(aej_step3, cv2.COLOR_BGR2RGB)
    cv2.line(aej_step3, (0, s_top), (W - 1, s_top), (0, 255, 255), 1)
    cv2.line(aej_step3, (0, s_bot), (W - 1, s_bot), (0, 255, 255), 1)

    xs = np.arange(W)
    pts_aej = np.column_stack([xs, AEJ]).astype(np.int32)
    aej_step4 = state["img_ori_rgb"].copy()
    cv2.polylines(aej_step4, [pts_aej], False, (255, 77, 77), 1, cv2.LINE_AA)

    return {
        "AEJ": AEJ,
        "grad_map": grad_map,
        "grad_norm": grad_norm,
        "w": w,
        "aej_1": aej_step1,
        "aej_2": aej_step2,
        "aej_3": aej_step3,
        "aej_4": aej_step4
    }

def detect_sc(state, AEJ, w, gamma_sc, gauss_ksize, dark_search_depth, sigma, preproc_method="2",
              tex_beta=0.0):
    if not state or AEJ is None or w is None:
        return None
    H, W = state["H"], state["W"]

    if "2" in preproc_method:
        grad_map_sc = stage_cost_map(state, gamma_sc, tex_beta)
    else:
        ks = int(gauss_ksize) | 1
        sc_blurred = cv2.GaussianBlur(w, (ks, ks), 0)
        sc_img = np.power(sc_blurred, gamma_sc)
        grad_map_sc = np.abs(np.gradient(sc_img, axis=0))

    grad_norm_sc = (grad_map_sc / (grad_map_sc.max() + 1e-8) * 255).astype(np.uint8)

    depth = int(dark_search_depth)
    row_off = np.arange(depth).reshape(-1, 1)
    starts_sc = (AEJ + 2).reshape(1, -1)
    search_r = starts_sc + row_off
    valid_sc = (search_r < H)
    search_r_clip = np.clip(search_r, 0, H - 1)
    c_idx = np.broadcast_to(np.arange(W).reshape(1, -1), search_r.shape)
    sg = grad_map_sc[search_r_clip, c_idx]
    sg = np.where(valid_sc, sg, np.inf)
    SC_boundary = starts_sc.ravel() + np.argmin(sg, axis=0)
    SC_boundary = np.clip(SC_boundary, 0, H - 1)
    SC_boundary = gaussian_filter1d(SC_boundary.astype(np.float32), sigma=sigma).astype(int)

    xs = np.arange(W)
    sc_step1 = cv2.applyColorMap(grad_norm_sc, cv2.COLORMAP_HOT)
    sc_step1 = cv2.cvtColor(sc_step1, cv2.COLOR_BGR2RGB)
    pts_sc = np.column_stack([xs, SC_boundary]).astype(np.int32)
    sc_step2 = sc_step1.copy()
    cv2.polylines(sc_step2, [pts_sc], False, (255, 255, 0), 1, cv2.LINE_AA)

    pts_sc_top = np.column_stack([xs, AEJ + 2]).astype(np.int32)
    pts_sc_bot = np.column_stack([xs, np.clip(AEJ + depth, 0, H - 1)]).astype(np.int32)
    cv2.polylines(sc_step1, [pts_sc_top], False, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.polylines(sc_step1, [pts_sc_bot], False, (255, 255, 255), 1, cv2.LINE_AA)

    return {
        "SC_boundary": SC_boundary,
        "grad_map_sc": grad_map_sc,
        "grad_norm_sc": grad_norm_sc,
        "sc_1": sc_step1,
        "sc_2": sc_step2
    }

def detect_edj(state, w, SC_boundary, AEJ, gamma_edj, gauss_ksize, edj_offset, edj_window, search_bottom, sigma, edj_method="3", fuzzy_sigma=3.0, preproc_method="2", tex_beta=1.0):
    if not state or SC_boundary is None or AEJ is None or w is None:
        return None
    H, W = state["H"], state["W"]
    xs = np.arange(W)

    is_decomp = "2" in preproc_method
    start_edj = SC_boundary + int(edj_offset)
    ks = 3 if is_decomp else int(gauss_ksize) | 1

    if "3" in edj_method:
        search_bottom = int(search_bottom)
        external_cost = stage_cost_map(state, gamma_edj, tex_beta) if is_decomp else None
        EDJ_boundary, edj_step1, edj_step2 = detect_edj_graph(
            w, SC_boundary, edj_offset, search_bottom, H, W, sigma, fuzzy_sigma, external_cost
        )
        edj_step3 = None
    else:
        if is_decomp:
            grad_map_edj = stage_cost_map(state, gamma_edj, tex_beta)
        else:
            edj_blurred = cv2.GaussianBlur(w, (ks, ks), 0)
            edj_img = np.power(edj_blurred, gamma_edj)
            grad_map_edj = np.abs(np.gradient(edj_img, axis=0))

        if "1" in edj_method:
            grad_norm_edj = (grad_map_edj / (grad_map_edj.max() + 1e-8) * 255).astype(np.uint8)
            
            edj_win = int(edj_window)
            row_off = np.arange(edj_win).reshape(-1, 1)
            start_edj = AEJ + int(edj_offset)
            search_r_edj = start_edj.reshape(1, -1) + row_off
            valid_edj = (search_r_edj < H)
            search_r_edj_clip = np.clip(search_r_edj, 0, H - 1)
            col_idx_edj = np.broadcast_to(np.arange(W).reshape(1, -1), search_r_edj.shape)
            
            eg = grad_map_edj[search_r_edj_clip, col_idx_edj]
            eg = np.where(valid_edj, eg, -np.inf)
            
            EDJ_boundary = start_edj + np.argmax(eg, axis=0)
            EDJ_boundary = np.clip(EDJ_boundary, 0, H - 1)
            EDJ_boundary = gaussian_filter1d(EDJ_boundary.astype(np.float32), sigma=sigma).astype(int)
            
            pts_edj_top = np.column_stack([xs, start_edj]).astype(np.int32)
            pts_edj_bot = np.column_stack([xs, np.clip(start_edj + edj_win, 0, H - 1)]).astype(np.int32)
            
        else: # "2" in edj_method
            # grad_map_edj = cv2.GaussianBlur(grad_map_edj, (ks, 1), 0)
            grad_norm_edj = (grad_map_edj / (grad_map_edj.max() + 1e-8) * 255).astype(np.uint8)
            
            search_bottom = int(search_bottom)
            r_grid = np.arange(H).reshape(-1, 1)
            valid_edj = (r_grid >= start_edj.reshape(1, -1)) & (r_grid <= search_bottom)
            
            eg = np.where(valid_edj, grad_map_edj, np.inf)
            has_valid = np.any(valid_edj, axis=0)
            EDJ_boundary = np.where(has_valid, np.argmin(eg, axis=0), start_edj)
            EDJ_boundary = np.clip(EDJ_boundary, 0, H - 1)
            
            from scipy.signal import medfilt
            EDJ_boundary = medfilt(EDJ_boundary, kernel_size=ks).astype(int)
            EDJ_boundary = gaussian_filter1d(EDJ_boundary.astype(np.float32), sigma=sigma).astype(int)
            
            pts_edj_top = np.column_stack([xs, start_edj]).astype(np.int32)
            pts_edj_bot = np.column_stack([xs, np.full(W, search_bottom)]).astype(np.int32)

        edj_step1 = cv2.applyColorMap(grad_norm_edj, cv2.COLORMAP_HOT)
        edj_step1 = cv2.cvtColor(edj_step1, cv2.COLOR_BGR2RGB)
        edj_step2 = edj_step1.copy()
        cv2.polylines(edj_step1, [pts_edj_top], False, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.polylines(edj_step1, [pts_edj_bot], False, (0, 200, 255), 1, cv2.LINE_AA)

        pts_edj = np.column_stack([xs, EDJ_boundary]).astype(np.int32)
        cv2.polylines(edj_step2, [pts_edj], False, (57, 255, 20), 1, cv2.LINE_AA)
        edj_step3 = None
   
    pts_edj = np.column_stack([xs, EDJ_boundary]).astype(np.int32)
    edj_step4 = state["img_ori_rgb"].copy()
    cv2.polylines(edj_step4, [pts_edj], False, (57, 255, 20), 1, cv2.LINE_AA)

    return {
        "EDJ": EDJ_boundary,
        "edj_1": edj_step1,
        "edj_2": edj_step2,
        "edj_3": edj_step3,
        "edj_4": edj_step4
    }

def generate_overlays(state, AEJ, SC_boundary, EDJ):
    if not state or AEJ is None or SC_boundary is None or EDJ is None:
        return None
    H, W = state["H"], state["W"]
    xs = np.arange(W)
    pts_aej = np.column_stack([xs, AEJ]).astype(np.int32)
    pts_sc = np.column_stack([xs, SC_boundary]).astype(np.int32)
    pts_edj = np.column_stack([xs, EDJ]).astype(np.int32)
    all_pts = [pts_aej, pts_sc, pts_edj]
    all_colors = [(255, 77, 77), (255, 255, 0), (57, 255, 20)]

    result = state["img_ori_rgb"].copy()
    for pts, c in zip(all_pts, all_colors):
        cv2.polylines(result, [pts], False, c, 1, cv2.LINE_AA)

    rows = np.arange(H).reshape(-1, 1)
    aej_r = AEJ.reshape(1, -1)
    sc_r = SC_boundary.reshape(1, -1)
    edj_r = EDJ.reshape(1, -1)

    mask_sc = (rows >= aej_r) & (rows < sc_r)
    mask_epi = (rows >= sc_r) & (rows < edj_r)
    mask_dermis = (rows >= edj_r) & (rows < np.minimum(edj_r + 50, H))

    img_sc = _build_layer_img(state, mask_sc, [0, 200, 255], all_pts, all_colors)
    img_epi = _build_layer_img(state, mask_epi, [255, 100, 255], all_pts, all_colors)
    img_dermis = _build_layer_img(state, mask_dermis, [50, 255, 100], all_pts, all_colors)

    combined = state["img_ori_rgb"].copy().astype(np.float32)
    for mask, color in [(mask_sc, [0, 200, 255]), (mask_epi, [255, 100, 255]), (mask_dermis, [50, 255, 100])]:
        ov = np.zeros_like(combined)
        ov[mask] = color
        combined[mask] = combined[mask] * 0.55 + ov[mask] * 0.45
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    for pts, c in zip(all_pts, all_colors):
        cv2.polylines(combined, [pts], False, c, 1, cv2.LINE_AA)

    return {
        "result": result,
        "combined": combined,
        "layer_sc": img_sc,
        "layer_epidermis": img_epi,
        "layer_dermis": img_dermis
    }


def save_results(state, results_dict, output_dir="results"):
    if not state or not results_dict:
        return "⚠️ Không có dữ liệu để lưu."
        
    img_stem = os.path.splitext(state["name"])[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(output_dir, f"{img_stem}_{stamp}")
    os.makedirs(folder, exist_ok=True)

    names = ["boundaries", "clahe", "combined", "layer_sc", "layer_epidermis", "layer_dermis"]
    keys = ["result", "clahe", "combined", "layer_sc", "layer_epidermis", "layer_dermis"]
    
    saved_count = 0
    for name, key in zip(names, keys):
        if key in results_dict:
            img = results_dict[key]
            # Convert RGB to BGR for OpenCV saving
            cv2.imwrite(os.path.join(folder, f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved_count += 1

    return f"✅ Đã lưu {saved_count} ảnh vào: {folder}/"
