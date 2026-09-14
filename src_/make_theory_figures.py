"""Sinh hình minh hoạ cho tài liệu lý thuyết từ chính pipeline đang chạy.

Chạy: python src/make_theory_figures.py [đường_dẫn_ảnh]
Kết quả ghi vào reports/figures/.
"""
import os
import sys
import glob

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oct_algo import load_image_state, get_decomposition, stage_cost_map, detect_aej, detect_sc, detect_edj
from oct_denoise import to_additive_domain, structure_tensor_coherence
from oct_cost import texture_energy, _normalize

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")
TV_WEIGHT, THRESH_SCALE, COH_THRESH, TEX_WIN = 0.1, 1.0, 0.3, 9
GAMMA, BETA = 3.5, 1.0

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titlesize": 10})


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def _show(ax, img, title, cmap="gray"):
    ax.imshow(img, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def fig_multiplicative(img):
    """Kiểm chứng miền nào làm speckle cộng tính trên dữ liệu thực tế.

    Nếu speckle là nhân, biên độ phần dư phải tỉ lệ với cường độ nền; nếu đã
    cộng tính, biên độ phải phẳng. Phần dư được lấy đúng cách pipeline làm
    (TV tách cấu trúc nền), một lần trên ảnh .pgm như hiện có và một lần sau
    khi áp thêm log.
    """
    from skimage.restoration import denoise_tv_chambolle

    lin = img.astype(np.float32) / 255.0
    log = np.log1p(lin * 255.0) / np.log(256.0)
    Us_lin = denoise_tv_chambolle(lin, weight=TV_WEIGHT).astype(np.float32)
    Us_log = denoise_tv_chambolle(log, weight=TV_WEIGHT).astype(np.float32)

    def binned_spread(residual, level, nbins=24):
        lo, hi = np.percentile(level, [2, 98])
        bins = np.linspace(lo, hi, nbins)
        idx = np.digitize(level.ravel(), bins)
        r = np.abs(residual.ravel())
        bx, by = [], []
        for i in range(1, len(bins)):
            s = r[idx == i]
            if s.size > 200:
                bx.append(0.5 * (bins[i - 1] + bins[i]))
                by.append(1.4826 * np.median(s))
        return np.array(bx), np.array(by)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    panels = [(lin - Us_lin, "Ảnh .pgm như hiện có\n(đã log ở khâu chuyển đổi)", "#2563eb"),
              (log - Us_log, "Sau khi áp thêm một phép log\n(log kép)", "#dc2626")]
    for ax, (res, name, color) in zip(axes, panels):
        bx, by = binned_spread(res, Us_lin)
        byn = by / by.mean()
        ax.plot(bx, byn, "o-", color=color, ms=4)
        verdict = "cộng tính ✔" if (byn.max() - byn.min()) < 0.25 else "phụ thuộc cường độ ✘"
        ax.set_title(f"{name}\nbiến thiên {(byn.max() - byn.min()) * 100:.0f}% — {verdict}")
        ax.set_xlabel("Cường độ mô nền")
        ax.set_ylabel("Biên độ phần dư (chuẩn hoá)")
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_ylim(0, 2.0)
    fig.suptitle("Điều kiện cộng tính đã được thoả sẵn: dữ liệu .pgm vốn đã nén log,\n"
                 "áp log lần nữa lại tạo ra sự phụ thuộc $1/I$ ngược lại", y=1.1)
    _save(fig, "fig1_multiplicative.png")


def fig_components(img, d):
    """Ba thành phần của phân rã log-domain."""
    base = to_additive_domain(img)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    _show(axes[0, 0], base, r"$\log I$ — ảnh đầu vào (đã ở miền log)")
    _show(axes[0, 1], d["Us_log"], r"$U_s$ — cấu trúc nền (TV)")
    _show(axes[1, 0], d["Ui"], r"$U_i$ — speckle mang thông tin")
    _show(axes[1, 1], d["Ur"], r"$U_r$ — nhiễu ngẫu nhiên (loại bỏ)")
    err = np.abs(d["Us_log"] + d["Ui"] + d["Ur"] - base).max()
    fig.suptitle(rf"Phân rã $\log I = U_s + U_i + U_r$   (sai số tái tạo {err:.1e})", y=0.99)
    fig.tight_layout()
    _save(fig, "fig2_components.png")


def fig_split_criterion(d):
    """Tiêu chí tách Ui/Ur: coherence định hướng + phân bố phần dư."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    im = axes[0].imshow(d["coherence"], cmap="magma", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Độ kết dính định hướng của $V$")
    axes[0].set_xticks([]); axes[0].set_yticks([]); axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.03)

    names = [("$V$ (phần dư)", d["Ui"] + d["Ur"], "#64748b"),
             ("$U_i$ (giữ lại)", d["Ui"], "#2563eb"),
             ("$U_r$ (vứt bỏ)", d["Ur"], "#dc2626")]
    vals = [structure_tensor_coherence(a).mean() for _, a, _ in names]
    axes[1].bar([n for n, _, _ in names], vals, color=[c for _, _, c in names])
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_ylabel("Coherence trung bình")
    axes[1].set_title("$U_i$ giữ phần có hướng,\n$U_r$ đẳng hướng hơn hẳn")
    axes[1].set_ylim(0, max(vals) * 1.25)

    z = (d["Ur"].ravel() - d["Ur"].mean()) / d["Ur"].std()
    axes[2].hist(z, bins=200, range=(-6, 6), density=True, color="#dc2626",
                 alpha=0.65, label="$U_r$ thực tế")
    xs = np.linspace(-6, 6, 300)
    axes[2].plot(xs, stats.norm.pdf(xs), "k--", lw=1.4, label="Gauss lý tưởng")
    axes[2].set_yscale("log")
    axes[2].set_title(f"Phân bố $U_r$\nskew={stats.skew(z):+.2f}  kurt={stats.kurtosis(z):+.1f}")
    axes[2].set_xlabel("Độ lệch chuẩn hoá")
    axes[2].legend(fontsize=8)

    fig.suptitle("Tiêu chí tách $U_i$ / $U_r$ và kiểm chứng phần bị loại bỏ", y=1.04)
    fig.tight_layout()
    _save(fig, "fig3_split_criterion.png")


def fig_channels(d):
    """Hai kênh của cost map."""
    G_base = np.abs(np.gradient(np.power(np.clip(d["Us_log"], 0, None), GAMMA), axis=0))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    _show(axes[0], texture_energy(d["Ui"], TEX_WIN), "Năng lượng speckle cục bộ", "viridis")
    _show(axes[1], G_base, r"$G_{base} = |\partial(U_s^\gamma)/\partial y|$", "hot")
    _show(axes[2], d["G_texture"], r"$G_{texture}$ — chuyển tiếp kết cấu", "hot")
    fig.suptitle("Hai kênh độc lập của bản đồ chi phí", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_channels.png")


def fig_core_fix(st, d):
    """So sánh cách tổng hợp sai của v1 với cost map hai kênh của v2."""
    I_v1 = d["Us_log"] + 1.0 * d["Ui"]
    G_v1 = np.abs(np.gradient(np.power(np.clip(I_v1, 0, None), GAMMA), axis=0))
    G_v2 = stage_cost_map(st, GAMMA, BETA)

    def npeaks(G):
        Gn = G / (G.max() + 1e-8)
        return np.mean([len(find_peaks(Gn[:, c], height=0.05)[0])
                        for c in range(0, G.shape[1], 10)])

    n1, n2 = npeaks(G_v1), npeaks(G_v2)
    col = G_v2.shape[1] // 2

    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, 0])
    _show(ax, G_v1 / (G_v1.max() + 1e-8), f"v1: $U_s + \\alpha U_i$ rồi lấy đạo hàm\n"
                                          f"{n1:.1f} đỉnh giả / cột", "hot")
    ax.axvline(col, color="#38bdf8", lw=1.2)
    ax = fig.add_subplot(gs[0, 1])
    _show(ax, G_v2 / (G_v2.max() + 1e-8), f"v2: $G_{{base}} + \\beta G_{{texture}}$\n"
                                          f"{n2:.1f} đỉnh giả / cột", "hot")
    ax.axvline(col, color="#38bdf8", lw=1.2)

    for i, (G, name, color) in enumerate([(G_v1, "v1", "#dc2626"), (G_v2, "v2", "#2563eb")]):
        ax = fig.add_subplot(gs[1, i])
        p = G[:, col] / (G[:, col].max() + 1e-8)
        pk, _ = find_peaks(p, height=0.05)
        ax.plot(p, color=color, lw=0.9)
        ax.plot(pk, p[pk], "v", color="k", ms=3.5, label=f"{len(pk)} đỉnh")
        ax.set_title(f"{name} — mặt cắt chi phí tại cột {col}")
        ax.set_xlabel("Độ sâu (pixel)")
        ax.set_ylabel("Chi phí (chuẩn hoá)")
        ax.legend(fontsize=8)

    fig.suptitle("Lỗi cốt lõi được sửa: cộng $U_i$ vào ảnh tạo ra rừng đỉnh giả tại từng hạt speckle",
                 y=1.0)
    fig.tight_layout()
    _save(fig, "fig5_core_fix.png")
    return n1, n2


def fig_beta_sweep(st):
    """Ảnh hưởng của trọng số beta lên cost map và đường EDJ."""
    betas = [0.0, 1.0, 3.0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for j, b in enumerate(betas):
        G = stage_cost_map(st, GAMMA, b)
        _show(axes[0, j], G / (G.max() + 1e-8), rf"$G_{{total}}$ với $\beta$ = {b}", "hot")

        r = detect_aej(st, 100, 280, GAMMA, 3, 41, 0.2, "2", TV_WEIGHT, b,
                       THRESH_SCALE, True, COH_THRESH, TEX_WIN)
        st.update(r)
        st.update(detect_sc(st, st["AEJ"], st["w"], GAMMA, 41, 18, 3, "2", b))
        st.update(detect_edj(st, st["w"], st["SC_boundary"], st["AEJ"], GAMMA, 41,
                             10, 50, 280, 3, "1", 5.0, "2", b))
        axes[1, j].imshow(st["img_ori_rgb"], aspect="auto")
        xs = np.arange(st["W"])
        for arr, c, lb in [(st["AEJ"], "#ff4d4d", "AEJ"), (st["SC_boundary"], "#ffff00", "SC"),
                           (st["EDJ"], "#39ff14", "EDJ")]:
            axes[1, j].plot(xs, arr, color=c, lw=1.0, label=lb)
        axes[1, j].set_title(rf"Ranh giới với $\beta$ = {b}")
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([]); axes[1, j].grid(False)
        if j == 0:
            axes[1, j].legend(fontsize=7, loc="lower right")
    fig.suptitle(r"$\beta$ điều chỉnh mức đóng góp của kênh chuyển tiếp kết cấu", y=1.0)
    fig.tight_layout()
    _save(fig, "fig6_beta_sweep.png")


def fig_branch_compare(st):
    """Kết quả dò biên của hai nhánh tiền xử lý."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, (m, name) in zip(axes, [("1", "Nhánh 1 — Truyền thống (NLM + Median)"),
                                    ("2", "Nhánh 2 — Phân rã log-domain")]):
        r = detect_aej(st, 100, 280, GAMMA, 3, 41, 0.2, m, TV_WEIGHT, BETA,
                       THRESH_SCALE, True, COH_THRESH, TEX_WIN)
        st.update(r)
        st.update(detect_sc(st, st["AEJ"], st["w"], GAMMA, 41, 18, 3, m, BETA))
        st.update(detect_edj(st, st["w"], st["SC_boundary"], st["AEJ"], GAMMA, 41,
                             10, 50, 280, 3, "1", 5.0, m, BETA))
        ax.imshow(st["img_ori_rgb"], aspect="auto")
        xs = np.arange(st["W"])
        for arr, c, lb in [(st["AEJ"], "#ff4d4d", "AEJ"), (st["SC_boundary"], "#ffff00", "SC"),
                           (st["EDJ"], "#39ff14", "EDJ")]:
            ax.plot(xs, arr, color=c, lw=1.0, label=lb)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig7_branch_compare.png")


def fig_speckle_zoom(st):
    """Kết cấu speckle ở mức phóng to và cái giá của việc làm mịn mạnh."""
    ori = st["img_ori"]
    H, W = ori.shape
    r0, r1, c0, c1 = 190, 330, 420, 640

    smoothed = cv2.GaussianBlur(st["img_clahe"].astype(np.float32) / 255.0, (41, 41), 0)

    fig = plt.figure(figsize=(12, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])

    ax = fig.add_subplot(gs[0])
    ax.imshow(ori, cmap="gray", aspect="auto")
    ax.add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, ec="#ef4444", fc="none", lw=1.8))
    ax.set_title("Ảnh B-scan đầy đủ")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    _show(fig.add_subplot(gs[1]), ori[r0:r1, c0:c1],
          "Phóng to: hạt speckle\n(cấu trúc vi mô còn nguyên)")
    _show(fig.add_subplot(gs[2]), smoothed[r0:r1, c0:c1],
          "Sau NLM + Gauss $ks{=}41$\n(cấu trúc vi mô bị xoá)")

    fig.tight_layout()
    _save(fig, "fig9_speckle_zoom.png")


def fig_metrics(st, d):
    """Bản đồ chi phí là gì, và hai chỉ số đọc trên mặt cắt một cột."""
    G = np.abs(np.gradient(np.power(np.clip(d["Us_log"], 0, None), GAMMA), axis=0))
    col = G.shape[1] // 2
    p = G[:, col] / (G[:, col].max() + 1e-8)

    pk, _ = find_peaks(p, height=0.05)
    i = int(np.argmax(p))
    half = p[i] / 2
    l = r = i
    while l > 0 and p[l] > half:
        l -= 1
    while r < len(p) - 1 and p[r] > half:
        r += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    ax.imshow(G / (G.max() + 1e-8), cmap="hot", aspect="auto")
    ax.axvline(col, color="#38bdf8", lw=1.6)
    ax.set_title("Bản đồ chi phí\n(sáng = khả năng có ranh giới cao)")
    ax.set_xlabel("Vị trí quét")
    ax.set_ylabel("Độ sâu")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.annotate(f"cột {col}", xy=(col, 30), xytext=(col + 90, 60), fontsize=8,
                color="#38bdf8", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.2))

    ax = axes[1]
    ax.plot(p, color="#2563eb", lw=1.2, label=f"Mặt cắt chi phí dọc cột {col}")
    ax.axhline(0.05, color="#dc2626", ls="--", lw=1, label="Ngưỡng đếm đỉnh (5% biên độ)")
    ax.plot(pk, p[pk], "v", color="#dc2626", ms=6,
            label=f"Đỉnh vượt ngưỡng ({len(pk)} ứng viên)")

    ax.annotate("", xy=(l, half), xytext=(r, half),
                arrowprops=dict(arrowstyle="<->", color="#16a34a", lw=2))
    ax.plot([i], [p[i]], "o", color="#16a34a", ms=7)
    ax.annotate(f"FWHM = {r - l} px", xy=((l + r) / 2, half), xytext=(r + 55, half + 0.14),
                fontsize=9, color="#16a34a", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1))
    ax.annotate("đỉnh cao nhất\n$\\to$ argmax chọn điểm này", xy=(i, p[i]),
                xytext=(i + 60, 0.86), fontsize=8.5, color="#16a34a",
                arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1))

    ax.set_xlabel("Độ sâu (pixel)")
    ax.set_ylabel("Chi phí (chuẩn hoá)")
    ax.set_title("Mặt cắt một cột: mỗi đỉnh là một ứng viên ranh giới")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    _save(fig, "fig10_metrics.png")


def fig_pareto(st, d):
    """Đánh đổi nhiễu/độ sắc nét: lọc Gauss truyền thống so với phân rã.

    Lọc tuyến tính chỉ có một núm xoay (bán kính) và núm đó buộc phải chọn giữa
    bản đồ chi phí sạch và biên sắc nét. Phân rã biến phân không nằm trên đường
    đánh đổi đó.
    """
    from oct_algo import prepare_work

    w = prepare_work(st, 0.2, True)

    def n_peaks(M):
        Mn = M / (M.max() + 1e-8)
        return np.mean([len(find_peaks(Mn[:, c], height=0.05)[0])
                        for c in range(0, M.shape[1], 10)])

    def fwhm(M):
        ws = []
        for c in range(0, M.shape[1], 20):
            p = M[:, c]
            i = int(np.argmax(p))
            half = p[i] / 2
            l = r = i
            while l > 0 and p[l] > half:
                l -= 1
            while r < len(p) - 1 and p[r] > half:
                r += 1
            ws.append(r - l)
        return np.mean(ws)

    def trad_cost(ks):
        return np.abs(np.gradient(np.power(cv2.GaussianBlur(w, (ks, ks), 0), GAMMA), axis=0))

    ks_list = [3, 7, 11, 17, 25, 33, 41, 51, 61]
    tx = [fwhm(trad_cost(k)) for k in ks_list]
    ty = [n_peaks(trad_cost(k)) for k in ks_list]

    G_base = np.abs(np.gradient(np.power(np.clip(d["Us_log"], 0, None), GAMMA), axis=0))
    bx, by = fwhm(G_base), n_peaks(G_base)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(tx, ty, "o-", color="#dc2626", ms=5, lw=1.5,
            label="Lọc Gauss truyền thống (quét bán kính)")
    for k, x, y in zip(ks_list, tx, ty):
        ax.annotate(f"ks={k}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                    color="#dc2626")
    ax.plot([bx], [by], "*", color="#2563eb", ms=20, zorder=5,
            label=r"Phân rã biến phân ($G_{base}$)")
    ax.annotate(rf"$G_{{base}}$: {bx:.1f} px, {by:.1f} đỉnh", (bx, by),
                textcoords="offset points", xytext=(12, -14), fontsize=9, color="#2563eb",
                fontweight="bold")

    ax.axvline(bx, color="#2563eb", ls=":", lw=1, alpha=0.5)
    ax.axhline(by, color="#2563eb", ls=":", lw=1, alpha=0.5)
    ax.fill_between([0, bx], 0, by, color="#2563eb", alpha=0.06)
    ax.text(bx * 0.52, by * 0.45, "vùng tốt hơn\ntrên cả hai trục", fontsize=8.5,
            color="#2563eb", ha="center", style="italic")

    ax.set_xlabel("Bề rộng đỉnh tại biên AEJ — FWHM (px)   ←  sắc nét hơn")
    ax.set_ylabel("Số đỉnh giả trên mỗi cột   ←  sạch hơn")
    ax.set_title("Lọc tuyến tính buộc phải đánh đổi sạch ↔ sắc nét;\n"
                 "phân rã biến phân đạt cả hai cùng lúc")
    ax.set_xlim(3, 20)
    ax.set_ylim(3.5, 12)
    ax.legend(fontsize=8.5, loc="upper right")
    _save(fig, "fig8_pareto.png")
    return list(zip(ks_list, tx, ty)), (bx, by)


def _dp_path(cost_map, r_starts, r_ends, H, W):
    """Đường đi ngắn nhất theo cột, liên thông 3 hướng — trả về đường THÔ chưa làm mượt."""
    INF = 1e9
    dp = np.full((H, W), INF)
    parent = np.zeros((H, W), dtype=np.int32)
    for r in range(r_starts[0], r_ends[0]):
        dp[r, 0] = cost_map[r, 0]
    for c in range(1, W):
        rs, re = r_starts[c], r_ends[c]
        for r in range(rs, re):
            best, bp = INF, r
            for dr in (-1, 0, 1):
                pr = r + dr
                if rs <= pr < re and dp[pr, c - 1] < best:
                    best, bp = dp[pr, c - 1], pr
            dp[r, c] = best + cost_map[r, c]
            parent[r, c] = bp
    rs_l, re_l = r_starts[W - 1], r_ends[W - 1]
    path = np.zeros(W, dtype=np.int32)
    path[W - 1] = rs_l + np.argmin(dp[rs_l:re_l, W - 1])
    for c in range(W - 2, -1, -1):
        path[c] = parent[path[c + 1], c + 1]
    return path


def fig_boundary_algorithms(st, d):
    """Ba thuật toán dò EDJ: quyết định theo cột độc lập vs tối ưu toàn cục."""
    H, W = st["H"], st["W"]
    G = stage_cost_map(st, GAMMA, BETA)
    AEJ, SC = st["AEJ"], st["SC_boundary"]
    off, win, bottom, sig = 10, 50, 280, 3.0

    # PP1 — argmax theo cột, ROI neo vào AEJ
    start1 = np.clip(AEJ + off, 0, H - 1)
    rows = np.arange(win).reshape(-1, 1) + start1.reshape(1, -1)
    valid = rows < H
    vals = np.where(valid, G[np.clip(rows, 0, H - 1), np.arange(W)], -np.inf)
    raw1 = start1 + np.argmax(vals, axis=0)

    # PP2 — argmin theo cột, ROI neo vào SC
    start2 = np.clip(SC + off, 0, H - 1)
    rg = np.arange(H).reshape(-1, 1)
    m2 = (rg >= start2.reshape(1, -1)) & (rg <= bottom)
    raw2 = np.where(m2.any(axis=0), np.argmin(np.where(m2, G, np.inf), axis=0), start2)

    # PP3 — đường đi ngắn nhất toàn cục, cùng ROI với PP2
    r_s = np.clip(SC + off, 0, H - 1)
    r_e = np.clip(np.maximum(r_s + 1, np.full(W, bottom)), 0, H)
    raw3 = _dp_path(1.0 - G / (G.max() + 1e-8), r_s, r_e, H, W)

    sm = lambda b, s: gaussian_filter1d(b.astype(np.float32), sigma=s)
    b1, b2, b3 = sm(raw1, sig), sm(raw2, sig), sm(raw3, 5.0)

    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.28, wspace=0.22)

    ori = st["img_ori"]
    specs = [(raw1, b1, "PP1 — Gradient Max (argmax)", "#ff4d4d", start1, start1 + win),
             (raw2, b2, "PP2 — Gradient Min (argmin)", "#ffd400", start2, np.full(W, bottom)),
             (raw3, b3, "PP3 — Đường đi ngắn nhất (DP)", "#39ff14", r_s, r_e - 1)]
    for k, (raw, sm_b, title, col, top, bot) in enumerate(specs):
        ax = fig.add_subplot(gs[0, k])
        _show(ax, ori, title)
        ax.plot(top, color="w", lw=0.7, alpha=0.55)
        ax.plot(bot, color="w", lw=0.7, alpha=0.55)
        ax.plot(raw, color=col, lw=0.6, alpha=0.45, label="thô (chưa làm mượt)")
        ax.plot(sm_b, color=col, lw=1.6, label="sau khi làm mượt")
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)

    # Bước nhảy giữa hai cột liền kề — thước đo tính liên tục
    ax = fig.add_subplot(gs[1, 0])
    jumps = [np.abs(np.diff(r)) for r in (raw1, raw2, raw3)]
    bins = np.arange(0, 26)
    for j, lbl, col in zip(jumps, ["PP1 argmax", "PP2 argmin", "PP3 DP"],
                           ["#ff4d4d", "#e0a800", "#2eb82e"]):
        ax.hist(np.clip(j, 0, 25), bins=bins, alpha=0.55, label=f"{lbl} (max={j.max():.0f})",
                color=col, density=True)
    ax.set_yscale("log")
    ax.set_xlabel("|y(x+1) − y(x)|  (pixel)")
    ax.set_ylabel("mật độ (log)")
    ax.set_title("Bước nhảy giữa hai cột liền kề")
    ax.legend(fontsize=8)

    # Mặt cắt một cột: argmax chọn gì
    ax = fig.add_subplot(gs[1, 1])
    c = W // 2
    lo, hi = int(start1[c]), int(min(start1[c] + win, H))
    prof = G[lo:hi, c]
    ax.plot(np.arange(lo, hi), prof, color="#333", lw=1.2)
    ax.axvline(raw1[c], color="#ff4d4d", ls="--", lw=1.4, label=f"argmax → y={raw1[c]}")
    pk, _ = find_peaks(prof / (prof.max() + 1e-8), height=0.35)
    ax.plot(pk + lo, prof[pk], "o", ms=4, color="#ff4d4d", alpha=0.6,
            label=f"{len(pk)} đỉnh cạnh tranh ≥35%")
    ax.set_xlabel("y (độ sâu, pixel)")
    ax.set_ylabel("$G_{total}$")
    ax.set_title(f"Mặt cắt cột x={c}: quyết định của PP1")
    ax.legend(fontsize=8)

    # Giá của ràng buộc liên tục: cùng ROI, cùng hàm mục tiêu của PP3
    ax = fig.add_subplot(gs[1, 2])
    Gn = G / (G.max() + 1e-8)
    C = 1.0 - Gn
    rg2 = np.arange(H).reshape(-1, 1)
    roi3 = (rg2 >= r_s.reshape(1, -1)) & (rg2 < r_e.reshape(1, -1))
    free = np.argmin(np.where(roi3, C, np.inf), axis=0)      # tối ưu từng cột, không ràng buộc
    cost_of = lambda b: C[np.clip(b.astype(int), 0, H - 1), np.arange(W)].sum()
    c_free, c_dp = cost_of(free), cost_of(raw3)
    costs = [c_free, c_dp]
    bars = ax.bar(["Tối ưu từng cột\n(không ràng buộc)", "PP3 — DP\n(|Δy| ≤ 1)"], costs,
                  color=["#9aa0a6", "#2eb82e"], alpha=0.85)
    for b, v in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"$\sum_x\, [1-\hat{G}(x,y(x))]$")
    ax.set_title("Giá của ràng buộc liên tục\n(cùng ROI, cùng hàm mục tiêu)")
    ax.grid(axis="x", alpha=0)
    ax.text(0.5, 0.06, f"chênh {100*(c_dp-c_free)/c_free:.1f}%  |  bước nhảy tối đa: "
            f"{np.abs(np.diff(free)).max():.0f} px  vs  {np.abs(np.diff(raw3)).max():.0f} px",
            transform=ax.transAxes, ha="center", fontsize=8,
            bbox=dict(fc="white", ec="#ccc", alpha=0.9))

    _save(fig, "fig11_boundary_algorithms.png")
    return [int(x.max()) for x in jumps], costs


def fig_beta_per_stage(st, d, path):
    """Vì sao beta=0 tốt nhất cho AEJ: texture không có tín hiệu ở đó, và G_base bị clip."""
    import glob as _glob
    folder = os.path.dirname(path)
    paths = sorted(_glob.glob(os.path.join(folder, "*.pgm")))[:4] or [path]

    # (a) Tương phản cường độ vs texture qua từng ranh giới, trung bình nhiều ảnh
    acc = {k: [[], []] for k in ("AEJ", "SC", "EDJ")}
    sat_acc = []
    for p in paths:
        s2 = load_image_state(p) if p != path else st
        d2 = get_decomposition(s2, TV_WEIGHT, THRESH_SCALE, COH_THRESH, TEX_WIN)
        H2, W2 = s2["H"], s2["W"]
        r = detect_aej(s2, 100, 280, GAMMA, 3, 41, 0.2, preproc_method="2", tv_weight=TV_WEIGHT,
                       tex_beta=0.0, thresh_scale=THRESH_SCALE, use_clahe=True,
                       coh_thresh=COH_THRESH, tex_win=TEX_WIN)
        s2.update(r)
        sc = detect_sc(s2, r["AEJ"], s2["w"], GAMMA, 41, 18, 3, "2", 0.0)
        ed = detect_edj(s2, s2["w"], sc["SC_boundary"], r["AEJ"], GAMMA, 41, 10, 50, 280, 3,
                        "1", 5.0, "2", 0.0)
        te = texture_energy(d2["Ui"], TEX_WIN)
        ori = s2["img_ori"].astype(np.float32)
        cols = np.arange(0, W2, 3)

        def contrast(ref, chan, w=15):
            a = np.mean([chan[max(0, ref[c] - w - 3):ref[c] - 3, c].mean() for c in cols])
            b = np.mean([chan[ref[c] + 3:ref[c] + w + 3, c].mean() for c in cols])
            return 100 * (b - a) / abs(a)

        for k, ref in (("AEJ", r["AEJ"]), ("SC", sc["SC_boundary"]), ("EDJ", ed["EDJ"])):
            acc[k][0].append(contrast(ref, ori))
            acc[k][1].append(contrast(ref, te))
        nb2 = _normalize(np.abs(np.gradient(np.power(np.clip(d2["Us_log"], 0, None), GAMMA), axis=0)))
        sat_acc.append(100 * (nb2[np.clip(r["AEJ"], 0, H2 - 1), np.arange(W2)] >= 0.999).mean())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    fig.subplots_adjust(hspace=0.32, wspace=0.24)

    ax = axes[0, 0]
    keys = ["AEJ", "SC", "EDJ"]
    ci = [abs(np.mean(acc[k][0])) for k in keys]
    tx = [abs(np.mean(acc[k][1])) for k in keys]
    x = np.arange(3); wbar = 0.36
    ax.bar(x - wbar/2, ci, wbar, label="Tương phản cường độ", color="#4c78a8")
    ax.bar(x + wbar/2, tx, wbar, label="Tương phản texture", color="#e45756")
    for i, (a, b) in enumerate(zip(ci, tx)):
        ax.text(i, max(a, b) * 1.06, f"tex/cđ = {b/max(a,1e-9):.2f}", ha="center", fontsize=8.5,
                bbox=dict(fc="white", ec="#ccc", alpha=0.9))
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylabel("|tương phản| qua ranh giới (%)")
    ax.set_title(f"(a) Kênh nào mang tín hiệu? (trung bình {len(paths)} ảnh)")
    ax.legend(fontsize=8.5)

    ax = axes[0, 1]
    nb = _normalize(np.abs(np.gradient(np.power(np.clip(d["Us_log"], 0, None), GAMMA), axis=0)))
    nt = _normalize(d["G_texture"])
    H, W = st["H"], st["W"]
    A0 = np.argmax(nb[100:280], axis=0) + 100
    c = W // 2
    yy = np.arange(100, 280)
    ax.plot(yy, nb[100:280, c], color="#4c78a8", lw=1.5, label=r"$\hat{G}_{base}$")
    ax.plot(yy, nt[100:280, c], color="#e45756", lw=1.2, alpha=0.85, label=r"$\hat{G}_{texture}$")
    ax.axhline(1.0, color="#4c78a8", ls=":", lw=1)
    ax.axvline(A0[c], color="k", ls="--", lw=1.2, label=f"AEJ (β=0) = {A0[c]}")
    ax.set_xlabel("y (độ sâu, pixel)"); ax.set_ylabel("giá trị đã chuẩn hoá")
    ax.set_title(f"(b) Cột x={c}: $G_{{base}}$ bị kẹp ở 1.0 → mất sức phân biệt")
    ax.legend(fontsize=8.5)

    ax = axes[1, 0]
    Gb = np.abs(np.gradient(np.power(np.clip(d["Us_log"], 0, None), GAMMA), axis=0))
    Gt = d["G_texture"]
    betas = np.arange(0, 2.01, 0.25)
    for lbl, f, col in [("phân vị 99 + clip (hiện tại)", lambda z: np.clip(z/np.percentile(z,99),0,1), "#e45756"),
                        ("phân vị 99, không clip", lambda z: z/np.percentile(z,99), "#4c78a8")]:
        nb_, nt_ = f(Gb), f(Gt)
        a0 = np.argmax(nb_[100:280], axis=0) + 100
        pct = [100*(np.argmax((nb_+b*nt_)[100:280],axis=0)+100 != a0).mean() for b in betas]
        ax.plot(betas, pct, "o-", color=col, lw=1.8, ms=4, label=lbl)
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("% cột có AEJ bị đổi")
    ax.set_title("(c) β>0 làm AEJ mất ổn định — clip khuếch đại mạnh")
    ax.legend(fontsize=8.5)

    ax = axes[1, 1]
    sat = np.mean(sat_acc)
    roi = _normalize(d["G_texture"])[100:160]
    txt = (f"Bão hoà $G_{{base}}$ trên đường AEJ: {sat:.0f}%\n"
           f"(đỉnh gradient thô: {100*(nb[np.clip(A0,0,H-1),np.arange(W)]>=0.999).mean():.0f}%)\n\n"
           f"$G_{{texture}}$ trong dải tìm AEJ (dòng 100–160):\n"
           f"   trung bình {roi.mean():.3f}, độ lệch chuẩn {roi.std():.3f}\n"
           f"   tỉ lệ đỉnh/nền = {roi.max()/roi.mean():.2f}  → gần như phẳng\n\n"
           f"Trong dải tìm AEJ:\n"
           f"   std($\\hat{{G}}_{{base}}$)    = {nb[100:280].std():.3f}\n"
           f"   std($\\hat{{G}}_{{texture}}$) = {nt[100:280].std():.3f}\n"
           f"   → tại β=1 kênh texture lấn át kênh nền")
    ax.axis("off")
    ax.text(0.02, 0.97, txt, va="top", ha="left", fontsize=10.5, linespacing=1.5,
            bbox=dict(fc="#f7f7f7", ec="#bbb", pad=10))
    ax.set_title("(d) Hai cơ chế cộng hưởng")

    _save(fig, "fig12_beta_per_stage.png")
    return {k: (np.mean(acc[k][0]), np.mean(acc[k][1])) for k in keys}, sat


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("data/outside-hand/out-pgm/*.pgm"))[0]

    st = load_image_state(path)
    d = get_decomposition(st, TV_WEIGHT, THRESH_SCALE, COH_THRESH, TEX_WIN)
    print(f"Sinh hình vào {OUT_DIR}/")

    fig_multiplicative(st["img_ori"])
    fig_components(st["img_ori"], d)
    fig_split_criterion(d)
    fig_channels(d)
    n1, n2 = fig_core_fix(st, d)
    fig_beta_sweep(st)
    fig_branch_compare(st)
    fig_speckle_zoom(st)
    fig_metrics(st, d)
    fig_pareto(st, d)
    jmax, jcosts = fig_boundary_algorithms(st, d)
    contr, sat = fig_beta_per_stage(st, d, path)

    print(f"\nĐỉnh giả/cột: v1 = {n1:.1f}  ->  v2 = {n2:.1f}")
    print(f"Bước nhảy tối đa PP1/PP2/PP3: {jmax}   chi phí tích luỹ: {[round(c) for c in jcosts]}")
    for k,(a,b) in contr.items():
        print(f"  {k}: cuong do {a:+.1f}%  texture {b:+.1f}%  ty le {abs(b)/max(abs(a),1e-9):.2f}")
    print(f"Bao hoa G_base tren AEJ: {sat:.1f}%")
    print(f"sigma_mad = {d['sigma_mad']:.5f}   tau trung bình = {d['tau']:.5f}")


if __name__ == "__main__":
    main()
