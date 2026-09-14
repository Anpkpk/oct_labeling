import numpy as np
import cv2
from skimage.restoration import denoise_tv_chambolle
import pywt

LOG_SCALE = np.log(256.0)

# Ảnh .pgm trong data/ đã được nén log sẵn: src/text2pgm-convert.cpp gọi
# std::log1p trên dữ liệu thô 12-bit trước khi chuẩn hoá phân vị và ghi file.
# Speckle vì thế đã cộng tính ngay trong ảnh đầu vào; áp log lần nữa là log kép,
# nó tạo ra sự phụ thuộc 1/I ngược lại vào cường độ (đo được: biên độ phần dư
# biến thiên 11% -> 59% theo mức cường độ). Đặt False cho ảnh cường độ tuyến tính.
INPUT_ALREADY_LOG = True


def to_additive_domain(img, input_is_log=INPUT_ALREADY_LOG):
    """Đưa ảnh về miền mà speckle mang tính cộng, chuẩn hoá [0, 1].

    Speckle OCT mang bản chất nhân (I = U_true * n_speckle + n_add), nên phân rã
    cộng chỉ hợp lệ sau phép log. Điều kiện đó có thể đã được thoả sẵn từ khâu
    tạo dữ liệu — xem INPUT_ALREADY_LOG.
    """
    img_f = img.astype(np.float32) / 255.0
    if input_is_log:
        return img_f
    return np.log1p(img_f * 255.0) / LOG_SCALE


def from_additive_domain(x, input_is_log=INPUT_ALREADY_LOG):
    """Nghịch đảo của to_additive_domain, trả về ảnh hiển thị [0, 1]."""
    if input_is_log:
        return x
    return np.expm1(x * LOG_SCALE) / 255.0


def estimate_noise_mad(detail_hh):
    """Ước lượng độ lệch chuẩn nhiễu bằng MAD trên subband chi tiết mịn nhất.

    Ước lượng Donoho: sigma = median(|d|) / 0.6745.
    """
    return float(np.median(np.abs(detail_hh)) / 0.6745)


def structure_tensor_coherence(V, sigma=1.5):
    """Độ kết dính định hướng của trường V, giá trị trong [0, 1].

    Speckle mang thông tin cấu trúc bộc lộ tính định hướng theo ranh giới mô,
    trong khi nhiễu cảm biến đẳng hướng nên có coherence thấp.
    """
    gx = cv2.Sobel(V, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(V, cv2.CV_32F, 0, 1, ksize=3)

    ksize = (0, 0)
    Jxx = cv2.GaussianBlur(gx * gx, ksize, sigma)
    Jxy = cv2.GaussianBlur(gx * gy, ksize, sigma)
    Jyy = cv2.GaussianBlur(gy * gy, ksize, sigma)

    trace = Jxx + Jyy
    diff = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2)
    return diff / (trace + 1e-8)


def _soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def bayes_shrink_tau(detail, sigma_n):
    """Ngưỡng BayesShrink thích ứng cho một subband: tau = sigma_n^2 / sigma_x.

    Ngưỡng universal sigma*sqrt(2 log N) quá gắt trên ảnh lớn (~5.2 sigma), nó
    dồn gần hết phần dư vào Ur và bỏ đói Ui. BayesShrink ước lượng phương sai
    tín hiệu của từng subband nên giữ được speckle có cấu trúc.
    """
    sigma_y2 = float(np.mean(detail ** 2))
    sigma_x = np.sqrt(max(sigma_y2 - sigma_n ** 2, 0.0))
    if sigma_x < 1e-8:
        return float(np.abs(detail).max())
    return sigma_n ** 2 / sigma_x


def _smoothstep_mask(coherence, thresh, width=0.15):
    """Mặt nạ mềm quanh ngưỡng coherence, tránh viền cứng do cắt nhị phân."""
    t = np.clip((coherence - thresh) / max(width, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def decompose_oct(img, weight_tv=0.1, thresh_scale=1.0, coh_thresh=0.3):
    """Phân rã ảnh OCT thành ba thành phần cộng tính trong miền log.

        log I = Us + Ui + Ur

    - Us: cấu trúc nền mô (TV, bảo toàn cạnh)
    - Ui: speckle mang thông tin cấu trúc (có định hướng)
    - Ur: nhiễu ngẫu nhiên cảm biến (loại bỏ)

    Tham số:
    - weight_tv: trọng số TV, càng lớn Us càng phẳng
    - thresh_scale: hệ số nhân trên ngưỡng nhiễu suy ra từ MAD (không phải
      ngưỡng tuyệt đối) — thang đo tự thích ứng theo mức nhiễu từng ảnh
    - coh_thresh: ngưỡng độ kết dính định hướng để giữ lại Ui
    """
    base = to_additive_domain(img)

    Us_log = denoise_tv_chambolle(base, weight=weight_tv).astype(np.float32)
    V = base - Us_log

    try:
        coeffs = pywt.wavedec2(V, "db4", level=2)
        sigma_mad = estimate_noise_mad(coeffs[-1][-1])

        new_coeffs = [coeffs[0]]
        taus = []
        for detail_level in coeffs[1:]:
            level_out = []
            for d in detail_level:
                t = thresh_scale * bayes_shrink_tau(d, sigma_mad)
                taus.append(t)
                level_out.append(_soft_threshold(d, t))
            new_coeffs.append(tuple(level_out))

        tau = float(np.mean(taus))
        Ui_wav = pywt.waverec2(new_coeffs, "db4")[: V.shape[0], : V.shape[1]]
        Ui_wav = Ui_wav.astype(np.float32)
    except Exception as e:
        print(f"Lỗi khi dùng Wavelet, bỏ qua bước lọc ngưỡng: {e}")
        sigma_mad, tau = 0.0, 0.0
        Ui_wav = V

    coherence = structure_tensor_coherence(V)
    Ui = Ui_wav * _smoothstep_mask(coherence, coh_thresh)
    Ur = V - Ui

    return {
        "Us_log": Us_log,
        "Us_lin": from_additive_domain(Us_log),
        "Ui": Ui,
        "Ur": Ur,
        "coherence": coherence,
        "sigma_mad": sigma_mad,
        "tau": tau,
    }
