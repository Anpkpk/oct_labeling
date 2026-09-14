import numpy as np
import cv2

NORM_PERCENTILE = 99.0


def _normalize(x):
    """Chuẩn hoá theo phân vị cao thay vì max.

    Chia cho max khiến một cạnh cực mạnh (thường là AEJ) ép toàn bộ phần còn
    lại của bản đồ về gần 0, làm hai kênh mất khả năng so sánh với nhau.
    """
    scale = np.percentile(x, NORM_PERCENTILE)
    if scale < 1e-8:
        scale = x.max() + 1e-8
    return np.clip(x / scale, 0.0, 1.0)


def texture_energy(Ui, win=9):
    """Năng lượng speckle cục bộ (RMS trong cửa sổ trượt).

    Đại lượng này mô tả *thống kê* của speckle tại mỗi vùng, không phải giá trị
    hạt thô — nền tảng để phát hiện nơi thống kê đổi khác.
    """
    win = int(win) | 1
    mean_sq = cv2.boxFilter(Ui.astype(np.float32) ** 2, -1, (win, win))
    return np.sqrt(np.maximum(mean_sq, 0.0))


def texture_transition_cost(tex_energy, win=9):
    """Mức thay đổi của thống kê speckle theo chiều sâu.

    Trường năng lượng phải được làm mượt ở thang lớn hơn kích thước hạt trước
    khi lấy đạo hàm; nếu không, đạo hàm bám theo dao động của từng hạt và tạo
    ra đúng thứ rừng đỉnh giả mà cost map hai kênh sinh ra để tránh.
    """
    sigma = max(float(win), 3.0)
    smoothed = cv2.GaussianBlur(tex_energy, (0, 0), sigma)
    return np.abs(np.gradient(smoothed, axis=0))


def build_cost_map(Us_log, G_texture, gamma, beta):
    """Cost map hai kênh: gradient cấu trúc nền + chuyển tiếp kết cấu.

        G_total = norm(|d(Us^gamma)/dy|) + beta * norm(G_texture)

    Gamma tác động lên Us *trước* khi lấy đạo hàm; nếu áp lên cost map sau đạo
    hàm thì nó là phép đơn điệu, không đổi kết quả argmax/argmin.
    """
    Us_gamma = np.power(np.clip(Us_log, 0.0, None), gamma)
    G_base = np.abs(np.gradient(Us_gamma, axis=0))

    G_total = _normalize(G_base) + beta * _normalize(G_texture)
    return G_total, G_base
