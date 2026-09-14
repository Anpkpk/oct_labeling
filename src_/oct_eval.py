import numpy as np
import matplotlib
matplotlib.use('Agg') # Tránh lỗi thread-safety trong backend
import matplotlib.pyplot as plt
import cv2

def evaluate_denoising(residual, grad_ori, grad_clean, w_index):
    """
    Phân tích thành phần bị loại bỏ và độ sắc nét của bản đồ chi phí.

    - residual: Phần dư bị vứt bỏ. Ở chế độ phân rã đây là Ur (nhiễu ngẫu
      nhiên) — phân bố của nó phải xấp xỉ Gauss nếu tiêu chí tách Ui/Ur đúng.
    - grad_ori: Bản đồ đạo hàm của ảnh gốc
    - grad_clean: Bản đồ đạo hàm / cost map sau xử lý
    - w_index: Vị trí cột (trục X) để cắt Line Profile
    """

    # 1. Ảnh phần dư (Residual Image)
    # Chuyển phần dư (có số âm) về hiển thị thang xám [0, 255]
    res_img = np.clip(residual + 0.5, 0, 1)
    res_img_u8 = (res_img * 255).astype(np.uint8)
    # Tăng cường tương phản để nhìn rõ nhiễu
    res_img_u8 = cv2.equalizeHist(res_img_u8)
    res_img_rgb = cv2.cvtColor(res_img_u8, cv2.COLOR_GRAY2RGB)
    
    # 2. Vẽ Histogram phần dư
    fig_hist = plt.figure(figsize=(6, 4))
    plt.hist(residual.ravel(), bins=100, color='gray', alpha=0.7, density=True)
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Phân bố phần dư (Kỳ vọng: Chuông Gauss tại 0)")
    plt.xlabel("Mức xám (I_ori - I_clean)")
    plt.ylabel("Mật độ")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 3. Vẽ Gradient Line Profile
    fig_profile = plt.figure(figsize=(8, 4))
    prof_ori = grad_ori[:, w_index]
    prof_clean = grad_clean[:, w_index]
    
    # Chuẩn hóa về [0, 1] để dễ so sánh hình dáng đỉnh
    if prof_ori.max() > 0: prof_ori = prof_ori / prof_ori.max()
    if prof_clean.max() > 0: prof_clean = prof_clean / prof_clean.max()
    
    plt.plot(prof_ori, label="Gradient Truyền Thống / Gốc", color='red', alpha=0.4, linestyle='--')
    plt.plot(prof_clean, label="Gradient Sau Phân Rã", color='blue', alpha=0.8, linewidth=2)
    plt.title(f"Mặt cắt Gradient tại cột X = {w_index} (Độ sắc nét cạnh)")
    plt.xlabel("Trục Y (Độ sâu, pixels)")
    plt.ylabel("Biên độ (Normalized)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    return res_img_rgb, fig_hist, fig_profile
