import cv2
import numpy as np
import os
from tqdm import tqdm

# Cấu hình đường dẫn
gt_dir = 'output/bicycle/test/ours_30000/gt'
render_dir = 'output/bicycle/test/ours_30000/renders'
output_dir = 'output/bicycle/test/ours_30000/error_maps'

# Tạo thư mục output nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Lấy danh sách file ảnh (giả sử tên file ở 2 folder là giống nhau)
render_files = [f for f in os.listdir(render_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

print(f"Tìm thấy {len(render_files)} ảnh để xử lý.")

for filename in tqdm(render_files):
    # 1. Đọc ảnh Render và Ground Truth
    render_path = os.path.join(render_dir, filename)
    gt_path = os.path.join(gt_dir, filename)
    
    img_render = cv2.imread(render_path)
    img_gt = cv2.imread(gt_path)

    if img_gt is None:
        print(f"Cảnh báo: Không tìm thấy ảnh GT tương ứng cho {filename}. Bỏ qua.")
        continue

    # Đảm bảo 2 ảnh cùng kích thước
    if img_render.shape != img_gt.shape:
        img_gt = cv2.resize(img_gt, (img_render.shape[1], img_render.shape[0]))

    # 2. Tính toán sai số L1 (Absolute Difference)
    # Công thức: |Render - GT|
    diff = cv2.absdiff(img_render.astype(np.float32), img_gt.astype(np.float32))
    
    # Lấy trung bình cộng của 3 kênh màu (B, G, R) theo đúng paper FastGS [cite: 228]
    error_map = np.mean(diff, axis=2)

    # 3. Chuẩn hóa và tạo Heatmap [cite: 231]
    # Chúng ta chuẩn hóa về [0, 255] để hiển thị
    # Lưu ý: Bạn có thể chỉnh sửa 'vmax' để làm nổi bật các lỗi nhỏ
    vmax = 50 # Giới hạn sai số tối đa để bản đồ nhiệt nhạy hơn với lỗi nhỏ
    normalized_error = np.clip((error_map / vmax) * 255, 0, 255).astype(np.uint8)
    
    # Áp dụng bảng màu JET (Đỏ = Lỗi cao, Xanh = Lỗi thấp)
    heatmap = cv2.applyColorMap(normalized_error, cv2.COLORMAP_JET)

    # 4. Lưu kết quả
    # Lưu ảnh Error Map riêng lẻ
    cv2.imwrite(os.path.join(output_dir, f"error_{filename}"), heatmap)
    
    # (Tùy chọn) Lưu ảnh so sánh 3 cột: GT | Render | Error
    combined = np.hstack((img_gt, img_render, heatmap))
    cv2.imwrite(os.path.join(output_dir, f"compare_{filename}"), combined)

print(f"\nHoàn thành! Kết quả đã được lưu tại: {output_dir}")