# Tạo file mới: utils/dash_utils.py
import math

def get_dash_resolution_scale(iteration, warmup_iters=3000):
    """
    Tính toán tỷ lệ scale ảnh. 
    Trong giai đoạn warmup, độ phân giải tăng dần từ 1/4 lên 1 (gốc).
    """
    if iteration >= warmup_iters:
        return 1.0
    
    # Chia giai đoạn warmup làm 3 mức: 1/4, 1/2, và 1
    if iteration < warmup_iters * 0.33:
        return 0.25
    elif iteration < warmup_iters * 0.66:
        return 0.5
    else:
        return 0.75

def get_primitive_budget(iteration, max_iterations, upper_bound):
    """
    Tính ngân sách điểm tối đa (Concave-up curve).
    Tăng chậm ở đầu và nhanh hơn ở cuối.
    """
    # Đường cong bậc 2 đơn giản mô phỏng DashGaussian
    progress = min(iteration / max_iterations, 1.0)
    budget = upper_bound * (progress ** 2) 
    
    # Đảm bảo luôn có một lượng ngân sách cơ bản lúc mới bắt đầu
    return max(int(budget), 10000)