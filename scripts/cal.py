import math

def compute_focal_lengths(width, height, fov_x_deg):
    # 将角度转换为弧度
    fov_x_rad = math.radians(fov_x_deg)
    
    # 计算 fx
    fx = width / (2 * math.tan(fov_x_rad / 2))
    
    # 计算 fy（保持像素比例）
    fy = fx * height / width
    
    return fx, fy

# 示例分辨率和水平视场角
width = 1080
height = 720
fov_x_deg = 90

fx, fy = compute_focal_lengths(width, height, fov_x_deg)

print(f"fx = {fx:.2f}")
print(f"fy = {fy:.2f}")
