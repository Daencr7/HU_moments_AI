import cv2
import numpy as np
import csv
import os

def compute_hu_moments(image_path):
    """Tính 7 đặc trưng Hu từ ảnh binary trắng–đen."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    # Chuyển ảnh từ trắng–đen sang 0–1
    binary = (img > 127).astype(np.uint8)  # trắng = 1, đen = 0

    # Tính moments
    moments = cv2.moments(binary * 255)  # OpenCV yêu cầu 0–255

    # Tính 7 đặc trưng Hu
    hu = cv2.HuMoments(moments).flatten()

    # Log transform để scale giá trị
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    
    return hu_log

def compute_hu_moments_new(matrix: np.ndarray):
    """Tính Hu Moments thủ công từ ảnh nhị phân (0/255)."""
    # Chuyển về ma trận 0/1 và ép kiểu int64
    matrix = (matrix > 0).astype(np.int64)

    rows, cols = matrix.shape
    M00 = np.sum(matrix)

    if M00 == 0:
        return [np.nan] * 7  # ảnh trắng hoàn toàn thì bỏ qua

    # Centroid
    x_c = np.sum([i * matrix[i, j] for i in range(rows) for j in range(cols)]) / M00
    y_c = np.sum([j * matrix[i, j] for i in range(rows) for j in range(cols)]) / M00

    # Tính các moment trung tâm
    M20 = M02 = M11 = M30 = M03 = M12 = M21 = 0.0
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                dx = i - x_c
                dy = j - y_c
                M20 += dx**2
                M02 += dy**2
                M11 += dx*dy
                M30 += dx**3
                M03 += dy**3
                M12 += dx*(dy**2)
                M21 += (dx**2)*dy

    # Hàm chuẩn hóa moment
    def eta(Mpq, p, q):
        return Mpq / (M00 ** (1 + (p + q) / 2))

    eta20 = eta(M20, 2, 0)
    eta02 = eta(M02, 0, 2)
    eta11 = eta(M11, 1, 1)
    eta30 = eta(M30, 3, 0)
    eta03 = eta(M03, 0, 3)
    eta12 = eta(M12, 1, 2)
    eta21 = eta(M21, 2, 1)

    # Hu Moments
    S1 = eta20 + eta02
    S2 = (eta20 - eta02)**2 + 4*(eta11**2)
    S3 = (eta30 - 3*eta12)**2 + (3*eta21 - eta03)**2
    S4 = (eta30 + eta12)**2 + (eta21 + eta03)**2
    S5 = ((eta30 - 3*eta12)*(eta30+eta12)*((eta30+eta12)**2 - 3*(eta21+eta03)**2) +
          (3*eta21 - eta03)*(eta21+eta03)*(3*(eta30+eta12)**2 - (eta21+eta03)**2))
    S6 = ((eta20 - eta02)*((eta30+eta12)**2 - (eta21+eta03)**2) +
          4*eta11*(eta30+eta12)*(eta21+eta03))
    S7 = ((3*eta21 - eta03)*(eta30+eta12)*((eta30+eta12)**2 - 3*(eta21+eta03)**2) -
          (eta30 - 3*eta12)*(eta21+eta03)*(3*(eta30+eta12)**2 - (eta21+eta03)**2))

    hu = [S1, S2, S3, S4, S5, S6, S7]
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log



# Thư mục chứa ảnh binary
folder_path = "./images/binary"  # Thư mục chứa ảnh binary
output_csv = "./data/hu_features_new.csv"

# Tạo thư mục data nếu chưa tồn tại
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Lưu kết quả
with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Ghi header
    header = ["Image", "Hu1", "Hu2", "Hu3", "Hu4", "Hu5", "Hu6", "Hu7", "label"]
    writer.writerow(header)

    # Lặp qua 50 ảnh
    
    for i in range(1, 51):
        image_name = f"{i}BI.png"
        image_path = os.path.join(folder_path, image_name)

        label = (i - 1) // 5 + 1 
        try:
            hu_features = compute_hu_moments_new(image_path)
            # Ghi vào CSV
            # Làm tròn 2 số sau dấu phẩy
            rounded_features = [round(x, 4) for x in hu_features]
            writer.writerow([image_name] + rounded_features + [label])
            print(f"Đã xử lý: {image_name}")
        except Exception as e:
            print(f"Lỗi với {image_name}: {e}")

print(f"\nĐã lưu 7 đặc trưng Hu của 50 ảnh vào file {output_csv}")
