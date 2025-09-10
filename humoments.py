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

# Thư mục chứa ảnh
folder_path = "./binary_images"  # nếu ảnh nằm cùng folder với script
output_csv = "./data/hu_features.csv"

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
        try:
            hu_features = compute_hu_moments(image_path)
            # Ghi vào CSV
            # Làm tròn 2 số sau dấu phẩy
            rounded_features = [round(x, 4) for x in hu_features]
            writer.writerow([image_name] + rounded_features)
            print(f"Đã xử lý: {image_name}")
        except Exception as e:
            print(f"Lỗi với {image_name}: {e}")

print(f"\nĐã lưu 7 đặc trưng Hu của 50 ảnh vào file {output_csv}")
