from PIL import Image
import numpy as np
import os

# Thư mục hiện tại (chứa script và ảnh)
folder_path = "."

# Lưu kết quả trung bình Red
red_means = {}

# Xử lý ảnh .jpg: 1.jpg và 2.jpg
for i in range(1, 3):
    filename = f"{i}.jpg"
    img_path = os.path.join(folder_path, filename)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        mask = np.all(img_np < [240, 240, 240], axis=-1)  # Lọc nền trắng
        red_values = img_np[:, :, 0][mask]
        red_mean = np.mean(red_values) if red_values.size > 0 else 0
        red_means[filename] = red_mean

# Xử lý ảnh .png: 3.png đến 22.png
for i in range(3, 23):
    filename = f"{i}.png"
    img_path = os.path.join(folder_path, filename)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        mask = np.all(img_np < [240, 240, 240], axis=-1)  # Lọc nền trắng
        red_values = img_np[:, :, 0][mask]
        red_mean = np.mean(red_values) if red_values.size > 0 else 0
        red_means[filename] = red_mean

# In kết quả
print("== GIÁ TRỊ TRUNG BÌNH KÊNH ĐỎ CỦA MỖI ẢNH ==")
for fname, val in sorted(red_means.items()):
    print(f"{fname}: {val:.2f}")

# for fname, val in red_means.items():
#     print(f"{fname}: {val:.2f}")
print()
print("Classify")
# set giá trị ngưỡng
lamda = 180

for fname, val in sorted(red_means.items()):
    if val >= lamda:
        print(f"Xoai ngot: {fname}")


