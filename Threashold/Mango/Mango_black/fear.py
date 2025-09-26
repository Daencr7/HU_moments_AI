from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Thư mục hiện tại
folder_path = "."

# Lưu kết quả Red mean
red_means = {}

# Đọc 1.jpg và 2.jpg
for i in range(1, 3):
    filename = f"{i}.jpg"
    img_path = os.path.join(folder_path, filename)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        # Lọc nền trắng (nếu có)
        mask = np.any(img_np > [0, 0, 0], axis=-1)
        red_values = img_np[:, :, 0][mask]
        red_mean = np.mean(red_values) if red_values.size > 0 else 0
        red_means[filename] = red_mean

# Đọc 3.png đến 22.png
for i in range(3, 23):
    filename = f"{i}.png"
    img_path = os.path.join(folder_path, filename)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        # Lọc nền đen
        mask = np.any(img_np > [0, 0, 0], axis=-1)
        red_values = img_np[:, :, 0][mask]
        red_mean = np.mean(red_values) if red_values.size > 0 else 0
        red_means[filename] = red_mean

# Phân lớp và in kết quả
print("== GIÁ TRỊ TRUNG BÌNH KÊNH ĐỎ VÀ PHÂN LỚP ==")
lamda = 180  # Threshold
classified = []

for fname, val in sorted(red_means.items(), key=lambda x: int(x[0].split('.')[0])):
    label = "ngọt" if val >= lamda else "chua"
    print(f"{fname}: {val:.2f} → Xoài {label}")
    classified.append((int(fname.split('.')[0]), val, label))

# === VẼ BIỂU ĐỒ TRÊN TRỤC OX ===
indices = [item[0] for item in classified]
red_values = [item[1] for item in classified]
labels = [item[2] for item in classified]

plt.figure(figsize=(10, 2))
    
# Vẽ chấm theo màu dựa trên nhãn
colors = ['red' if lbl == 'ngọt' else 'green' for lbl in labels]
plt.scatter(red_values, [0]*len(red_values), color=colors, s=50)

# Đánh số thứ tự trên từng chấm
for i in range(len(indices)):
    plt.text(red_values[i], 0.05, str(indices[i]), ha='center', fontsize=9)

# Vẽ đường ngưỡng dọc
plt.axvline(x=lamda, color='black', linestyle='--', label=f'Ngưỡng = {lamda}')

# Tùy chỉnh giao diện
plt.yticks([])  # Ẩn trục Oy
plt.xlabel('Giá trị trung bình kênh đỏ (Red)')
plt.title('Phân bố Red Mean của các ảnh xoài')
plt.grid(True, axis='x')
plt.legend()
plt.tight_layout()
plt.show()

