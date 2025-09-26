from PIL import Image
import numpy as np
import os

# Thư mục chứa ảnh đầu vào và đầu ra
input_folder = "."  # hoặc "Mango"
output_folder = "./Mango_black"
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua tất cả ảnh
for i in range(1, 23):
    filename = f"{i}.{'jpg' if i < 3 else 'png'}"
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    if os.path.exists(input_path):
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)

        # Tạo mask cho pixel nền trắng (gần trắng)
        white_mask = np.all(img_np > [220, 220, 220], axis=-1)

        # Đổi các pixel nền trắng thành đen
        img_np[white_mask] = [0, 0, 0]

        # Lưu ảnh mới
        new_img = Image.fromarray(img_np)
        new_img.save(output_path)

# print("✅ Đã chuyển nền trắng thành đen và lưu vào thư mục 'output_black_background'.")
