import cv2
import os

# Thư mục chứa ảnh gốc và nơi lưu ảnh binary
input_folder = './'    # nếu ảnh cùng folder với script
output_folder = './binary_images/'

# Tạo folder lưu nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Lặp qua 50 ảnh
for i in range(1, 51):
    input_path = os.path.join(input_folder, f'{i}.png')
    output_path = os.path.join(output_folder, f'{i}BI.png')
    
    # Đọc ảnh
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Không đọc được ảnh {input_path}, bỏ qua.")
        continue

    # Nếu ảnh có alpha channel (4 kênh), convert sang grayscale
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    # Chuyển sang binary, object trắng background đen
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Lưu ảnh
    cv2.imwrite(output_path, binary)
    print(f"Đã lưu {output_path}")
