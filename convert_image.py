import cv2
import os
import glob

# Thư mục chứa ảnh gốc và nơi lưu ảnh binary
input_folder = './images/original/'    # Thư mục chứa ảnh gốc
output_folder = './images/binary/'     # Thư mục lưu ảnh binary

# Tạo folder lưu nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Lấy danh sách tất cả các file ảnh trong thư mục input
image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

# Sắp xếp để đảm bảo thứ tự
image_files.sort()

# Lặp qua tất cả các ảnh
for i, input_path in enumerate(image_files, 1):
    # Tạo tên file đầu ra
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
