import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import pandas as pd

def compute_hog_features(image_path, win_size=(64, 128), block_size=(16, 16), 
                        block_stride=(8, 8), cell_size=(8, 8), nbins=9):
    # Bước 1: Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        return None
    
    # Bước 2: Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Lưu ảnh xám vào thư mục grey
    gray_dir = "./images/grey/"
    os.makedirs(gray_dir, exist_ok=True)
    gray_filename = os.path.join(gray_dir, os.path.basename(image_path))
    cv2.imwrite(gray_filename, gray)
    
    # Bước 3: Thay đổi kích thước ảnh về kích thước cửa sổ mong muốn
    resized = cv2.resize(gray, win_size)
    
    # Lưu ảnh đã resize vào thư mục resize
    resize_dir = "./images/resize/"
    os.makedirs(resize_dir, exist_ok=True)
    resize_filename = os.path.join(resize_dir, os.path.basename(image_path))
    cv2.imwrite(resize_filename, resized)
    
    # Bước 4: Khởi tạo HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Bước 5: Tính toán đặc trưng HOG và chỉ lấy 50 đặc trưng đầu tiên
    hog_features = hog.compute(resized)
    
    # Làm phẳng mảng và chỉ lấy 50 đặc trưng đầu tiên
    hog_features_flattened = hog_features.flatten()
    return hog_features_flattened[:50]  # Chỉ trả về 50 đặc trưng đầu tiên

def extract_hog_features_to_csv(input_folder, output_file):
    # Tạo thư mục chứa kết quả nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Lấy danh sách tất cả file ảnh trong thư mục đầu vào
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # Sắp xếp để đảm bảo thứ tự
    image_files.sort()
    
    # Danh sách nhãn 3 bit (từ 000 đến 111)
    labels = [f"{i:03b}" for i in range(8)]
    
    # Danh sách lưu trữ kết quả
    results = []
    
    # Xử lý từng ảnh
    for i, image_path in enumerate(tqdm(image_files, desc="Đang trích xuất đặc trưng")):
        # Lấy tên file ảnh
        image_name = os.path.basename(image_path)
        
        # Xác định nhãn (mỗi 10 ảnh sẽ có cùng nhãn, sau đó lặp lại)
        label = labels[(i // 10) % len(labels)]
        
        # Trích xuất đặc trưng HOG
        hog_features = compute_hog_features(image_path)
        
        if hog_features is not None:
            # Làm tròn các đặc trưng HOG về 4 chữ số thập phân
            rounded_features = [round(float(x), 4) for x in hog_features]
            # Thêm tên ảnh, đặc trưng đã làm tròn và nhãn vào danh sách kết quả
            result = [image_name] + rounded_features + [label]
            results.append(result)
    
    # Tạo DataFrame từ kết quả
    columns = ['image_name'] + [f'hog_{i+1}' for i in range(len(hog_features))] + ['label']
    df = pd.DataFrame(results, columns=columns)
    
    # Lưu vào file CSV
    df.to_csv(output_file, index=False)
    print(f"Đã lưu đặc trưng HOG vào {output_file}")
    print(f"Tổng số ảnh đã xử lý: {len(results)}")
    
    # Lấy danh sách các nhãn duy nhất và chuyển đổi sang chuỗi
    unique_labels = sorted(set(str(label) for label in df['label']))
    print(f"Các nhãn đã sử dụng: {', '.join(unique_labels)}")
    
    # In ra số lượng ảnh cho mỗi nhãn
    print("\nThống kê số lượng ảnh cho mỗi nhãn:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        print(f"- {label}: {count} ảnh")

if __name__ == "__main__":
    # Thư mục chứa ảnh gốc
    input_folder = "./images/original/"
    
    # File CSV đầu ra
    output_folder = "./data/HOG_data/"
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "hog_features.csv")
    
    # Tạo thư mục lưu ảnh xám nếu chưa tồn tại
    gray_dir = "./images/grey/"
    os.makedirs(gray_dir, exist_ok=True)
    
    # Trích xuất đặc trưng HOG và lưu vào file CSV
    extract_hog_features_to_csv(input_folder, output_file)
