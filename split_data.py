import os
import pandas as pd
from sklearn.datasets import load_iris

base_dir = "./data/iris_split"   # sửa đường dẫn theo ý bạn
os.makedirs(base_dir, exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["target_name"] = df["target"].map(dict(enumerate(iris.target_names)))

for flower in iris.target_names:               # setosa, versicolor, virginica
    flower_df = df[df["target_name"] == flower].reset_index(drop=True)
    flower_dir = os.path.join(base_dir, flower)
    os.makedirs(flower_dir, exist_ok=True)
    
    # chia 5 folder, mỗi folder 10 mẫu
    for i in range(5):
        folder_dir = os.path.join(flower_dir, f"Folder{i+1}")
        os.makedirs(folder_dir, exist_ok=True)
        
        part = flower_df.iloc[i*10:(i+1)*10].reset_index(drop=True)
        file_path = os.path.join(folder_dir, "samples.csv")  # 1 file csv chứa 10 mẫu
        part.to_csv(file_path, index=False, encoding="utf-8")
        print(f"Saved {file_path} ({len(part)} rows)")

print("\n✅ Hoàn tất. Cấu trúc lưu: base_dir/<loai_hoa>/Folder1/samples.csv ...")
