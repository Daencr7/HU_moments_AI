import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns

def compute_hu_moments(image_path):
    """Tính 7 Hu Moments từ ảnh nhị phân (đối tượng trắng, nền đen)"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không đọc được ảnh: {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (128, 128))


    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    if np.mean(binary) < 128:
        binary = cv2.bitwise_not(binary)

    binary_dir = "./images/binary/"
    os.makedirs(binary_dir, exist_ok=True)
    cv2.imwrite(os.path.join(binary_dir, os.path.basename(image_path)), binary)

    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments).flatten()

    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    return hu_moments, hu_moments_log


def extract_hu_features_to_csv(input_folder, output_file, mode="a", write_header=False, log=True, is_full=False):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Lấy danh sách file ảnh
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    image_files.sort()

    num_labels = 5
    labels = [f"{i:03b}" for i in range(num_labels)]

    all_results = []  # ✅ Dùng biến này để chứa toàn bộ dữ liệu

    for i, image_path in enumerate(tqdm(image_files, desc=f"Extracting Hu Moments from {input_folder}")):
        image_name = os.path.basename(image_path)

        if is_full:
            label = (i // 10) + 1
        else:
            label = labels[i // 2]

        hu_features, hu_features_log = compute_hu_moments(image_path)
        if hu_features is None:
            continue

        features = hu_features_log if log else hu_features
        formatted_features = [f"{float(x):.12e}" for x in features]

        # ✅ Lưu một dòng dữ liệu đúng format
        row = [image_name] + formatted_features + [label]
        all_results.append(row)

    feature_number = len(all_results[0]) - 2  # trừ image_name & label
    columns = ['image_name'] + [f'hu_{i+1}' for i in range(feature_number)] + ['label']
    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv(output_file, mode=mode, header=write_header, index=False)

def load_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    X_train = train_df.drop(columns = ['image_name','label']).values
    y_train = train_df['label'].values

    X_test = test_df.drop(columns = ['image_name','label']).values
    y_test = test_df['label'].values

    return X_train, y_train, X_test, y_test

def save(input_folder, output_folder,log=True, mode="append"):
    os.makedirs(output_folder, exist_ok=True)
    if log == True:
        if mode == "append":
            output_file = os.path.join(output_folder, "hu_features_log.csv")
            write_header = not os.path.exists(output_file)
            extract_hu_features_to_csv(input_folder, output_file, mode="a", write_header=write_header)
        else:
            folder_name = os.path.basename(input_folder.rstrip("/"))
            output_file = os.path.join(output_folder, f"hu_features_log{folder_name}.csv")
            extract_hu_features_to_csv(input_folder, output_file, mode="w", write_header=True)
    else:
        if mode == "append":
            output_file = os.path.join(output_folder, "hu_features.csv")
            write_header = not os.path.exists(output_file)
            extract_hu_features_to_csv(input_folder, output_file, mode="a", write_header=write_header)
        else:
            folder_name = os.path.basename(input_folder.rstrip("/"))
            output_file = os.path.join(output_folder, f"hu_features{folder_name}.csv")
            extract_hu_features_to_csv(input_folder, output_file, mode="w", write_header=True)

        
# --- Hàm vẽ và LƯU ma trận nhầm lẫn ---
def save_confusion_matrix(cm, labels=None, title="Confusion Matrix", cmap='Blues', save_path="confusion_matrix.png"):
    """
    Vẽ và lưu ma trận nhầm lẫn ra file ảnh PNG
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=title, xlabel='Predicted', ylabel='True')

    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

    # Hiển thị số trong từng ô
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)  # Giải phóng bộ nhớ

def train_and_evaluate(X_train, y_train, X_test, y_test, class_names=None, save_dir="./results", log = True):
    os.makedirs(save_dir, exist_ok=True)
    cm_dir = os.path.join(save_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    results = []

    for k in [3, 4, 5]:

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)

        acc_log = accuracy_score(y_test, predict)
        recall_log = recall_score(y_test, predict, average='macro', zero_division=0)
        prec_log = precision_score(y_test, predict, average='macro', zero_division=0)
        f1_log = f1_score(y_test, predict, average='macro', zero_division=0)

        cm_log = confusion_matrix(y_test, predict)
        cm_path_log = os.path.join(cm_dir, f"K{k}_Hu_Goc.png")
        save_confusion_matrix(cm_log, labels=class_names, title=f"K={k} - Hu Moments (Gốc)", cmap='Blues', save_path=cm_path_log)

        results.append({
            'K': k,
            'Type': 'Hu Moments {} '.format("log" if log else "gốc" ),
            'Accuracy': acc_log,
        })


  
    # --- Tổng kết kết quả ---
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(save_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv, index=False)

    return results_df



def kfold_by_folder(base_dir, folders, model, verbose=True):
    data_dict = {}
    for folder in folders:
        path = os.path.join(base_dir, folder, f'hu_features_log{folder}.csv')
        df = pd.read_csv(path)
        # Bỏ cột 'image_name' vì không phải đặc trưng số
        if 'image_name' in df.columns:
            df = df.drop(columns=['image_name'])
        data_dict[folder] = df

    results = []


    for test_folder in folders:
        if verbose:
            print(f"\n===== Fold {test_folder} =====")

        test_df = data_dict[test_folder]

        train_folders = [f for f in folders if f != test_folder]
        train_df = pd.concat([data_dict[f] for f in train_folders], ignore_index=True)

        # Chia dữ liệu
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']
        X_test = test_df.drop(columns=['label'])
        y_test = test_df['label']

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            "fold": test_folder,
            "accuracy": acc
        })

        if verbose:
            print("Accuracy:", acc)
            print("Confusion matrix:\n", cm)
            print("Classification report:\n", classification_report(y_test, y_pred))

    return pd.DataFrame(results)

def save_file_csv(folder_num):
    input_folder = f"./images/data/{folder_num}"

    # Xác định đầu ra và chế độ ghi file
    output_folder = './data/HU/{}'.format(folder_num)
    mode = "w"  # test ghi mới

    os.makedirs(output_folder, exist_ok=True)

    save(input_folder=input_folder, output_folder=output_folder, log=True, mode=mode)
    save(input_folder=input_folder, output_folder=output_folder, log=False, mode=mode)

def save_all():
    for folder_num in range(1, 6):
        save_file_csv(folder_num)

def save_file_csv(): 
    save(input_folder="./images/data/1", output_folder='./data/HU/data_train', log = False, mode = "append") 
    save(input_folder="./images/data/1", output_folder='./data/HU/data_train', log = True, mode = "append") 
    save(input_folder="./images/data/2", output_folder='./data/HU/data_train', log = False, mode = "append") 
    save(input_folder="./images/data/2", output_folder='./data/HU/data_train', log = True, mode = "append") 
    save(input_folder="./images/data/3", output_folder='./data/HU/data_train', log = False, mode = "append") 
    save(input_folder="./images/data/3", output_folder='./data/HU/data_train', log = True, mode = "append") 
    save(input_folder="./images/data/4", output_folder='./data/HU/data_train', log = False, mode = "append") 
    save(input_folder="./images/data/4", output_folder='./data/HU/data_train', log = True, mode = "append") 
    save(input_folder="./images/data/5", output_folder='./data/HU/data_test', log = False, mode = "w") 
    save(input_folder="./images/data/5", output_folder='./data/HU/data_test', log = True, mode = "w")


def main():
    # save_file_csv()
    # === Đường dẫn dữ liệu ===
    train_csv_log = "./data/HU/data_train/hu_features_log.csv"
    test_csv_log  = "./data/HU/data_test/hu_features_log5.csv"
    train_csv    = "./data/HU/data_train/hu_features.csv"
    test_csv     = "./data/HU/data_test/hu_features5.csv"

    
    X_train_log, y_train_log, X_test_log, y_test_log = load_data(train_csv_log, test_csv_log)
    X_train, y_train, X_test, y_test = load_data(train_csv, test_csv)

    # === Train và đánh giá ===
    results_1 = train_and_evaluate(X_train_log, y_train_log, X_test_log, y_test_log, class_names=None, save_dir="./data/HU/results_log", log = True)

    results_2 = train_and_evaluate(X_train, y_train, X_test, y_test, class_names=None, save_dir="./data/HU/results", log = False)

    results_1.to_csv("./data/HU/results_knn.csv", index=False)
    
    results_2.to_csv("./data/HU/results_knn1.csv", header=False, index=False)
    print(results_1)
    print(results_2)
    # === KFold Evaluation ===
    # save_all()
    folders = ['1', '2', '3', '4', '5']
    base_dir = './data/HU'

    knn = KNeighborsClassifier(n_neighbors=3)

    results_df = kfold_by_folder(base_dir, folders, knn, verbose=False)
    print("Kết quả")
    print(results_df)


if __name__ == "__main__":



    main()

