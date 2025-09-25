import os
import numpy as np  
from sklearn.metrics import pairwise_distances


def split_data_test_and_train(input_file):
    with open(input_file, 'r') as file:
        next(file)  # Bỏ qua header
        lines = file.readlines()


    train_data = []
    test_data = []
    for i, line in enumerate(lines, start=1):  # đếm từ 1
        row = line.strip().split(',')
        if (i) % 10 in [9, 0]:  
            print(f"Adding to test data: {row}")
            test_data.append(row)

        else:
            train_data.append(row)
    
    return train_data, test_data

# 4. Template matching: tính khoảng cách test đến tất cả train
def normalize_data(data):
    data_array = np.array(data, dtype=float)
    min_vals = data_array.min(axis=0)
    max_vals = data_array.max(axis=0)
    ranges = max_vals - min_vals
    norm_data = (data_array - min_vals) / ranges
    return norm_data

def euclidean_distance(x_test, train):   
    return [float(np.sqrt(np.sum((x_test - train[i])**2))) for i in range(len(train))]


def normalize(input_file):
    train_data, test_data = split_data_test_and_train(input_file)

    norm_train = normalize_data(train_data)
    norm_test = normalize_data(test_data)
    
    return norm_train, norm_test

def cal_euclidean_distance(train_data, test_data):
    distances = []
    for test_point in test_data:
        distances.append(euclidean_distance(test_point, train_data))
    return distances


if __name__ == "__main__":
    input_folder = "./data/hu_data/"
    os.makedirs(input_folder, exist_ok=True)
    input_file = os.path.join(input_folder, "hu_features.csv")
    output_folder = "./data/template_matching/"

    train_data, test_data = split_data_test_and_train(input_file)
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    label_test = [row[-1] for row in test_data]
    label_train = [row[-1] for row in train_data]
    train_data = [row[1:] for row in train_data]
    test_data = [row[1:] for row in test_data]

    norm_train = normalize_data(train_data)
    norm_test = normalize_data(test_data)
    print(f"Normalized train data shape: {norm_train.shape}")
    print(f"Normalized test data shape: {norm_test.shape}")
    dis = euclidean_distance(norm_test[0], norm_train)


    distances = cal_euclidean_distance(norm_train, norm_test)
    label_pred = []
    for d in distances:
        i = float(np.argmin(d))
        label_pred.append(label_train[int(i)])


    distances1 = pairwise_distances(norm_train, norm_test, metric='euclidean')
    label_pred1 = []
    for d in distances:
        i = float(np.argmin(d))
        label_pred1.append(label_train[int(i)])

    # Lưu kết quả
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "euclidean_results.csv")
    with open(output_file, 'w') as f:
        f.write("Lable,PredictedLabel\n")
        for lable_true, label_pred in zip(label_test,label_pred):
            f.write(f"{lable_true},{label_pred}\n")






