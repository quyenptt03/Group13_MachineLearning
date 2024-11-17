# File: 2115272_fruit_classification.py
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# Đường dẫn và khởi tạo
train_dir = './dataset/MY_data/train'
test_dir = './dataset/MY_data/test'

# Hàm tạo DataFrame
def create_dataframe(directory):
    data = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        for file in os.listdir(label_dir):
            data.append({'Filepath': os.path.join(label_dir, file), 'Label': label})
    return pd.DataFrame(data)

# Khởi tạo ResNet50
feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Hàm trích xuất đặc trưng
def extract_features(image_paths, model):
    features = []
    for path in image_paths:
        img = load_img(path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        features.append(model.predict(img_array).flatten())
    return np.array(features)

# Dữ liệu train/test
train_df = create_dataframe(train_dir)
test_df = create_dataframe(test_dir)

train_features = extract_features(train_df['Filepath'], feature_extractor)
test_features = extract_features(test_df['Filepath'], feature_extractor)

# SVM: Huấn luyện và đánh giá
svm_model = SVC(kernel='linear', probability=True, class_weight='balanced')
svm_model.fit(train_features, train_df['Label'])
predictions = svm_model.predict(test_features)
accuracy = accuracy_score(test_df['Label'], predictions)

print("Độ chính xác của SVM trên tập test:", accuracy)

# Biểu đồ Confusion Matrix cho SVM
cm = confusion_matrix(test_df['Label'], predictions, labels=np.unique(test_df['Label']))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(test_df['Label']), yticklabels=np.unique(test_df['Label']))
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Lưu mô hình SVM
with open("svm_fruit_classifier.pkl", "wb") as file:
    pickle.dump(svm_model, file)

# So sánh với các mô hình khác
models = {
    "SVM": SVC(kernel='linear', probability=True, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

def evaluate_models(train_features, train_labels, test_features, test_labels):
    results = []
    for model_name, model in models.items():
        # Huấn luyện mô hình
        model.fit(train_features, train_labels)
        # Dự đoán
        predictions = model.predict(test_features)
        # Tính các metric
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        f1 = f1_score(test_labels, predictions, average='weighted')
        # Lưu kết quả
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
    return pd.DataFrame(results)

# Đánh giá các mô hình
results_df = evaluate_models(train_features, train_df['Label'], test_features, test_df['Label'])
print("\nSo sánh hiệu năng giữa các mô hình:\n", results_df)

# Vẽ biểu đồ so sánh
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
for metric in metrics:
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["Model"], results_df[metric], color='skyblue')
    plt.title(f"So sánh {metric} giữa các mô hình")
    plt.ylabel(metric)
    plt.xlabel("Mô hình")
    plt.ylim(0, 1)
    plt.show()
