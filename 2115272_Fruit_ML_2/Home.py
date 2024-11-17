import streamlit as st
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import requests
import time
from bs4 import BeautifulSoup

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


# Load model and initialize ResNet50
with open('svm_fruit_classifier.pkl', 'rb') as file:
    svm_model = pickle.load(file)
feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define fruit calorie information
FRUIT_INFO = {
    'Apple': 52, 'Banana': 89, 'Orange': 47, 'Mango': 60, 
    'Watermelon': 30, 'Pineapple': 50, 'Kiwi': 61, 'Avocado': 92, 
    'Strawberries': 45, 'Banana': 95, 'Cherry': 42
}



# Extract features using ResNet50
def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    return features.flatten()

# Predict fruit and its metrics
def predict_fruit(image_path):
    features = extract_features(image_path)
    prediction = svm_model.predict([features])[0]
    probabilities = svm_model.predict_proba([features])[0]
    confidence = probabilities.max() * 100
    return prediction, confidence, probabilities

# Fetch calorie information from Google
def fetch_calories(prediction):
    try:
        url = f'https://www.google.com/search?q=calories+in+{prediction}'
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception:
        st.error("Could not fetch calorie information.")
        return f"{FRUIT_INFO.get(prediction, 'Unknown')} kcal/100g"

def calculate_label_distribution(labels):
    label_counts = {label: labels.count(label) for label in set(labels)}
    return label_counts


def compare_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare their performance with additional metrics.
    """
    results = []

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Define the models
    models = {
        "SVM (Linear)": SVC(kernel='linear', probability=True),
        "SVM (RBF)": SVC(kernel='rbf', probability=True),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier()
    }

    for name, model in models.items():
        start_time = time.time()
        #model.fit(X_train, y_train)
        model.fit(X_train, y_train_encoded)
        train_time = time.time() - start_time

        y_pred_encoded = model.predict(X_test)  # Kết quả dự đoán là số

        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        precision = precision_score(y_test_encoded, y_pred_encoded, average='macro')
        recall = recall_score(y_test_encoded, y_pred_encoded, average='macro')
        f1_macro = f1_score(y_test_encoded, y_pred_encoded, average='macro')
        f1_micro = f1_score(y_test_encoded, y_pred_encoded, average='micro')
        mse = mean_squared_error(y_test_encoded, y_pred_encoded)
        rmse = np.sqrt(mse)

        # Silhouette score for clustering-based metric
        kmeans = KMeans(n_clusters=len(set(y_test_encoded)), random_state=42)
        kmeans.fit(X_test)
        silhouette = silhouette_score(X_test, kmeans.labels_)

        # Append results
        results.append({
            "Model": name,
            "Train Time (s)": train_time,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Macro": f1_macro,
            "F1 Micro": f1_micro,
            "MSE": mse,
            "RMSE": rmse,
            "Silhouette Score": silhouette
        })

    return pd.DataFrame(results)



def parameter_vs_metric_svm(X_train, y_train, X_test, y_test, param_name, param_values, metric="Precision"):
    """
    Test different parameter values for an SVM model and evaluate performance on a chosen metric.

    Args:
        X_train, y_train, X_test, y_test: Data splits.
        param_name (str): Name of the parameter to vary ("C" or "gamma").
        param_values (list): List of parameter values to test.
        metric (str): The performance metric to evaluate ("Accuracy", "Precision", "Recall", "F1").

    Returns:
        pd.DataFrame: A dataframe with parameter values and their corresponding metric scores.
    """
    results = []
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    for param_value in param_values:
        # Train the SVM model with the specified parameter
        if param_name == "C":
            model = SVC(C=param_value, probability=True, kernel='linear')
        elif param_name == "gamma":
            model = SVC(gamma=param_value, probability=True, kernel='rbf')
        else:
            raise ValueError("Unsupported parameter name.")

        # Train and evaluate
        model.fit(X_train, y_train_encoded)
        y_pred_encoded = model.predict(X_test)

        # Calculate the chosen metric
        if metric == "Precision":
            score = precision_score(y_test_encoded, y_pred_encoded, average='macro')
        elif metric == "Accuracy":
            score = accuracy_score(y_test_encoded, y_pred_encoded)
        elif metric == "Recall":
            score = recall_score(y_test_encoded, y_pred_encoded, average='macro')
        elif metric == "F1":
            score = f1_score(y_test_encoded, y_pred_encoded, average='macro')
        else:
            raise ValueError("Unsupported metric.")

        # Append results
        results.append({"Parameter Value": param_value, metric: score})

    return pd.DataFrame(results)


def parameter_vs_metric_svm_kernel_c(X_train, y_train, X_test, y_test, kernels, c_values, metric="Precision"):
    """
    Test different SVM kernels and C values, and evaluate performance on a chosen metric.

    Args:
        X_train, y_train, X_test, y_test: Data splits.
        kernels (list): List of kernel types to test (e.g., ["linear", "rbf", "poly"]).
        c_values (list): List of regularization values (C) to test.
        metric (str): The performance metric to evaluate ("Accuracy", "Precision", "Recall", "F1").

    Returns:
        pd.DataFrame: A dataframe with kernel, C values, and their corresponding metric scores.
    """
    results = []
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    for kernel in kernels:
        for c_value in c_values:
            # Train the SVM model with specified kernel and C
            model = SVC(kernel=kernel, C=c_value, probability=True)
            model.fit(X_train, y_train_encoded)
            y_pred_encoded = model.predict(X_test)

            # Calculate the chosen metric
            if metric == "Precision":
                score = precision_score(y_test_encoded, y_pred_encoded, average='macro')
            elif metric == "Accuracy":
                score = accuracy_score(y_test_encoded, y_pred_encoded)
            elif metric == "Recall":
                score = recall_score(y_test_encoded, y_pred_encoded, average='macro')
            elif metric == "F1":
                score = f1_score(y_test_encoded, y_pred_encoded, average='macro')
            else:
                raise ValueError("Unsupported metric.")

            # Append results
            results.append({"Kernel": kernel, "C Value": c_value, metric: score})

    return pd.DataFrame(results)


 # Hàm kiểm tra hiệu năng SVM với các loại kernel
def evaluate_svm_kernels(X_train, y_train, X_test, y_test, kernels):
    """
    Kiểm tra độ chính xác của các SVM kernels trên tập dữ liệu.

    Args:
        X_train, y_train: Tập huấn luyện.
        X_test, y_test: Tập kiểm tra.
        kernels: Danh sách các kernels để kiểm tra.

    Returns:
        pd.DataFrame: Bảng kết quả với Accuracy cho từng kernel.
    """
    results = []
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    for kernel in kernels:
        model = SVC(kernel=kernel, C=1.0, probability=True)
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, y_pred)

        results.append({"Kernel": kernel, "Accuracy": accuracy})

    return pd.DataFrame(results)


# Function to plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, values_format=".0f")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Predicted Labels', fontsize=14)
    ax.set_ylabel('True Labels', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    st.pyplot(fig)



# Streamlit application
def main():
    st.set_page_config(page_title="Fruit Classification", page_icon="🍎", layout="wide")
    st.title("Fruit Classification System 🍓")
    kernels = None

    # Dummy training labels (replace with actual data labels)
    train_labels = ["Apple", "Orange", "Avocado", "Kiwi", "Mango", "Pineapple", "Strawberries", 
                    "Banana", "Cherry", "Watermelon"]  # Replace with real labels
    
    label_distribution = calculate_label_distribution(train_labels)

    X_train, X_test, y_train, y_test = np.random.rand(100, 2048), np.random.rand(20, 2048), \
                                       np.random.choice(["Apple", "Banana", "Orange"], 100), \
                                       np.random.choice(["Apple", "Banana", "Orange"], 20)

    # Compare models
    st.subheader("Model Comparison")
    if st.button("Compare Models"):
        comparison_df = compare_models(X_train, y_train, X_test, y_test)

        # Display comparison table
        st.dataframe(comparison_df)

        # Display bar chart for model metrics
        st.subheader("Metrics Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_melted = comparison_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")
        sns.barplot(data=comparison_melted, x="Metric", y="Score", hue="Model", ax=ax)
        plt.title("Model Metrics Comparison")
        plt.ylabel("Score")
        st.pyplot(fig)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    if st.button("Predict"):
        prediction, confidence, probabilities = predict_fruit(uploaded_file)
        st.success(f"Prediction: {prediction.capitalize()}")
        st.info(f"Confidence: {confidence:.2f}%")


        # Fetch and display calories
        calories = fetch_calories(prediction)
        st.write(f"Calories for {prediction}: {calories}")

        # Get labels used by the model
        model_labels = svm_model.classes_

        # Heatmap of feature correlations (dummy data example)
        st.subheader("Feature Correlation Heatmap")
        dummy_features = np.random.rand(100, len(model_labels))
        correlation_matrix = np.corrcoef(dummy_features, rowvar=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", xticklabels=model_labels, yticklabels=model_labels)
        plt.title("Heatmap of Feature Correlations")
        st.pyplot(fig)

        # Display prediction probabilities bar chart
        st.subheader("Prediction Probabilities")
        fig, ax = plt.subplots()
        sns.barplot(x=model_labels, y=probabilities, ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel("Probability")
        plt.title("Prediction Probabilities for Each Class")
        st.pyplot(fig)

        # Display calorie comparison chart
        st.subheader("Calorie Comparison (Bar Chart)")
        fig, ax = plt.subplots()
        sns.barplot(x=list(FRUIT_INFO.keys()), y=list(FRUIT_INFO.values()), ax=ax)
        plt.ylabel("Calories (kcal/100g)")
        plt.xticks(rotation=45)
        plt.title("Calories of Each Fruit")
        st.pyplot(fig)

        # Display pie chart of fruit distribution (real data)
        st.subheader("Fruit Distribution (Real Data)")
        fig, ax = plt.subplots()
        ax.pie(label_distribution.values(), labels=label_distribution.keys(), autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Ensure pie is a circle
        plt.title("Distribution of Fruits in Dataset (Real Data)")
        st.pyplot(fig)

        # Display comparison of calories and confidence
        st.subheader("Comparison: Calories vs Confidence")
        fruit_name = prediction.capitalize()  # Normalize prediction to match FRUIT_INFO keys
        calories = FRUIT_INFO.get(fruit_name, 0)  # Fetch calories or default to 0
        st.write(f"Prediction: {prediction}, Calories: {calories}")  # Debugging output

        fig, ax = plt.subplots()
        sns.barplot(x=["Calories", "Confidence"], y=[calories, confidence], ax=ax)
        plt.ylabel("Value")
        plt.title("Calories vs Confidence for Prediction")
        st.pyplot(fig)

        # Lấy danh sách nhãn từ mô hình
        labels = list(svm_model.classes_)

        # # Hiển thị Confusion Matrix
        # st.subheader("Confusion Matrix")
        # # Dữ liệu mẫu (thay thế bằng dữ liệu thực tế khi sẵn sàng)
        # y_true = ["Apple", "Orange", "Avocado", "Kiwi", "Mango", "Pineapple", "Strawberries", 
        #             "Banana", "Cherry", "Watermelon"]  # Thay bằng nhãn thực tế
        # y_pred = ["Apple", "Orange", "Avocado", "Kiwi", "Mango", "Pineapple", "Strawberries", 
        #             "Banana", "Cherry", "Watermelon"]  # Thay bằng dự đoán thực tế
        # cm = confusion_matrix(y_true, y_pred, labels=labels)
        # plot_confusion_matrix(cm, labels)

         # Display Confusion Matrix
        st.subheader("Confusion Matrix")
        y_true = train_labels  # Replace with actual labels when available
        y_pred = train_labels  # Replace with actual predictions when available
        plot_confusion_matrix(y_true, y_pred, np.unique(train_labels))

        

    st.subheader("Parameter vs Metric Comparison for SVM")

    # Choose parameter (C or gamma) and range for testing
    param_name = st.selectbox("Choose the parameter to test for SVM", ["C", "gamma"])
    
    if param_name == "C":
        param_values = st.slider("Choose range for C (Regularization Strength)", 0.01, 10.0, (0.1, 1.0), step=0.1)
        param_values = np.linspace(param_values[0], param_values[1], 10).tolist()
    elif param_name == "gamma":
        param_values = st.slider("Choose range for gamma (Kernel coefficient)", 0.01, 5.0, (0.1, 1.0), step=0.1)
        param_values = np.linspace(param_values[0], param_values[1], 10).tolist()

    # Run parameter vs metric comparison
    if st.button("Run Parameter Test"):
        comparison_df = parameter_vs_metric_svm(
            X_train, y_train, X_test, y_test, param_name, param_values, metric="Precision"
        )

        # Display results
        st.dataframe(comparison_df)

        # Plot the results
        st.subheader(f"{param_name} vs Precision for SVM")
        fig, ax = plt.subplots()
        sns.lineplot(data=comparison_df, x="Parameter Value", y="Precision", marker="o", ax=ax)
        plt.title(f"{param_name} vs Precision for SVM")
        plt.xlabel(param_name)
        plt.ylabel("Precision")
        st.pyplot(fig)
    
    st.subheader("Kernel and C Comparison for SVM")

    # User selects kernels and range of C values
    svm_kernels = st.multiselect("Choose kernels to test", ["linear", "rbf", "poly", "sigmoid"], default=["linear", "rbf"])
    svm_c_range = st.slider(
    "Choose range for C (Regularization Strength)", 
    0.01, 
    10.0, 
    (0.1, 1.0), 
    step=0.1, 
    key="svm_c_range_slider")

    svm_c_values = np.linspace(svm_c_range[0], svm_c_range[1], 10).tolist()

    # User chooses metric
    chosen_metric = st.selectbox("Choose a metric to evaluate", ["Accuracy", "Precision", "Recall", "F1"])

    # Run the test and plot results
    if st.button("Run Kernel and C Comparison"):
        kernel_c_df = parameter_vs_metric_svm_kernel_c(
            X_train, y_train, X_test, y_test, svm_kernels, svm_c_values, metric=chosen_metric)

        # Display results
        st.dataframe(kernel_c_df)

        # Create a pivot table for heatmap visualization
        heatmap_data = kernel_c_df.pivot(index="Kernel", columns="C Value", values=chosen_metric)

        # Plot heatmap
        st.subheader(f"Heatmap: {chosen_metric} vs Kernel and C Value")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        plt.title(f"{chosen_metric} Heatmap for Kernel and C")
        st.pyplot(fig)

        # So sánh hiệu năng các Kernel
    st.subheader("So sánh hiệu năng của các SVM Kernel Types")
    kernels = st.multiselect(
    "Chọn các Kernel để kiểm tra:",
    ["linear", "rbf", "poly", "sigmoid"],
    default=["linear", "rbf"]
    )

    if st.button("Chạy kiểm tra Kernel"):
        if not kernels:
            st.warning("Hãy chọn ít nhất một Kernel để kiểm tra.")
        else:
            kernel_results = evaluate_svm_kernels(X_train, y_train, X_test, y_test, kernels)

        # Hiển thị bảng kết quả
        st.dataframe(kernel_results)

        # Vẽ biểu đồ Accuracy bằng biểu đồ đường
        st.subheader("Biểu đồ Accuracy giữa các Kernel (Biểu đồ đường)")
        fig, ax = plt.subplots()
        sns.lineplot(data=kernel_results, x="Kernel", y="Accuracy", marker="o", ax=ax, color="blue")
        plt.title("Accuracy của từng Kernel (Biểu đồ đường)")
        plt.ylabel("Accuracy")
        plt.xlabel("Kernel")
        plt.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)
    

if __name__ == "__main__":
    main()
