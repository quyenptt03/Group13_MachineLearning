import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and ResNet50 feature extractor
model_path = 'svm_fruit_classifier.pkl'
with open(model_path, 'rb') as file:
    svm_model = pickle.load(file)

feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Labels for fruits
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage',
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn',
    10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes',
    15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas',
    25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato',
    33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

# Function to extract features
def extract_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array)
    return feature.flatten()

# Predict fruit class
def predict_fruit(image_path):
    features = extract_features(image_path, feature_extractor)
    prediction = svm_model.predict([features])[0]
    probabilities = svm_model.predict_proba([features])[0]
    return labels[prediction], probabilities

# Display heat map
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Bar chart for probabilities
def plot_probabilities(probabilities):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=list(labels.values()), y=probabilities, palette='viridis')
    plt.xticks(rotation=45)
    plt.ylabel('Confidence')
    plt.title('Confidence Distribution Across Classes')
    st.pyplot(fig)

# Pie chart for probabilities
def plot_pie_chart(probabilities):
    fig, ax = plt.subplots(figsize=(8, 8))
    top_indices = np.argsort(probabilities)[-5:][::-1]
    top_labels = [list(labels.values())[i] for i in top_indices]
    top_probs = probabilities[top_indices]
    ax.pie(top_probs, labels=top_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Top 5 Predictions')
    st.pyplot(fig)

# Streamlit app
def run():
    st.set_page_config(page_title="Fruits Classification", page_icon="🍍")
    st.title("Fruits Classification System")

    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])

    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, caption="Uploaded Image", use_column_width=False)

        if st.button("Predict"):
            prediction, probabilities = predict_fruit(img_file)
            st.success(f"Predicted: {prediction.capitalize()}")

            # Display probabilities
            plot_probabilities(probabilities)
            plot_pie_chart(probabilities)

            # Sample confusion matrix (to be replaced with actual test data metrics)
            cm = confusion_matrix([1, 0, 2], [1, 0, 2])  # Replace with actual data
            plot_confusion_matrix(cm, list(labels.values()))

run()
