import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

model = load_model('./model/emotionModel.h5')
data_cat = ['angry', 'happy', 'sad']
# data_cat = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', "surprised"]
img_height = 256
img_width = 256

st.header('Human Emotion Detection')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    image_load = Image.open(uploaded_file)
    image_load = image_load.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)  

    st.image(image_load, caption="Uploaded Image", use_container_width=True)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])
    
    emotion = data_cat[np.argmax(score)]

    st.markdown(f"<h2 style='font-size: 48px;'>Emotion in image is: {emotion}</h2>", unsafe_allow_html=True)
