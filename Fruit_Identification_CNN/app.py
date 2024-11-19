


import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

model = load_model('./Image_classify1811.keras') 
data_cat = ['apple','avocado', 
 'banana', 'cherry', 
 'kiwi',
 'mango',
 'orange',
 'pineapple',
 'strawberries',
 'watermelon']
img_height = 180
img_width = 180


st.header('Image Classification Model')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","jfif"])

if uploaded_file is not None:
    image_load = Image.open(uploaded_file)
    image_load = image_load.resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image_load)
    img_bat = tf.expand_dims(img_arr, 0)
    
    st.image(image_load, caption="Uploaded Image", width=400)
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    st.write(f'Fruit in image is: {data_cat[np.argmax(score)]}')
    st.write(f'With accuracy of: {np.max(score) * 100:.2f}%')
