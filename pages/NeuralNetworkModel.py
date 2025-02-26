import streamlit as st
import pandas as pd
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("css/style.css")
load_css(css_path)

st.markdown('<h1 class="st-KNN">Pokemon Type & Attack Prediction with CNN</h1>', unsafe_allow_html=True)

df = pd.read_csv("data/Pokemoncopy.csv")
st.write(df.head())

def preprocess_image(image):
    """ฟังก์ชันพรีโปรเซสภาพ (resize และ normalize)"""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def build_cnn_model():
    """ฟังก์ชันสร้างโมเดล CNN ด้วย VGG16 เป็น Transfer Learning"""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(df["Type1"].unique()), activation='softmax')  # ใช้ softmax สำหรับหลายประเภท
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    model = build_cnn_model()

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_data = datagen.flow_from_directory(
        'data/pokemon_images',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    model.fit(train_data, epochs=20, steps_per_epoch=200)

    model.save('pokemon_cnn.h5')

try:
    model = tf.keras.models.load_model("pokemon_cnn.h5")
except:
    model = build_cnn_model()

uploaded_file = st.file_uploader("อัปโหลดภาพโปเกมอน", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปโปเกมอน", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    prediction_confidence = np.max(prediction)

    pokemon_info = df.iloc[predicted_label]
    predicted_name = pokemon_info["Name"]
    predicted_type = pokemon_info["Type1"]
    predicted_attack = pokemon_info["Attack"]

    st.markdown(f"### ผลลัพธ์การทำนาย:")
    st.write(f"**ชื่อ:** {predicted_name}")
    st.write(f"**ประเภท:** {predicted_type}")
    st.write(f"**พลังโจมตี:** {predicted_attack}")
    st.write(f"**ความมั่นใจในการทำนาย:** {prediction_confidence*100:.2f} %")
