import streamlit as st
import pandas as pd
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
import os
from sklearn.preprocessing import LabelEncoder

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("css/style.css")
load_css(css_path)

st.markdown('<h1 class="st-KNN">Pokemon Type & Attack Prediction with CNN</h1>', unsafe_allow_html=True)

df = pd.read_csv("data/Pokemoncopy.csv")
st.subheader("ตัวอย่างข้อมูล")
st.write(df.head())

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def build_cnn_model_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8 , activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    model = build_cnn_model_vgg16()
    print("⚙️ เริ่มการฝึกโมเดล...")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2
    )

    base_path = "train/pokemon_images"
    if not os.path.exists(base_path):
        print(f"⚠️ โฟลเดอร์ `{base_path}` ไม่มีอยู่! โปรดตรวจสอบไฟล์ภาพ")
        return

    train_data = datagen.flow_from_directory(
        base_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    if train_data.samples == 0:
        print("⚠️ ไม่มีภาพในโฟลเดอร์ `train/pokemon_images` เทรนไม่ได้!")
        return

    steps_per_epoch = train_data.samples // train_data.batch_size
    model.fit(train_data, epochs=50, batch_size=32, steps_per_epoch=steps_per_epoch)

    model.save("model/pokemon_cnn.h5")
 

model = tf.keras.models.load_model("model/pokemon_cnn.h5")

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
