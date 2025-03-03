import pandas as pd
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import streamlit as st
import pathlib
import random
import base64
from io import BytesIO

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("css/style.css")
load_css(css_path)

df = pd.read_csv("data/Pokemoncopy.csv")

pokemon_names = ["Charmander", "Pikachu"]
filtered_df = df[df["Name"].isin(pokemon_names)]

filtered_df = filtered_df[["Name", "Type1", "Attack", "HP"]]

st.markdown('<h1 class="st-KNN">Pokemon Prediction with CNN</h1>', unsafe_allow_html=True)

st.subheader("ข้อมูล Pokémon ทั้งหมด")
st.dataframe(df) 

st.subheader("ข้อมูล Pokémon (Charmander และ Pikachu)")
st.write(filtered_df)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)

def build_cnn_model_simplified():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Flatten(),
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2, 
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' 
)

def train_model():
    model = build_cnn_model_simplified()

    base_path = "train/pokemon_images"
    train_data = datagen.flow_from_directory(
        base_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True 
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6)

    model.fit(
        train_data,
        epochs=100,
        validation_data=train_data,
        callbacks=[reduce_lr]
    )

    model.save("model/pokemon_cnn.h5")

model = tf.keras.models.load_model("model/pokemon_cnn.h5")

st.subheader("")
st.subheader("สามารถดาวน์โหลดภาพตัวอย่างของ Charmander และ Pikachu เพื่อทดสอบโมเดลได้")


def get_pokemon_image(pokemon_name , size = (300,300)):
        folder_path = f"train/pokemon_images/{pokemon_name}/"
        image_files = list(pathlib.Path(folder_path).glob("*.jpg"))
        if image_files:
            return random.choice(image_files) 
        return None
    
charmander_image = get_pokemon_image("Charmander")
if charmander_image:
        image = Image.open(charmander_image)
        image = image.resize((300, 300))
        st.image(image, caption="Charmander ตัวอย่าง", use_container_width=True)
        img_buffer = BytesIO()
        image.save(img_buffer, format="JPEG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        href = f'<a href="data:file/jpg;base64,{img_base64}" download="Charmander_sample.jpg">📥 ดาวน์โหลด Charmander</a>'
        st.markdown(href, unsafe_allow_html=True)

st.subheader("")
pikachu_image_path = get_pokemon_image("Pikachu")
if pikachu_image_path:
        image = Image.open(pikachu_image_path)
        image = image.resize((300, 300))
        st.image(image, caption="Pikachu ตัวอย่าง", use_container_width=True)

        img_buffer = BytesIO()
        image.save(img_buffer, format="JPEG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        href = f'<a href="data:file/jpg;base64,{img_base64}" download="Pikachu_sample.jpg">📥 ดาวน์โหลด Pikachu</a>'
        st.markdown(href, unsafe_allow_html=True)
            



uploaded_file = st.file_uploader("อัปโหลดภาพโปเกมอน", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปโปเกมอน", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    prediction_confidence = np.max(prediction)

    predicted_name = "Charmander" if predicted_label == 0 else "Pikachu"
    predicted_type = filtered_df[filtered_df["Name"] == predicted_name]["Type1"].values[0]
    predicted_attack = filtered_df[filtered_df["Name"] == predicted_name]["Attack"].values[0]
    predicted_hp = filtered_df[filtered_df["Name"] == predicted_name]["HP"].values[0]

    st.markdown(f"### ผลลัพธ์การทำนาย:")
    st.write(f"**ชื่อ:** {predicted_name}")
    st.write(f"**ประเภท:** {predicted_type}")
    st.write(f"**พลังโจมตี:** {predicted_attack}")
    st.write(f"**เลือด:** {predicted_hp}")
    st.write(f"**ความมั่นใจในการทำนาย:** {prediction_confidence*100:.2f} %")
    
    st.write(f"**ความมั่นใจในการทำนาย Pikachu:** {prediction[0][1] * 100:.2f}%")
    st.write(f"**ความมั่นใจในการทำนาย Charmander:** {prediction[0][0] * 100:.2f}%")
    
    
    