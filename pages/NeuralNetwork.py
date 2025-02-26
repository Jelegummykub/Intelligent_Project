import streamlit as st
import pathlib

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
css_path = pathlib.Path("css/style.css")
load_css(css_path)

st.markdown('<h1 class="st-header">Neural Network </h1>', unsafe_allow_html=True)

st.subheader("การเตรียม Dataset")

st.markdown(
    """
    <div class="content-section">
        <p class="st-element11">
        - ในส่วนของการเตรียมข้อมูล ผมได้เตรียมข้อมูลมาจากเว็บ 
        <a href="https://www.kaggle.com/datasets/abcsds/pokemon" target="_blank">Kaggle</a> 
        โดยผมอยากทราบว่าหากใส่รูปนี้ไปจะเป็น Pokemon ตัวไหน มีธาตุอะไร หรือว่า พลังโจมตีเท่าไร ผมจึงได้นำมาทดสอบใน Neural Network 
        โดยใช้ Algorithm Convolutional Neural Network (CNN) ในการทำนายจากรูปภาพของ Pokemon
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# st.image("public/images/pokemon.jpg")

st.subheader("ทฤษฎีของ Convolutional Neural Network (CNN)")

st.markdown(
    """
    <div class="content-section">
        <p class="st-element11">
        - Convolutional Neural Network (CNN) เป็นโมเดล Deep Learning ที่ใช้ในการจำแนกประเภทของภาพ โดยมีการใช้ Convolutional Layer และ Pooling Layer
        ในการจำแนกประเภทของภาพ และใช้ Fully Connected Layer ในการจำแนกประเภทของภาพ โดยมีการใช้ Activation Function แบบ ReLU ใน Convolutional Layer
        และใช้ Activation Function แบบ Softmax ใน Fully Connected Layer สำหรับการจำแนกประเภทของภาพ
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("ขั้นตอนในการพัฒนาโมเดล Convolutional Neural Network (CNN)")

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">ขั้นตอนนี้ผมได้ทำการอ่านข้อมูลจากไฟล์ CSV และ สร้างฟังก์ชันพรีโปรเซสภาพ</h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    df = pd.read_csv("data/Pokemoncopy.csv")
    st.write(df.head())

    def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">จากนั้นผมได้ทำการสร้างโมเดล CNN และทำการฝึกโมเดล ด้วย VGG16</h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python

    def build_cnn_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(df["Type1"].unique()), activation='softmax')
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
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">ในขั้นตอนนี้ผมได้ให้ user สามารถ upload รูปภาพของ Pokemon และทำการทำนายประเภทของ Pokemon ที่อยู่ในรูปภาพ และแสดงผลลัพธ์</h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
   
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
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
    <h3> สามารถดูโมเดล Convolutional Neural Network (CNN)</h3>
    <div class="st-container1">
        <div class="st-card1">
            <h4 class="st-element1">Model Neural Network </h4>
            <a href="/NeuralNetworkModel">
             <button class="st-button">Click me</button>
            </a> 
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
    <div class="content-section">
        <p class="st-element11">
        - สามารถดู code ของ Convolutional Neural Network (CNN) ได้ที่
        <a href="https://github.com/Jelegummykub/Intelligent_Project/blob/main/pages/NeuralNetworkModel.py" target="_blank">github.com/Jelegummy</a> 
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



