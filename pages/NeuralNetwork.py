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
        <h5 class="st-element1">ขั้นตอนนี้ผมได้ทำการอ่านข้อมูลจากไฟล์ CSV และ แสดงข้อมูล</h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    df = pd.read_csv("data/Pokemoncopy.csv")

    pokemon_names = ["Charmander", "Pikachu"]
    filtered_df = df[df["Name"].isin(pokemon_names)]

    filtered_df = filtered_df[["Name", "Type1", "Attack", "HP"]]

    st.markdown('<h1 class="st-KNN">Pokemon Prediction with CNN</h1>', unsafe_allow_html=True)

    st.subheader("ข้อมูล Pokémon ทั้งหมด")
    st.dataframe(df) 

    st.subheader("ข้อมูล Pokémon (Charmander และ Pikachu)")
    st.write(filtered_df)
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">จากนั้นผมได้ทำการสร้างโมเดล CNN และ สร้าง ImageDataGenerator สำหรับการขยายข้อมูล</h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python

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
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">ในขั้นตอนนี้ผมได้ทำการ train ข้อมุลโดยให้ epochs = 100</h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
   
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
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">ในขั้นตอนนี้ผมได้ให้ user สามารถ upload รูปภาพของ Pokemon และทำการทำนาย Pokemon ที่อยู่ในรูปภาพ และแสดงผลลัพธ์</h5>
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



