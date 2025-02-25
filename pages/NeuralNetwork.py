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
        <a href="https://www.kaggle.com/datasets/navtiw/pokemon" target="_blank">Kaggle</a> 
        โดยผมสนใจ พลังการโจมตีของ Pokemon ผมจึงได้ทำมาทดสอบใน Neural Network 
        โดยใช้ Algorithm Multi-layer Perceptron (MLP) ในการทำนายพลังการโจมดีของ Pokemon
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.image("public/images/pokemon.jpg")

st.subheader("ทฤษฎีของ Multi-layer Perceptron (MLP)")

st.markdown(
    """
    <div class="content-section">
        <p class="st-element11">
        - Multi-layer Perceptron (MLP) เป็น Neural Network ที่มีชั้นซ่อนมากกว่า 1 ชั้น โดยมีชั้น Input และ Output ที่ชัดเจน
        และมีชั้นซ่อนที่มีหน่วยประมวลผลหลายๆ หน่วย โดยการสร้างเชื่อมต่อระหว่างหน่วยประมวลผลของชั้นก่อนหน้า
        และหน่วยประมวลผลของชั้นถัดไป ทำให้สามารถทำนายข้อมูลที่มีความซับซ้อนได้ดี และมีความแม่นยำสูง
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("ขั้นตอนในการพัฒนาโมเดล Multi-layer Perceptron (MLP)")

st.markdown(
    """
    ```python
    # ขั้นตอนนี้ผมได้ทำการอ่านข้อมูลจากไฟล์ CSV และแสดงตัวอย่างของชุดข้อมูล
    df = pd.read_csv("data/pokemon.csv")

    st.subheader("")
    st.subheader("ตัวอย่างข้อมูล")
    # แสดงตัวอย่างของชุดข้อมูล
    st.write(df.head())
    
    ```
    
    ```python

    # จากนั้นทำการเตรียมข้อมูล โดยการแปลงข้อมูลเชิงหมวดหมู่ (เพศ, หมวดหมู่, ความสามารถ, จุดอ่อน ฯลฯ) และเลือกคุณลักษณะที่ใช้ในการฝึกโมเดล
    label_encoder = LabelEncoder()

    df['gender'] = label_encoder.fit_transform(df['gender'].astype(str))
    df['category'] = label_encoder.fit_transform(df['category'].astype(str))
    df['abilities'] = label_encoder.fit_transform(df['abilities'].astype(str))
    df['weakness'] = label_encoder.fit_transform(df['weakness'].astype(str))

    features = df[['gender', 'category', 'abilities', 'weakness', 'height', 'weight', 'attack', 'defense', 'hp', 'special_attack', 'special_defense', 'speed']]

    target = df['attack']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    ```
    
    ```python
   # จากนั้นผมได้ทำการสร้างและฝึกโมเดล MLP และทำการทำนาย
    mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)

    y_pred = mlp_model.predict(X_test)

    st.markdown("### ผลการประเมินโมเดล:")

    report = classification_report(y_test, y_pred, output_dict=True)

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    plt.figure(figsize=(10, 6))
    ```
    
    ```python
    # ทำการแสดงกราฟ Scatter Plot เพื่อแสดงผลการทำนาย
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='การทำนายที่สมบูรณ์แบบ')

    plt.title("Predicted vs Actual Attack Values")
    plt.xlabel("Actual Attack")
    plt.ylabel("Predicted Attack")

    st.pyplot(plt)
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
    <h3> สามารถดูโมเดล Multi-layer Perceptron (MLP) </h3>
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
        - สามารถดู code ของ Multi-layer Perceptron (MLP) ได้ที่
        <a href="https://github.com/Jelegummykub/Intelligent_Project/blob/main/pages/NeuralNetworkModel.py" target="_blank">github.com/Jelegummy</a> 
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



