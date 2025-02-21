import streamlit as st
import pathlib

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
css_path = pathlib.Path("css/style.css")
load_css(css_path)

st.markdown('<h1 class="st-header">Machine Learning</h1>', unsafe_allow_html=True)

st.subheader("การเตรียม Dataset")

st.markdown(
    """
    <div class="content-section">
        <p class="st-element11">
        - ในส่วนของการเตรียมข้อมูล ผมได้เตรียมข้อมูลมาจากเว็บ 
        <a href="https://www.kaggle.com/competitions/titanic" target="_blank">Kaggle</a> 
        โดยผมสนใจอัตราการรอดชีวิตของเรือ Titanic ผมจึงได้นำข้อมูลมาใช้ในการทำ Machine Learning 
        โดยใช้ Algorithm K-Nearest Neighbors (KNN) และ Algorithm Support Vector Machine (SVM) 
        ในการทำนายอัตราการรอดชีวิตของผู้โดยสาร
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("ทฤษฎีของ K-Nearest Neighbors (KNN)")

st.markdown(
    """
    <div class="content-section">
        <p class="st-element11">
        - K-Nearest Neighbors (KNN) เป็น Algorithm ที่ใช้ในการจำแนกประเภทข้อมูล โดยการคำนวณระยะห่างระหว่างจุดข้อมูล 
        และจุดข้อมูลใหม่ที่ต้องการทำนาย จากนั้นจะเลือกจุดข้อมูลที่ใกล้ที่สุด K จุด และทำนายประเภทข้อมูลของจุดข้อมูลใหม่
        โดยใช้ค่าเฉลี่ยของประเภทข้อมูลของจุดข้อมูลที่ใกล้ที่สุด K จุด ในการทำนาย
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("ขั้นตอนในการพัฒนาโมเดล K-Nearest Neighbors (KNN)")

st.markdown(
    """
    ```python
    # การอ่านข้อมูลจากไฟล์ CSV
    df = pd.read_csv("data/Titanic.csv")

    # การสุ่มข้อมูลจำนวน 30 แถวจาก DataFrame
    df_sample = df.sample(n=30, random_state=42)

    # การเลือกคอลัมน์ที่สนใจและการจัดการข้อมูลที่ขาดหาย
    interested = pd.DataFrame({
        'Pclass': df_sample['Pclass'],  # เลือกคอลัมน์ Pclass
        'Sex': df_sample['Sex'].map({'male': 0, 'female': 1}),  # แปลงคอลัมน์ Sex ให้เป็น 0 และ 1
        'Age': df_sample['Age'].fillna(df['Age'].mean()),  # แทนค่าที่หายไปใน Age ด้วยค่าเฉลี่ย
        'Fare': df_sample['Fare'],  # เลือกคอลัมน์ Fare
        'Survived': df_sample['Survived']  # เลือกคอลัมน์ Survived
    })

    # แสดงข้อมูลตัวอย่าง
    st.subheader("ตัวอย่างข้อมูล")
    st.write(interested.head())

    # แยกข้อมูลออกเป็น features (X) และ label (y)
    X = interested.drop(columns=['Survived'])  # ลบคอลัมน์ Survived เพื่อใช้เป็นข้อมูล features
    y = interested['Survived']  # ใช้คอลัมน์ Survived เป็น label

    # สร้างโมเดล KNN (K-Nearest Neighbors)
    knn = KNeighborsClassifier(n_neighbors=3)  # สร้างโมเดล KNN โดยเลือก k = 3
    knn.fit(X, y)  # ฝึกโมเดลด้วยข้อมูล features (X) และ label (y)

    # ทำนายผลลัพธ์
    y_pred = knn.predict(X)  # ทำนายผลลัพธ์โดยใช้โมเดลที่ฝึกแล้ว

    # คำนวณความแม่นยำ (Accuracy)
    accuracy = accuracy_score(y, y_pred)
    st.subheader(f"Accuracy: :red[{accuracy:.2f}]")  # แสดงผลลัพธ์ความแม่นยำ

    # คำนวณ Precision, Recall, F1-score
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    st.subheader(f"Precision: :green[{precision:.2f}]")  # แสดงผล Precision
    st.subheader(f"Recall: :orange[{recall:.2f}]")  # แสดงผล Recall
    st.subheader(f"F1-score: :blue[{f1:.2f}]")  # แสดงผล F1-score

    # คำนวณและแสดง Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    st.subheader("Confusion Matrix : ")
    st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))  # แสดง Confusion Matrix

    # การแสดงชื่อหัวข้อ
    st.markdown('<h1 class="st-KNN">Titanic Survival Prediction</h1>', unsafe_allow_html=True)

    # การแสดงกราฟ Scatter Plot เพื่อแสดงผลการทำนาย
    fig, ax = plt.subplots()  # สร้างกราฟใหม่
    scatter = ax.scatter(interested['Age'], interested['Fare'], c=y_pred, cmap=plt.cm.viridis)  # สร้างกราฟ Scatter
    ax.set_xlabel("Age")  # กำหนดชื่อแกน X
    ax.set_ylabel("Fare")  # กำหนดชื่อแกน Y
    ax.set_title("Prediction Survived")  # กำหนดชื่อกราฟ
    plt.colorbar(scatter, label="Survived (0=No, 1=Yes)")  # แสดงแถบสีเพื่อบ่งบอกค่าผลการทำนาย

    # แสดงกราฟใน Streamlit
    st.pyplot(fig)
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("ทฤษฎีของ Support Vector Machine (SVM)")

st.markdown(
    """
    <div class="content-section">
        <p class="st-element11">
        - Support Vector Machine (SVM) เป็น Algorithm ที่ใช้ในการจำแนกประเภทข้อมูล โดยการสร้างเส้นแบ่งแยกข้อมูลที่มีความแตกต่างอย่างชัดเจน
        ระหว่างประเภทข้อมูล โดยหาเส้นแบ่งที่มีระยะห่างระหว่างจุดข้อมูลที่ใกล้ที่สุดจากเส้นแบ่งมากที่สุดผลการทำนายจะมีความแม่นยำสูง
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.subheader("ขั้นตอนในการพัฒนาโมเดล Support Vector Machine (SVM)")

st.markdown(
    """
    ```python
    # อ่านข้อมูลจากไฟล์ CSV
    df = pd.read_csv("data/Titanic.csv")

    # เลือกข้อมูลตัวอย่าง 30 แถวจาก DataFrame
    df_sample = df.sample(n=30, random_state=42)

    # สร้าง DataFrame ใหม่ที่สนใจเฉพาะบางคอลัมน์
    interested = pd.DataFrame({
        'Pclass': df_sample['Pclass'],  # คอลัมน์ Pclass
        'Sex': df_sample['Sex'].map({'male': 0, 'female': 1}),  # แปลงคอลัมน์ Sex จาก male/female เป็น 0/1
        'Age': df_sample['Age'].fillna(df['Age'].mean()),  # เติมค่า Age ที่หายไปด้วยค่าเฉลี่ยจากคอลัมน์ Age ทั้งหมด
        'Fare': df_sample['Fare'],  # คอลัมน์ Fare
        'Survived': df_sample['Survived']  # คอลัมน์ Survived (เป้าหมายที่เราต้องทำนาย)
    })

    # แสดงตัวอย่างข้อมูลที่เราเลือก
    st.subheader("ตัวอย่างข้อมูล")
    st.write(interested.head())

    # แยกคุณสมบัติ (features) และเป้าหมาย (target)
    X = interested.drop(columns=['Survived'])  # คุณสมบัติที่ใช้ในการทำนาย
    y = interested['Survived']  # เป้าหมายที่ต้องการทำนาย

    # สร้างโมเดล SVM โดยใช้ kernel แบบ linear
    svm = SVC(kernel='linear')
    svm.fit(X, y)  # ฝึกโมเดลด้วยข้อมูล

    # ทำนายผลจากข้อมูลที่ฝึก
    y_pred = svm.predict(X)

    # คำนวณค่าต่างๆ เช่น Accuracy, Precision, Recall, F1-score
    accuracy = accuracy_score(y, y_pred)
    st.subheader(f"Accuracy: :red[{accuracy:.2f}]")

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    # แสดง Precision, Recall และ F1-score
    st.subheader(f"Precision: :green[{precision:.2f}]")
    st.subheader(f"Recall: :orange[{recall:.2f}]")
    st.subheader(f"F1-score: :blue[{f1:.2f}]")

    # สร้าง confusion matrix และแสดงผล
    cm = confusion_matrix(y, y_pred)
    st.subheader("Confusion Matrix : ")
    st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))

    # สร้างกราฟที่แสดงผลการทำนาย
    st.markdown('<h1 class="st-KNN">Titanic Survival Prediction</h1>', unsafe_allow_html=True)

    # สร้างกราฟ scatter plot ที่แสดงผลการทำนาย
    fig, ax = plt.subplots()
    scatter = ax.scatter(interested['Age'], interested['Fare'], c=y_pred, cmap=plt.cm.viridis)
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")
    ax.set_title("Prediction Survived")
    plt.colorbar(scatter, label="Survived (0=No, 1=Yes)")

    # แสดงกราฟ
    st.pyplot(fig)
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
    <h3> สามารถดูโมเดล K-Nearest Neighbors (KNN) และ Support Vector Machine (SVM) </h3>
    <div class="st-container1">
        <div class="st-card1">
            <h2 class="st-element1">Model Machine Learning</h2>
            <a href="/MachineLearningModel">
             <button class="st-button">Click me</button>
            </a> 
        </div>
    </div>
    """,
    unsafe_allow_html=True
)



