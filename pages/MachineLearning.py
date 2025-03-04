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

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">ขั้นตอนนี้ผมได้ทำการอ่านข้อมูลจากไฟล์ CSV และสุ่มข้อมูลจำนวน 100 แถวจาก DataFrame พร้อมทั้งแสดงตัวอย่างข้อมูลที่เราสนใจ
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    df = pd.read_csv("data/Titanic.csv")

    df_sample = df.sample(n=100, random_state=42)

    interested = pd.DataFrame({
        'Pclass': df_sample['Pclass'],
        'Sex': df_sample['Sex'].map({'male': 0, 'female': 1}),
        'Age': df_sample['Age'].fillna(df['Age'].mean()),
        'Fare': df_sample['Fare'],
        'Survived': df_sample['Survived']
    })

    st.subheader("ตัวอย่างข้อมูล")
    st.write(interested.head())
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")


st.markdown(
    """
        <h5 class="st-element1">จากนั้นทำการแยกข้อมูลออกเป็น features (X) และ label (y) และสร้างโมเดล K-Nearest Neighbors (KNN) โดยเลือก k = 3
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    
    X = interested.drop(columns=['Survived'])
    y = interested['Survived']

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    y_pred = knn.predict(X)
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">ขั้นตอนนี้ผมได้คำนวณความแม่นยำ Accuracy, Precision, Recall, F1-score, Confusion Matrix และแสดงผลลัพธ์ที่ได้
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    accuracy = accuracy_score(y, y_pred)
    st.subheader(f"Accuracy: :red[{accuracy:.2f}]")

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    st.subheader(f"Precision: :green[{precision:.2f}]")
    st.subheader(f"Recall: :orange[{recall:.2f}]")
    st.subheader(f"F1-score: :blue[{f1:.2f}]")

    cm = confusion_matrix(y, y_pred)
    st.subheader("Confusion Matrix : ")
    st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">ทำการแสดงกราฟ Scatter Plot เพื่อแสดงผลการทำนาย
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    fig, ax = plt.subplots()
    scatter = ax.scatter(interested['Age'], interested['Fare'], c=y_pred, cmap=plt.cm.viridis)
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")
    ax.set_title("Prediction Survived")
    plt.colorbar(scatter, label="Survived (0=No, 1=Yes)")

    st.pyplot(fig)
    
    ```
    """,
    unsafe_allow_html=True
)


st.subheader("")

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


st.subheader("")


st.markdown(
    """
        <h5 class="st-element1">ขั้นตอนนี้ผมได้ทำการอ่านข้อมูลจากไฟล์ CSV และสุ่มข้อมูลจำนวน 100 แถวจาก DataFrame พร้อมทั้งแสดงตัวอย่างข้อมูลที่เราสนใจ
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    
    df = pd.read_csv("data/Titanic.csv")

    df_sample = df.sample(n=100, random_state=42)

    interested = pd.DataFrame({
        'Pclass': df_sample['Pclass'],
        'Sex': df_sample['Sex'].map({'male': 0, 'female': 1}),
        'Age': df_sample['Age'].fillna(df['Age'].mean()),
        'Fare': df_sample['Fare'],
        'Survived': df_sample['Survived']
    })

    st.subheader("ตัวอย่างข้อมูล")
    st.write(interested.head())
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")

st.markdown(
    """
        <h5 class="st-element1">ขั้นตอนนี้คือผมได้แยกคุณสมบัติ (features) และเป้าหมาย (target) และสร้างโมเดล Support Vector Machine (SVM) โดยใช้ kernel แบบ rbf และได้แบ่งข้อมูลเป็นชุด train โดยใช้ test_size=0.2
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    
    X = interested.drop(columns=['Survived'])
    y = interested['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = SVC(kernel='rbf', gamma='scale')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")


st.markdown(
    """
        <h5 class="st-element1">คำนวณค่า Accuracy, Precision, Recall, F1-score, Confusion Matrix และแสดงผลลัพธ์ที่ได้
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    
    accuracy = accuracy_score(y, y_pred)
    st.subheader(f"Accuracy: :red[{accuracy:.2f}]")

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    st.subheader(f"Precision: :green[{precision:.2f}]")
    st.subheader(f"Recall: :orange[{recall:.2f}]")
    st.subheader(f"F1-score: :blue[{f1:.2f}]")

    cm = confusion_matrix(y, y_pred)
    st.subheader("Confusion Matrix : ")
    st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))
    
    ```
    """,
    unsafe_allow_html=True
)

st.subheader("")


st.markdown(
    """
        <h5 class="st-element1">ทำการแสดงกราฟ Scatter Plot เพื่อแสดงผลการทำนาย
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    ```python
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(interested['Age'], interested['Fare'], c=y_pred, cmap=plt.cm.viridis)
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")
    ax.set_title("Prediction Survived")
    plt.colorbar(scatter, label="Survived (0=No, 1=Yes)")

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
            <h4 class="st-element1">Model Machine Learning</h4>
            <a href="/MachineLearningModel">
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
        - สามารถดู code ของ Model K-Nearest Neighbors (KNN) และ Support Vector Machine (SVM) ได้ที่
        <a href="https://github.com/Jelegummykub/Intelligent_Project/blob/main/pages/MachineLearningModel.py" target="_blank">github.com/Jelegummy</a> 
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



