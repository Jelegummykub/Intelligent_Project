import streamlit as st
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("css/style.css")
load_css(css_path)

st.subheader("ผมได้เตรียมโมเดลสำหรับการทำนายไว้ ได้แก่ K-Nearest Neighbors (KNN) และ Support Vector Machine (SVM)")
st.image("public/images/titanic.jpg")

model_option = st.selectbox("", ["กรุณาเลือกโมเดล", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"], index=0)

if model_option == "กรุณาเลือกโมเดล":
    st.subheader("")
    

    

else:
    st.markdown(f'<h1 class="st-KNN">Titanic Survival Prediction with {model_option}</h1>', unsafe_allow_html=True)
    
    df = pd.read_csv("data/Titanic.csv")
    df_sample = df.sample(n=50 if model_option == "KNN" else 30, random_state=42)
    
    interested = pd.DataFrame({
        'Pclass': df_sample['Pclass'],
        'Sex': df_sample['Sex'].map({'male': 0, 'female': 1}),
        'Age': df_sample['Age'].fillna(df['Age'].mean()),  
        'Fare': df_sample['Fare'],
        'Survived': df_sample['Survived']
    })
    
    st.subheader("ตัวอย่างข้อมูล")
    st.write(interested.head())
    
    X = interested.drop(columns=['Survived'])
    y = interested['Survived']
    
    if model_option == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=3)
    else:
        model = SVC(kernel='linear')
    
    model.fit(X, y)
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    st.subheader(f"Accuracy: :red[{accuracy:.2f}]")
    st.subheader(f"Precision: :green[{precision:.2f}]")
    st.subheader(f"Recall: :orange[{recall:.2f}]")
    st.subheader(f"F1-score: :blue[{f1:.2f}]")
    
    cm = confusion_matrix(y, y_pred)
    st.subheader("Confusion Matrix : ")
    st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(interested['Age'], interested['Fare'], c=y_pred, cmap=plt.cm.viridis)
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")
    ax.set_title("Prediction Survived")
    plt.colorbar(scatter, label="Survived (0=No, 1=Yes)")
    
    st.pyplot(fig)