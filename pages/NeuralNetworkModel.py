import streamlit as st
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("css/style.css")
load_css(css_path)

st.markdown('<h1 class="st-KNN">Pokemon Attack Prediction with MLP</h1>', unsafe_allow_html=True)


df = pd.read_csv("data/pokemon.csv")

st.subheader("")

st.subheader("ตัวอย่างข้อมูล")

st.write(df.head())

label_encoder = LabelEncoder()

df['gender'] = label_encoder.fit_transform(df['gender'].astype(str))
df['category'] = label_encoder.fit_transform(df['category'].astype(str))
df['abilities'] = label_encoder.fit_transform(df['abilities'].astype(str))
df['weakness'] = label_encoder.fit_transform(df['weakness'].astype(str))

features = df[['gender', 'category', 'abilities', 'weakness', 'height', 'weight', 'attack', 'defense', 'hp', 'special_attack', 'special_defense', 'speed']]

target = df['attack']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)

y_pred = mlp_model.predict(X_test)

st.markdown("### ผลการประเมินโมเดล:")

report = classification_report(y_test, y_pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

plt.figure(figsize=(10, 6))

st.subheader("")


st.markdown('<h1 class="st-KNN">Pokemon Attack Prediction</h1>', unsafe_allow_html=True)


sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='การทำนายที่สมบูรณ์แบบ')

plt.title("Predicted vs Actual Attack Values")
plt.xlabel("Actual Attack")
plt.ylabel("Predicted Attack")

st.pyplot(plt)