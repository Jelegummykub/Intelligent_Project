import streamlit as st
import pathlib

def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
css_path = pathlib.Path("css/style.css")
load_css(css_path)

st.markdown('<h1 class="st-header">Project Intelligent</h1>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="st-container">
        <div class="st-card">
            <h2 class="st-element1">Machine Learning</h2>
            <a href="/MachineLearning">
             <button class="st-button">Click me</button>
            </a> 
        </div>
         <div class="st-card1">
            <h2 class="st-element1">Neural Network</h2>
             <a href="/NeuralNetwork">
             <button class="st-button1">Click me</button>
            </a> 
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
