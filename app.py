import streamlit as st
from src.inference import prediction

st.title("Customer Complaint Classifier")

text = st.text_area("Enter your complaint:")

if st.button("Predict"):
    if text.strip() != "":
        result = prediction(text)
        st.write("Prediction:", result)
    else:
        st.write("Please enter some text")