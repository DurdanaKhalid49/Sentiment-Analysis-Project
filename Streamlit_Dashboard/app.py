import streamlit as st
import joblib
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(".."))
from text_cleaner import clean_text 
# Load model and label encoder
pipeline = joblib.load("../model/sentiment_pipeline_model.joblib")
label_encoder = joblib.load("../model/label_encoder.joblib")

# Streamlit Dashboard
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🎯 Multiclass Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("### Enter your text below:")

user_input = st.text_area("✍️ Text to Analyze", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_series = pd.Series([user_input])
        prediction = pipeline.predict(input_series)[0]
        sentiment = label_encoder.inverse_transform([prediction])[0]
        
        # Styled Output
        if sentiment == "positive":
            st.success(f" **Predicted Sentiment:** {sentiment.upper()}")
        elif sentiment == "negative":
            st.error(f" **Predicted Sentiment:** {sentiment.upper()}")
        else:
            st.info(f"💬 **Predicted Sentiment:** {sentiment.upper()}")

# Optional Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Made by Durdana Khalid</p>", unsafe_allow_html=True)
