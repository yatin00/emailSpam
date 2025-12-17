# ---------------------------
# 1. Import Libraries
# ---------------------------
import streamlit as st
import joblib
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------------------
# 2. Load Model & Vectorizer
# ---------------------------
model = joblib.load('models/spam_detection_model.pkl')    # Your trained Logistic Regression model
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')   # Your saved TF-IDF vectorizer

# ---------------------------
# 3. Preprocessing Function
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\S+@\S+', '', text)          # Remove emails
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)              # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces
    return ' '.join([word for word in text.split() if word not in stop_words])

# ---------------------------
# 4. Streamlit Interface
# ---------------------------
st.set_page_config(page_title="Spam Email Detector", page_icon="📧")
st.title("📧 Spam Email Detector")
st.write("Enter any text below to check if it's Spam or Not Spam:")

user_input = st.text_area("Enter text here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze!")
    else:
        # Preprocess and transform input
        clean_input = clean_text(user_input)
        vector_input = vectorizer.transform([clean_input])
        
        # Prediction
        try:
            proba = model.predict_proba(vector_input)[0][1]
        except AttributeError:
            score = model.decision_function(vector_input)[0]
            proba = 1 / (1 + np.exp(-score))
        
        # Show result
        if proba > 0.5:
            st.error(f"🚫 This is SPAM! Probability: {proba:.2f}")
        else:
            st.success(f"✅ This is NOT SPAM. Probability: {proba:.2f}")
