import streamlit as st
import pickle
import numpy as np

# ==========================
# Load Saved Models
# ==========================
tf = pickle.load(open("tfidfssss.pkl", "rb"))
mn = pickle.load(open("mnssss.pkl", "rb"))
le = pickle.load(open("encoderssss.pkl", "rb"))

# ==========================
# Streamlit Page Setup
# ==========================
st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️‍♂️")
st.title("📰 Fake News Detection App")
st.write("This app uses a trained **TF-IDF + Naive Bayes** model to predict whether a news article is *Fake* or *Real*.")

# ==========================
# User Input
# ==========================
news_text = st.text_area("✍️ Enter News Content Here", height=200)

# ==========================
# Predict Button
# ==========================
if st.button("🔍 Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess and transform
        vectorized_input = tf.transform([news_text]).toarray()
        prediction = mn.predict(vectorized_input)
        label = le.inverse_transform(prediction)[0]

        # Display result
        if label.lower() == "fake":
            st.error("🚨 The news appears to be **FAKE**.")
        else:
            st.success("✅ The news appears to be **REAL**.")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Scikit-learn, and TF-IDF.")
