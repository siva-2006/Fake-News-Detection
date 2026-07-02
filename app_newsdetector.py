import streamlit as st
from transformers import pipeline

# Configure the Streamlit page
st.set_page_config(page_title="Fake News Detection Chatbot", layout="centered")

st.title("📰 Fake News Detection Chatbot")
st.write("Paste a news article or snippet below to verify its authenticity.")

# Cache the model so it only loads once when the app starts
@st.cache_resource
def load_model():
    # Make sure the folder name exactly matches where your model files are saved
    return pipeline("text-classification", model="./transformer_fake_news_model", tokenizer="./transformer_fake_news_model")

classifier = load_model()

# User input text box
user_input = st.text_area("Enter News Text:", height=200)

if st.button("Detect News"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Get the prediction from the transformer
        with st.spinner("Analyzing text..."):
            result = classifier(user_input)[0]
        
        # Process output (Hugging Face outputs LABEL_1 for true, LABEL_0 for fake based on our mapping)
        if result['label'] == 'LABEL_1':
            st.success("**Prediction:** True News")
        else:
            st.error("**Prediction:** Fake News")
            
        # Display the confidence score
        confidence = result['score'] * 100
        st.info(f"**Confidence Score:** {confidence:.2f}%")
