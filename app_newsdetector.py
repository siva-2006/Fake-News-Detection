import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Fake News Detection Chatbot", layout="centered")

st.title("📰 Fake News Detection Chatbot")

# Load the fine-tuned transformer model and tokenizer
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="./transformer_fake_news_model", tokenizer="./transformer_fake_news_model")

classifier = load_model()

# Initialize the chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Fake News Detection Assistant. Paste a news headline or article snippet below, and I will verify its authenticity!"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input via the chat input box at the bottom of the screen
if prompt := st.chat_input("Paste news article or snippet here..."):
    
    # 1. Display the user's message in the chat container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Add the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. The Assistant processes the text and responds
    with st.chat_message("assistant"):
        with st.spinner("Analyzing text..."):
            # Get the prediction from the transformer
            result = classifier(prompt)[0]
            confidence = result['score'] * 100
            
            # Format the output based on the prediction
            if result['label'] == 'LABEL_1':
                response = f"**Prediction:** True News ✅\n\n**Confidence Score:** {confidence:.2f}%"
            else:
                response = f"**Prediction:** Fake News 🚨\n\n**Confidence Score:** {confidence:.2f}%"
            
            # Display the response
            st.markdown(response)
    
    # 4. Add the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
