## 📰 Fake News Detection Chatbot Overview

* **Objective:** Design and deploy an intelligent conversational chatbot to classify news articles as "Fake" or "True" using a context-aware transformer architecture.
* **Dataset:** Utilizes the popular **"Fake and Real News Dataset"** from Kaggle, containing roughly 45,000 articles divided into true and fake news. 
* **Methodology:** Features are extracted by merging article headlines and body text to capture complete semantic context. The text is processed using `DistilBertTokenizerFast` and fed into a fine-tuned `distilbert-base-uncased` sequence classification model. 
* **Deployment:** Implemented as a conversational interface via **Streamlit**, allowing users to separate input for a news title and its content. The app dynamically predicts the article's authenticity and returns a corresponding confidence score.
* **Website link:** https://fake-news-detector-ppoc.streamlit.app/
