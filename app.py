import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Train and save model if not exists
def train_model():
    texts = [
        "I love this movie", "This is amazing", "Absolutely wonderful experience",
        "Best thing ever", "I really enjoyed this", "Fantastic and brilliant",
        "Great product highly recommend", "So happy with this purchase",
        "Excellent quality and service", "Very good and satisfying",
        "I hate this movie", "This is terrible", "Worst experience ever",
        "Absolutely horrible", "I really disliked this", "Very disappointing",
        "Bad product do not buy", "So unhappy with this purchase",
        "Poor quality and service", "Very bad and unsatisfying"
    ]
    labels = [1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    return model, vectorizer

# ✅ Load model
model, vectorizer = train_model()

# ✅ Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

# ✅ Title
st.title("😊 Sentiment Analyzer")
st.write("Type any text below and I'll tell you if it's **Positive** or **Negative!**")
st.divider()

# ✅ Text input
user_input = st.text_area("Enter your text here:", height=150, placeholder="e.g. I love this product!")

# ✅ Button
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)[0]
        confidence = model.predict_proba(input_vectorized)[0]
        score = round(max(confidence) * 100, 2)

        st.divider()
        if prediction == 1:
            st.success(f"😊 Positive Sentiment!")
            st.metric(label="Confidence Score", value=f"{score}%")