import streamlit as st
import pickle

# ✅ Load the saved AI model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

# ✅ Title and description
st.title("😊 Sentiment Analyzer")
st.write("Type any text below and I'll tell you if it's **Positive** or **Negative!**")
st.divider()

# ✅ Text input box
user_input = st.text_area("Enter your text here:", height=150, placeholder="e.g. I love this product!")

# ✅ Button
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        # ✅ Convert text to numbers
        input_vectorized = vectorizer.transform([user_input])

        # ✅ Predict sentiment
        prediction = model.predict(input_vectorized)[0]

        # ✅ Get confidence score
        confidence = model.predict_proba(input_vectorized)[0]
        score = round(max(confidence) * 100, 2)

        st.divider()

        # ✅ Show result
        if prediction == 1:
            st.success(f"😊 Positive Sentiment!")
            st.metric(label="Confidence Score", value=f"{score}%")
            st.progress(int(score))
        else:
            st.error(f"😞 Negative Sentiment!")
            st.metric(label="Confidence Score", value=f"{score}%")
            st.progress(int(score))