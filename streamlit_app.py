
import streamlit as st
import joblib
import re

st.set_page_config(
    page_title="Sentiment Classifier",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_models():
    saved = joblib.load("best_sentiment_model.joblib")
    return saved["model"], saved["vectorizer"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("💬 Sentiment Classifier")
st.write("Analyze the sentiment of your customer reviews.")

try:
    model, vectorizer = load_models()

    review = st.text_area(
        "Enter your customer review:",
        placeholder="Example: This product is amazing!"
    )

    if st.button("Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            cleaned_text = clean_text(review)
            text_vector = vectorizer.transform([cleaned_text])

            prediction = model.predict(text_vector)[0]

            st.subheader("Prediction")

            if prediction == "positive":
                st.success(f"Positive 😊")
            elif prediction == "negative":
                st.error(f"Negative 😞")
            else:
                st.info(f"Neutral 😐")

            st.write("**Algorithm:** SVM")

except Exception as e:
    st.error(f"Error loading the models: {e}")
