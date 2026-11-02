
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
    saved = joblib.load("sentiment_models.joblib")
    return saved["models"], saved["vectorizer"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("💬 Sentiment Classifier")
st.write("Analyze the sentiment of your customer reviews.")

try:
    models, vectorizer = load_models()

    model_name = st.selectbox(
        "Choose an algorithm:",
        ["SVM", "Random Forest", "Naive Bayes"]
    )

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

            prediction = models[model_name].predict(text_vector)[0]

            st.subheader("Prediction")

            if prediction == "positive":
                st.success(f"Positive 😊")
            elif prediction == "negative":
                st.error(f"Negative 😞")
            else:
                st.info(f"Neutral 😐")

            st.write(f"**Algorithm:** {model_name}")

except Exception as e:
    st.error(f"Error loading the models: {e}")
