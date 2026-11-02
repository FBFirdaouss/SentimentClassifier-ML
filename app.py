from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os

app = Flask(__name__, static_folder="frontend")
CORS(app)

saved = joblib.load("sentiment_models.joblib")
models = saved["models"]
vectorizer = saved["vectorizer"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@app.route("/", methods=["GET"])
def home():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Please provide a text"}), 400

    text = data["text"]
    model_name = data.get("model", "SVM")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text must not be empty"}), 400

    if model_name not in models:
        return jsonify({"error": f"Model {model_name} not found"}), 400

    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    model = models[model_name]
    prediction = model.predict(text_vector)[0]

    return jsonify({
        "text": text,
        "model": model_name,
        "sentiment": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))