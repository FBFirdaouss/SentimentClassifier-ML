
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Charger les données de test
test = pd.read_csv("dataset/test_all.csv")
test = test.dropna(subset=["sentence"])

def clean_text(text):
    return " ".join(str(text).lower().split())

X_test_text = test["sentence"].apply(clean_text)
y_test = test["label"]

# Charger les modèles et le vectorizer
saved = joblib.load("sentiment_models.joblib")
models = saved["models"]
vectorizer = saved["vectorizer"]

# Transformer les textes avec le même TF-IDF
X_test = vectorizer.transform(X_test_text)

# Évaluer chaque modèle
for name, model in models.items():
    print(f"\n===== {name} =====")

    predictions = model.predict(X_test)

    print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
    print(classification_report(y_test, predictions))
