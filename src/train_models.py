
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report

# 1. Charger les données
train = pd.read_csv("dataset/train_all.csv")
validation = pd.read_csv("dataset/val_all.csv")

# 2. Nettoyer les textes
def clean_text(text):
    return " ".join(str(text).lower().split())

X_train_text = train["sentence"].dropna().apply(clean_text)
y_train = train.loc[X_train_text.index, "label"]

X_val_text = validation["sentence"].dropna().apply(clean_text)
y_val = validation.loc[X_val_text.index, "label"]

# 3. Extraction TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

# 4. Définir les modèles
models = {
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Naive Bayes": MultinomialNB()
}

trained_models = {}
best_score = 0
best_model_name = None

# 5. Entraîner et comparer
for name, model in models.items():
    print(f"\n===== {name} =====")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    score = accuracy_score(y_val, predictions)

    print("Accuracy :", round(score, 4))
    print(classification_report(y_val, predictions))

    trained_models[name] = model

    if score > best_score:
        best_score = score
        best_model_name = name



# 7. Sauvegarder les trois modèles et TF-IDF
joblib.dump(
    {
        "models": trained_models,
        "vectorizer": vectorizer,
        "best_model_name": best_model_name
    },
    "sentiment_models.joblib"
)

print("\nMeilleur modèle :", best_model_name)
print("Meilleure accuracy :", round(best_score, 4))
print("Les trois modèles ont été sauvegardés.")
