
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger les données
train = pd.read_csv("dataset/train_all.csv")
validation = pd.read_csv("dataset/val_all.csv")
test = pd.read_csv("dataset/test_all.csv")

# Nettoyer les textes
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Supprimer les lignes sans texte
train = train.dropna(subset=["sentence"])
validation = validation.dropna(subset=["sentence"])
test = test.dropna(subset=["sentence"])

# Appliquer le nettoyage
train["sentence"] = train["sentence"].apply(clean_text)
validation["sentence"] = validation["sentence"].apply(clean_text)
test["sentence"] = test["sentence"].apply(clean_text)

# Initialiser TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)

# Apprendre le vocabulaire sur Train uniquement
X_train = vectorizer.fit_transform(train["sentence"])

# Transformer Validation et Test
X_val = vectorizer.transform(validation["sentence"])
X_test = vectorizer.transform(test["sentence"])

print("Dimensions des données :")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

print("\nExemple de vecteur TF-IDF :")
print(X_train[0].toarray())
