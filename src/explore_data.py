import pandas as pd

train = pd.read_csv("dataset/train_all.csv")
validation = pd.read_csv("dataset/val_all.csv")
test = pd.read_csv("dataset/test_all.csv")

print("Colonnes :", train.columns.tolist())

print("\nPremières lignes :")
print(train.head())

print("\nValeurs manquantes :")
print(train.isnull().sum())

print("\nRépartition des sentiments :")
print(train["label"].value_counts())