import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from utils import clean_text

fake = pd.read_csv("Fake.csv"); fake["label"] = 0
real = pd.read_csv("True.csv"); real["label"] = 1
df = pd.concat([fake, real], ignore_index=True)

df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)
df = df[df["content"].str.len() > 20].drop_duplicates("content")
df = df.sample(frac=1, random_state=42)
print("Articles after cleaning:", len(df))

X_tr, X_te, y_tr, y_te = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
vec = TfidfVectorizer(stop_words="english", max_df=0.7, min_df=2,
                      ngram_range=(1, 2), max_features=50000)
Xtr = vec.fit_transform(X_tr)
Xte = vec.transform(X_te)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50, random_state=42),
}
rows, fitted = [], {}
for name, m in models.items():
    m.fit(Xtr, y_tr)
    pred = m.predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary")
    rows.append(dict(name=name, accuracy=round(accuracy_score(y_te, pred) * 100, 2),
                     precision=round(p * 100, 2), recall=round(r * 100, 2),
                     f1=round(f * 100, 2)))
    fitted[name] = m
    print(rows[-1])

best = fitted["Logistic Regression"]
cm = confusion_matrix(y_te, best.predict(Xte)).tolist()
words = np.array(vec.get_feature_names_out())
order = np.argsort(best.coef_[0])

metrics = dict(
    dataset=dict(total=int(len(df)), fake=int((df.label == 0).sum()),
                 real=int((df.label == 1).sum()), train=int(len(X_tr)), test=int(len(X_te))),
    models=rows, deployed="Logistic Regression", confusion=cm,
    top_real=words[order[-15:]][::-1].tolist(),
    top_fake=words[order[:15]].tolist(),
)

pickle.dump(best, open("model.pkl", "wb"))
pickle.dump(vec, open("vectorizer.pkl", "wb"))
json.dump(metrics, open("metrics.json", "w"), indent=2)
print("Saved model.pkl, vectorizer.pkl, metrics.json")