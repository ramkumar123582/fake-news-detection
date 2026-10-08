import csv
import io
import json
import os
import pickle
from urllib.parse import quote

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from utils import clean_text

app = Flask("app")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

model = pickle.load(open("model.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))
words = np.array(vec.get_feature_names_out())
metrics = json.load(open("metrics.json"))

EXAMPLES = {
    "Write my own": "",
    "Sample: sounds like normal reporting": (
        "The Senate voted 51-49 on Thursday to approve a budget resolution after hours "
        "of debate over tax policy. Republican leaders said the vote clears the way for "
        "committee work, while Democrats argued the plan would raise the deficit, "
        "according to a statement from the Senate Budget Committee."),
    "Sample: sounds like clickbait": (
        "SHOCKING! You will not believe what the mainstream media is hiding from you. "
        "Insiders reveal the secret plot that the establishment wants to bury. "
        "Share this before they delete it!!!"),
}


def predict(texts):
    X = vec.transform([clean_text(t) for t in texts])
    proba = model.predict_proba(X)
    out = []
    for i in range(X.shape[0]):
        p_real = proba[i][1]
        label = "REAL" if p_real >= 0.5 else "FAKE"
        out.append((label, round(max(p_real, 1 - p_real) * 100, 1), X[i]))
    return out


def explain(row, n=8):
    contrib = row.multiply(model.coef_[0]).tocoo()
    pairs = sorted(zip(contrib.col, contrib.data), key=lambda p: abs(p[1]), reverse=True)[:n]
    return [dict(word=words[i], side="real" if v > 0 else "fake") for i, v in pairs]


def render(**kw):
    kw.setdefault("tab", "single")
    return render_template("index.html", metrics=metrics, examples=EXAMPLES, **kw)


@app.route("/")
def home():
    return render()


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("news", "").strip()
    if not text:
        return render(error="Paste a news article or headline first.", text=text)
    label, conf, row = predict([text])[0]
    result = dict(label=label, conf=conf, top=explain(row),
                  short=len(text.split()) < 30)
    return render(result=result, text=text)


@app.route("/batch", methods=["POST"])
def batch():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"):
        return render(tab="batch", batch_error="Upload a .csv file.")
    try:
        df = pd.read_csv(f).head(1000)
    except Exception:
        return render(tab="batch", batch_error="Could not read this CSV file.")
    if "text" not in df.columns:
        return render(tab="batch",
                      batch_error="The CSV needs a column named 'text' (optional: 'title').")
    texts = (df["title"].fillna("") + " " + df["text"].fillna("")
             if "title" in df.columns else df["text"].fillna("")).astype(str).tolist()
    preds = predict(texts)
    rows = [dict(snippet=t[:110] + ("..." if len(t) > 110 else ""), label=p[0], conf=p[1])
            for t, p in zip(texts, preds)]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["text", "prediction", "confidence_percent"])
    for t, p in zip(texts, preds):
        w.writerow([t, p[0], p[1]])
    summary = dict(total=len(rows),
                   fake=sum(r["label"] == "FAKE" for r in rows),
                   real=sum(r["label"] == "REAL" for r in rows))
    return render(tab="batch", rows=rows[:50], summary=summary,
                  csv_uri="data:text/csv;charset=utf-8," + quote(buf.getvalue()))


if not os.environ.get("RENDER"):
    app.run(debug=False)