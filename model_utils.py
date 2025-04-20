import re, pickle
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import (
    TfidfVectorizer, CountVectorizer, HashingVectorizer
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

LABEL_ENCODER_PATH = "label_encoder.pkl"


# --------------------------------------------------------------------- #
#  Pre‑processing
# --------------------------------------------------------------------- #
def _clean_transcript(text: str) -> str:
    t = re.sub(r"\d+\.\d+s\s+\d+\.\d+s\s+SPEAKER_\d{2}:", "", text)
    return re.sub(r"\s+", " ", t).strip().lower()


def preprocess_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    df["clean_transcript"] = df["transcript"].apply(_clean_transcript)
    df["clean_nature"]     = df["nature"].str.strip().str.lower()

    le = LabelEncoder()
    df["nature_label"] = le.fit_transform(df["clean_nature"])
    return df, le


# --------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------- #
def train_and_evaluate_encodings(df: pd.DataFrame, le: LabelEncoder) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_transcript"], df["nature_label"], test_size=0.2, random_state=42
    )
    vecs = {
        "TfidfVectorizer": TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                           stop_words="english"),
        "CountVectorizer": CountVectorizer(max_features=5000, ngram_range=(1, 2),
                                           stop_words="english"),
        "HashingVectorizer": HashingVectorizer(n_features=5000, ngram_range=(1, 2),
                                               stop_words="english",
                                               alternate_sign=False)
    }
    out = {}
    for name, vec in vecs.items():
        Xtr = vec.fit_transform(X_train) if name != "HashingVectorizer" else vec.transform(X_train)
        Xte = vec.transform(X_test)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
        y_pred = clf.predict(Xte)

        out[name] = {"accuracy": accuracy_score(y_test, y_pred),
                     "report":   classification_report(y_test, y_pred)}

    return out


def train_and_save_model(df: pd.DataFrame, le: LabelEncoder) -> str:
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_transcript"], df["nature_label"], test_size=0.2, random_state=42
    )
    vec = CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000).fit(Xtr, y_train)

    with open("model_CountVectorizer.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("vectorizer_CountVectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    acc = accuracy_score(y_test, clf.predict(Xte))
    return f"CountVectorizer model saved (accuracy {acc:.3f})"


# --------------------------------------------------------------------- #
#  Inference
# --------------------------------------------------------------------- #
def load_count_model():
    with open("model_CountVectorizer.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("vectorizer_CountVectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return clf, vec, le
