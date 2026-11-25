"""The ML approach: TF-IDF text features + Logistic Regression.

TF-IDF turns each ticket into numbers (words that are rare but meaningful get
higher weight). Logistic Regression then learns which words point to which
label and gives a probability for each label.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triage.config import LOGREG_C, SEED


def build_text_classifier(C: float = LOGREG_C, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def top_terms(pipeline: Pipeline, text: str, label: str, k: int = 5):
    """Words in `text` that pushed the model towards `label` (simple explanation)."""
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    x = tfidf.transform([text])
    class_index = list(clf.classes_).index(label)
    contributions = x.multiply(clf.coef_[class_index]).tocoo()
    names = tfidf.get_feature_names_out()
    pairs = sorted(zip(contributions.col, contributions.data), key=lambda p: -p[1])
    return [names[i] for i, v in pairs if v > 0][:k]
