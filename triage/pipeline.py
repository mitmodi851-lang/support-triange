"""TriageSystem: validation + models + safeguards in one easy-to-use object."""
import pandas as pd

from triage import config
from triage.model import build_text_classifier, top_terms
from triage.validation import InvalidTicketError, clean_ticket_text

DISCLAIMER = ("AI suggestion only. A support agent must confirm the category, urgency and team "
              "before any action is taken.")


class TriageSystem:
    def __init__(self, C: float = config.LOGREG_C, threshold: float = config.CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self.category_model = build_text_classifier(C)
        self.urgency_model = build_text_classifier(C)
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "TriageSystem":
        if train_df is None or len(train_df) == 0:
            raise ValueError("Training data is empty.")
        missing = {"text", "category", "urgency"} - set(train_df.columns)
        if missing:
            raise ValueError(f"Training data is missing columns: {sorted(missing)}")
        self.category_model.fit(train_df["text"], train_df["category"])
        self.urgency_model.fit(train_df["text"], train_df["urgency"])
        self.is_fitted = True
        return self

    # ----------------------------------------------------------------- single
    def triage(self, text) -> dict:
        """Return a result dictionary. Never raises for bad user input."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before triage().")
        try:
            cleaned, warnings = clean_ticket_text(text)
        except InvalidTicketError as err:
            return {"status": "invalid_input", "error": str(err), "needs_human_review": True,
                    "review_reasons": ["invalid_input"], "disclaimer": DISCLAIMER}

        cat_probs = self.category_model.predict_proba([cleaned])[0]
        urg_probs = self.urgency_model.predict_proba([cleaned])[0]
        cat_classes, urg_classes = self.category_model.classes_, self.urgency_model.classes_
        cat_idx, urg_idx = int(cat_probs.argmax()), int(urg_probs.argmax())
        category, urgency = str(cat_classes[cat_idx]), str(urg_classes[urg_idx])
        cat_conf, urg_conf = float(cat_probs[cat_idx]), float(urg_probs[urg_idx])

        reasons = []
        if cat_conf < self.threshold:
            reasons.append("low_category_confidence")
        if urg_conf < self.threshold:
            reasons.append("low_urgency_confidence")
        if urgency == "High":
            reasons.append("high_urgency_needs_human_confirmation")
        if any("Very short" in w for w in warnings):
            reasons.append("very_short_text")

        ranked = sorted(zip(cat_classes, cat_probs), key=lambda p: -p[1])[:3]
        return {
            "status": "ok",
            "text": cleaned,
            "category": category,
            "category_confidence": round(cat_conf, 3),
            "category_top3": [(str(c), round(float(p), 3)) for c, p in ranked],
            "urgency": urgency,
            "urgency_confidence": round(urg_conf, 3),
            "recommended_team": config.CATEGORY_TO_TEAM[category],
            "key_terms": top_terms(self.category_model, cleaned, category),
            "needs_human_review": bool(reasons),
            "review_reasons": reasons,
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        }

    # ------------------------------------------------------------------ batch
    def triage_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Triage many tickets. Bad rows get status=invalid_input instead of stopping the run."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Columns present: {list(df.columns)}")
        rows = []
        for value in df[text_column].tolist():
            r = self.triage(value)
            rows.append({
                "text": r.get("text", value),
                "status": r["status"],
                "category": r.get("category"),
                "category_confidence": r.get("category_confidence"),
                "urgency": r.get("urgency"),
                "urgency_confidence": r.get("urgency_confidence"),
                "recommended_team": r.get("recommended_team"),
                "needs_human_review": r["needs_human_review"],
                "review_reasons": ", ".join(r["review_reasons"]),
                "error": r.get("error", ""),
            })
        return pd.DataFrame(rows)


_CACHE = {}


def load_default_system() -> TriageSystem:
    """Train on the bundled simulated tickets (takes about a second). Cached per process."""
    if "system" not in _CACHE:
        train = pd.read_csv(config.TRAIN_PATH)
        _CACHE["system"] = TriageSystem().fit(train)
    return _CACHE["system"]
