"""Simple rule-based baselines. The ML model must beat these to be worth using."""
import re

# Keywords per category. Hand-written, deliberately simple.
CATEGORY_KEYWORDS = {
    "Billing & Payments": ["invoice", "charge", "charged", "billing", "bill", "billed", "payment", "pay", "card", "price", "receipt", "gst", "tax"],
    "Technical Issue": ["error", "crash", "crashes", "bug", "slow", "freeze", "freezes", "app", "website", "api", "loading", "install", "server", "timeout", "feature"],
    "Account Access": ["password", "login", "log in", "sign in", "locked", "account", "otp", "username", "verification", "two-factor", "profile"],
    "Shipping & Delivery": ["delivery", "deliver", "delivered", "shipping", "shipped", "parcel", "package", "courier", "tracking", "dispatch", "order"],
    "Refund & Cancellation": ["refund", "cancel", "cancelled", "cancellation", "return", "returned", "money back", "chargeback"],
}
DEFAULT_CATEGORY = "General Inquiry"

HIGH_WORDS = ["urgent", "urgently", "immediately", "asap", "right now", "emergency", "today", "tomorrow", "midnight", "fraud", "locked out", "unauthorized", "without permission"]
MEDIUM_WORDS = ["not working", "error", "failed", "wrong", "damaged", "missing", "cancel", "crash", "not received", "declined", "problem", "stuck"]


def _count_hits(text: str, words) -> int:
    text = text.lower()
    return sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", text))


class KeywordBaseline:
    """Counts keyword hits. No training; same output every time."""

    def predict_category(self, texts):
        out = []
        for t in texts:
            scores = {c: _count_hits(t, kws) for c, kws in CATEGORY_KEYWORDS.items()}
            best = max(scores, key=scores.get)   # ties -> first category in dict order
            out.append(best if scores[best] > 0 else DEFAULT_CATEGORY)
        return out

    def predict_urgency(self, texts):
        out = []
        for t in texts:
            if _count_hits(t, HIGH_WORDS) > 0:
                out.append("High")
            elif _count_hits(t, MEDIUM_WORDS) > 0:
                out.append("Medium")
            else:
                out.append("Low")
        return out
