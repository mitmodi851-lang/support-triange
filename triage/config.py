"""Central configuration. Change settings here, not inside the logic files."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
TRAIN_PATH = DATA_DIR / "train_tickets.csv"   # generated, simulated tickets (training only)
EVAL_PATH = DATA_DIR / "eval_tickets.csv"     # hand-written tickets (evaluation only)

SEED = 42  # one seed used everywhere so results are reproducible

CATEGORIES = [
    "Billing & Payments",
    "Technical Issue",
    "Account Access",
    "Shipping & Delivery",
    "Refund & Cancellation",
    "General Inquiry",
]
URGENCY_LEVELS = ["Low", "Medium", "High"]

# Routing rule: the predicted category decides the team.
CATEGORY_TO_TEAM = {
    "Billing & Payments": "Billing Team",
    "Technical Issue": "Technical Support",
    "Account Access": "Identity & Access Team",
    "Shipping & Delivery": "Logistics Support",
    "Refund & Cancellation": "Refunds & Retention Team",
    "General Inquiry": "Customer Care (General)",
}

# --- Model and safeguard settings -------------------------------------------
LOGREG_C = 10.0               # regularisation strength of Logistic Regression
CONFIDENCE_THRESHOLD = 0.45   # below this, a person must review (chosen up-front, NOT tuned on the evaluation set)
MIN_WORDS = 3                 # very short tickets are too vague to trust
MAX_TEXT_CHARS = 5000         # longer text is truncated (protects the app)
