import math

import pandas as pd
import pytest

from triage.baseline import KeywordBaseline
from triage.pipeline import TriageSystem


def test_result_structure(system):
    r = system.triage("I was charged twice for my plan and need the money back immediately.")
    assert r["status"] == "ok"
    for key in ("category", "urgency", "recommended_team", "category_confidence", "needs_human_review", "disclaimer"):
        assert key in r
    assert 0 <= r["category_confidence"] <= 1
    assert math.isclose(sum(p for _, p in system.triage("password reset email never arrives")["category_top3"]), 1, abs_tol=0.5)


def test_team_matches_category(system):
    from triage.config import CATEGORY_TO_TEAM
    r = system.triage("My parcel has not been delivered and tracking is stuck.")
    assert r["recommended_team"] == CATEGORY_TO_TEAM[r["category"]]


@pytest.mark.parametrize("bad", [None, "", "   ", 123])
def test_invalid_input_does_not_crash(system, bad):
    r = system.triage(bad)
    assert r["status"] == "invalid_input"
    assert r["needs_human_review"] is True
    assert "error" in r


def test_gibberish_goes_to_human_review(system):
    r = system.triage("zxqv wplk mnbv")
    assert r["status"] == "ok" and r["needs_human_review"] is True


def test_very_short_goes_to_human_review(system):
    r = system.triage("help")
    assert "very_short_text" in r["review_reasons"]


def test_high_urgency_always_needs_confirmation(system):
    r = system.triage("Our entire production system is down and customers cannot log in, fix this immediately!")
    if r["urgency"] == "High":
        assert r["needs_human_review"] is True


def test_training_is_reproducible(train_df):
    a, b = TriageSystem().fit(train_df), TriageSystem().fit(train_df)
    texts = ["refund for my cancelled order please", "app crashes on startup", "what are your office hours"]
    assert [a.triage(t)["category"] for t in texts] == [b.triage(t)["category"] for t in texts]
    assert a.triage(texts[0])["category_confidence"] == b.triage(texts[0])["category_confidence"]


def test_unfitted_system_raises():
    with pytest.raises(RuntimeError):
        TriageSystem().triage("hello there friend")


def test_fit_rejects_bad_training_data():
    with pytest.raises(ValueError):
        TriageSystem().fit(pd.DataFrame())
    with pytest.raises(ValueError):
        TriageSystem().fit(pd.DataFrame({"text": ["a b c"]}))


def test_batch_handles_bad_rows(system):
    df = pd.DataFrame({"text": ["Please refund my order", "", None, "The app keeps crashing on login page"]})
    out = system.triage_dataframe(df)
    assert list(out["status"]) == ["ok", "invalid_input", "invalid_input", "ok"]
    with pytest.raises(ValueError):
        system.triage_dataframe(pd.DataFrame({"wrong_column": ["x"]}))


def test_keyword_baseline_basics():
    kb = KeywordBaseline()
    assert kb.predict_category(["I need a refund"]) == ["Refund & Cancellation"]
    assert kb.predict_category(["hello there"]) == ["General Inquiry"]
    assert kb.predict_urgency(["this is urgent"]) == ["High"]
    assert kb.predict_urgency(["nice product"]) == ["Low"]
