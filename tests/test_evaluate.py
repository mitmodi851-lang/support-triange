import pytest

from triage.evaluate import compute_metrics, run_evaluation


def test_metrics_perfect_and_known_values():
    labels = ["a", "b"]
    perfect = compute_metrics(["a", "b", "a"], ["a", "b", "a"], labels)
    assert perfect["accuracy"] == 1.0 and perfect["macro_f1"] == 1.0
    mixed = compute_metrics(["a", "a", "b", "b"], ["a", "b", "b", "b"], labels)
    assert mixed["accuracy"] == pytest.approx(0.75)
    assert mixed["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert mixed["per_class"]["a"]["recall"] == pytest.approx(0.5)
    assert mixed["confusion_matrix"] == [[1, 1], [0, 2]]


def test_full_evaluation_runs_and_beats_majority():
    res = run_evaluation(write_files=False)
    assert res["leakage_overlap_count"] == 0
    for task in ("category", "urgency"):
        m = res["main_eval"][task]
        assert m["TF-IDF + Logistic Regression"]["accuracy"] > m["Majority class"]["accuracy"]
