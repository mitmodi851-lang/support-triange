"""Evaluation: baselines vs ML model, on the SAME hand-written evaluation tickets.

Run:  python -m triage.evaluate
Writes everything to evaluation/results/ (metrics.json, a Markdown report, error CSVs).
All numbers come from running the code - nothing is typed in by hand.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

from triage import config
from triage.baseline import KeywordBaseline
from triage.data_generation import generate_training_tickets
from triage.model import build_text_classifier
from triage.pipeline import TriageSystem


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, labels):
    """Accuracy + macro precision/recall/F1 + per-class numbers + confusion matrix."""
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp), "macro_recall": float(mr),
        "macro_f1": float(mf), "weighted_f1": float(wf),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, lab in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels),
    }


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower()).strip()


def check_no_leakage(train_df, eval_df):
    """The evaluation tickets must never appear in the training data."""
    overlap = set(train_df["text"].map(_normalise)) & set(eval_df["text"].map(_normalise))
    return sorted(overlap)


def _md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(x):
    return f"{x:.3f}"


# --------------------------------------------------------------------------- main run
def run_evaluation(write_files: bool = True) -> dict:
    seed = config.SEED
    if not config.TRAIN_PATH.exists():
        config.TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_training_tickets().to_csv(config.TRAIN_PATH, index=False)
    train = pd.read_csv(config.TRAIN_PATH)
    evald = pd.read_csv(config.EVAL_PATH)

    leaks = check_no_leakage(train, evald)
    if leaks:
        raise RuntimeError(f"Data leakage: {len(leaks)} evaluation tickets also appear in training data.")

    results = {"seed": seed, "n_train": len(train), "n_eval": len(evald),
               "train_distribution": {"category": train["category"].value_counts().to_dict(),
                                      "urgency": train["urgency"].value_counts().to_dict()},
               "eval_distribution": {"category": evald["category"].value_counts().to_dict(),
                                     "urgency": evald["urgency"].value_counts().to_dict()},
               "leakage_overlap_count": len(leaks)}

    # ---- A. Main comparison on the hand-written evaluation set ---------------
    system = TriageSystem().fit(train)
    keyword = KeywordBaseline()
    texts = evald["text"].tolist()
    tasks = {"category": config.CATEGORIES, "urgency": config.URGENCY_LEVELS}
    main = {}
    for task, labels in tasks.items():
        majority = DummyClassifier(strategy="most_frequent").fit(train["text"], train[task])
        model = system.category_model if task == "category" else system.urgency_model
        keyword_pred = keyword.predict_category(texts) if task == "category" else keyword.predict_urgency(texts)
        preds = {"Majority class": majority.predict(texts).tolist(),
                 "Keyword rules (baseline)": keyword_pred,
                 "TF-IDF + Logistic Regression": model.predict(texts).tolist()}
        main[task] = {name: compute_metrics(evald[task], p, labels) for name, p in preds.items()}
        results.setdefault("_preds", {})[task] = preds
    results["main_eval"] = main

    # ---- B. Development check: hold out whole templates ---------------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_idx, te_idx = next(splitter.split(train, groups=train["template_id"]))
    dev_train, dev_test = train.iloc[tr_idx], train.iloc[te_idx]
    dev_system = TriageSystem().fit(dev_train)
    dev = {}
    for task, labels in tasks.items():
        model = dev_system.category_model if task == "category" else dev_system.urgency_model
        kw = keyword.predict_category(dev_test["text"].tolist()) if task == "category" else keyword.predict_urgency(dev_test["text"].tolist())
        dev[task] = {"Keyword rules (baseline)": compute_metrics(dev_test[task], kw, labels),
                     "TF-IDF + Logistic Regression": compute_metrics(dev_test[task], model.predict(dev_test["text"]), labels)}
    results["dev_split"] = {"n_train": len(dev_train), "n_test": len(dev_test),
                            "templates_held_out": int(dev_test["template_id"].nunique()), "metrics": dev}

    # ---- C. Sensitivity of the main hyper-parameter (grouped CV, training data only)
    cv = GroupKFold(n_splits=5)
    sens = {}
    for C in [1.0, 10.0, 100.0]:
        scores = cross_val_score(build_text_classifier(C), train["text"], train["category"],
                                 groups=train["template_id"], cv=cv, scoring="f1_macro")
        sens[str(C)] = {"mean_macro_f1": float(scores.mean()), "std": float(scores.std())}
    results["c_sensitivity_grouped_cv"] = sens

    # ---- D. Safeguard check: do low-confidence flags catch the mistakes? ----
    triaged = [system.triage(t) for t in texts]
    cat_ok = np.array([r["category"] == t for r, t in zip(triaged, evald["category"])])
    low_conf = np.array(["low_category_confidence" in r["review_reasons"] for r in triaged])
    flagged = np.array([r["needs_human_review"] for r in triaged])
    results["safeguards"] = {
        "threshold": config.CONFIDENCE_THRESHOLD,
        "tickets_flagged_for_review": int(flagged.sum()),
        "share_flagged": float(flagged.mean()),
        "low_category_confidence_count": int(low_conf.sum()),
        "category_accuracy_when_confident": float(cat_ok[~low_conf].mean()) if (~low_conf).any() else None,
        "category_accuracy_when_low_confidence": float(cat_ok[low_conf].mean()) if low_conf.any() else None,
        "category_errors_total": int((~cat_ok).sum()),
        "category_errors_caught_by_low_confidence_flag": int(((~cat_ok) & low_conf).sum()),
    }
    cat_conf_arr = np.array([r["category_confidence"] for r in triaged])
    results["threshold_table_info_only"] = [
        {"threshold": t, "share_below_threshold": float((cat_conf_arr < t).mean()),
         "accuracy_at_or_above": float(cat_ok[cat_conf_arr >= t].mean()) if (cat_conf_arr >= t).any() else None}
        for t in (0.3, 0.4, 0.45, 0.5, 0.6)]
    high_true = (evald["urgency"] == "High").to_numpy()
    high_pred = np.array([r["urgency"] == "High" for r in triaged])
    results["safeguards"]["high_urgency_true"] = int(high_true.sum())
    results["safeguards"]["high_urgency_found"] = int((high_true & high_pred).sum())

    # ---- E. Error analysis -------------------------------------------------
    err_rows = []
    for r, (_, row) in zip(triaged, evald.iterrows()):
        if r["category"] != row["category"] or r["urgency"] != row["urgency"]:
            err_rows.append({"ticket_id": row["ticket_id"], "text": row["text"],
                             "true_category": row["category"], "pred_category": r["category"],
                             "category_confidence": r["category_confidence"],
                             "true_urgency": row["urgency"], "pred_urgency": r["urgency"],
                             "urgency_confidence": r["urgency_confidence"],
                             "flagged_for_review": r["needs_human_review"]})
    errors = pd.DataFrame(err_rows)
    results["n_tickets_with_any_error"] = len(errors)
    ml_cat = main["category"]["TF-IDF + Logistic Regression"]
    labels = ml_cat["labels"]
    pair_counts = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j and ml_cat["confusion_matrix"][i][j] > 0:
                pair_counts.append((ml_cat["confusion_matrix"][i][j], a, b))
    results["top_category_confusions"] = [{"count": c, "true": a, "predicted": b} for c, a, b in sorted(pair_counts, reverse=True)[:5]]

    preds_for_json = results.pop("_preds")
    if write_files:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
        errors.to_csv(config.RESULTS_DIR / "error_analysis.csv", index=False)
        for task in tasks:
            cm = pd.DataFrame(main[task]["TF-IDF + Logistic Regression"]["confusion_matrix"],
                              index=[f"true: {l}" for l in tasks[task]], columns=[f"pred: {l}" for l in tasks[task]])
            cm.to_csv(config.RESULTS_DIR / f"confusion_matrix_{task}.csv")
        (config.RESULTS_DIR / "evaluation_report.md").write_text(build_report(results, errors))
    results["_preds"] = preds_for_json
    return results


# --------------------------------------------------------------------------- report text
def build_report(res: dict, errors: pd.DataFrame) -> str:
    out = ["# Evaluation Report - Project 1: Intelligent Support Triage", "",
           "_Generated automatically by `python -m triage.evaluate`. Every number below was computed by running the code._", "",
           f"- Random seed: {res['seed']}",
           f"- Training tickets (simulated, template-generated): {res['n_train']}",
           f"- Evaluation tickets (hand-written, never used for training): {res['n_eval']}",
           f"- Exact-text overlap between training and evaluation sets: {res['leakage_overlap_count']} (must be 0)", "",
           "## 1. Class balance", ""]
    for task in ("category", "urgency"):
        labels = config.CATEGORIES if task == "category" else config.URGENCY_LEVELS
        rows = [[l, res["train_distribution"][task].get(l, 0), res["eval_distribution"][task].get(l, 0)] for l in labels]
        out += [f"**{task.title()}**", "", _md_table(["Label", "Train", "Evaluation"], rows), ""]

    out += ["## 2. Main comparison on the hand-written evaluation set", "",
            f"All systems are scored on the same {res['n_eval']} tickets. Macro averages treat every class equally, "
            "which is why they are the headline metric for these imbalanced classes.", ""]
    for task in ("category", "urgency"):
        rows = []
        for name, m in res["main_eval"][task].items():
            rows.append([name, _fmt(m["accuracy"]), _fmt(m["macro_precision"]), _fmt(m["macro_recall"]), _fmt(m["macro_f1"]), _fmt(m["weighted_f1"])])
        out += [f"### {task.title()} prediction", "", _md_table(["System", "Accuracy", "Macro precision", "Macro recall", "Macro F1", "Weighted F1"], rows), ""]
        ml = res["main_eval"][task]["TF-IDF + Logistic Regression"]
        rows = [[l, _fmt(v["precision"]), _fmt(v["recall"]), _fmt(v["f1"]), v["support"]] for l, v in ml["per_class"].items()]
        out += ["Per-class results for TF-IDF + Logistic Regression:", "", _md_table(["Class", "Precision", "Recall", "F1", "Support"], rows), ""]
        header = ["True \\ Predicted"] + [l[:14] for l in ml["labels"]]
        rows = [[l] + row for l, row in zip(ml["labels"], ml["confusion_matrix"])]
        out += ["Confusion matrix (rows = true, columns = predicted):", "", _md_table(header, rows), ""]

    dv = res["dev_split"]
    out += ["## 3. Development check (held-out templates from the simulated data)", "",
            f"Train: {dv['n_train']} tickets, test: {dv['n_test']} tickets from {dv['templates_held_out']} templates the model never saw. "
            "Because there are only about 10 templates per category, holding out 25% of them removes a lot of vocabulary, so this is a HARSH and noisy stress test of "
            "generalisation to unseen phrasing. It is not an upper bound and not a measure of real-world accuracy. It is reported so that the weaker side of the evidence is visible too.", ""]
    for task in ("category", "urgency"):
        rows = [[n, _fmt(m["accuracy"]), _fmt(m["macro_f1"])] for n, m in dv["metrics"][task].items()]
        out += [f"**{task.title()}**", "", _md_table(["System", "Accuracy", "Macro F1"], rows), ""]

    out += ["## 4. Hyper-parameter sensitivity (5-fold grouped cross-validation, training data only)", "",
            _md_table(["C (regularisation)", "Mean macro F1", "Std"], [[k, _fmt(v["mean_macro_f1"]), _fmt(v["std"])] for k, v in res["c_sensitivity_grouped_cv"].items()]), "",
            f"The project uses C = {config.LOGREG_C}. Folds are grouped by template so the same template never appears in both train and validation.", ""]

    sg = res["safeguards"]
    fmt_opt = lambda v: "n/a" if v is None else _fmt(v)
    out += ["## 5. Safeguard check (human-review flags)", "",
            f"- Confidence threshold: {sg['threshold']} (fixed in advance, not tuned on the evaluation set)",
            f"- Tickets flagged for human review (any reason): {sg['tickets_flagged_for_review']} of {res['n_eval']} ({sg['share_flagged']:.0%})",
            f"- Tickets with low CATEGORY confidence: {sg['low_category_confidence_count']}",
            f"- Category accuracy when confident: {fmt_opt(sg['category_accuracy_when_confident'])}; when low confidence: {fmt_opt(sg['category_accuracy_when_low_confidence'])}",
            f"- Category mistakes caught by the low-confidence flag: {sg['category_errors_caught_by_low_confidence_flag']} of {sg['category_errors_total']}",
            f"- High-urgency tickets found: {sg['high_urgency_found']} of {sg['high_urgency_true']} (all predicted-High tickets are also flagged for human confirmation)", "",
            "Category-confidence threshold table (for information only; the threshold was NOT chosen from this table, "
            "because choosing it on the evaluation set would be data leakage):", "",
            _md_table(["Threshold", "Share of tickets below it", "Category accuracy at or above it"],
                      [[t["threshold"], f"{t['share_below_threshold']:.0%}", fmt_opt(t["accuracy_at_or_above"])] for t in res["threshold_table_info_only"]]), "",
            "## 6. Error analysis", "",
            f"Tickets with at least one wrong prediction (category or urgency): {res['n_tickets_with_any_error']} of {res['n_eval']}. Full list: `error_analysis.csv`.", "",
            "Most frequent category confusions (ML model):", "",
            _md_table(["Count", "True category", "Predicted category"], [[c["count"], c["true"], c["predicted"]] for c in res["top_category_confusions"]]), ""]
    if len(errors):
        out += ["First 15 errors:", "", _md_table(["ID", "Text", "True cat.", "Pred. cat.", "True urg.", "Pred. urg.", "Flagged"],
                [[r.ticket_id, r.text[:80] + ("..." if len(r.text) > 80 else ""), r.true_category, r.pred_category, r.true_urgency, r.pred_urgency, r.flagged_for_review]
                 for r in errors.head(15).itertuples()]), ""]
    out += ["## 7. Reading these results honestly", "",
            f"- The evaluation set has only {res['n_eval']} tickets, so one ticket changes accuracy by about {100 / res['n_eval']:.1f} percentage points. Differences of a few points are not meaningful.",
            "- Training data is simulated from templates and the evaluation data was written by one person, so neither represents real customers.",
            "- These results do NOT show the system is ready for production. See the README for limitations.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    res = run_evaluation()
    ml_c, ml_u = res["main_eval"]["category"], res["main_eval"]["urgency"]
    for name in ml_c:
        print(f"{name:32s} category acc={ml_c[name]['accuracy']:.3f} macroF1={ml_c[name]['macro_f1']:.3f} | "
              f"urgency acc={ml_u[name]['accuracy']:.3f} macroF1={ml_u[name]['macro_f1']:.3f}")
    print(f"\nFull report: {config.RESULTS_DIR / 'evaluation_report.md'}")
