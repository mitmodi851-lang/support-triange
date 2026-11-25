# Project 1 - Requirements and Design

## Problem framing
- **Stakeholder:** support team lead (primary); support agents (users).
- **Inputs:** free-text ticket (English). **Outputs:** category, urgency, team, confidence, human-review flag with reasons.
- **Scope in:** 6 categories, 3 urgency levels, single and batch triage, evaluation, safeguards.
- **Scope out:** real customer data, replying to customers, ticket-system integration, multilingual support, LLM calls.
- **Constraints:** simulated data, runs offline, beginner-maintainable, reproducible.

## Success measures (defined before evaluation)
| Measure | Target for this prototype |
|---|---|
| Beats majority-class baseline on both tasks | Yes |
| Compared with a keyword baseline on identical data | Yes, reported honestly even if no gain |
| Macro-F1 reported (class imbalance) | Yes |
| Invalid input never crashes | Yes (tested) |
| No train/eval leakage | 0 overlapping tickets (tested) |
| Every claim backed by a generated result | Yes |
No numeric accuracy target was set because the data is simulated and small; the goal is a defensible comparison, not a high score.

## Requirements and where they are met
| # | Requirement | Where |
|---|---|---|
| R1 | Accept ticket text | `triage/pipeline.py`, `app.py` |
| R2 | Classify category | `triage/model.py` |
| R3 | Estimate urgency | `triage/model.py` |
| R4 | Recommend team | `triage/config.py` (`CATEGORY_TO_TEAM`) |
| R5 | Simulated evaluation dataset | `data/eval_tickets.csv`, `data/train_tickets.csv` |
| R6 | Baseline + AI approach, same data | `triage/baseline.py`, `triage/evaluate.py` |
| R7 | Accuracy, precision, recall, F1 | `evaluation/results/evaluation_report.md` |
| R8 | Error analysis | `error_analysis.csv`, README section 10 |
| R9 | Empty/invalid/uncertain input | `triage/validation.py`, review flags, tests |
| R10 | Human review / responsible use | `pipeline.py` flags, README section 13 |
| R11 | Limitations, future work | README sections 12 and 14 |

## Decision log
| Decision | Why | Trade-off |
|---|---|---|
| TF-IDF + Logistic Regression | Fast, explainable, works with tiny data, no GPU/API | Weaker than transformers on meaning |
| Two separate classifiers | Category and urgency are different questions | Two models to maintain |
| Training on simulated templates, evaluating on hand-written tickets | Avoids testing on near-copies of training data | Hand-written set is small and single-annotator |
| Grouped split by `template_id` | Variants of one template are near-duplicates; random splits would leak | Fewer templates per fold |
| Threshold 0.45 fixed in advance | Tuning on the evaluation set would leak | Flags 70% of tickets |
| Force-review predicted High | Escalations are costly to get wrong | Adds review workload |
| No saved model file | Training takes ~1s; avoids stale model files | Re-trains on each app start |
| Macro averages as headline | Urgency classes are imbalanced | Less intuitive than accuracy |
