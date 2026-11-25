# Project 1 - Intelligent Support Triage

An AI-assisted workflow that reads a customer-support ticket and suggests **(1) a category, (2) an urgency level and (3) the team that should handle it**. It also says when a human must look at the ticket first.

> Built on **simulated data only** for the Arrowstack AI Internship. It is a learning prototype, not a production system.

## 1. Problem statement
Support agents lose time reading and sorting tickets by hand, and urgent tickets can wait in the same queue as simple questions. Given only the ticket text, the system must suggest a category, an urgency level and a team, and must **not** pretend to be certain when it is not.

**Primary stakeholder:** a support team lead. **Decision enabled:** which queue a ticket goes to and which tickets get looked at first. **Measurable benefit (hypothesis, not measured here):** less manual sorting time and fewer urgent tickets sitting unseen.

## 2. Objectives
- Classify tickets into 6 categories and 3 urgency levels; map category -> team.
- Compare a **simple baseline** with an **ML model** on the same evaluation data.
- Report accuracy, precision, recall and F1; analyse wrong predictions.
- Handle empty/invalid/uncertain input and keep a human in the loop.

## 3. Features
- Single-ticket triage with explanation (key words, runner-up categories) and confidence.
- Batch triage from a CSV (bad rows are reported, never silently dropped).
- Human-review flags: low confidence, very short text, and every predicted High urgency.
- Reproducible evaluation (`python -m triage.evaluate`) that writes a report, confusion matrices and an error list.
- Streamlit dashboard, 34 automated tests.

## 4. Technology stack
Python 3.11+ (tested on 3.12), pandas, scikit-learn (TF-IDF + Logistic Regression), Streamlit, pytest. No paid API and no internet are needed. An LLM could be added later (see Future improvements); none is used here.

## 5. Architecture
```
Project-1-Support-Triage/
  app.py                    Streamlit dashboard
  triage/
    config.py               labels, team mapping, thresholds, seed
    data_generation.py      builds simulated training tickets (seeded)
    validation.py           input checks (empty, symbols, too long ...)
    baseline.py             keyword-rule baseline
    model.py                TF-IDF + Logistic Regression
    pipeline.py             TriageSystem: validate -> predict -> safeguards
    evaluate.py             baseline vs ML evaluation + report writer
  data/train_tickets.csv    366 simulated tickets (training only)
  data/eval_tickets.csv     66 hand-written tickets (evaluation only)
  evaluation/results/       generated report, metrics.json, errors, confusion matrices
  tests/                    pytest suite
  docs/REQUIREMENTS_AND_DESIGN.md
```
Flow: `text -> validate/clean -> TF-IDF -> two classifiers (category, urgency) -> team lookup -> review flags -> result`.

## 6. Dataset
| File | What | Size |
|---|---|---|
| `data/train_tickets.csv` | Generated from 61 hand-written templates with random greetings/slots. Each row has a `template_id`. | 366 rows |
| `data/eval_tickets.csv` | Written separately by hand in different wording, including 6 deliberately ambiguous tickets. | 66 rows |

Labels follow one rule: **High** = service blocked, money/security at risk or a deadline within about a day; **Medium** = a problem with a workaround or a few days of time; **Low** = information request or minor matter. Labels were assigned by one person (single annotator), so some are debatable.
No real customer data is used. Training and evaluation tickets are checked for exact overlap (must be 0; this is a test).

**Class balance:** categories are roughly equal in training (60-66 each) and evaluation (10-12 each). Urgency is imbalanced: training Low 150 / Medium 132 / High 84, evaluation 26 / 27 / 13. That is why **macro** averages are reported.

## 7. Installation
```bash
# from the repository root (Windows: use  venv\Scripts\activate)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd Project-1-Support-Triage
```

## 8. Run
```bash
python -m triage.data_generation     # optional: regenerate training data (identical every time)
python -m triage.evaluate            # regenerate the evaluation report
streamlit run app.py                 # open the dashboard (http://localhost:8501)
```
The model is trained in about a second when the app starts, so no model file is stored. Same seed + same data = same model.

## 9. Example input and output
Input: `I was charged twice for my Pro plan today and the money left my account. Please fix this immediately.`
```json
{"status": "ok", "category": "Billing & Payments", "category_confidence": 0.91,
 "urgency": "High", "urgency_confidence": 0.955, "recommended_team": "Billing Team",
 "needs_human_review": true, "review_reasons": ["high_urgency_needs_human_confirmation"]}
```
Input: empty text -> `{"status": "invalid_input", "error": "Ticket text is empty. Please describe the problem.", "needs_human_review": true}`

(These outputs were copied from a real run.)

## 10. Evaluation methodology and actual results
- **Main test:** train on the simulated tickets, score on the 66 hand-written tickets. Baselines: *majority class* and *keyword rules*. Same data for every system.
- **Dev check:** hold out whole templates (grouped split) from the simulated data.
- **Leakage controls:** separate files, a test that no ticket text appears in both, and grouped splits/cross-validation.

Results from `evaluation/results/evaluation_report.md` (seed 42):

| Task | System | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|---|
| Category | Majority class | 0.152 | 0.025 | 0.167 | 0.044 |
| Category | Keyword rules | 0.758 | 0.832 | 0.764 | 0.749 |
| Category | TF-IDF + LogReg | 0.758 | 0.762 | 0.764 | 0.753 |
| Urgency | Majority class | 0.394 | 0.131 | 0.333 | 0.188 |
| Urgency | Keyword rules | 0.576 | 0.667 | 0.587 | 0.570 |
| Urgency | TF-IDF + LogReg | 0.712 | 0.714 | 0.695 | 0.701 |

**What this does and does not show**
- Both approaches clearly beat the majority class. For **category**, the ML model only *ties* the keyword baseline - no ML advantage is demonstrated. For **urgency**, ML is better on this evaluation set (+0.13 accuracy), but on the harsher held-out-template check the keyword baseline scored higher (see report, section 3). The evidence on ML vs rules is therefore **mixed**.
- 66 tickets is tiny: one ticket moves accuracy by 1.5 points, so small differences are noise.
- The keyword baseline was written after reading the training templates, which may flatter it.
- **Safeguards:** with the fixed threshold 0.45, 46 of 66 tickets (70%) are flagged for review - too many to save much work. Flags are still informative: category accuracy is 0.893 when confident vs 0.658 when low-confidence, and 13 of the 16 category mistakes were flagged. The threshold was fixed in advance and *not* tuned on the evaluation set.

### Error analysis (from `error_analysis.csv`)
- **16 category errors, 19 urgency errors** (30 tickets have at least one error).
- **Refund & Cancellation is the weakest class** (recall 0.50): refund requests mention money, delivery or cards and get pulled to Billing or Shipping (e.g. "Package says delivered but I want my money back").
- **Short how-to questions** ("Is there a keyboard shortcut...", "Can you add an option...") look like General Inquiry or Account tickets because training templates for them are few.
- **Multi-topic tickets** have one label but two topics, so some "errors" are arguable. One ambiguous ticket was predicted wrongly at 54% confidence and was *not* flagged.
- **Urgency: 5 of 13 High tickets were missed** (e.g. "error code 503 ... entire office", "login from a device I do not recognise"). Only predicted-High tickets are force-flagged, so a missed urgent ticket can pass without review. This is the most important weakness to fix.
- The "key terms" explanation sometimes shows filler words ("for my", "was"), so it is a rough guide, not a real justification.

## 11. Testing
```bash
python -m pytest -q
```
34 tests cover input validation, data/label checks, **no train/eval leakage**, reproducibility, result structure, batch handling, metric calculations, the full evaluation run and a headless smoke test of the app. Last run: **34 passed**.

## 12. Limitations
- Simulated, template-based training data and a single-annotator evaluation set: not representative of real customers.
- English only; no spelling-error, slang or multi-language robustness testing.
- Small evaluation set; no confidence intervals; results can change with different tickets.
- Missed High-urgency tickets (see above); 70% of tickets flagged for review.
- Category -> team is a fixed rule; it ignores team workload, customer tier or ticket history.
- Confidence scores come from Logistic Regression and are not calibrated probabilities.

## 13. Responsible-use considerations
- **Human in the loop:** output is a suggestion; an agent confirms before any action. The tool never replies to or closes a ticket.
- **Privacy:** only simulated text is included. Real tickets contain personal data and need consent, masking and retention rules before use.
- **Fairness:** the model might perform worse for non-native English writers or other languages. Check error rates per customer group before any real use.
- **Automation bias:** a confident-looking label can make agents stop thinking. The UI shows confidence and reasons and avoids "auto-resolved" language.
- **Not evidence of production readiness.**

## 14. Future improvements
1. Collect real, anonymised, consented tickets and have 2+ people label them; measure agreement.
2. Flag tickets when `urgency` could be High (low threshold on P(High)) so urgent tickets are not missed.
3. Calibrate probabilities and choose the threshold on a separate validation set.
4. Try sentence embeddings or a small fine-tuned transformer; compare with this baseline using cross-validation and confidence intervals.
5. Optional LLM (API or local) for hard/ambiguous tickets, always behind the human-review step.
6. Multi-label output for multi-topic tickets; add language detection.
7. Monitoring: track agent corrections and re-train regularly.

See `docs/REQUIREMENTS_AND_DESIGN.md` for requirements, acceptance criteria and the decision log.
