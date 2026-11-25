# Evaluation Report - Project 1: Intelligent Support Triage

_Generated automatically by `python -m triage.evaluate`. Every number below was computed by running the code._

- Random seed: 42
- Training tickets (simulated, template-generated): 366
- Evaluation tickets (hand-written, never used for training): 66
- Exact-text overlap between training and evaluation sets: 0 (must be 0)

## 1. Class balance

**Category**

| Label | Train | Evaluation |
|---|---|---|
| Billing & Payments | 60 | 11 |
| Technical Issue | 60 | 11 |
| Account Access | 60 | 11 |
| Shipping & Delivery | 60 | 11 |
| Refund & Cancellation | 60 | 12 |
| General Inquiry | 66 | 10 |

**Urgency**

| Label | Train | Evaluation |
|---|---|---|
| Low | 150 | 26 |
| Medium | 132 | 27 |
| High | 84 | 13 |

## 2. Main comparison on the hand-written evaluation set

All systems are scored on the same 66 tickets. Macro averages treat every class equally, which is why they are the headline metric for these imbalanced classes.

### Category prediction

| System | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Majority class | 0.152 | 0.025 | 0.167 | 0.044 | 0.040 |
| Keyword rules (baseline) | 0.758 | 0.832 | 0.764 | 0.749 | 0.750 |
| TF-IDF + Logistic Regression | 0.758 | 0.762 | 0.764 | 0.753 | 0.750 |

Per-class results for TF-IDF + Logistic Regression:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Billing & Payments | 0.727 | 0.727 | 0.727 | 11 |
| Technical Issue | 0.889 | 0.727 | 0.800 | 11 |
| Account Access | 0.786 | 1.000 | 0.880 | 11 |
| Shipping & Delivery | 0.727 | 0.727 | 0.727 | 11 |
| Refund & Cancellation | 0.750 | 0.500 | 0.600 | 12 |
| General Inquiry | 0.692 | 0.900 | 0.783 | 10 |

Confusion matrix (rows = true, columns = predicted):

| True \ Predicted | Billing & Paym | Technical Issu | Account Access | Shipping & Del | Refund & Cance | General Inquir |
|---|---|---|---|---|---|---|
| Billing & Payments | 8 | 0 | 2 | 0 | 1 | 0 |
| Technical Issue | 0 | 8 | 1 | 0 | 0 | 2 |
| Account Access | 0 | 0 | 11 | 0 | 0 | 0 |
| Shipping & Delivery | 1 | 0 | 0 | 8 | 1 | 1 |
| Refund & Cancellation | 2 | 0 | 0 | 3 | 6 | 1 |
| General Inquiry | 0 | 1 | 0 | 0 | 0 | 9 |

### Urgency prediction

| System | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| Majority class | 0.394 | 0.131 | 0.333 | 0.188 | 0.223 |
| Keyword rules (baseline) | 0.576 | 0.667 | 0.587 | 0.570 | 0.535 |
| TF-IDF + Logistic Regression | 0.712 | 0.714 | 0.695 | 0.701 | 0.713 |

Per-class results for TF-IDF + Logistic Regression:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.818 | 0.692 | 0.750 | 26 |
| Medium | 0.656 | 0.778 | 0.712 | 27 |
| High | 0.667 | 0.615 | 0.640 | 13 |

Confusion matrix (rows = true, columns = predicted):

| True \ Predicted | Low | Medium | High |
|---|---|---|---|
| Low | 18 | 7 | 1 |
| Medium | 3 | 21 | 3 |
| High | 1 | 4 | 8 |

## 3. Development check (held-out templates from the simulated data)

Train: 270 tickets, test: 96 tickets from 16 templates the model never saw. Because there are only about 10 templates per category, holding out 25% of them removes a lot of vocabulary, so this is a HARSH and noisy stress test of generalisation to unseen phrasing. It is not an upper bound and not a measure of real-world accuracy. It is reported so that the weaker side of the evidence is visible too.

**Category**

| System | Accuracy | Macro F1 |
|---|---|---|
| Keyword rules (baseline) | 0.688 | 0.639 |
| TF-IDF + Logistic Regression | 0.344 | 0.311 |

**Urgency**

| System | Accuracy | Macro F1 |
|---|---|---|
| Keyword rules (baseline) | 0.812 | 0.772 |
| TF-IDF + Logistic Regression | 0.562 | 0.558 |

## 4. Hyper-parameter sensitivity (5-fold grouped cross-validation, training data only)

| C (regularisation) | Mean macro F1 | Std |
|---|---|---|
| 1.0 | 0.450 | 0.149 |
| 10.0 | 0.462 | 0.154 |
| 100.0 | 0.459 | 0.149 |

The project uses C = 10.0. Folds are grouped by template so the same template never appears in both train and validation.

## 5. Safeguard check (human-review flags)

- Confidence threshold: 0.45 (fixed in advance, not tuned on the evaluation set)
- Tickets flagged for human review (any reason): 46 of 66 (70%)
- Tickets with low CATEGORY confidence: 38
- Category accuracy when confident: 0.893; when low confidence: 0.658
- Category mistakes caught by the low-confidence flag: 13 of 16
- High-urgency tickets found: 8 of 13 (all predicted-High tickets are also flagged for human confirmation)

Category-confidence threshold table (for information only; the threshold was NOT chosen from this table, because choosing it on the evaluation set would be data leakage):

| Threshold | Share of tickets below it | Category accuracy at or above it |
|---|---|---|
| 0.3 | 18% | 0.833 |
| 0.4 | 39% | 0.925 |
| 0.45 | 58% | 0.893 |
| 0.5 | 65% | 0.913 |
| 0.6 | 88% | 1.000 |

## 6. Error analysis

Tickets with at least one wrong prediction (category or urgency): 30 of 66. Full list: `error_analysis.csv`.

Most frequent category confusions (ML model):

| Count | True category | Predicted category |
|---|---|---|
| 3 | Refund & Cancellation | Shipping & Delivery |
| 2 | Technical Issue | General Inquiry |
| 2 | Refund & Cancellation | Billing & Payments |
| 2 | Billing & Payments | Account Access |
| 1 | Technical Issue | Account Access |

First 15 errors:

| ID | Text | True cat. | Pred. cat. | True urg. | Pred. urg. | Flagged |
|---|---|---|---|---|---|---|
| E002 | Need the GST invoice for our October purchase with our company registration numb... | Billing & Payments | Billing & Payments | Low | Medium | True |
| E004 | My card keeps getting declined on your payment page. I tried three different car... | Billing & Payments | Billing & Payments | Medium | High | True |
| E005 | Can I pay my annual fee in two instalments? | Billing & Payments | Account Access | Low | Medium | True |
| E009 | Our finance team wants to switch to invoicing instead of card payment from next ... | Billing & Payments | Refund & Cancellation | Low | Low | False |
| E012 | Is there a keyboard shortcut to switch between projects? | Technical Issue | General Inquiry | Low | Low | True |
| E014 | The website looks broken on Safari, buttons overlap each other. | Technical Issue | General Inquiry | Low | Medium | True |
| E015 | I keep getting error code 503 on every page. Nothing loads for my entire office. | Technical Issue | Technical Issue | High | Medium | True |
| E016 | The charts on the analytics page display wrong numbers compared to the exported ... | Technical Issue | Technical Issue | Medium | Low | False |
| E017 | Can you add an option to change the font size in the editor? | Technical Issue | Account Access | Low | Low | True |
| E021 | I cannot log in. It says my account is disabled. | Account Access | Account Access | Medium | High | True |
| E023 | I got an email about a login from a device I do not recognise. Please lock my ac... | Account Access | Account Access | High | Medium | False |
| E026 | My OTP never arrives on my mobile number so I am unable to sign in. | Account Access | Account Access | Medium | High | True |
| E027 | Our new employee needs an account with admin rights, can you set it up sometime ... | Account Access | Account Access | Low | Medium | True |
| E032 | Can you tell me when my parcel will arrive? I ordered last Thursday. | Shipping & Delivery | Shipping & Delivery | Low | Medium | True |
| E034 | The box was open and one of the two items is missing. | Shipping & Delivery | Refund & Cancellation | Medium | Medium | True |

## 7. Reading these results honestly

- The evaluation set has only 66 tickets, so one ticket changes accuracy by about 1.5 percentage points. Differences of a few points are not meaningful.
- Training data is simulated from templates and the evaluation data was written by one person, so neither represents real customers.
- These results do NOT show the system is ready for production. See the README for limitations.
