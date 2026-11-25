"""Streamlit demo for Project 1. Run:  streamlit run app.py"""
import json
import pandas as pd
import streamlit as st

from triage import config
from triage.pipeline import load_default_system

st.set_page_config(page_title="Support Triage", page_icon="🎫", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.result-card {border: 1px solid #C9DAD8; border-radius: 10px; padding: 1rem 1.2rem; background: #F7FAFA; color: #14262B;}
.chip {display:inline-block; padding: 2px 12px; border-radius: 999px; font-weight: 600; font-size: 0.9rem; border: 1px solid;}
.chip-High {background:#FDE8E6; color:#8A1C12; border-color:#8A1C12;}
.chip-Medium {background:#FFF3D6; color:#6B4A00; border-color:#6B4A00;}
.chip-Low {background:#E4F3EA; color:#16502B; border-color:#16502B;}
.review {border-left: 4px solid #8A1C12; background:#FDF2F1; padding:.6rem 1rem; border-radius:6px; color:#14262B;}
.ok {border-left: 4px solid #16502B; background:#EEF8F1; padding:.6rem 1rem; border-radius:6px; color:#14262B;}
@media (max-width: 640px) {.block-container {padding: 1rem .6rem;}}
</style>
""", unsafe_allow_html=True)

SAMPLES = {
    "(choose a sample)": "",
    "Double charge": "I was charged twice for my Pro plan today and the money left my account. Please fix this immediately.",
    "Locked out": "I changed my phone number and now I cannot get past the security check. I have a client meeting in 30 minutes.",
    "Late parcel": "Tracking says my parcel is out for delivery since three days ago but nothing has arrived.",
    "App crash": "The app crashes every time I open the reports page on my phone.",
    "Simple question": "Do you offer a discount for students?",
    "Ambiguous": "Charged for an order I cancelled and I never got the item either.",
}


@st.cache_resource(show_spinner="Training the model on the simulated tickets...")
def get_system():
    return load_default_system()


def _use_sample():
    st.session_state["ticket_text"] = SAMPLES[st.session_state["sample_choice"]]


def show_result(r: dict):
    if r["status"] != "ok":
        st.error(f"Cannot triage this input: {r['error']}")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Category", r["category"], f"confidence {r['category_confidence']:.0%}", delta_color="off")
    c2.markdown(f"**Urgency**<br><span class='chip chip-{r['urgency']}'>{r['urgency']}</span><br>"
                f"<small>confidence {r['urgency_confidence']:.0%}</small>", unsafe_allow_html=True)
    c3.metric("Recommended team", r["recommended_team"])
    if r["needs_human_review"]:
        st.markdown("<div class='review'><b>Human review needed.</b> Reasons: "
                    + ", ".join(x.replace("_", " ") for x in r["review_reasons"]) + "</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='ok'><b>High-confidence suggestion.</b> An agent should still confirm it.</div>", unsafe_allow_html=True)
    for w in r["warnings"]:
        st.warning(w)
    with st.expander("Why this category? (explanation and runner-up categories)"):
        st.write("Words that pushed the model towards this category:", ", ".join(r["key_terms"]) or "none found")
        st.dataframe(pd.DataFrame(r["category_top3"], columns=["Category", "Probability"]), hide_index=True)
        st.caption("Probabilities are the model's own estimates, not guarantees of correctness.")
    st.caption(r["disclaimer"])


st.title("Intelligent Support Triage")
st.write("Paste a customer ticket. The tool suggests a category, an urgency level and a team, "
         "and tells you when a person should look at it first.")

tab_single, tab_batch, tab_eval, tab_about = st.tabs(["Triage a ticket", "Batch (CSV)", "Evaluation results", "About and safeguards"])

with tab_single:
    st.selectbox("Try a sample ticket", list(SAMPLES), key="sample_choice", on_change=_use_sample)
    text = st.text_area("Ticket text", key="ticket_text", height=140, placeholder="Describe the customer's problem...")
    if st.button("Triage ticket", type="primary"):
        with st.spinner("Analysing..."):
            show_result(get_system().triage(text))

with tab_batch:
    st.write("Upload a CSV with a **text** column. Invalid rows are reported, not skipped silently.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            out = get_system().triage_dataframe(df)
            st.dataframe(out, hide_index=True)
            st.download_button("Download results (CSV)", out.to_csv(index=False), "triage_results.csv", "text/csv")
        except Exception as err:   # unreadable file, wrong columns, etc.
            st.error(f"Could not process the file: {err}")

with tab_eval:
    path = config.RESULTS_DIR / "metrics.json"
    if not path.exists():
        st.info("No results yet. Run `python -m triage.evaluate` first.")
    else:
        m = json.loads(path.read_text())
        st.write(f"Scored on **{m['n_eval']} hand-written tickets** that were never used for training. "
                 "A set this small cannot prove production readiness.")
        for task in ("category", "urgency"):
            rows = [{"System": n, "Accuracy": round(v["accuracy"], 3), "Macro precision": round(v["macro_precision"], 3),
                     "Macro recall": round(v["macro_recall"], 3), "Macro F1": round(v["macro_f1"], 3)}
                    for n, v in m["main_eval"][task].items()]
            st.subheader(task.title())
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        sg = m["safeguards"]
        st.write(f"Human-review flags: {sg['tickets_flagged_for_review']} of {m['n_eval']} tickets flagged. "
                 f"Category accuracy when confident: {sg['category_accuracy_when_confident']:.1%}, "
                 f"when low confidence: {sg['category_accuracy_when_low_confidence']:.1%}.")
        err = config.RESULTS_DIR / "error_analysis.csv"
        if err.exists():
            with st.expander("Misclassified evaluation tickets"):
                st.dataframe(pd.read_csv(err), hide_index=True)

with tab_about:
    st.markdown("""
**What it is.** A prototype built on simulated data for a learning internship.

**Safeguards**
- Empty, non-text or symbol-only input is rejected with a clear message.
- Low-confidence, very short and every *High urgency* prediction is marked for human review.
- The tool only suggests. It never sends, closes or escalates a ticket by itself.

**Limits.** Trained on 366 template-generated tickets, English only, six categories.
Confidence scores are not probabilities of being correct. See the README for the full list.
""")
