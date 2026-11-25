from triage import config
from triage.data_generation import generate_training_tickets
from triage.evaluate import check_no_leakage


def test_files_have_required_columns(train_df, eval_df):
    for df in (train_df, eval_df):
        assert {"text", "category", "urgency"} <= set(df.columns)
        assert df["text"].notna().all() and (df["text"].str.strip() != "").all()


def test_labels_are_valid(train_df, eval_df):
    for df in (train_df, eval_df):
        assert set(df["category"]) <= set(config.CATEGORIES)
        assert set(df["urgency"]) <= set(config.URGENCY_LEVELS)
        assert set(df["category"]) == set(config.CATEGORIES)   # every class is present


def test_no_train_eval_leakage(train_df, eval_df):
    assert check_no_leakage(train_df, eval_df) == []


def test_generation_is_reproducible(train_df):
    again = generate_training_tickets()
    assert again["text"].tolist() == train_df["text"].tolist()
