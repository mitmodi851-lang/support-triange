import pandas as pd
import pytest

from triage import config
from triage.pipeline import TriageSystem


@pytest.fixture(scope="session")
def train_df():
    return pd.read_csv(config.TRAIN_PATH)


@pytest.fixture(scope="session")
def eval_df():
    return pd.read_csv(config.EVAL_PATH)


@pytest.fixture(scope="session")
def system(train_df):
    return TriageSystem().fit(train_df)
