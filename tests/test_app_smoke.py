"""Starts the Streamlit app headlessly and checks that it renders without errors."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_empty_ticket_shows_error_not_crash():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area[0].set_value("   ").run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.error) >= 1


def test_valid_ticket_shows_result():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area[0].set_value("I was charged twice for my Pro plan today, please fix this immediately.").run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.error) == 0
    assert len(at.metric) >= 2   # category and team cards are shown
