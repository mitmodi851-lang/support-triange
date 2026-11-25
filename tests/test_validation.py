import pytest

from triage.config import MAX_TEXT_CHARS
from triage.validation import InvalidTicketError, clean_ticket_text


@pytest.mark.parametrize("bad", [None, "", "   ", "\n\t", "12345 !!!", 42, ["a"]])
def test_invalid_inputs_raise(bad):
    with pytest.raises(InvalidTicketError):
        clean_ticket_text(bad)


def test_whitespace_is_normalised():
    cleaned, _ = clean_ticket_text("  My   invoice \n is wrong  ")
    assert cleaned == "My invoice is wrong"


def test_short_text_warns():
    _, warnings = clean_ticket_text("help me")
    assert any("Very short" in w for w in warnings)


def test_long_text_is_truncated():
    cleaned, warnings = clean_ticket_text("invoice " * 2000)
    assert len(cleaned) <= MAX_TEXT_CHARS
    assert any("truncated" in w for w in warnings)


def test_non_english_script_warns():
    _, warnings = clean_ticket_text("मेरा पासवर्ड काम नहीं कर रहा है please help")
    assert any("non-English" in w for w in warnings)
