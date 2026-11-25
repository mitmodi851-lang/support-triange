"""Input checking. Bad input must produce a clear message, never a crash."""
import re

from triage.config import MAX_TEXT_CHARS, MIN_WORDS


class InvalidTicketError(ValueError):
    """Raised when a ticket cannot be processed at all."""


def clean_ticket_text(text):
    """Validate and tidy a ticket. Returns (cleaned_text, warnings).

    Raises InvalidTicketError for: None, non-text, empty/blank, or text
    with no letters at all (for example only symbols or digits).
    """
    if text is None:
        raise InvalidTicketError("Ticket text is missing (None).")
    if not isinstance(text, str):
        raise InvalidTicketError(f"Ticket text must be a string, got {type(text).__name__}.")
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise InvalidTicketError("Ticket text is empty. Please describe the problem.")
    if not re.search(r"[A-Za-z\u0900-\u097F\u0A80-\u0AFF]", cleaned):
        raise InvalidTicketError("Ticket text has no readable words (only numbers or symbols).")

    warnings = []
    if len(cleaned) > MAX_TEXT_CHARS:
        cleaned = cleaned[:MAX_TEXT_CHARS]
        warnings.append(f"Text was longer than {MAX_TEXT_CHARS} characters and was truncated.")
    if len(cleaned.split()) < MIN_WORDS:
        warnings.append(f"Very short ticket (fewer than {MIN_WORDS} words): prediction is unreliable.")
    if re.search(r"[\u0900-\u097F\u0A80-\u0AFF]", cleaned):
        warnings.append("Contains non-English script; the model was trained on English text only.")
    return cleaned, warnings
