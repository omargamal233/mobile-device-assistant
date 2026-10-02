import re
import unicodedata

SUSPICIOUS_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"ignore previous",
    r"disregard (all |any )?(the )?(previous|prior|above)",
    r"system prompt",
    r"you are now",
    r"act as",
    r"reveal (your|the) (instructions|prompt)",
    r"new instructions",
    r"</?\s*external_data\s*>",   # attempt to break out of the data tags
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]

MAX_INPUT_LENGTH = 2000


class UnsafeInputError(ValueError):
    """Raised when user input looks like a prompt-injection attempt."""


def _normalize(text: str) -> str:
    # Remove zero-width/control characters, fold lookalike Unicode, collapse whitespace
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cf", "Cc") or ch in "\n\t ")
    return re.sub(r"\s+", " ", text).strip()


def check_input(text: str) -> str:
    """Return the cleaned text, or raise UnsafeInputError if it looks suspicious."""
    if len(text) > MAX_INPUT_LENGTH:
        raise UnsafeInputError("Input is too long.")

    cleaned = _normalize(text)
    for pattern in _COMPILED:
        if pattern.search(cleaned):
            raise UnsafeInputError("Input contains suspicious content.")
    return cleaned
