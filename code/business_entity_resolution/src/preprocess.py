import re
import anyascii
from config import HONORIFICS, LEGAL_SUFFIXES, STOP_WORDS

def clean_text(s: str) -> str:
    """Normalize text: convert any unicode script to ASCII, lowercase, strip domains and punctuation."""
    if not s:
        return ""
    # Transliterate any unicode script (Devanagari, Telugu, Tamil, French accents) to ASCII
    s = anyascii.anyascii(s).lower()
    # Strip domain extensions (e.g. .com, .org, .net, .in, .fr, .co.in)
    s = re.sub(r"\.(com|org|net|in|co\.in|fr|io|biz|info)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", " ", s)
    # Normalize whitespace
    return re.sub(r"\s+", " ", s).strip()

def get_tokens(s: str) -> list[str]:
    """Return list of cleaned tokens."""
    return clean_text(s).split()

def get_core_name(tokens: list[str]) -> list[str]:
    """Strip honorifics, legal suffixes, and stop words from name tokens."""
    core = [
        t for t in tokens
        if t not in HONORIFICS and t not in LEGAL_SUFFIXES and t not in STOP_WORDS and len(t) > 1
    ]
    return core if core else [t for t in tokens if len(t) > 1]

def normalize_number(num_str: str) -> str:
    """Normalize street/building numbers by stripping leading zeros (e.g. '01600' -> '1600')."""
    m = re.match(r"^0*(\d+[a-z]?)$", num_str)
    return m.group(1) if m else num_str

def extract_addr_numbers(tokens: list[str]) -> list[str]:
    """Extract and normalize numerical address tokens (street numbers, unit numbers, pin codes)."""
    nums = []
    for t in tokens:
        if re.search(r"\d", t):
            norm = normalize_number(t)
            if len(norm) <= 8:  # filter out phone numbers
                nums.append(norm)
    return nums
