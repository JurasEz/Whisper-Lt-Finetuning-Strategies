import re
import string
from typing import Iterable

EXTRA_PUNCT = "„“”–—…“”‘’«»"
PUNCT_TABLE = str.maketrans({ch: " " for ch in (string.punctuation + EXTRA_PUNCT)})
BRACKET_PATTERNS = [r"\[[^\]]*\]", r"\([^\)]*\)", r"\{[^\}]*\}", r"<[^>]*>"]
LT_STOPWORDS = {
    "ir", "bei", "kad", "tai", "yra", "buvo", "bus", "su", "iš", "į", "už", "po", "per",
    "ant", "dėl", "ar", "o", "bet", "jei", "nes", "kaip", "kur", "kas", "kiek", "čia",
    "ten", "mes", "jūs", "jie", "jos", "jis", "ji", "mano", "tavo", "savo", "mūsų",
    "jūsų", "labai", "taip", "ne", "jo", "jos", "jų", "tu", "aš", "mus", "jus", "man",
    "tau", "jam", "jai", "juos", "jas", "šis", "ši", "šie", "šios", "tas", "ta", "tie",
    "tos", "vis", "dar", "tik", "čia", "ten", "nėra", "buvo", "būti", "esu", "esi", "yra",
}


def normalize_text(text: str, noise_labels: Iterable[str] | None = None) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    for pattern in BRACKET_PATTERNS:
        text = re.sub(pattern, " ", text)
    if noise_labels:
        for label in noise_labels:
            text = text.replace(label.lower(), " ")
    text = text.translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str, min_len: int = 4, max_keywords: int = 12) -> list[str]:
    words = []
    seen = set()
    for token in normalize_text(text).split():
        if len(token) < min_len:
            continue
        if token in LT_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        words.append(token)
        if len(words) >= max_keywords:
            break
    return words
