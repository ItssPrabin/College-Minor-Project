
import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data silently on first import
for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(_pkg, quiet=True)

# ── Shared constants ──────────────────────────────────────────────────────────
NOISE_PHRASES = [
    r"curriculum vitae", r"\bresume\b", r"personal name", r"personal information",
    r"references available upon request", r"linkedin", r"github", r"portfolio",
    r"novypro", r"tableau public",
]

SECTION_START_PATTERN = re.compile(
    r"(?i)\b(resume\s+objective|objective|professional\s+summary|summary|profile|"
    r"experience|work\s+experience|employment\s+history|education|skills|"
    r"technical\s+skills|languages)\b"
)

HEADER_WORDS = 60

_stops = set(stopwords.words("english"))
_lemma = WordNetLemmatizer()


# ── Stage 1: shared base cleaner ──────────────────────────────────────────────

def _basic_clean(text: str) -> str:
    cleaned = []
    for ch in str(text):
        if ch in "\n\r\t":
            cleaned.append(ch)
        elif unicodedata.category(ch).startswith("C"):
            continue
        elif ch.isprintable():
            cleaned.append(ch)
    text = "".join(cleaned)
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"(?:[A-Z]\s){3,}[A-Z]", lambda m: m.group(0).replace(" ", ""), text)
    return text


def _remove_contact_and_pii(text: str) -> str:
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"(?<!\w)\(?\+?\d[\d\s().-]{6,}\d\)?(?!\w)", " ", text)
    text = re.sub(
        r"\b\d{1,5}\s+(?:[A-Za-z0-9.\-\' ]{1,8}\s+){0,6}"
        r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|"
        r"Court|Ct|Circle|Cir|Place|Pl|Terrace|Ter|Parkway|Pkwy|Trail|Trl|"
        r"Highway|Hwy|Rodovia)\b[^,.;\n]{0,60}",
        " ", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\b[A-Za-z]+(?:\s+[A-Za-z]+){0,3}\s*,\s*[A-Za-z]{2}\s+\d{5}(?:-\d{4})?\b", " ", text)
    text = re.sub(
        r"(?i)\b(address|country of residence|nationality|date of birth|dob|"
        r"marital status)\s*:?\s*.{0,120}?(?=\b(?:Phone|Email|Address|Country|"
        r"Nationality|Objective|Summary|Experience|Education|Skills)\b|[.\n]|$)",
        " ", text,
    )
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(
        r"\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|uk|in|biz|info)(?:/[^\s]*)?",
        " ", text, flags=re.IGNORECASE,
    )
    return text


def _remove_boilerplate(text: str) -> str:
    section_match = SECTION_START_PATTERN.search(text[:1200])
    if section_match and section_match.start() > 0:
        text = text[section_match.start():]
    for phrase in NOISE_PHRASES:
        text = re.sub(phrase, " ", text, flags=re.IGNORECASE)
    orphan = re.compile(
        r"(?i)\b(phone(?:\s*number)?|mobile|cell|tel(?:ephone)?|e[-\s]?mail)"
        r"\s*[:\-]?\s*(?=[,()\[\]\s]|$)"
    )
    return orphan.sub(" ", text)


def _split_header_body(text: str, header_words: int = HEADER_WORDS) -> tuple[str, str]:
    words = text.split()
    return " ".join(words[:header_words]), " ".join(words[header_words:])


def _remove_names_places_spacy(text: str, nlp) -> str:
    doc = nlp(text)
    spans = [(e.start_char, e.end_char) for e in doc.ents if e.label_ in {"PERSON", "GPE", "LOC"}]
    for start, end in sorted(spans, key=lambda s: -s[0]):
        text = text[:start] + " " + text[end:]
    return text


def _final_normalize(text: str) -> str:
    text = re.sub(r"[•▪◦‣➤→✓❖✦|]", " ", text)
    text = re.sub(r"[^\w\s.,!?;:\-]", " ", text)
    text = re.sub(r"[-]+[–]+[—]+[.]+[_]+", " ", text)
    text = re.sub(r"\(\s*\)", " ", text)
    text = re.sub(r"\[\s*\]", " ", text)
    text = re.sub(r"\s*,\s*,+", ",", text)
    text = re.sub(r"(^|\s),", r"\1", text)
    text = re.sub(r"[.,;:]{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_for_sbert(text: str, nlp=None) -> str:
    
    text = _basic_clean(text)
    text = _remove_contact_and_pii(text)
    text = _remove_boilerplate(text)

    if nlp is not None:
        header, body = _split_header_body(text)
        header = _remove_names_places_spacy(header, nlp)
        text = header + " " + body

    text = re.sub(r"[•▪◦‣➤→✓|]", " ", text)
    text = re.sub(r"[^\w\s.,;:\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_for_w2v(text: str) -> list[str]:
    
    text = _basic_clean(text)
    text = _remove_contact_and_pii(text)
    text = _remove_boilerplate(text)
    text = _final_normalize(text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = word_tokenize(text)
    words = [_lemma.lemmatize(w) for w in words if w not in _stops and len(w) > 2]
    return words


def preprocess_jd_for_sbert(text: str) -> str:
    text = "".join(
        ch for ch in str(text)
        if ch in "\n\r\t" or (not unicodedata.category(ch).startswith("C") and ch.isprintable())
    )
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[•▪◦‣➤→✓|]", " ", text)
    text = re.sub(r"[^\w\s.,;:\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_jd_for_w2v(text: str) -> list[str]:
    text = "".join(
        ch for ch in str(text)
        if ch in "\n\r\t" or (not unicodedata.category(ch).startswith("C") and ch.isprintable())
    )
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = word_tokenize(text)
    words = [_lemma.lemmatize(w) for w in words if w not in _stops and len(w) > 2]
    return words
