
import os
import re
import pdfplumber


# ── Constants ─────────────────────────────────────────────────────────────────
MIN_TEXT_LENGTH = 50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_merged_tokens(text: str, max_word_len: int = 25) -> bool:
    return any(len(w) > max_word_len for w in text.split())


def _rebuild_from_words(page, space_gap_ratio: float = 0.3) -> str:
   
    words = page.extract_words(
        x_tolerance=1.5,
        y_tolerance=3,
        keep_blank_chars=False,
        use_text_flow=False,
    )
    if not words:
        return ""

    lines: dict = {}
    for w in words:
        key = round(w["top"], 1)
        lines.setdefault(key, []).append(w)

    out_lines = []
    for top in sorted(lines.keys()):
        line_words = sorted(lines[top], key=lambda w: w["x0"])
        parts, prev_x1 = [], None
        for w in line_words:
            if prev_x1 is not None:
                gap = w["x0"] - prev_x1
                avg_char_w = (w["x1"] - w["x0"]) / max(len(w["text"]), 1)
                if gap > avg_char_w * space_gap_ratio:
                    parts.append(" ")
            parts.append(w["text"])
            prev_x1 = w["x1"]
        out_lines.append("".join(parts))

    return "\n".join(out_lines)


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> tuple[str | None, str | None]:
    
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(
                    x_tolerance=1.5,
                    y_tolerance=3,
                    keep_blank_chars=False,
                )
                if not page_text:
                    continue
                if _has_merged_tokens(page_text):
                    page_text = _rebuild_from_words(page)
                text += page_text + "\n"
    except Exception as exc:
        return None, str(exc)

    stripped = text.strip()
    if len(stripped) < MIN_TEXT_LENGTH:
        return None, f"Extracted text too short ({len(stripped)} chars)"
    return stripped, None


def extract_text_from_bytes(file_bytes: bytes) -> tuple[str | None, str | None]:
   
    import io
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(
                    x_tolerance=1.5,
                    y_tolerance=3,
                    keep_blank_chars=False,
                )
                if not page_text:
                    continue
                if _has_merged_tokens(page_text):
                    page_text = _rebuild_from_words(page)
                text += page_text + "\n"
    except Exception as exc:
        return None, str(exc)

    stripped = text.strip()
    if len(stripped) < MIN_TEXT_LENGTH:
        return None, f"Extracted text too short ({len(stripped)} chars)"
    return stripped, None


def extract_resume_batch(pdf_files: list) -> list[dict]:
    
    results = []
    for idx, (filename, file_bytes) in enumerate(pdf_files):
        text, err = extract_text_from_bytes(file_bytes)
        results.append({
            "resume_id": idx,
            "filename": filename,
            "raw_text": text or "",
            "error": err,
        })
    return results
