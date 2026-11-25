from __future__ import annotations

import re

FOREIGN_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff]")
LATIN_EXTENDED = re.compile(r"[\u0100-\u024f]")
OPTION_KEYS = ("option_a", "option_b", "option_c", "option_d")

FLAG_FOREIGN_QUESTION = "ocr_foreign_script_in_question"
FLAG_FOREIGN_OPTION = "ocr_foreign_script_in_option"
FLAG_GARBLED_QUESTION = "ocr_garbled_question"
FLAG_GARBLED_OPTION = "ocr_garbled_option"
FLAG_DUPLICATE_OPTIONS = "duplicate_options"

OCR_FLAGS = (FLAG_FOREIGN_QUESTION, FLAG_FOREIGN_OPTION, FLAG_GARBLED_QUESTION, FLAG_GARBLED_OPTION)


def _options(row: dict) -> list[str]:
    return [str(row.get(k) or "").strip() for k in OPTION_KEYS]


def text_quality_flags(row: dict) -> list[str]:
    question = str(row.get("question_text") or "")
    options = _options(row)
    flags: list[str] = []
    if FOREIGN_SCRIPT.search(question):
        flags.append(FLAG_FOREIGN_QUESTION)
    if any(FOREIGN_SCRIPT.search(o) for o in options):
        flags.append(FLAG_FOREIGN_OPTION)
    if LATIN_EXTENDED.search(question):
        flags.append(FLAG_GARBLED_QUESTION)
    if any(LATIN_EXTENDED.search(o) for o in options):
        flags.append(FLAG_GARBLED_OPTION)
    spaced = [re.sub(r"\s+", " ", o) for o in options]
    if all(spaced) and len(set(spaced)) < 4:
        flags.append(FLAG_DUPLICATE_OPTIONS)
    return flags


def is_text_corrupt(row: dict) -> bool:
    return bool(text_quality_flags(row))


def offending_snippet(row: dict, width: int = 40) -> str:
    fields = [("question_text", str(row.get("question_text") or ""))] + [(k, str(row.get(k) or "")) for k in OPTION_KEYS]
    for name, text in fields:
        m = FOREIGN_SCRIPT.search(text) or LATIN_EXTENDED.search(text)
        if m:
            lo = max(0, m.start() - width // 2)
            return f"{name}: {text[lo:m.end() + width // 2]}".replace("\n", " ")
    spaced = [re.sub(r"\s+", " ", o) for o in _options(row)]
    if all(spaced) and len(set(spaced)) < 4:
        return "options: " + " | ".join(spaced)[:80]
    return ""
