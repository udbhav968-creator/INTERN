from __future__ import annotations

import math

TRUE_TOKENS = frozenset({"true", "1", "yes", "correct"})
FALSE_TOKENS = frozenset({"false", "0", "no", "incorrect"})


def parse_correct(value) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in TRUE_TOKENS:
            return True
        if token in FALSE_TOKENS:
            return False
    return None


def clean_time(value) -> float | None:
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    return t if math.isfinite(t) and t > 0 else None


def clean_responses(responses: list[dict]) -> tuple[list[dict], dict]:
    latest: dict[str, dict] = {}
    without_id = 0
    with_id = 0
    for r in responses or []:
        qid = r.get("question_id") if isinstance(r, dict) else None
        if qid is None:
            without_id += 1
            continue
        with_id += 1
        latest[qid] = r
    cleaned: list[dict] = []
    invalid = 0
    for qid, r in latest.items():
        raw = r.get("is_correct")
        parsed = parse_correct(raw)
        if raw is not None and parsed is None:
            invalid += 1
            continue
        item = dict(r)
        item["is_correct"] = parsed
        item["time_taken_seconds"] = clean_time(r.get("time_taken_seconds"))
        cleaned.append(item)
    return cleaned, {
        "duplicate_responses_dropped": with_id - len(latest),
        "invalid_is_correct_dropped": invalid,
        "responses_without_question_id": without_id,
        "duplicate_policy": "last answer per question wins",
    }
