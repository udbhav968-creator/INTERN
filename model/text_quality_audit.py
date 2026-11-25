from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .sample_paper_ingestion import load_accepted_sample_rows
from .text_quality_gate import FOREIGN_SCRIPT, offending_snippet, text_quality_flags

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
SAMPLE_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_samples_001_to_101_verified.json"
FORMS_PATH = MODEL_DIR / "assessment_forms.json"
REPORT_PATH = MODEL_DIR / "text_quality_report.json"
REVIEW_CSV = PROJECT_ROOT / "data" / "review" / "ocr_corruption_review.csv"

SUSPECT_PAPER_SHARE = 0.25
PI_HINT = "the character 元 is a common OCR misread of the Greek letter π; confirm against the source paper before editing"


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _form_membership() -> dict[str, int]:
    if not FORMS_PATH.exists():
        return {}
    forms = json.loads(FORMS_PATH.read_text(encoding="utf-8")).get("forms", [])
    return {qid: f["form_index"] for f in forms for qid in f["question_ids"]}


def _scan(rows: list[dict]) -> list[tuple[dict, list[str]]]:
    return [(r, f) for r in rows if (f := text_quality_flags(r))]


def _hint(row: dict) -> str:
    text = " ".join(str(row.get(k) or "") for k in ("question_text", "option_a", "option_b", "option_c", "option_d"))
    found = set(FOREIGN_SCRIPT.findall(text))
    return PI_HINT if found == {"元"} and row.get("subject") == "Physics" else ""


def run(save: bool = True) -> dict:
    pyq = _load(PYQ_PATH)
    raw = _load(SAMPLE_PATH)
    accepted, _ = load_accepted_sample_rows()
    forms = _form_membership()

    pyq_hits = _scan(pyq)
    raw_hits = _scan(raw)
    accepted_hits = _scan(accepted)
    form_hits = sorted(qid for qid in forms if qid in {r["question_id"] for r, _ in pyq_hits + accepted_hits})

    per_paper = defaultdict(lambda: [0, 0])
    for r in raw:
        per_paper[r.get("paper_source", "")][0] += 1
    for r, _ in raw_hits:
        per_paper[r.get("paper_source", "")][1] += 1
    suspect = sorted(
        (
            {"paper": p, "rows": n, "flagged": k, "share": round(k / n, 4)}
            for p, (n, k) in per_paper.items()
            if n and k / n >= SUSPECT_PAPER_SHARE
        ),
        key=lambda d: -d["share"],
    )

    review = []
    for bank, hits in (("pyq", pyq_hits), ("sample_accepted", accepted_hits), ("sample_raw", raw_hits)):
        for r, flags in hits:
            review.append(
                {
                    "bank": bank,
                    "question_id": r["question_id"],
                    "subject": r.get("subject"),
                    "chapter": r.get("chapter"),
                    "paper_source": r.get("paper_source"),
                    "flags": ";".join(flags),
                    "snippet": offending_snippet(r),
                    "in_assessment_form": forms.get(r["question_id"], ""),
                    "ocr_hint": _hint(r),
                    "action": "content team: fix text from source paper or quarantine",
                }
            )

    def by_flag(hits):
        return dict(Counter(f for _, flags in hits for f in flags))

    report = {
        "rule": "foreign-script characters, Latin-Extended garble, and exact duplicate options (model/text_quality_gate.py)",
        "pyq": {"rows": len(pyq), "flagged": len(pyq_hits), "by_flag": by_flag(pyq_hits), "flagged_ids": sorted(r["question_id"] for r, _ in pyq_hits)},
        "sample_raw": {"rows": len(raw), "flagged": len(raw_hits), "by_flag": by_flag(raw_hits)},
        "sample_accepted": {"rows": len(accepted), "flagged": len(accepted_hits), "by_flag": by_flag(accepted_hits)},
        "assessment_forms_with_flagged_questions": form_hits,
        "suspect_papers": suspect,
        "invariants": {
            "accepted_set_has_no_flagged_rows": len(accepted_hits) == 0,
            "assessment_forms_have_no_flagged_questions": len(form_hits) == 0,
        },
        "pyq_policy": "flagged PYQs are excluded from ML-built forms and mocks; the PYQ file itself is not edited (content team decision)",
    }
    report["passed"] = all(report["invariants"].values())
    if save:
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
        fields = ["bank", "question_id", "subject", "chapter", "paper_source", "flags", "snippet", "in_assessment_form", "ocr_hint", "action"]
        with REVIEW_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(review)
    return report


if __name__ == "__main__":
    rep = run()
    print(json.dumps({k: rep[k] for k in ("pyq", "sample_raw", "sample_accepted", "assessment_forms_with_flagged_questions", "invariants", "passed")}, indent=2))
    print("suspect papers:", [(s["paper"], s["flagged"], s["rows"]) for s in rep["suspect_papers"][:5]])
