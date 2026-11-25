from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

SHORT_YEAR_QUESTIONS = 170


def exam_year(row: dict) -> int | None:
    m = re.search(r"(20\d{2})", str(row.get("paper_source", "")))
    return int(m.group(1)) if m else None


def pyq_chapter_weightage(pyq: list[dict]) -> tuple[list[dict], dict]:
    per_year = Counter()
    chapter_year = defaultdict(Counter)
    subject = {}
    missing_year = 0
    for r in pyq:
        y = exam_year(r)
        if y is None:
            missing_year += 1
            continue
        per_year[y] += 1
        chapter_year[r["chapter"]][y] += 1
        subject[r["chapter"]] = r.get("subject", "")
    years = sorted(per_year)
    out = []
    for chapter, cnt in chapter_year.items():
        shares = [cnt[y] / per_year[y] for y in years]
        recent = [cnt[y] / per_year[y] for y in years[-3:]]
        mean_share, recent_share = float(np.mean(shares)), float(np.mean(recent))
        out.append(
            {
                "subject": subject[chapter],
                "chapter": chapter,
                "years_present": sum(1 for y in years if cnt[y] > 0),
                "mean_questions_per_paper": round(float(np.mean([cnt[y] for y in years])), 2),
                "mean_share": round(mean_share, 4),
                "last3_share": round(recent_share, 4),
                "trend": "rising" if recent_share > 1.2 * mean_share else ("falling" if recent_share < 0.8 * mean_share else "stable"),
            }
        )
    meta = {
        "years": years,
        "questions_per_year": {str(y): per_year[y] for y in years},
        "short_years": [y for y in years if per_year[y] < SHORT_YEAR_QUESTIONS],
        "rows_without_year": missing_year,
    }
    return sorted(out, key=lambda r: -r["mean_share"]), meta


def taxonomy_chapter_weightage(taxonomy: dict) -> tuple[dict[str, dict], dict]:
    entries = taxonomy.get("Chapter_Weightage_And_Marks", [])
    per_year = Counter()
    chapter_year = defaultdict(Counter)
    subjects = defaultdict(set)
    for e in entries:
        subjects[e["chapter"]].add(e["subject"])
        for y, v in e.get("yearly_weightage", {}).items():
            chapter_year[e["chapter"]][int(y)] += int(v.get("question_count", 0))
            per_year[int(y)] += int(v.get("question_count", 0))
    years = sorted(per_year)
    out = {}
    for chapter, cnt in chapter_year.items():
        shares = [cnt[y] / per_year[y] for y in years]
        recent = [cnt[y] / per_year[y] for y in years[-3:]]
        mean_share, recent_share = float(np.mean(shares)), float(np.mean(recent))
        out[chapter] = {
            "subject": "/".join(sorted(subjects[chapter])),
            "chapter": chapter,
            "years_present": sum(1 for y in years if cnt[y] > 0),
            "mean_questions_per_paper": round(float(np.mean([cnt[y] for y in years])), 2),
            "mean_share": round(mean_share, 6),
            "last3_share": round(recent_share, 6),
            "trend": "rising" if recent_share > 1.2 * mean_share else ("falling" if recent_share < 0.8 * mean_share else "stable"),
        }
    meta = {"years": years, "questions_per_year": {str(y): per_year[y] for y in years}, "chapters": len(out)}
    return out, meta


def cross_check_with_bank(taxonomy_rows: dict[str, dict], taxonomy_meta: dict, bank_rows: list[dict]) -> dict:
    from scipy.stats import spearmanr

    bank_weight, bank_meta = pyq_chapter_weightage(bank_rows)
    bank_by_chapter = {r["chapter"]: r for r in bank_weight}
    shared = sorted(set(taxonomy_rows) & set(bank_by_chapter))
    rho = float(spearmanr([taxonomy_rows[c]["mean_share"] for c in shared], [bank_by_chapter[c]["mean_share"] for c in shared]).statistic)
    tax_total = sum(taxonomy_meta["questions_per_year"].values())
    bank_total = sum(bank_meta["questions_per_year"].values())
    gap_by_year = {y: taxonomy_meta["questions_per_year"].get(y, 0) - bank_meta["questions_per_year"].get(y, 0) for y in taxonomy_meta["questions_per_year"]}
    return {
        "taxonomy_questions": tax_total,
        "bank_questions": bank_total,
        "bank_missing_vs_taxonomy": tax_total - bank_total,
        "bank_missing_by_year": gap_by_year,
        "chapters_only_in_bank": sorted(set(bank_by_chapter) - set(taxonomy_rows)),
        "chapters_only_in_taxonomy": sorted(set(taxonomy_rows) - set(bank_by_chapter)),
        "chapter_share_spearman": round(rho, 4),
    }
