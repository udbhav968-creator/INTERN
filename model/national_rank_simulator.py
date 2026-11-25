from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_FACTS_PATH = Path(__file__).resolve().parent / "neet_reference_facts.json"
_FACTS = json.loads(_FACTS_PATH.read_text(encoding="utf-8"))

SCORE_ANCHORS = np.array(_FACTS["score_percentile_anchors"]["score"], dtype=np.float64)
PERCENTILE_ANCHORS = np.array(_FACTS["score_percentile_anchors"]["percentile"], dtype=np.float64)
_NUMERIC_FACTS = ("registered", "appeared", "qualified", "general_cutoff_score", "topper_score")
NEET_YEAR_FACTS = {int(y): {k: v for k, v in f.items() if k in _NUMERIC_FACTS} for y, f in _FACTS["years"].items()}
FACTS_PROVENANCE = {
    "years": {int(y): {"source": f["source"], "verification": f["verification"]} for y, f in _FACTS["years"].items()},
    "score_percentile_anchors": _FACTS["score_percentile_anchors"]["status"],
    "college_tiers": _FACTS["college_tiers"]["status"],
}
REFERENCE_EXAM_YEAR = int(_FACTS["reference_exam_year"])
TOTAL_NEET_CANDIDATES = NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["appeared"]

QUALIFYING_CUTOFF_SOURCE = {
    "exam_year": REFERENCE_EXAM_YEAR,
    "category": "General/EWS 50th percentile",
    "general_cutoff_score": NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["general_cutoff_score"],
    "source": "NTA NEET UG 2026 result press coverage (re-exam June 21, result July 16, 2026)",
}

MODEL_ASSUMPTIONS = {
    "candidate_count": {"value": TOTAL_NEET_CANDIDATES, "status": "sourced", "basis": "NEET UG 2026 candidates appeared"},
    "qualifying_cutoff": {"value": QUALIFYING_CUTOFF_SOURCE["general_cutoff_score"], "status": "sourced", "basis": QUALIFYING_CUTOFF_SOURCE["source"]},
    "score_to_percentile_curve": {"status": "approximate", "basis": "hand-set piecewise-linear anchors; NTA does not publish a full score-percentile table; year-to-year shifts of tens of marks are normal"},
    "college_tier_thresholds": {"status": "approximate", "basis": "indicative closing scores/ranks from past counselling rounds; vary by year, category, quota and round"},
    "rank_formula": {"status": "derived", "basis": "rank = (1 - percentile/100) x candidates appeared"},
}

COLLEGE_TIERS = [
    *_FACTS["college_tiers"]["tiers"],
    {"id": "qualifying_floor", "name": "NEET Qualifying Cutoff", "score_threshold": QUALIFYING_CUTOFF_SOURCE["general_cutoff_score"], "air_threshold": NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["qualified"]},
]


def score_to_neet_percentile(score: float) -> float:
    clamped = max(0.0, min(720.0, float(score)))
    return float(np.interp(clamped, SCORE_ANCHORS, PERCENTILE_ANCHORS))


def score_to_all_india_rank(score: float, n_candidates: int = TOTAL_NEET_CANDIDATES) -> int:
    pct = score_to_neet_percentile(score)
    fraction_above = (100.0 - pct) / 100.0
    rank = int(round(fraction_above * n_candidates))
    return max(1, rank)


def simulate_national_rank_and_college_cutoffs(
    total_scores_trials: np.ndarray,
    conformal_score_interval: list[int] | tuple[int, int],
    mean_score: int | float,
    n_candidates: int = TOTAL_NEET_CANDIDATES,
) -> dict[str, Any]:
    low_score, high_score = conformal_score_interval
    air_best = score_to_all_india_rank(float(high_score), n_candidates=n_candidates)
    air_worst = score_to_all_india_rank(float(low_score), n_candidates=n_candidates)
    air_median = score_to_all_india_rank(float(mean_score), n_candidates=n_candidates)

    pct_low = round(score_to_neet_percentile(float(low_score)), 3)
    pct_high = round(score_to_neet_percentile(float(high_score)), 3)
    pct_median = round(score_to_neet_percentile(float(mean_score)), 3)

    college_admissions: list[dict[str, Any]] = []
    for tier in COLLEGE_TIERS:
        th = tier["score_threshold"]
        prob = float(np.mean(total_scores_trials >= th) * 100.0) if len(total_scores_trials) > 0 else (100.0 if mean_score >= th else 0.0)
        college_admissions.append({
            "college_id": tier["id"],
            "college_tier": tier["name"],
            "target_score_720": th,
            "target_air_cutoff": tier["air_threshold"],
            "admission_probability_percent": round(prob, 1),
            "threshold_status": "sourced" if tier["id"] == "qualifying_floor" else "approximate",
            "status": "SECURE" if prob >= 80.0 else ("COMPETITIVE" if prob >= 40.0 else ("REACH" if prob >= 10.0 else "UNLIKELY")),
        })

    return {
        "total_national_candidates": n_candidates,
        "conformal_air_interval": [air_best, air_worst],
        "projected_air_median": air_median,
        "conformal_percentile_interval": [pct_low, pct_high],
        "projected_percentile_median": pct_median,
        "college_admission_probabilities": college_admissions,
        "reference_exam_year": REFERENCE_EXAM_YEAR,
        "model_assumptions": MODEL_ASSUMPTIONS,
    }
