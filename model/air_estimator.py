from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .assessment_blueprint import MARKS_CORRECT, MARKS_WRONG, SUBJECT_ORDER, load_forms, score_initial_assessment
from .national_rank_simulator import (
    COLLEGE_TIERS,
    MODEL_ASSUMPTIONS,
    NEET_YEAR_FACTS,
    REFERENCE_EXAM_YEAR,
    score_to_all_india_rank,
    score_to_neet_percentile,
)
from .pyq_weightage import pyq_chapter_weightage, taxonomy_chapter_weightage
from .model_provenance import reliability_summary
from .response_hygiene import clean_responses
from .student_profile_initializer import _bank_by_id, _p_correct, initialize_student_profile
from .syllabus import DEFAULT_SYLLABUS_MODE, NMC_REDUCED_SYLLABUS_EXCLUSIONS, in_syllabus, normalise_syllabus_mode

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
CHAPTER_SUMMARY_PATH = PROJECT_ROOT / "data" / "taxonomy" / "chapter_wise_neet_pyq_summary.json"
RELABEL_PATH = MODEL_DIR / "pyq_difficulty_relabel_proposal.json"
TAXONOMY_PATH = PROJECT_ROOT / "data" / "taxonomy" / "NEET_Unified_Weightage_And_Marks_2013_2026.json"

REPORT_VERSION = "air_estimate_v3"
NEET_QUESTIONS = 180
FULL_MOCK_MIN_QUESTIONS = 144
MIN_SCORED_PER_SUBJECT = 10
TARGET_MASTERY = 0.85
MOCK_DIFFICULTY_TOLERANCE_MARKS = 6.0
Z90 = 1.645
FOCUS_CHAPTERS = 10
TIER_NAMES = ("basic", "medium", "advanced")
LABEL_SCENARIOS = ("v21_labels", "labels_uninformative")
DISPLAY_SIGNIFICANT_DIGITS = 2
WATCHLIST_CHAPTERS = 8
MIN_RISING_QUESTIONS_PER_PAPER = 2.0


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@lru_cache(maxsize=1)
def _weightage() -> tuple[dict[str, dict], dict]:
    pyq = json.loads(PYQ_PATH.read_text(encoding="utf-8"))
    bank_rows, bank_meta = pyq_chapter_weightage(pyq)
    bank = {r["chapter"]: r for r in bank_rows}
    if not TAXONOMY_PATH.exists():
        return bank, {**bank_meta, "source": "pyq_bank_only", "chapters_from_taxonomy": 0, "chapters_from_bank_fallback": len(bank)}
    taxonomy, tax_meta = taxonomy_chapter_weightage(json.loads(TAXONOMY_PATH.read_text(encoding="utf-8")))
    merged = dict(taxonomy)
    fallback = sorted(set(bank) - set(taxonomy))
    for ch in fallback:
        merged[ch] = bank[ch]
    total = sum(v["mean_share"] for v in merged.values())
    merged = {ch: {**v, "mean_share": v["mean_share"] / total, "last3_share": v["last3_share"] / total} for ch, v in merged.items()}
    return merged, {
        "years": tax_meta["years"],
        "questions_per_year": tax_meta["questions_per_year"],
        "short_years": [],
        "source": "taxonomy_with_bank_fallback",
        "chapters_from_taxonomy": len(taxonomy),
        "chapters_from_bank_fallback": fallback,
    }


@lru_cache(maxsize=1)
def _chapter_summary() -> dict[str, dict]:
    return json.loads(CHAPTER_SUMMARY_PATH.read_text(encoding="utf-8")).get("chapters", {})


@lru_cache(maxsize=1)
def _chapter_mean_b() -> dict[str, float]:
    by_chapter = defaultdict(list)
    for q in _bank_by_id("unified").values():
        if q.get("source_type") == "pyq":
            by_chapter[q["chapter"]].append(q["b"])
    return {ch: round(float(np.mean(v)), 3) for ch, v in by_chapter.items()}


@lru_cache(maxsize=1)
def _reference_form() -> list[dict]:
    bank = _bank_by_id("unified")
    return [bank[q] for q in load_forms()["forms"][0]["question_ids"] if q in bank]


def _expected_marks(questions: list[dict], theta: float) -> float:
    p = np.array([float(_p_correct(np.array(theta), q)) for q in questions])
    return float(np.sum(MARKS_CORRECT * p + MARKS_WRONG * (1.0 - p)))


def _scored(cleaned: list[dict], bank: dict[str, dict]) -> list[tuple[dict, dict]]:
    return [(r, bank[r["question_id"]]) for r in cleaned if r.get("is_correct") is not None and r["question_id"] in bank]


def _evidence(cleaned: list[dict], bank: dict[str, dict]) -> dict:
    scored = _scored(cleaned, bank)
    per_subject = {s: 0 for s in SUBJECT_ORDER}
    chapters = set()
    for _, q in scored:
        per_subject[q["subject"]] = per_subject.get(q["subject"], 0) + 1
        chapters.add(q["chapter"])
    sufficient = all(per_subject[s] >= MIN_SCORED_PER_SUBJECT for s in SUBJECT_ORDER)
    if sufficient and len(scored) >= FULL_MOCK_MIN_QUESTIONS:
        scope = "full_mock"
    elif sufficient:
        scope = "partial"
    elif len(chapters) == 1:
        scope = "single_chapter"
    else:
        scope = "partial_insufficient"
    return {
        "scored_questions": len(scored),
        "per_subject": per_subject,
        "chapters_touched": len(chapters),
        "scope": scope,
        "sufficient_for_air": sufficient,
    }


def _mock_difficulty(scored: list[tuple[dict, dict]]) -> dict | None:
    if not scored:
        return None
    qs = [q for _, q in scored]
    ref = _reference_form()
    this_avg = _expected_marks(qs, 0.0) * (NEET_QUESTIONS / len(qs))
    ref_avg = _expected_marks(ref, 0.0)
    gap = this_avg - ref_avg
    label = "similar to" if abs(gap) <= MOCK_DIFFICULTY_TOLERANCE_MARKS else ("easier than" if gap > 0 else "harder than")
    return {
        "questions": len(qs),
        "mean_item_difficulty_b": round(float(np.mean([q["b"] for q in qs])), 3),
        "reference_mean_item_difficulty_b": round(float(np.mean([q["b"] for q in ref])), 3),
        "average_student_marks_on_this_paper_per_180": round(this_avg, 1),
        "average_student_marks_on_reference_form": round(ref_avg, 1),
        "gap_marks": round(gap, 1),
        "verdict": f"{label} the NEET-style reference form",
        "tolerance_marks": MOCK_DIFFICULTY_TOLERANCE_MARKS,
    }


def _subject_wise(scored: list[tuple[dict, dict]]) -> dict:
    out = {s: {"attempted": 0, "correct": 0, "wrong": 0, "marks": 0} for s in SUBJECT_ORDER}
    for r, q in scored:
        row = out[q["subject"]]
        row["attempted"] += 1
        if r["is_correct"]:
            row["correct"] += 1
            row["marks"] += MARKS_CORRECT
        else:
            row["wrong"] += 1
            row["marks"] += MARKS_WRONG
    for row in out.values():
        row["accuracy"] = round(row["correct"] / row["attempted"], 4) if row["attempted"] else None
    return out


def _round_significant(value: float, digits: int = DISPLAY_SIGNIFICANT_DIGITS) -> int:
    if value <= 0:
        return 0
    magnitude = 10 ** max(0, int(math.floor(math.log10(value))) - digits + 1)
    return int(round(value / magnitude) * magnitude)


@lru_cache(maxsize=1)
def _label_model() -> tuple[dict, dict, dict]:
    by_cell, by_subject = defaultdict(list), defaultdict(list)
    for q in _bank_by_id("unified").values():
        if q.get("source_type") == "pyq":
            by_cell[(q["subject"], q["tier"])].append(q["b"])
            by_subject[q["subject"]].append(q["b"])
    probs = {}
    if RELABEL_PATH.exists():
        probs = {i["question_id"]: i["proposed_probabilities"] for i in json.loads(RELABEL_PATH.read_text(encoding="utf-8")).get("items", [])}
    return (
        {k: float(np.mean(v)) for k, v in by_cell.items()},
        {k: float(np.mean(v)) for k, v in by_subject.items()},
        probs,
    )


def _scenario_items(items: list[dict], scenario: str) -> list[dict]:
    cell_mean, subject_mean, probs = _label_model()
    out = []
    for q in items:
        if q.get("source_type") != "pyq":
            out.append(q)
            continue
        subject = q["subject"]
        if scenario == "v21_labels":
            p = probs.get(q["question_id"])
            if p is None:
                out.append(q)
                continue
            b = q["b"] - cell_mean[(subject, q["tier"])] + sum(p[t] * cell_mean[(subject, t)] for t in TIER_NAMES)
        else:
            b = subject_mean[subject]
        out.append({**q, "b": b})
    return out


def _label_sensitivity(cleaned: list[dict], scored: list[tuple[dict, dict]], equated: float) -> dict:
    items = [q for _, q in scored]
    reference = _reference_form()
    base_theta = score_initial_assessment(cleaned, items)["theta_eap"]
    base_marks = _expected_marks(reference, base_theta)
    out = {}
    for name in LABEL_SCENARIOS:
        theta = score_initial_assessment(cleaned, _scenario_items(items, name))["theta_eap"]
        delta = _expected_marks(_scenario_items(reference, name), theta) - base_marks
        out[name] = {"theta": round(theta, 4), "equated_marks": round(equated + delta, 1), "delta_marks": round(delta, 1)}
    return out


def _range_if_assumptions_wrong(lo: float, hi: float, sensitivity: dict) -> list[int]:
    ref_cut = NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["general_cutoff_score"]
    best, worst = [], []
    for delta in [0.0] + [v["delta_marks"] for v in sensitivity.values()]:
        for facts in NEET_YEAR_FACTS.values():
            shift = ref_cut - facts["general_cutoff_score"] + delta
            n = facts["appeared"]
            best.append(score_to_all_india_rank(float(np.clip(hi + shift, 0.0, 720.0)), n_candidates=n))
            worst.append(score_to_all_india_rank(float(np.clip(lo + shift, 0.0, 720.0)), n_candidates=n))
    return [min(best), max(worst)]


def _year_scenarios(equated: float) -> dict:
    ref_cut = NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["general_cutoff_score"]
    out = {}
    for year, facts in sorted(NEET_YEAR_FACTS.items()):
        shift = ref_cut - facts["general_cutoff_score"]
        adjusted = float(np.clip(equated + shift, 0.0, 720.0))
        n = facts["appeared"]
        out[str(year)] = {
            "general_cutoff": facts["general_cutoff_score"],
            "candidates_appeared": n,
            "score_on_reference_scale": round(adjusted, 1),
            "percentile": round(score_to_neet_percentile(adjusted), 3),
            "air": score_to_all_india_rank(adjusted, n_candidates=n),
        }
    return out


def _college_reach(equated: float, lo: float, hi: float) -> list[dict]:
    sd = max(1.0, (hi - lo) / (2.0 * Z90))
    out = []
    for tier in COLLEGE_TIERS:
        prob = 100.0 * (1.0 - _normal_cdf((tier["score_threshold"] - equated) / sd))
        out.append(
            {
                "college_tier": tier["name"],
                "target_score_720": tier["score_threshold"],
                "target_air_cutoff": tier["air_threshold"],
                "probability_percent": round(prob, 1),
                "status": "SECURE" if prob >= 80.0 else ("COMPETITIVE" if prob >= 40.0 else ("REACH" if prob >= 10.0 else "UNLIKELY")),
                "threshold_status": "sourced" if tier["id"] == "qualifying_floor" else "approximate",
            }
        )
    return out


def _chapter_rows(profile: dict, current: list[tuple[dict, dict]], equated: float | None, target: float, mode: str) -> list[dict]:
    weight, _ = _weightage()
    included = {ch: v for ch, v in weight.items() if in_syllabus(ch, mode)}
    included_share = sum(v["mean_share"] for v in included.values()) or 1.0
    summary = _chapter_summary()
    mean_b = _chapter_mean_b()
    attempt = defaultdict(lambda: [0, 0])
    for r, q in current:
        attempt[q["chapter"]][0] += 1
        attempt[q["chapter"]][1] += 1 if r["is_correct"] else 0
    rows = []
    for ts in profile["topic_states"]:
        ch = ts["topic_id"]
        w = weight.get(ch, {})
        s = summary.get(ch, {})
        expected_q = NEET_QUESTIONS * w.get("mean_share", 0.0) / included_share if in_syllabus(ch, mode) else 0.0
        mastery = ts["mastery_prob"]
        gain = expected_q * (MARKS_CORRECT - MARKS_WRONG) * max(0.0, target - mastery)
        air_gain = None
        if equated is not None:
            air_gain = score_to_all_india_rank(equated) - score_to_all_india_rank(min(720.0, equated + gain))
        total_pyq = s.get("total_pyqs") or 0
        attempted, correct = attempt[ch]
        rows.append(
            {
                "chapter": ch,
                "subject": ts["subject"],
                "mastery": mastery,
                "level": ts["level"],
                "in_syllabus": in_syllabus(ch, mode),
                "nmc_reduced_excluded": ch in NMC_REDUCED_SYLLABUS_EXCLUSIONS,
                "evidence": "measured" if ts["observed"] else "prior_only",
                "attempted_in_this_attempt": attempted,
                "correct_in_this_attempt": correct,
                "expected_questions_per_neet_paper": round(expected_q, 2),
                "pyq_trend": w.get("trend"),
                "pyq_mean_share": w.get("mean_share"),
                "pyq_last3_share": w.get("last3_share"),
                "chapter_difficulty": {
                    "mean_irt_b": mean_b.get(ch),
                    "pyq_advanced_share": round(s.get("advanced_pyqs", 0) / total_pyq, 3) if total_pyq else None,
                    "mean_ideal_time_sec": s.get("mean_ideal_time_sec"),
                },
                "marks_at_stake_if_target_reached": round(gain, 1),
                "air_places_gained_if_target_reached": air_gain,
            }
        )
    return sorted(rows, key=lambda r: (-r["marks_at_stake_if_target_reached"], r["chapter"]))


def _watchlist(chapter_rows: list[dict]) -> list[dict]:
    rising = [
        r for r in chapter_rows
        if r["pyq_trend"] == "rising"
        and r["in_syllabus"]
        and r["expected_questions_per_neet_paper"] >= MIN_RISING_QUESTIONS_PER_PAPER
        and r["mastery"] < TARGET_MASTERY
    ]
    rising.sort(key=lambda r: -(r["pyq_last3_share"] or 0.0) / max(r["pyq_mean_share"] or 1e-9, 1e-9))
    return [
        {
            "chapter": r["chapter"],
            "subject": r["subject"],
            "mastery": r["mastery"],
            "pyq_mean_share": r["pyq_mean_share"],
            "pyq_last3_share": r["pyq_last3_share"],
        }
        for r in rising[:WATCHLIST_CHAPTERS]
    ]


def _assumptions(meta: dict, mode: str) -> dict:
    return {
        "syllabus_mode": {"value": mode, "excluded_chapters_in_nmc_reduced": sorted(NMC_REDUCED_SYLLABUS_EXCLUSIONS)},
        "weightage_source": {k: meta.get(k) for k in ("source", "chapters_from_taxonomy", "chapters_from_bank_fallback")},
        "target_mastery": {"value": TARGET_MASTERY, "kind": "scenario parameter, not a measurement"},
        "minimum_scored_per_subject_for_air": {"value": MIN_SCORED_PER_SUBJECT, "kind": "policy threshold, not statistically derived"},
        "mock_difficulty_tolerance_marks": {"value": MOCK_DIFFICULTY_TOLERANCE_MARKS, "kind": "same tolerance as the equated-form gate"},
        "score_to_percentile_curve": MODEL_ASSUMPTIONS["score_to_percentile_curve"],
        "paper_year_scenarios": "same marks re-read on the curve shifted by the difference in sourced general cut-off scores (2024-2026); an approximation, not a published distribution",
        "college_probabilities": "normal approximation to the 90% score interval against approximate tier thresholds",
        "ability_basis": "IRT 3PL with priors; no real student attempts exist yet to fit item parameters",
        "chapter_keys": "Thermodynamics is one chapter key for Physics and Chemistry (known issue)",
        "chapter_marks_at_stake": "each chapter assumes only that chapter reaches the target; the values are not additive",
    }


def _reliability(sensitivity: dict | None) -> dict:
    out = dict(reliability_summary())
    if sensitivity:
        out["difficulty_label_sensitivity_marks"] = {k: v["delta_marks"] for k, v in sensitivity.items()}
        deltas = [v["delta_marks"] for v in sensitivity.values()]
        biggest = max(abs(d) for d in deltas)
        all_lower = all(d <= -MOCK_DIFFICULTY_TOLERANCE_MARKS for d in deltas)
        out["direction_of_label_risk"] = "all tested alternatives lower the estimate" if all_lower else ("all tested alternatives raise the estimate" if all(d >= MOCK_DIFFICULTY_TOLERANCE_MARKS for d in deltas) else "mixed")
        out["statement"] = (
            f"Indicative range only. It rests on prior item parameters, not on real students, and changing the difficulty-label assumption alone moves the equated marks by up to {biggest:.0f}. "
            + ("Every tested alternative lowers the estimate, so the upper half of the range deserves extra caution. " if all_lower else "")
            + "The wider range_if_assumptions_wrong includes that and the 2024-2026 paper-year variation."
        )
    else:
        out["statement"] = "No rank is given: the evidence does not cover all three subjects."
    return out


def build_air_report(
    responses: list[dict],
    history: list[dict] | None = None,
    target_mastery: float = TARGET_MASTERY,
    syllabus_mode: str = DEFAULT_SYLLABUS_MODE,
) -> dict:
    mode = normalise_syllabus_mode(syllabus_mode)
    bank = _bank_by_id("unified")
    current_clean, hygiene = clean_responses(responses)
    current_scored = _scored(current_clean, bank)
    base = {"report_version": REPORT_VERSION, "syllabus_mode": mode, "data_hygiene": hygiene}
    if not current_scored and not history:
        return {**base, "status": "no_scored_responses", "air": None, "reason": "no scored answers were provided"}

    evidence = _evidence(current_clean, bank)
    source = "attempt"
    used = list(responses)
    if history:
        combined = list(history) + list(responses)
        combined_evidence = _evidence(clean_responses(combined)[0], bank)
        if not evidence["sufficient_for_air"] and combined_evidence["sufficient_for_air"]:
            evidence, source, used = combined_evidence, "attempt_plus_history", combined
        elif not evidence["sufficient_for_air"]:
            used = combined
    profile = initialize_student_profile(used)
    predicted = profile["predicted_neet_score"]
    _, meta = _weightage()

    air = None
    reason = None
    equated = None
    sensitivity = None
    if evidence["sufficient_for_air"]:
        equated = float(predicted["expected_marks"])
        lo, hi = predicted["interval_90"]
        air = {
            "estimate": score_to_all_india_rank(equated),
            "range_90": [score_to_all_india_rank(hi), score_to_all_india_rank(lo)],
            "percentile": round(score_to_neet_percentile(equated), 3),
            "reference_exam_year": REFERENCE_EXAM_YEAR,
            "candidates_appeared": NEET_YEAR_FACTS[REFERENCE_EXAM_YEAR]["appeared"],
            "curve_status": MODEL_ASSUMPTIONS["score_to_percentile_curve"]["status"],
            "paper_year_scenarios": _year_scenarios(equated),
        }
        ranks = [v["air"] for v in air["paper_year_scenarios"].values()]
        air["paper_year_spread"] = [min(ranks), max(ranks)]
        sensitivity = _label_sensitivity(clean_responses(used)[0], _scored(clean_responses(used)[0], bank), equated)
        air["range_if_assumptions_wrong"] = _range_if_assumptions_wrong(lo, hi, sensitivity)
        air["display"] = {
            "estimate": _round_significant(air["estimate"]),
            "range_90": [_round_significant(v) for v in air["range_90"]],
            "range_if_assumptions_wrong": [_round_significant(v) for v in air["range_if_assumptions_wrong"]],
            "note": f"rounded to {DISPLAY_SIGNIFICANT_DIGITS} significant digits to avoid false precision",
        }
    else:
        reason = (
            f"AIR needs at least {MIN_SCORED_PER_SUBJECT} scored questions in each of Physics, Chemistry and Biology; "
            f"this evidence has {evidence['per_subject']}"
        )

    chapters = _chapter_rows(profile, current_scored, equated, target_mastery, mode)
    return {
        **base,
        "status": "ok" if air else "insufficient_evidence_for_air",
        "evidence": {**evidence, "source": source},
        "scores": {
            "raw_marks_this_attempt": sum(MARKS_CORRECT if r["is_correct"] else MARKS_WRONG for r, _ in current_scored),
            "equated_neet_marks": equated,
            "interval_90": predicted["interval_90"] if air else None,
            "theta": profile["global_theta"],
            "theta_sem": profile["global_theta_sem"],
        },
        "mock_difficulty": _mock_difficulty(current_scored),
        "sensitivity": {"difficulty_labels": sensitivity, "meaning": "same answers re-estimated if PYQ difficulty followed the v2.1 human-trained labels, or if the labels carried no information; the production estimate uses the formula labels"} if sensitivity else None,
        "reliability": _reliability(sensitivity),
        "air": air,
        "reason": reason,
        "college_reach": _college_reach(equated, *predicted["interval_90"]) if air else None,
        "subject_wise": _subject_wise(current_scored),
        "chapter_wise": chapters,
        "focus_chapters": chapters[:FOCUS_CHAPTERS],
        "trend_watchlist": _watchlist(chapters),
        "assumptions": _assumptions(meta, mode),
    }
