import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.air_estimator import build_air_report  # noqa: E402
from model.national_rank_simulator import COLLEGE_TIERS, PERCENTILE_ANCHORS, SCORE_ANCHORS, score_to_all_india_rank  # noqa: E402
from model.student_profile_initializer import _bank_by_id, _p_correct  # noqa: E402

OUT = ROOT / "docs" / "air_validation_simulation.json"


def answer(bank, qids, theta, rng):
    return [
        {"question_id": q, "is_correct": bool(rng.random() < float(_p_correct(np.array(theta), bank[q]))), "time_taken_seconds": float(bank[q]["ideal_time_sec"])}
        for q in qids
    ]


def extreme_paper(bank, hard):
    ids = []
    for subject, n in (("Physics", 45), ("Chemistry", 45), ("Biology", 90)):
        pool = sorted((q for q in bank.values() if q["subject"] == subject and q["source_type"] == "pyq"), key=lambda q: q["b"], reverse=hard)
        ids += [q["question_id"] for q in pool[:n]]
    return ids


def main():
    bank = _bank_by_id("unified")
    forms = json.loads((ROOT / "model" / "assessment_forms.json").read_text(encoding="utf-8"))["forms"]
    out = {"note": "all students are simulated from the model's own item parameters; this validates the estimator, not real students"}

    calibration = []
    for theta in (-1.5, -1.0, 0.0, 1.0, 2.0, 2.5):
        errs, sems, airs = [], [], []
        for k in range(120):
            rng = np.random.default_rng(31000 + 1000 * int(theta * 2 + 10) + k)
            rep = build_air_report(answer(bank, forms[0]["question_ids"], theta, rng))
            errs.append(rep["scores"]["theta"] - theta)
            sems.append(rep["scores"]["theta_sem"])
            airs.append(rep["air"]["estimate"])
        e, s = np.array(errs), np.array(sems)
        calibration.append(
            {
                "true_theta": theta,
                "n": len(e),
                "mean_error": round(float(e.mean()), 4),
                "rmse": round(float(np.sqrt((e ** 2).mean())), 4),
                "mean_claimed_sem": round(float(s.mean()), 4),
                "coverage_95": round(float((np.abs(e) <= 1.96 * s).mean()), 4),
                "median_air": int(np.median(airs)),
            }
        )
    out["calibration"] = calibration

    papers = {
        "NEET-style form 0": forms[0]["question_ids"],
        "NEET-style form 1": forms[1]["question_ids"],
        "NEET-style form 2": forms[2]["question_ids"],
        "deliberately hard": extreme_paper(bank, True),
        "deliberately easy": extreme_paper(bank, False),
    }
    equating = []
    for name, qids in papers.items():
        raws, eqs, airs, widths = [], [], [], []
        for k in range(24):
            rng = np.random.default_rng(52000 + k)
            rep = build_air_report(answer(bank, qids, 0.5, rng))
            raws.append(rep["scores"]["raw_marks_this_attempt"])
            eqs.append(rep["scores"]["equated_neet_marks"])
            airs.append(rep["air"]["estimate"])
            widths.append(rep["air"]["range_90"][1] - rep["air"]["range_90"][0])
        equating.append(
            {
                "paper": name,
                "true_theta": 0.5,
                "mean_raw_marks": round(float(np.mean(raws)), 1),
                "mean_equated_marks": round(float(np.mean(eqs)), 1),
                "median_air": int(np.median(airs)),
                "median_range_width": int(np.median(widths)),
                "verdict": rep["mock_difficulty"]["verdict"],
            }
        )
    out["equating"] = equating

    sensitivity = []
    for theta in (-1.0, 0.0, 1.0, 2.0):
        rows = []
        for k in range(30):
            rng = np.random.default_rng(81000 + 1000 * int(theta + 5) + k)
            rep = build_air_report(answer(bank, forms[0]["question_ids"], theta, rng))
            d = rep["sensitivity"]["difficulty_labels"]
            rows.append(
                {
                    "formula": rep["scores"]["equated_neet_marks"],
                    "v21": d["v21_labels"]["equated_marks"],
                    "uninformative": d["labels_uninformative"]["equated_marks"],
                    "estimate": rep["air"]["estimate"],
                    "width_90": rep["air"]["range_90"][1] - rep["air"]["range_90"][0],
                    "range_wrong": rep["air"]["range_if_assumptions_wrong"],
                }
            )
        sensitivity.append(
            {
                "true_theta": theta,
                "students": len(rows),
                "mean_marks_formula": round(float(np.mean([r["formula"] for r in rows])), 1),
                "mean_marks_v21": round(float(np.mean([r["v21"] for r in rows])), 1),
                "mean_marks_uninformative": round(float(np.mean([r["uninformative"] for r in rows])), 1),
                "median_air": int(np.median([r["estimate"] for r in rows])),
                "median_range_if_assumptions_wrong": [int(np.median([r["range_wrong"][0] for r in rows])), int(np.median([r["range_wrong"][1] for r in rows]))],
            }
        )
    out["label_sensitivity"] = sensitivity

    rng = np.random.default_rng(2026)
    sample = build_air_report(answer(bank, forms[0]["question_ids"], 0.8, rng))
    out["sample_year_scenarios"] = sample["air"]["paper_year_scenarios"]
    out["sample_scores"] = sample["scores"]
    out["sample_focus_chapters"] = sample["focus_chapters"][:8]
    out["sample_watchlist"] = sample["trend_watchlist"]
    out["sample_college_reach"] = sample["college_reach"]
    out["sample_air"] = {k: sample["air"][k] for k in ("estimate", "range_90", "percentile", "paper_year_spread", "range_if_assumptions_wrong", "display")}
    out["sample_reliability"] = sample["reliability"]
    out["sample_sensitivity"] = sample["sensitivity"]

    out["rank_curve"] = {
        "score_anchors": [float(x) for x in SCORE_ANCHORS],
        "percentile_anchors": [float(x) for x in PERCENTILE_ANCHORS],
        "air_at_score": {str(s): score_to_all_india_rank(s) for s in range(0, 721, 40)},
        "college_tiers": [{"name": t["name"], "score": t["score_threshold"], "air": t["air_threshold"]} for t in COLLEGE_TIERS],
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
