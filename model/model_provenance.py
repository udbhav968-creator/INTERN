from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
ATTEMPTS_PATH = PROJECT_ROOT / "data" / "training" / "student_attempts.csv"
REPORT_PATH = MODEL_DIR / "model_provenance_report.json"

BASES = (
    "fitted_on_real_students",
    "fitted_on_real_human_labels",
    "fitted_on_formula_derived_labels",
    "hand_designed_prior",
    "design_objective_simulated",
    "rule_based",
    "math_on_priors",
    "built_waiting_for_data",
)
CLAIMS_ACCURACY = ("fitted_on_real_human_labels", "fitted_on_formula_derived_labels")


def _load(name: str, default=None):
    path = MODEL_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _get(d, *keys):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def real_student_attempts() -> int:
    if not ATTEMPTS_PATH.exists():
        return 0
    with ATTEMPTS_PATH.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _stage_output(run: dict, name: str) -> dict:
    for s in run.get("stages", []):
        if s.get("stage") == name:
            return s.get("output") or {}
    return {}


def build_provenance() -> dict:
    attempts = real_student_attempts()
    run = _load("ml_pipeline_run.json", {})
    label = _load("label_validity_report.json", {})
    v21 = _load("difficulty_model_v21_report.json", {})
    valid = _load("validation_report.json", {})
    ncert = _load("ncert_rag_svd_meta.json", {})
    ingest = _load("sample_paper_ingestion_report.json", {})
    chapters = _load("chapter_topic_evaluation_report.json", {})
    harness = _load("harness_benchmark_report.json", {})
    equiv = _load("assessment_equivalence_report.json", {})
    student = _load("student_models/student_response_training_report.json", {})
    promo = {d.get("candidate"): d for d in _load("promotion_gate_report.json", {}).get("decisions", [])}
    from .national_rank_simulator import FACTS_PROVENANCE

    kappa = _get(label, "heuristic_rule_vs_human_verified_sample_tiers", "cohens_kappa")
    cold = _get(harness, "linucb_cold_start_stochastic_ablation") or {}
    below = _get(chapters, "chapters", "reliable_below_majority") or []
    res = _get(ingest, "chapter_resolution", "classifier_metrics_by_subject") or {}
    sr_status = student.get("status")
    selftest = _stage_output(run, "student_pipeline_self_test")
    air_self = _stage_output(run, "air_estimator_self_test")

    def m(id_, name, job, basis, parameters, evidence, limits, real_labels=False):
        return {
            "id": id_,
            "name": name,
            "job": job,
            "basis": basis,
            "parameters_come_from": parameters,
            "evidence": evidence,
            "uses_real_human_labels": real_labels,
            "validated_on_real_students": False,
            "limitations": limits,
        }

    models = [
        m("irt_items_pyq", "3PL IRT item parameters (PYQs)", "difficulty, discrimination, guessing per question", "hand_designed_prior",
          "hand-designed formula: per-tier base values plus constant adjustments for time, text length, explanation steps, diagram and question type",
          {"accuracy": None},
          ["no response data exists to fit them", "the time input is itself derived from the tier, so the chain is circular", f"formula tiers agree with human tiers at kappa {kappa}"]),
        m("irt_ability", "IRT ability estimate (EAP)", "student ability theta from answers", "math_on_priors",
          "Bayesian posterior under the item parameters above and a N(0,1) prior",
          {"simulated_calibration_95_coverage": air_self.get("interval_95_coverage")},
          ["calibration holds only for simulated students drawn from the same parameters", "no real student has been scored against a ground truth"]),
        m("cold_start_ridge", "Cold-start ridge (new items)", "a, b, c for unseen questions", "fitted_on_formula_derived_labels",
          "ridge regression on targets that are derived from the question features",
          {"r2_is_high_by_construction": True}, ["R-squared is high because targets are computed from the inputs"]),
        m("bkt", "Bayesian Knowledge Tracing", "chapter mastery and forgetting", "hand_designed_prior",
          "hand-set guess, slip, transit and half-life constants per tier; p_init from a hand-written mixture clipped to 0.22-0.54",
          {"accuracy": None}, ["real EM trainer is built but skipped until 200 students exist"]),
        m("dina", "DINA skill diagnosis", "8 thinking skills per student", "hand_designed_prior",
          "regex keyword rules decide which skills a question needs (unvalidated Q-matrix)", {"accuracy": None},
          ["excluded from real training because the Q-matrix is not validated"]),
        m("linucb", "LinUCB question picker", "chooses the next questions", "design_objective_simulated",
          "reward is a design objective (0.65 ZPD alignment + 0.35 Fisher information); warm start from a synthetic reward",
          {"simulated_regret_reduction_percent": cold.get("regret_reduction_vs_random_percent"), "reward_noise_sd": cold.get("reward_noise_sd")},
          ["regret is measured on the simulated reward, not on student learning", "A/B test against random needs 100 students per arm"]),
        m("difficulty_rf", "Difficulty random forest (production)", "PYQ tier", "fitted_on_formula_derived_labels",
          "trained to reproduce formula tiers", {"holdout_accuracy": _get(valid, "random_forest_holdout_evaluation", "holdout_accuracy"), "kappa_vs_human_labels": kappa},
          ["accuracy measures how well the formula is re-learned, not real difficulty"]),
        m("difficulty_v21", "Difficulty v2.1 (candidate)", "tier from question text", "fitted_on_real_human_labels",
          "TF-IDF + logistic regression on content-team tiers, 5-fold split by paper",
          {"accuracy": _get(v21, "v21_tfidf_logreg", "accuracy"), "macro_f1": _get(v21, "v21_tfidf_logreg", "macro_f1"), "ece": _get(v21, "v21_tfidf_logreg", "ece_top_label"),
           "majority_baseline": v21.get("majority_baseline"), "rows": v21.get("n_rows"), "chapters_not_beating_majority": len(below),
           "promotion_decision": _get(promo, "difficulty_v21_text", "decision")},
          ["trained on sample papers; domain shift to PYQs is unmeasured", "how the human tiers were produced is not visible to the ML side", "not promoted: needs the blind human review"], real_labels=True),
        m("temperature", "Temperature calibration", "calibrate the forest's confidence", "fitted_on_formula_derived_labels",
          "fitted then rejected on the untouched holdout", {"decision": _get(valid, "temperature_calibration_summary", "temperature_decision")}, ["not in use"]),
        m("ncert_retriever", "NCERT retriever (TF-IDF + SVD)", "finds the right NCERT chapter", "fitted_on_real_human_labels",
          "TF-IDF and 32-dimensional SVD over NCERT notes and PYQs",
          {"top1": _get(ncert, "heldout_chapter_retrieval", "top1_chapter_accuracy"), "top3": _get(ncert, "heldout_chapter_retrieval", "top3_chapter_accuracy"), "chance_top1": round(1 / 65, 4), "chance_top3": round(3 / 65, 4)},
          ["measures chapter retrieval, not tutoring quality", "semantic challenger is on HOLD"], real_labels=True),
        m("chapter_resolver", "Chapter resolver", "fixes chapter labels at ingestion", "fitted_on_real_human_labels",
          "TF-IDF + logistic regression; acts only at >= 90% held-out precision",
          {s: {"top1": v.get("holdout_top1_accuracy"), "coverage_at_threshold": v.get("holdout_coverage_at_threshold"), "precision_at_threshold": v.get("holdout_precision_at_threshold")} for s, v in res.items()},
          ["low coverage: most generic labels stay quarantined"], real_labels=True),
        m("near_duplicate_guard", "Near-duplicate guard", "prevents repeated questions in mocks", "rule_based", "TF-IDF cosine with a chosen threshold", {"accuracy": None}, ["no labeled accuracy measured"]),
        m("score_simulator", "Monte Carlo score simulator", "NEET score range", "math_on_priors", "posterior predictive on the equated reference form", {"validated_against_real_scores": False}, ["no real NEET scores exist to validate against"]),
        m("rank_curve", "Score-to-percentile-to-AIR curve", "rank from marks", "hand_designed_prior",
          "hand-set anchors; only candidate counts and cut-offs are sourced", {"anchors_status": FACTS_PROVENANCE["score_percentile_anchors"], "facts_verification": sorted({v["verification"] for v in FACTS_PROVENANCE["years"].values()})},
          ["NTA publishes no full score-percentile table", "facts were not re-verified in the audit"]),
        m("equated_forms", "Equated 180-question forms", "equal-difficulty papers", "math_on_priors", "item swaps to match information curves",
          {"max_expected_marks_gap": equiv.get("max_expected_marks_gap_across_theta"), "tolerance": _get(equiv, "tolerances", "expected_marks_gap")}, ["equality holds under the model's own parameters, not proven on students"]),
        m("study_plan", "Study plan, importance, leverage, drift limits", "chapter priorities and monitoring", "rule_based", "rules, heuristic weights and chosen thresholds", {"accuracy": None}, ["no outcome data to validate against"]),
        m("student_training", "Student-response training (IRT MML-EM, BKT EM)", "fit real parameters from answers", "built_waiting_for_data",
          "needs 200 students and 20,000 answers", {"status": sr_status, "real_attempts": attempts, "simulated_self_test": selftest or None}, ["self-test uses simulated students only"]),
        m("air_estimator", "AIR estimator", "post-attempt rank range", "math_on_priors", "composes the items above with the sourced cut-offs",
          {"simulated_interval_95_coverage": air_self.get("interval_95_coverage")}, ["calibration is on simulated students", "range widens when label and year assumptions are varied"]),
        m("text_quality_gate", "Text-quality gate", "rejects OCR-corrupt rows", "rule_based", "character ranges and exact duplicate options", {"accuracy": None}, ["not measured against labeled corruption"]),
    ]
    return {
        "real_student_attempts": attempts,
        "student_training_status": sr_status,
        "formula_vs_human_label_kappa": kappa,
        "models": models,
    }


def check_consistency(prov: dict) -> list[str]:
    problems = []
    attempts = prov["real_student_attempts"]
    for mdl in prov["models"]:
        if mdl["basis"] not in BASES:
            problems.append(f"{mdl['id']}: unknown basis {mdl['basis']}")
        if not mdl["limitations"]:
            problems.append(f"{mdl['id']}: no limitations stated")
        if mdl["basis"] == "fitted_on_real_students" and attempts == 0:
            problems.append(f"{mdl['id']}: claims real-student fitting with 0 attempts")
        if mdl["basis"] == "hand_designed_prior" and any(k.endswith("accuracy") and v is not None for k, v in mdl["evidence"].items()):
            problems.append(f"{mdl['id']}: a prior must not carry an accuracy")
        if mdl["basis"] in CLAIMS_ACCURACY and not any(isinstance(v, (int, float, dict, str, bool)) and v is not None for v in mdl["evidence"].values()):
            problems.append(f"{mdl['id']}: fitted model without evidence")
        if mdl["validated_on_real_students"] and attempts == 0:
            problems.append(f"{mdl['id']}: validated on real students with 0 attempts")
        if mdl["basis"] == "fitted_on_real_human_labels" and not mdl["uses_real_human_labels"]:
            problems.append(f"{mdl['id']}: basis and uses_real_human_labels disagree")
    ids = [m["id"] for m in prov["models"]]
    if len(ids) != len(set(ids)):
        problems.append("duplicate model ids")
    return problems


@lru_cache(maxsize=1)
def reliability_summary() -> dict:
    from .national_rank_simulator import FACTS_PROVENANCE

    prov = build_provenance()
    attempts = prov["real_student_attempts"]
    unverified = sorted({v["verification"] for v in FACTS_PROVENANCE["years"].values()} - {"verified"})
    return {
        "basis": "prior_based_no_real_student_calibration" if attempts == 0 else "partly_calibrated_on_real_students",
        "real_student_attempts_used_for_calibration": attempts,
        "item_difficulty_labels": f"PYQ item difficulty is formula-derived (kappa {prov['formula_vs_human_label_kappa']} against human labels)",
        "score_to_rank_curve": FACTS_PROVENANCE["score_percentile_anchors"],
        "neet_facts_verification": unverified or ["verified"],
        "models_by_basis": {b: sum(1 for m in prov["models"] if m["basis"] == b) for b in BASES if any(m["basis"] == b for m in prov["models"])},
    }


def run(save: bool = True) -> dict:
    prov = build_provenance()
    prov["problems"] = check_consistency(prov)
    prov["passed"] = not prov["problems"]
    if save:
        REPORT_PATH.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return prov


if __name__ == "__main__":
    r = run()
    print(json.dumps({"models": len(r["models"]), "attempts": r["real_student_attempts"], "problems": r["problems"], "passed": r["passed"]}, indent=2))
