import json
import tempfile
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODEL_DIR.parent
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"


def _pyq():
    return json.loads(PYQ_PATH.read_text(encoding="utf-8"))


def ml_suite_40_text_quality_gate_ingestion_rejection_and_form_invariants():
    from model.assessment_blueprint import load_question_bank
    from model.chapter_resolver import ChapterResolver
    from model.sample_paper_ingestion import ingest
    from model.text_quality_audit import run as audit_run
    from model.text_quality_gate import (
        FLAG_DUPLICATE_OPTIONS,
        FLAG_FOREIGN_OPTION,
        FLAG_FOREIGN_QUESTION,
        FLAG_GARBLED_OPTION,
        FLAG_GARBLED_QUESTION,
        is_text_corrupt,
        offending_snippet,
        text_quality_flags,
    )

    clean = {"question_text": "Find the value of the constant", "option_a": "3.14", "option_b": "α < β < γ", "option_c": "π", "option_d": "CO", "subject": "Physics"}
    assert text_quality_flags(clean) == [] and not is_text_corrupt(clean) and offending_snippet(clean) == ""
    assert text_quality_flags(dict(clean, option_a="CO", option_d="Co", option_b="Ca", option_c="Cu")) == []
    assert text_quality_flags(dict(clean, option_a="H2O", option_b="h2o")) == []
    assert FLAG_FOREIGN_OPTION in text_quality_flags(dict(clean, option_c="6元RaT"))
    assert FLAG_FOREIGN_QUESTION in text_quality_flags(dict(clean, question_text="4元0a0m nucleus"))
    assert FLAG_GARBLED_QUESTION in text_quality_flags(dict(clean, question_text="A įystem of rotating crops"))
    assert FLAG_GARBLED_OPTION in text_quality_flags(dict(clean, option_b="Ċ"))
    assert FLAG_DUPLICATE_OPTIONS in text_quality_flags(dict(clean, option_a=",", option_b=",", option_c=",", option_d="and"))
    assert FLAG_DUPLICATE_OPTIONS in text_quality_flags(dict(clean, option_a="x  y", option_b="x y"))
    assert offending_snippet(dict(clean, option_c="6元RaT")) != ""

    pyq = _pyq()
    resolver = ChapterResolver(pyq)
    resolver.fit([])
    rows = []
    for i, r in enumerate(pyq[:12]):
        x = dict(r)
        x["question_id"] = f"TQ_{i:03d}"
        x["paper_source"] = "Sample Paper 1"
        x["question_text"] = r["question_text"] + f" [text quality variant {i}]"
        x["has_diagram"] = False
        x["image_asset"] = ""
        rows.append(x)
    corrupt = [
        dict(rows[0], question_id="TQ_BAD_CJK", option_c="6元RaT", question_text="cjk option variant"),
        dict(rows[1], question_id="TQ_BAD_LATX", question_text="A įystem of rotating crops variant"),
        dict(rows[2], question_id="TQ_BAD_DUP", option_a=",", option_b=",", option_c=",", option_d="and", question_text="duplicate option variant"),
    ]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        src = tdp / "samples.json"
        src.write_text(json.dumps(rows + corrupt), encoding="utf-8")
        rep = ingest(
            sample_path=src,
            params_path=tdp / "p.json",
            report_path=tdp / "r.json",
            rejected_path=tdp / "x.json",
            assets_root=tdp,
            resolver=resolver,
        )
        assert rep["status"] == "ingested"
        assert rep["accepted"] == 12 and rep["rejected"] == 3, rep["rejection_reasons"]
        reasons = rep["rejection_reasons"]
        assert reasons.get(FLAG_FOREIGN_OPTION, 0) >= 1
        assert reasons.get(FLAG_GARBLED_QUESTION, 0) >= 1
        assert reasons.get(FLAG_DUPLICATE_OPTIONS, 0) >= 1

    rep = audit_run(save=False)
    assert rep["invariants"]["accepted_set_has_no_flagged_rows"], rep["sample_accepted"]
    assert rep["invariants"]["assessment_forms_have_no_flagged_questions"], rep["assessment_forms_with_flagged_questions"]
    assert rep["passed"]
    flagged_pyq = set(rep["pyq"]["flagged_ids"])
    served = {q["question_id"] for q in load_question_bank()}
    assert not (served & flagged_pyq)
    assert len(served) == len(pyq) - len(flagged_pyq)


def ml_suite_41_chapter_topic_statistics_coverage_ratchet_and_weightage():
    from model.chapter_topic_evaluation import (
        KNOWN_ISSUES_PATH,
        REPORT_PATH,
        chapter_tier_coverage,
        group_metrics,
        multi_subject_chapters,
        pyq_zero_tier_keys,
        ratchet,
        wilson,
    )
    from model.pyq_weightage import exam_year, pyq_chapter_weightage
    from model.sample_paper_ingestion import load_accepted_sample_rows

    lo, hi = wilson(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97
    assert wilson(0, 0) == (0.0, 0.0)

    y = np.array(["basic"] * 5 + ["medium"] * 40)
    small_big = {g["group"]: g for g in group_metrics(y, y, ["A"] * 5 + ["B"] * 40, 30)}
    assert small_big["A"]["reliability"] == "insufficient_support" and small_big["B"]["reliability"] == "ok"
    y2 = np.array(["medium"] * 30 + ["basic"] * 10)
    g = group_metrics(y2, np.array(["medium"] * 40), ["X"] * 40, 30)[0]
    assert g["accuracy"] == g["majority_baseline"] == 0.75 and g["beats_majority"] is False
    assert g["accuracy_ci95"][0] <= g["accuracy"] <= g["accuracy_ci95"][1]

    pyq_rows = [{"chapter": "C", "subject": "Physics", "difficulty_level": "medium"}]
    sample_rows = [{"chapter": "C", "subject": "Physics", "difficulty_level": "basic"}]
    row = chapter_tier_coverage(pyq_rows, sample_rows)[0]
    assert row["zero_tiers_pyq_only"] == "basic,advanced" and row["zero_tiers_pyq_plus_sample"] == "advanced"

    known = {"pyq_zero_tier_chapters": ["P::A::basic"], "chapters_in_multiple_subjects": ["T"]}
    assert ratchet(["P::A::basic"], ["T"], known)["passed"]
    worse = ratchet(["P::A::basic", "P::B::medium"], ["T"], known)
    assert not worse["passed"] and worse["new_zero_tiers"] == ["P::B::medium"]
    better = ratchet([], ["T"], known)
    assert better["passed"] and better["resolved_zero_tiers"] == ["P::A::basic"]
    assert not ratchet(["P::A::basic"], ["T", "U"], known)["passed"]

    assert exam_year({"paper_source": "NEET 2023"}) == 2023
    assert exam_year({"paper_source": "RE-NEET 2026"}) == 2026
    assert exam_year({"paper_source": "Sample Paper 5"}) is None

    pyq = _pyq()
    accepted, _ = load_accepted_sample_rows()
    weightage, meta = pyq_chapter_weightage(pyq)
    assert meta["rows_without_year"] == 0
    assert meta["years"] == list(range(2013, 2027))
    assert sum(meta["questions_per_year"].values()) == len(pyq)
    assert set(meta["short_years"]) <= set(meta["years"])
    assert abs(sum(w["mean_share"] for w in weightage) - 1.0) < 0.01
    assert {w["trend"] for w in weightage} <= {"rising", "falling", "stable"}

    known_real = json.loads(KNOWN_ISSUES_PATH.read_text(encoding="utf-8"))
    gate = ratchet(pyq_zero_tier_keys(pyq), list(multi_subject_chapters(pyq, accepted)), known_real)
    assert gate["passed"], gate
    assert multi_subject_chapters(pyq, accepted).get("Thermodynamics") == ["Chemistry", "Physics"]

    assert REPORT_PATH.exists(), "run python -m model.chapter_topic_evaluation or the full pipeline first"
    rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert "GroupKFold" in rep["evaluation"] and "not_measured" in rep
    assert 0.0 < rep["overall_accuracy"] < 1.0 and rep["coverage"]["ratchet"]["passed"]
    for c in rep["by_chapter"]:
        assert c["reliability"] in ("ok", "insufficient_support")
        assert c["accuracy_ci95"][0] <= c["accuracy"] <= c["accuracy_ci95"][1]


def ml_suite_42_response_hygiene_and_attempt_to_chapter_to_rank_path():
    from model.assessment_blueprint import score_initial_assessment
    from model.national_rank_simulator import score_to_all_india_rank
    from model.response_hygiene import clean_responses, clean_time, parse_correct
    from model.student_profile_initializer import _bank_by_id, _p_correct, initialize_student_profile

    assert [parse_correct(v) for v in (True, False, 1, 0, "true", "FALSE", " 1 ", "no", None)] == [True, False, True, False, True, False, True, False, None]
    assert [parse_correct(v) for v in ("maybe", 2, -1, float("nan"), [], {})] == [None] * 6
    assert clean_time(-5) is None and clean_time(0) is None and clean_time(float("inf")) is None and clean_time("x") is None
    assert clean_time("12.5") == 12.5

    dup = [{"question_id": "Q1", "is_correct": False}, {"question_id": "Q1", "is_correct": True}, {"question_id": "Q2", "is_correct": "maybe"}, {"is_correct": True}]
    cleaned, hygiene = clean_responses(dup)
    assert [(c["question_id"], c["is_correct"]) for c in cleaned] == [("Q1", True)]
    assert hygiene["duplicate_responses_dropped"] == 1 and hygiene["invalid_is_correct_dropped"] == 1 and hygiene["responses_without_question_id"] == 1
    assert clean_responses([]) [0] == [] and clean_responses(None)[0] == []

    bank = _bank_by_id("unified")
    forms = json.loads((MODEL_DIR / "assessment_forms.json").read_text(encoding="utf-8"))["forms"]
    form = forms[0]["question_ids"]

    def answers(theta: float, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        return [
            {"question_id": q, "is_correct": bool(rng.random() < float(_p_correct(np.array(theta), bank[q]))), "time_taken_seconds": float(bank[q]["ideal_time_sec"])}
            for q in form
        ]

    base = answers(0.5, 11)
    once = initialize_student_profile(base)
    twice = initialize_student_profile(base + base)
    assert once["marks"] == twice["marks"] and once["global_theta"] == twice["global_theta"]
    assert twice["data_hygiene"]["duplicate_responses_dropped"] == len(form)
    assert once["marks"] <= 720 and twice["marks"] <= 720
    assert score_initial_assessment(base)["marks"] == score_initial_assessment(base + base)["marks"]

    only = form[0]
    assert initialize_student_profile([{"question_id": only, "is_correct": "false", "time_taken_seconds": 10}])["marks"] == -1
    assert initialize_student_profile([{"question_id": only, "is_correct": False}, {"question_id": only, "is_correct": True}])["marks"] == 4
    empty = initialize_student_profile([])
    assert empty["prior_only"] is True and empty["responses_scored"] == 0 and once["prior_only"] is False
    odd_times = initialize_student_profile([{"question_id": form[0], "is_correct": True, "time_taken_seconds": -5}])
    assert all(t["avg_time_ratio"] is None for t in odd_times["topic_states"])

    previous_score, previous_rank = -1e9, 10 ** 9
    for theta in (-1.0, 0.0, 1.0, 2.0):
        out = initialize_student_profile(answers(theta, 1000 + int(theta * 10)))
        assert abs(out["global_theta"] - theta) < 0.35, (theta, out["global_theta"])
        score = out["predicted_neet_score"]["expected_marks"]
        rank = score_to_all_india_rank(score)
        assert score > previous_score and rank < previous_rank
        assert out["priority_chapters"] and all(0.0 <= s["mastery_prob"] <= 1.0 for s in out["topic_states"])
        previous_score, previous_rank = score, rank

    ranks = [score_to_all_india_rank(s) for s in range(0, 721, 20)]
    assert all(a >= b for a, b in zip(ranks, ranks[1:]))
    assert score_to_all_india_rank(720) == 1 and score_to_all_india_rank(0) > 1_500_000


def ml_suite_43_post_attempt_air_estimate_difficulty_trends_chapters_and_evidence():
    import json as _json

    from model.air_estimator import (
        FULL_MOCK_MIN_QUESTIONS,
        MIN_SCORED_PER_SUBJECT,
        NEET_QUESTIONS,
        TARGET_MASTERY,
        build_air_report,
    )
    from model.national_rank_simulator import NEET_YEAR_FACTS
    from model.student_profile_initializer import _bank_by_id, _p_correct

    bank = _bank_by_id("unified")
    forms = _json.loads((MODEL_DIR / "assessment_forms.json").read_text(encoding="utf-8"))["forms"]

    def answer(qids, theta, seed):
        rng = np.random.default_rng(seed)
        return [
            {"question_id": q, "is_correct": bool(rng.random() < float(_p_correct(np.array(theta), bank[q]))), "time_taken_seconds": float(bank[q]["ideal_time_sec"])}
            for q in qids
        ]

    def extreme_paper(hard):
        ids = []
        for subject, n in (("Physics", 45), ("Chemistry", 45), ("Biology", 90)):
            pool = sorted((q for q in bank.values() if q["subject"] == subject and q["source_type"] == "pyq"), key=lambda q: q["b"], reverse=hard)
            ids += [q["question_id"] for q in pool[:n]]
        return ids

    form0, form1 = forms[0]["question_ids"], forms[1]["question_ids"]
    rep = build_air_report(answer(form0, 0.5, 3))
    assert rep["status"] == "ok" and rep["evidence"]["scope"] == "full_mock" and rep["evidence"]["sufficient_for_air"]
    assert _json.loads(_json.dumps(rep)) == rep
    air = rep["air"]
    assert 1 <= air["estimate"] <= 2_000_000 and air["range_90"][0] <= air["range_90"][1]
    assert air["curve_status"] == "approximate" and 0.0 < air["percentile"] < 100.0
    assert set(rep["assumptions"]) >= {"target_mastery", "score_to_percentile_curve", "paper_year_scenarios", "college_probabilities"}
    assert rep["scores"]["raw_marks_this_attempt"] <= 720 and rep["scores"]["theta_sem"] > 0

    previous_marks, previous_air = -1e9, 10 ** 9
    for theta in (-1.0, 0.0, 1.0, 2.0):
        r = build_air_report(answer(form0, theta, 500 + int(theta * 10)))
        assert r["scores"]["equated_neet_marks"] > previous_marks and r["air"]["estimate"] < previous_air
        previous_marks, previous_air = r["scores"]["equated_neet_marks"], r["air"]["estimate"]

    same = [np.mean([build_air_report(answer(f, 0.5, s))["scores"]["equated_neet_marks"] for s in range(8)]) for f in (form0, form1)]
    assert abs(same[0] - same[1]) < 20.0

    hard, easy = extreme_paper(True), extreme_paper(False)
    r_hard, r_easy = build_air_report(answer(hard, 0.5, 1)), build_air_report(answer(easy, 0.5, 1))
    raw_gap = r_easy["scores"]["raw_marks_this_attempt"] - r_hard["scores"]["raw_marks_this_attempt"]
    equated_gap = r_easy["scores"]["equated_neet_marks"] - r_hard["scores"]["equated_neet_marks"]
    assert raw_gap > 400 and 0 <= equated_gap < 0.5 * raw_gap
    assert r_hard["mock_difficulty"]["verdict"].startswith("harder") and r_easy["mock_difficulty"]["verdict"].startswith("easier")
    assert rep["mock_difficulty"]["verdict"].startswith("similar")
    width = lambda x: x["air"]["range_90"][1] - x["air"]["range_90"][0]
    assert width(r_hard) > width(rep) and width(r_easy) > width(rep)

    scen = rep["air"]["paper_year_scenarios"]
    assert set(scen) == {str(y) for y in NEET_YEAR_FACTS}
    by_cutoff = sorted(scen.values(), key=lambda v: v["general_cutoff"])
    assert all(a["percentile"] >= b["percentile"] for a, b in zip(by_cutoff, by_cutoff[1:]))
    assert rep["air"]["paper_year_spread"] == [min(v["air"] for v in scen.values()), max(v["air"] for v in scen.values())]

    probs = [c["probability_percent"] for c in sorted(rep["college_reach"], key=lambda c: -c["target_score_720"])]
    assert all(a <= b + 1e-9 for a, b in zip(probs, probs[1:]))
    assert rep["college_reach"][-1]["status"] == "SECURE" and rep["college_reach"][-1]["threshold_status"] == "sourced"

    chapters = rep["chapter_wise"]
    assert abs(sum(c["expected_questions_per_neet_paper"] for c in chapters) - NEET_QUESTIONS) < 2.0
    gains = [c["marks_at_stake_if_target_reached"] for c in chapters]
    assert gains == sorted(gains, reverse=True) and min(gains) >= 0.0
    assert rep["focus_chapters"] == chapters[:len(rep["focus_chapters"])]
    assert all(c["air_places_gained_if_target_reached"] >= 0 for c in chapters)
    assert all(c["marks_at_stake_if_target_reached"] == 0.0 for c in chapters if c["mastery"] >= TARGET_MASTERY)
    assert all(c["pyq_trend"] in ("rising", "falling", "stable", None) for c in chapters)
    lenient = build_air_report(answer(form0, 0.5, 3), target_mastery=0.5)
    assert sum(c["marks_at_stake_if_target_reached"] for c in lenient["chapter_wise"]) <= sum(gains)
    assert all(w["mastery"] < TARGET_MASTERY for w in rep["trend_watchlist"])
    assert sum(r["attempted"] for r in rep["subject_wise"].values()) == len(form0)

    subject_ids = {}
    for qid in form0:
        subject_ids.setdefault(bank[qid]["chapter"], []).append(qid)
    one_chapter = max(subject_ids.values(), key=len)
    partial = build_air_report(answer(one_chapter, 0.5, 4))
    assert partial["status"] == "insufficient_evidence_for_air" and partial["air"] is None and partial["reason"]
    assert partial["evidence"]["scope"] == "single_chapter" and partial["college_reach"] is None
    assert partial["chapter_wise"] and sum(c["attempted_in_this_attempt"] for c in partial["chapter_wise"]) == len(one_chapter)
    assert all(c["air_places_gained_if_target_reached"] is None for c in partial["chapter_wise"])
    with_history = build_air_report(answer(one_chapter, 0.5, 4), history=answer(form0, 0.5, 5))
    assert with_history["status"] == "ok" and with_history["evidence"]["source"] == "attempt_plus_history"

    assert build_air_report([])["status"] == "no_scored_responses"
    assert build_air_report([{"question_id": "NOPE", "is_correct": True}])["status"] == "no_scored_responses"
    base = answer(form0, 0.5, 3)
    twice = build_air_report(base + base)
    assert twice["scores"] == rep["scores"] and twice["air"]["estimate"] == rep["air"]["estimate"]
    assert twice["data_hygiene"]["duplicate_responses_dropped"] == len(form0)
    assert FULL_MOCK_MIN_QUESTIONS <= NEET_QUESTIONS and MIN_SCORED_PER_SUBJECT >= 1


def ml_suite_44_syllabus_modes_weightage_sources_forms_freeze_and_cross_dataset_gate():
    import ast
    import json as _json

    from model.air_estimator import NEET_QUESTIONS, build_air_report
    from model.assessment_blueprint import existing_attempt_rows, main as build_forms
    from model.chapter_topic_evaluation import KNOWN_ISSUES_PATH, TAXONOMY_PATH, cross_dataset_gate
    from model.pyq_weightage import cross_check_with_bank, taxonomy_chapter_weightage
    from model.student_profile_initializer import _bank_by_id, _p_correct
    from model.syllabus import NMC_REDUCED_SYLLABUS_EXCLUSIONS, SYLLABUS_MODES, in_syllabus, normalise_syllabus_mode

    candidates = [PROJECT_ROOT / "backend" / "app" / "paper_generator.py", PROJECT_ROOT.parent / "backend" / "app" / "paper_generator.py"]
    source = next((c.read_text(encoding="utf-8") for c in candidates if c.exists()), "")
    backend_set = None
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "NMC_REDUCED_SYLLABUS_EXCLUSIONS" for t in node.targets):
            backend_set = {e.value for e in node.value.elts}
    if backend_set is not None:
        assert backend_set == set(NMC_REDUCED_SYLLABUS_EXCLUSIONS)

    assert normalise_syllabus_mode(None) == "all" and normalise_syllabus_mode(" NMC_Reduced ") == "nmc_reduced"
    assert SYLLABUS_MODES == ("all", "nmc_reduced")
    assert normalise_syllabus_mode("") == "all"
    for bad in ("reduced", "full", "nmc"):
        try:
            normalise_syllabus_mode(bad)
            raise AssertionError("invalid mode accepted")
        except ValueError:
            pass
    assert in_syllabus("Kinematics", "nmc_reduced") and not in_syllabus("Surface Chemistry", "nmc_reduced") and in_syllabus("Surface Chemistry", "all")

    bank = _bank_by_id("unified")
    form = _json.loads((MODEL_DIR / "assessment_forms.json").read_text(encoding="utf-8"))["forms"][0]["question_ids"]
    rng = np.random.default_rng(44)
    answers = [{"question_id": q, "is_correct": bool(rng.random() < float(_p_correct(np.array(0.4), bank[q]))), "time_taken_seconds": 60.0} for q in form]
    full, reduced = build_air_report(answers), build_air_report(answers, syllabus_mode="nmc_reduced")
    assert full["syllabus_mode"] == "all" and reduced["syllabus_mode"] == "nmc_reduced"
    assert full["scores"] == reduced["scores"] and full["air"]["estimate"] == reduced["air"]["estimate"]
    for rep in (full, reduced):
        assert abs(sum(c["expected_questions_per_neet_paper"] for c in rep["chapter_wise"]) - NEET_QUESTIONS) < 2.0
    excluded = [c for c in reduced["chapter_wise"] if c["chapter"] in NMC_REDUCED_SYLLABUS_EXCLUSIONS]
    assert excluded and all(c["expected_questions_per_neet_paper"] == 0.0 and c["marks_at_stake_if_target_reached"] == 0.0 and not c["in_syllabus"] for c in excluded)
    assert all(c["nmc_reduced_excluded"] for c in excluded) and all(c["in_syllabus"] for c in full["chapter_wise"])
    assert sum(c["expected_questions_per_neet_paper"] for c in full["chapter_wise"] if c["chapter"] in NMC_REDUCED_SYLLABUS_EXCLUSIONS) > 0
    assert not {w["chapter"] for w in reduced["trend_watchlist"]} & set(NMC_REDUCED_SYLLABUS_EXCLUSIONS)
    included_full = {c["chapter"]: c["expected_questions_per_neet_paper"] for c in full["chapter_wise"] if c["in_syllabus"]}
    assert all(c["expected_questions_per_neet_paper"] >= included_full[c["chapter"]] for c in reduced["chapter_wise"] if c["in_syllabus"])
    try:
        build_air_report(answers, syllabus_mode="bogus")
        raise AssertionError("invalid syllabus_mode accepted")
    except ValueError:
        pass

    src = full["assumptions"]["weightage_source"]
    assert src["source"] == "taxonomy_with_bank_fallback" and src["chapters_from_taxonomy"] >= 60
    assert set(src["chapters_from_bank_fallback"]) <= set(NMC_REDUCED_SYLLABUS_EXCLUSIONS)

    taxonomy_rows, taxonomy_meta = taxonomy_chapter_weightage(_json.loads(TAXONOMY_PATH.read_text(encoding="utf-8")))
    assert taxonomy_meta["years"] == list(range(2013, 2027))
    assert abs(sum(v["mean_share"] for v in taxonomy_rows.values()) - 1.0) < 1e-3
    cross = cross_check_with_bank(taxonomy_rows, taxonomy_meta, _pyq())
    known = _json.loads(KNOWN_ISSUES_PATH.read_text(encoding="utf-8"))
    assert cross_dataset_gate(cross, known)["passed"], cross
    assert not cross_dataset_gate({**cross, "bank_missing_vs_taxonomy": cross["bank_missing_vs_taxonomy"] + 1}, known)["passed"]
    assert not cross_dataset_gate({**cross, "chapter_share_spearman": 0.5}, known)["passed"]
    assert cross["bank_missing_vs_taxonomy"] == cross["taxonomy_questions"] - cross["bank_questions"] > 0

    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "attempts.csv"
        empty.write_text("student_id,question_id,is_correct\n", encoding="utf-8")
        assert existing_attempt_rows(empty) == 0 and existing_attempt_rows(Path(td) / "missing.csv") == 0
        used = Path(td) / "used.csv"
        used.write_text("student_id,question_id,is_correct\ns1,q1,1\ns2,q2,0\n", encoding="utf-8")
        assert existing_attempt_rows(used) == 2
        try:
            build_forms([], attempts_path=used)
            raise AssertionError("forms rebuilt although student attempts exist")
        except SystemExit as exc:
            assert "refusing to rebuild" in str(exc)


def ml_suite_45_provenance_facts_file_label_sensitivity_and_honest_ranges():
    import copy
    import json as _json

    from model.air_estimator import LABEL_SCENARIOS, _round_significant, _scenario_items, build_air_report
    from model.model_provenance import BASES, build_provenance, check_consistency, real_student_attempts, reliability_summary
    from model.national_rank_simulator import COLLEGE_TIERS, FACTS_PROVENANCE, NEET_YEAR_FACTS, PERCENTILE_ANCHORS, SCORE_ANCHORS
    from model.student_profile_initializer import _bank_by_id, _p_correct

    prov = build_provenance()
    assert check_consistency(prov) == [] and prov["real_student_attempts"] == real_student_attempts()
    by_id = {m["id"]: m for m in prov["models"]}
    assert {"irt_items_pyq", "bkt", "dina", "linucb", "difficulty_rf", "difficulty_v21", "rank_curve", "student_training", "air_estimator"} <= set(by_id)
    assert by_id["irt_items_pyq"]["basis"] == "hand_designed_prior" and by_id["bkt"]["basis"] == "hand_designed_prior" and by_id["dina"]["basis"] == "hand_designed_prior"
    assert by_id["difficulty_v21"]["basis"] == "fitted_on_real_human_labels" and by_id["difficulty_v21"]["evidence"]["accuracy"] > by_id["difficulty_v21"]["evidence"]["majority_baseline"]
    assert by_id["difficulty_rf"]["basis"] == "fitted_on_formula_derived_labels" and by_id["linucb"]["basis"] == "design_objective_simulated"
    assert all(m["basis"] in BASES and m["limitations"] and m["validated_on_real_students"] is False for m in prov["models"])
    if prov["real_student_attempts"] == 0:
        assert by_id["student_training"]["basis"] == "built_waiting_for_data"
    broken = copy.deepcopy(prov)
    broken["models"][0]["basis"] = "fitted_on_real_students"
    assert any("0 attempts" in p for p in check_consistency(broken))
    broken = copy.deepcopy(prov)
    broken["models"][0]["evidence"] = {"accuracy": 0.9}
    assert any("prior must not carry an accuracy" in p for p in check_consistency(broken))
    broken = copy.deepcopy(prov)
    broken["models"][1]["basis"] = "made_up"
    broken["models"][2]["limitations"] = []
    assert len(check_consistency(broken)) >= 2

    facts = _json.loads((MODEL_DIR / "neet_reference_facts.json").read_text(encoding="utf-8"))
    assert all(f["source"] and f["verification"] for f in facts["years"].values())
    assert facts["score_percentile_anchors"]["status"] == "hand_set_approximate"
    assert list(SCORE_ANCHORS) == facts["score_percentile_anchors"]["score"] and list(PERCENTILE_ANCHORS) == facts["score_percentile_anchors"]["percentile"]
    assert all(a < b for a, b in zip(SCORE_ANCHORS, SCORE_ANCHORS[1:])) and all(a < b for a, b in zip(PERCENTILE_ANCHORS, PERCENTILE_ANCHORS[1:]))
    assert set(NEET_YEAR_FACTS) == {int(y) for y in facts["years"]} and set(FACTS_PROVENANCE["years"]) == set(NEET_YEAR_FACTS)
    assert COLLEGE_TIERS[-1]["id"] == "qualifying_floor" and COLLEGE_TIERS[-1]["score_threshold"] == NEET_YEAR_FACTS[facts["reference_exam_year"]]["general_cutoff_score"]
    assert all(a["score_threshold"] > b["score_threshold"] for a, b in zip(COLLEGE_TIERS, COLLEGE_TIERS[1:]))
    assert reliability_summary()["neet_facts_verification"] == ["not_verified_in_audit"] or "verified" in reliability_summary()["neet_facts_verification"]

    assert [_round_significant(v) for v in (129637, 866, 12, 0, 262430, 1999895)] == [130000, 870, 12, 0, 260000, 2000000]

    bank = _bank_by_id("unified")
    pyq_item = next(q for q in bank.values() if q["source_type"] == "pyq")
    sample_item = next(q for q in bank.values() if q["source_type"] == "sample_paper")
    for name in LABEL_SCENARIOS:
        out = _scenario_items([pyq_item, sample_item], name)
        assert out[1] == sample_item and out[0]["a"] == pyq_item["a"] and out[0]["c"] == pyq_item["c"]
    flat = _scenario_items([q for q in bank.values() if q["source_type"] == "pyq" and q["subject"] == "Physics"][:20], "labels_uninformative")
    assert len({round(q["b"], 9) for q in flat}) == 1

    form = _json.loads((MODEL_DIR / "assessment_forms.json").read_text(encoding="utf-8"))["forms"][0]["question_ids"]

    def answers(theta, seed, qids=form):
        rng = np.random.default_rng(seed)
        return [{"question_id": q, "is_correct": bool(rng.random() < float(_p_correct(np.array(theta), bank[q])))} for q in qids]

    previous = 10 ** 9
    for theta in (-0.5, 0.5, 1.5):
        rep = build_air_report(answers(theta, 450 + int(theta * 10)))
        air = rep["air"]
        assert set(rep["sensitivity"]["difficulty_labels"]) == set(LABEL_SCENARIOS)
        assert all(np.isfinite(v["delta_marks"]) for v in rep["sensitivity"]["difficulty_labels"].values())
        assert air["range_if_assumptions_wrong"][0] <= air["range_90"][0] and air["range_if_assumptions_wrong"][1] >= air["range_90"][1]
        assert abs(air["display"]["estimate"] - air["estimate"]) <= 0.05 * air["estimate"] and air["display"]["estimate"] < previous
        previous = air["display"]["estimate"]
        rel = rep["reliability"]
        assert rel["basis"] == "prior_based_no_real_student_calibration" or rel["real_student_attempts_used_for_calibration"] > 0
        assert "marks" in rel["statement"] and rel["direction_of_label_risk"] in ("all tested alternatives lower the estimate", "all tested alternatives raise the estimate", "mixed")
        assert _json.loads(_json.dumps(rep)) == rep
    chapter_ids = [q for q in form if bank[q]["chapter"] == bank[form[0]]["chapter"]][:5]
    partial = build_air_report(answers(0.5, 3, chapter_ids))
    assert partial["status"] == "insufficient_evidence_for_air" and partial["sensitivity"] is None and "No rank" in partial["reliability"]["statement"]
