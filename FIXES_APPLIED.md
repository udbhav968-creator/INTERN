# Klassed ML: honest-model fixes, AIR feature, final audit (Udbhav Yadav)

Built and verified on Python 3.13.14 with the pinned libraries. Extract over the root of the INTERN repo on a branch.
Open docs/KLASSED_ML_PROJECT_REPORT.html for the full report.

## Verified (final code)
- run_full_ml_pipeline --verify-reproducible : 18/18 stages PASSED, 7/7 metrics within 0.005
- model.tests.test_ml_standalone_suite : 45/45 PASSED     backend/tests/test_v2_engine.py : 31/31 PASSED
- verify_artifact_integrity VERIFIED (21 + 14), serving_runtime_parity --strict OK, evaluation_harness PASSED, pyflakes 0 warnings

## This round: making the system honest (what no code can fix is listed in the report, section 8-9)
- model/model_provenance.py + model/model_provenance_report.json + GET /mlops/model_provenance:
  every model states its real basis (fitted on real students / real human labels / formula-derived labels / hand-designed prior /
  simulated objective / rule-based / math on priors / waiting for data). Pipeline stage model_provenance_audit fails on any
  claim the data cannot support (e.g. real-student fitting with 0 attempts, an accuracy on a prior).
- model/neet_reference_facts.json: NEET facts, score-percentile anchors and college tiers moved out of code, each with source and
  verification status (all marked not_verified_in_audit). national_rank_simulator.py now loads them; values unchanged.
- AIR estimator v3: label-sensitivity block (v2.1 labels, no-information labels), range_if_assumptions_wrong (labels + 2024-2026 paper years),
  reliability block with plain statement, 2-significant-digit display values.
- Finding: re-estimating the same answers under either alternative difficulty-label assumption LOWERS equated marks by tens of marks,
  so the production estimate may be optimistic. This is surfaced in every response; the real fix needs human labels or student data.
- suite ml_suite_45, backend test_31 extended, scripts regenerate the report and the AIR validation numbers.

## Earlier in this audit (still included)
text-quality gate, response hygiene, syllabus modes, cross-dataset gate, forms freeze guard, chapter/topic evaluation,
GET /students/{id}/air_estimate, report scripts. See docs/AIR_ESTIMATE_FEATURE.md and the report.

## Regenerated artifacts (built on Linux; another OS gives identical metrics but different bytes)
sample_paper_irt_parameters.json, sample_papers_rejected.json, ingestion report, assessment_forms.json, registries, model cards, trained model files.
