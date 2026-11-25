# Post-attempt AIR estimate

Code: `model/air_estimator.py` (`build_air_report`). API: `GET /students/{student_id}/air_estimate[?paper_number=N]` (auth required).
Sample output (simulated student): `docs/air_estimate_sample_simulated.json`.

## What goes in
Responses of one attempt (`question_id`, `is_correct`, `time_taken_seconds`) and optionally earlier responses as `history`.
The endpoint reads `TestAttempt`: the requested paper (default: the latest) is the attempt, all other papers are history.
Duplicates collapse (last answer per question wins); invalid values are dropped and counted in `data_hygiene`.

## How each parameter is used
| Parameter | How |
|---|---|
| Mock difficulty | IRT 3PL equating: ability is estimated from which questions were answered, so a harder paper does not lower the estimate. `mock_difficulty` reports the paper against the equated reference form (tolerance 6 marks). |
| Chapter-wise performance | Chapter mastery (IRT + BKT with hierarchical shrinkage) from the attempt. Chapters with no answers are marked `prior_only`. |
| Previous-year trends | PYQ chapter weightage 2013 to 2026: mean share, last-3-year share, rising/falling/stable. `expected_questions_per_neet_paper` = 180 x mean share. `trend_watchlist` = rising chapters where mastery is below target. |
| Chapter difficulty | Mean IRT b of the chapter's PYQs, advanced-tier share, mean ideal time. |
| Real NEET facts | Only the sourced facts in `NEET_YEAR_FACTS` (2024 to 2026 candidates appeared and general cut-off). `paper_year_scenarios` re-reads the same marks on the curve shifted by the difference in cut-off scores. |
| AIR | Equated marks to percentile to rank. `range_90` comes from the 90% score interval; `paper_year_spread` from the year scenarios. |
| Where to focus | `marks_at_stake_if_target_reached` = expected questions x 5 x (target mastery - mastery); `air_places_gained_if_target_reached` is the rank gain if only that chapter reached the target. Values are not additive. |

## When AIR is not given
AIR needs 10 scored questions in each of Physics, Chemistry and Biology. A single-chapter or partial test returns `status: insufficient_evidence_for_air`, `air: null`, the chapter-wise table, and a `reason`.
If earlier attempts together satisfy the rule, `evidence.source` is `attempt_plus_history` and AIR is returned.
No scored answers: `status: no_scored_responses`.

## Limits (also in `assumptions` in every response)
- The score-to-percentile curve is hand-set and approximate; NTA publishes no full table. Treat AIR as an indicative range.
- No real student attempts exist, so item parameters are priors, not fitted values.
- Papers far easier or harder than NEET carry little information: equating narrows but does not remove the gap and the AIR range widens. Measured on simulated students: for the same ability a deliberately hard and a deliberately easy paper gave raw marks -52 and 708 and equated marks 269 and 428, against about 412 on NEET-style forms.
- Target mastery (0.85) and the 10-question minimum are scenario/policy values, not measurements.
- Thermodynamics is one chapter key for Physics and Chemistry (known issue).
- The PYQ bank has fewer questions for 2025 and 2026 than other years, which affects trend flags.

## Verified
Simulated calibration over 480 students: mean ability error near 0, RMSE close to the stated standard error, 92 to 99% coverage of the 95% interval.
Suite `ml_suite_43`, backend `test_31`, and pipeline stage `air_estimator_self_test` cover monotonicity, equating, year scenarios, chapter table, partial evidence, history, duplicates and the HTTP contract.
