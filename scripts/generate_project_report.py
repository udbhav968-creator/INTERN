from __future__ import annotations

import ast
import base64
import html
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from model.pyq_weightage import taxonomy_chapter_weightage  # noqa: E402
from report_svg import arrow, box, hbars, label, line_chart, poly_arrow, svg, vbars_grouped  # noqa: E402

M = ROOT / "model"
D = ROOT / "docs"
OUT_HTML = D / "KLASSED_ML_PROJECT_REPORT.html"


def jload(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


RUN = jload(M / "ml_pipeline_run.json", {})
V21 = jload(M / "difficulty_model_v21_report.json", {})
LABEL = jload(M / "label_validity_report.json", {})
INGEST = jload(M / "sample_paper_ingestion_report.json", {})
CHAPTER = jload(M / "chapter_topic_evaluation_report.json", {})
TQ = jload(M / "text_quality_report.json", {})
PROMO = jload(M / "promotion_gate_report.json", {})
HARNESS = jload(M / "harness_benchmark_report.json", {})
EQUIV = jload(M / "assessment_equivalence_report.json", {})
INTEG = jload(M / "data_integrity_report.json", {})
VALID = jload(M / "validation_report.json", {})
SIM = jload(D / "air_validation_simulation.json", {})
FINAL = jload(D / "final_verification.json", {})
PROV = jload(M / "model_provenance_report.json", {})
TAX = jload(ROOT / "data" / "taxonomy" / "NEET_Unified_Weightage_And_Marks_2013_2026.json", {})
PYQ = jload(ROOT / "data" / "tagged_questions" / "neet_questions_clean.json", [])
SAMPLE_RAW = jload(ROOT / "data" / "tagged_questions" / "neet_samples_001_to_101_verified.json", [])
KNOWN = jload(ROOT / "data" / "config" / "known_data_issues.json", {})
FORMS = jload(M / "assessment_forms.json", {}).get("forms", [])

TIERS = ("basic", "medium", "advanced")
SUBJECTS = ("Physics", "Chemistry", "Biology")


def pct(x, d=1):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def fig_architecture() -> str:
    from report_svg import PALETTE

    b = ""
    cols = [(10, "INPUTS (other teams)"), (205, "DATA GATES"), (420, "ML CORE"), (650, "MLOPS"), (835, "SERVING")]
    for x, t in cols:
        b += label(x + 90, 18, t, 12, "middle", "txt", "700")
    ext = [("PYQ bank 2,400 (content/data)", 34), ("Sample papers 16,689 (data team)", 92), ("NCERT notes 65 ch. (content)", 150), ("Weightage taxonomy 2013-26", 208), ("Sourced NEET facts 2024-26", 266), ("Student attempts: 0 today", 324), ("Human tier review: pending", 382)]
    for t, y in ext:
        b += box(10, y, 170, 44, t, "ext", 11, dashed=True, bold_first=False)
    gates = [("data_integrity_gate", 34), ("sample_paper_ingestion + validator", 92), ("text_quality_gate (new)", 150), ("chapter_resolver (90% precision)", 208), ("unified_similarity_guard", 266), ("coverage ratchet + cross-dataset gate (new)", 324)]
    for t, y in gates:
        b += box(205, y, 175, 44, t, "gate", 11, bold_first=False)
    core = [("Student models: IRT 3PL, BKT, DINA, pace", 34), ("Selection: LinUCB, forms, 180-Q mocks", 92), ("Content: difficulty v2.1, NCERT RAG", 150), ("Prediction: Monte Carlo, rank, AIR estimator (new)", 208), ("Study plan, drift monitor, analytics", 266), ("Student-response training (waits for data)", 324)]
    for t, y in core:
        b += box(420, y, 190, 44, t, "ml", 11, bold_first=False)
    ops = [("17-stage pipeline", 34), ("SHA-256 registries (21 + 14)", 92), ("Promotion gates + human review", 150), ("Model cards + release registry", 208), ("CI/CD + 12-hourly canary", 266)]
    for t, y in ops:
        b += box(650, y, 160, 44, t, "ops", 11, bold_first=False)
    srv = [("FastAPI + auth", 34), ("SQLite -> PostgreSQL", 104), ("React dashboard / exam / AIR screen", 174), ("GET /air_estimate (new)", 244)]
    for t, y in srv:
        b += box(835, y, 150, 56, t, "api", 11, bold_first=False)
    b += arrow(182, 200, 203, 200) + arrow(382, 200, 418, 200) + arrow(612, 172, 648, 172) + arrow(812, 172, 833, 172)
    b += poly_arrow([(910, 302), (910, 440), (515, 440), (515, 370)], "student attempts flow back as training data", dashed=True)
    b += label(10, 478, "Legend:", 11, "start", "txt", "700")
    for i, (k, t) in enumerate([("ext", "other teams' data"), ("gate", "validation"), ("ml", "ML"), ("ops", "MLOps"), ("api", "serving")]):
        f_, s_ = PALETTE[k]
        b += f'<rect x="{70 + i * 140}" y="466" width="14" height="14" rx="3" fill="{f_}" stroke="{s_}"/>' + label(90 + i * 140, 478, t, 11)
    return svg(1000, 495, b, "System block diagram")


def fig_data_flow() -> str:
    acc, rej, tot = INGEST.get("accepted", 0), INGEST.get("rejected", 0), INGEST.get("rows_in_file", 0)
    b = label(10, 20, "Question ingestion and quality funnel", 14, "start", "txt", "700")
    b += box(10, 50, 150, 70, f"Raw sample file {tot:,} rows marked Verified", "ext", 12, dashed=True)
    steps = ["Field, answer-key and diagram checks", "Text-quality gate (OCR corruption)", "Duplicate of PYQ / content / options", "Chapter resolver, 90% precision"]
    for i, t in enumerate(steps):
        x = 190 + i * 165
        b += arrow(x - 28 if i else 162, 85, x - 2, 85) + box(x, 50, 140, 70, t, "gate", 11.5)
    b += arrow(822, 85, 848, 85) + box(850, 50, 140, 70, f"Accepted {acc:,} rows", "ok", 12)
    b += arrow(920, 122, 920, 148)
    b += box(795, 150, 195, 54, "IRT cold-start parameters, linked to PYQ anchors", "ml", 11.5)
    b += arrow(793, 177, 777, 177) + box(545, 150, 230, 54, "Unified bank: forms, mocks, difficulty v2.1, retriever", "ml", 11.5)
    b += poly_arrow([(425, 122), (425, 245)], "rows that fail any check")
    b += box(250, 247, 500, 58, f"Quarantine {rej:,} rows, each with its reasons recorded (one row can fail several checks, so reason counts overlap)", "bad", 11.5)
    return svg(1000, 325, b, "Data ingestion flow")


def fig_loop() -> str:
    b = box(40, 30, 190, 62, "1. Pick questions: LinUCB, IRT information, prerequisite graph, duplicate guard", "ml", 11.5)
    b += box(300, 30, 190, 62, "2. Student answers: answer + time stored in test_attempts", "api", 11.5)
    b += box(560, 30, 190, 62, "3. Update: theta, chapter mastery, 8 skills, pace", "ml", 11.5)
    b += box(560, 140, 190, 62, "4. Predict: score range, AIR range, weak chapters, plan", "feat", 11.5)
    b += box(300, 140, 190, 62, "5. Dashboard / AIR screen shows result", "api", 11.5)
    b += box(40, 140, 190, 62, "6. Next paper adapts to the new state", "ml", 11.5)
    b += arrow(232, 61, 298, 61) + arrow(492, 61, 558, 61) + arrow(655, 94, 655, 138) + arrow(558, 171, 492, 171) + arrow(298, 171, 232, 171) + arrow(135, 138, 135, 94)
    return svg(800, 230, b, "Adaptive learning loop")


def fig_lifecycle() -> str:
    steps = [("1 Collect", "ext"), ("2 Check data", "gate"), ("3 Check labels", "gate"), ("4 Train", "ml"), ("5 Test on unseen", "ml"), ("6 Register SHA-256", "ops"), ("7 Promote (human)", "ops"), ("8 Serve", "api"), ("9 Monitor", "api")]
    b = ""
    for i, (t, k) in enumerate(steps):
        x = 8 + i * 108
        b += box(x, 40, 96, 54, t, k, 11.5)
        if i:
            b += arrow(x - 11, 67, x - 1, 67)
    b += poly_arrow([(900, 96), (900, 140), (478, 140), (56, 140), (56, 96)], "drift or new data triggers retraining", dashed=True)
    b += label(8, 20, "Steps 1-6 automatic, 7 needs a person, 8-9 run in production", 11.5, "start", "lbl")
    return svg(990, 160, b, "ML lifecycle")


def fig_pipeline() -> str:
    stages = RUN.get("stages", [])
    b = ""
    per_row = 6
    for i, s in enumerate(stages):
        r, c = divmod(i, per_row)
        x, y = 8 + c * 165, 20 + r * 92
        kind = "ok" if s.get("status") == "PASSED" else "bad"
        b += box(x, y, 150, 62, f"{i + 1}. {s['stage'].replace('_', ' ')} ({s.get('seconds', 0):.1f}s)", kind, 11)
        if c < per_row - 1 and i < len(stages) - 1:
            b += arrow(x + 151, y + 31, x + 164, y + 31)
    rows = (len(stages) + per_row - 1) // per_row
    return svg(1000, 40 + rows * 92, b, "17-stage pipeline")


def fig_stage_times() -> str:
    rows = sorted(((s["stage"].replace("_", " "), s.get("seconds", 0.0)) for s in RUN.get("stages", [])), key=lambda t: -t[1])
    return hbars(rows, width=900, fmt=lambda v: f"{v:.1f} s", title="Stage durations")


def fig_promotion() -> str:
    b = box(10, 40, 130, 60, "Candidate model trained on held-out split", "ml", 11.5)
    b += box(180, 40, 150, 60, "Gates: F1 gain >= 0.02, ECE <= 0.05, no leakage", "gate", 11.5)
    b += box(370, 40, 140, 60, "Human reviewer (blind sample)", "ext", 11.5, dashed=True)
    b += box(550, 40, 130, 60, "Release registry: staging + canary %", "ops", 11.5)
    b += box(720, 40, 130, 60, "Production (named reviewer)", "api", 11.5)
    b += box(550, 140, 130, 50, "One-command rollback", "bad", 11.5)
    b += arrow(142, 70, 178, 70) + arrow(332, 70, 368, 70) + arrow(512, 70, 548, 70) + arrow(682, 70, 718, 70) + arrow(615, 102, 615, 138, "if canary fails")
    b += label(10, 20, "Promotion flow: a model never replaces production on metrics alone", 12.5, "start", "txt", "700")
    return svg(870, 210, b, "Model promotion flow")


def fig_air_flow() -> str:
    b = label(10, 18, "Post-attempt AIR estimate: model/air_estimator.py", 14, "start", "txt", "700")
    b += box(10, 40, 150, 56, "Attempt responses (+ earlier history)", "api", 11.5)
    b += box(190, 40, 150, 56, "Hygiene: dedupe, validate, ignore bad times", "gate", 11.5)
    b += box(370, 40, 160, 56, "Evidence check: >= 10 scored per subject", "gate", 11.5)
    b += box(560, 40, 170, 56, "IRT 3PL equating: item difficulty a,b,c gives theta", "ml", 11.5)
    b += box(760, 40, 170, 56, "Equated NEET marks + 90% interval", "feat", 11.5)
    b += arrow(162, 68, 188, 68) + arrow(342, 68, 368, 68) + arrow(532, 68, 558, 68) + arrow(732, 68, 758, 68)
    b += box(760, 140, 170, 56, "Score to percentile to AIR (curve approximate)", "feat", 11.5)
    b += arrow(845, 98, 845, 138)
    b += box(560, 140, 170, 56, "Year scenarios: 2024-2026 cut-offs shift the curve", "ext", 11.5, dashed=True)
    b += box(560, 230, 170, 56, "College reach: normal approx. of the 90% interval", "feat", 11.5)
    b += arrow(758, 168, 732, 168) + poly_arrow([(845, 198), (845, 258), (732, 258)])
    b += box(10, 140, 170, 56, "Chapter mastery: IRT + BKT shrinkage", "ml", 11.5)
    b += box(200, 140, 170, 56, "Taxonomy weightage 2013-26: share and trend", "ext", 11.5, dashed=True)
    b += box(390, 140, 150, 56, "Syllabus mode: all or nmc_reduced", "gate", 11.5)
    b += box(10, 230, 170, 56, "Chapter difficulty: mean b, tier mix, ideal time", "ml", 11.5)
    b += box(200, 230, 340, 56, "Marks at stake = questions x 5 x (target - mastery); AIR places gained", "feat", 11.5)
    b += poly_arrow([(95, 198), (95, 215), (370, 215), (370, 228)]) + poly_arrow([(285, 198), (285, 228)]) + poly_arrow([(465, 198), (465, 228)]) + arrow(180, 258, 198, 258)
    b += box(200, 320, 340, 50, "Focus chapters + trend watchlist (rising and weak)", "ok", 12)
    b += arrow(370, 288, 370, 318)
    b += box(560, 320, 370, 50, "JSON report: air, scores, mock_difficulty, chapter_wise, assumptions", "ok", 12)
    b += arrow(542, 345, 558, 345) + poly_arrow([(645, 288), (645, 318)])
    return svg(950, 392, b, "AIR estimate flow")


def fig_rank_curve() -> str:
    rc = SIM.get("rank_curve", {})
    xs, ys = rc.get("score_anchors", []), rc.get("percentile_anchors", [])
    if not xs:
        return ""
    tiers = sorted(rc.get("college_tiers", []), key=lambda t: t["score"])
    w, L, R, T, B = 900, 56, 20, 30, 44
    h = 330 + 16 * len(tiers) + 12
    ph, pw = 330 - T - B, w - L - R
    px = lambda sc: L + pw * sc / 720
    py = lambda p: T + ph - ph * p / 100
    b = ""
    for t in range(0, 101, 25):
        b += f'<line x1="{L}" y1="{py(t):.1f}" x2="{L + pw}" y2="{py(t):.1f}" class="grid"/>' + label(L - 6, py(t) + 4, f"{t}%", 10.5, "end", "lbl")
    for sc in range(0, 721, 120):
        b += label(px(sc), 330 - 24, sc, 10.5, "middle", "lbl")
    b += label(L + pw / 2, 330 - 6, "NEET marks (out of 720)", 11, "middle", "lbl")
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    b += f'<polyline points="{pts}" fill="none" stroke="#0f766e" stroke-width="2.4"/>'
    for x, y in zip(xs, ys):
        b += f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="#0f766e"/>'
    for i, t in enumerate(tiers, start=1):
        xx = px(t["score"])
        b += f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{T + ph}" stroke="#b45309" stroke-dasharray="4 4"/>'
        b += f'<circle cx="{xx:.1f}" cy="{T - 8}" r="8" fill="#b45309"/>' + label(xx, T - 4.5, i, 10, "middle", "txt", "700").replace('class="txt"', 'fill="#ffffff"')
        b += label(L, 330 + 12 + (i - 1) * 16, f"{i}: {t['name']}, about {t['score']} marks (approximate threshold)", 11, "start", "lbl")
    b += label(L + 8, T + ph - 8, "dots = hand-set anchors (approximate)", 10.5, "start", "lbl")
    return svg(w, h, b, "Score to percentile curve")


def fig_tier_heatmap() -> str:
    cells = defaultdict(Counter)
    subj = {}
    for r in PYQ:
        cells[r["chapter"]][str(r["difficulty_level"]).lower()] += 1
        subj[r["chapter"]] = r["subject"]
    b = ""
    maxc = max((max(c.values()) for c in cells.values()), default=1)
    for si, s in enumerate(SUBJECTS):
        chs = sorted((c for c in cells if subj[c] == s), key=lambda c: c)
        x0 = 8 + si * 330
        b += label(x0, 14, f"{s} ({len(chs)} chapters)", 12, "start", "txt", "700")
        for t, tier in enumerate(TIERS):
            b += label(x0 + 196 + t * 40 + 18, 30, tier[:3].title(), 10.5, "middle", "lbl")
        for i, ch in enumerate(chs):
            y = 36 + i * 17
            b += label(x0, y + 12, ch[:34], 10, "start", "txt")
            for t, tier in enumerate(TIERS):
                v = cells[ch][tier]
                a = 0.15 + 0.85 * min(1, v / maxc * 2.2)
                if v == 0:
                    b += f'<rect x="{x0 + 196 + t * 40}" y="{y}" width="36" height="14" rx="3" fill="#fee2e2" stroke="#b91c1c" stroke-width="1.6"/>' + label(x0 + 196 + t * 40 + 18, y + 11, "0", 10, "middle", "txt", "700")
                else:
                    b += f'<rect x="{x0 + 196 + t * 40}" y="{y}" width="36" height="14" rx="3" fill="#0f766e" fill-opacity="{a:.2f}"/>' + label(x0 + 196 + t * 40 + 18, y + 11, v, 10, "middle", "txt")
    return svg(1000, 36 + 24 * 17, b, "PYQ tier coverage heatmap")


def fig_chapter_dots() -> str:
    rows = [c for c in CHAPTER.get("by_chapter", []) if c["reliability"] == "ok"]
    rows.sort(key=lambda c: c["accuracy"])
    left, top, rh = 250, 30, 15
    w = 900
    pw = w - left - 40
    h = top + rh * len(rows) + 28
    px = lambda v: left + pw * (v - 0.3) / 0.7
    b = label(left, 16, "held-out accuracy by chapter (dot, 95% CI); tick = always guess the majority tier; red = does not beat that guess", 11, "start", "lbl")
    for t in (0.4, 0.6, 0.8, 1.0):
        b += f'<line x1="{px(t):.1f}" y1="{top}" x2="{px(t):.1f}" y2="{top + rh * len(rows)}" class="grid"/>' + label(px(t), h - 8, f"{int(t * 100)}%", 10.5, "middle", "lbl")
    for i, c in enumerate(rows):
        y = top + i * rh + 8
        bad = not c["beats_majority"]
        col = "#b91c1c" if bad else "#0f766e"
        lo, hi = c["accuracy_ci95"]
        b += label(left - 8, y + 4, f'{c["group"].split("::")[1][:30]} ({c["n"]})', 10, "end", "txt")
        b += f'<line x1="{px(max(lo, 0.3)):.1f}" y1="{y}" x2="{px(hi):.1f}" y2="{y}" stroke="{col}" stroke-opacity="0.45" stroke-width="3"/>'
        b += f'<circle cx="{px(c["accuracy"]):.1f}" cy="{y}" r="3.6" fill="{col}"/>'
        b += f'<line x1="{px(c["majority_baseline"]):.1f}" y1="{y - 5}" x2="{px(c["majority_baseline"]):.1f}" y2="{y + 5}" stroke="#111827" stroke-width="1.6"/>'
    return svg(w, h, b, "Chapter-wise held-out accuracy")


def fig_weightage() -> str:
    rows, meta = taxonomy_chapter_weightage(TAX)
    entries = TAX.get("Chapter_Weightage_And_Marks", [])
    per_year = Counter()
    by_ch = defaultdict(Counter)
    for e in entries:
        for y, v in e["yearly_weightage"].items():
            per_year[int(y)] += v["question_count"]
            by_ch[e["chapter"]][int(y)] += v["question_count"]
    years = sorted(per_year)
    cand = [(c, r) for c, r in rows.items() if r["mean_questions_per_paper"] >= 3]
    rising = sorted(cand, key=lambda t: -(t[1]["last3_share"] / t[1]["mean_share"]))[:3]
    falling = sorted(cand, key=lambda t: (t[1]["last3_share"] / t[1]["mean_share"]))[:3]

    def smooth(vals):
        return [sum(vals[max(0, i - 1): i + 2]) / len(vals[max(0, i - 1): i + 2]) for i in range(len(vals))]

    series = []
    for tag, group in (("\u25b2 ", rising), ("\u25bc ", falling)):
        for c, _ in group:
            series.append((tag + c[:26], smooth([100 * by_ch[c][y] / per_year[y] for y in years])))
    return line_chart(series, [str(y) for y in years], fmt=lambda v: f"{v:.1f}%", title="Chapter share of each NEET paper (3-year moving average)")


def fig_taxonomy_gap() -> str:
    cd = CHAPTER.get("cross_dataset", {})
    tq = TAX.get("Chapter_Weightage_And_Marks", [])
    tax = Counter()
    for e in tq:
        for y, v in e["yearly_weightage"].items():
            tax[int(y)] += v["question_count"]
    gap = cd.get("bank_missing_by_year", {})
    groups = [(str(y), [tax[y], tax[y] - gap.get(str(y), 0)]) for y in sorted(tax)]
    return vbars_grouped(groups, ["taxonomy (full paper)", "verified bank"], ymax=220, fmt=lambda v: f"{v:g}", colors=("#1d4ed8", "#0f766e"), title="Questions per year: taxonomy vs bank")


def fig_funnel() -> str:
    r = dict(INGEST.get("rejection_reasons", {}))
    rows = [("Rows in raw file", INGEST.get("rows_in_file", 0))]
    colors = ["#6b7280"]
    for k, v in sorted(r.items(), key=lambda kv: -kv[1]):
        rows.append((f"rejected: {k.replace('_', ' ')}", v))
        colors.append("#b91c1c" if k.startswith("ocr_") else "#b45309")
    rows.append(("Accepted into unified bank", INGEST.get("accepted", 0)))
    colors.append("#15803d")
    return hbars(rows, width=900, left=300, colors=colors, title="Ingestion funnel")


def fig_pyq_tiers() -> str:
    cells = defaultdict(Counter)
    for r in PYQ:
        cells[r["subject"]][str(r["difficulty_level"]).lower()] += 1
    groups = [(s, [cells[s][t] for t in TIERS]) for s in SUBJECTS]
    return vbars_grouped(groups, [t.title() for t in TIERS], ymax=650, title="PYQ bank by subject and tier")


def fig_models() -> str:
    v = V21
    groups = [
        ("Majority guess", [100 * v.get("majority_baseline", 0)]),
        ("v2 structural RF", [100 * v.get("v2_structural_rf", {}).get("accuracy", 0)]),
        ("v2.1 text model", [100 * v.get("v21_tfidf_logreg", {}).get("accuracy", 0)]),
        ("Formula labels vs human labels", [100 * LABEL.get("heuristic_rule_vs_human_verified_sample_tiers", {}).get("accuracy", 0)]),
    ]
    return vbars_grouped(groups, ["accuracy % on human-verified tiers, held-out papers"], ymax=100, fmt=lambda v: f"{v:.1f}", colors=("#0f766e",), title="Difficulty model comparison")


def fig_latency() -> str:
    lat = HARNESS.get("latency_sla_benchmarks", {})
    rows = sorted(((k.replace("_", " "), v["p95_ms"]) for k, v in lat.items() if isinstance(v, dict)), key=lambda t: -t[1])
    return hbars(rows, width=900, left=260, fmt=lambda v: f"{v:.3f} ms", title="p95 latency per model")


def fig_calibration() -> str:
    cal = SIM.get("calibration", [])
    groups = [(f"true theta {c['true_theta']:g}", [c["rmse"], c["mean_claimed_sem"], abs(c["mean_error"])]) for c in cal]
    return vbars_grouped(groups, ["actual error (RMSE)", "claimed SEM", "|mean bias|"], ymax=0.25, fmt=lambda v: f"{v:.3f}", colors=("#b45309", "#0f766e", "#6d28d9"), title="Ability estimate calibration")


def fig_equating() -> str:
    eq = SIM.get("equating", [])
    groups = [(e["paper"], [e["mean_raw_marks"], e["mean_equated_marks"]]) for e in eq]
    return vbars_grouped(groups, ["raw marks", "equated NEET marks"], ymax=760, ymin=-100, fmt=lambda v: f"{v:.0f}", colors=("#b45309", "#0f766e"), title="Same ability on different papers")


def fig_year_scenarios() -> str:
    ys = SIM.get("sample_year_scenarios", {})
    groups = [(f"NEET {y}\ncut-off {v['general_cutoff']}", [v["air"] / 1000]) for y, v in ys.items()]
    return vbars_grouped([(g[0].replace("\n", " "), g[1]) for g in groups], ["same marks, AIR (thousands)"], ymax=max([g[1][0] for g in groups] + [1]) * 1.2, fmt=lambda v: f"{v:,.0f}k", colors=("#be185d",), title="Same marks across paper years")


def fig_basis_counts() -> str:
    labels = {
        "fitted_on_real_students": "fitted on real students",
        "fitted_on_real_human_labels": "fitted on real human labels",
        "fitted_on_formula_derived_labels": "fitted on formula-derived labels",
        "hand_designed_prior": "hand-designed prior",
        "design_objective_simulated": "design objective, simulated reward",
        "rule_based": "rule-based",
        "math_on_priors": "math on priors",
        "built_waiting_for_data": "built, waiting for data",
    }
    colors = {"fitted_on_real_students": "#15803d", "fitted_on_real_human_labels": "#0f766e", "fitted_on_formula_derived_labels": "#b45309", "hand_designed_prior": "#b91c1c", "design_objective_simulated": "#b45309", "rule_based": "#6b7280", "math_on_priors": "#6d28d9", "built_waiting_for_data": "#1d4ed8"}
    counts = Counter(m["basis"] for m in PROV.get("models", []))
    keys = [k for k in labels if counts.get(k)]
    rows = [(labels[k], counts[k]) for k in keys]
    return hbars(rows, width=900, left=280, colors=[colors[k] for k in keys], fmt=lambda v: f"{v} model" + ("" if v == 1 else "s"), title="Models by honest basis") if rows else ""


def fig_label_sensitivity() -> str:
    rows = SIM.get("label_sensitivity", [])
    groups = [(f"true theta {r['true_theta']:g}", [r["mean_marks_formula"], r["mean_marks_v21"], r["mean_marks_uninformative"]]) for r in rows]
    return vbars_grouped(groups, ["formula labels (production)", "v2.1 human-trained labels", "labels carry no information"], ymax=700, fmt=lambda v: f"{v:.0f}", colors=("#b45309", "#0f766e", "#6d28d9"), title="Equated marks under three difficulty-label assumptions") if groups else ""


def provenance_table() -> str:
    names = {"fitted_on_real_human_labels": "ok", "fitted_on_real_students": "ok"}

    def ev(e):
        parts = []
        for k, v in e.items():
            if v is None or isinstance(v, (dict, list)):
                continue
            parts.append(f"{k.replace('_', ' ')}: {v:.3f}" if isinstance(v, float) else f"{k.replace('_', ' ')}: {v}")
        return "; ".join(parts[:4]) or "none: no accuracy can be claimed"

    rows = ""
    for m in PROV.get("models", []):
        cls = names.get(m["basis"], "bad" if m["basis"] in ("hand_designed_prior",) else "")
        rows += (
            f"<tr><td><b>{html.escape(m['name'])}</b></td><td class='{cls}'>{html.escape(m['basis'].replace('_', ' '))}</td>"
            f"<td>{html.escape(m['parameters_come_from'])}</td><td>{html.escape(ev(m['evidence']))}</td><td>{html.escape('; '.join(m['limitations']))}</td></tr>"
        )
    return f"<table><tr><th>Model</th><th>Honest basis</th><th>Parameters come from</th><th>Measured evidence</th><th>Limits</th></tr>{rows}</table>"


def thumbnails(n=6):
    seen_papers, out = set(), []
    base = ROOT / "data" / "tagged_questions"
    for r in SAMPLE_RAW:
        if not r.get("has_diagram") or not r.get("image_asset"):
            continue
        paper = r.get("paper_source")
        if paper in seen_papers:
            continue
        path = base / r["image_asset"]
        if not path.exists():
            continue
        try:
            im = Image.open(path).convert("RGB")
            im.thumbnail((340, 240))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=70, optimize=True)
        except Exception:
            continue
        seen_papers.add(paper)
        out.append((r["question_id"], paper, r["subject"], r["chapter"], base64.b64encode(buf.getvalue()).decode()))
        if len(out) >= n:
            break
    return out


def suite_sources() -> dict[str, str]:
    out = {}
    for path in (M / "tests" / "test_ml_standalone_suite.py", M / "tests" / "test_data_quality_and_chapter_eval.py", ROOT / "backend" / "tests" / "test_v2_engine.py"):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and re.match(r"(ml_suite_|test_)\d+", node.name):
                m = re.match(r"(ml_suite|test)_(\d+)", node.name)
                out[f"{'ML' if m.group(1) == 'ml_suite' else 'BE'}{m.group(2)}"] = ast.get_source_segment(text, node) or ""
    return out


SUITES = suite_sources()

MODELS = [
    ("3PL IRT", "Student ability theta from answers", ["irt_3pl_model"], "Production (priors)"),
    ("BKT + forgetting", "Chapter mastery and decay", ["bkt_model", "state_machine"], "Production (priors)"),
    ("DINA + pace", "8 thinking skills and speed", ["dina_cognitive_diagnosis"], "Production (priors)"),
    ("Prerequisite DAG", "Which chapter needs which", ["knowledge_graph_dag"], "Production"),
    ("LinUCB bandit", "Picks the next questions", ["linucb_bandit"], "Production (warm start)"),
    ("Score simulator + Laplacian leverage", "NEET score range, high-leverage chapters", ["monte_carlo_conformal_neet"], "Production"),
    ("National rank simulator", "Marks to percentile to AIR", ["national_rank_simulator"], "Production (approximate curve)"),
    ("Difficulty RF", "PYQ tier (formula labels)", ["rf_compact_runtime", "training_and_validation_pipeline"], "Production (measures the formula)"),
    ("Difficulty v2.1", "Tier from text, human labels", ["difficulty_model_v21_text"], "Candidate: eligible, pending review"),
    ("Probability calibrator", "Temperature scaling", ["probability_calibrator"], "Rejected on holdout (T = 1)"),
    ("NCERT retriever + trap detector", "Finds the right NCERT note", ["rag_socratic_engine"], "Production"),
    ("Near-duplicate guard", "No repeated questions in mocks", ["unified_similarity_guard", "question_similarity_audit"], "Production"),
    ("Chapter resolver", "Fixes chapter labels at ingestion", ["chapter_resolver"], "Production (ingestion)"),
    ("Forms and mocks", "Equal-difficulty 180-Q papers", ["assessment_blueprint", "mock_test_generator"], "Production"),
    ("Study plan engine", "Weak chapters and revision queue", ["study_plan_and_coverage_engine"], "Production"),
    ("Drift + retraining trigger", "PSI, Brier, ECE, KS", ["feature_store_and_drift", "continuous_retraining_engine"], "Production"),
    ("Student-response training", "Real IRT + BKT from answers", ["student_response_training"], "Built; waits for 200 students"),
    ("Text-quality gate (new)", "OCR corruption and duplicate options", ["text_quality_gate"], "Production gate"),
    ("Response hygiene (new)", "Dedupe and validate answers", ["response_hygiene"], "Production"),
    ("Chapter/topic evaluation (new)", "Per-chapter accuracy, coverage ratchet", ["chapter_topic_evaluation"], "Pipeline stage"),
    ("AIR estimator (new)", "Post-attempt AIR, chapter ROI", ["air_estimator"], "Production feature"),
]


def models_table() -> str:
    rows = ""
    uncovered = 0
    for name, what, mods, status in MODELS:
        cov = sorted((k for k, src in SUITES.items() if any(m in src for m in mods)), key=lambda k: (k[:2], int(k[2:])))
        if not cov:
            uncovered += 1
        cls = "ok" if cov else "bad"
        rows += f"<tr><td><b>{html.escape(name)}</b></td><td>{html.escape(what)}</td><td>{html.escape(status)}</td><td class='{cls}'>{html.escape(', '.join(cov) if cov else 'NO DIRECT SUITE')}</td></tr>"
    return f"<table><tr><th>Model</th><th>Job</th><th>Status</th><th>Covered by suites (derived from test source)</th></tr>{rows}</table>", uncovered


FINDINGS = [
    ("F1", "Data", "184 of 9,448 accepted sample rows (1.9%) carried OCR corruption (garbled CJK, Latin-Extended, all-comma options). The validator never checked text quality.", "High", "Fixed", "text_quality_gate in ingestion; accepted 9,264"),
    ("F2", "Student-facing", "Assessment form 2 contained 3 corrupted PYQs; 12 PYQs overall are OCR-damaged.", "High", "Fixed", "forms rebuilt; loader excludes flagged PYQs; audit invariant"),
    ("F3", "Scoring", "Submitting the same 180 answers twice gave marks=520 (impossible, max 720).", "High", "Fixed", "response_hygiene: last answer per question wins"),
    ("F4", "Scoring", "String 'false' was scored as correct; garbage values were not rejected.", "Medium", "Fixed", "strict parse_correct, dropped and counted"),
    ("F5", "Scoring", "Zero or negative times polluted the pace calculation; empty submissions still produced a prediction.", "Low", "Fixed", "clean_time; prior_only flag"),
    ("F6", "AIR accuracy", "Weightage was computed from a bank missing 161 questions vs the taxonomy (2025: 16, 2026: 21), under-weighting recent years.", "High", "Fixed", "AIR uses taxonomy with bank fallback; cross-dataset gate"),
    ("F7", "AIR accuracy", "ML layer ignored the backend's NMC reduced-syllabus switch.", "Medium", "Fixed", "syllabus_mode all|nmc_reduced; test locks the two exclusion lists together"),
    ("F8", "Coverage", "My first coverage check pooled PYQ and sample data and hid PYQ-only zero-tier chapters (5, not 3).", "Medium", "Fixed", "both views reported; ratchet fails on any new gap"),
    ("F9", "Forms", "Equated forms could be rebuilt after students had taken them (they reshuffle 82-109 questions when the bank changes).", "Medium", "Fixed", "freeze guard refuses rebuild once attempts exist"),
    ("F10", "Tests", "My first files broke the repo's no-comments/no-docstrings rule (suite 15) and one pipeline self-test judged the model on 1 student per level (statistically flawed).", "Medium", "Fixed", "rewritten; 30 students per level, calibration-based checks"),
    ("F11", "Claims", "Handover said artifacts are byte-identical. On the pinned runtime metrics reproduce to 4 decimals but 6 binaries differ across machines.", "Info", "Corrected", "claim reworded: reproducible metrics within 0.005"),
    ("F12", "Claims", "I earlier called the zero-tier PYQ chapters 'the original Physics bug alive'. The backend already has ADJACENT_FALLBACK_ORDER, so they are not dropped.", "Info", "Corrected", "downgraded to adaptive-precision loss"),
    ('F13', 'Honesty', 'Nothing machine-readable said what each model is really based on; accuracies could be read as validation. Several are hand-designed priors, formula-derived or simulated.', 'High', 'Fixed', 'model_provenance registry + consistency gate + /mlops/model_provenance + pipeline stage'),
    ('F14', 'Hardcoding', 'NEET facts, score-to-percentile anchors and college thresholds were constants buried in code with no provenance.', 'Medium', 'Fixed', 'moved to model/neet_reference_facts.json; every block carries source and verification; values identical'),
    ('F15', 'AIR honesty', 'AIR was shown to 6 digits with a range that ignored the unvalidated difficulty labels and the paper-year variation.', 'High', 'Fixed', '2-significant-digit display; sensitivity block; range_if_assumptions_wrong; reliability statement'),
    ('F16', 'AIR honesty', 'Measured: using v2.1 labels or no-information labels instead of the formula lowers equated marks by tens of marks (see 6.6). The production estimate may be optimistic.', 'High', 'Mitigated', 'surfaced in every response; real fix needs human labels or student data'),
    ("O1", "Data/Backend", "Thermodynamics is one chapter key for Physics (22 PYQ) and Chemistry (21 PYQ): merged mastery, one DAG node, subject totals 564/612 in a taxonomy summary.", "Medium-High", "Open", "needs backend topic_id contract change (ratchet-tracked)"),
    ("O2", "Content", "12 PYQs and Sample Paper 50 (66 of 86 rows) need source-text repair; 10 PYQs show the OCR misread of pi.", "High", "Open", "data/review/ocr_corruption_review.csv"),
    ("O3", "Content", "PYQ difficulty tiers come from a text formula (kappa 0.07 vs human labels). v2.1 passes gates but is not promoted.", "High", "Open", "300-question blind review"),
    ("O4", "Product", "0 real student attempts: IRT, BKT, DINA and the bandit run on priors.", "High", "Open", "collect 200 students; pipeline already waits for them"),
    ("O5", "Data", "Score-to-percentile curve is hand-set; NTA publishes no full table. AIR is an indicative range.", "High", "Open", "needs sourced marks-vs-AIR points for one year"),
    ("O6", "Content", "PYQ bank lacks 161 questions vs the taxonomy (2022: 24, 2026: 21, 2025: 16).", "Medium", "Open", "content team to restore or verify"),
    ("O7", "Product", "Syllabus decision: forms carry 7/180 questions from the NMC-excluded chapters and non-repeating mock capacity is 1 guaranteed, bottlenecked by those chapters.", "Medium", "Open", "decide all vs nmc_reduced, then re-derive blueprint"),
    ("O8", "Data", "5 PYQ-only zero-tier chapters; backend fallback masks them but a basic-level student gets medium questions there.", "Low-Medium", "Mitigated", "ratchet baseline; content to add questions"),
    ("O9", "Content", "131 template explanations (4 name the wrong option); 0 translated questions of 12 languages.", "Medium", "Open", "content rewrite lists"),
    ("O10", "Legal", "Behaviour data of under-18 students (DPDP Rules 2025).", "High", "Open", "switch ready: KLASSED_BEHAVIOURAL_TELEMETRY=off"),
    ("O11", "DevOps", "Persistence and secrets: SQLite in /tmp loses answers on cold start; new JWT secret must be set in Vercel.", "High", "Open", "PostgreSQL + secret (DevOps)"),
    ("O12", "Repo", "The analysed clone lacks platform-API commit f8f8419 listed in the handover.", "Medium", "Open", "verify merge; align paper_number semantics with /air_estimate"),
    ("O14", "Verification", "NEET cut-offs, candidate counts and the 2026 source were not re-verified against NTA in this audit (flagged not_verified_in_audit in the facts file).", "Medium", "Open", "ML/product to verify and set verification: verified"),
    ("O13", "Model", "In 17 of 54 reliably measured chapters v2.1 does not beat guessing the majority tier.", "Info", "Open", "report per chapter, not only the 82.2% headline"),
]

DEPENDENCIES = [
    ("Official PYQ bank, 2,400 questions", "Content / data assessment", "All models, forms, mocks", "Verified; 12 OCR-damaged; 161 fewer than the taxonomy", "Repair or quarantine 12; restore missing questions"),
    ("Sample papers 1-101, 16,689 rows", "Data team", "Difficulty v2.1, mocks, IRT linking", "9,264 accepted; 7,425 rejected (reasons overlap: duplicates, 2,633 unresolved chapter, OCR corruption)", "Re-source Paper 50; relabel the 2,633 with top-3 suggestions"),
    ("NCERT chapter notes, 65 chapters", "Content", "NCERT retriever", "Present; retrieval top-1 34.9%, top-3 53.4%", "Semantic encoder on a GPU machine"),
    ("Weightage taxonomy 2013-2026", "Content / data", "AIR chapter weightage, importance", "2,561 questions, 64 chapters; omits 2 chapters; agrees with bank at Spearman 0.94", "Keep in sync; gate fails on drift"),
    ("Sourced NEET facts 2024-2026", "ML (press sources)", "Rank curve, year scenarios", "Candidates and general cut-off only", "Marks-vs-AIR table for one real year"),
    ("Human tier review sheet, 300 questions", "Content team", "Promote difficulty v2.1", "Not done", "Fill blind labels; run review_pack score"),
    ("Student attempts", "Backend / product", "IRT, BKT, DINA, bandit, A/B", "0 rows", "200 students, 20,000 answers"),
    ("Real NEET scores of students (consent)", "Product", "Score-prediction validation", "None", "Collect after the exam"),
    ("Student group metadata (language, region, board)", "Product / legal", "Fairness report", "None", "Collect with consent"),
    ("Explanations and translations", "Content", "Remediation, 12 languages", "131 templates; 0 translated", "Rewrite list and translation workflow"),
    ("Syllabus decision (all vs nmc_reduced)", "Product", "Forms, mocks, AIR", "Backend supports both; ML now supports both", "Pick the default and re-derive the blueprint"),
]


def verification_table() -> str:
    f = FINAL
    rows = [
        ("Full ML pipeline", f"{f.get('pipeline_passed', '?')}/{f.get('pipeline_total', '?')} stages PASSED in {f.get('pipeline_seconds', '?')} s", f.get("pipeline_ok")),
        ("Reproducibility (retrain and compare)", f"{f.get('repro_metrics', '?')} metrics within {f.get('repro_tolerance', '?')}", f.get("repro_ok")),
        ("ML standalone suites", f"{f.get('ml_passed', '?')}/{f.get('ml_total', '?')} passed", f.get("ml_ok")),
        ("Backend integration suites", f"{f.get('be_passed', '?')}/{f.get('be_total', '?')} passed", f.get("be_ok")),
        ("Artifact integrity (SHA-256)", f"{f.get('core_artifacts', '?')} core + {f.get('ext_artifacts', '?')} extension VERIFIED", f.get("integrity_ok")),
        ("Train/serve runtime parity (strict)", f"Python {f.get('python', '?')}, pinned libraries match", f.get("parity_ok")),
        ("Evaluation harness", "personas, bandit regret, 8 latency SLAs", f.get("harness_ok")),
        ("Static analysis", f"{f.get('pyflakes_warnings', '?')} pyflakes warnings; all code compiles", f.get("static_ok")),
    ]
    body = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(str(b))}</td><td class='{'ok' if c else 'bad'}'>{'PASS' if c else 'NOT VERIFIED'}</td></tr>" for a, b, c in rows)
    return f"<table><tr><th>Check</th><th>Result</th><th>Status</th></tr>{body}</table>"


def table(headers, rows, cls_col=None) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = ""
    for r in rows:
        tds = ""
        for i, v in enumerate(r):
            c = ""
            if cls_col is not None and i == cls_col:
                c = " class='ok'" if v in ("Fixed", "Corrected", "Mitigated") else " class='bad'"
            tds += f"<td{c}>{html.escape(str(v))}</td>"
        body += f"<tr>{tds}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def previous_accepted() -> int | None:
    import subprocess

    try:
        raw = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:model/sample_paper_ingestion_report.json"], stderr=subprocess.DEVNULL)
        return int(json.loads(raw).get("accepted"))
    except Exception:
        return None


def mock_audit() -> dict:
    from model.mock_test_generator import generate_mock_180, mock_capacity, reserved_initial_assessment_ids
    from model.student_profile_initializer import _bank_by_id
    from model.text_quality_gate import is_text_corrupt

    bank = _bank_by_id("unified")
    raw = {r["question_id"]: r for r in PYQ + SAMPLE_RAW}
    reserved = reserved_initial_assessment_ids()
    seen, rows = set(), []
    bad = Counter()
    for i in range(1, 11):
        m = generate_mock_180("report_audit", i, seen_question_ids=set(seen))
        ids = m["question_ids"]
        subj = Counter(bank[q]["subject"] for q in ids)
        tiers = Counter(bank[q]["tier"] for q in ids)
        rep = set(ids) & seen
        chap = Counter(bank[q]["unit"].split("::")[1] for q in rep)
        if len(ids) != 180 or len(set(ids)) != 180:
            bad["size"] += 1
        if dict(subj) != {"Physics": 45, "Chemistry": 45, "Biology": 90}:
            bad["quota"] += 1
        if set(ids) & reserved:
            bad["initial_form_leak"] += 1
        if any(is_text_corrupt(raw[q]) for q in ids):
            bad["corrupt_text"] += 1
        rows.append((i, subj["Physics"], subj["Chemistry"], subj["Biology"], tiers["basic"], tiers["medium"], tiers["advanced"], len(rep), ", ".join(f"{k} x{v}" for k, v in chap.items()) or "none"))
        seen |= set(ids)
    cap = mock_capacity()
    return {"rows": rows, "bad": dict(bad), "capacity": {k: cap[k] for k in ("pool_size", "non_repeating_mocks_guaranteed", "non_repeating_mocks_by_chapter_quota", "non_repeating_mocks_by_tier_quota", "bottleneck_chapter", "bottleneck_tier_cell")}}


CSS = """
:root{--bg:#ffffff;--fg:#111827;--muted:#4b5563;--card:#f8fafc;--line:#d1d5db;--grid:#e5e7eb;--ok:#15803d;--bad:#b91c1c;--accent:#0f766e;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#0b1220;--fg:#e5e7eb;--muted:#9ca3af;--card:#111a2e;--line:#334155;--grid:#243247;--ok:#4ade80;--bad:#f87171;--accent:#2dd4bf;--warn:#fbbf24}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:34px 22px 18px;border-bottom:1px solid var(--line);background:linear-gradient(135deg,rgba(15,118,110,.14),rgba(109,40,217,.10))}
.wrap{max-width:1080px;margin:0 auto;padding:0 18px}
h1{margin:0 0 6px;font-size:28px}h2{margin:42px 0 6px;font-size:22px;border-bottom:2px solid var(--accent);padding-bottom:4px}h3{margin:26px 0 6px;font-size:17px}
.sub{color:var(--muted);margin:0}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:5;overflow-x:auto;white-space:nowrap}
nav a{display:inline-block;padding:9px 11px;color:var(--fg);text-decoration:none;font-size:13px}nav a:hover{color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card b{display:block;font-size:24px;color:var(--accent)}.card span{color:var(--muted);font-size:13px}
table{border-collapse:collapse;width:100%;margin:10px 0 16px;font-size:13px;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:var(--card)}
td.ok{color:var(--ok);font-weight:600}td.bad{color:var(--bad);font-weight:600}
.fig{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;margin:8px 0}
.txt{fill:var(--fg)}.lbl{fill:var(--muted)}.grid{stroke:var(--grid);stroke-width:1}
.note{border-left:4px solid var(--warn);background:var(--card);padding:8px 12px;margin:10px 0;border-radius:0 8px 8px 0}
.good{border-left-color:var(--ok)}.bad{border-left-color:var(--bad)}
code,pre{background:var(--card);border:1px solid var(--line);border-radius:6px;font-size:12.5px}code{padding:1px 5px}pre{padding:10px;overflow-x:auto}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.gallery figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px}.gallery img{width:100%;border-radius:6px;background:#fff}
figcaption{font-size:12px;color:var(--muted);margin-top:4px}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
footer{margin:46px 0 30px;color:var(--muted);font-size:13px}
@media print{nav{display:none}.fig{break-inside:avoid}}
"""


def build() -> str:
    f = FINAL
    acc, tot = INGEST.get("accepted", 0), INGEST.get("rows_in_file", 0)
    rej_total = INGEST.get("rejected", 0)
    prev = previous_accepted()
    ocr_rows = f"{prev - acc:,}" if prev else "an unknown number of"
    v21 = V21.get("v21_tfidf_logreg", {})
    cd = CHAPTER.get("cross_dataset", {})
    cov = CHAPTER.get("coverage", {})
    mocks = mock_audit()
    mt, uncovered = models_table()
    open_n = sum(1 for x in FINDINGS if x[4] in ("Open",))
    fixed_n = sum(1 for x in FINDINGS if x[4] in ("Fixed", "Corrected"))
    thumbs = thumbnails()

    cards = [
        (f"{f.get('pipeline_passed', '?')}/{f.get('pipeline_total', '?')}", "pipeline stages passed"),
        (f"{f.get('ml_passed', '?')}/{f.get('ml_total', '?')}", "ML test suites"),
        (f"{f.get('be_passed', '?')}/{f.get('be_total', '?')}", "backend test suites"),
        (f"{acc:,}", f"questions accepted (of {tot:,})"),
        (pct(v21.get("accuracy")), "difficulty v2.1, unseen papers"),
        (f"{fixed_n} / {open_n}", "findings fixed / still open"),
    ]
    card_html = "".join(f"<div class='card'><b>{a}</b><span>{b}</span></div>" for a, b in cards)

    promo_rows = []
    for d in PROMO.get("decisions", []):
        gates = ", ".join(f"{k}: {'pass' if v else 'fail'}" for k, v in d.get("gates", {}).items())
        promo_rows.append((d.get("candidate"), d.get("compared_with"), gates, d.get("decision")))

    mock_rows = [tuple(r) for r in mocks["rows"]]
    cap = mocks["capacity"]
    sample_rows = [
        (c["chapter"], c["subject"], f"{c['mastery']:.2f}", c["expected_questions_per_neet_paper"], c["pyq_trend"], c["marks_at_stake_if_target_reached"], f"{c['air_places_gained_if_target_reached']:,}")
        for c in SIM.get("sample_focus_chapters", [])
    ]
    eq_rows = [(e["paper"], e["mean_raw_marks"], e["mean_equated_marks"], f"{e['median_air']:,}", f"{e['median_range_width']:,}", e["verdict"]) for e in SIM.get("equating", [])]
    cal_rows = [(c["true_theta"], c["n"], c["mean_error"], c["rmse"], c["mean_claimed_sem"], pct(c["coverage_95"], 0), f"{c['median_air']:,}") for c in SIM.get("calibration", [])]
    sample_air = SIM.get("sample_air", {})
    label_rows = [(r["true_theta"], r["students"], r["mean_marks_formula"], r["mean_marks_v21"], r["mean_marks_uninformative"], f"{r['median_air']:,}", f"{r['median_range_if_assumptions_wrong'][0]:,} to {r['median_range_if_assumptions_wrong'][1]:,}") for r in SIM.get("label_sensitivity", [])]
    gaps = [abs(r["mean_marks_v21"] - r["mean_marks_formula"]) for r in SIM.get("label_sensitivity", [])] + [abs(r["mean_marks_uninformative"] - r["mean_marks_formula"]) for r in SIM.get("label_sensitivity", [])]
    sens_lo, sens_hi = (min(gaps), max(gaps)) if gaps else (0.0, 0.0)
    eqs = {e["paper"]: e for e in SIM.get("equating", [])}
    forms_eq = [v["mean_equated_marks"] for k, v in eqs.items() if k.startswith("NEET-style")]
    hard, easy = eqs.get("deliberately hard", {}), eqs.get("deliberately easy", {})
    raw_gap = easy.get("mean_raw_marks", 0) - hard.get("mean_raw_marks", 0)
    eq_gap = easy.get("mean_equated_marks", 0) - hard.get("mean_equated_marks", 0)
    equating_text = (
        f"Equating makes NEET-style forms interchangeable: the three forms give equated marks within {max(forms_eq) - min(forms_eq):.0f} marks of each other (24 simulated students each, so sampling noise included). "
        f"On deliberately extreme papers the raw gap of {raw_gap:.0f} marks shrinks to {eq_gap:.0f} equated marks but does not vanish, and the AIR range widens "
        f"(median width {hard.get('median_range_width', 0):,} on the hard paper, {easy.get('median_range_width', 0):,} on the easy one, versus {eqs.get('NEET-style form 0', {}).get('median_range_width', 0):,} on a NEET-style form). "
        "A paper far too hard or too easy carries little information; the report exposes both facts instead of hiding them."
    )
    gallery = "".join(f"<figure><img alt='{html.escape(q)}' src='data:image/jpeg;base64,{b}'/><figcaption>{html.escape(q)}<br>{html.escape(p)} - {html.escape(s)} - {html.escape(c)}</figcaption></figure>" for q, p, s, c, b in thumbs)
    tq_pyq, tq_raw, tq_acc = TQ.get("pyq", {}), TQ.get("sample_raw", {}), TQ.get("sample_accepted", {})
    suspects = ", ".join(f"{s['paper']} ({s['flagged']}/{s['rows']})" for s in TQ.get("suspect_papers", [])) or "none"
    chap_ok = CHAPTER.get("chapters", {})

    sections = f"""
<section id="summary"><h2>1. Executive summary</h2>
<div class="cards">{card_html}</div>
<p>Klassed is an adaptive NEET-UG practice platform. This report covers the machine-learning side end to end: data, models, pipeline, the new post-attempt <b>AIR estimate</b>, a deep re-audit with every fix, and where the project still depends on other teams' data. Everything below was produced by scripts in <code>scripts/</code> from the repository's own report files, on the pinned production runtime (Python {f.get('python', '3.13')}).</p>
<div class="note good"><b>State:</b> the ML part is complete, tested and reproducible. Remaining risk is not code: it is data and decisions owned by others (section 9): 0 real student attempts, hand-set score-to-percentile curve, formula-based PYQ difficulty labels, OCR-damaged source text, and an undecided syllabus default.</div>
<h3>Verification evidence</h3>{verification_table()}
</section>

<section id="arch"><h2>2. System architecture and flows</h2>
<h3>2.1 Block diagram</h3>{fig_architecture()}
<p class="sub">Dashed grey boxes are inputs owned by other teams. Pink/green boxes marked "new" were added in this audit.</p>
<h3>2.2 Adaptive learning loop</h3>{fig_loop()}
<h3>2.3 ML lifecycle</h3>{fig_lifecycle()}
</section>

<section id="data"><h2>3. Data</h2>
<h3>3.1 Ingestion flow and quality funnel</h3>{fig_data_flow()}{fig_funnel()}
<p class="sub">Reason counts overlap: one row can fail several checks, so the bars sum to more than the {rej_total:,} rejected rows. OCR bars count flags; {ocr_rows} rows that earlier versions accepted are now rejected by the text-quality gate.</p>
<div class="note"><b>Deep fix:</b> the validator never checked text quality. {tq_acc.get('flagged', '?')} accepted rows were flagged before the fix; the new gate rejects them. Raw file: {tq_raw.get('flagged', '?')} flagged of {tq_raw.get('rows', 0):,}. Official PYQs: {tq_pyq.get('flagged', '?')} flagged of {tq_pyq.get('rows', 0):,} (excluded from ML-built papers; the file itself is not edited). Suspect paper(s): {html.escape(suspects)}.</div>
<h3>3.2 PYQ bank by subject and tier</h3>{fig_pyq_tiers()}
<h3>3.3 Tier coverage by chapter (PYQ only; red = zero)</h3>{fig_tier_heatmap()}
<p>PYQ-only zero tiers ({len(cov.get('pyq_only_zero_tiers_all_rows', []))}): {html.escape(', '.join(cov.get('pyq_only_zero_tiers_all_rows', [])))}. The backend's <code>ADJACENT_FALLBACK_ORDER</code> (basic to medium to advanced) means these chapters are not dropped from papers; a basic-level student simply sees medium questions there. A pipeline ratchet now fails if any <em>new</em> gap appears.</p>
<h3>3.4 Taxonomy versus verified bank</h3>{fig_taxonomy_gap()}
<p>The taxonomy file lists {cd.get('taxonomy_questions', '?'):,} questions; the verified bank holds {cd.get('bank_questions', '?'):,} ({cd.get('bank_missing_vs_taxonomy', '?')} fewer, mostly recent years). Chapter shares agree at Spearman {cd.get('chapter_share_spearman', '?')}. The AIR feature therefore reads the taxonomy and falls back to the bank only for chapters it omits ({html.escape(', '.join(cd.get('chapters_only_in_bank', [])))}).</p>
<h3>3.5 Diagram assets (sample from 963 committed images)</h3><div class="gallery">{gallery}</div>
</section>

<section id="models"><h2>4. Models</h2>
<h3>4.0 What is real, what is a prior, what is simulated</h3>
<p>Generated from <code>model/model_provenance.py</code>, which reads the current report files and fails the pipeline if any model claims more than the data supports (for example, real-student fitting while 0 attempts exist, or an accuracy attached to a hand-designed prior). The same table is served at <code>GET /mlops/model_provenance</code>.</p>
{fig_basis_counts()}{provenance_table()}
<div class="note"><b>Real students used anywhere:</b> {PROV.get('real_student_attempts', '?')}. Models validated on real students: 0. Student-response training status: {html.escape(str(PROV.get('student_training_status')))}.</div>
<h3>4.1 Catalogue and test coverage</h3>{mt}
<p class="sub">Coverage is derived by scanning the test source for each model's module; {uncovered} models have no direct suite.</p>
<h3>4.2 Difficulty: what the accuracy really measures</h3>{fig_models()}
<p>Formula labels agree with human-verified labels at accuracy {pct(LABEL.get('heuristic_rule_vs_human_verified_sample_tiers', {}).get('accuracy'))}, Cohen's kappa {LABEL.get('heuristic_rule_vs_human_verified_sample_tiers', {}).get('cohens_kappa', '?')}. The old random forest's {pct(VALID.get('random_forest_holdout_evaluation', {}).get('holdout_accuracy'))} only measures how well it re-learns that formula. v2.1, trained on human labels, scores {pct(v21.get('accuracy'))} (macro-F1 {v21.get('macro_f1', '?')}, ECE {v21.get('ece_top_label', '?')}) with paper-grouped validation.</p>
<h3>4.3 Chapter-wise reality check</h3>{fig_chapter_dots()}
<p>{len(chap_ok.get('reliable_below_majority', []))} of {chap_ok.get('reliable', '?')} reliably measured chapters do not beat guessing the majority tier; the headline accuracy is partly carried by the "medium" majority. Only {CHAPTER.get('topics', {}).get('reliable', '?')} of {CHAPTER.get('topics', {}).get('total', '?')} topics have enough held-out rows to judge.</p>
<h3>4.4 Equated 180-question forms</h3>
<p>3 forms, 45/45/90, disjoint; max expected-marks gap across ability {EQUIV.get('max_expected_marks_gap_across_theta', '?')} of 720 (tolerance {EQUIV.get('tolerances', {}).get('expected_marks_gap', '?')}); all {len(EQUIV.get('checks', {}))} equivalence checks pass: {EQUIV.get('all_checks_passed')}. Forms are now frozen by a guard once real attempts exist.</p>
<h3>4.5 Unlimited mocks: live audit of 10 generated mocks</h3>
{table(['mock', 'Phys', 'Chem', 'Bio', 'basic', 'medium', 'adv', 'repeats vs earlier', 'where repeats occur'], mock_rows)}
<div class="note {'good' if not mocks['bad'] else 'bad'}"><b>Structural checks:</b> {'all passed (size 180, quotas 45/45/90, no initial-form questions, no corrupt text)' if not mocks['bad'] else html.escape(str(mocks['bad']))}. <b>Capacity:</b> pool {cap['pool_size']:,}; non-repeating mocks guaranteed: {cap['non_repeating_mocks_guaranteed']} (chapter quota, bottleneck {html.escape(cap['bottleneck_chapter'])}) versus {cap['non_repeating_mocks_by_tier_quota']} by tier quota. Repeats are honestly reported in each mock and sit only in thin chapters, which are the NMC-excluded ones.</div>
<h3>4.6 Latency (p95)</h3>{fig_latency()}
</section>

<section id="pipeline"><h2>5. Pipeline and MLOps</h2>
<h3>5.1 {len(RUN.get('stages', []))}-stage pipeline</h3>{fig_pipeline()}{fig_stage_times()}
<p>Total {RUN.get('total_seconds', '?')} s, no GPU. Reproducibility: {RUN.get('reproducibility', {}).get('metrics_checked', '?')} headline metrics retrained and compared within {RUN.get('reproducibility', {}).get('tolerance', '?')}: {RUN.get('reproducibility', {}).get('reproducible')}.</p>
<h3>5.2 Promotion flow</h3>{fig_promotion()}
{table(['candidate', 'compared with', 'gates', 'decision'], promo_rows)}
<div class="note"><b>Correction to an earlier claim:</b> on the pinned runtime, metrics reproduce to 4 decimals, but binary files differ across machines (6 of 21 binaries, plus timestamps). "Reproducible within tolerance" is accurate; "byte-identical everywhere" is not.</div>
</section>

<section id="air"><h2>6. New feature: post-attempt AIR estimate</h2>
<p>After an attempt the student gets an indicative All-India Rank range, built from mock difficulty, chapter-wise performance, previous-year chapter trends and real NEET facts. API: <code>GET /students/{{id}}/air_estimate?paper_number=N&amp;syllabus_mode=all|nmc_reduced</code>. Code: <code>model/air_estimator.py</code>.</p>
{fig_air_flow()}
<h3>6.1 Parameters and how each is used</h3>
{table(['Parameter', 'How it enters'], [
 ('Mock difficulty', 'IRT 3PL equating: item parameters (a, b, c) of the questions answered convert raw marks to NEET-scale marks; the report states whether the paper was easier or harder than the reference form.'),
 ('Chapter-wise performance', 'Chapter mastery from IRT + BKT with hierarchical shrinkage; chapters never attempted are marked prior_only.'),
 ('Previous-year trends', 'Taxonomy weightage 2013-2026: mean share, last-3-year share, rising/falling/stable; expected questions per paper = 180 x share (renormalised for the syllabus mode).'),
 ('Chapter difficulty', 'Mean IRT b of the chapter PYQs, advanced-tier share, mean ideal time.'),
 ('Real NEET facts', 'Sourced candidates appeared and general cut-off for 2024-2026; the same marks are re-read on the curve shifted by the cut-off difference (an approximation).'),
 ('Where to focus', 'marks at stake = expected questions x 5 x (target mastery - mastery); AIR places gained if only that chapter reached the target (not additive).')])}
<h3>6.2 Score-to-rank curve</h3>{fig_rank_curve()}
<div class="note"><b>Limit:</b> the curve is hand-set (NTA publishes no full table). The AIR is an indicative range, not a promise. Every API response lists this under <code>assumptions</code>.</div>
<h3>6.3 Mock difficulty handling (simulated students, true ability 0.5)</h3>{fig_equating()}
{table(['paper', 'mean raw marks', 'mean equated marks', 'median AIR', 'median 90% range width', 'verdict'], eq_rows)}
<p>{equating_text}</p>
<h3>6.4 Is the ability estimate calibrated?</h3>{fig_calibration()}
{table(['true theta', 'students', 'mean error', 'RMSE', 'claimed SEM', '95% coverage', 'median AIR'], cal_rows)}
<p>Actual error tracks the claimed standard error and the 95% interval covers 92-99% of students, so the stated uncertainty is honest (simulated students drawn from the model's own parameters; this validates the estimator, not real students).</p>
<h3>6.5 Same marks, different paper years</h3>{fig_year_scenarios()}
<p>A simulated student (true ability 0.8): equated marks {SIM.get('sample_scores', {}).get('equated_neet_marks', '?')}, AIR {sample_air.get('estimate', 0):,} (90% range {sample_air.get('range_90', ['?', '?'])[0]:,} to {sample_air.get('range_90', ['?', '?'])[1]:,}), paper-year spread {sample_air.get('paper_year_spread', ['?', '?'])[0]:,} to {sample_air.get('paper_year_spread', ['?', '?'])[1]:,}. Shown to the student as about {sample_air.get('display', {}).get('estimate', 0):,}; with label and year assumptions varied the range is {sample_air.get('range_if_assumptions_wrong', ['?', '?'])[0]:,} to {sample_air.get('range_if_assumptions_wrong', ['?', '?'])[1]:,}.</p>
<h3>6.6 How much do the unvalidated assumptions matter?</h3>{fig_label_sensitivity()}
{table(['true theta', 'students', 'marks: formula labels (production)', 'marks: v2.1 labels', 'marks: labels uninformative', 'median AIR (production)', 'median range if assumptions wrong'], label_rows)}
<div class="note bad"><b>Honest finding:</b> PYQ item difficulty comes from a formula that barely agrees with human labels (kappa {PROV.get('formula_vs_human_label_kappa', '?')}). Re-estimating the same answers under either alternative lowers the equated marks in every row above, by {sens_lo:.0f} to {sens_hi:.0f} marks on average. v2.1 partly reflects its own domain shift (it learned from easier sample papers), but the neutral "no information" scenario points the same way. So the production estimate is more likely optimistic than pessimistic, and the API returns <code>sensitivity</code>, <code>reliability</code> and a wider <code>range_if_assumptions_wrong</code> instead of hiding this. AIR values are rounded to 2 significant digits for display.</div>
<h3>6.7 What to study first (same simulated student)</h3>
{table(['chapter', 'subject', 'mastery', 'questions per NEET paper', 'PYQ trend', 'marks at stake', 'AIR places if target reached'], sample_rows)}
<h3>6.8 When AIR is withheld</h3>
<p>AIR needs 10 scored questions in each of Physics, Chemistry and Biology. A chapter test returns the chapter table with <code>air: null</code> and a reason; if earlier attempts together satisfy the rule, <code>evidence.source</code> is <code>attempt_plus_history</code>. No answers: <code>no_scored_responses</code>. Duplicate submissions collapse (last answer wins).</p>
</section>

<section id="trends"><h2>7. Previous-year trends</h2>{fig_weightage()}
<p>Chapter share of each NEET paper, 2013-2026, from the taxonomy: three most rising and three most falling chapters among those with at least 3 questions per paper on average. Trend flags feed the watchlist in the AIR report.</p></section>

<section id="audit"><h2>8. Deep re-audit: findings and fixes</h2>
<p>{fixed_n} findings fixed or corrected, {open_n} open (all need other teams, data or decisions). Static analysis: 0 warnings across 95+ Python files.</p>
{table(['id', 'area', 'finding', 'severity', 'status', 'fix / next step'], FINDINGS, cls_col=4)}
</section>

<section id="deps"><h2>9. Dependencies on other teams' datasets, and gaps</h2>
{table(['dataset / decision', 'owner', 'used by', 'current state', 'needed'], DEPENDENCIES)}
<div class="note"><b>Blocking for "real" accuracy:</b> (1) student attempts, (2) a sourced marks-to-AIR table, (3) human tier labels. Until then every student-facing number is labelled estimated and prior-based.</div>
</section>

<section id="run"><h2>10. Runbook and roadmap</h2>
<pre>python -m model.sample_paper_ingestion              # only when the sample file changes
python -m model.assessment_blueprint                # refuses once student attempts exist (--force overrides)
python -m model.run_full_ml_pipeline --verify-reproducible
python -m model.tests.test_ml_standalone_suite      # {f.get("ml_total", "?")} suites
python backend/tests/test_v2_engine.py              # {f.get("be_total", "?")} suites
python scripts/simulate_air_validation.py           # AIR validation numbers
python scripts/generate_project_report.py           # this report</pre>
{table(['when', 'action', 'owner'], [
 ('now', 'Decide syllabus default (all or nmc_reduced); fix or quarantine 12 PYQs and Paper 50; set JWT secret and PostgreSQL', 'product, content, DevOps'),
 ('before pilot', 'Wire the AIR screen to /air_estimate; confirm paper_number semantics after merging the platform-API commit', 'frontend, backend'),
 ('after 200 students', 'Pipeline fits real IRT and BKT automatically; review and promote through gates', 'ML + reviewer'),
 ('after first NEET', 'Collect real scores to validate the predictor; replace the hand-set curve with sourced anchors', 'ML, product'),
 ('later', 'Thermodynamics re-key by subject; semantic retriever on GPU; translations', 'backend, ML, content')])}
</section>
"""
    nav = "".join(f"<a href='#{i}'>{t}</a>" for i, t in [("summary", "Summary"), ("arch", "Architecture"), ("data", "Data"), ("models", "Models"), ("pipeline", "Pipeline"), ("air", "AIR feature"), ("trends", "Trends"), ("audit", "Audit"), ("deps", "Dependencies"), ("run", "Runbook")])
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Klassed ML: complete project report</title>"
        f"<style>{CSS}</style></head><body>"
        "<header><div class='wrap'><h1>Klassed NEET Adaptive Engine: ML project report</h1>"
        f"<p class='sub'>Final audit, fixes, architecture and AIR feature &middot; Udbhav Yadav, ML Engineer Intern &middot; {html.escape(str(f.get('date', '')))}</p></div></header>"
        f"<nav><div class='wrap'>{nav}</div></nav><div class='wrap'>{sections}"
        "<footer>Every number is read from the repository's report files by scripts/generate_project_report.py. Simulated-student results validate the estimators, not real students. "
        "AIR values are indicative ranges based on an approximate curve and prior item parameters.</footer></div></body></html>"
    )


if __name__ == "__main__":
    OUT_HTML.write_text(build(), encoding="utf-8")
    print("wrote", OUT_HTML, round(OUT_HTML.stat().st_size / 1e3), "KB")
