from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from .difficulty_model_v21_text import N_FOLDS, TIERS, _ordered_proba, build_v21_pipeline, featurize_with_text
from .pyq_weightage import cross_check_with_bank, pyq_chapter_weightage, taxonomy_chapter_weightage
from .sample_paper_ingestion import load_accepted_sample_rows
from .text_quality_gate import is_text_corrupt

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
PYQ_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
KNOWN_ISSUES_PATH = PROJECT_ROOT / "data" / "config" / "known_data_issues.json"
TAXONOMY_PATH = PROJECT_ROOT / "data" / "taxonomy" / "NEET_Unified_Weightage_And_Marks_2013_2026.json"
REPORT_PATH = MODEL_DIR / "chapter_topic_evaluation_report.json"
CHAPTER_CSV = PROJECT_ROOT / "docs" / "chapter_wise_difficulty_metrics.csv"
TOPIC_CSV = PROJECT_ROOT / "docs" / "topic_wise_difficulty_metrics.csv"
COVERAGE_CSV = PROJECT_ROOT / "docs" / "chapter_tier_coverage.csv"

MIN_SUPPORT_CHAPTER = 30
MIN_SUPPORT_TOPIC = 20


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray, keys: list[str], min_support: int) -> list[dict]:
    idx = defaultdict(list)
    for i, k in enumerate(keys):
        idx[k].append(i)
    out = []
    for k, ii in idx.items():
        ii = np.array(ii)
        t, p = y_true[ii], y_pred[ii]
        n = len(ii)
        correct = int((t == p).sum())
        majority = Counter(t.tolist()).most_common(1)[0][1]
        out.append(
            {
                "group": k,
                "n": n,
                "accuracy": round(correct / n, 4),
                "accuracy_ci95": wilson(correct, n),
                "macro_f1": round(float(f1_score(t, p, average="macro", labels=sorted(set(t.tolist())), zero_division=0)), 4),
                "majority_baseline": round(majority / n, 4),
                "beats_majority": correct > majority,
                "tier_mix": dict(Counter(t.tolist())),
                "reliability": "ok" if n >= min_support else "insufficient_support",
            }
        )
    return sorted(out, key=lambda r: (-r["n"], r["group"]))


def out_of_fold_predictions() -> tuple[list[dict], np.ndarray, np.ndarray]:
    rows, _ = load_accepted_sample_rows()
    df = featurize_with_text(rows)
    labels = df["label"].to_numpy()
    splits = list(GroupKFold(n_splits=N_FOLDS).split(df, labels, groups=df["group"]))
    proba = np.zeros((len(df), 3))
    for tr, te in splits:
        proba[te] = _ordered_proba(build_v21_pipeline().fit(df.iloc[tr], labels[tr]), df.iloc[te])
    pred = np.array([TIERS[i] for i in proba.argmax(axis=1)])
    return rows, labels, pred


def chapter_tier_coverage(pyq: list[dict], sample: list[dict]) -> list[dict]:
    cells = defaultdict(lambda: {"pyq": Counter(), "sample": Counter(), "subject": ""})
    for source, rows in (("pyq", pyq), ("sample", sample)):
        for r in rows:
            c = cells[r["chapter"]]
            c["subject"] = r.get("subject", c["subject"])
            c[source][str(r.get("difficulty_level", "")).lower()] += 1
    out = []
    for chapter, c in sorted(cells.items(), key=lambda kv: (kv[1]["subject"], kv[0])):
        row = {"subject": c["subject"], "chapter": chapter}
        zero_union, zero_pyq = [], []
        for t in TIERS:
            row[f"pyq_{t}"] = c["pyq"][t]
            row[f"sample_{t}"] = c["sample"][t]
            row[f"total_{t}"] = c["pyq"][t] + c["sample"][t]
            if row[f"total_{t}"] == 0:
                zero_union.append(t)
            if c["pyq"][t] == 0:
                zero_pyq.append(t)
        row["zero_tiers_pyq_plus_sample"] = ",".join(zero_union)
        row["zero_tiers_pyq_only"] = ",".join(zero_pyq)
        out.append(row)
    return out


def pyq_zero_tier_keys(pyq: list[dict]) -> list[str]:
    cells = defaultdict(Counter)
    subject = {}
    for r in pyq:
        cells[r["chapter"]][str(r.get("difficulty_level", "")).lower()] += 1
        subject.setdefault(r["chapter"], r["subject"])
    return sorted(f"{subject[ch]}::{ch}::{t}" for ch, c in cells.items() for t in TIERS if c[t] == 0)


def multi_subject_chapters(pyq: list[dict], sample: list[dict]) -> dict[str, list[str]]:
    seen = defaultdict(set)
    for r in list(pyq) + list(sample):
        seen[r["chapter"]].add(r["subject"])
    return {ch: sorted(s) for ch, s in sorted(seen.items()) if len(s) > 1}


def cross_dataset_gate(cross: dict, known: dict) -> dict:
    limits = known.get("taxonomy_vs_bank", {})
    max_missing = limits.get("max_bank_missing_questions")
    min_rho = limits.get("min_chapter_share_spearman")
    problems = []
    if max_missing is not None and cross["bank_missing_vs_taxonomy"] > max_missing:
        problems.append(f"bank is missing {cross['bank_missing_vs_taxonomy']} questions vs taxonomy (baseline {max_missing})")
    if min_rho is not None and cross["chapter_share_spearman"] < min_rho:
        problems.append(f"chapter share agreement {cross['chapter_share_spearman']} below {min_rho}")
    return {"problems": problems, "passed": not problems}


def ratchet(current_zero: list[str], current_multi: list[str], known: dict) -> dict:
    known_zero = set(known.get("pyq_zero_tier_chapters", []))
    known_multi = set(known.get("chapters_in_multiple_subjects", []))
    new_zero = sorted(set(current_zero) - known_zero)
    new_multi = sorted(set(current_multi) - known_multi)
    return {
        "new_zero_tiers": new_zero,
        "resolved_zero_tiers": sorted(known_zero - set(current_zero)),
        "known_zero_tiers_still_open": sorted(known_zero & set(current_zero)),
        "new_multi_subject_chapters": new_multi,
        "known_multi_subject_chapters_still_open": sorted(known_multi & set(current_multi)),
        "passed": not new_zero and not new_multi,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list, tuple)) else v) for k, v in r.items()})


def run(save: bool = True) -> dict:
    rows, y, pred = out_of_fold_predictions()
    by_subject = group_metrics(y, pred, [r["subject"] for r in rows], MIN_SUPPORT_CHAPTER)
    by_chapter = group_metrics(y, pred, [f'{r["subject"]}::{r["chapter"]}' for r in rows], MIN_SUPPORT_CHAPTER)
    by_topic = group_metrics(y, pred, [f'{r["subject"]}::{r["chapter"]}::{r.get("topic_tag") or "untagged"}' for r in rows], MIN_SUPPORT_TOPIC)

    pyq = json.loads(PYQ_PATH.read_text(encoding="utf-8"))
    pyq_served = [r for r in pyq if not is_text_corrupt(r)]
    coverage = chapter_tier_coverage(pyq, rows)
    weightage, weightage_meta = pyq_chapter_weightage(pyq)
    known = json.loads(KNOWN_ISSUES_PATH.read_text(encoding="utf-8")) if KNOWN_ISSUES_PATH.exists() else {}
    zero_all = pyq_zero_tier_keys(pyq)
    zero_served = pyq_zero_tier_keys(pyq_served)
    multi = multi_subject_chapters(pyq, rows)
    gate = ratchet(zero_all, list(multi), known)
    taxonomy_rows, taxonomy_meta = taxonomy_chapter_weightage(json.loads(TAXONOMY_PATH.read_text(encoding="utf-8")))
    cross = cross_check_with_bank(taxonomy_rows, taxonomy_meta, pyq)
    cross_gate = cross_dataset_gate(cross, known)

    reliable_chapters = [c for c in by_chapter if c["reliability"] == "ok"]
    reliable_topics = [t for t in by_topic if t["reliability"] == "ok"]
    report = {
        "model": "difficulty v2.1 (TF-IDF + logistic regression), pipeline from model/difficulty_model_v21_text.py",
        "evaluation": f"out-of-fold, GroupKFold({N_FOLDS}) by paper_source, no paper in both train and test",
        "rows": len(rows),
        "overall_accuracy": round(float((y == pred).mean()), 4),
        "overall_accuracy_ci95": wilson(int((y == pred).sum()), len(y)),
        "chapters": {
            "total": len(by_chapter),
            "reliable": len(reliable_chapters),
            "reliable_below_majority": [c["group"] for c in reliable_chapters if not c["beats_majority"]],
            "worst_reliable": sorted(reliable_chapters, key=lambda c: c["accuracy"])[:5],
        },
        "topics": {
            "total": len(by_topic),
            "reliable": len(reliable_topics),
            "note": f"topics with fewer than {MIN_SUPPORT_TOPIC} held-out rows are listed but not trustworthy",
        },
        "coverage": {
            "chapters": len(coverage),
            "pyq_only_zero_tiers_all_rows": zero_all,
            "pyq_only_zero_tiers_ml_served_rows": zero_served,
            "zero_tiers_pyq_plus_sample": [f'{c["subject"]}::{c["chapter"]}::{c["zero_tiers_pyq_plus_sample"]}' for c in coverage if c["zero_tiers_pyq_plus_sample"]],
            "chapters_in_multiple_subjects": multi,
            "ratchet": gate,
        },
        "cross_dataset": {**cross, "gate": cross_gate},
        "pyq_weightage": weightage_meta,
        "by_subject": by_subject,
        "by_chapter": by_chapter,
        "by_topic": by_topic,
        "pyq_chapter_weightage": weightage,
        "not_measured": "student-level chapter mastery accuracy: needs real student attempts, 0 exist today",
    }
    if save:
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        cols = ["group", "n", "accuracy", "accuracy_ci95", "macro_f1", "majority_baseline", "beats_majority", "tier_mix", "reliability"]
        write_csv(CHAPTER_CSV, by_chapter, cols)
        write_csv(TOPIC_CSV, by_topic, cols)
        write_csv(COVERAGE_CSV, coverage, list(coverage[0].keys()))
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("rows", "overall_accuracy", "overall_accuracy_ci95", "chapters", "topics", "coverage")}, indent=2))
