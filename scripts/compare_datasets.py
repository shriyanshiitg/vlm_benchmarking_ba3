"""
compare_datasets.py
-------------------
Compares the auto-generated VQA dataset against the manually curated one.

Produces:
  - Per-patient coverage summary (which anatomical structures are covered)
  - Closed/open split comparison
  - Series mapping agreement between the two datasets
  - Side-by-side question listing for manual review

Usage:
    python scripts/compare_datasets.py
    python scripts/compare_datasets.py \\
        --manual   data/clinical_vqa_dataset.jsonl \\
        --generated data/auto_generated_vqa_dataset.jsonl
"""

import os
import sys
import json
import argparse
from collections import defaultdict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        sys.exit(f"ERROR: File not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def short_id(study_id: str) -> str:
    return f"...{study_id[-10:]}" if len(study_id) > 10 else study_id


def group_by_study(records: list) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[r.get("study_id", "UNKNOWN")].append(r)
    return dict(groups)


def print_separator(char="=", width=72):
    print(char * width)


# ── Core analysis ──────────────────────────────────────────────────────────────

def summarise_dataset(records: list, label: str) -> None:
    """Prints a high-level summary of a dataset."""
    total   = len(records)
    closed  = sum(1 for r in records if r.get("answer_type") == "CLOSED")
    open_   = total - closed
    by_type = defaultdict(int)
    by_series = defaultdict(int)
    for r in records:
        by_type[r.get("target_series_type", "?")] += 1
        by_series[r.get("target_series", "?")] += 1

    print(f"\n{label} ({total} total)")
    print(f"  Closed (Yes/No): {closed}  |  Open: {open_}")
    print(f"  By series type:")
    for k, v in sorted(by_type.items()):
        print(f"    {k:20s}: {v}")
    print(f"  By series number (target_series):")
    for k, v in sorted(by_series.items(), key=lambda x: (x[0] or "99")):
        print(f"    Series {str(k):>4s}: {v} questions")


def compare_study(study_id: str, manual_qs: list, gen_qs: list) -> None:
    """Detailed side-by-side comparison for a single study."""
    print(f"\nPatient {short_id(study_id)}")
    print_separator("-", 72)

    # Closed/open counts
    m_closed = sum(1 for q in manual_qs if q.get("answer_type") == "CLOSED")
    g_closed = sum(1 for q in gen_qs   if q.get("answer_type") == "CLOSED")
    print(f"  QA count  — Manual: {len(manual_qs)}  |  Generated: {len(gen_qs)}")
    print(f"  Closed    — Manual: {m_closed}  |  Generated: {g_closed}")
    print(f"  Open      — Manual: {len(manual_qs) - m_closed}  |  Generated: {len(gen_qs) - g_closed}")

    # Anatomical structures covered
    m_structs = {q.get("anatomical_structure", "?") for q in manual_qs}
    g_structs = {q.get("anatomical_structure", "?") for q in gen_qs}
    print(f"\n  Structures covered:")
    print(f"    Manual:    {sorted(m_structs)}")
    print(f"    Generated: {sorted(g_structs)}")
    only_manual = m_structs - g_structs
    only_gen    = g_structs - m_structs
    if only_manual:
        print(f"    ⚠  In manual only:    {sorted(only_manual)}")
    if only_gen:
        print(f"    ⚠  In generated only: {sorted(only_gen)}")

    # Series mapping comparison — keyed by target_series_type
    # Note: manual dataset stores target_series (number) but not target_series_type,
    # so we key by the number directly for the manual side
    m_series_map = {(q.get("target_series_type") or q.get("target_series") or "?"): q.get("target_series") for q in manual_qs}
    g_series_map = {}
    for q in gen_qs:
        t = q.get("target_series_type")
        s = q.get("target_series")
        if t not in g_series_map:
            g_series_map[t] = s

    print(f"\n  Series mapping comparison:")
    all_types = sorted(set(m_series_map) | set(g_series_map), key=lambda x: x or "")
    for t in all_types:
        m_s = m_series_map.get(t, "—")
        g_s = g_series_map.get(t, "—")
        match = "✓" if m_s == g_s else "✗"
        print(f"    {match} {t:20s}  Manual→{m_s}  Generated→{g_s}")

    # Hallucinatation check: any generated "Yes" where manual says "No" for same structure?
    m_neg_structs = {
        q.get("anatomical_structure", "").lower()
        for q in manual_qs
        if q.get("answer_type") == "CLOSED" and q.get("answer", "").lower() == "no"
    }
    gen_pos_structs = {
        q.get("anatomical_structure", "").lower()
        for q in gen_qs
        if q.get("answer_type") == "CLOSED" and q.get("answer", "").lower() == "yes"
    }
    hallucinated = m_neg_structs & gen_pos_structs
    if hallucinated:
        print(f"\n  ⚠  POTENTIAL HALLUCINATIONS (generated 'Yes' where manual says 'No'):")
        for s in sorted(hallucinated):
            print(f"    • {s}")
    else:
        print("\n  ✓  No polarity conflicts detected.")

    # Full question listing
    print(f"\n  {'—'*68}")
    print(f"  MANUAL questions:")
    for i, q in enumerate(manual_qs, 1):
        print(f"    {i:2d}. [{q.get('answer_type', '?'):6s}] [{q.get('target_series_type','?'):16s}]")
        print(f"        Q: {q.get('question','')}")
        print(f"        A: {q.get('answer','')}")

    print(f"\n  GENERATED questions:")
    for i, q in enumerate(gen_qs, 1):
        series_tag = (
            f"→ Series {q.get('target_series','?')}"
            if q.get("target_series")
            else "→ Series UNMAPPED ⚠"
        )
        print(
            f"    {i:2d}. [{q.get('answer_type', '?'):6s}] "
            f"[{q.get('target_series_type','?'):16s}] {series_tag}"
        )
        print(f"        Q: {q.get('question','')}")
        print(f"        A: {q.get('answer','')}")


# ── Validation checks ──────────────────────────────────────────────────────────

def validate_generated(gen_records: list) -> None:
    """Runs hard checks on generated records and prints a pass/fail report."""
    print_separator()
    print("VALIDATION CHECKS on generated dataset")
    print_separator()
    issues = []

    for i, r in enumerate(gen_records):
        prefix = f"  [Row {i+1}, ...{r.get('study_id','?')[-10:]}]"

        if r.get("answer_type") == "CLOSED":
            ans = str(r.get("answer", "")).strip().lower()
            if ans not in ("yes", "no"):
                issues.append(f"{prefix} CLOSED answer is not Yes/No: '{r.get('answer')}'")

        if r.get("target_series") is None:
            issues.append(f"{prefix} target_series is None (series mapping failed)")

        if not r.get("question", "").strip().endswith("?"):
            issues.append(f"{prefix} Question does not end with '?': '{r.get('question','')[:60]}'")

        if r.get("answer_type") == "OPEN":
            words = len(str(r.get("answer", "")).split())
            if words > 10:
                issues.append(f"{prefix} Open answer too long ({words} words): '{r.get('answer')}'")

    if issues:
        print(f"  {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print(f"  ✓ All {len(gen_records)} generated records passed validation.")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare auto-generated VQA dataset to the manual ground-truth dataset."
    )
    parser.add_argument(
        "--manual",
        default=os.path.join(_PROJECT_ROOT, "data", "clinical_vqa_dataset.jsonl"),
    )
    parser.add_argument(
        "--generated",
        default=os.path.join(_PROJECT_ROOT, "data", "auto_generated_vqa_dataset.jsonl"),
    )
    args = parser.parse_args()

    print_separator()
    print("VQA Dataset Comparison: Manual vs Auto-Generated")
    print_separator()
    print(f"  Manual:    {args.manual}")
    print(f"  Generated: {args.generated}")

    manual    = load_jsonl(args.manual)
    generated = load_jsonl(args.generated)

    # High-level summaries
    summarise_dataset(manual,    "MANUAL dataset")
    summarise_dataset(generated, "GENERATED dataset")

    # Per-patient comparison
    print_separator()
    print("PER-PATIENT COMPARISON")
    manual_by_study = group_by_study(manual)
    gen_by_study    = group_by_study(generated)

    all_studies = sorted(set(manual_by_study) | set(gen_by_study))
    for study_id in all_studies:
        m_qs = manual_by_study.get(study_id, [])
        g_qs = gen_by_study.get(study_id, [])
        compare_study(study_id, m_qs, g_qs)

    # Hard validation
    print()
    validate_generated(generated)

    print_separator()
    print("COMPARISON COMPLETE")
    print(
        "Review the output above. If generated questions cover the same structures\n"
        "and have no polarity conflicts, the pipeline is validated."
    )
    print_separator()


if __name__ == "__main__":
    main()
