"""
fewshot_analysis.py
===================
Local analysis script for the few-shot experiment (Notebook 10).

Run AFTER placing all 6 output JSONL files into:
    outputs/_archive/fewshot_experiment/

Expected files:
    gemma3_4b__slake_0shot.jsonl
    gemma3_4b__slake_1shot.jsonl
    gemma3_4b__slake_3shot.jsonl
    llava16_7b__slake_0shot.jsonl
    llava16_7b__slake_1shot.jsonl
    llava16_7b__slake_3shot.jsonl

Usage:
    python3 scripts/fewshot_analysis.py

Outputs:
    docs/report_fewshot_experiment.md    Full analysis report
    results/fig_fewshot_f1.png           F1 line chart (0-shot → 1-shot → 3-shot)
    results/fig_fewshot_closed.png       Closed Acc line chart
    results/fewshot_results.json         All metrics in machine-readable JSON
"""

import json, os, re
import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings("ignore")

# ── Try importing matplotlib (optional — chart generation) ────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print("matplotlib not found — skipping chart generation.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEWSHOT_DIR = os.path.join(BASE, "outputs", "_archive", "fewshot_experiment")
RESULTS_DIR = os.path.join(BASE, "results")
DOCS_DIR    = os.path.join(BASE, "docs")

os.makedirs(FEWSHOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# File registry: (model_short, n_shot) -> filename
FILES = {
    ("Gemma-3-4B",   0): "gemma3_4b__slake_0shot.jsonl",
    ("Gemma-3-4B",   1): "gemma3_4b__slake_1shot.jsonl",
    ("Gemma-3-4B",   3): "gemma3_4b__slake_3shot.jsonl",
    ("LLaVA-1.6-7B", 0): "llava16_7b__slake_0shot.jsonl",
    ("LLaVA-1.6-7B", 1): "llava16_7b__slake_1shot.jsonl",
    ("LLaVA-1.6-7B", 3): "llava16_7b__slake_3shot.jsonl",
}

# Reference baselines from the full-dataset SLAKE evaluation (Section 8)
# These are used for comparison in the discussion but NOT as substitutes for
# the 0-shot runs — the 0-shot runs on the 200-sample subset may differ slightly.
FULL_DATASET_BASELINES = {
    "Gemma-3-4B":   {"f1": 0.4214, "closed_acc": 0.6827, "f1_open": 0.2450},
    "LLaVA-1.6-7B": {"f1": 0.3698, "closed_acc": 0.5841, "f1_open": 0.2512},
    # MedGemma-4B is included for reference context
    "MedGemma-4B":  {"f1": 0.7050, "closed_acc": 0.8558, "f1_open": 0.5581},
}

SMOOTHING = SmoothingFunction().method1


# ---------------------------------------------------------------------------
# Metric helpers  (same as scot_extension_analysis.py)
# ---------------------------------------------------------------------------

def norm(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def token_f1(pred: str, gt: str) -> float:
    p, g = norm(pred).split(), norm(gt).split()
    if not p or not g:
        return 0.0
    pc, gc = Counter(p), Counter(g)
    common = sum((pc & gc).values())
    if not common:
        return 0.0
    pr = common / len(p)
    rc = common / len(g)
    return 2 * pr * rc / (pr + rc)


def bleu4(pred: str, gt: str) -> float:
    h = norm(pred).split()
    r = norm(gt).split()
    if not h or not r:
        return 0.0
    return sentence_bleu([r], h, smoothing_function=SMOOTHING)


def closed_correct(pred: str, gt: str) -> bool:
    pn, gn = norm(pred), norm(gt)
    if pn == gn:
        return True
    for w in ("yes", "no"):
        if w in pn and w in gn:
            return True
    return False


def compute_metrics(records: list) -> dict:
    valid  = [r for r in records if "error" not in r]
    closed = [r for r in valid if r.get("is_closed")]
    open_r = [r for r in valid if not r.get("is_closed")]

    f1s   = [token_f1(r["prediction"], r["ground_truth"]) for r in valid]
    f1c   = [token_f1(r["prediction"], r["ground_truth"]) for r in closed]
    f1o   = [token_f1(r["prediction"], r["ground_truth"]) for r in open_r]
    bleus = [bleu4(r["prediction"], r["ground_truth"])    for r in valid]
    c_acc = [1 if closed_correct(r["prediction"], r["ground_truth"]) else 0
             for r in closed]

    # Per question-type breakdown — field is 'content_type' in Kaggle output
    qtypes = {}
    for qtype in ["Modality", "Organ", "Abnormality"]:
        # Match both short ('MOD') and full ('Modality') forms
        qt_recs = [r for r in valid if
                   r.get("content_type", r.get("q_type", "")).strip() == qtype or
                   r.get("content_type", r.get("q_type", "")).strip().upper() == qtype[:3].upper()]
        if qt_recs:
            qtypes[qtype] = float(np.mean(
                [token_f1(r["prediction"], r["ground_truth"]) for r in qt_recs]
            ))

    return {
        "n":          len(valid),
        "n_closed":   len(closed),
        "n_open":     len(open_r),
        "f1":         float(np.mean(f1s))   if f1s   else 0.0,
        "f1_closed":  float(np.mean(f1c))   if f1c   else 0.0,
        "f1_open":    float(np.mean(f1o))   if f1o   else 0.0,
        "closed_acc": float(np.mean(c_acc)) if c_acc else 0.0,
        "bleu":       float(np.mean(bleus)) if bleus else 0.0,
        "qtypes":     qtypes,
        "records":    valid,
    }


# ---------------------------------------------------------------------------
# Paired permutation test
# ---------------------------------------------------------------------------

def paired_permutation_test(scores_a, scores_b, n_iter=10_000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    observed = abs(np.mean(a) - np.mean(b))
    count = 0
    for _ in range(n_iter):
        mask   = rng.integers(0, 2, n).astype(bool)
        perm_a = np.where(mask, a, b)
        perm_b = np.where(mask, b, a)
        if abs(np.mean(perm_a) - np.mean(perm_b)) >= observed:
            count += 1
    return count / n_iter


def fmt_p(p: float) -> str:
    if p < 0.001: return f"{p:.4f} ★★★"
    if p < 0.01:  return f"{p:.4f} ★★"
    if p < 0.05:  return f"{p:.4f} ★"
    return f"{p:.4f} ns"


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def delta_str(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f} pp"


# ---------------------------------------------------------------------------
# Load all files
# ---------------------------------------------------------------------------

print("=" * 65)
print("Few-Shot Experiment — Local Analysis")
print("=" * 65)

results  = {}  # (model_short, n_shot) -> metrics dict
missing  = []

for (model, n_shot), filename in FILES.items():
    path = os.path.join(FEWSHOT_DIR, filename)
    if not os.path.exists(path):
        missing.append(f"  MISSING: {filename}  <- needed for {model} {n_shot}-shot")
        continue
    records = [json.loads(l) for l in open(path, encoding="utf-8")]
    m = compute_metrics(records)
    results[(model, n_shot)] = m
    print(f"  {model:16s} {n_shot}-shot  N={m['n']:3d}  "
          f"F1={pct(m['f1'])}  ClosedAcc={pct(m['closed_acc'])}  OpenF1={pct(m['f1_open'])}")

if missing:
    print("\nMISSING FILES (run the Kaggle notebook for these configs first):")
    for m in missing:
        print(m)

if not results:
    print("\nNo results yet. Exiting.")
    exit(0)

# ---------------------------------------------------------------------------
# Significance tests: 0-shot vs 1-shot, 0-shot vs 3-shot (per model)
# ---------------------------------------------------------------------------

print("\nRunning significance tests ...")
sig = {}  # (model, "0v1") or (model, "0v3") -> p_f1

MODELS = ["Gemma-3-4B", "LLaVA-1.6-7B"]
for model in MODELS:
    for comparison in [("0v1", 0, 1), ("0v3", 0, 3)]:
        label, n_a, n_b = comparison
        if (model, n_a) not in results or (model, n_b) not in results:
            continue

        recs_a = results[(model, n_a)]["records"]
        recs_b = results[(model, n_b)]["records"]

        # Align by idx
        by_idx_a = {r["idx"]: r for r in recs_a}
        by_idx_b = {r["idx"]: r for r in recs_b}
        common   = sorted(set(by_idx_a) & set(by_idx_b))

        f1s_a = [token_f1(by_idx_a[i]["prediction"], by_idx_a[i]["ground_truth"]) for i in common]
        f1s_b = [token_f1(by_idx_b[i]["prediction"], by_idx_b[i]["ground_truth"]) for i in common]

        p = paired_permutation_test(f1s_a, f1s_b)
        sig[(model, label)] = p
        print(f"  {model:16s}  {n_a}-shot vs {n_b}-shot:  p={fmt_p(p)}")

# ---------------------------------------------------------------------------
# Determine conclusion
# ---------------------------------------------------------------------------

# Gap at 0-shot between Gemma-3 and MedGemma (from full dataset baseline)
gap_0shot = {
    "Gemma-3-4B":   FULL_DATASET_BASELINES["MedGemma-4B"]["f1"]
                    - FULL_DATASET_BASELINES["Gemma-3-4B"]["f1"],
    "LLaVA-1.6-7B": FULL_DATASET_BASELINES["MedGemma-4B"]["f1"]
                    - FULL_DATASET_BASELINES["LLaVA-1.6-7B"]["f1"],
}

# Check if few-shot significantly improved either model
def gap_closed(model, threshold=0.30):
    """Returns True if 3-shot F1 reduced the MedGemma gap by >= threshold fraction."""
    if (model, 0) not in results or (model, 3) not in results:
        return None
    f1_0 = results[(model, 0)]["f1"]
    f1_3 = results[(model, 3)]["f1"]
    delta = f1_3 - f1_0
    original_gap = gap_0shot[model]
    fraction_closed = delta / original_gap if original_gap > 0 else 0
    return fraction_closed, delta

conclusions = {}
for model in MODELS:
    gc = gap_closed(model)
    if gc is None:
        conclusions[model] = "insufficient_data"
    elif gc[0] >= 0.30:
        conclusions[model] = "gap_substantially_closed"
    elif gc[1] > 0.01 and sig.get((model, "0v3"), 1.0) < 0.05:
        conclusions[model] = "significant_improvement_gap_not_closed"
    elif gc[1] > 0:
        conclusions[model] = "marginal_improvement_ns"
    else:
        conclusions[model] = "no_improvement_or_degradation"

print(f"\nConclusions:")
for model, c in conclusions.items():
    print(f"  {model:16s}: {c}")

# ---------------------------------------------------------------------------
# Generate charts
# ---------------------------------------------------------------------------

if HAVE_MPL and results:
    shots = [0, 1, 3]
    colors = {"Gemma-3-4B": "#4285F4", "LLaVA-1.6-7B": "#EA4335"}

    # F1 chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (metric_key, metric_label) in zip(axes, [
        ("f1", "Overall Token F1 (%)"),
        ("closed_acc", "Closed-Ended Accuracy (%)"),
    ]):
        for model in MODELS:
            ys = []
            for n in shots:
                if (model, n) in results:
                    ys.append(results[(model, n)][metric_key] * 100)
                else:
                    ys.append(None)
            valid_shots = [s for s, y in zip(shots, ys) if y is not None]
            valid_ys    = [y for y in ys if y is not None]
            ax.plot(valid_shots, valid_ys, "o-", color=colors[model],
                    label=model, linewidth=2, markersize=8)

        # MedGemma reference line (from full dataset)
        ref_val = FULL_DATASET_BASELINES["MedGemma-4B"][
            "f1" if metric_key == "f1" else "closed_acc"
        ] * 100
        ax.axhline(ref_val, color="gray", linestyle="--", linewidth=1.5,
                   label=f"MedGemma-4B (full-set ref, {ref_val:.1f}%)")
        ax.set_xticks([0, 1, 3])
        ax.set_xticklabels(["0-shot", "1-shot", "3-shot"])
        ax.set_ylabel(metric_label)
        ax.set_title(f"SLAKE — {metric_label} vs Shot Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "fig_fewshot_f1.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nChart saved: {chart_path}")

# ---------------------------------------------------------------------------
# Save machine-readable JSON
# ---------------------------------------------------------------------------

json_out = {}
for (model, n_shot), m in results.items():
    key = f"{model}_{n_shot}shot"
    json_out[key] = {
        "model": model, "n_shot": n_shot,
        "n": m["n"], "n_closed": m["n_closed"], "n_open": m["n_open"],
        "f1": round(m["f1"], 4),
        "f1_closed": round(m["f1_closed"], 4),
        "f1_open": round(m["f1_open"], 4),
        "closed_acc": round(m["closed_acc"], 4),
        "bleu": round(m["bleu"], 4),
        "qtypes": {k: round(v, 4) for k, v in m["qtypes"].items()},
        "sig_0v1": round(sig.get((model, "0v1"), 1.0), 4),
        "sig_0v3": round(sig.get((model, "0v3"), 1.0), 4),
    }

json_path = os.path.join(RESULTS_DIR, "fewshot_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_out, f, indent=2)
print(f"JSON saved : {json_path}")

# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------

def conclusion_paragraph(model, conclusion, f1_0, f1_1, f1_3, p_0v1, p_0v3):
    d1 = delta_str(f1_1 - f1_0)
    d3 = delta_str(f1_3 - f1_0)
    gap = gap_0shot.get(model, 0)

    if conclusion == "gap_substantially_closed":
        return (
            f"Few-shot prompting **substantially closes the gap** for {model}. "
            f"Moving from 0-shot to 3-shot improves overall F1 by {d3} "
            f"(p={fmt_p(p_0v3)}), recovering a meaningful fraction of the "
            f"{pct(gap)} gap that separates generalist models from MedGemma-4B "
            f"on SLAKE. This result supports the hypothesis that in-context learning "
            f"can serve as an efficient clinical deployment path for generalist VLMs, "
            f"without any fine-tuning."
        )
    elif conclusion == "significant_improvement_gap_not_closed":
        return (
            f"Few-shot prompting produces a **statistically significant but modest improvement** "
            f"for {model} ({d3}, p={fmt_p(p_0v3)}). While the gain is real, it does not "
            f"substantially close the {pct(gap)} gap to MedGemma-4B. This suggests that "
            f"in-context examples provide some task-format guidance but cannot replicate the "
            f"domain knowledge embedded through medical pre-training."
        )
    elif conclusion == "marginal_improvement_ns":
        return (
            f"Few-shot prompting produces **marginal, non-significant improvement** for {model} "
            f"(3-shot: {d3}, p={fmt_p(p_0v3)}). The {pct(gap)} gap to MedGemma-4B is essentially "
            f"unchanged. The domain bottleneck is architectural, not informational — providing "
            f"examples of correct medical VQA answers cannot substitute for the structural "
            f"knowledge encoded by medical pre-training."
        )
    elif conclusion == "no_improvement_or_degradation":
        return (
            f"Few-shot prompting **does not improve and may slightly degrade** performance for {model} "
            f"(3-shot: {d3}, p={fmt_p(p_0v3)}). The model's in-context learning capability is "
            f"insufficient to benefit from clinical examples in the zero-shot medical VQA setting. "
            f"This confirms that the domain gap is architectural rather than informational."
        )
    else:
        return f"Insufficient data to draw conclusions for {model}."


# Build main results table rows
def make_row(model, n_shot):
    if (model, n_shot) not in results:
        return f"| {model} | {n_shot}-shot | — | — | — | — | — |\n"
    m = results[(model, n_shot)]
    p_label = ""
    if n_shot == 1:
        p = sig.get((model, "0v1"), 1.0)
        ref = results.get((model, 0))
        delta = delta_str(m["f1"] - ref["f1"]) if ref else "—"
        p_label = fmt_p(p)
    elif n_shot == 3:
        p = sig.get((model, "0v3"), 1.0)
        ref = results.get((model, 0))
        delta = delta_str(m["f1"] - ref["f1"]) if ref else "—"
        p_label = fmt_p(p)
    else:
        delta = "—"
        p_label = "—"
    return (
        f"| {model} | {n_shot}-shot | {pct(m['f1'])} | {delta} | "
        f"{pct(m['closed_acc'])} | {pct(m['f1_open'])} | {p_label} |\n"
    )


table_rows = ""
for model in MODELS:
    for n in [0, 1, 3]:
        table_rows += make_row(model, n)

# Per question-type table
qtype_rows = ""
for model in MODELS:
    for n in [0, 1, 3]:
        if (model, n) not in results:
            continue
        qt = results[(model, n)]["qtypes"]
        qtype_rows += (
            f"| {model} | {n}-shot | "
            f"{pct(qt.get('Modality', qt.get('MOD', 0)))} | "
            f"{pct(qt.get('Organ', qt.get('ORG', 0)))} | "
            f"{pct(qt.get('Abnormality', qt.get('ABN', 0)))} |\n"
        )

# Conclusion paragraphs
conclusion_texts = {}
for model in MODELS:
    f1_0 = results.get((model, 0), {}).get("f1", 0.0)
    f1_1 = results.get((model, 1), {}).get("f1", 0.0)
    f1_3 = results.get((model, 3), {}).get("f1", 0.0)
    p_0v1 = sig.get((model, "0v1"), 1.0)
    p_0v3 = sig.get((model, "0v3"), 1.0)
    if conclusions.get(model, "insufficient_data") != "insufficient_data":
        conclusion_texts[model] = conclusion_paragraph(
            model, conclusions[model], f1_0, f1_1, f1_3, p_0v1, p_0v3
        )
    else:
        conclusion_texts[model] = "Insufficient data."

# Overall verdict
all_conclusions = set(conclusions.values())
if "gap_substantially_closed" in all_conclusions:
    overall_verdict = (
        "**In-context learning can partially close the domain gap.** "
        "Few-shot prompting with curated clinical examples provides measurable benefit "
        "for at least one generalist model. This suggests that few-shot prompting is a "
        "viable low-cost adaptation strategy for clinical deployment when fine-tuning "
        "is not available."
    )
elif "no_improvement_or_degradation" in all_conclusions or \
     "marginal_improvement_ns" in all_conclusions:
    overall_verdict = (
        "**In-context learning does not close the domain gap.** "
        "Neither generalist model benefits significantly from clinical few-shot examples. "
        "The performance gap between generalist and medical VLMs is driven by architectural "
        "differences — specifically, the domain-specific knowledge encoded during medical "
        "pre-training — and cannot be bridged through prompt engineering alone. "
        "This validates why medical fine-tuning is mandatory for competitive clinical VQA."
    )
else:
    overall_verdict = (
        "Results are mixed across models. See per-model conclusions below."
    )

md = f"""# Few-Shot Experiment Report
## VLM Medical VQA Benchmark — Section G2.1

**Research Question:** Does in-context learning (few-shot prompting) close the performance
gap between generalist and medical VLMs without fine-tuning?

**Date:** July 2026  
**Notebook:** `notebooks/10_fewshot_experiment.ipynb`

---

## 1. Experimental Setup

### 1.1 Models and Conditions

| Model | Type | Parameters | Shot Conditions |
|---|---|---|---|
| Gemma-3-4B-IT | Generalist | 4B (fp16) | 0-shot, 1-shot, 3-shot |
| LLaVA-1.6-Mistral-7B | Generalist | 7B (4-bit NF4) | 0-shot, 1-shot, 3-shot |

**Reference (not re-run, from full benchmark):**

| Model | Type | Full-Dataset SLAKE F1 |
|---|---|---|
| MedGemma-4B-IT | Medical | 70.50% |

### 1.2 Test Subset — Stratified 200-Sample SLAKE Subset

The 200-sample subset was drawn from the SLAKE EN test split using stratified sampling
across 6 buckets: 3 question content types (Modality, Organ, Abnormality) × 2 answer
types (Closed/Open), targeting approximately 33 samples per bucket. Seed: 42.
No test samples were used as few-shot examples.

### 1.3 Few-Shot Examples

Examples were drawn exclusively from the SLAKE **training split**. One example per
question content type was selected (Modality, Organ, Abnormality), prioritising
questions with short, unambiguous ground-truth answers (1–3 words).

- **1-shot condition:** Uses the Modality exemplar only.
- **3-shot condition:** Uses all three exemplars in order: Modality → Organ → Abnormality.
- Both conditions use the **zero-shot prompt format** for the target question, prepending
  example turns as prior conversation history (user image + question → assistant answer).

### 1.4 Statistical Tests

Paired permutation test (10,000 iterations, seed 42) on matched question indices,
comparing 0-shot vs. 1-shot and 0-shot vs. 3-shot within each model.

---

## 2. Results

### 2.1 Main Results Table

| Model | Condition | Overall F1 | ΔF1 vs 0-shot | Closed Acc | Open F1 | p (vs 0-shot) |
|---|---|---|---|---|---|---|
{table_rows.strip()}

*Statistical significance: ★★★ p<0.001, ★★ p<0.01, ★ p<0.05, ns p≥0.05.*
*Paired permutation test, 10,000 iterations.*

### 2.2 Reference Context — MedGemma-4B on Full SLAKE Dataset

| Model | F1 | Closed Acc | Open F1 |
|---|---|---|---|
| MedGemma-4B (Section 8, full SLAKE) | {pct(FULL_DATASET_BASELINES['MedGemma-4B']['f1'])} | {pct(FULL_DATASET_BASELINES['MedGemma-4B']['closed_acc'])} | {pct(FULL_DATASET_BASELINES['MedGemma-4B']['f1_open'])} |
| Gemma-3-4B (Section 8, full SLAKE) | {pct(FULL_DATASET_BASELINES['Gemma-3-4B']['f1'])} | {pct(FULL_DATASET_BASELINES['Gemma-3-4B']['closed_acc'])} | {pct(FULL_DATASET_BASELINES['Gemma-3-4B']['f1_open'])} |
| LLaVA-1.6-7B (Section 8, full SLAKE) | {pct(FULL_DATASET_BASELINES['LLaVA-1.6-7B']['f1'])} | {pct(FULL_DATASET_BASELINES['LLaVA-1.6-7B']['closed_acc'])} | {pct(FULL_DATASET_BASELINES['LLaVA-1.6-7B']['f1_open'])} |

*Gap to MedGemma-4B (full-dataset): Gemma-3-4B = {pct(gap_0shot['Gemma-3-4B'])}, LLaVA-1.6-7B = {pct(gap_0shot['LLaVA-1.6-7B'])}.*

### 2.3 Per Question-Type F1 Breakdown

| Model | Condition | Modality F1 | Organ F1 | Abnormality F1 |
|---|---|---|---|---|
{qtype_rows.strip()}

---

## 3. Conclusions

### 3.1 Overall Verdict

{overall_verdict}

### 3.2 Per-Model Analysis

**Gemma-3-4B:**
{conclusion_texts.get('Gemma-3-4B', 'N/A')}

**LLaVA-1.6-7B:**
{conclusion_texts.get('LLaVA-1.6-7B', 'N/A')}

---

## 4. Scientific Interpretation

### 4.1 What This Tells Us About the Domain Gap

The core finding of the prior benchmark work is that domain pre-training trumps parameter
count: MedGemma-4B (4B, medical) comprehensively outperforms LLaVA-1.6-7B (7B, generalist)
on SLAKE despite being 3B parameters smaller. The few-shot experiment asks whether the
informational content of the domain gap — clinical question-answer patterns — can be injected
via in-context learning rather than through fine-tuning.

If few-shot substantially closes the gap: the bottleneck is **informational** — the model has
the visual capability to answer clinical questions but needs format examples to do so correctly.
Few-shot prompting becomes a viable low-cost deployment strategy.

If few-shot does not close the gap: the bottleneck is **architectural** — the missing capability
is the domain-specific visual feature extraction and clinical reasoning learned during medical
pre-training, which examples cannot replicate. Fine-tuning is mandatory.

### 4.2 Relationship to S-CoT Finding

The S-CoT experiment showed that structured prompting degrades MedGemma-4B on SLAKE (−5.0 pp F1,
p < 0.001) but not on VQA-RAD or for HuatuoGPT-7B. The few-shot experiment adds complementary
evidence: it tests whether a different form of prompt enrichment — exemplars rather than structure —
benefits generalist models. Together, the two experiments map the full landscape of what prompt
engineering can and cannot do for medical VQA.

---

## 5. Paper-Ready Paragraph (Section G2 — Few-Shot Experiment)

> We evaluated whether in-context learning can close the performance gap between
> generalist and medical VLMs without fine-tuning. Gemma-3-4B and LLaVA-v1.6-7B were
> evaluated on a stratified 200-sample subset of the SLAKE EN test split under 0-shot,
> 1-shot, and 3-shot conditions. Few-shot examples were drawn exclusively from the
> SLAKE training split, covering one Modality, one Organ, and one Abnormality question
> per condition, using a standard multi-turn conversation format. Statistical significance
> was assessed by paired permutation test (10,000 iterations).
> [INSERT RESULT SUMMARY: e.g., "Neither model showed statistically significant F1
> improvement under few-shot prompting (Gemma-3-4B 3-shot: ΔF1 = X pp, p = Y;
> LLaVA-1.6-7B 3-shot: ΔF1 = X pp, p = Y), indicating that the domain gap is driven
> by architectural differences rather than prompt format, and that medical fine-tuning
> cannot be substituted by in-context learning."]

---

## 6. Generated Files

| File | Description |
|---|---|
| `outputs/_archive/fewshot_experiment/*.jsonl` | Per-run inference outputs (6 files) |
| `results/fewshot_results.json` | All metrics in machine-readable JSON |
| `results/fig_fewshot_f1.png` | F1 line chart: 0-shot → 1-shot → 3-shot |
| `docs/report_fewshot_experiment.md` | This report |
| `scripts/fewshot_analysis.py` | This analysis script (re-runnable) |
| `notebooks/10_fewshot_experiment.ipynb` | Kaggle inference notebook |
"""

out_md = os.path.join(DOCS_DIR, "report_fewshot_experiment.md")
with open(out_md, "w", encoding="utf-8") as f:
    f.write(md)

print(f"\nReport written to: {out_md}")
print("\nNext step: fill in the paper-ready paragraph with your actual numbers.")
print("=" * 65)
