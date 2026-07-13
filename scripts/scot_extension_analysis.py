"""
S-CoT Extension Analysis — Phase 2 (local, post-Kaggle)
========================================================
Run this after you have placed both Kaggle output files into:
  outputs/_archive/scot_experiment/

Computes F1, Closed Acc, Open Acc, BLEU for all 4 S-CoT runs, compares
each to its v2 baseline, runs a paired permutation test for significance,
and writes a full analysis into docs/report_scot_extension.md.

Usage:
    python3 scripts/scot_extension_analysis.py
"""

import json, os, re, math
import numpy as np
from collections import Counter
from scipy.stats import pearsonr
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCOT_FILES = {
    # (model_short, dataset_label): scot_jsonl_path, baseline_jsonl_path
    ("MedGemma-4B", "SLAKE"): {
        "scot":     os.path.join(BASE, "outputs/_archive/scot_experiment",
                                 "google_medgemma-4b-it__slake_scot.jsonl"),
        "baseline": os.path.join(BASE, "outputs/inference",
                                 "google_medgemma-4b-it__slake_v2.jsonl"),
    },
    ("HuatuoGPT-7B", "SLAKE"): {
        "scot":     os.path.join(BASE, "outputs/_archive/scot_experiment",
                                 "FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_scot.jsonl"),
        "baseline": os.path.join(BASE, "outputs/inference",
                                 "FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_7b_v2.jsonl"),
    },
    ("MedGemma-4B", "VQA-RAD"): {
        "scot":     os.path.join(BASE, "outputs/_archive/scot_experiment",
                                 "google_medgemma-4b-it__vqa_rad_scot.jsonl"),
        "baseline": os.path.join(BASE, "outputs/inference",
                                 "google_medgemma-4b-it__vqa_rad_v2.jsonl"),
    },
}

OUT_MD = os.path.join(BASE, "docs", "report_scot_extension.md")

SMOOTHING = SmoothingFunction().method1

# ---------------------------------------------------------------------------
# Metric helpers
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


def compute_metrics(records):
    """Return dict of overall/closed/open F1, closed acc, BLEU."""
    valid = [r for r in records if "error" not in r]
    closed = [r for r in valid if r.get("is_closed")]
    open_r = [r for r in valid if not r.get("is_closed")]

    f1s  = [token_f1(r["prediction"], r["ground_truth"]) for r in valid]
    f1c  = [token_f1(r["prediction"], r["ground_truth"]) for r in closed]
    f1o  = [token_f1(r["prediction"], r["ground_truth"]) for r in open_r]
    bleus = [bleu4(r["prediction"], r["ground_truth"]) for r in valid]
    c_acc = ([1 if closed_correct(r["prediction"], r["ground_truth"]) else 0
               for r in closed] if closed else [])

    return {
        "n":          len(valid),
        "n_closed":   len(closed),
        "n_open":     len(open_r),
        "f1":         float(np.mean(f1s))  if f1s  else 0.0,
        "f1_closed":  float(np.mean(f1c))  if f1c  else 0.0,
        "f1_open":    float(np.mean(f1o))  if f1o  else 0.0,
        "closed_acc": float(np.mean(c_acc)) if c_acc else 0.0,
        "bleu":       float(np.mean(bleus)) if bleus else 0.0,
        "records":    valid,
    }


# ---------------------------------------------------------------------------
# Paired permutation test (two-sided)
# ---------------------------------------------------------------------------

def paired_permutation_test(scores_a, scores_b, n_iter=10_000, seed=42):
    """
    H0: mean(scores_a) == mean(scores_b)
    Returns p-value (two-sided).
    """
    rng = np.random.default_rng(seed)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = min(len(scores_a), len(scores_b))
    scores_a, scores_b = scores_a[:n], scores_b[:n]
    observed = abs(np.mean(scores_a) - np.mean(scores_b))
    count = 0
    for _ in range(n_iter):
        mask   = rng.integers(0, 2, n).astype(bool)
        perm_a = np.where(mask, scores_a, scores_b)
        perm_b = np.where(mask, scores_b, scores_a)
        if abs(np.mean(perm_a) - np.mean(perm_b)) >= observed:
            count += 1
    return count / n_iter


def fmt_p(p):
    if p < 0.001:
        return f"{p:.4f} ***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f} ns"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def load_jsonl(path):
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path, encoding="utf-8")]


print("Loading S-CoT and baseline files...")
results = {}
missing = []

for (model, dataset), paths in SCOT_FILES.items():
    scot_records = load_jsonl(paths["scot"])
    base_records = load_jsonl(paths["baseline"])

    if scot_records is None:
        missing.append(f"{model}/{dataset} SCoT — {paths['scot']}")
        continue
    if base_records is None:
        missing.append(f"{model}/{dataset} baseline — {paths['baseline']}")
        continue

    scot_m = compute_metrics(scot_records)
    base_m = compute_metrics(base_records)

    # Permutation tests on F1 (align indices)
    scot_by_idx = {r["idx"]: r for r in scot_m["records"]}
    base_by_idx = {r["idx"]: r for r in base_m["records"]}
    common_idx  = sorted(set(scot_by_idx) & set(base_by_idx))

    scot_f1s = [token_f1(scot_by_idx[i]["prediction"], scot_by_idx[i]["ground_truth"])
                for i in common_idx]
    base_f1s = [token_f1(base_by_idx[i]["prediction"], base_by_idx[i]["ground_truth"])
                for i in common_idx]

    p_f1 = paired_permutation_test(scot_f1s, base_f1s)

    results[(model, dataset)] = {
        "scot": scot_m,
        "base": base_m,
        "delta_f1":         scot_m["f1"]         - base_m["f1"],
        "delta_closed_acc": scot_m["closed_acc"]  - base_m["closed_acc"],
        "delta_open_f1":    scot_m["f1_open"]     - base_m["f1_open"],
        "delta_bleu":       scot_m["bleu"]         - base_m["bleu"],
        "p_f1":             p_f1,
        "n_paired":         len(common_idx),
    }
    print(f"  {model}/{dataset}: SCoT N={scot_m['n']}  Base N={base_m['n']}  "
          f"ΔF1={scot_m['f1']-base_m['f1']:+.3f}  p={p_f1:.4f}")

if missing:
    print("\nMISSING FILES — run Kaggle notebooks first:")
    for m in missing:
        print(f"  {m}")

if not results:
    print("\nNo results available yet. Exiting.")
    exit(0)

# ---------------------------------------------------------------------------
# Determine architecture conclusion
# ---------------------------------------------------------------------------

degraded = []
improved = []
for key, r in results.items():
    if r["delta_f1"] < -0.01:
        degraded.append(key)
    elif r["delta_f1"] > 0.01:
        improved.append(key)

n_deg = len(degraded)
n_tot = len(results)

if n_deg == n_tot:
    conclusion_label = "Architecture-Agnostic Failure"
    conclusion_body  = (
        "S-CoT degrades performance across **all** model–dataset combinations tested. "
        "The failure is architecture-agnostic: both MedGemma-4B (dense, PaliGemma-based) "
        "and HuatuoGPT-7B (Qwen2.5VL-based, larger) are hurt by the structured prompt. "
        "This confirms that generative drag is a property of sub-10B VLMs in zero-shot "
        "medical VQA, not an artefact of MedGemma's specific training regime."
    )
elif n_deg == 0:
    conclusion_label = "Architecture-Agnostic Improvement"
    conclusion_body  = (
        "S-CoT improves performance across all runs. This contradicts the original "
        "MedGemma/SLAKE finding and suggests the original degradation was "
        "dataset- or run-specific."
    )
elif improved:
    imp_str = ", ".join(f"{m}/{d}" for m, d in improved)
    deg_str = ", ".join(f"{m}/{d}" for m, d in degraded)
    conclusion_label = "Architecture-Specific or Dataset-Specific Failure"
    conclusion_body  = (
        f"S-CoT improves {imp_str} but degrades {deg_str}. "
        "The failure is not universal, suggesting that either backbone architecture "
        "or dataset characteristics (question length, visual complexity) modulate "
        "the impact of structured prompting."
    )
else:
    conclusion_label = "Partial Degradation"
    conclusion_body  = (
        "S-CoT degrades some but not all combinations. Further analysis needed."
    )

# ---------------------------------------------------------------------------
# Generate markdown report
# ---------------------------------------------------------------------------

def pct(v):
    return f"{v*100:.2f}%"

def delta_str(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.2f} pp"

def sig_row(r):
    p = r["p_f1"]
    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    return f"{p:.4f} ({stars})"


rows = ""
for (model, dataset), r in results.items():
    sc, ba = r["scot"], r["base"]
    rows += (
        f"| {model} | {dataset} | {pct(ba['f1'])} | {pct(sc['f1'])} | "
        f"**{delta_str(r['delta_f1'])}** | {pct(ba['closed_acc'])} | "
        f"{pct(sc['closed_acc'])} | {delta_str(r['delta_closed_acc'])} | "
        f"{pct(ba['f1_open'])} | {pct(sc['f1_open'])} | "
        f"{delta_str(r['delta_open_f1'])} | {sig_row(r)} |\n"
    )

md = f"""# S-CoT Extension — Results Report
## VLM Medical VQA Benchmark

This report documents the extension of the Structured Chain-of-Thought (S-CoT)
experiment to two new model–dataset combinations, testing whether the
performance degradation originally observed on MedGemma-4B / SLAKE is
architecture-agnostic.

---

## 1. Experiment Summary

| Model | Dataset | S-CoT Run | Baseline |
|---|---|---|---|
| MedGemma-4B | SLAKE | Original experiment | `outputs/inference/google_medgemma-4b-it__slake_v2.jsonl` |
| HuatuoGPT-7B | SLAKE | **New — Run A** | `outputs/inference/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_7b_v2.jsonl` |
| MedGemma-4B | VQA-RAD | **New — Run B** | `outputs/inference/google_medgemma-4b-it__vqa_rad_v2.jsonl` |

**S-CoT Prompt (identical across all runs):**
```
{{question}}

Please reason step by step using this exact structure:

Step 1 - Modality: Identify the imaging modality (e.g., CT, MRI, X-ray).
Step 2 - Anatomy: Name the primary organ or anatomical structure visible.
Step 3 - Observation: Write exactly one short sentence answering the core
         question based on Steps 1 and 2.
Step 4 - Conclusion: State your definitive answer (if possible, a single word)
         in the exact format 'Final Answer: X'.
```

---

## 2. Results

| Model | Dataset | Base F1 | SCoT F1 | ΔF1 | Base Closed | SCoT Closed | ΔClosed | Base Open | SCoT Open | ΔOpen | p (F1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
{rows.strip()}

Statistical significance: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05.  
Paired permutation test, 10,000 iterations, on matched question indices.

---

## 3. Conclusion

### {conclusion_label}

{conclusion_body}

---

## 4. Diagnostic Analysis

### 4.1 Generative Drag Hypothesis

The original hypothesis was that a 4B model's limited attention capacity is
overwhelmed by the overhead of generating the 4-step structured output, causing
it to "forget" the visual grounding established in earlier steps.

{"The current results support this hypothesis across architectures." if n_deg == n_tot else "The current results partially support or refute this hypothesis — see conclusions above."}

### 4.2 Per-Combination Breakdown

"""

for (model, dataset), r in results.items():
    sc, ba = r["scot"], r["base"]
    direction = "degradation" if r["delta_f1"] < -0.01 else ("improvement" if r["delta_f1"] > 0.01 else "no change")
    md += (
        f"**{model} / {dataset}** (N={sc['n']}): "
        f"F1 {pct(ba['f1'])} → {pct(sc['f1'])} ({delta_str(r['delta_f1'])}, "
        f"p={r['p_f1']:.4f}). "
        f"Net effect: **{direction}**.\n\n"
    )

md += f"""
---

## 5. Updated Section 14 — Paper-Ready Paragraph

```
The S-CoT intervention was subsequently extended to two additional
model–dataset combinations: HuatuoGPT-Vision-7B-Qwen2.5VL evaluated on SLAKE
and MedGemma-4B evaluated on VQA-RAD. The same four-step structured prompt
was applied without modification. {"Results showed performance degradation in all new combinations (see Table 14.2), confirming that the generative-drag effect is architecture-agnostic and not limited to MedGemma's specific training regime. The combined evidence across three model–dataset pairs supports the conclusion that rigid structured prompting is harmful for sub-10B VLMs in zero-shot medical VQA, regardless of backbone architecture or dataset domain." if n_deg == n_tot else "Results were mixed across combinations (see Table 14.2). Architecture and dataset characteristics modulate the impact of structured prompting, suggesting that the original MedGemma/SLAKE finding was not fully generalisable."}
```

---

## 6. Generated Outputs

| File | Description |
|---|---|
| `docs/report_scot_extension.md` | This report |
| `scripts/scot_extension_analysis.py` | Analysis script (re-runnable) |
| `notebooks/09_scot_extension.ipynb` | Kaggle notebook for Runs A and B |
"""

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)

print(f"\nReport written to: {OUT_MD}")
print(f"Conclusion: {conclusion_label}")
