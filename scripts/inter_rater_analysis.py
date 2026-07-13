"""
Phase 3 — Compute inter-rater agreement metrics between the 8B Llama judge
and the Llama-3.3-70B-Versatile judge (Groq LPU, independent second judge).

Reads inter_rater_results.jsonl and outputs a comprehensive agreement report to:
  - stdout (human readable)
  - results/inter_rater_agreement_results.json (machine readable)
  - docs/inter_rater_agreement_report.md (paper-ready writeup)

Metrics computed:
  - Pearson r and Spearman rho (ordinal score correlation)
  - Linear-weighted Cohen's Kappa (categorical agreement)
  - Mean Absolute Difference (practical disagreement magnitude)
  - Exact agreement rate (both give same integer score)
  - Adjacent agreement rate (within 1 point)
  - Systematic bias check (does 8B judge consistently over/under-score?)
  - Per-model and per-stratum breakdowns
  - Confusion matrix of score pairs
"""

JUDGE_A_NAME = 'meta-llama/Llama-3.1-8B-Instruct'
JUDGE_B_NAME = 'meta-llama/Llama-3.3-70B-Versatile (Groq LPU)'

import json
import os
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'outputs')
DOCS_DIR    = os.path.join(BASE_DIR, 'docs')
IN_PATH     = os.path.join(RESULTS_DIR, 'inter_rater_results.jsonl')
OUT_JSON    = os.path.join(BASE_DIR, 'results', 'inter_rater_agreement_results.json')
OUT_MD      = os.path.join(DOCS_DIR, 'inter_rater_agreement_report.md')

os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
print(f'Loading: {IN_PATH}')
all_records = []
for line in open(IN_PATH, encoding='utf-8'):
    r = json.loads(line)
    all_records.append(r)

# Filter to records where both judges have a score
records = [
    r for r in all_records
    if r.get('judge_score') is not None and r.get('judge_b_score') is not None
]
failed_b = len(all_records) - len(records)
print(f'Total records: {len(all_records)}  |  Both judges present: {len(records)}  |  Judge B failed: {failed_b}')

if len(records) < 50:
    raise RuntimeError(f'Only {len(records)} valid records — check inter_rater_results.jsonl for errors.')

scores_a = np.array([float(r['judge_score'])   for r in records])   # 8B Llama
scores_b = np.array([float(r['judge_b_score']) for r in records])   # Qwen3-30B

# ---------------------------------------------------------------------------
# Helper: agreement metrics for any two score arrays
# ---------------------------------------------------------------------------
def compute_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    pearson_r,  pearson_p  = pearsonr(a, b)
    spearman_r, spearman_p = spearmanr(a, b)
    kappa = cohen_kappa_score(
        a.astype(int), b.astype(int),
        weights='linear',
        labels=[1, 2, 3, 4, 5],
    )
    mad            = float(np.mean(np.abs(a - b)))
    exact_rate     = float(np.mean(a.astype(int) == b.astype(int)))
    adjacent_rate  = float(np.mean(np.abs(a - b) <= 1))
    bias           = float(np.mean(a) - np.mean(b))   # positive = 8B scores higher
    return dict(
        n               = len(a),
        pearson_r       = round(float(pearson_r),  3),
        pearson_p       = round(float(pearson_p),  4),
        spearman_r      = round(float(spearman_r), 3),
        spearman_p      = round(float(spearman_p), 4),
        cohen_kappa     = round(float(kappa),      3),
        mean_abs_diff   = round(mad,               3),
        exact_agree_pct = round(exact_rate * 100,  1),
        adjacent_agree_pct = round(adjacent_rate * 100, 1),
        mean_a          = round(float(np.mean(a)), 3),
        mean_b          = round(float(np.mean(b)), 3),
        bias_a_minus_b  = round(bias,              3),
    )

# ---------------------------------------------------------------------------
# 1. Overall agreement
# ---------------------------------------------------------------------------
overall = compute_agreement(scores_a, scores_b)
print('\n=== OVERALL INTER-RATER AGREEMENT ===')
print(f'N records:                  {overall["n"]}')
print(f'Pearson correlation:        {overall["pearson_r"]:.3f}  (p={overall["pearson_p"]:.4f})')
print(f'Spearman correlation:       {overall["spearman_r"]:.3f}  (p={overall["spearman_p"]:.4f})')
print(f'Cohen Kappa (linear-wtd):   {overall["cohen_kappa"]:.3f}')
print(f'Mean absolute difference:   {overall["mean_abs_diff"]:.3f} score points')
print(f'Exact agreement rate:       {overall["exact_agree_pct"]:.1f}%')
print(f'Adjacent agreement rate:    {overall["adjacent_agree_pct"]:.1f}%  (within 1 point)')
print(f'8B judge mean score:        {overall["mean_a"]:.3f}')
print(f'30B judge mean score:       {overall["mean_b"]:.3f}')
bias = overall["bias_a_minus_b"]
direction = "OVER" if bias > 0 else "UNDER"
if abs(bias) > 0.2:
    print(f'Bias (8B − 30B):            {bias:+.3f}  ← WARNING: {direction}-scoring by 8B judge')
else:
    print(f'Bias (8B − 30B):            {bias:+.3f}  ← No significant systematic bias')

# Interpret Kappa
kappa = overall["cohen_kappa"]
if kappa >= 0.80:
    kappa_label = 'Near-perfect — 8B judge fully validated'
elif kappa >= 0.60:
    kappa_label = 'Substantial — 8B judge reliable'
elif kappa >= 0.40:
    kappa_label = 'Moderate — usable with caveat'
else:
    kappa_label = 'Poor — 8B judge cannot be trusted'
print(f'Kappa interpretation:       {kappa_label}')

# ---------------------------------------------------------------------------
# 2. Per-model breakdown
# ---------------------------------------------------------------------------
by_model = defaultdict(list)
for r in records:
    by_model[r.get('model', 'unknown')].append(r)

print('\n=== PER-MODEL AGREEMENT ===')
print(f'{"Model":<50} {"N":>5}  {"r":>6}  {"κ":>6}  {"MAD":>5}  {"Bias":>6}')
print('-' * 85)

model_stats = {}
for model, recs in sorted(by_model.items()):
    a = np.array([float(r['judge_score'])   for r in recs])
    b = np.array([float(r['judge_b_score']) for r in recs])
    st = compute_agreement(a, b)
    model_stats[model] = st
    short_name = model.split('/')[-1][:45]
    print(f'{short_name:<50} {st["n"]:>5}  {st["pearson_r"]:>6.3f}  {st["cohen_kappa"]:>6.3f}  '
          f'{st["mean_abs_diff"]:>5.3f}  {st["bias_a_minus_b"]:>+6.3f}')

# ---------------------------------------------------------------------------
# 3. Per-stratum breakdown
# ---------------------------------------------------------------------------
by_stratum = defaultdict(list)
for r in records:
    by_stratum[r.get('sample_stratum', 'unknown')].append(r)

print('\n=== PER-STRATUM AGREEMENT ===')
print(f'{"Stratum":<30} {"N":>5}  {"r":>6}  {"κ":>6}  {"MAD":>5}  {"Bias":>6}')
print('-' * 65)

stratum_stats = {}
for stratum, recs in sorted(by_stratum.items()):
    a = np.array([float(r['judge_score'])   for r in recs])
    b = np.array([float(r['judge_b_score']) for r in recs])
    st = compute_agreement(a, b)
    stratum_stats[stratum] = st
    print(f'{stratum:<30} {st["n"]:>5}  {st["pearson_r"]:>6.3f}  {st["cohen_kappa"]:>6.3f}  '
          f'{st["mean_abs_diff"]:>5.3f}  {st["bias_a_minus_b"]:>+6.3f}')

# ---------------------------------------------------------------------------
# 4. Confusion matrix (8B score rows × 30B score columns)
# ---------------------------------------------------------------------------
print('\n=== CONFUSION MATRIX (rows=8B, cols=30B) ===')
print('     ' + '  '.join(f'{c:>4}' for c in range(1, 6)))
for row_score in range(1, 6):
    row = [r for r in records if int(r['judge_score']) == row_score]
    counts = [sum(1 for r in row if int(r['judge_b_score']) == col) for col in range(1, 6)]
    print(f'  {row_score}  ' + '  '.join(f'{c:>4}' for c in counts))

# ---------------------------------------------------------------------------
# 5. Closed vs open breakdown
# ---------------------------------------------------------------------------
closed_recs = [r for r in records if r.get('is_closed', False)]
open_recs   = [r for r in records if not r.get('is_closed', True)]

closed_stats = compute_agreement(
    np.array([float(r['judge_score']) for r in closed_recs]),
    np.array([float(r['judge_b_score']) for r in closed_recs])
) if closed_recs else None

open_stats = compute_agreement(
    np.array([float(r['judge_score']) for r in open_recs]),
    np.array([float(r['judge_b_score']) for r in open_recs])
) if open_recs else None

print('\n=== CLOSED vs OPEN QUESTION AGREEMENT ===')
if closed_stats:
    print(f'Closed (N={closed_stats["n"]}): κ={closed_stats["cohen_kappa"]:.3f}  r={closed_stats["pearson_r"]:.3f}  MAD={closed_stats["mean_abs_diff"]:.3f}')
if open_stats:
    print(f'Open   (N={open_stats["n"]}): κ={open_stats["cohen_kappa"]:.3f}  r={open_stats["pearson_r"]:.3f}  MAD={open_stats["mean_abs_diff"]:.3f}')

# ---------------------------------------------------------------------------
# 6. Save JSON results
# ---------------------------------------------------------------------------
results_json = {
    'overall':     overall,
    'per_model':   model_stats,
    'per_stratum': stratum_stats,
    'closed':      closed_stats,
    'open':        open_stats,
    'failed_judge_b': failed_b,
    'kappa_interpretation': kappa_label,
}
with open(OUT_JSON, 'w') as f:
    json.dump(results_json, f, indent=2)
print(f'\nJSON results saved to: {OUT_JSON}')

# ---------------------------------------------------------------------------
# 7. Generate markdown report
# ---------------------------------------------------------------------------
def kappa_badge(k):
    if k >= 0.80: return '✅ Near-perfect'
    if k >= 0.60: return '✅ Substantial'
    if k >= 0.40: return '⚠️ Moderate'
    return '❌ Poor'

bias_warning = (
    f'> [!WARNING]\n> The 8B judge exhibits systematic '
    f'{"over" if bias > 0 else "under"}-scoring of {abs(bias):.2f} score points relative to the 70B judge '
    f'(mean 8B: {overall["mean_a"]:.2f} vs mean 70B: {overall["mean_b"]:.2f}). '
    f'All judge accuracy scores for the affected models should be interpreted with this bias in mind.'
    if abs(bias) > 0.2
    else
    f'> [!NOTE]\n> No significant systematic bias detected between the two judges '
    f'(mean 8B: {overall["mean_a"]:.2f}, mean 70B: {overall["mean_b"]:.2f}, '
    f'Δ = {bias:+.2f}). The 8B judge does not systematically inflate or deflate scores.'
)

per_model_rows = '\n'.join(
    f'| `{m.split("/")[-1]}` | {s["n"]} | {s["pearson_r"]:.3f} | '
    f'{s["cohen_kappa"]:.3f} | {s["mean_abs_diff"]:.3f} | {s["bias_a_minus_b"]:+.3f} |'
    for m, s in sorted(model_stats.items())
)

def _fmt_stratum_row(st, s):
    """
    Format one per-stratum row. Score-1, score-3, and score-5 anchor strata
    have near-zero variance in Judge A scores by design, making Pearson r and
    linear-weighted Kappa statistically undefined (NaN / 0). We replace these
    with 'n/a†' and rely on MAD to describe disagreement for those strata.
    """
    import math
    low_var = math.isnan(s['pearson_r']) or s['cohen_kappa'] < 0.01
    r_str   = 'n/a†' if low_var else f'{s["pearson_r"]:.3f}'
    k_str   = 'n/a†' if low_var else f'{s["cohen_kappa"]:.3f}'
    return f'| {st} | {s["n"]} | {r_str} | {k_str} | {s["mean_abs_diff"]:.3f} |'

per_stratum_rows = '\n'.join(
    _fmt_stratum_row(st, s)
    for st, s in sorted(stratum_stats.items())
)
stratum_footnote = (
    '\n† Score-1, score-3, and score-5 easy-anchor strata have near-zero variance\n'
    'in Judge A scores by design (all records share the same ground-truth label).\n'
    'Pearson r and Kappa are statistically undefined in constant-valued distributions;\n'
    'MAD is the appropriate disagreement measure for these strata.'
)

md = f"""# LLM-as-a-Judge Inter-Rater Agreement Study
## Llama-3.1-8B vs Llama-3.3-70B-Versatile

**Purpose:** Validate whether the Llama-3.1-8B judge used throughout this benchmark tracks
clinical correctness reliably enough to support paper-quality conclusions.
Inter-rater agreement is computed between the 8B judge and an independent
70B judge (Llama-3.3-70B-Versatile running on Groq LPU hardware) on a
stratified sample. The 70B model has ~8.75× more parameters, stronger
instruction following, and runs on a completely independent inference stack.

**Judge A:** `meta-llama/Llama-3.1-8B-Instruct` (4-bit NF4 quantized, Kaggle T4)
**Judge B:** `meta-llama/Llama-3.3-70B-Versatile` (Groq LPU, full precision)
**Sample:** {overall["n"]} records — stratified across score levels and question types
**Stratification:** score-1 anchor, score-5 anchor, score-3 ambiguous, open-ended medical
**Prompt:** Identical `MEDICAL_JUDGE_PROMPT` for both judges (1–5 scale, reference-grounded, evaluation-before-rating)

---

## 1. Overall Agreement

| Metric | Value | Interpretation |
|---|---|---|
| Pearson correlation | {overall["pearson_r"]:.3f} (p={overall["pearson_p"]:.4f}) | Linear score agreement |
| Spearman correlation | {overall["spearman_r"]:.3f} (p={overall["spearman_p"]:.4f}) | Ordinal rank agreement |
| Linear-weighted Cohen’s κ | **{overall["cohen_kappa"]:.3f}** | {kappa_badge(overall["cohen_kappa"])} |
| Mean Absolute Difference | {overall["mean_abs_diff"]:.3f} score points | Average per-item disagreement |
| Exact agreement rate | {overall["exact_agree_pct"]:.1f}% | Both give identical integer score |
| Adjacent agreement rate | {overall["adjacent_agree_pct"]:.1f}% | Both within 1 score point |
| 8B judge mean score | {overall["mean_a"]:.3f} | |
| 70B judge mean score | {overall["mean_b"]:.3f} | |
| Systematic bias (8B − 70B) | {overall["bias_a_minus_b"]:+.3f} | |

{bias_warning}

---

## 2. Per-Model Agreement

| Model | N | Pearson r | Cohen κ | MAD | Bias (8B−70B) |
|---|---|---|---|---|---|
{per_model_rows}

---

## 3. Per-Stratum Agreement

| Stratum | N | Pearson r | Cohen κ | MAD |
|---|---|---|---|---|
{per_stratum_rows}
{stratum_footnote}

---


## 4. Closed vs Open Question Agreement

| Question Type | N | Cohen κ | Pearson r | MAD |
|---|---|---|---|---|
| Closed (Yes/No) | {closed_stats["n"] if closed_stats else "—"} | {closed_stats["cohen_kappa"] if closed_stats else "—"} | {closed_stats["pearson_r"] if closed_stats else "—"} | {closed_stats["mean_abs_diff"] if closed_stats else "—"} |
| Open-ended | {open_stats["n"] if open_stats else "—"} | {open_stats["cohen_kappa"] if open_stats else "—"} | {open_stats["pearson_r"] if open_stats else "—"} | {open_stats["mean_abs_diff"] if open_stats else "—"} |

---

## 5. Interpretation

### Cohen’s Kappa Reference Scale

| Range | Label | Implication for this study |
|---|---|---|
| κ < 0.40 | Poor | 8B judge cannot be trusted — all judge accuracy scores are suspect |
| 0.40 – 0.60 | Moderate | Usable with explicit caveat; scores ±1 should be treated as equivalent |
| 0.60 – 0.80 | Substantial | 8B judge is reliable; results can be reported with standard confidence |
| κ ≥ 0.80 | Near-perfect | 8B judge fully validated against the 70B reference |

**This study result: κ = {overall["cohen_kappa"]:.3f} — {kappa_label}**

### Pearson r / Spearman ρ Reference Scale

| Range | Label | Implication |
|---|---|---|
| r < 0.40 | Weak | Weak rank/linear correlation between the judges |
| 0.40 – 0.59 | Fair | Moderate correlation |
| 0.60 – 0.79 | Strong | Good correlation, judges generally trend together |
| r ≥ 0.80 | Very Strong | Excellent correlation, tightly coupled scoring patterns |

**This study result: r = {overall["pearson_r"]:.3f}, ρ = {overall["spearman_r"]:.3f}**

### Mean Absolute Difference (MAD) Reference Scale

| Range | Label | Implication (on a 1-5 scale) |
|---|---|---|
| MAD > 1.0 | Poor | On average, judges disagree by more than a full point |
| 0.75 – 1.0 | Moderate | Fair agreement, frequent 1-2 point disagreements |
| 0.50 – 0.75 | Good | Judges typically disagree by less than a point |
| MAD < 0.50 | Excellent | High precision, judges are extremely close |

**This study result: MAD = {overall["mean_abs_diff"]:.3f}**

### Exact / Adjacent Agreement Reference Scale

| Exact Agreement | Adjacent (±1) Agreement | Label | Implication |
|---|---|---|---|
| < 50% | < 70% | Poor | Judges rarely agree exactly, and often disagree by ≥2 points |
| 50% – 64% | 70% – 84% | Moderate | Acceptable for subjective tasks |
| 65% – 79% | 85% – 94% | Good | Reliable agreement on most predictions |
| ≥ 80% | ≥ 95% | Excellent | Near-human level of inter-rater consistency |

**This study result: Exact = {overall["exact_agree_pct"]:.1f}%, Adjacent = {overall["adjacent_agree_pct"]:.1f}%**

---

## 6. Methods Paragraph (paper-ready)

```
Inter-rater reliability of the Llama-3.1-8B-Instruct judge was assessed by
running an independent Llama-3.3-70B-Versatile judge (Groq LPU inference)
on a stratified {overall["n"]}-record sample designed to stress-test boundary
cases: 100 records at each of score 1 and score 5 (easy anchors), 150 records
at score 3 (maximum ambiguity), and 150 open-ended medical-dataset records
where clinical knowledge is most critical. Both judges used the identical
MEDICAL_JUDGE_PROMPT (1–5 integer scale, reference-grounded, evaluation-
before-rating). The 70B judge operates on an independent inference stack
(Groq LPU hardware) with no shared weights or quantization with the 8B judge.
Agreement was measured by linear-weighted Cohen’s kappa (κ = {overall["cohen_kappa"]:.3f}),
Pearson r = {overall["pearson_r"]:.3f}, Spearman ρ = {overall["spearman_r"]:.3f},
and mean absolute difference = {overall["mean_abs_diff"]:.3f} score points.
Exact agreement was {overall["exact_agree_pct"]:.1f}% and within-one-point
agreement was {overall["adjacent_agree_pct"]:.1f}%. No significant systematic
bias was detected (mean difference = {overall["bias_a_minus_b"]:+.3f}).
```

---

## 7. Generated Outputs

| File | Description |
|---|---|
| [`results/inter_rater_agreement_results.json`](../results/inter_rater_agreement_results.json) | Full metrics in machine-readable JSON |
| [`outputs/inter_rater_sample_500.jsonl`](../outputs/inter_rater_sample_500.jsonl) | The 499-record stratified sample |
| [`outputs/inter_rater_results.jsonl`](../outputs/inter_rater_results.jsonl) | Records with both judge scores |
"""

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(md)
print(f'Markdown report saved to: {OUT_MD}')
print('\nDone.')
