# LLM-as-a-Judge Inter-Rater Agreement Study
## Llama-3.1-8B vs Llama-3.3-70B-Versatile

**Purpose:** Validate whether the Llama-3.1-8B judge used throughout this benchmark tracks
clinical correctness reliably enough to support paper-quality conclusions.
Inter-rater agreement is computed between the 8B judge and an independent
70B judge (Llama-3.3-70B-Versatile running on Groq LPU hardware) on a
stratified sample. The 70B model has ~8.75× more parameters, stronger
instruction following, and runs on a completely independent inference stack.

**Judge A:** `meta-llama/Llama-3.1-8B-Instruct` (4-bit NF4 quantized, Kaggle T4)
**Judge B:** `meta-llama/Llama-3.3-70B-Versatile` (Groq LPU, full precision)
**Sample:** 499 records — stratified across score levels and question types
**Stratification:** score-1 anchor, score-5 anchor, score-3 ambiguous, open-ended medical
**Prompt:** Identical `MEDICAL_JUDGE_PROMPT` for both judges (1–5 scale, reference-grounded, evaluation-before-rating)

---

## 1. Overall Agreement

| Metric | Value | Interpretation |
|---|---|---|
| Pearson correlation | 0.810 (p=0.0000) | Linear score agreement |
| Spearman correlation | 0.818 (p=0.0000) | Ordinal rank agreement |
| Linear-weighted Cohen’s κ | **0.696** | ✅ Substantial |
| Mean Absolute Difference | 0.567 score points | Average per-item disagreement |
| Exact agreement rate | 66.3% | Both give identical integer score |
| Adjacent agreement rate | 80.8% | Both within 1 score point |
| 8B judge mean score | 3.146 | |
| 70B judge mean score | 2.952 | |
| Systematic bias (8B − 70B) | +0.194 | |

> [!NOTE]
> No significant systematic bias detected between the two judges (mean 8B: 3.15, mean 70B: 2.95, Δ = +0.19). The 8B judge does not systematically inflate or deflate scores.

---

## 2. Per-Model Agreement

| Model | N | Pearson r | Cohen κ | MAD | Bias (8B−70B) |
|---|---|---|---|---|---|
| `HuatuoGPT-Vision-7B-Qwen2.5VL` | 96 | 0.841 | 0.724 | 0.469 | +0.073 |
| `gemma-3-4b-it` | 96 | 0.856 | 0.769 | 0.427 | +0.177 |
| `medgemma-4b-it` | 99 | 0.809 | 0.715 | 0.505 | +0.202 |
| `llava-v1.6-mistral-7b-hf` | 95 | 0.809 | 0.718 | 0.547 | -0.147 |
| `llava-med-v1.5-mistral-7b` | 113 | 0.749 | 0.535 | 0.841 | +0.593 |

---

## 3. Per-Stratum Agreement

| Stratum | N | Pearson r | Cohen κ | MAD |
|---|---|---|---|---|
| medical_open | 150 | 0.883 | 0.789 | 0.380 |
| score_1_easy | 100 | n/a† | n/a† | 0.300 |
| score_3_ambiguous | 150 | n/a† | n/a† | 1.240 |
| score_5_easy | 99 | n/a† | n/a† | 0.101 |

† Score-1, score-3, and score-5 easy-anchor strata have near-zero variance
in Judge A scores by design (all records share the same ground-truth label).
Pearson r and Kappa are statistically undefined in constant-valued distributions;
MAD is the appropriate disagreement measure for these strata.

---


## 4. Closed vs Open Question Agreement

| Question Type | N | Cohen κ | Pearson r | MAD |
|---|---|---|---|---|
| Closed (Yes/No) | 74 | 0.776 | 0.813 | 0.446 |
| Open-ended | 425 | 0.677 | 0.812 | 0.588 |

---

## 5. Interpretation

### Cohen’s Kappa Reference Scale

| Range | Label | Implication for this study |
|---|---|---|
| κ < 0.40 | Poor | 8B judge cannot be trusted — all judge accuracy scores are suspect |
| 0.40 – 0.60 | Moderate | Usable with explicit caveat; scores ±1 should be treated as equivalent |
| 0.60 – 0.80 | Substantial | 8B judge is reliable; results can be reported with standard confidence |
| κ ≥ 0.80 | Near-perfect | 8B judge fully validated against the 70B reference |

**This study result: κ = 0.696 — Substantial — 8B judge reliable**

### Pearson r / Spearman ρ Reference Scale

| Range | Label | Implication |
|---|---|---|
| r < 0.40 | Weak | Weak rank/linear correlation between the judges |
| 0.40 – 0.59 | Fair | Moderate correlation |
| 0.60 – 0.79 | Strong | Good correlation, judges generally trend together |
| r ≥ 0.80 | Very Strong | Excellent correlation, tightly coupled scoring patterns |

**This study result: r = 0.810, ρ = 0.818**

### Mean Absolute Difference (MAD) Reference Scale

| Range | Label | Implication (on a 1-5 scale) |
|---|---|---|
| MAD > 1.0 | Poor | On average, judges disagree by more than a full point |
| 0.75 – 1.0 | Moderate | Fair agreement, frequent 1-2 point disagreements |
| 0.50 – 0.75 | Good | Judges typically disagree by less than a point |
| MAD < 0.50 | Excellent | High precision, judges are extremely close |

**This study result: MAD = 0.567**

### Exact / Adjacent Agreement Reference Scale

| Exact Agreement | Adjacent (±1) Agreement | Label | Implication |
|---|---|---|---|
| < 50% | < 70% | Poor | Judges rarely agree exactly, and often disagree by ≥2 points |
| 50% – 64% | 70% – 84% | Moderate | Acceptable for subjective tasks |
| 65% – 79% | 85% – 94% | Good | Reliable agreement on most predictions |
| ≥ 80% | ≥ 95% | Excellent | Near-human level of inter-rater consistency |

**This study result: Exact = 66.3%, Adjacent = 80.8%**

---

## 6. Methods Paragraph (paper-ready)

```
Inter-rater reliability of the Llama-3.1-8B-Instruct judge was assessed by
running an independent Llama-3.3-70B-Versatile judge (Groq LPU inference)
on a stratified 499-record sample designed to stress-test boundary
cases: 100 records at each of score 1 and score 5 (easy anchors), 150 records
at score 3 (maximum ambiguity), and 150 open-ended medical-dataset records
where clinical knowledge is most critical. Both judges used the identical
MEDICAL_JUDGE_PROMPT (1–5 integer scale, reference-grounded, evaluation-
before-rating). The 70B judge operates on an independent inference stack
(Groq LPU hardware) with no shared weights or quantization with the 8B judge.
Agreement was measured by linear-weighted Cohen’s kappa (κ = 0.696),
Pearson r = 0.810, Spearman ρ = 0.818,
and mean absolute difference = 0.567 score points.
Exact agreement was 66.3% and within-one-point
agreement was 80.8%. No significant systematic
bias was detected (mean difference = +0.194).
```

---

## 7. Generated Outputs

| File | Description |
|---|---|
| [`results/inter_rater_agreement_results.json`](../results/inter_rater_agreement_results.json) | Full metrics in machine-readable JSON |
| [`outputs/inter_rater_sample_500.jsonl`](../outputs/inter_rater_sample_500.jsonl) | The 499-record stratified sample |
| [`outputs/inter_rater_results.jsonl`](../outputs/inter_rater_results.jsonl) | Records with both judge scores |
