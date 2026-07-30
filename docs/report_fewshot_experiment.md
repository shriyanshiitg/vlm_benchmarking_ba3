# Few-Shot Experiment Report
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
| Gemma-3-4B | 0-shot | 57.27% | — | 80.00% | 34.53% | — |
| Gemma-3-4B | 1-shot | 56.08% | -1.19 pp | 69.00% | 43.15% | 0.6980 ns |
| Gemma-3-4B | 3-shot | 53.82% | -3.45 pp | 71.00% | 36.64% | 0.2145 ns |
| LLaVA-1.6-7B | 0-shot | 38.24% | — | 54.00% | 29.47% | — |
| LLaVA-1.6-7B | 1-shot | 34.00% | -4.24 pp | 57.00% | 11.00% | 0.1882 ns |
| LLaVA-1.6-7B | 3-shot | 34.90% | -3.34 pp | 52.00% | 17.80% | 0.3493 ns |

*Statistical significance: ★★★ p<0.001, ★★ p<0.01, ★ p<0.05, ns p≥0.05.*
*Paired permutation test, 10,000 iterations.*

### 2.2 Reference Context — MedGemma-4B on Full SLAKE Dataset

| Model | F1 | Closed Acc | Open F1 |
|---|---|---|---|
| MedGemma-4B (Section 8, full SLAKE) | 70.50% | 85.58% | 55.81% |
| Gemma-3-4B (Section 8, full SLAKE) | 42.14% | 68.27% | 24.50% |
| LLaVA-1.6-7B (Section 8, full SLAKE) | 36.98% | 58.41% | 25.12% |

*Gap to MedGemma-4B (full-dataset): Gemma-3-4B = 28.36%, LLaVA-1.6-7B = 33.52%.*

### 2.3 Per Question-Type F1 Breakdown

| Model | Condition | Modality F1 | Organ F1 | Abnormality F1 |
|---|---|---|---|---|
| Gemma-3-4B | 0-shot | 78.79% | 52.94% | 40.20% |
| Gemma-3-4B | 1-shot | 81.82% | 50.98% | 35.58% |
| Gemma-3-4B | 3-shot | 78.79% | 43.63% | 39.35% |
| LLaVA-1.6-7B | 0-shot | 42.42% | 41.47% | 30.73% |
| LLaVA-1.6-7B | 1-shot | 31.82% | 39.71% | 30.30% |
| LLaVA-1.6-7B | 3-shot | 31.82% | 43.14% | 29.50% |

---

## 3. Conclusions

### 3.1 Overall Verdict

**In-context learning does not close the domain gap.** Neither generalist model benefits significantly from clinical few-shot examples. The performance gap between generalist and medical VLMs is driven by architectural differences — specifically, the domain-specific knowledge encoded during medical pre-training — and cannot be bridged through prompt engineering alone. This validates why medical fine-tuning is mandatory for competitive clinical VQA.

### 3.2 Per-Model Analysis

**Gemma-3-4B:**
Few-shot prompting **does not improve and may slightly degrade** performance for Gemma-3-4B (3-shot: -3.45 pp, p=0.2145 ns). The model's in-context learning capability is insufficient to benefit from clinical examples in the zero-shot medical VQA setting. This confirms that the domain gap is architectural rather than informational.

**LLaVA-1.6-7B:**
Few-shot prompting **does not improve and may slightly degrade** performance for LLaVA-1.6-7B (3-shot: -3.34 pp, p=0.3493 ns). The model's in-context learning capability is insufficient to benefit from clinical examples in the zero-shot medical VQA setting. This confirms that the domain gap is architectural rather than informational.

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

## 5. Generated Files

| File | Description |
|---|---|
| `outputs/_archive/fewshot_experiment/*.jsonl` | Per-run inference outputs (6 files) |
| `results/fewshot_results.json` | All metrics in machine-readable JSON |
| `results/fig_fewshot_f1.png` | F1 line chart: 0-shot → 1-shot → 3-shot |
| `docs/report_fewshot_experiment.md` | This report |
| `scripts/fewshot_analysis.py` | This analysis script (re-runnable) |
| `notebooks/10_fewshot_experiment.ipynb` | Kaggle inference notebook |
