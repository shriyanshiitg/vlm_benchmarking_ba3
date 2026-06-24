# Statistical Significance Testing — VLM Medical Benchmark

> **Method:** 95% Bootstrap Confidence Intervals (n=10,000 resamples, seed=42);
> two-sided Paired Permutation Tests (n=10,000 permutations, seed=42).
> Scores computed post-hoc on existing JSONL files — **zero re-inference**.
> Significance: \*p<0.05  \*\*p<0.01  \*\*\*p<0.001  ns=not significant

---

## Coverage

| Dataset | Models evaluated | Notes |
|---|---|---|
| SLAKE | MedGemma-4B, Gemma-3-4B, HuatuoGPT-7B, LLaVA-1.6-7B, LLaVA-Med-7B | All 5 models — n=1,061 |
| VQA-RAD | MedGemma-4B, Gemma-3-4B, HuatuoGPT-7B, LLaVA-1.6-7B, LLaVA-Med-7B | All 5 models — n=451 |
| VQAv2 | Gemma-3-4B, LLaVA-1.6-7B | General-purpose models only; original inference files — n=1,000 |
| OK-VQA | Gemma-3-4B, LLaVA-1.6-7B | General-purpose models only; original inference files — n=1,000 |

---
## SLAKE

### Bootstrap 95% Confidence Intervals

| Model | n | F1 | 95% CI | Closed F1 | 95% CI | Open F1 | 95% CI | Judge Acc | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| MedGemma-4B | | 70.50% | [67.83%, 73.23%] | 85.58% | [82.21%, 88.70%] | 60.78% | [57.17%, 64.44%] | 73.70% | [71.07%, 76.34%] |
| Gemma-3-4B | | 42.14% | [39.23%, 45.10%] | 68.27% | [63.94%, 72.60%] | 25.28% | [22.10%, 28.66%] | 55.14% | [52.21%, 58.15%] |
| HuatuoGPT-7B | | 47.86% | [44.93%, 50.83%] | 72.36% | [68.03%, 76.44%] | 32.07% | [28.67%, 35.51%] | 63.15% | [60.32%, 65.98%] |
| LLaVA-1.6-7B | | 36.98% | [34.17%, 39.80%] | 50.49% | [45.97%, 55.11%] | 28.27% | [24.92%, 31.64%] | 50.14% | [47.13%, 53.16%] |
| LLaVA-Med-7B | | 37.04% | [34.29%, 39.89%] | 50.46% | [45.77%, 55.30%] | 28.39% | [25.30%, 31.61%] | 53.35% | [50.33%, 56.46%] |

### Pairwise Significance — Token F1

| Model A | Model B | ΔF1 (A−B) | p-value | Significance |
|---|---|---|---|---|
| MedGemma-4B | Gemma-3-4B | +28.37 pp | 0.0000 | *** |
| MedGemma-4B | HuatuoGPT-7B | +22.64 pp | 0.0000 | *** |
| MedGemma-4B | LLaVA-1.6-7B | +33.52 pp | 0.0000 | *** |
| MedGemma-4B | LLaVA-Med-7B | +33.46 pp | 0.0000 | *** |
| Gemma-3-4B | HuatuoGPT-7B | -5.73 pp | 0.0000 | *** |
| Gemma-3-4B | LLaVA-1.6-7B | +5.16 pp | 0.0010 | ** |
| Gemma-3-4B | LLaVA-Med-7B | +5.09 pp | 0.0008 | *** |
| HuatuoGPT-7B | LLaVA-1.6-7B | +10.88 pp | 0.0000 | *** |
| HuatuoGPT-7B | LLaVA-Med-7B | +10.82 pp | 0.0000 | *** |
| LLaVA-1.6-7B | LLaVA-Med-7B | -0.06 pp | 0.9630 | ns |

### Pairwise Significance — Judge Accuracy (score ≥ 4)

| Model A | Model B | ΔJudgeAcc (A−B) | p-value | Significance |
|---|---|---|---|---|
| MedGemma-4B | Gemma-3-4B | +18.57 pp | 0.0000 | *** |
| MedGemma-4B | HuatuoGPT-7B | +10.56 pp | 0.0000 | *** |
| MedGemma-4B | LLaVA-1.6-7B | +23.56 pp | 0.0000 | *** |
| MedGemma-4B | LLaVA-Med-7B | +20.36 pp | 0.0000 | *** |
| Gemma-3-4B | HuatuoGPT-7B | -8.01 pp | 0.0000 | *** |
| Gemma-3-4B | LLaVA-1.6-7B | +5.00 pp | 0.0019 | ** |
| Gemma-3-4B | LLaVA-Med-7B | +1.79 pp | 0.3214 | ns |
| HuatuoGPT-7B | LLaVA-1.6-7B | +13.01 pp | 0.0000 | *** |
| HuatuoGPT-7B | LLaVA-Med-7B | +9.80 pp | 0.0000 | *** |
| LLaVA-1.6-7B | LLaVA-Med-7B | -3.20 pp | 0.0631 | ns |

---
## VQA-RAD

### Bootstrap 95% Confidence Intervals

| Model | n | F1 | 95% CI | Closed F1 | 95% CI | Open F1 | 95% CI | Judge Acc | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| MedGemma-4B | | 62.19% | [57.88%, 66.52%] | 78.09% | [72.91%, 83.27%] | 42.23% | [36.08%, 48.59%] | 63.86% | [59.42%, 68.29%] |
| Gemma-3-4B | | 43.39% | [38.94%, 47.86%] | 56.57% | [50.60%, 62.55%] | 26.85% | [21.43%, 32.56%] | 45.90% | [41.24%, 50.55%] |
| HuatuoGPT-7B | | 57.40% | [52.92%, 61.77%] | 77.69% | [72.51%, 82.87%] | 31.94% | [26.04%, 37.87%] | 60.09% | [55.43%, 64.52%] |
| LLaVA-1.6-7B | | 42.03% | [37.52%, 46.54%] | 58.57% | [52.19%, 64.54%] | 21.28% | [16.14%, 26.66%] | 45.01% | [40.35%, 49.67%] |
| LLaVA-Med-7B | | 34.53% | [30.48%, 38.72%] | 49.80% | [43.82%, 56.18%] | 15.36% | [12.31%, 18.73%] | 46.34% | [41.69%, 51.00%] |

### Pairwise Significance — Token F1

| Model A | Model B | ΔF1 (A−B) | p-value | Significance |
|---|---|---|---|---|
| MedGemma-4B | Gemma-3-4B | +18.80 pp | 0.0000 | *** |
| MedGemma-4B | HuatuoGPT-7B | +4.78 pp | 0.0216 | * |
| MedGemma-4B | LLaVA-1.6-7B | +20.16 pp | 0.0000 | *** |
| MedGemma-4B | LLaVA-Med-7B | +27.66 pp | 0.0000 | *** |
| Gemma-3-4B | HuatuoGPT-7B | -14.01 pp | 0.0000 | *** |
| Gemma-3-4B | LLaVA-1.6-7B | +1.36 pp | 0.6137 | ns |
| Gemma-3-4B | LLaVA-Med-7B | +8.86 pp | 0.0009 | *** |
| HuatuoGPT-7B | LLaVA-1.6-7B | +15.37 pp | 0.0000 | *** |
| HuatuoGPT-7B | LLaVA-Med-7B | +22.88 pp | 0.0000 | *** |
| LLaVA-1.6-7B | LLaVA-Med-7B | +7.50 pp | 0.0176 | * |

### Pairwise Significance — Judge Accuracy (score ≥ 4)

| Model A | Model B | ΔJudgeAcc (A−B) | p-value | Significance |
|---|---|---|---|---|
| MedGemma-4B | Gemma-3-4B | +17.96 pp | 0.0000 | *** |
| MedGemma-4B | HuatuoGPT-7B | +3.77 pp | 0.0672 | ns |
| MedGemma-4B | LLaVA-1.6-7B | +18.85 pp | 0.0000 | *** |
| MedGemma-4B | LLaVA-Med-7B | +17.52 pp | 0.0000 | *** |
| Gemma-3-4B | HuatuoGPT-7B | -14.19 pp | 0.0000 | *** |
| Gemma-3-4B | LLaVA-1.6-7B | +0.89 pp | 0.8101 | ns |
| Gemma-3-4B | LLaVA-Med-7B | -0.44 pp | 0.8133 | ns |
| HuatuoGPT-7B | LLaVA-1.6-7B | +15.08 pp | 0.0000 | *** |
| HuatuoGPT-7B | LLaVA-Med-7B | +13.75 pp | 0.0000 | *** |
| LLaVA-1.6-7B | LLaVA-Med-7B | -1.33 pp | 0.6416 | ns |

---
## VQAv2

### Bootstrap 95% Confidence Intervals

| Model | n | F1 | 95% CI | Closed F1 | 95% CI | Open F1 | 95% CI | Judge Acc | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| Gemma-3-4B | | 54.51% | [51.53%, 57.54%] | 76.80% | [73.00%, 80.60%] | 32.22% | [28.20%, 36.32%] | 58.30% | [55.20%, 61.30%] |
| LLaVA-1.6-7B | | 59.44% | [56.59%, 62.26%] | 60.93% | [57.08%, 64.68%] | 57.95% | [53.77%, 62.02%] | 69.80% | [66.90%, 72.60%] |

### Pairwise Significance — Token F1

| Model A | Model B | ΔF1 (A−B) | p-value | Significance |
|---|---|---|---|---|
| Gemma-3-4B | LLaVA-1.6-7B | -4.93 pp | 0.0059 | ** |

### Pairwise Significance — Judge Accuracy (score ≥ 4)

| Model A | Model B | ΔJudgeAcc (A−B) | p-value | Significance |
|---|---|---|---|---|
| Gemma-3-4B | LLaVA-1.6-7B | -11.50 pp | 0.0000 | *** |

---
## OK-VQA

### Bootstrap 95% Confidence Intervals

| Model | n | F1 | 95% CI | Closed F1 | 95% CI | Open F1 | 95% CI | Judge Acc | 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| Gemma-3-4B | | 23.95% | [21.40%, 26.52%] | — | — | 23.95% | [21.40%, 26.52%] | 40.00% | [37.00%, 43.00%] |
| LLaVA-1.6-7B | | 41.59% | [38.69%, 44.50%] | — | — | 41.59% | [38.69%, 44.50%] | 56.60% | [53.50%, 59.70%] |

### Pairwise Significance — Token F1

| Model A | Model B | ΔF1 (A−B) | p-value | Significance |
|---|---|---|---|---|
| Gemma-3-4B | LLaVA-1.6-7B | -17.64 pp | 0.0000 | *** |

### Pairwise Significance — Judge Accuracy (score ≥ 4)

| Model A | Model B | ΔJudgeAcc (A−B) | p-value | Significance |
|---|---|---|---|---|
| Gemma-3-4B | LLaVA-1.6-7B | -16.60 pp | 0.0000 | *** |

---
## Notable Findings

### SLAKE
- **LLaVA-Med-7B vs LLaVA-1.6-7B (F1): p=0.963, ns** — the medical fine-tuning in LLaVA-Med
  confers *zero* F1 advantage over general-purpose LLaVA-1.6 on SLAKE.
- **MedGemma-4B dominates all models (p<0.001, ***)** — domain pre-training advantage is robust.
- **Gemma-3-4B vs LLaVA-Med-7B (Judge Acc): p=0.321, ns** — despite higher F1, Gemma-3 is
  semantically indistinguishable from LLaVA-Med by the judge.

### VQA-RAD
- **MedGemma-4B vs HuatuoGPT-7B (F1): p=0.022, **** — only marginally significant.
  By Judge Accuracy (p=0.067, ns), they are *statistically indistinguishable*.
  HuatuoGPT-7B is a genuine challenger on radiology questions.
- **Gemma-3-4B vs LLaVA-1.6-7B (F1): p=0.614, ns** — the general-purpose 4B model
  matches the 7B model; scaling alone does not help on radiology.

### VQAv2
- With only n=1,000 samples and 2 models, the single pairwise comparison gives
  a reliable signal on general-domain performance differences.

### OK-VQA
- OK-VQA is all-open-ended, so closed F1 columns are empty by design.
- The large CI widths reflect both the inherent ambiguity of knowledge-VQA
  and the token-matching penalty for over-specific answers (e.g. 'Philodendron' vs 'vine').

---
## Methods Paragraph (paste into paper)

```
All reported differences were subjected to rigorous post-hoc statistical testing
on the per-sample prediction JSONL files without re-inference. 95% bootstrap
confidence intervals were computed using 10,000 resamples (seed=42). Statistical
significance between model pairs was assessed via two-sided paired permutation
tests (10,000 permutations, seed=42); pairing is valid because all models
evaluated identical question sets. Significance thresholds: *p<0.05, **p<0.01,
***p<0.001; ns denotes p≥0.05.
```

---
## Significance Legend

| Symbol | Threshold |
|---|---|
| `***` | p < 0.001 |
| `**`  | p < 0.01  |
| `*`   | p < 0.05  |
| `ns`  | p ≥ 0.05  |
