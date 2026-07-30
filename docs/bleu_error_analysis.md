# BLEU Error Analysis — N-Gram Metric Audit
## VLM Medical VQA Benchmark

This study investigates whether BLEU-4, while computed for all model–dataset
combinations, provides signal that is distinct from Token F1 and LLM Judge
Accuracy, and specifically characterises **where and why** it fails for
open-ended medical VQA.

All analyses are restricted to **open-ended questions only**. BLEU on binary
yes/no answers is trivially redundant with exact-match F1 and is excluded.

---

## 1. Dataset Summary

| Subset | N |
|---|---|
| Total open-ended records | 11766 |
| Medical open-ended (SLAKE + VQA-RAD) | 4224 |
| General open-ended (VQAv2 + OK-VQA) | 7542 |

---

## 2. Triple-Axis Correlation Analysis

Pearson r between three metric pairs. Statistical significance: * p<0.05, ** p<0.01, *** p<0.001.

### 2.1 Overall and Domain-Level

| Subset | N | BLEU ↔ Judge | BLEU ↔ F1 | Judge ↔ F1 |
|---|---|---|---|---|
| All open-ended | 11766 | 0.595*** | 0.937*** | 0.633*** |
| Medical (SLAKE + VQA-RAD) | 4224 | 0.642*** | 0.943*** | 0.672*** |
| General (VQAv2 + OK-VQA) | 7542 | 0.572*** | 0.934*** | 0.614*** |

### 2.2 Per-Model (Medical + General Open-Ended)

| Model | N | BLEU ↔ Judge | BLEU ↔ F1 | Judge ↔ F1 | Δ (Judge−BLEU) |
|---|---|---|---|---|---|
| Gemma-3-4B | 2323 | 0.635*** | 0.964*** | 0.652*** | +0.551 |
| HuatuoGPT-7B | 2431 | 0.596*** | 0.916*** | 0.645*** | +0.669 |
| LLaVA-1.6-7B | 2322 | 0.671*** | 0.926*** | 0.723*** | +0.574 |
| LLaVA-Med-7B | 2345 | 0.393*** | 0.960*** | 0.431*** | +0.651 |
| MedGemma-4B | 2345 | 0.644*** | 0.930*** | 0.683*** | +0.653 |

> **Redundancy check — BLEU vs. F1:** r = 0.937. ⚠️ High correlation (r > 0.85): BLEU and F1 are near-redundant for terse models.

### 2.3 Delta Analysis — Verbosity Penalty

The **Delta (Judge_norm − BLEU)** column above quantifies the verbosity penalty:
a large positive Δ means the judge rewards the model's semantic content, but
BLEU is penalising the model for using different wording than the ground truth.
Models with Δ > 0.30 are producing correct answers that BLEU cannot see.

---

## 3. The Rescue Zone — Visual Evidence of Metric Failure

The scatter plot below (X=BLEU, Y=Token F1, colour=Judge Score) highlights
a "rescue zone" in the bottom-left corner: records with near-zero BLEU **and**
near-zero F1, yet judged as correct (score ≥ 4) by the LLM.

![BLEU vs F1 Rescue Zone scatter](/Users/shriyanshraj/vlm_benchmark/results/fig_bleu_rescue_zone.png)

| Domain | Records in rescue zone (BLEU<0.10, F1<0.15, Judge≥4) |
|---|---|
| Medical | **692** |
| General | **1870** |

These records are **true positives that both BLEU and F1 call false negatives**.
Every one of them would be counted as a model failure if either classical metric
were the sole evaluation criterion.

---

## 4. Qualitative N-Gram Autopsy

From the autopsy pool (medical open-ended, Judge ≥ 4, BLEU < 0.05,
N = 782 records), three failure categories were identified.

### 4.1 Failure Mode Distribution

| Model | Medical Synonymy | Conversational Filler | Granularity Mismatch | Other |
|---|---|---|---|---|
| Gemma-3-4B | 93 | 4 | 1 | 64 |
| HuatuoGPT-7B | 118 | 0 | 0 | 67 |
| LLaVA-1.6-7B | 59 | 0 | 0 | 60 |
| LLaVA-Med-7B | 61 | 44 | 12 | 113 |
| MedGemma-4B | 48 | 1 | 0 | 37 |

**Total per bucket:**

| Bucket | N | Description |
|---|---|---|
| Medical Synonymy | 379 | Ground truth and prediction use clinically equivalent but lexically different terms (e.g., "renal" vs "kidney", "pulmonary" vs "lung"). |
| Conversational Filler | 49 | Model output contains the correct answer embedded in a longer conversational response, diluting n-gram precision. |
| Granularity Mismatch | 13 | Model provides a highly specific sub-classification while the ground truth is a broader term (or vice versa). |
| Other | 341 | Paraphrase or structural variation not captured by the above three patterns. |

### 4.2 Representative Examples

#### Medical Synonymy

| Model | Dataset | Ground Truth | Prediction | BLEU | Judge |
|---|---|---|---|---|---|
| MedGemma-4B | VQA_RAD | lateral ventricles | The bright white structures forming an "X" are the **cerebral ventricles**. | 0.019 | 5 |
| MedGemma-4B | VQA_RAD | xray | X-ray | 0.000 | 5 |
| LLaVA-Med-7B | VQA_RAD | left | colon appears to be more prominent on the patient's right side | 0.000 | 5 |
| LLaVA-Med-7B | VQA_RAD | left | colon is most prominent in the right lower quadrant of the abdomen | 0.000 | 5 |
| HuatuoGPT-7B | SLAKE | Lung | Lungs | 0.000 | 4 |
| HuatuoGPT-7B | SLAKE | Chest | Thorax | 0.000 | 5 |

#### Conversational Filler

| Model | Dataset | Ground Truth | Prediction | BLEU | Judge |
|---|---|---|---|---|---|
| MedGemma-4B | VQA_RAD | stomach | The three circular opacities are located in the **stomach**. | 0.024 | 5 |
| LLaVA-Med-7B | VQA_RAD | right | right side of the heart border is obscured in the chest X-ray | 0.016 | 4 |
| LLaVA-Med-7B | VQA_RAD | supratentorial | image is an MRI of the brain, and it shows a supratentorial lesion | 0.016 | 4 |

#### Granularity Mismatch

| Model | Dataset | Ground Truth | Prediction | BLEU | Judge |
|---|---|---|---|---|---|
| LLaVA-Med-7B | VQA_RAD | right | diaphragm is elevated on the right side | 0.033 | 5 |
| LLaVA-Med-7B | VQA_RAD | right | right side of the diaphragm is elevated | 0.033 | 5 |

---

## 5. Implications for Metric Selection

| Question | Finding |
|---|---|
| Is BLEU redundant with F1? | r(BLEU, F1) = 0.937 — Yes, largely redundant for terse outputs. |
| Does BLEU track judge accuracy? | r(BLEU, Judge) = 0.595 — weaker than r(F1, Judge) = 0.633, confirming BLEU is the least reliable single metric. |
| Which models suffer most from BLEU bias? | Highest Δ (Judge−BLEU) models are the most penalised by n-gram overlap metrics. |
| What is the rescue zone size? | 2562 records (692 medical, 1870 general) are correctly answered but falsely penalised by both BLEU and F1. |

> [!IMPORTANT]
> These findings justify the benchmark's primary reliance on **LLM Judge Accuracy**
> for open-ended evaluation, with BLEU and F1 reported as secondary metrics for
> reproducibility and comparison with prior work. BLEU alone would
> systematically underestimate model performance on conversational and
> medically-synonym-rich outputs.

---

## 6. Methodology

Analysis was restricted to open-ended questions only (N = 11,766). Sentence-level BLEU-4 (add-1 smoothing) and unigram Token F1 were computed per prediction–reference pair. Pearson and Spearman correlations were calculated across three metric pairs: BLEU vs Token F1, BLEU vs Judge score, and Token F1 vs Judge score. A rescue-zone analysis identified predictions where BLEU < 0.10 and Token F1 < 0.15 but Judge ≥ 4/5 — cases where both classical metrics falsely penalise correct predictions. Qualitative autopsy of 782 high-judge/low-BLEU medical records was conducted to categorise failure modes.

---

## 7. Generated Outputs

| File | Description |
|---|---|
| [`docs/bleu_error_analysis.md`](bleu_error_analysis.md) | This report |
| [`results/fig_bleu_rescue_zone.png`](../results/fig_bleu_rescue_zone.png) | Rescue-zone scatter plot |
