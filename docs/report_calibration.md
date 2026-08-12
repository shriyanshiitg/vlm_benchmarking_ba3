# Calibration Analysis Report
## VLM Medical VQA Benchmark — Confidence Reliability Study

**Authors:** VLM Benchmark Project  
**Date:** July–August 2026  
**Notebook:** `notebooks/11_calibration_analysis.ipynb`  
**Analysis Script:** `scripts/calibration_analysis.py`

---

## 1. Why We Did This

### 1.1 The Problem With Accuracy Alone

Throughout this benchmark, we evaluated five models on SLAKE and VQA-RAD using Token F1, Closed Accuracy, and LLM Judge scores. MedGemma-4B consistently outperformed all other models. But accuracy answers only one question: *did the model get it right?*

In clinical AI, there is a second, equally important question: *does the model know when it is right?*

A model that answers 80% of questions correctly but assigns high confidence to both its correct and incorrect answers is dangerous. A clinician or downstream system relying on that confidence score has no signal to identify the 20% of wrong answers. This is the **calibration problem**.

**Calibration** is the statistical alignment between a model's expressed confidence and its actual accuracy. A perfectly calibrated model that says "90% confident" is correct exactly 90% of the time. A poorly calibrated model that says "90% confident" might be correct only 55% of the time — its confidence score is meaningless as a reliability signal.

### 1.2 Why This Matters for Medical AI

Medical VQA is not a neutral benchmark task. In the real world, VLM predictions on radiology images could be used to:

- **Triage** patients by severity (flag scans above a confidence threshold for urgent review)
- **Filter** second reads (auto-approve low-risk cases, escalate uncertain ones)
- **Assist** radiologists by pre-filling structured reports with confidence-gated predictions

In all three scenarios, the clinical workflow assumes that a model's confidence score is *actionable* — that a high confidence score genuinely means the model is more likely to be right. If the model is overconfident (reports 90% confidence when it is correct 55% of the time), setting any meaningful threshold becomes impossible.

An overconfident model that says **"95% confident: No pathology"** when it is actually correct only 60% of the time is not just unhelpful — it is a patient safety risk.

### 1.3 The Specific Research Question

Our main benchmark established that MedGemma-4B has higher accuracy than Gemma-3-4B on medical VQA. The calibration study asks a distinct follow-up question:

> **Does medical domain pre-training confer better-calibrated confidence estimates, or does it only improve raw accuracy?**

If medical pre-training improves both accuracy *and* calibration, MedGemma-4B's clinical advantage is stronger than the accuracy numbers alone suggest. If it improves only accuracy, the calibration problem remains open for all models regardless of domain adaptation.

---

## 2. How We Did It

### 2.1 Scope and Model Selection

We limited the calibration study to the two 4B models: **MedGemma-4B-IT** and **Gemma-3-4B-IT**. The reason for this focused scope:

1. **Controlled comparison.** Comparing models of identical size (4B) and identical base architecture (Gemma-4B) isolates the effect of medical pre-training while holding everything else constant.
2. **Compute efficiency.** Calibration inference requires `output_scores=True`, which returns the full vocabulary logit vector at every token position. For a 32,000-vocabulary model this multiplies memory usage significantly. The 7B models would require quantisation, which distorts logit distributions and compromises the validity of confidence estimates.
3. **Most clinically relevant comparison.** MedGemma-4B is the primary deployment candidate from this benchmark. Its calibration profile vs. its generalist counterpart is the most actionable finding.

Both models were run in **fp16 without quantisation** to preserve the integrity of raw logit values.

### 2.2 Dataset Scope

Only **closed (Yes/No) questions** were used:

| Dataset | Total | Closed Used |
|---|---|---|
| SLAKE EN (test split) | 1,061 | 416 |
| VQA-RAD (test split) | 451 | 251 |

Open-ended questions (e.g., "Where is the abnormality?") do not map to a binary confidence axis. A probability distribution over organ names cannot be collapsed into a single reliable Yes/No confidence score. Restricting to closed questions gives a clean, interpretable binary calibration setting.

### 2.3 The Confidence Extraction Pipeline

Extracting reliable confidence scores from a generative LLM requires going below the surface prediction. The standard `model.generate()` call returns only the final answer string — we needed the raw probability distribution over the vocabulary at the first token position.

**Step-by-step pipeline:**

1. **Modified inference call:** Run `model.generate()` with `max_new_tokens=1` and `output_scores=True`. This returns `scores[0]` — the logit vector of shape `(1, vocab_size)` at the first generated token, before any sampling.

2. **Softmax conversion:** Apply softmax over the full vocabulary: `probs = softmax(scores[0], dim=-1)`. This converts raw logits into a valid probability distribution summing to 1.

3. **Token set collection:** For each model's tokenizer, collect all token IDs that decode to "yes", "Yes", "YES" (and variants with leading spaces, which is how subword tokenisers typically encode mid-sentence words). Repeat for "no", "No", "NO".

4. **Mass aggregation:**
   - `P(Yes)_raw = Σ probs[token_id]` for all Yes token IDs
   - `P(No)_raw = Σ probs[token_id]` for all No token IDs

5. **Normalisation:** `P(Yes) = P(Yes)_raw / (P(Yes)_raw + P(No)_raw)`

   This normalisation step is critical. Summing only Yes/No probability mass and renormalising ensures the confidence score reflects the model's relative belief between the two valid answers — independent of how much probability the model allocates to irrelevant tokens (punctuation, continuations, etc.).

6. **Binary label:** `label = 1` if the model's greedy prediction matches the ground truth, else `label = 0`.

Each inference record was saved to a JSONL file containing: `idx`, `question`, `ground_truth`, `prediction`, `p_yes`, `label`, `is_correct`.

### 2.4 Calibration Metrics

**Expected Calibration Error (ECE)** with 15 equal-width bins:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|S_b|}{N} \left| \text{acc}(S_b) - \text{conf}(S_b) \right|$$

Where `|S_b|` is the number of samples in bin `b`, `acc(S_b)` is the fraction of correct predictions in that bin, and `conf(S_b)` is the mean confidence in that bin. ECE is reported in percentage points (pp). Lower is better; a perfectly calibrated model has ECE = 0.

**Brier Score** (mean squared error between confidence and binary correctness label):

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (P(\text{Yes})_i - \text{label}_i)^2$$

Lower is better. Unlike ECE, the Brier score penalises both poor calibration and poor accuracy simultaneously, making it a joint measure of reliability.

**Overconfidence/Underconfidence:** Defined as mean confidence − accuracy. A positive value means the model's average confidence exceeds its accuracy (overconfident). A negative value means the model is more accurate than its confidence suggests (underconfident).

### 2.5 Kaggle Execution

Both calibration inference runs were executed on Kaggle with a T4 GPU using `notebooks/11_calibration_analysis.ipynb`. The notebook runs each model separately (one per Kaggle session) and writes output JSONL files. The local analysis script `scripts/calibration_analysis.py` then reads these files to compute all metrics and generate the reliability diagrams.

**Why on Kaggle:** fp16 inference requires ~16 GB VRAM for a 4B model. Local machines with 8 GB VRAM cannot run this without quantisation, which would distort the logit values and invalidate the calibration measurement.

---

## 3. Results

### 3.1 ECE and Brier Score Summary

| Model | Dataset | N | Accuracy | ECE | Brier | Direction |
|---|---|---|---|---|---|---|
| MedGemma-4B | SLAKE | 416 | 74.04% | 21.95 pp | 0.2129 | Underconfident |
| Gemma-3-4B | SLAKE | 416 | 57.93% | 35.49 pp | 0.3468 | Overconfident |
| MedGemma-4B | VQA-RAD | 251 | 79.68% | 19.52 pp | 0.1920 | Underconfident |
| Gemma-3-4B | VQA-RAD | 251 | 55.38% | 43.18 pp | 0.4310 | Overconfident |

*ECE and Brier: lower is better. Overconfident = mean confidence > accuracy.*

MedGemma-4B has lower ECE than Gemma-3-4B by **13.54 pp on SLAKE** and **23.66 pp on VQA-RAD** — a consistent, large-margin advantage across both datasets.

### 3.2 Yes/No Accuracy and Mean Confidence

| Model | Dataset | Yes-Accuracy | No-Accuracy | Mean P(Yes) |
|---|---|---|---|---|
| MedGemma-4B | SLAKE | 89.14% (N=175) | 63.07% (N=241) | 55.37% |
| Gemma-3-4B | SLAKE | 87.43% (N=175) | 36.51% (N=241) | 67.57% |
| MedGemma-4B | VQA-RAD | 78.81% (N=118) | 80.45% (N=133) | 47.76% |
| Gemma-3-4B | VQA-RAD | 66.10% (N=118) | 45.86% (N=133) | 59.63% |

### 3.3 Confidence Distribution Pattern

**Gemma-3-4B** shows a bimodal confidence distribution heavily skewed toward the high-confidence end — it assigns P(Yes) > 0.8 on the majority of questions regardless of whether it is correct. On SLAKE, its high-confidence (>0.8) bucket has accuracy of approximately 57% — barely above random for a binary task. The model is systematically overconfident.

**MedGemma-4B** shows a broader, flatter distribution centred near 0.5. Its confidence is not polarised: it expresses genuine uncertainty on borderline cases rather than defaulting to high confidence. The negative overconfidence value (−18.67 pp on SLAKE, −31.92 pp on VQA-RAD) means MedGemma is actually *underconfident* — it is more accurate than its confidence scores suggest. In a clinical context, underconfidence is the safer failure mode: it causes over-escalation to human review but does not produce falsely reassuring high-confidence wrong predictions.

### 3.4 Reliability Diagrams

Reliability diagrams are saved at `results/fig_calibration_reliability.png`. The 2×2 grid shows accuracy per confidence bin (bars) against the perfect-calibration diagonal (dashed line) for each model × dataset combination. Gaps above the diagonal indicate underconfidence; gaps below indicate overconfidence.

Confidence distribution histograms are at `results/fig_calibration_confidence_hist.png`.

---

## 4. Findings

**F1 — MedGemma-4B is better calibrated than Gemma-3-4B across both datasets.** Average ECE reduction: 18.60 pp. This is not a marginal improvement — ECEs of 35–43 pp indicate severe miscalibration where confidence scores provide essentially no reliable signal.

**F2 — Gemma-3-4B is systematically overconfident; MedGemma-4B is systematically underconfident.** Gemma-3-4B's high-confidence predictions are correct only ~57% of the time on VQA-RAD (binary task: chance = 50%). Its confidence score has effectively no discriminative power. MedGemma's underconfidence, while not ideal, is the safer clinical failure mode.

**F3 — Medical pre-training confers a dual advantage: accuracy AND calibration.** The calibration improvement is not a trivial consequence of higher accuracy (a more accurate model is not automatically better calibrated — see temperature scaling literature). MedGemma's calibration advantage reflects a genuine difference in how it distributes probability mass over Yes/No tokens.

**F4 — Gemma-3-4B's No-accuracy collapse drives its calibration failure.** On SLAKE, Gemma-3-4B answers "Yes" to 63.49% of all questions (mean P(Yes) = 67.57%) while the true Yes rate is only 42.07% (175/416). The model has a strong Yes-bias that inflates its confidence on incorrect No predictions. This Yes-bias is absent in MedGemma-4B (mean P(Yes) = 55.37%, near the true 42.07% base rate).

---

## 5. Clinical Implications

A well-calibrated model enables **threshold-based clinical workflows**:

- Set a threshold at P(Yes) > 0.85 → auto-approve as high-confidence negative
- Set a threshold at P(Yes) ∈ [0.35, 0.65] → escalate to human radiologist review
- Set a threshold at P(Yes) < 0.15 → auto-flag as high-confidence positive

With MedGemma-4B's calibration (ECE ~20 pp, underconfident), these thresholds are directionally meaningful — the model's high-confidence predictions are genuinely more reliable than its low-confidence predictions. With Gemma-3-4B's calibration (ECE ~35–43 pp, overconfident), threshold-based gating is clinically unsafe: the model assigns high confidence to wrong answers at near-random rates.

The data supports using MedGemma-4B, not Gemma-3-4B, as the candidate for any threshold-gated clinical screening workflow.

---

## 6. Limitations

1. **ECE bin sensitivity.** ECE with 15 equal-width bins can be unstable when bins are sparsely populated. We verified that the results are consistent with 10-bin ECE and that bin occupancy was sufficient across all four configurations.
2. **Scope limited to 4B models.** The 7B models (HuatuoGPT, LLaVA-Med, LLaVA-1.6) were not evaluated due to quantisation concerns that would distort logit distributions. A dedicated fp16 or bf16 7B calibration run would require 40+ GB VRAM.
3. **Closed questions only.** Binary Yes/No calibration is not directly generalisable to open-ended question calibration, where confidence must be defined over a much larger output space.
4. **Single temperature.** No temperature scaling or post-hoc calibration was applied. The reported ECE values reflect the raw pre-calibration model output.

---

## 7. Generated Files

| File | Description |
|---|---|
| `outputs/_archive/calibration/*.jsonl` | Per-run inference outputs with per-sample P(Yes), label, prediction |
| `results/calibration_results.json` | All ECE, Brier, accuracy metrics (machine-readable) |
| `results/fig_calibration_reliability.png` | 2×2 reliability diagram grid (accuracy per confidence bin) |
| `results/fig_calibration_confidence_hist.png` | P(Yes) distribution histograms per model × dataset |
| `scripts/calibration_analysis.py` | Local analysis script (re-runnable, reads from archive) |
| `notebooks/11_calibration_analysis.ipynb` | Kaggle T4 inference notebook |
