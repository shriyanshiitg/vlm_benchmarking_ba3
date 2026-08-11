# Research Gap Analysis & Project Status
## VLM Medical VQA Benchmark — Updated 26 July 2026

---

## What Has Been Completed

The following items have been **fully addressed** and are no longer open:

| Item | Status | Key Result |
|---|---|---|
| HuatuoGPT not evaluated on DICOM | ✅ Done | 77.8% closed acc, F1 = 52.1% |
| No multi-slice DICOM evaluation | ✅ Done | 2×2 grid at 20/40/60/80% depth; all 4 models evaluated |
| Medical models not tested on general datasets | ✅ Done | MedGemma, LLaVA-Med, HuatuoGPT all on VQAv2 + OK-VQA |
| LLM-as-a-Judge on general-domain results | ✅ Done | Judge results for all 3 medical models on VQAv2/OK-VQA |
| No statistical significance testing | ✅ Done | 10k-iteration bootstrap CI + paired permutation tests; 5 figures generated |
| LLM Judge Reliability Not Formally Validated | ✅ Done | Cohen's κ = 0.696 (Substantial) vs independent 70B judge on 499-record stratified sample |
| BLEU Underused in Error Analysis | ✅ Done | Full 5-phase BLEU audit; 2,562 rescue-zone records identified; report in `docs/bleu_error_analysis.md` |
| S-CoT Tested on Only One Model/Dataset | ✅ Done | Extended to HuatuoGPT/SLAKE and MedGemma/VQA-RAD; finding is architecture-specific, not general |
| No automated QA generation pipeline for DICOM | ✅ Done | End-to-end LLM pipeline implemented; 16 QA pairs generated and validated on 2-patient dataset |
| Few-Shot Experiment Not Conducted | ✅ Done | 200-sample SLAKE subset, 0/1/3-shot × 2 models; no significant improvement at any shot count (Gemma-3 3-shot: ΔF1 = −3.45 pp, p = 0.215; LLaVA 3-shot: ΔF1 = −3.34 pp, p = 0.349); domain bottleneck confirmed architectural |
| Calibration Analysis Not Conducted | ✅ Done | MedGemma-4B vs Gemma-3-4B on 416 SLAKE + 251 VQA-RAD closed questions; MedGemma ECE = 21.95/19.52 pp vs Gemma-3 ECE = 35.49/43.18 pp; MedGemma is better calibrated and underconfident; Gemma-3 is overconfident with high-conf accuracy of only ~57% |

The project now has a **complete 5-model × 4-dataset × 7-metric** evaluation matrix with statistically validated results, a clinical DICOM evaluation covering 4 models × 2 evaluation strategies (single-slice + multi-slice), a full cross-domain analysis in both directions (generalist → medical and medical → general), five completed methodological validation studies (judge reliability, BLEU audit, S-CoT extension, few-shot experiment, calibration analysis), and an automated QA generation pipeline for the private DICOM dataset.

---

## Completed Work — Detailed Summaries

### Judge Reliability Study

**Report:** `docs/inter_rater_agreement_report.md`

To formally validate the Llama-3.1-8B judge used throughout the benchmark, an independent Llama-3.3-70B-Versatile judge (running on Groq LPU hardware — a completely independent inference stack at full precision) was run on a stratified 499-record sample designed to stress-test boundary cases. The sample covered score-1 anchors, score-5 anchors, score-3 ambiguous cases, and open-ended medical questions. Both judges used the identical `MEDICAL_JUDGE_PROMPT`.

| Metric | Value | Interpretation |
|---|---|---|
| Linear-weighted Cohen's κ | **0.696** | ✅ Substantial agreement |
| Pearson r | 0.810 | Very strong linear correlation |
| Spearman ρ | 0.818 | Very strong rank correlation |
| Mean Absolute Difference | 0.567 score points | Good (judges rarely disagree by more than half a point) |
| Exact agreement rate | 66.3% | — |
| Adjacent (±1) agreement rate | 80.8% | — |
| Systematic bias (8B − 70B) | +0.194 | No significant inflation or deflation |

The 8B judge is reliable for reporting paper-quality conclusions. The highest disagreement is on LLaVA-Med outputs (κ = 0.535), which is expected — LLaVA-Med's verbose sentence-form answers are inherently more ambiguous to score than single-word responses.

---

### BLEU Correlation Analysis

**Report:** `docs/bleu_error_analysis.md`

A full audit was conducted across all 11,766 open-ended predictions to determine whether BLEU-4 provides evaluation signal distinct from Token F1 and LLM Judge Accuracy, and to characterise where and why it fails in medical VQA.

**Core correlation findings:**

| Metric Pair | Correlation | Interpretation |
|---|---|---|
| BLEU ↔ Token F1 | r = 0.937 | Near-redundant — BLEU adds almost no information beyond F1 for terse models |
| BLEU ↔ Judge Score | r = 0.595 | Weakest metric pair — BLEU is the least reliable proxy for semantic correctness |
| F1 ↔ Judge Score | r = 0.633 | Stronger than BLEU-Judge, but still modest |

**Rescue zone analysis:** 2,562 predictions scored BLEU < 0.10 and F1 < 0.15 but received Judge ≥ 4/5. These are true positives that both classical metrics falsely call failures — 692 in the medical domain and 1,870 in the general domain.

**Qualitative autopsy** of 782 high-judge/low-BLEU medical records identified three failure categories: Medical Synonymy (379 cases, e.g., "Lung" vs "Lungs", "Chest" vs "Thorax"), Conversational Filler (49 cases — correct answer embedded in a sentence), and Granularity Mismatch (13 cases — specific sub-classification vs broader ground-truth term).

These findings justify the benchmark's primary reliance on LLM Judge Accuracy for open-ended evaluation, with BLEU and Token F1 retained as secondary metrics for reproducibility.

---

### S-CoT Extension — Multiple Model/Dataset Pairs

**Report:** `docs/report_scot_extension.md`

The original S-CoT experiment showed that a structured 4-step chain-of-thought prompt degraded MedGemma-4B on SLAKE by −5.0 pp F1 (p < 0.001). The open question was whether this failure was architecture-agnostic or specific to that one combination. Two additional model–dataset pairs were tested using the identical prompt.

| Model | Dataset | N | Baseline F1 | S-CoT F1 | ΔF1 | p-value |
|---|---|---|---|---|---|---|
| MedGemma-4B | SLAKE | 440 | 70.5% | 65.5% | **−5.0 pp** | < 0.001 ★★★ |
| HuatuoGPT-7B | SLAKE | 1,061 | 47.9% | 47.1% | −0.8 pp | 0.570 ns |
| MedGemma-4B | VQA-RAD | 451 | 62.5% | 61.6% | −0.9 pp | 0.576 ns |

**Conclusion:** The S-CoT degradation is not architecture-agnostic. It is specific to MedGemma-4B on SLAKE. The same 4B model that fails under structured prompting on SLAKE is unaffected on VQA-RAD. The 7B HuatuoGPT model, with a different backbone and approximately twice the parameters, shows no statistically significant degradation on the same SLAKE data — its open-ended F1 even improves marginally (+1.3 pp).

The most likely explanation is that generative drag is real but interacts with both model capacity (the 7B model sustains structured generation better than the 4B) and dataset visual complexity (SLAKE's heterogeneous multi-organ CT/MRI imagery demands more visual working memory than VQA-RAD's more homogeneous chest X-rays).

---

### Automated DICOM QA Generation Pipeline

**Report:** `docs/automated_qa_pipeline_strategy.md`

An end-to-end automated pipeline was built and validated to replace the manual QA construction process used for the 2-patient clinical dataset. The pipeline takes any radiology report text as input, calls the Gemini API (free tier) with a carefully designed clinical extraction prompt, validates each returned QA pair against seven hard rules, maps the abstract series type to an actual DICOM series number using fuzzy string matching on metadata already embedded in the CSV, and saves a `.jsonl` file ready for the existing evaluation harness.

Validated on both existing patients: 16 QA pairs generated, 15/16 series-mapped, zero hallucinations detected. Every generated answer was directly traceable to a stated finding in the corresponding radiologist report, and all answer polarities matched the manual ground truth. The pipeline is ready to scale to the full dataset on arrival with no code changes.

---

### Few-Shot Experiment — In-Context Learning vs the Domain Gap

**Report:** `docs/report_fewshot_experiment.md` | **Section in main report:** `docs/report.md` Section 15

To determine whether in-context learning can close the performance gap between generalist and medical VLMs without fine-tuning, Gemma-3-4B and LLaVA-1.6-7B were evaluated on a stratified 200-sample SLAKE subset under 0-shot, 1-shot, and 3-shot conditions. Few-shot examples were drawn exclusively from the SLAKE training split (one Modality, one Organ, one Abnormality example), prepended as prior conversation turns.

| Model | Condition | Overall F1 | ΔF1 | Closed Acc | Open F1 | p-value |
|---|---|---|---|---|---|---|
| Gemma-3-4B | 0-shot | 57.27% | — | 80.00% | 34.53% | — |
| Gemma-3-4B | 1-shot | 56.08% | −1.19 pp | 69.00% | 43.15% | 0.698 ns |
| Gemma-3-4B | 3-shot | 53.82% | **−3.45 pp** | 71.00% | 36.64% | 0.215 ns |
| LLaVA-1.6-7B | 0-shot | 38.24% | — | 54.00% | 29.47% | — |
| LLaVA-1.6-7B | 1-shot | 34.00% | −4.24 pp | 57.00% | 11.00% | 0.188 ns |
| LLaVA-1.6-7B | 3-shot | 34.90% | **−3.34 pp** | 52.00% | 17.80% | 0.349 ns |

**Conclusion:** Neither model showed statistically significant improvement at any shot count. All four permutation tests returned p ≥ 0.05. F1 trends slightly negative under few-shot for both models. The largest degradations are in Organ and Abnormality questions — the categories most dependent on domain-specific anatomical knowledge. Combined with the S-CoT and parameter-scaling findings, this establishes convergent evidence that **the domain gap is architectural, not informational**, and that medical fine-tuning is mandatory rather than merely beneficial.

### Calibration Analysis — Confidence Reliability on Closed Questions

**Report:** `docs/report_calibration.md` | **Section in main report:** `docs/report.md` Section 16

MedGemma-4B and Gemma-3-4B were evaluated on all closed (Yes/No) questions from SLAKE EN (416 records each) and VQA-RAD (251 records each). Confidence was extracted as `P(Yes)` from the softmax distribution at the first generated token using `output_scores=True`, `max_new_tokens=1`. ECE was computed with 15 equal-width bins; Brier score provides a complementary metric.

| Model | Dataset | Accuracy | ECE | Brier | Overconfidence |
|---|---|---|---|---|---|
| MedGemma-4B | SLAKE | 74.04% | **21.95 pp** | 0.2129 | −18.7 pp (underconfident) |
| MedGemma-4B | VQA-RAD | 79.68% | **19.52 pp** | 0.1920 | −31.9 pp (underconfident) |
| Gemma-3-4B | SLAKE | 57.93% | 35.49 pp | 0.3468 | +9.6 pp (overconfident) |
| Gemma-3-4B | VQA-RAD | 55.38% | 43.18 pp | 0.4310 | +4.3 pp (overconfident) |

**Conclusion:** MedGemma-4B is substantially better calibrated on both datasets (ECE Δ = −13.54 pp on SLAKE, −23.66 pp on VQA-RAD). Gemma-3-4B is systematically overconfident — when it predicts with high confidence, it is correct only ~57% of the time (barely above random for a binary task). MedGemma-4B is underconfident, which is the safer failure mode for clinical deployment. The claim is confirmed: **MedGemma-4B is not just more accurate — it is better calibrated**, making its confidence scores more actionable for threshold-based clinical workflows.

---

### G3 — Clinical Validity Gaps *(Not Addressable Without External Resources)*

**DICOM QA pairs not radiologist-verified.** The 16 QA pairs were constructed from radiologist reports by the researcher, not validated by a clinician. This is the same methodology used to build VQA-RAD, but the limitation must be explicitly acknowledged. Mitigation: document that questions were derived verbatim from radiologist impression statements.

**DICOM limited to knee MRI.** Results from 2 patients do not generalise to other modalities. Mitigation: frame in the paper as a methodology proof-of-concept, not a definitive clinical claim.

---

## Recommended Next Steps (Prioritized)

| Priority | Task | Estimated Effort | Output |
|---|---|---|---|
| **1 — Medium** | **Modality-specific leaderboard** — filter SLAKE/VQA-RAD JSONL by CT/MRI/X-ray metadata | 0.5 days | Per-modality performance table for paper appendix |
| **2 — Lower** | **Ensemble mechanism** — majority vote (closed) + score aggregation (open) on existing SLAKE JSONLs | 1 day | Does model combination outperform best single model? |

---

## Current Project Strength Assessment

The project already contains, without any additional experiments:

- A **statistically validated** (bootstrap CI + permutation test) 5-model × 4-dataset benchmark — most student projects omit significance testing entirely
- **Named, quantified failure modes** — attention hijacking, visual blindness, pathology bias, verbosity penalty, catastrophic forgetting paradox — this is analysis depth, not just number-reporting
- **The catastrophic forgetting paradox** — F1 ~12% but Judge Accuracy ~70–82% for medical models on general data — a genuinely counterintuitive finding with a mechanistic explanation
- **Private clinical DICOM data** with a multi-slice pipeline — very few interns touch proprietary clinical imaging
- **A formally validated evaluation framework** — the judge reliability study (κ = 0.696) and BLEU audit together justify the entire metric stack
- **A negative experimental result with scope clarified** — S-CoT degrades only MedGemma-4B on SLAKE, not HuatuoGPT or VQA-RAD; the extension turns a single-point finding into a nuanced architectural claim
- **Convergent empirical evidence across three experiments** — parameter scaling (Section 8), structured prompting (Section 14), and few-shot prompting (Section 15) all independently confirm that the domain gap is architectural, not informational
- **A reproducible automated pipeline** — the QA generation pipeline means the entire clinical evaluation can be scaled to the full dataset with a single command
- **A confirmed calibration advantage** — MedGemma-4B achieves ECE of 21.95/19.52 pp vs Gemma-3-4B’s 35.49/43.18 pp on SLAKE/VQA-RAD; the claim that medical pre-training produces not just more accurate but better-calibrated confidence estimates is now empirically validated (Section 16)

The core experimental programme is complete. All five planned methodological validation studies have been conducted and reported. Remaining optional directions (modality-specific leaderboard, ensemble mechanism) would strengthen the paper further but are not required for a complete, defensible submission.

---

## Things to Do Next

*Based on a full review of all completed experiments (Sections 8–16 of `docs/report.md`) and the initial task objective — "Evaluate Mistral and Gemma model performance across VLM and LLM task categories" — the following gaps remain open. Each item is grounded in a specific finding from the existing data.*

---

### T1 — Modality-Specific Performance Leaderboard ✅ Done
**Importance:** Medium | **Effort:** 0.5 days | **Type:** Analysis (no new inference)

**Result:** Post-hoc decomposition of all five SLAKE JSONL files by the `modality` field. MedGemma-4B leads every modality on both Token F1 and Judge accuracy. Key findings: X-Ray is the strongest modality for MedGemma (77.50% F1, 79.50% Judge); CT is the hardest for generalist models (LLaVA-1.6: 32.93% F1, 42.37% Judge); LLaVA-1.6 has a 17.74 pp X-Ray vs CT Judge accuracy gap, confirming its domain-specific spatial failure. VQA-RAD excluded — no per-image modality metadata available.

**Section in main report:** `docs/report.md` Section 17. **Chart:** `results/fig_modality_leaderboard.png`.

---

### T2 — Calibration Analysis Extended to All Five Models
**Importance:** Medium | **Effort:** 2–3 days (Kaggle T4) | **Type:** New experiment

**Gap:** Section 16 measured calibration (ECE, Brier score) for only two models — MedGemma-4B and Gemma-3-4B. Three models evaluated in the full benchmark (HuatuoGPT-7B, LLaVA-1.6-7B, LLaVA-Med-7B) have no calibration measurement. The error analysis (C3) showed HuatuoGPT has a very different score distribution (lowest catastrophic failure rate at 21.81% anatomical hallucination) and LLaVA-Med has the most dangerous partial-competence profile. Whether these behavioral differences translate into calibration differences is unknown.

**What it adds:** HuatuoGPT's low anatomical hallucination rate suggests it may be more calibrated than other 7B models. LLaVA-Med's confident hallucination pattern suggests it may have the worst ECE of all five models. Confirming this would extend the calibration claim from a 2-model comparison to a full 5-model ranking, making the result much more publishable.

**Deliverable:** Three additional Kaggle runs (HuatuoGPT, LLaVA-1.6, LLaVA-Med on SLAKE or VQA-RAD closed questions). Extend `scripts/calibration_analysis.py` and add to `docs/report_calibration.md`.

---

### T3 — Spatial Reasoning Targeted Evaluation
**Importance:** Medium | **Effort:** 1 day (analysis) | **Type:** Deep analysis of existing data

**Gap:** Error analysis A3 found extremely high spatial failure rates across all models — Gemma-3 fails 73.44% of spatial questions on SLAKE and 88.81% on VQAv2. MedGemma cuts this to 40.00% on SLAKE but still fails nearly half of all spatial queries. Critically, an inversion was found: Gemma-3 is *worse* spatially on general images (88.81%) than on medical images (60.98%), which is an unexpected and unexplained result. This finding is currently one paragraph in the error report with no dedicated analysis.

**What it adds:** A systematic breakdown of spatial failure types — left/right orientation errors, counting failures, relative size errors, anatomical plane identification (axial/coronal/sagittal) — would quantify exactly *which* spatial sub-skill is broken for each model. The axial/coronal/sagittal plane question is entirely unique to medical imaging and is one of the hardest zero-shot tasks (Gemma-3: 22.41% Judge accuracy on Plane questions). This is a standalone, publishable finding about spatial grounding in medical VLMs.

**Deliverable:** Add Section to `docs/error_analysis_report.md` with a sub-type breakdown of spatial failures (orientation, counting, plane identification) across all 5 models on SLAKE and VQA-RAD.

---

### T4 — Ensemble / Model Combination Experiment ✅ Done
**Importance:** Low-Medium | **Effort:** 1 day | **Type:** Analysis (no new inference)

**Gap:** The failure overlap analysis (C2) showed that MedGemma and HuatuoGPT share 623 wins but also 232 failures — approximately 22% of questions that *neither* medical model can answer. Conversely, there are questions only HuatuoGPT gets right that MedGemma fails (47 exclusive wins), and vice versa (159 exclusive wins for MedGemma). A simple ensemble — majority vote for closed questions, best-score selection for open questions — could theoretically capture these complementary strengths.

**What it adds:** If ensemble outperforms the best single model, it demonstrates that the models make different errors and are complementary, which is a publishable system-level finding. If it does not, it quantifies the ceiling of model combination without fine-tuning. Either result is informative. Requires only post-hoc computation on existing JSONL files.

**Deliverable:** `scripts/ensemble_analysis.py` using existing SLAKE/VQA-RAD JSONL files. Add results table to `docs/report.md`.

---

### T5 — Knowledge-Grounded (KG) Question Failure Analysis
**Importance:** Medium | **Effort:** 1 day | **Type:** Deep analysis of existing data

**Gap:** The SLAKE content-type breakdown (C1) revealed that Knowledge-Grounded (KG) questions are the second-hardest category for all models: MedGemma achieves only 52.03% Judge accuracy, HuatuoGPT 47.30%, and Gemma-3 a mere 32.43%. KG questions require external medical knowledge not present in the image (e.g., "What is the clinical significance of this finding?"). This is a qualitatively different failure mode from spatial or modality failures — it is a *knowledge retrieval* failure, not a visual grounding failure. The 148 KG questions in SLAKE have never been analyzed as a distinct category.

**What it adds:** Characterizing the exact type of knowledge required (pharmacology, clinical protocols, epidemiology, anatomy definitions) and which model retrieves it best would produce a standalone finding about knowledge-intensive medical VQA. It would also reveal whether HuatuoGPT's PubMedVision training gives it an advantage specifically on biomedical knowledge questions.

**Deliverable:** Filter existing SLAKE JSONL outputs to KG questions only. Categorize KG questions by knowledge type. Add as sub-section to `docs/error_analysis_report.md`.

---

### T6 — LLM-Only (Text-Only) Baseline
**Importance:** High | **Effort:** 2 days | **Type:** New experiment

**Gap:** Every experiment in this benchmark uses vision-language models — models that receive both an image and a question. There is no text-only baseline. A critical open question in medical VQA is: *how much of the answer can be inferred from the question text alone, without looking at the image?* If a model achieves 60% accuracy without seeing the image, then the 70% accuracy of MedGemma is not evidence of visual grounding — it is evidence of dataset bias in question phrasing. This is a known issue in VQA benchmarks generally (the "language prior" problem) and has never been measured here.

**What it adds:** Running the same SLAKE and VQA-RAD questions through the text-only versions of Gemma-3-4B and MedGemma-4B (without the image) would establish a language-prior baseline. A large gap between text-only and vision-enabled performance confirms genuine visual grounding. A small gap would be a significant methodological concern. This experiment is straightforward and requires only modifying the inference pipeline to pass no image.

**Deliverable:** A short inference run on SLAKE closed questions (416 records, no image). Compare text-only accuracy vs. vision-enabled accuracy per model. Add as limitation mitigation section to `docs/report.md`.

---

### T7 — LLaVA-Med Failure Mode Report
**Importance:** Medium | **Effort:** 0.5 days | **Type:** Analysis of existing data

**Gap:** The error analysis identified LLaVA-Med as "the most clinically dangerous model in the cohort" — highest anatomical hallucination rate (47.17%), concentrated Score 3–4 distribution (medically plausible but wrong), and near-zero drift rate (always structurally compliant, always confidently wrong). These findings are scattered across multiple sections (A2, B3, B4, C3, A4) with no consolidated model card. Given that LLaVA-Med is a widely cited model that practitioners might consider deploying, a consolidated failure mode report would be a valuable standalone contribution.

**What it adds:** A dedicated LLaVA-Med failure analysis would synthesize all existing findings into a single deployment-risk assessment, covering hallucination taxonomy, calibration status (if T2 is conducted), closed/open performance gap, and comparison against MedGemma on matched questions. Could be published as a model card or safety assessment.

**Deliverable:** A new document `docs/llava_med_failure_analysis.md` synthesizing all existing findings about LLaVA-Med's failure patterns.

---

### T8 — Prompt Template Sensitivity Study ✅ Done
**Importance:** Low | **Effort:** 1–2 days (Kaggle T4) | **Type:** New experiment

**Gap:** The v2 prompt protocol (Section 6.2) was designed specifically based on the MedGemma technical report and boosted F1 by +14.55 pp over v1. However, all five models use the *identical* prompt template. There is no measurement of whether this template is optimal for non-MedGemma architectures. HuatuoGPT uses a Qwen2.5-VL backbone with a different instruction format; its optimal prompt may be different. LLaVA-Med was trained on PMC-15M with a conversational format that may respond better to different phrasing.

**What it adds:** Testing 2–3 minor prompt variants (e.g., removing the radiology-specific framing for VQA-RAD, changing "Final Answer:" to "Answer:") on a 200-sample subset for HuatuoGPT and LLaVA-Med would determine whether the v2 template systematically underestimates their capabilities. If HuatuoGPT improves with a different template, all its numbers should be re-reported. If it does not, the benchmark's prompt universality is validated.

**Deliverable:** A 200-sample prompt sensitivity table for HuatuoGPT and LLaVA-Med. Add as a methodology note to `docs/report.md` Section 6.

---

*Items T1, T5, T7 require only analysis of existing data and can be completed without any additional Kaggle compute. Items T6 and T2 require new inference runs and should be prioritised by their research significance.*
