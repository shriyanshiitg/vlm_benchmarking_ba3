# Research Gap Analysis & Future Directions
## VLM Medical VQA Benchmark — Updated 8 July 2026

---

## What Has Been Completed

The following items from the original gap analysis have been **fully addressed** and are no longer open:

| Item | Status |
|---|---|
| HuatuoGPT not evaluated on DICOM | ✅ Done — 77.8% closed acc, F1 = 52.1% |
| No multi-slice DICOM evaluation | ✅ Done — 2×2 grid at 20/40/60/80% depth; all 4 models evaluated |
| Medical models not tested on general datasets | ✅ Done — MedGemma, LLaVA-Med, HuatuoGPT all on VQAv2 + OK-VQA |
| LLM-as-a-Judge on general-domain results | ✅ Done — judge results for all 3 medical models on VQAv2/OK-VQA in report.md |
| No statistical significance testing | ✅ Done — 10k-iteration bootstrap CI + paired permutation tests on all JSONL files; 5 figures generated |
| LLM Judge Reliability Not Formally Validated | ✅ Done — inter-rater agreement study completed (Cohen’s κ = 0.696, Substantial) against 70B judge |
| BLEU Underused in Error Analysis | ✅ Done — full 5-phase BLEU audit completed; 2,562 rescue-zone records identified; report in docs/bleu_error_analysis.md |

The project now has a **complete 5-model × 4-dataset × 7-metric** evaluation matrix with statistically validated results, a clinical DICOM evaluation covering 4 models × 2 evaluation formats (single-slice + multi-slice), and a full cross-domain analysis in both directions (generalist → medical and medical → general).

---

## Remaining Gaps

### G1 — Metric and Evaluation Gaps (Still Open)


#### G1.2 — Calibration Analysis Absent *(High Priority, Clinically Novel)*
The benchmark measures *what* models predict but not *how confident* they are. For closed (Yes/No) questions, model calibration — whether a model's stated confidence matches its actual accuracy — is a critical safety property in clinical AI. An overconfident wrong prediction is more dangerous than a correctly uncertain one.

**What's missing:** Per-model reliability diagrams and Expected Calibration Error (ECE) scores derived from token-level softmax probabilities on the "Yes"/"No" vocabulary tokens.

**Why it matters:** "MedGemma is not just more accurate but better calibrated, making it safer for clinical screening" is a publishable, clinically grounded claim that no other benchmark in this space makes.

**Implementation note:** Requires `output_scores=True` in HuggingFace `generate()` and `max_new_tokens=1` to capture the first-token distribution cleanly. Only needs to run on closed questions from SLAKE + VQA-RAD for MedGemma vs. Gemma-3 (the most informative pair). Runs in ~30 mins on a T4.

---


### G2 — Experimental Depth Gaps (Still Open)

#### G2.1 — Few-Shot Experiment Not Conducted *(High Priority)*
All evaluations are zero-shot. The core finding — that domain pre-training trumps parameter count — could be meaningfully extended by asking: **does in-context learning (few-shot) close this gap without fine-tuning?**

**Design:** Evaluate Gemma-3-4B and LLaVA-1.6-7B on a stratified 200-sample subset of SLAKE under 0-shot, 1-shot, and 3-shot conditions. Few-shot examples should be curated from the training split (never the test split), covering one Modality, one Organ, and one Abnormality question.

**Either result is publishable:**
- If few-shot *closes* the gap → domain adaptation is possible without fine-tuning; few-shot is an efficient clinical deployment path
- If few-shot *does not close* the gap → proves the domain bottleneck is architectural, not informational; validates why fine-tuning is mandatory

**Effort:** Medium. Loop modification to inference scripts + 200-sample run per condition. ~2–3 hours of T4 compute total across all conditions.

---

#### G2.2 — S-CoT Tested on Only One Model/Dataset
The S-CoT negative result (MedGemma SLAKE: −9.37 pp F1 from structured prompting) is a valid finding but lacks generalizability. Does the same generative drag affect HuatuoGPT, which has a different backbone and higher base CoT hallucination rate (34.86%)? Does it hold on VQA-RAD where questions tend to be shorter?

**What's missing:** Run the same S-CoT protocol on HuatuoGPT/SLAKE and on MedGemma/VQA-RAD.

**Effort:** Low — prompt is already designed, infrastructure exists. ~1 hour of T4 compute.

---

### G3 — Clinical Validity Gaps (Not Addressable Without External Resources)

#### G3.1 — DICOM QA Pairs Not Radiologist-Verified
The 16 QA pairs were constructed by the researcher from radiologist reports, not validated by a clinician. This is the same methodology used to build VQA-RAD, but the limitation stands and should be explicitly acknowledged in the paper.

**Mitigation (no clinical collaborator):** Document that questions were derived verbatim from radiologist impression statements — this is the strongest defensible claim available.

---

#### G3.2 — DICOM Limited to Knee MRI
Results from 2-patient knee MRI do not generalize to chest X-ray, brain CT, or abdominal imaging.

**Mitigation:** Frame in the paper as a "methodology proof-of-concept" for private DICOM evaluation, not as a definitive clinical claim. Acknowledge this as an open direction.

---

## Recommended Next Steps (Prioritized)

Given approximately **2 remaining weeks + possible 1-month extension:**

### This Week (High-Impact, No New Large-Scale Inference)

| # | Task | Effort | Output |
|---|---|---|---|
| **1** | **Judge reliability study** — 100 manual annotations on stratified SLAKE sample + free API (Groq Llama-3-70B or Gemini-Flash) on same 100 samples | 1–2 days | Inter-rater agreement (Cohen's κ / Spearman ρ); validates the entire judge framework |
| **2** | **BLEU correlation analysis** — correlate per-sample BLEU with Judge score on open-ended questions across all models | 0.5 days | Answers whether BLEU is a useful proxy or redundant with Judge |
| **3** | **S-CoT extension** — run existing S-CoT prompt on HuatuoGPT/SLAKE and MedGemma/VQA-RAD | 1 day | Determines if S-CoT failure is architecture-agnostic |

### Next Week (Requires New Inference)

| # | Task | Effort | Output |
|---|---|---|---|
| **4** | **Few-shot experiment** — 0/1/3-shot on Gemma-3 + LLaVA-1.6, 200-sample SLAKE subset | 2–3 days | Clean table + binary conclusion; converts the project from descriptive to prescriptive |
| **5** | **Calibration analysis** — closed-question token probabilities for MedGemma vs Gemma-3 on SLAKE + VQA-RAD | 2–3 days | Reliability diagrams + ECE scores; clinically novel, no other benchmark does this |

### If Extension Granted (Good Additions, Lower Priority)

| # | Task | Effort | Output |
|---|---|---|---|
| **6** | **Modality-specific leaderboard** — filter SLAKE/VQA-RAD JSONL by CT/MRI/X-ray metadata | 0.5 days | Per-modality performance table for paper appendix |
| **7** | **Ensemble mechanism** — majority vote (closed) + score aggregation (open) on existing SLAKE JSONLs | 1 day | Answers if model combination outperforms best single model |

---

## Current CV Strength Assessment

Even without the remaining experiments, the project already contains:

- **Statistically validated** (bootstrap CI + permutation test) 5-model × 4-dataset benchmark — most student projects omit significance testing entirely
- **Named, quantified failure modes** — attention hijacking, visual blindness, pathology bias, verbosity penalty, catastrophic forgetting paradox — this is analysis depth, not just number-reporting
- **The catastrophic forgetting paradox** — F1 ~12% but Judge Acc ~70–82% for medical models on general data — a genuinely counterintuitive, novel finding
- **Private clinical DICOM data** with multi-slice pipeline — very few interns touch proprietary clinical imaging
- **A negative experimental result** (S-CoT degrades performance) with a mechanistic explanation — negative results that explain *why* are considered valid contributions

**The single addition that would most strengthen both the paper and CV:** the few-shot experiment, because it converts the project's central finding from descriptive ("medical training trumps parameter count") to prescriptive ("here is whether and how the gap can be closed without retraining"), which is what practitioners actually want to know.
