# Research Gap Analysis & Future Directions
## VLM Medical VQA Benchmark — Internship Extension Opportunities

---

## What You've Already Done (Stronger Than You Think)

Before discussing gaps, it's worth appreciating the scope of what's already here, because this is **genuinely solid internship-level research work**:

| Dimension | What You Did |
|---|---|
| **Scale** | 5 models × 4 datasets × 7,500+ samples evaluated |
| **Metric breadth** | Token F1, Normalized F1, BLEU, BERTScore, LLM-as-a-Judge — all implemented from scratch |
| **Engineering depth** | Prompt engineering (v1→v2 evolution), context-aware NLP extractor, quantized 7B inference, dataset streaming |
| **Analysis quality** | 10-part failure taxonomy with named failure modes (attention hijacking, visual blindness, pathology bias, etc.) |
| **Experimental work** | S-CoT intervention with negative result (which is a valid finding) |
| **Real clinical data** | DICOM pipeline on private MRI data — very few interns touch proprietary clinical imaging data |

**On the CV question:** Benchmarking _with this level of analysis depth_ is competitive. The distinction is between "I ran some models through a benchmark" vs. "I designed an evaluation framework, identified failure modes, and conducted controlled experiments." You're much closer to the latter.

---

## Identified Gaps (Honest Assessment)

### G1 — Incomplete Work Within Current Scope
These are things your own reports acknowledge as "next steps" but haven't been done yet.

- **DICOM LLM-as-a-Judge not run** — Your DICOM report explicitly lists this as Step 1. The verbosity bias inflating LLaVA-Med's apparent failure is likely rescued by the judge (same pattern seen in B4 of error analysis).
- **HuatuoGPT not evaluated on DICOM** — Listed as Step 2. The clinical leaderboard is incomplete.
- **No multi-slice evaluation** — You use single middle slices. Radiologists examine 20–34 slices. This is flagged as a known limitation.
- **Weak coverage of general models on medical datasets** — Only Gemma-3 and LLaVA-1.6 were tested on general datasets. No HuatuoGPT or LLaVA-Med on VQAv2/OK-VQA cross-domain tests.

### G2 — Metric and Evaluation Gaps
- **No calibration analysis** — You know *what* models predict but not *how confident* they are. Does MedGemma know when it doesn't know? Overconfident wrong predictions are the worst clinical failure mode.
- **LLM judge reliability not validated** — You computed Pearson correlation (0.61–0.79) but never checked inter-rater agreement between your Llama-3.1 judge and a stronger judge (e.g., GPT-4o on a small subset). If the judge is biased, all judge results are suspect.
- **No statistical significance testing** — All result differences (e.g., MedGemma 70.5% vs HuatuoGPT 47.86% on SLAKE F1) are presented without confidence intervals or p-values. For a rigorous paper, these are required.
- **BLEU is underused** — You compute it but rarely discuss it in error analysis beyond passing mentions.

### G3 — Experimental Depth Gaps
- **S-CoT was only tested on MedGemma/SLAKE** — The negative result is real, but it's from one model on one dataset. Does it hold for HuatuoGPT? For VQA-RAD? The finding could be either more general or model-specific.
- **Prompt sensitivity analysis missing** — You found a +14.55pp boost from v1→v2. What happens with other prompt variants? The current conclusion that "v2 is optimal" is based on a single comparison.
- **No few-shot experiments** — Everything is zero-shot. A simple 1-shot or 3-shot experiment would immediately tell you whether the performance gaps are learnable from examples or require actual fine-tuning.

### G4 — Clinical Validity Gaps
- **QA pairs were manually constructed (not radiologist-verified)** — For the DICOM evaluation, you made the QA pairs yourself from the radiologist report. This is a reasonable starting point, but the questions haven't been validated by a radiologist as being the "right" clinical questions to ask from a given MRI.
- **Only knee MRI** — The DICOM evaluation is a single anatomy (knee), single modality (MRI). Results don't generalize to chest X-ray, brain CT, abdominal imaging, etc.
- **No clinician-in-the-loop evaluation** — All "ground truth" is text-based. For the DICOM evaluation, a radiologist hasn't reviewed the model predictions and labeled them as clinically acceptable or dangerous.

---

## Possible Future Directions

These are ranked roughly from **lowest effort / most completable** to **highest impact / most publishable**.

---

### Direction 1 — Complete the DICOM Pipeline (Low Effort, High Completeness)
**What:** Run the remaining 3 pending steps from your own DICOM report:
1. LLM-as-a-Judge on DICOM results
2. HuatuoGPT evaluation on DICOM
3. Multi-slice evaluation (3–5 slices per question instead of 1)

**Why it matters for CV:** Turns the DICOM section from "started an interesting thing" to "completed a clinical evaluation on private MRI data." Multi-slice is novel enough to mention specifically.

**Effort:** Low. You already have the infrastructure. Multi-slice is a loop change in the DICOM notebook.

---

### Direction 2 — Few-Shot Prompting Experiment (Medium Effort, Strong Finding)
**What:** Take 1–3 clinically annotated examples per question type and run a few-shot evaluation on the weakest models (Gemma-3, LLaVA-1.6) on SLAKE and VQA-RAD.

**Hypothesis:** Either (a) few-shot closes the gap substantially → shows that domain adaptation is possible without fine-tuning, or (b) few-shot doesn't help → proves the gap is architectural, not just prompt-related. Either result is informative.

**Why it matters for CV:** Few-shot is a standard technique, but "does few-shot close the medical domain gap" is a specific, answerable research question with a clean result table. It extends the narrative from benchmarking to a controlled adaptation experiment.

**Effort:** Medium. Need to curate 3–5 high-quality few-shot examples per dataset and add an `examples` parameter to your inference loop.

---

### Direction 3 — Calibration Analysis (Medium Effort, Clinically Novel)
**What:** Extract the model's confidence (log-probability of the predicted answer token, or softmax probability on "Yes"/"No" tokens for closed questions) and compute calibration curves (reliability diagrams, ECE — Expected Calibration Error).

**Hypothesis:** Medical fine-tuning doesn't just improve accuracy — it should improve calibration. A model saying "Yes" with 90% confidence when it's right 90% of the time is well-calibrated. A model saying "Yes" with 90% confidence when it's right 50% of the time is dangerous.

**Why it matters for CV/research:** Calibration in medical AI is a known open problem. The paper would be: *"We show that MedGemma is not just more accurate but better calibrated than generalist VLMs, making it safer for clinical screening use."* That's a publishable claim.

**Effort:** Medium-High. Requires modifying inference scripts to extract token probabilities (via `scores` in HuggingFace `generate()` output). Not trivial but very doable.

---

### Direction 4 — LLM-as-a-Judge Reliability Study (Medium Effort, Methodologically Important)
**What:** Take a stratified sample of ~200 predictions (covering all 5 models, all question types) from your existing SLAKE results. Run them through three judges: your Llama-3.1-8B judge, a GPT-4o API call, and human annotation (you annotating them yourself). Compute inter-rater agreement (Cohen's κ or Spearman ρ).

**Why it matters:** Your entire judge framework rests on the assumption that Llama-3.1-8B is a valid judge. The Pearson correlation proves it tracks F1, but not that it tracks *clinical correctness*. A judge validation study would either confirm your methodology or reveal a systematic bias that needs correction.

**For CV:** The section title "LLM-as-a-Judge Reliability Validation in Medical VQA" is publishable standalone. There's active 2024–2025 literature on evaluating evaluators.

**Effort:** Medium. GPT-4o API calls on 200 samples are cheap (<$5). Manual annotation by you takes a few hours.

---

### Direction 5 — Systematic Prompt Sensitivity Analysis (Medium Effort, Good Paper Section)
**What:** Design a grid of 5–8 prompt variants (ranging from minimal "Answer concisely" to structured "Final Answer: X" to various few-shot and CoT variants) and evaluate all 5 models on a fixed 100-sample subset of SLAKE. Plot a heatmap of accuracy × prompt variant × model.

**Why it matters:** Your v1→v2 comparison showed a +14.55pp jump. But v2 was designed for MedGemma — is it optimal for all models? You show in the error analysis that LLaVA-Med ignores the "Final Answer:" constraint entirely. Maybe a completely different prompt structure is optimal for each model family. This is the "prompt engineering as scientific variable" angle.

**For CV:** "Prompt Sensitivity in Medical VLM Evaluation" is a clean methodology paper and directly relevant to anyone building production medical AI systems.

---

### Direction 6 — Automated QA Generation from Radiology Reports (High Impact, Scalable)
**What:** Use an LLM (e.g., GPT-4o or Llama-3-70B) to automatically extract QA pairs from the free-text radiologist reports in your DICOM dataset CSV. For each report: extract findings → generate closed (yes/no) and open-ended questions → generate ground truth answers from the report text.

**Why it matters:** Your DICOM report's biggest limitation is 16 QA pairs from 2 patients. With automated extraction, 2 patients could become 100+ QA pairs. If you had access to even 20–30 patients, you'd have 800+ pairs and could run proper statistical analysis.

**For CV:** "Automated clinical QA dataset construction from radiology reports" is a pipeline that other researchers would actually use. It turns your DICOM work from "a small sample study" into "a replicable methodology for building medical VQA datasets from proprietary PACS data."

**Effort:** Medium-High. Requires prompt engineering the LLM extractor, validation of generated QA pairs, and testing on your existing 2-patient ground truth.

---

### Direction 7 — Ensemble / Voting Mechanism (Medium Effort, Practical Contribution)
**What:** Given that you have 5 models' predictions on the same questions, explore whether an ensemble (majority vote on closed questions, confidence-weighted average on open) outperforms any single model. Your C2 failure overlap analysis already shows that ~22% of SLAKE is irreducible noise, but some questions are answered correctly by only one model. Can you recover those via ensemble?

**Hypothesis:** MedGemma's exclusive wins (230 questions) + HuatuoGPT's exclusive wins (47 questions) may be partially disjoint. If so, a clinical ensemble could significantly outperform either model alone.

**For CV:** "Model Ensemble for Medical VQA" is practical, grounded in your existing data, and requires no new inference runs — just post-hoc analysis of your JSONL files.

---

### Direction 8 — Modality-Specific Leaderboard (Low Effort, High Clarity)
**What:** SLAKE and VQA-RAD contain questions about CT, MRI, and X-ray images, but you currently aggregate them all together. Break your results by imaging modality. Does MedGemma's superiority hold equally for CT vs MRI vs X-ray? Does LLaVA-1.6's modality confusion (29.84% confusion rate) occur equally across all three, or is it primarily confused between CT and MRI (which look more similar) vs. X-ray (which is fundamentally different)?

**For CV:** Adds a clinically meaningful dimension to the existing analysis. X-ray AI (chest screening) vs. MRI AI (soft tissue) are practically different deployment contexts. Separating their performance gives radiologists a more actionable result.

**Effort:** Very Low. Filter your existing JSONL results by modality metadata that already exists in SLAKE/VQA-RAD.

---

## Summary Table: Directions at a Glance

| Direction | Effort | CV Impact | Research Impact | Requires New Inference? |
|---|---|---|---|---|
| 1. Complete DICOM pipeline | Low | Medium | Low | Yes (HuatuoGPT on DICOM) |
| 2. Few-shot experiment | Medium | High | High | Yes |
| 3. Calibration analysis | Medium-High | High | High | Partial (modify scripts) |
| 4. Judge reliability study | Medium | High | Medium | No (GPT-4o API + manual) |
| 5. Prompt sensitivity grid | Medium | Medium | Medium | Yes (subset only) |
| 6. Automated QA generation | Medium-High | High | Very High | No (post-processing) |
| 7. Ensemble mechanism | Low | Medium | Medium | No (post-hoc on JSONL) |
| 8. Modality-specific leaderboard | Very Low | Low-Medium | Medium | No (filter existing data) |

---

## What Would Look Best on a CV

If I had to rank which directions are most CV-impactful, in order:

1. **Calibration analysis** — "Model confidence calibration in medical VLMs" is a specific, publishable claim with clinical stakes.
2. **Few-shot experiment** — "Does few-shot prompting close the medical domain gap?" is a clean research question with a binary, memorable answer.
3. **Automated QA generation pipeline** — Turns your DICOM work into a replicable open-source tool, which is the most practical contribution.
4. **Judge reliability study** — Strengthens the methodological foundation of everything else you've done.

---

## What to Ask Your Mentor

When you discuss with your mentor, the key questions to resolve are:

1. **Is there any path to accessing more of the private DICOM dataset?** If yes, Direction 6 (automated QA generation) becomes the clear top priority and dramatically raises the clinical impact.
2. **Is the goal a paper or just strong internship output?** If paper: calibration + judge reliability. If CV: few-shot + DICOM completion are most demonstrable.
3. **Are there clinical collaborators?** A radiologist reviewing model predictions for 30 minutes would add a "clinician-validated" claim that very few papers have.
4. **What's the remaining timeline?** Some directions (ensemble, modality leaderboard) can be done in 1–2 days. Others (few-shot, calibration) need 1–2 weeks.
