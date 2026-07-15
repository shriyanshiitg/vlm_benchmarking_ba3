# Progress Report — Post July 2nd, 2026
## VLM Medical VQA Benchmark

**Prepared for:** Next Meeting  
**Period Covered:** 2 July 2026 – 16 July 2026  
**Author:** Shriyansh Raj

---

## Executive Summary

This two-week period covered four major workstreams:

1. **LLM-as-a-Judge general domain validation** — confirmed models retain general visual reasoning despite medical fine-tuning
2. **Inter-rater agreement study** — formally validated the 8B judge's clinical reliability against a 70B reference model
3. **BLEU audit** — proved n-gram metrics structurally fail on medical synonymy and justified our metric hierarchy
4. **S-CoT extension experiment** — refuted the architecture-agnostic hypothesis; generative drag is conditionally real

---

## 1. LLM-as-a-Judge: General Domain Evaluation (VQAv2 + OK-VQA)

### Motivation
Token-matching metrics (F1, BLEU) penalise models for formatting differences unrelated to knowledge. The LLM judge evaluates semantic correctness instead, separating formatting artefacts from real capability.

### Results

| Model | VQAv2 Judge Acc | OK-VQA Judge Acc | Key Insight |
|---|---|---|---|
| **MedGemma-4B** | 70.50% | 58.10% | Retains general reasoning despite medical specialisation |
| **LLaVA-Med-7B** | 73.50% | 60.20% | General visual reasoning preserved after medical fine-tuning |
| **HuatuoGPT-7B** | **82.10%** | **65.90%** | Best general results — blended training prevents catastrophic forgetting |

### Interpretation

HuatuoGPT-7B's blended training (medical + general data) achieves highest general-domain performance, confirming it avoids the catastrophic formatting corruption seen in purely medically fine-tuned models. LLaVA-Med's relatively strong numbers suggest its fine-tuning preserved cross-domain capabilities better than its medical-domain scores alone suggest.

---

## 2. LLM-as-a-Judge Inter-Rater Agreement Study

### Motivation
To be paper-publishable, our 8B judge must be validated against an independent, stronger reference. We tested agreement between:
- **Judge A:** `Llama-3.1-8B-Instruct` (4-bit NF4, Kaggle T4) — used throughout the benchmark
- **Judge B:** `Llama-3.3-70B-Versatile` (Groq LPU, full precision, 8.75× larger)

### Sample Design
499 records, stratified across four strata:
- 100 score-1 anchors (clear failures)
- 99 score-5 anchors (clear successes)
- 150 score-3 ambiguous (maximum disagreement zone)
- 150 open-ended medical records (clinical knowledge critical)

### Agreement Metrics

| Metric | Value | Benchmark | Status |
|---|---|---|---|
| Cohen's κ (linear-weighted) | **0.696** | ≥ 0.60 = Substantial | ✅ |
| Pearson r | **0.810** (p<0.0001) | ≥ 0.80 = Very Strong | ✅ |
| Spearman ρ | **0.818** (p<0.0001) | ≥ 0.80 = Very Strong | ✅ |
| Mean Absolute Difference | **0.567** score points | < 0.75 = Good | ✅ |
| Exact agreement | **66.3%** | ≥ 65% = Good | ✅ |
| Adjacent (±1) agreement | **80.8%** | ≥ 85% = Good | ⚠️ |
| Systematic bias (8B − 70B) | **+0.194** | No significant bias | ✅ |

### Per-Model Agreement

| Model | Cohen κ | Pearson r | MAD |
|---|---|---|---|
| Gemma-3-4B | 0.769 | 0.856 | 0.427 |
| MedGemma-4B | 0.715 | 0.809 | 0.505 |
| HuatuoGPT-7B | 0.724 | 0.841 | 0.469 |
| LLaVA-1.6-7B | 0.718 | 0.809 | 0.547 |
| LLaVA-Med-7B | 0.535 | 0.749 | 0.841 |

> ⚠️ **LLaVA-Med-7B has the weakest agreement (κ = 0.535, MAD = 0.841).** Its conversational response style makes the 8B judge's scoring less reliable for this specific model. Results for LLaVA-Med should be interpreted with this caveat.

### Conclusion
**The 8B judge is formally validated.** κ = 0.696 (Substantial) and r = 0.810 (Very Strong) meet the threshold for paper-quality reporting. The 8B judge does not systematically inflate or deflate scores.

---

## 3. BLEU-4 Error Analysis — N-Gram Metric Audit

### Motivation
BLEU is computed across all model–dataset combinations but was never examined to determine whether it provides signal *distinct* from Token F1 and LLM Judge Accuracy.

### Dataset
11,766 open-ended predictions (SLAKE, VQA-RAD, VQAv2, OK-VQA combined).

### Triple-Axis Correlation

| Metric Pair | r (all open-ended) | Interpretation |
|---|---|---|
| BLEU ↔ Token F1 | **0.937** | Near-redundant — BLEU adds almost no new information |
| BLEU ↔ Judge Accuracy | 0.595 | Weak proxy for semantic correctness |
| F1 ↔ Judge Accuracy | 0.633 | F1 tracks semantics better than BLEU |

### The Rescue Zone
**2,562 predictions** have BLEU < 0.10 AND Token F1 < 0.15, yet the LLM judge scores them ≥ 4/5 (correct). These are **true positives falsely penalised by both classical metrics**:

| Domain | Rescue Zone Records |
|---|---|
| Medical (SLAKE + VQA-RAD) | 692 |
| General (VQAv2 + OK-VQA) | 1,870 |

### Why Classical Metrics Fail — Root Cause Autopsy

From 782 high-judge/low-BLEU medical records:

| Failure Mode | N | Example |
|---|---|---|
| **Medical synonymy** | 379 | GT: "kidney" → Pred: "renal" — clinically identical, zero n-gram overlap |
| **Conversational filler** | 49 | GT: "stomach" → Pred: "The three circular opacities are located in the **stomach**." |
| **Granularity mismatch** | 13 | GT: "right" → Pred: "right side of the diaphragm is elevated" |

### Decision
LLM Judge Accuracy is confirmed as the **primary metric** for open-ended evaluation. BLEU and F1 are retained as secondary metrics for reproducibility and comparison with prior work only.

---

## 4. Repository Maintenance & Documentation (End of Month 1)

- Cleaned all macOS junk files, Jupyter checkpoints, backup files (~4 MB removed)
- Reorganised `docs/` and `results/` folder structure
- Removed `research_gap_analysis.md` from GitHub history (privacy) and added to `.gitignore`
- Finalised and committed all comprehensive benchmark reports
- Pushed clean repository to GitHub

---

## 5. S-CoT Extension Experiment

### Background
Section 14 of the main report showed that a structured 4-step Chain-of-Thought prompt (S-CoT) degraded MedGemma-4B performance on SLAKE by **−5.0 pp F1** (p < 0.001). The hypothesis was *generative drag* — the model's limited attention capacity is overwhelmed by the structured output format, diluting visual grounding.

**Open question:** Is this an architecture-agnostic property of sub-10B VLMs, or is it specific to MedGemma-4B on SLAKE?

### Extension Design
The **identical S-CoT prompt** was applied to two new model–dataset pairs:
- **Run A:** HuatuoGPT-Vision-7B (Qwen2.5VL backbone) on SLAKE
- **Run B:** MedGemma-4B on VQA-RAD

Baselines from the Section 8 evaluation were used. Statistical comparison via paired permutation test (10,000 iterations).

### Results

| Model | Dataset | N | Base F1 | SCoT F1 | **ΔF1** | Base Closed | SCoT Closed | ΔClosed | p (F1) |
|---|---|---|---|---|---|---|---|---|---|
| MedGemma-4B | SLAKE | 440 | 70.5% | 65.5% | **−5.0 pp** | 85.6% | 81.1% | −4.5 pp | **<0.001 ★★★** |
| HuatuoGPT-7B | SLAKE | 1,061 | 47.9% | 47.1% | −0.8 pp | 72.4% | 68.0% | −4.3 pp | 0.570 ns |
| MedGemma-4B | VQA-RAD | 451 | 62.5% | 61.6% | −0.9 pp | 78.1% | 78.1% | +0.0 pp | 0.576 ns |

*★★★ p<0.001; ns = not significant (p≥0.05). Paired permutation test.*

### Conclusion

> **The architecture-agnostic hypothesis is refuted.**

The S-CoT degradation is **specific to MedGemma-4B on SLAKE** and does not generalise to other model–dataset combinations:

- HuatuoGPT-7B (7B params, Qwen2.5VL backbone) shows **no significant degradation** on the same SLAKE dataset (−0.8 pp, p = 0.570)
- MedGemma-4B itself shows **no significant degradation** on VQA-RAD (−0.9 pp, p = 0.576)

**Revised interpretation:** Generative drag is conditionally real. It emerges from the interaction of *limited model capacity* (4B parameters) with *dataset visual complexity* (SLAKE's heterogeneous multi-organ CT/MRI imaging). The 7B model can sustain structured generation without losing visual grounding; and MedGemma-4B copes fine with VQA-RAD's less visually complex chest X-rays.

**Paper-ready paragraph:**
> The S-CoT intervention was extended to HuatuoGPT-Vision-7B on SLAKE and MedGemma-4B on VQA-RAD. Results showed no statistically significant degradation in either new combination (HuatuoGPT-7B/SLAKE: ΔF1 = −0.8 pp, p = 0.570; MedGemma-4B/VQA-RAD: ΔF1 = −0.9 pp, p = 0.576), in contrast to the original MedGemma-4B/SLAKE finding (ΔF1 = −5.0 pp, p < 0.001). These results indicate that S-CoT-induced performance degradation is neither architecture-agnostic nor universally harmful. Instead, it appears to emerge from the interaction of limited model capacity with dataset visual complexity, suggesting that the generative drag hypothesis applies within a specific regime rather than as a general property of structured prompting.

---

## 6. Open Items & Next Steps

| Priority | Item | Status |
|---|---|---|
| 🔴 High | Model calibration analysis (ECE/reliability diagrams for closed-ended questions) | Not started |
| 🔴 High | Few-shot experiment (0/1/3-shot on 200-sample SLAKE subset) | Not started |
| 🟡 Medium | Clinical verification of DICOM series-to-question mappings | Not started |

---

## Appendix — Files Produced This Period

| File | Description |
|---|---|
| `docs/inter_rater_agreement_report.md` | Full inter-rater study with reference scales |
| `docs/bleu_error_analysis.md` | BLEU audit, rescue zone, qualitative autopsy |
| `docs/report_scot_extension.md` | S-CoT extension full results |
| `docs/report.md` (Section 14.4) | Main report updated with extension conclusion |
| `scripts/scot_extension_analysis.py` | Re-runnable analysis script |
| `notebooks/09_scot_extension.ipynb` | Kaggle inference notebook for Runs A & B |
| `outputs/_archive/scot_experiment/` | Raw JSONL outputs for all 3 S-CoT runs |
