# S-CoT Extension — Structured Chain-of-Thought on New Model/Dataset Pairs

**VLM Medical VQA Benchmark — Section 14 Extension**  
**Date:** July 2026

---

## 1. Motivation

The original S-CoT experiment (Section 14, `report.md`) showed that a structured 4-step
chain-of-thought prompt degraded MedGemma-4B performance on SLAKE by −5.0 pp F1
(p < 0.001). The hypothesis was **generative drag**: sub-10B VLMs have insufficient
attention capacity to maintain visual grounding while generating long structured outputs.

A critical open question remained: is this failure **architecture-agnostic** (a property
of all small VLMs) or **model/dataset-specific** (a quirk of MedGemma-4B on SLAKE)?

This extension tests two new model–dataset pairs to answer that question.

---

## 2. Experimental Setup

### 2.1 Models and Datasets

| Run | Model | Dataset | New? |
|---|---|---|---|
| Original | MedGemma-4B (`google/medgemma-4b-it`) | SLAKE (EN test) | — |
| **Run A** | HuatuoGPT-Vision-7B (`FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`) | SLAKE (EN test) | ✅ |
| **Run B** | MedGemma-4B (`google/medgemma-4b-it`) | VQA-RAD (test) | ✅ |

### 2.2 S-CoT Prompt (identical across all runs)

```
{question}

Please reason step by step using this exact structure:

Step 1 - Modality: Identify the imaging modality (e.g., CT, MRI, X-ray).
Step 2 - Anatomy: Name the primary organ or anatomical structure visible.
Step 3 - Observation: Write exactly one short sentence answering the core
         question based on Steps 1 and 2.
Step 4 - Conclusion: State your definitive answer (if possible, a single word)
         in the exact format 'Final Answer: X'.
```

Closed-ended questions additionally prepend: `Answer the question with yes or no.`

### 2.3 Baselines

Baselines are from the Section 8 evaluation (`outputs/inference/*_v2.jsonl`),
using the MedGemma paper prompt protocol (concise-answer + `Final Answer: X` format).

### 2.4 Statistical Test

Paired permutation test (10,000 iterations) on matched question indices.

---

## 3. Results

| Model | Dataset | N | Base F1 | SCoT F1 | ΔF1 | Base Closed | SCoT Closed | ΔClosed | Base Open | SCoT Open | ΔOpen | p(F1) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MedGemma-4B | SLAKE | 440 | 70.5% | 65.5% | **−5.0 pp** | 85.6% | 81.1% | −4.5 pp | 60.7% | 56.9% | −3.8 pp | **<0.001 ★★★** |
| HuatuoGPT-7B | SLAKE | 1061 | 47.9% | 47.1% | −0.8 pp | 72.4% | 68.0% | −4.3 pp | 32.1% | 33.4% | +1.3 pp | 0.570 ns |
| MedGemma-4B | VQA-RAD | 451 | 62.5% | 61.6% | −0.9 pp | 78.1% | 78.1% | +0.0 pp | 42.9% | 40.8% | −2.1 pp | 0.576 ns |

*Paired permutation test, 10,000 iterations. ★★★ p<0.001; ns p≥0.05.*

---

## 4. Conclusion

### Finding: Architecture-Specific and Dataset-Specific Failure

The S-CoT degradation is **not architecture-agnostic**. It is specific to
**MedGemma-4B on SLAKE** and does not generalise to other combinations:

**MedGemma-4B / SLAKE** (original finding confirmed):
- −5.0 pp overall F1 (p < 0.001), −4.5 pp closed, −3.8 pp open
- Significant degradation across all question types

**HuatuoGPT-7B / SLAKE** (Run A):
- −0.8 pp overall F1 (p = 0.570, not significant)
- Open F1 actually *improves* +1.3 pp
- The 7B model (Qwen2.5VL backbone, ~2× parameters) sustains structured generation without losing visual grounding

**MedGemma-4B / VQA-RAD** (Run B):
- −0.9 pp overall F1 (p = 0.576, not significant)
- Closed accuracy unchanged (78.1% → 78.1%)
- The same model that degrades on SLAKE is unaffected on VQA-RAD

### Scientific Interpretation

| Hypothesis | Evidence |
|---|---|
| **Generative drag** (capacity-limited model overwhelmed by structured output) | Supported for MedGemma-4B/SLAKE only |
| **Architecture-agnostic failure** (all sub-10B VLMs degrade) | **Refuted** — HuatuoGPT-7B shows no degradation on same SLAKE data |
| **Dataset-specific failure** (SLAKE visual complexity amplifies drag) | Partially supported — MedGemma-4B fails on SLAKE but not VQA-RAD |

**Most likely explanation:** Generative drag is real but interacts with both
*model capacity* (7B sustains structured output better than 4B) and *dataset
visual complexity* (SLAKE's diverse multi-organ CT/MRI requires more visual
working memory than VQA-RAD's more homogeneous chest X-rays).

---

## 5. Summary

The S-CoT intervention was extended to two additional model–dataset pairs: HuatuoGPT-Vision-7B-Qwen2.5VL on SLAKE and MedGemma-4B on VQA-RAD, using the identical four-step structured prompt and baselines from the main evaluation (Section 8). Results showed no statistically significant degradation in either new combination (HuatuoGPT-7B/SLAKE: ΔF1 = −0.8 pp, p = 0.570; MedGemma-4B/VQA-RAD: ΔF1 = −0.9 pp, p = 0.576), in contrast to the original MedGemma-4B/SLAKE finding (ΔF1 = −5.0 pp, p < 0.001). The S-CoT-induced performance degradation is therefore neither architecture-agnostic nor universally harmful. It appears to emerge from the interaction of limited model capacity (4B parameters) with dataset visual complexity (SLAKE’s heterogeneous multi-organ imaging), suggesting that the generative drag hypothesis applies within a specific regime rather than as a general property of structured prompting in medical VQA.

---

## 6. File References

| File | Description |
|---|---|
| `outputs/_archive/scot_experiment/google_medgemma-4b-it__slake_scot.jsonl` | MedGemma-4B SCoT / SLAKE (440 records) |
| `outputs/_archive/scot_experiment/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_scot.jsonl` | HuatuoGPT-7B SCoT / SLAKE (1061 records) |
| `outputs/_archive/scot_experiment/google_medgemma-4b-it__vqa_rad_scot.jsonl` | MedGemma-4B SCoT / VQA-RAD (451 records) |
| `scripts/scot_extension_analysis.py` | Analysis script (re-runnable) |
| `notebooks/09_scot_extension.ipynb` | Kaggle inference notebook |
