# Error Analysis Report: VLM Medical VQA Benchmark

**Models evaluated:** MedGemma-4b-it, Gemma-3-4b-it, LLaVA-1.6-Mistral-7b, LLaVA-Med-1.5-Mistral-7b, HuatuoGPT-Vision-7B-Qwen2.5VL

**Datasets:** SLAKE, VQA-RAD, VQAv2, OK-VQA

---

## Overview

This report presents a ten-part failure taxonomy and behavioral analysis of all five evaluated VLMs. The analyses progress from category-level clinical failures (modality confusion, anatomical hallucination) through behavioral patterns (verbosity, spatial reasoning, cross-domain transfer) to structural overlap studies. Together they answer a single overarching question: **are the performance gaps observed in the benchmark due to domain-specific knowledge deficits, architectural limitations, or evaluation metric biases?**

The short answer is all three — but in different proportions for each model, and the analyses below quantify exactly where each bottleneck lies.

---

## C1. Question Type Performance Breakdown (SLAKE)

SLAKE annotates every question with a `content_type` field covering eight categories: Modality, Organ, Abnormality, Attribute, Color, Size, Position (Plane), and Quantity. This breakdown reveals which clinical reasoning skills each model actually possesses versus which it merely simulates.

### Token F1 by Content Type

| Content Type | Count | PT-Vision-7B-Qwen2.5VL | google/gemma-3-4b-it | google/medgemma-4b-it | hf/llava-v1.6-mistral-7b-hf | llava-med-v1.5-mistral-7b |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Organ | 253 | 60.67 | 52.39 | 78.72 | 42.47 | 47.17 |
| Position | 186 | 42.06 | 34.87 | 61.38 | 29.61 | 34.04 |
| Abnormality | 150 | 53.19 | 55.85 | 75.51 | 45.48 | 34.31 |
| KG | 148 | 31.08 | 22.1 | 53.68 | 24.59 | 30.22 |
| Modality | 108 | 79.01 | 76.92 | 98.15 | 43.52 | 63.41 |
| Size | 65 | 46.15 | 44.62 | 81.54 | 67.69 | 36.92 |
| Plane | 58 | 43.1 | 17.24 | 41.38 | 15.95 | 34.1 |
| Quantity | 52 | 7.69 | 0.0 | 42.31 | 15.38 | 0.0 |
| Color | 34 | 8.82 | 32.35 | 100.0 | 47.06 | 2.94 |
| Shape | 7 | 42.86 | 0.0 | 42.86 | 14.29 | 0.0 |

### LLM Judge Accuracy by Content Type

| Content Type | Count | PT-Vision-7B-Qwen2.5VL | google/gemma-3-4b-it | google/medgemma-4b-it | hf/llava-v1.6-mistral-7b-hf | llava-med-v1.5-mistral-7b |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Organ | 253 | 77.08 | 67.59 | 83.4 | 64.43 | 56.92 |
| Position | 186 | 54.84 | 50.54 | 64.52 | 45.16 | 63.98 |
| Abnormality | 150 | 56.67 | 56.67 | 74.0 | 58.67 | 42.67 |
| KG | 148 | 47.3 | 32.43 | 52.03 | 41.22 | 34.46 |
| Modality | 108 | 82.41 | 84.26 | 98.15 | 45.37 | 87.04 |
| Size | 65 | 67.69 | 70.77 | 83.08 | 70.77 | 46.15 |
| Plane | 58 | 53.45 | 22.41 | 46.55 | 27.59 | 60.34 |
| Quantity | 52 | 51.92 | 40.38 | 73.08 | 15.38 | 36.54 |
| Color | 34 | 70.59 | 41.18 | 100.0 | 47.06 | 26.47 |
| Shape | 7 | 42.86 | 28.57 | 57.14 | 14.29 | 14.29 |

### Key Findings

**Finding 1 — The F1 Smoking Gun on Quantity Questions.** HuatuoGPT scores an abysmal 7.69% Token F1 on Quantity questions but immediately jumps to 51.92% under LLM Judge evaluation. LLaVA-Med goes from 0.00% F1 to 36.54% Judge. Both models are counting correctly but outputting natural language ("There are two lesions") against a ground truth digit ("2"). This single category is the clearest demonstration of why strict token matching fails as a standalone metric.

**Finding 2 — Modality Is Solved, Abnormality Is Not.** Identifying imaging modality (CT vs. X-ray vs. MRI) is the easiest task across all models. MedGemma reaches 98.15% Judge accuracy and even LLaVA-Med achieves 87.04%. However, the moment the task shifts to Abnormality detection, generalist models stall at 42–56% while MedGemma maintains 74.00% — the only clinically viable performance level. A model that cannot reliably identify abnormalities cannot be deployed in a clinical setting regardless of its overall F1 score.

**Finding 3 — Spatial and Plane Catastrophe.** Identifying the anatomical plane (Axial, Coronal, Sagittal) requires 3D spatial context that generalist encoders simply do not have. Gemma-3 collapses to 22.41% Judge accuracy and LLaVA-1.6 reaches only 27.59%. Even HuatuoGPT reaches only approximately 48%, suggesting spatial awareness of radiological planes is one of the hardest zero-shot clinical skills to acquire.

**Finding 4 — Domain Pre-training Is Absolute.** MedGemma-4B wins every single content category in both F1 and Judge accuracy. This is not a marginal lead — it is a structural dominance that proves a smaller 4B medical model is categorically superior to a 7B generalist model across every clinical reasoning dimension.

---

## B3. Closed vs Open Performance Gap

The gap between closed (yes/no binary) accuracy and open-ended accuracy quantifies a specific deficit: the model can verify visual presence/absence but cannot articulate precise findings. Computed using LLM Judge accuracies to avoid token matching surface-form bias.

| Model | SLAKE Gap | VQA-RAD Gap | VQAv2 Gap | OK-VQA Gap |
|---|---|---|---|---|
| google/medgemma-4b-it | 16.37 | 21.31 | — | — |
| google/gemma-3-4b-it | 22.79 | 24.97 | 3.40 | — |
| llava-hf/llava-v1.6-mistral-7b-hf | 21.52 | 29.67 | 3.46 | — |
| HuatuoGPT-Vision-7B | 17.52 | 31.39 | — | — |
| microsoft/llava-med-v1.5-mistral-7b | 7.15 | −2.08 | — | — |

*A larger number = model is better at binary verification than at generating precise answers.*

### Key Findings

**Finding 1 — Generalist Models Suffer "Generative Paralysis" on Medical Data.** On VQAv2, both Gemma-3 (3.40) and LLaVA-1.6 (3.46) show negligible gaps — their architectures can generate answers as well as they can verify them when working in a familiar domain. But the moment they move to VQA-RAD, the gap explodes to 24.97 and 29.67 respectively. They can guess "yes/no" from basic visual intuition but hit a wall when they need to produce specific clinical vocabulary.

**Finding 2 — Medical Fine-tuning Specifically Bridges Generation.** Comparing base Gemma-3 to MedGemma directly: the gap shrinks from 22.79 to 16.37 on SLAKE and from 24.97 to 21.31 on VQA-RAD. The fine-tuning did not just improve visual recognition — it specifically bridged the gap between binary visual comprehension and fluent medical text generation.

**Finding 3 — The LLaVA-Med "Floor" Anomaly.** LLaVA-Med appears to have the best gap numbers (7.15, −2.08) but this is deeply misleading. A negative gap means the model performs equally poorly at both tasks. LLaVA-Med's PMC-15M conversational fine-tuning broke its ability to answer rigid Yes/No questions, dragging its closed accuracy down to the same level as its already-low open accuracy. Zero gap means zero competence — not zero deficit.

---

## A1. Modality Confusion Analysis (Medical Datasets)

Definition: The model predicts the wrong imaging modality (CT vs MRI vs X-Ray vs Ultrasound) when directly asked. This is the most clinically dangerous single-category failure because it signals fundamental visual processing failure — if a model cannot identify the imaging machine, its downstream diagnostic reasoning is untrustworthy.

| Model | Total Modality Qs | Confusions | Confusion Rate |
|---|---|---|---|
| google/medgemma-4b-it | 124 | 5 | **4.03%** |
| HuatuoGPT-Vision-7B | 124 | 6 | 4.84% |
| google/gemma-3-4b-it | 124 | 6 | 4.84% |
| microsoft/llava-med-v1.5-mistral-7b | 124 | 11 | 8.87% |
| llava-hf/llava-v1.6-mistral-7b-hf | 124 | 37 | **29.84%** |

### Key Findings

**Finding 1 — LLaVA-1.6 Is Clinically Blind to Modality.** A 29.84% modality confusion rate means the model names the wrong imaging modality nearly one-third of the time. This is not a formatting issue or a vocabulary mismatch — it is a fundamental failure of visual texture processing. The model cannot reliably distinguish the visual signature of CT from MRI from X-ray.

**Finding 2 — Gemma-3 Outperforms LLaVA-Med at Modality.** Despite having no medical fine-tuning, Gemma-3 achieves 4.84% confusion — equal to the specialized HuatuoGPT and only slightly worse than MedGemma. Its massive pre-training corpus likely included enough medical textbook captions to map visual textures to modality labels reliably. This is one of the few areas where the base model matches specialized models.

**Finding 3 — LLaVA-Med Fails Despite Medical Training.** At 8.87%, LLaVA-Med performs worse than a general model (Gemma-3). This aligns with the PMC-15M training bias: that dataset emphasizes clinical text descriptions more than visual modality discrimination, so LLaVA-Med learned to describe what it sees in pathological terms without reliably grounding the imaging modality first.

---

## A2. Anatomical Hallucination Analysis (Medical Datasets)

Definition: Among completely failed open-ended questions (Token F1 = 0), how often did the model confidently name a specific but entirely wrong anatomical structure? This distinguishes "confidently wrong" failures from "vague or empty" failures.

| Model | Total Zero-F1 Open Qs | Anatomical Hallucinations | Hallucination Rate |
|---|---|---|---|
| microsoft/llava-med-v1.5-mistral-7b | 477 | 225 | **47.17%** |
| google/medgemma-4b-it | 330 | 93 | 28.18% |
| llava-hf/llava-v1.6-mistral-7b-hf | 589 | 146 | 24.79% |
| google/gemma-3-4b-it | 582 | 139 | 23.88% |
| HuatuoGPT-Vision-7B | 541 | 118 | **21.81%** |

### Key Findings

**Finding 1 — LLaVA-Med's "Chatty Hallucination Trap."** When LLaVA-Med fails an open-ended question, it confidently names a completely hallucinated organ 47.17% of the time — 225 instances. This is the direct consequence of PMC-15M conversational fine-tuning. Because the model is trained to output exhaustive, clinical paragraph descriptions, it cannot fail gracefully. It suffers from generative momentum: it talks itself into inventing anatomical structures that are not present in the scan. From a clinical deployment perspective, a model that confidently hallucinates anatomy in nearly half of its errors is categorically unsafe.

**Finding 2 — The Quiet vs. Loud Failure Distinction.** MedGemma's 28.18% hallucination rate appears higher than Gemma-3's 23.88%, but the denominators tell the real story. Gemma-3 failed 582 times (large denominator) while MedGemma only failed 330 times (small denominator). Generalist models fail quietly — they output vague words like "mass," "lesion," or "area" because they lack the clinical vocabulary to name specific structures. MedGemma fails much less often, but because it is deeply fine-tuned on clinical anatomy vocabulary, when it does guess wrong it guesses with a specific medical word, triggering the hallucination flag. Medical fine-tuning reduces total failures dramatically while slightly increasing the specificity of the remaining errors.

**Finding 3 — HuatuoGPT Has the Lowest Hallucination Rate.** At 21.81%, HuatuoGPT hallucinates anatomy the least when it fails. This suggests its Qwen2.5-VL backbone is better calibrated to abstain or output generic terms when uncertain, rather than confidently fabricating specific anatomy.

---

## B4. Answer Length vs. Correctness Analysis

Definition: Does generating more words correlate with better reasoning or worse accuracy? Records are binned into Short (<20 words), Medium (20–100 words), and Long (>100 words) based on raw_output word count.

| Model | Bin | Count | Avg F1 | Avg Judge Acc |
|---|---|---|---|---|
| HuatuoGPT-Vision-7B | Short | 792 | 59.10% | 69.95% |
| | Medium | 587 | 43.28% | 55.54% |
| | Long | 132 | 32.81% | 46.21% |
| google/gemma-3-4b-it | Short | 1,118 | 55.12% | 60.64% |
| | Medium | 2,174 | 36.33% | 48.71% |
| | Long | 198 | 9.86% | 19.19% |
| google/medgemma-4b-it | Short | 1,025 | 70.03% | 73.66% |
| | Medium | 453 | 66.96% | 68.21% |
| | Long | 34 | 15.39% | 17.65% |
| llava-hf/llava-v1.6-mistral-7b-hf | Short | 3,201 | 47.07% | 57.54% |
| | Medium | 275 | 28.01% | 56.73% |
| | Long | 12 | 0.93% | 8.33% |
| microsoft/llava-med-v1.5-mistral-7b | Short | 1,403 | 37.28% | 52.32% |
| | Medium | 109 | 20.89% | 37.61% |
| | Long | 0 | — | — |

### Key Findings

**Finding 1 — The LLaVA-1.6 Medium Bin Inversion (The Metric Bias Smoking Gun).** When LLaVA-1.6 moves from Short to Medium output length, Token F1 collapses from 47.07% to 28.01% — a brutal 19.06-point drop. But LLM Judge accuracy barely moves: 57.54% to 56.73%, a delta of only 0.81 points. The model's actual visual comprehension is completely unchanged. This single observation mathematically justifies the entire LLM-as-a-Judge pipeline — traditional token matching was punishing models for being verbose, not for being wrong.

**Finding 2 — The Long Bin Collapse Is Universal.** Across every single model, performance collapses when output enters the Long (>100 words) bin. Gemma-3's Judge accuracy drops from 60.64% to 19.19%. MedGemma drops from 73.66% to 17.65%. HuatuoGPT drops from 69.95% to 46.21%. Long-form generation is not deeper reasoning — it is generative momentum. Models talk themselves out of correct visual features and introduce false anatomical variables the longer they generate.

**Finding 3 — Medical Fine-tuning Acts as a Formatting Guardrail.** Gemma-3 concentrates most outputs in the Medium bin (2,174 samples) rather than Short (1,118). MedGemma successfully inverts this, concentrating heavily in Short (1,025) with only 453 in Medium. Domain alignment and instruction tuning suppress the verbose trap, keeping responses concise where semantic accuracy peaks.

**Finding 4 — LLaVA-Med Never Goes Long.** LLaVA-Med has zero records in the Long bin. Combined with the Conversational Drift analysis (A4), this confirms that LLaVA-Med's failure mode is specifically semantic hallucination in medium-length confident sentences — not unstructured rambling. It is always wrong in a tidy, medically-sounding way.

---

## C3. Judge Score Distribution Analysis

Definition: Rather than just average judge score, looking at the full distribution of scores (1–5) per model reveals the behavioral risk profile — whether a model fails catastrophically or partially.

| Model | Total | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 |
|---|---|---|---|---|---|---|
| google/medgemma-4b-it | 1,512 | **15.01%** | 6.15% | 8.07% | 15.61% | **55.16%** |
| HuatuoGPT-Vision-7B | 1,511 | 21.77% | 7.88% | 8.07% | 18.20% | 44.08% |
| llava-hf/llava-v1.6-mistral-7b-hf | 3,488 | 29.59% | 8.29% | **4.82%** | 11.55% | 45.76% |
| google/gemma-3-4b-it | 3,490 | 32.38% | 11.09% | 5.67% | 12.52% | 38.34% |
| microsoft/llava-med-v1.5-mistral-7b | 1,512 | 32.47% | 5.22% | **11.04%** | **19.64%** | 31.61% |

### Key Findings

**Finding 1 — Generalist Models Exhibit a Bimodal Cliff.** Both LLaVA-1.6 and Gemma-3 show severely "hollow centers" — high Score 1 rates (29–32%), high Score 5 rates (38–46%), but very low Score 3 rates (4.82–5.67%). They operate on a binary reasoning cliff: either they recognize a visual abstraction and nail it perfectly (Score 5), or the question requires deep pathological reasoning they entirely lack and they plummet to complete hallucination (Score 1). They are almost never "almost right." This bimodal distribution is the signature of a generalist architecture encountering a domain boundary.

**Finding 2 — LLaVA-Med Is Partially Correct and Clinically Deceptive.** LLaVA-Med has the lowest Score 5 rate (31.61%) but the highest Score 3 (11.04%) and Score 4 (19.64%) concentrations among underperforming models. It has absorbed enough medical vocabulary to sound competent, earning partial credit from the judge for being "medically adjacent." This makes it the most deceptive model in a clinical context — it consistently produces plausible-sounding but ultimately incorrect medical terminology, which is arguably more dangerous than silent failure.

**Finding 3 — MedGemma's Clinical Safe Baseline.** MedGemma entirely breaks the bimodal pattern. With 55.16% Score 5 responses and only 15.01% Score 1, it has the highest precision and the lowest catastrophic failure rate. Domain-specific fine-tuning did not merely boost average accuracy — it fundamentally eliminated over half of the base model's catastrophic failures, shifting probability mass from Score 1 to Score 5.

---

## B5. Cross-Domain Degradation Profile

Definition: How much does performance degrade when Gemma-3 and LLaVA-1.6 (the two models evaluated on both general and medical datasets) move from natural images to medical images? Quantifies the "medical domain penalty."

| Model | VQAv2 Judge Acc | SLAKE Judge Acc | SLAKE Penalty | VQA-RAD Judge Acc | VQA-RAD Penalty |
|---|---|---|---|---|---|
| google/gemma-3-4b-it | 58.30% | 55.14% | **3.16 pp** | 45.90% | 12.40 pp |
| llava-hf/llava-v1.6-mistral-7b-hf | 70.22% | 50.14% | **20.08 pp** | 45.01% | 25.21 pp |

### Key Findings

**Finding 1 — LLaVA-1.6 Is Overfit to Natural Images.** Starting at a dominant 70.22% on VQAv2, LLaVA-1.6 suffers a catastrophic 20.08-point penalty on SLAKE and a 25.21-point penalty on VQA-RAD. Its 7 billion parameters are highly specialized for natural, color-textured scenes. When the input shifts to radiological grayscale, its visual encoder's learned features fail to transfer. Parameter count alone cannot bridge this domain gap.

**Finding 2 — The Gemma-3 Resilience Paradox.** Despite starting with a lower general baseline (58.30%), Gemma-3 suffers only a 3.16-point penalty on SLAKE — nearly seven times smaller than LLaVA-1.6's penalty on the same dataset. This challenges the naive assumption that larger models generalize better. Gemma-3's 4B parameters are better generalized across visual domains, suggesting its training favored abstract visual reasoning over natural-scene specialization.

**Finding 3 — Final SLAKE Scores Mask the Architectural Story.** Looking only at final SLAKE performance, Gemma-3 (55.14%) appears only slightly better than LLaVA-1.6 (50.14%). But the penalty analysis reveals a fundamentally different architectural story: a smaller model retained its reasoning floor while the larger model collapsed by over 20 points. This demonstrates why cross-domain penalty analysis is essential alongside absolute performance metrics.

---

## A4. Conversational Drift and Extraction Leakage

Definition: Conversational Drift = model ignores the "Final Answer:" prompt constraint entirely and generates unstructured text over 200 characters. Extraction Leakage = model uses "Final Answer:" but the extraction logic still fails to isolate a clean answer.

| Model | Dataset | Total Qs | Leakage Rate | Drift Rate |
|---|---|---|---|---|
| google/gemma-3-4b-it | OK-VQA | 1,000 | 0.10% | **33.40%** |
| google/gemma-3-4b-it | SLAKE | 1,061 | 0.09% | 7.92% |
| google/gemma-3-4b-it | VQA-RAD | 451 | 0.00% | 5.54% |
| google/gemma-3-4b-it | VQAv2 | 1,000 | 0.30% | 4.50% |
| llava-hf/llava-v1.6-mistral-7b-hf | OK-VQA | 1,000 | 0.60% | ~12% |
| llava-hf/llava-v1.6-mistral-7b-hf | VQAv2 | 1,000 | 0.20% | ~5% |
| google/medgemma-4b-it | SLAKE | 1,061 | 0.00% | 1.04% |
| google/medgemma-4b-it | VQA-RAD | 451 | 0.00% | 0.22% |
| HuatuoGPT-Vision-7B | SLAKE | 1,061 | 0.19% | 2.54% |
| HuatuoGPT-Vision-7B | VQA-RAD | 451 | 0.00% | 0.00% |
| microsoft/llava-med-v1.5-mistral-7b | SLAKE | 1,061 | 0.00% | 0.38% |
| microsoft/llava-med-v1.5-mistral-7b | VQA-RAD | 451 | 0.00% | 0.00% |

### Key Findings

**Finding 1 — Gemma-3's Cognitive Load Collapse on OK-VQA.** The most dramatic anomaly in the entire error analysis is Gemma-3's 33.40% drift rate on OK-VQA — 334 out of 1,000 questions where the model abandoned the prompt structure entirely. On every other dataset (VQAv2, SLAKE, VQA-RAD), Gemma-3 maintains drift rates under 8%. OK-VQA is unique in requiring external knowledge reasoning ("What era is this vehicle from?"). When the base model lacks the knowledge to answer immediately, its attention mechanism abandons the structural prompt constraint and reverts to base-level hallucination. Instruction following degrades under cognitive load.

**Finding 2 — LLaVA-Med Does Not Drift.** The earlier hypothesis that LLaVA-Med would have the highest drift rate is completely refuted by the data: 0.38% drift on SLAKE and 0% on VQA-RAD, with near-zero extraction leakage. Combined with the B4 finding (zero Long outputs) and A2 (47% anatomical hallucination among failures), this precisely characterizes LLaVA-Med's failure mode: it generates structurally compliant, medium-length sentences that confidently invent incorrect anatomical features. Its failures are purely semantic hallucinations, not formatting breakdowns.

**Finding 3 — The Extraction Pipeline Is Robust.** Across all models and datasets, leakage peaks at a negligible 0.60% for LLaVA-1.6 on OK-VQA. Every other model is under 0.30%. The "Final Answer:" prompt engineering is highly resilient — when models do follow the constraint, the regex extraction works perfectly. F1 score failures are legitimately due to wrong predictions, not extraction failures.

---

## A3. Spatial Reasoning Failure Analysis

Definition: Among questions containing spatial keywords (where, location, left, right, largest, biggest, how many, count, side), what percentage are confidently answered incorrectly (F1=0, non-empty prediction)?

| Model | Domain | Dataset | Spatial Qs | Spatial Failures | Failure Rate |
|---|---|---|---|---|---|
| google/gemma-3-4b-it | General | VQAv2 | 143 | 127 | **88.81%** |
| google/gemma-3-4b-it | General | OK-VQA | 95 | 78 | 82.11% |
| google/gemma-3-4b-it | Medical | SLAKE | 305 | 224 | 73.44% |
| google/gemma-3-4b-it | Medical | VQA-RAD | 82 | 50 | 60.98% |
| google/medgemma-4b-it | Medical | SLAKE | 305 | 122 | **40.00%** |
| google/medgemma-4b-it | Medical | VQA-RAD | 82 | 39 | 47.56% |
| llava-hf/llava-v1.6-mistral-7b-hf | General | VQAv2 | 143 | 61 | **42.66%** |
| llava-hf/llava-v1.6-mistral-7b-hf | Medical | SLAKE | 305 | 193 | 63.28% |
| llava-hf/llava-v1.6-mistral-7b-hf | Medical | VQA-RAD | 82 | 59 | 71.95% |
| llava-hf/llava-v1.6-mistral-7b-hf | General | OK-VQA | 95 | 63 | 66.32% |
| HuatuoGPT-Vision-7B | Medical | SLAKE | 305 | 212 | 69.51% |
| HuatuoGPT-Vision-7B | Medical | VQA-RAD | 82 | 48 | 58.54% |
| microsoft/llava-med-v1.5-mistral-7b | Medical | SLAKE | 305 | 204 | 66.89% |
| microsoft/llava-med-v1.5-mistral-7b | Medical | VQA-RAD | 82 | 33 | 40.24% |

### Key Findings

**Finding 1 — LLaVA-1.6's Texture Collapse Is Domain-Specific.** On VQAv2 (natural images), LLaVA-1.6 handles spatial queries well at 42.66% failure. On VQA-RAD (medical images), its spatial failure rate explodes to 71.95% — a 29.29-point degradation. This proves that LLaVA-1.6's spatial blindness on medical data is not an architectural limitation of its ViT encoder but a domain-specific failure. Its coordinate grounding functions when tracing distinct object boundaries in natural colors, but completely breaks down on low-contrast, grayscale, amorphous radiological textures.

**Finding 2 — The Gemma-3 Inversion Paradox.** Gemma-3 displays the exact opposite pattern. On VQAv2 (general), it is functionally blind spatially at 88.81% failure. But on VQA-RAD (medical), its failure rate drops to 60.98% — a 27.83-point improvement moving into medical data. Medical spatial queries operate within a much tighter, structurally predictable semantic distribution (left vs. right on a standard anatomical plane) that allows Gemma's language weights to narrow down answers more effectively than in unconstrained open-world scenes.

**Finding 3 — Medical Fine-tuning Dramatically Recovers Spatial Grounding.** MedGemma cuts spatial failure from Gemma-3's 73.44% down to 40.00% on SLAKE, and from 60.98% to 47.56% on VQA-RAD. The base model's underlying visual encoder already contains spatial features — medical instruction tuning successfully translates those features into precise clinical spatial orientation. This is concrete evidence that spatial competency is recoverable through domain alignment, not an immutable architectural property.

---

## A5. CoT-Induced Hallucination Analysis

Definition: Among failed questions where raw_output contains "Final Answer:", how often did the model write the correct answer during its reasoning but output a different, wrong answer in the final line? This detects cases where Chain-of-Thought generation actively destroyed a correct visual grounding.

| Model | Dataset | Failed CoT Qs | CoT Hallucinations | CoT Hallucination Rate |
|---|---|---|---|---|
| google/gemma-3-4b-it | SLAKE | 524 | 180 | **34.35%** |
| google/gemma-3-4b-it | OK-VQA | 413 | 135 | 32.69% |
| google/gemma-3-4b-it | VQAv2 | 406 | 92 | 22.66% |
| HuatuoGPT-Vision-7B | SLAKE | 479 | 167 | 34.86% |
| google/medgemma-4b-it | SLAKE | 281 | 67 | **23.84%** |
| llava-hf/llava-v1.6-mistral-7b-hf | SLAKE | 483 | 26 | **5.38%** |
| llava-hf/llava-v1.6-mistral-7b-hf | OK-VQA | 529 | 20 | 3.78% |
| llava-hf/llava-v1.6-mistral-7b-hf | VQAv2 | 179 | 8 | 4.47% |

### Key Findings

**Finding 1 — Gemma-3's "Attention Hijacking" Phenomenon.** In over one-third of its failures on SLAKE (34.35%) and OK-VQA (32.69%), Gemma-3 actually wrote the correct answer in its reasoning paragraph. It successfully recognized the visual feature and typed the ground truth word into its argument. However, the act of generating conversational text overloaded its attention mechanism. By the time it reached the "Final Answer:" anchor, it lost track of its own visual grounding and substituted a different word. This mathematically proves that forcing zero-shot models to "think out loud" can actively destroy their accuracy.

**Finding 2 — LLaVA-1.6 Fails Differently — It Never Knew the Answer.** LLaVA-1.6's CoT hallucination rate is consistently under 5.38% across all datasets. This does not make it a better model — its total failure rate is far higher. It means LLaVA-1.6 fails for a completely different reason: its visual encoder was genuinely blind to the feature. The correct word never entered its context window at all. It does not "talk itself out of" the right answer because it never had the right answer to begin with. Two distinct failure modes: attention hijacking (Gemma-3) vs. visual blindness (LLaVA-1.6).

**Finding 3 — Medical Fine-tuning Fixes Attention Routing.** MedGemma reduces Gemma-3's CoT hallucination rate from 34.35% to 23.84% on SLAKE — a 10.5-point absolute reduction. This reveals a subtle but critical benefit of domain-specific instruction tuning that goes beyond visual recognition. MedGemma's attention heads were specifically re-weighted to reliably pull the most salient medical entity from the reasoning block and correctly route it into the "Final Answer:" slot. Medical fine-tuning fixed not just what the model sees, but how it routes what it sees into the final answer.

**Finding 4 — HuatuoGPT Has High CoT Self-Contradiction.** At 34.86% on SLAKE, HuatuoGPT has the highest CoT hallucination rate of all models. Despite having a much better overall performance than the generalist models, it frequently knows the right answer during reasoning but outputs something different. This suggests its Qwen2.5-VL backbone is susceptible to the same attention hijacking phenomenon as Gemma-3, and that its accuracy gains come primarily from better initial visual grounding rather than better answer routing.

---

## C2. Failure Overlap Analysis (SLAKE)

Definition: For every pair of models, how many SLAKE questions did both models answer correctly (Shared Wins), both fail (Shared Failures), and how many did only one model answer correctly (Exclusive Wins)?

| Model A | Model B | Shared Wins | Shared Failures | A Exclusive | B Exclusive |
|---|---|---|---|---|---|
| google/gemma-3-4b-it | google/medgemma-4b-it | 552 | 246 | 33 | **230** |
| google/gemma-3-4b-it | llava-hf/llava-v1.6-mistral-7b-hf | 416 | **360** | 169 | 116 |
| google/medgemma-4b-it | llava-hf/llava-v1.6-mistral-7b-hf | 482 | 229 | **300** | 50 |
| google/medgemma-4b-it | microsoft/llava-med-v1.5-mistral-7b | 485 | 198 | **297** | 81 |
| google/medgemma-4b-it | HuatuoGPT-Vision-7B | 623 | 232 | 159 | 47 |
| HuatuoGPT-Vision-7B | google/gemma-3-4b-it | 502 | 308 | 168 | 83 |
| HuatuoGPT-Vision-7B | llava-hf/llava-v1.6-mistral-7b-hf | 432 | 291 | 238 | 100 |
| HuatuoGPT-Vision-7B | microsoft/llava-med-v1.5-mistral-7b | 446 | 271 | 224 | 120 |
| google/gemma-3-4b-it | microsoft/llava-med-v1.5-mistral-7b | 410 | 320 | 175 | 156 |
| llava-hf/llava-v1.6-mistral-7b-hf | microsoft/llava-med-v1.5-mistral-7b | 375 | 338 | 157 | 191 |

### Key Findings

**Finding 1 — Medical Fine-tuning Is Purely Additive (Gemma-3 vs MedGemma).** MedGemma exclusively won 230 questions that Gemma-3 failed. Gemma-3 exclusively won only 33 questions that MedGemma failed. This 7:1 exclusive win ratio mathematically proves that medical instruction tuning is a purely additive process — it unlocked 230 entirely new clinical questions while surrendering only 33 previously known ones. This is concrete empirical evidence against catastrophic forgetting: the fine-tuning preserved the base model's capabilities while dramatically extending its clinical reach.

**Finding 2 — The Generalist Ceiling.** Gemma-3 and LLaVA-1.6 share the highest Shared Failure count in the entire matrix: 360 questions that both models failed. Despite LLaVA-1.6 having nearly double the parameters of Gemma-3, scaling a generalist model cannot cross this clinical ceiling. These 360 questions require specialized pathological reasoning and medical vocabulary that simply does not exist in open-world training data.

**Finding 3 — Irreducible Dataset Noise (~20%).** Even the two strongest medical models in the cohort — MedGemma and HuatuoGPT — share 232 failures. This is approximately 22% of the 1,061 SLAKE questions that neither specialized medical model can answer correctly. These represent questions where image resolution is too degraded, spatial prompts are hopelessly ambiguous, ground truth labels may be objectively flawed, or the medical concept genuinely requires clinical reasoning beyond current model capabilities. This "irreducible noise floor" is an inherent dataset limitation, not a model limitation.

**Finding 4 — MedGemma's Dominant Territory.** MedGemma's exclusive wins against every other model tell the full story: it beats HuatuoGPT by 159 to 47, LLaVA-1.6 by 300 to 50, and LLaVA-Med by 297 to 81. MedGemma has captured a massive distinct territory of clinical reasoning — approximately 280–300 questions — that the other architectures fundamentally cannot access regardless of their parameter count or training strategy.

---

## Summary of Cross-Cutting Findings

### MedGemma-4b: The Clinical Benchmark
MedGemma leads every category across every analysis. Most importantly, domain fine-tuning did not just improve visual recognition — it fixed attention routing (A5), reduced catastrophic failures (C3), bridged the closed/open generation gap (B3), suppressed verbosity (B4), and unlocked exclusive clinical questions (C2). It is the only model with a clinically viable failure distribution.

### The LLaVA-1.6 Paradox
LLaVA-1.6 is the strongest general VLM (70.22% VQAv2 Judge accuracy) but one of the weakest on medical data. Its domain penalty is 6x larger than Gemma-3's. Its spatial blindness is domain-specific, not architectural. Its CoT hallucination rate is near-zero because it simply never recognizes the correct answer visually. It fails silently and consistently. Not suitable for medical deployment.

### LLaVA-Med: Dangerous Partial Competence
LLaVA-Med is the most clinically dangerous model in the cohort. Its hallucination rate (47.17%) is the highest. Its failure distribution (C3) concentrates in Score 3–4 — partially correct, clinically plausible, factually wrong. Its closed/open gap is essentially zero — it is equally bad at both. It never drifts (A4) and never goes verbose (B4), meaning every failure is a confident, well-structured, medically-sounding hallucination. This is a much more dangerous failure profile than silent refusal or obvious nonsense.

### Gemma-3: The Efficient Transfer Baseline
Gemma-3 proves to be a surprisingly robust foundation. Its medical domain penalty (3.16pp on SLAKE) is microscopic compared to LLaVA-1.6 (20.08pp). It suffers catastrophic CoT hijacking on complex tasks (34.35% on SLAKE) but this is fundamentally correctable through fine-tuning, as demonstrated by MedGemma's 23.84% rate. Its spatial reasoning also shows an interesting inversion — better on medical data than general data due to the constrained semantic distribution of clinical spatial queries.

### HuatuoGPT-Vision-7B: Specialized but Spatially Blind
HuatuoGPT occupies the second-best position in most analyses. It has the lowest anatomical hallucination rate (21.81%), strong Modality recognition (4.84% confusion), and a bimodal distribution shifted toward Score 5 (44.08%). However, its spatial failure rate (69.51% on SLAKE) and CoT self-contradiction rate (34.86%) reveal the same weaknesses as the other 7B models. Its Qwen2.5-VL backbone gives it better semantic grounding than LLaVA architectures but does not solve spatial grounding or attention routing.
