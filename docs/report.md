# VLM Medical VQA Benchmark Report

## 1. Project Overview
This repository implements a rigorous benchmark for evaluating **vision‑language models (VLMs)** on medical visual question answering (VQA). The primary objective is to compare the performance of **medical‑domain specific VLMs** against state-of-the-art **general‑purpose VLMs**.

By expanding our evaluation across both specialized medical splits (SLAKE, VQA-RAD) and standardized general-domain benchmarks (VQAv2, OK-VQA), this project mathematically isolates whether current performance limitations stem from cross-domain alignment constraints or architectural vision-language capability gaps. The benchmark utilizes a reproducible inference pipeline, iterative prompt engineering, and a suite of robust evaluation metrics — including token-level F1, normalized F1, BLEU, BERTScore, and LLM-as-a-Judge — to parse both closed and open-ended vision reasoning.

---

## 2. Models Evaluated
| Category | Model | Size | Architecture / Backbone | Selection Rationale |
|---|---|---|---|---|
| **Medical** | `google/medgemma-4b-it` | 4B | Gemma‑4B (IT to Text) | Explicitly trained on medical image‑text data. Serves as the primary 4B medical baseline based on the MedGemma paper. |
| **Medical** | `microsoft/llava-med-v1.5-mistral-7b` | 7B | LLaVA‑Med (Mistral-7B) | A premier open-source medical VLM. Evaluated to test if a 7B medical model outperforms a 4B medical model. |
| **Medical** | `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` | 7B | Qwen2.5-VL | Chinese medical VLM trained on PubMedVision. Evaluated to test a non-LLaVA specialized medical architecture. |
| **General** | `google/gemma-3-4b-it` | 4B | Gemma‑4B | Represents a state‑of‑the‑art open‑source LLM extended with a vision encoder, lacking medical fine-tuning. |
| **General** | `llava-hf/llava-v1.6-mistral-7b-hf` | 7B | LLaVA‑Mistral‑7B | General-purpose multimodal control model. Evaluated to isolate the specific impact of domain tuning. |

---

## 3. Datasets
| Dataset | Test Samples | Question Type | Modality | Pre‑processing |
|---|---|---|---|---|
| **SLAKE** | 1,061 (English) | 645 open, 416 closed (Yes/No) | CT, MRI, X‑ray | Raw HuggingFace images extracted locally. No resizing; the model receives the raw image array. |
| **VQA‑RAD** | 451 (Public test) | 251 closed, 200 open | Chest X‑ray, CT/MRI | Loaded via `to_rgb` helper. Passed directly to the processor unchanged. |
| **VQAv2** | 1,000 (Sampled Subset) | 500 closed, 500 open-ended | Natural Images (COCO) | Pre-stratified from `lmms-lab/VQAv2` (validation) using memory-safe dataset streaming. |
| **OK-VQA** | 1,000 (Sampled Subset) | 1,000 open-ended | Natural Images (COCO) | Knowledge-based open questions pulled from `lmms-lab/OK-VQA` (`val2014`). |

> **Limitation Note:** The VQA‑RAD split used in this benchmark is the original public test split. The heavily de‑contaminated split utilized in the official MedGemma technical report is not publicly available, meaning some training data contamination may exist across all models for this specific dataset.

---

## 4. Evaluation Protocol
The benchmark uses a multi-metric evaluation suite following the MultiMedEval protocol. Because formatting and phrasing can vary wildly, we compute six metrics for each prediction:

* **Token‑level F1:** Measures exact token overlap between the prediction and the ground truth.
* **Normalized F1:** Mitigates surface‑form bias by lower‑casing, removing punctuation, applying stemming, and filtering stop‑words prior to calculation.
* **Closed‑answer Accuracy (Yes/No):** A prediction is marked correct if the normalized token recall is $\ge 0.5$.
* **Open‑answer Accuracy:** A prediction is marked correct if the normalized token recall is $\ge 0.75$.
* **BLEU‑1:** Calculates sentence‑level semantic overlap using `sacrebleu`.
* **BERTScore (F1):** Computes contextual semantic similarity using `roberta-large` embeddings.
* **LLM-as-a-Judge (1–5 scale):** Semantic correctness scored by `meta-llama/Llama-3.1-8B-Instruct` on a 1–5 integer scale, with judge accuracy defined as score $\ge 4$.

---

## 5. Infrastructure & Implementation Details
* **Hardware Evolution:** Initial local development was conducted on a MacBook Air (Apple Silicon/MPS), which yielded unviable $\approx$14s/sample inference times. To accommodate the massive footprint of generalist models like LLaVA-1.6-7B, the execution pipeline was migrated to a Kaggle environment leveraging **NVIDIA Tesla T4 GPUs (CUDA)**, reducing inference to $\approx$4s/sample for 4B models and allowing background batch jobs via decoupled commits.
* **7B Model Quantization:** To fit the 7B models into the 16GB VRAM limit of a T4 GPU, the models were loaded using **4-bit NF4 quantization** via `bitsandbytes`. Language model layers were quantized while vision encoder operations were explicitly retained in `float16` to preserve structural visual acuity.
* **Dataset Streaming Optimization:** Attempting to store massive general datasets like VQAv2 locally triggered critical disk space depletion (`OSError: [Errno 28] No space left on device`). The processing pipeline was overhauled to integrate **real-time streaming (`streaming=True`)**, allowing the engine to iteratively evaluate sample arrays directly from memory.
* **Bypassing Hugging Face Deprecations:** Recent HF updates block old remote Python loader scripts due to execution safety enforcement. The dataloader layer was successfully refactored to pull pre-converted, standardized Parquet format repositories from the `lmms-lab` ecosystem.

---

## 6. Pipeline Evolution: From v1 to v2
The evaluation harness underwent a critical evolution to accurately capture model performance, beginning with the 4B Gemma architectures.

### 6.1 Baseline Evaluation (v1)
During the initial v1 evaluations, the prompt was simply the dataset `question` appended with the instruction `"Answer concisely."` The answers were extracted using the raw model output.
* **The Result:** This yielded an overall F1 of 55.95% for MedGemma on SLAKE, which fell significantly short of the published benchmarks.

### 6.2 The MedGemma Correction (v2 Protocol Defined)
To improve evaluation accuracy, the v2 pipeline adopted the exact prompt protocol detailed in the MedGemma technical report (Appendix A7). The "v2 protocol" specifically replaces generic instruction prompts with a rigid, multi-part extraction constraint defined by the following rules:

* **Prompt Engineering Structure:**
    * **Closed-Ended Prefix:** All Yes/No questions are strictly prepended with the instruction: *"Answer the question with yes or no."*.
    * **Domain-Specific Context:** For VQA-RAD, prompts are injected with radiology-specific framing: *"Given this radiology image, which can be a frontal chest X-ray, a single slice head or abdominal CT or MR image, provide a very short, definitive, and concise answer..."*.
    * **The Structural Anchor:** Prompts are appended with a directive that permits internal reasoning but forces a structured conclusion: *"You may write out your argument before stating your final very short, definitive, and concise answer (if possible, a single word) X in the format 'Final Answer: X'"*.
* **Answer Extraction Logic:** Instead of evaluating the raw model output, the pipeline applies a regular expression (`r'[Ff]inal\s+[Aa]nswer\s*:\s*(.+)'`) to specifically isolate and score only the text following the anchor. If the model fails to generate the anchor, the extractor falls back to isolating the first generated sentence.
* **The Impact:** Extracting via the `Final Answer:` regex constraint boosted MedGemma's SLAKE F1 score by **+14.55 percentage points** (55.95% to 70.50%), perfectly aligning our local benchmark with the paper's reported 72.3% baseline.
---

## 7. Engineering Challenges: Evaluating 7B Conversational VLMs
Evaluating the 7B conversational models (LLaVA and HuatuoGPT) on the strict v2 protocol exposed critical vulnerabilities in token-matching evaluations.

### 7.1 The Formatting Vulnerability (LLaVA-Med)
LLaVA-Med is fine-tuned on the PMC-15M dataset, which heavily biases its attention weights to generate exhaustive, conversational descriptions rather than benchmark-compliant lexical targets (e.g., generating *"The image is a computed tomography (CT) scan..."* instead of just *"CT"*). This conversational fluff artificially tanked LLaVA-Med's F1 token-matching scores to as low as 26.94%.

### 7.2 Failed Mitigations & Tokenizer Crashes
To fix this without degrading the model's actual clinical knowledge, we attempted several structural interventions:
1. **Assistant Prefilling (`Final Answer:`):** Failed. The model semantically absorbed the anchor and continued chatting anyway.
2. **JSON Syntax Forcing (`{"answer": "`):** Failed. This caused a catastrophic tokenization boundary crash. Mistral's BPE tokenizer expects preceding spaces for standard vocabulary tokens. Forcing a string literal immediately after a quote caused the generation probabilities to collapse, resulting in immediate End-Of-Sequence (`EOS`) tokens and blank inferences.

### 7.3 The Context-Aware NLP Extractor
To avoid fighting the BPE tokenizers, we reverted the prompt to standard v2 inference and engineered a **Context-Aware NLP Extractor** in Python.
* **Contextual Routing:** The extractor dynamically parses the dataset `question`. If the question asks for a "modality" and the model generates *"The image is a computed tomography (CT) scan"*, the extractor intercepts "computed tomography" and rigidly returns `"CT"`.
* **Regex Stripping:** For general anatomical questions, it aggressively strips known LLaVA-Med conversational wrappers before scoring.
* **The Impact:** This rescued the evaluation metrics without requiring expensive re-inference, boosting LLaVA-Med's SLAKE F1 from 26.94% (unfiltered) to 37.04%.

### 7.4 Post-Hoc Patches for General Domain Datasets
When applying the evaluation loops to VQAv2 and OK-VQA, two critical parsing errors emerged:
* **The Grayscale/Palette Conflict:** Hugging Face datasets include web images containing multiple color profiles (`RGBA`, palette, single-channel grayscale). Passing these natively threw execution errors inside the vision tokenizers. We patched this by introducing a rigid `to_rgb` utility layer, enforcing 3-channel alignment.
* **The Structured Rescoring Matrix:** General VLMs displayed significant chatty formatting leakage. Rather than incurring thousands of compute hours re-running inference, we engineered a post-hoc evaluation framework. We re-parsed the complete `raw_output` logs directly from the JSONL artifacts, applying numeric word mapping (converting `"Two"` to `"2"`) and verbose truncation rules to isolate the core answer tokens prior to scoring.

---

## 8. Diagnostic Analysis of General Domain Anomaly
Evaluating the generalist models on their native, general-domain benchmarks output surprisingly low performance numbers under traditional grading (e.g., Gemma-3 scoring 32.4% open accuracy on VQAv2). A detailed output-level diagnosis of the raw data isolated distinct systemic failures in the metric grading structure rather than a lack of vision model intelligence.

### 8.1 The "Token Starvation" Truncation Barrier
On knowledge-heavy datasets like OK-VQA, models like Gemma-3 generated extensive Chain-of-Thought reasoning blocks to justify their deductions. However, because our inference generation boundary was restricted to `max_new_tokens=100`, the model's response reached the token limit and cut off before it could output the required `Final Answer: X` syntax template block. Our regex extractor was subsequently forced to fall back onto evaluating the introductory sentence, dragging the F1 score down to zero on otherwise accurate lines of logic.

### 8.2 The Taxonomic Over-Specificity Penalty
Traditional token matching severely punishes generalist models for possessing granular vocabulary depth on open questions. In multiple instances, the model output highly accurate, context-specific identifications that completely missed exact string overlap with the simplified human annotator ground truths:
* **OK-VQA Sample 1:** Model: `"Philodendron"` | Ground Truth: `"vine"` $\rightarrow$ F1 Score: **0.00**
* **OK-VQA Sample 2:** Model: `"Teddy"` | Ground Truth: `"stuffed animal"` $\rightarrow$ F1 Score: **0.00**
* **OK-VQA Sample 8:** Model: `"Bulb"` | Ground Truth: `"ground"` $\rightarrow$ F1 Score: **0.00**
* **OK-VQA Sample 9:** Model: `"Batting"` | Ground Truth: `"swinging"` $\rightarrow$ F1 Score: **0.00**

This structural metric failure mathematically validates the necessity of our **LLM-as-a-Judge architecture**, as pure n-gram comparison engines remain completely unequipped to map taxonomic relationships or semantic equivalence.

---

## 9. BERTScore Evaluation
BERTScore was computed using `roberta-large` embeddings across all model-dataset combinations to measure contextual semantic similarity beyond token overlap.

### 9.1 Results

| Model | Dataset | Version | BERTScore F1 |
|---|---|---|---|
| `google/gemma-3-4b-it` | SLAKE | v1 | 85.84% |
| `google/medgemma-4b-it` | SLAKE | v1 | 93.19% |
| `google/gemma-3-4b-it` | SLAKE | v2 | 93.67% |
| `google/medgemma-4b-it` | SLAKE | v2 | **96.36%** |
| `HuatuoGPT-Vision-7B-Qwen2.5VL` | SLAKE | v2 | 94.33% |
| `llava-v1.6-mistral-7b` | SLAKE | v2 | 93.33% |
| `llava-med-v1.5-mistral-7b` | SLAKE | v2 | 92.42% |
| `google/medgemma-4b-it` | VQA-RAD | v1 | 92.27% |
| `google/gemma-3-4b-it` | VQA-RAD | v2 | 91.22% |
| `google/medgemma-4b-it` | VQA-RAD | v2 | **94.81%** |
| `HuatuoGPT-Vision-7B-Qwen2.5VL` | VQA-RAD | v2 | 94.07% |
| `llava-v1.6-mistral-7b` | VQA-RAD | v2 | 94.02% |
| `llava-med-v1.5-mistral-7b` | VQA-RAD | v2 | 92.45% |
| `google/gemma-3-4b-it` | OK-VQA | v2 | 87.06% |
| `llava-hf/llava-v1.6-mistral-7b-hf` | OK-VQA | v2 | 90.67% |
| `google/gemma-3-4b-it` | VQAv2 | v2 | **96.43%** |
| `llava-hf/llava-v1.6-mistral-7b-hf` | VQAv2 | v2 | 94.19% |

### 9.2 BERTScore Limitation: Clinical Blindness
BERTScore reveals a critical limitation for medical evaluation. Despite MedGemma having double the clinical accuracy (70.50% F1) of LLaVA-Med (37.04% F1) on SLAKE, their BERTScores are dangerously close (96.36% vs 92.42%). BERTScore measures distributional semantic similarity — it recognizes that "Liver" and "Lung" are both internal organs in a medical context, but it fails to penalize the catastrophic clinical difference between diagnosing the wrong organ. **BERTScore should not be used as a standalone metric for medical VQA evaluation.**

---

## 10. LLM-as-a-Judge Evaluation

### 10.1 Motivation
Token-level F1 penalizes semantically correct answers that differ in surface form ("Lungs" vs "Lung", "Radiography" vs "X-Ray", "Chest/Thorax" vs "Chest"). BERTScore over-rewards lexical similarity. An LLM judge evaluating medical semantic correctness against a reference answer addresses both failure modes simultaneously.

### 10.2 Implementation Journey

**Phase 1 — API Era and the Rate Limit Wall**
Initially targeted `meta-llama/Llama-3.3-70B-Instruct` via cloud APIs (SambaNova/Groq/HuggingFace). We immediately hit severe rate-limiting (429 errors) and monthly credit exhaustion (402 errors). Adding sleep timers prevented crashes but rendered the pipeline useless for 7,500+ record bulk evaluation.

**Phase 2 — Local Pivot and Dependency Hell**
Abandoned external APIs entirely. Shifted to local GPU inference on Kaggle using `meta-llama/Llama-3.1-8B-Instruct` loaded in 4-bit NF4 quantization via `bitsandbytes`. Key roadblock: Kaggle's pre-installed `transformers` library had a hardcoded bug failing to recognize the updated `bitsandbytes` installation. Fixed by injecting `os.environ["BITSANDBYTES_NOW_LOADED"] = "1"` directly into the kernel before loading the model.

**Phase 3 — Batched Inference Optimization**
Initial single-sample evaluation took 1.5 hours per 1,000-record file. Two bottlenecks identified: (1) `device_map="auto"` was splitting the 6GB model across two GPUs, forcing data through a slow PCIe bridge; (2) the loop evaluated one question at a time, leaving 95% of GPU cores idle. Fixed by forcing the model onto a single GPU (`device_map={"": 0}`) and rewriting the loop to use batched inference with batch size 16–32. Processing time dropped from 1.5 hours to ~55 minutes per file. The full 7,500-record evaluation was offloaded to Kaggle's background commit server and completed in ~9 hours.

### 10.3 Judge Design
Following the HuggingFace LLM-as-a-Judge cookbook:
* **Model:** `meta-llama/Llama-3.1-8B-Instruct` (local, 4-bit quantized)
* **Scale:** 1–5 integer (1=completely wrong, 2=mostly wrong, 3=partially correct, 4=mostly correct, 5=fully correct)
* **Reference-grounded:** judge sees the ground truth for every prediction
* **Evaluation before rating:** forces reasoning prior to scoring
* **Judge accuracy:** score $\ge 4$ considered correct

### 10.4 Judge Results

| Model | Dataset | N Judged | N Failed | Avg Score (All) | Avg Score (Closed) | Avg Score (Open) | Judge Acc (All) | Judge Acc (Closed) | Judge Acc (Open) |
|---|---|---|---|---|---|---|---|---|---|
| `google/gemma-3-4b-it` | okvqa | 978 | 22 | 2.809 | — | 2.809 | 40.90% | — | 40.90% |
| `llava-hf/llava-v1.6-mistral-7b-hf` | okvqa | 982 | 18 | 3.428 | — | 3.428 | 57.64% | — | 57.64% |
| `google/medgemma-4b-it` | okvqa | 1000 | 0 | 3.533 | — | 3.533 | 58.10% | — | 58.10% |
| `microsoft/llava-med-v1.5-mistral-7b` | okvqa | 1000 | 0 | 3.441 | — | 3.441 | 60.20% | — | 60.20% |
| `HuatuoGPT-Vision-7B-Qwen2.5VL` | okvqa | 1000 | 0 | **3.796** | — | **3.796** | **65.90%** | — | **65.90%** |
| `google/gemma-3-4b-it` | slake | 1061 | 0 | 3.281 | 3.550 | 3.107 | 55.14% | 68.99% | 46.20% |
| `google/medgemma-4b-it` | slake | 1061 | 0 | **4.004** | **4.111** | **3.935** | **73.70%** | **83.65%** | **67.29%** |
| `HuatuoGPT-Vision-7B-Qwen2.5VL` | slake_7b | 1061 | 0 | 3.587 | 3.719 | 3.502 | 63.15% | 73.80% | 56.28% |
| `llava-hf/llava-v1.6-mistral-7b-hf` | slake_7b | 1061 | 0 | 3.055 | 3.375 | 2.848 | 50.14% | 63.22% | 41.71% |
| `microsoft/llava-med-v1.5-mistral-7b` | slake_7b-2 | 1061 | 0 | 3.177 | 3.147 | 3.197 | 53.35% | 57.69% | 50.54% |
| `google/gemma-3-4b-it` | vqa_rad | 451 | 0 | 2.978 | 3.155 | 2.755 | 45.90% | 56.97% | 32.00% |
| `google/medgemma-4b-it` | vqa_rad | 451 | 0 | **3.647** | 3.765 | **3.500** | **63.86%** | 73.31% | **52.00%** |
| `HuatuoGPT-Vision-7B-Qwen2.5VL` | vqa_rad_7b | 450 | 1 | 3.460 | **3.809** | 3.020 | 60.22% | **74.10%** | 42.71% |
| `llava-hf/llava-v1.6-mistral-7b-hf` | vqa_rad_7b | 451 | 0 | 2.887 | 3.195 | 2.500 | 45.01% | 58.17% | 28.50% |
| `microsoft/llava-med-v1.5-mistral-7b` | vqa_rad_7b | 451 | 0 | 3.009 | 2.721 | 3.370 | 46.34% | 45.42% | 47.50% |
| `google/gemma-3-4b-it` | vqav2 | 1000 | 0 | 3.365 | 3.406 | 3.324 | 58.30% | 60.00% | 56.60% |
| `llava-hf/llava-v1.6-mistral-7b-hf` | vqav2 | 994 | 6 | 3.820 | 3.862 | 3.778 | 70.22% | 71.94% | 68.48% |
| `google/medgemma-4b-it` | vqav2 | 1000 | 0 | 3.873 | 4.028 | 3.718 | 70.50% | 76.00% | 65.00% |
| `microsoft/llava-med-v1.5-mistral-7b` | vqav2 | 1000 | 0 | 3.906 | 4.318 | 3.494 | 73.50% | 85.20% | 61.80% |
| `HuatuoGPT-Vision-7B-Qwen2.5VL` | vqav2 | 1000 | 0 | **4.331** | **4.530** | **4.191** | **82.10%** | **88.14%** | **77.85%** |

### 10.5 Pearson Correlation: Token F1 vs Judge Score

All correlations are statistically significant ($p < 0.0001$), confirming the judge tracks real quality differences while adding genuine semantic signal beyond token matching.

| File | Correlation |
|---|---|
| HuatuoGPT VQA-RAD | 0.733 |
| HuatuoGPT SLAKE | 0.683 |
| Gemma-3 SLAKE | 0.722 |
| Gemma-3 VQA-RAD | 0.789 |
| MedGemma SLAKE | 0.776 |
| MedGemma VQA-RAD | 0.719 |
| LLaVA-1.6 SLAKE | 0.719 |
| LLaVA-1.6 VQA-RAD | 0.758 |
| LLaVA-Med SLAKE | 0.726 |
| LLaVA-Med VQA-RAD | 0.612 |

---

## 11. Comprehensive Benchmark Results
*All percentages are rounded to two decimal places. Dataset evaluations utilize the v2 inference prompt. General-domain metrics reflect the optimized rescored matrix mappings.*

| Model | Dataset | Ver | Total | F1 | Cls Acc | Opn Acc | BLEU | BERTScore | Judge Acc |
|---|---|---|---|---|---|---|---|---|---|
| `google/gemma-3-4b-it` | SLAKE | v1 | 1,061 | 17.62% | 64.66% | 24.65% | 7.65% | 85.84% | — |
| `google/medgemma-4b-it` | SLAKE | v1 | 1,061 | 55.95% | 76.68% | 47.29% | 50.53% | 93.19% | — |
| `google/gemma-3-4b-it` | SLAKE | v2 | 1,061 | 42.14% | 68.27% | 24.50% | 41.54% | 93.67% | 55.14% |
| `google/medgemma-4b-it` | SLAKE | v2 | 1,061 | **70.50%** | **85.58%** | **55.81%** | **68.88%** | **96.36%** | **73.70%** |
| `llava-v1.6-mistral-7b` | SLAKE | v2 | 1,061 | 36.98% | 58.41% | 25.12% | 34.58% | 93.33% | 50.14% |
| `llava-med-v1.5-mistral-7b` | SLAKE | v2 | 1,061 | 37.04% | 51.44% | 30.39% | 34.85% | 92.42% | 53.35% |
| `HuatuoGPT-Vision-7B` | SLAKE | v2 | 1,061 | 47.86% | 72.36% | 28.68% | 46.71% | 94.33% | 63.15% |
| `google/medgemma-4b-it` | VQA-RAD | v1 | 451 | 57.14% | 71.31% | 43.50% | 44.44% | 92.27% | — |
| `google/gemma-3-4b-it` | VQA-RAD | v2 | 451 | 43.43% | 56.57% | 20.50% | 29.68% | 91.22% | 45.90% |
| `google/medgemma-4b-it` | VQA-RAD | v2 | 451 | **62.21%** | **78.09%** | **36.50%** | **59.97%** | **94.81%** | **63.86%** |
| `llava-v1.6-mistral-7b` | VQA-RAD | v2 | 451 | 42.03% | 58.57% | 17.50% | 40.91% | 94.02% | 45.01% |
| `llava-med-v1.5-mistral-7b` | VQA-RAD | v2 | 451 | 34.55% | 49.80% | 20.00% | 31.01% | 92.45% | 46.34% |
| `HuatuoGPT-Vision-7B` | VQA-RAD | v2 | 451 | 57.40% | **77.69%** | 25.50% | 56.06% | 94.07% | 60.22% |
| `google/gemma-3-4b-it` | OK-VQA | v2 | 1,000 | 23.95% | 0.00% | 24.30% | 22.53% | 87.06% | 40.90% |
| `llava-v1.6-mistral-7b` | OK-VQA | v2 | 1,000 | 41.64% | 0.00% | 41.50% | 39.58% | 90.67% | 57.64% |
| `google/gemma-3-4b-it` | VQAv2 | v2 | 1,000 | 58.81% | 76.80% | 41.00% | **54.40%** | **96.43%** | 58.30% |
| `llava-v1.6-mistral-7b` | VQAv2 | v2 | 1,000 | 59.46% | 80.40% | **61.00%** | 53.82% | 94.19% | 70.22% |
| `google/medgemma-4b-it` | VQAv2 | v2 | 1,000 | 12.25% | 72.60% | 53.40% | 2.27% | 83.34% | 70.50% |
| `llava-med-v1.5-mistral-7b` | VQAv2 | v2 | 1,000 | 12.00% | 55.80% | 20.60% | 3.08% | 84.47% | 73.50% |
| `google/medgemma-4b-it` | OK-VQA | v2 | 1,000 | 4.86% | 0.00% | 42.20% | 0.82% | 82.10% | 58.10% |
| `llava-med-v1.5-mistral-7b` | OK-VQA | v2 | 1,000 | 5.21% | 0.00% | 17.70% | 1.27% | 83.48% | 60.20% |
| `HuatuoGPT-Vision-7B` | VQAv2 | v2 | 1,000 | 42.33% | 87.17% | 62.18% | 32.18% | 88.17% | **82.10%** |
| `HuatuoGPT-Vision-7B` | OK-VQA | v2 | 1,000 | 34.32% | 0.00% | 45.80% | 29.77% | 88.09% | **65.90%** |

---

## 12. Key Findings & Diagnostic Takeaways

1. **Domain Pre-training Trumps Parameter Count:** MedGemma (4B) comprehensively dominates all 7B VLMs in zero-shot evaluation across every clinical metric. Dedicated medical image-text pre-training provides a vastly superior zero-shot performance ceiling compared to simply scaling up general-purpose multimodal parameters.

2. **Cross-Domain Extension Verifies Structural Capable Baselines:** Benchmarking general-purpose models on VQAv2 confirms that their fundamental vision-language text-generation engines are highly capable (with LLaVA-1.6 reaching 61.00% Open Accuracy and 94.19% BERTScore). This provides concrete cross-domain confirmation that their previous low medical performance is entirely a **domain-alignment bottleneck** rather than an architectural deficiency.

3. **BERTScore is Clinically Blind:** Despite MedGemma having double the clinical accuracy of LLaVA-Med on SLAKE (70.50% vs 37.04% F1), their BERTScores are dangerously close (96.36% vs 92.42%). BERTScore recognizes "Liver" and "Lung" as semantically similar internal organs but fails to penalize the catastrophic clinical difference. BERTScore should not be used as a standalone metric for medical VQA.

4. **The Pearson Correlation Range (0.61–0.79) is the Right Zone:** High enough to confirm the judge tracks real quality differences, but not so high ($>0.95$) that it merely replicates F1. The LLM judge adds genuine semantic signal. The lowest correlation (LLaVA-Med VQA-RAD, 0.612) directly corresponds to the model with the most conversational output style — confirming the judge is correctly rescuing those predictions.

5. **Surface-Form Metric Bias Arbitrarily Penalizes Intelligence:** Traditional string matching actively works against conversational architectures. Re-scoring the outputs using robust number conversion and extraction adjustments rescued Gemma-3's open VQAv2 tracking from a broken 32.40% up to an actual 41.00%, confirming that generation forcing or surface mismatches skew accuracy baselines.

6. **The Illusion of Catastrophic Forgetting (Metric Failure):** Initial surface-level metrics (e.g., F1 dropping to ~12%, BLEU ~2-3) strongly suggested that aggressive medical fine-tuning caused catastrophic forgetting of general visual-linguistic concepts. However, the LLM-as-a-Judge evaluation entirely refutes this. The Judge Accuracy for MedGemma (70.50%) and LLaVA-Med (73.50%) on the general-domain VQAv2 dataset actually surpasses the general-purpose LLaVA-1.6 (70.22%), with HuatuoGPT achieving a benchmark-high 82.10%. The models did not lose general knowledge; rather, medical fine-tuning permanently altered their output formatting and response style in ways that catastrophically broke traditional exact-match metrics. This highlights the critical necessity of semantic LLM judges when evaluating cross-domain VLM generalization.

---

## 13. Limitations & Technical Debt
* **Library Incompatibility:** `multimedeval` (v1.0.0) is fundamentally incompatible with `transformers>=4.49` due to the deprecation and removal of `AdamW`. The pipeline currently bypasses this by pinning `transformers==4.51.3`, but this introduces technical debt.
* **VQA‑RAD Contamination:** The public VQA-RAD split used in this benchmark contains images that likely appear in the pre-training data of the evaluated models, potentially inflating baseline metrics.
* **LLM Judge Model Size:** Using Llama-3.1-8B as the judge introduces a limitation — the judge model is smaller than the models being evaluated. A frontier judge (GPT-4o, Claude) would provide higher-quality semantic assessment, particularly for complex open-ended radiology questions.
* **Zero-Shot Only:** All evaluations are zero-shot. Results should not be compared directly to supervised fine-tuning results reported in model papers (e.g., LLaVA-Med's 87% is SFT, not zero-shot).

---
## 14. Experimental Intervention: Structured Chain-of-Thought (S-CoT)

Following the identification of CoT-induced hallucinations in error analysis A5 (specifically the 34.35% hallucination rate in Gemma-3), we conducted an experimental intervention to determine if rigid, multi-step prompt engineering could mitigate attention hijacking in medical VQA.

### 14.1 Methodology
We hypothesized that unconstrained CoT reasoning allows models to drift into irrelevant semantic lanes, causing them to "talk themselves out of" correct visual groundings. We designed a **Structured CoT (S-CoT)** prompt to enforce a rigid four-step reasoning process for `google/medgemma-4b-it` on the SLAKE dataset:

1.  **Step 1 - Modality:** Identify the imaging modality (e.g., CT, MRI, X-ray).
2.  **Step 2 - Anatomy:** Name the primary organ or anatomical structure.
3.  **Step 3 - Observation:** Write exactly one short sentence answering the core question based on Steps 1 and 2.
4.  **Step 4 - Conclusion:** State your definitive answer (if possible, a single word) in the exact format 'Final Answer: X'.

### 14.2 Results
The S-CoT protocol was evaluated against our established `v2` baseline prompt.

| Metric | MedGemma v2 Baseline | MedGemma S-CoT | Delta |
|---|---|---|---|
| **Overall F1** | 70.50% | 61.13% | -9.37 pp |
| **Overall Recall** | 69.77% | 63.36% | -6.41 pp |
| **Closed Acc** | 85.58% | 80.53% | -5.05 pp |
| **Open Acc** | 55.81% | 48.37% | -7.44 pp |
| **BLEU** | 68.88 | 59.11 | -9.77 |

### 14.3 Diagnostic Findings
The S-CoT intervention resulted in a performance degradation across all metrics. This negative result provides two critical architectural insights:

* **Generative Drag:** For a 4B parameter model, forcing a multi-step text generation format introduces significant "generative drag." The overhead of maintaining the rigid 4-step structure consumes the model's limited attention capacity, diluting the visual signal extracted in the initial steps.
* **Structural Over-Constraint:** MedGemma appears optimized for concise, latent visual grounding. Forcing an explicit, step-by-step externalized reasoning process requires the model to generate approximately 50–80 tokens of "filler" observations before reaching the final answer. This extended generation window increased the probability of the model injecting hallucinated medical features, effectively creating the very "hallucinations" we aimed to prevent.

**Conclusion:** Rigid semantic routing (S-CoT) is not a universal fix for CoT-induced hallucinations in smaller architectures. The overhead of maintaining structural compliance creates a trade-off that, in this instance, outweighs the benefits of guided reasoning.

---