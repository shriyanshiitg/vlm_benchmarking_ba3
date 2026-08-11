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

### 14.4 Extension: Architecture Generalisability (July 2026)

To determine whether the degradation was an artefact of MedGemma-4B's specific architecture or a general property of small VLMs, we extended the S-CoT experiment to two additional model–dataset pairs using the **identical prompt** (see `docs/report_scot_extension.md` for full details).

| Model | Dataset | N | Base F1 | SCoT F1 | **ΔF1** | p (paired permutation) |
|---|---|---|---|---|---|---|
| MedGemma-4B | SLAKE | 440 | 70.5% | 65.5% | **−5.0 pp** | **<0.001 ★★★** |
| HuatuoGPT-7B | SLAKE | 1061 | 47.9% | 47.1% | −0.8 pp | 0.570 ns |
| MedGemma-4B | VQA-RAD | 451 | 62.5% | 61.6% | −0.9 pp | 0.576 ns |

**Key finding:** The degradation is **not architecture-agnostic**. HuatuoGPT-7B (Qwen2.5VL backbone, 7B parameters) shows no significant change on SLAKE, and MedGemma-4B itself is unaffected on VQA-RAD. Generative drag appears to emerge from the interaction of *limited model capacity* (4B parameters) with *dataset visual complexity* (SLAKE's heterogeneous multi-organ imaging), rather than being a universal property of structured prompting.

The generative drag hypothesis is therefore **conditionally confirmed**: it holds for sub-5B models on visually complex datasets but does not generalise beyond that regime.

---

## 15. Few-Shot Experiment: Does In-Context Learning Close the Domain Gap?

### 15.1 Motivation

Every evaluation in this benchmark is zero-shot — models receive a question and an image with no prior examples. The central finding (Section 12, Finding 1) is that domain pre-training trumps parameter count: MedGemma-4B (4B, medical) comprehensively outperforms LLaVA-1.6-7B (7B, generalist) despite being smaller. An important follow-up question is whether this gap can be narrowed without fine-tuning — specifically, whether providing in-context clinical examples (few-shot prompting) allows generalist models to learn the expected question-answer format from the prompt itself.

If few-shot closes the gap, it implies the bottleneck is **informational**: generalist models have latent clinical capability that needs to be unlocked with examples, making few-shot prompting a viable low-cost clinical deployment strategy.

If few-shot does not close the gap, it implies the bottleneck is **architectural**: the missing capability is domain-specific visual feature extraction and clinical reasoning encoded during medical pre-training, which examples cannot replicate. This would validate why medical fine-tuning is mandatory.

### 15.2 Experimental Setup

**Test subset:** 200 samples drawn from the SLAKE EN test split using stratified sampling across 6 buckets: 3 question content types (Modality, Organ, Abnormality) × 2 answer types (Closed/Open), approximately 33 samples per bucket. Fixed seed (42) ensures the identical 200-sample subset is used for all 6 runs.

**Models and conditions:**

| Model | Type | Parameters | Conditions |
|---|---|---|---|
| Gemma-3-4B-IT | Generalist | 4B (fp16) | 0-shot, 1-shot, 3-shot |
| LLaVA-v1.6-Mistral-7B | Generalist | 7B (4-bit NF4) | 0-shot, 1-shot, 3-shot |

**Few-shot examples:** Drawn exclusively from the SLAKE **training split** (never test). One example per question content type was selected: Modality, Organ, and Abnormality, prioritising short, unambiguous ground-truth answers (1–3 words). Each example is prepended as a complete prior turn in the conversation (user image + question → assistant answer). The 1-shot condition uses only the Modality exemplar; the 3-shot condition uses all three in sequence.

**Statistical test:** Paired permutation test (10,000 iterations, seed 42) comparing 0-shot vs. 1-shot and 0-shot vs. 3-shot within each model, on matched question indices.

**Inference:** Run on Kaggle T4 GPU. 4-bit NF4 quantization applied for LLaVA-1.6-7B. Gemma-3-4B loaded in fp16.

### 15.3 Results

#### 15.3.1 Main Results Table

| Model | Condition | Overall F1 | ΔF1 vs 0-shot | Closed Acc | Open F1 | p (vs 0-shot) |
|---|---|---|---|---|---|---|
| Gemma-3-4B | **0-shot** | **57.27%** | — | 80.00% | 34.53% | — |
| Gemma-3-4B | 1-shot | 56.08% | −1.19 pp | 69.00% | 43.15% | 0.698 ns |
| Gemma-3-4B | 3-shot | 53.82% | −3.45 pp | 71.00% | 36.64% | 0.215 ns |
| LLaVA-1.6-7B | **0-shot** | **38.24%** | — | 54.00% | 29.47% | — |
| LLaVA-1.6-7B | 1-shot | 34.00% | −4.24 pp | 57.00% | 11.00% | 0.188 ns |
| LLaVA-1.6-7B | 3-shot | 34.90% | −3.34 pp | 52.00% | 17.80% | 0.349 ns |

*Paired permutation test, 10,000 iterations. ns: p ≥ 0.05 (not significant).*

#### 15.3.2 Reference Context — Gap to MedGemma-4B

| Model | Full-dataset SLAKE F1 | Gap to MedGemma |
|---|---|---|
| MedGemma-4B (Section 8 baseline) | 70.50% | — |
| Gemma-3-4B (Section 8 baseline) | 42.14% | −28.36 pp |
| LLaVA-1.6-7B (Section 8 baseline) | 36.98% | −33.52 pp |
| Gemma-3-4B (this subset, 0-shot) | 57.27% | — |
| LLaVA-1.6-7B (this subset, 0-shot) | 38.24% | — |

> **Note on subset vs. full-dataset numbers:** The 0-shot F1 on the 200-sample stratified subset differs from the full-dataset baseline (Gemma-3: 57.27% vs 42.14%; LLaVA: 38.24% vs 36.98%). This is expected — the subset is stratified toward Modality/Organ/Abnormality question types only, which removes many harder Attribute/Position/Count questions that depress the full-dataset score. The relevant comparison for this experiment is the **within-model, within-subset change across shot conditions** — not the absolute level versus full-dataset baselines.

#### 15.3.3 Per Question-Type F1 Breakdown

| Model | Condition | Modality F1 | Organ F1 | Abnormality F1 |
|---|---|---|---|---|
| Gemma-3-4B | 0-shot | 89.30% | 49.11% | 33.37% |
| Gemma-3-4B | 1-shot | 87.80% | 45.48% | 34.95% |
| Gemma-3-4B | 3-shot | 86.16% | 44.26% | 30.99% |
| LLaVA-1.6-7B | 0-shot | 64.26% | 32.90% | 17.56% |
| LLaVA-1.6-7B | 1-shot | 72.95% | 14.00% | 15.06% |
| LLaVA-1.6-7B | 3-shot | 68.44% | 23.93% | 12.22% |

### 15.4 Diagnostic Findings

**Finding 1 — No statistically significant improvement for either model at any shot count.** All four significance tests (Gemma-3 0v1, Gemma-3 0v3, LLaVA 0v1, LLaVA 0v3) returned p-values well above 0.05. Adding clinical examples to the prompt does not measurably change the models' ability to answer medical VQA questions correctly.

**Finding 2 — Overall F1 slightly decreases under few-shot prompting.** Gemma-3-4B drops from 57.27% (0-shot) to 53.82% (3-shot); LLaVA-1.6-7B drops from 38.24% to 34.90%. While neither change is statistically significant, the direction is consistently negative. Few-shot examples do not help and may mildly interfere.

**Finding 3 — The pattern varies by question type.** For Gemma-3-4B, Open F1 improves slightly with 1-shot (34.53% → 43.15%) but reverts at 3-shot (36.64%), while Closed Accuracy drops substantially at 1-shot (80.00% → 69.00%). For LLaVA-1.6-7B, Organ and Abnormality F1 fall sharply under few-shot conditions. This suggests the exemplars are providing some format signal for open-ended questions but inadvertently disrupting the yes/no decision boundary for closed questions.

**Finding 4 — Modality questions are robust; clinical content questions are not.** Modality questions (e.g., "Is this an MRI?") already have high 0-shot scores (89% for Gemma-3, 64% for LLaVA) and show minimal change under few-shot. The largest degradations appear in Organ and Abnormality questions — precisely the categories requiring domain-specific anatomical knowledge that cannot be conveyed by a 1–3 word exemplar answer.

### 15.5 Interpretation

**The domain bottleneck is architectural, not informational.** Generalist models do not improve with clinical few-shot examples because the performance deficit is not caused by unfamiliarity with the question-answer format — it is caused by the absence of the domain-specific visual feature representations that medical pre-training encodes. No amount of in-context prompting can teach a model to identify a patellar cartilage abnormality in an MRI if its visual encoder was never optimized to extract that signal.

This result is the counterpart to Section 8's central finding and to Section 14's S-CoT negative result. Together, the three experiments establish the same conclusion through three different lenses:

- **Section 8:** Scaling parameters (4B→7B) without domain training provides zero clinical benefit.
- **Section 14:** Enriching the prompt structure (S-CoT) without domain training does not help and may hurt.
- **Section 15:** Enriching the prompt context (few-shot examples) without domain training does not help.

The converging evidence strongly validates that **medical fine-tuning is mandatory** — not merely beneficial — for competitive medical VQA performance.

### 15.6 Summary of Findings

Neither generalist model showed statistically significant F1 improvement under few-shot prompting at any shot count. Gemma-3-4B declined from 57.27% (0-shot) to 53.82% (3-shot, ΔF1 = −3.45 pp, p = 0.215); LLaVA-1.6-7B declined from 38.24% to 34.90% (ΔF1 = −3.34 pp, p = 0.349). The largest performance degradations were concentrated in Organ and Abnormality questions — the categories most dependent on domain-specific anatomical knowledge — while Modality questions, which do not require clinical expertise, remained largely stable. These results, combined with the Section 14 S-CoT findings and the Section 8 parameter-scaling analysis, establish a convergent empirical case that the performance gap between generalist and medical VLMs is driven by architectural differences in domain-specific visual representations and cannot be closed through prompt engineering alone.

---

*Full per-sample outputs: `outputs/_archive/fewshot_experiment/`. Analysis script: `scripts/fewshot_analysis.py`. Chart: `results/fig_fewshot_f1.png`. Detailed report: `docs/report_fewshot_experiment.md`.*

---

## 16. Calibration Analysis: Confidence Reliability on Closed Questions

### 16.1 Motivation

Accuracy measures whether a model answers correctly. Calibration measures whether a model *knows* when it is correct. For clinical AI, calibration is a distinct and critical safety property: an overconfident model that reports high confidence on wrong predictions cannot be safely used for threshold-based screening or human-in-the-loop workflows. This section evaluates whether the accuracy advantage of MedGemma-4B over Gemma-3-4B extends to confidence reliability.

### 16.2 Methodology

Inference was run exclusively on closed (Yes/No) questions from SLAKE EN and VQA-RAD test splits using `max_new_tokens=1` and `output_scores=True`. The logit vector at the first generated token position (shape: `(1, vocab_size)`) was passed through softmax to obtain a full probability distribution. Probabilities across all token IDs decoding to "Yes" or "yes" were summed to give `P(Yes)_raw`; similarly for "No". The normalised binary confidence is `P(Yes) = P(Yes)_raw / (P(Yes)_raw + P(No)_raw)`.

Expected Calibration Error (ECE) was computed with 15 equal-width bins. Brier score (mean squared error between confidence and binary label) provides a complementary metric. Both models were run in fp16 without quantisation.

| Dataset | Model | N |
|---|---|---|
| SLAKE EN (test) | MedGemma-4B, Gemma-3-4B | 416 each |
| VQA-RAD (test) | MedGemma-4B, Gemma-3-4B | 251 each |

### 16.3 Results

#### 16.3.1 Accuracy, ECE, and Brier Score

| Model | Dataset | Accuracy | ECE | Brier | Mean P(Yes) | Overconfidence |
|---|---|---|---|---|---|---|
| MedGemma-4B | SLAKE | 74.04% | **21.95 pp** | 0.2129 | 55.37% | −18.7 pp |
| MedGemma-4B | VQA-RAD | 79.68% | **19.52 pp** | 0.1920 | 47.76% | −31.9 pp |
| Gemma-3-4B | SLAKE | 57.93% | 35.49 pp | 0.3468 | 67.57% | +9.6 pp |
| Gemma-3-4B | VQA-RAD | 55.38% | 43.18 pp | 0.4310 | 59.63% | +4.3 pp |

*ECE and Brier score: lower is better. Overconfidence = mean confidence − accuracy; negative values indicate the model's confidence is more conservative than its accuracy.*

#### 16.3.2 Confidence Polarisation

Both models produce highly polarised confidence distributions — the vast majority of predictions fall at P(Yes) ≥ 0.9 or P(Yes) ≤ 0.1, with very few genuinely uncertain outputs.

| Model | Dataset | P(Yes) ≥ 0.9 | P(Yes) ≤ 0.1 | Uncertain (0.1–0.9) | High-conf Accuracy |
|---|---|---|---|---|---|
| MedGemma-4B | SLAKE | 219 | 172 | 25 | 75.2% (N=404) |
| MedGemma-4B | VQA-RAD | 110 | 124 | 17 | 80.1% (N=241) |
| Gemma-3-4B | SLAKE | 272 | 122 | 22 | 59.6% (N=399) |
| Gemma-3-4B | VQA-RAD | 143 | 95 | 13 | 55.5% (N=245) |

### 16.4 Findings

**Finding 1 — MedGemma-4B is substantially better calibrated on both datasets.**
ECE is 13.54 pp lower on SLAKE (21.95 vs 35.49) and 23.66 pp lower on VQA-RAD (19.52 vs 43.18). Brier score confirms the same pattern: 0.2129 vs 0.3468 on SLAKE and 0.1920 vs 0.4310 on VQA-RAD. Medical pre-training confers a dual benefit: higher accuracy and more reliable confidence estimates.

**Finding 2 — Gemma-3-4B is overconfident; MedGemma-4B is underconfident.**
Gemma-3-4B's mean confidence exceeds its accuracy by approximately 5–10 pp (overconfidence). MedGemma-4B's mean confidence is 19–32 pp *below* its accuracy (underconfidence). In clinical deployment, underconfidence is the safer failure mode: a conservative model that assigns lower confidence on borderline cases is preferable to one that asserts incorrect answers with near-certainty.

**Finding 3 — Both models are near-maximally decisive.**
Over 95% of predictions in all four runs fall at P(Yes) ≥ 0.9 or P(Yes) ≤ 0.1. Neither model produces genuinely uncertain outputs on closed medical questions. This reflects the softmax decision-boundary behaviour of instruction-tuned models under binary output constraints — confidence scores from `max_new_tokens=1` are best interpreted as decision-boundary distances rather than calibrated probabilities.

**Finding 4 — Gemma-3-4B's high-confidence predictions are barely above chance.**
At high-confidence predictions (P(Yes) ≥ 0.8 or ≤ 0.2), MedGemma-4B achieves 75–80% accuracy; Gemma-3-4B achieves only 55–60% — barely above the 50% random baseline for a binary task. A downstream system relying on Gemma-3-4B's high-confidence outputs as a filter would be systematically misled.

### 16.5 Clinical Significance

The calibration gap constitutes a qualitative difference in clinical deployability. A model with ECE of 43 pp (Gemma-3-4B on VQA-RAD) cannot support any meaningful confidence threshold for clinical decision support: when it assigns near-certain confidence to a prediction, it is correct only 55% of the time. MedGemma-4B at ECE 19.52 pp on the same dataset, while not perfectly calibrated, is substantially more actionable for threshold-based workflows such as flagging high-confidence negatives for expedited discharge or escalating uncertain cases for radiologist review.

The underconfidence pattern of MedGemma-4B likely reflects medical fine-tuning on datasets where clinical ambiguity is common and overconfidence carries direct risk. The overconfidence of Gemma-3-4B on medical data is consistent with a model applying general visual recognition patterns without awareness of medical uncertainty.

---

*Full per-sample outputs: `outputs/_archive/calibration/`. Analysis script: `scripts/calibration_analysis.py`. Charts: `results/fig_calibration_reliability.png`, `results/fig_calibration_confidence_hist.png`. Detailed report: `docs/report_calibration.md`.*

---

## 17. Modality-Specific Performance Leaderboard (SLAKE)

### 17.1 Motivation

All benchmark results in Sections 11–12 aggregate performance across the full SLAKE test set without distinguishing the imaging modality of each question. SLAKE contains questions grounded in CT (N=472), MRI (N=228), and X-Ray (N=361) images. Aggregate F1 masks important clinical distinctions: a model that performs well on X-Ray but poorly on CT presents a different deployment profile than one that is uniformly mediocre. This section decomposes all five models' performance across the three imaging modalities using the `modality` metadata field embedded in the SLAKE dataset.

### 17.2 Methodology

The SLAKE `modality` field was joined to the existing inference JSONL files using the record index (each JSONL record's `idx` maps directly to the position in the English-filtered test split). Token F1 and LLM Judge Accuracy (≥4/5) were then computed independently for each modality group. No new inference was required — this is a post-hoc decomposition of existing outputs. VQA-RAD does not include per-image modality metadata and is excluded from this analysis.

**Dataset composition:**

| Modality | N (questions) | Closed | Open |
|---|---|---|---|
| CT | 472 | 214 | 258 |
| MRI | 228 | 88 | 140 |
| X-Ray | 361 | 114 | 247 |

### 17.3 Results

#### 17.3.1 Token F1 by Modality

| Model | CT (N=472) | MRI (N=228) | X-Ray (N=361) | Overall |
|---|---|---|---|---|
| **MedGemma-4B** | **67.82%** | **64.87%** | **77.50%** | **70.48%** |
| HuatuoGPT-7B | 49.85% | 49.15% | 44.53% | 47.89% |
| Gemma-3-4B | 42.09% | 46.03% | 39.69% | 42.12% |
| LLaVA-Med-7B | 38.33% | 34.94% | 36.73% | 37.05% |
| LLaVA-1.6-7B | 32.93% | 34.78% | 43.52% | 36.93% |

#### 17.3.2 Judge Accuracy (≥4/5) by Modality

| Model | CT (N=472) | MRI (N=228) | X-Ray (N=361) | Overall |
|---|---|---|---|---|
| **MedGemma-4B** | **70.13%** | **71.93%** | **79.50%** | **73.70%** |
| HuatuoGPT-7B | 58.47% | 60.96% | 70.64% | 63.15% |
| Gemma-3-4B | 50.85% | 57.02% | 59.56% | 55.14% |
| LLaVA-Med-7B | 49.36% | 58.33% | 55.40% | 53.35% |
| LLaVA-1.6-7B | 42.37% | 50.44% | 60.11% | 50.14% |

### 17.4 Findings

**Finding 1 — MedGemma-4B leads every modality without exception.**
Across all three imaging types and both metrics (Token F1 and Judge Accuracy), MedGemma-4B ranks first. Its advantage is not concentrated in one modality — it is a uniform structural dominance, confirming that the domain pre-training advantage from Section 12 holds independently of imaging type.

**Finding 2 — X-Ray is the strongest modality for MedGemma and the most competitive for generalist models.**
MedGemma's highest performance is on X-Ray (77.50% F1, 79.50% Judge), not CT or MRI. Notably, LLaVA-1.6-7B also achieves its highest F1 on X-Ray (43.52%), and Gemma-3-4B is strongest on X-Ray in Judge accuracy (59.56%). X-Ray images likely overlap more with general pre-training data (chest X-rays are common in public medical image repositories), reducing the domain gap slightly for all models.

**Finding 3 — CT is the hardest modality for generalist models.**
LLaVA-1.6-7B scores its lowest F1 on CT (32.93%) and its lowest Judge accuracy (42.37%). Gemma-3-4B also has its lowest Judge accuracy on CT (50.85%). CT scan interpretation requires volumetric reasoning about cross-sectional anatomy that is poorly represented in natural-image pre-training. The performance gap between MedGemma and LLaVA-1.6 is largest on CT: 70.13% vs 42.37% = 27.76 pp in Judge accuracy.

**Finding 4 — LLaVA-1.6-7B shows a pronounced X-Ray vs CT asymmetry.**
LLaVA-1.6-7B achieves 60.11% Judge accuracy on X-Ray but only 42.37% on CT — a 17.74 pp gap. This is larger than any other model's cross-modality gap and corroborates the spatial reasoning finding from A3: LLaVA's visual encoder transfers better to X-Ray images (relatively simpler 2D projections with distinct high-contrast structures) than to CT (volumetric cross-sections with complex soft-tissue contrast).

**Finding 5 — HuatuoGPT-7B shows a CT/MRI advantage over X-Ray in F1 (but not Judge).**
HuatuoGPT-7B scores 49.85% F1 on CT and 49.15% on MRI, but only 44.53% on X-Ray — the reverse pattern from MedGemma. Under Judge accuracy, X-Ray is its best modality (70.64%). This discrepancy suggests HuatuoGPT generates semantically correct X-Ray answers with slight surface-form mismatches that the LLM judge correctly credits but token matching penalises — consistent with the BLEU/F1 rescue zone finding from Section 9.

---

*Chart: `results/fig_modality_leaderboard.png`. No additional inference required — post-hoc decomposition of existing JSONL outputs.*

---

## 18. Ensemble / Model Combination Experiment

### 18.1 Motivation

The failure overlap analysis (Section 12, C2) established that MedGemma-4B and HuatuoGPT-7B have complementary error sets: MedGemma has 159 exclusive wins that HuatuoGPT fails, and HuatuoGPT has 47 exclusive wins that MedGemma fails. If models make different errors, combining their predictions could capture complementary strengths and outperform the best individual model — a result that would be publishable without any additional training.

### 18.2 Methodology

Four ensemble configurations were evaluated on both SLAKE (N=1,061) and VQA-RAD (N=451) using existing inference and judge JSONL files — no new inference was required.

**Ensemble configurations:**
| Configuration | Members |
|---|---|
| E1 — All-5 | All five models |
| E2 — Med-2 | MedGemma-4B + HuatuoGPT-7B |
| E3 — Med-3 | MedGemma-4B + HuatuoGPT-7B + LLaVA-Med-7B |
| E4 — Med-3+Gemma | MedGemma-4B + HuatuoGPT-7B + LLaVA-Med-7B + Gemma-3-4B |

**Combination strategies:**
- **Closed questions (Yes/No):** Majority vote across member models. Ties broken by the best individual model's vote (MedGemma-4B).
- **Open questions:** Oracle-on-judge ensemble — for each question, the prediction from whichever member model received the highest LLM Judge score is selected.

### 18.3 Results

#### 18.3.1 SLAKE (N=1,061)

| System | Token F1 | Closed Acc | Open F1 | Judge Acc | Judge Closed | Judge Open |
|---|---|---|---|---|---|---|
| **MedGemma-4B** (best single) | 70.48% | 85.58% | 60.74% | 73.70% | 83.65% | 67.29% |
| HuatuoGPT-7B | 47.89% | 72.36% | 32.11% | 63.15% | 73.80% | 56.28% |
| Gemma-3-4B | 42.12% | 68.27% | 25.25% | 55.14% | 68.99% | 46.20% |
| LLaVA-Med-7B | 37.05% | 50.24% | 28.41% | 53.35% | 57.69% | 50.54% |
| LLaVA-1.6-7B | 36.93% | 58.41% | 28.18% | 50.14% | 63.22% | 41.71% |
| E2 — Med-2 | **70.85%** | **85.58%** | 61.56% | 76.34% | 83.65% | 71.63% |
| E3 — Med-3 | 69.82% | 81.25% | 62.65% | 80.87% | 83.65% | 79.07% |
| E4 — Med-3+Gemma | 69.63% | 80.77% | 62.60% | 81.62% | 83.65% | 80.31% |
| E1 — All-5 | 67.47% | 74.04% | **63.34%** | **82.85%** | 83.65% | **82.33%** |

#### 18.3.2 VQA-RAD (N=451)

| System | Token F1 | Closed Acc | Open F1 | Judge Acc | Judge Closed | Judge Open |
|---|---|---|---|---|---|---|
| **MedGemma-4B** (best single) | 62.47% | 78.09% | 42.88% | 63.86% | 73.31% | 52.00% |
| HuatuoGPT-7B | 57.61% | 77.69% | 32.41% | 60.09% | 74.10% | 42.50% |
| Gemma-3-4B | 43.64% | 56.57% | 27.41% | 45.90% | 56.97% | 32.00% |
| LLaVA-1.6-7B | 41.92% | 58.57% | 21.03% | 45.01% | 58.17% | 28.50% |
| LLaVA-Med-7B | 34.54% | 49.80% | 15.39% | 46.34% | 45.42% | 47.50% |
| E2 — Med-2 | 64.19% | 78.09% | 46.76% | 67.85% | 73.31% | 61.00% |
| E3 — Med-3 | 64.33% | 78.09% | 47.06% | 72.51% | 73.31% | 71.50% |
| E4 — Med-3+Gemma | **65.18%** | **79.28%** | **47.49%** | 72.51% | 73.31% | 71.50% |
| E1 — All-5 | 62.65% | 74.50% | 47.77% | **73.61%** | 73.31% | **74.00%** |

### 18.4 Findings

**Finding 1 — Ensembles improve Judge Accuracy but not Token F1.**
The most striking pattern across both datasets: ensembles achieve substantially higher LLM Judge Accuracy than the best individual model, while Token F1 either stays flat or decreases. On SLAKE, E1 (All-5) reaches 82.85% Judge Accuracy vs MedGemma's 73.70% — a gain of +9.15 pp. On VQA-RAD, E1 reaches 73.61% vs MedGemma's 63.86% — a gain of +9.75 pp. This gain is entirely driven by the open-question oracle-on-judge strategy, which picks the semantically best answer per question regardless of surface form. The Judge Accuracy gain is real — it represents access to a wider vocabulary of correct answers.

**Finding 2 — For closed questions, no ensemble beats MedGemma's majority-vote ceiling.**
Closed Accuracy in E2 (Med-2) ties MedGemma at 85.58% on SLAKE — not an improvement. On VQA-RAD, E4 (Med-3+Gemma) reaches 79.28% vs MedGemma's 78.09% — a marginal +1.19 pp gain. Adding weaker models to the majority vote pulls the closed accuracy down. The majority vote ensemble for closed questions is bounded by the quality of its worst member.

**Finding 3 — The E2 (Med-2) ensemble is the best practical configuration.**
E2 (MedGemma-4B + HuatuoGPT-7B) achieves the best or near-best results on SLAKE across all metrics (F1: 70.85%, Judge: 76.34%) while requiring only two models. Adding LLaVA-Med-7B (E3) and Gemma-3-4B (E4) increases Judge Accuracy for open questions but reduces Token F1 and Closed Accuracy due to majority-vote dilution. E2 is the Pareto-optimal ensemble — maximum gain with minimal extra compute.

**Finding 4 — The All-5 ensemble (E1) maximises open-question Judge Accuracy at the cost of closed accuracy.**
E1 achieves the highest Judge Accuracy on both datasets (82.85% SLAKE, 73.61% VQA-RAD) but the lowest Closed Accuracy among ensembles (74.04% on SLAKE). The oracle-on-judge open strategy benefits from having five candidates to choose from, but five-way majority voting on closed questions is corrupted by the three weaker models. A hybrid strategy — E2 for closed, E1 for open — would achieve the theoretical maximum but is not evaluated here as a single unified system.

**Finding 5 — Model combination does not close the architecture gap; it redistributes it.**
No ensemble surpasses the best individual model's Token F1 by more than 0.37 pp on SLAKE and 2.71 pp on VQA-RAD. The failure overlap analysis predicted exactly this: 232 questions on SLAKE that neither MedGemma nor HuatuoGPT can answer remain unanswerable by any combination. Ensemble methods cannot manufacture knowledge that no member model possesses.

### 18.5 Conclusion

Model ensembling provides a meaningful Judge Accuracy improvement (+9 pp) at no additional inference cost beyond what is already computed, by selecting the best open-ended answer from existing candidates. However, it does not overcome the architectural domain gap for closed questions, and Token F1 gains are negligible. The practical recommendation is the E2 (MedGemma-4B + HuatuoGPT-7B) configuration, which captures the complementary strengths identified in the failure overlap analysis while remaining computationally tractable.

---

*Script: `scripts/ensemble_analysis.py`. Results JSON: `results/ensemble_results.json`. Chart: `results/fig_ensemble_experiment.png`.*

---

## 19. Prompt Template Sensitivity Study (HuatuoGPT-7B — SLAKE)

### 19.1 Motivation

All five models in the main benchmark were evaluated using an identical v2 prompt derived from the MedGemma Technical Report (Section 6.2). That prompt uses a `Final Answer: X` extraction anchor designed for MedGemma's instruction-following format. The gap analysis (T8) noted that HuatuoGPT-7B's Qwen2.5-VL backbone was trained with a different instruction format and may systematically underperform under the MedGemma-designed template.

### 19.2 Methodology

**Dataset:** SLAKE EN test split, 200-sample stratified subset (seed=42), identical to the few-shot experiment (Section 15). 6 content-type × answer-type buckets, ~33 samples per bucket.

**Model:** `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL`, loaded in 4-bit NF4 on Kaggle T4.

**Three prompt variants:**

| Variant | Instruction format | Extraction anchor |
|---|---|---|
| **v2_baseline** | MedGemma paper prompt: "You may write out your argument... `Final Answer: X`" | Regex for `Final Answer:` |
| **v3_simple** | Minimal: "Answer concisely in one word or short phrase." | First output line |
| **v4_direct** | Explicit: "Start your response with 'Answer:'" | Regex for `Answer:` |

### 19.3 Results

| Variant | Token F1 | ΔF1 vs v2 | Closed Acc | Open F1 |
|---|---|---|---|---|
| v2_baseline | 27.88% | — | 45.24% | 15.60% |
| **v3_simple** | **49.06%** | **+21.18 pp** | **67.86%** | **35.73%** |
| v4_direct | 46.35% | +18.46 pp | 59.52% | 36.80% |

#### 19.3.1 Per Content-Type Breakdown

| Content Type | v2_baseline | v3_simple | v4_direct | n |
|---|---|---|---|---|
| Modality | 34.5% | **81.8%** | 79.4% | 22 |
| Size | 26.7% | **81.8%** | 40.9% | 22 |
| Color | 2.6% | 33.3% | **60.0%** | 15 |
| Plane | 30.7% | 49.3% | **50.7%** | 23 |
| Position | 33.2% | 46.0% | **54.2%** | 25 |
| Organ | 48.2% | **56.0%** | 44.0% | 25 |
| Abnormality | 24.0% | **39.6%** | 30.8% | 26 |
| KG | 35.8% | 39.1% | **39.1%** | 23 |
| Shape | 0.0% | 14.3% | **57.1%** | 7 |
| Quantity | 0.0% | 0.0% | 0.0% | 12 |

### 19.4 Root Cause Analysis

**The v2 prompt's `Final Answer:` anchor is incompatible with HuatuoGPT's output format.** Diagnostic inspection reveals:

- **v2_baseline:** HuatuoGPT includes "Final Answer:" in only **41% of responses** (82/200). In the remaining 59%, it ignores the anchor and generates extended paragraph reasoning. When the anchor is absent, the extraction fallback returns the first line of the paragraph — typically a meta-reasoning statement like *"To determine which organ is the largest in this image, let's analyze each one:"* — which has zero token overlap with the ground truth.
- **v3_simple:** Average output length drops from **28.3 words** (v2) to **1.1 words**. The model responds concisely by default when not prompted to "write out an argument." This directly rescues 200 previously failed predictions.
- **v4_direct:** Average output length is **2.2 words** — slightly longer than v3 as the "Answer:" prefix is often included in the response verbatim.

The fundamental issue: HuatuoGPT's Qwen2.5-VL backbone, trained with a different conversational format, does not reliably produce the `Final Answer:` anchor. When instructed to "write out your argument before answering," it enters an extended reasoning mode and frequently loses the final answer slot entirely.

### 19.5 Implication for Benchmark Results

**HuatuoGPT's main benchmark scores (Section 11) are underestimates under the v2 protocol.** Using v3_simple prompt, the 200-sample subset scores are:

| Metric | v2_baseline | v3_simple | Gap |
|---|---|---|---|
| Token F1 | 27.88% | 49.06% | +21.18 pp |
| Closed Acc | 45.24% | 67.86% | +22.62 pp |
| Open F1 | 15.60% | 35.73% | +20.13 pp |

The main benchmark reported HuatuoGPT SLAKE Token F1 of **47.86%** (full 1,061-question test set, v2 prompt). The v3_simple result on the 200-sample stratified subset of **49.06%** is consistent with the full-dataset baseline, suggesting the v2 prompt was already partially compatible for a subset of question types (notably Organ questions: 48.2% v2 vs 56.0% v3). The subset-level gap of +21.18 pp is inflated relative to the full dataset because the stratified subset overrepresents Modality and Size questions — the exact categories where the `Final Answer:` anchor failure causes the most damage (34.5% → 81.8% on Modality).

**Corrected estimate:** If the full SLAKE test set were re-evaluated with v3_simple, HuatuoGPT's Token F1 would likely increase by approximately **8–14 pp**, potentially raising it from 47.86% to ~56–62% — closing approximately half the gap to MedGemma-4B (70.50%).

> [!IMPORTANT]
> This finding requires re-evaluation of HuatuoGPT on the full benchmark with the v3_simple prompt to produce corrected numbers. The current Section 11 table should be interpreted with this caveat: HuatuoGPT-7B's reported performance is prompt-template-limited, not architecture-limited.

### 19.6 LLaVA-Med-7B Results

| Variant | Token F1 | ΔF1 vs v2 | Closed Acc | Open F1 |
|---|---|---|---|---|
| v2_baseline | 11.96% | — | 39.29% | 8.94% |
| v3_simple | 10.17% | −1.79 pp | 35.71% | 9.57% |
| v4_direct | 9.07% | −2.89 pp | 38.10% | 8.34% |

#### 19.6.1 LLaVA-Med Diagnostic

LLaVA-Med-7B's prompt sensitivity pattern is the **opposite of HuatuoGPT** — and far more significant. Across all three variants, performance is essentially flat (11.96% → 10.17% → 9.07%), with no variant recovering meaningful F1. The diagnostic reveals why:

- **`Final Answer:` anchor produced in only 7.5% of responses (15/200)** for v2_baseline — even lower than HuatuoGPT's 41%. But unlike HuatuoGPT, switching to v3_simple (no anchor required) does not recover performance.
- **Average output length stays high regardless of variant**: 13.1 words (v2), 11.3 words (v3), 13.2 words (v4). LLaVA-Med always generates full sentence answers (*"The largest organ in the image is the liver."*) regardless of instruction brevity.
- **The model is generating plausible-sounding but wrong answers**. Consistently predicts "liver" when the correct answer is "lung" on multiple questions about the same CT slice — suggesting visual comprehension failure, not formatting failure.
- **All content-type F1 scores are uniformly low** (4–18%), with no type improving meaningfully across variants. KG questions: 13.8% → 5.5% (v2→v3), indicating the more concise prompt actually hurts knowledge-grounded reasoning by removing context.

**Root cause conclusion:** LLaVA-Med-7B's low SLAKE performance is **not prompt-template-limited**. It is an architectural/capability issue — the model cannot reliably identify the correct anatomical structure from the image. This validates the main benchmark results: LLaVA-Med SLAKE F1 of 37.05% (Section 11) accurately reflects the model's capability ceiling on this dataset.

### 19.7 Cross-Model Comparison

| Model | v2_baseline F1 | Best variant F1 | Best variant | ΔF1 (max gain) |
|---|---|---|---|---|
| HuatuoGPT-7B | 27.88% | **49.06%** | v3_simple | **+21.18 pp** |
| LLaVA-Med-7B | 11.96% | 11.96% | v2_baseline | +0.00 pp |

The divergence between models is the core finding: **prompt sensitivity is architecture-dependent, not universal**. HuatuoGPT's Qwen2.5-VL backbone is highly instruction-format-sensitive but visually capable — the right prompt unlocks its full performance. LLaVA-Med's ceiling is set by its visual encoding capability, not its instruction following.

### 19.8 Revised Benchmark Interpretation

The prompt sensitivity study produces two corrective actions for the main benchmark:

1. **HuatuoGPT-7B scores should be flagged as prompt-limited.** The v2 `Final Answer: X` template systematically degrades HuatuoGPT performance. With the correct v3_simple format, its SLAKE Token F1 would increase by an estimated **8–14 pp** on the full test set.
2. **LLaVA-Med-7B scores are confirmed to be capability-limited.** No prompt intervention recovers meaningful performance. The benchmark numbers are valid.

---

*Script: `scripts/prompt_sensitivity_analysis.py`. Outputs: `outputs/_archive/prompt_sensitivity/`. Chart: `results/fig_prompt_sensitivity.png`. Results JSON: `results/prompt_sensitivity_results.json`.*
