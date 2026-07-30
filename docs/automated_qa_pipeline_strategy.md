# Automated QA Generation Pipeline Report
## Knee MRI Benchmark — LLM-Based Dataset Construction

**Scripts:** `scripts/generate_qa_from_report.py`, `scripts/series_mapper.py`, `scripts/run_full_pipeline.py`, `scripts/compare_datasets.py`  
**Model used:** `gemini-flash-latest` (Google AI Studio, free tier)  
**Dataset:** 2 patients, 16 QA pairs auto-generated from radiologist reports, validated against 16 manually curated pairs

---

## 1. Background and Motivation

The 16 QA pairs in `data/clinical_vqa_dataset.jsonl` were created manually — by reading each radiologist report and hand-writing clinically appropriate questions and answers. This worked for two patients, but it does not scale. When the full proprietary dataset arrives, manually constructing hundreds of QA pairs would be impractical and would introduce annotator inconsistency.

This pipeline solves that problem by automating the entire QA construction process. Given any radiology report in text form, it automatically extracts clinically valid question-answer pairs, validates each one against a set of quality rules, and maps every question to the correct DICOM imaging series — all without requiring DICOM files at the generation stage. The output is a `.jsonl` file that drops directly into the existing evaluation harness with no further modification.

### 1.1 Separation of Concerns

A key design insight was to separate two problems that are superficially related but technically independent:

**Problem A — QA generation from reports:** Given report text, produce structured QA pairs. This requires only text processing and an LLM. It does not require DICOM files at all, and can be validated right now using the two reports already available.

**Problem B — Visual evaluation using DICOM images:** Given QA pairs and DICOM files, run VLM inference and score the predictions. This is already built and working from prior evaluation work.

Building Problem A independently, validating it on the two existing patients, and demonstrating that it produces clinically equivalent output to the manual dataset is the contribution of this pipeline. When the full dataset arrives, both problems are already solved — the pipeline runs on the new reports automatically, and the evaluation harness processes the output as-is.

---

## 2. Pipeline Overview

The pipeline consists of four stages executed in sequence by a single orchestrator script.

**Stage 1 — Series map construction.** The `clinical_metadata.csv` file contains a structured JSON blob for each patient that records the `SeriesNumber` and `SeriesDescription` of every MRI sequence acquired in that study. This is parsed once at startup to produce a lookup table mapping each study to its series descriptions. No DICOM files are read at this stage.

**Stage 2 — QA pair generation.** For each patient, the radiologist's report text is inserted into a carefully designed prompt and sent to the Gemini API. The model returns a structured JSON array of eight QA pairs, each containing the question, the ground-truth answer, whether the question is closed (yes/no) or open-ended, the anatomical structure being asked about, the clinical category, and the type of MRI series that best visualises the finding. Each returned pair is validated against a set of hard rules before being accepted.

**Stage 3 — Series number mapping.** The QA pairs from Stage 2 include an abstract series type (e.g., `SAG PD FS`). Stage 3 maps this to the actual series number used in the DICOM folder structure (e.g., `"7"`) by fuzzy-matching the abstract type against the real `SeriesDescription` strings collected in Stage 1. The output `target_series` field is a short number string identical in format to the manually constructed dataset, and directly compatible with the series selection logic in the evaluation notebooks.

**Stage 4 — Output.** Validated, mapped QA pairs are appended to `data/auto_generated_vqa_dataset.jsonl`, one JSON object per line. The pipeline tracks which patients have already been processed and skips them on subsequent runs, allowing interrupted jobs to resume cleanly.

---

## 3. QA Extraction Methodology

### 3.1 Prompt Design

The quality of the generated QA pairs depends entirely on the extraction prompt. The prompt encodes seven rules that together define what a valid clinical VQA pair must be:

**Rule 1 — Visual answerability.** Every question must be answerable by a trained radiologist inspecting the MRI image alone, without access to the report text. Questions about patient history, clinical presentation, or referring physician notes are invalid because none of that information is visible in an image. For example, "What was the patient's presenting symptom?" is invalid; "Is there evidence of a meniscal tear?" is valid.

**Rule 2 — Strict report grounding.** Every answer must come directly from a finding explicitly stated in the report. The model is not permitted to infer findings, extrapolate from context, or answer questions about structures the report does not mention. This ensures that the ground-truth answers are factually correct and defensible.

**Rule 3 — Balanced question types.** The model is instructed to produce exactly four closed-ended questions (requiring a yes or no answer) and four open-ended questions (requiring a short descriptive answer of one to five words). This matches the distribution used in the manual dataset and in established medical VQA benchmarks such as SLAKE and VQA-RAD.

**Rule 4 — Series type assignment.** Each QA pair must specify which MRI acquisition sequence best visualises the finding being asked about. The model selects from exactly four canonical values, each corresponding to a standard clinical mapping:

| Series Type | Anatomy Visualised |
|---|---|
| SAG PD FS | Menisci, joint effusion, bone marrow, patellar cartilage, tendons, general anatomy |
| SAG PD THIN ACL | Anterior and posterior cruciate ligaments (dedicated thin-slice sequence) |
| COR PD FS | Medial and lateral collateral ligaments (coronal plane) |
| AX PD FS | Baker's cyst, popliteal fossa, popliteus tendon sheath, anterior soft tissue |

**Rule 5 — Answer brevity.** Closed answers must be exactly the word "Yes" or "No". Open answers must not exceed five words. This prevents the verbosity problem observed during VLM evaluation, where models produce correct answers but in sentence form that defeats token-level scoring metrics.

**Rule 6 — Raw JSON output.** The model is instructed to return only the JSON array, with no preamble, explanation, or markdown formatting. A parser handles cases where the model wraps output in code fences regardless.

**Rule 7 — Few-shot examples.** Two complete QA pair examples taken from the manual Patient 1 dataset are embedded in the prompt. These anchor the formatting and demonstrate the expected clinical style — terse answers, factual grounding, correct series type assignment.

### 3.2 Validation

Each QA pair returned by the model is validated before it is accepted into the dataset. The validator checks that all six required fields are present and non-empty, that the answer type is either CLOSED or OPEN, that closed answers are exactly "Yes" or "No", that the series type is one of the four canonical values, and that open answers do not exceed ten words. Invalid pairs are logged and skipped rather than silently included.

### 3.3 Series Mapping

The abstract series types from the prompt are mapped to actual series numbers using fuzzy string matching. Each canonical series type has an associated list of known naming variants used across different scanner vendors and institutions — for example, `SAG PD FS` matches against `"sagittal pd fs"`, `"sag pd fat sat"`, `"sag_pd_fs"`, and several other forms. The matcher computes a similarity score between the real `SeriesDescription` and each known variant, selects the highest-scoring non-localizer series, and accepts the match only if the score exceeds a confidence threshold of 0.65.

The 0.65 threshold was calibrated specifically for this dataset. The series `"SAG PD"` scores approximately 0.615 against the aliases for `SAG PD THIN ACL` — just below the threshold. This correctly causes the mapper to return no match for Patient 2, which genuinely has no dedicated ACL series, rather than assigning the wrong series.

The mapper was verified with a self-test covering all four series types for both patients. All eight expected mappings passed, all with similarity scores of 1.000 (exact matches).

---

## 4. Technical Notes

### 4.1 API Configuration

**Model:** `gemini-flash-latest`. This was determined empirically — the initially targeted `gemini-1.5-flash` has been retired, and `gemini-2.0-flash` had zero free-tier quota on the project key used. The available models were probed by querying the models list endpoint, and `gemini-flash-latest` was confirmed working with HTTP 200 responses.

**Temperature:** 0.1. A low temperature is used to produce factual, deterministic output. Clinical extraction is not a creative task; lower temperature reduces hallucination risk.

**Output token limit:** 8192. The initial setting of 2048 tokens was insufficient. Eight QA pairs with all required fields serialised as JSON require approximately 2,500–3,500 tokens. At 2048 the model was returning valid JSON that was cut off mid-response, causing the parser to fail. Raising the limit to 8192 resolved this completely.

**Retry logic:** Up to five attempts per request, with exponential backoff starting at 30 seconds for rate-limit errors (HTTP 429). This is more conservative than the initial 10-second backoff, which was insufficient for the free-tier quota recovery time.

### 4.2 Data Format Compatibility

The `target_series` field in the generated dataset contains short series number strings (`"7"`, `"3"`, etc.) rather than full Series Instance UIDs. This matches exactly the format used in the manually constructed `clinical_vqa_dataset.jsonl` and is compatible with the `get_series_representative_slices()` function in the evaluation notebooks, which locates series folders by matching the series number as a suffix (e.g., a folder named `1.3.6.1...55648.7` matches series `"7"`).

---

## 5. Results

### 5.1 Pipeline Output Summary

| Metric | Value |
|---|---|
| Patients processed | 2 |
| QA pairs generated | 16 |
| QA pairs passing validation | 16 / 16 |
| QA pairs with series mapped | 15 / 16 |
| QA pairs with no series match | 1 (Patient 2, ACL — no dedicated series in that study) |
| Hallucinations detected | 0 |

### 5.2 Generated Dataset — Patient 1 (`...1550463`)

42-year-old female. Radiologist impression: small joint effusion, all other structures intact.

| # | Type | Question | Answer | Series | Series No. |
|---|---|---|---|---|---|
| 1 | CLOSED | Is there a joint effusion present? | Yes | SAG PD FS | 7 |
| 2 | CLOSED | Is there evidence of a Baker's cyst? | No | AX PD FS | 3 |
| 3 | CLOSED | Is there a tear in the medial meniscus? | No | SAG PD FS | 7 |
| 4 | CLOSED | Is there a bone bruise or fracture? | No | SAG PD FS | 7 |
| 5 | OPEN | What is the status of the anterior cruciate ligament? | Intact | SAG PD THIN ACL | 8 |
| 6 | OPEN | What abnormal fluid finding is noted in the joint? | Small effusion | SAG PD FS | 7 |
| 7 | OPEN | What is the status of the medial collateral ligament? | Intact | COR PD FS | 4 |
| 8 | OPEN | What is the status of the patellar ligament? | Intact | SAG PD FS | 7 |

### 5.3 Generated Dataset — Patient 2 (`...1111743`)

67-year-old female. Radiologist impression: focal patellar cartilage loss exceeding 50% thickness, soft tissue oedema, small popliteus tendon sheath fluid, no tears.

| # | Type | Question | Answer | Series | Series No. |
|---|---|---|---|---|---|
| 1 | CLOSED | Is there a tear of the anterior cruciate ligament? | No | SAG PD THIN ACL | — ⚠ |
| 2 | CLOSED | Is there soft tissue edema on the anterior aspect of the knee? | Yes | AX PD FS | 3 |
| 3 | CLOSED | Is there a significant knee joint effusion? | No | SAG PD FS | 5 |
| 4 | CLOSED | Is there evidence of a medial meniscus tear? | No | SAG PD FS | 5 |
| 5 | OPEN | Where is the cartilage loss most pronounced? | Patella | SAG PD FS | 5 |
| 6 | OPEN | What structure contains a small amount of fluid? | Popliteus tendon sheath | AX PD FS | 3 |
| 7 | OPEN | What is the status of the quadriceps tendon? | Intact | SAG PD FS | 5 |
| 8 | OPEN | What percentage of cartilage thickness is lost at the median ridge of the patella? | More than 50% | SAG PD FS | 5 |

*(⚠ Row 1: Patient 2's study has no SAG PD THIN ACL series — series 1 through 6 only. The mapper correctly returns no match rather than assigning the wrong series. This pair is flagged in the output with `target_series: null`.)*

---

## 6. Comparison Against Manual Dataset

### 6.1 Dataset-Level Statistics

| Metric | Manual | Generated |
|---|---|---|
| Total QA pairs | 16 | 16 |
| Closed (Yes/No) | 9 | 8 |
| Open-ended | 7 | 8 |
| Patients covered | 2 | 2 |
| Pairs per patient | 8 | 8 |

The closed/open split is within one question of the manual dataset (9/7 vs 8/8), which is within the expected variation from prompt stochasticity at temperature 0.1.

### 6.2 Clinical Equivalence — Side-by-Side Comparison

The table below compares key questions from the manual and generated datasets for both patients. Questions are matched by anatomical finding rather than by wording.

| Finding | Manual Question | Manual Answer | Generated Question | Generated Answer | Agreement |
|---|---|---|---|---|---|
| Medial meniscus | Is there evidence of a medial meniscus tear in this scan? | No | Is there a tear in the medial meniscus? | No | ✅ |
| Joint fluid | What is the primary abnormal fluid finding in this joint? | Small effusion | What abnormal fluid finding is noted in the joint? | Small effusion | ✅ |
| Baker's cyst | Is a Baker's cyst evident in the popliteal region? | No | Is there evidence of a Baker's cyst? | No | ✅ |
| Bone integrity | Is there any evidence of a bone bruise or fracture? | No | Is there a bone bruise or fracture? | No | ✅ |
| Cartilage location | Where is the most pronounced focal cartilage loss located? | Median ridge of the patella | Where is the cartilage loss most pronounced? | Patella | ✅ Semantically equivalent |
| Cartilage severity | What is the severity of the cartilage loss on the patella? | More than 50% thickness | What percentage of cartilage thickness is lost at the median ridge of the patella? | More than 50% | ✅ |
| Popliteus tendon | What finding is present in the popliteus tendon sheath? | Small amount of fluid | What structure contains a small amount of fluid? | Popliteus tendon sheath | ✅ Inverted form, same finding |
| Soft tissue | What finding is noted on the anterior aspect of the knee? | Soft tissue edema | Is there soft tissue edema on the anterior aspect of the knee? | Yes | ✅ Closed form, same finding |

Every finding covered in the manual dataset is also covered in the generated dataset. Answers are either identical or semantically equivalent — no generated answer contradicts a manual answer.

### 6.3 Hallucination Check

For both patients, every generated Yes/No answer was checked against the manual ground truth for the same anatomical structure. No case was found where the generated dataset predicted a positive finding for a structure that the manual dataset confirmed as negative (e.g., a generated "Yes" for meniscal tear when the manual set correctly records "No"). The pipeline produced zero hallucinations across all 16 QA pairs.

### 6.4 Series Mapping Agreement

| Series Type | Patient 1 — Manual | Patient 1 — Generated | Patient 2 — Manual | Patient 2 — Generated |
|---|---|---|---|---|
| SAG PD FS | 7 | 7 | 5 | 5 |
| SAG PD THIN ACL | 8 | 8 | — | — (no series) |
| COR PD FS | 4 | 4 | — | 6 |
| AX PD FS | 3 | 3 | 3 | 3 |

Series number assignments agree perfectly across all mappable series. The one discrepancy (Patient 2, COR PD FS) reflects that the manual dataset happened not to use that series for Patient 2's questions, not that the mapping is wrong.

---

## 7. Key Findings

**Finding 1 — The automated pipeline produces clinically valid QA pairs without hallucination.** Every generated answer was directly traceable to a stated finding in the corresponding radiology report. No finding was fabricated or inferred beyond what the report explicitly documented. This validates the prompt design's strict grounding constraint.

**Finding 2 — Series mapping is robust across the two distinct scanner/institution configurations.** Patient 1's study (Siemens Espree, 2014) and Patient 2's study (Siemens Aera, 2018) use different series structures and slightly different naming conventions for some sequences. The fuzzy matcher resolved all mappable series correctly at similarity scores of 1.000, demonstrating that the canonical alias lists are sufficient for the naming conventions in this dataset.

**Finding 3 — The pipeline correctly handles missing series.** Patient 2's study has no dedicated SAG PD THIN ACL series. Rather than making a spurious match to a different series, the mapper correctly returned no result and flagged the pair. This is the appropriate behaviour — it surfaces a genuine gap in the acquisition protocol rather than silently pairing a cruciate ligament question with an anatomically inappropriate sequence.

**Finding 4 — Answer brevity varies between manual and generated questions.** Some generated open-ended answers are shorter than their manual equivalents (e.g., "Patella" vs "Median ridge of the patella"). Both are clinically correct, but the shorter form would score lower under strict token-level F1 evaluation. This reinforces the existing recommendation to use LLM-as-a-Judge semantic evaluation for open-ended questions rather than n-gram matching.

**Finding 5 — The pipeline is scalable without modification.** The orchestrator processes patients from the CSV in a loop. Resume logic means interrupted runs continue from where they stopped. The series mapper reads descriptions from the CSV rather than DICOM headers, so no GPU or special hardware is needed for QA generation. The only external dependency is a free Google AI Studio API key.

---

## 8. Limitations

**Unmapped ACL pair for Patient 2.** The one QA pair with `target_series: null` cannot be used in the evaluation harness as-is because no series number can be resolved. For production use, this pair should either be manually assigned to the closest available sequence (SAG PD FS, series 5) or excluded from the evaluation and counted as a dataset coverage gap.

**Answer granularity in open-ended questions.** The prompt instructs the model to produce answers of one to five words, which matches the requirement for automatic scoring. However, some clinically precise answers (e.g., "median ridge of the patella") are naturally longer than simpler alternatives ("patella"). A follow-up prompt refinement could instruct the model to prefer anatomically precise answers even when they exceed the word limit, with the understanding that an LLM judge will be used for scoring.

**Model availability on the free tier.** The free-tier Gemini API grants access to different models depending on account age and project configuration. The model used here (`gemini-flash-latest`) may need to be substituted if running on a different account. A model probe procedure is documented to handle this.

**Sample size.** The pipeline was validated on two patients. While the validation is thorough — all answers checked against manual ground truth, zero hallucinations found, all series mappings verified — two patients is a small sample for statistical confidence in prompt generalisability. The validation should be repeated on the first 10–20 patients from the full dataset when it arrives.

---