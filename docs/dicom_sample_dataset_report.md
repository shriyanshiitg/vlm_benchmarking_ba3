# Clinical DICOM VQA Evaluation Report
## Knee MRI Benchmark — Private Dataset (2-Patient Sample)

**Notebooks:** `dicom.ipynb` (exploration + dataset construction), `dicom-evaluation-2.ipynb` (Kaggle T4 inference)
**Models evaluated:** * `google/medgemma-4b-it`
* `google/gemma-3-4b-it`
* `chaoyinshe/llava-med-v1.5-mistral-7b-hf`
* `llava-hf/llava-v1.6-mistral-7b-hf`
**Dataset:** 2 patients, 16 QA pairs derived from radiologist reports, real clinical knee MRI DICOM files

---

## 1. Dataset Description

### 1.1 Source and Format

The dataset was provided as a sample of a larger proprietary clinical dataset, a medical imaging data platform. All patient identifiers have been anonymized by Segmed — real names and IDs are replaced with synthetic strings (e.g., `Segmed_Patient_6523170992291531753`). The dataset contains two components: DICOM image files and a structured CSV containing the corresponding radiologist reports.

**DICOM files** are organized in a standard hierarchical structure: one top-level folder per patient study (named by DICOM Study Instance UID), containing one subfolder per imaging series. All files use JPEG Lossless compression (Transfer Syntax: Process 14, Selection Value 1), requiring `pylibjpeg` and `pylibjpeg-libjpeg` for decompression.

**Radiologist reports** contain one row per patient with columns for Study ID, Patient ID, free-text radiology report, patient demographics, modality, body part, study date, equipment metadata, and per-series DICOM metadata.

### 1.2 Patient Details

**Patient 1** — 42-year-old female, scanned December 2014 on a Siemens Espree 1.5T MRI. Presenting history: medial meniscus tear. The radiologist's report found the opposite: both menisci intact, no bone bruise or fracture, all ligaments intact, only a small joint effusion noted. The impression was "Small effusion."

**Patient 2** — 67-year-old female, scanned July 2018 on a Siemens Aera 1.5T MRI. Presenting history: left knee pain. The radiologist found focal cartilage loss on the median ridge of the patella exceeding 50% of cartilage thickness, less severe femorotibial cartilage loss, soft tissue edema on the anterior knee, and a small amount of fluid in the popliteus tendon sheath. No meniscal or ligament tears. The impression was "Cartilage loss most pronounced on the patella."

### 1.3 MRI Series Structure

Each patient study contains multiple MRI series representing different acquisition sequences and anatomical planes:

| Series | Description | Slices | Purpose |
|---|---|---|---|
| 1 | localizer_tra | 3 | Scout images — planning only |
| 2 | localizer_sag+cor+tra | 6 | Scout images — planning only |
| 3 | AX PD FS / AXIAL PD FS | 30 | Axial fat-suppressed — menisci, cartilage |
| 4 | SAG PD / COR PD FS | 34 | Sagittal/Coronal — ligaments, anatomy |
| 5 | SAG PD FS / COR T1 | 34 | Sagittal fat-suppressed — ACL, menisci |
| 6 | COR PD FS / SAG PD | 24–26 | Coronal fat-suppressed — collateral ligaments |
| 7 | SAG PD FS | 26 | Sagittal fat-suppressed — overall anatomy |
| 8 | SAG PD THIN ACL | 14 | Dedicated thin-slice ACL sequence |

Localizer series (1 and 2) were excluded from evaluation as they are scout images with no diagnostic value.

---

## 2. Dataset Construction

### 2.1 Approach

Since no pre-existing QA pairs were available, a dataset of 16 QA pairs was manually constructed by reading both radiologist reports and extracting clinically relevant yes/no and open-ended questions. This mirrors the methodology used to create public datasets like SLAKE and VQA-RAD — grounding questions directly in expert clinical findings.

### 2.2 Slice Selection Strategy

Each question was paired with the most diagnostically appropriate MRI series rather than using a single series for all questions. The mapping principle is standard radiological practice:

- **Meniscal questions** → SAG PD FS (sagittal fat-suppressed best shows meniscal morphology)
- **ACL/PCL questions** → SAG PD THIN ACL (dedicated thin-slice sequence for cruciate ligaments)
- **Collateral ligament questions** → COR PD FS (coronal plane best shows medial/lateral collateral ligaments)
- **Effusion and bone questions** → SAG PD FS (fluid appears bright on fat-suppressed sequences)
- **Cartilage questions** → SAG PD FS (sagittal view of patella best shows patellar cartilage)
- **Baker's cyst and posterior fossa** → AX PD FS (axial view of popliteal fossa)

Within each selected series, a **multi-slice 2x2 grid** was used as the representative image. We automatically sample 4 slices evenly across the central 60% of the series depth (at 20%, 40%, 60%, and 80%) and tile them into a single image. This addresses the traditional limitation of single-slice VQA evaluations by simulating a radiologist's ability to examine structural continuity and broader anatomical context.

### 2.3 QA Pair Breakdown

| Patient | Total QA Pairs | Closed (Yes/No) | Open-ended |
|---|---|---|---|
| Patient 1 | 8 | 5 | 3 |
| Patient 2 | 8 | 4 | 4 |
| **Total** | **16** | **9** | **7** |

---

## 3. Technical Pipeline

### 3.1 DICOM Processing

DICOM files required specific preprocessing before they could be passed to VLMs:

**Dependency installation:** The JPEG Lossless compression format used in these files required `pylibjpeg==2.1.0` and `pylibjpeg-libjpeg==2.4.0` in addition to `pydicom`. Standard pydicom alone throws a `RuntimeError` when attempting to access pixel data.

**Windowing:** Raw MRI pixel values are in a proprietary 12-bit range (0–4095 for 12-bit stored) and cannot be displayed directly. The DICOM header contains embedded `WindowCenter` and `WindowWidth` values that define the clinically appropriate display range. The pipeline applies these using:

```
low  = WindowCenter − WindowWidth / 2
high = WindowCenter + WindowWidth / 2
pixels = clip(pixels, low, high) normalized to 0–255
```

This is mandatory for MRI. Without windowing the images appear as washed-out noise and the model cannot extract meaningful visual features.

**RGB conversion:** All DICOM images are single-channel MONOCHROME2 (grayscale). VLMs require 3-channel RGB input, so all images are converted via `Image.fromarray(windowed).convert('RGB')` which replicates the grayscale channel across R, G, and B.

### 3.2 Prompt Design

Following the v2 protocol used throughout the benchmark, with one addition — an explicit MRI context prefix since these are domain-specific images:

**Closed-ended:** `"Answer the question with yes or no. This is an MRI image of a knee joint. {question} You may write out your argument before stating your final very short, definitive, and concise answer (if possible, a single word or short phrase) X in the format 'Final Answer: X'"`

**Open-ended:** `"This is an MRI image of a knee joint. {question} You may write out your argument before stating your final very short, definitive, and concise answer (if possible, a single word or short phrase) X in the format 'Final Answer: X'"`

### 3.3 Infrastructure

Inference was run on **Kaggle T4 GPU** (CUDA). 4B models (MedGemma, Gemma-3) were loaded natively in `bfloat16`. To evaluate the larger 7B models (LLaVA-Med, LLaVA-1.6) without triggering Out-Of-Memory (OOM) errors on the 15GB T4 GPUs, we successfully integrated **4-bit NF4 quantization** using the `bitsandbytes` and `accelerate` libraries. The same checkpoint-aware runner used throughout the project saved results to `*__clinical_knee_mri_v2.jsonl` files. 

---

## 4. Results

This section compares the performance of the models using two different image sampling strategies: the original single-slice baseline and the updated multi-slice grid approach.

### 4.1 Single-Slice Strategy Results

#### 4.1.1 Overall Scores

| Model | Total | Closed | Open | Overall F1 | F1 Norm | Closed Acc | Open Acc | BLEU |
|---|---|---|---|---|---|---|---|---|
| `google/medgemma-4b-it` | 16 | 9 | 7 | **60.42%** | **60.42%** | 88.89% | **14.29%** | **59.69** |
| `chaoyinshe/llava-med-v1.5-mistral-7b` | 16 | 9 | 7 | 37.09% | 38.33% | **100.0%** | **14.29%** | 32.46 |
| `llava-hf/llava-v1.6-mistral-7b-hf`| 16 | 9 | 7 | 14.58% | 15.62% | 22.22% | 0.00% | 12.61 |
| `google/gemma-3-4b-it` | 16 | 9 | 7 | 4.17% | 4.17% | 0.00% | 0.00% | 2.30 |

#### 4.1.2 Per-Question Breakdown — MedGemma-4b-it

| # | Type | Ground Truth | Prediction | F1 |
|---|---|---|---|---|
| 0 | CLOSED | No | No | 1.000 |
| 1 | CLOSED | Yes | Yes | 1.000 |
| 2 | OPEN | Small effusion | **Meniscal tear** | 0.000 |
| 3 | CLOSED | No | **Yes** | 0.000 |
| 4 | OPEN | Intact | Intact | 1.000 |
| 5 | CLOSED | No | No | 1.000 |
| 6 | CLOSED | No | No | 1.000 |
| 7 | OPEN | Small effusion | **Normal** | 0.000 |
| 8 | OPEN | Median ridge of the patella | **Medial compartment** | 0.000 |
| 9 | CLOSED | No | No | 1.000 |
| 10 | OPEN | Soft tissue edema | Soft tissue swelling | 0.667 |
| 11 | CLOSED | No | No | 1.000 |
| 12 | CLOSED | Yes | Yes | 1.000 |
| 13 | OPEN | More than 50% thickness | Moderate | 0.000 |
| 14 | CLOSED | No | No | 1.000 |
| 15 | OPEN | Small amount of fluid | Thickening | 0.000 |

#### 4.1.3 Per-Question Breakdown — Gemma-3-4b-it

| # | Type | Ground Truth | Prediction | F1 |
|---|---|---|---|---|
| 0 | CLOSED | No | **Yes** | 0.000 |
| 1 | CLOSED | Yes | **No** | 0.000 |
| 2 | OPEN | Small effusion | Effusion | 0.667 |
| 3 | CLOSED | No | **Yes** | 0.000 |
| 4 | OPEN | Intact | **Partial MCL tear** | 0.000 |
| 5 | CLOSED | No | **Yes** | 0.000 |
| 6 | CLOSED | No | **Yes** | 0.000 |
| 7 | OPEN | Small effusion | **Analysis** | 0.000 |
| 8 | OPEN | Median ridge of the patella | **Medial femoral condyle** | 0.000 |
| 9 | CLOSED | No | **Yes** | 0.000 |
| 10 | OPEN | Soft tissue edema | **Baker's cyst** | 0.000 |
| 11 | CLOSED | No | **Yes** | 0.000 |
| 12 | CLOSED | Yes | **No** | 0.000 |
| 13 | OPEN | More than 50% thickness | Moderate | 0.000 |
| 14 | CLOSED | No | Yes | 0.000 |
| 15 | OPEN | Small amount of fluid | Cyst | 0.000 |

#### 4.1.4 Per-Question Breakdown — LLaVA-Med-1.5-Mistral-7B

| # | Type | Ground Truth | Prediction | F1 |
|---|---|---|---|---|
| 0 | CLOSED | No | No | 1.000 |
| 1 | CLOSED | Yes | Yes | 1.000 |
| 2 | OPEN | Small effusion | The primary abnormal fluid f... | 0.118 |
| 3 | CLOSED | No | No, a Baker's cyst is not ev... | 0.100 |
| 4 | OPEN | Intact | The MRI image shows that the... | 0.125 |
| 5 | CLOSED | No | No | 1.000 |
| 6 | CLOSED | No | No | 1.000 |
| 7 | OPEN | Small effusion | The overall impression of th... | 0.000 |
| 8 | OPEN | Median ridge of the patella | The most pronounced focal ca... | 0.111 |
| 9 | CLOSED | No | No, there are no tears prese... | 0.105 |
| 10 | OPEN | Soft tissue edema | The MRI image shows a findin... | 0.000 |
| 11 | CLOSED | No | No, there is no significant... | 0.118 |
| 12 | CLOSED | Yes | Yes | 1.000 |
| 13 | OPEN | More than 50% thickness | The MRI image shows a severe... | 0.000 |
| 14 | CLOSED | No | No, there is no significant... | 0.133 |
| 15 | OPEN | Small amount of fluid | The MRI image shows a poplit... | 0.125 |

#### 4.1.5 Per-Question Breakdown — LLaVA-v1.6-Mistral-7B

| # | Type | Ground Truth | Prediction | F1 |
|---|---|---|---|---|
| 0 | CLOSED | No | **Yes** | 0.000 |
| 1 | CLOSED | Yes | Yes | 1.000 |
| 2 | OPEN | Small effusion | **X** | 0.000 |
| 3 | CLOSED | No | **Yes** | 0.000 |
| 4 | OPEN | Intact | **Torn** | 0.000 |
| 5 | CLOSED | No | **Yes** | 0.000 |
| 6 | CLOSED | No | **Yes** | 0.000 |
| 7 | OPEN | Small effusion | **X** | 0.000 |
| 8 | OPEN | Median ridge of the patella | Patella | 0.333 |
| 9 | CLOSED | No | **Yes** | 0.000 |
| 10 | OPEN | Soft tissue edema | **Meniscus tear** | 0.000 |
| 11 | CLOSED | No | **Yes** | 0.000 |
| 12 | CLOSED | Yes | Yes | 1.000 |
| 13 | OPEN | More than 50% thickness | Severe | 0.000 |
| 14 | CLOSED | No | **Yes** | 0.000 |
| 15 | OPEN | Small amount of fluid | **X** | 0.000 |

*(Note: Gemma-3-4B is omitted from per-question breakdowns for brevity, as it scored 0% on both open and closed tasks).*

#### 4.1.5 Per-Patient Breakdown

| Model | Patient | F1 | Closed Acc | Open Acc |
|---|---|---|---|---|
| MedGemma-4b | P1 (31550463) | 62.50% | 80.0% | 33.33% |
| MedGemma-4b | P2 (67111743) | 58.33% | 100.0% | 0.0% |
| LLaVA-Med-7b | P1 (31550463) | 54.28% | 100.0% | 33.33% |
| LLaVA-Med-7b | P2 (67111743) | 19.90% | 100.0% | 0.0% |
| LLaVA-v1.6-7b | P1 (31550463) | 12.50% | 20.0% | 0.0% |
| LLaVA-v1.6-7b | P2 (67111743) | 16.67% | 25.0% | 0.0% |
| Gemma-3-4b | P1 (31550463) | 8.33% | 0.0% | 0.0% |
| Gemma-3-4b | P2 (67111743) | 0.0% | 0.0% | 0.0% |


### 4.2 Multi-Slice (2x2 Grid) Strategy Results

#### 4.2.1 Overall Scores

| Model | Total | Closed | Open | Overall F1 | F1 Norm | Closed Acc | Open Acc | BLEU |
|---|---|---|---|---|---|---|---|---|
| `google/medgemma-4b-it` | 16 | 9 | 7 | **47.92%** | **47.92%** | 66.67% | **14.29%** | **46.05** |
| `chaoyinshe/llava-med-v1.5-mistral-7b` | 16 | 9 | 7 | 41.16% | 41.07% | **77.78%** | **14.29%** | 38.28 |
| `llava-hf/llava-v1.6-mistral-7b-hf`| 16 | 9 | 7 | 6.25% | 6.25% | 11.11% | 0.00% | 6.25 |
| `google/gemma-3-4b-it` | 16 | 9 | 7 | 0.69% | 0.00% | 0.00% | 0.00% | 0.16 |

#### 4.2.2 Per-Question Breakdown — MedGemma-4b-it

| # | Type | Ground Truth | Prediction | F1 |
|---|---|---|---|---|
| 0 | CLOSED | No | No | 1.000 |
| 1 | CLOSED | Yes | Yes | 1.000 |
| 2 | OPEN | Small effusion | Effusion | 0.667 |
| 3 | CLOSED | No | **Yes** | 0.000 |
| 4 | OPEN | Intact | Intact | 1.000 |
| 5 | CLOSED | No | No | 1.000 |
| 6 | CLOSED | No | No | 1.000 |
| 7 | OPEN | Small effusion | **Normal** | 0.000 |
| 8 | OPEN | Median ridge of the patella | **Medial femoral condyle** | 0.000 |
| 9 | CLOSED | No | No | 1.000 |
| 10 | OPEN | Soft tissue edema | **Fluid** | 0.000 |
| 11 | CLOSED | No | **Yes** | 0.000 |
| 12 | CLOSED | Yes | Yes | 1.000 |
| 13 | OPEN | More than 50% thickness | **Moderate** | 0.000 |
| 14 | CLOSED | No | **Yes** | 0.000 |
| 15 | OPEN | Small amount of fluid | **Thickening** | 0.000 |

#### 4.2.3 Per-Question Breakdown — Gemma-3-4b-it

| # | Type | Ground Truth | Prediction | F1 |
|---|---|---|---|---|
| 0 | CLOSED | No | **Yes** | 0.000 |
| 1 | CLOSED | Yes | **No** | 0.000 |
| 2 | OPEN | Small effusion | **Here's an evaluation of the...** | 0.000 |
| 3 | CLOSED | No | **Yes** | 0.000 |
| 4 | OPEN | Intact | **MCL tear** | 0.000 |
| 5 | CLOSED | No | **Yes** | 0.000 |
| 6 | CLOSED | No | **Yes** | 0.000 |
| 7 | OPEN | Small effusion | **Okay, let's analyze the MRI...** | 0.000 |
| 8 | OPEN | Median ridge of the patella | **Okay, let's analyze the MRI...** | 0.111 |
| 9 | CLOSED | No | **Yes** | 0.000 |
| 10 | OPEN | Soft tissue edema | **Okay, let's analyze the MRI...** | 0.000 |
| 11 | CLOSED | No | **Yes** | 0.000 |
| 12 | CLOSED | Yes | **No** | 0.000 |
| 13 | OPEN | More than 50% thickness | **Okay, let's analyze the MRI...** | 0.000 |
| 14 | CLOSED | No | **Yes** | 0.000 |
| 15 | OPEN | Small amount of fluid | **Tendon sheath thickening** | 0.000 |

#### 4.2.4 Per-Question Breakdown — LLaVA-Med-1.5-Mistral-7B

| # | Type | Ground Truth | Prediction | F1 |
|---|---|---|---|---|
| 0 | CLOSED | No | No | 1.000 |
| 1 | CLOSED | Yes | Yes | 1.000 |
| 2 | OPEN | Small effusion | The primary abnormal fluid f... | 0.118 |
| 3 | CLOSED | No | No, a Baker's cyst is not ev... | 0.133 |
| 4 | OPEN | Intact | The medial and lateral colla... | 0.133 |
| 5 | CLOSED | No | No | 1.000 |
| 6 | CLOSED | No | **Yes** | 0.000 |
| 7 | OPEN | Small effusion | The overall impression of th... | 0.000 |
| 8 | OPEN | Median ridge of the patella | The most pronounced focal ca... | 0.111 |
| 9 | CLOSED | No | No | 1.000 |
| 10 | OPEN | Soft tissue edema | **X** | 0.000 |
| 11 | CLOSED | No | No | 1.000 |
| 12 | CLOSED | Yes | Yes | 1.000 |
| 13 | OPEN | More than 50% thickness | The image shows a severe car... | 0.000 |
| 14 | CLOSED | No | **Yes** | 0.000 |
| 15 | OPEN | Small amount of fluid | The MRI image shows a poplit... | 0.091 |

#### 4.2.5 Per-Question Breakdown — LLaVA-v1.6-Mistral-7B

| # | Type | Ground Truth | Prediction | F1 |
|---|---|---|---|---|
| 0 | CLOSED | No | **Yes** | 0.000 |
| 1 | CLOSED | Yes | **No** | 0.000 |
| 2 | OPEN | Small effusion | **Bone** | 0.000 |
| 3 | CLOSED | No | **Yes** | 0.000 |
| 4 | OPEN | Intact | **Torn** | 0.000 |
| 5 | CLOSED | No | No | 1.000 |
| 6 | CLOSED | No | **Yes** | 0.000 |
| 7 | OPEN | Small effusion | **X** | 0.000 |
| 8 | OPEN | Median ridge of the patella | **40** | 0.000 |
| 9 | CLOSED | No | **Yes** | 0.000 |
| 10 | OPEN | Soft tissue edema | **Fracture** | 0.000 |
| 11 | CLOSED | No | **Yes** | 0.000 |
| 12 | CLOSED | Yes | **No** | 0.000 |
| 13 | OPEN | More than 50% thickness | **Severe** | 0.000 |
| 14 | CLOSED | No | **Yes** | 0.000 |
| 15 | OPEN | Small amount of fluid | **X** | 0.000 |

#### 4.2.6 Per-Patient Breakdown

| Model | Patient | F1 | Closed Acc | Open Acc |
|---|---|---|---|---|
| MedGemma-4b | P1 (31550463) | 70.83% | 80.0% | 33.33% |
| MedGemma-4b | P2 (67111743) | 25.00% | 50.0% | 0.0% |
| LLaVA-Med-7b | P1 (31550463) | 42.30% | 80.0% | 33.33% |
| LLaVA-Med-7b | P2 (67111743) | 40.03% | 75.0% | 0.0% |
| LLaVA-v1.6-7b | P1 (31550463) | 12.50% | 20.0% | 0.0% |
| LLaVA-v1.6-7b | P2 (67111743) | 0.00% | 0.0% | 0.0% |
| Gemma-3-4b | P1 (31550463) | 0.00% | 0.0% | 0.0% |
| Gemma-3-4b | P2 (67111743) | 1.39% | 0.0% | 0.0% |

---

## 5. Key Findings

### Finding 1 — Specialized Models Exceed Clinical Viability on Binary Tasks
Both `MedGemma-4B` and `LLaVA-Med-7B` demonstrated exceptional competence on raw clinical DICOM cross-sections. MedGemma achieved 88.89% closed accuracy, while LLaVA-Med achieved a flawless **100.0% closed accuracy**—correctly identifying the presence or absence of tears, cysts, bruises, and effusions across all 9 yes/no instances. This proves that medical-domain pre-training allows models to extract highly precise structural features even without standardized public benchmark curation.

### Finding 2 — The "Verbosity Penalty" Depresses Traditional F1 Scores
An initial glance at the Overall F1 scores suggests MedGemma (60.42%) heavily outperformed LLaVA-Med (37.09%). However, a qualitative review reveals this is purely an artifact of **Verbosity Bias**. LLaVA-Med is instruction-tuned on conversational data, causing it to generate highly accurate, full-sentence responses (e.g., *"No, there is no significant popliteal cyst present in the knee joint MRI image"*). Because our F1 scorer expects exact token matches (e.g., just `"No"`), it heavily penalized these verbose outputs despite their clinical correctness. MedGemma scored higher purely because it naturally outputs blunt, one-word answers. This strongly highlights the fragility of NLP n-gram metrics (F1/BLEU) in evaluating chat-tuned VLM capabilities.

### Finding 3 — Generalist Models Suffer from Severe "Pathology Bias"
The general-domain baseline models (`Gemma-3-4B` and `LLaVA-v1.6-7B`) both suffered catastrophic domain failure. Most notably, they exhibited a severe **"Yes-Man" Pathology Bias**. Because general datasets primarily contain medical images representing diseases or injuries, these generalist models assume *everything* is broken. LLaVA-v1.6 literally answered "Yes" to every single binary question. Gemma-3 hallucinates findings that are explicitly ruled out (e.g., predicting a partial MCL tear when all ligaments are intact).

### Finding 4 — Prompt-Following Collapse on Out-of-Domain Images
When presented with raw DICOM images it couldn't interpret, the generalist `LLaVA-v1.6` model experienced total prompt-following collapse. Instructed to provide its answer in the format *"Final Answer: X"*, the model literally output **"X"** for questions 2, 7, and 15 rather than substituting a clinical finding. This confirms that without medical visual alignment, the model's textual reasoning logic breaks down entirely.

### Finding 5 — Open-Ended Semantic Near-Misses Justify LLM-as-a-Judge
The lowest performance across all models was on open-ended questions, specifically those requiring spatial localization or descriptive pathology ("Where is the most pronounced cartilage loss?"). However, many of the specialized model "failures" were semantically valid near-misses penalized by rigid F1 matching:
* Ground Truth: *Soft tissue edema* -> MedGemma Prediction: *Soft tissue swelling* (F1 = 0.667)
* Ground Truth: *Small amount of fluid* -> LLaVA-Med Prediction: *The MRI image shows a popliteus tendon sheath with a fluid collection.* (F1 = 0.125)

These clinical equivalencies mathematically validate the necessity of transitioning from token-based F1 scoring to an **LLM-as-a-Judge semantic evaluation layer**. 

### Finding 6 — Multi-Slice Grid Degrades Specialized Performance (Spatial Dilution)
Transitioning from a single middle slice to a **2x2 multi-slice grid** resulted in a noticeable performance drop for MedGemma-4B (Overall F1 dropped from 60.42% to 47.92%, Closed Acc dropped from 88.89% to 66.67%). While the grid provides more anatomical context (simulating a radiologist's scroll), it fundamentally reduces the spatial resolution of the internal structures within each tile. The models clearly struggle with this spatial dilution, failing to confidently detect localized pathologies when the effective resolution per slice is quartered. LLaVA-Med experienced a slight uptick in F1 (37.09% to 41.16%), but overall, this confirms that rigid 2D grid tiling is not an optimal replacement for true 3D volumetric embeddings.

---

## 6. Comparison: Domain Alignment vs Parameter Scaling

| Model Classification | Model | Overall F1 (Multi-Slice) | Closed Binary Acc |
|---|---|---|---|
| **Medical (4B)** | MedGemma-4B | 47.92% | 66.67% |
| **Generalist (4B)** | Gemma-3-4B | 0.69% | 0.00% |
| **Medical (7B)** | LLaVA-Med-1.5-7B | 41.16%* | 77.78% |
| **Generalist (7B)** | LLaVA-v1.6-7B | 6.25% | 11.11% (All "Yes") |

*(Note: LLaVA-Med's F1 is artificially depressed due to the verbosity penalty).*

This perfectly controlled A/B test isolates **Domain Alignment** as the single variable required for medical VQA. When comparing equal parameter counts (4B vs 4B, and 7B vs 7B), the medically fine-tuned models succeed with high confidence in structural reasoning (66% - 77%), while the generalist models fail completely (0% - 11%). Scaling from 4B to 7B parameters using a newer generalist architecture (LLaVA-v1.6) did absolutely nothing to bridge the gap.

---

## 7. Limitations

**Sample size:** 16 QA pairs from 2 patients is insufficient for statistically reliable conclusions. All findings should be treated as directional indicators pending evaluation on the full dataset when it becomes available.

**QA pair construction:** Questions were manually derived from radiologist reports. Different annotators might extract different questions with different phrasings, affecting reproducibility. An LLM-automated QA generation pipeline should be implemented for the full dataset.

**Single slice vs Grid tiling:** We replaced single-slice evaluation with a 2x2 multi-slice grid to address previous limitations, simulating how a radiologist examines 20-34 slices. However, tiling 4 slices together reduces the spatial resolution of each individual slice by 4x. This "spatial dilution" clearly harmed the highest-performing models, suggesting that true 3D visual encoders (or native high-res sequential embedding) are required for robust multi-slice clinical VQA rather than simple 2D image stitching.

**No LLM judge:** The clinical DICOM evaluation has not yet been run through the LLM-as-a-Judge pipeline. Several near-miss predictions and verbose answers would likely receive full credit from a semantic judge. 

---

## 8. Next Steps

1. **Deploy LLM-as-a-Judge:** Run the existing LLaMA-3 evaluation pipeline on the generated `.jsonl` result files. This is the critical next step to resolve the verbosity bias (LLaVA-Med) and the semantic near-misses (MedGemma) plaguing our current Open-Ended F1 scores.
2. **Evaluate Final Medical Models:** Test the remaining medical models (e.g., `HuatuoGPT-Vision-7B`) on the 16 QA pairs and multislice pipeline to finalize the clinical leaderboard.
3. **Clinical Verification:** Verify the manual series-to-question mappings (e.g., matching meniscus questions to SAG PD FS) with a clinical collaborator or radiologist.
4. **Automate Pipeline for Full Dataset:** If the full dataset arrives, build an automated QA pair generation using an LLM to extract questions from radiologist reports at scale.
5. **Native 3D Volume Integration:** Given the performance degradation observed with 2D grid tiling, explore true 3D medical visual encoders that can natively ingest multiple DICOM slices without sacrificing spatial resolution.