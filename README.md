# VLM Medical VQA Benchmark

A rigorous, reproducible benchmark comparing **medical-domain VLMs** against **general-purpose VLMs** on medical visual question answering (VQA). The project evaluates five models across four datasets using a six-metric evaluation suite, with post-hoc statistical significance testing.

> **Internship research project** — all evaluations are zero-shot, run on Kaggle T4 GPUs.

---

## Models Evaluated

| Category | Model | Size |
|---|---|---|
| Medical | `google/medgemma-4b-it` | 4B |
| Medical | `microsoft/llava-med-v1.5-mistral-7b` | 7B |
| Medical | `FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` | 7B |
| General | `google/gemma-3-4b-it` | 4B |
| General | `llava-hf/llava-v1.6-mistral-7b-hf` | 7B |

## Datasets

| Dataset | Samples | Type |
|---|---|---|
| [SLAKE](https://www.med-vqa.com/slake/) | 1,061 | Medical VQA (CT / MRI / X-ray) |
| [VQA-RAD](https://osf.io/89kps/) | 451 | Radiology VQA |
| [VQAv2](https://visualqa.org/) | 1,000 (sampled) | General VQA |
| [OK-VQA](https://okvqa.allenai.org/) | 1,000 (sampled) | Knowledge-based VQA |

---

## Repository Structure

```
vlm_benchmark/
│
├── README.md
├── requirements.txt
│
├── docs/                              # All reports and analysis documents
│   ├── report.md                      # Main benchmark report (start here)
│   ├── error_analysis_report.md
│   ├── dicom_sample_dataset_report.md
│   ├── research_gap_analysis.md
│   └── statistical_significance_results.md
│
├── notebooks/                         # Jupyter / Kaggle notebooks (numbered pipeline)
│   ├── 01_dataset_verification.ipynb
│   ├── 02_inference_harness.ipynb     # v1 baseline inference
│   ├── 03_inference_harness_v2.ipynb  # v2 prompt-engineered inference (primary)
│   ├── 04_llm_judge.ipynb             # LLM-as-a-Judge pipeline
│   ├── 05_error_analysis.ipynb
│   ├── 06_statistical_significance.ipynb
│   ├── 07-medical-on-general-2.ipynb
│   ├── LLMasaJudgeMedonGen.ipynb
│   ├── dicom-evaluation-multislice-huatuogpt.ipynb
│   └── dicom/                         # DICOM-specific experiments
│       ├── dicom.ipynb
│       └── dicom-evaluation.ipynb
│
├── scripts/                           # Standalone Python scripts
│   ├── make_notebook.py
│   ├── score_new_outputs.py
│   └── statistical_significance.py   # Post-hoc bootstrap CI + permutation tests
│
├── outputs/
│   ├── inference/                     # Active v2 JSONL predictions (14 files)
│   ├── judge/                         # LLM-judge JSONL files + summary CSV
│   ├── error_analysis_csvs/           # Per-error-type analysis CSVs (A1–C3)
│   ├── dicom_evaluation/              # DICOM clinical VQA outputs
│   └── _archive/                      # Superseded files (v1, rescored, S-CoT)
│       ├── v1_predictions/
│       ├── rescored/
│       ├── scot_experiment/
│       └── judge_dry_run/
│
├── results/                           # Final computed outputs
│   ├── findings.json
│   ├── statistical_significance_results.json
│   └── statistical_significance_summary.md
│
└── data/                              # Local dataset assets (see .gitignore)
    ├── clinical_metadata.csv
    ├── clinical_vqa_dataset.jsonl
    ├── dicom_files/                   # ⚠ Not tracked — patient DICOM data
    └── slake_imgs/                    # ⚠ Not tracked — 212 MB image archive
```

---

## Key Results (v2 Protocol, Zero-Shot)

| Model | SLAKE F1 | SLAKE 95% CI | VQA-RAD F1 | VQA-RAD 95% CI |
|---|---|---|---|---|
| **MedGemma-4B** | **70.50%** | [67.83%, 73.23%] | **62.19%** | [57.88%, 66.52%] |
| HuatuoGPT-7B | 47.86% | [44.93%, 50.83%] | 57.40% | [52.92%, 61.77%] |
| Gemma-3-4B | 42.14% | [39.23%, 45.10%] | 43.39% | [38.94%, 47.86%] |
| LLaVA-Med-7B | 37.04% | [34.29%, 39.89%] | 34.53% | [30.48%, 38.72%] |
| LLaVA-1.6-7B | 36.98% | [34.17%, 39.80%] | 42.03% | [37.52%, 46.54%] |

> All differences verified with paired permutation tests (n=10,000). See [`results/statistical_significance_results.json`](results/statistical_significance_results.json).

### Notable Findings

- **Domain pre-training > parameter count:** MedGemma-4B outperforms all 7B models on medical VQA (`p<0.001`).
- **LLaVA-Med ≡ LLaVA-1.6 on SLAKE F1** (`p=0.963, ns`): medical fine-tuning confers zero advantage over the general-purpose model.
- **MedGemma vs HuatuoGPT on VQA-RAD** is only marginally significant (`p=0.022`); indistinguishable by Judge Accuracy (`p=0.067, ns`).

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Token-level F1 | Exact token overlap (primary metric) |
| Normalized F1 | Stemmed, stopword-filtered F1 |
| Closed Accuracy | Recall ≥ 0.5 for Yes/No questions |
| Open Accuracy | Recall ≥ 0.75 for open questions |
| BLEU-1 | N-gram overlap via `sacrebleu` |
| BERTScore | Contextual similarity via `roberta-large` |
| LLM-as-a-Judge | Semantic score 1–5 via `Llama-3.1-8B-Instruct` |

---

## Running the Statistical Analysis

```bash
# From the repo root
pip install numpy
python scripts/statistical_significance.py
# → results/statistical_significance_results.md
# → results/statistical_significance_results.json
```

---

## Infrastructure

- **Hardware:** Kaggle NVIDIA Tesla T4 (16 GB VRAM)
- **Quantization:** 4-bit NF4 via `bitsandbytes` for 7B models
- **Inference speed:** ~4s/sample (4B), ~6s/sample (7B quantized)
- **Full pipeline runtime:** ~9 hours for all 7,500+ LLM-judge evaluations

## Limitations

- All evaluations are **zero-shot** — not comparable to supervised fine-tuning results
- VQA-RAD uses the public split which may have training contamination
- LLM judge uses an 8B model; a frontier judge (GPT-4o) would be more reliable

---

## Citation / Report

See [`docs/report.md`](docs/report.md) for the full technical report including pipeline evolution, engineering challenges, and diagnostic analysis.
