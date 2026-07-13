"""
Generates notebooks/08_inter_rater_agreement.ipynb — the Kaggle notebook
that runs Judge B (Qwen3-30B-A3B via HF Inference API) on the 499-record
inter-rater sample and saves inter_rater_results.jsonl.

Run locally:  python3 scripts/make_inter_rater_notebook.py
Then upload the generated notebook to Kaggle.
"""

import json

cells = []

def add_md(text: str):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def add_code(text: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })


# ── Title ────────────────────────────────────────────────────────────────────
add_md("""# Inter-Rater Agreement Study — Judge B Scoring
## VLM Medical VQA Benchmark

**Purpose:** Run a second, larger judge (Qwen3-30B-A3B via HuggingFace Inference API)
on the stratified 499-record sample produced by `scripts/build_inter_rater_sample.py`.
This validates whether the primary Llama-3.1-8B judge used throughout the benchmark
produces scores that agree with a significantly stronger model.

**Inputs:**
- `inter_rater_sample_500.jsonl` — uploaded as a Kaggle dataset (see setup notes)

**Outputs:**
- `inter_rater_results.jsonl` — each record has both `judge_score` (8B, already present)
  and `judge_b_score` (30B, computed here)

**No GPU required** — all inference is done via the HuggingFace Inference API.

---

### ⚠️ Setup before running
1. Upload `outputs/inter_rater_sample_500.jsonl` from your local machine as a
   Kaggle dataset named `inter-rater-sample`.
2. Add your HuggingFace token as a Kaggle Secret named `HF_TOKEN`.
3. Run all cells. Expected runtime: ~25–40 minutes for 499 API calls.
""")


# ── Cell 1: Install ───────────────────────────────────────────────────────────
add_md("## Cell 1 — Install dependencies")
add_code("""import subprocess
subprocess.run(['pip', 'install', 'huggingface_hub', 'tqdm', '-q'])
print('Done.')
""")


# ── Cell 2: Imports & auth ────────────────────────────────────────────────────
add_md("## Cell 2 — Imports and HuggingFace login")
add_code("""import os, json, re, time
from tqdm.auto import tqdm
from huggingface_hub import InferenceClient, login

# Read HF token from Kaggle secrets
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
hf_token = secrets.get_secret('HF_TOKEN')
login(token=hf_token, add_to_git_credential=False)
print('Logged in to HuggingFace.')
""")


# ── Cell 3: Paths ─────────────────────────────────────────────────────────────
add_md("## Cell 3 — Configure paths")
add_code("""# Input — uploaded Kaggle dataset
SAMPLE_PATH = '/kaggle/input/inter-rater-sample/inter_rater_sample_500.jsonl'

# Output — saved to working directory, then download manually
OUT_PATH = '/kaggle/working/inter_rater_results.jsonl'

# Checkpoint: resume if kernel was interrupted
print(f'Input:  {SAMPLE_PATH}')
print(f'Output: {OUT_PATH}')
""")


# ── Cell 4: Load sample ───────────────────────────────────────────────────────
add_md("## Cell 4 — Load the stratified sample")
add_code("""records = []
for line in open(SAMPLE_PATH, encoding='utf-8'):
    records.append(json.loads(line))

print(f'Loaded {len(records)} records.')

# Quick breakdown
from collections import Counter
stratum_counts = Counter(r.get('sample_stratum', 'unknown') for r in records)
model_counts   = Counter(r.get('model', 'unknown').split('/')[-1] for r in records)

print('\\nStrata:')
for k, v in sorted(stratum_counts.items()):
    print(f'  {k}: {v}')

print('\\nModels:')
for k, v in sorted(model_counts.items()):
    print(f'  {k}: {v}')
""")


# ── Cell 5: Judge B setup ─────────────────────────────────────────────────────
add_md("""## Cell 5 — Initialize Judge B (Qwen3-30B-A3B)

Using `Qwen/Qwen3-30B-A3B` — a 30B MoE model (~3B active parameters at inference)
available free on the HuggingFace Inference API.

**Critical:** We use the **identical prompt** as the Llama-3.1-8B judge. Only the model
changes. This is required for a valid inter-rater comparison.
""")
add_code("""JUDGE_B_MODEL = 'Qwen/Qwen3-30B-A3B'

judge_b_client = InferenceClient(
    provider="hf-inference",
    api_key=hf_token,
)

# Exact same prompt used in 04_llm_judge.ipynb — do NOT modify
MEDICAL_JUDGE_PROMPT = \"\"\"
You are an expert medical evaluator assessing the quality of answers to medical visual question answering (VQA) tasks.

You will be given:
- A medical question about a radiology or pathology image
- A reference answer (ground truth)
- A predicted answer from a vision-language model

Your task is to rate how correct the predicted answer is compared to the reference answer.
Focus on medical correctness and semantic equivalence, not exact wording.

Use this scale:
1: Completely wrong — the predicted answer is medically incorrect or entirely irrelevant
2: Mostly wrong — contains a relevant medical concept but misses the key point
3: Partially correct — captures the general idea but with a meaningful medical error or omission
4: Mostly correct — semantically equivalent to the reference with minor phrasing differences (e.g. 'Lungs' vs 'Lung')
5: Fully correct — matches the reference answer in medical meaning, possibly with different but equivalent phrasing

Provide your feedback as follows:

Feedback:::
Evaluation: (your medical reasoning for the rating, 1-2 sentences)
Total rating: (your rating, as a single integer between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question, reference answer, and predicted answer.

Question: {question}
Reference answer: {reference}
Predicted answer: {prediction}

Provide your feedback. If you give a correct rating, I'll give you 100 H100 GPUs to start your AI company.
Feedback:::
Evaluation: \"\"\"

print(f'Judge B model: {JUDGE_B_MODEL}')
print(f'Prompt length: {len(MEDICAL_JUDGE_PROMPT)} characters')
""")


# ── Cell 6: Score extraction function ────────────────────────────────────────
add_md("## Cell 6 — Score extraction and smoke test")
add_code("""def extract_judge_score(answer: str):
    \"\"\"
    Extract the integer score from the judge's response.
    Looks for 'Total rating:' then grabs the first number.
    Returns None if extraction fails.
    \"\"\"
    try:
        if 'Total rating:' in answer:
            rating_text = answer.split('Total rating:')[1]
        else:
            rating_text = answer
        digits = re.findall(r'\\d+(?:\\.\\d+)?', rating_text)
        if digits:
            score = float(digits[0])
            return max(1.0, min(5.0, score))
        return None
    except Exception as e:
        print(f'Extraction error: {e}')
        return None


def judge_b_single(question: str, reference: str, prediction: str, retries: int = 3) -> dict:
    \"\"\"
    Call Judge B on a single triple using raw text_generation.
    Qwen3-30B-A3B does not support the 'conversational' task on the HF
    Inference API free tier, so we bypass chat.completions and call
    text_generation directly, wrapping the prompt in Qwen's native ChatML
    format so the model still behaves as an assistant.
    Retries up to 3 times on API errors with exponential backoff.
    \"\"\"
    raw_prompt = MEDICAL_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        prediction=prediction,
    )
    # Manually apply Qwen's ChatML template (system + user + assistant prefix)
    formatted_prompt = (
        "<|im_start|>system\\nYou are a helpful and strict evaluator.<|im_end|>\\n"
        f"<|im_start|>user\\n{raw_prompt}<|im_end|>\\n"
        "<|im_start|>assistant\\n"
    )
    for attempt in range(retries):
        try:
            raw_text = judge_b_client.text_generation(
                formatted_prompt,
                model=JUDGE_B_MODEL,
                max_new_tokens=250,
                temperature=0.01,       # near-deterministic scoring
                return_full_text=False, # generated tokens only
            )
            score = extract_judge_score(raw_text)
            return {'judge_b_response': raw_text, 'judge_b_score': score}
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s
                print(f'  Attempt {attempt+1} failed: {e}. Retrying in {wait}s...')
                time.sleep(wait)
            else:
                return {'judge_b_response': str(e), 'judge_b_score': None}


# Smoke test — three known examples
print('Smoke test on 3 known examples:')
test_cases = [
    ('What modality is used to take this image?', 'CT',    'Computed tomography (CT)', '~5'),
    ('What modality is used to take this image?', 'CT',    'MRI',                      '~1'),
    ('Which part of the body does this image belong to?', 'Chest', 'Chest/Thorax',     '~4-5'),
]
for q, ref, pred, expected in test_cases:
    result = judge_b_single(q, ref, pred)
    print(f'  Q: {q[:60]}')
    print(f'  Ref: {ref}  |  Pred: {pred}  |  Expected: {expected}  |  Got: {result["judge_b_score"]}')
    print()
""")


# ── Cell 7: Main evaluation loop ──────────────────────────────────────────────
add_md("""## Cell 7 — Main evaluation loop

Iterates over all 499 records, calls Judge B for each, and appends results
to `inter_rater_results.jsonl` with checkpoint/resume support.

Expected runtime: ~25–40 minutes (approximately 3–5 seconds per API call).
""")
add_code("""# Load already-completed records (checkpoint resume)
completed_keys = set()
if os.path.exists(OUT_PATH):
    for line in open(OUT_PATH, encoding='utf-8'):
        r = json.loads(line)
        key = (r.get('model', ''), r.get('idx', 0), r.get('dataset', ''))
        completed_keys.add(key)
    print(f'Resuming: {len(completed_keys)} records already done.')
else:
    print('Starting fresh.')

# Run evaluation
failed = 0
with open(OUT_PATH, 'a', encoding='utf-8') as f_out:
    for record in tqdm(records, desc='Judge B scoring'):
        key = (record.get('model', ''), record.get('idx', 0), record.get('dataset', ''))
        if key in completed_keys:
            continue

        result = judge_b_single(
            question   = record.get('question', ''),
            reference  = record.get('ground_truth', ''),
            prediction = record.get('prediction', ''),
        )

        # Merge Judge B scores into the original record
        out_record = {**record, **result}
        f_out.write(json.dumps(out_record) + '\\n')
        f_out.flush()

        if result['judge_b_score'] is None:
            failed += 1

        # Polite delay to stay within free-tier rate limits
        time.sleep(0.5)

print(f'\\nDone. Total: {len(records)}  |  Failed (null score): {failed}')
print(f'Output saved to: {OUT_PATH}')
""")


# ── Cell 8: Quick validation ──────────────────────────────────────────────────
add_md("## Cell 8 — Quick validation of results")
add_code("""import numpy as np

results = [json.loads(l) for l in open(OUT_PATH)]
valid   = [r for r in results if r.get('judge_score') is not None and r.get('judge_b_score') is not None]

print(f'Total records in output:        {len(results)}')
print(f'Records with both scores:       {len(valid)}')
print(f'Records with null judge_b:      {len(results) - len(valid)}')

if valid:
    a = np.array([r['judge_score']   for r in valid])
    b = np.array([r['judge_b_score'] for r in valid])
    print(f'\\n8B  judge — mean: {a.mean():.3f}, std: {a.std():.3f}')
    print(f'30B judge — mean: {b.mean():.3f}, std: {b.std():.3f}')
    print(f'Mean absolute difference: {np.mean(np.abs(a - b)):.3f}')
    print(f'Exact agreement: {np.mean(a.astype(int) == b.astype(int))*100:.1f}%')

    print('\\n✅ Download inter_rater_results.jsonl from the Output tab.')
    print('   Place it in: outputs/inter_rater_results.jsonl')
    print('   Then run:    python3 scripts/inter_rater_analysis.py')
""")


# ── Build notebook ────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = 'notebooks/08_inter_rater_agreement.ipynb'
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f'Notebook generated: {out_path}')
print('\nNext steps:')
print('  1. Upload outputs/inter_rater_sample_500.jsonl to Kaggle as dataset "inter-rater-sample"')
print('  2. Upload notebooks/08_inter_rater_agreement.ipynb to Kaggle')
print('  3. Add HF_TOKEN as a Kaggle Secret')
print('  4. Run all cells (~25-40 min)')
print('  5. Download inter_rater_results.jsonl → place in outputs/')
print('  6. Run: python3 scripts/inter_rater_analysis.py')
