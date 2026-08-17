#!/usr/bin/env python3
"""
Run LLM-as-a-Judge on HuatuoGPT v3_corrected SLAKE inference output.

Judge model : meta-llama/Llama-3.1-8B-Instruct (HF Inference API, no GPU)
Input       : outputs/inference/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_v3_corrected.jsonl
Output      : outputs/judge/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_v3_corrected_judged.jsonl

Resume support: re-running picks up from last completed record.

Usage:
  HF_TOKEN=hf_xxx python3 scripts/run_judge_v3_corrected.py
  OR set HF_TOKEN in your shell before running.
"""

import os, sys, json, re, time
from tqdm.auto import tqdm
from huggingface_hub import InferenceClient

# ── Paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE, 'outputs', 'inference',
    'FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_v3_corrected.jsonl')
OUTPUT_DIR = os.path.join(BASE, 'outputs', 'judge')
OUT_FILE   = os.path.join(OUTPUT_DIR, os.path.basename(INPUT_FILE).replace('.jsonl', '_judged.jsonl'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Judge config ─────────────────────────────────────────────────────────────
JUDGE_MODEL         = 'meta-llama/Llama-3.1-8B-Instruct'
SLEEP_BETWEEN_CALLS = 1.2   # seconds — stays well within free-tier rate limits
MAX_NEW_TOKENS      = 200
PASS_THRESHOLD      = 4.0   # score >= 4 → correct

MEDICAL_JUDGE_PROMPT = """\
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

Provide your evaluation using this exact format:
Evaluation: <your evaluation here>
Total rating: <integer 1-5>

Question: {question}
Reference answer: {ground_truth}
Predicted answer: {prediction}
"""

def extract_judge_score(answer: str) -> float | None:
    try:
        if 'Total rating:' in answer:
            rating_text = answer.split('Total rating:')[1]
        else:
            rating_text = answer
        digits = re.findall(r'\d+(?:\.\d+)?', rating_text)
        if digits:
            return max(1.0, min(5.0, float(digits[0])))
        return None
    except Exception:
        return None


def judge_record(client: InferenceClient, record: dict) -> dict:
    prompt = MEDICAL_JUDGE_PROMPT.format(
        question     = record.get('question', ''),
        ground_truth = record.get('ground_truth', ''),
        prediction   = record.get('prediction', '') or record.get('raw_output', ''),
    )

    retries = 3
    for attempt in range(retries):
        try:
            resp = client.chat_completion(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=MAX_NEW_TOKENS,
                temperature=0.1,
            )
            response = resp.choices[0].message.content
            score = extract_judge_score(response)
            return {**record,
                    'judge_response': response,
                    'judge_score'   : score,
                    'judge_correct' : (score is not None and score >= PASS_THRESHOLD)}
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"\n  [retry {attempt+1}/{retries}] Error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  [FAIL] idx={record.get('idx')}: {e}")
                return {**record, 'judge_response': str(e), 'judge_score': None, 'judge_correct': None}


def main():
    # ── Token ────────────────────────────────────────────────────────────────
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        print("Run: HF_TOKEN=hf_xxx python3 scripts/run_judge_v3_corrected.py")
        sys.exit(1)
    print(f"Judge model : {JUDGE_MODEL}")
    print(f"Input       : {INPUT_FILE}")
    print(f"Output      : {OUT_FILE}")

    # ── Load input ───────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    records = [json.loads(l) for l in open(INPUT_FILE) if l.strip()]
    print(f"\nLoaded {len(records)} records from input file.")

    # ── Resume: load already-completed records ────────────────────────────────
    completed = {}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed[int(r['idx'])] = r
                except Exception:
                    pass
        print(f"Resuming: {len(completed)} / {len(records)} already judged.")

    # ── Smoke test ───────────────────────────────────────────────────────────
    client = InferenceClient(model=JUDGE_MODEL, timeout=120, token=hf_token)
    try:
        test = client.chat_completion(
            messages=[{'role': 'user', 'content': 'Say OK.'}],
            max_tokens=5,
        )
        reply = test.choices[0].message.content
        print(f"Smoke test passed: {repr(reply[:30])}\n")
    except Exception as e:
        print(f"\nERROR: Smoke test failed: {e}")
        print("Check your HF token and network connection.")
        sys.exit(1)

    # ── Judge loop ────────────────────────────────────────────────────────────
    todo = [r for r in records if int(r['idx']) not in completed]
    print(f"Records to judge: {len(todo)}")

    scores_so_far = [r['judge_score'] for r in completed.values() if r.get('judge_score') is not None]

    with open(OUT_FILE, 'a') as f_out:
        for record in tqdm(todo, desc='LLM Judge', unit='rec'):
            judged = judge_record(client, record)
            f_out.write(json.dumps(judged) + '\n')
            f_out.flush()

            if judged['judge_score'] is not None:
                scores_so_far.append(judged['judge_score'])

            time.sleep(SLEEP_BETWEEN_CALLS)

    # ── Final summary ─────────────────────────────────────────────────────────
    all_judged = [json.loads(l) for l in open(OUT_FILE) if l.strip()]
    valid  = [r for r in all_judged if r.get('judge_score') is not None]
    closed = [r for r in valid if r.get('is_closed') in [True, 'True'] or r.get('answer_type') == 'CLOSED']
    open_  = [r for r in valid if r.get('is_closed') not in [True, 'True'] and r.get('answer_type') != 'CLOSED']

    def acc(recs): return sum(1 for r in recs if r.get('judge_correct')) / len(recs) * 100 if recs else 0.0
    def avg(recs): return sum(r['judge_score'] for r in recs) / len(recs) if recs else 0.0

    print(f"\n{'='*55}")
    print(f"  HuatuoGPT-7B — SLAKE v3_corrected — Judge Results")
    print(f"{'='*55}")
    print(f"  N judged        : {len(valid)} / {len(all_judged)}")
    print(f"  Mean score      : {avg(valid):.3f} / 5.0")
    print(f"  Judge Acc (≥4)  : {acc(valid):.2f}%")
    print(f"  Judge Acc Closed: {acc(closed):.2f}%  (N={len(closed)})")
    print(f"  Judge Acc Open  : {acc(open_):.2f}%  (N={len(open_)})")
    print(f"\n  Output: {OUT_FILE}")


if __name__ == '__main__':
    main()
