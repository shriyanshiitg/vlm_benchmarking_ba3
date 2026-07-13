"""
Phase 1 — Build stratified 500-record sample for inter-rater agreement study.

Reads all *_judged.jsonl files from outputs/judge/ and selects a sample
designed to stress-test the 8B Llama judge specifically:
  - 100 records where 8B judge scored 5 (easy anchor — both should agree)
  - 100 records where 8B judge scored 1 (easy anchor — both should agree)
  - 150 records where 8B judge scored 3 (ambiguous — most likely to disagree)
  - 150 open-ended medical records (where clinical knowledge matters most)

Within each stratum, samples proportionally across all five models so no
single model dominates the per-model breakdown analysis.

Output: outputs/inter_rater_sample_500.jsonl
"""

import json
import os
import random
from collections import defaultdict

random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_DIR  = os.path.join(BASE_DIR, 'outputs', 'judge')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
OUT_PATH   = os.path.join(OUTPUT_DIR, 'inter_rater_sample_500.jsonl')

MEDICAL_DATASETS = {'slake', 'vqa_rad', 'slake_7b', 'vqa_rad_7b', 'slake_7b-2'}

# Normalise the two LLaVA-Med IDs that appear in the judged files to a single
# canonical name so per-model stats are not split across two keys.
MODEL_ALIASES = {
    'chaoyinshe/llava-med-v1.5-mistral-7b-hf': 'microsoft/llava-med-v1.5-mistral-7b',
}

# ---------------------------------------------------------------------------
# Load all judged records
# ---------------------------------------------------------------------------
judged_files = sorted(
    f for f in os.listdir(JUDGE_DIR)
    if f.endswith('_judged.jsonl')
)

print(f'Found {len(judged_files)} judged files:')
for f in judged_files:
    print(f'  {f}')

all_records = []
skipped = 0
for fname in judged_files:
    path = os.path.join(JUDGE_DIR, fname)
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r.get('judge_score') is None:
            skipped += 1
            continue
        # Normalise model name
        r['model'] = MODEL_ALIASES.get(r.get('model', ''), r.get('model', ''))
        all_records.append(r)

print(f'\nTotal records loaded: {len(all_records)}  (skipped {skipped} with null judge_score)')

# ---------------------------------------------------------------------------
# Helper: proportional per-model sampling from a pool
# ---------------------------------------------------------------------------
def proportional_sample(pool: list, n: int) -> list:
    """Sample n records from pool, proportionally across models."""
    if len(pool) <= n:
        return list(pool)

    by_model = defaultdict(list)
    for r in pool:
        by_model[r.get('model', 'unknown')].append(r)

    # Determine per-model quota (proportional to pool size, rounded down)
    total_pool = len(pool)
    quotas = {}
    for model, recs in by_model.items():
        quotas[model] = max(1, round(n * len(recs) / total_pool))

    # Adjust for rounding drift
    while sum(quotas.values()) > n:
        biggest = max(quotas, key=quotas.get)
        quotas[biggest] -= 1
    while sum(quotas.values()) < n:
        biggest = max(by_model, key=lambda m: len(by_model[m]))
        quotas[biggest] += 1

    sampled = []
    for model, quota in quotas.items():
        pool_m = by_model[model]
        sampled.extend(random.sample(pool_m, min(quota, len(pool_m))))

    return sampled

# ---------------------------------------------------------------------------
# Build score buckets
# ---------------------------------------------------------------------------
score_buckets = defaultdict(list)
for r in all_records:
    score_buckets[int(r['judge_score'])].append(r)

print('\nScore distribution in full dataset:')
for s in sorted(score_buckets):
    print(f'  Score {s}: {len(score_buckets[s])} records')

# ---------------------------------------------------------------------------
# Medical open-ended pool (dataset-specific stress test)
# ---------------------------------------------------------------------------
medical_open_pool = [
    r for r in all_records
    if not r.get('is_closed', True)
    and r.get('dataset', '') in MEDICAL_DATASETS
]
print(f'\nMedical open-ended pool: {len(medical_open_pool)} records')

# ---------------------------------------------------------------------------
# Sample each stratum
# ---------------------------------------------------------------------------
strata = {
    'score_5_easy':       proportional_sample(score_buckets[5], 100),
    'score_1_easy':       proportional_sample(score_buckets[1], 100),
    'score_3_ambiguous':  proportional_sample(score_buckets[3], 150),
    'medical_open':       proportional_sample(medical_open_pool, 150),
}

# Tag each record with the stratum it came from (useful for analysis later)
for stratum_name, records in strata.items():
    for r in records:
        r['sample_stratum'] = stratum_name

print('\nStratum sizes before deduplication:')
for name, recs in strata.items():
    print(f'  {name}: {len(recs)}')

# ---------------------------------------------------------------------------
# Merge and deduplicate by (model, idx, dataset)
# ---------------------------------------------------------------------------
combined = []
for recs in strata.values():
    combined.extend(recs)

seen = set()
deduped = []
for r in combined:
    key = (r.get('model', ''), r.get('idx', 0), r.get('dataset', ''))
    if key not in seen:
        seen.add(key)
        deduped.append(r)

print(f'\nFinal sample size after deduplication: {len(deduped)}')

# ---------------------------------------------------------------------------
# Print per-model and per-stratum breakdown
# ---------------------------------------------------------------------------
by_model_count = defaultdict(int)
by_stratum_count = defaultdict(int)
for r in deduped:
    by_model_count[r.get('model', 'unknown')] += 1
    by_stratum_count[r.get('sample_stratum', 'unknown')] += 1

print('\nPer-model breakdown in final sample:')
for model, cnt in sorted(by_model_count.items()):
    print(f'  {model}: {cnt}')

print('\nPer-stratum breakdown in final sample:')
for stratum, cnt in sorted(by_stratum_count.items()):
    print(f'  {stratum}: {cnt}')

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    for r in deduped:
        f.write(json.dumps(r) + '\n')

print(f'\nSaved {len(deduped)} records to: {OUT_PATH}')
print('\nNext step: run the Kaggle notebook 08_inter_rater_agreement.ipynb')
print('using this file to get Judge B scores from Qwen3-30B-A3B.')
