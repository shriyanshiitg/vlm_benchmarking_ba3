"""
Merges the retry results for the 51 records that previously had null
judge_b_score back into inter_rater_results.jsonl.

Usage:
    python3 scripts/merge_inter_rater_retry.py \
        --retry  outputs/inter_rater_retry_51_results.jsonl

The merged, complete file is written back to:
    outputs/inter_rater_results.jsonl   (in-place, with a .bak backup)

After merging, re-run the analysis:
    python3 scripts/inter_rater_analysis.py
"""

import argparse
import json
import os
import shutil

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, 'outputs', 'inter_rater_results.jsonl')

parser = argparse.ArgumentParser(description='Merge retry results into inter_rater_results.jsonl')
parser.add_argument('--retry', required=True, help='Path to the retry results JSONL (the 51 records with judge_b_score filled in)')
args = parser.parse_args()

RETRY_PATH = args.retry
if not os.path.exists(RETRY_PATH):
    raise FileNotFoundError(f'Retry file not found: {RETRY_PATH}')

# ---------------------------------------------------------------------------
# Load retry results — index by (model, idx, dataset)
# ---------------------------------------------------------------------------
retry_map = {}
retry_null = 0
for line in open(RETRY_PATH, encoding='utf-8'):
    r = json.loads(line)
    key = (r.get('model', ''), r.get('idx', 0), r.get('dataset', ''))
    if r.get('judge_b_score') is None:
        retry_null += 1
        print(f'  WARNING: retry record still has null judge_b_score: {key}')
    retry_map[key] = r

print(f'Retry records loaded: {len(retry_map)}  (still null: {retry_null})')

# ---------------------------------------------------------------------------
# Load original results and patch in the retry scores
# ---------------------------------------------------------------------------
original = []
for line in open(RESULTS_PATH, encoding='utf-8'):
    original.append(json.loads(line))

print(f'Original records: {len(original)}')

patched  = 0
still_null = 0
merged = []
for r in original:
    key = (r.get('model', ''), r.get('idx', 0), r.get('dataset', ''))
    if r.get('judge_b_score') is None and key in retry_map:
        retry_rec = retry_map[key]
        r['judge_b_score']    = retry_rec.get('judge_b_score')
        r['judge_b_response'] = retry_rec.get('judge_b_response')
        patched += 1
    if r.get('judge_b_score') is None:
        still_null += 1
    merged.append(r)

print(f'Patched: {patched}')
print(f'Still null after merge: {still_null}')

# ---------------------------------------------------------------------------
# Back up and write
# ---------------------------------------------------------------------------
backup_path = RESULTS_PATH + '.bak'
shutil.copy2(RESULTS_PATH, backup_path)
print(f'Backup saved to: {backup_path}')

with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    for r in merged:
        f.write(json.dumps(r) + '\n')

print(f'Merged file written to: {RESULTS_PATH}')
print(f'\nTotal records: {len(merged)}')
print(f'Records with both scores: {len(merged) - still_null}')
print('\nNext step: python3 scripts/inter_rater_analysis.py')
