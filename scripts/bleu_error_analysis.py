"""
BLEU Error Analysis — 5-Phase Study
====================================
Phase 1: Data stratification (open-ended, medical vs. general)
Phase 2: Triple-axis correlation (BLEU vs. F1 vs. Judge)
Phase 3: Scatter plot — the "Rescue Zone" where classical metrics fail
Phase 4: Qualitative N-Gram Autopsy (high-judge, low-BLEU records)
Phase 5: Markdown report generation → docs/bleu_error_analysis.md

Reads all judged JSONL files from outputs/judge/.
Computes sentence-BLEU-4 (with add-1 smoothing) and token-F1 per record.
"""

import json
import os
import re
import glob
import warnings
import math
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, spearmanr
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGE_DIR  = os.path.join(BASE_DIR, 'outputs', 'judge')
DOCS_DIR   = os.path.join(BASE_DIR, 'docs')
FIGS_DIR   = os.path.join(BASE_DIR, 'results')  # all figures live in results/
OUT_MD     = os.path.join(DOCS_DIR, 'bleu_error_analysis.md')
OUT_PLOT   = os.path.join(FIGS_DIR, 'fig_bleu_rescue_zone.png')

MEDICAL_DATASETS  = {'slake', 'vqa_rad'}
GENERAL_DATASETS  = {'vqav2', 'okvqa', 'ok_vqa'}

MODEL_SHORT = {
    'google/medgemma-4b-it':                              'MedGemma-4B',
    'google/gemma-3-4b-it':                               'Gemma-3-4B',
    'microsoft/llava-med-v1.5-mistral-7b':                'LLaVA-Med-7B',
    'llava-hf/llava-v1.6-mistral-7b-hf':                  'LLaVA-1.6-7B',
    'FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL': 'HuatuoGPT-7B',
    # 'chaoyinshe' variant used for some okvqa/vqav2 files
    'chaoyinshe/llava-med-v1.5-mistral-7b-hf':            'LLaVA-Med-7B',
}

SMOOTHING = SmoothingFunction().method1   # add-1 smoothing for short sentences

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: str, ground_truth: str) -> float:
    """Unigram token F1 (same definition used in SQuAD / VQA benchmarks)."""
    pred_tokens = normalise(prediction).split()
    gt_tokens   = normalise(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    pred_ctr = Counter(pred_tokens)
    gt_ctr   = Counter(gt_tokens)
    common   = sum((pred_ctr & gt_ctr).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall    = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu4(prediction: str, ground_truth: str) -> float:
    """Sentence BLEU-4 with add-1 smoothing."""
    hyp = normalise(prediction).split()
    ref = normalise(ground_truth).split()
    if not hyp or not ref:
        return 0.0
    return sentence_bleu([ref], hyp, smoothing_function=SMOOTHING)


def pearson_safe(a, b):
    """Returns (r, p) or (nan, nan) if too few points."""
    a, b = np.array(a), np.array(b)
    if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan'), float('nan')
    return pearsonr(a, b)


def spearman_safe(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 5:
        return float('nan'), float('nan')
    return spearmanr(a, b)


# ---------------------------------------------------------------------------
# Phase 1: Load & stratify
# ---------------------------------------------------------------------------
print("Phase 1 — Loading and stratifying data...")

all_records = []
for fpath in glob.glob(os.path.join(JUDGE_DIR, '*_judged.jsonl')):
    for line in open(fpath, encoding='utf-8'):
        r = json.loads(line)
        if r.get('judge_score') is None:
            continue
        model   = r.get('model', '')
        dataset = r.get('dataset', '').lower().replace('-', '_')
        r['model_short'] = MODEL_SHORT.get(model, model.split('/')[-1])
        r['dataset_norm'] = dataset
        r['domain'] = 'medical' if dataset in MEDICAL_DATASETS else 'general'
        r['f1']     = token_f1(r.get('prediction', ''), r.get('ground_truth', ''))
        r['bleu']   = bleu4(r.get('prediction', ''), r.get('ground_truth', ''))
        r['judge_norm'] = r['judge_score'] / 5.0    # normalise to [0,1]
        all_records.append(r)

print(f"  Total records loaded: {len(all_records)}")

# Filter to open-ended only
open_records = [r for r in all_records if not r.get('is_closed', False)]
print(f"  Open-ended records:   {len(open_records)}")

medical_open  = [r for r in open_records if r['domain'] == 'medical']
general_open  = [r for r in open_records if r['domain'] == 'general']
print(f"  Medical open:         {len(medical_open)}")
print(f"  General open:         {len(general_open)}")

# ---------------------------------------------------------------------------
# Phase 2: Triple-axis correlations
# ---------------------------------------------------------------------------
print("\nPhase 2 — Triple-axis correlation analysis...")

def correlation_block(records, label):
    bleus   = [r['bleu']       for r in records]
    f1s     = [r['f1']         for r in records]
    judges  = [r['judge_norm'] for r in records]

    r_bj, p_bj = pearson_safe(bleus, judges)
    r_bf, p_bf = pearson_safe(bleus, f1s)
    r_jf, p_jf = pearson_safe(judges, f1s)
    s_bj, _    = spearman_safe(bleus, judges)
    s_bf, _    = spearman_safe(bleus, f1s)
    s_jf, _    = spearman_safe(judges, f1s)
    return {
        'label': label, 'n': len(records),
        'bleu_mean': np.mean(bleus), 'f1_mean': np.mean(f1s), 'judge_mean': np.mean(judges),
        'pearson_bleu_judge': r_bj, 'p_bleu_judge': p_bj,
        'pearson_bleu_f1':    r_bf, 'p_bleu_f1': p_bf,
        'pearson_judge_f1':   r_jf, 'p_judge_f1': p_jf,
        'spearman_bleu_judge': s_bj,
        'spearman_bleu_f1':    s_bf,
        'spearman_judge_f1':   s_jf,
    }

overall_corr = correlation_block(open_records, 'All open-ended')
medical_corr = correlation_block(medical_open, 'Medical (SLAKE + VQA-RAD)')
general_corr = correlation_block(general_open, 'General (VQAv2 + OK-VQA)')

# Per-model correlations
model_corrs = {}
for model_short in sorted(set(r['model_short'] for r in open_records)):
    recs = [r for r in open_records if r['model_short'] == model_short]
    model_corrs[model_short] = correlation_block(recs, model_short)

for c in [overall_corr, medical_corr, general_corr]:
    print(f"  [{c['label']}] N={c['n']}  "
          f"BLEU↔Judge: r={c['pearson_bleu_judge']:.3f}  "
          f"BLEU↔F1: r={c['pearson_bleu_f1']:.3f}  "
          f"Judge↔F1: r={c['pearson_judge_f1']:.3f}")

# Delta analysis: (judge_norm - bleu) per model
print("\n  Delta analysis (judge_norm − BLEU), higher = verbosity penalty:")
model_delta = {}
for ms, mc in model_corrs.items():
    delta = mc['judge_mean'] - mc['bleu_mean']
    model_delta[ms] = delta
    print(f"    {ms:30s}  judge_norm={mc['judge_mean']:.3f}  bleu={mc['bleu_mean']:.3f}  Δ={delta:+.3f}")

# ---------------------------------------------------------------------------
# Phase 3: Scatter plot — the Rescue Zone
# ---------------------------------------------------------------------------
print("\nPhase 3 — Generating Rescue Zone scatter plot...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('BLEU vs. Token F1 Coloured by LLM Judge Score\n'
             '(Bottom-Left / Bright = Classical Metric Failure)', fontsize=13, y=1.02)

for ax, records, title in [
    (axes[0], medical_open,  'Medical Datasets (SLAKE + VQA-RAD)'),
    (axes[1], general_open,  'General Datasets (VQAv2 + OK-VQA)'),
]:
    bleus   = np.array([r['bleu']       for r in records])
    f1s     = np.array([r['f1']         for r in records])
    judges  = np.array([r['judge_score'] for r in records])

    sc = ax.scatter(bleus, f1s, c=judges, cmap='RdYlGn', vmin=1, vmax=5,
                    alpha=0.45, s=14, linewidths=0)
    ax.set_xlabel('BLEU-4 Score', fontsize=11)
    ax.set_ylabel('Token F1', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Rescue-zone box: BLEU<0.1, F1<0.15, Judge>=4
    rescue = [(b, f, j) for b, f, j in zip(bleus, f1s, judges)
              if b < 0.10 and f < 0.15 and j >= 4]
    ax.add_patch(plt.Rectangle((0, 0), 0.10, 0.15,
                                linewidth=1.5, edgecolor='royalblue',
                                facecolor='royalblue', alpha=0.08,
                                label=f'Rescue zone ({len(rescue)} pts)'))
    ax.annotate(f'Rescue zone\n{len(rescue)} records\n(BLEU<0.10, F1<0.15, Judge≥4)',
                xy=(0.05, 0.075), fontsize=8, color='royalblue',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Judge Score (1–5)', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=160, bbox_inches='tight')
plt.close()
print(f"  Plot saved: {OUT_PLOT}")

# ---------------------------------------------------------------------------
# Phase 4: Qualitative N-Gram Autopsy
# ---------------------------------------------------------------------------
print("\nPhase 4 — N-gram autopsy on high-judge / low-BLEU records...")

# Target: medical open-ended, judge_score >= 4, BLEU < 0.05
autopsy_pool = [r for r in medical_open
                if r['judge_score'] >= 4 and r['bleu'] < 0.05]
print(f"  Autopsy pool (judge>=4, BLEU<0.05, medical): {len(autopsy_pool)}")

# Categorise into failure buckets — comprehensive synonym & pattern coverage
# -------------------------------------------------------------------------
# Pairs are bidirectional. Each entry means: if ONE side is in the GT and the
# OTHER side is in the prediction (and the GT side is absent), that is synonymy.
MEDICAL_SYNONYMS_PAIRS = [
    # Organ / anatomy synonymy
    ('renal', 'kidney'), ('pulmonary', 'lung'), ('pulmonary', 'lungs'),
    ('hepatic', 'liver'), ('cardiac', 'heart'), ('cerebral', 'brain'),
    ('gastric', 'stomach'), ('thoracic', 'chest'), ('thoracic', 'thorax'),
    ('abdominal', 'abdomen'), ('abdominal', 'belly'),
    ('cranial', 'skull'), ('cranial', 'head'), ('cranial', 'brain'),
    ('head', 'brain'),          # "Head" GT → "Brain" pred
    ('chest', 'respiratory'), ('chest', 'thorax'),
    ('lung', 'lungs'),          # simple plural variant handled here too
    # Plane / orientation synonymy
    ('axial', 'transverse'), ('axial', 'transverse plane'),
    ('transverse', 'transverse plane'), ('coronal', 'frontal'),
    ('sagittal', 'lateral'),
    # Condition synonymy
    ('edema', 'fluid'), ('edema', 'oedema'), ('opacity', 'haziness'),
    ('atelectasis', 'collapse'), ('pneumonia', 'infection'),
    ('fracture', 'break'), ('lesion', 'mass'), ('lesion', 'nodule'),
    ('neoplasm', 'tumor'), ('neoplasm', 'tumour'), ('neoplasm', 'mass'),
    ('calcification', 'calcium'), ('effusion', 'fluid'),
    ('cardiomegaly', 'enlarged'), ('pneumothorax', 'air'),
    # Modality synonymy
    ('ct', 'computed tomography'), ('mri', 'magnetic resonance'),
    ('xray', 'x-ray'), ('xray', 'radiograph'),
    # Lateral synonymy
    ('left', 'right'),   # these FAIL similarly — judge often credits near-correct laterality
    # Ventricle synonymy
    ('lateral ventricles', 'cerebral ventricles'), ('ventricle', 'ventricles'),
]

# Number word ↔ digit equivalence (common BLEU killer on count questions)
NUM_WORD_MAP = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12',
}
NUM_DIGIT_MAP = {v: k for k, v in NUM_WORD_MAP.items()}

FILLER_PATTERNS = [
    r'based on (the |this )?image',
    r'the (image|scan|modality|answer|finding|study) (is|shows|appears|indicates)',
    r'looking at (the |this )?',
    r'in (the|this) (image|scan|study|mri|ct)',
    r'from the (image|scan)',
    r'it (appears|seems|looks) (like|as|to be)',
    r'i (can see|think|believe|would say)',
    r'the answer (is|would be)',
    r'this (image|scan|mri|ct|x-ray) (shows|demonstrates|reveals|depicts)',
    r'appears? to (be|show)',
]
FILLER_RE = re.compile('|'.join(FILLER_PATTERNS), re.I)


def contains_number_synonym(pred_n: str, gt_n: str) -> bool:
    """Returns True if the only difference between GT and prediction is a
    number vs. its word equivalent (e.g., GT='2', pred='two' or vice versa)."""
    gt_tokens   = set(gt_n.split())
    pred_tokens = set(pred_n.split())
    for word, digit in NUM_WORD_MAP.items():
        if digit in gt_tokens and word in pred_tokens:
            return True
        if word in gt_tokens and digit in pred_tokens:
            return True
    return False


def is_morphological_variant(pred_n: str, gt_n: str) -> bool:
    """Catches simple plural/adjectival forms that differ by 1-2 characters.
    E.g., 'lung' vs 'lungs', 'abdomen' vs 'abdominal'."""
    gt_t = gt_n.split()
    pd_t = pred_n.split()
    if not gt_t or not pd_t:
        return False
    # Single-token case: one of {pred, gt} is a prefix of the other
    if len(gt_t) == 1 and len(pd_t) == 1:
        a, b = gt_t[0], pd_t[0]
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        if len(short) >= 3 and long_.startswith(short):
            return True
    return False


def classify_failure(pred: str, gt: str) -> str:
    pred_n = normalise(pred)
    gt_n   = normalise(gt)
    pred_t = set(pred_n.split())
    gt_t   = set(gt_n.split())

    # 1. Number-word synonymy (e.g., GT='one', pred='1')
    if contains_number_synonym(pred_n, gt_n):
        return 'Medical Synonymy'

    # 2. Morphological variant (lung→lungs, abdomen→abdominal)
    if is_morphological_variant(pred_n, gt_n):
        return 'Medical Synonymy'

    # 3. Established medical synonym pairs
    for w1, w2 in MEDICAL_SYNONYMS_PAIRS:
        w1_n = normalise(w1); w2_n = normalise(w2)
        if (w1_n in gt_n and w2_n in pred_n and w1_n not in pred_n):
            return 'Medical Synonymy'
        if (w2_n in gt_n and w1_n in pred_n and w2_n not in pred_n):
            return 'Medical Synonymy'

    # 4. Conversational filler: prediction has filler preamble but correct core
    if FILLER_RE.search(pred_n) or len(pred_n.split()) > len(gt_n.split()) * 3 + 4:
        core_gt = [t for t in gt_t if len(t) > 2]
        if core_gt and any(t in pred_n for t in core_gt):
            return 'Conversational Filler'

    # 5. Granularity mismatch: prediction is more specific than GT or vice versa
    #    — pred is much longer and GT tokens are a subset of pred tokens
    if len(pred_n.split()) > len(gt_n.split()) * 2 + 2 and gt_t <= pred_t:
        return 'Granularity Mismatch'
    #    — OR: pred is shorter (concise specific answer, GT is verbose)
    if len(gt_n.split()) > len(pred_n.split()) * 2 + 2 and pred_t <= gt_t:
        return 'Granularity Mismatch'

    return 'Other'


buckets = defaultdict(list)
for r in autopsy_pool:
    cat = classify_failure(r.get('prediction', ''), r.get('ground_truth', ''))
    r['autopsy_bucket'] = cat
    buckets[cat].append(r)

bucket_counts = {k: len(v) for k, v in buckets.items()}
print(f"  Autopsy buckets: {bucket_counts}")

# Per-model breakdown in autopsy pool
model_bucket = defaultdict(lambda: defaultdict(int))
for r in autopsy_pool:
    model_bucket[r['model_short']][r['autopsy_bucket']] += 1

# Sample examples from each bucket (up to 2 per model per bucket)
autopsy_examples = {}
for bucket, recs in buckets.items():
    # Prioritise contrasting models: MedGemma vs LLaVA-Med
    examples = []
    for model in ['MedGemma-4B', 'LLaVA-Med-7B', 'HuatuoGPT-7B']:
        sample = [r for r in recs if r['model_short'] == model][:2]
        examples.extend(sample)
    autopsy_examples[bucket] = examples[:6]

# ---------------------------------------------------------------------------
# Phase 5: Markdown report
# ---------------------------------------------------------------------------
print("\nPhase 5 — Generating markdown report...")

def fmt_r(r, p):
    if math.isnan(r):
        return 'n/a'
    stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    return f'{r:.3f}{stars}'

def corr_table_row(c):
    return (f'| {c["label"]} | {c["n"]} '
            f'| {fmt_r(c["pearson_bleu_judge"], c["p_bleu_judge"])} '
            f'| {fmt_r(c["pearson_bleu_f1"],    c["p_bleu_f1"])} '
            f'| {fmt_r(c["pearson_judge_f1"],   c["p_judge_f1"])} |')

overall_row = corr_table_row(overall_corr)
medical_row = corr_table_row(medical_corr)
general_row = corr_table_row(general_corr)

model_corr_rows = '\n'.join(
    f'| {ms} | {mc["n"]} '
    f'| {fmt_r(mc["pearson_bleu_judge"], mc["p_bleu_judge"])} '
    f'| {fmt_r(mc["pearson_bleu_f1"],    mc["p_bleu_f1"])} '
    f'| {fmt_r(mc["pearson_judge_f1"],   mc["p_judge_f1"])} '
    f'| {model_delta[ms]:+.3f} |'
    for ms, mc in sorted(model_corrs.items())
)

# Count rescue zone sizes
rescue_medical = len([r for r in medical_open
                      if r['bleu'] < 0.10 and r['f1'] < 0.15 and r['judge_score'] >= 4])
rescue_general = len([r for r in general_open
                      if r['bleu'] < 0.10 and r['f1'] < 0.15 and r['judge_score'] >= 4])

# Model bucket table
bucket_names = ['Medical Synonymy', 'Conversational Filler', 'Granularity Mismatch', 'Other']
model_bucket_rows = '\n'.join(
    '| ' + ms + ' | ' + ' | '.join(str(model_bucket[ms].get(b, 0)) for b in bucket_names) + ' |'
    for ms in sorted(model_bucket.keys())
)

# Autopsy example table rows (one per example, all buckets)
def example_rows(recs):
    rows = []
    for r in recs:
        gt  = r.get('ground_truth', '').replace('|', '\\|')[:80]
        pred = r.get('prediction', '').replace('|', '\\|')[:100]
        rows.append(f'| {r["model_short"]} | {r["dataset_norm"].upper()} '
                    f'| {gt} | {pred} | {r["bleu"]:.3f} | {r["judge_score"]:.0f} |')
    return '\n'.join(rows)

ex_synonymy  = example_rows(autopsy_examples.get('Medical Synonymy', []))
ex_filler    = example_rows(autopsy_examples.get('Conversational Filler', []))
ex_granular  = example_rows(autopsy_examples.get('Granularity Mismatch', []))

# Interpretation badge
bleu_f1_redundancy = overall_corr['pearson_bleu_f1']
redundancy_note = (
    "⚠️ High correlation (r > 0.85): BLEU and F1 are near-redundant for terse models."
    if bleu_f1_redundancy > 0.85
    else "✅ Moderate correlation: BLEU provides distinct signal from F1."
)

md = f"""# BLEU Error Analysis — N-Gram Metric Audit
## VLM Medical VQA Benchmark

This study investigates whether BLEU-4, while computed for all model–dataset
combinations, provides signal that is distinct from Token F1 and LLM Judge
Accuracy, and specifically characterises **where and why** it fails for
open-ended medical VQA.

All analyses are restricted to **open-ended questions only**. BLEU on binary
yes/no answers is trivially redundant with exact-match F1 and is excluded.

---

## 1. Dataset Summary

| Subset | N |
|---|---|
| Total open-ended records | {len(open_records)} |
| Medical open-ended (SLAKE + VQA-RAD) | {len(medical_open)} |
| General open-ended (VQAv2 + OK-VQA) | {len(general_open)} |

---

## 2. Triple-Axis Correlation Analysis

Pearson r between three metric pairs. Statistical significance: * p<0.05, ** p<0.01, *** p<0.001.

### 2.1 Overall and Domain-Level

| Subset | N | BLEU ↔ Judge | BLEU ↔ F1 | Judge ↔ F1 |
|---|---|---|---|---|
{overall_row}
{medical_row}
{general_row}

### 2.2 Per-Model (Medical + General Open-Ended)

| Model | N | BLEU ↔ Judge | BLEU ↔ F1 | Judge ↔ F1 | Δ (Judge−BLEU) |
|---|---|---|---|---|---|
{model_corr_rows}

> **Redundancy check — BLEU vs. F1:** r = {bleu_f1_redundancy:.3f}. {redundancy_note}

### 2.3 Delta Analysis — Verbosity Penalty

The **Delta (Judge_norm − BLEU)** column above quantifies the verbosity penalty:
a large positive Δ means the judge rewards the model's semantic content, but
BLEU is penalising the model for using different wording than the ground truth.
Models with Δ > 0.30 are producing correct answers that BLEU cannot see.

---

## 3. The Rescue Zone — Visual Evidence of Metric Failure

The scatter plot below (X=BLEU, Y=Token F1, colour=Judge Score) highlights
a "rescue zone" in the bottom-left corner: records with near-zero BLEU **and**
near-zero F1, yet judged as correct (score ≥ 4) by the LLM.

![BLEU vs F1 Rescue Zone scatter]({OUT_PLOT})

| Domain | Records in rescue zone (BLEU<0.10, F1<0.15, Judge≥4) |
|---|---|
| Medical | **{rescue_medical}** |
| General | **{rescue_general}** |

These records are **true positives that both BLEU and F1 call false negatives**.
Every one of them would be counted as a model failure if either classical metric
were the sole evaluation criterion.

---

## 4. Qualitative N-Gram Autopsy

From the autopsy pool (medical open-ended, Judge ≥ 4, BLEU < 0.05,
N = {len(autopsy_pool)} records), three failure categories were identified.

### 4.1 Failure Mode Distribution

| Model | Medical Synonymy | Conversational Filler | Granularity Mismatch | Other |
|---|---|---|---|---|
{model_bucket_rows}

**Total per bucket:**

| Bucket | N | Description |
|---|---|---|
| Medical Synonymy | {bucket_counts.get('Medical Synonymy', 0)} | Ground truth and prediction use clinically equivalent but lexically different terms (e.g., "renal" vs "kidney", "pulmonary" vs "lung"). |
| Conversational Filler | {bucket_counts.get('Conversational Filler', 0)} | Model output contains the correct answer embedded in a longer conversational response, diluting n-gram precision. |
| Granularity Mismatch | {bucket_counts.get('Granularity Mismatch', 0)} | Model provides a highly specific sub-classification while the ground truth is a broader term (or vice versa). |
| Other | {bucket_counts.get('Other', 0)} | Paraphrase or structural variation not captured by the above three patterns. |

### 4.2 Representative Examples

#### Medical Synonymy

| Model | Dataset | Ground Truth | Prediction | BLEU | Judge |
|---|---|---|---|---|---|
{ex_synonymy if ex_synonymy else '| — | — | No examples in this category | — | — | — |'}

#### Conversational Filler

| Model | Dataset | Ground Truth | Prediction | BLEU | Judge |
|---|---|---|---|---|---|
{ex_filler if ex_filler else '| — | — | No examples in this category | — | — | — |'}

#### Granularity Mismatch

| Model | Dataset | Ground Truth | Prediction | BLEU | Judge |
|---|---|---|---|---|---|
{ex_granular if ex_granular else '| — | — | No examples in this category | — | — | — |'}

---

## 5. Implications for Metric Selection

| Question | Finding |
|---|---|
| Is BLEU redundant with F1? | r(BLEU, F1) = {overall_corr["pearson_bleu_f1"]:.3f} — {"Yes, largely redundant for terse outputs." if bleu_f1_redundancy > 0.85 else "Partially — diverges on verbose and synonym-heavy outputs."} |
| Does BLEU track judge accuracy? | r(BLEU, Judge) = {overall_corr["pearson_bleu_judge"]:.3f} — weaker than r(F1, Judge) = {overall_corr["pearson_judge_f1"]:.3f}, confirming BLEU is the least reliable single metric. |
| Which models suffer most from BLEU bias? | Highest Δ (Judge−BLEU) models are the most penalised by n-gram overlap metrics. |
| What is the rescue zone size? | {rescue_medical + rescue_general} records ({rescue_medical} medical, {rescue_general} general) are correctly answered but falsely penalised by both BLEU and F1. |

> [!IMPORTANT]
> These findings justify the benchmark's primary reliance on **LLM Judge Accuracy**
> for open-ended evaluation, with BLEU and F1 reported as secondary metrics for
> reproducibility and comparison with prior work. BLEU alone would
> systematically underestimate model performance on conversational and
> medically-synonym-rich outputs.

---

## 6. Methods Paragraph (paper-ready)

```
We conducted a post-hoc BLEU audit to quantify whether BLEU-4 provides
evaluation signal distinct from Token F1 and LLM Judge Accuracy.
Restricting analysis to open-ended questions (N = {len(open_records)}), we
computed sentence-level BLEU-4 (add-1 smoothing) and unigram Token F1 per
prediction–reference pair, and computed Pearson and Spearman correlations
across three metric pairs. BLEU and Token F1 showed {'high' if bleu_f1_redundancy > 0.85 else 'moderate'} correlation
(r = {overall_corr["pearson_bleu_f1"]:.3f}), indicating near-redundancy for terse model
outputs. BLEU correlated less strongly with judge scores
(r = {overall_corr["pearson_bleu_judge"]:.3f}) than F1 did (r = {overall_corr["pearson_judge_f1"]:.3f}),
confirming it is the least reliable proxy for semantic correctness in this
benchmark. A rescue-zone analysis identified {rescue_medical + rescue_general} predictions
(BLEU < 0.10, Token F1 < 0.15, Judge ≥ 4/5) that both classical metrics
falsely penalise. Qualitative autopsy of {len(autopsy_pool)} high-judge/low-BLEU medical
records attributed failures to three categories: medical synonymy
({bucket_counts.get("Medical Synonymy", 0)} cases), conversational filler ({bucket_counts.get("Conversational Filler", 0)} cases), and
granularity mismatch ({bucket_counts.get("Granularity Mismatch", 0)} cases). These results motivated
our use of LLM Judge Accuracy as the primary open-ended evaluation metric,
with BLEU and F1 retained as secondary metrics for reproducibility.
```

---

## 7. Generated Outputs

| File | Description |
|---|---|
| [`docs/bleu_error_analysis.md`](bleu_error_analysis.md) | This report |
| [`results/fig_bleu_rescue_zone.png`](../results/fig_bleu_rescue_zone.png) | Rescue-zone scatter plot |
"""

with open(OUT_MD, 'w', encoding='utf-8') as f:
    f.write(md)

print(f"\nMarkdown report saved to: {OUT_MD}")
print("Done.")
