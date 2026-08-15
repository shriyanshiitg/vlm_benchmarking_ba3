"""
HuatuoGPT Corrected Inference Analysis
=======================================
Compares v2_baseline (existing) vs v3_corrected (new) on the full SLAKE EN
test set (1,061 questions). Also runs LLM Judge scoring if judge file exists.

Usage:
  python3 scripts/huatuo_corrected_analysis.py

Expects:
  outputs/inference/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_v3_corrected.jsonl
  (place the Kaggle output here before running)
"""

import json, re, os, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V2_PATH = os.path.join(BASE, 'outputs', 'inference',
    'FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_7b_v2.jsonl')
V3_PATH = os.path.join(BASE, 'outputs', 'inference',
    'FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_v3_corrected.jsonl')
V3_JUDGE_PATH = os.path.join(BASE, 'outputs', 'judge',
    'FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_v3_corrected_judged.jsonl')

# ── Helpers ───────────────────────────────────────────────────────────────────
def norm(t):
    t = re.sub(r'[^\w\s]', ' ', str(t).lower())
    return re.sub(r'\s+', ' ', t).strip()

def token_f1(pred, gt):
    p, g = norm(pred).split(), norm(gt).split()
    if not p or not g: return 0.0
    pc, gc = collections.Counter(p), collections.Counter(g)
    common = sum((pc & gc).values())
    return 2 * common / (len(p) + len(g)) if common else 0.0

def load_jsonl(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                out[int(r['idx'])] = r
            except: pass
    return out

def score(records, judge_map=None):
    f1s, cls_c, opn = [], [], []
    j_all, j_cl, j_op = [], [], []
    ct_f1 = collections.defaultdict(list)

    for idx, r in sorted(records.items()):
        pred = r.get('prediction', '') or r.get('raw_output', '') or ''
        gt   = r.get('ground_truth', '')
        is_cl = r.get('is_closed') in [True, 'True'] or r.get('answer_type') == 'CLOSED'
        f1 = token_f1(pred, gt)
        f1s.append(f1)
        ct_f1[r.get('content_type', 'unknown')].append(f1)

        if is_cl:
            p0 = norm(pred).split()
            g0 = norm(gt).split()
            cls_c.append(1 if (p0 and g0 and p0[0] == g0[0]) else 0)
        else:
            opn.append(f1)

        if judge_map and idx in judge_map:
            jc = 1 if float(judge_map[idx].get('judge_score', 0) or 0) >= 4.0 else 0
            j_all.append(jc)
            (j_cl if is_cl else j_op).append(jc)

    def a(l): return sum(l) / len(l) * 100 if l else 0.0
    return {
        'n': len(records),
        'f1': a(f1s), 'cls_acc': a(cls_c), 'open_f1': a(opn),
        'judge_all': a(j_all), 'judge_cl': a(j_cl), 'judge_op': a(j_op),
        'ct_f1': {ct: a(vals) for ct, vals in ct_f1.items()},
    }

# ── Load files ────────────────────────────────────────────────────────────────
v2 = load_jsonl(V2_PATH)
v3 = load_jsonl(V3_PATH)
v3_judge = load_jsonl(V3_JUDGE_PATH)

if not v2:
    print(f"ERROR: v2 file missing at {V2_PATH}")
    exit(1)
if not v3:
    print(f"ERROR: v3 corrected file missing at {V3_PATH}")
    print("Place the Kaggle output file there and re-run.")
    exit(1)

print(f"v2 records: {len(v2)}")
print(f"v3 records: {len(v3)}")
print(f"v3 judge:   {len(v3_judge)} {'(loaded)' if v3_judge else '(not yet available)'}")

# ── Score ─────────────────────────────────────────────────────────────────────
s2 = score(v2)
s3 = score(v3, v3_judge if v3_judge else None)

print(f"\n{'='*65}")
print(f"  HuatuoGPT-7B — SLAKE — v2 vs v3_corrected")
print(f"{'='*65}")
print(f"\n  {'Metric':<20} {'v2 (baseline)':>16} {'v3 corrected':>14} {'Δ':>10}")
print("  " + "-" * 62)
for metric, label in [('f1','Token F1'), ('cls_acc','Closed Acc'), ('open_f1','Open F1'),
                       ('judge_all','Judge Acc'), ('judge_cl','Judge Closed'), ('judge_op','Judge Open')]:
    v2v = s2[metric]
    v3v = s3[metric]
    if v3v == 0 and metric.startswith('judge') and not v3_judge:
        delta = '(judge pending)'
        print(f"  {label:<20} {v2v:>14.2f}% {'—':>14}  {delta:>10}")
    else:
        delta = f"{v3v - v2v:+.2f} pp"
        marker = ' ✓' if v3v > v2v else ''
        print(f"  {label:<20} {v2v:>14.2f}% {v3v:>13.2f}%  {delta:>10}{marker}")

# ── Per content-type ──────────────────────────────────────────────────────────
print(f"\n  Per Content-Type Token F1\n  {'Content Type':<16} {'v2':>8} {'v3':>8} {'Δ':>8}")
print("  " + "-" * 42)
all_cts = sorted(set(list(s2['ct_f1'].keys()) + list(s3['ct_f1'].keys())))
for ct in all_cts:
    v = s2['ct_f1'].get(ct, 0)
    w = s3['ct_f1'].get(ct, 0)
    print(f"  {ct:<16} {v:>7.1f}% {w:>7.1f}% {w-v:>+7.1f}pp")

# ── Chart ─────────────────────────────────────────────────────────────────────
metrics_label = ['Token F1', 'Closed Acc', 'Open F1']
metrics_key   = ['f1', 'cls_acc', 'open_f1']
v2_vals = [s2[k] for k in metrics_key]
v3_vals = [s3[k] for k in metrics_key]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0F172A')
ax.set_facecolor('#1E293B')
for sp in ax.spines.values(): sp.set_color('#334155')

x = np.arange(len(metrics_label))
w = 0.35
b1 = ax.bar(x - w/2, v2_vals, w, color='#475569', edgecolor='#0F172A', label='v2 baseline (Section 11)', alpha=0.85)
b2 = ax.bar(x + w/2, v3_vals, w, color='#2563EB', edgecolor='#0F172A', label='v3_simple corrected', alpha=0.90)

for bar, v in zip(b1, v2_vals): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, color='#CBD5E1')
for bar, v in zip(b2, v3_vals): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=10, color='#93C5FD', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metrics_label, color='#E2E8F0', fontsize=12)
ax.set_ylim(0, 100)
ax.set_ylabel('Score (%)', color='#94A3B8', fontsize=12)
ax.set_title('HuatuoGPT-7B — SLAKE: v2 Baseline vs v3_simple Corrected',
             color='#F1F5F9', fontsize=13, fontweight='bold', pad=12)
ax.grid(axis='y', color='#334155', linewidth=0.5, linestyle='--', alpha=0.5)
ax.tick_params(colors='#CBD5E1')
ax.legend(fontsize=10, framealpha=0.2, facecolor='#1E293B', edgecolor='#475569', labelcolor='#E2E8F0')

plt.tight_layout()
out_img = os.path.join(BASE, 'results', 'fig_huatuo_v3_corrected.png')
plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor='#0F172A')
plt.close()
print(f"\nChart saved: {out_img}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out_json = os.path.join(BASE, 'results', 'huatuo_corrected_results.json')
with open(out_json, 'w') as f:
    json.dump({'v2': s2, 'v3_corrected': s3}, f, indent=2)
print(f"JSON saved: {out_json}")
