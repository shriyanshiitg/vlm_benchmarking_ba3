"""
Prompt Template Sensitivity Analysis
======================================
Run after downloading Kaggle outputs into:
  outputs/_archive/prompt_sensitivity/

Computes Token F1, Closed Accuracy, Open F1 per variant per model,
generates a comparison bar chart, and produces the delta table
vs the v2 baseline.
"""

import json, re, os, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR   = os.path.join(BASE, 'outputs', '_archive', 'prompt_sensitivity')
OUT_DIR  = os.path.join(BASE, 'results')
os.makedirs(IN_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = {
    'HuatuoGPT-7B': 'FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL',
    'LLaVA-Med-7B': 'microsoft_llava-med-v1.5-mistral-7b',
}
VARIANTS = ['v2_baseline', 'v3_simple', 'v4_direct']

def norm(t):
    t = re.sub(r'[^\w\s]', ' ', str(t).lower())
    return re.sub(r'\s+', ' ', t).strip()

def token_f1(pred, gt):
    p, g = norm(pred).split(), norm(gt).split()
    if not p or not g: return 0.0
    pc, gc = collections.Counter(p), collections.Counter(g)
    common = sum((pc & gc).values())
    if common == 0: return 0.0
    return 2 * common / (len(p) + len(g))

def score_file(path):
    records = [json.loads(l) for l in open(path)]
    f1s, cls, opn = [], [], []
    for r in records:
        f1 = token_f1(r.get('prediction', ''), r.get('ground_truth', ''))
        f1s.append(f1)
        if r.get('is_closed'):
            p0 = norm(r.get('prediction','')).split()
            g0 = norm(r.get('ground_truth','')).split()
            cls.append(1 if (p0 and g0 and p0[0]==g0[0]) else 0)
        else:
            opn.append(f1)
    avg = lambda l: sum(l)/len(l)*100 if l else 0.0
    return {'f1': avg(f1s), 'cls_acc': avg(cls), 'open_f1': avg(opn), 'n': len(records)}

# ── Load results ──────────────────────────────────────────────────────────────
results = {}
print(f"\n{'='*65}")
print(f"  Prompt Template Sensitivity — Results")
print(f"{'='*65}")

for model_name, safe in MODELS.items():
    results[model_name] = {}
    print(f"\n  {model_name}")
    print(f"  {'Variant':<16} {'F1':>7} {'ClsAcc':>8} {'OpenF1':>8}  {'ΔF1':>7}")
    print("  " + "-" * 50)
    baseline_f1 = None
    for var in VARIANTS:
        fname = f'{safe}__{var}.jsonl'
        path  = os.path.join(IN_DIR, fname)
        if not os.path.exists(path):
            print(f"  {var:<16}  MISSING: {path}")
            results[model_name][var] = None
            continue
        s = score_file(path)
        results[model_name][var] = s
        if var == 'v2_baseline':
            baseline_f1 = s['f1']
        delta = f"{s['f1']-baseline_f1:+.2f}pp" if baseline_f1 is not None and var != 'v2_baseline' else "—"
        print(f"  {var:<16} {s['f1']:>6.2f}% {s['cls_acc']:>7.2f}% {s['open_f1']:>7.2f}%  {delta:>7}")

# ── Chart ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0F172A')

VARIANT_LABELS = {'v2_baseline': 'v2 Baseline\n(Final Answer: X)', 
                  'v3_simple': 'v3 Simple\n(Answer concisely)', 
                  'v4_direct': 'v4 Direct\n(Answer: prefix)'}
COLORS = {'f1': '#2563EB', 'cls_acc': '#16A34A', 'open_f1': '#D97706'}

for ax, model_name in zip(axes, MODELS.keys()):
    ax.set_facecolor('#1E293B')
    for sp in ax.spines.values(): sp.set_color('#334155')
    ax.tick_params(colors='#CBD5E1')

    data = results.get(model_name, {})
    available = [v for v in VARIANTS if data.get(v) is not None]
    if not available:
        ax.text(0.5, 0.5, 'No data yet\n(run on Kaggle first)', ha='center', va='center',
                color='#94A3B8', fontsize=12, transform=ax.transAxes)
        ax.set_title(model_name, color='#F1F5F9', fontsize=13, fontweight='bold')
        continue

    x     = np.arange(len(available))
    width = 0.26
    labels = [VARIANT_LABELS[v] for v in available]

    for j, (metric, color, label) in enumerate([('f1', '#2563EB', 'Token F1'),
                                                  ('cls_acc', '#16A34A', 'Closed Acc'),
                                                  ('open_f1', '#D97706', 'Open F1')]):
        vals = [data[v][metric] for v in available]
        bars = ax.bar(x + (j-1)*width, vals, width, color=color, alpha=0.85,
                      edgecolor='#0F172A', linewidth=0.5, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8,
                    color='#E2E8F0', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color='#E2E8F0', fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Score (%)', color='#94A3B8', fontsize=11)
    ax.set_title(f'{model_name} — Prompt Sensitivity (SLAKE N=200)',
                 color='#F1F5F9', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='y', color='#334155', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.2,
              facecolor='#1E293B', edgecolor='#475569', labelcolor='#E2E8F0')

plt.suptitle('Prompt Template Sensitivity Study — SLAKE 200-sample subset',
             color='#F1F5F9', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out_img = os.path.join(OUT_DIR, 'fig_prompt_sensitivity.png')
plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor='#0F172A')
plt.close()
print(f"\nChart saved: {out_img}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
out_json = os.path.join(OUT_DIR, 'prompt_sensitivity_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f"JSON saved: {out_json}")
print("\nDone. Now run this after downloading Kaggle outputs.")
