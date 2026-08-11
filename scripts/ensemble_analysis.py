"""
Ensemble / Model Combination Analysis
======================================
Strategy:
  - Closed (Yes/No): majority vote across selected models.
    Tie-breaking: prefer the best single model (MedGemma-4B).
  - Open: pick the prediction from whichever model has the highest
    judge score for that question (oracle-on-judge ensemble).

Combinations evaluated:
  E1 — All-5:       all five models
  E2 — Med-2:       MedGemma-4B + HuatuoGPT-7B
  E3 — Med-3:       MedGemma-4B + HuatuoGPT-7B + LLaVA-Med-7B
  E4 — Med-3+Gemma: MedGemma-4B + HuatuoGPT-7B + LLaVA-Med-7B + Gemma-3-4B
"""

import json, re, os, collections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SLAKE_INF = {
    'MedGemma-4B':  'outputs/inference/google_medgemma-4b-it__slake_v2.jsonl',
    'Gemma-3-4B':   'outputs/inference/google_gemma-3-4b-it__slake_v2.jsonl',
    'HuatuoGPT-7B': 'outputs/inference/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_7b_v2.jsonl',
    'LLaVA-1.6-7B': 'outputs/inference/llava-hf_llava-v1.6-mistral-7b-hf__slake_7b_v2.jsonl',
    'LLaVA-Med-7B': 'outputs/inference/microsoft_llava-med-v1.5-mistral-7b__slake_7b_v2-2.jsonl',
}
SLAKE_JUDGE = {
    'MedGemma-4B':  'outputs/judge/google_medgemma-4b-it__slake_v2_judged.jsonl',
    'Gemma-3-4B':   'outputs/judge/google_gemma-3-4b-it__slake_v2_judged.jsonl',
    'HuatuoGPT-7B': 'outputs/judge/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__slake_7b_v2_judged.jsonl',
    'LLaVA-1.6-7B': 'outputs/judge/llava-hf_llava-v1.6-mistral-7b-hf__slake_7b_v2_judged.jsonl',
    'LLaVA-Med-7B': 'outputs/judge/microsoft_llava-med-v1.5-mistral-7b__slake_7b_v2-2_judged.jsonl',
}
VRAD_INF = {
    'MedGemma-4B':  'outputs/inference/google_medgemma-4b-it__vqa_rad_v2.jsonl',
    'Gemma-3-4B':   'outputs/inference/google_gemma-3-4b-it__vqa_rad_v2.jsonl',
    'HuatuoGPT-7B': 'outputs/inference/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__vqa_rad_7b_v2.jsonl',
    'LLaVA-1.6-7B': 'outputs/inference/llava-hf_llava-v1.6-mistral-7b-hf__vqa_rad_7b_v2.jsonl',
    'LLaVA-Med-7B': 'outputs/inference/microsoft_llava-med-v1.5-mistral-7b__vqa_rad_7b_v2.jsonl',
}
VRAD_JUDGE = {
    'MedGemma-4B':  'outputs/judge/google_medgemma-4b-it__vqa_rad_v2_judged.jsonl',
    'Gemma-3-4B':   'outputs/judge/google_gemma-3-4b-it__vqa_rad_v2_judged.jsonl',
    'HuatuoGPT-7B': 'outputs/judge/FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__vqa_rad_7b_v2_judged.jsonl',
    'LLaVA-1.6-7B': 'outputs/judge/llava-hf_llava-v1.6-mistral-7b-hf__vqa_rad_7b_v2_judged.jsonl',
    'LLaVA-Med-7B': 'outputs/judge/microsoft_llava-med-v1.5-mistral-7b__vqa_rad_7b_v2_judged.jsonl',
}

ENSEMBLES = {
    'E1 All-5':       ['MedGemma-4B','HuatuoGPT-7B','Gemma-3-4B','LLaVA-Med-7B','LLaVA-1.6-7B'],
    'E2 Med-2':       ['MedGemma-4B','HuatuoGPT-7B'],
    'E3 Med-3':       ['MedGemma-4B','HuatuoGPT-7B','LLaVA-Med-7B'],
    'E4 Med3+Gemma':  ['MedGemma-4B','HuatuoGPT-7B','LLaVA-Med-7B','Gemma-3-4B'],
}
TIEBREAK = 'MedGemma-4B'


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

def norm_yn(t):
    t = norm(str(t))
    if t.startswith('yes'): return 'yes'
    if t.startswith('no'):  return 'no'
    return t.split()[0] if t.split() else ''

def load_inf(path):
    out = {}
    with open(os.path.join(BASE, path)) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                out[int(r['idx'])] = r
            except Exception: pass
    return out

def load_judge(path):
    out = {}
    full = os.path.join(BASE, path)
    if not os.path.exists(full): return out
    with open(full) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                out[int(r['idx'])] = float(r.get('judge_score', 0) or 0)
            except Exception: pass
    return out

def score_single(inf_map, judge_map):
    f1_all, f1_cl, f1_op, cls_corr = [], [], [], []
    j_all, j_cl, j_op = [], [], []
    for idx, r in sorted(inf_map.items()):
        pred = r.get('prediction','') or r.get('raw_output','') or ''
        gt   = r.get('ground_truth','')
        is_cl = r.get('is_closed') in [True,'True']
        f1 = token_f1(pred, gt)
        f1_all.append(f1)
        if is_cl:
            f1_cl.append(f1)
            cls_corr.append(1 if norm_yn(pred)==norm_yn(gt) else 0)
        else:
            f1_op.append(f1)
        if idx in judge_map:
            jc = 1 if judge_map[idx] >= 4.0 else 0
            j_all.append(jc)
            (j_cl if is_cl else j_op).append(jc)
    def a(l): return sum(l)/len(l)*100 if l else 0.0
    return dict(f1_all=a(f1_all), f1_cl=a(f1_cl), f1_op=a(f1_op),
                cls_acc=a(cls_corr), judge_all=a(j_all),
                judge_cl=a(j_cl), judge_op=a(j_op), n=len(inf_map))

def score_ensemble(members, all_inf, all_judge):
    idx_sets = [set(all_inf[m].keys()) for m in members if m in all_inf]
    common = sorted(set.intersection(*idx_sets))
    f1_all, f1_cl, f1_op, cls_corr = [], [], [], []
    j_all, j_cl, j_op = [], [], []

    for idx in common:
        ref  = all_inf[members[0]][idx]
        gt   = ref.get('ground_truth','')
        is_cl = ref.get('is_closed') in [True,'True']

        if is_cl:
            votes = [norm_yn(all_inf[m][idx].get('prediction','') or
                             all_inf[m][idx].get('raw_output',''))
                     for m in members if m in all_inf]
            cnt = collections.Counter(votes)
            top_n = cnt.most_common(1)[0][1]
            winners = [v for v,c in cnt.items() if c==top_n]
            if len(winners)==1:
                pred = winners[0]
            else:
                tb = norm_yn(all_inf[TIEBREAK][idx].get('prediction','')) if TIEBREAK in all_inf else ''
                pred = tb if tb in winners else winners[0]
            f1 = token_f1(pred, gt)
            f1_cl.append(f1); f1_all.append(f1)
            cls_corr.append(1 if pred==norm_yn(gt) else 0)
            # judge from tiebreak model
            bm = TIEBREAK if TIEBREAK in members else members[0]
            js = all_judge[bm].get(idx)
            if js is not None:
                jc = 1 if js>=4.0 else 0
                j_cl.append(jc); j_all.append(jc)
        else:
            best_s, best_m = -1, members[0]
            for m in members:
                if m in all_judge:
                    s = all_judge[m].get(idx, -1)
                    if s > best_s: best_s, best_m = s, m
            pred = (all_inf[best_m][idx].get('prediction','') or
                    all_inf[best_m][idx].get('raw_output','') or '')
            f1 = token_f1(pred, gt)
            f1_op.append(f1); f1_all.append(f1)
            if best_s >= 0:
                jc = 1 if best_s>=4.0 else 0
                j_op.append(jc); j_all.append(jc)

    def a(l): return sum(l)/len(l)*100 if l else 0.0
    return dict(f1_all=a(f1_all), f1_cl=a(f1_cl), f1_op=a(f1_op),
                cls_acc=a(cls_corr), judge_all=a(j_all),
                judge_cl=a(j_cl), judge_op=a(j_op), n=len(common))

def run(inf_paths, judge_paths, label):
    print(f'\n{"="*72}\n  {label}\n{"="*72}')
    all_inf   = {m: load_inf(p)   for m,p in inf_paths.items()}
    all_judge = {m: load_judge(p) for m,p in judge_paths.items()}
    rows = {}
    hdr = f'  {"Name":<22} {"F1":>7} {"ClsAcc":>8} {"OpenF1":>8} {"Judge":>8} {"J-Cls":>7} {"J-Opn":>7}'
    sep = '  ' + '-'*72
    print(f'\n  Individual Baselines\n{sep}\n{hdr}\n{sep}')
    for m in inf_paths:
        s = score_single(all_inf[m], all_judge[m])
        rows[m] = s
        print(f'  {m:<22} {s["f1_all"]:>6.2f}% {s["cls_acc"]:>7.2f}% {s["f1_op"]:>7.2f}% {s["judge_all"]:>7.2f}% {s["judge_cl"]:>6.2f}% {s["judge_op"]:>6.2f}%')
    print(f'\n  Ensembles\n{sep}\n{hdr}\n{sep}')
    for ename, members in ENSEMBLES.items():
        avail = [m for m in members if m in all_inf]
        if len(avail) < 2: continue
        s = score_ensemble(avail, all_inf, all_judge)
        rows[ename] = s
        print(f'  {ename:<22} {s["f1_all"]:>6.2f}% {s["cls_acc"]:>7.2f}% {s["f1_op"]:>7.2f}% {s["judge_all"]:>7.2f}% {s["judge_cl"]:>6.2f}% {s["judge_op"]:>6.2f}%')
    return rows

if __name__ == '__main__':
    slake = run(SLAKE_INF, SLAKE_JUDGE, 'SLAKE (N=1,061)')
    vrad  = run(VRAD_INF,  VRAD_JUDGE,  'VQA-RAD (N=451)')
    out_path = os.path.join(BASE, 'results', 'ensemble_results.json')
    with open(out_path, 'w') as f:
        json.dump({'slake': slake, 'vqa_rad': vrad}, f, indent=2)
    print(f'\nSaved: {out_path}')
