import json, os, re, string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sacrebleu.metrics import BLEU
import torch
from evaluate import load as load_metric
import warnings
warnings.filterwarnings('ignore')

# ── Number normalisation map ───────────────────────────────────────────
NUM_WORDS = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5',
    'six':'6','seven':'7','eight':'8','nine':'9','ten':'10',
    'eleven':'11','twelve':'12','thirteen':'13','fourteen':'14',
    'fifteen':'15','sixteen':'16','seventeen':'17','eighteen':'18',
    'nineteen':'19','twenty':'20'
}

def normalize_numbers(text: str) -> str:
    tokens = text.lower().split()
    return ' '.join(NUM_WORDS.get(t, t) for t in tokens)

def tokenize_answer_fixed(text: str) -> list:
    text = re.sub(r'\\*+', '', text).lower()
    text = normalize_numbers(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)

def token_f1_fixed(prediction: str, ground_truth: str) -> dict:
    pred_tokens = tokenize_answer_fixed(prediction)
    gt_tokens   = tokenize_answer_fixed(ground_truth)
    if not pred_tokens or not gt_tokens:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    pred_set = Counter(pred_tokens)
    gt_set   = Counter(gt_tokens)
    common   = sum((pred_set & gt_set).values())
    precision = common / len(pred_tokens)
    recall    = common / len(gt_tokens)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {'f1': f1, 'precision': precision, 'recall': recall}

def is_correct_fixed(prediction: str, ground_truth: str, is_closed: bool) -> bool:
    scores    = token_f1_fixed(prediction, ground_truth)
    threshold = 0.5 if is_closed else 0.75
    return scores['recall'] >= threshold

def extract_final_answer_fixed(text: str) -> str:
    clean = re.sub(r'\\*+', '', text).strip()
    match = re.search(r'[Ff]inal\\s+[Aa]nswer\\s*:\\s*(.+)', clean, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        answer = answer.split('\\n')[0].strip()
        answer = answer.strip(string.punctuation + ' ')
        return answer
    return re.split(r'(?<=[.!?])\\s', clean)[0].strip()

bleu_metric_fixed = BLEU(effective_order=True)

def compute_bleu_fixed(prediction: str, ground_truth: str) -> float:
    return bleu_metric_fixed.sentence_score(
        hypothesis=prediction.lower(),
        references=[ground_truth.lower()]
    ).score

bertscore = load_metric("bertscore")

def rescore_general_jsonl(path: str) -> dict:
    records  = [json.loads(l) for l in open(path)]
    records  = [r for r in records if 'error' not in r]

    # Re-extract and score
    rescored = []
    for r in records:
        raw      = r.get('raw_output', '')
        new_pred = extract_final_answer_fixed(raw) if raw else r['prediction']
        corrected = dict(r)
        corrected['prediction'] = new_pred
        rescored.append(corrected)

    closed   = [r for r in rescored if r.get('is_closed', False)]
    open_    = [r for r in rescored if not r.get('is_closed', False)]

    def avg_f1(recs):
        if not recs: return 0.0
        return sum(token_f1_fixed(r['prediction'], r['ground_truth'])['f1']
                   for r in recs) / len(recs)

    def accuracy(recs, is_closed):
        if not recs: return 0.0
        return sum(is_correct_fixed(r['prediction'], r['ground_truth'], is_closed)
                   for r in recs) / len(recs)

    def avg_bleu(recs):
        if not recs: return 0.0
        return sum(compute_bleu_fixed(r['prediction'], r['ground_truth'])
                   for r in recs) / len(recs)
                   
    # Compute BERTScore
    preds = [r['prediction'] for r in rescored]
    refs = [r['ground_truth'] for r in rescored]
    
    if len(preds) > 0:
        bs_results = bertscore.compute(predictions=preds, references=refs, model_type="roberta-large", device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        bertscore_f1 = sum(bs_results['f1']) / len(bs_results['f1'])
    else:
        bertscore_f1 = 0.0

    return {
        'dataset':      os.path.basename(path),
        'n_total':      len(rescored),
        'overall_f1':   round(avg_f1(rescored) * 100, 2),
        'closed_acc':   round(accuracy(closed, True)  * 100, 2),
        'open_acc':     round(accuracy(open_,  False) * 100, 2),
        'bleu':         round(avg_bleu(rescored), 2),
        'bertscore_f1': round(bertscore_f1 * 100, 2)
    }

OUTPUT_DIR = '/Users/shriyanshraj/vlm_benchmark/outputs/inference'

files_to_score = [
    'google_medgemma-4b-it__okvqa_v2-2.jsonl',
    'chaoyinshe_llava-med-v1.5-mistral-7b-hf__okvqa_v2-2.jsonl',
    'FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__vqav2_v2.jsonl',
    'FreedomIntelligence_HuatuoGPT-Vision-7B-Qwen2.5VL__okvqa_v2.jsonl'
]

print("Calculating metrics...")
for fname in files_to_score:
    path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(path):
        scores = rescore_general_jsonl(path)
        print(f"\\nFile: {fname}")
        print(f"F1: {scores['overall_f1']}% | Cls Acc: {scores['closed_acc']}% | Opn Acc: {scores['open_acc']}% | BLEU: {scores['bleu']} | BERTScore: {scores['bertscore_f1']}%")
    else:
        print(f"File not found: {path}")

