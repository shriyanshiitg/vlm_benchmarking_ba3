"""
calibration_analysis.py
=======================
Local analysis script for the calibration experiment (Notebook 11).

Run AFTER placing all 4 output JSONL files in:
    outputs/_archive/calibration/

Expected files:
    medgemma_4b__slake__calibration.jsonl
    medgemma_4b__vqa_rad__calibration.jsonl
    gemma3_4b__slake__calibration.jsonl
    gemma3_4b__vqa_rad__calibration.jsonl

Usage:
    python3 scripts/calibration_analysis.py

Outputs:
    results/fig_calibration_reliability.png    2×2 reliability diagram grid
    results/fig_calibration_confidence_hist.png Confidence distribution histograms
    results/calibration_results.json           All ECE + metrics (machine-readable)
    docs/report_calibration.md                 Full analysis report
"""

import json, os, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print("matplotlib not found — skipping chart generation. Install with: pip install matplotlib")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_DIR    = os.path.join(BASE, "outputs", "_archive", "calibration")
RESULTS_DIR  = os.path.join(BASE, "results")
DOCS_DIR     = os.path.join(BASE, "docs")

os.makedirs(CALIB_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── File registry ──────────────────────────────────────────────────────────────
FILES = {
    ("MedGemma-4B", "slake"):   "medgemma_4b__slake__calibration.jsonl",
    ("MedGemma-4B", "vqa_rad"): "medgemma_4b__vqa_rad__calibration.jsonl",
    ("Gemma-3-4B",  "slake"):   "gemma3_4b__slake__calibration.jsonl",
    ("Gemma-3-4B",  "vqa_rad"): "gemma3_4b__vqa_rad__calibration.jsonl",
}

DATASET_LABELS = {"slake": "SLAKE", "vqa_rad": "VQA-RAD"}
MODEL_COLORS   = {"MedGemma-4B": "#1a73e8", "Gemma-3-4B": "#ea4335"}
N_BINS         = 15   # equal-width bins from 0 to 1

print("=" * 65)
print("Calibration Analysis")
print("=" * 65)

# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_ece(confidences, labels, n_bins=15):
    """
    Expected Calibration Error.
    confidences : array of P(Yes) values in [0, 1]
    labels      : array of 1 (GT=yes) or 0 (GT=no)
    Returns ECE, and per-bin (acc, conf, count) for reliability diagram.
    """
    bins       = np.linspace(0, 1, n_bins + 1)
    bin_accs   = np.zeros(n_bins)
    bin_confs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_accs[b]   = labels[mask].mean()
            bin_confs[b]  = confidences[mask].mean()
            bin_counts[b] = mask.sum()

    ece = np.sum(bin_counts / len(confidences) * np.abs(bin_accs - bin_confs))
    return float(ece), bin_accs, bin_confs, bin_counts


def compute_brier(confidences, labels):
    """Brier score: mean squared error between confidence and binary label."""
    return float(np.mean((confidences - labels) ** 2))


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if "error" not in r and r.get("p_yes_norm") is not None:
                    records.append(r)
            except:
                pass
    return records


def to_arrays(records):
    """Convert records to numpy arrays of confidence and binary label."""
    confidences = np.array([r["p_yes_norm"] for r in records])
    labels      = np.array([1 if r["ground_truth"].lower() == "yes" else 0
                             for r in records])
    correct     = np.array([1 if r["correct"] else 0 for r in records])
    return confidences, labels, correct


# ── Load data ──────────────────────────────────────────────────────────────────
results = {}  # (model, dataset) -> metrics dict
missing = []

for (model, dataset), filename in FILES.items():
    path = os.path.join(CALIB_DIR, filename)
    if not os.path.exists(path):
        missing.append(f"  MISSING: {filename}  ({model} / {DATASET_LABELS[dataset]})")
        continue

    records = load_records(path)
    if not records:
        missing.append(f"  EMPTY: {filename}")
        continue

    conf, labels, correct = to_arrays(records)
    ece, bin_accs, bin_confs, bin_counts = compute_ece(conf, labels, N_BINS)
    brier = compute_brier(conf, labels)

    accuracy     = correct.mean()
    mean_conf    = conf.mean()
    overconf     = float(mean_conf - accuracy)  # positive = overconfident
    n_yes        = int(labels.sum())
    n_no         = len(labels) - n_yes

    # Accuracy on yes vs no subsets
    yes_mask = labels == 1
    no_mask  = labels == 0
    yes_acc  = float(correct[yes_mask].mean()) if yes_mask.sum() > 0 else 0.0
    no_acc   = float(correct[no_mask].mean())  if no_mask.sum()  > 0 else 0.0

    results[(model, dataset)] = {
        "model": model, "dataset": dataset, "n": len(records),
        "n_yes": n_yes, "n_no": n_no,
        "accuracy":   round(accuracy, 4),
        "yes_acc":    round(yes_acc, 4),
        "no_acc":     round(no_acc, 4),
        "mean_conf":  round(float(mean_conf), 4),
        "ece":        round(ece, 4),
        "brier":      round(brier, 4),
        "overconf":   round(overconf, 4),
        "bin_accs":   bin_accs.tolist(),
        "bin_confs":  bin_confs.tolist(),
        "bin_counts": bin_counts.tolist(),
        "confidences": conf.tolist(),
        "labels":      labels.tolist(),
    }

    print(f"  {model:14s} / {DATASET_LABELS[dataset]:7s}  "
          f"N={len(records):3d}  Acc={accuracy*100:.2f}%  "
          f"ECE={ece*100:.2f}pp  Brier={brier:.4f}  "
          f"MeanConf={mean_conf*100:.2f}%")

if missing:
    print("\nMissing files:")
    for m in missing:
        print(m)

if not results:
    print("\nNo data loaded. Run the Kaggle notebook first.")
    sys.exit(0)

# ── Reliability diagrams ───────────────────────────────────────────────────────
if HAVE_MPL and results:
    MODELS   = ["MedGemma-4B", "Gemma-3-4B"]
    DATASETS = ["slake", "vqa_rad"]
    bins_x   = np.linspace(1 / (2 * N_BINS), 1 - 1 / (2 * N_BINS), N_BINS)

    # Fig 1 — Reliability diagram grid (2 rows × 2 cols)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Reliability Diagrams — MedGemma-4B vs Gemma-3-4B", fontsize=14, fontweight="bold")

    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes[row][col]
            key = (model, dataset)
            if key not in results:
                ax.set_visible(False)
                continue

            m = results[key]
            ba   = np.array(m["bin_accs"])
            bc   = np.array(m["bin_confs"])
            bcnt = np.array(m["bin_counts"])
            active = bcnt > 0

            # Perfect calibration diagonal
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")

            # Gap bars (confidence - accuracy) — shaded red/blue
            for b in range(N_BINS):
                if bcnt[b] == 0:
                    continue
                x = bins_x[b]
                w = 0.9 / N_BINS
                if bc[b] > ba[b]:
                    ax.bar(x, bc[b] - ba[b], bottom=ba[b], width=w,
                           color="#ea4335", alpha=0.35, label="Overconfidence" if b == 0 else "")
                else:
                    ax.bar(x, ba[b] - bc[b], bottom=bc[b], width=w,
                           color="#1a73e8", alpha=0.35, label="Underconfidence" if b == 0 else "")

            # Accuracy per bin
            ax.bar(bins_x[active], ba[active], width=0.9 / N_BINS,
                   color=MODEL_COLORS[model], alpha=0.75, label="Accuracy per bin")

            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence P(Yes)" if row == 1 else "")
            ax.set_ylabel("Accuracy" if col == 0 else "")
            ax.set_title(f"{model} — {DATASET_LABELS[dataset]}\n"
                         f"Acc={m['accuracy']*100:.1f}%  ECE={m['ece']*100:.2f} pp  Brier={m['brier']:.4f}",
                         fontsize=10)
            ax.grid(True, alpha=0.3)

            # ECE annotation
            ax.text(0.05, 0.92, f"ECE = {m['ece']*100:.2f} pp",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    fig1_path = os.path.join(RESULTS_DIR, "fig_calibration_reliability.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nReliability diagram saved: {fig1_path}")

    # Fig 2 — Confidence histograms
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7))
    fig2.suptitle("Confidence Distribution P(Yes) — MedGemma-4B vs Gemma-3-4B",
                  fontsize=13, fontweight="bold")

    for row, dataset in enumerate(DATASETS):
        for col, model in enumerate(MODELS):
            ax = axes2[row][col]
            key = (model, dataset)
            if key not in results:
                ax.set_visible(False)
                continue

            m    = results[key]
            conf = np.array(m["confidences"])
            labs = np.array(m["labels"])

            ax.hist(conf[labs == 1], bins=20, color="#2ecc71", alpha=0.6,
                    label="GT=Yes", density=True)
            ax.hist(conf[labs == 0], bins=20, color="#e74c3c", alpha=0.6,
                    label="GT=No", density=True)
            ax.axvline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)

            ax.set_title(f"{model} — {DATASET_LABELS[dataset]}", fontsize=10)
            ax.set_xlabel("P(Yes)" if row == 1 else "")
            ax.set_ylabel("Density" if col == 0 else "")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "fig_calibration_confidence_hist.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confidence histogram saved: {fig2_path}")

# ── Save JSON ──────────────────────────────────────────────────────────────────
json_out = {}
for (model, dataset), m in results.items():
    key = f"{model}__{dataset}"
    json_out[key] = {
        "model": model, "dataset": dataset,
        "n": m["n"], "n_yes": m["n_yes"], "n_no": m["n_no"],
        "accuracy":  m["accuracy"],
        "yes_acc":   m["yes_acc"],
        "no_acc":    m["no_acc"],
        "mean_conf": m["mean_conf"],
        "ece":       m["ece"],
        "brier":     m["brier"],
        "overconf":  m["overconf"],
    }

json_path = os.path.join(RESULTS_DIR, "calibration_results.json")
with open(json_path, "w") as f:
    json.dump(json_out, f, indent=2)
print(f"JSON saved: {json_path}")

# ── Determine conclusion ───────────────────────────────────────────────────────
def ece_conclusion(med_ece, gen_ece, dataset_label):
    diff = gen_ece - med_ece
    if diff > 0.03:
        return (f"MedGemma-4B is better calibrated on {dataset_label} "
                f"(ECE: {med_ece*100:.2f} pp vs {gen_ece*100:.2f} pp, "
                f"Δ = +{diff*100:.2f} pp in favour of MedGemma).")
    elif abs(diff) <= 0.03:
        return (f"Calibration is comparable on {dataset_label} "
                f"(MedGemma ECE: {med_ece*100:.2f} pp, Gemma-3 ECE: {gen_ece*100:.2f} pp, "
                f"Δ = {diff*100:.2f} pp — within margin).")
    else:
        return (f"Gemma-3-4B is better calibrated on {dataset_label} "
                f"(ECE: {gen_ece*100:.2f} pp vs {med_ece*100:.2f} pp, "
                f"Δ = {abs(diff)*100:.2f} pp in favour of Gemma-3).")


conclusions = {}
for dataset in ["slake", "vqa_rad"]:
    med = results.get(("MedGemma-4B", dataset))
    gen = results.get(("Gemma-3-4B",  dataset))
    if med and gen:
        conclusions[dataset] = ece_conclusion(med["ece"], gen["ece"], DATASET_LABELS[dataset])

print("\nConclusions:")
for ds, c in conclusions.items():
    print(f"  {ds}: {c}")

# ── Generate report ────────────────────────────────────────────────────────────
def fmt(v, pct=False, pp=False):
    if v is None: return "—"
    if pct:  return f"{v*100:.2f}%"
    if pp:   return f"{v*100:.2f} pp"
    return f"{v:.4f}"

def overconf_str(v):
    if v is None: return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.2f} pp"


rows_main = ""
for dataset in ["slake", "vqa_rad"]:
    for model in ["MedGemma-4B", "Gemma-3-4B"]:
        m = results.get((model, dataset))
        if not m:
            rows_main += f"| {model} | {DATASET_LABELS[dataset]} | — | — | — | — | — |\n"
        else:
            rows_main += (
                f"| {model} | {DATASET_LABELS[dataset]} | {m['n']} | "
                f"{fmt(m['accuracy'], pct=True)} | "
                f"{fmt(m['ece'], pp=True)} | "
                f"{fmt(m['brier'])} | "
                f"{overconf_str(m['overconf'])} |\n"
            )

rows_detail = ""
for dataset in ["slake", "vqa_rad"]:
    for model in ["MedGemma-4B", "Gemma-3-4B"]:
        m = results.get((model, dataset))
        if not m:
            continue
        rows_detail += (
            f"| {model} | {DATASET_LABELS[dataset]} | "
            f"{fmt(m['yes_acc'], pct=True)} (N={m['n_yes']}) | "
            f"{fmt(m['no_acc'], pct=True)} (N={m['n_no']}) | "
            f"{fmt(m['mean_conf'], pct=True)} |\n"
        )

conclusion_block = "\n\n".join(
    f"**{DATASET_LABELS[ds]}:** {txt}" for ds, txt in conclusions.items()
)

# Overall calibration verdict
all_ece_diffs = []
for ds in ["slake", "vqa_rad"]:
    med = results.get(("MedGemma-4B", ds))
    gen = results.get(("Gemma-3-4B", ds))
    if med and gen:
        all_ece_diffs.append(gen["ece"] - med["ece"])

if all_ece_diffs:
    avg_diff = np.mean(all_ece_diffs)
    if avg_diff > 0.03:
        overall_verdict = (
            f"MedGemma-4B is systematically better calibrated than Gemma-3-4B across both "
            f"datasets (average ECE reduction: {avg_diff*100:.2f} pp). Medical pre-training "
            f"therefore confers a dual advantage: higher accuracy **and** better-calibrated "
            f"confidence estimates. In a clinical screening context, a well-calibrated model "
            f"produces confidence scores that are actionable — a 90% confidence prediction "
            f"is correct approximately 90% of the time — which is a distinct safety property "
            f"beyond raw accuracy."
        )
        claim_validated = True
    elif abs(avg_diff) <= 0.03:
        overall_verdict = (
            f"MedGemma-4B and Gemma-3-4B show comparable calibration across both datasets "
            f"(average ECE difference: {avg_diff*100:.2f} pp). The accuracy advantage of "
            f"MedGemma-4B is therefore not accompanied by a systematic calibration advantage, "
            f"suggesting that medical pre-training primarily improves predictive accuracy "
            f"rather than confidence reliability in this setting."
        )
        claim_validated = False
    else:
        overall_verdict = (
            f"Gemma-3-4B shows marginally better calibration than MedGemma-4B on average "
            f"(average ECE difference: {abs(avg_diff)*100:.2f} pp in favour of Gemma-3). "
            f"This is a counterintuitive finding: MedGemma-4B is more accurate but "
            f"Gemma-3-4B is more reliably calibrated. The result may reflect that domain "
            f"fine-tuning on medical data increases model confidence in ways that outpace "
            f"accuracy gains, leading to mild overconfidence on specific question types."
        )
        claim_validated = False
else:
    overall_verdict = "Insufficient data for overall verdict."
    claim_validated = False

md = f"""# Calibration Analysis Report
## VLM Medical VQA Benchmark — Section 16

**Research Question:** Is MedGemma-4B not only more accurate but also better calibrated
than Gemma-3-4B on closed (Yes/No) medical VQA questions? A well-calibrated model's
confidence scores reflect its true accuracy — a property with direct clinical implications.

**Date:** July 2026
**Notebook:** `notebooks/11_calibration_analysis.ipynb`

---

## 1. Methodology

### 1.1 Calibration Measurement

Calibration was measured using the **Expected Calibration Error (ECE)** with {N_BINS} equal-width
bins. ECE is defined as the weighted average absolute difference between confidence and accuracy
across all bins:

$$\\text{{ECE}} = \\sum_{{b=1}}^{{B}} \\frac{{|S_b|}}{{N}} \\left| \\text{{acc}}(S_b) - \\text{{conf}}(S_b) \\right|$$

Lower ECE indicates better calibration. A perfectly calibrated model has ECE = 0.

The **Brier score** (mean squared error between confidence and binary label) provides a
complementary calibration metric that is sensitive to both accuracy and reliability.

### 1.2 Confidence Extraction

Confidence was extracted as follows:
1. Run inference with `max_new_tokens=1`, `output_scores=True`.
2. `scores[0]` — the logit vector at the first generated token position, shape `(1, vocab_size)`.
3. Apply softmax over the full vocabulary to obtain a probability distribution.
4. Sum probabilities across all token IDs that decode to "Yes" or "yes" → `P(Yes)_raw`.
5. Sum probabilities across all token IDs for "No" / "no" → `P(No)_raw`.
6. Normalise: `P(Yes) = P(Yes)_raw / (P(Yes)_raw + P(No)_raw)`.

Normalisation against only the Yes/No probability mass is the standard approach for binary
calibration — it avoids dilution from irrelevant vocabulary items.

### 1.3 Scope

| Model | Type | Parameters | Datasets |
|---|---|---|---|
| MedGemma-4B-IT | Medical | 4B (fp16) | SLAKE EN, VQA-RAD |
| Gemma-3-4B-IT | Generalist | 4B (fp16) | SLAKE EN, VQA-RAD |

Only closed (Yes/No) questions were used. Open-ended questions do not have a binary decision
axis and cannot be mapped to a single confidence estimate.

| Dataset | Closed questions used |
|---|---|
| SLAKE EN (test split) | {results.get(('MedGemma-4B','slake'), {}).get('n', '—')} |
| VQA-RAD (test split) | {results.get(('MedGemma-4B','vqa_rad'), {}).get('n', '—')} |

---

## 2. Results

### 2.1 ECE and Brier Score Summary

| Model | Dataset | N | Accuracy | ECE | Brier | Overconfidence |
|---|---|---|---|---|---|---|
{rows_main.strip()}

*ECE and Brier score: lower is better. Overconfidence = mean confidence − accuracy;
positive values indicate the model is more confident than its accuracy warrants.*

### 2.2 Yes / No Accuracy and Mean Confidence

| Model | Dataset | Yes-Accuracy | No-Accuracy | Mean P(Yes) |
|---|---|---|---|---|
{rows_detail.strip()}

### 2.3 Reliability Diagrams

Reliability diagrams are available at `results/fig_calibration_reliability.png`.
Each subplot shows accuracy per confidence bin against the perfect-calibration diagonal.
Gaps above the diagonal indicate underconfidence; gaps below indicate overconfidence.

Confidence distributions are available at `results/fig_calibration_confidence_hist.png`.

---

## 3. Findings

### 3.1 Per-Dataset Conclusions

{conclusion_block}

### 3.2 Overall Conclusion

{overall_verdict}

### 3.3 Clinical Significance

Calibration is a distinct safety property from accuracy. An overconfident model that says
"95% confident: No pathology" when it is correct only 60% of the time poses a direct risk
in clinical screening — the numeric confidence score cannot be trusted. A well-calibrated
model allows clinicians and downstream systems to set meaningful confidence thresholds
(e.g., escalate to human review if P(Yes) ∈ [0.3, 0.7]).

{"The data confirms that MedGemma-4B's clinical advantage extends beyond accuracy to confidence reliability." if claim_validated else "The data shows that the calibration gap is not as pronounced as expected, which is itself an informative negative result."}

---

## 4. Generated Files

| File | Description |
|---|---|
| `outputs/_archive/calibration/*.jsonl` | Per-run inference outputs with per-sample P(Yes) |
| `results/calibration_results.json` | All ECE, Brier, accuracy metrics (machine-readable) |
| `results/fig_calibration_reliability.png` | 2×2 reliability diagram grid |
| `results/fig_calibration_confidence_hist.png` | Confidence distribution histograms |
| `scripts/calibration_analysis.py` | This analysis script (re-runnable) |
| `notebooks/11_calibration_analysis.ipynb` | Kaggle inference notebook |
"""

report_path = os.path.join(DOCS_DIR, "report_calibration.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(md)
print(f"\nReport written: {report_path}")
print("=" * 65)
