"""
run_full_pipeline.py
--------------------
End-to-end automated pipeline:
  1. Reads radiology reports from clinical_metadata.csv
  2. Calls Gemini 1.5 Flash (free) to extract QA pairs per patient
  3. Maps each QA pair's target_series_type → actual series number
     using series metadata already embedded in the CSV (no DICOM reading)
  4. Validates and saves results to data/auto_generated_vqa_dataset.jsonl

Usage:
    export GEMINI_API_KEY="your_key_here"
    python scripts/run_full_pipeline.py

Optional flags:
    --metadata_csv      Path to CSV (default: data/clinical_metadata.csv)
    --output_jsonl      Output path (default: data/auto_generated_vqa_dataset.jsonl)
    --api_key           Gemini API key (or use GEMINI_API_KEY env var)
    --study_id          Process only this specific study_id (optional filter)
    --skip_existing     If output file exists, skip studies already in it
"""

import os
import sys
import json
import time
import argparse
import pandas as pd

# Make sure we can import sibling scripts regardless of CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.generate_qa_from_report import generate_qa_pairs, save_to_jsonl
from scripts.series_mapper import (
    build_series_map_from_csv,
    add_series_numbers_to_dataset,
)

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_CSV     = os.path.join(_PROJECT_ROOT, "data", "clinical_metadata.csv")
DEFAULT_OUTPUT  = os.path.join(_PROJECT_ROOT, "data", "auto_generated_vqa_dataset.jsonl")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_completed_studies(output_jsonl: str) -> set:
    """Returns the set of study_ids already written to the output file."""
    done = set()
    if not os.path.exists(output_jsonl):
        return done
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "study_id" in obj:
                    done.add(obj["study_id"])
            except json.JSONDecodeError:
                pass
    return done


def print_separator(char="=", width=70):
    print(char * width)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    metadata_csv:  str,
    output_jsonl:  str,
    api_key:       str,
    study_id_filter: str | None = None,
    skip_existing:  bool = True,
) -> None:

    # ── Load CSV ──────────────────────────────────────────────────────────────
    if not os.path.exists(metadata_csv):
        sys.exit(f"ERROR: Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    if "Study ID" not in df.columns or "Report" not in df.columns:
        sys.exit(
            "ERROR: Expected columns 'Study ID' and 'Report' in CSV. "
            f"Found: {df.columns.tolist()}"
        )

    # ── Build series map from CSV (no DICOM reading needed) ───────────────────
    print_separator()
    print("Step 1 — Building series map from CSV metadata ...")
    csv_series_map = build_series_map_from_csv(metadata_csv)
    print(f"  Loaded series metadata for {len(csv_series_map)} studies.\n")

    # ── Filter rows ───────────────────────────────────────────────────────────
    if study_id_filter:
        df = df[df["Study ID"] == study_id_filter]
        if df.empty:
            sys.exit(f"ERROR: study_id '{study_id_filter}' not found in CSV.")

    if skip_existing:
        completed = load_completed_studies(output_jsonl)
        if completed:
            print(
                f"Skipping {len(completed)} already-processed studies "
                f"(--skip_existing is on)."
            )
            df = df[~df["Study ID"].isin(completed)]

    if df.empty:
        print("All studies already processed. Nothing to do.")
        return

    total = len(df)
    print(f"Step 2 — Processing {total} patient(s) ...\n")

    # ── Per-patient loop ──────────────────────────────────────────────────────
    all_saved   = 0
    all_skipped = 0

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        study_id    = str(row["Study ID"]).strip()
        report_text = str(row["Report"]).strip()

        print_separator("-", 70)
        print(f"Patient {row_idx}/{total}  |  study_id: ...{study_id[-10:]}")
        print_separator("-", 70)

        if not report_text or report_text.lower() in ("nan", ""):
            print("  ⚠  Empty report text — skipping.\n")
            all_skipped += 1
            continue

        # Step A: Generate QA pairs via Gemini
        try:
            qa_pairs = generate_qa_pairs(report_text, study_id, api_key)
        except Exception as e:
            print(f"  ✗  QA generation failed: {e}\n")
            all_skipped += 1
            continue

        if not qa_pairs:
            print("  ⚠  No valid QA pairs generated — skipping.\n")
            all_skipped += 1
            continue

        # Step B: Map target_series_type → series number
        print(f"\n  Mapping series types for {len(qa_pairs)} QA pairs ...")
        qa_pairs = add_series_numbers_to_dataset(
            qa_pairs       = qa_pairs,
            csv_series_map = csv_series_map,
            verbose        = False,   # set True for detailed per-series output
        )

        # Warn about unmapped pairs (target_series = None)
        unmapped = [q for q in qa_pairs if q.get("target_series") is None]
        if unmapped:
            print(
                f"  ⚠  {len(unmapped)} QA pair(s) could not be mapped to a series "
                f"(target_series=None). They are still saved — review manually."
            )
        mapped_ok = len(qa_pairs) - len(unmapped)
        print(f"  Mapping done: {mapped_ok} mapped, {len(unmapped)} unmapped.\n")

        # Step C: Save
        save_to_jsonl(qa_pairs, output_jsonl)
        all_saved += len(qa_pairs)

        # Brief pause between patients to respect free API limits
        if row_idx < total:
            time.sleep(2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_separator()
    print("PIPELINE COMPLETE")
    print(f"  Patients processed:   {total - all_skipped}")
    print(f"  Patients skipped:     {all_skipped}")
    print(f"  Total QA pairs saved: {all_saved}")
    print(f"  Output file:          {output_jsonl}")
    print_separator()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full automated pipeline: reports → QA pairs → series mapping → .jsonl"
    )
    parser.add_argument(
        "--metadata_csv",   default=DEFAULT_CSV,
        help=f"Path to clinical_metadata.csv (default: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--output_jsonl",   default=DEFAULT_OUTPUT,
        help=f"Output .jsonl path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--api_key",        default="",
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--study_id",       default=None,
        help="Optional: process only this study_id (for testing)"
    )
    parser.add_argument(
        "--no_skip_existing", action="store_true",
        help="Reprocess all studies even if output file already has them"
    )
    args = parser.parse_args()

    key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit(
            "ERROR: No API key provided.\n"
            "  Option 1: export GEMINI_API_KEY='your_key'\n"
            "  Option 2: python scripts/run_full_pipeline.py --api_key 'your_key'\n"
            "\nGet a free key at: https://aistudio.google.com/app/apikey"
        )

    run_pipeline(
        metadata_csv    = args.metadata_csv,
        output_jsonl    = args.output_jsonl,
        api_key         = key,
        study_id_filter = args.study_id,
        skip_existing   = not args.no_skip_existing,
    )
