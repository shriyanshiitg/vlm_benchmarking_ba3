"""
generate_qa_from_report.py
--------------------------
Takes a radiology report text and calls the Gemini 1.5 Flash API (free tier)
to extract structured QA pairs for clinical VQA evaluation.

Usage (CLI):
    export GEMINI_API_KEY="your_key"
    python scripts/generate_qa_from_report.py \\
        --report_text data/patient1_report.txt \\
        --study_id "1.3.6.1.4.1.55648.52489457771155006307501275967031550463" \\
        --output_jsonl data/generated_vqa_dataset.jsonl

Usage (import):
    from scripts.generate_qa_from_report import generate_qa_pairs, save_to_jsonl
"""

import os
import json
import time
import argparse
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL   = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={{api_key}}"
)

VALID_SERIES_TYPES = {"SAG PD FS", "SAG PD THIN ACL", "COR PD FS", "AX PD FS"}

# Prompt template lives next to this script
_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_generator_prompt.txt")


# ── Prompt loading ─────────────────────────────────────────────────────────────

def load_prompt_template() -> str:
    if not os.path.exists(_PROMPT_PATH):
        raise FileNotFoundError(
            f"Prompt template not found: {_PROMPT_PATH}\n"
            "Make sure qa_generator_prompt.txt is in the same directory as this script."
        )
    with open(_PROMPT_PATH, "r") as f:
        return f.read()


# ── Gemini API call ────────────────────────────────────────────────────────────

def call_gemini(prompt: str, api_key: str, retries: int = 5) -> str:
    """
    Calls Gemini 2.0 Flash via the REST API.
    Returns the raw text response. Raises RuntimeError on repeated failure.
    """
    url = GEMINI_URL.format(api_key=api_key)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":    0.1,   # Low temperature = factual, deterministic output
            "maxOutputTokens": 8192, # Must be large enough for 8 full QA pairs in JSON
        },
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            print(f"  [Attempt {attempt+1}/{retries}] HTTP {status}: {e}")
            if status == 429:
                # Rate limited — use aggressive backoff: 30s, 60s, 120s, 240s, 480s
                wait = 30 * (2 ** attempt)
                print(f"  Rate limited. Waiting {wait}s before retry ...")
                time.sleep(wait)
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  [Attempt {attempt+1}/{retries}] Error: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini API failed after {retries} retries.")


# ── JSON parsing ───────────────────────────────────────────────────────────────

def extract_json_from_response(raw_text: str) -> list:
    """
    Parses the LLM response into a Python list of QA dicts.
    Handles cases where the model wraps output in markdown code fences.
    """
    text = raw_text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)

    # Find the JSON array boundaries
    start = text.find("[")
    end   = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(
            f"No JSON array found in LLM response. Raw response was:\n{raw_text[:500]}"
        )
    return json.loads(text[start:end])


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_qa_pair(qa: dict) -> list:
    """
    Returns a list of validation error strings.
    An empty list means the QA pair is valid.
    """
    errors = []
    required = [
        "question", "answer", "answer_type",
        "target_series_type", "anatomical_structure", "clinical_category",
    ]
    for field in required:
        if field not in qa or not str(qa[field]).strip():
            errors.append(f"Missing or empty field: '{field}'")

    atype = qa.get("answer_type", "")
    if atype not in ("CLOSED", "OPEN"):
        errors.append(f"answer_type must be 'CLOSED' or 'OPEN', got: '{atype}'")

    if atype == "CLOSED":
        answer = qa.get("answer", "").strip().lower()
        if answer not in ("yes", "no"):
            errors.append(
                f"CLOSED answer must be exactly 'Yes' or 'No', got: '{qa.get('answer')}'"
            )

    if qa.get("target_series_type") not in VALID_SERIES_TYPES:
        errors.append(
            f"target_series_type '{qa.get('target_series_type')}' is not one of "
            f"{sorted(VALID_SERIES_TYPES)}"
        )

    answer_words = len(str(qa.get("answer", "")).split())
    if atype == "OPEN" and answer_words > 10:
        errors.append(
            f"Open answer too long ({answer_words} words): '{qa.get('answer')}' "
            f"— should be 1-5 words"
        )

    return errors


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_qa_pairs(report_text: str, study_id: str, api_key: str) -> list:
    """
    Core pipeline function.

    Args:
        report_text:  The full text of the radiology report.
        study_id:     The DICOM Study Instance UID for this patient.
        api_key:      Gemini API key.

    Returns:
        List of validated QA pair dicts, each with 'study_id' injected.
    """
    template = load_prompt_template()
    prompt   = template.replace("{report_text}", report_text.strip())

    print(f"  Calling Gemini API for study: ...{study_id[-10:]}")
    raw_response = call_gemini(prompt, api_key)

    print("  Parsing JSON response ...")
    qa_pairs = extract_json_from_response(raw_response)
    print(f"  Extracted {len(qa_pairs)} QA pairs. Validating ...")

    valid_pairs = []
    skipped     = 0
    for i, qa in enumerate(qa_pairs):
        errors = validate_qa_pair(qa)
        if errors:
            skipped += 1
            print(f"    ⚠  QA #{i+1} SKIPPED — {len(errors)} validation error(s):")
            for err in errors:
                print(f"       • {err}")
        else:
            qa["study_id"] = study_id
            valid_pairs.append(qa)
            print(
                f"    ✓  #{i+1} [{qa['answer_type']:6s}] "
                f"[{qa['target_series_type']:16s}] "
                f"{qa['question'][:55]}..."
            )

    print(
        f"  Validation done: {len(valid_pairs)} valid, {skipped} skipped "
        f"out of {len(qa_pairs)} extracted."
    )
    return valid_pairs


def save_to_jsonl(qa_pairs: list, output_path: str) -> None:
    """Appends QA pairs to a .jsonl file (one JSON object per line)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + "\n")
    print(f"  Saved {len(qa_pairs)} pairs → {output_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate clinical VQA pairs from a radiology report using Gemini 1.5 Flash (free)."
    )
    parser.add_argument(
        "--report_text", required=True,
        help="Path to a .txt file containing the radiology report text.",
    )
    parser.add_argument(
        "--study_id", required=True,
        help="DICOM Study Instance UID for this patient.",
    )
    parser.add_argument(
        "--output_jsonl", default="data/generated_vqa_dataset.jsonl",
        help="Path to the output .jsonl file (pairs are appended).",
    )
    parser.add_argument(
        "--api_key", default="",
        help="Gemini API key. Falls back to GEMINI_API_KEY environment variable.",
    )
    args = parser.parse_args()

    key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise SystemExit(
            "ERROR: No API key provided. Use --api_key or set the GEMINI_API_KEY "
            "environment variable."
        )

    with open(args.report_text, "r", encoding="utf-8") as f:
        report = f.read()

    pairs = generate_qa_pairs(report, args.study_id, key)
    if pairs:
        save_to_jsonl(pairs, args.output_jsonl)
    else:
        print("WARNING: No valid QA pairs generated. Output file not written.")
