"""
series_mapper.py
----------------
Maps a target_series_type string (e.g., "SAG PD FS") to an actual
series NUMBER (e.g., "7") that matches the existing clinical_vqa_dataset.jsonl
format and is compatible with get_series_representative_slices() in the
evaluation notebooks.

Two mapping strategies are available:

  1. CSV-based (preferred, no DICOM reading required):
     Reads SeriesDescription from the structured JSON embedded in the
     'Metadata On Series Level' column of clinical_metadata.csv.
     Use: map_series_type_from_csv(...)

  2. DICOM-based (fallback, reads actual DICOM headers):
     Reads SeriesDescription from the first .dcm file in each series folder.
     Use: map_series_type_from_dicom(...)

Both return the series NUMBER as a string (e.g., "7"), matching the format
used in target_series fields across the project.
"""

import os
import json
import pandas as pd
import pydicom
from difflib import SequenceMatcher

# ── Canonical Keyword Aliases ──────────────────────────────────────────────────
# Each target_series_type maps to a list of known SeriesDescription strings
# (or substrings) that represent it across different scanner vendors/sites.
# All strings are compared case-insensitively after normalisation.

SERIES_TYPE_KEYWORDS = {
    "SAG PD FS": [
        "sag pd fs",
        "sagittal pd fs",
        "sag pd fat sat",
        "sag_pd_fs",
        "sagittal pd fat sat",
        "sag pd fatsat",
        "sag_pd_fat_sat",
        "sag pd fatsup",
    ],
    "SAG PD THIN ACL": [
        "sag pd thin acl",
        "sag_pd_thin_acl",
        "thin acl",
        "sag acl",
        "sagittal acl",
        "thin slice acl",
        "sag thin acl",
    ],
    "COR PD FS": [
        "cor pd fs",
        "coronal pd fs",
        "cor pd fat sat",
        "cor_pd_fs",
        "coronal pd fat sat",
        "cor pd fatsat",
        "cor_pd_fat_sat",
    ],
    "AX PD FS": [
        "ax pd fs",
        "axial pd fs",
        "ax pd fat sat",
        "axial pd fat sat",
        "ax_pd_fs",
        "axial_pd_fs",
        "axial pd fatsat",
        "ax pd fatsat",
    ],
}

# Series descriptions that indicate planning scouts — never diagnostically useful
EXCLUDED_KEYWORDS = ["localizer", "loc_", "_loc", "scout", "3-plane", "3plane", "survey"]

VALID_SERIES_TYPES = set(SERIES_TYPE_KEYWORDS.keys())

# Minimum fuzzy match score to accept a series (0.0–1.0)
# Minimum fuzzy score to accept a match (0.0–1.0).
# 0.65 is conservative: requires meaningful token overlap.
# "SAG PD" vs "SAG PD THIN ACL" aliases scores ~0.615 → falls below this,
# correctly returning None when no dedicated ACL series exists.
MIN_CONFIDENCE = 0.65


# ── String helpers ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, collapse underscores/hyphens to spaces."""
    return text.lower().replace("_", " ").replace("-", " ").strip()


def _fuzzy_score(a: str, b: str) -> float:
    """Longest-common-subsequence similarity ratio."""
    return SequenceMatcher(None, a, b).ratio()


def _is_excluded(description: str) -> bool:
    norm = _normalize(description)
    return any(excl in norm for excl in EXCLUDED_KEYWORDS)


def _best_match(series_descriptions: dict, target_series_type: str) -> tuple:
    """
    Given a dict of {series_number: SeriesDescription} and a target type,
    returns (best_series_number, best_score).
    """
    aliases       = SERIES_TYPE_KEYWORDS[target_series_type]
    best_series   = None
    best_score    = 0.0

    for series_num, raw_desc in series_descriptions.items():
        if _is_excluded(raw_desc):
            continue
        norm_desc = _normalize(raw_desc)
        score     = max(_fuzzy_score(norm_desc, _normalize(alias)) for alias in aliases)
        if score > best_score:
            best_score  = score
            best_series = series_num

    return best_series, best_score


# ── Strategy 1: Map from CSV metadata (no DICOM reading needed) ────────────────

def build_series_map_from_csv(metadata_csv: str) -> dict:
    """
    Parses clinical_metadata.csv and builds a lookup:
        {study_id: {series_number_str: SeriesDescription}}

    The 'Metadata On Series Level' column contains a JSON blob like:
        {"<StudyUID>.<SeriesNumber>": {"SeriesNumber": "7", "SeriesDescription": "SAG PD FS", ...}, ...}

    Returns a dict keyed by Study ID, each value being a dict of
    {series_number: description} e.g. {"7": "SAG PD FS", "8": "SAG PD THIN ACL"}.
    """
    df = pd.read_csv(metadata_csv)
    result = {}

    for _, row in df.iterrows():
        study_id = str(row["Study ID"]).strip()
        raw_meta = row.get("Metadata On Series Level", "")

        if pd.isna(raw_meta) or not str(raw_meta).strip():
            result[study_id] = {}
            continue

        try:
            meta = json.loads(str(raw_meta))
        except json.JSONDecodeError as e:
            print(f"  [Warning] Could not parse series metadata for {study_id[-10:]}: {e}")
            result[study_id] = {}
            continue

        series_descriptions = {}
        for _uid, series_info in meta.items():
            series_num  = str(series_info.get("SeriesNumber", "")).strip()
            description = str(series_info.get("SeriesDescription", "")).strip()
            if series_num and description:
                series_descriptions[series_num] = description

        result[study_id] = series_descriptions

    return result


def map_series_type_from_csv(
    study_id:          str,
    target_series_type: str,
    csv_series_map:    dict,
    verbose:           bool = True,
) -> str | None:
    """
    Maps a target_series_type to a series number using the pre-built CSV map.

    Args:
        study_id:            DICOM Study Instance UID.
        target_series_type:  One of VALID_SERIES_TYPES.
        csv_series_map:      Built by build_series_map_from_csv().
        verbose:             If True, prints per-series scores.

    Returns:
        Series number as a string (e.g., "7"), or None if no confident match.
    """
    if target_series_type not in VALID_SERIES_TYPES:
        raise ValueError(
            f"Unknown target_series_type: '{target_series_type}'. "
            f"Valid options: {sorted(VALID_SERIES_TYPES)}"
        )

    series_descriptions = csv_series_map.get(study_id, {})
    if not series_descriptions:
        print(f"  [Warning] No series metadata found in CSV for study: ...{study_id[-10:]}")
        return None

    if verbose:
        print(f"  Matching '{target_series_type}' for study ...{study_id[-10:]}:")
        for sn, desc in sorted(series_descriptions.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            excl_tag = " [LOCALIZER-SKIP]" if _is_excluded(desc) else ""
            print(f"    Series {sn:>2}: {desc!r}{excl_tag}")

    best_series, best_score = _best_match(series_descriptions, target_series_type)

    if best_series is None or best_score < MIN_CONFIDENCE:
        print(
            f"  ⚠  No confident match for '{target_series_type}' "
            f"(best score: {best_score:.3f} < threshold {MIN_CONFIDENCE})"
        )
        return None

    if verbose:
        desc = series_descriptions.get(best_series, "?")
        print(f"  → Best match: Series {best_series} ({desc!r}, score={best_score:.3f})")

    return best_series


# ── Strategy 2: Map from DICOM headers (fallback) ─────────────────────────────

def _read_series_descriptions_from_dicom(study_dir: str) -> dict:
    """
    Reads SeriesDescription from the first .dcm file in each subdirectory.
    Returns {series_number: description} where series_number is the digit(s)
    after the last '.' in the folder name (e.g., '...55648.7' → '7').
    """
    descriptions = {}
    if not os.path.isdir(study_dir):
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    for folder_name in sorted(os.listdir(study_dir)):
        folder_path = os.path.join(study_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Extract series number from folder name (last segment after '.')
        series_num = folder_name.rsplit(".", 1)[-1]
        if not series_num.isdigit():
            continue

        dcm_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".dcm"))
        if not dcm_files:
            continue

        try:
            ds   = pydicom.dcmread(
                os.path.join(folder_path, dcm_files[0]),
                stop_before_pixels=True,  # Fast — no pixel data needed
            )
            desc = str(getattr(ds, "SeriesDescription", "")).strip()
            descriptions[series_num] = desc
        except Exception as e:
            print(f"  [Warning] Could not read DICOM header from {folder_path}: {e}")

    return descriptions


def map_series_type_from_dicom(
    study_dir:          str,
    target_series_type: str,
    verbose:            bool = True,
) -> str | None:
    """
    Maps a target_series_type to a series number by reading DICOM headers.
    Use this when the CSV metadata is unavailable or for a new dataset.

    Args:
        study_dir:           Absolute path to the study directory.
        target_series_type:  One of VALID_SERIES_TYPES.
        verbose:             If True, prints per-series scores.

    Returns:
        Series number as a string (e.g., "7"), or None if no confident match.
    """
    if target_series_type not in VALID_SERIES_TYPES:
        raise ValueError(
            f"Unknown target_series_type: '{target_series_type}'. "
            f"Valid options: {sorted(VALID_SERIES_TYPES)}"
        )

    series_descriptions = _read_series_descriptions_from_dicom(study_dir)
    if not series_descriptions:
        print(f"  [Warning] No DICOM series found in: {study_dir}")
        return None

    if verbose:
        print(f"  Matching '{target_series_type}' via DICOM headers in {os.path.basename(study_dir)}:")
        for sn, desc in sorted(series_descriptions.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
            excl_tag = " [LOCALIZER-SKIP]" if _is_excluded(desc) else ""
            print(f"    Series {sn:>2}: {desc!r}{excl_tag}")

    best_series, best_score = _best_match(series_descriptions, target_series_type)

    if best_series is None or best_score < MIN_CONFIDENCE:
        print(
            f"  ⚠  No confident match for '{target_series_type}' "
            f"(best score: {best_score:.3f} < threshold {MIN_CONFIDENCE})"
        )
        return None

    if verbose:
        desc = series_descriptions.get(best_series, "?")
        print(f"  → Best match: Series {best_series} ({desc!r}, score={best_score:.3f})")

    return best_series


# ── Batch augmentation ─────────────────────────────────────────────────────────

def add_series_numbers_to_dataset(
    qa_pairs:       list,
    csv_series_map: dict,
    verbose:        bool = False,
) -> list:
    """
    Adds 'target_series' (the series number string, e.g., "7") to each QA pair
    in a dataset, using the CSV-based mapper.

    Args:
        qa_pairs:       List of QA pair dicts (output of generate_qa_pairs).
        csv_series_map: Built by build_series_map_from_csv().
        verbose:        If True, prints matching details for every pair.

    Returns:
        The same list with 'target_series' field added/updated on each dict.
    """
    augmented = []
    for qa in qa_pairs:
        study_id            = qa.get("study_id", "")
        target_series_type  = qa.get("target_series_type", "")

        series_num = map_series_type_from_csv(
            study_id           = study_id,
            target_series_type = target_series_type,
            csv_series_map     = csv_series_map,
            verbose            = verbose,
        )
        qa["target_series"] = series_num
        augmented.append(qa)
    return augmented


# ── Quick self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = "data/clinical_metadata.csv"
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found. Run from the project root.")

    print("=" * 65)
    print("Series Mapper — Self Test")
    print("=" * 65)

    csv_map = build_series_map_from_csv(csv_path)
    print(f"\nLoaded metadata for {len(csv_map)} studies.\n")

    # Expected results based on the actual CSV metadata
    expected = {
        "1.3.6.1.4.1.55648.52489457771155006307501275967031550463": {
            "SAG PD FS":       "7",
            "SAG PD THIN ACL": "8",
            "COR PD FS":       "4",
            "AX PD FS":        "3",
        },
        "1.3.6.1.4.1.55648.6974691373277018395743151067111743": {
            "SAG PD FS":       "5",
            "SAG PD THIN ACL": None,  # Patient 2 has no dedicated ACL series → expect None
            "COR PD FS":       "6",
            "AX PD FS":        "3",
        },
    }

    all_pass = True
    for study_id, mappings in expected.items():
        short = study_id[-10:]
        print(f"\nPatient ...{short}:")
        for target, expected_series in mappings.items():
            result = map_series_type_from_csv(
                study_id=study_id,
                target_series_type=target,
                csv_series_map=csv_map,
                verbose=True,
            )
            status = "✓ PASS" if result == expected_series else f"✗ FAIL (expected {expected_series!r})"
            if result != expected_series:
                all_pass = False
            print(f"  {status} — {target} → Series {result!r}\n")

    print("=" * 65)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED — review output above")
    print("=" * 65)
