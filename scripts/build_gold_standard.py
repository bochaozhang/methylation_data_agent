"""
Gold-standard builder: multi-annotator curation tooling for the evaluation set
consumed by scripts/gold_standard.py.

WHY THIS EXISTS
---------------
Z.ai's review recommended a ~100-record gold standard built by multi-expert
annotation, where every disagreement is either discussed to consensus or simply
DISCARDED, so that every surviving record is certainly correct. We already have
two independent annotation efforts that can serve as two annotators:
    (a) 梦帆's CRC marker curation
    (b) our own 2021 CRC literature curation
This script is the tooling around that protocol. It does NOT annotate anything
itself and never asks an LLM for a gold value — humans produce the labels, this
script only imports, cross-checks, and promotes them.

RELATIONSHIP TO scripts/gold_standard.py  (this format EXTENDS that one)
------------------------------------------------------------------------
scripts/gold_standard.py holds GOLD_STANDARD: a list of records shaped like

    {
        "pmid": "41796341",
        "gold_verified": True,
        "cancer_type": "CRC",
        "sample_type": "plasma_cfdna",
        "sample_size_case": 636,
        "sample_size_control": 59,
        "performance_metrics": {"auc_validation": None},
        "dataset_ids_include": ["GSE203944"],
        "dataset_ids_exclude": ["GSE50132"],
        "notes": "...",
    }

and compare_record() scores a prediction against it with these rules:
    * scalar fields in _SCALAR_FIELDS  -> exact match (case/space-insensitive)
    * performance_metrics: {...}       -> match per sub-key present
    * dataset_ids_include: [...]       -> all of these must appear in predicted
    * dataset_ids_exclude: [...]       -> none of these may appear in predicted
    * a field left OUT entirely        -> skipped when scoring, not a miss
    * gold_verified=False              -> record excluded from the summary

Records emitted by this script (gold_standard_accepted.json) use exactly those
keys, so they are drop-in: they can be pasted into GOLD_STANDARD, or loaded and
passed straight to compare_record(). Two additions on top, both ignored by the
existing scorer because it only reads keys it knows:

    "provenance": {<gold field>: {value, annotators, evidence: [...]}}
    "annotation": {"annotators": [...], "accepted_at": ..., "unconfirmed_fields": [...]}

That is the whole point of the extension: the existing format records WHAT the
gold value is, this one also records WHO said so, from WHICH quote, and WHERE in
the paper — which is what makes a disagreement adjudicable instead of a
coin-flip.

FIELD NAME MAPPING (annotation schema -> gold_standard.py key)
--------------------------------------------------------------
    cancer_type            -> cancer_type
    sample_type            -> sample_type
    n_case                 -> sample_size_case
    n_control              -> sample_size_control
    auc + cohort qualifier -> performance_metrics.auc_{training,validation,external}
    dataset_ids @primary   -> dataset_ids_include
    dataset_ids @reference -> dataset_ids_exclude
    markers                -> markers_or_panel   (recorded, not scored today)

NEGATIVE GOLD LABELS
--------------------
Writing NONE in the value column means "the paper explicitly does not report
this", which is a real and important gold label — PMID 41796341's gold truth is
that there is NO AUC in the abstract (the LLM hallucinated 0.91 from the
specificity). NONE on a field maps to a null gold value, which compare_record()
scores as "the extractor must also produce null here". Leave the row out
entirely if you simply have not checked the field yet; that is different from
NONE and is scored differently (skipped vs. required-null).

NETWORK
-------
Phases 1-3 are fully offline. The only network path is the optional
--verify-pmids, which fetches titles through tools.ncbi_search (the shared
backoff/rate-limit path) and caches every result in <out-dir>/pmid_cache.json,
so re-runs and re-analysis never re-hit NCBI. Proxy is resolved from the
environment first, per project convention:
    NCBI_PROXY -> HTTPS_PROXY -> config/settings.yaml geo.proxy (intentionally blank)

USAGE
-----
    # one-time: write the CSV template + JSON schema + example rows for annotators
    python scripts/build_gold_standard.py --make-template

    # Phase 3: ingest a filled spreadsheet (rejected rows are reported, not dropped)
    python scripts/build_gold_standard.py --import-csv annotations_evren.csv --annotator evren

    # Phase 2: cross-check two or more annotators
    python scripts/build_gold_standard.py --agreement annotations_evren.json annotations_mengfan.json

    # progress against the 100-record target
    python scripts/build_gold_standard.py --status
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = "1.0"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "gold_standard"
DEFAULT_TARGET = 100

# ------------------------------------------------------------------ #
#  PHASE 1 — annotation schema                                        #
# ------------------------------------------------------------------ #
# One annotation = one (pmid, field, value) claim by one annotator, carrying the
# evidence that justifies it. Multi-valued fields (dataset_ids, markers) get one
# row per value, so every single dataset ID carries its own quote.

CSV_COLUMNS = [
    "pmid",
    "field",
    "value",
    "value_qualifier",
    "evidence_quote",
    "evidence_location",
    "annotator",
    "annotated_at",
    "confidence",
    "notes",
]

CANCER_TYPES = ["CRC", "lung", "breast", "liver", "gastric", "pancreatic", "multi", "other"]
SAMPLE_TYPES = ["tissue", "plasma_cfdna", "serum_cfdna", "wbc", "whole_blood", "mixed", "unknown"]
AUC_COHORTS = ["training", "validation", "external"]
DATASET_ROLES = ["primary", "reference"]
MARKER_KINDS = ["CpG", "gene", "DMR", "panel"]
CONFIDENCES = ["high", "medium", "low"]

# Recommended evidence_location vocabulary. Free text is accepted (a warning is
# printed) because real papers put the number in places no enum anticipates.
LOCATION_VOCAB = [
    "title", "abstract", "results", "methods", "discussion",
    "table_1", "figure_1", "supplementary", "data_availability", "geo_metadata",
]

# Values that mean "the paper explicitly does not report this" -> null gold value.
NONE_TOKENS = {"none", "null", "na", "n/a", "absent", "not_reported", "not reported"}

# value_qualifier requirement per field: "required" / "optional" / "none"
FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    "cancer_type": {
        "kind": "categorical",
        "enum": CANCER_TYPES,
        "qualifier": "none",
        "multi": False,
        "gold_key": "cancer_type",
        "help": "Primary cancer studied. One of: " + " | ".join(CANCER_TYPES),
    },
    "sample_type": {
        "kind": "categorical",
        "enum": SAMPLE_TYPES,
        "qualifier": "none",
        "multi": False,
        "gold_key": "sample_type",
        "help": (
            "PRIMARY sample type of the study's main cohort. tissue and plasma_cfdna "
            "are NOT interchangeable. One of: " + " | ".join(SAMPLE_TYPES)
        ),
    },
    "n_case": {
        "kind": "int",
        "enum": None,
        "qualifier": "optional",  # optional cohort tag, e.g. discovery/validation
        "multi": False,
        "gold_key": "sample_size_case",
        "help": "Number of cancer cases. Integer, or NONE if the paper never states it.",
    },
    "n_control": {
        "kind": "int",
        "enum": None,
        "qualifier": "optional",
        "multi": False,
        "gold_key": "sample_size_control",
        "help": "Number of non-cancer controls. Integer, or NONE.",
    },
    "auc": {
        "kind": "float",
        "enum": None,
        "qualifier": "required",  # WHICH cohort this AUC belongs to
        "multi": False,           # one per cohort; the cohort is the qualifier
        "gold_key": "performance_metrics",
        "help": (
            "AUC as a 0-1 decimal. value_qualifier MUST name the cohort: "
            + " | ".join(AUC_COHORTS)
            + ". Write NONE if the paper reports no AUC at all (do not convert "
              "sensitivity/specificity into an AUC)."
        ),
    },
    "dataset_ids": {
        "kind": "accession",
        "enum": None,
        "qualifier": "required",  # primary|reference provenance tag
        "multi": True,
        "gold_key": "dataset_ids",
        "help": (
            "One accession per row (GSE.../TCGA-.../EGA.../PRJNA...). value_qualifier "
            "MUST be: primary (holds the study's own experimental data) or reference "
            "(reference panel, background filter, normalisation or annotation source). "
            "Write NONE if the paper releases no dataset."
        ),
    },
    "markers": {
        "kind": "marker",
        "enum": None,
        "qualifier": "optional",  # CpG|gene|DMR|panel
        "multi": True,
        "gold_key": "markers_or_panel",
        "help": (
            "One marker per row: a CpG ID (cg08122047), gene symbol (SEPT9), DMR or "
            "panel name. value_qualifier may be: " + " | ".join(MARKER_KINDS)
        ),
    },
}

PMID_RE = re.compile(r"^\d{7,8}$")
ACCESSION_RE = re.compile(
    r"^(GSE\d+|GSM\d+|GPL\d+|TCGA[-_][A-Z]+|E-MTAB-\d+|PRJ[EDN][ABZ]\d+|"
    r"SRP\d+|EGA[SD]\d+|phs\d+(\.v\d+)?(\.p\d+)?|dbGaP:\S+)$",
    re.IGNORECASE,
)


def annotation_json_schema() -> Dict[str, Any]:
    """The documented JSON Schema for an annotation file (Phase 1 deliverable)."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Methylation gold-standard annotation file",
        "description": (
            "One annotator's per-record, per-field claims with provenance. Consumed by "
            "scripts/build_gold_standard.py --agreement, which promotes unanimous records "
            "into gold_standard_accepted.json in the record format used by "
            "scripts/gold_standard.py GOLD_STANDARD."
        ),
        "type": "object",
        "required": ["schema_version", "annotator", "annotations"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "annotator": {"type": "string", "minLength": 1},
            "created_at": {"type": "string", "format": "date-time"},
            "source_csv": {"type": ["string", "null"]},
            "annotations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "pmid", "field", "value", "evidence_quote",
                        "evidence_location", "annotator", "annotated_at", "confidence",
                    ],
                    "properties": {
                        "pmid": {
                            "type": "string",
                            "pattern": PMID_RE.pattern,
                            "description": "PubMed ID, 7-8 digits, no 'PMID:' prefix.",
                        },
                        "field": {
                            "type": "string",
                            "enum": list(FIELD_SPECS),
                            "description": "Which gold field this row asserts.",
                        },
                        "value": {
                            "type": ["string", "number", "null"],
                            "description": (
                                "The asserted value. null encodes an explicit negative "
                                "gold label ('the paper does not report this'), written "
                                "as NONE in the CSV. Absent row = not yet checked."
                            ),
                        },
                        "value_qualifier": {
                            "type": ["string", "null"],
                            "description": (
                                "Required for auc (cohort: " + "/".join(AUC_COHORTS) + ") "
                                "and dataset_ids (provenance: " + "/".join(DATASET_ROLES) + "). "
                                "Optional marker kind for markers."
                            ),
                        },
                        "evidence_quote": {
                            "type": "string",
                            "minLength": 10,
                            "description": (
                                "Verbatim sentence from the paper that supports the value. "
                                "Non-empty is mandatory: a value without a quote cannot be "
                                "adjudicated when annotators disagree."
                            ),
                        },
                        "evidence_location": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Where the quote lives. Recommended: " + ", ".join(LOCATION_VOCAB),
                        },
                        "annotator": {"type": "string", "minLength": 1},
                        "annotated_at": {"type": "string", "format": "date"},
                        "confidence": {"type": "string", "enum": CONFIDENCES},
                        "notes": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "x-field-specs": {
            name: {
                "kind": spec["kind"],
                "enum": spec["enum"],
                "value_qualifier": spec["qualifier"],
                "one_row_per_value": spec["multi"],
                "maps_to_gold_key": spec["gold_key"],
                "help": spec["help"],
            }
            for name, spec in FIELD_SPECS.items()
        },
    }


# ------------------------------------------------------------------ #
#  PHASE 3 — CSV importer + validation                                #
# ------------------------------------------------------------------ #

class RowError(Exception):
    """A row-level validation failure carrying every reason at once."""

    def __init__(self, reasons: List[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def _clean(v: Any) -> str:
    return str(v).strip() if v is not None else ""


def _is_none_token(raw: str) -> bool:
    return raw.lower() in NONE_TOKENS


def validate_row(
    raw_row: Dict[str, str],
    row_num: int,
    annotator: str,
    default_date: str,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate one spreadsheet row.

    Returns (annotation, warnings). Raises RowError with EVERY reason found, so an
    annotator fixing a spreadsheet sees all problems in one pass instead of one
    per re-run.
    """
    reasons: List[str] = []
    warnings: List[str] = []

    # strip only a LEADING "PMID:"/"PMID " prefix — a blanket replace would mangle
    # the value and then report the mangled form back to the annotator.
    pmid = re.sub(r"^\s*pmid[:\s]*", "", _clean(raw_row.get("pmid")), flags=re.IGNORECASE).strip()
    field = _clean(raw_row.get("field")).lower()
    raw_value = _clean(raw_row.get("value"))
    qualifier = _clean(raw_row.get("value_qualifier")).lower()
    quote = _clean(raw_row.get("evidence_quote"))
    location = _clean(raw_row.get("evidence_location")).lower()
    row_annotator = _clean(raw_row.get("annotator")) or annotator
    annotated_at = _clean(raw_row.get("annotated_at")) or default_date
    confidence = _clean(raw_row.get("confidence")).lower() or "medium"
    notes = _clean(raw_row.get("notes")) or None

    # --- pmid ---
    if not pmid:
        reasons.append("pmid is empty")
    elif not PMID_RE.match(pmid):
        reasons.append(f"pmid {pmid!r} is not well-formed (expected 7-8 digits, no prefix)")

    # --- field ---
    spec: Optional[Dict[str, Any]] = FIELD_SPECS.get(field)
    if not field:
        reasons.append("field is empty")
    elif spec is None:
        reasons.append(f"field {field!r} is not one of {list(FIELD_SPECS)}")

    # --- value ---
    value: Any = None
    if not raw_value:
        reasons.append(
            "value is empty — leave the whole row out if the field was not checked, "
            "or write NONE if the paper explicitly does not report it"
        )
    elif _is_none_token(raw_value):
        value = None  # explicit negative gold label
    elif spec is not None:
        kind = spec["kind"]
        if kind == "categorical":
            match = next((e for e in spec["enum"] if e.lower() == raw_value.lower()), None)
            if match is None:
                reasons.append(f"value {raw_value!r} not in {field} enum {spec['enum']}")
            else:
                value = match
        elif kind == "int":
            try:
                value = int(str(raw_value).replace(",", ""))
                if value < 0:
                    reasons.append(f"{field} must be >= 0, got {value}")
            except ValueError:
                reasons.append(f"{field} value {raw_value!r} is not an integer")
        elif kind == "float":
            try:
                value = float(raw_value)
            except ValueError:
                reasons.append(f"{field} value {raw_value!r} is not a number")
            else:
                if not 0.0 <= value <= 1.0:
                    reasons.append(
                        f"auc {value} is outside 0-1 (percentages must be written as decimals)"
                    )
        elif kind == "accession":
            value = raw_value.upper().replace(" ", "")
            if not ACCESSION_RE.match(value):
                warnings.append(
                    f"dataset id {value!r} does not look like a public accession "
                    "(assay/trial/platform names are usually NOT dataset_ids)"
                )
        else:  # marker
            value = raw_value

    # --- value_qualifier ---
    if spec is not None:
        need = spec["qualifier"]
        if field == "auc" and value is not None:
            if not qualifier:
                reasons.append("auc requires value_qualifier naming the cohort: " + "/".join(AUC_COHORTS))
            elif qualifier not in AUC_COHORTS:
                reasons.append(f"auc value_qualifier {qualifier!r} not in {AUC_COHORTS}")
        elif field == "dataset_ids" and value is not None:
            if not qualifier:
                reasons.append("dataset_ids requires value_qualifier: " + "/".join(DATASET_ROLES))
            elif qualifier not in DATASET_ROLES:
                reasons.append(f"dataset_ids value_qualifier {qualifier!r} not in {DATASET_ROLES}")
        elif field == "markers" and qualifier:
            if qualifier not in [k.lower() for k in MARKER_KINDS]:
                warnings.append(f"markers value_qualifier {qualifier!r} not in {MARKER_KINDS}")
        elif need == "none" and qualifier:
            warnings.append(f"{field} takes no value_qualifier; {qualifier!r} ignored")
            qualifier = ""

    # --- evidence ---
    if not quote:
        reasons.append("evidence_quote is empty (a value with no quote cannot be adjudicated)")
    elif len(quote) < 10:
        reasons.append(f"evidence_quote {quote!r} is too short to be a real quote (<10 chars)")
    if not location:
        reasons.append("evidence_location is empty")
    elif location not in LOCATION_VOCAB and not re.match(r"^(table|figure|supp\w*)[_ ]?\S*$", location):
        warnings.append(f"evidence_location {location!r} is outside the recommended vocabulary")

    # --- bookkeeping ---
    if not row_annotator:
        reasons.append("annotator is empty and no --annotator was given")
    if confidence not in CONFIDENCES:
        reasons.append(f"confidence {confidence!r} not in {CONFIDENCES}")
    try:
        datetime.strptime(annotated_at, "%Y-%m-%d")
    except ValueError:
        reasons.append(f"annotated_at {annotated_at!r} is not an ISO date (YYYY-MM-DD)")

    if reasons:
        raise RowError(reasons)

    return (
        {
            "pmid": pmid,
            "field": field,
            "value": value,
            "value_qualifier": qualifier or None,
            "evidence_quote": quote,
            "evidence_location": location,
            "annotator": row_annotator,
            "annotated_at": annotated_at,
            "confidence": confidence,
            "notes": notes,
        },
        warnings,
    )


def import_csv(csv_path: Path, annotator: str, out_dir: Path) -> Path:
    """Phase 3: ingest a filled spreadsheet into the JSON annotation schema."""
    if not csv_path.exists():
        raise SystemExit(f"[error] CSV not found: {csv_path}")

    default_date = date.today().isoformat()
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    warned: List[Tuple[int, str, List[str]]] = []

    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing_cols = [c for c in ("pmid", "field", "value", "evidence_quote") if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise SystemExit(
                f"[error] {csv_path.name} is missing required column(s): {missing_cols}\n"
                f"        expected header: {','.join(CSV_COLUMNS)}\n"
                f"        run --make-template to regenerate a correct template."
            )
        for i, raw_row in enumerate(reader, start=2):  # row 1 is the header
            if not any(_clean(v) for v in raw_row.values()):
                continue  # blank spreadsheet filler row
            try:
                ann, warnings = validate_row(raw_row, i, annotator, default_date)
            except RowError as err:
                rejected.append({
                    "row": i,
                    "pmid": _clean(raw_row.get("pmid")),
                    "field": _clean(raw_row.get("field")),
                    "value": _clean(raw_row.get("value")),
                    "reasons": err.reasons,
                })
                continue
            if warnings:
                warned.append((i, f"{ann['pmid']}/{ann['field']}", warnings))
            accepted.append(ann)

    # de-duplicate identical claims (same annotator re-pasting rows)
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for ann in accepted:
        key = (ann["pmid"], ann["field"], str(ann["value"]), ann["value_qualifier"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ann)
    dupes = len(accepted) - len(deduped)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"annotations_{annotator}.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "annotator": annotator,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(csv_path),
        "annotations": deduped,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- report ----------------
    pmids = sorted({a["pmid"] for a in deduped})
    print(f"\n{'=' * 68}")
    print(f"IMPORT  {csv_path.name}  ->  {out_path}")
    print('=' * 68)
    print(f"  annotator          : {annotator}")
    print(f"  rows accepted      : {len(deduped)}"
          + (f"  ({dupes} duplicate rows collapsed)" if dupes else ""))
    print(f"  rows rejected      : {len(rejected)}")
    print(f"  distinct PMIDs     : {len(pmids)}")

    by_field: Dict[str, int] = defaultdict(int)
    for a in deduped:
        by_field[a["field"]] += 1
    if by_field:
        print("  claims per field   :")
        for f in FIELD_SPECS:
            if by_field.get(f):
                print(f"      {f:<14} {by_field[f]}")

    if warned:
        print(f"\n  WARNINGS ({len(warned)} rows kept, but check them):")
        for row_num, where, msgs in warned:
            for m in msgs:
                print(f"    row {row_num:<4} {where:<24} {m}")

    if rejected:
        print(f"\n  REJECTED ROWS ({len(rejected)}) — nothing was silently dropped:")
        for r in rejected:
            head = f"    row {r['row']:<4} pmid={r['pmid'] or '?':<10} field={r['field'] or '?':<12}"
            print(head + f" value={r['value']!r}")
            for reason in r["reasons"]:
                print(f"          - {reason}")
        rej_path = out_dir / f"rejected_{annotator}.csv"
        with open(rej_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row", "pmid", "field", "value", "reasons"])
            for r in rejected:
                w.writerow([r["row"], r["pmid"], r["field"], r["value"], " | ".join(r["reasons"])])
        print(f"\n  Fix these rows in the spreadsheet and re-run. Machine-readable copy:")
        print(f"    {rej_path}")

    return out_path


# ------------------------------------------------------------------ #
#  PHASE 2 — inter-annotator agreement                                #
# ------------------------------------------------------------------ #
# Comparison units. A "unit" is the smallest thing two annotators can agree or
# disagree about:
#   scalar fields  -> unit key = (pmid, field, qualifier)         one value
#   dataset_ids    -> unit key = (pmid, "dataset_ids", accession) label primary|
#                     reference|absent  (absent = the annotator did not list it
#                     at all, which is itself a judgement — the GSE50132 case)
#   markers        -> set comparison per pmid (open vocabulary, no kappa)

SET_FIELDS = {"markers"}
KAPPA_FIELDS = ["cancer_type", "sample_type", "dataset_provenance"]
ABSENT = "__absent__"


def load_annotation_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[error] annotation file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "annotations" not in data:
        raise SystemExit(f"[error] {path} has no 'annotations' key (not an annotation file?)")
    if data.get("schema_version") != SCHEMA_VERSION:
        print(f"[warn] {path.name} schema_version={data.get('schema_version')!r}, "
              f"expected {SCHEMA_VERSION!r} — continuing")
    if not data.get("annotator"):
        data["annotator"] = path.stem.replace("annotations_", "")
    return data


def _norm_value(field: str, value: Any, auc_tol: int) -> Any:
    """Normalise for comparison. None stays None (explicit negative label)."""
    if value is None:
        return None
    if field == "auc":
        return round(float(value), auc_tol)
    if isinstance(value, str):
        return value.strip().lower()
    return value


def build_units(
    data: Dict[str, Any], auc_tol: int
) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]]:
    """
    Split one annotator's claims into:
      scalar_units: {(pmid, field, qualifier): annotation}
      set_units:    {(pmid, field): {value: annotation}}
    """
    scalar_units: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    set_units: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for ann in data["annotations"]:
        field = ann["field"]
        pmid = ann["pmid"]
        qual = (ann.get("value_qualifier") or "")
        if field == "dataset_ids":
            # unit = the accession itself; the label under test is its provenance
            key_val = ABSENT if ann["value"] is None else str(ann["value"]).upper()
            set_units[(pmid, "dataset_ids")][key_val] = ann
        elif field in SET_FIELDS:
            key_val = ABSENT if ann["value"] is None else str(ann["value"]).strip().lower()
            set_units[(pmid, field)][key_val] = ann
        else:
            scalar_units[(pmid, field, qual)] = ann
    return scalar_units, dict(set_units)


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> Optional[float]:
    """
    Cohen's kappa for two annotators over paired categorical labels.

    Returns None if there are no paired units. Returns 1.0 when both annotators
    used a single identical label everywhere (kappa is undefined there — chance
    agreement is 1 — but reporting perfect agreement is the honest reading, and
    the caller also prints the raw agreement rate and n).
    """
    n = len(labels_a)
    if n == 0:
        return None
    categories = sorted(set(labels_a) | set(labels_b))
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = 0.0
    for cat in categories:
        pa = labels_a.count(cat) / n
        pb = labels_b.count(cat) / n
        expected += pa * pb
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else 0.0
    return (observed - expected) / (1.0 - expected)


def kappa_label(k: Optional[float]) -> str:
    if k is None:
        return "n/a"
    if k < 0.0:
        return "worse than chance"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def _fmt(value: Any) -> str:
    return "NONE (explicitly not reported)" if value is None else repr(value)


def compute_agreement(
    files: List[Dict[str, Any]], auc_tol: int
) -> Dict[str, Any]:
    """
    Cross-check every pair of annotators over the PMIDs they both touched.

    Produces per-field agreement rates, Cohen's kappa for the categorical fields,
    and the full disagreement list (both values side by side, with both quotes).
    """
    annotators = [f["annotator"] for f in files]
    units = {f["annotator"]: build_units(f, auc_tol) for f in files}
    pmids_by_ann = {
        f["annotator"]: {a["pmid"] for a in f["annotations"]} for f in files
    }

    all_pmids = sorted(set().union(*pmids_by_ann.values())) if pmids_by_ann else []
    overlap_pmids = sorted(set.intersection(*pmids_by_ann.values())) if pmids_by_ann else []

    # field -> [(agree: bool, ...)] over all annotator pairs on overlapping pmids
    field_hits: Dict[str, List[bool]] = defaultdict(list)
    kappa_pairs: Dict[str, Dict[str, Tuple[List[str], List[str]]]] = defaultdict(dict)
    disagreements: List[Dict[str, Any]] = []
    # per-pmid conflict tracking for accept/dispute
    conflicts_by_pmid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    single_annotator_units: Dict[str, List[str]] = defaultdict(list)

    for i, a_name in enumerate(annotators):
        for b_name in annotators[i + 1:]:
            a_scalar, a_sets = units[a_name]
            b_scalar, b_sets = units[b_name]
            pair = f"{a_name} vs {b_name}"
            shared_pmids = pmids_by_ann[a_name] & pmids_by_ann[b_name]

            cat_labels: Dict[str, Tuple[List[str], List[str]]] = {
                f: ([], []) for f in KAPPA_FIELDS
            }

            # ---------- scalar units ----------
            keys = {k for k in a_scalar if k[0] in shared_pmids} | {
                k for k in b_scalar if k[0] in shared_pmids
            }
            for key in sorted(keys):
                pmid, field, qual = key
                a_ann, b_ann = a_scalar.get(key), b_scalar.get(key)
                field_key = f"{field}[{qual}]" if qual else field
                if a_ann is None or b_ann is None:
                    only = a_name if a_ann else b_name
                    single_annotator_units[pmid].append(f"{field_key} (only {only})")
                    # An AUC one annotator reports for a cohort the other never
                    # mentions is not agreement — but it is not a contradiction
                    # either. Tracked as unconfirmed, not scored.
                    continue
                a_val = _norm_value(field, a_ann["value"], auc_tol)
                b_val = _norm_value(field, b_ann["value"], auc_tol)
                agree = a_val == b_val
                field_hits[field].append(agree)
                if field in cat_labels:
                    cat_labels[field][0].append(str(a_val))
                    cat_labels[field][1].append(str(b_val))
                if not agree:
                    conflict = {
                        "pmid": pmid,
                        "field": field_key,
                        "values": [
                            {"annotator": a_name, **_evidence_of(a_ann)},
                            {"annotator": b_name, **_evidence_of(b_ann)},
                        ],
                    }
                    disagreements.append({"pair": pair, **conflict})
                    conflicts_by_pmid[pmid].append(conflict)

            # ---------- set units: dataset provenance ----------
            for pmid in sorted(shared_pmids):
                a_ds = a_sets.get((pmid, "dataset_ids"), {})
                b_ds = b_sets.get((pmid, "dataset_ids"), {})
                if not a_ds and not b_ds:
                    continue
                if not a_ds or not b_ds:
                    only = a_name if a_ds else b_name
                    single_annotator_units[pmid].append(f"dataset_ids (only {only})")
                    continue
                ids = sorted((set(a_ds) | set(b_ds)) - {ABSENT})
                if not ids:  # both said NONE
                    field_hits["dataset_ids"].append(True)
                    continue
                for ds_id in ids:
                    a_ann, b_ann = a_ds.get(ds_id), b_ds.get(ds_id)
                    a_lab = (a_ann.get("value_qualifier") or "primary") if a_ann else ABSENT
                    b_lab = (b_ann.get("value_qualifier") or "primary") if b_ann else ABSENT
                    agree = a_lab == b_lab
                    field_hits["dataset_ids"].append(agree)
                    cat_labels["dataset_provenance"][0].append(a_lab)
                    cat_labels["dataset_provenance"][1].append(b_lab)
                    if not agree:
                        conflict = {
                            "pmid": pmid,
                            "field": f"dataset_ids:{ds_id}",
                            "values": [
                                {"annotator": a_name,
                                 **(_evidence_of(a_ann) if a_ann else _absent_evidence(ds_id))},
                                {"annotator": b_name,
                                 **(_evidence_of(b_ann) if b_ann else _absent_evidence(ds_id))},
                            ],
                        }
                        disagreements.append({"pair": pair, **conflict})
                        conflicts_by_pmid[pmid].append(conflict)

            # ---------- set units: markers ----------
            for pmid in sorted(shared_pmids):
                a_mk = set(a_sets.get((pmid, "markers"), {})) - {ABSENT}
                b_mk = set(b_sets.get((pmid, "markers"), {})) - {ABSENT}
                a_has = (pmid, "markers") in a_sets
                b_has = (pmid, "markers") in b_sets
                if not a_has and not b_has:
                    continue
                if not (a_has and b_has):
                    single_annotator_units[pmid].append(
                        f"markers (only {a_name if a_has else b_name})"
                    )
                    continue
                agree = a_mk == b_mk
                field_hits["markers"].append(agree)
                if not agree:
                    conflict = {
                        "pmid": pmid,
                        "field": "markers (set)",
                        "jaccard": round(
                            len(a_mk & b_mk) / len(a_mk | b_mk), 3
                        ) if (a_mk | b_mk) else 1.0,
                        "values": [
                            {"annotator": a_name, "value": sorted(a_mk),
                             "evidence_quote": "; ".join(
                                 sorted({v["evidence_quote"] for v in a_sets[(pmid, 'markers')].values()})
                             )[:400],
                             "evidence_location": "", "confidence": ""},
                            {"annotator": b_name, "value": sorted(b_mk),
                             "evidence_quote": "; ".join(
                                 sorted({v["evidence_quote"] for v in b_sets[(pmid, 'markers')].values()})
                             )[:400],
                             "evidence_location": "", "confidence": ""},
                        ],
                    }
                    disagreements.append({"pair": pair, **conflict})
                    conflicts_by_pmid[pmid].append(conflict)

            for f, (la, lb) in cat_labels.items():
                kappa_pairs[f][pair] = (la, lb)

    # ---------------- summaries ----------------
    field_agreement = {
        f: {
            "n_units": len(hits),
            "n_agree": sum(hits),
            "rate": round(sum(hits) / len(hits), 4) if hits else None,
        }
        for f, hits in sorted(field_hits.items())
    }
    kappas: Dict[str, Dict[str, Any]] = {}
    for f, pairs in kappa_pairs.items():
        kappas[f] = {}
        for pair, (la, lb) in pairs.items():
            k = cohens_kappa(la, lb)
            kappas[f][pair] = {
                "kappa": round(k, 4) if k is not None else None,
                "n": len(la),
                "interpretation": kappa_label(k),
                "categories_used": sorted(set(la) | set(lb)),
            }

    return {
        "annotators": annotators,
        "n_pmids_total": len(all_pmids),
        "pmids_all_annotators": overlap_pmids,
        "field_agreement": field_agreement,
        "cohens_kappa": kappas,
        "disagreements": disagreements,
        "conflicts_by_pmid": dict(conflicts_by_pmid),
        "single_annotator_units": dict(single_annotator_units),
        "pmids_by_annotator": {k: sorted(v) for k, v in pmids_by_ann.items()},
    }


def _evidence_of(ann: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "value": ann["value"],
        "value_qualifier": ann.get("value_qualifier"),
        "evidence_quote": ann["evidence_quote"],
        "evidence_location": ann["evidence_location"],
        "confidence": ann["confidence"],
    }


def _absent_evidence(ds_id: str) -> Dict[str, Any]:
    return {
        "value": None,
        "value_qualifier": ABSENT,
        "evidence_quote": f"(did not list {ds_id} as a dataset at all)",
        "evidence_location": "",
        "confidence": "",
    }


# ------------------------------------------------------------------ #
#  Promotion to the scripts/gold_standard.py record format            #
# ------------------------------------------------------------------ #

def to_gold_record(
    pmid: str,
    anns: List[Dict[str, Any]],
    annotators: List[str],
    unconfirmed: List[str],
) -> Dict[str, Any]:
    """
    Fold agreed annotations into ONE record in the exact shape that
    scripts/gold_standard.py compare_record() consumes, plus provenance.

    Only annotated fields are emitted: an absent key is 'no ground truth yet'
    and is skipped by the scorer, while an emitted null is 'the extractor must
    also produce null here'.
    """
    record: Dict[str, Any] = {"pmid": pmid, "gold_verified": True}
    provenance: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    include: List[str] = []
    exclude: List[str] = []
    markers: List[Dict[str, Any]] = []
    notes: List[str] = []

    def add_prov(gold_key: str, value: Any, sources: List[Dict[str, Any]]) -> None:
        provenance.setdefault(gold_key, {"value": value, "annotators": [], "evidence": []})
        for s in sources:
            provenance[gold_key]["annotators"].append(s["annotator"])
            provenance[gold_key]["evidence"].append({
                "annotator": s["annotator"],
                "quote": s["evidence_quote"],
                "location": s["evidence_location"],
                "confidence": s["confidence"],
                "annotated_at": s["annotated_at"],
            })
        provenance[gold_key]["annotators"] = sorted(set(provenance[gold_key]["annotators"]))

    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for a in anns:
        vkey = "" if a["value"] is None else str(a["value"])
        grouped[(a["field"], a.get("value_qualifier") or "", vkey)].append(a)

    for (field, qual, _vkey), sources in sorted(grouped.items()):
        spec = FIELD_SPECS[field]
        value = sources[0]["value"]
        if field in ("cancer_type", "sample_type", "n_case", "n_control"):
            record[spec["gold_key"]] = value
            add_prov(spec["gold_key"], value, sources)
        elif field == "auc":
            key = f"auc_{qual}" if qual else "auc_validation"
            metrics[key] = value
            add_prov(f"performance_metrics.{key}", value, sources)
        elif field == "dataset_ids":
            if value is None:
                notes.append("annotators agree the paper releases no dataset")
                add_prov("dataset_ids", None, sources)
                continue
            (include if qual == "primary" else exclude).append(str(value))
            add_prov(f"dataset_ids:{value}", qual or "primary", sources)
        elif field == "markers":
            if value is None:
                continue
            markers.append({"id": str(value), "type": qual or None})
            add_prov(f"markers:{value}", value, sources)
        for s in sources:
            if s.get("notes"):
                notes.append(f"[{s['annotator']}] {s['notes']}")

    if metrics:
        record["performance_metrics"] = metrics
    if include:
        record["dataset_ids_include"] = sorted(set(include))
    if exclude:
        record["dataset_ids_exclude"] = sorted(set(exclude))
    if markers:
        record["markers_or_panel"] = markers
    if notes:
        record["notes"] = " | ".join(dict.fromkeys(notes))

    record["provenance"] = provenance
    record["annotation"] = {
        "annotators": sorted(annotators),
        "n_annotators": len(set(annotators)),
        "accepted_at": date.today().isoformat(),
        "unconfirmed_fields": sorted(set(unconfirmed)),
    }
    return record


def partition_records(
    files: List[Dict[str, Any]],
    result: Dict[str, Any],
    min_annotators: int,
    strict: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split every PMID into accepted / disputed / pending, per Z.ai's protocol:
      accepted — annotated independently by >= min_annotators, ZERO conflicts
      disputed — at least one conflicting field -> discuss or discard
      pending  — fewer than min_annotators have annotated it yet
    In --strict, a record with any field only one annotator checked is disputed
    rather than accepted.
    """
    by_pmid_ann: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for f in files:
        for a in f["annotations"]:
            by_pmid_ann[a["pmid"]][f["annotator"]].append(a)

    accepted, disputed, pending = [], [], []
    for pmid in sorted(by_pmid_ann):
        anns_by_annotator = by_pmid_ann[pmid]
        annotators = sorted(anns_by_annotator)
        flat = [a for lst in anns_by_annotator.values() for a in lst]
        conflicts = result["conflicts_by_pmid"].get(pmid, [])
        unconfirmed = result["single_annotator_units"].get(pmid, [])

        if len(annotators) < min_annotators:
            pending.append({
                "pmid": pmid,
                "annotators": annotators,
                "n_claims": len(flat),
                "reason": f"only {len(annotators)} annotator(s), need {min_annotators}",
            })
            continue
        if conflicts or (strict and unconfirmed):
            disputed.append({
                "pmid": pmid,
                "annotators": annotators,
                "status": "disputed",
                "resolution": None,  # fill in: "discussed -> <value>" or "discarded"
                "conflicts": conflicts,
                "unconfirmed_fields": sorted(set(unconfirmed)),
                "n_conflicts": len(conflicts),
            })
            continue
        accepted.append(to_gold_record(pmid, flat, annotators, unconfirmed))
    return accepted, disputed, pending


# ------------------------------------------------------------------ #
#  Reporting                                                          #
# ------------------------------------------------------------------ #

def print_agreement_report(
    result: Dict[str, Any],
    accepted: List[Dict[str, Any]],
    disputed: List[Dict[str, Any]],
    pending: List[Dict[str, Any]],
    target: int,
) -> None:
    print(f"\n{'=' * 68}")
    print("INTER-ANNOTATOR AGREEMENT")
    print('=' * 68)
    print(f"  annotators              : {', '.join(result['annotators'])}")
    for ann, pmids in result["pmids_by_annotator"].items():
        print(f"    {ann:<20} {len(pmids)} PMIDs")
    print(f"  PMIDs seen (union)      : {result['n_pmids_total']}")
    print(f"  PMIDs all annotators    : {len(result['pmids_all_annotators'])}")

    print(f"\n  Per-field agreement rate (over doubly-annotated units)")
    print(f"  {'field':<20} {'agree/units':>14}   rate")
    for field, stats in result["field_agreement"].items():
        rate = "n/a" if stats["rate"] is None else f"{stats['rate'] * 100:.1f}%"
        print(f"  {field:<20} {stats['n_agree']:>6}/{stats['n_units']:<7} {rate:>8}")
    if not result["field_agreement"]:
        print("    (no overlapping units — annotators have no PMIDs in common yet)")

    print(f"\n  Cohen's kappa (categorical fields)")
    any_kappa = False
    for field, pairs in result["cohens_kappa"].items():
        for pair, stats in pairs.items():
            if stats["n"] == 0:
                continue
            any_kappa = True
            k = "n/a" if stats["kappa"] is None else f"{stats['kappa']:.3f}"
            print(f"    {field:<20} {pair:<28} k={k:>7}  n={stats['n']:<4} ({stats['interpretation']})")
    if not any_kappa:
        print("    (no paired categorical units yet)")

    # ---------------- disagreement report ----------------
    print(f"\n{'=' * 68}")
    print(f"DISAGREEMENT REPORT — {len(result['disagreements'])} conflicting field(s)")
    print('=' * 68)
    if not result["disagreements"]:
        print("  No conflicts. Every doubly-annotated field agrees.")
    for d in result["disagreements"]:
        print(f"\n  PMID {d['pmid']}  |  field: {d['field']}  |  {d['pair']}")
        if "jaccard" in d:
            print(f"  set overlap (Jaccard): {d['jaccard']}")
        for v in d["values"]:
            print(f"    - {v['annotator']:<12} value={_fmt(v['value'])}"
                  + (f"  [{v['value_qualifier']}]" if v.get("value_qualifier") else "")
                  + (f"  conf={v['confidence']}" if v.get("confidence") else ""))
            quote = (v["evidence_quote"] or "").replace("\n", " ")
            if len(quote) > 220:
                quote = quote[:217] + "..."
            print(f"      quote  : \"{quote}\"")
            if v.get("evidence_location"):
                print(f"      located: {v['evidence_location']}")
        print("      -> resolve: discuss to consensus, or DISCARD the record (Z.ai protocol)")

    # ---------------- coverage ----------------
    print(f"\n{'=' * 68}")
    print("COVERAGE")
    print('=' * 68)
    n_acc = len(accepted)
    bar_len = 40
    filled = min(bar_len, int(bar_len * n_acc / target)) if target else 0
    print(f"  accepted (all annotators agree) : {n_acc}")
    print(f"  disputed (discuss or discard)   : {len(disputed)}")
    print(f"  pending  (needs a 2nd annotator): {len(pending)}")
    print(f"\n  [{'#' * filled}{'.' * (bar_len - filled)}] {n_acc}/{target} "
          f"({n_acc / target * 100:.0f}% of the target gold standard)" if target else "")
    if n_acc < target:
        print(f"  {target - n_acc} more accepted records needed.")
        if pending:
            print(f"  Fastest path: get a 2nd annotator onto the {len(pending)} pending PMIDs "
                  f"(that alone would add up to {len(pending)} records).")
        if disputed:
            print(f"  Then adjudicate the {len(disputed)} disputed records.")
    else:
        print("  TARGET REACHED.")


def write_disagreement_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Disagreement report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Annotators: {', '.join(result['annotators'])}",
        "",
        "Protocol (Z.ai): each conflict below is either **discussed to consensus** "
        "or the record is **discarded**. Nothing lands in the gold standard while a "
        "conflict is open.",
        "",
        "## Summary",
        "",
        "| field | agree | units | rate |",
        "|---|---|---|---|",
    ]
    for field, s in result["field_agreement"].items():
        rate = "n/a" if s["rate"] is None else f"{s['rate'] * 100:.1f}%"
        lines.append(f"| {field} | {s['n_agree']} | {s['n_units']} | {rate} |")
    lines += ["", "| categorical field | pair | kappa | n | reading |", "|---|---|---|---|---|"]
    for field, pairs in result["cohens_kappa"].items():
        for pair, s in pairs.items():
            if s["n"] == 0:
                continue
            k = "n/a" if s["kappa"] is None else f"{s['kappa']:.3f}"
            lines.append(f"| {field} | {pair} | {k} | {s['n']} | {s['interpretation']} |")

    lines += ["", "## Conflicts", ""]
    if not result["disagreements"]:
        lines.append("None — every doubly-annotated field agrees.")
    for d in result["disagreements"]:
        lines += [f"### PMID {d['pmid']} — `{d['field']}`  ({d['pair']})", ""]
        lines += ["| annotator | value | evidence quote | location | confidence |",
                  "|---|---|---|---|---|"]
        for v in d["values"]:
            quote = (v["evidence_quote"] or "").replace("|", "\\|").replace("\n", " ")
            val = "NONE (not reported)" if v["value"] is None else str(v["value"])
            if v.get("value_qualifier"):
                val += f" [{v['value_qualifier']}]"
            lines.append(
                f"| {v['annotator']} | {val} | {quote} | {v.get('evidence_location', '')} "
                f"| {v.get('confidence', '')} |"
            )
        lines += ["", "**Resolution:** _(discussed -> value, or DISCARDED)_", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ #
#  Template / schema emission                                         #
# ------------------------------------------------------------------ #

TEMPLATE_README = """# Gold-standard annotation — how to fill this in

Goal: ~{target} records where **every** annotator agrees, so each surviving record is
certainly correct. Conflicts are discussed or the record is discarded — never averaged.

## Files here
| file | what it is |
|---|---|
| `annotation_template.csv` | empty spreadsheet — copy it, rename to `annotations_<you>.csv`, fill it in |
| `annotation_example.csv`  | the two already-verified records, filled in, as a worked example |
| `annotation_schema.json`  | the JSON Schema the importer validates against |

## Rules that matter
1. **One row per claim.** A paper with 3 dataset IDs gets 3 `dataset_ids` rows.
2. **Every row needs a verbatim quote.** If you cannot quote it, you cannot annotate it.
   A value with no quote is unadjudicable when two annotators differ, so it is rejected.
3. **NONE is a real answer.** If the paper reports no AUC, write `NONE` in `value` and quote
   the sentence that reports sensitivity/specificity instead. That is a gold label meaning
   "the extractor must also return null here" — it is how PMID 41796341's known bug is caught.
   **Leave the row out entirely** if you simply have not checked that field. Skipped and
   explicitly-null are scored differently.
4. **auc needs a cohort** in `value_qualifier`: training / validation / external. An AUC with
   no cohort is the tissue-vs-cfDNA mixup waiting to happen (PMID 40860669).
5. **dataset_ids needs a provenance tag** in `value_qualifier`:
   - `primary`   = the study's own experimental data
   - `reference` = reference panel / background filter / normalisation / annotation source
   Assay names, trial registry IDs and platform names (ColonAiQ, NCT..., HYGEIA) are **not**
   dataset IDs — the importer warns about these.
6. **Annotate independently.** Do not look at another annotator's file first; the agreement
   number is only meaningful if the two passes are independent.

## Columns
{columns}

## When you are done
```
python scripts/build_gold_standard.py --import-csv annotations_<you>.csv --annotator <you>
```
Fix any rejected rows it reports, re-run, then the coordinator runs `--agreement`.
"""


def make_template(out_dir: Path, target: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tpl = out_dir / "annotation_template.csv"
    with open(tpl, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(CSV_COLUMNS)

    schema_path = out_dir / "annotation_schema.json"
    schema_path.write_text(
        json.dumps(annotation_json_schema(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Worked example built from the two records already verified in
    # scripts/gold_standard.py — including both of its known-bug cases.
    today = date.today().isoformat()
    example_rows = [
        ("41796341", "cancer_type", "CRC", "", "colorectal cancer (CRC) early detection",
         "title", "example", today, "high", ""),
        ("41796341", "sample_type", "plasma_cfdna", "", "plasma cell-free DNA samples from 636 participants",
         "abstract", "example", today, "high", ""),
        ("41796341", "n_case", "636", "", "plasma cell-free DNA samples from 636 participants",
         "abstract", "example", today, "high", ""),
        ("41796341", "auc", "NONE", "validation",
         "sensitivity of 87.82% and specificity of 91.88%", "abstract", "example", today, "high",
         "abstract reports NO AUC; the LLM previously hallucinated 0.91 from the specificity"),
        ("40860669", "sample_type", "tissue", "",
         "genome-wide methylation profiling of tumour tissue", "abstract", "example", today, "high",
         "primary cohort is tissue, not cfDNA"),
        ("40860669", "auc", "0.922", "validation",
         "AUC of 0.922 in the tissue validation cohort", "abstract", "example", today, "high",
         "this is the TISSUE AUC; the cfDNA AUC is 0.728 (n=33)"),
        ("40860669", "dataset_ids", "GSE50132", "reference",
         "GSE50132 was used as a mouse WBC background-filter reference panel",
         "methods", "example", today, "high", "reference panel, NOT primary study data"),
    ]
    ex = out_dir / "annotation_example.csv"
    with open(ex, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        w.writerows(example_rows)

    col_doc = "\n".join(
        f"- `{c}` — " + {
            "pmid": "PubMed ID, 7-8 digits, no `PMID:` prefix",
            "field": "one of: " + ", ".join(f"`{f}`" for f in FIELD_SPECS),
            "value": "the value, or `NONE` for an explicit 'not reported'",
            "value_qualifier": "cohort for `auc`, provenance for `dataset_ids`, kind for `markers`",
            "evidence_quote": "verbatim sentence from the paper (mandatory, >=10 chars)",
            "evidence_location": "where the quote lives: " + ", ".join(f"`{l}`" for l in LOCATION_VOCAB),
            "annotator": "your handle (may be left blank; `--annotator` fills it)",
            "annotated_at": "YYYY-MM-DD (blank = import date)",
            "confidence": "high / medium / low (blank = medium)",
            "notes": "free text, carried into the gold record",
        }[c]
        for c in CSV_COLUMNS
    )
    field_doc = "\n".join(f"\n### `{name}`\n{spec['help']}" for name, spec in FIELD_SPECS.items())
    readme = out_dir / "ANNOTATION_GUIDE.md"
    readme.write_text(
        TEMPLATE_README.format(target=target, columns=col_doc) + "\n## Field reference\n" + field_doc + "\n",
        encoding="utf-8",
    )

    print(f"\n{'=' * 68}")
    print("TEMPLATE WRITTEN")
    print('=' * 68)
    for p in (tpl, ex, schema_path, readme):
        print(f"  {p}")
    print("\n  Give annotators ANNOTATION_GUIDE.md + a copy of annotation_template.csv.")
    print("  They fill it in a spreadsheet, save as CSV, then:")
    print("    python scripts/build_gold_standard.py --import-csv annotations_<name>.csv --annotator <name>")


# ------------------------------------------------------------------ #
#  Optional PMID verification (the only network path)                 #
# ------------------------------------------------------------------ #

def _resolve_proxy() -> str:
    """
    Project convention: config/settings.yaml keeps geo.proxy intentionally BLANK
    and the real value is per-machine in the environment. Reading the config key
    alone sends traffic out unproxied, which NCBI rate-limit-blocks.
    """
    proxy = os.environ.get("NCBI_PROXY") or os.environ.get("HTTPS_PROXY") or ""
    if proxy:
        return proxy
    try:
        import yaml  # local import: this script must run without yaml for offline phases
        cfg = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
        return (cfg.get("geo") or {}).get("proxy") or ""
    except Exception:
        return ""


def verify_pmids(pmids: List[str], out_dir: Path) -> Dict[str, Any]:
    """
    Fetch titles for PMIDs to confirm they exist, via tools.ncbi_search (shared
    rate-limit + backoff path — never hit the endpoints directly). Everything is
    cached to <out-dir>/pmid_cache.json so re-analysis never re-hits NCBI.
    """
    cache_path = out_dir / "pmid_cache.json"
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    todo = [p for p in pmids if p not in cache]
    print(f"\n[verify] {len(pmids)} PMIDs: {len(pmids) - len(todo)} cached, {len(todo)} to fetch")
    if todo:
        proxy = _resolve_proxy()
        print(f"[verify] proxy: {proxy or '(none — expect NCBI rate-limit blocks)'}")
        if proxy:
            os.environ.setdefault("HTTPS_PROXY", proxy)
            os.environ.setdefault("HTTP_PROXY", proxy)
        try:
            from tools.ncbi_search import efetch_abstracts
        except Exception as exc:  # noqa: BLE001
            print(f"[verify] cannot import tools.ncbi_search ({exc}); skipping verification")
            return cache
        try:
            fetched = efetch_abstracts(todo)
        except Exception as exc:  # noqa: BLE001
            print(f"[verify] fetch failed ({exc}); keeping cache, skipping verification")
            return cache
        found = {r["pmid"]: r for r in fetched if r.get("pmid")}
        for pmid in todo:
            rec = found.get(pmid)
            cache[pmid] = {
                "exists": rec is not None,
                "title": (rec or {}).get("title", ""),
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
            }
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[verify] cache updated: {cache_path}")

    missing = [p for p in pmids if not cache.get(p, {}).get("exists", True)]
    if missing:
        print(f"[verify] WARNING — {len(missing)} PMID(s) not found in PubMed: {missing}")
    else:
        print("[verify] all PMIDs resolve in PubMed")
    return cache


# ------------------------------------------------------------------ #
#  Commands                                                           #
# ------------------------------------------------------------------ #

def cmd_agreement(args: argparse.Namespace, out_dir: Path) -> int:
    files = [load_annotation_file(Path(p)) for p in args.agreement]
    if len({f["annotator"] for f in files}) < len(files):
        raise SystemExit("[error] two annotation files carry the same annotator name")
    if len(files) < 2:
        raise SystemExit("[error] --agreement needs >= 2 annotation files")

    if args.verify_pmids:
        all_pmids = sorted({a["pmid"] for f in files for a in f["annotations"]})
        verify_pmids(all_pmids, out_dir)

    result = compute_agreement(files, args.auc_decimals)
    accepted, disputed, pending = partition_records(
        files, result, args.min_annotators, args.strict
    )
    print_agreement_report(result, accepted, disputed, pending, args.target)

    out_dir.mkdir(parents=True, exist_ok=True)
    acc_path = out_dir / "gold_standard_accepted.json"
    dis_path = out_dir / "gold_standard_disputed.json"
    sum_path = out_dir / "agreement_summary.json"
    md_path = out_dir / "disagreement_report.md"

    acc_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "annotators": result["annotators"],
        "target": args.target,
        "n_accepted": len(accepted),
        "note": (
            "Records use the scripts/gold_standard.py GOLD_STANDARD field names "
            "(compare_record-compatible); 'provenance' and 'annotation' are extra keys "
            "the existing scorer ignores."
        ),
        "records": accepted,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    dis_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "discuss to consensus, or discard the record (Z.ai)",
        "n_disputed": len(disputed),
        "records": disputed,
        "pending_second_annotator": pending,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {k: v for k, v in result.items() if k != "conflicts_by_pmid"}
    summary["counts"] = {
        "accepted": len(accepted), "disputed": len(disputed),
        "pending": len(pending), "target": args.target,
    }
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_disagreement_markdown(result, md_path)

    print(f"\n  WROTE")
    for p in (acc_path, dis_path, sum_path, md_path):
        print(f"    {p}")
    print(f"\n  Next: paste records from gold_standard_accepted.json into GOLD_STANDARD in")
    print(f"  scripts/gold_standard.py (they are already in its record format), then run it.")
    return 0


def cmd_status(out_dir: Path, target: int) -> int:
    print(f"\n{'=' * 68}")
    print(f"GOLD STANDARD STATUS   ({out_dir})")
    print('=' * 68)
    if not out_dir.exists():
        print(f"  No annotation directory yet. Start with:")
        print(f"    python scripts/build_gold_standard.py --make-template")
        return 0

    ann_files = sorted(out_dir.glob("annotations_*.json"))
    if not ann_files:
        print("  No annotation files yet (expected annotations_<annotator>.json).")
        print("  Run --make-template, hand the CSV to annotators, then --import-csv.")
    pmids_by_ann: Dict[str, set] = {}
    for p in ann_files:
        data = load_annotation_file(p)
        pmids = {a["pmid"] for a in data["annotations"]}
        pmids_by_ann[data["annotator"]] = pmids
        print(f"  {data['annotator']:<18} {len(data['annotations']):>4} claims  "
              f"{len(pmids):>3} PMIDs   ({p.name})")

    if len(pmids_by_ann) >= 2:
        print(f"\n  Overlap (PMIDs annotated by both — the only ones that can be accepted):")
        names = sorted(pmids_by_ann)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = pmids_by_ann[a] & pmids_by_ann[b]
                print(f"    {a} ∩ {b}: {len(both)}")

    acc_path = out_dir / "gold_standard_accepted.json"
    dis_path = out_dir / "gold_standard_disputed.json"
    n_acc = n_dis = 0
    if acc_path.exists():
        n_acc = json.loads(acc_path.read_text(encoding="utf-8")).get("n_accepted", 0)
    if dis_path.exists():
        d = json.loads(dis_path.read_text(encoding="utf-8"))
        n_dis = d.get("n_disputed", 0)
        n_pend = len(d.get("pending_second_annotator", []))
    else:
        n_pend = 0

    bar_len = 40
    filled = min(bar_len, int(bar_len * n_acc / target)) if target else 0
    print(f"\n  accepted : {n_acc}")
    print(f"  disputed : {n_dis}")
    print(f"  pending  : {n_pend}")
    print(f"  [{'#' * filled}{'.' * (bar_len - filled)}] {n_acc}/{target} "
          f"({n_acc / target * 100:.0f}%)")
    if not acc_path.exists():
        print(f"\n  (no --agreement run yet — accepted/disputed counts are 0 by default)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_gold_standard.py",
        description=(
            "Build a multi-annotator gold standard for scripts/gold_standard.py. "
            "Records survive only when every annotator agrees; conflicts are discussed "
            "or discarded."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s --make-template\n"
            "  %(prog)s --import-csv annotations_evren.csv --annotator evren\n"
            "  %(prog)s --agreement annotations_evren.json annotations_mengfan.json\n"
            "  %(prog)s --status\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--import-csv", metavar="CSV", help="Phase 3: ingest a filled annotator spreadsheet")
    mode.add_argument("--agreement", nargs="+", metavar="JSON",
                      help="Phase 2: cross-check >= 2 annotation JSON files")
    mode.add_argument("--status", action="store_true", help="progress against the target")
    mode.add_argument("--make-template", action="store_true",
                      help="Phase 1: write the CSV template, JSON schema and annotator guide")
    mode.add_argument("--print-schema", action="store_true", help="dump the JSON schema to stdout")

    parser.add_argument("--annotator", help="annotator handle (required with --import-csv)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"working directory for annotation artefacts (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"target number of accepted records (default: {DEFAULT_TARGET})")
    parser.add_argument("--min-annotators", type=int, default=2,
                        help="independent annotators required before a record can be accepted (default: 2)")
    parser.add_argument("--strict", action="store_true",
                        help="also dispute records where any field was checked by only one annotator")
    parser.add_argument("--auc-decimals", type=int, default=3,
                        help="decimals AUC values are rounded to before comparison (default: 3)")
    parser.add_argument("--verify-pmids", action="store_true",
                        help="check PMIDs resolve in PubMed (network; cached in pmid_cache.json)")

    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)

    if args.print_schema:
        print(json.dumps(annotation_json_schema(), indent=2, ensure_ascii=False))
        return 0
    if args.make_template:
        make_template(out_dir, args.target)
        return 0
    if args.status:
        return cmd_status(out_dir, args.target)
    if args.import_csv:
        if not args.annotator:
            raise SystemExit("[error] --import-csv requires --annotator <name>")
        import_csv(Path(args.import_csv), args.annotator, out_dir)
        return 0
    if args.agreement:
        return cmd_agreement(args, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
