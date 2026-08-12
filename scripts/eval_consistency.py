"""
Run-to-run stability + cost baseline for orchestrator_v2 (AstaBench-style).

WHY: a performance number on its own is not interpretable. AstaBench's argument
is that every reported score needs two companions — how much the score moves if
you just run it again (stability), and what it cost to produce (cost). We have
neither. This script produces both for agents/orchestrator_v2.py.

METHOD: run the SAME query N times (default 5) through the real
run_methyagent_v2 — real LLM, real NCBI — and then measure:

  STABILITY (computed over "core PMIDs": papers found in EVERY run, so we are
  measuring extraction disagreement, not retrieval disagreement)
    - per-field agreement : mean fraction of runs holding the MODAL value of
                            cancer_type / sample_type / auc / dataset_ids
    - unanimity           : fraction of core PMIDs where a field is byte-identical
                            across ALL runs (the strict version of the above)
    - set-level           : mean pairwise Jaccard of the PMID set and of the
                            accession set (this IS retrieval stability)
    - headline            : overall field-level agreement rate, all fields pooled

  COST (per run, then mean +/- SD)
    - wall time, search_papers calls, tool calls (attempted + executed),
      LLM calls, and tokens when the backend reports them

INSTRUMENTATION: nothing in agents/ or utils/ is modified. run_methyagent_v2
already accepts an `llm=` override, and that one model object is threaded into
every downstream call (search_and_extract -> extraction_reviewer,
evaluate_geo_dataset, review_geo_verdict). So a single LangChain callback
handler attached to it sees the whole run. Tool-call counts are read straight
off the convergence guards the orchestrator already keeps (tool_calls /
refused_calls / search_calls).

NCBI COURTESY: N full runs hit PubMed/GEO N times. --sleep (default 20s) spaces
them out, and the script is RESUMABLE — every run is written to disk the moment
it finishes, so a crash or a rate-limit ban keeps the runs already paid for.
--analyze-only recomputes all metrics from those saved JSONs with zero network.

Layout:
    data/eval/consistency_<ts>/meta.json        session parameters
    data/eval/consistency_<ts>/run_01.json ...  one file per completed run
    data/eval/consistency_<ts>.json             final report (metrics + costs)

Run:
    python scripts/eval_consistency.py --query "..." --runs 5 --sleep 20
    python scripts/eval_consistency.py --resume data/eval/consistency_<ts>
    python scripts/eval_consistency.py --analyze-only data/eval/consistency_<ts>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import traceback
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_dotenv() -> None:
    """Populate os.environ from .env without clobbering the real environment."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

import yaml  # noqa: E402

DEFAULT_QUERY = "breast cancer plasma cfDNA methylation EPIC early detection"

# Fields compared across runs. "auc" is synthesised from performance_metrics —
# see _paper_fields(); the other three are read straight off the paper record.
COMPARED_FIELDS = ("cancer_type", "sample_type", "auc", "dataset_ids")

# Stand-in for "this run did not report the field at all". Kept distinct from a
# literal empty string so that "absent" and "extracted as empty" do not merge.
MISSING = "<missing>"


# --------------------------------------------------------------------------- #
#  Proxy resolution                                                            #
#                                                                              #
#  config/settings.yaml ships geo.proxy INTENTIONALLY BLANK — the real value is #
#  per-machine and lives in the environment. A naive cfg["geo"]["proxy"] read   #
#  therefore runs unproxied and gets rate-limit-blocked by NCBI. Resolve as     #
#  NCBI_PROXY -> HTTPS_PROXY -> config, and publish the winner back into        #
#  NCBI_PROXY so the orchestrator's own lookup (which checks NCBI_PROXY first)  #
#  picks it up without us editing that module.                                  #
# --------------------------------------------------------------------------- #
def resolve_proxy(config: Dict[str, Any]) -> str:
    proxy = (
        os.environ.get("NCBI_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or (config.get("geo") or {}).get("proxy")
        or ""
    )
    if proxy and not os.environ.get("NCBI_PROXY"):
        os.environ["NCBI_PROXY"] = proxy
    return proxy


# --------------------------------------------------------------------------- #
#  Cost meter                                                                  #
# --------------------------------------------------------------------------- #
def build_usage_meter():
    """
    A LangChain callback handler that counts LLM calls and sums token usage.

    Attached to the single chat-model instance handed to run_methyagent_v2, which
    is the same object every downstream tool uses — so this counts the whole run
    (orchestrator ReAct turns + Stage 1/2 extraction + both reviewer passes), not
    just the top-level loop.

    Token fields are best-effort: ChatOpenAI-backed backends populate
    llm_output["token_usage"] and/or message.usage_metadata, ChatZhipuAI may
    populate neither. Call counts do not depend on that and are always valid.
    """
    from langchain_core.callbacks.base import BaseCallbackHandler

    class UsageMeter(BaseCallbackHandler):
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._seen_runs: set = set()
            self.llm_calls = 0
            self.llm_errors = 0
            self.usage_reports = 0  # how many calls actually reported tokens
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0

        # -- call counting ------------------------------------------------- #
        def _count(self, run_id: Any) -> None:
            with self._lock:
                # Chat models fire on_chat_model_start; some paths also fire
                # on_llm_start. Dedupe on run_id so one call counts once.
                if run_id is not None:
                    if run_id in self._seen_runs:
                        return
                    self._seen_runs.add(run_id)
                self.llm_calls += 1

        def on_llm_start(self, serialized, prompts, **kwargs) -> None:  # noqa: D102
            self._count(kwargs.get("run_id"))

        def on_chat_model_start(self, serialized, messages, **kwargs) -> None:  # noqa: D102
            self._count(kwargs.get("run_id"))

        def on_llm_error(self, error, **kwargs) -> None:  # noqa: D102
            with self._lock:
                self.llm_errors += 1

        # -- token accounting ---------------------------------------------- #
        def on_llm_end(self, response, **kwargs) -> None:  # noqa: D102
            prompt = completion = total = 0
            found = False

            llm_output = getattr(response, "llm_output", None) or {}
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if isinstance(usage, dict) and usage:
                prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                completion = int(
                    usage.get("completion_tokens") or usage.get("output_tokens") or 0
                )
                total = int(usage.get("total_tokens") or 0)
                found = bool(prompt or completion or total)

            if not found:
                # Fallback: langchain-core >=0.3 puts usage on the message itself.
                for gen_list in getattr(response, "generations", None) or []:
                    for gen in gen_list or []:
                        meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                        if not meta:
                            continue
                        prompt += int(meta.get("input_tokens") or 0)
                        completion += int(meta.get("output_tokens") or 0)
                        total += int(meta.get("total_tokens") or 0)
                        found = True

            if not found:
                return
            if not total:
                total = prompt + completion

            with self._lock:
                self.usage_reports += 1
                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.total_tokens += total

        def snapshot(self) -> Dict[str, Any]:
            with self._lock:
                return {
                    "llm_calls": self.llm_calls,
                    "llm_errors": self.llm_errors,
                    "usage_reports": self.usage_reports,
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                    "tokens_available": self.usage_reports > 0,
                }

    return UsageMeter()


def build_metered_llm(config: Dict[str, Any]) -> Tuple[Any, Any, bool]:
    """
    Fresh chat model + usage meter for one run.

    Returns (llm, meter, attached). `attached` is False if the model instance
    refused the callbacks assignment — the run still proceeds, we just report
    llm_calls as unavailable rather than silently reporting zero.
    """
    from utils.llm_factory import get_llm

    llm = get_llm(config["llm"])
    meter = build_usage_meter()
    try:
        existing = list(getattr(llm, "callbacks", None) or [])
        llm.callbacks = existing + [meter]
        attached = True
    except Exception as exc:  # pragma: no cover - backend-dependent
        print(f"  ! could not attach usage meter to the LLM ({exc}); "
              f"LLM-call/token cost will be unavailable")
        attached = False
    return llm, meter, attached


# --------------------------------------------------------------------------- #
#  Field normalisation                                                         #
# --------------------------------------------------------------------------- #
def _norm_text(value: Any) -> str:
    if value is None:
        return MISSING
    s = str(value).strip().lower()
    return s or MISSING


def _norm_auc(paper: Dict[str, Any]) -> str:
    """
    Canonical AUC signature for one paper.

    Compared as the whole (training, validation, external) triple rather than a
    single number: which slot a model fills is itself part of the extraction
    decision, and collapsing to "the AUC" would hide a run that moved a value
    from validation to external while looking perfectly stable.
    """
    metrics = paper.get("performance_metrics") or {}
    if not isinstance(metrics, dict):
        return MISSING
    parts = []
    any_value = False
    for short, key in (("t", "auc_training"), ("v", "auc_validation"), ("e", "auc_external")):
        raw = metrics.get(key)
        if raw is None or raw == "":
            parts.append(f"{short}=-")
            continue
        any_value = True
        try:
            parts.append(f"{short}={round(float(raw), 4)}")
        except (TypeError, ValueError):
            parts.append(f"{short}={str(raw).strip().lower()}")
    return "|".join(parts) if any_value else MISSING


def _norm_accessions(value: Any) -> List[str]:
    """Uppercase, de-duplicate and sort an accession list (order is not signal)."""
    if not value:
        return []
    if isinstance(value, str):
        items: Iterable[Any] = [v for v in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]
    out = {str(v).strip().upper() for v in items if str(v).strip()}
    return sorted(out)


def _paper_fields(paper: Dict[str, Any]) -> Dict[str, str]:
    """The four compared fields of one paper record, as comparable strings."""
    accessions = _norm_accessions(paper.get("dataset_ids"))
    return {
        "cancer_type": _norm_text(paper.get("cancer_type")),
        "sample_type": _norm_text(paper.get("sample_type")),
        "auc": _norm_auc(paper),
        "dataset_ids": ",".join(accessions) if accessions else MISSING,
    }


def extract_run_view(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce a full orchestrator report to just what stability needs.

    Keeps only the first record per PMID: run_trace already dedupes by PMID, but
    records with no PMID cannot be deduped and are excluded here since they
    cannot be matched across runs either.
    """
    papers: Dict[str, Dict[str, str]] = {}
    accessions: set = set()
    for paper in report.get("papers") or []:
        pmid = str(paper.get("pmid") or "").strip()
        accessions.update(_norm_accessions(paper.get("dataset_ids")))
        if not pmid or pmid in papers:
            continue
        papers[pmid] = _paper_fields(paper)

    evaluated = sorted({
        str(e.get("accession") or "").strip().upper()
        for e in (report.get("gse_evaluated") or [])
        if str(e.get("accession") or "").strip()
    })
    writes = sorted({
        str(a).strip().upper() for a in (report.get("registry_writes") or []) if str(a).strip()
    })

    return {
        "papers": papers,
        "pmids": sorted(papers),
        "accessions": sorted(accessions),
        "gse_evaluated": evaluated,
        "registry_writes": writes,
        "papers_found": report.get("papers_found"),
        "agent_summary": report.get("agent_summary"),
    }


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0  # two runs that both found nothing agree perfectly
    return len(sa & sb) / len(sa | sb)


def mean_pairwise_jaccard(sets: List[Sequence[str]]) -> Optional[float]:
    pairs = list(combinations(range(len(sets)), 2))
    if not pairs:
        return None
    return sum(jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def compute_stability(views: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Field agreement, unanimity and set-level overlap across N run views.

    Field metrics are restricted to CORE PMIDs — papers present in every run —
    because a field cannot disagree in a run that never saw the paper. Retrieval
    variance is measured separately by the Jaccard numbers.
    """
    n = len(views)
    pmid_sets = [v["pmids"] for v in views]
    core = sorted(set.intersection(*[set(s) for s in pmid_sets])) if views else []

    per_field: Dict[str, Any] = {}
    per_pmid: List[Dict[str, Any]] = []
    all_agreements: List[float] = []
    all_unanimous: List[bool] = []

    field_agreements: Dict[str, List[float]] = {f: [] for f in COMPARED_FIELDS}
    field_unanimous: Dict[str, List[bool]] = {f: [] for f in COMPARED_FIELDS}

    for pmid in core:
        entry: Dict[str, Any] = {"pmid": pmid, "fields": {}}
        for field in COMPARED_FIELDS:
            values = [v["papers"][pmid][field] for v in views]
            counts = Counter(values)
            modal_value, modal_count = counts.most_common(1)[0]
            agreement = modal_count / n
            unanimous = modal_count == n

            field_agreements[field].append(agreement)
            field_unanimous[field].append(unanimous)
            all_agreements.append(agreement)
            all_unanimous.append(unanimous)

            entry["fields"][field] = {
                "agreement": round(agreement, 4),
                "unanimous": unanimous,
                "modal_value": modal_value,
                "distinct_values": len(counts),
                # Only kept when they disagree — otherwise the report bloats with
                # N copies of the same string for every stable field.
                "values": None if unanimous else values,
            }
        per_pmid.append(entry)

    for field in COMPARED_FIELDS:
        vals = field_agreements[field]
        uni = field_unanimous[field]
        per_field[field] = {
            "agreement": round(statistics.fmean(vals), 4) if vals else None,
            "unanimity": round(sum(uni) / len(uni), 4) if uni else None,
            "n_core_pmids": len(vals),
        }

    return {
        "n_runs": n,
        "core_pmids": core,
        "n_core_pmids": len(core),
        "pmid_set_sizes": [len(s) for s in pmid_sets],
        "per_field": per_field,
        "per_pmid": per_pmid,
        "set_level": {
            "mean_pairwise_jaccard_pmids": _round(mean_pairwise_jaccard(pmid_sets)),
            "mean_pairwise_jaccard_accessions": _round(
                mean_pairwise_jaccard([v["accessions"] for v in views])
            ),
            "mean_pairwise_jaccard_gse_evaluated": _round(
                mean_pairwise_jaccard([v["gse_evaluated"] for v in views])
            ),
            "mean_pairwise_jaccard_registry_writes": _round(
                mean_pairwise_jaccard([v["registry_writes"] for v in views])
            ),
        },
        "headline": {
            "field_agreement_rate": (
                round(statistics.fmean(all_agreements), 4) if all_agreements else None
            ),
            "field_unanimity_rate": (
                round(sum(all_unanimous) / len(all_unanimous), 4) if all_unanimous else None
            ),
            "n_field_observations": len(all_agreements),
        },
    }


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(value, digits)


def mean_sd(values: Sequence[float]) -> Dict[str, Any]:
    """Mean and sample SD (ddof=1); SD is None for n<2 rather than a fake 0."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.fmean(clean), 4),
        "sd": round(statistics.stdev(clean), 4) if len(clean) > 1 else None,
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        "n": len(clean),
    }


COST_KEYS = (
    ("wall_time_s", "wall time (s)"),
    ("search_calls", "search_papers calls"),
    ("tool_calls", "tool calls (attempted)"),
    ("tool_calls_executed", "tool calls (executed)"),
    ("refused_calls", "tool calls (refused)"),
    ("llm_calls", "LLM calls"),
    ("prompt_tokens", "prompt tokens"),
    ("completion_tokens", "completion tokens"),
    ("total_tokens", "total tokens"),
    ("papers_found", "papers found"),
)


def compute_cost(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in runs if r.get("ok")]
    out: Dict[str, Any] = {"n_runs_ok": len(ok), "n_runs_failed": len(runs) - len(ok)}
    for key, _label in COST_KEYS:
        out[key] = mean_sd([r.get("cost", {}).get(key) for r in ok])
    out["tokens_available"] = any(r.get("cost", {}).get("tokens_available") for r in ok)
    return out


# --------------------------------------------------------------------------- #
#  Session I/O (resumability)                                                  #
# --------------------------------------------------------------------------- #
def run_path(session_dir: Path, index: int) -> Path:
    return session_dir / f"run_{index:02d}.json"


def load_saved_runs(session_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for path in sorted(session_dir.glob("run_*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  ! skipping unreadable {path.name}: {exc}")
    runs.sort(key=lambda r: r.get("index", 0))
    return runs


def save_run(session_dir: Path, record: Dict[str, Any]) -> Path:
    path = run_path(session_dir, record["index"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half-run behind
    return path


# --------------------------------------------------------------------------- #
#  One run                                                                     #
# --------------------------------------------------------------------------- #
def execute_run(
    index: int,
    query: str,
    config: Dict[str, Any],
    config_path: str,
    save_log: bool,
) -> Dict[str, Any]:
    """Execute one full orchestrator_v2 run and return its persisted record."""
    from agents.orchestrator_v2 import run_methyagent_v2

    llm, meter, attached = build_metered_llm(config)

    started = datetime.now().astimezone().isoformat()
    t0 = time.perf_counter()
    error = None
    report: Dict[str, Any] = {}
    try:
        report = run_methyagent_v2(
            query, config_path=config_path, llm=llm, save_log=save_log
        )
    except Exception as exc:  # keep prior runs; one bad run must not kill the eval
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    wall = time.perf_counter() - t0

    usage = meter.snapshot()
    tool_calls = report.get("tool_calls")
    refused = report.get("refused_calls")
    executed = (
        tool_calls - refused
        if isinstance(tool_calls, int) and isinstance(refused, int)
        else None
    )

    cost = {
        "wall_time_s": round(wall, 3),
        "search_calls": report.get("search_calls"),
        "tool_calls": tool_calls,
        "refused_calls": refused,
        "tool_calls_executed": executed,
        "skipped_duplicate_calls": len(report.get("skipped_duplicate_calls") or []),
        "llm_calls": usage["llm_calls"] if attached else None,
        "llm_errors": usage["llm_errors"] if attached else None,
        "prompt_tokens": usage["prompt_tokens"] if usage["tokens_available"] else None,
        "completion_tokens": (
            usage["completion_tokens"] if usage["tokens_available"] else None
        ),
        "total_tokens": usage["total_tokens"] if usage["tokens_available"] else None,
        "tokens_available": usage["tokens_available"],
        "meter_attached": attached,
        "papers_found": report.get("papers_found"),
    }

    return {
        "index": index,
        "ok": error is None,
        "error": error,
        "started_at": started,
        "query": query,
        "cost": cost,
        "view": extract_run_view(report) if error is None else None,
        "log_path": report.get("log_path"),
    }


# --------------------------------------------------------------------------- #
#  Reporting                                                                   #
# --------------------------------------------------------------------------- #
def _fmt(stat: Dict[str, Any], digits: int = 2) -> str:
    if stat.get("mean") is None:
        return "n/a"
    mean = f"{stat['mean']:.{digits}f}"
    if stat.get("sd") is None:
        return mean
    return f"{mean} ± {stat['sd']:.{digits}f}"


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def print_summary(result: Dict[str, Any]) -> None:
    stab = result.get("stability")
    cost = result["cost"]
    runs = result["runs"]

    print()
    print("=" * 72)
    print("  CONSISTENCY + COST BASELINE — orchestrator_v2")
    print("=" * 72)
    print(f"  query    : {result['query']}")
    print(f"  backend  : {result['backend']} / {result['model']}")
    print(f"  runs     : {cost['n_runs_ok']} ok, {cost['n_runs_failed']} failed")
    print(f"  session  : {result['session_dir']}")

    print()
    print("-" * 72)
    print("  STABILITY")
    print("-" * 72)
    if not stab:
        print("  not computed — fewer than 2 successful runs")
    else:
        print(f"  PMIDs per run          : {stab['pmid_set_sizes']}")
        print(f"  core PMIDs (in all {stab['n_runs']}) : {stab['n_core_pmids']}")
        head = stab["headline"]
        print()
        print(f"  >> FIELD AGREEMENT RATE : {_pct(head['field_agreement_rate'])}"
              f"   ({head['n_field_observations']} field observations)")
        print(f"  >> FIELD UNANIMITY RATE : {_pct(head['field_unanimity_rate'])}")
        print()
        print(f"  {'field':<16}{'agreement':>12}{'unanimity':>12}")
        for field in COMPARED_FIELDS:
            f = stab["per_field"][field]
            print(f"  {field:<16}{_pct(f['agreement']):>12}{_pct(f['unanimity']):>12}")
        sl = stab["set_level"]
        print()
        print("  set-level (mean pairwise Jaccard)")
        print(f"    PMID set          : {_pct(sl['mean_pairwise_jaccard_pmids'])}")
        print(f"    accession set     : {_pct(sl['mean_pairwise_jaccard_accessions'])}")
        print(f"    GSE evaluated     : {_pct(sl['mean_pairwise_jaccard_gse_evaluated'])}")
        print(f"    registry writes   : {_pct(sl['mean_pairwise_jaccard_registry_writes'])}")

        unstable = [
            (e["pmid"], field, e["fields"][field]["values"])
            for e in stab["per_pmid"]
            for field in COMPARED_FIELDS
            if not e["fields"][field]["unanimous"]
        ]
        if unstable:
            print()
            print(f"  disagreements ({len(unstable)}):")
            for pmid, field, values in unstable[:15]:
                shown = [str(v)[:60] for v in values]
                print(f"    PMID {pmid} · {field}: {shown}")
            if len(unstable) > 15:
                print(f"    ... and {len(unstable) - 15} more (see JSON)")

    print()
    print("-" * 72)
    print("  COST PER RUN (mean ± SD)")
    print("-" * 72)
    for key, label in COST_KEYS:
        digits = 2 if key == "wall_time_s" else (1 if "token" not in key else 0)
        print(f"  {label:<26}{_fmt(cost[key], digits):>20}")
    if not cost["tokens_available"]:
        print("  (token usage not reported by this backend — call counts are exact)")

    print()
    print("  per-run detail")
    print(f"  {'#':<4}{'ok':<5}{'wall_s':>9}{'search':>8}{'tools':>7}{'llm':>7}{'tokens':>10}{'papers':>8}")
    for r in runs:
        c = r.get("cost", {})
        print(
            f"  {r['index']:<4}{'y' if r.get('ok') else 'N':<5}"
            f"{_num(c.get('wall_time_s'), 2):>9}{_num(c.get('search_calls')):>8}"
            f"{_num(c.get('tool_calls')):>7}{_num(c.get('llm_calls')):>7}"
            f"{_num(c.get('total_tokens')):>10}{_num(c.get('papers_found')):>8}"
        )
    print()
    print(f"  report written: {result['report_path']}")
    print("=" * 72)


def _num(value: Any, digits: int = 0) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def analyze(
    runs: List[Dict[str, Any]],
    query: str,
    config: Dict[str, Any],
    session_dir: Path,
    report_path: Path,
) -> Dict[str, Any]:
    ok_views = [r["view"] for r in runs if r.get("ok") and r.get("view")]
    stability = compute_stability(ok_views) if len(ok_views) >= 2 else None
    if len(ok_views) == 1:
        print("  ! only 1 successful run — stability needs at least 2")

    llm_cfg = config.get("llm", {}) or {}
    return {
        "schema": "eval_consistency/1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "query": query,
        "backend": llm_cfg.get("backend"),
        "model": (
            os.environ.get("ZHIPU_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or llm_cfg.get("model")
            or "(from env)"
        ),
        "orchestrator_guards": config.get("orchestrator", {}),
        "compared_fields": list(COMPARED_FIELDS),
        "session_dir": str(session_dir),
        "report_path": str(report_path),
        "stability": stability,
        "cost": compute_cost(runs),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run-to-run stability + cost baseline for orchestrator_v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/eval_consistency.py --query \"...\" --runs 5 --sleep 20\n"
            "  python scripts/eval_consistency.py --resume data/eval/consistency_<ts>\n"
            "  python scripts/eval_consistency.py --analyze-only data/eval/consistency_<ts>\n"
        ),
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="query to repeat")
    # default=None so a resume can tell "user asked for N" from "user said nothing",
    # and keep the session's original N in the latter case.
    parser.add_argument("--runs", type=int, default=None, help="number of runs (default 5)")
    parser.add_argument(
        "--sleep", type=float, default=20.0,
        help="seconds between runs — NCBI courtesy pause (default 20)",
    )
    parser.add_argument("--config", default=str(ROOT / "config" / "settings.yaml"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "eval"))
    parser.add_argument(
        "--resume", metavar="SESSION_DIR",
        help="continue an interrupted session directory (keeps its completed runs)",
    )
    parser.add_argument(
        "--analyze-only", metavar="SESSION_DIR",
        help="recompute metrics from saved runs — no network, no LLM calls",
    )
    parser.add_argument(
        "--retry-failed", action="store_true",
        help="on resume, re-execute runs that previously errored",
    )
    parser.add_argument(
        "--no-run-log", action="store_true",
        help="skip the orchestrator's own per-run JSON under data/methylation",
    )
    args = parser.parse_args()
    runs_requested = args.runs is not None
    if args.runs is None:
        args.runs = 5

    config_path = str(Path(args.config).resolve())
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- session setup (new / resume / analyze-only) --------------------- #
    session_arg = args.analyze_only or args.resume
    if session_arg:
        session_dir = Path(session_arg).resolve()
        if not session_dir.is_dir():
            print(f"ERROR: session dir not found: {session_dir}")
            return 2
        meta_path = session_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        query = meta.get("query", args.query)
        stamp = meta.get("stamp", session_dir.name.replace("consistency_", ""))
        # Resuming means "finish this session", so the session's own N wins unless
        # the user explicitly asked for a different one on the command line.
        target_runs = args.runs if runs_requested else int(meta.get("runs", args.runs))
        if args.resume and meta.get("query") and args.query != meta["query"]:
            if args.query != DEFAULT_QUERY:
                print("  ! --query differs from the session's query — a session must "
                      "compare runs of ONE query, so the session's is used:")
                print(f"    {query}")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = (out_dir / f"consistency_{stamp}").resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
        query = args.query
        target_runs = args.runs
        (session_dir / "meta.json").write_text(
            json.dumps(
                {
                    "query": query,
                    "runs": target_runs,
                    "sleep": args.sleep,
                    "stamp": stamp,
                    "config_path": config_path,
                    "created_at": datetime.now().astimezone().isoformat(),
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    report_path = out_dir / f"consistency_{stamp}.json"
    saved = load_saved_runs(session_dir)

    # ---- execute the missing runs ---------------------------------------- #
    if not args.analyze_only:
        proxy = resolve_proxy(config)
        print(f"Session : {session_dir}")
        print(f"Query   : {query}")
        print(f"Runs    : {target_runs} (target) | {len(saved)} already saved")
        print(f"Proxy   : {proxy or '(none — expect NCBI rate limiting)'}")
        print(f"Sleep   : {args.sleep}s between runs")

        done = {r["index"]: r for r in saved}
        if args.retry_failed:
            done = {i: r for i, r in done.items() if r.get("ok")}

        executed_any = False
        for index in range(1, target_runs + 1):
            if index in done:
                print(f"\n[run {index}/{target_runs}] already saved — skipping")
                continue
            if executed_any and args.sleep > 0:
                print(f"  sleeping {args.sleep}s before the next run (NCBI courtesy)...")
                time.sleep(args.sleep)

            print(f"\n[run {index}/{target_runs}] starting at "
                  f"{datetime.now().strftime('%H:%M:%S')}")
            record = execute_run(
                index=index,
                query=query,
                config=config,
                config_path=config_path,
                save_log=not args.no_run_log,
            )
            executed_any = True
            path = save_run(session_dir, record)  # persisted immediately: resumable
            done[index] = record
            c = record["cost"]
            status = "ok" if record["ok"] else f"FAILED ({record['error']})"
            print(f"[run {index}/{target_runs}] {status} | {c['wall_time_s']}s | "
                  f"papers={_num(c['papers_found'])} tools={_num(c['tool_calls'])} "
                  f"llm={_num(c['llm_calls'])} tokens={_num(c['total_tokens'])}")
            print(f"  saved -> {path}")

        saved = load_saved_runs(session_dir)

    if not saved:
        print("ERROR: no runs available to analyze.")
        return 1

    result = analyze(saved, query, config, session_dir, report_path)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    # Second copy inside the session dir so a session is self-contained.
    (session_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
