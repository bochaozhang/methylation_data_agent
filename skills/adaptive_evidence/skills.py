"""
Real evidence-fetch Skills for the adaptive agent (path B).

Each Skill wraps an EXISTING production fetch utility (no new network code) and is
adapted to a LangChain StructuredTool via skills.base.to_tool(), so the model can
`bind_tools` and call them. Tool bodies return truncated, model-readable text; on
any failure they return a "(... failed: ...)" string so the loop never crashes.

Clients come from the SkillContext baked in by to_tool(): ctx.geo_client (GEOClient)
and ctx.lit_client (LiteratureClient).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from skills.base import Skill, SkillContext, to_tool

_OBS_LIMIT = 4000  # cap observation text fed back to the model


def _truncate(s: Optional[str], n: int = _OBS_LIMIT) -> str:
    if not s:
        return "(empty)"
    s = str(s)
    return s if len(s) <= n else s[:n] + f"\n…[truncated, {len(s)} chars total]"


# --------------------------------------------------------------------------- #
#  Pydantic args schemas (one per tool signature)                              #
# --------------------------------------------------------------------------- #

class _PmidArgs(BaseModel):
    pmid: str = Field(..., description="PubMed ID (PMID), digits only")


class _QueryArgs(BaseModel):
    query: str = Field(..., description="PubMed search query (e.g. the GSE accession or dataset title)")


class _AccArgs(BaseModel):
    accession: str = Field(..., description="GSE accession, e.g. GSE123456")
    group: str = Field("", description="optional sample-type hint to expand (e.g. unknown, plasma)")


class _ConcludeArgs(BaseModel):
    outcome: str = Field(..., description="final outcome: download | exclude | manual_review")
    reason: str = Field(..., description="one sentence: what the samples are and why this outcome")
    confirmed_sample_type: str = Field(
        "unknown",
        description="plasma|tumor|adjacent|normal|wbc|cfdna|serum|whole_blood|cell_line|other|unknown",
    )
    confirmed_cancer_type: Optional[str] = Field(None, description="canonical English cancer name, or null")
    notes: str = Field("", description="caveats; empty string if none")
    reasoning: str = Field("", description="short chain: what evidence confirmed the decision")


# --------------------------------------------------------------------------- #
#  Skills                                                                      #
# --------------------------------------------------------------------------- #

class FetchAbstractSkill(Skill):
    name = "fetch_abstract"
    description = (
        "Fetch the PubMed abstract for a PMID. Use when the dataset HAS a PMID but the "
        "abstract was not fetched or came back empty."
    )
    args_schema = _PmidArgs

    def run(self, ctx: SkillContext, pmid: str = "") -> str:
        geo = ctx.geo_client
        if geo is None or not pmid:
            return "(no geo_client / empty pmid)"
        try:
            ab = geo.fetch_pubmed_abstract(str(pmid))
            return _truncate(ab) if ab else f"(no abstract for PMID {pmid})"
        except Exception as e:  # noqa: BLE001
            return f"(fetch_abstract failed: {e})"


class ReverseLookupSkill(Skill):
    name = "pubmed_reverse_lookup"
    description = (
        "Search PubMed by GSE accession/title to find a linked paper and return its "
        "abstract + PMID. Use when the dataset has NO PMID."
    )
    args_schema = _QueryArgs

    def run(self, ctx: SkillContext, query: str = "") -> str:
        lit = ctx.lit_client
        if lit is None or not query:
            return "(no lit_client / empty query)"
        try:
            papers = lit.search_pubmed(str(query), max_results=5)
            if not papers:
                return f"(no PubMed papers found for: {query})"
            out = []
            for p in papers[:3]:
                pmid = p.get("pmid") or p.get("uid") or "?"
                out.append(
                    f"PMID {pmid}: {(p.get('title') or '')[:160]}\n"
                    f"Abstract: {_truncate(p.get('abstract'), 1500)}"
                )
            return "\n\n".join(out)
        except Exception as e:  # noqa: BLE001
            return f"(pubmed_reverse_lookup failed: {e})"


class FullTextSkill(Skill):
    name = "fetch_full_text"
    description = (
        "Fetch PMC open-access full text (focused on Methods / Data Availability). Use "
        "when the abstract is available but insufficient (treatment status, sample "
        "counts, or file types unclear). Returns 'not open access' if unavailable."
    )
    args_schema = _PmidArgs

    def run(self, ctx: SkillContext, pmid: str = "") -> str:
        lit = ctx.lit_client
        if lit is None or not pmid:
            return "(no lit_client / empty pmid)"
        try:
            txt = lit.get_pmc_data_availability(str(pmid)) or lit.get_pmc_fulltext(str(pmid))
            return _truncate(txt) if txt else f"(PMID {pmid}: not open access / no PMC full text)"
        except Exception as e:  # noqa: BLE001
            return f"(fetch_full_text failed: {e})"


class SuppTableSkill(Skill):
    name = "fetch_supplementary_table"
    description = (
        "List GEO supplementary files for the series and preview the first text-like "
        "one. Use to confirm sample types / counts / file forms from metadata files."
    )
    args_schema = _AccArgs

    def run(self, ctx: SkillContext, accession: str = "", group: str = "") -> str:
        geo = ctx.geo_client
        if geo is None or not accession:
            return "(no geo_client / empty accession)"
        try:
            urls = geo._get_supplementary_urls(str(accession)) or []
            if not urls:
                return f"(no supplementary files for {accession})"
            lines = [f"Supplementary files ({len(urls)}):"]
            lines += [f"- {u.rsplit('/', 1)[-1]}" for u in urls[:15]]
            for u in urls[:3]:
                try:
                    head = geo.fetch_file_head(u, max_bytes=4096)
                    txt = head.decode("utf-8", errors="ignore") if isinstance(head, bytes) else str(head)
                    if txt.strip() and not txt.startswith("\x00"):
                        lines.append(f"\nPreview [{u.rsplit('/', 1)[-1]}]:\n" + _truncate(txt, 2500))
                        break
                except Exception:  # noqa: BLE001  -- binary/unreadable file; skip
                    continue
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"(fetch_supplementary_table failed: {e})"


class MoreGsmSkill(Skill):
    name = "fetch_more_gsm"
    description = (
        "Fetch more representative GSM sample details (wider sampling) to clarify sample "
        "types, especially for 'unknown' groups."
    )
    args_schema = _AccArgs

    def run(self, ctx: SkillContext, accession: str = "", group: str = "") -> str:
        geo = ctx.geo_client
        if geo is None or not accession:
            return "(no geo_client / empty accession)"
        try:
            wanted = (group or "").strip()
            details = geo.get_representative_gsm_details(str(accession), wanted_sample_type=wanted)
            if not details:
                details = geo.get_all_gsm_metadata(str(accession))
            if not details:
                return f"(no GSM details for {accession})"
            lines = [f"GSM details ({len(details)}):"]
            for g in details[:20]:
                ch = g.get("characteristics") or {}
                chs = "; ".join(f"{k}: {v}" for k, v in list(ch.items())[:6])
                lines.append(
                    f"- GSM {g.get('gsm','?')} [group={g.get('group','?')}]: "
                    f"source={g.get('source_name','')!r}, molecule={g.get('molecule','')!r}, {chs}"
                )
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            return f"(fetch_more_gsm failed: {e})"


class ConcludeSkill(Skill):
    """Its run() is vestigial — the agent intercepts `conclude` and reads args directly
    to build the merged verdict. Kept so the tool set is uniform and self-describing."""
    name = "conclude"
    description = (
        "Conclude: stop gathering evidence and emit the final verdict. Call this once you "
        "have enough evidence, or once you determine the dataset should be excluded / "
        "cannot be resolved (then outcome=manual_review)."
    )
    args_schema = _ConcludeArgs

    def run(self, ctx: SkillContext, outcome: str = "manual_review", reason: str = "",
            confirmed_sample_type: str = "unknown", confirmed_cancer_type: Optional[str] = None,
            notes: str = "", reasoning: str = "") -> Dict[str, Any]:
        return {
            "outcome": outcome, "reason": reason,
            "confirmed_sample_type": confirmed_sample_type,
            "confirmed_cancer_type": confirmed_cancer_type,
            "notes": notes, "reasoning": reasoning,
        }


_EVIDENCE_SKILL_CLASSES = [
    FetchAbstractSkill, ReverseLookupSkill, FullTextSkill,
    SuppTableSkill, MoreGsmSkill, ConcludeSkill,
]


def build_tools(ctx: SkillContext) -> List[Any]:
    """Instantiate all evidence skills and adapt them to StructuredTools via to_tool()."""
    return [to_tool(S())(ctx) for S in _EVIDENCE_SKILL_CLASSES]
