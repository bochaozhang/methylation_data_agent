"""
Agent 2: LiteratureAgent

Searches PubMed, PMC full text, and bioRxiv for methylation papers,
extracts dataset accession numbers, checks the shared registry to
avoid duplicates, and downloads only datasets not already covered
by Agent 1 (DatabaseAgent).

Sample type awareness:
  When the user specifies a sample type (e.g. cfDNA, plasma, WBC),
  LiteratureAgent adds sample type terms to the PubMed search query
  and filters literature-mined datasets by sample type keywords
  in title/abstract.

LLM-assisted PDF extraction (Layer 2+3):
  - Triggered when regex finds no accessions in a PDF
  - Bilingual prompt (English + Chinese)
  - Three confidence levels: high (auto-download), medium (pending_review), low (discard)
  - GEO API verification filters hallucinated accessions
  - DOI-keyed SQLite cache avoids redundant LLM calls
"""
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from registry.registry import Registry
from state.graph_state import MethyAgentState
from tools.download_tools import DownloadEngine, build_geo_download_tasks
from tools.geo_tools import GEOClient
from tools.parser_tools import SAMPLE_TYPE_PUBMED_TERMS, SAMPLE_TYPE_RELATED, TCGA_CODE_TO_ENGLISH
from tools.pubmed_tools import LiteratureClient, PDFSupplementaryParser
from utils.logger import get_logger
from utils.llm_factory import get_llm

logger = get_logger(__name__)


class LiteratureAgent:
    """
    Agent 2: Mines literature for methylation datasets and supplements
    the registry with datasets not found by DatabaseAgent.

    Args:
        config: Full settings dict from settings.yaml.
        registry: Shared Registry instance (same as Agent 1).
    """

    def __init__(self, config: Dict[str, Any], registry: Registry):
        self.config = config
        self.registry = registry
        self.llm = get_llm(config["llm"])

        ncbi_key = os.environ.get(config["geo"].get("api_key_env", ""), "")
        self.lit_client = LiteratureClient(ncbi_api_key=ncbi_key or None)
        self.geo_client = GEOClient(api_key=ncbi_key or None)

        lit_cfg = config.get("literature", {})
        self.max_papers = lit_cfg.get("max_papers_per_query", 50)
        self.sources = lit_cfg.get("sources", ["pubmed", "pmc", "biorxiv"])
        self.extract_supplementary = lit_cfg.get("extract_supplementary", True)

        # LLM extraction configuration
        llm_ext_cfg = lit_cfg.get("llm_extraction", {})
        self.llm_extraction_enabled = llm_ext_cfg.get("enabled", True)
        self.llm_extraction_trigger = llm_ext_cfg.get("trigger", "regex_fallback")
        self.llm_min_confidence = llm_ext_cfg.get("min_confidence", "medium")
        self.llm_verify_geo = llm_ext_cfg.get("verify_with_geo_api", True)
        self.llm_cache_db = llm_ext_cfg.get(
            "cache_db_path", "/workspace/methyagent_llm_cache.db"
        )

        # Initialize PDF parser with LLM support
        self.pdf_parser = PDFSupplementaryParser(
            llm=self.llm if self.llm_extraction_enabled else None,
            geo_client=self.geo_client if self.llm_verify_geo else None,
            cache_db_path=self.llm_cache_db,
            model_name=config.get("llm", {}).get("model", "unknown"),
            verify_accessions=self.llm_verify_geo,
            min_confidence=self.llm_min_confidence,
        )

        self.downloader = DownloadEngine(
            output_dir=config["download"]["output_dir"],
            max_concurrent=config["download"]["max_concurrent"],
            retry_attempts=config["download"]["retry_attempts"],
            retry_delay=config["download"]["retry_delay"],
            chunk_size_mb=config["download"]["chunk_size_mb"],
            timeout=config["download"]["timeout"],
        )

    # ------------------------------------------------------------------ #
    #  LangGraph node entry point                                          #
    # ------------------------------------------------------------------ #

    def run(self, state: MethyAgentState) -> MethyAgentState:
        """
        Main LangGraph node function.
        Runs after DatabaseAgent. Reads the registry to know what's
        already downloaded, then mines literature for additional datasets.
        """
        intent = state.get("parsed_intent", {})
        logger.info(f"LiteratureAgent starting. Intent: {intent}")

        papers_found = []
        lit_candidates = []
        downloaded = []
        failed = []
        skipped = []
        pending_review = []
        errors = list(state.get("error_log", []))

        # ---- Step 1: Build search queries ----
        pubmed_query = self._build_pubmed_query(intent)
        year_start = intent.get("year_start")
        year_end = intent.get("year_end")

        logger.info(f"PubMed query: {pubmed_query}")

        # ---- Step 2: Search literature sources ----
        if "pubmed" in self.sources or "pmc" in self.sources:
            try:
                pubmed_papers = self.lit_client.search_pubmed(
                    query=pubmed_query,
                    max_results=self.max_papers,
                    year_start=year_start,
                    year_end=year_end,
                )
                papers_found.extend(pubmed_papers)
                logger.info(f"PubMed: {len(pubmed_papers)} papers found")
            except Exception as e:
                logger.error(f"PubMed search failed: {e}")
                errors.append(f"LiteratureAgent PubMed: {e}")

        if "biorxiv" in self.sources:
            try:
                biorxiv_papers = self.lit_client.search_biorxiv(
                    query=pubmed_query,
                    year_start=year_start,
                    year_end=year_end,
                    max_results=30,
                )
                papers_found.extend(biorxiv_papers)
                logger.info(f"bioRxiv: {len(biorxiv_papers)} preprints found")
            except Exception as e:
                logger.warning(f"bioRxiv search failed: {e}")

        # ---- Step 3: Extract accessions from papers ----
        fetch_fulltext = "pmc" in self.sources
        enriched_papers = self.lit_client.mine_accessions_from_papers(
            papers_found, fetch_fulltext=fetch_fulltext
        )

        # Collect all unique accessions from literature
        sample_types = intent.get("sample_types", [])
        all_lit_accessions = self._collect_accessions(enriched_papers, sample_types=sample_types)
        logger.info(f"Total unique accessions from literature: {len(all_lit_accessions)}")

        # ---- Step 4: Dedup against registry (Agent 1 results) ----
        existing_accessions = self.registry.get_accession_set()

        new_accessions = []
        for acc_info in all_lit_accessions:
            acc = acc_info["accession"]
            if acc in existing_accessions:
                logger.info(f"Skipping {acc}: already covered by DatabaseAgent")
                skipped.append(acc)
            else:
                new_accessions.append(acc_info)
                lit_candidates.append(acc_info)

        logger.info(
            f"Literature: {len(new_accessions)} new accessions "
            f"(skipped {len(skipped)} already in registry)"
        )

        # ---- Step 5: Fetch metadata and register new datasets ----
        download_tasks = []
        for acc_info in new_accessions:
            acc = acc_info["accession"]
            source = acc_info.get("source", "GEO")
            pmid = acc_info.get("pmid")
            needs_review = acc_info.get("needs_review", False)
            llm_evidence = acc_info.get("llm_evidence", None)

            # Fetch GEO metadata for array datasets
            metadata = None
            if source == "GEO" and acc.startswith("GSE"):
                try:
                    metadata = self.geo_client.get_series_metadata(acc)
                except Exception as e:
                    logger.warning(f"Could not fetch metadata for {acc}: {e}")

            # Determine download status
            if needs_review:
                dl_status = Registry.STATUS_PENDING_REVIEW
                pending_review.append(acc)
            else:
                dl_status = "pending"

            # Register in registry
            self.registry.upsert_dataset(
                accession=acc,
                source=source,
                discovered_by="agent2",
                data_type=metadata.get("data_type") if metadata else None,
                cancer_type=metadata.get("cancer_type") if metadata else None,
                platform=(
                    metadata.get("platform_canonical") if metadata else None
                ),
                sample_count=metadata.get("sample_count") if metadata else None,
                year=metadata.get("year") if metadata else acc_info.get("year"),
                title=metadata.get("title") if metadata else acc_info.get("title"),
                paper_pmid=pmid,
                sample_type=metadata.get("sample_type") if metadata else None,
                download_status=dl_status,
                needs_review=needs_review,
                llm_evidence=llm_evidence,
            )
            self.registry.log_event(
                acc, "start",
                f"Registered by LiteratureAgent (PMID: {pmid}, needs_review={needs_review})"
            )

            # Only queue download for high-confidence (not pending_review) datasets
            if needs_review:
                logger.info(
                    "Skipping download for %s: pending human review (LLM medium confidence)",
                    acc,
                )
                continue

            # Build download tasks
            if metadata and source == "GEO":
                tasks = build_geo_download_tasks(
                    metadata, self.config["download"]["output_dir"]
                )
            elif acc_info.get("direct_url"):
                tasks = [{
                    "accession": acc,
                    "url": acc_info["direct_url"],
                    "filename": None,
                    "subdir": acc,
                }]
            else:
                # Fallback: try GEO series matrix
                prefix = acc[:-3] + "nnn" if acc.startswith("GSE") else acc
                fallback_url = (
                    f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{acc}/"
                    f"matrix/{acc}_series_matrix.txt.gz"
                )
                tasks = [{
                    "accession": acc,
                    "url": fallback_url,
                    "filename": f"{acc}_series_matrix.txt.gz",
                    "subdir": acc,
                }]

            download_tasks.extend(tasks)

        # ---- Step 6: Execute downloads ----
        if download_tasks:
            results = self.downloader.download_many_sync(download_tasks)

            acc_results: Dict[str, List] = {}
            for r in results:
                acc = r["accession"]
                acc_results.setdefault(acc, []).append(r)

            for acc, acc_res in acc_results.items():
                all_done = all(r["status"] == "done" for r in acc_res)

                if all_done:
                    local_path = acc_res[0]["local_path"]
                    file_size = sum(r.get("file_size_bytes", 0) for r in acc_res)
                    self.registry.update_status(
                        acc, "done",
                        local_path=str(local_path),
                        file_size_bytes=file_size,
                    )
                    self.registry.log_event(acc, "done", f"Downloaded by LiteratureAgent")
                    downloaded.append(acc)
                else:
                    error_msgs = [r.get("error", "") for r in acc_res if r["status"] == "failed"]
                    self.registry.update_status(acc, "failed")
                    self.registry.log_event(acc, "error", "; ".join(error_msgs))
                    failed.append(acc)
                    errors.append(f"LiteratureAgent: {acc} failed: {'; '.join(error_msgs)}")

        # ---- Step 7: Handle supplementary material links (with LLM) ----
        if self.extract_supplementary:
            supp_results = self._process_supplementary_links_v2(
                enriched_papers, existing_accessions
            )
            downloaded.extend(supp_results.get("downloaded", []))
            failed.extend(supp_results.get("failed", []))
            pending_review.extend(supp_results.get("pending_review", []))
            errors.extend(supp_results.get("errors", []))

        summary_msg = self._generate_summary_message(
            papers_found, lit_candidates, downloaded, failed, skipped, pending_review
        )

        return {
            **state,
            "papers_found": enriched_papers,
            "lit_candidates": lit_candidates,
            "lit_downloaded": downloaded,
            "lit_failed": failed,
            "lit_skipped": skipped,
            "lit_pending_review": pending_review,
            "error_log": errors,
            "messages": [AIMessage(content=summary_msg, name="LiteratureAgent")],
        }

    # ------------------------------------------------------------------ #
    #  Helper methods                                                      #
    # ------------------------------------------------------------------ #

    def _build_pubmed_query(self, intent: Dict[str, Any]) -> str:
        """Build a PubMed search query from parsed intent."""
        # Use LLM-generated query if available
        if intent.get("pubmed_search_query"):
            return intent["pubmed_search_query"]

        # Build manually
        parts = []

        cancer_type = intent.get("cancer_type", {})
        if isinstance(cancer_type, dict):
            mesh = cancer_type.get("mesh_term") or cancer_type.get("display", "")
            if mesh:
                parts.append(f'"{mesh}"[MeSH Terms]')
        elif isinstance(cancer_type, str) and cancer_type:
            parts.append(f'"{cancer_type}"[Title/Abstract]')
        elif intent.get("cancer_type_display"):
            # Rule-based parser: use canonical English name via TCGA code
            # (handles Chinese displays and partial English matches)
            if intent.get("cancer_type_code") and intent["cancer_type_code"] in TCGA_CODE_TO_ENGLISH:
                english = TCGA_CODE_TO_ENGLISH[intent["cancer_type_code"]]
                parts.append(f'"{english}"[Title/Abstract]')
            else:
                parts.append(f'"{intent["cancer_type_display"]}"[Title/Abstract]')

        platform = intent.get("platform")
        if platform:
            platform_terms = {
                "EPIC": "(EPIC OR HumanMethylationEPIC OR 850K)",
                "450K": "(HumanMethylation450 OR 450K)",
                "WGBS": "(WGBS OR whole genome bisulfite sequencing)",
                "RRBS": "(RRBS OR reduced representation bisulfite)",
            }
            parts.append(platform_terms.get(platform, platform))

        # Sample type — add to PubMed query for targeted literature search
        sample_type = intent.get("sample_type")
        sample_types = intent.get("sample_types", [])
        if sample_type and sample_type in SAMPLE_TYPE_PUBMED_TERMS:
            parts.append(SAMPLE_TYPE_PUBMED_TERMS[sample_type])
        elif sample_types:
            # Combine multiple sample types with OR
            pubmed_terms = []
            seen = set()
            for st in sample_types:
                if st in SAMPLE_TYPE_PUBMED_TERMS and st not in seen:
                    pubmed_terms.append(SAMPLE_TYPE_PUBMED_TERMS[st])
                    seen.add(st)
            # Also add related sample types
            for st in sample_types:
                for related in SAMPLE_TYPE_RELATED.get(st, set()):
                    if related in SAMPLE_TYPE_PUBMED_TERMS and related not in seen:
                        pubmed_terms.append(SAMPLE_TYPE_PUBMED_TERMS[related])
                        seen.add(related)
            if pubmed_terms:
                if len(pubmed_terms) == 1:
                    parts.append(pubmed_terms[0])
                else:
                    parts.append("(" + " OR ".join(pubmed_terms) + ")")

        parts.append("DNA methylation[MeSH Terms]")

        return " AND ".join(parts) if parts else "DNA methylation"

    def _collect_accessions(
        self, papers: List[Dict[str, Any]], sample_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect all unique accessions from enriched papers.
        Optionally filter by sample type keywords in paper title/abstract.

        Returns list of dicts with accession, source, pmid, year, title.
        """
        seen = set()
        result = []

        # Build sample type filter keywords if specified
        sample_type_keywords = {}
        if sample_types:
            # Keywords that indicate the paper uses the wanted sample types
            wanted_keywords = {
                "cfdna": ["cfdna", "cell-free dna", "circulating dna", "ctdna",
                           "游离dna", "循环dna"],
                "plasma": ["plasma", "blood plasma", "血浆"],
                "serum": ["serum", "blood serum", "血清"],
                "wbc": ["wbc", "leukocyte", "buffy coat", "pbmc",
                         "白细胞", "血细胞"],
                "whole_blood": ["whole blood", "全血"],
                "tumor": ["tumor", "tumour", "cancer tissue", "primary tumor",
                           "肿瘤组织", "癌组织"],
                "adjacent": ["adjacent normal", "paratumor", "peritumoral",
                             "癌旁"],
                "normal": ["normal tissue", "healthy tissue", "healthy control",
                            "正常组织"],
                "non_cancer": ["non-cancer", "benign", "control tissue",
                                "非癌", "良性"],
            }
            # Expand with related types
            expanded = set(sample_types)
            for st in sample_types:
                expanded.update(SAMPLE_TYPE_RELATED.get(st, set()))
            for st in expanded:
                if st in wanted_keywords:
                    sample_type_keywords[st] = wanted_keywords[st]

        for paper in papers:
            accessions = paper.get("accessions", {})
            pmid = paper.get("pmid")
            year = paper.get("year")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            paper_text = (title + " " + abstract).lower()

            # Check if paper mentions wanted sample types
            paper_matches_sample_type = True  # Default: keep
            if sample_type_keywords:
                paper_matches_sample_type = False
                for st_type, keywords in sample_type_keywords.items():
                    for kw in keywords:
                        if kw in paper_text:
                            paper_matches_sample_type = True
                            break
                    if paper_matches_sample_type:
                        break

            if not paper_matches_sample_type:
                logger.debug(
                    f"Skipping paper PMID {pmid}: no matching sample type keywords "
                    f"in title/abstract"
                )
                continue

            for acc in accessions.get("geo", []):
                if acc not in seen and acc.startswith("GSE"):
                    seen.add(acc)
                    result.append({
                        "accession": acc,
                        "source": "GEO",
                        "pmid": pmid,
                        "year": year,
                        "title": f"From paper: {title[:100]}",
                        "needs_review": False,
                    })

            for acc in accessions.get("tcga", []):
                if acc not in seen:
                    seen.add(acc)
                    result.append({
                        "accession": acc,
                        "source": "TCGA",
                        "pmid": pmid,
                        "year": year,
                        "title": f"From paper: {title[:100]}",
                        "needs_review": False,
                    })

        return result

    def _process_supplementary_links_v2(
        self,
        papers: List[Dict[str, Any]],
        existing_accessions: set,
    ) -> Dict[str, List]:
        """
        For papers with PMC IDs, find and download direct data links from
        supplementary materials. Uses three-layer LLM extraction pipeline
        for PDF files when regex finds no accessions.
        """
        downloaded = []
        failed = []
        pending_review_list = []
        errors = []
        download_tasks = []

        for paper in papers:
            pmc_id = paper.get("pmc_id")
            pmid = paper.get("pmid")
            doi = paper.get("doi", "")
            if not pmc_id:
                continue

            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
            try:
                links = self.lit_client.parse_supplementary_links(pmc_url)
            except Exception as e:
                logger.debug(f"Supplementary parsing failed for PMC {pmc_id}: {e}")
                continue

            for url in links:
                # Check if this is a PDF — use LLM pipeline
                if url.lower().endswith(".pdf") and self.llm_extraction_enabled:
                    try:
                        pdf_result = self.pdf_parser.parse_pdf_with_llm(
                            pdf_url=url,
                            doi=doi,
                        )

                        # Register high-confidence accessions for auto-download
                        for acc in pdf_result["accessions"]["high_confidence"]:
                            if acc in existing_accessions:
                                continue
                            self.registry.upsert_dataset(
                                accession=acc,
                                source="GEO",
                                discovered_by="agent2_llm",
                                paper_pmid=pmid,
                                paper_doi=doi,
                                download_status="pending",
                                needs_review=False,
                                llm_evidence=f"LLM high-confidence from {url}",
                            )
                            self.registry.log_event(
                                acc, "start",
                                f"LLM high-confidence extraction from PDF (DOI: {doi})"
                            )
                            # Queue for download
                            prefix = acc[:-3] + "nnn" if acc.startswith("GSE") else acc
                            fallback_url = (
                                f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{acc}/"
                                f"matrix/{acc}_series_matrix.txt.gz"
                            )
                            download_tasks.append({
                                "accession": acc,
                                "url": fallback_url,
                                "filename": f"{acc}_series_matrix.txt.gz",
                                "subdir": acc,
                            })

                        # Register medium-confidence accessions as pending_review
                        for acc in pdf_result["accessions"]["pending_review"]:
                            if acc in existing_accessions:
                                continue
                            self.registry.upsert_dataset(
                                accession=acc,
                                source="GEO",
                                discovered_by="agent2_llm",
                                paper_pmid=pmid,
                                paper_doi=doi,
                                download_status=Registry.STATUS_PENDING_REVIEW,
                                needs_review=True,
                                llm_evidence=f"LLM medium-confidence from {url}",
                            )
                            self.registry.log_event(
                                acc, "pending_review",
                                f"LLM medium-confidence extraction from PDF (DOI: {doi})"
                            )
                            pending_review_list.append(acc)
                            logger.info(
                                "Registered %s as pending_review (LLM medium confidence, DOI: %s)",
                                acc, doi,
                            )

                    except Exception as exc:
                        logger.warning(
                            "LLM PDF extraction failed for %s: %s", url[:60], exc
                        )
                        errors.append(f"LLM PDF {url[:60]}: {exc}")

                else:
                    # Non-PDF supplementary file: direct download
                    pseudo_acc = Registry.url_to_accession(url)
                    if pseudo_acc in existing_accessions:
                        continue

                    self.registry.upsert_dataset(
                        accession=pseudo_acc,
                        source="LITERATURE",
                        discovered_by="agent2",
                        paper_pmid=pmid,
                        download_status="pending",
                        title=f"Supplementary from PMID {pmid}",
                    )
                    download_tasks.append({
                        "accession": pseudo_acc,
                        "url": url,
                        "filename": None,
                        "subdir": f"supp_{pmid}",
                    })

        if download_tasks:
            results = self.downloader.download_many_sync(download_tasks)
            for r in results:
                acc = r["accession"]
                if r["status"] == "done":
                    self.registry.update_status(
                        acc, "done",
                        local_path=r["local_path"],
                        file_size_bytes=r.get("file_size_bytes"),
                    )
                    downloaded.append(acc)
                else:
                    self.registry.update_status(acc, "failed")
                    failed.append(acc)
                    if r.get("error"):
                        errors.append(f"Supplementary {acc}: {r['error']}")

        return {
            "downloaded": downloaded,
            "failed": failed,
            "pending_review": pending_review_list,
            "errors": errors,
        }

    def _generate_summary_message(
        self,
        papers: List,
        candidates: List,
        downloaded: List,
        failed: List,
        skipped: List,
        pending_review: List,
    ) -> str:
        papers_with_acc = sum(1 for p in papers if p.get("has_accessions"))
        return (
            f"LiteratureAgent completed.\n"
            f"  Papers searched: {len(papers)} "
            f"(PubMed/PMC: {sum(1 for p in papers if p.get('source') == 'pubmed')}, "
            f"bioRxiv: {sum(1 for p in papers if p.get('source') == 'biorxiv')})\n"
            f"  Papers with accessions: {papers_with_acc}\n"
            f"  New accessions from literature: {len(candidates)}\n"
            f"  Already covered by DatabaseAgent (skipped): {len(skipped)}\n"
            f"  Successfully downloaded: {len(downloaded)}\n"
            f"  Pending human review (LLM medium confidence): {len(pending_review)}\n"
            f"  Failed: {len(failed)}"
        )
