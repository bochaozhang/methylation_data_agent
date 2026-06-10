"""
Literature search tools for MethyAgent (Agent 2).

Searches PubMed, PMC full text, and bioRxiv for methylation papers,
then extracts GEO/TCGA accession numbers from abstracts, full text,
and supplementary materials.

Includes PDFSupplementaryParser with three-layer LLM-assisted extraction:
  Layer 1: Regex (zero cost)
  Layer 2: LLM extraction with confidence levels + DOI cache
  Layer 3: GEO API verification (hallucination filter)
"""
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from tools.parser_tools import extract_accessions
from utils.logger import get_logger

logger = get_logger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BIORXIV_API = "https://api.biorxiv.org/details/biorxiv"
PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles"


class LiteratureClient:
    """
    Client for searching PubMed, PMC, and bioRxiv for methylation papers
    and extracting dataset accession numbers.

    Args:
        ncbi_api_key: Optional NCBI API key for higher rate limits.
    """

    def __init__(self, ncbi_api_key: Optional[str] = None, geo_email: Optional[str] = None):
        self.ncbi_api_key = ncbi_api_key
        self.geo_email = geo_email
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MethyAgent/1.0 (methylation literature miner)"
        })
        self._rate_delay = 0.11 if ncbi_api_key else 0.34

    def _ncbi_get(self, endpoint: str, params: Dict) -> requests.Response:
        """
        Rate-limited GET to NCBI E-utilities.

        Detects NCBI abuse redirect (302 → misuse.ncbi.nlm.nih.gov) and
        automatically retries without the API key after a short delay.
        """
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        # NCBI requires email & tool for E-utilities access;
        # missing these is a common cause of IP bans (302 → misuse.ncbi.nlm.nih.gov)
        if self.geo_email:
            params["email"] = self.geo_email
        params.setdefault("tool", "MethyAgent")
        url = f"{EUTILS_BASE}/{endpoint}"
        time.sleep(self._rate_delay)
        resp = self.session.get(url, params=params, timeout=30, allow_redirects=True)

        # Detect NCBI abuse redirect — key may be flagged or request rate too high
        if "misuse.ncbi.nlm.nih.gov" in resp.url:
            logger.warning(
                f"NCBI abuse redirect detected for {endpoint}. "
                "Retrying without API key after 5s delay."
            )
            time.sleep(5)
            params_no_key = {k: v for k, v in params.items() if k != "api_key"}
            resp = self.session.get(url, params=params_no_key, timeout=30)

        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------ #
    #  PubMed search                                                       #
    # ------------------------------------------------------------------ #

    def search_pubmed(
        self,
        query: str,
        max_results: int = 50,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search PubMed and return paper metadata with accessions.

        Args:
            query: PubMed search string (supports MeSH terms).
            max_results: Maximum number of papers to retrieve.
            year_start: Filter by publication year (start).
            year_end: Filter by publication year (end).

        Returns:
            List of paper dicts with pmid, title, abstract, accessions.
        """
        # Add date filter to query
        if year_start and year_end:
            query += f' AND ("{year_start}/01/01"[PDAT] : "{year_end}/12/31"[PDAT])'
        elif year_start:
            query += f' AND ("{year_start}/01/01"[PDAT] : "3000/12/31"[PDAT])'

        # Step 1: esearch to get PMIDs
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        resp = self._ncbi_get("esearch.fcgi", params)
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        logger.info(f"PubMed search '{query[:60]}...' → {len(pmids)} papers")

        if not pmids:
            return []

        # Step 2: efetch to get abstracts and metadata
        return self._fetch_pubmed_records(pmids)

    def _fetch_pubmed_records(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch full PubMed records for a list of PMIDs."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids[:200]),
            "retmode": "xml",
            "rettype": "abstract",
        }
        resp = self._ncbi_get("efetch.fcgi", params)

        papers = []
        try:
            root = ET.fromstring(resp.content)
            for article in root.findall(".//PubmedArticle"):
                paper = _parse_pubmed_article(article)
                if paper:
                    papers.append(paper)
        except ET.ParseError as e:
            logger.error(f"XML parse error for PubMed records: {e}")

        return papers

    # ------------------------------------------------------------------ #
    #  PMC full text                                                       #
    # ------------------------------------------------------------------ #

    def get_pmc_fulltext(self, pmid: str) -> Optional[str]:
        """
        Retrieve PMC full text for a paper (if open access).

        Returns the full text as plain string, or None if not available.
        """
        # First check if PMC ID exists for this PMID
        params = {
            "db": "pmc",
            "linkname": "pubmed_pmc",
            "id": pmid,
            "retmode": "json",
        }
        resp = self._ncbi_get("elink.fcgi", params)
        data = resp.json()

        pmc_ids = []
        for linkset in data.get("linksets", []):
            for linksetdb in linkset.get("linksetdbs", []):
                if linksetdb.get("linkname") == "pubmed_pmc":
                    pmc_ids = linksetdb.get("links", [])
                    break

        if not pmc_ids:
            return None

        pmc_id = pmc_ids[0]

        # Fetch full text XML from PMC
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml",
        }
        try:
            resp = self._ncbi_get("efetch.fcgi", params)
            return _extract_text_from_pmc_xml(resp.content)
        except Exception as e:
            logger.debug(f"Could not fetch PMC full text for PMID {pmid}: {e}")
            return None

    def get_pmc_data_availability(self, pmid: str) -> Optional[str]:
        """
        Extract only the Data Availability / Methods section from PMC full text.
        More efficient than fetching the entire paper.
        """
        full_text = self.get_pmc_fulltext(pmid)
        if not full_text:
            return None

        # Extract relevant sections
        sections = []
        for section_name in ["data availability", "data access", "methods", "materials and methods"]:
            pattern = re.compile(
                rf"(?i){re.escape(section_name)}[:\s]{{0,5}}(.{{0,3000}}?)(?=\n[A-Z]{{3,}}|\Z)",
                re.DOTALL,
            )
            match = pattern.search(full_text)
            if match:
                sections.append(match.group(0))

        return "\n\n".join(sections) if sections else full_text[:2000]

    # ------------------------------------------------------------------ #
    #  bioRxiv search                                                      #
    # ------------------------------------------------------------------ #

    def search_biorxiv(
        self,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        max_results: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Search bioRxiv/medRxiv preprints for methylation papers.

        Uses Europe PMC (https://europepmc.org) which indexes bioRxiv/medRxiv
        preprints with full-text keyword search — far more effective than the
        bioRxiv date-range API which does not support keyword queries.

        Falls back to the bioRxiv date-range API if Europe PMC is unavailable.
        """
        # Build Europe PMC query
        # Strip MeSH syntax for plain-text search
        query_clean = re.sub(r'\[.*?\]', '', query).strip()
        epmc_query = f"{query_clean} SRC:PPR"  # PPR = preprints (bioRxiv/medRxiv)

        if year_start:
            epmc_query += f" FIRST_PDATE:[{year_start}-01-01 TO {year_end or '3000'}-12-31]"

        try:
            resp = self.session.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": epmc_query,
                    "format": "json",
                    "pageSize": min(max_results, 100),
                    "resultType": "core",
                    "sort": "CITED desc",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Europe PMC search failed: {e}. Falling back to bioRxiv API.")
            return self._search_biorxiv_fallback(query, year_start, year_end, max_results)

        papers = []
        for item in data.get("resultList", {}).get("result", []):
            doi = item.get("doi", "")
            abstract = item.get("abstractText", "") or item.get("abstract", "")
            accessions = extract_accessions(abstract)

            papers.append({
                "pmid": item.get("pmid"),
                "doi": doi,
                "title": item.get("title", ""),
                "abstract": abstract,
                "year": item.get("pubYear"),
                "source": "biorxiv",
                "accessions": accessions,
                "has_accessions": any(accessions[k] for k in ("geo", "tcga")),
            })

        logger.info(
            f"bioRxiv/preprint search (EuropePMC) '{query_clean[:50]}' "
            f"→ {len(papers)} preprints (total hits: {data.get('hitCount', '?')})"
        )
        return papers

    def _search_biorxiv_fallback(
        self,
        query: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        max_results: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Fallback bioRxiv search using the date-range API.
        Only practical for narrow date ranges (≤1 year) due to lack of
        keyword search support in the bioRxiv API.
        """
        from datetime import date

        start_date = f"{year_start or 2020}-01-01"
        end_date = f"{year_end or date.today().year}-12-31"

        query_clean = re.sub(r'\[.*?\]', '', query).lower()
        query_terms = [
            t.strip().strip('"')
            for t in re.split(r'\s+AND\s+|\s+OR\s+|\s+', query_clean)
            if len(t.strip().strip('"')) > 3
        ]
        required_terms = [t for t in query_terms if "methylat" in t] or ["methylat"]
        optional_terms = [t for t in query_terms if t not in required_terms]

        papers = []
        cursor = 0

        for _ in range(5):  # max 5 pages in fallback
            url = f"{BIORXIV_API}/{start_date}/{end_date}/{cursor}/json"
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"bioRxiv fallback API error (cursor={cursor}): {e}")
                break

            collection = data.get("collection", [])
            if not collection:
                break

            for item in collection:
                combined = (item.get("title", "") + " " + item.get("abstract", "")).lower()
                if not all(t in combined for t in required_terms):
                    continue
                if optional_terms and not any(t in combined for t in optional_terms):
                    continue

                doi = item.get("doi", "")
                abstract = item.get("abstract", "")
                accessions = extract_accessions(abstract)
                papers.append({
                    "pmid": None,
                    "doi": doi,
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "year": _parse_year_from_date(item.get("date", "")),
                    "source": "biorxiv",
                    "accessions": accessions,
                    "has_accessions": any(accessions[k] for k in ("geo", "tcga")),
                })
                if len(papers) >= max_results:
                    break

            if len(papers) >= max_results:
                break

            total = int(data.get("messages", [{}])[0].get("total") or 0)
            cursor += len(collection)
            if cursor >= total:
                break

        logger.info(f"bioRxiv fallback search → {len(papers)} preprints")
        return papers

    # ------------------------------------------------------------------ #
    #  Supplementary material parsing                                      #
    # ------------------------------------------------------------------ #

    def parse_supplementary_links(self, paper_url: str) -> List[str]:
        """
        Scrape a paper's supplementary material page for data download links.

        Supports: PMC, Nature, Springer, Elsevier, bioRxiv.

        Returns:
            List of URLs pointing to methylation data files.
        """
        try:
            resp = self.session.get(paper_url, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.debug(f"Could not fetch supplementary page {paper_url}: {e}")
            return []

        methylation_file_patterns = re.compile(
            r"\.(txt|csv|tsv|xlsx|gz|zip|tar|bed|cov|idat)(\.gz)?$",
            re.IGNORECASE,
        )
        methylation_keywords = re.compile(
            r"(beta|methylat|idat|bismark|cpg|cov|matrix|supplement)",
            re.IGNORECASE,
        )

        data_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            link_text = a_tag.get_text(strip=True)

            if methylation_file_patterns.search(href) or methylation_keywords.search(href):
                # Make absolute URL
                if href.startswith("http"):
                    data_links.append(href)
                elif href.startswith("/"):
                    from urllib.parse import urlparse
                    parsed = urlparse(paper_url)
                    data_links.append(f"{parsed.scheme}://{parsed.netloc}{href}")

        return list(set(data_links))

    # ------------------------------------------------------------------ #
    #  Combined pipeline                                                   #
    # ------------------------------------------------------------------ #

    def mine_accessions_from_papers(
        self,
        papers: List[Dict[str, Any]],
        fetch_fulltext: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        For each paper, extract all accession numbers from abstract + full text.

        Args:
            papers: List of paper dicts from search_pubmed() or search_biorxiv().
            fetch_fulltext: Whether to fetch PMC full text for papers without accessions.

        Returns:
            Papers enriched with 'accessions' and 'supplementary_links' fields.
        """
        enriched = []
        for paper in papers:
            # Extract from abstract first
            abstract = paper.get("abstract", "")
            accessions = extract_accessions(abstract)

            # If no accessions found and full text available, try PMC
            if fetch_fulltext and not any(accessions[k] for k in ("geo", "tcga")):
                pmid = paper.get("pmid")
                if pmid:
                    data_section = self.get_pmc_data_availability(pmid)
                    if data_section:
                        more_accessions = extract_accessions(data_section)
                        for key in ("geo", "tcga", "arrayexpress"):
                            accessions[key] = list(set(accessions[key] + more_accessions[key]))

            paper["accessions"] = accessions
            paper["has_accessions"] = any(accessions[k] for k in ("geo", "tcga"))
            enriched.append(paper)

        return enriched


# ------------------------------------------------------------------ #
#  PDF Supplementary Parser with LLM-assisted extraction              #
# ------------------------------------------------------------------ #

class PDFSupplementaryParser:
    """
    Three-layer pipeline for extracting accession numbers from PDF supplementary materials.

    Layer 1: Regex extraction (zero cost, existing logic)
    Layer 2: LLM extraction (triggered only when regex returns empty)
              - Bilingual prompt (English + Chinese)
              - Three confidence levels: high / medium / low
              - DOI-keyed SQLite cache
    Layer 3: GEO API verification (filters LLM hallucinations)

    Usage::

        parser = PDFSupplementaryParser(llm=my_llm, geo_client=geo_client)
        result = parser.parse_pdf_with_llm(
            pdf_url="https://...",
            doi="10.1038/...",
        )
        # result["accessions"]["high_confidence"] → auto-download list
        # result["accessions"]["pending_review"]  → needs human review
    """

    def __init__(
        self,
        llm: Any = None,
        geo_client: Any = None,
        cache_db_path: str = "/workspace/methyagent_llm_cache.db",
        model_name: str = "unknown",
        verify_accessions: bool = True,
        min_confidence: str = "medium",
    ) -> None:
        """
        Args:
            llm: LangChain-compatible chat model. If None, LLM layer is disabled.
            geo_client: GEOClient instance for accession verification.
            cache_db_path: Path to SQLite cache for LLM results.
            model_name: Model identifier for logging.
            verify_accessions: Whether to run GEO API verification (Layer 3).
            min_confidence: Minimum confidence level to include in results
                            ('high', 'medium', or 'low').
        """
        self.llm = llm
        self.geo_client = geo_client
        self.verify_accessions = verify_accessions
        self.min_confidence = min_confidence
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "MethyAgent/1.0 (PDF supplementary parser)"
        })

        # Lazy-initialize LLM extractor and section extractor
        self._llm_extractor = None
        self._section_extractor = None

        if llm is not None:
            try:
                from tools.llm_accession_extractor import LLMAccessionExtractor
                self._llm_extractor = LLMAccessionExtractor(
                    llm=llm,
                    cache_db_path=cache_db_path,
                    model_name=model_name,
                )
            except ImportError:
                logger.warning("llm_accession_extractor not available; LLM layer disabled")

        try:
            from tools.pdf_section_extractor import PDFSectionExtractor
            self._section_extractor = PDFSectionExtractor()
        except ImportError:
            logger.warning("pdf_section_extractor not available; using full text")

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def parse_pdf_with_llm(
        self,
        pdf_url: str,
        doi: str = "",
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Three-layer accession extraction pipeline for a PDF URL.

        Parameters
        ----------
        pdf_url : str
            Direct URL to a PDF file.
        doi : str
            Paper DOI used as LLM cache key.
        headers : dict, optional
            Additional HTTP headers (e.g., for journal authentication).

        Returns
        -------
        dict with keys:
            accessions : dict
                high_confidence : list[str]  — verified, auto-download
                pending_review  : list[str]  — needs human confirmation
            data_links      : list[str]      — direct file download URLs
            extraction_method : str          — 'regex' | 'llm' | 'regex+llm'
            llm_used        : bool
            cache_hit       : bool
            pages_parsed    : int
            error           : str or None
        """
        result: Dict[str, Any] = {
            "accessions": {"high_confidence": [], "pending_review": []},
            "data_links": [],
            "extraction_method": "regex",
            "llm_used": False,
            "cache_hit": False,
            "pages_parsed": 0,
            "error": None,
        }

        # --- Step 1: Download and parse PDF ---
        pdf_text, pages, data_links, pages_parsed = self._download_and_parse_pdf(
            pdf_url, headers
        )
        result["data_links"] = data_links
        result["pages_parsed"] = pages_parsed

        if not pdf_text:
            result["error"] = "Could not extract text from PDF"
            return result

        # --- Layer 1: Regex extraction ---
        regex_accessions = extract_accessions(pdf_text)
        geo_accessions = regex_accessions.get("geo", [])
        tcga_accessions = regex_accessions.get("tcga", [])
        all_regex = list(set(geo_accessions + tcga_accessions))

        if all_regex:
            # Regex succeeded — skip LLM
            result["accessions"]["high_confidence"] = all_regex
            result["extraction_method"] = "regex"
            logger.info(
                "PDF regex extraction: %d accessions from %s",
                len(all_regex), pdf_url[:60],
            )
            return result

        # --- Layer 2: LLM extraction (triggered because regex returned empty) ---
        if self._llm_extractor is None:
            logger.debug("LLM extractor not configured; skipping Layer 2")
            return result

        # Extract relevant sections to minimize token usage
        section_text = self._get_section_text(pdf_text, pages)
        if not section_text:
            logger.debug("No relevant sections found in PDF; skipping LLM")
            return result

        logger.info("Regex found 0 accessions; triggering LLM extraction for %s", doi or pdf_url[:60])
        llm_result = self._llm_extractor.extract(
            text=section_text,
            doi=doi,
            pdf_url=pdf_url,
        )
        result["llm_used"] = True
        result["cache_hit"] = llm_result.cache_hit
        result["extraction_method"] = "llm"

        if llm_result.error:
            logger.warning("LLM extraction error: %s", llm_result.error)
            result["error"] = f"LLM error: {llm_result.error}"
            return result

        # Collect accessions by confidence
        high_candidates = llm_result.high_confidence
        medium_candidates = llm_result.medium_confidence

        # --- Layer 3: GEO API verification ---
        if self.verify_accessions and self.geo_client is not None:
            high_candidates, medium_candidates = self._verify_candidates(
                high_candidates, medium_candidates
            )

        result["accessions"]["high_confidence"] = high_candidates
        result["accessions"]["pending_review"] = medium_candidates

        logger.info(
            "LLM extraction complete: high=%d pending=%d cache_hit=%s doi=%s",
            len(high_candidates),
            len(medium_candidates),
            llm_result.cache_hit,
            doi or "N/A",
        )
        return result

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _download_and_parse_pdf(
        self,
        pdf_url: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> tuple:
        """
        Download PDF and extract text using pdfplumber.

        Returns (full_text, pages_list, data_links, pages_parsed).
        """
        full_text = ""
        pages: List[str] = []
        data_links: List[str] = []
        pages_parsed = 0

        try:
            req_headers = dict(self._session.headers)
            if headers:
                req_headers.update(headers)

            resp = self._session.get(pdf_url, headers=req_headers, timeout=30, stream=True)
            resp.raise_for_status()

            # Check content type
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not pdf_url.lower().endswith(".pdf"):
                # Might be HTML — try to extract text directly
                soup = BeautifulSoup(resp.text, "lxml")
                full_text = soup.get_text(separator="\n")
                pages = [full_text]
                pages_parsed = 1
                return full_text, pages, data_links, pages_parsed

            # Parse PDF with pdfplumber
            import io
            try:
                import pdfplumber
            except ImportError:
                logger.warning("pdfplumber not installed; cannot parse PDF")
                return full_text, pages, data_links, pages_parsed

            pdf_bytes = resp.content
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_parsed = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)

                    # Extract hyperlinks from page annotations
                    for annot in (page.annots or []):
                        uri = annot.get("uri", "")
                        if uri and _is_data_link(uri):
                            data_links.append(uri)

            full_text = "\n".join(pages)

        except Exception as exc:
            logger.warning("PDF download/parse failed for %s: %s", pdf_url[:60], exc)

        return full_text, pages, list(set(data_links)), pages_parsed

    def _get_section_text(
        self, full_text: str, pages: List[str]
    ) -> str:
        """Extract relevant sections using PDFSectionExtractor."""
        if self._section_extractor is not None:
            result = self._section_extractor.extract(full_text, pages)
            best = result.best_text
            if best:
                return best

        # Fallback: return first 3000 chars
        return full_text[:3000]

    def _verify_candidates(
        self,
        high_candidates: List[str],
        medium_candidates: List[str],
    ) -> tuple:
        """
        Run GEO API verification on LLM-extracted accessions.

        Verified accessions stay in their tier.
        Unverified accessions are discarded (logged).
        """
        all_candidates = list(set(high_candidates + medium_candidates))
        if not all_candidates:
            return [], []

        try:
            verification = self.geo_client.batch_verify_accessions(all_candidates)
        except Exception as exc:
            logger.warning("Batch verification failed: %s; skipping verification", exc)
            return high_candidates, medium_candidates

        verified_high = [a for a in high_candidates if verification.get(a.upper(), True)]
        verified_medium = [a for a in medium_candidates if verification.get(a.upper(), True)]

        discarded = [
            a for a in all_candidates
            if not verification.get(a.upper(), True)
        ]
        if discarded:
            logger.info(
                "GEO API verification discarded %d hallucinated accessions: %s",
                len(discarded),
                discarded,
            )

        return verified_high, verified_medium


# ------------------------------------------------------------------ #
#  Helper functions                                                    #
# ------------------------------------------------------------------ #

def _is_data_link(url: str) -> bool:
    """Return True if URL looks like a data file link."""
    data_extensions = (
        ".txt", ".csv", ".tsv", ".gz", ".zip", ".tar",
        ".bed", ".cov", ".idat", ".xlsx",
    )
    url_lower = url.lower()
    return any(url_lower.endswith(ext) for ext in data_extensions)


def _parse_pubmed_article(article_elem: ET.Element) -> Optional[Dict[str, Any]]:
    """Parse a PubmedArticle XML element into a dict."""
    try:
        pmid_elem = article_elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else None

        title_elem = article_elem.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()) if title_elem is not None else ""

        # Abstract (may have multiple AbstractText elements)
        abstract_parts = []
        for ab in article_elem.findall(".//AbstractText"):
            label = ab.get("Label", "")
            text = "".join(ab.itertext())
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Year
        year_elem = article_elem.find(".//PubDate/Year")
        year = int(year_elem.text) if year_elem is not None else None

        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # DOI
        doi = None
        for id_elem in article_elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break

        # PMC ID
        pmc_id = None
        for id_elem in article_elem.findall(".//ArticleId"):
            if id_elem.get("IdType") == "pmc":
                pmc_id = id_elem.text
                break

        # Extract accessions from abstract
        accessions = extract_accessions(abstract)

        return {
            "pmid": pmid,
            "doi": doi,
            "pmc_id": pmc_id,
            "title": title,
            "abstract": abstract,
            "year": year,
            "journal": journal,
            "source": "pubmed",
            "accessions": accessions,
            "has_accessions": any(accessions[k] for k in ("geo", "tcga")),
        }
    except Exception as e:
        logger.debug(f"Error parsing PubMed article: {e}")
        return None


def _extract_text_from_pmc_xml(xml_bytes: bytes) -> str:
    """Extract plain text from PMC full text XML."""
    try:
        root = ET.fromstring(xml_bytes)
        texts = []
        for elem in root.iter():
            if elem.text:
                texts.append(elem.text.strip())
            if elem.tail:
                texts.append(elem.tail.strip())
        return " ".join(t for t in texts if t)
    except ET.ParseError:
        # Fallback: strip XML tags with regex
        text = xml_bytes.decode("utf-8", errors="ignore")
        return re.sub(r"<[^>]+>", " ", text)


def _parse_year_from_date(date_str: str) -> Optional[int]:
    """Parse year from a date string like '2024-03-15'."""
    match = re.search(r"\b(20\d{2})\b", date_str)
    return int(match.group(1)) if match else None
