"""
GEO (Gene Expression Omnibus) tools for MethyAgent.

Uses NCBI E-utilities REST API to:
  - Search GEO for methylation datasets
  - Fetch dataset metadata (platform, sample count, data type)
  - Resolve download URLs for SOFT files and supplementary data
  - Verify accession numbers exist in NCBI databases (for LLM hallucination filtering)
"""
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

# NCBI E-utilities base URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# GEO FTP base for supplementary files
GEO_FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

# Methylation-related platform accessions (Illumina arrays)
METHYLATION_PLATFORMS = {
    "GPL13534",  # HumanMethylation450
    "GPL21145",  # MethylationEPIC
    "GPL23976",  # MethylationEPIC v2
    "GPL8490",   # HumanMethylation27
}

# Data type detection keywords
ARRAY_KEYWORDS = ["450k", "epic", "methylation array", "illumina methylation", "hm450", "hm850"]
SEQ_KEYWORDS = ["wgbs", "rrbs", "bisulfite sequencing", "whole genome bisulfite", "reduced representation"]

# Accession prefix → NCBI database mapping for verification
_ACCESSION_DB_MAP: Dict[str, str] = {
    "GSE": "gse",
    "GSM": "gsm",
    "GPL": "gpl",
    "GDS": "gds",
    "SRP": "sra",
    "SRR": "sra",
    "SRX": "sra",
}


class GEOClient:
    """
    Client for NCBI GEO E-utilities API.

    Args:
        api_key: Optional NCBI API key (raises rate limit from 3 to 10 req/s).
        base_url: E-utilities base URL.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = EUTILS_BASE):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MethyAgent/1.0 (methylation data collector)"})
        self._rate_limit_delay = 0.11 if api_key else 0.34  # seconds between requests

    def _get(self, endpoint: str, params: Dict) -> requests.Response:
        """Make a rate-limited GET request to E-utilities."""
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        time.sleep(self._rate_limit_delay)
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        max_results: int = 100,
        db: str = "gds",
    ) -> List[str]:
        """
        Search GEO DataSets (gds) or GEO Series (gse) for a query.

        Args:
            query: NCBI search string.
            max_results: Maximum number of UIDs to return.
            db: NCBI database ('gds' for DataSets, 'gse' for Series via 'gse').

        Returns:
            List of GEO UIDs (integers as strings).
        """
        params = {
            "db": db,
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "usehistory": "y",
        }
        resp = self._get("esearch.fcgi", params)
        data = resp.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        logger.info(f"GEO search '{query[:60]}...' → {len(ids)} results")
        return ids

    def search_gse(self, query: str, max_results: int = 100) -> List[str]:
        """
        Search GEO Series (GSE) directly.
        Returns list of GSE accession strings (e.g. ['GSE124600', ...]).

        NCBI E-utilities uses db='gds' for GEO DataSets/Series searches.
        The db='gse' alias is not reliably supported across all API versions.
        """
        # Primary: use db='gds' which reliably returns GEO Series UIDs
        params = {
            "db": "gds",
            "term": query + " AND GSE[Entry Type]",
            "retmax": max_results,
            "retmode": "json",
        }
        try:
            resp = self._get("esearch.fcgi", params)
            data = resp.json()
            uids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"GEO search (gds) '{query[:60]}' → {len(uids)} UIDs")
        except Exception as e:
            logger.warning(f"GEO search failed: {e}")
            uids = []

        # Fallback: try without Entry Type filter
        if not uids:
            try:
                params2 = {
                    "db": "gds",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json",
                }
                resp2 = self._get("esearch.fcgi", params2)
                uids = resp2.json().get("esearchresult", {}).get("idlist", [])
                logger.info(f"GEO search fallback (gds, no filter) → {len(uids)} UIDs")
            except Exception as e2:
                logger.error(f"GEO search fallback also failed: {e2}")
                return []

        if not uids:
            return []

        # Convert GDS UIDs to GSE accessions via esummary
        return self._uids_to_accessions(uids, db="gds")

    def _uids_to_accessions(self, uids: List[str], db: str = "gds") -> List[str]:
        """Convert numeric UIDs to GSE accession strings."""
        if not uids:
            return []
        params = {
            "db": db,
            "id": ",".join(uids[:200]),  # API limit
            "retmode": "json",
        }
        resp = self._get("esummary.fcgi", params)
        data = resp.json()
        result = data.get("result", {})
        accessions = []
        for uid in uids:
            item = result.get(uid, {})

            # Primary: 'accession' field — may be 'GSE309199' (full) or just '309199'
            acc = item.get("accession", "")
            if acc:
                acc_upper = acc.upper()
                if acc_upper.startswith("GSE"):
                    accessions.append(acc_upper)
                    continue
                # Numeric-only: prepend GSE
                if acc.isdigit():
                    accessions.append(f"GSE{acc}")
                    continue

            # Fallback: 'gse' field — may be '309199' (no prefix) or 'GSE309199'
            gse_val = item.get("gse", "")
            if isinstance(gse_val, str) and gse_val:
                gse_upper = gse_val.upper()
                if gse_upper.startswith("GSE"):
                    accessions.append(gse_upper)
                elif gse_val.isdigit():
                    accessions.append(f"GSE{gse_val}")
            elif isinstance(gse_val, list):
                for g in gse_val:
                    if isinstance(g, str) and g:
                        accessions.append(f"GSE{g}" if g.isdigit() else g.upper())

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for a in accessions:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        logger.info(f"_uids_to_accessions: {len(uids)} UIDs → {len(unique)} GSE accessions")
        return unique

    # ------------------------------------------------------------------ #
    #  Metadata                                                            #
    # ------------------------------------------------------------------ #

    def get_series_metadata(self, accession: str) -> Dict[str, Any]:
        """
        Fetch metadata for a GSE series.

        Returns dict with: title, summary, platform, sample_count,
        data_type, year, supplementary_files.

        Uses db='gds' throughout — the correct NCBI database for GEO Series.
        db='gse' is not a valid E-utilities database name.
        """
        # Step 1: resolve accession → UID using db='gds'
        params = {
            "db": "gds",
            "term": f"{accession}[Accession]",
            "retmode": "json",
        }
        resp = self._get("esearch.fcgi", params)
        uids = resp.json().get("esearchresult", {}).get("idlist", [])

        if not uids:
            logger.warning(f"No UID found for {accession} in db=gds")
            return {"accession": accession, "error": "not_found"}

        uid = uids[0]

        # Step 2: fetch summary using db='gds'
        params = {"db": "gds", "id": uid, "retmode": "json"}
        resp = self._get("esummary.fcgi", params)
        data = resp.json().get("result", {}).get(uid, {})

        # Parse platform info — gds esummary stores platforms in 'gpl' field.
        # The value may be:
        #   - a string like '13534' (numeric, no GPL prefix)  ← most common
        #   - a string like 'GPL13534' (with prefix)
        #   - a list of dicts with 'accession' key
        #   - a list of strings
        platforms = []
        gpl_field = data.get("gpl", "")

        def _normalise_gpl(val: str) -> str:
            """Ensure GPL prefix is present."""
            val = val.strip()
            if not val:
                return ""
            return val if val.upper().startswith("GPL") else f"GPL{val}"

        if isinstance(gpl_field, list):
            for gpl_item in gpl_field:
                if isinstance(gpl_item, dict):
                    raw = gpl_item.get("accession", "")
                    if raw:
                        platforms.append(_normalise_gpl(raw))
                elif isinstance(gpl_item, str) and gpl_item.strip():
                    # Each list item may itself be semicolon-separated
                    for part in gpl_item.split(";"):
                        if part.strip():
                            platforms.append(_normalise_gpl(part))
        elif isinstance(gpl_field, str) and gpl_field.strip():
            # String may be semicolon-separated: "13534;570" or "GPL13534;GPL21145"
            for part in gpl_field.split(";"):
                if part.strip():
                    platforms.append(_normalise_gpl(part))

        # Detect data type from title/summary
        title = data.get("title", "")
        summary = data.get("summary", "")
        combined_text = (title + " " + summary).lower()
        data_type = _detect_data_type(combined_text, platforms)

        # Parse year — gds uses 'pdat' or 'submissiondate'
        pub_date = (
            data.get("pdat", "")
            or data.get("pubdate", "")
            or data.get("submissiondate", "")
        )
        year = _parse_year(pub_date)

        # Sample count — gds uses 'n_samples' or 'samplecount'
        sample_count = int(
            data.get("n_samples", 0)
            or data.get("samplecount", 0)
            or 0
        )

        # Supplementary file URLs
        supp_files = self._get_supplementary_urls(accession)

        return {
            "accession": accession,
            "title": title,
            "summary": summary[:500] if summary else "",
            "platforms": platforms,
            "platform_canonical": _canonical_platform(platforms, combined_text),
            "sample_count": sample_count,
            "data_type": data_type,
            "year": year,
            "supplementary_files": supp_files,
            "source": "GEO",
        }

    def batch_get_series_metadata(
        self,
        accessions: List[str],
        batch_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Fetch metadata for multiple GSE accessions efficiently.

        Instead of N×2 API calls (one esearch + one esummary per accession),
        this method:
          1. Batch-searches all accessions in one esearch call
          2. Batch-fetches all summaries in one esummary call

        Args:
            accessions: List of GSE accession strings.
            batch_size: Max accessions per API batch (default 50).

        Returns:
            List of metadata dicts (same format as get_series_metadata).
        """
        if not accessions:
            return []

        all_meta = []

        for i in range(0, len(accessions), batch_size):
            batch = accessions[i : i + batch_size]

            # Step 1: batch esearch — one OR query for all accessions
            term = " OR ".join(f"{acc}[Accession]" for acc in batch)
            try:
                resp = self._get("esearch.fcgi", {
                    "db": "gds",
                    "term": term,
                    "retmax": len(batch) * 2,  # allow for multi-platform entries
                    "retmode": "json",
                })
                uids = resp.json().get("esearchresult", {}).get("idlist", [])
            except Exception as e:
                logger.warning(f"batch_get_series_metadata esearch failed: {e}")
                # Fall back to individual fetches
                for acc in batch:
                    meta = self.get_series_metadata(acc)
                    all_meta.append(meta)
                continue

            if not uids:
                logger.warning(f"batch_get_series_metadata: no UIDs for batch {batch}")
                continue

            # Step 2: batch esummary — one call for all UIDs
            try:
                resp = self._get("esummary.fcgi", {
                    "db": "gds",
                    "id": ",".join(uids),
                    "retmode": "json",
                })
                result = resp.json().get("result", {})
            except Exception as e:
                logger.warning(f"batch_get_series_metadata esummary failed: {e}")
                for acc in batch:
                    meta = self.get_series_metadata(acc)
                    all_meta.append(meta)
                continue

            # Build accession → uid map from results
            acc_to_data: Dict[str, Any] = {}
            for uid in uids:
                item = result.get(uid, {})
                acc_raw = item.get("accession", "")
                if not acc_raw:
                    continue
                acc_norm = acc_raw.upper()
                if not acc_norm.startswith("GSE"):
                    acc_norm = f"GSE{acc_norm}" if acc_norm.isdigit() else acc_norm
                acc_to_data[acc_norm] = (uid, item)

            # Parse metadata for each requested accession
            for acc in batch:
                acc_upper = acc.upper()
                if acc_upper not in acc_to_data:
                    logger.warning(f"batch_get_series_metadata: {acc} not in esummary result")
                    all_meta.append({"accession": acc, "error": "not_found"})
                    continue

                uid, data = acc_to_data[acc_upper]

                def _normalise_gpl(val: str) -> str:
                    val = val.strip()
                    if not val:
                        return ""
                    return val if val.upper().startswith("GPL") else f"GPL{val}"

                platforms = []
                gpl_field = data.get("gpl", "")
                if isinstance(gpl_field, list):
                    for gpl_item in gpl_field:
                        if isinstance(gpl_item, dict):
                            raw = gpl_item.get("accession", "")
                            if raw:
                                platforms.append(_normalise_gpl(raw))
                        elif isinstance(gpl_item, str) and gpl_item.strip():
                            for part in gpl_item.split(";"):
                                if part.strip():
                                    platforms.append(_normalise_gpl(part))
                elif isinstance(gpl_field, str) and gpl_field.strip():
                    for part in gpl_field.split(";"):
                        if part.strip():
                            platforms.append(_normalise_gpl(part))

                title = data.get("title", "")
                summary = data.get("summary", "")
                combined_text = (title + " " + summary).lower()
                data_type = _detect_data_type(combined_text, platforms)

                pub_date = (
                    data.get("pdat", "")
                    or data.get("pubdate", "")
                    or data.get("submissiondate", "")
                )
                year = _parse_year(pub_date)

                sample_count = int(
                    data.get("n_samples", 0)
                    or data.get("samplecount", 0)
                    or 0
                )

                supp_files = self._get_supplementary_urls(acc_upper)

                all_meta.append({
                    "accession": acc_upper,
                    "title": title,
                    "summary": summary[:500] if summary else "",
                    "platforms": platforms,
                    "platform_canonical": _canonical_platform(platforms, combined_text),
                    "sample_count": sample_count,
                    "data_type": data_type,
                    "year": year,
                    "supplementary_files": supp_files,
                    "source": "GEO",
                })

        logger.info(f"batch_get_series_metadata: fetched {len(all_meta)}/{len(accessions)} records")
        return all_meta

    def get_accession_metadata(self, accession: str) -> Dict[str, Any]:
        """Alias for get_series_metadata — handles GSE accessions."""
        return self.get_series_metadata(accession)

    # ------------------------------------------------------------------ #
    #  Supplementary file discovery                                        #
    # ------------------------------------------------------------------ #

    def _get_supplementary_urls(self, accession: str) -> List[str]:
        """
        Build FTP/HTTPS URLs for supplementary files of a GSE series.

        GEO FTP structure:
          /geo/series/GSE124nnn/GSE124600/suppl/
        """
        # Derive the FTP subdirectory (e.g. GSE124600 → GSE124nnn)
        prefix = accession[:-3] + "nnn"  # GSE124600 → GSE124nnn
        base_url = f"{GEO_FTP_BASE}/{prefix}/{accession}/suppl/"

        try:
            resp = self.session.get(base_url, timeout=15)
            if resp.status_code == 200:
                # Parse directory listing for methylation-related files
                files = _parse_ftp_listing(resp.text, base_url)
                return files
        except Exception as e:
            logger.debug(f"Could not list supplementary files for {accession}: {e}")

        # Return the base URL as fallback
        return [base_url]

    # ------------------------------------------------------------------ #
    #  Methylation-specific filtering                                      #
    # ------------------------------------------------------------------ #

    def filter_methylation_datasets(
        self,
        accessions: List[str],
        platform_filter: Optional[str] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch metadata for a list of accessions and filter to methylation datasets.

        Args:
            accessions: List of GSE accession strings.
            platform_filter: Optional platform filter ('450K', 'EPIC', 'WGBS', 'RRBS').
            year_start: Optional start year filter.
            year_end: Optional end year filter.

        Returns:
            List of metadata dicts for datasets passing the filter.
        """
        # Use batch fetch to reduce API calls (N×2 → 2 calls per batch of 50)
        try:
            all_meta = self.batch_get_series_metadata(accessions)
        except Exception as e:
            logger.warning(f"batch_get_series_metadata failed, falling back to individual: {e}")
            all_meta = []
            for acc in accessions:
                try:
                    all_meta.append(self.get_series_metadata(acc))
                except Exception as e2:
                    logger.warning(f"Error fetching metadata for {acc}: {e2}")

        results = []
        for meta in all_meta:
            acc = meta.get("accession", "?")
            if meta.get("error"):
                logger.debug(f"Skipping {acc}: metadata error={meta.get('error')!r}")
                continue

            data_type = meta.get("data_type", "unknown")
            platform_canonical = meta.get("platform_canonical")
            logger.debug(
                f"{acc}: data_type={data_type}, platform={platform_canonical}, "
                f"year={meta.get('year')}, title={meta.get('title','')[:50]}"
            )

            # Platform filter — only exclude if platform is known AND mismatches
            if platform_filter:
                canonical = meta.get("platform_canonical")
                if canonical and canonical != platform_filter:
                    logger.debug(f"Skipping {acc}: platform {canonical} != {platform_filter}")
                    continue
                # If canonical is None/unknown, keep the dataset (don't over-filter)

            # Year filter
            year = meta.get("year")
            if year_start and year and year < year_start:
                logger.debug(f"Skipping {acc}: year {year} < {year_start}")
                continue
            if year_end and year and year > year_end:
                logger.debug(f"Skipping {acc}: year {year} > {year_end}")
                continue

            logger.info(f"Accepted {acc}: data_type={data_type}, platform={platform_canonical}")
            results.append(meta)

        logger.info(f"filter_methylation_datasets: {len(results)}/{len(accessions)} datasets passed")
        return results
    # ------------------------------------------------------------------ #
    #  Accession verification (LLM hallucination filter)                  #
    # ------------------------------------------------------------------ #

    def verify_accession(self, accession: str) -> bool:
        """
        Verify that an accession number actually exists in NCBI databases.

        Used as Layer 3 of the LLM extraction pipeline to filter hallucinated
        accessions before they enter the registry.

        Supports: GSE, GSM, GPL, GDS (→ GEO), SRP/SRR/SRX (→ SRA).

        Args:
            accession: Accession string to verify (e.g. "GSE124600").

        Returns:
            True if the accession exists in NCBI, False otherwise.
        """
        accession = accession.strip().upper()
        if not accession:
            return False

        # Determine target NCBI database
        db = _accession_to_db(accession)
        if db is None:
            logger.debug("verify_accession: unknown prefix for %s", accession)
            return False

        try:
            params = {
                "db": db,
                "term": f"{accession}[Accession]",
                "retmode": "json",
                "retmax": 1,
            }
            resp = self._get("esearch.fcgi", params)
            data = resp.json()
            count = int(data.get("esearchresult", {}).get("count", 0))
            exists = count > 0
            logger.debug("verify_accession: %s → db=%s count=%d exists=%s", accession, db, count, exists)
            return exists
        except Exception as exc:
            logger.warning("verify_accession failed for %s: %s", accession, exc)
            # On network error, return True to avoid discarding valid accessions
            return True

    def batch_verify_accessions(
        self,
        accessions: List[str],
        batch_size: int = 20,
    ) -> Dict[str, bool]:
        """
        Verify multiple accessions efficiently using batched NCBI OR queries.

        Groups accessions by database type and issues one esearch per group,
        reducing the number of API calls compared to individual verify_accession().

        Args:
            accessions: List of accession strings to verify.
            batch_size: Maximum accessions per NCBI query (default 20).

        Returns:
            Dict mapping accession → True/False.
        """
        if not accessions:
            return {}

        # Group by NCBI database
        db_groups: Dict[str, List[str]] = {}
        unknown: List[str] = []
        for acc in accessions:
            acc_upper = acc.strip().upper()
            db = _accession_to_db(acc_upper)
            if db:
                db_groups.setdefault(db, []).append(acc_upper)
            else:
                unknown.append(acc_upper)

        results: Dict[str, bool] = {acc.upper(): False for acc in accessions}

        # Mark unknown-prefix accessions as unverifiable (return True to be safe)
        for acc in unknown:
            results[acc] = True
            logger.debug("batch_verify: unknown prefix for %s, skipping", acc)

        # Process each database group in batches
        for db, db_accessions in db_groups.items():
            for i in range(0, len(db_accessions), batch_size):
                batch = db_accessions[i : i + batch_size]
                # Build OR query: GSE124600[Accession] OR GSE200234[Accession] OR ...
                term = " OR ".join(f"{acc}[Accession]" for acc in batch)
                try:
                    params = {
                        "db": db,
                        "term": term,
                        "retmode": "json",
                        "retmax": len(batch),
                        "usehistory": "n",
                    }
                    resp = self._get("esearch.fcgi", params)
                    data = resp.json()
                    found_ids = data.get("esearchresult", {}).get("idlist", [])

                    if not found_ids:
                        # None in this batch exist
                        continue

                    # Fetch summaries to map UIDs back to accession strings
                    sum_params = {
                        "db": db,
                        "id": ",".join(found_ids),
                        "retmode": "json",
                    }
                    sum_resp = self._get("esummary.fcgi", sum_params)
                    sum_data = sum_resp.json().get("result", {})

                    for uid in found_ids:
                        item = sum_data.get(uid, {})
                        # esummary returns 'accession' for gse/gsm/gpl
                        found_acc = (
                            item.get("accession", "")
                            or item.get("experimentaccession", "")  # SRA
                        ).upper()
                        if found_acc and found_acc in results:
                            results[found_acc] = True

                except Exception as exc:
                    logger.warning(
                        "batch_verify failed for db=%s batch=%s: %s", db, batch, exc
                    )
                    # On error, mark batch as verified to avoid false negatives
                    for acc in batch:
                        results[acc] = True

        verified_count = sum(1 for v in results.values() if v)
        logger.info(
            "batch_verify_accessions: %d/%d verified", verified_count, len(accessions)
        )
        return results


# ------------------------------------------------------------------ #
#  Helper functions                                                    #
# ------------------------------------------------------------------ #

def _accession_to_db(accession: str) -> Optional[str]:
    """Map accession prefix to NCBI database name."""
    acc_upper = accession.upper()
    for prefix, db in _ACCESSION_DB_MAP.items():
        if acc_upper.startswith(prefix):
            return db
    return None


def _detect_data_type(text: str, platforms: List[str]) -> str:
    """Detect whether dataset is array-based or sequencing-based."""
    # Check platform accessions first
    for p in platforms:
        if p in METHYLATION_PLATFORMS:
            return "array"

    # Check text keywords
    for kw in ARRAY_KEYWORDS:
        if kw in text:
            return "array"
    for kw in SEQ_KEYWORDS:
        if kw in text:
            return "sequencing"

    if "methylat" in text:
        return "array"  # Default assumption for methylation without seq keywords

    return "unknown"


def _canonical_platform(platforms: List[str], text: str) -> Optional[str]:
    """Map platform accessions or text to canonical platform name."""
    for p in platforms:
        if p in ("GPL21145", "GPL23976"):
            return "EPIC"
        if p == "GPL13534":
            return "450K"
        if p == "GPL8490":
            return "27K"

    text_lower = text.lower()
    if "epic" in text_lower or "850k" in text_lower or "hm850" in text_lower:
        return "EPIC"
    if "450k" in text_lower or "hm450" in text_lower or "humanmethylation450" in text_lower:
        return "450K"
    if "wgbs" in text_lower or "whole genome bisulfite" in text_lower:
        return "WGBS"
    if "rrbs" in text_lower or "reduced representation" in text_lower:
        return "RRBS"

    return None


def _parse_year(date_str: str) -> Optional[int]:
    """Extract year from a date string like '2024 Jan 15' or '2024-01-15'."""
    if not date_str:
        return None
    match = re.search(r"\b(20\d{2})\b", date_str)
    return int(match.group(1)) if match else None


def _parse_ftp_listing(html: str, base_url: str) -> List[str]:
    """Parse an FTP/HTTP directory listing and return methylation file URLs."""
    methylation_extensions = (
        ".txt.gz", ".csv.gz", ".tsv.gz", ".idat.gz",
        ".txt", ".csv", ".tsv", ".bed.gz", ".cov.gz",
    )
    methylation_keywords = (
        "beta", "mvalue", "methylation", "idat", "bismark",
        "cpg", "cov", "matrix",
    )

    urls = []
    # Match href links in directory listing
    for match in re.finditer(r'href="([^"]+)"', html):
        filename = match.group(1)
        if filename.startswith("?") or filename.startswith("/"):
            continue
        fname_lower = filename.lower()
        if any(fname_lower.endswith(ext) for ext in methylation_extensions):
            if any(kw in fname_lower for kw in methylation_keywords):
                urls.append(base_url + filename)

    return urls
