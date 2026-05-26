"""
TCGA / GDC (Genomic Data Commons) tools for MethyAgent.

Uses the GDC REST API to:
  - Search for DNA methylation files by cancer type, platform, year
  - Retrieve file metadata (UUID, size, md5sum)
  - Download public-access Level 3 methylation data (beta values)

Public data (Level 3 processed) requires no authentication.
Controlled data (Level 1/2 raw) requires a GDC token — not implemented here.

GDC API docs: https://docs.gdc.cancer.gov/API/Users_Guide/
"""
import time
from typing import Any, Dict, List, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

GDC_API_BASE = "https://api.gdc.cancer.gov"

# GDC experimental strategies for methylation
METHYLATION_STRATEGIES = ["Methylation Array"]

# GDC data types for methylation
METHYLATION_DATA_TYPES = [
    "Methylation Beta Value",
    "Bisulfite Sequence Alignment",
]

# Platform name mapping
PLATFORM_TO_GDC = {
    "450K": "Illumina Human Methylation 450",
    "EPIC": "Illumina Methylation EPIC",
    "WGBS": "Illumina",  # WGBS uses sequencing platforms
}

# TCGA project → cancer type display name
TCGA_PROJECTS = {
    "TCGA-BRCA": "Breast Invasive Carcinoma",
    "TCGA-LUAD": "Lung Adenocarcinoma",
    "TCGA-LUSC": "Lung Squamous Cell Carcinoma",
    "TCGA-COAD": "Colon Adenocarcinoma",
    "TCGA-LIHC": "Liver Hepatocellular Carcinoma",
    "TCGA-STAD": "Stomach Adenocarcinoma",
    "TCGA-PRAD": "Prostate Adenocarcinoma",
    "TCGA-OV": "Ovarian Serous Cystadenocarcinoma",
    "TCGA-CESC": "Cervical Squamous Cell Carcinoma",
    "TCGA-PAAD": "Pancreatic Adenocarcinoma",
    "TCGA-BLCA": "Bladder Urothelial Carcinoma",
    "TCGA-KIRC": "Kidney Renal Clear Cell Carcinoma",
    "TCGA-THCA": "Thyroid Carcinoma",
    "TCGA-SKCM": "Skin Cutaneous Melanoma",
    "TCGA-GBM": "Glioblastoma Multiforme",
    "TCGA-LAML": "Acute Myeloid Leukemia",
    "TCGA-HNSC": "Head and Neck Squamous Cell Carcinoma",
    "TCGA-UCEC": "Uterine Corpus Endometrial Carcinoma",
    "TCGA-KIRP": "Kidney Renal Papillary Cell Carcinoma",
    "TCGA-LGG": "Brain Lower Grade Glioma",
}


class GDCClient:
    """
    Client for the GDC REST API (public data only).

    Args:
        api_base: GDC API base URL.
        token: Optional GDC token for controlled-access data.
    """

    def __init__(
        self,
        api_base: str = GDC_API_BASE,
        token: Optional[str] = None,
    ):
        self.api_base = api_base
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "MethyAgent/1.0",
        })
        if token:
            self.session.headers["X-Auth-Token"] = token

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        """POST to GDC API with retry logic."""
        url = f"{self.api_base}/{endpoint}"
        for attempt in range(3):
            try:
                resp = self.session.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"GDC rate limit hit, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"GDC API failed after 3 attempts: {url}")

    def _get(self, endpoint: str, params: Dict = None) -> Dict:
        """GET from GDC API."""
        url = f"{self.api_base}/{endpoint}"
        resp = self.session.get(url, params=params or {}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------ #
    #  File search                                                         #
    # ------------------------------------------------------------------ #

    def search_methylation_files(
        self,
        cancer_type_code: Optional[str] = None,
        platform: Optional[str] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        max_results: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Search GDC for methylation files matching the given criteria.

        Args:
            cancer_type_code: TCGA cancer code, e.g. 'BRCA' (without 'TCGA-' prefix).
            platform: Canonical platform name ('450K', 'EPIC', 'WGBS').
            year_start: Filter by file creation year (start).
            year_end: Filter by file creation year (end).
            max_results: Maximum number of files to return.

        Returns:
            List of file metadata dicts.
        """
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "files.experimental_strategy",
                        "value": METHYLATION_STRATEGIES,
                    },
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_type",
                        "value": METHYLATION_DATA_TYPES,
                    },
                },
                {
                    "op": "=",
                    "content": {
                        "field": "files.access",
                        "value": "open",
                    },
                },
            ],
        }

        # Add cancer type filter
        if cancer_type_code:
            project_id = (
                cancer_type_code
                if cancer_type_code.startswith("TCGA-")
                else f"TCGA-{cancer_type_code}"
            )
            filters["content"].append({
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project_id,
                },
            })

        # Add platform filter
        if platform and platform in PLATFORM_TO_GDC:
            filters["content"].append({
                "op": "=",
                "content": {
                    "field": "files.platform",
                    "value": PLATFORM_TO_GDC[platform],
                },
            })

        payload = {
            "filters": filters,
            "fields": (
                "file_id,file_name,file_size,md5sum,data_type,"
                "experimental_strategy,platform,cases.project.project_id,"
                "cases.case_id,created_datetime,updated_datetime"
            ),
            "format": "JSON",
            "size": max_results,
        }

        data = self._post("files", payload)
        hits = data.get("data", {}).get("hits", [])
        logger.info(
            f"GDC search (cancer={cancer_type_code}, platform={platform}) → {len(hits)} files"
        )

        # Apply year filter post-hoc (GDC doesn't support year-only filter)
        if year_start or year_end:
            hits = _filter_by_year(hits, year_start, year_end)

        return hits

    def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """
        Get summary info for a TCGA project (sample counts, data types).

        Args:
            project_id: Full project ID, e.g. 'TCGA-BRCA'.
        """
        params = {
            "filters": f'{{"op":"=","content":{{"field":"project_id","value":"{project_id}"}}}}',
            "fields": "project_id,name,primary_site,disease_type,summary",
        }
        data = self._get("projects", params)
        hits = data.get("data", {}).get("hits", [])
        return hits[0] if hits else {}

    def get_file_download_url(self, file_id: str) -> str:
        """Return the GDC download URL for a given file UUID."""
        return f"{self.api_base}/data/{file_id}"

    def get_bulk_download_manifest(self, file_ids: List[str]) -> str:
        """
        Generate a GDC download manifest for multiple files.
        The manifest can be used with the gdc-client tool for bulk downloads.

        Returns:
            Manifest content as a string (TSV format).
        """
        payload = {"ids": file_ids}
        resp = self.session.post(
            f"{self.api_base}/manifest",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.text

    # ------------------------------------------------------------------ #
    #  Metadata helpers                                                    #
    # ------------------------------------------------------------------ #

    def files_to_dataset_records(
        self, files: List[Dict], cancer_type_code: str
    ) -> List[Dict[str, Any]]:
        """
        Convert GDC file hits to registry-compatible dataset records.

        Groups files by project (one record per project/platform combination).
        """
        from collections import defaultdict
        import re as _re

        groups = defaultdict(list)
        for f in files:
            project = _extract_project(f)
            platform = f.get("platform", "unknown")
            key = f"{project}|{platform}"
            groups[key].append(f)

        records = []
        for key, group_files in groups.items():
            project, platform = key.split("|", 1)
            canonical_platform = _gdc_platform_to_canonical(platform)
            year = _extract_year_from_files(group_files)

            records.append({
                "accession": project,
                "source": "TCGA",
                "data_type": "array" if "Methylation" in platform else "sequencing",
                "cancer_type": cancer_type_code,
                "platform": canonical_platform,
                "sample_count": len(group_files),
                "year": year,
                "title": TCGA_PROJECTS.get(project, project),
                "file_ids": [f["file_id"] for f in group_files],
                "total_size_bytes": sum(f.get("file_size", 0) for f in group_files),
            })

        return records


# ------------------------------------------------------------------ #
#  Helper functions                                                    #
# ------------------------------------------------------------------ #

def _extract_project(file_hit: Dict) -> str:
    """Extract project ID from a GDC file hit."""
    cases = file_hit.get("cases", [])
    if cases:
        return cases[0].get("project", {}).get("project_id", "TCGA-UNKNOWN")
    return "TCGA-UNKNOWN"


def _gdc_platform_to_canonical(platform: str) -> str:
    """Map GDC platform string to canonical platform name."""
    p = platform.lower()
    if "epic" in p or "850" in p:
        return "EPIC"
    if "450" in p:
        return "450K"
    if "27k" in p or "27000" in p:
        return "27K"
    if "bisulfite" in p or "wgbs" in p:
        return "WGBS"
    return platform


def _filter_by_year(
    files: List[Dict],
    year_start: Optional[int],
    year_end: Optional[int],
) -> List[Dict]:
    """Filter GDC file hits by creation year."""
    import re

    filtered = []
    for f in files:
        date_str = f.get("created_datetime", "") or f.get("updated_datetime", "")
        match = re.search(r"\b(20\d{2})\b", date_str)
        if match:
            year = int(match.group(1))
            if year_start and year < year_start:
                continue
            if year_end and year > year_end:
                continue
        filtered.append(f)
    return filtered


def _extract_year_from_files(files: List[Dict]) -> Optional[int]:
    """Extract the most common year from a list of GDC file hits."""
    import re
    from collections import Counter

    years = []
    for f in files:
        date_str = f.get("created_datetime", "") or ""
        match = re.search(r"\b(20\d{2})\b", date_str)
        if match:
            years.append(int(match.group(1)))

    if years:
        return Counter(years).most_common(1)[0][0]
    return None
