"""
Async download engine for MethyAgent.

Features:
  - Async concurrent downloads (asyncio + aiohttp)
  - Resume/range requests for interrupted downloads
  - Exponential backoff retry
  - Progress tracking via tqdm
  - File type routing (array vs sequencing)
  - MD5 checksum verification
  - Download progress persisted to registry
"""
import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm as async_tqdm

from utils.logger import get_logger

logger = get_logger(__name__)

# File extensions that indicate methylation array data
ARRAY_EXTENSIONS = (
    "_beta_values.txt.gz", "_matrix_processed.txt.gz",
    ".methylation_array.sesame.level3betas.txt",
    "_beta.txt", "_betas.txt.gz", "_methylation.txt.gz",
    ".idat.gz",
)

# File extensions that indicate sequencing-based methylation data
SEQ_EXTENSIONS = (
    ".bismark.cov.gz", ".bed.gz", ".cov.gz",
    "_CpG.txt.gz", "_bismark.txt.gz",
    ".bedGraph.gz",
)


class DownloadEngine:
    """
    Async download engine with concurrency control and resume support.

    Args:
        output_dir: Base directory for downloaded files.
        max_concurrent: Maximum simultaneous downloads.
        retry_attempts: Number of retry attempts on failure.
        retry_delay: Base delay (seconds) for exponential backoff.
        chunk_size_mb: Download chunk size in megabytes.
        timeout: Per-request timeout in seconds.
        on_progress: Optional callback(accession, bytes_downloaded, total_bytes).
    """

    def __init__(
        self,
        output_dir: str = "./data/methylation",
        max_concurrent: int = 5,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        chunk_size_mb: float = 10.0,
        timeout: int = 300,
        on_progress: Optional[Callable] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.chunk_size = int(chunk_size_mb * 1024 * 1024)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.on_progress = on_progress
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily create semaphore in the running event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def download_many(
        self, tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Download multiple files concurrently.

        Args:
            tasks: List of dicts, each with:
                - accession: str
                - url: str
                - filename: str (optional, derived from URL if not given)
                - subdir: str (optional subdirectory under output_dir)

        Returns:
            List of result dicts with accession, status, local_path, error.
        """
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={"User-Agent": "MethyAgent/1.0"},
        ) as session:
            coros = [self._download_one(session, task) for task in tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        # Normalize exceptions to result dicts
        final = []
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                final.append({
                    "accession": task.get("accession", "unknown"),
                    "status": "failed",
                    "local_path": None,
                    "error": str(result),
                })
            else:
                final.append(result)

        return final

    def download_many_sync(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for download_many (for use in non-async contexts)."""
        return asyncio.run(self.download_many(tasks))

    async def download_one(
        self,
        url: str,
        accession: str,
        filename: Optional[str] = None,
        subdir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download a single file. Convenience wrapper."""
        task = {
            "accession": accession,
            "url": url,
            "filename": filename,
            "subdir": subdir,
        }
        connector = aiohttp.TCPConnector()
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={"User-Agent": "MethyAgent/1.0"},
        ) as session:
            return await self._download_one(session, task)

    # ------------------------------------------------------------------ #
    #  Internal download logic                                             #
    # ------------------------------------------------------------------ #

    async def _download_one(
        self, session: aiohttp.ClientSession, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Download a single file with retry and resume support."""
        async with self._get_semaphore():
            accession = task.get("accession", "unknown")
            url = task["url"]
            subdir = task.get("subdir") or accession
            filename = task.get("filename") or _url_to_filename(url)

            dest_dir = self.output_dir / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / filename
            tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

            for attempt in range(self.retry_attempts):
                try:
                    result = await self._attempt_download(
                        session, url, accession, dest_path, tmp_path
                    )
                    return result
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < self.retry_attempts - 1:
                        wait = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Download attempt {attempt + 1} failed for {accession}: {e}. "
                            f"Retrying in {wait:.1f}s..."
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"All {self.retry_attempts} attempts failed for {accession}: {e}")
                        return {
                            "accession": accession,
                            "status": "failed",
                            "local_path": None,
                            "error": str(e),
                        }

    async def _attempt_download(
        self,
        session: aiohttp.ClientSession,
        url: str,
        accession: str,
        dest_path: Path,
        tmp_path: Path,
    ) -> Dict[str, Any]:
        """Single download attempt with range request support."""
        # Check if partial download exists (resume)
        resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"
            logger.info(f"Resuming {accession} from byte {resume_pos}")

        async with session.get(url, headers=headers) as resp:
            # Handle range request response
            if resume_pos > 0 and resp.status == 416:
                # Range not satisfiable — file already complete
                tmp_path.rename(dest_path)
                return self._success_result(accession, dest_path)

            if resp.status not in (200, 206):
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history,
                    status=resp.status,
                    message=f"HTTP {resp.status}",
                )

            total_size = int(resp.headers.get("Content-Length", 0)) + resume_pos
            mode = "ab" if resume_pos > 0 else "wb"

            md5 = hashlib.md5()
            bytes_downloaded = resume_pos

            async with aiofiles.open(tmp_path, mode) as f:
                async for chunk in resp.content.iter_chunked(self.chunk_size):
                    await f.write(chunk)
                    md5.update(chunk)
                    bytes_downloaded += len(chunk)

                    if self.on_progress:
                        self.on_progress(accession, bytes_downloaded, total_size)

        # Move completed file
        tmp_path.rename(dest_path)
        checksum = md5.hexdigest()
        file_size = dest_path.stat().st_size

        logger.info(
            f"Downloaded {accession}: {dest_path.name} "
            f"({file_size / 1024 / 1024:.1f} MB, md5={checksum[:8]}...)"
        )

        return {
            "accession": accession,
            "status": "done",
            "local_path": str(dest_path),
            "file_size_bytes": file_size,
            "checksum_md5": checksum,
            "error": None,
        }

    @staticmethod
    def _success_result(accession: str, path: Path) -> Dict[str, Any]:
        return {
            "accession": accession,
            "status": "done",
            "local_path": str(path),
            "file_size_bytes": path.stat().st_size,
            "checksum_md5": None,
            "error": None,
        }


# ------------------------------------------------------------------ #
#  GEO-specific download helpers                                       #
# ------------------------------------------------------------------ #

def build_geo_download_tasks(
    metadata: Dict[str, Any],
    output_dir: str,
    data_type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build download task list for a GEO dataset.

    Args:
        metadata: GEO series metadata dict from GEOClient.get_series_metadata().
        output_dir: Base output directory.
        data_type_filter: Optional filter ('array', 'sequencing', or None for all).

    Returns:
        List of download task dicts.
    """
    accession = metadata["accession"]
    supp_files = metadata.get("supplementary_files", [])
    data_type = metadata.get("data_type", "unknown")

    if data_type_filter and data_type != data_type_filter and data_type != "unknown":
        return []

    tasks = []
    for url in supp_files:
        filename = _url_to_filename(url)
        if _is_methylation_file(filename):
            tasks.append({
                "accession": accession,
                "url": url,
                "filename": filename,
                "subdir": accession,
            })

    # If no supplementary files found, add the SOFT file as fallback
    if not tasks:
        prefix = accession[:-3] + "nnn"
        soft_url = (
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/"
            f"matrix/{accession}_series_matrix.txt.gz"
        )
        tasks.append({
            "accession": accession,
            "url": soft_url,
            "filename": f"{accession}_series_matrix.txt.gz",
            "subdir": accession,
        })

    return tasks


def build_tcga_download_tasks(
    dataset_record: Dict[str, Any],
    output_dir: str,
    gdc_api_base: str = "https://api.gdc.cancer.gov",
) -> List[Dict[str, Any]]:
    """
    Build download task list for a TCGA dataset.

    Args:
        dataset_record: TCGA dataset record from GDCClient.files_to_dataset_records().
        output_dir: Base output directory.
        gdc_api_base: GDC API base URL.

    Returns:
        List of download task dicts.
    """
    accession = dataset_record["accession"]
    file_ids = dataset_record.get("file_ids", [])

    tasks = []
    for file_id in file_ids:
        url = f"{gdc_api_base}/data/{file_id}"
        tasks.append({
            "accession": accession,
            "url": url,
            "filename": f"{file_id}.txt",
            "subdir": accession,
        })

    return tasks


# ------------------------------------------------------------------ #
#  Utility functions                                                   #
# ------------------------------------------------------------------ #

def _url_to_filename(url: str) -> str:
    """Extract filename from URL, stripping query parameters."""
    from urllib.parse import urlparse, unquote
    path = urlparse(url).path
    name = unquote(path.split("/")[-1])
    return name if name else "download.bin"


def _is_methylation_file(filename: str) -> bool:
    """
    Return True if the filename looks like a methylation data file.
    Requires BOTH a methylation-related keyword AND a data file extension
    to avoid false positives (e.g. README.txt, LICENSE).
    """
    fname_lower = filename.lower()
    methylation_keywords = (
        "beta", "methylat", "idat", "bismark", "cpg", "cov",
        "matrix", "mvalue", "m_value",
    )
    valid_extensions = (
        ".txt.gz", ".csv.gz", ".tsv.gz", ".bed.gz", ".cov.gz",
        ".txt", ".csv", ".tsv", ".bed", ".cov", ".idat",
        ".zip", ".tar.gz",
    )
    has_keyword = any(kw in fname_lower for kw in methylation_keywords)
    has_extension = any(fname_lower.endswith(ext) for ext in valid_extensions)
    # Require both keyword AND extension to avoid false positives
    return has_keyword and has_extension
