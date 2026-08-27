"""
geo-download Phase-2 ⑤ — tar/zip member-level handling.

Supplementary archives (tar / tar.gz / zip) are expanded into their members
BEFORE the junk filter: a compressed archive read as raw bytes otherwise looks
"unparseable" to inspect_matrix_head and gets junk-dropped, losing every member
inside. Extraction is conservative — on any failure the archive is kept as one
unit and flows through the normal pipeline.
"""
from __future__ import annotations

import gzip
import os
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_ARCHIVE_EXTS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".zip")


def is_archive(path: str) -> bool:
    """Extension or magic-bytes check for tar/zip family archives.

    NOTE: a plain .txt.gz matrix is NOT an archive — gzip alone never
    qualifies; .tar.gz/tgz must be detected by peeling the gzip layer and
    checking the tar "ustar" magic.
    """
    name = (path or "").lower()
    if name.endswith((".tar", ".tgz", ".tar.bz2", ".tbz2", ".zip")):
        return True
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
    except OSError:
        return False
    if magic.startswith(b"PK\x03\x04"):
        return True  # zip
    if magic.startswith(b"\x1f\x8b"):
        # gzip member — archive only if it's a tar inside (ustar magic at 257).
        try:
            with gzip.open(path, "rb") as f:
                f.seek(257)
                return f.read(5) == b"ustar"
        except Exception:
            return False
    # raw tar: "ustar" magic at offset 257
    try:
        with open(path, "rb") as f:
            f.seek(257)
            return f.read(5) == b"ustar"
    except OSError:
        return False


def extract_archive_members(path: str, out_dir: str,
                            max_members: int = 500) -> Optional[List[str]]:
    """
    Extract an archive's file members into out_dir (skipping directories,
    hidden files, __MACOSX). Returns extracted paths, or None on any failure
    (caller keeps the archive as one unit — never deletes on failure).
    """
    out = Path(out_dir)
    extracted: List[str] = []
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                infos = [i for i in zf.infolist()
                         if not i.is_dir() and _keep_member(i.filename)]
                if len(infos) > max_members:
                    logger.info(f"extract_archive_members({os.path.basename(path)}): "
                                f"{len(infos)} members > {max_members}, keeping archive whole")
                    return None
                for info in infos:
                    dest = _safe_dest(out, info.filename)
                    if not dest:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    extracted.append(str(dest))
        else:
            with tarfile.open(path, "r:*") as tf:
                infos = [m for m in tf.getmembers()
                         if m.isfile() and _keep_member(m.name)]
                if len(infos) > max_members:
                    logger.info(f"extract_archive_members({os.path.basename(path)}): "
                                f"{len(infos)} members > {max_members}, keeping archive whole")
                    return None
                for m in infos:
                    dest = _safe_dest(out, m.name)
                    if not dest:
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        continue
                    with fobj, open(dest, "wb") as dst:
                        dst.write(fobj.read())
                    extracted.append(str(dest))
        if not extracted:
            return None
        logger.info(f"extract_archive_members({os.path.basename(path)}): "
                    f"{len(extracted)} member(s) extracted")
        return extracted
    except Exception as e:
        logger.warning(f"extract_archive_members({os.path.basename(path)}): "
                       f"{e} → keeping archive as one unit")
        return None


def detect_per_sample_set(paths: List[str], n_gsms: int,
                          tolerance: float = 0.0) -> bool:
    """
    Do these paths look like one-file-per-sample (Tier-3 per-GSM downloads,
    per-sample tar members)? True when the file count ≈ the download-GSM count
    or every filename carries a GSM id.
    """
    if not paths or not n_gsms:
        return False
    import re
    gsm_re = re.compile(r"GSM\d+", re.IGNORECASE)
    if all(gsm_re.search(os.path.basename(p) or "") for p in paths):
        return True
    return abs(len(paths) - n_gsms) <= tolerance * n_gsms


def _keep_member(name: str) -> bool:
    base = os.path.basename(name)
    norm = name.replace("\\", "/")
    return (bool(base) and not base.startswith((".", "__", "_"))
            and ".." not in norm.split("/")
            and "__MACOSX" not in norm
            and not base.endswith(".DS_Store"))


def _safe_dest(out: Path, member_name: str) -> Optional[Path]:
    """Reject path-traversal members; place the rest under out using their
    basename (flatten — member dirs are noise for our purposes)."""
    base = os.path.basename(member_name.replace("\\", "/"))
    if not base or base in (".", ".."):
        return None
    dest = out / base
    if not str(dest.resolve()).startswith(str(out.resolve())):
        return None
    return dest
