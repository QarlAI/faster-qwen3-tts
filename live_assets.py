"""live_assets.py — load builtin voices from the QarlAI/live-assets git repo
instead of the GCS bucket.

Part of the GCP->Azure migration; the voice counterpart of pipeqarl PR #119.
One HTTPS GET of the whole-repo codeload tarball at startup, from which only the
``voices/qwen3/`` subtree is extracted into the local voices dir. No GCS, no
per-file GitHub calls.

Enabled only when LIVE_ASSETS_REPO + LIVE_ASSETS_BRANCH are set; otherwise
``sync_voices()`` returns False and the caller falls back to the existing GCS
download.
"""

import io
import logging
import os
import tarfile
import time

import requests

from config import VOICE_CACHE_PREFIX, VOICES_DIR

logger = logging.getLogger(__name__)

_CODELOAD = "https://codeload.github.com"
_FETCH_TIMEOUT_S = 10.0
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_S = 1.0


def _repo_branch() -> tuple[str, str]:
    return (
        os.environ.get("LIVE_ASSETS_REPO", "").strip(),
        os.environ.get("LIVE_ASSETS_BRANCH", "").strip(),
    )


def is_enabled() -> bool:
    repo, branch = _repo_branch()
    return bool(repo and branch)


def _token() -> str:
    # live-assets shares the avatar_profiles org PAT; reuse it when a dedicated
    # LIVE_ASSETS_TOKEN is not configured.
    return (
        os.environ.get("LIVE_ASSETS_TOKEN", "").strip()
        or os.environ.get("AVATAR_PROFILES_TOKEN", "").strip()
    )


def _voices_subpath() -> str:
    """In-repo path holding the qwen3 voices. Defaults to the GCS prefix
    (``voices/qwen3``) so repo and bucket layouts stay aligned."""
    return (
        os.environ.get("LIVE_ASSETS_VOICES_PREFIX", "").strip()
        or VOICE_CACHE_PREFIX
        or "voices/qwen3"
    ).strip("/")


def _dest_dir() -> str:
    return os.environ.get("LIVE_ASSETS_CACHE_DIR", "").strip() or VOICES_DIR


def _parse_repo(repo: str) -> str:
    repo = repo.strip().rstrip("/")
    if repo.startswith("http://") or repo.startswith("https://"):
        path = repo.split("github.com/", 1)[-1]
        return path[:-4] if path.endswith(".git") else path
    return repo


def _tarball_url(repo: str, branch: str) -> str:
    return f"{_CODELOAD}/{_parse_repo(repo)}/tar.gz/refs/heads/{branch}"


def _fetch_tarball(repo: str, branch: str, token: str) -> bytes:
    """GET the whole-repo tarball from codeload with retry/backoff. Raises on
    final failure."""
    url = _tarball_url(repo, branch)
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    logger.info(f"live-assets: fetching {repo}@{branch}")
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=_FETCH_TIMEOUT_S)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_error = e
            if attempt < _MAX_ATTEMPTS:
                logger.warning(
                    f"live-assets: fetch attempt {attempt}/{_MAX_ATTEMPTS} "
                    f"failed ({e}); retrying"
                )
                time.sleep(_BACKOFF_BASE_S * attempt)
    raise RuntimeError(
        f"live-assets: all {_MAX_ATTEMPTS} fetch attempts failed: {last_error}"
    )


def sync_voices() -> bool:
    """Fetch the live-assets repo and extract its qwen3 voices into the voices
    dir. Never raises; returns True on success, False when disabled or on
    failure (so the caller keeps its GCS fallback)."""
    repo, branch = _repo_branch()
    if not repo or not branch:
        return False

    try:
        tar_bytes = _fetch_tarball(repo, branch, _token())
    except Exception as e:
        logger.warning(f"live-assets: fetch failed; using GCS fallback: {e}")
        return False

    subpath = _voices_subpath()
    dest_dir = os.path.abspath(_dest_dir())
    os.makedirs(dest_dir, exist_ok=True)

    try:
        count = 0
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                # Strip the GitHub root dir "<repo>-<sha>/".
                rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
                if not rel.startswith(f"{subpath}/"):
                    continue
                inner = rel[len(subpath) + 1:]  # path under voices/qwen3/
                if not inner:
                    continue
                dest = os.path.abspath(os.path.join(dest_dir, inner))
                # Path-traversal guard.
                if not dest.startswith(dest_dir + os.sep):
                    continue
                # Idempotent: skip files already present at the same size.
                if os.path.exists(dest) and os.path.getsize(dest) == member.size:
                    continue
                src = tar.extractfile(member)
                if src is None:
                    continue
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(src.read())
                count += 1
    except Exception as e:
        logger.warning(f"live-assets: extract failed; using GCS fallback: {e}")
        return False

    logger.info(
        f"live-assets: loaded {count} voice file(s) from {repo}@{branch} "
        f"({subpath}/) into {dest_dir}"
    )
    return True
