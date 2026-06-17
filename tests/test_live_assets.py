"""Tests for live_assets.sync_voices — fetching builtin voices from the
live-assets git repo instead of GCS."""

import io
import tarfile

import pytest

import live_assets


def _make_tarball(files: dict[str, bytes], root: str = "live-assets-abc123") -> bytes:
    """Build a gzipped tarball mirroring GitHub's codeload layout (everything
    nested under a single ``<repo>-<sha>/`` root dir)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path, data in files.items():
            info = tarfile.TarInfo(name=f"{root}/{path}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def repo_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_ASSETS_REPO", "QarlAI/live-assets")
    monkeypatch.setenv("LIVE_ASSETS_BRANCH", "develop")
    monkeypatch.setenv("LIVE_ASSETS_VOICES_PREFIX", "voices/qwen3")
    monkeypatch.setenv("LIVE_ASSETS_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_disabled_returns_false(monkeypatch):
    monkeypatch.delenv("LIVE_ASSETS_REPO", raising=False)
    monkeypatch.delenv("LIVE_ASSETS_BRANCH", raising=False)
    assert live_assets.sync_voices() is False


def test_extracts_only_qwen3_voices_flat(repo_env, monkeypatch):
    tar = _make_tarball({
        "voices/qwen3/english-male.wav": b"WAVDATA",
        "voices/qwen3/english-male.txt": b"a transcript",
        "voices/qwen3/english-male.pt": b"PTDATA",
        "voices/voxcpm/vx-ref/reference.wav": b"OTHER",   # different provider
        "backgrounds/office.png": b"IMG",                  # different asset type
        "README.md": b"docs",                              # top-level noise
    })
    monkeypatch.setattr(live_assets, "_fetch_tarball", lambda *a, **k: tar)

    assert live_assets.sync_voices() is True

    assert (repo_env / "english-male.wav").read_bytes() == b"WAVDATA"
    assert (repo_env / "english-male.txt").read_bytes() == b"a transcript"
    assert (repo_env / "english-male.pt").read_bytes() == b"PTDATA"
    # Only the qwen3 voices are extracted; nothing else.
    assert sorted(p.name for p in repo_env.iterdir()) == [
        "english-male.pt", "english-male.txt", "english-male.wav",
    ]


def test_idempotent_skips_same_size(repo_env, monkeypatch):
    (repo_env / "english-male.wav").write_bytes(b"WAVDATA")
    tar = _make_tarball({"voices/qwen3/english-male.wav": b"WAVDATA"})
    monkeypatch.setattr(live_assets, "_fetch_tarball", lambda *a, **k: tar)

    # Already present at the same size — still succeeds, leaves file intact.
    assert live_assets.sync_voices() is True
    assert (repo_env / "english-male.wav").read_bytes() == b"WAVDATA"


def test_path_traversal_member_skipped(repo_env, monkeypatch):
    tar = _make_tarball({
        "voices/qwen3/english-male.wav": b"OK",
        "voices/qwen3/../../evil.txt": b"PWNED",
    })
    monkeypatch.setattr(live_assets, "_fetch_tarball", lambda *a, **k: tar)

    assert live_assets.sync_voices() is True
    assert (repo_env / "english-male.wav").exists()
    # The traversal member must not escape the destination dir.
    assert not (repo_env.parent / "evil.txt").exists()


def test_fetch_failure_returns_false(repo_env, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(live_assets, "_fetch_tarball", _boom)

    assert live_assets.sync_voices() is False
