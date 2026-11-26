"""Tests for the Hugging Face upload guard.

The upload script must refuse to publish anything that is not the verified release: a stale
``release-*`` folder, a hand-edited file, or a model card for another version. These tests exercise
that gate without touching the network.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from test_release_budget import _release

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "upload_hf.py"

#: An empty HF home for the tests: a real token from `hf auth login` in the developer's home must
#: not turn "push without a token" into a test that reaches the network.
HF_HOME = tempfile.mkdtemp(prefix="moth-hfhome-")


def _fixture(root: Path) -> tuple[Path, Path]:
    """A tiny verified release plus the card that documents it."""
    release = root / "release"
    release.mkdir()
    _release(release)
    card = root / "CARD.md"
    card.write_text("# MOTH-50 0.0.1 — model card\n\nTest card.\n", encoding="utf-8")
    return release, card


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PYTHONPATH": str(REPO),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HF_HOME": HF_HOME,
    }
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _args(root: Path, release: Path, card: Path) -> list[str]:
    return [
        "--repo",
        "someone/moth-test",
        "--release",
        str(release),
        "--card",
        str(card),
    ]


def test_dry_run_accepts_a_matching_release(tmp_path: Path) -> None:
    release, card = _fixture(tmp_path)
    result = _run(tmp_path, *_args(tmp_path, release, card))
    assert result.returncode == 0, result.stderr
    assert "verified release 0.0.1" in result.stdout
    assert "README.md" in result.stdout
    assert "nothing uploaded" in result.stdout


def test_modified_file_is_refused(tmp_path: Path) -> None:
    release, card = _fixture(tmp_path)
    (release / "model.int8.onnx").write_bytes(b"tampered")
    result = _run(tmp_path, *_args(tmp_path, release, card))
    assert result.returncode == 2
    assert "sha256 mismatch" in result.stderr


def test_missing_file_is_refused(tmp_path: Path) -> None:
    release, card = _fixture(tmp_path)
    (release / "vocab.json").unlink()
    result = _run(tmp_path, *_args(tmp_path, release, card))
    assert result.returncode == 2
    assert "missing release file" in result.stderr


def test_missing_card_is_refused(tmp_path: Path) -> None:
    release, card = _fixture(tmp_path)
    card.unlink()
    result = _run(tmp_path, *_args(tmp_path, release, card))
    assert result.returncode == 2
    assert "missing model card" in result.stderr


def test_card_for_a_different_release_is_refused(tmp_path: Path) -> None:
    """A card that documents another version publishes true numbers for the wrong artifact."""
    release, card = _fixture(tmp_path)
    card.write_text("# MOTH-50 0.0.0 — old card\n", encoding="utf-8")
    result = _run(tmp_path, *_args(tmp_path, release, card))
    assert result.returncode == 2
    assert "different release" in result.stderr


def test_push_without_token_fails_before_the_network(tmp_path: Path) -> None:
    release, card = _fixture(tmp_path)
    result = _run(tmp_path, *_args(tmp_path, release, card), "--push")
    assert result.returncode == 2
    assert "HF_TOKEN" in result.stderr


@pytest.mark.skipif(
    not (REPO / "runs" / "E-023" / "release-0.9.8").is_dir(),
    reason="release-0.9.8 not built on this machine",
)
def test_the_real_release_passes_the_guard() -> None:
    """The built package must match its own ``config.json`` and the card must name its version."""
    release = REPO / "runs" / "E-023" / "release-0.9.8"
    version = json.loads((release / "config.json").read_text(encoding="utf-8"))["version"]
    card = REPO / "MODEL_CARD.md"
    result = _run(REPO, *_args(REPO, release, card))
    assert result.returncode == 0, result.stderr
    assert f"verified release {version}" in result.stdout
