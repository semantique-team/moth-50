"""Tests for the static Space staging and upload guard."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from publish_space import stage
from test_release_budget import _release

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "publish_space.py"
HF_HOME = tempfile.mkdtemp(prefix="moth-space-hfhome-")


def test_stage_copies_the_verified_release(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release)
    destination = tmp_path / "space"

    report = stage(destination, release)

    assert report["problems"] == []
    assert report["version"] == "0.0.1"
    assert report["files"] > 0 and report["bytes"] > 0
    for name in ("index.html", "README.md", "logo.png", "ort.js", "release/NOTICE"):
        assert (destination / name).is_file(), name
    assert (destination / "release" / "model.int8.onnx").is_file()


def test_stage_refuses_a_tampered_release(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release)
    (release / "model.int8.onnx").write_bytes(b"tampered")

    report = stage(tmp_path / "space", release)

    assert report["problems"]
    assert "sha256 mismatch" in report["problems"][0]


def test_push_without_token_fails_before_the_network(tmp_path: Path) -> None:
    release = tmp_path / "release"
    release.mkdir()
    _release(release)
    env = {
        "PYTHONPATH": str(REPO),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HF_HOME": HF_HOME,
    }
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--release",
            str(release),
            "--dest",
            str(tmp_path / "space"),
            "--push",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 2
    assert "HF_TOKEN" in result.stderr
