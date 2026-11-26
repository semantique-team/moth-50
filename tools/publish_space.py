"""Publish the MOTH-50 static browser demo as a Hugging Face Space.

Assembles a staging directory from files already in the repository (the demo page, the vendored
JS runtime, the verified ``runs/E-023/release-0.9.8`` and the ONNX Runtime files the WASM backend
needs) and uploads it. Only the page, the Semantique brand assets and the Space card are
versioned; the binaries are copied from the release and ``node_modules/`` at build time.

    uv run --frozen -- python tools/publish_space.py --check          # stage + verify, no network
    uv run --frozen -- python tools/publish_space.py --serve          # also serve it locally
    HF_TOKEN=hf_... uv run --frozen -- python tools/publish_space.py \
      --push --repo semantique-team/moth-50-demo

Why a **static** Space and not Gradio: the point of this model is that it runs in the user's
browser and their audio never leaves it. A Gradio Space would run the model on Hugging Face's CPU
and quietly tell the opposite story. Serving without COOP/COEP means no SharedArrayBuffer, so the
runtime falls back to 1 thread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "moth-stt"))

from moth_release import MAX_BUDGET_BYTES, validate_release

SPACE_SRC = REPO / "runs/E-023/space"
WEB = REPO / "vendor/moth-stt-web"
RELEASE = REPO / "runs/E-023/release-0.9.8"

#: Files that define the page and its brand.
SPACE_FILES = (
    "index.html",
    "README.md",
    "semantique-logo.svg",
    "semantique-thumbnail.png",
    "logo.png",
    "logo-white.png",
    "favicon.png",
)
#: The vendored runtime. ``ort.wasm.min.mjs`` dynamically imports the threaded build, which loads
#: the ``.wasm`` — all three must sit in the same directory, so the Space keeps the node_modules
#: layout.
JS_FILES = ("ort.js", "decode.js", "lm.js", "mel.js", "mic.js", "mic-worklet.js", "vad.js")
ORT_DIR = "node_modules/onnxruntime-web/dist"
ORT_FILES = (
    "ort.wasm.min.mjs",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.wasm",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def release_files(config: dict) -> tuple[str, ...]:
    """The five runtime assets named by the release config, plus config and NOTICE."""
    return (
        config["model"]["file"],
        config["tokenizer"]["vocab"],
        config["features"]["mel_filters"],
        config["vad"]["file"],
        config["decode"]["lm"],
        "config.json",
        "NOTICE",
    )


def stage(destination: Path, release: Path = RELEASE) -> dict:
    """Build the Space tree; returns a report of what went in and whether it verifies."""
    try:
        config, report = validate_release(release, MAX_BUDGET_BYTES)
    except (OSError, ValueError) as exc:
        return {"files": 0, "bytes": 0, "problems": [str(exc)], "version": "unknown"}

    if destination.exists():
        shutil.rmtree(destination)
    (destination / "release").mkdir(parents=True)
    (destination / ORT_DIR).mkdir(parents=True)

    problems: list[str] = []
    for name in SPACE_FILES:
        source = SPACE_SRC / name
        if not source.is_file():
            problems.append(f"missing {source}")
            continue
        shutil.copy2(source, destination / name)
    for name in JS_FILES:
        source = WEB / name
        if not source.is_file():
            problems.append(f"missing {source}")
            continue
        shutil.copy2(source, destination / name)
    for name in ORT_FILES:
        source = WEB / ORT_DIR / name
        if not source.is_file():
            problems.append(f"missing {source}")
            continue
        shutil.copy2(source, destination / ORT_DIR / name)

    # The demo loads ./release/ and verifies these sha256 values in the browser before running.
    for name in release_files(config):
        source = release / name
        if not source.is_file():
            problems.append(f"missing {source}")
            continue
        shutil.copy2(source, destination / "release" / name)
        expected = report["files"].get(name, {}).get("sha256")
        if expected and sha256(source) != expected:
            problems.append(f"{name}: does not match the recorded release")

    total = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    return {
        "files": sum(1 for path in destination.rglob("*") if path.is_file()),
        "bytes": total,
        "problems": problems,
        "version": config["version"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="semantique-team/moth-50-demo")
    parser.add_argument(
        "--push", action="store_true", help="upload (default: stage and check only)"
    )
    parser.add_argument("--check", action="store_true", help="stage and verify, no network")
    parser.add_argument("--serve", action="store_true", help="serve the staged site locally")
    parser.add_argument("--dest", default=None, help="staging directory (default: a temp dir)")
    parser.add_argument("--private", action="store_true", help="create the Space private")
    parser.add_argument("--release", default=str(RELEASE), help="release directory to stage")
    args = parser.parse_args(argv)

    destination = Path(args.dest) if args.dest else Path(tempfile.mkdtemp(prefix="moth-space-"))
    report = stage(destination, Path(args.release))
    print(json.dumps(report, indent=1))
    print(f"staged in {destination}")
    if report["problems"]:
        for problem in report["problems"]:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    if args.serve:
        print(f"\nserving {destination} on http://127.0.0.1:8767 (Ctrl-C to stop)")
        return subprocess.call(
            [sys.executable, "-m", "http.server", "8767", "--bind", "127.0.0.1"],
            cwd=destination,
        )

    if not args.push:
        print("\nnothing uploaded (staged only). Use --push and HF_TOKEN to publish.")
        return 0

    # Three ways in, in order: the environment, then the token `hf auth login` stored (which keeps
    # the secret out of shell history and out of any chat), then refuse.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        from huggingface_hub import get_token

        token = get_token()
    if not token:
        print(
            "error: --push needs HF_TOKEN in the environment, or a token from `uv run hf auth login`",
            file=sys.stderr,
        )
        return 2

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        args.repo, repo_type="space", space_sdk="static", private=args.private, exist_ok=True
    )
    api.upload_folder(
        folder_path=str(destination),
        repo_id=args.repo,
        repo_type="space",
        commit_message=f"MOTH-50 {report['version']} — static browser demo",
    )
    print(f"\ndone: https://huggingface.co/spaces/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
