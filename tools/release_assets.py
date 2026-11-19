"""Pinned third-party release assets: the Moth 0.8.1 A3 LM and graph, Silero VAD.

The E-005 readout and package loaded ``../moth-stt/release/lm_bpe3.bin``: a stale,
gitignored copy of the LM shipped from 0.5.0 to 0.8.0 (sha256 ``dcf1d59f…``), not
the 0.8.1 A3 LM (``3be1177b…``) that produced the E-002 reference hypotheses. Every
MOTH-50 tool now takes its LM from here and checks the sha256 before decoding, so
a path can no longer silently select a different LM.

    python tools/release_assets.py fetch           # every pinned asset
    python tools/release_assets.py fetch lm        # only the A3 LM
    python tools/release_assets.py check

Files go to the gitignored ``data/raw/``; the sources and licences are pinned in
``data/sources.lock.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Hugging Face revision of the published ``semantique-team/moth-stt`` 0.8.1 release.
MOTH_081_REVISION = "1257327e3622f3ac6c5653a32f87b793049c36b7"
MOTH_081_URL = f"https://huggingface.co/semantique-team/moth-stt/resolve/{MOTH_081_REVISION}"
MOTH_081_DIR = REPO / "data/raw/moth-0.8.1"

#: ``snakers4/silero-vad`` commit ``add v6.2 model``: its ``silero_vad.onnx`` is the
#: file Moth 0.8.1 and E-005 ship (sha256 ``1a153a22…``).
SILERO_REVISION = "bfdc0193023f121ea5b3cc7b176dbed570a68a59"
SILERO_URL = (
    f"https://raw.githubusercontent.com/snakers4/silero-vad/{SILERO_REVISION}/src/silero_vad/data"
)
SILERO_DIR = REPO / "data/raw/silero-vad"

#: The LM shipped from 0.5.0 to 0.8.0. E-005 decoded with it by mistake; it is
#: recognised so a readout can name it, and never selected by default.
PRE_A3_LM_SHA256 = "dcf1d59f075858c7e5f210804b5aabde847fb5d33e8d716e0d534f7c7886003e"


@dataclass(frozen=True)
class Asset:
    key: str
    path: Path
    url: str
    sha256: str
    size: int
    description: str


A3_LM = Asset(
    key="lm",
    path=MOTH_081_DIR / "lm_bpe3.bin",
    url=f"{MOTH_081_URL}/lm_bpe3.bin",
    sha256="3be1177b0ca9f18343efb8d00f0d96365226595c9de32d992720aa4e0b0f16f1",
    size=5_381_992,
    description="Moth 0.8.1 A3 trigram LM (E-031), the E-002 reference decoder LM",
)
MOTH_081_MODEL = Asset(
    key="moth-0.8.1-model",
    path=MOTH_081_DIR / "model.int8.onnx",
    url=f"{MOTH_081_URL}/model.int8.onnx",
    sha256="7bbfad7e288640538c5c0147f81738bfd63021727b8760b1f3cbbcc4f939764d",
    size=11_910_677,
    description="Moth 0.8.1 INT8 acoustic graph, the reference system for S1-dev",
)
SILERO_VAD = Asset(
    key="vad",
    path=SILERO_DIR / "silero_vad.onnx",
    url=f"{SILERO_URL}/silero_vad.onnx",
    sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    size=2_327_524,
    description="Silero VAD, 8 kHz + 16 kHz graph (shipped by Moth 0.8.1 and E-005)",
)
SILERO_VAD_16K = Asset(
    key="vad-16k",
    path=SILERO_DIR / "silero_vad_16k_op15.onnx",
    url=f"{SILERO_URL}/silero_vad_16k_op15.onnx",
    sha256="7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49",
    size=1_289_603,
    description="Silero VAD, 16 kHz-only graph from the same commit (E-012 candidate)",
)

ASSETS = {asset.key: asset for asset in (A3_LM, MOTH_081_MODEL, SILERO_VAD, SILERO_VAD_16K)}


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(asset: Asset, path: Path | None = None) -> Path:
    """Return the asset path after checking its size and sha256; raise with a fetch hint."""
    path = Path(path) if path is not None else asset.path
    hint = f"fetch it with `python tools/release_assets.py fetch {asset.key}`"
    if not path.is_file():
        raise FileNotFoundError(f"{asset.description} missing at {path}; {hint}")
    size = path.stat().st_size
    if size != asset.size:
        raise ValueError(f"{path}: {size} bytes, expected {asset.size} ({asset.description})")
    digest = sha256(path)
    if digest != asset.sha256:
        raise ValueError(f"{path}: sha256 {digest} != {asset.sha256} ({asset.description})")
    return path


def lm_label(digest: str) -> str:
    if digest == A3_LM.sha256:
        return "a3-0.8.1"
    if digest == PRE_A3_LM_SHA256:
        return "pre-a3-0.5.0-0.8.0"
    return "other"


def resolve_lm(path: Path | None = None, *, allow_other: bool = False) -> dict:
    """The decoder LM: the pinned A3 file by default, any other file only on request.

    Gate readouts compare against hypotheses decoded with A3, so a different LM is
    an experiment (E-004), not a default; ``allow_other`` makes that explicit.
    """
    if path is None:
        resolved = verify(A3_LM)
        return {"path": resolved, "sha256": A3_LM.sha256, "label": "a3-0.8.1", "is_a3": True}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LM not found: {path}")
    digest = sha256(path)
    label = lm_label(digest)
    if label != "a3-0.8.1" and not allow_other:
        raise ValueError(
            f"{path} (sha256 {digest[:12]}…, {label}) is not the 0.8.1 A3 LM "
            f"({A3_LM.sha256[:12]}…). Gate readouts and releases use A3; pass "
            "--allow-other-lm only for an LM experiment such as E-004."
        )
    return {"path": path, "sha256": digest, "label": label, "is_a3": label == "a3-0.8.1"}


def fetch(asset: Asset, *, force: bool = False) -> Path:
    """Download ``asset`` to its pinned path and verify it; skip a verified existing file."""
    if asset.path.is_file() and not force:
        return verify(asset)
    asset.path.parent.mkdir(parents=True, exist_ok=True)
    partial = asset.path.with_name(asset.path.name + ".part")
    with urllib.request.urlopen(asset.url, timeout=120) as response, partial.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    try:
        verify(asset, partial)
    except ValueError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(asset.path)
    return asset.path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_assets.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_parser = sub.add_parser("fetch", help="download and verify pinned assets")
    fetch_parser.add_argument("keys", nargs="*", help=f"any of {', '.join(ASSETS)} (default: all)")
    fetch_parser.add_argument("--force", action="store_true")
    sub.add_parser("check", help="verify every pinned asset that is present")
    args = parser.parse_args(argv)

    if args.command == "fetch":
        unknown = sorted(set(args.keys) - set(ASSETS))
        if unknown:
            parser.error(f"unknown asset(s) {unknown}; choose from {sorted(ASSETS)}")
        for key in args.keys or list(ASSETS):
            path = fetch(ASSETS[key], force=args.force)
            print(f"{key}: {path} ok")
        return 0

    status = 0
    for key, asset in ASSETS.items():
        try:
            verify(asset)
            print(f"{key}: ok ({asset.path})")
        except FileNotFoundError:
            print(f"{key}: missing ({asset.path})")
        except ValueError as error:
            print(f"{key}: BAD — {error}")
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
