"""Repackage a release with a corrected NOTICE; every runtime asset stays byte-identical.

D37: the ``0.9.7-e023-r1`` NOTICE describes the E-034c LM as Moth 0.8.1's LM
"reused unchanged". This tool copies the five runtime assets of an existing
release, takes ``NOTICE`` from a fresh ``tools/export_release.py`` run (written by
the corrected ``tools/release_notice.py``), changes only the config ``version``,
and refuses to finish unless every asset keeps its declared sha256 and the
package validates within the 50 MB budget. No readout is needed: the graph, LM,
VAD, vocabulary, mel filters and decoder settings are the same bytes.

    python tools/renotice_release.py --release runs/E-023/release-0.9.7 \\
      --notice runs/E-023/release-0.9.8-scratch/NOTICE --version 0.9.8-e023-r1 \\
      --out-dir runs/E-023/release-0.9.8 --report runs/E-023/export-report-0.9.8.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "moth-stt"))

from moth_release import MAX_BUDGET_BYTES, validate_release


def renotice(release: Path, notice: Path, version: str, out_dir: Path) -> dict:
    """Write the repackaged release to ``out_dir`` and return its report."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"refusing to write into a non-empty {out_dir}")
    config, before = validate_release(release, MAX_BUDGET_BYTES)
    source_version = config["version"]
    if not version.strip() or version == source_version:
        raise ValueError(f"the new version must differ from {source_version!r}")
    new_notice = notice.read_bytes()
    if not new_notice.strip():
        raise ValueError(f"{notice} is empty")
    if new_notice == (release / "NOTICE").read_bytes():
        raise ValueError(f"{notice} is the NOTICE already in {release}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in config["files"]:
        shutil.copyfile(release / name, out_dir / name)
    (out_dir / "NOTICE").write_bytes(new_notice)
    config = {**config, "version": version}
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    _, after = validate_release(out_dir, MAX_BUDGET_BYTES)
    for name in config["files"]:
        if after["files"][name]["sha256"] != before["files"][name]["sha256"]:
            raise ValueError(f"{name} changed while repackaging")
    return {
        "tool": "tools/renotice_release.py",
        "source_release": str(release),
        "source_version": source_version,
        "version": version,
        "notice_from": str(notice),
        "changed": ["NOTICE", "config.json"],
        "unchanged_assets": {name: after["files"][name]["sha256"] for name in config["files"]},
        "notice": {
            "before": before["files"]["NOTICE"],
            "after": after["files"]["NOTICE"],
        },
        "config": {
            "before": before["files"]["config.json"],
            "after": after["files"]["config.json"],
        },
        "total_bytes": {"before": before["total_bytes"], "after": after["total_bytes"]},
        "budget_bytes": after["budget_bytes"],
        "remaining_bytes": after["remaining_bytes"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, required=True, help="the release to repackage")
    parser.add_argument("--notice", type=Path, required=True, help="the corrected NOTICE")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    report = renotice(args.release, args.notice, args.version, args.out_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
