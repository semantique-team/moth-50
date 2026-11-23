"""Score browser hypotheses against a Python readout (E-016).

Reads the reference manifest and a Python hypothesis JSONL, scores each browser
probe JSON from ``tools/browser_probe.mjs`` and reports WER/ETA, normalized
identity and the paired bootstrap difference. The browser files may come from
different audio front-ends (raw 48 kHz, Python-FFT 16 kHz) or segmentation
settings; the reference manifest is always the one used for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "moth-stt"))

import moth_eval
from moth_text import normalize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_hyps(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid = str(row["id"])
        if uid in result:
            raise SystemExit(f"{path}: duplicate id {uid}")
        result[uid] = row["hyp"]
    return result


def _load_browser(path: Path) -> tuple[dict[str, str], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    hyps = {str(row["id"]): row["hyp"] for row in data["hyps"]}
    meta = {
        "environment": data.get("environment"),
        "browser": data.get("browser"),
        "threads": data.get("threads"),
        "segment": data.get("segment"),
        "errors": data.get("errors"),
    }
    return hyps, meta


def _overall(result: dict) -> dict:
    overall = result["overall"]
    return {
        "n": overall["n"],
        "ref_words": overall["ref_words"],
        "wer": overall["wer"],
        "wer_ci": overall["wer_ci"],
        "eta": overall["eta"],
        "insertion_rate": overall["insertion_rate"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-hyp", type=Path, required=True)
    parser.add_argument("--browser", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = {str(row["id"]) for row in rows}
    reference = _load_hyps(args.reference_hyp)
    if set(reference) != ids:
        raise SystemExit("reference hypotheses do not cover exactly the manifest ids")
    reference_result = moth_eval.score_rows(rows, reference)
    report = {
        "tool": "tools/browser_parity.py",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "reference_hyp": str(args.reference_hyp),
        "reference_hyp_sha256": _sha256(args.reference_hyp),
        "reference": _overall(reference_result),
        "variants": {},
    }

    for path in args.browser:
        hyps, meta = _load_browser(path)
        if set(hyps) != ids:
            raise SystemExit(f"{path}: hypotheses do not cover exactly the manifest ids")
        result = moth_eval.score_rows(rows, hyps)
        paired = moth_eval._compare(result, reference_result, args.bootstrap, args.seed)
        differing = []
        for row in rows:
            uid = str(row["id"])
            lang = row.get("lang") if row.get("lang") in ("it", "en") else "it"
            if normalize(hyps[uid], lang) != normalize(reference[uid], lang):
                differing.append(uid)
        report["variants"][path.stem] = {
            "browser_file": str(path),
            "browser_file_sha256": _sha256(path),
            **meta,
            "metrics": _overall(result),
            "identical": len(rows) - len(differing),
            "identical_ratio": (len(rows) - len(differing)) / len(rows),
            "paired": {
                "wer_diff": paired["wer_diff"],
                "wer_diff_ci": paired["wer_diff_ci"],
                "wer_tie": paired["wer_tie"],
                "eta_diff": paired["eta_diff"],
                "eta_diff_ci": paired["eta_diff_ci"],
                "eta_tie": paired["eta_tie"],
            },
            "differing_ids": differing,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, variant in report["variants"].items():
        metrics = variant["metrics"]
        paired = variant["paired"]
        print(
            f"{name}: WER {metrics['wer']:.4f} ETA {metrics['eta']:.4f} "
            f"identical {variant['identical']}/{metrics['n']} "
            f"delta {paired['wer_diff'] * 100:+.4f} pp "
            f"CI [{paired['wer_diff_ci'][0] * 100:+.4f}, {paired['wer_diff_ci'][1] * 100:+.4f}] "
            f"tie={paired['wer_tie']}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
