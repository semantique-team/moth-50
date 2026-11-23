"""E-027 listening audit: a seeded sample per voice/mode and the drop rule.

``sample`` draws ``--per-stratum`` clips (default 10) from every voice/mode
stratum of the TTS manifest with ``Random("<seed>:<stratum>")`` and writes
them with an empty ``verdict``. A listener sets each verdict to ``ok`` or
``fail`` (``fail``: unintelligible, wrong or missing words, truncated, noise).
``score`` applies the pre-registered rule: a stratum with **2 or more** fails
is dropped, and the manifest is rewritten without it before the mel shards,
the share and any training. The audit judges the audio against its text only;
an Italian accent on English terms is not a failure.

    python tools/e027_audit.py sample --manifest data/manifests/tts_e027.jsonl \\
      --out runs/E-027/audit.jsonl
    python tools/e027_audit.py score --audit runs/E-027/audit.jsonl \\
      --manifest data/manifests/tts_e027.jsonl --report runs/E-027/audit-report.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

MAX_FAILS = 1


def stratum(row: dict) -> str:
    return f"{row['speaker'].removeprefix('tts-')}-{row['tts']['mode']}"


def sample_audit(rows: list[dict], per_stratum: int, seed: int) -> list[dict]:
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    sample = []
    for name in sorted(groups):
        members = sorted(groups[name], key=lambda row: row["id"])
        chosen = random.Random(f"{seed}:{name}").sample(members, min(per_stratum, len(members)))
        for row in sorted(chosen, key=lambda row: row["id"]):
            sample.append(
                {
                    "id": row["id"],
                    "stratum": name,
                    "audio": row["audio"],
                    "text": row["text"],
                    "verdict": None,
                    "note": "",
                }
            )
    return sample


def score_audit(audit: list[dict], max_fails: int = MAX_FAILS) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in audit:
        if row.get("verdict") not in ("ok", "fail"):
            raise ValueError(f"{row['id']}: verdict must be 'ok' or 'fail'")
        entry = result.setdefault(row["stratum"], {"ok": 0, "fail": 0})
        entry[row["verdict"]] += 1
    for entry in result.values():
        entry["keep"] = entry["fail"] <= max_fails
    return result


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("--manifest", type=Path, required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--per-stratum", type=int, default=10)
    sample.add_argument("--seed", type=int, default=0)
    score = sub.add_parser("score")
    score.add_argument("--audit", type=Path, required=True)
    score.add_argument("--manifest", type=Path, required=True, help="rewritten if a stratum drops")
    score.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "sample":
        rows = sample_audit(_jsonl(args.manifest), args.per_stratum, args.seed)
        _write(args.out, rows)
        print(f"wrote {len(rows)} audit rows to {args.out}")
        return 0

    result = score_audit(_jsonl(args.audit))
    dropped = sorted(name for name, entry in result.items() if not entry["keep"])
    rows = _jsonl(args.manifest)
    kept = [row for row in rows if stratum(row) not in dropped]
    if dropped:
        _write(args.manifest, kept)
    report = {
        "tool": "tools/e027_audit.py",
        "audit": str(args.audit),
        "max_fails": MAX_FAILS,
        "strata": result,
        "dropped_strata": dropped,
        "manifest": str(args.manifest),
        "clips_before": len(rows),
        "clips_after": len(kept),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
