"""Summarize the E-018 teacher bake-off and apply the pre-registered rule.

Reads every ``runs/E-018/<system>/gold_train.metrics.json`` plus its
``meta.json``, reports WER/ETA/italianized ETA/insertion/RTFx, and picks the
teacher: WER <= 7.0 %, ETA >= 85 %, italianized ETA >= 72.3 %; among qualifiers
the lowest WER, with RTFx as the tie-break within 0.5 pp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WER_GATE = 0.07
ETA_GATE = 0.85
ITALIANIZED_GATE = 0.723
TIE_PP = 0.005


def _model_bytes(meta: dict) -> int | None:
    if meta.get("size_bytes"):
        return int(meta["size_bytes"])
    model_dir = meta.get("model_dir")
    if not model_dir:
        return None
    path = Path(model_dir)
    if not path.is_dir():
        return None
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("runs/E-018"))
    parser.add_argument("--out", type=Path, default=Path("runs/E-018/summary.json"))
    args = parser.parse_args(argv)

    rows = []
    for directory in sorted(args.root.iterdir()):
        metrics_path = directory / "gold_train.metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        meta_path = directory / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        overall = metrics["overall"]
        italianized = metrics["breakdown"]["category"].get("italianized", {}).get("eta")
        rtfx = meta.get("rtfx")
        if rtfx is None and meta.get("sets"):
            values = [entry.get("rtfx") for entry in meta["sets"].values() if entry.get("rtfx")]
            rtfx = sum(values) / len(values) if values else None
        row = {
            "system": meta.get("system") or directory.name,
            "dir": str(directory),
            "clips": overall["n"],
            "ref_words": overall["ref_words"],
            "wer": overall["wer"],
            "wer_ci": overall.get("wer_ci"),
            "eta": overall["eta"],
            "italianized_eta": italianized,
            "insertion_rate": overall["insertion_rate"],
            "rtfx": rtfx,
            "model_bytes": _model_bytes(meta),
            "language": meta.get("language"),
            "device": meta.get("device"),
        }
        row["label"] = row["system"] + (f" ({row['language']})" if row["language"] else "")
        row["qualifies"] = bool(
            row["wer"] <= WER_GATE
            and row["eta"] >= ETA_GATE
            and (row["italianized_eta"] or 0) >= ITALIANIZED_GATE
        )
        rows.append(row)

    qualifiers = [row for row in rows if row["qualifies"]]
    selected = None
    if qualifiers:
        best = min(row["wer"] for row in qualifiers)
        close = [row for row in qualifiers if row["wer"] <= best + TIE_PP]
        selected = max(close, key=lambda row: row["rtfx"] or 0).copy()
        selected["rule"] = "lowest WER; within 0.5 pp the faster RTFx"
        selected["best_wer_in_tie"] = best
        selected["tie_members"] = [row["system"] for row in close]

    excluded_path = args.root / "excluded.json"
    excluded = (
        json.loads(excluded_path.read_text(encoding="utf-8")) if excluded_path.is_file() else []
    )
    report = {
        "tool": "tools/e018_summary.py",
        "gates": {
            "wer_max": WER_GATE,
            "eta_min": ETA_GATE,
            "italianized_eta_min": ITALIANIZED_GATE,
            "tie_pp": TIE_PP,
        },
        "teachers": rows,
        "excluded": excluded,
        "selected": selected,
        "recommendation": (
            f"teacher {selected['system']}"
            if selected
            else "no teacher qualifies; E-019 keeps only monolingual Italian sources"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'system':38s} {'WER':>7s} {'ETA':>7s} {'it-ETA':>7s} {'RTFx':>7s}  qualifies")
    for row in rows:
        print(
            f"{row['label']:38s} {row['wer'] * 100:7.2f} {row['eta'] * 100:7.2f} "
            f"{(row['italianized_eta'] or 0) * 100:7.2f} "
            f"{(row['rtfx'] or 0):7.3f}  {row['qualifies']}"
        )
    for row in excluded:
        print(f"{row.get('system', '?'):38s} excluded: {row.get('reason', '')}")
    print(report["recommendation"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
