"""Seed-mean comparison for the E-002 §7 (P6) gate and future final runs.

The protocol compares a candidate's seeds against a control and takes the
**seed mean** as the verdict statistic: per-utterance error and ETA-hit counts
are averaged across seeds, then a paired utterance bootstrap (1,000 resamples,
seed 0) gives the difference and its 95 % CI. Each seed is also reported on its
own; no seed is ever selected on S1-dev.

With ``--macro`` the metric is the equal-language macro WER over the manifest
``lang`` values (the dev-macro gate); without it the metric is pooled WER/ETA
for one mixed set (S1-dev). Differences are candidate − control: negative WER is
better, positive ETA is better. ``--control`` may be repeated: a multi-seed
control (E-027's same-length fine-tune) is then its own seed mean.

    python tools/seed_mean.py --macro \\
      --manifest ../moth-stt/data/manifests/mls_it_dev.jsonl \\
      --manifest ../moth-stt/data/manifests/ls_en_dev.jsonl \\
      --control runs/E-004/readout-e034c-int8.hyp.jsonl \\
      --seed runs/E-011/p6/s0-dev-macro.hyp.jsonl \\
      --seed runs/E-011/p6/s1-dev-macro.hyp.jsonl \\
      --out runs/E-011/p6/dev-seed-mean.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "moth-stt"))

from moth_eval import read_hypotheses, score_rows

BOOTSTRAP_LOW = 0.025
BOOTSTRAP_HIGH = 0.975


def _errors(stats: dict) -> float:
    return stats["sub"] + stats["del"] + stats["ins"]


def metric(per_utterance: dict[str, dict], groups: dict[str, list[str]]) -> dict:
    """Macro (mean over groups) or pooled WER/ETA for one system."""
    if len(groups) == 1:
        ids = next(iter(groups.values()))
        ref = sum(per_utterance[uid]["ref_words"] for uid in ids)
        err = sum(_errors(per_utterance[uid]) for uid in ids)
        en_ref = sum(per_utterance[uid]["en_ref"] for uid in ids)
        en_hit = sum(per_utterance[uid]["en_hit"] for uid in ids)
        return {"wer": err / ref if ref else 0.0, "eta": en_hit / en_ref if en_ref else None}
    wer = []
    eta = []
    for ids in groups.values():
        ref = sum(per_utterance[uid]["ref_words"] for uid in ids)
        wer.append(sum(_errors(per_utterance[uid]) for uid in ids) / ref if ref else 0.0)
        en_ref = sum(per_utterance[uid]["en_ref"] for uid in ids)
        if en_ref:
            eta.append(sum(per_utterance[uid]["en_hit"] for uid in ids) / en_ref)
    return {"wer": sum(wer) / len(wer), "eta": sum(eta) / len(eta) if eta else None}


def _resample(groups: dict[str, list[str]], rng: random.Random) -> dict[str, list[str]]:
    return {lang: [rng.choice(ids) for _ in ids] for lang, ids in groups.items()}


def _diff(
    system_a: dict,
    system_b: dict,
    groups: dict[str, list[str]],
    field: str,
    resamples: int,
    seed: int,
) -> tuple[float | None, list[float] | None]:
    if metric(system_a, groups)[field] is None or metric(system_b, groups)[field] is None:
        return None, None
    point = metric(system_a, groups)[field] - metric(system_b, groups)[field]
    if resamples <= 0:
        return point, None
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = _resample(groups, rng)
        values.append(metric(system_a, sample)[field] - metric(system_b, sample)[field])
    values.sort()
    lo = values[min(resamples - 1, int(BOOTSTRAP_LOW * resamples))]
    hi = values[min(resamples - 1, int(BOOTSTRAP_HIGH * resamples))]
    return point, [lo, hi]


def mean_system(systems: list[dict], ids: list[str]) -> dict:
    """Average per-utterance counts across seeds; reference counts are shared."""
    result = {}
    for uid in ids:
        base = systems[0][uid]
        result[uid] = {
            "ref_words": base["ref_words"],
            "sub": sum(s[uid]["sub"] for s in systems) / len(systems),
            "del": sum(s[uid]["del"] for s in systems) / len(systems),
            "ins": sum(s[uid]["ins"] for s in systems) / len(systems),
            "en_ref": base["en_ref"],
            "en_hit": sum(s[uid]["en_hit"] for s in systems) / len(systems),
        }
    return result


def control_system(systems: list[dict], ids: list[str]) -> dict:
    """One control readout as is, or the seed mean of several control seeds (E-027)."""
    return systems[0] if len(systems) == 1 else mean_system(systems, ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--control",
        type=Path,
        action="append",
        required=True,
        help="control hypotheses; repeat it for a multi-seed control (its seed mean)",
    )
    parser.add_argument("--seed", type=Path, action="append", required=True)
    parser.add_argument(
        "--macro",
        action="store_true",
        help="equal-language macro over row['lang'] (default: pooled)",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--category",
        action="store_true",
        help="also report per-category metrics and paired comparisons",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    manifest_rows = []
    for path in args.manifest:
        current = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows.extend(current)
        manifest_rows.append({"path": str(path), "rows": len(current)})
    ids = [str(row["id"]) for row in rows]
    groups = {"all": ids}
    if args.macro:
        groups = {}
        for row in rows:
            groups.setdefault(row["lang"], []).append(str(row["id"]))

    def scored(path: Path) -> dict:
        result = score_rows(rows, read_hypotheses(path), bootstrap=0)
        missing = set(ids) - set(result["per_utterance"])
        if missing:
            raise SystemExit(f"{path}: {len(missing)} manifest ids have no hypothesis")
        return result["per_utterance"]

    control = control_system([scored(path) for path in args.control], ids)
    seeds = [scored(path) for path in args.seed]
    mean = mean_system(seeds, ids)

    systems = {"control": metric(control, groups), "seed_mean": metric(mean, groups)}
    for index, per_utterance in enumerate(seeds):
        systems[f"seed_{index}"] = metric(per_utterance, groups)

    comparisons = {}
    for name, system in [("seed_mean", mean), *[(f"seed_{i}", s) for i, s in enumerate(seeds)]]:
        wer_diff, wer_ci = _diff(
            system, control, groups, "wer", args.bootstrap, args.bootstrap_seed
        )
        eta_diff, eta_ci = _diff(
            system, control, groups, "eta", args.bootstrap, args.bootstrap_seed
        )
        comparisons[name] = {
            "wer_diff": wer_diff,
            "wer_ci": wer_ci,
            "eta_diff": eta_diff,
            "eta_ci": eta_ci,
        }

    categories = None
    if args.category:
        category_groups: dict[str, list[str]] = {}
        for row in rows:
            category = row.get("category")
            if category is not None:
                category_groups.setdefault(str(category), []).append(str(row["id"]))
        categories = {}
        for category, category_ids in category_groups.items():
            group = {"all": category_ids}
            category_comparisons = {}
            for name, system in [
                ("seed_mean", mean),
                *[(f"seed_{i}", s) for i, s in enumerate(seeds)],
            ]:
                wer_diff, wer_ci = _diff(
                    system, control, group, "wer", args.bootstrap, args.bootstrap_seed
                )
                eta_diff, eta_ci = _diff(
                    system, control, group, "eta", args.bootstrap, args.bootstrap_seed
                )
                category_comparisons[name] = {
                    "wer_diff": wer_diff,
                    "wer_ci": wer_ci,
                    "eta_diff": eta_diff,
                    "eta_ci": eta_ci,
                }
            categories[category] = {
                "n": len(category_ids),
                "control": metric(control, group),
                "seed_mean": metric(mean, group),
                "comparisons": category_comparisons,
            }

    report = {
        "tool": "tools/seed_mean.py",
        "mode": "macro" if args.macro else "pooled",
        "manifests": manifest_rows,
        "rows": len(rows),
        "groups": {lang: len(ids) for lang, ids in groups.items()},
        "control": {
            "path": str(args.control[0]) if len(args.control) == 1 else None,
            "paths": [str(path) for path in args.control],
            "seeds": len(args.control),
            "metrics": systems["control"],
        },
        "seeds": [
            {"path": str(p), "metrics": systems[f"seed_{i}"]} for i, p in enumerate(args.seed)
        ],
        "seed_mean": {"metrics": systems["seed_mean"], "seeds": len(seeds)},
        "comparisons": comparisons,
        "categories": categories,
        "bootstrap": {"resamples": args.bootstrap, "seed": args.bootstrap_seed},
        "convention": "difference = candidate - control; negative WER better, positive ETA better",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name in ["seed_mean", *[f"seed_{i}" for i in range(len(seeds))]]:
        row = comparisons[name]
        print(
            f"{name:10s} WER diff {row['wer_diff'] * 100:+.3f} pp {row['wer_ci']}  "
            f"ETA diff {(row['eta_diff'] or 0) * 100:+.3f} pp {row['eta_ci']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
