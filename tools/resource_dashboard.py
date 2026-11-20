#!/usr/bin/env python3
"""Render a self-contained resource dashboard for the E-010/E-011 runs.

Reads the trainer's ``train.log.jsonl`` (MPS memory, step time, throughput) and
the CPU/RAM samples written by ``tools/sample_resources.py``. The output is one
HTML file with inline SVG: no external assets and no plotting dependency.

    python tools/resource_dashboard.py --out runs/E-011/resource-dashboard.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path

COLORS = {
    "base+4": "#2563eb",
    "base+6": "#7c3aed",
    "seed 0": "#0f766e",
    "seed 1": "#b45309",
    "MPS": "#0f766e",
    "system": "#2563eb",
    "process": "#b45309",
    "swap": "#b91c1c",
    "wired": "#64748b",
    "compressed": "#a855f7",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def rolling_median(values: list[float], window: int = 9) -> list[float]:
    out = []
    for index in range(len(values)):
        start = max(0, index - window // 2)
        end = min(len(values), index + window // 2 + 1)
        out.append(statistics.median(values[start:end]))
    return out


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if high <= low:
        high = low + 1
    raw = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiplier in (1, 2, 2.5, 5, 10):
        step = multiplier * magnitude
        if raw <= step:
            break
    first = math.ceil(low / step) * step
    ticks = []
    value = first
    while value <= high + 1e-9:
        ticks.append(value)
        value += step
    return ticks


def fmt_tick(value: float, decimals: int) -> str:
    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def line_chart(
    series: list[dict],
    *,
    title: str,
    x_label: str,
    y_label: str,
    width: int = 860,
    height: int = 300,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    x_decimals: int = 0,
    y_decimals: int = 1,
    note: str | None = None,
) -> str:
    left, right, top, bottom = 68, 18, 40, 48
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = [point for item in series for point in item["points"]]
    if not points:
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            f'font-size="13" fill="#94a3b8">nessun campione ancora</text></svg>'
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if x_range:
        x_min, x_max = x_range
    else:
        x_min, x_max = min(xs), max(xs)
        if x_max == x_min:
            x_max = x_min + 1
    if y_range:
        y_min, y_max = y_range
    else:
        y_min, y_max = min(ys), max(ys)
        pad = max((y_max - y_min) * 0.08, 1e-6)
        y_min, y_max = y_min - pad, y_max + pad
        y_min = max(0, y_min)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (1 - (value - y_min) / (y_max - y_min)) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    parts.append(
        f'<text x="{left}" y="22" font-size="14" font-weight="600" fill="#0f172a">'
        f"{html.escape(title)}</text>"
    )
    for tick in nice_ticks(y_min, y_max, 5):
        y = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="10.5" '
            f'fill="#64748b">{fmt_tick(tick, y_decimals)}</text>'
        )
    for tick in nice_ticks(x_min, x_max, 6):
        x = sx(tick)
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 19}" text-anchor="middle" font-size="10.5" '
            f'fill="#64748b">{fmt_tick(tick, x_decimals)}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        'stroke="#94a3b8" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{left + plot_w}" y="{height - 8}" text-anchor="end" font-size="10.5" '
        f'fill="#64748b">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="14" y="{top + 6}" font-size="10.5" fill="#64748b" '
        f'transform="rotate(-90 14 {top + 6})" text-anchor="end">{html.escape(y_label)}</text>'
    )
    for item in series:
        color = item.get("color", "#0f766e")
        width_px = item.get("width", 2.0)
        opacity = item.get("opacity", 1.0)
        path = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in item["points"])
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="{width_px}" '
            f'stroke-opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round" '
            f'points="{path}"/>'
        )
    legend_x = left
    for item in series:
        color = item.get("color", "#0f766e")
        parts.append(
            f'<line x1="{legend_x}" y1="34" x2="{legend_x + 16}" y2="34" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x + 21}" y="38" font-size="10.5" fill="#334155">'
            f"{html.escape(item['name'])}</text>"
        )
        legend_x += 30 + 6.2 * len(item["name"])
    if note:
        parts.append(
            f'<text x="{left + plot_w}" y="34" text-anchor="end" font-size="10" '
            f'fill="#94a3b8">{html.escape(note)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resource_dashboard.py", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/E-011/resource-dashboard.html"))
    args = parser.parse_args(argv)

    e010 = {
        arm: read_jsonl(Path(f"runs/E-010/{arm}/train.log.jsonl"))
        for arm in ("base+4-enc1e-4", "base+6-enc1e-4")
    }
    e011 = {
        path.parent.name: read_jsonl(path)
        for path in sorted(Path("runs/E-011").glob("*/train.log.jsonl"))
    }
    samples = read_jsonl(Path("runs/E-011/resource-samples.jsonl"))

    e011_series = []
    for name, rows in e011.items():
        label = "seed 0" if name.endswith("s0") else ("seed 1" if name.endswith("s1") else name)
        e011_series.append(
            {
                "name": label,
                "color": COLORS.get(label, "#0f766e"),
                "points": [(row["step"], row["memory_gb"]) for row in rows if "memory_gb" in row],
            }
        )

    e011_steps = [(row["step"], row["step_seconds"]) for row in e011.get("base+6-e011-s0", [])]
    e011_step_series = []
    if e011_steps:
        xs = [point[0] for point in e011_steps]
        ys = [point[1] for point in e011_steps]
        smooth = rolling_median(ys, 11)
        e011_step_series = [
            {
                "name": "step time",
                "color": "#94a3b8",
                "width": 1.0,
                "opacity": 0.55,
                "points": list(zip(xs, ys)),
            },
            {
                "name": "mediana mobile (11)",
                "color": "#b45309",
                "width": 2.2,
                "points": list(zip(xs, smooth)),
            },
        ]

    compare_series = []
    for arm, label in (("base+4-enc1e-4", "E-010 base+4"), ("base+6-enc1e-4", "E-010 base+6")):
        rows = e010[arm]
        compare_series.append(
            {
                "name": label,
                "color": COLORS[label.split()[-1]],
                "points": [
                    (row["step"] / 4000 * 100, row["memory_gb"])
                    for row in rows
                    if "memory_gb" in row
                ],
            }
        )
    for name, rows in e011.items():
        if not rows:
            continue
        label = "E-011 seed 0" if name.endswith("s0") else "E-011 seed 1"
        color_key = "seed 0" if name.endswith("s0") else "seed 1"
        compare_series.append(
            {
                "name": label,
                "color": COLORS.get(color_key, "#0f766e"),
                "points": [
                    (row["step"] / 40000 * 100, row["memory_gb"])
                    for row in rows
                    if "memory_gb" in row
                ],
            }
        )

    cpu_series = []
    ram_series = []
    if samples:
        epochs = [row["epoch"] for row in samples]
        start = min(epochs)
        cpu_series = [
            {
                "name": "CPU sistema (user+sys)",
                "color": COLORS["system"],
                "points": [
                    (
                        (row["epoch"] - start) / 60,
                        row.get("sys_user_pct", 0) + row.get("sys_system_pct", 0),
                    )
                    for row in samples
                ],
            },
            {
                "name": "CPU processo training",
                "color": COLORS["process"],
                "points": [
                    ((row["epoch"] - start) / 60, row.get("proc_cpu_pct") or 0) for row in samples
                ],
            },
        ]
        ram_series = [
            {
                "name": "RAM sistema usata",
                "color": COLORS["system"],
                "points": [
                    ((row["epoch"] - start) / 60, row["mem_used_bytes"] / 1e9) for row in samples
                ],
            },
            {
                "name": "RSS processo",
                "color": COLORS["process"],
                "points": [
                    ((row["epoch"] - start) / 60, (row.get("proc_rss_bytes") or 0) / 1e9)
                    for row in samples
                ],
            },
            {
                "name": "memoria MPS",
                "color": COLORS["MPS"],
                "points": [
                    ((row["epoch"] - start) / 60, row.get("mps_gb") or 0) for row in samples
                ],
            },
            {
                "name": "swap usato",
                "color": COLORS["swap"],
                "points": [
                    ((row["epoch"] - start) / 60, (row.get("swap_used_bytes") or 0) / 1e9)
                    for row in samples
                ],
            },
        ]

    latest = samples[-1] if samples else {}
    latest_log = e011.get("base+6-e011-s0", [])
    latest_row = latest_log[-1] if latest_log else {}
    step = latest_row.get("step")
    peak = max((row["memory_gb"] for row in e011.get("base+6-e011-s0", [])), default=0)
    avg_step = (
        statistics.mean(row["step_seconds"] for row in latest_log[-200:]) if latest_log else 0
    )
    dev_wer = None
    dev_files = sorted(Path("runs/E-011/base+6-e011-s0").glob("dev-*.json"))
    if dev_files:
        dev_wer = json.loads(dev_files[-1].read_text(encoding="utf-8"))["wer"] * 100
    elapsed_min = (latest.get("epoch", 0) - (samples[0]["epoch"] if samples else 0)) / 60

    def card(label: str, value: str, hint: str = "") -> str:
        return (
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div>'
            f'<div class="hint">{html.escape(hint)}</div></div>'
        )

    cards = [
        card("Step", f"{step:,}" if step else "—", "di 40.000" if step else ""),
        card("Greedy dev", f"{dev_wer:.2f} %" if dev_wer is not None else "—", "ultima eval"),
        card("MPS memoria", f"{latest_row.get('memory_gb', 0):.2f} GB", f"picco {peak:.2f} GB"),
        card("Step time", f"{avg_step:.3f} s", "media ultimi 200 log"),
        card(
            "CPU sistema",
            f"{latest.get('sys_user_pct', 0) + latest.get('sys_system_pct', 0):.0f} %",
            "user+sys",
        ),
        card("RAM usata", f"{latest.get('mem_used_bytes', 0) / 1e9:.1f} GB", "di 16 GB"),
        card("RSS training", f"{(latest.get('proc_rss_bytes') or 0) / 1e9:.2f} GB", "processo"),
        card("Swap", f"{(latest.get('swap_used_bytes') or 0) / 1e9:.2f} GB", "di 4.1 GB"),
    ]

    first_sample = samples[0]["ts"] if samples else "—"
    last_sample = samples[-1]["ts"] if samples else "—"
    charts = [
        line_chart(
            e011_series,
            title="E-011 — memoria MPS per step",
            x_label="step",
            y_label="GB",
            x_range=(0, 40000),
            note="train.log.jsonl",
        ),
        line_chart(
            e011_step_series,
            title="E-011 — tempo per step (rumore e mediana mobile)",
            x_label="step",
            y_label="secondi",
            x_range=(0, 40000),
            note="train.log.jsonl",
        ),
        line_chart(
            compare_series,
            title="Memoria MPS — confronto a pari progresso",
            x_label="% del run pianificato",
            y_label="GB",
            x_range=(0, 100),
            note="E-010 4k / E-011 40k",
        ),
        line_chart(
            cpu_series,
            title="CPU — campionamento live",
            x_label="minuti dall'inizio del campionamento",
            y_label="% CPU",
            y_range=(0, 100),
            x_decimals=1,
            note=f"{first_sample} → {last_sample}",
        ),
        line_chart(
            ram_series,
            title="RAM e memoria unificata — campionamento live",
            x_label="minuti dall'inizio del campionamento",
            y_label="GB",
            x_decimals=1,
            note=f"{first_sample} → {last_sample}",
        ),
    ]

    page = f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60">
<title>MOTH-50 — risorse E-010/E-011</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 28px; background: #f8fafc; color: #0f172a;
         font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; letter-spacing: -0.02em; }}
  .sub {{ color: #64748b; margin: 0 0 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px; margin-bottom: 20px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 10px 12px; }}
  .label {{ color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }}
  .value {{ font-size: 19px; font-weight: 650; margin-top: 2px; }}
  .hint {{ color: #94a3b8; font-size: 10.5px; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 8px 10px; margin-bottom: 14px; overflow: hidden; }}
  svg {{ width: 100%; height: auto; display: block; }}
  .note {{ color: #64748b; font-size: 12px; margin-top: 8px; }}
  code {{ background: #eef2f7; padding: 1px 5px; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>MOTH-50 — risorse dei training</h1>
  <p class="sub">E-010 (base+4 / base+6, 4.000 step) ed E-011 (base+6, 40.000 step, seed 0 in corso).
  Memoria e step time dai log del trainer; CPU/RAM dal campionatore live.</p>
  <div class="cards">{"".join(cards)}</div>
  {"".join(f'<div class="chart">{chart}</div>' for chart in charts)}
  <p class="note">CPU/RAM: macOS non conserva lo storico, quindi il grafico parte dal primo campione
  (<code>{html.escape(first_sample)}</code>, intervallo 15 s). Il campionatore continua fino a 30 minuti
  dopo la fine del training. La "RAM sistema usata" è active + wired + compressed di <code>vm_stat</code>;
  "memoria MPS" è <code>torch.mps.driver_allocated_memory()</code> dal log del trainer.
  Campioni live: {len(samples)} ({elapsed_min:.0f} min). La pagina si aggiorna
  automaticamente ogni 60 s dal file locale.</p>
</body>
</html>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(args.out)
    print(f"wrote {args.out} ({len(page) / 1024:.0f} KiB, {len(samples)} live samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
