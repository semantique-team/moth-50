#!/usr/bin/env python3
"""Measure the real INT8 ONNX size and CPU latency of MOTH-50 candidates.

Builds each candidate with Moth's ``stt.model.MothSTT`` (random weights), exports
it with Moth's ``stt.export.export_fp32`` and quantizes it with
``stt.export.quantize_int8`` (dynamic, per-channel QInt8), i.e. the exact 0.8.1
release recipe. Quantized size does not depend on the weight values, so random
weights give the shipped graph size. Latency is ONNX Runtime CPU, batch 1, on
random log-mel input; it is a relative cost, not a browser measurement.

Needs a ``moth-stt`` checkout (frozen repository, not the published release
folder) and its environment: torch, onnx, onnxruntime.

Usage:
    python tools/probe_export.py --moth-stt ../moth-stt \
        --out runs/probe-2026-11-17/export.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

CANDIDATES = {
    "tiny+2": ("tiny", 2),
    "base+0": ("base", 0),
    "base+2": ("base", 2),
    "base+4": ("base", 4),
    "base+6": ("base", 6),
    "base+7": ("base", 7),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--moth-stt", type=Path, default=Path("../moth-stt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--seconds", type=float, nargs="+", default=[10.0, 30.0])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--reps", type=int, default=7)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.moth_stt.resolve()))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from stt.export import export_fp32, quantize_int8
    from stt.model import EncoderConfig, MothSTT

    encoders = {
        "tiny": EncoderConfig(),
        "base": EncoderConfig(width=512, n_layers=6, n_heads=8, ffn_dim=2048),
    }

    def latency(path: Path, seconds: float, threads: int) -> float:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        mel = (rng.standard_normal((1, 80, int(seconds * 100))) * 0.1).astype(np.float32)
        session.run(None, {"mel": mel})
        times = []
        for _ in range(args.reps):
            start = time.perf_counter()
            session.run(None, {"mel": mel})
            times.append(time.perf_counter() - start)
        return statistics.median(times)

    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for name in args.candidates:
            encoder, extra = CANDIDATES[name]
            torch.manual_seed(0)
            model = MothSTT(
                vocab_size=1024, encoder_config=encoders[encoder], head_blocks=extra
            ).eval()
            params = sum(parameter.numel() for parameter in model.parameters())
            fp32 = Path(scratch) / f"{name}.fp32.onnx"
            int8 = Path(scratch) / f"{name}.int8.onnx"
            export_fp32(model, fp32)
            quantize_int8(fp32, int8)
            graph = onnx.load(str(int8))
            ops: dict[str, int] = {}
            for node in graph.graph.node:
                ops[node.op_type] = ops.get(node.op_type, 0) + 1
            row = {
                "candidate": name,
                "params": params,
                "fp32_bytes": fp32.stat().st_size,
                "int8_bytes": int8.stat().st_size,
                "int8_bytes_per_param": round(int8.stat().st_size / params, 5),
                "quantized_ops": {
                    op: count for op, count in sorted(ops.items()) if "Integer" in op
                },
                "latency_s": {
                    f"{seconds:g}s_{threads}t": round(latency(int8, seconds, threads), 4)
                    for seconds in args.seconds
                    for threads in args.threads
                },
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    report = {
        "tool": "tools/probe_export.py",
        "recipe": "stt.export.export_fp32 + quantize_int8 (dynamic, per-channel QInt8)",
        "weights": "random (size is value-independent)",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "reps": args.reps,
        "candidates": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
