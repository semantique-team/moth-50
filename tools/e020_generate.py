"""Generate in-domain mixed IT–EN dictation text for E-020.

The generator is a pinned open-weight LLM; prompts and seeds are committed in
``experiments/E-025-prompts.json``. Seeds come only from training-side terms
(gold train + ``data/en_terms.txt``), never from S1-dev, T or A. Output is
filtered separately by ``tools/e020_filter.py`` before it enters an LM corpus.

    uv run --frozen --project ../moth-stt --extra training -- \\
      python tools/e020_generate.py --model-dir /path/to/Qwen2.5-1.5B-Instruct \\
      --out runs/E-025/generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MOTH = ROOT.parent / "moth-stt"
sys.path.insert(0, str(MOTH))

from stt.text import normalize

PROMPTS = ROOT / "experiments" / "E-025-prompts.json"
TRAIN = ROOT.parent / "moth-cs-dataset" / "manifests" / "train.jsonl"
TERMS = MOTH / "data" / "en_terms.txt"


def _clean(text: str) -> str:
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    text = re.sub(r"^[\s\-*0-9.)]+", "", text)
    text = text.strip("\"'“”‘’ ")
    return " ".join(text.split())


def _seed_pools() -> dict[str, list[str]]:
    rows = [
        json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    pools: dict[str, set[str]] = {
        key: set()
        for key in (
            "brand",
            "acronym",
            "switch",
            "italianized",
            "single_word",
            "phrase",
            "control_en",
        )
    }
    for row in rows:
        words = row.get("norm", "").split()
        terms = []
        for start, end in row.get("en_spans") or []:
            terms.extend(normalize(term) for term in words[start:end])
        category = row.get("category")
        if category in pools:
            pools[category].update(term for term in terms if term and " " not in term)
    listed = [
        normalize(line)
        for line in TERMS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    listed = [term for term in listed if term and " " not in term]
    pools["control_en"].update(listed)
    for key in ("single_word", "phrase", "switch", "italianized"):
        pools[key].update(listed)
    for key in ("brand", "acronym"):
        if not pools[key]:
            pools[key].update(listed)
    return {key: sorted(values) for key, values in pools.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = parser.parse_args(argv)

    spec = json.loads(args.prompts.read_text(encoding="utf-8"))
    count = int(spec["count"])
    if args.limit is not None:
        count = min(count, args.limit)
    seed = int(spec.get("seed", 0))
    rng = random.Random(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), local_files_only=True, torch_dtype=torch.bfloat16
    )
    model.eval()
    model.to(args.device)
    device = torch.device(args.device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    pools = _seed_pools()
    categories = spec["categories"]
    names = list(categories)
    weights = [int(categories[name]["weight"]) for name in names]
    allocation = {
        name: max(1, round(count * weight / sum(weights))) for name, weight in zip(names, weights)
    }
    while sum(allocation.values()) > count:
        name = max(allocation, key=allocation.get)
        allocation[name] -= 1
    while sum(allocation.values()) < count:
        name = min(allocation, key=allocation.get)
        allocation[name] += 1

    tasks = []
    for name in names:
        instruction = categories[name]["instruction"]
        examples = categories[name].get("examples") or []
        for _ in range(allocation[name]):
            if name in ("control_it", "control_en"):
                terms = []
                user = instruction
            else:
                pool = pools[name] or pools["single_word"]
                terms = rng.sample(pool, min(4, len(pool)))
                user = instruction.format(terms=", ".join(terms))
            if examples:
                user = f"{user}\nEsempi da imitare:\n" + "\n".join(f"- {item}" for item in examples)
            tasks.append((name, terms, user))
    rng.shuffle(tasks)

    rows = []
    started = time.time()
    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": spec["system"]},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _, _, user in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(spec["decoding"]["max_new_tokens"]),
                do_sample=True,
                temperature=float(spec["decoding"]["temperature"]),
                top_p=float(spec["decoding"]["top_p"]),
                pad_token_id=tokenizer.pad_token_id,
            )
        for index, (name, terms, user) in enumerate(batch):
            text = tokenizer.decode(
                generated[index][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            text = _clean(text)
            if not text or "```" in text or len(text) > 400:
                continue
            rows.append(
                {
                    "id": f"E025-{len(rows) + 1:05d}",
                    "text": text,
                    "category": name,
                    "seed_terms": terms,
                    "prompt": user,
                }
            )
        if len(rows) % 100 == 0 or start + args.batch_size >= len(tasks):
            print(f"{len(rows)}/{len(tasks)} generated", flush=True)
    wall = time.time() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "tool": "tools/e020_generate.py",
        "model": spec["model"],
        "revision": spec["revision"],
        "prompts": str(args.prompts),
        "count_requested": count,
        "count_written": len(rows),
        "allocation": allocation,
        "device": args.device,
        "wall_seconds": round(wall, 1),
        "output": str(args.out),
    }
    args.out.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
