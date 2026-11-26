"""Publish the verified MOTH-50 release to a Hugging Face model repository.

Uploads the *verified* release folder exactly as ``tools/export_release.py`` built it, plus the
model card as ``README.md`` and the NOTICE. Refuses to run when the release does not match its own
``config.json`` hashes or the model card documents another version, so a stale or hand-edited
release can never be published.

Dry run by default: it prints the plan and the file hashes without touching the network.

    # 1. check what would be uploaded
    uv run --frozen -- python tools/upload_hf.py --repo semantique-team/moth-50

    # 2. publish (needs a token with write access)
    HF_TOKEN=hf_... uv run --frozen -- python tools/upload_hf.py \
      --repo semantique-team/moth-50 --push

Token: create a *write* token at https://huggingface.co/settings/tokens. Use ``--private`` for a
staged repo; switch it public later from the model page. The token is read from the environment
only — never from a file in the repository.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "vendor" / "moth-stt"))

from moth_release import MAX_BUDGET_BYTES, validate_release

#: The package this repository currently publishes. The card moves with the release it documents.
RELEASE = REPO / "runs/E-023/release-0.9.8"
CARD = REPO / "MODEL_CARD.md"

#: Files the runtime downloads. ``config.json`` and ``NOTICE`` travel with them.
ALSO = ("config.json", "NOTICE")


def payload(config: dict) -> tuple[str, ...]:
    """The five runtime assets named by the release config, in a stable order."""
    return (
        config["model"]["file"],
        config["tokenizer"]["vocab"],
        config["features"]["mel_filters"],
        config["vad"]["file"],
        config["decode"]["lm"],
    )


def check(release: Path = RELEASE, card: Path = CARD) -> tuple[dict, dict, list[str]]:
    """Verify a release folder and its model card; return the config, report and any problems."""
    try:
        config, report = validate_release(release, MAX_BUDGET_BYTES)
    except (OSError, ValueError) as exc:
        return {}, {}, [str(exc)]

    problems: list[str] = []
    if not card.is_file():
        problems.append(f"missing model card: {card}")
    else:
        text = card.read_text(encoding="utf-8")
        heading = next((line for line in text.splitlines() if line.startswith("# ")), "")
        # A card that documents a different release is worse than no card: it publishes true
        # numbers for the wrong artifact.
        if config["version"] not in heading:
            problems.append(
                f"model card is for a different release: its title is {heading.strip()!r}, "
                f"expected it to name {config['version']}"
            )
    return config, report, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="target repo, e.g. semantique-team/moth-50")
    parser.add_argument("--push", action="store_true", help="actually upload (default: dry run)")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--release", default=str(RELEASE))
    parser.add_argument("--card", default=str(CARD))
    args = parser.parse_args(argv)

    release, card = Path(args.release), Path(args.card)
    config, report, problems = check(release, card)
    if problems:
        print("refusing to publish — the release folder is not the verified one:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nrebuild it first with tools/export_release.py, then re-run this guard.",
            file=sys.stderr,
        )
        return 2

    names = (*payload(config), *ALSO)
    total = report["total_bytes"]
    print(f"verified release {config['version']} — {total} bytes ({total / 1e6:.2f} MB)")
    for name in names:
        entry = report["files"][name]
        print(f"  {name:<20} {entry['bytes']:>10}  sha256 {entry['sha256'][:16]}…")
    print(f"  README.md (from {card})")
    print(
        f"\ntarget: https://huggingface.co/{args.repo}  ({'private' if args.private else 'public'})"
    )

    if not args.push:
        print("\ndry run: nothing uploaded. Re-run with --push and HF_TOKEN set.")
        return 0

    # Three ways in, in order: the environment, then the token `hf auth login` stored (which keeps
    # the secret out of shell history and out of any chat), then refuse.
    import os

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
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"model card: MOTH-50 {config['version']}",
    )
    for name in names:
        api.upload_file(
            path_or_fileobj=str(release / name),
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"release {config['version']}: {name}",
        )
        print(f"  uploaded {name}", flush=True)
    print(f"\ndone: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
