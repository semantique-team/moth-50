"""Serve a MOTH-50 release with the vendored web runtime.

The upstream page loads ``../release/`` relative to ``/web/index.html``.  This
tool builds a temporary root with ``web/`` pointing at
``vendor/moth-stt-web/`` and ``release/`` pointing at the requested package, so
the vendored runtime never has to be copied next to a release.

Examples::

    npm ci --prefix vendor/moth-stt-web
    python tools/serve_web.py --release runs/E-023/release-0.9.7
    python tools/serve_web.py --release runs/E-023/release-0.9.7 --no-isolation

The default server sets COOP/COEP, which enables WASM threads and a
multi-threaded session.  ``--no-isolation`` uses the same root but the stdlib
server without those headers, reproducing the 1-thread E-016 measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "vendor" / "moth-stt-web"
DEFAULT_RELEASE = ROOT / "runs" / "E-023" / "release-0.9.7"


def _check(release: Path) -> int:
    config_path = release / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"no release config at {config_path}")
    if not (WEB / "node_modules" / "onnxruntime-web" / "dist").is_dir():
        raise SystemExit(
            "the vendored runtime has no npm dependencies; run "
            "`npm ci --prefix vendor/moth-stt-web` first"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    files = ["NOTICE", "config.json", *config.get("files", {})]
    total = sum((release / name).stat().st_size for name in files if (release / name).is_file())
    print(f"release {config.get('version', '?')}: {release} ({total:,} B)")
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--release",
        type=Path,
        default=DEFAULT_RELEASE,
        help=f"release directory (default: {DEFAULT_RELEASE.relative_to(ROOT)})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="omit COOP/COEP so the page runs the 1-thread WASM fallback",
    )
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="serve an extra directory at /NAME (repeatable), e.g. dataset=../moth-cs-dataset",
    )
    args = parser.parse_args(argv)

    release = args.release.resolve()
    _check(release)
    mounts: list[tuple[str, Path]] = []
    for item in args.mount:
        name, separator, value = item.partition("=")
        if not separator or not name or "/" in name or name in {"web", "release"}:
            raise SystemExit(f"invalid --mount {item!r}; expected NAME=PATH")
        source = Path(value).resolve()
        if not source.is_dir():
            raise SystemExit(f"mount {name}: no directory at {source}")
        mounts.append((name, source))
    with tempfile.TemporaryDirectory(prefix="moth-web-") as temp:
        root = Path(temp)
        (root / "web").symlink_to(WEB, target_is_directory=True)
        (root / "release").symlink_to(release, target_is_directory=True)
        for name, source in mounts:
            (root / name).symlink_to(source, target_is_directory=True)
            print(f"mount /{name} -> {source}")
        if args.no_isolation:
            command = [
                sys.executable,
                "-m",
                "http.server",
                str(args.port),
                "--bind",
                args.host,
                "--directory",
                str(root),
            ]
        else:
            command = [
                sys.executable,
                str(WEB / "server.py"),
                "--root",
                str(root),
                "--host",
                args.host,
                "--port",
                str(args.port),
            ]
        print(
            f"serving http://{args.host}:{args.port}/web/index.html"
            f"  ({'no isolation, 1 thread' if args.no_isolation else 'COOP/COEP, threads'})"
        )
        try:
            return subprocess.run(command, check=False).returncode
        except KeyboardInterrupt:
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
