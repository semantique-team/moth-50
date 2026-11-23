"""E-027 TTS rendering: the E-027 script spoken by the pinned permissive voices.

Voices (every file pinned by sha256 in ``data/sources.lock.json``):

| key | engine | voice | modes |
|---|---|---|---|
| sara | Kokoro-82M v1.0 | ``if_sara`` | it, mix |
| nicola | Kokoro-82M v1.0 | ``im_nicola`` | it, mix |
| paola | Piper | ``it_IT-paola-medium`` | it |
| serena | Piper | ``it_IT-serena-medium`` | it |

Mode ``it`` reads the whole sentence with the Italian G2P (espeak-ng ``it``):
English terms come out Italian-accented. Mode ``mix`` (Kokoro only) phonemizes
the English-tagged runs with espeak-ng ``en-us`` and the rest with ``it``, then
speaks the joined phonemes with the same Italian voice; it is skipped for rows
without English. Every clip is peak-normalized to 0.7 and must pass the QC
rules (duration, speaking rate) or is dropped with a counted reason.

    uv run --frozen --project ../moth-stt --extra training \\
      --with 'kokoro>=0.9.4' --with 'misaki[en]>=0.9.4' --with 'piper-tts>=1.5' -- \\
      python tools/e027_tts_render.py --script runs/E-027/tts-script.jsonl \\
      --models data/raw/tts --audio-out data/raw/tts_e027 \\
      --manifest data/manifests/tts_e027.jsonl --report runs/E-027/render-report.json

The paola voice needs Piper 1.5 or later (its model card). The installed
package versions are recorded in the report; the arm is rendered once, with
those versions. ``--limit N`` renders the first N script rows (the smoke
test); ``--resume`` keeps clips that already exist. The engines are imported
only when rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "sources.lock.json"

KOKORO_REPO = "hexgrad/Kokoro-82M"
#: ``sources.lock.json`` id -> folder under ``--models`` (``hf download --local-dir``).
LOCK_DIRS = {"kokoro-82m": "kokoro-82m", "piper-voices-it": "piper"}
KOKORO_DIR = LOCK_DIRS["kokoro-82m"]
PIPER_DIR = LOCK_DIRS["piper-voices-it"] + "/it/it_IT"

PEAK = 0.7
MIN_SECONDS = 0.8
MAX_SECONDS = 30.0
MIN_WORDS_PER_SECOND = 0.8
MAX_WORDS_PER_SECOND = 6.0
#: Kokoro truncates phoneme strings above 510 symbols; refuse instead.
MAX_PHONEMES = 500


@dataclass(frozen=True)
class Voice:
    engine: str
    name: str
    modes: tuple[str, ...]
    files: tuple[str, ...]


def _kokoro(name: str) -> Voice:
    return Voice(
        "kokoro",
        name,
        ("it", "mix"),
        (
            f"{KOKORO_DIR}/kokoro-v1_0.pth",
            f"{KOKORO_DIR}/config.json",
            f"{KOKORO_DIR}/voices/{name}.pt",
        ),
    )


def _piper(speaker: str) -> Voice:
    name = f"it_IT-{speaker}-medium"
    folder = f"{PIPER_DIR}/{speaker}/medium"
    return Voice("piper", name, ("it",), (f"{folder}/{name}.onnx", f"{folder}/{name}.onnx.json"))


VOICES = {
    "sara": _kokoro("if_sara"),
    "nicola": _kokoro("im_nicola"),
    "paola": _piper("paola"),
    "serena": _piper("serena"),
}


def load_pins(lock: Path = LOCK) -> dict[str, str]:
    """Pinned sha256 of every TTS file, keyed by its path under ``--models``."""
    sources = {source["id"]: source for source in json.loads(lock.read_text("utf-8"))["sources"]}
    return {
        f"{folder}/{name}": sha
        for lock_id, folder in LOCK_DIRS.items()
        for name, sha in sources[lock_id]["files"].items()
    }


def runs(segments: list[list[str]], mode: str) -> list[tuple[str, str]]:
    """Consecutive same-language pieces joined: ``[(text, "it"|"en"), ...]``.

    Mode ``it`` is one Italian run over the whole sentence.
    """
    if mode == "it":
        return [(" ".join(piece for piece, _ in segments), "it")]
    if mode != "mix":
        raise ValueError(f"unknown mode {mode!r}")
    joined: list[tuple[str, str]] = []
    for piece, tag in segments:
        if joined and joined[-1][1] == tag:
            joined[-1] = (f"{joined[-1][0]} {piece}", tag)
        else:
            joined.append((piece, tag))
    return joined


def renders(row: dict, voice: Voice) -> list[str]:
    """The modes of ``voice`` that apply to ``row`` (``mix`` needs English)."""
    has_english = any(tag == "en" for _, tag in row["segments"])
    return [mode for mode in voice.modes if mode == "it" or has_english]


def peak_normalize(audio: np.ndarray, peak: float = PEAK) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    top = float(np.max(np.abs(audio))) if audio.size else 0.0
    return audio if top == 0.0 else audio * (peak / top)


def qc_reason(seconds: float, words: int) -> str | None:
    """The QC drop reason for one clip, or ``None`` if it passes."""
    if seconds < MIN_SECONDS:
        return "too_short"
    if seconds > MAX_SECONDS:
        return "too_long"
    rate = words / seconds
    if rate < MIN_WORDS_PER_SECOND:
        return "too_slow"
    if rate > MAX_WORDS_PER_SECOND:
        return "too_fast"
    return None


def clip_id(row_id: str, voice_key: str, mode: str) -> str:
    return f"{row_id}-{voice_key}-{mode}"


def manifest_row(
    row: dict, voice_key: str, voice: Voice, mode: str, audio: str, seconds: float, sha: str
) -> dict:
    lang = row["lang"]
    return {
        "id": clip_id(row["id"], voice_key, mode),
        "audio": audio,
        "text": row["text"],
        "norm": row["norm"],
        "duration": round(seconds, 3),
        "source": "tts_e027",
        "split": "train",
        "speaker": f"tts-{voice_key}",
        "lang": lang,
        "category": row["category"] if lang == "mix" else None,
        "tts": {"engine": voice.engine, "voice": voice.name, "mode": mode},
        "script_id": row["id"],
        "audio_sha256": sha,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pins(models: Path, voices: list[str], pins: dict[str, str]) -> dict[str, str]:
    needed = sorted({name for key in voices for name in VOICES[key].files})
    problems = []
    for name in needed:
        path = models / name
        if name not in pins:
            problems.append(f"no pin for {name} in {LOCK}")
        elif not path.is_file():
            problems.append(f"missing {path}")
        elif file_sha256(path) != pins[name]:
            problems.append(f"sha256 mismatch {path}")
    if problems:
        raise SystemExit("pinned TTS files:\n" + "\n".join(problems))
    return {name: pins[name] for name in needed}


class KokoroEngine:
    sample_rate = 24000

    def __init__(self, models: Path, device: str) -> None:
        from kokoro import KModel, KPipeline
        from misaki import espeak

        root = models / KOKORO_DIR
        model = KModel(
            repo_id=KOKORO_REPO,
            config=str(root / "config.json"),
            model=str(root / "kokoro-v1_0.pth"),
        )
        model = model.to(device).eval()
        self.pipeline = KPipeline(lang_code="i", repo_id=KOKORO_REPO, model=model)
        self.english = espeak.EspeakG2P(language="en-us")
        self.voices = root / "voices"

    def phonemes(self, pieces: list[tuple[str, str]]) -> str:
        parts = []
        for text, tag in pieces:
            g2p = self.english if tag == "en" else self.pipeline.g2p
            phonemes, _ = g2p(text)
            parts.append(phonemes.strip())
        return " ".join(part for part in parts if part)

    def render(self, voice: Voice, pieces: list[tuple[str, str]]) -> tuple[np.ndarray, int]:
        phonemes = self.phonemes(pieces)
        if len(phonemes) > MAX_PHONEMES:
            raise ValueError(f"phonemes_over_{MAX_PHONEMES}")
        voice_path = str(self.voices / f"{voice.name}.pt")
        chunks = []
        for result in self.pipeline.generate_from_tokens(phonemes, voice=voice_path, speed=1.0):
            if result.audio is not None:
                chunks.append(np.asarray(result.audio.detach().cpu().numpy(), dtype=np.float32))
        if not chunks:
            raise ValueError("no_audio")
        return np.concatenate(chunks), self.sample_rate


class PiperEngine:
    def __init__(self, models: Path) -> None:
        from piper import PiperVoice

        self._load = PiperVoice.load
        self.models = models
        self.loaded: dict[str, object] = {}

    def render(self, voice: Voice, pieces: list[tuple[str, str]]) -> tuple[np.ndarray, int]:
        if voice.name not in self.loaded:
            onnx, config = (self.models / name for name in voice.files)
            self.loaded[voice.name] = self._load(str(onnx), config_path=str(config))
        text = " ".join(text for text, _ in pieces)
        chunks = list(self.loaded[voice.name].synthesize(text))
        if not chunks:
            raise ValueError("no_audio")
        audio = np.concatenate([np.asarray(c.audio_float_array, np.float32) for c in chunks])
        return audio, int(chunks[0].sample_rate)


def _versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for package in ("kokoro", "misaki", "piper-tts", "onnxruntime", "torch", "soundfile"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True, help="dir with the pinned files")
    parser.add_argument("--audio-out", type=Path, required=True, help="audio root for the WAVs")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--voice", action="append", choices=sorted(VOICES), default=None)
    parser.add_argument("--limit", type=int, default=0, help="first N script rows (smoke)")
    parser.add_argument("--device", default="cpu", help="Kokoro torch device")
    parser.add_argument("--resume", action="store_true", help="keep WAVs that already exist")
    args = parser.parse_args(argv)

    import soundfile

    voice_keys = args.voice or sorted(VOICES)
    pins = verify_pins(args.models, voice_keys, load_pins())
    rows = [
        json.loads(line)
        for line in args.script.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    engines: dict[str, KokoroEngine | PiperEngine] = {}
    if any(VOICES[key].engine == "kokoro" for key in voice_keys):
        engines["kokoro"] = KokoroEngine(args.models, args.device)
    if any(VOICES[key].engine == "piper" for key in voice_keys):
        engines["piper"] = PiperEngine(args.models)

    started = time.time()
    manifest: list[dict] = []
    dropped: Counter[str] = Counter()
    seconds_by: defaultdict[str, float] = defaultdict(float)
    clips_by: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        words = len(row["norm"].split())
        for key in voice_keys:
            voice = VOICES[key]
            for mode in renders(row, voice):
                stratum = f"{key}-{mode}"
                relative = f"{key}/{mode}/{clip_id(row['id'], key, mode)}.wav"
                path = args.audio_out / relative
                if args.resume and path.is_file():
                    info = soundfile.info(str(path))
                    seconds = info.frames / info.samplerate
                else:
                    pieces = runs(row["segments"], mode)
                    try:
                        audio, rate = engines[voice.engine].render(voice, pieces)
                    except ValueError as error:
                        dropped[f"{stratum}:{error}"] += 1
                        continue
                    audio = peak_normalize(audio)
                    seconds = audio.size / rate
                    reason = qc_reason(seconds, words)
                    if reason:
                        dropped[f"{stratum}:{reason}"] += 1
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    soundfile.write(str(path), audio, rate, subtype="PCM_16")
                manifest.append(
                    manifest_row(row, key, voice, mode, relative, seconds, file_sha256(path))
                )
                seconds_by[stratum] += seconds
                clips_by[stratum] += 1
        if index % 100 == 0:
            print(f"{index}/{len(rows)} script rows, {len(manifest)} clips", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )
    report = {
        "tool": "tools/e027_tts_render.py",
        "script": str(args.script),
        "script_sha256": file_sha256(args.script),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "audio_root": str(args.audio_out),
        "voices": {key: asdict(VOICES[key]) for key in voice_keys},
        "pins": pins,
        "versions": _versions(),
        "script_rows": len(rows),
        "clips": len(manifest),
        "hours": round(sum(seconds_by.values()) / 3600, 3),
        "by_stratum": {
            key: {"clips": clips_by[key], "hours": round(seconds_by[key] / 3600, 3)}
            for key in sorted(clips_by)
        },
        "dropped": dict(sorted(dropped.items())),
        "qc": {
            "peak": PEAK,
            "seconds": [MIN_SECONDS, MAX_SECONDS],
            "words_per_second": [MIN_WORDS_PER_SECOND, MAX_WORDS_PER_SECOND],
            "max_phonemes": MAX_PHONEMES,
        },
        "wall_seconds": round(time.time() - started, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in ("clips", "hours", "by_stratum", "dropped")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
