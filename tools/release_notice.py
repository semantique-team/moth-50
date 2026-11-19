"""The MOTH-50 release NOTICE.

E-005 shipped Moth 0.8.1's NOTICE byte for byte (sha256 ``6ca7672f…``): it names
``openai/whisper-tiny`` as the encoder and lists only Moth's sources. This module
writes the notice for what a MOTH-50 package actually contains: the Whisper
encoder of the run, the extra blocks, the reused Moth LM and tokenizer, the VAD
file, and the MOTH-50 training sources pinned in ``data/sources.lock.json``.
"""

from __future__ import annotations

from collections.abc import Iterable

#: MOTH-50 training sources, by ``data/sources.lock.json`` id.
TRAINING_SOURCES = ("moth-cs", "common-voice-it", "mls-italian", "librispeech", "musan")


def moth_dataset_lines(sources: Iterable) -> list[str]:
    """Moth's attribution lines, built exactly as ``stt.export.build_notice`` builds them."""
    return list(
        dict.fromkeys(
            f"- {source.citation or source.name} — {source.license} — {source.url}"
            for source in sources
            if not source.license.lower().startswith("verify")
        )
    )


def _lock_line(source: dict) -> str:
    version = f" {source['version']}" if source.get("version") else ""
    return f"- {source['id']}{version} — {source['license']} — {source['url']}"


def compose_notice(
    *,
    whisper_mit: str,
    silero_mit: str,
    lock: dict,
    moth_lines: list[str],
    encoder_preset: str,
    head_blocks: int,
    lm_sha256: str,
    vad_description: str,
) -> str:
    sources = {source["id"]: source for source in lock["sources"]}
    missing = [key for key in TRAINING_SOURCES if key not in sources]
    if missing:
        raise KeyError(f"data/sources.lock.json lacks training sources {missing}")
    encoder = f"openai/whisper-{encoder_preset}"
    whisper = sources.get(f"whisper-{encoder_preset}")
    revision = f", Hugging Face revision {whisper['revision']}" if whisper else ""
    silero = sources.get("silero-vad")
    silero_revision = f", commit {silero['revision'][:7]}" if silero else ""
    blocks = f"{head_blocks} additional transformer block{'' if head_blocks == 1 else 's'}"
    encoder_line = (
        f"The encoder of this model is derived from OpenAI's Whisper (``{encoder}``{revision});"
    )
    vad_line = (
        f"The no-speech guard is Silero VAD (``snakers4/silero-vad``{silero_revision}; "
        f"{vad_description}),"
    )
    return "\n".join(
        [
            "MOTH-50 — third-party notices and attributions",
            "",
            encoder_line,
            f"it is fine-tuned with {blocks} and a CTC head. Whisper is distributed",
            "under the MIT License; the notice follows verbatim.",
            "",
            whisper_mit,
            "",
            vad_line,
            "shipped unmodified as ``silero_vad.onnx``. Silero VAD is distributed under the",
            "MIT License; the notice follows verbatim.",
            "",
            silero_mit,
            "",
            f"The decoder LM (``lm_bpe3.bin``, sha256 {lm_sha256}) and the SentencePiece",
            "tokenizer behind ``vocab.json`` are reused unchanged from Moth 0.8.1",
            "(semantique-team/moth-stt, Apache-2.0). Moth built them from these sources:",
            *moth_lines,
            "",
            "Datasets used to train this model:",
            *(_lock_line(sources[key]) for key in TRAINING_SOURCES),
            "- utterance-level IT+EN splices derived from the Common Voice, MLS and",
            "  LibriSpeech training audio above",
            "",
            "Evaluation-only sets (never used for training) are listed in the model card.",
            "",
            "The model repository itself is released under Apache-2.0 (see LICENSE).",
            "",
        ]
    )
