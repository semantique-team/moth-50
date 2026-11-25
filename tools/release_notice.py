"""The MOTH-50 release NOTICE.

E-005 shipped Moth 0.8.1's NOTICE byte for byte (sha256 ``6ca7672f…``): it names
``openai/whisper-tiny`` as the encoder and lists only Moth's sources. This module
writes the notice for what a MOTH-50 package actually contains: the Whisper
encoder of the run, the extra blocks, the decoder LM and the reused Moth
tokenizer, the VAD file, and the MOTH-50 training sources pinned in
``data/sources.lock.json``.

The LM paragraph depends on the LM (D37): Moth 0.8.1's A3 LM is reused
unchanged, while E-034c and any other non-A3 LM were built by MOTH-50 and must
not be described as Moth's file.
"""

from __future__ import annotations

from collections.abc import Iterable

from release_assets import A3_LM

#: MOTH-50 training sources, by ``data/sources.lock.json`` id.
TRAINING_SOURCES = ("moth-cs", "common-voice-it", "mls-italian", "librispeech", "musan")

#: Unpacked (MOTHLMB1) sha256 of the E-034c product LM (D25).
E034C_LM_SHA256 = "4b122fbed685c3e4be14cad63e26ebe47fc3d311f379faa850bfbf29574ad5b6"

_TOKENIZER_LINES = (
    "The SentencePiece tokenizer behind ``vocab.json`` is reused unchanged from Moth",
    "0.8.1 (semantique-team/moth-stt, Apache-2.0).",
)


def lm_lines(lm_sha256: str, lm_file_sha256: str | None = None) -> list[str]:
    """The NOTICE paragraph for the decoder LM and the tokenizer.

    ``lm_sha256`` is the source LM file; ``lm_file_sha256`` the shipped
    ``lm_bpe3.bin`` when it was packed (MOTHLMB2). The A3 wording is kept byte for
    byte so the historical A3 packages re-export unchanged.
    """
    if lm_sha256 == A3_LM.sha256:
        return [
            f"The decoder LM (``lm_bpe3.bin``, sha256 {lm_sha256}) and the SentencePiece",
            "tokenizer behind ``vocab.json`` are reused unchanged from Moth 0.8.1",
            "(semantique-team/moth-stt, Apache-2.0). Moth built them from these sources:",
        ]
    if lm_sha256 == E034C_LM_SHA256:
        origin = [
            "The decoder LM ``lm_bpe3.bin`` is MOTH-50's E-034c LM: Moth's E-034 BPE",
            "trigram recipe, rebuilt by MOTH-50 from Moth's frozen E-031 text corpus",
            "without the 900 sentences shared with the S1-dev evaluation set.",
        ]
        sources = "Moth built the tokenizer and that text corpus from these sources:"
    else:
        origin = [
            "The decoder LM ``lm_bpe3.bin`` was built by MOTH-50 and is not part of the",
            "Moth 0.8.1 release; its text corpus is documented with the run that built it.",
        ]
        sources = "Moth built the tokenizer from these sources:"
    if lm_file_sha256 and lm_file_sha256 != lm_sha256:
        hashes = [
            f"Shipped file sha256 {lm_file_sha256},",
            f"packed losslessly (MOTHLMB2) from sha256 {lm_sha256}.",
        ]
    else:
        hashes = [f"File sha256 {lm_sha256}."]
    return [*origin, *hashes, *_TOKENIZER_LINES, sources]


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
    lm_file_sha256: str | None = None,
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
            *lm_lines(lm_sha256, lm_file_sha256),
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
