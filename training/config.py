"""Configuration for the MOTH-50 trainer.

The trainer is deliberately separate from Moth's ``stt/train.py``: it reuses
Moth's model, tokenizer, mel cache, collate, batching and evaluator, but it owns
the loop, the encoder configuration, activation checkpointing and the 50 MB
accounting. Configuration is TOML (stdlib ``tomllib``), so comments are allowed
without an extra dependency.

Example:
    [model]
    preset = "base"
    head_blocks = 6
    checkpointing = true

    [data]
    tokenizer = "data/tokenizer/moth_bpe.model"
    max_seconds = 192
    max_clips = 32
    augment = true
    [[data.train]]
    manifest = "../moth-stt/data/manifests/mls_it_train.jsonl"
    shards = "data/shards/mls_it_train"

    [optimizer]
    lr_head = 1e-4
    lr_encoder = 1e-5

    [train]
    epochs = 4
    warmup_steps = 1500
    ...
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

#: Encoder presets. ``base`` is the MOTH-50 primary; ``tiny`` exists for
#: regression checks against Moth 0.8.1.
ENCODER_PRESETS = {
    "tiny": {"width": 384, "n_layers": 4, "n_heads": 6, "ffn_dim": 1536},
    "base": {"width": 512, "n_layers": 6, "n_heads": 8, "ffn_dim": 2048},
}

#: A fully explicit encoder still needs every field; presets keep configs short.
ENCODER_FIELDS = ("width", "n_layers", "n_heads", "ffn_dim")


@dataclass(frozen=True)
class DataSource:
    manifest: Path
    shards: Path
    repeat: int = 1


@dataclass(frozen=True)
class ModelConfig:
    preset: str = "base"
    head_blocks: int = 6
    vocab_size: int = 1024
    checkpointing: bool = True
    overrides: dict[str, int] = field(default_factory=dict)
    #: Converted Whisper encoder to load before training (``tools/convert_whisper_encoder.py``).
    init: Path | None = None
    #: Start the extra blocks as exact identity (review P3) so every E-003 arm
    #: begins from the same function and depth is the only difference.
    identity_heads: bool = True

    def encoder(self) -> dict[str, int]:
        if self.preset not in ENCODER_PRESETS:
            raise ValueError(f"unknown encoder preset {self.preset!r}")
        values = dict(ENCODER_PRESETS[self.preset])
        values.update(self.overrides)
        missing = set(ENCODER_FIELDS) - set(values)
        if missing:
            raise ValueError(f"encoder config missing {sorted(missing)}")
        return {field: int(values[field]) for field in ENCODER_FIELDS}


@dataclass(frozen=True)
class DataConfig:
    tokenizer: Path
    train: list[DataSource]
    dev: list[DataSource]
    max_seconds: float = 192.0
    max_clips: int = 32
    augment: bool = True
    seed: int = 0
    dev_limit: int = 0
    eval_max_seconds: float = 192.0
    eval_max_clips: int = 32


@dataclass(frozen=True)
class OptimizerConfig:
    lr_head: float = 1e-4
    lr_encoder: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip: float = 1.0


@dataclass(frozen=True)
class ScheduleConfig:
    warmup_steps: int = 1500
    min_factor: float = 0.05


#: Which weights of a trainer checkpoint a fine-tune starts from (``training/ema.py``).
FINETUNE_WEIGHTS = ("auto", "model", "ema")


@dataclass(frozen=True)
class FinetuneConfig:
    """Start from a trained MOTH-50 checkpoint with a fresh optimizer and schedule."""

    init_checkpoint: Path
    #: Pinned sha256 of the checkpoint file; the trainer refuses a different file.
    sha256: str | None = None
    weights: str = "ema"


@dataclass(frozen=True)
class TrainConfig:
    path: Path
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    epochs: int = 4
    max_steps: int = 0
    log_every: int = 50
    eval_every_steps: int = 1000
    checkpoint_every_steps: int = 1000
    empty_cache_every: int = 10
    seed: int = 0
    autocast: str = "bf16"
    #: Weight EMA decay (``training/ema.py``); 0 disables it, as in E-003/E-005.
    ema_decay: float = 0.0
    #: Fine-tune from a trained checkpoint instead of ``model.init`` (E-027).
    finetune: FinetuneConfig | None = None


def override_seed(config: TrainConfig, seed: int) -> TrainConfig:
    """Same recipe, different init/augmentation/batch seed (E-003 second-seed rule)."""
    return replace(config, seed=seed, data=replace(config.data, seed=seed))


def _table(document: dict, name: str, *, optional: bool = False) -> dict:
    value = document.get(name, {} if optional else None)
    if value is None:
        raise ValueError(f"missing [{name}] table")
    if not isinstance(value, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return value


def _bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _int(value, name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _float(value, name: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _path(parent: Path, value, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


def _sources(parent: Path, value, name: str, *, required: bool) -> list[DataSource]:
    if value is None:
        if required:
            raise ValueError(f"missing [[{name}]] entries")
        return []
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array of tables")
    sources = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise TypeError(f"{name}[{index}] must be a table")
        try:
            manifest = entry["manifest"]
            shards = entry["shards"]
        except KeyError as error:
            raise ValueError(f"{name}[{index}] needs manifest and shards") from error
        sources.append(
            DataSource(
                manifest=_path(parent, manifest, f"{name}[{index}].manifest"),
                shards=_path(parent, shards, f"{name}[{index}].shards"),
                repeat=_int(entry.get("repeat", 1), f"{name}[{index}].repeat", 1),
            )
        )
    return sources


def load_config(path: Path | str) -> TrainConfig:
    """Load and validate a TOML training config. Relative paths are config-relative."""
    path = Path(path).resolve()
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    parent = path.parent

    model_doc = _table(document, "model")
    preset = model_doc.get("preset", "base")
    if preset not in ENCODER_PRESETS:
        raise ValueError(f"model.preset must be one of {sorted(ENCODER_PRESETS)}")
    overrides = {
        field: _int(model_doc[field], f"model.{field}", 1)
        for field in ENCODER_FIELDS
        if field in model_doc
    }
    model = ModelConfig(
        preset=preset,
        head_blocks=_int(model_doc.get("head_blocks", 6), "model.head_blocks"),
        vocab_size=_int(model_doc.get("vocab_size", 1024), "model.vocab_size", 1),
        checkpointing=_bool(model_doc.get("checkpointing", True), "model.checkpointing"),
        overrides=overrides,
        init=(
            _path(parent, model_doc["init"], "model.init")
            if model_doc.get("init") is not None
            else None
        ),
        identity_heads=_bool(model_doc.get("identity_heads", True), "model.identity_heads"),
    )

    data_doc = _table(document, "data")
    if "tokenizer" not in data_doc:
        raise ValueError("missing data.tokenizer")
    data = DataConfig(
        tokenizer=_path(parent, data_doc["tokenizer"], "data.tokenizer"),
        train=_sources(parent, data_doc.get("train"), "data.train", required=True),
        dev=_sources(parent, data_doc.get("dev"), "data.dev", required=False),
        max_seconds=_float(data_doc.get("max_seconds", 192.0), "data.max_seconds", positive=True),
        max_clips=_int(data_doc.get("max_clips", 32), "data.max_clips", 1),
        augment=_bool(data_doc.get("augment", True), "data.augment"),
        seed=_int(data_doc.get("seed", 0), "data.seed"),
        dev_limit=_int(data_doc.get("dev_limit", 0), "data.dev_limit"),
        eval_max_seconds=_float(
            data_doc.get("eval_max_seconds", 192.0), "data.eval_max_seconds", positive=True
        ),
        eval_max_clips=_int(data_doc.get("eval_max_clips", 32), "data.eval_max_clips", 1),
    )

    optimizer_doc = _table(document, "optimizer", optional=True)
    betas = optimizer_doc.get("betas", [0.9, 0.98])
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("optimizer.betas must be a two-element array")
    optimizer = OptimizerConfig(
        lr_head=_float(optimizer_doc.get("lr_head", 1e-4), "optimizer.lr_head", positive=True),
        lr_encoder=_float(
            optimizer_doc.get("lr_encoder", 1e-5), "optimizer.lr_encoder", positive=True
        ),
        betas=(_float(betas[0], "optimizer.betas[0]"), _float(betas[1], "optimizer.betas[1]")),
        eps=_float(optimizer_doc.get("eps", 1e-8), "optimizer.eps", positive=True),
        weight_decay=_float(optimizer_doc.get("weight_decay", 0.01), "optimizer.weight_decay"),
        grad_clip=_float(optimizer_doc.get("grad_clip", 1.0), "optimizer.grad_clip", positive=True),
    )

    schedule_doc = _table(document, "schedule", optional=True)
    schedule = ScheduleConfig(
        warmup_steps=_int(schedule_doc.get("warmup_steps", 1500), "schedule.warmup_steps"),
        min_factor=_float(schedule_doc.get("min_factor", 0.05), "schedule.min_factor"),
    )

    train_doc = _table(document, "train", optional=True)
    autocast = train_doc.get("autocast", "bf16")
    if autocast not in ("bf16", "fp16", "off"):
        raise ValueError("train.autocast must be bf16, fp16 or off")
    ema_decay = _float(train_doc.get("ema_decay", 0.0), "train.ema_decay")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("train.ema_decay must be 0 (off) or in (0, 1)")
    finetune = _finetune(parent, document.get("finetune"))
    if finetune is not None and model.init is not None:
        raise ValueError("finetune.init_checkpoint replaces model.init; remove model.init")
    return TrainConfig(
        path=path,
        model=model,
        data=data,
        optimizer=optimizer,
        schedule=schedule,
        epochs=_int(train_doc.get("epochs", 4), "train.epochs", 1),
        max_steps=_int(train_doc.get("max_steps", 0), "train.max_steps"),
        log_every=_int(train_doc.get("log_every", 50), "train.log_every", 1),
        eval_every_steps=_int(train_doc.get("eval_every_steps", 1000), "train.eval_every_steps"),
        checkpoint_every_steps=_int(
            train_doc.get("checkpoint_every_steps", 1000), "train.checkpoint_every_steps"
        ),
        empty_cache_every=_int(train_doc.get("empty_cache_every", 10), "train.empty_cache_every"),
        seed=_int(train_doc.get("seed", data.seed), "train.seed"),
        autocast=autocast,
        ema_decay=ema_decay,
        finetune=finetune,
    )


def _finetune(parent: Path, value) -> FinetuneConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("[finetune] must be a TOML table")
    if "init_checkpoint" not in value:
        raise ValueError("finetune.init_checkpoint is required")
    sha256 = value.get("sha256")
    if sha256 is not None and not (
        isinstance(sha256, str)
        and len(sha256) == 64
        and all(char in "0123456789abcdef" for char in sha256)
    ):
        raise ValueError("finetune.sha256 must be 64 lowercase hex characters")
    weights = value.get("weights", "ema")
    if weights not in FINETUNE_WEIGHTS:
        raise ValueError(f"finetune.weights must be one of {FINETUNE_WEIGHTS}")
    return FinetuneConfig(
        init_checkpoint=_path(parent, value["init_checkpoint"], "finetune.init_checkpoint"),
        sha256=sha256,
        weights=weights,
    )
