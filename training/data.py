"""Mel-shard datasets for the MOTH-50 trainer.

Reuses Moth's ``MelDataset``, ``AugmentConfig``, ``collate`` and
``plan_batches``. Each configured source is a manifest JSONL plus a shard
directory containing ``index.json``; the trainer builds them with
``python -m stt.data.build_mel`` from the pinned sources.
"""

from __future__ import annotations

from stt.data.build_mel import load_index
from stt.data.dataset import AugmentConfig, MelDataset
from stt.manifest import load_manifest, validate_training_rows

from training.config import DataConfig, DataSource


class MultiMelDataset:
    """Several Moth mel-shard datasets addressed as one index space."""

    def __init__(
        self,
        sources: list[DataSource],
        *,
        augment: AugmentConfig | None,
        seed: int,
        limit: int = 0,
        role: str,
    ) -> None:
        self.parts: list[tuple[MelDataset, int]] = []
        self.rows: list[dict] = []
        self.durations: list[float] = []
        for source in sources:
            manifest, shards = source.manifest, source.shards
            rows = load_manifest(manifest)
            if limit:
                rows = rows[:limit]
            if source.repeat > 1:
                rows = rows * source.repeat
            index_path = shards / "index.json"
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"{role}: missing {index_path}; build it with "
                    f"`python -m stt.data.build_mel --manifest {manifest} --audio-root ... "
                    f"--out-dir {shards}`"
                )
            index = load_index(index_path)
            missing = [row["id"] for row in rows if row["id"] not in index]
            if missing:
                raise ValueError(
                    f"{role}: {len(missing)} manifest rows missing from {shards} "
                    f"(first: {missing[:3]}); rebuild the shards from this manifest"
                )
            dataset = MelDataset(index, shards, rows, augment=augment, seed=seed)
            self.parts.append((dataset, len(self.rows)))
            self.rows.extend(rows)
            self.durations.extend(float(row["duration"]) for row in rows)
        if not self.rows:
            raise ValueError(f"{role}: no rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int) -> dict:
        for dataset, offset in reversed(self.parts):
            if position >= offset:
                return dataset[position - offset]
        raise IndexError(position)

    def set_epoch(self, epoch: int) -> None:
        for dataset, _ in self.parts:
            dataset.epoch = epoch


def load_datasets(config: DataConfig) -> tuple[MultiMelDataset, MultiMelDataset | None]:
    """Load dev first, then train, enforcing Moth's no-test/no-S3 training rules."""
    dev = None
    dev_rows: list[dict] = []
    if config.dev:
        dev = MultiMelDataset(
            config.dev,
            augment=None,
            seed=config.seed,
            limit=config.dev_limit,
            role="dev",
        )
        dev_rows = dev.rows
    train = MultiMelDataset(
        config.train,
        augment=AugmentConfig() if config.augment else None,
        seed=config.seed,
        role="train",
    )
    validate_training_rows(train.rows, dev_rows)
    return train, dev
