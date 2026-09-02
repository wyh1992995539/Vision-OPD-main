"""Deterministic fixed-width sampling without dropping a partial final batch."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
import random
from typing import Generic, Iterator, Sized, TypeVar

T_co = TypeVar("T_co", covariant=True)


class Sampler(Generic[T_co]):
    """Minimal sampler protocol; DataLoader only requires iter and len."""

    pass


@dataclass(frozen=True)
class SampleReference:
    """Dataset index plus the optimization weight of this occurrence."""

    index: int
    sample_weight: float
    is_padding: bool


def build_epoch_references(
    dataset_size: int,
    *,
    multiple: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    padding_source_index: int = 0,
) -> list[SampleReference]:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if not 0 <= padding_source_index < dataset_size:
        raise ValueError("padding_source_index is outside the dataset")

    order = list(range(dataset_size))
    if shuffle:
        random.Random(seed + epoch).shuffle(order)

    references = [SampleReference(index, 1.0, False) for index in order]
    padding_rows = ceil(dataset_size / multiple) * multiple - dataset_size
    references.extend(
        SampleReference(padding_source_index, 0.0, True)
        for _ in range(padding_rows)
    )
    return references


class FullCoveragePaddingSampler(Sampler[SampleReference]):
    """Yield every real row once, then deterministic zero-weight padding rows."""

    def __init__(
        self,
        data_source: Sized,
        *,
        multiple: int,
        shuffle: bool,
        seed: int,
        padding_source_index: int = 0,
    ) -> None:
        self.data_source = data_source
        self.multiple = int(multiple)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.padding_source_index = int(padding_source_index)
        self.epoch = 0
        # Validate eagerly so configuration mistakes fail before GPU startup.
        build_epoch_references(
            len(self.data_source),
            multiple=self.multiple,
            shuffle=False,
            seed=self.seed,
            epoch=0,
            padding_source_index=self.padding_source_index,
        )

    def __iter__(self) -> Iterator[SampleReference]:
        references = build_epoch_references(
            len(self.data_source),
            multiple=self.multiple,
            shuffle=self.shuffle,
            seed=self.seed,
            epoch=self.epoch,
            padding_source_index=self.padding_source_index,
        )
        self.epoch += 1
        return iter(references)

    def __len__(self) -> int:
        return ceil(len(self.data_source) / self.multiple) * self.multiple

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        epoch = int(state_dict.get("epoch", 0))
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self.epoch = epoch
