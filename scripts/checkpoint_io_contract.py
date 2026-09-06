"""Bind Pilot evidence to the checkpoint implementation actually selected."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "verl/utils/checkpoint/fsdp_checkpoint_manager.py",
    "verl/utils/checkpoint/shard_io.py",
    "verl/trainer/config/config.py",
    "scripts/run_vopd_2gpu.sh",
)


def checkpoint_io_contract():
    return {"strategy": "fsdp_sequential_shards_fsync_fadvise_v1", "flush_reclaim": True,
            "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}}


def checkpoint_io_matches(receipt):
    return receipt.get("checkpoint_io_contract") == checkpoint_io_contract()
