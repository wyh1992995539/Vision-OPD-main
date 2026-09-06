import copy

from omegaconf import OmegaConf

from scripts.checkpoint_io_contract import checkpoint_io_contract, checkpoint_io_matches


def test_current_contract_passes_but_historical_or_changed_code_does_not():
    contract = checkpoint_io_contract()
    assert checkpoint_io_matches({"checkpoint_io_contract": contract})
    assert not checkpoint_io_matches({"status": "PASS", "stage_gate_pass": True})
    changed = copy.deepcopy(contract)
    name = next(iter(changed["source_hashes"]))
    changed["source_hashes"][name] = "old-source-hash"
    assert not checkpoint_io_matches({"checkpoint_io_contract": changed})
    changed = copy.deepcopy(contract)
    changed["flush_reclaim"] = False
    assert not checkpoint_io_matches({"checkpoint_io_contract": changed})


def test_new_checkpoint_field_is_accepted_by_structured_config():
    from verl.trainer.config.config import CheckpointConfig

    config = OmegaConf.structured(CheckpointConfig)
    assert config.fsdp_flush_reclaim is False
    config.fsdp_flush_reclaim = True
    restored = OmegaConf.to_object(config)
    assert restored.fsdp_flush_reclaim is True
    assert restored.save_contents == ["model", "optimizer", "extra"]
