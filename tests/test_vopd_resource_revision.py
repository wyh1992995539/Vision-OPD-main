import copy
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from scripts.build_day11_static_gate import resource_changes_match, semantic_changes

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_serializes_graph_sizes_for_installed_compilation_api():
    text = (ROOT / "scripts/run_vopd_2gpu.sh").read_text()
    code = text.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    for config in ("vopd_6241.yaml", "vopd_6241_pilot_16.yaml", "vopd_6241_pilot_64.yaml"):
        result = subprocess.run(
            [sys.executable, "-", str(ROOT / "configs" / config), str(ROOT)],
            input=code, text=True, capture_output=True, check=True,
        )
        values = dict(item.split("=", 1) for item in shlex.split(result.stdout))
        assert json.loads(values["VLLM_CUDAGRAPH_CAPTURE_SIZES"]) == [1, 2, 4, 8]
        assert values["ACTOR_PARAM_OFFLOAD"] == values["ACTOR_OPTIMIZER_OFFLOAD"] == "true"
        assert values["REF_PARAM_OFFLOAD"] == "true"
        assert float(values["ROLLOUT_GPU_MEMORY_UTILIZATION"]) == 0.40
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="$VLLM_CUDAGRAPH_CAPTURE_SIZES"' in text
    assert 'actor_rollout_ref.rollout.cudagraph_capture_sizes=' not in text


def test_resource_amendment_verifies_real_config_and_rejects_data_changes():
    before = {"actor": {"parameter_offload": False}, "data": {"max_prompt_length": 8192}}
    after = copy.deepcopy(before)
    after["actor"]["parameter_offload"] = True
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audited.yaml"
        path.write_text(yaml.safe_dump(before))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        amendment = {"audited_config": {"path": str(path), "sha256": digest},
                     "semantic_changes": semantic_changes(before, after)}
        assert resource_changes_match(amendment, after, digest)
        changed_prompt = copy.deepcopy(after)
        changed_prompt["data"]["max_prompt_length"] = 4096
        assert not resource_changes_match(amendment, changed_prompt, digest)
        amendment["semantic_changes"] = semantic_changes(before, changed_prompt)
        assert not resource_changes_match(amendment, changed_prompt, digest)
        path.write_text("modified: true\n")
        assert not resource_changes_match(amendment, after, digest)


def test_vllm_adapter_preserves_explicit_length_and_defaults_only_when_unset():
    # Execute the real constructor's config-initialization block without importing GPU engines.
    import ast
    from types import SimpleNamespace
    source = ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "vLLMHttpServer")
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    body = []
    for node in init.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "rollout_mode"
            for target in node.targets
        ):
            break
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.If)):
            body.append(node)
    code = compile(ast.Module(body=body, type_ignores=[]), str(source), "exec")
    for configured, expected in ((9216, 9216), (4096, 4096), (None, 262144)):
        worker = SimpleNamespace()
        namespace = {
            "self": worker, "config": SimpleNamespace(max_model_len=configured),
            "model_config": SimpleNamespace(hf_config={}), "HFModelConfig": object, "RolloutConfig": object,
            "omega_conf_to_dataclass": lambda value, **kwargs: value,
            "get_max_position_embeddings": lambda config: 262144,
        }
        exec(code, namespace)
        assert worker.config.max_model_len == expected
