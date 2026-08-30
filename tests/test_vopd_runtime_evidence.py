import importlib.util
from pathlib import Path

import torch
from torch import nn


MODULE_PATH = Path(__file__).parents[1] / "verl" / "utils" / "runtime_evidence.py"
SPEC = importlib.util.spec_from_file_location("runtime_evidence", MODULE_PATH)
runtime_evidence = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runtime_evidence)


def test_runtime_probe_observes_optimizer_then_ema_boundaries():
    student = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    teacher = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    teacher.load_state_dict(student.state_dict())
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

    inputs = torch.ones(2, 4)
    loss = student(inputs).sum()
    loss.backward()

    assert runtime_evidence.non_none_gradient_count(teacher) == 0
    student_before_optimizer = runtime_evidence.capture_parameter_probe(student)
    teacher_before_optimizer = runtime_evidence.capture_parameter_probe(teacher)
    assert sum(values.numel() for _, _, values in student_before_optimizer) > 0

    optimizer.step()

    assert runtime_evidence.parameter_probe_max_abs_delta(student, student_before_optimizer) > 0.0
    assert runtime_evidence.parameter_probe_max_abs_delta(teacher, teacher_before_optimizer) == 0.0

    teacher_before_ema = runtime_evidence.capture_parameter_probe(teacher)
    runtime_evidence.update_ema_parameters(teacher, student, update_rate=0.5)
    assert runtime_evidence.parameter_probe_max_abs_delta(teacher, teacher_before_ema) > 0.0


def test_zero_rate_does_not_change_teacher():
    student = nn.Linear(4, 1)
    teacher = nn.Linear(4, 1)
    teacher.load_state_dict(student.state_dict())
    teacher_before_ema = runtime_evidence.capture_parameter_probe(teacher)

    runtime_evidence.update_ema_parameters(teacher, student, update_rate=0.0)
    assert runtime_evidence.parameter_probe_max_abs_delta(teacher, teacher_before_ema) == 0.0
