"""Low-overhead parameter probes used to produce training runtime evidence."""

import torch
from torch import nn
from torch.distributed.tensor import DTensor


ParameterProbe = list[tuple[int, torch.Tensor, torch.Tensor]]


def evenly_spaced_indices(length: int, count: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return valid int64 indices without floating-point endpoint rounding."""
    if length <= 0 or count <= 0:
        raise ValueError("Index length and count must be positive.")
    count = min(length, count)
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    positions = torch.arange(count, dtype=torch.long, device=device)
    return positions * (length - 1) // (count - 1)


def local_parameter_tensor(parameter: nn.Parameter) -> torch.Tensor:
    tensor = parameter.detach()
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor


def capture_parameter_probe(
    module: nn.Module,
    max_parameters: int = 32,
    samples_per_parameter: int = 16,
) -> ParameterProbe:
    """Capture a small deterministic sample without cloning full parameters."""
    if max_parameters <= 0 or samples_per_parameter <= 0:
        raise ValueError("Parameter probe limits must be positive.")

    parameters = list(module.parameters())
    eligible = [
        index for index, parameter in enumerate(parameters) if local_parameter_tensor(parameter).numel() > 0
    ]
    if not eligible:
        return []

    if len(eligible) > max_parameters:
        positions = evenly_spaced_indices(len(eligible), max_parameters).tolist()
        selected = [eligible[position] for position in positions]
    else:
        selected = eligible

    probe = []
    for parameter_index in selected:
        flat = local_parameter_tensor(parameters[parameter_index]).reshape(-1)
        sample_count = min(samples_per_parameter, flat.numel())
        sample_indices = evenly_spaced_indices(flat.numel(), sample_count, device=flat.device)
        sample_values = flat.index_select(0, sample_indices).float().clone()
        probe.append((parameter_index, sample_indices, sample_values))
    return probe


def parameter_probe_max_abs_delta(module: nn.Module, probe: ParameterProbe) -> float:
    parameters = list(module.parameters())
    max_delta = 0.0
    for parameter_index, sample_indices, before in probe:
        if parameter_index >= len(parameters):
            raise RuntimeError("Module parameter structure changed while collecting runtime evidence.")
        flat = local_parameter_tensor(parameters[parameter_index]).reshape(-1)
        if sample_indices.numel() and sample_indices.max().item() >= flat.numel():
            raise RuntimeError("Module parameter shape changed while collecting runtime evidence.")
        after = flat.index_select(0, sample_indices).float()
        max_delta = max(max_delta, (after - before).abs().max().item())
    return max_delta


def non_none_gradient_count(module: nn.Module) -> int:
    return sum(parameter.grad is not None for parameter in module.parameters())


@torch.no_grad()
def update_ema_parameters(teacher_module: nn.Module, student_module: nn.Module, update_rate: float) -> None:
    for teacher_param, student_param in zip(
        teacher_module.parameters(),
        student_module.parameters(),
        strict=True,
    ):
        student_data = student_param.data.to(device=teacher_param.device)
        teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
