import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)

# 1. 构造一批假输入
batch_size = 8
input_dim = 4
vocab_size = 6

inputs = torch.randn(batch_size, input_dim)

# 2. Student 和 Teacher 都输出 logits
student = nn.Linear(input_dim, vocab_size)
teacher = nn.Linear(input_dim, vocab_size)

# Teacher 只负责提供目标分布，不参与优化
teacher.eval()
for parameter in teacher.parameters():
    parameter.requires_grad_(False)

optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

# 保存更新前的参数，用于验证谁发生了变化
student_before = copy.deepcopy(student.state_dict())
teacher_before = copy.deepcopy(teacher.state_dict())

# 3. Teacher forward：不构建计算图
with torch.no_grad():
    teacher_logits = teacher(inputs)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

# 4. Student forward：需要保留计算图
student_logits = student(inputs)
student_log_probs = F.log_softmax(student_logits, dim=-1)

# KL(Teacher || Student)
loss = F.kl_div(
    student_log_probs,
    teacher_probs,
    reduction="batchmean",
)

# 5. 反向传播并更新 Student
optimizer.zero_grad()
loss.backward()

student_grad_norm = sum(
    parameter.grad.norm().item()
    for parameter in student.parameters()
    if parameter.grad is not None
)

teacher_has_grad = any(
    parameter.grad is not None
    for parameter in teacher.parameters()
)

optimizer.step()

# 6. 验证参数是否发生变化
student_changed = any(
    not torch.equal(student_before[name], parameter)
    for name, parameter in student.state_dict().items()
)

teacher_changed = any(
    not torch.equal(teacher_before[name], parameter)
    for name, parameter in teacher.state_dict().items()
)

print(f"loss: {loss.item():.6f}")
print(f"student grad norm: {student_grad_norm:.6f}")
print(f"teacher has grad: {teacher_has_grad}")
print(f"student parameters changed: {student_changed}")
print(f"teacher parameters changed: {teacher_changed}")

# 自动验收
assert student_grad_norm > 0
assert teacher_has_grad is False
assert student_changed is True
assert teacher_changed is False

print("\nDay 1 Student–Teacher minimal example: PASS")