import copy

import torch
from torch import nn

from cifar_dino.train import ema_update


def test_ema_update():
    student = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    teacher = copy.deepcopy(student).requires_grad_(False)
    old = [p.clone() for p in teacher.parameters()]
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p))
    ema_update(teacher, student, 0.9)
    for t, o, s in zip(teacher.parameters(), old, student.parameters()):
        assert torch.allclose(t, 0.9 * o + 0.1 * s)
