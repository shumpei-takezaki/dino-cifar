import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..distributed import all_reduce_sum


def dino_pairs(n_teacher: int, n_student: int) -> list[tuple[int, int]]:
    """(teacher_view, student_view) pairs; the identical global view is excluded."""
    return [(t, s) for t in range(n_teacher) for s in range(n_student) if s != t]


class DINOLoss(nn.Module):
    """DINO cross-view self-distillation on CLS prototypes (Caron et al., 2021).

    Teacher: softmax((logits - center) / t_temp); student: log_softmax(logits / s_temp).
    Cross-entropy averaged over all (teacher global, student view) pairs except same-view pairs.
    The center is an EMA of the (all-reduced) teacher batch mean.
    """

    def __init__(self, out_dim: int, student_temp: float, teacher_temp: float, center_momentum: float):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, n_student: int, n_teacher: int = 2) -> Tensor:
        """student_logits: [n_student*B, K] (view-major), teacher_logits: [n_teacher*B, K]."""
        s = (student_logits / self.student_temp).log_softmax(-1).chunk(n_student)
        t = F.softmax((teacher_logits - self.center) / self.teacher_temp, dim=-1).detach().chunk(n_teacher)
        pairs = dino_pairs(n_teacher, n_student)
        return sum((-t[i] * s[j]).sum(-1).mean() for i, j in pairs) / len(pairs)

    @torch.no_grad()
    def update_center(self, teacher_logits: Tensor) -> None:
        stats = all_reduce_sum(torch.cat([teacher_logits.sum(0), teacher_logits.new_tensor([len(teacher_logits)])]))
        batch_center = stats[:-1] / stats[-1]
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
