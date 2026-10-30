import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..distributed import all_reduce_sum


class iBOTPatchLoss(nn.Module):
    """iBOT masked-patch self-distillation (Zhou et al., 2022; DINOv2 ``iBOTPatchLoss``).

    Cross-entropy between the teacher's (unmasked view) patch prototypes and the student's
    (masked view) patch prototypes, **only at masked positions**. Per sample it averages over
    that sample's masked patches, then averages over all samples (unmasked samples add 0),
    matching DINOv2 ``forward_masked``. Keeps its own center, separate from the DINO CLS center.
    """

    def __init__(self, out_dim: int, student_temp: float, teacher_temp: float, center_momentum: float):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_logits: Tensor, teacher_logits: Tensor, masks: Tensor) -> Tensor:
        """student/teacher_logits: [B, N, K]; masks: [B, N] bool, True = masked."""
        t = F.softmax((teacher_logits - self.center) / self.teacher_temp, dim=-1).detach()
        ce = (-t * (student_logits / self.student_temp).log_softmax(-1)).sum(-1)  # [B, N]
        m = masks.float()
        return ((ce * m).sum(-1) / m.sum(-1).clamp(min=1.0)).mean()

    @torch.no_grad()
    def update_center(self, teacher_logits: Tensor, masks: Tensor) -> None:
        """EMA of the teacher mean over masked positions only (as in DINOv2)."""
        masked = teacher_logits[masks]
        stats = all_reduce_sum(torch.cat([masked.sum(0), masked.new_tensor([len(masked)])]))
        if stats[-1] > 0:
            self.center.mul_(self.center_momentum).add_(stats[:-1] / stats[-1], alpha=1 - self.center_momentum)
