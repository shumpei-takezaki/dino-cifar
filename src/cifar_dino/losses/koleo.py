import torch
import torch.nn.functional as F
from torch import Tensor


def koleo_loss(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Kozachenko-Leonenko entropy regularizer (DINOv2 ``KoLeoLoss``).

    x: [B, D] student *backbone* CLS embeddings of a single view (not head logits).
    Returns -mean(log(distance to nearest neighbour in the batch)) on L2-normalized features.
    """
    if x.ndim != 2:
        raise ValueError(f"koleo_loss expects [B, D] features, got {tuple(x.shape)}")
    x = F.normalize(x.float(), dim=-1, eps=eps)
    with torch.no_grad():
        dots = x @ x.T
        dots.fill_diagonal_(-1.0)  # exclude self
        nn_idx = dots.argmax(dim=1)
    return -torch.log(F.pairwise_distance(x, x[nn_idx]) + eps).mean()
