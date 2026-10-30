import torch.nn.functional as F
from torch import Tensor, nn


class DINOHead(nn.Module):
    """MLP -> bottleneck -> L2 normalize -> weight-normalized prototype layer (g fixed to 1).

    ``F.linear(x, normalize(W))`` is exactly ``weight_norm`` with a frozen unit gain
    (official ``norm_last_layer=True``), without the parametrization machinery.
    Shared between DINO (CLS tokens) and iBOT (patch tokens), as in DINOv2's default.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 4096,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
        num_layers: int = 3,
        use_batch_norm: bool = False,
        weight_norm_last_layer: bool = True,
    ):
        super().__init__()
        if not weight_norm_last_layer:
            raise NotImplementedError("only weight_norm_last_layer=True is supported")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [bottleneck_dim]
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers += [nn.BatchNorm1d(dims[i + 1])] if use_batch_norm else []
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = self.mlp(x.reshape(-1, shape[-1]))  # accepts [B, D] or [B, N, D]
        x = F.normalize(x, dim=-1, eps=1e-6)
        x = F.linear(x, F.normalize(self.last_layer.weight, dim=1))
        return x.reshape(*shape[:-1], -1)
