"""ViT backbone following the official DINO / DINOv2 conventions.

Token order: [CLS] [REG_1..REG_R] [PATCH_1..PATCH_N]. iBOT masking replaces patch
embeddings with ``mask_token`` *before* positional embeddings and Transformer blocks
(DINOv2 ``prepare_tokens_with_masks``). Positional embeddings are bicubically
interpolated so 16x16 local crops work alongside 32x32 global crops.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(1 - self.p)
        return x * keep / (1 - self.p)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.1,
        num_register_tokens: int = 0,
    ):
        super().__init__()
        if img_size % patch_size:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads, mlp_ratio, qkv_bias, p) for p in dpr)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.register_tokens, std=1e-6)
        self.apply(_init_linear)

    def interpolate_pos_encoding(self, h: int, w: int) -> Tensor:
        """Positional embedding for an h x w patch grid (bicubic, as in official DINO/DINOv2)."""
        n = self.pos_embed.shape[1] - 1
        if h * w == n and h == w:
            return self.pos_embed
        side = int(math.sqrt(n))
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patch_pos = patch_pos.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode="bicubic", align_corners=False)
        return torch.cat([cls_pos, patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)], dim=1)

    def prepare_tokens(self, x: Tensor, masks: Tensor | None = None) -> Tensor:
        B, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"input {H}x{W} is not divisible by patch_size={self.patch_size}")
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, D]
        if masks is not None:
            if masks.shape != x.shape[:2]:
                raise ValueError(f"masks shape {tuple(masks.shape)} != [B, N_patches] {tuple(x.shape[:2])}")
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.interpolate_pos_encoding(H // self.patch_size, W // self.patch_size)
        # Registers go after CLS and get no positional embedding (official DINOv2).
        return torch.cat([x[:, :1], self.register_tokens.expand(B, -1, -1), x[:, 1:]], dim=1)

    def forward_features(self, x: Tensor, masks: Tensor | None = None) -> dict[str, Tensor]:
        """Returns final-LayerNorm CLS [B,D], register tokens [B,R,D] and patch tokens [B,N,D]."""
        x = self.prepare_tokens(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        r = self.num_register_tokens
        return {"cls": x[:, 0], "reg_tokens": x[:, 1 : 1 + r], "patch_tokens": x[:, 1 + r :]}

    def forward(self, x: Tensor, masks: Tensor | None = None) -> Tensor:
        return self.forward_features(x, masks)["cls"]


def _init_linear(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_vit_tiny_patch4(**kwargs) -> VisionTransformer:
    """ViT-Tiny/4 for 32x32 CIFAR images; any kwarg overrides the default."""
    cfg = dict(img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0)
    return VisionTransformer(**{**cfg, **kwargs})
