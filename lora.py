"""Minimal LoRA (Hu et al. 2021) for nn.Linear, no external deps.

    inject_lora(model.layers, r=16, alpha=32, targets=("qkv", "out_proj", "ff"))
    ... train ...
    torch.save(lora_state_dict(model), "lora.pt")   # only A/B (+ anything trainable)
    merge_lora(model)                                # fold into base weights for deployment
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B(A(dropout(x)))  with W, b frozen."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("r must be > 0")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.merged = False
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # B = 0 -> starts as identity delta

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        lora = F.linear(self.lora_dropout(x).to(self.lora_A.dtype), self.lora_A)
        lora = F.linear(lora, self.lora_B) * self.scaling
        return out + lora.to(out.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight += self.delta_weight().to(self.base.weight.dtype)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
            self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, scaling={self.scaling:g}, in={self.in_features}, out={self.out_features}"


def _matches(name: str, targets: Sequence[str]) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    parent = name.rsplit(".", 2)[-2] if name.count(".") >= 1 else ""
    for t in targets:
        if leaf == t or name.endswith("." + t):
            return True
        # "ff" matches the Linear children of the nn.Sequential named ff (ff.0, ff.3)
        if parent == t:
            return True
    return False


def inject_lora(
    root: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    targets: Sequence[str] = ("qkv", "out_proj", "ff"),
) -> list[str]:
    """Replace matching nn.Linear modules under `root` with LoRALinear, in place.

    Returns the list of replaced module names. Everything under `root` that is
    not a LoRA parameter is frozen.
    """
    replaced: list[str] = []
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear) or not _matches(name, targets):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        replaced.append(name)
    for n, p in root.named_parameters():
        if "lora_A" not in n and "lora_B" not in n:
            p.requires_grad_(False)
    return replaced


def lora_modules(root: nn.Module) -> Iterable[LoRALinear]:
    return (m for m in root.modules() if isinstance(m, LoRALinear))


def merge_lora(root: nn.Module) -> None:
    for m in lora_modules(root):
        m.merge()


def unmerge_lora(root: nn.Module) -> None:
    for m in lora_modules(root):
        m.unmerge()


def trainable_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    """State dict of every parameter that requires grad (LoRA A/B, head, unfrozen extras)."""
    names = {n for n, p in root.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in root.state_dict().items() if k in names}


def count_params(root: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in root.parameters())
    trainable = sum(p.numel() for p in root.parameters() if p.requires_grad)
    return trainable, total
