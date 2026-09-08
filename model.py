"""Memory Transformer for masked IQ reconstruction.

Burtsev / RMT style: learnable memory tokens are prepended and appended to
the IQ sequence and participate in full bidirectional attention. Masked I/Q
values are replaced by a learnable mask token (not dropped), so they also
attend and are attended to. I and Q are masked independently.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class MemoryTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = dropout

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        b, t, _ = q.shape
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )
        attn = attn.transpose(1, 2).contiguous().view(b, t, -1)
        x = x + self.drop_path(self.resid_drop(self.out_proj(attn)))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class MemoryTransformer(nn.Module):
    """16-layer Memory Transformer encoder for IQ masked prediction.

    Input:  iq   [B, L, 2]  sample-level RMS-normalized I/Q
            mask [B, L, 2]  True = this I or Q value is masked
    Output: pred [B, L, 2]  reconstructed I/Q at every position
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 16,
        n_heads: int = 8,
        d_ff: int = 2048,
        seq_len: int = 128,
        n_memory: int = 16,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        mask_ratio: float = 0.3,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_memory = n_memory
        self.mask_ratio = mask_ratio
        self.grad_checkpoint = grad_checkpoint

        self.i_proj = nn.Linear(1, d_model)
        self.q_proj = nn.Linear(1, d_model)
        self.mask_token_i = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token_q = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.memory = nn.Parameter(torch.zeros(1, n_memory, d_model))
        self.embed_drop = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path, n_layers).tolist() if drop_path > 0 else [0.0] * n_layers
        self.layers = nn.ModuleList(
            [
                MemoryTransformerLayer(d_model, n_heads, d_ff, dropout, dpr[i])
                for i in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.memory, std=0.02)
        nn.init.normal_(self.mask_token_i, std=0.02)
        nn.init.normal_(self.mask_token_q, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def make_mask(self, iq: torch.Tensor, ratio: Optional[float] = None) -> torch.Tensor:
        """Independent Bernoulli mask over I and Q. Shape [B, L, 2]."""
        ratio = self.mask_ratio if ratio is None else ratio
        b, t, _ = iq.shape
        mask = torch.rand(iq.shape, device=iq.device, dtype=torch.float32) < ratio
        # keep at least one masked value per sample so the loss is well-defined.
        # Vectorized (no .item()/branch) so torch.compile sees a single graph.
        empty = ~mask.flatten(1).any(dim=1)  # [B]
        idx = torch.randint(0, t, (b,), device=iq.device)
        ch = torch.randint(0, 2, (b,), device=iq.device)
        force = torch.zeros_like(mask)
        force[torch.arange(b, device=iq.device), idx, ch] = True
        return mask | (force & empty[:, None, None])

    def embed(self, iq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replace masked I/Q with mask tokens, then add memory tokens.

        Sequence layout: [mem_read | IQ tokens | mem_write]
        Mask tokens stay in the sequence and participate in attention.
        """
        i_emb = self.i_proj(iq[..., 0:1])
        q_emb = self.q_proj(iq[..., 1:2])
        i_emb = torch.where(mask[..., 0:1], self.mask_token_i.to(dtype=i_emb.dtype), i_emb)
        q_emb = torch.where(mask[..., 1:2], self.mask_token_q.to(dtype=q_emb.dtype), q_emb)
        h = i_emb + q_emb + self.pos_embed[:, : iq.size(1)]
        h = self.embed_drop(h)
        mem = self.memory.expand(iq.size(0), -1, -1).to(dtype=h.dtype)
        return torch.cat([mem, h, mem], dim=1)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        # Optional activation checkpointing: each block only stashes its input and
        # recomputes QKV / attention / FFN in backward. On 32GB GPUs this trades
        # ~25% throughput for ~8x batch; measured on RTX 5090 it is *slower* than
        # micro-batch 512 without checkpointing, so config defaults to off.
        use_ckpt = self.grad_checkpoint and self.training and h.requires_grad
        for layer in self.layers:
            if use_ckpt:
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return self.norm(h)

    def forward(
        self,
        iq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = self.make_mask(iq)
        h = self.embed(iq, mask)
        h = self.encode(h)
        iq_h = h[:, self.n_memory : self.n_memory + iq.size(1)]
        pred = self.head(iq_h)
        return pred, mask

    def masked_mse(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = (pred - target) ** 2
        denom = mask.float().sum().clamp_min(1.0)
        return (diff * mask.float()).sum() / denom

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> MemoryTransformer:
    m = cfg["model"]
    t = cfg.get("data", {})
    return MemoryTransformer(
        d_model=m.get("d_model", 512),
        n_layers=m.get("n_layers", 16),
        n_heads=m.get("n_heads", 8),
        d_ff=m.get("d_ff", 2048),
        seq_len=m.get("seq_len", t.get("seq_len", 128)),
        n_memory=m.get("n_memory", 16),
        dropout=m.get("dropout", 0.1),
        drop_path=m.get("drop_path", 0.0),
        mask_ratio=t.get("mask_ratio", 0.3),
        grad_checkpoint=m.get("grad_checkpoint", True),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    net = MemoryTransformer()
    x = torch.randn(4, 128, 2)
    pred, mask = net(x)
    loss = net.masked_mse(pred, x, mask)
    print(
        f"params={net.param_count()/1e6:.2f}M  pred={tuple(pred.shape)}  "
        f"mask_frac={mask.float().mean():.3f}  loss={loss.item():.4f}"
    )
    loss.backward()
    print("backward ok")
