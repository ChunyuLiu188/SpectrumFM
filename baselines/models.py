"""Wrappers: sample-level RMS IQ (B, L, 2) -> official baseline logits."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from official.awn import AWN  # noqa: E402
from official.mccfn import MCCFN  # noqa: E402


class AWNWrap(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = AWN(num_classes=n_classes)

    def forward(self, iq):  # (B, L, 2)
        logits, regu = self.net(iq.transpose(1, 2).contiguous())
        extra = sum(regu) if regu else iq.new_zeros(())
        return logits, extra


class MCCFNWrap(nn.Module):
    def __init__(self, n_classes: int, seq_len: int = 128):
        super().__init__()
        self.net = MCCFN(num_classes=n_classes, sig_len=seq_len)

    def forward(self, iq):
        logits, _ = self.net(iq.transpose(1, 2).contiguous())
        return logits, iq.new_zeros(())


class IQGCA(nn.Module):
    """Paper-faithful lightweight IQGCA (Fu et al., TVT 2026). No official code released.

    3D I/Q is projected onto IQ / It / Qt planes; each plane is a graph over time
    nodes. A graph-coarsening attention (GCA) block pools nodes, then a linear head.
    """

    def __init__(self, n_classes: int, seq_len: int = 128, d: int = 64, n_heads: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.qkv = nn.Linear(d, 3 * d)
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.coarse = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, max(8, seq_len // 8)))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def _gca(self, h, plane):
        # plane: (B, L, L) adjacency weights in [0, 1]
        B, L, d = h.shape
        qkv = self.qkv(h).view(B, L, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn + plane.unsqueeze(1).clamp_min(0).log().clamp_min(-8)
        attn = attn.softmax(-1)
        h2 = (attn @ v).transpose(1, 2).reshape(B, L, d)
        h = self.norm(h + self.proj(h2))
        s = self.coarse(h).softmax(1)  # (B, L, C) assignment
        pooled = s.transpose(1, 2) @ h  # (B, C, d)
        return pooled.mean(1)

    def forward(self, iq):
        i, q = iq[..., 0], iq[..., 1]
        # three planes: temporal corr of I, of Q, and of IQ product
        def corr(a):
            a = a - a.mean(-1, keepdim=True)
            g = a.unsqueeze(-1) * a.unsqueeze(-2)
            scale = a.pow(2).mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
            return (g / (scale.unsqueeze(-1) * scale.unsqueeze(-2))).abs()

        h = self.embed(iq)
        f = (self._gca(h, corr(i)) + self._gca(h, corr(q)) + self._gca(h, corr(i * q))) / 3
        return self.head(f), iq.new_zeros(())


class AvgNetWrap(nn.Module):
    """Official AvgNet architecture (Xuan et al., TNSE 2022), device-safe."""

    def __init__(self, n_classes: int, num_filter: int = 11):
        super().__init__()
        from math import ceil
        from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

        class GNN(nn.Module):
            def __init__(self, inn, hid, out, lin=True):
                super().__init__()
                self.convs = nn.ModuleList(
                    [DenseSAGEConv(inn if i == 0 else hid, hid if i < 5 else out) for i in range(6)]
                )
                self.bns = nn.ModuleList([nn.BatchNorm1d(hid if i < 5 else out) for i in range(6)])
                self.lin = nn.Linear(5 * hid + out, out) if lin else None

            def _bn(self, i, x):
                b, n, c = x.shape
                return self.bns[i](x.reshape(-1, c)).view(b, n, -1)

            def forward(self, x, adj, mask=None):
                xs = []
                for i, conv in enumerate(self.convs):
                    x = self._bn(i, F.relu(conv(x, adj, mask)))
                    xs.append(x)
                x = torch.cat(xs, -1)
                return F.relu(self.lin(x)) if self.lin is not None else x

        self.num_filter = num_filter
        self.add_conv1d = nn.ModuleList(
            [nn.Conv1d(1, 1, kernel_size=i + 2) for i in range(num_filter)]
        )
        n1 = ceil(0.25 * 128)
        n2 = ceil(0.25 * n1)
        self.gnn1_pool = GNN(2, 64, n1)
        self.gnn1_embed = GNN(2, 64, 64, lin=False)
        self.gnn2_pool = GNN(6 * 64, 64, n2)
        self.gnn2_embed = GNN(6 * 64, 64, 64, lin=False)
        self.gnn3_embed = GNN(6 * 64, 64, 64, lin=False)
        self.lin1 = nn.Linear(6 * 64, 3 * 64)
        self.lin2 = nn.Linear(3 * 64, 64)
        self.lin3 = nn.Linear(2 * 64, 64)
        self.lin4 = nn.Linear(64, n_classes)

    def _adj(self, ch):  # (B, 1, L)
        B, _, L = ch.shape
        adj = ch.new_zeros(B, L, L)
        for i, conv in enumerate(self.add_conv1d):
            d = F.relu(conv(ch).squeeze(1), inplace=True)
            adj = adj + torch.diag_embed(d, offset=i + 1)
        return adj + adj.transpose(1, 2)

    def _branch(self, x_iq, adj):
        from torch_geometric.nn import dense_diff_pool

        s = self.gnn1_pool(x_iq, adj)
        x = self.gnn1_embed(x_iq, adj)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        x = self.gnn3_embed(x, adj).mean(1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x, l1 + l2, e1 + e2

    def forward(self, iq):
        x_in = iq.unsqueeze(1)  # (B, 1, L, 2)
        xi = x_in[..., 0]
        xq = x_in[..., 1]
        x_iq = torch.cat([xi.transpose(1, 2), xq.transpose(1, 2)], 2) * 10
        hi, l1, e1 = self._branch(x_iq, self._adj(xi))
        hq, l2, e2 = self._branch(x_iq, self._adj(xq))
        logits = self.lin4(F.relu(self.lin3(torch.cat([hi, hq], 1))))
        return logits, 0.1 * (l1 + l2 + e1 + e2)


class STFGCNWrap(nn.Module):
    """Official STF-GCN (Shao et al., TCCN 2025) with IQ+STFT built from (B,L,2)."""

    def __init__(self, n_classes: int, seq_len: int = 128):
        super().__init__()
        from math import ceil
        from torch.nn.init import trunc_normal_
        from torch_geometric.nn.dense import DenseSAGEConv, DenseGATConv, dense_diff_pool

        class Fusion(nn.Module):
            def __init__(self, c, drop=0.0):
                super().__init__()
                self.Conv = nn.Sequential(nn.Conv1d(c, c, 1), nn.BatchNorm1d(c), nn.GELU(), nn.Conv1d(c, c, 1))
                self.drop = nn.Dropout(drop)
                for m in self.modules():
                    if isinstance(m, nn.Conv1d):
                        trunc_normal_(m.weight, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

            def forward(self, x, stft):
                return self.drop(self.Conv(torch.cat((x, stft), 1)))

        class GNN(nn.Module):
            def __init__(self, inn, hid, out, lin=True):
                super().__init__()
                self.conv1 = DenseSAGEConv(inn, hid)
                self.bn1 = nn.BatchNorm1d(hid)
                self.conv2 = DenseSAGEConv(hid, hid)
                self.bn2 = nn.BatchNorm1d(hid)
                self.conv3 = DenseSAGEConv(hid, hid)
                self.bn3 = nn.BatchNorm1d(hid)
                self.conv4 = DenseSAGEConv(hid, out)
                self.bn4 = nn.BatchNorm1d(out)
                self.lin = nn.Linear(3 * hid + out, out) if lin else None

            def _bn(self, i, x):
                b, n, c = x.shape
                return getattr(self, f"bn{i}")(x.reshape(-1, c)).view(b, n, -1)

            def forward(self, x, adj, mask=None):
                x1 = self._bn(1, F.relu(self.conv1(x, adj, mask)))
                x2 = self._bn(2, F.relu(self.conv2(x1, adj, mask)))
                x3 = self._bn(3, F.relu(self.conv3(x2, adj, mask)))
                x4 = self._bn(4, F.relu(self.conv4(x3, adj, mask)))
                x = torch.cat([x1, x2, x3, x4], -1)
                return F.relu(self.lin(x)) if self.lin is not None else x

        class AGNN(nn.Module):
            def __init__(self, inn, hid, out):
                super().__init__()
                self.gat1 = DenseGATConv(inn, hid, 1)
                self.bn1 = nn.BatchNorm1d(hid)
                self.gat2 = DenseGATConv(hid, out, 1)
                self.bn2 = nn.BatchNorm1d(out)

            def _bn(self, i, x):
                b, n, c = x.shape
                return getattr(self, f"bn{i}")(x.reshape(-1, c)).view(b, n, -1)

            def forward(self, x, adj, mask=None):
                x1 = self._bn(1, F.relu(self.gat1(x, adj, mask)))
                x2 = self._bn(2, F.relu(self.gat2(x1, adj, mask)))
                return torch.cat([x1, x2], -1)

        self.seq_len = seq_len
        self.num_filter = 11
        hid = 32
        self.bniq = nn.BatchNorm1d(2)
        self.iq_conv1d = nn.Conv1d(2, hid // 2, 3, padding=1, groups=2)
        self.bnstft = nn.BatchNorm2d(1)
        self.stft_conv2d = nn.Conv2d(1, hid // 2, kernel_size=(40, 1))
        self.fusion = Fusion(hid)
        self.LSTM = nn.LSTM(hid, hid, bidirectional=True, batch_first=True, num_layers=2, dropout=0.2)
        self.pool = nn.MaxPool1d(4, 4)
        self.gmp = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten())
        n1 = ceil(0.25 * seq_len)
        self.gnn1_pool = GNN(hid // 2, hid * 2, n1)
        self.gnn1_embed = GNN(hid // 2, hid * 2, hid * 2, lin=False)
        self.gat1_embed = AGNN(hid * 2 * 4, hid * 2 * 2, hid * 2 * 2)
        n2 = ceil(0.25 * n1)
        self.gnn2_pool = GNN(hid * 2 * 4, hid * 2, n2)
        self.gnn2_embed = GNN(hid * 2 * 4, hid * 2, hid * 2, lin=False)
        self.gat2_embed = AGNN(hid * 2 * 4, hid * 2 * 2, hid * 2 * 2)
        self.gnn3_embed = GNN(hid * 2 * 4, hid * 2, hid * 2, lin=False)
        self.gap = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.linear = nn.Linear(hid * 2 * 4, n_classes)

    def _stft(self, iq):
        z = torch.complex(iq[..., 0].float(), iq[..., 1].float())
        win = torch.hann_window(40, device=z.device, dtype=z.real.dtype)
        spec = torch.stft(z, n_fft=80, hop_length=1, win_length=40, window=win, return_complex=True, center=True)
        mag = spec.abs()[:, :40]
        return F.interpolate(mag, size=iq.size(1), mode="linear", align_corners=False)

    def _adj(self, x):
        B, N, C = x.shape
        adj = x.new_zeros(B, N, N)
        for i in range(self.num_filter):
            if N - 1 - i <= 0:
                break
            other = self.gmp(x[:, 0 : -1 - i] * x[:, i + 1 :])
            other = F.relu(other, inplace=True)
            adj = adj + torch.diag_embed(other, offset=i + 1)
        return adj + adj.transpose(1, 2)

    def forward(self, iq):
        from torch_geometric.nn.dense import dense_diff_pool

        x_iq = self.iq_conv1d(self.bniq(iq.transpose(1, 2)))
        x_stft = self.stft_conv2d(self.bnstft(self._stft(iq).unsqueeze(1))).squeeze(2)
        x = self.fusion(x_iq, x_stft)
        x, _ = self.LSTM(x.transpose(1, 2))
        x = self.pool(x)
        adj = self._adj(x)
        s = self.gnn1_pool(x, adj)
        x1 = self.gnn1_embed(x, adj)
        x1, adj1, _, _ = dense_diff_pool(x1, adj, s)
        x1 = self.gat1_embed(x1, adj1)
        s = self.gnn2_pool(x1, adj1)
        x2 = self.gnn2_embed(x1, adj1)
        x2, adj2, _, _ = dense_diff_pool(x2, adj1, s)
        x2 = self.gat2_embed(x2, adj2)
        x3 = self.gnn3_embed(x2, adj2)
        return self.linear(self.gap(x3.transpose(1, 2))), iq.new_zeros(())


BUILDERS = {
    "awn": AWNWrap,
    "mccfn": MCCFNWrap,
    "iqgca": IQGCA,
    "avgnet": AvgNetWrap,
    "stfgcn": STFGCNWrap,
}

CNN_MODELS = {"awn", "mccfn", "iqgca"}
GNN_MODELS = {"avgnet", "stfgcn"}


def build_baseline(name: str, n_classes: int, seq_len: int = 128) -> nn.Module:
    name = name.lower()
    if name not in BUILDERS:
        raise KeyError(name)
    cls = BUILDERS[name]
    if name in {"mccfn", "iqgca", "stfgcn"}:
        return cls(n_classes, seq_len)
    return cls(n_classes)
