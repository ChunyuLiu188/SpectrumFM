#!/usr/bin/env python3
"""LoRA / full / linear-probe fine-tuning of the pretrained Memory Transformer (multi-GPU DDP).

Tasks
  rml2016a  11-way AMC, 8:2 stratified split, per-SNR accuracy
  rml2016b  10-way AMC, 8:2 stratified split, per-SNR accuracy
  manytx    150-way RF fingerprint; day-1 8:2 split, then test on days 2/3/4
  wtc       3-way wireless technology classification (WiFi/LTE/ZigBee, finetune.mat), per-SNR accuracy

Launch (4 GPUs):
  torchrun --nproc_per_node=4 finetune.py --task rml2016a --seed 42
  bash run_finetune.sh                # all tasks x seeds {42,123,456}
Single GPU also works: python finetune.py --task rml2016a

Schedule (default): 20 epochs, lr 1e-3 for epochs 1-10, then lr * 0.1 = 1e-4.
Split (default 8:2 here; run_finetune.sh uses 6:2:2 = train/val/test, stratified on class[,snr]).

Outputs in --out-dir/<task>/seed<seed>/:
  last.pt          trainable weights only (LoRA A/B + head [+ extras]) after the final epoch
  last_merged.pt   full backbone+head state_dict with LoRA folded in (load without lora.py)
  best.pt          same as last.pt but at the best --val-frac epoch (only if --val-frac > 0)
  results.json, log.jsonl, confusion_*.npy, args.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import rms_normalize_torch  # noqa: E402
from finetune_data import Task, load_task  # noqa: E402
from lora import LoRALinear, count_params, inject_lora, trainable_state_dict  # noqa: E402
from model import MemoryTransformer  # noqa: E402


# ----------------------------------------------------------------------------- distributed
def dist_setup() -> Tuple[int, int, int]:
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def gather_preds(idx: torch.Tensor, pred: torch.Tensor, n: int) -> Optional[np.ndarray]:
    """Rank-sharded predictions -> full array on rank 0 (None elsewhere)."""
    if not is_dist():
        out = np.empty(n, dtype=np.int64)
        out[idx.cpu().numpy()] = pred.cpu().numpy()
        return out
    parts: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * dist.get_world_size()
    dist.all_gather_object(parts, (idx.cpu().numpy(), pred.cpu().numpy()))
    if dist.get_rank() != 0:
        return None
    out = np.empty(n, dtype=np.int64)
    for i, p in parts:  # type: ignore[misc]
        out[i] = p
    return out


# ----------------------------------------------------------------------------- model
class IQClassifier(nn.Module):
    """Pretrained encoder + pooling + linear head.

    Sequences longer than the pretraining length (ManyTx: 256 vs 128):
      long_mode="split":  cut into 128-windows, encode each, average pooled features
      long_mode="interp": linearly interpolate pos_embed to L (made trainable)
    Each window is RMS-normalized on its own, exactly like pretraining chunks.
    """

    def __init__(self, backbone: MemoryTransformer, n_classes: int, pool: str = "mean",
                 long_mode: str = "split", head_dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.long_mode = long_mode
        d = backbone.d_model
        feat = 2 * d if pool == "meanmem" else d
        self.head = nn.Sequential(nn.LayerNorm(feat), nn.Dropout(head_dropout), nn.Linear(feat, n_classes))
        self.pos_long: Optional[nn.Parameter] = None

    def init_pos_long(self, L: int) -> None:
        bb = self.backbone
        with torch.no_grad():
            p = F.interpolate(bb.pos_embed.detach().transpose(1, 2), size=L, mode="linear", align_corners=True)
        self.pos_long = nn.Parameter(p.transpose(1, 2).contiguous())

    def _embed(self, iq: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        h = bb.i_proj(iq[..., 0:1]) + bb.q_proj(iq[..., 1:2])
        L = iq.size(1)
        if L <= bb.seq_len:
            pos = bb.pos_embed[:, :L]
        else:
            if self.pos_long is None or self.pos_long.shape[1] != L:
                self.init_pos_long(L)
            pos = self.pos_long
        h = bb.embed_drop(h + pos.to(h.dtype))
        mem = bb.memory.expand(iq.size(0), -1, -1).to(h.dtype)
        return torch.cat([mem, h, mem], dim=1)

    def _pool(self, h: torch.Tensor, L: int) -> torch.Tensor:
        m = self.backbone.n_memory
        sig = h[:, m : m + L]
        if self.pool == "mean":
            return sig.mean(1)
        if self.pool == "mem":  # write-memory tokens (tail), the RMT "summary"
            return h[:, m + L :].mean(1)
        if self.pool == "cls":
            return h[:, m + L]
        if self.pool == "meanmem":
            return torch.cat([sig.mean(1), h[:, m + L :].mean(1)], dim=-1)
        raise ValueError(self.pool)

    def features(self, iq: torch.Tensor) -> torch.Tensor:
        """iq: (B, L, 2) raw (un-normalized). Returns (B, feat)."""
        bb = self.backbone
        B, L, _ = iq.shape
        n_win = 1
        if L > bb.seq_len and self.long_mode == "split":
            n_win = L // bb.seq_len
            iq = iq[:, : n_win * bb.seq_len].reshape(B * n_win, bb.seq_len, 2)
        iq = rms_normalize_torch(iq)
        h = bb.encode(self._embed(iq))
        f = self._pool(h, iq.size(1))
        if n_win > 1:
            f = f.view(B, n_win, -1).mean(1)
        return f

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(iq))


def build_backbone(ckpt: Optional[str], dropout: float, drop_path: float, verbose: bool = True) -> Tuple[MemoryTransformer, dict]:
    cfg: dict = {}
    state = None
    if ckpt and ckpt.lower() != "none":
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = obj["model"] if "model" in obj else obj
        cfg = obj.get("cfg", {})
    m = cfg.get("model", {})
    bb = MemoryTransformer(
        d_model=m.get("d_model", 512), n_layers=m.get("n_layers", 16), n_heads=m.get("n_heads", 8),
        d_ff=m.get("d_ff", 2048), seq_len=m.get("seq_len", 128), n_memory=m.get("n_memory", 16),
        dropout=dropout, drop_path=drop_path, grad_checkpoint=False,
    )
    if state is not None:
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        missing, unexpected = bb.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"checkpoint missing keys: {list(missing)[:5]}")
        if verbose:
            print(f"loaded {ckpt} (ignored: {list(unexpected)})", flush=True)
    elif verbose:
        print("no checkpoint: random init backbone", flush=True)
    return bb, cfg


def build_classifier(args_d: dict, n_classes: int, seq_len: int, verbose: bool = True) -> Tuple[IQClassifier, dict, dict]:
    """Shared by finetune.py and eval_finetune.py: backbone + LoRA + head with the right trainable set."""
    backbone, pre_cfg = build_backbone(args_d["ckpt"], args_d["dropout"], args_d["drop_path"], verbose)
    model = IQClassifier(backbone, n_classes, pool=args_d["pool"], long_mode=args_d["long_mode"],
                         head_dropout=args_d["head_dropout"])
    if seq_len > backbone.seq_len and args_d["long_mode"] == "interp":
        model.init_pos_long(seq_len)
    counts = setup_trainable(model, args_d, verbose)
    return model, pre_cfg, counts


def setup_trainable(model: IQClassifier, a: dict, verbose: bool = True) -> Dict[str, int]:
    bb = model.backbone
    for p in bb.parameters():
        p.requires_grad_(False)
    mode = a["mode"]
    if mode == "lora":
        targets = tuple(t.strip() for t in a["lora_targets"].split(",") if t.strip())
        names = inject_lora(bb.layers, r=a["lora_r"], alpha=a["lora_alpha"], dropout=a["lora_dropout"], targets=targets)
        if verbose:
            print(f"LoRA r={a['lora_r']} alpha={a['lora_alpha']} on {len(names)} linears: targets={targets}", flush=True)
    elif mode == "full":
        for p in bb.parameters():
            p.requires_grad_(True)
    elif mode != "linear":
        raise ValueError(mode)
    extras = {e.strip() for e in a["train_extra"].split(",") if e.strip() and e.strip() != "none"}
    if "norm" in extras:
        for mod in bb.modules():
            if isinstance(mod, nn.LayerNorm):
                for p in mod.parameters():
                    p.requires_grad_(True)
    if "memory" in extras:
        bb.memory.requires_grad_(True)
    if "embed" in extras:
        for p in list(bb.i_proj.parameters()) + list(bb.q_proj.parameters()) + [bb.pos_embed]:
            p.requires_grad_(True)
    for p in model.head.parameters():
        p.requires_grad_(True)
    if model.pos_long is not None:
        model.pos_long.requires_grad_(True)
    tr, tot = count_params(model)
    return {"trainable": tr, "total": tot}


def merged_state_dict(model: IQClassifier) -> Dict[str, torch.Tensor]:
    """Full state_dict with LoRA deltas folded into base Linear weights (plain MemoryTransformer keys).

    Non-mutating: the live model keeps its separate A/B, so this is safe to call mid-training.
    """
    merged: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                merged[f"{name}.weight"] = (mod.base.weight + mod.delta_weight().to(mod.base.weight.dtype)).detach().cpu()
    sd: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if "lora_A" in k or "lora_B" in k:
            continue
        key = k.replace(".base.", ".")
        sd[key] = merged.get(key, v.detach().cpu().clone())
    return sd


# ----------------------------------------------------------------------------- splits
def build_splits(a: dict):
    """Task + (train, val) arrays from args. Shared with eval_finetune.py so splits match exactly.

    --split "8:2"   -> train 80% / test 20%, no val
    --split "6:2:2" -> train 60% / val 20% / test 20%   (all stratified on class[,snr])
    ManyTx: shares apply within the training day; days 2-4 are always full extra test sets.
    """
    parts = [float(p) for p in str(a.get("split", "8:2")).split(":")]
    if len(parts) not in (2, 3) or any(p <= 0 for p in parts):
        raise ValueError(f"--split must be a:b or a:b:c, got {a.get('split')}")
    parts = [p / sum(parts) for p in parts]
    train_share = parts[0]
    val_share_of_train = 0.0
    if len(parts) == 3:  # load train+val as one pool, then carve val out of it (stratified)
        train_share = parts[0] + parts[1]
        val_share_of_train = parts[1] / train_share
    elif a.get("val_frac", 0) and a["val_frac"] > 0:
        val_share_of_train = a["val_frac"]
    if a.get("train_frac") is not None:  # legacy override
        train_share = a["train_frac"]
    split_seed = a.get("split_seed", 0)
    task: Task = load_task(a["task"], cache_dir=Path(a["cache_dir"]), train_frac=train_share, seed=split_seed,
                           equalized=a.get("equalized", 0), train_day=a.get("train_day", 0),
                           per_cell=a.get("wtc_per_cell", 9200))
    Xtr, ytr = task.X_train, task.y_train
    Xva = yva = None
    if val_share_of_train > 0:
        from finetune_data import stratified_split

        snr = task.meta_train.get("snr")
        strata = ytr * 1000 + (snr + 100) if snr is not None else ytr
        tr_i, va_i = stratified_split(strata, 1.0 - val_share_of_train, split_seed + 1)
        Xva, yva = Xtr[va_i], ytr[va_i]
        Xtr, ytr = Xtr[tr_i], ytr[tr_i]
    if a.get("max_train") and len(ytr) > a["max_train"]:
        sel = np.random.default_rng(split_seed).choice(len(ytr), a["max_train"], replace=False)
        Xtr, ytr = Xtr[sel], ytr[sel]
    return task, Xtr, ytr, Xva, yva


# ----------------------------------------------------------------------------- data on device
class TensorBatches:
    """Whole split resident on each GPU; rank-sharded shuffled batches, no DataLoader."""

    def __init__(self, X: np.ndarray, y: np.ndarray, device: torch.device, rank: int = 0, world: int = 1) -> None:
        self.X = torch.from_numpy(np.ascontiguousarray(X)).to(device)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).to(device)
        self.n = len(self.y)
        self.rank, self.world = rank, world

    def train_batches(self, bs: int, epoch_seed: int):
        """Same permutation on every rank (seeded), disjoint equal-length shards."""
        g = torch.Generator(device="cpu")
        g.manual_seed(epoch_seed)
        perm = torch.randperm(self.n, generator=g)
        per = self.n // self.world
        mine = perm[self.rank * per : (self.rank + 1) * per].to(self.X.device)
        for i in range(0, per, bs):
            j = mine[i : i + bs]
            yield self.X[j], self.y[j]

    def eval_batches(self, bs: int):
        mine = torch.arange(self.rank, self.n, self.world, device=self.X.device)
        for i in range(0, len(mine), bs):
            j = mine[i : i + bs]
            yield j, self.X[j], self.y[j]


def augment(x: torch.Tensor, phase: bool, flip: bool) -> torch.Tensor:
    """Label-preserving IQ augmentation (AMC): random phase rotation, random time reversal."""
    if phase:
        th = torch.rand(x.size(0), 1, device=x.device) * (2 * math.pi)
        c, s = torch.cos(th), torch.sin(th)
        i, q = x[..., 0], x[..., 1]
        x = torch.stack([i * c - q * s, i * s + q * c], dim=-1)
    if flip:
        m = torch.rand(x.size(0), device=x.device) < 0.5
        x = torch.where(m[:, None, None], x.flip(1), x)
    return x


@torch.no_grad()
def predict(model: nn.Module, data: TensorBatches, bs: int, amp: bool) -> Optional[np.ndarray]:
    """Full prediction array on rank 0, None on other ranks."""
    model.eval()
    idxs, preds = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for j, xb, _ in data.eval_batches(bs):
            idxs.append(j)
            preds.append(model(xb).argmax(-1))
    idx = torch.cat(idxs) if idxs else torch.empty(0, dtype=torch.long, device=data.X.device)
    pred = torch.cat(preds) if preds else torch.empty(0, dtype=torch.long, device=data.X.device)
    return gather_preds(idx, pred, data.n)


def accuracy_report(pred: np.ndarray, y: np.ndarray, meta: Dict[str, np.ndarray], n_classes: int) -> dict:
    from sklearn.metrics import f1_score, precision_score
    rep = {
        "acc": float((pred == y).mean()),
        "n": int(len(y)),
        "precision": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
    }
    for key in ("snr", "rx", "eq"):
        if key in meta:
            vals = meta[key]
            rep[f"acc_by_{key}"] = {int(v): float((pred[vals == v] == y[vals == v]).mean()) for v in np.unique(vals)}
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    rep["_confusion"] = cm
    return rep


def print_report(name: str, rep: dict) -> None:
    print(f"{name}: acc {rep['acc']*100:.2f}%  (n={rep['n']})")
    if "acc_by_snr" in rep:
        row = "  ".join(f"{int(s):>3d}:{a*100:5.1f}" for s, a in sorted(rep["acc_by_snr"].items(), key=lambda kv: int(kv[0])))
        print("   per-SNR %:", row)


# ----------------------------------------------------------------------------- args
def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["rml2016a", "rml2016b", "manytx", "wtc"])
    ap.add_argument("--ckpt", default="./ckpts/best.pt", help="pretrained .pt or 'none'")
    ap.add_argument("--mode", default="lora", choices=["lora", "full", "linear"])
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="qkv,out_proj,ff", help="comma list among qkv,out_proj,ff")
    ap.add_argument("--train-extra", default="none", help="comma list among norm,memory,embed (also trained)")
    ap.add_argument("--pool", default="mean", choices=["mean", "mem", "cls", "meanmem"])
    ap.add_argument("--long-mode", default="split", choices=["split", "interp"])
    ap.add_argument("--dropout", type=float, default=0.1, help="backbone dropout during fine-tuning")
    ap.add_argument("--drop-path", type=float, default=0.0)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256, help="per-GPU batch; global = bs * n_gpus")
    ap.add_argument("--eval-bs", type=int, default=1024)
    ap.add_argument("--eval-extra-every", type=int, default=5, help="evaluate cross-day extra sets every N epochs (always at the end)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay-epoch", type=int, default=10, help="after this many epochs lr *= lr-decay")
    ap.add_argument("--lr-decay", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--head-lr", type=float, default=None, help="default = --lr")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--aug-phase", action="store_true", help="random phase rotation (AMC only; hurts fingerprinting)")
    ap.add_argument("--aug-flip", action="store_true")
    ap.add_argument("--split", default="8:2", help="train:test or train:val:test, e.g. 8:2 or 6:2:2 (stratified). "
                    "With a val part, best.pt is selected on val and also reported")
    ap.add_argument("--train-frac", type=float, default=None, help="(legacy) overrides --split train share")
    ap.add_argument("--val-frac", type=float, default=0.0, help="(legacy) extra val carved out of TRAIN when --split has no val part")
    ap.add_argument("--split-seed", type=int, default=0, help="data split seed (fixed across --seed so all seeds share the split)")
    ap.add_argument("--equalized", type=int, default=0, help="manytx: 0 raw, 1 equalized, -1 both")
    ap.add_argument("--train-day", type=int, default=0, help="manytx: index of the training day (0 = 2021_03_01)")
    ap.add_argument("--wtc-per-cell", type=int, default=9200,
                    help="wtc: samples per (class, snr) cell before the split; 9200 -> 220,800 total (~RML2016.10a). 0 = all 2.06M")
    ap.add_argument("--max-train", type=int, default=0, help="subsample train set (quick experiments)")
    ap.add_argument("--seed", type=int, default=42, help="init / shuffle seed")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out-dir", default="./runs/finetune")
    ap.add_argument("--run-name", default=None, help="default <task>/seed<seed>[_<mode>]")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--steps", type=int, default=0, help="debug: stop epoch after N steps")
    return ap


# ----------------------------------------------------------------------------- main
def main() -> int:
    args = get_parser().parse_args()
    rank, world, local = dist_setup()
    main_rank = rank == 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    amp = device.type == "cuda" and not args.no_amp

    run_name = args.run_name or f"{args.task}/seed{args.seed}" + ("" if args.mode == "lora" else f"_{args.mode}")
    out_dir = Path(args.out_dir) / run_name
    if main_rank:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    log_f = open(out_dir / "log.jsonl", "a") if main_rank else None

    def log(rec: dict) -> None:
        if main_rank and log_f:
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    # ---- data (identical split on every rank and every seed: split_seed)
    task, Xtr, ytr, Xva, yva = build_splits(vars(args))
    if main_rank:
        print(f"split={args.split} ", end="")
        print(f"task={task.name} classes={task.n_classes} L={task.seq_len} train={len(ytr)} "
              f"val={0 if yva is None else len(yva)} test={len(task.y_test)} "
              f"extra={ {k: len(v[1]) for k, v in task.extra_tests.items()} } gpus={world} global_bs={args.bs*world}", flush=True)
    train = TensorBatches(Xtr, ytr, device, rank, world)
    val = TensorBatches(Xva, yva, device, rank, world) if Xva is not None else None
    test = TensorBatches(task.X_test, task.y_test, device, rank, world)
    # Extra days stay on CPU until final eval — ManyTx day2/3/4 is ~800MB and
    # previously OOM'd 32GB cards when all splits lived on GPU with L=256.
    extras = {}

    # ---- model
    model, pre_cfg, counts = build_classifier(vars(args), task.n_classes, task.seq_len, verbose=main_rank)
    model.to(device)
    if main_rank:
        print(f"mode={args.mode} trainable={counts['trainable']/1e6:.3f}M / {counts['total']/1e6:.2f}M "
              f"({100*counts['trainable']/counts['total']:.2f}%) pool={args.pool} long_mode={args.long_mode}", flush=True)
    ddp = DDP(model, device_ids=[local]) if world > 1 else model
    fwd = torch.compile(ddp) if args.compile and device.type == "cuda" else ddp

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    decay = lambda ps: [p for p in ps if p.ndim >= 2]  # noqa: E731
    no_decay = lambda ps: [p for p in ps if p.ndim < 2]  # noqa: E731
    groups = [
        {"params": decay(body_params), "lr": args.lr, "weight_decay": args.wd},
        {"params": no_decay(body_params), "lr": args.lr, "weight_decay": 0.0},
        {"params": decay(head_params), "lr": args.head_lr or args.lr, "weight_decay": args.wd},
        {"params": no_decay(head_params), "lr": args.head_lr or args.lr, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8, fused=device.type == "cuda")
    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step: int, epoch: int) -> float:
        f = args.lr_decay if epoch > args.lr_decay_epoch else 1.0  # step schedule: 1e-3 -> 1e-4 after epoch 10
        if args.warmup_steps and step < args.warmup_steps:
            f *= (step + 1) / args.warmup_steps
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        return f

    def save(tag: str, epoch: int, extra: dict) -> None:
        if not main_rank:
            return
        meta = {"args": vars(args), "epoch": epoch, "pretrain_cfg": pre_cfg, "class_names": task.class_names,
                "seq_len": task.seq_len, **extra}
        torch.save({"state": trainable_state_dict(model), **meta}, out_dir / f"{tag}.pt")
        torch.save({"model": merged_state_dict(model), **meta}, out_dir / f"{tag}_merged.pt")

    # ---- train
    best_val, best_epoch = -1.0, -1
    best_state: Optional[Dict[str, torch.Tensor]] = None
    step = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = torch.zeros((), device=device)
        tot_correct = torch.zeros((), device=device, dtype=torch.long)
        seen = torch.zeros((), device=device, dtype=torch.long)
        for i, (xb, yb) in enumerate(train.train_batches(args.bs, epoch_seed=args.seed * 1000 + epoch)):
            if args.steps and i >= args.steps:
                break
            xb = augment(xb, args.aug_phase, args.aug_flip)
            set_lr(step, epoch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = fwd(xb)
                loss = F.cross_entropy(logits.float(), yb, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], args.grad_clip)
            opt.step()
            step += 1
            tot_loss += loss.detach() * len(yb)
            tot_correct += (logits.argmax(-1) == yb).sum()
            seen += len(yb)
        if is_dist():
            for t in (tot_loss, tot_correct, seen):
                dist.all_reduce(t)
        rec = {"epoch": epoch, "step": step, "lr": opt.param_groups[0]["lr"],
               "train_loss": round((tot_loss / seen.clamp_min(1)).item(), 5),
               "train_acc": round((tot_correct / seen.clamp_min(1)).item(), 5),
               "epoch_time_s": round(time.time() - t0, 1)}
        if val is not None:
            pv = predict(fwd, val, args.eval_bs, amp)
            improved = torch.zeros(1, device=device)
            if main_rank:
                rec["val_acc"] = round(float((pv == yva).mean()), 5)
                improved[0] = float(rec["val_acc"] > best_val)
            if is_dist():
                dist.broadcast(improved, 0)  # every rank must agree (weights are identical under DDP)
            if improved.item():
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in trainable_state_dict(model).items()}
                if main_rank:
                    best_val = rec["val_acc"]
                    save("best", epoch, {"val_acc": best_val})
                    rec["saved_best"] = True
        pt = predict(fwd, test, args.eval_bs, amp)
        if main_rank:
            rec["test_acc"] = round(float((pt == task.y_test).mean()), 5)
        if extras and (epoch % max(1, args.eval_extra_every) == 0 or epoch == args.epochs):
            for k, d in extras.items():
                pe = predict(fwd, d, args.eval_bs, amp)
                if main_rank:
                    rec[f"{k}_acc"] = round(float((pe == task.extra_tests[k][1]).mean()), 5)
        rec["eval_time_s"] = round(time.time() - t0 - rec["epoch_time_s"], 1)
        log(rec)

    save("last", args.epochs, {})
    if is_dist():
        dist.barrier()

    # ---- final report: last-epoch weights, plus best-val weights if a val split exists
    results = {"task": task.name, "mode": args.mode, "seed": args.seed, "epochs": args.epochs, "split": args.split,
               "n_train": int(train.n), "n_val": int(val.n) if val is not None else 0, "n_test": int(test.n),
               "best_val_epoch": best_epoch if val is not None else None, "best_val_acc": best_val if val is not None else None,
               "trainable_params": counts["trainable"], "total_params": counts["total"],
               "train_time_min": round((time.time() - t_start) / 60, 1), "class_names": task.class_names}

    def evaluate_all(tag: str) -> None:
        pt = predict(fwd, test, args.eval_bs, amp)
        if main_rank:
            rep = accuracy_report(pt, task.y_test, task.meta_test, task.n_classes)
            np.save(out_dir / f"confusion_test_{tag}.npy", rep.pop("_confusion"))
            results[f"test@{tag}"] = rep
        for k, (Xex, yex, meta) in task.extra_tests.items():
            d = TensorBatches(Xex, yex, device, rank, world)
            pe = predict(fwd, d, args.eval_bs, amp)
            if main_rank:
                rep = accuracy_report(pe, yex, meta, task.n_classes)
                np.save(out_dir / f"confusion_{k}_{tag}.npy", rep.pop("_confusion"))
                results[f"{k}@{tag}"] = rep
            del d
            if device.type == "cuda":
                torch.cuda.empty_cache()

    evaluate_all("last")
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        evaluate_all("best")
    if main_rank:
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"\n==== RESULTS {run_name} (seed {args.seed}, split {args.split}) ====")
        for k in results:
            if isinstance(results[k], dict) and "acc" in results[k]:
                print_report(k, results[k])
        if best_state is not None:
            print(f"best.pt = epoch {best_epoch} (val {best_val*100:.2f}%)")
        print(f"saved -> {out_dir}  (last.pt / best.pt = LoRA+head, *_merged.pt = full model)")
        if log_f:
            log_f.close()
    if is_dist():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
