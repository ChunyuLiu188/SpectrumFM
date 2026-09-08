#!/usr/bin/env python3
"""Train AWN / MCCFN / IQGCA / AvgNet / STF-GCN on the same 6:2:2 RMS-normalized splits.

  torchrun --nproc_per_node=4 baselines/train.py --model awn --task rml2016a --seed 42
  bash run_baselines.sh          # 2016a x 5 models x 3 seeds
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import rms_normalize_torch  # noqa: E402
from finetune import TensorBatches, accuracy_report, build_splits, dist_setup, is_dist, print_report  # noqa: E402
from baselines.models import CNN_MODELS, GNN_MODELS, build_baseline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CNN_MODELS | GNN_MODELS))
    ap.add_argument("--task", default="rml2016a", choices=["rml2016a", "rml2016b", "wtc"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="6:2:2")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=0, help="0 = 256 CNN / 64 GNN")
    ap.add_argument("--eval-bs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay-epoch", type=int, default=20)
    ap.add_argument("--lr-decay", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--out-dir", default="./runs/baselines")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()

    rank, world, local = dist_setup()
    main_rank = rank == 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if args.bs <= 0:
        args.bs = 64 if args.model in GNN_MODELS else 256
    if args.eval_bs <= 0:
        args.eval_bs = args.bs * 2

    cfg = dict(task=args.task, cache_dir=args.cache_dir, split=args.split, split_seed=args.split_seed,
               train_frac=None, val_frac=0.0, max_train=0, equalized=0, train_day=0, wtc_per_cell=9200)
    task, Xtr, ytr, Xva, yva = build_splits(cfg)
    if main_rank:
        print(f"model={args.model} task={args.task} split={args.split} train={len(ytr)} "
              f"val={0 if yva is None else len(yva)} test={len(task.y_test)} gpus={world} bs={args.bs}", flush=True)

    train = TensorBatches(Xtr, ytr, device, rank, world)
    val = TensorBatches(Xva, yva, device, rank, world) if Xva is not None else None
    test = TensorBatches(task.X_test, task.y_test, device, rank, world)

    model = build_baseline(args.model, task.n_classes, task.seq_len).to(device)
    if main_rank:
        n = sum(p.numel() for p in model.parameters())
        print(f"params={n/1e6:.3f}M", flush=True)
    ddp = DDP(model, device_ids=[local]) if world > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))

    run = Path(args.out_dir) / args.task / args.model / f"seed{args.seed}"
    if main_rank:
        run.mkdir(parents=True, exist_ok=True)
        (run / "args.json").write_text(json.dumps(vars(args), indent=2))
    log_f = open(run / "log.jsonl", "a") if main_rank else None

    class _P(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(rms_normalize_torch(x))[0]

        def eval(self):
            self.m.eval()
            return self

        def train(self, mode=True):
            self.m.train(mode)
            return self

    best_val, best_state, best_epoch = -1.0, None, -1
    t0_all = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        lr = args.lr * (args.lr_decay if epoch > args.lr_decay_epoch else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        tot_loss = torch.zeros((), device=device)
        tot_ok = torch.zeros((), device=device)
        seen = torch.zeros((), device=device)
        t0 = time.time()
        for i, (xb, yb) in enumerate(train.train_batches(args.bs, args.seed * 1000 + epoch)):
            if args.steps and i >= args.steps:
                break
            xb = rms_normalize_torch(xb)
            logits, extra = ddp(xb)
            loss = F.cross_entropy(logits.float(), yb) + extra
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.detach() * len(yb)
            tot_ok += (logits.argmax(-1) == yb).sum()
            seen += len(yb)
        if is_dist():
            for t in (tot_loss, tot_ok, seen):
                dist.all_reduce(t)
        rec = {"epoch": epoch, "lr": lr, "train_loss": round((tot_loss / seen.clamp_min(1)).item(), 5),
               "train_acc": round((tot_ok / seen.clamp_min(1)).item(), 5), "epoch_time_s": round(time.time() - t0, 1)}

        from finetune import predict
        proxy = _P(ddp)
        if val is not None:
            pv = predict(proxy, val, args.eval_bs, amp=device.type == "cuda")
            improved = torch.zeros(1, device=device)
            if main_rank:
                rec["val_acc"] = round(float((pv == yva).mean()), 5)
                improved[0] = float(rec["val_acc"] > best_val)
            if is_dist():
                dist.broadcast(improved, 0)
            if improved.item():
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if main_rank:
                    best_val = rec["val_acc"]
                    torch.save({"model": best_state, "args": vars(args), "epoch": epoch,
                                "class_names": task.class_names}, run / "best.pt")
                    rec["saved_best"] = True
        pt = predict(proxy, test, args.eval_bs, amp=device.type == "cuda")
        if main_rank:
            rec["test_acc"] = round(float((pt == task.y_test).mean()), 5)
            print(json.dumps(rec), flush=True)
            if log_f:
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

    if main_rank:
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "args": vars(args), "epoch": args.epochs, "class_names": task.class_names}, run / "last.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
    proxy = _P(ddp)
    pred = predict(proxy, test, args.eval_bs, amp=device.type == "cuda")
    if main_rank:
        rep = accuracy_report(pred, task.y_test, task.meta_test, task.n_classes)
        np.save(run / "confusion_test.npy", rep.pop("_confusion"))
        results = {"model": args.model, "task": args.task, "seed": args.seed, "split": args.split,
                   "epochs": args.epochs, "best_epoch": best_epoch, "best_val": best_val,
                   "train_time_min": round((time.time() - t0_all) / 60, 1),
                   "test@best" if best_state is not None else "test@last": rep}
        (run / "results.json").write_text(json.dumps(results, indent=2))
        print_report("test", rep)
        print(f"saved -> {run}")
        if log_f:
            log_f.close()
    if is_dist():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
