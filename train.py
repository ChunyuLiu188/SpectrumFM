#!/usr/bin/env python3
"""Pretrain Memory Transformer with DeepSpeed ZeRO + bf16 on multi-GPU.

Launch (4x RTX 5090):
  bash run_pretrain.sh                       # deepspeed --num_gpus=4 train.py
  deepspeed --num_gpus=4 train.py --config config.yaml
  deepspeed --num_gpus=4 train.py --dummy --steps 20   # smoke test
  deepspeed --num_gpus=4 train.py --resume ./ckpts

Notes
  * Every DeepSpeed collective (save_checkpoint, all_reduce, evaluate) runs on
    ALL ranks. Only file writes / printing are gated on rank 0.
  * LR schedule is DeepSpeed's WarmupCosineLR so it is stepped by engine.step()
    and restored on resume.
  * Data is read as contiguous pre-shuffled blocks (see dataset.BlockIQDataset).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset

from dataset import BlockSampler, block_dataset_from_cache, rms_normalize_torch
from model import build_model


# ----------------------------------------------------------------------------- utils
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_distributed() -> tuple[int, int, int]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    return local_rank, rank, world_size


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def to_engine_tensor(engine, iq: torch.Tensor) -> torch.Tensor:
    iq = iq.to(engine.device, non_blocking=True)
    iq = rms_normalize_torch(iq)  # fp32 sample-level RMS, then cast
    if engine.bfloat16_enabled():
        iq = iq.bfloat16()
    elif engine.fp16_enabled():
        iq = iq.half()
    return iq


class DummyBlockDataset(Dataset):
    """Random IQ blocks for --dummy smoke tests. Item = one micro-batch."""

    def __init__(self, n_samples: int, block_size: int, seq_len: int) -> None:
        self.n = n_samples
        self.bs = block_size
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, start: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(int(start))
        return torch.randn(self.bs, self.seq_len, 2, generator=g)


# ----------------------------------------------------------------------------- deepspeed config
def build_ds_config(cfg: dict, template_path: Path, total_steps: int) -> dict:
    ds_cfg = json.loads(Path(template_path).read_text())
    train_cfg = cfg["train"]
    micro = int(train_cfg["batch_size"])
    accum = int(train_cfg.get("grad_accum", 1))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    lr = float(train_cfg["lr"])
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    warmup = int(train_cfg.get("warmup_steps", 2000))

    ds_cfg["train_micro_batch_size_per_gpu"] = micro
    ds_cfg["gradient_accumulation_steps"] = accum
    ds_cfg["train_batch_size"] = micro * accum * world_size
    ds_cfg["bf16"] = {"enabled": True}
    ds_cfg["fp16"] = {"enabled": False}
    ds_cfg["gradient_clipping"] = float(train_cfg.get("grad_clip", 1.0))
    ds_cfg["steps_per_print"] = int(train_cfg.get("log_every", 50)) * 20  # DS's own log is redundant with ours

    zero = ds_cfg.setdefault("zero_optimization", {})
    zero["stage"] = int(train_cfg.get("zero_stage", 2))

    opt = ds_cfg.setdefault("optimizer", {"type": "AdamW"}).setdefault("params", {})
    opt["lr"] = lr
    opt["weight_decay"] = float(train_cfg.get("weight_decay", 0.01))
    opt["torch_adam"] = True
    opt["fused"] = True  # torch.optim.AdamW(fused=True) on the fp32 ZeRO partitions

    # Stepped inside engine.step(); state saved/restored with the checkpoint.
    ds_cfg["scheduler"] = {
        "type": "WarmupCosineLR",
        "params": {
            "total_num_steps": max(1, total_steps),
            "warmup_num_steps": max(1, min(warmup, total_steps)),
            "warmup_min_ratio": 0.0,
            "warmup_type": "linear",
            "cos_min_ratio": min_lr / lr,
        },
    }
    return ds_cfg


# ----------------------------------------------------------------------------- eval / ckpt
@torch.no_grad()
def evaluate(engine, loader, seed: int) -> dict:
    """Runs on ALL ranks (each on its own shard of val); metrics are all-reduced."""
    engine.eval()
    model = engine.module
    # fixed mask RNG so val is comparable across epochs
    rng_state = torch.cuda.get_rng_state(engine.device)
    torch.cuda.manual_seed(seed)
    sums = torch.zeros(3, dtype=torch.float64, device=engine.device)  # mse_sum, mae_sum, n_masked
    for iq in loader:
        iq = to_engine_tensor(engine, iq)
        pred, mask = engine(iq)
        m = mask.float()
        diff = (pred.float() - iq.float())
        sums[0] += (diff.square() * m).sum().double()
        sums[1] += (diff.abs() * m).sum().double()
        sums[2] += m.sum().double()
    torch.cuda.set_rng_state(rng_state, engine.device)
    all_reduce_sum(sums)
    engine.train()
    n = max(1.0, float(sums[2]))
    return {"val_mse": float(sums[0]) / n, "val_mae": float(sums[1]) / n}


def save_all(engine, save_dir: Path, tag: str, step: int, epoch: int, cfg: dict, extra: dict | None = None) -> None:
    """Collective DS checkpoint on every rank + plain state_dict on rank 0."""
    client = {"step": step, "epoch": epoch, "cfg": cfg, **(extra or {})}
    engine.save_checkpoint(str(save_dir), tag=tag, client_state=client)
    if is_main():
        torch.save({"model": engine.module.state_dict(), **client}, save_dir / f"{tag}.pt")


# ----------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    parser.add_argument("--batch-size", type=int, default=None, help="micro batch per GPU")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", default=None, help="DeepSpeed checkpoint dir (uses its `latest` tag)")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.resume:
        train_cfg["resume"] = args.resume
    if args.no_compile:
        model_cfg["compile"] = False
    if args.log_every:
        train_cfg["log_every"] = args.log_every
    if args.save_dir:
        train_cfg["save_dir"] = args.save_dir

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    local_rank, rank, world_size = ensure_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed + rank)  # different mask RNG per rank; data split is handled by the sampler

    micro = int(train_cfg["batch_size"])
    accum = int(train_cfg.get("grad_accum", 1))
    epochs = int(train_cfg.get("epochs", 10))
    seq_len = int(data_cfg.get("seq_len", 128))
    workers = int(data_cfg.get("num_workers", 4))

    # ---- data: one DataLoader item == one micro-batch (contiguous pre-shuffled block)
    if args.dummy:
        n_dummy = micro * 64
        train_ds: Dataset = DummyBlockDataset(n_dummy, micro, seq_len)
        val_ds: Dataset | None = DummyBlockDataset(micro * 8, micro, seq_len)
        workers = 0
    else:
        cache = Path(data_cfg["cache_dir"])
        train_ds = block_dataset_from_cache(cache, "train", micro, rms=False)
        try:
            val_ds = block_dataset_from_cache(cache, "val", micro, rms=False)
        except Exception:
            val_ds = None

    train_sampler = BlockSampler(len(train_ds), micro, world_size, rank, seed=seed, shuffle=True, multiple_of=accum)
    loader_kw = dict(
        batch_size=None,  # dataset already returns whole micro-batches
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kw)
    val_loader = None
    if val_ds is not None and len(val_ds) >= micro * world_size:
        val_sampler = BlockSampler(len(val_ds), micro, world_size, rank, seed=seed, shuffle=False)
        val_loader = DataLoader(val_ds, sampler=val_sampler, **{**loader_kw, "num_workers": min(workers, 2), "persistent_workers": False})

    micro_per_epoch = len(train_loader)
    if micro_per_epoch < accum:
        raise RuntimeError(f"too few samples for micro={micro} x accum={accum} x world={world_size}")
    steps_per_epoch = micro_per_epoch // accum
    total_steps = epochs * steps_per_epoch
    if args.steps:
        total_steps = min(total_steps, args.steps)

    # ---- model
    model = build_model(cfg)
    use_compile = bool(model_cfg.get("compile", True)) and device.type == "cuda"
    if use_compile:
        # in-place: keeps state_dict keys clean (no `_orig_mod.` prefix)
        model.compile(dynamic=False)

    ds_path = Path(train_cfg.get("ds_config") or Path(__file__).parent / "ds_config.json")
    ds_cfg = build_ds_config(cfg, ds_path, total_steps)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_cfg)

    if is_main():
        n_params = sum(p.numel() for p in engine.module.parameters())
        print(
            f"world={world_size} params={n_params/1e6:.2f}M train_samples={len(train_ds)} "
            f"val_samples={len(val_ds) if val_ds is not None else 0} micro={micro} accum={accum} "
            f"global_batch={micro*accum*world_size} steps/epoch={steps_per_epoch} total_steps={total_steps} "
            f"zero={ds_cfg['zero_optimization']['stage']} bf16=True compile={use_compile} "
            f"grad_ckpt={engine.module.grad_checkpoint} workers={workers}",
            flush=True,
        )

    # ---- logging / ckpt dirs
    save_dir = Path(train_cfg.get("save_dir", "./emlm_ckpts"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 2000))
    writer = None
    if is_main() and not args.dummy:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_dir = Path(train_cfg.get("tb_dir", "./tb_logs")) / time.strftime("%Y%m%d-%H%M%S")
            writer = SummaryWriter(str(tb_dir))
            writer.add_text("config", "```\n" + yaml.safe_dump(cfg) + "\n```", 0)
            print(f"tensorboard -> {tb_dir}", flush=True)
        except Exception as exc:  # tensorboard is optional
            print(f"[warn] tensorboard disabled: {exc}", flush=True)

    # ---- resume (collective)
    start_step = 0
    start_epoch = 0
    best_val = float("inf")
    resume_path = train_cfg.get("resume")
    if resume_path:
        _, client = engine.load_checkpoint(resume_path)
        if client:
            start_step = int(client.get("step", engine.global_steps))
            start_epoch = int(client.get("epoch", 0))
            best_val = float(client.get("best_val", best_val))
        if is_main():
            print(f"resumed from {resume_path} step={start_step} epoch={start_epoch}", flush=True)

    engine.train()
    step = start_step
    running = torch.zeros((), device=device)
    running_n = 0
    t0 = time.time()
    epoch = start_epoch
    stop = False

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        it = iter(train_loader)
        # mid-epoch resume: skip micro-batches already consumed (block loads are cheap)
        done_in_epoch = (step - epoch * steps_per_epoch) * accum
        if 0 < done_in_epoch < micro_per_epoch:
            it = itertools.islice(it, done_in_epoch, None)
            if is_main():
                print(f"skipping {done_in_epoch} micro-batches of epoch {epoch+1}", flush=True)

        for iq in it:
            iq = to_engine_tensor(engine, iq)
            pred, mask = engine(iq)
            loss = engine.module.masked_mse(pred, iq, mask)
            engine.backward(loss)
            boundary = engine.is_gradient_accumulation_boundary()
            engine.step()

            running += loss.detach()
            running_n += 1
            if not boundary:
                continue
            step += 1

            if step % log_every == 0:
                avg = all_reduce_sum(running / running_n) / world_size
                dt = time.time() - t0
                sps = log_every * micro * accum * world_size / max(dt, 1e-6)
                lr = float(engine.get_lr()[0])
                if is_main():
                    print(
                        f"epoch {epoch+1}/{epochs} step {step}/{total_steps} loss {avg.item():.5f} "
                        f"lr {lr:.2e} {sps:,.0f} samples/s {dt/log_every*1000:.0f} ms/step",
                        flush=True,
                    )
                    if writer:
                        writer.add_scalar("train/loss", avg.item(), step)
                        writer.add_scalar("train/lr", lr, step)
                        writer.add_scalar("train/samples_per_s", sps, step)
                        writer.add_scalar("train/epoch", epoch + (step - epoch * steps_per_epoch) / steps_per_epoch, step)
                running.zero_()
                running_n = 0
                t0 = time.time()

            if not args.dummy and step % save_every == 0:
                save_all(engine, save_dir, "last", step, epoch, cfg, {"best_val": best_val})
                t0 = time.time()

            if step >= total_steps:
                stop = True
                break

        # ---- end of epoch (or --steps reached): eval + checkpoint on all ranks
        metrics: dict = {}
        if val_loader is not None:
            metrics = evaluate(engine, val_loader, seed=seed)
            if is_main():
                print(json.dumps({"epoch": epoch + 1, "step": step, **metrics}), flush=True)
                if writer:
                    for k, v in metrics.items():
                        writer.add_scalar(k.replace("val_", "val/"), v, step)
        improved = bool(metrics) and metrics["val_mse"] < best_val
        if improved:
            best_val = metrics["val_mse"]
        # epoch counter stored in the ckpt: only bump if this epoch really completed
        # (a --steps stop mid-epoch must resume inside the same epoch and skip ahead)
        ep_done = epoch + 1 if step >= (epoch + 1) * steps_per_epoch else epoch
        if not args.dummy:
            if improved:
                save_all(engine, save_dir, "best", step, ep_done, cfg, {"best_val": best_val, **metrics})
            save_all(engine, save_dir, "last", step, ep_done, cfg, {"best_val": best_val, **metrics})
            if is_main():
                (save_dir / f"epoch_{epoch+1:03d}.pt").write_bytes((save_dir / "last.pt").read_bytes())
                print(f"saved {save_dir / f'epoch_{epoch+1:03d}.pt'}", flush=True)
        t0 = time.time()
        if stop:
            break

    if writer:
        writer.close()
    if is_main():
        print(f"finished step={step} epoch={epoch+1} best_val={best_val:.6f} ckpt={save_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
