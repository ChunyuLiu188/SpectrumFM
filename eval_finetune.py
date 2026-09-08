#!/usr/bin/env python3
"""Evaluate saved fine-tuned weights (no training).

  python eval_finetune.py ./runs/finetune/rml2016a/seed42            # re-run test (+ extra sets)
  python eval_finetune.py ./runs/finetune/manytx/seed42 --equalized 1  # override task options
  python eval_finetune.py RUN_DIR --npy my_iq.npy                    # predict a (N, L, 2) array

Accepts a run dir (uses last.pt) or a direct path to last.pt / best.pt / *_merged.pt.
Also usable as a library:
    from eval_finetune import load_finetuned
    model = load_finetuned("./runs/finetune/rml2016a/seed42/last.pt").cuda().eval()
    logits = model(iq_tensor)      # iq: (B, L, 2) raw float32, un-normalized
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune import IQClassifier, TensorBatches, accuracy_report, build_classifier, build_splits, predict, print_report  # noqa: E402


def load_finetuned(path: str | Path) -> IQClassifier:
    path = Path(path)
    if path.is_dir():
        path = path / "last.pt"
    obj = torch.load(path, map_location="cpu", weights_only=False)
    a = obj["args"]
    n_classes = len(obj["class_names"])
    seq_len = int(obj.get("seq_len", 128))
    if "model" in obj:  # *_merged.pt: plain weights, LoRA already folded in
        a = dict(a, mode="linear", ckpt="none")
        model, _, _ = build_classifier(a, n_classes, seq_len, verbose=False)
        model.load_state_dict(obj["model"], strict=True)
    else:  # last.pt / best.pt: rebuild pretrained backbone + LoRA, then load trainable weights
        model, _, _ = build_classifier(a, n_classes, seq_len, verbose=False)
        missing, unexpected = model.load_state_dict(obj["state"], strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys: {list(unexpected)[:5]}")
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run dir or .pt file")
    ap.add_argument("--npy", default=None, help="(N, L, 2) or (N, 2, L) float array to classify instead of the task test set")
    ap.add_argument("--equalized", type=int, default=None)
    ap.add_argument("--train-day", type=int, default=None)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--out", default=None, help="write predictions/metrics json here")
    args = ap.parse_args()

    path = Path(args.run)
    ckpt_path = path / "last.pt" if path.is_dir() else path
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = obj["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_finetuned(ckpt_path).to(device)
    print(f"loaded {ckpt_path} (task={a['task']} mode={a['mode']} seed={a['seed']} epoch={obj['epoch']})")

    if args.npy:
        X = np.load(args.npy).astype(np.float32)
        if X.ndim == 3 and X.shape[1] == 2 and X.shape[2] != 2:
            X = X.transpose(0, 2, 1)
        data = TensorBatches(X, np.zeros(len(X), np.int64), device)
        pred = predict(model, data, args.bs, amp=device.type == "cuda")
        names = obj["class_names"]
        out = {"pred": pred.tolist(), "pred_names": [names[i] for i in pred]}
        print(json.dumps({"n": len(pred), "first10": out["pred_names"][:10]}))
        if args.out:
            Path(args.out).write_text(json.dumps(out))
        return 0

    a = dict(a)
    if args.equalized is not None:
        a["equalized"] = args.equalized
    if args.train_day is not None:
        a["train_day"] = args.train_day
    a["max_train"] = 0
    task, _, _, Xva, yva = build_splits(a)  # identical split to training
    results = {}
    sets = {"test": (task.X_test, task.y_test, task.meta_test), **task.extra_tests}
    if Xva is not None:
        sets = {"val": (Xva, yva, {}), **sets}
    for name, (X, y, meta) in sets.items():
        pred = predict(model, TensorBatches(X, y, device), args.bs, amp=device.type == "cuda")
        rep = accuracy_report(pred, y, meta, task.n_classes)
        rep.pop("_confusion")
        results[name] = rep
        print_report(name, rep)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
