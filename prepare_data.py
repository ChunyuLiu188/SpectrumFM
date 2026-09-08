#!/usr/bin/env python3
"""Extract archives, convert all radio IQ to 128-chunks, mix into shuffled shards.

RMS is NOT applied here — training does sample-level RMS on the GPU batch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dataset import (
    arrays_to_chunks,
    discover_raw_files,
    iter_raw_arrays,
    load_h5_slice,
    plan_h5_slices,
)

# Extract members from these archives (RadarComm hdf5 already unpacked).
ARCHIVE_PLAN = [
    {
        "kind": "zip",
        "path": "RML2018/radioml2018.zip",
        "members": ["GOLD_XYZ_OSC.0001_1024.hdf5"],
    },
    {
        "kind": "zip",
        "path": "DeepRadar/deepradar.zip",
        "members": ["X_train.mat", "X_test.mat", "X_val.mat"],
    },
    {
        "kind": "zip",
        "path": "POWDER/channelMeasurements.zip",
        "suffixes": [".hdf5", ".h5"],
    },
    {
        "kind": "7z",
        "path": "HKDD/HKDD_AMC12.7z",
        "members": [
            "HKDD_AMC12/HKDD_AMC12_train.mat",
            "HKDD_AMC12/HKDD_AMC12_test.mat",
        ],
    },
    {
        "kind": "7z",
        "path": "HKDD/HKDD_AMC36.7z.001",
        "members": [
            "HKDD_AMC36/HKDD_AMC36_train.mat",
            "HKDD_AMC36/HKDD_AMC36_test.mat",
        ],
    },
]


def safe_stem(root: Path, path: Path) -> str:
    rel = path.relative_to(root) if path.is_relative_to(root) else Path(path.name)
    text = str(rel.with_suffix("")).replace("\\", "/")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or path.stem


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _7z_extract(archive: Path, dest: Path, members: list[str] | None = None, threads: int = 64) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["7z", "e", "-y", f"-mmt={threads}", f"-o{dest}", str(archive)]
    if members:
        cmd.extend(members)
    run(cmd)
    if members:
        out: list[Path] = []
        for member in members:
            target = dest / Path(member).name
            if target.exists():
                out.append(target)
                continue
            found = list(dest.rglob(Path(member).name))
            if not found:
                raise FileNotFoundError(f"{member} not extracted from {archive}")
            out.append(found[0])
        return out
    return [p for p in dest.rglob("*") if p.is_file()]


def extract_zip(archive: Path, dest: Path, members: list[str] | None, suffixes: list[str] | None) -> list[Path]:
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        if members:
            pick = [n for n in names if Path(n).name in set(members) or n in set(members)]
        elif suffixes:
            pick = [n for n in names if any(n.lower().endswith(s) for s in suffixes)]
        else:
            pick = [n for n in names if not n.endswith("/")]
        reuse, need = [], []
        for name in pick:
            target = dest / Path(name).name
            info = zf.getinfo(name)
            if target.exists() and target.stat().st_size == info.file_size:
                print(f"  reuse {target}", flush=True)
                reuse.append(target)
            else:
                if target.exists():
                    target.unlink()
                need.append(name)
    if need:
        print(f"  7z -mmt extract {len(need)} members", flush=True)
        extracted = _7z_extract(archive, dest, need, threads=64)
        return reuse + extracted
    return reuse


def extract_7z_member(archive: Path, dest: Path, member: str) -> Path:
    return _7z_extract(archive, dest, [member], threads=64)[0]


def extract_archives(root: Path, extract_root: Path) -> list[Path]:
    extracted: list[Path] = []
    for job in ARCHIVE_PLAN:
        archive = root / job["path"]
        if not archive.exists():
            print(f"[skip missing archive] {archive}", flush=True)
            continue
        dest = extract_root / archive.stem.replace(".7z", "")
        print(f"[extract] {archive} -> {dest}", flush=True)
        if job["kind"] == "zip":
            files = extract_zip(archive, dest, job.get("members"), job.get("suffixes"))
        else:
            files = extract_7z(archive, dest)
        extracted.extend(files)
    return extracted


def write_source_shards(
    path: Path,
    roots: list[Path],
    out_dir: Path,
    seq_len: int,
    shard_size: int,
) -> list[str]:
    root = next((r for r in roots if path.is_relative_to(r)), roots[0])
    name = safe_stem(root, path)
    marker = out_dir / f"{name}.done"
    if marker.exists():
        return list(json.loads(marker.read_text()))

    buffer: list[np.ndarray] = []
    n_buf = 0
    shard_id = 0
    written: list[str] = []
    n_total = 0

    def flush(force: bool = False) -> None:
        nonlocal buffer, n_buf, shard_id, n_total
        if n_buf == 0:
            return
        if not force and n_buf < shard_size:
            return
        arr = np.concatenate(buffer, axis=0)
        buffer, n_buf = [], 0
        while arr.shape[0] >= shard_size:
            chunk, arr = arr[:shard_size], arr[shard_size:]
            out = out_dir / f"{name}_{shard_id:04d}.npy"
            np.save(out, np.ascontiguousarray(chunk, dtype=np.float32))
            written.append(str(out))
            n_total += chunk.shape[0]
            shard_id += 1
        if arr.shape[0] > 0:
            if force:
                out = out_dir / f"{name}_{shard_id:04d}.npy"
                np.save(out, np.ascontiguousarray(arr, dtype=np.float32))
                written.append(str(out))
                n_total += arr.shape[0]
                shard_id += 1
            else:
                buffer = [arr]
                n_buf = arr.shape[0]

    try:
        for raw in iter_raw_arrays(path):
            chunks = arrays_to_chunks(raw, seq_len=seq_len)
            if chunks.shape[0] == 0:
                continue
            buffer.append(chunks)
            n_buf += chunks.shape[0]
            flush(force=False)
        flush(force=True)
    except Exception as exc:
        print(f"[skip] {path}: {exc}", file=sys.stderr, flush=True)
        return []

    marker.write_text(json.dumps(written, indent=2))
    print(f"[ok] {path} -> {n_total} chunks, {len(written)} shards", flush=True)
    return written


def _convert_worker(payload: dict) -> list[str]:
    kind = payload.get("kind", "file")
    if kind == "slice":
        return _convert_slice_worker(payload)
    return write_source_shards(
        Path(payload["path"]),
        [Path(r) for r in payload["roots"]],
        Path(payload["out_dir"]),
        payload["seq_len"],
        payload["shard_size"],
    )


def _convert_slice_worker(payload: dict) -> list[str]:
    path = Path(payload["path"])
    out_dir = Path(payload["out_dir"])
    seq_len = int(payload["seq_len"])
    part = int(payload["part"])
    try:
        raw = load_h5_slice(path, payload["dset"], int(payload["axis"]), int(payload["start"]), int(payload["end"]))
        chunks = arrays_to_chunks(raw, seq_len=seq_len)
        if chunks.shape[0] == 0:
            return []
        out = out_dir / f"{payload['name']}_p{part:04d}.npy"
        np.save(out, np.ascontiguousarray(chunks, dtype=np.float32))
        return [str(out)]
    except Exception as exc:
        print(f"[skip slice] {path} {payload.get('dset')}[{payload.get('start')}:{payload.get('end')}]: {exc}", file=sys.stderr, flush=True)
        return []


def _dump_one_shard(item: tuple) -> str:
    arr, path = item
    np.save(path, np.ascontiguousarray(arr))
    return str(path)


def mix_shards(shard_paths: list[str], out_dir: Path, shard_size: int, val_ratio: float, seed: int) -> dict:
    if not shard_paths:
        raise RuntimeError("no shards to mix")
    counts = [int(np.load(p, mmap_mode="r").shape[0]) for p in shard_paths]
    n = int(sum(counts))
    probe = np.load(shard_paths[0], mmap_mode="r")
    seq_len, ch = int(probe.shape[1]), int(probe.shape[2])
    print(f"[mix] concatenating {len(shard_paths)} shards, {n} samples x {seq_len} into RAM", flush=True)
    mixed = np.empty((n, seq_len, ch), dtype=np.float32)
    i = 0
    for path, k in tqdm(list(zip(shard_paths, counts)), desc="concat"):
        part = np.load(path, mmap_mode="r")
        mixed[i : i + k] = part
        i += k
        try:
            os.remove(path)
        except OSError:
            pass
    print("[mix] shuffling", flush=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(mixed)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else 0
    val = mixed[:n_val]
    train = mixed[n_val:]
    del mixed

    def dump(arr: np.ndarray, prefix: str, n_write: int = 32) -> list[str]:
        if arr.shape[0] == 0:
            return []
        n_shards = max(1, (arr.shape[0] + shard_size - 1) // shard_size)
        items = []
        for i in range(n_shards):
            sl = arr[i * shard_size : (i + 1) * shard_size]
            items.append((sl, out_dir / f"{prefix}_{i:04d}.npy"))
        if len(items) == 1 or n_write <= 1:
            return [_dump_one_shard(it) for it in items]
        with ProcessPoolExecutor(max_workers=min(n_write, len(items))) as ex:
            return list(ex.map(_dump_one_shard, items))

    print("[mix] writing mixed shards", flush=True)
    train_paths = dump(train, "mixed_train")
    val_paths = dump(val, "mixed_val")
    print(f"[mix] train={train.shape[0]} val={val.shape[0]}", flush=True)
    return {
        "n_train": int(train.shape[0]),
        "n_val": int(val.shape[0]),
        "train": train_paths,
        "val": val_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=200_000)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--no-mix", action="store_true")
    args = parser.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg["root"])
    out_dir = Path(data_cfg["cache_dir"])
    extract_root = Path(data_cfg.get("extract_dir", "./extract"))
    seq_len = int(data_cfg.get("seq_len", 128))
    val_ratio = float(data_cfg.get("val_ratio", 0.01))
    seed = int(cfg.get("train", {}).get("seed", 42))
    skip_names = data_cfg.get("skip_names", [])
    skip_suffixes = data_cfg.get("skip_suffixes", [])
    workers = int(args.workers or data_cfg.get("prepare_workers", 64))

    out_dir.mkdir(parents=True, exist_ok=True)
    roots = [root, extract_root]
    all_shards: list[str] = []

    def convert_files(files: list[Path], label: str) -> None:
        nonlocal all_shards
        if not files:
            return
        payloads: list[dict] = []
        for p in files:
            name = safe_stem(next((r for r in roots if p.is_relative_to(r)), roots[0]), p)
            marker = out_dir / f"{name}.done"
            if marker.exists():
                all_shards.extend(json.loads(marker.read_text()))
                print(f"  reuse {p}", flush=True)
                continue
            n_file = max(1, len(files))
            inner = max(1, workers // min(n_file, workers))
            slices = plan_h5_slices(p, inner)
            if slices:
                print(f"  {p} -> {len(slices)} slices", flush=True)
                for i, (dset, axis, start, end) in enumerate(slices):
                    payloads.append(
                        {
                            "kind": "slice",
                            "path": str(p),
                            "roots": [str(r) for r in roots],
                            "out_dir": str(out_dir),
                            "seq_len": seq_len,
                            "shard_size": args.shard_size,
                            "name": name,
                            "dset": dset,
                            "axis": axis,
                            "start": start,
                            "end": end,
                            "part": i,
                        }
                    )
            else:
                print(f"  {p}", flush=True)
                payloads.append(
                    {
                        "kind": "file",
                        "path": str(p),
                        "roots": [str(r) for r in roots],
                        "out_dir": str(out_dir),
                        "seq_len": seq_len,
                        "shard_size": args.shard_size,
                    }
                )

        if not payloads:
            return
        n_w = max(1, min(workers, len(payloads)))
        print(f"[{label}] {len(payloads)} jobs, workers={n_w}", flush=True)
        done_by_file: dict[str, list[str]] = {}
        if n_w == 1:
            results = [_convert_worker(pl) for pl in payloads]
            for pl, res in zip(payloads, results):
                all_shards.extend(res)
                done_by_file.setdefault(pl["path"], []).extend(res)
        else:
            with ProcessPoolExecutor(max_workers=n_w) as ex:
                futs = {ex.submit(_convert_worker, pl): pl for pl in payloads}
                for fut in tqdm(as_completed(futs), total=len(futs), desc=label):
                    pl = futs[fut]
                    res = fut.result()
                    all_shards.extend(res)
                    done_by_file.setdefault(pl["path"], []).extend(res)
        for fpath, shards in done_by_file.items():
            p = Path(fpath)
            name = safe_stem(next((r for r in roots if p.is_relative_to(r)), roots[0]), p)
            marker = out_dir / f"{name}.done"
            if not marker.exists():
                marker.write_text(json.dumps(shards, indent=2))
                n_chunks = 0
                for s in shards:
                    try:
                        n_chunks += int(np.load(s, mmap_mode="r").shape[0])
                    except OSError:
                        pass
                print(f"[ok] {p} -> {n_chunks} chunks, {len(shards)} shards", flush=True)

    # 1) already-unpacked IQ (Hisar / pretrain.mat / RML22 / RadarComm hdf5)
    convert_files(
        discover_raw_files(root, skip_names=skip_names, skip_suffixes=skip_suffixes),
        "convert-unpacked",
    )

    # 2) one archive at a time so peak disk stays under 200G
    if not args.no_extract:
        for job in ARCHIVE_PLAN:
            archive = root / job["path"]
            if not archive.exists():
                print(f"[skip missing archive] {archive}", flush=True)
                continue
            dest = extract_root / archive.stem.replace(".7z", "")
            dest.mkdir(parents=True, exist_ok=True)

            def member_done(filename: str) -> bool:
                marker = out_dir / f"{safe_stem(extract_root, dest / Path(filename).name)}.done"
                return marker.exists()

            if job.get("members") and all(member_done(m) for m in job["members"]):
                print(f"[skip converted] {archive}", flush=True)
                continue
            print(f"[extract] {archive} -> {dest}", flush=True)
            if job["kind"] == "zip":
                files = extract_zip(archive, dest, job.get("members"), job.get("suffixes"))
                iq_files = [
                    p
                    for p in files
                    if p.suffix.lower() in {".mat", ".hdf5", ".h5", ".npy"}
                    and not p.name.startswith("Y_")
                    and not p.name.startswith("lbl_")
                ]
                convert_files(iq_files, f"convert-{archive.stem}")
                print(f"[cleanup] {dest}", flush=True)
                shutil.rmtree(dest, ignore_errors=True)
            else:
                members = job.get("members") or []
                pending = [m for m in members if not member_done(m)]
                if not pending:
                    print(f"[skip converted] {archive}", flush=True)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                print(f"[extract] {archive} :: {len(pending)} members, 7z -mmt=64", flush=True)
                files = _7z_extract(archive, dest, pending, threads=64)
                convert_files(files, f"convert-{archive.stem}")
                print(f"[cleanup] {dest}", flush=True)
                shutil.rmtree(dest, ignore_errors=True)

    source_shards = [p for p in all_shards if Path(p).exists()]
    print(f"source shards: {len(source_shards)}", flush=True)

    if args.no_mix:
        man = {"seq_len": seq_len, "train": source_shards, "val": [], "all": source_shards}
        (out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
        return 0

    mixed = mix_shards(source_shards, out_dir, args.shard_size, val_ratio, seed)
    for p in source_shards:
        try:
            os.remove(p)
        except OSError:
            pass
    for marker in out_dir.glob("*.done"):
        marker.unlink(missing_ok=True)

    if extract_root.exists():
        print(f"[cleanup] remove {extract_root}", flush=True)
        shutil.rmtree(extract_root, ignore_errors=True)

    manifest = {
        "seq_len": seq_len,
        "n_train": mixed["n_train"],
        "n_val": mixed["n_val"],
        "n_train_shards": len(mixed["train"]),
        "n_val_shards": len(mixed["val"]),
        "train": mixed["train"],
        "val": mixed["val"],
        "mixed": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("n_train", "n_val", "n_train_shards", "n_val_shards")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
