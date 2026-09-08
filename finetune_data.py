"""Downstream datasets for fine-tuning: RML2016.10a / 10b (AMC) and WiSig ManyTx (RF fingerprint).

Every loader returns a `Task`:
    X_train, y_train, X_test, y_test            float32 (N, L, 2) / int64
    meta_test: {"snr": ...}                     for per-SNR accuracy (RML)
    extra_tests: {"day2": (X, y, meta), ...}    ManyTx cross-day sets
    class_names

Splits are stratified and seeded. Raw arrays are cached once as .npz in
`cache_dir` so the 3.5 GB pickle is not re-read every run.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RADIO_ROOT = Path(os.environ.get("RADIO_ROOT", "data"))
DEFAULT_CACHE = Path(os.environ.get("EMLM_CACHE", "cache"))

RML2016A_PATHS = [
    RADIO_ROOT / "RML2016.10a_dict.pkl",
    RADIO_ROOT / "RML2016" / "RML2016.10a_dict.pkl",
]
RML2016B_PATHS = [
    RADIO_ROOT / "RML2016.10b.dat",
    RADIO_ROOT / "RML2016" / "RML2016.10b.dat",
]
MANYTX_PATH = Path(os.environ.get("MANYTX_PKL", str(RADIO_ROOT / "ManyTx.pkl")))

MANYTX_DAYS = ["2021_03_01", "2021_03_08", "2021_03_15", "2021_03_23"]


@dataclass
class Task:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: List[str]
    meta_train: Dict[str, np.ndarray] = field(default_factory=dict)
    meta_test: Dict[str, np.ndarray] = field(default_factory=dict)
    extra_tests: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]] = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def seq_len(self) -> int:
        return int(self.X_train.shape[1])


# ----------------------------------------------------------------------------- helpers
def _first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"none of {[str(p) for p in paths]} exist")


def stratified_split(
    strata: np.ndarray, train_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-stratum shuffled split. Returns (train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        k = int(round(len(idx) * train_frac))
        tr.append(idx[:k])
        te.append(idx[k:])
    tr = np.concatenate(tr)
    te = np.concatenate(te)
    rng.shuffle(tr)
    rng.shuffle(te)
    return tr, te


def _load_rml_pickle(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")  # python2 pickles
    mods = sorted({k[0] for k in d})
    xs, ys, snrs = [], [], []
    for (mod, snr), arr in d.items():
        arr = np.asarray(arr, dtype=np.float32)  # (N, 2, 128)
        if arr.ndim == 3 and arr.shape[1] == 2:
            arr = arr.transpose(0, 2, 1)  # -> (N, 128, 2)
        xs.append(np.ascontiguousarray(arr))
        ys.append(np.full(arr.shape[0], mods.index(mod), dtype=np.int64))
        snrs.append(np.full(arr.shape[0], int(snr), dtype=np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(snrs), mods


def _cached(cache_dir: Path, name: str, builder):
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    out = builder()
    np.savez(f, **out)
    return out


# ----------------------------------------------------------------------------- RML2016
def _load_rml(name: str, paths: List[Path], cache_dir: Path, train_frac: float, seed: int) -> Task:
    path = _first_existing(paths)

    def build():
        X, y, snr, mods = _load_rml_pickle(path)
        return {"X": X, "y": y, "snr": snr, "mods": np.array(mods)}

    d = _cached(cache_dir, name, build)
    X, y, snr, mods = d["X"], d["y"], d["snr"], [str(m) for m in d["mods"]]
    # stratify on (class, snr) so every SNR bucket keeps the 8:2 ratio
    strata = y * 1000 + (snr + 100)
    tr, te = stratified_split(strata, train_frac, seed)
    return Task(
        name=name,
        X_train=X[tr],
        y_train=y[tr],
        X_test=X[te],
        y_test=y[te],
        class_names=mods,
        meta_train={"snr": snr[tr]},
        meta_test={"snr": snr[te]},
    )


def load_rml2016a(cache_dir: Path = DEFAULT_CACHE, train_frac: float = 0.8, seed: int = 0) -> Task:
    return _load_rml("rml2016a", RML2016A_PATHS, cache_dir, train_frac, seed)


def load_rml2016b(cache_dir: Path = DEFAULT_CACHE, train_frac: float = 0.8, seed: int = 0) -> Task:
    return _load_rml("rml2016b", RML2016B_PATHS, cache_dir, train_frac, seed)


# ----------------------------------------------------------------------------- WiSig ManyTx
def _load_manytx_raw(path: Path) -> dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    tx_list, rx_list = list(d["tx_list"]), list(d["rx_list"])
    days, eqs = list(d["capture_date_list"]), list(d["equalized_list"])
    xs, tx_i, rx_i, day_i, eq_i = [], [], [], [], []
    for ti, per_tx in enumerate(d["data"]):
        for ri, per_rx in enumerate(per_tx):
            for di, per_day in enumerate(per_rx):
                for ei, arr in enumerate(per_day):
                    arr = np.asarray(arr)
                    if arr.ndim != 3 or arr.shape[0] == 0:
                        continue
                    n = arr.shape[0]
                    xs.append(arr.astype(np.float32))  # (n, 256, 2)
                    tx_i.append(np.full(n, ti, np.int64))
                    rx_i.append(np.full(n, ri, np.int64))
                    day_i.append(np.full(n, di, np.int64))
                    eq_i.append(np.full(n, eqs[ei], np.int64))
    return {
        "X": np.concatenate(xs),
        "tx": np.concatenate(tx_i),
        "rx": np.concatenate(rx_i),
        "day": np.concatenate(day_i),
        "eq": np.concatenate(eq_i),
        "tx_list": np.array(tx_list),
        "rx_list": np.array(rx_list),
        "days": np.array(days),
    }


def load_manytx(
    cache_dir: Path = DEFAULT_CACHE,
    train_frac: float = 0.8,
    seed: int = 0,
    equalized: int = 0,
    train_day: int = 0,
) -> Task:
    """Day-1 8:2 split (stratified per TX) for train/test; days 2-4 as extra test sets.

    equalized: 0 raw captures, 1 equalized captures, -1 both.
    """
    d = _cached(cache_dir, "manytx", lambda: _load_manytx_raw(MANYTX_PATH))
    X, tx, rx, day, eq = d["X"], d["tx"], d["rx"], d["day"], d["eq"]
    days = [str(s) for s in d["days"]]
    tx_names = [str(s) for s in d["tx_list"]]
    keep = np.ones(len(X), bool) if equalized < 0 else (eq == equalized)

    d1 = np.flatnonzero(keep & (day == train_day))
    tr_rel, te_rel = stratified_split(tx[d1], train_frac, seed)
    tr, te = d1[tr_rel], d1[te_rel]

    extra = {}
    for di in range(len(days)):
        if di == train_day:
            continue
        idx = np.flatnonzero(keep & (day == di))
        if len(idx) == 0:
            continue
        extra[f"day{di+1}_{days[di]}"] = (X[idx], tx[idx], {"rx": rx[idx], "eq": eq[idx]})

    return Task(
        name="manytx",
        X_train=X[tr],
        y_train=tx[tr],
        X_test=X[te],
        y_test=tx[te],
        class_names=tx_names,
        meta_train={"rx": rx[tr], "eq": eq[tr]},
        meta_test={"rx": rx[te], "eq": eq[te]},
        extra_tests=extra,
    )


# ----------------------------------------------------------------------------- WTC (finetune.mat)
WTC_PATH = Path(os.environ.get("WTC_MAT", str(RADIO_ROOT / "TecRec" / "finetune.mat")))
# label row 0 in {0,1,2}; assumed order WiFi / LTE / ZigBee (accuracy is unaffected by naming)
WTC_CLASSES = ["WiFi", "LTE", "ZigBee"]


def _load_wtc_raw(path: Path, chunk: int = 65536) -> dict:
    """MATLAB v7.3: out (256, N) = stacked [I(128); Q(128)] per column (same convention as
    pretrain.mat / dataset.to_nt2), label (2, N) = [class; snr]."""
    import h5py

    with h5py.File(path, "r") as f:
        lab = np.asarray(f["label"]).astype(np.int64)  # (2, N)
        out = f["out"]
        t, n = out.shape
        if t != 256:
            raise ValueError(f"expected 256 rows in out, got {t}")
        X = np.empty((n, 128, 2), dtype=np.float32)
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            blk = np.asarray(out[:, s:e], dtype=np.float32)  # (256, k)
            X[s:e, :, 0] = blk[:128].T
            X[s:e, :, 1] = blk[128:].T
    return {"X": X, "y": lab[0], "snr": lab[1], "mods": np.array(WTC_CLASSES)}


def load_wtc(cache_dir: Path = DEFAULT_CACHE, train_frac: float = 0.8, seed: int = 0, per_cell: int = 9200) -> Task:
    """per_cell: samples kept per (class, snr) cell before splitting; 0 = all 2.06M.
    Default 9200 -> 3 x 8 x 9200 = 220,800 samples, same scale as RML2016.10a (220k)."""
    if not WTC_PATH.exists():
        raise FileNotFoundError(WTC_PATH)
    d = _cached(cache_dir, "wtc", lambda: _load_wtc_raw(WTC_PATH))
    X, y, snr, mods = d["X"], d["y"], d["snr"], [str(m) for m in d["mods"]]
    strata = y * 1000 + (snr + 100)
    if per_cell and per_cell > 0:
        rng = np.random.default_rng(seed + 7)  # fixed by split seed -> identical subset for every run seed
        keep = []
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            keep.append(rng.choice(idx, min(per_cell, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        X, y, snr, strata = X[keep], y[keep], snr[keep], strata[keep]
    tr, te = stratified_split(strata, train_frac, seed)
    return Task(
        name="wtc",
        X_train=X[tr],
        y_train=y[tr],
        X_test=X[te],
        y_test=y[te],
        class_names=mods,
        meta_train={"snr": snr[tr]},
        meta_test={"snr": snr[te]},
    )


LOADERS = {
    "rml2016a": load_rml2016a,
    "rml2016b": load_rml2016b,
    "manytx": load_manytx,
    "wtc": load_wtc,
}


def load_task(name: str, **kw) -> Task:
    if name not in LOADERS:
        raise KeyError(f"unknown task {name}; choose from {sorted(LOADERS)}")
    fn = LOADERS[name]
    if name != "manytx":
        kw.pop("equalized", None)
        kw.pop("train_day", None)
    if name != "wtc":
        kw.pop("per_cell", None)
    return fn(**kw)


if __name__ == "__main__":
    import sys

    for n in sys.argv[1:] or ["rml2016a", "rml2016b", "manytx"]:
        t = load_task(n)
        print(
            f"{t.name}: train {t.X_train.shape} test {t.X_test.shape} classes={t.n_classes} "
            f"extra={ {k: v[0].shape[0] for k, v in t.extra_tests.items()} }"
        )
