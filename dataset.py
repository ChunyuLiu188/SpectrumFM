"""IQ dataset utilities: layout conversion, 128-chunking, sample-level RMS."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

SEQ_LEN_DEFAULT = 128
SKIP_SUFFIXES = {
    ".zip",
    ".7z",
    ".csv",
    ".txt",
    ".wav",
    ".docx",
    ".md",
    ".json",
    ".png",
    ".jpg",
}
IQ_SUFFIXES = {".mat", ".hdf5", ".h5", ".npy", ".npz", ".pkl", ".pickle"}
SPECIAL_NAMES = {"RML22.01A", "RML2016.10a", "RML2016.10b", "RML2016.04c"}
LABEL_KEYS = {
    "label",
    "labels",
    "y",
    "Y",
    "snr",
    "SNR",
    "mods",
    "modulation",
    "classes",
    "dumvar",
}


TIME_LIKE = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}


def rms_normalize(iq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Sample-level RMS over I and Q jointly. Prefer rms_normalize_torch in training."""
    rms = np.sqrt(np.mean(np.square(iq, dtype=np.float64), axis=(-2, -1), keepdims=True) + eps)
    return (iq / rms).astype(np.float32, copy=False)


def rms_normalize_torch(iq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Vectorized sample-level RMS on a batch [B, T, 2]. I and Q jointly, not batch stats."""
    orig = iq.dtype
    x = iq.float()
    rms = x.square().mean(dim=(-2, -1), keepdim=True).clamp_min(eps).sqrt()
    return (x / rms).to(dtype=orig)


def _as_float_iq(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if getattr(arr.dtype, "names", None) and {"real", "imag"} <= set(arr.dtype.names):
        arr = np.stack([arr["real"], arr["imag"]], axis=-1)
    if np.iscomplexobj(arr):
        arr = np.stack([arr.real, arr.imag], axis=-1)
    return np.asarray(arr, dtype=np.float32)


def to_nt2(arr: np.ndarray) -> np.ndarray:
    """Convert a raw array into (N, T, 2).

    Heuristics covering common radio IQ layouts under data.root:
      - rank-3 with a size-2 axis: that axis is I/Q
      - (N, 2, T) / (2, T) / (T, 2)
      - (256, N) MATLAB: stacked [I(128); Q(128)]
      - rank-1 even length: interleaved I/Q
      - otherwise the largest axis is treated as samples / time
    """
    arr = _as_float_iq(arr)
    if arr.size == 0:
        return np.zeros((0, 0, 2), dtype=np.float32)

    if arr.ndim == 1:
        n = arr.shape[0]
        if n % 2 == 0:
            return arr.reshape(1, n // 2, 2)
        pad = np.zeros(1, dtype=np.float32)
        return np.stack([arr, np.concatenate([arr[1:], pad])[:n]], axis=-1)[None]

    if arr.ndim == 2:
        a, b = arr.shape
        if a == 2:
            return np.transpose(arr, (1, 0))[None]  # (1, T, 2)
        if b == 2:
            return arr[None]
        # (T, N) vs (N, T): time-like dims are 128/256/512/1024...
        if a in TIME_LIKE and b not in TIME_LIKE:
            t, n = a, b
            x = arr  # (T, N)
        elif b in TIME_LIKE and a not in TIME_LIKE:
            n, t = a, b
            x = arr.T  # (T, N)
        elif a < b:
            t, n = a, b
            x = arr
        else:
            n, t = a, b
            x = arr.T
        if t == 256:
            # stacked I/Q of length 128, common in MATLAB pretrain/finetune.mat
            return np.stack([x[:128], x[128:]], axis=-1).transpose(1, 0, 2)
        if t in TIME_LIKE and t != 256:
            q = np.zeros((t, n), dtype=np.float32)
            return np.stack([x, q], axis=-1).transpose(1, 0, 2)
        if t % 2 == 0:
            return x.reshape(t // 2, 2, n).transpose(2, 0, 1)
        q = np.zeros((t, n), dtype=np.float32)
        return np.stack([x, q], axis=-1).transpose(1, 0, 2)

    if arr.ndim >= 3:
        while 1 in arr.shape and arr.ndim > 3:
            arr = np.squeeze(arr)
        ch_axes = [i for i, s in enumerate(arr.shape) if s == 2]
        if arr.ndim > 3 and ch_axes:
            ch = ch_axes[-1]
            rest = [i for i in range(arr.ndim) if i != ch]
            time_cands = [i for i in rest if arr.shape[i] in TIME_LIKE]
            time_axis = time_cands[-1] if time_cands else rest[-1]
            sample_axes = [i for i in rest if i != time_axis]
            x = np.transpose(arr, (*sample_axes, time_axis, ch))
            return np.ascontiguousarray(x.reshape(-1, x.shape[-2], 2), dtype=np.float32)
        if arr.ndim > 3:
            arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        shape = arr.shape
        ch_axes = [i for i, s in enumerate(shape) if s == 2]
        if ch_axes and arr.ndim == 3:
            ch = ch_axes[0] if len(ch_axes) == 1 else ch_axes[-1]
            axes = [i for i in range(3) if i != ch]
            a0, a1 = axes
            s0, s1 = shape[a0], shape[a1]
            if s0 in TIME_LIKE and s1 not in TIME_LIKE:
                time_axis, sample_axis = a0, a1
            elif s1 in TIME_LIKE and s0 not in TIME_LIKE:
                time_axis, sample_axis = a1, a0
            elif s0 in TIME_LIKE and s1 in TIME_LIKE:
                time_axis, sample_axis = (a0, a1) if s0 >= s1 else (a1, a0)
            else:
                sample_axis = a0 if s0 >= s1 else a1
                time_axis = a1 if sample_axis == a0 else a0
            x = np.transpose(arr, (sample_axis, time_axis, ch))
            return np.ascontiguousarray(x, dtype=np.float32)
        t = shape[-1]
        n = int(np.prod(shape[:-1]))
        return to_nt2(arr.reshape(n, t))

    raise ValueError(f"unsupported IQ array shape {arr.shape}")


def chunk_nt2(iq: np.ndarray, seq_len: int = SEQ_LEN_DEFAULT) -> np.ndarray:
    """Non-overlapping chunk along time. Drop the remainder. (N, T, 2) -> (N', L, 2)."""
    if iq.ndim != 3 or iq.shape[-1] != 2:
        raise ValueError(f"expected (N, T, 2), got {iq.shape}")
    n, t, _ = iq.shape
    if t == seq_len:
        return np.ascontiguousarray(iq, dtype=np.float32)
    n_chunks = t // seq_len
    if n_chunks <= 0:
        return np.zeros((0, seq_len, 2), dtype=np.float32)
    iq = iq[:, : n_chunks * seq_len]
    iq = iq.reshape(n, n_chunks, seq_len, 2).reshape(n * n_chunks, seq_len, 2)
    return np.ascontiguousarray(iq, dtype=np.float32)


def sample_axis_of(shape: Tuple[int, ...]) -> int:
    """Largest axis is treated as the sample/batch axis for slicing big files."""
    return int(np.argmax(shape))


class ShardIQDataset(Dataset):
    """Reads prepare_data.py shards: float32 memmap arrays of shape (N, L, 2)."""

    def __init__(self, shard_paths: Sequence[Path], rms: bool = False) -> None:
        self.paths = [Path(p) for p in shard_paths]
        self.rms = rms
        self._mms: List[np.ndarray] = []
        self.lengths: List[int] = []
        for p in self.paths:
            mm = np.load(p, mmap_mode="r")
            if mm.ndim != 3 or mm.shape[-1] != 2:
                raise ValueError(f"{p} has shape {mm.shape}, expected (N, L, 2)")
            self._mms.append(mm)
            self.lengths.append(int(mm.shape[0]))
        self.cum = np.cumsum([0] + self.lengths)

    def __len__(self) -> int:
        return int(self.cum[-1])

    def _loc(self, idx: int) -> Tuple[int, int]:
        shard = int(np.searchsorted(self.cum, idx, side="right") - 1)
        return shard, idx - int(self.cum[shard])

    def __getitem__(self, idx: int) -> torch.Tensor:
        si, li = self._loc(idx)
        iq = np.array(self._mms[si][li], dtype=np.float32, copy=True)
        if self.rms:
            iq = rms_normalize(iq)
        return torch.from_numpy(iq)

    def get_block(self, start: int, size: int) -> torch.Tensor:
        """Contiguous samples [start, start+size) as one (size, L, 2) tensor.

        One memmap slice per touched shard instead of `size` random reads.
        """
        end = min(start + size, len(self))
        parts: List[np.ndarray] = []
        pos = start
        while pos < end:
            si, li = self._loc(pos)
            take = min(end - pos, self.lengths[si] - li)
            parts.append(np.asarray(self._mms[si][li : li + take], dtype=np.float32))
            pos += take
        # np.array(copy=True): materialize the memmap slice into a private, writable
        # buffer (torch.from_numpy on a read-only memmap view is undefined behaviour)
        arr = np.array(parts[0], dtype=np.float32) if len(parts) == 1 else np.concatenate(parts, axis=0)
        if self.rms:
            arr = rms_normalize(arr)
        return torch.from_numpy(arr)


class BlockIQDataset(ShardIQDataset):
    """Each item is a whole micro-batch: `block_size` contiguous pre-shuffled samples.

    prepare_data.py globally shuffles every chunk before writing shards, so a
    contiguous block is already an i.i.d. mix of all source datasets. Use with
    DataLoader(batch_size=None, sampler=BlockSampler(...)).
    """

    def __init__(self, shard_paths: Sequence[Path], block_size: int, rms: bool = False) -> None:
        super().__init__(shard_paths, rms=rms)
        self.block_size = int(block_size)

    def __getitem__(self, start: int) -> torch.Tensor:  # type: ignore[override]
        return self.get_block(int(start), self.block_size)


class BlockSampler(Sampler[int]):
    """Yields block start offsets for BlockIQDataset, sharded across ranks.

    Per epoch: a random grid offset in [0, block_size) so batch composition
    changes between epochs, then shuffled block order. Every rank gets the same
    number of blocks (drop_last semantics) so collective ops stay in lock-step.
    """

    def __init__(
        self,
        n_samples: int,
        block_size: int,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        multiple_of: int = 1,
    ) -> None:
        self.n = int(n_samples)
        self.bs = int(block_size)
        self.world = int(world_size)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0
        n_blocks = (self.n - (self.bs - 1)) // self.bs  # worst-case offset keeps blocks full
        per_rank = n_blocks // self.world
        # keep per-rank micro-batches a multiple of grad_accum so an epoch is a whole
        # number of optimizer steps (no accumulation bleeding across epochs)
        self.per_rank = max(0, per_rank // max(1, multiple_of) * max(1, multiple_of))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.per_rank

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        off = int(rng.integers(0, self.bs)) if self.shuffle else 0
        starts = np.arange(off, self.n - self.bs + 1, self.bs, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(starts)
        starts = starts[: self.per_rank * self.world]
        return iter(starts[self.rank :: self.world].tolist())


class DummyIQDataset(Dataset):
    def __init__(self, n: int = 2048, seq_len: int = 128) -> None:
        self.n = n
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(idx)
        i = torch.randn(self.seq_len, generator=g)
        q = torch.randn(self.seq_len, generator=g)
        return torch.stack([i, q], dim=-1)


def load_manifest(cache_dir: Path) -> dict:
    path = Path(cache_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"cache manifest not found: {path}. Run prepare_data.py first.")
    return json.loads(path.read_text())


def dataset_from_cache(cache_dir: Path, split: str = "train", rms: bool = False) -> ShardIQDataset:
    man = load_manifest(cache_dir)
    paths = [Path(p) for p in man[split]]
    if not paths:
        raise RuntimeError(f"no {split} shards in {cache_dir}")
    return ShardIQDataset(paths, rms=rms)


def block_dataset_from_cache(cache_dir: Path, split: str, block_size: int, rms: bool = False) -> BlockIQDataset:
    man = load_manifest(cache_dir)
    paths = [Path(p) for p in man[split]]
    if not paths:
        raise RuntimeError(f"no {split} shards in {cache_dir}")
    return BlockIQDataset(paths, block_size=block_size, rms=rms)


def is_download_incomplete(path: Path) -> bool:
    flag = path.parent / f"{path.name}.BaiduPCS-Go-downloading"
    return flag.exists()


def discover_raw_files(root: Path, skip_names: Sequence[str] = (), skip_suffixes: Sequence[str] = ()) -> List[Path]:
    root = Path(root)
    skip_names_l = {n.lower() for n in skip_names}
    skip_suf = {s.lower() if s.startswith(".") else f".{s.lower()}" for s in skip_suffixes}
    skip_suf |= SKIP_SUFFIXES
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name.endswith(".BaiduPCS-Go-downloading"):
            continue
        if is_download_incomplete(p):
            continue
        if p.name.lower() in skip_names_l:
            continue
        if p.name.startswith("Y_") or p.name.startswith("lbl_"):
            continue
        suf = p.suffix.lower()
        if suf in skip_suf:
            continue
        if suf in IQ_SUFFIXES or p.name in SPECIAL_NAMES:
            out.append(p)
    return out


def _iter_numpy(path: Path) -> Iterator[np.ndarray]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        for k in data.files:
            if k.lower() in LABEL_KEYS:
                continue
            arr = data[k]
            if np.asarray(arr).size >= 128:
                yield np.asarray(arr)
        return
    yield np.load(path, allow_pickle=True)


def _iter_pickle(path: Path) -> Iterator[np.ndarray]:
    with open(path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key in LABEL_KEYS:
                continue
            arr = np.asarray(v)
            if arr.size >= 128:
                yield arr
        return
    yield np.asarray(obj)


def _h5_should_skip(name: str, shape: Tuple[int, ...], dtype) -> bool:
    low = name.lower().split("/")[-1]
    if low in LABEL_KEYS:
        return True
    if dtype is None:
        return True
    dt = np.dtype(dtype)
    if dt.kind not in "cbiuf":
        return True
    if int(np.prod(shape)) < 128:
        return True
    if "#refs#" in name or name.startswith("#") or "refs" in low:
        return True
    return False


def _iter_h5_dataset(ds, batch: int = 8192) -> Iterator[np.ndarray]:
    shape = tuple(ds.shape)
    if len(shape) == 1:
        yield np.asarray(ds[()], dtype=np.float32)
        return
    if len(shape) == 0:
        return
    axis = sample_axis_of(shape)
    n = shape[axis]
    # many tiny vectors stored as one dataset already handled; slice the long axis
    if n <= batch or n < 256:
        yield np.asarray(ds[()])
        return
    slicer = [slice(None)] * len(shape)
    for start in range(0, n, batch):
        slicer[axis] = slice(start, min(start + batch, n))
        yield np.asarray(ds[tuple(slicer)])


def _iter_hdf5(path: Path, batch: int = 8192) -> Iterator[np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        items: List[Tuple[str, object]] = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                items.append((name, obj))

        f.visititems(visitor)
        small_names = set()
        small: List[Tuple[str, object]] = []
        large: List[Tuple[str, object]] = []
        for n, obj in items:
            if not isinstance(obj, h5py.Dataset):
                continue
            if _h5_should_skip(n, tuple(obj.shape), obj.dtype):
                continue
            if obj.ndim <= 1 and int(np.prod(obj.shape)) <= 4096:
                small.append((n, obj))
                small_names.add(n)
            else:
                large.append((n, obj))

        # RadarComm-style: many (256,) vectors -> stack to (N, 256)
        buf: List[np.ndarray] = []
        for _, obj in small:
            arr = np.asarray(obj[()], dtype=np.float32).reshape(-1)
            buf.append(arr)
            if len(buf) >= batch:
                yield np.stack(buf, axis=0)
                buf = []
        if buf:
            yield np.stack(buf, axis=0)

        for _, obj in large:
            for sl in _iter_h5_dataset(obj, batch=batch):
                yield sl


def _iter_mat(path: Path, batch: int = 8192) -> Iterator[np.ndarray]:
    # MATLAB v7.3 is hdf5
    with open(path, "rb") as f:
        magic = f.read(16)
    if magic.startswith(b"MATLAB 7.3") or magic.startswith(b"\x89HDF"):
        yield from _iter_hdf5(path, batch=batch)
        return
    from scipy.io import loadmat

    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    for k, v in mat.items():
        if k.startswith("__") or k.lower() in LABEL_KEYS:
            continue
        arr = np.asarray(v)
        if arr.size >= 128:
            yield arr


def iter_raw_arrays(path: Path, batch: int = 8192) -> Iterator[np.ndarray]:
    suf = path.suffix.lower()
    name = path.name
    if suf in {".npy", ".npz"}:
        yield from _iter_numpy(path)
    elif suf in {".pkl", ".pickle"} or name in SPECIAL_NAMES:
        try:
            yield from _iter_pickle(path)
        except Exception:
            if suf in {".mat", ".hdf5", ".h5"}:
                yield from _iter_hdf5(path, batch=batch)
            else:
                raise
    elif suf in {".hdf5", ".h5"}:
        yield from _iter_hdf5(path, batch=batch)
    elif suf == ".mat":
        yield from _iter_mat(path, batch=batch)
    else:
        # extensionless pickle (RML22.01A)
        try:
            yield from _iter_pickle(path)
        except Exception as exc:
            raise RuntimeError(f"cannot load {path}: {exc}") from exc


def arrays_to_chunks(arr: np.ndarray, seq_len: int = SEQ_LEN_DEFAULT) -> np.ndarray:
    return chunk_nt2(to_nt2(arr), seq_len=seq_len)


def is_hdf_container(path: Path) -> bool:
    suf = path.suffix.lower()
    if suf in {".hdf5", ".h5"}:
        return True
    if suf != ".mat":
        return False
    with open(path, "rb") as f:
        magic = f.read(16)
    return magic.startswith(b"MATLAB 7.3") or magic.startswith(b"\x89HDF")


def plan_h5_slices(path: Path, n_workers: int, min_per_slice: int = 2048) -> Optional[List[Tuple[str, int, int, int]]]:
    """Split large HDF5/MAT v7.3 datasets along the sample axis.

    Returns [(dataset_name, axis, start, end), ...] or None to fall back to
    sequential iter_raw_arrays (RadarComm-style many tiny datasets).
    """
    if n_workers <= 1 or not is_hdf_container(path):
        return None
    import h5py

    large: List[Tuple[str, int, int]] = []
    small_n = 0
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            nonlocal small_n
            if not isinstance(obj, h5py.Dataset):
                return
            shape = tuple(obj.shape)
            if _h5_should_skip(name, shape, obj.dtype):
                return
            if obj.ndim <= 1 and int(np.prod(shape)) <= 4096:
                small_n += 1
                return
            axis = sample_axis_of(shape)
            large.append((name, axis, int(shape[axis])))

        f.visititems(visitor)

    if small_n > 16 or not large:
        return None

    jobs: List[Tuple[str, int, int, int]] = []
    for name, axis, n in large:
        n_parts = min(n_workers, max(1, n // min_per_slice))
        if n_parts <= 1:
            jobs.append((name, axis, 0, n))
            continue
        step = (n + n_parts - 1) // n_parts
        for i in range(n_parts):
            start = i * step
            end = min((i + 1) * step, n)
            if start < end:
                jobs.append((name, axis, start, end))
    return jobs if len(jobs) > 1 else None


def load_h5_slice(path: Path, dset: str, axis: int, start: int, end: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        ds = f[dset]
        slicer = [slice(None)] * ds.ndim
        slicer[axis] = slice(start, end)
        return np.asarray(ds[tuple(slicer)])
