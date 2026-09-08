# SpectrumFM v2 (EMLM)

**English** | [中文](README_zh.md)

This branch is **SpectrumFM version 2**: a Memory Transformer pretrained with masked I/Q reconstruction, then adapted to downstream spectrum tasks with LoRA.

- Repository: [ChunyuLiu188/SpectrumFM](https://github.com/ChunyuLiu188/SpectrumFM)
- This code: branch [`v2`](https://github.com/ChunyuLiu188/SpectrumFM/tree/v2)
- Version 1 (JSAC 2025 paper code): branch [`main`](https://github.com/ChunyuLiu188/SpectrumFM/tree/main)

```bash
git clone https://github.com/ChunyuLiu188/SpectrumFM.git
cd SpectrumFM
git checkout v2
```

---

## 1. What changed from v1

| | v1 (`main`) | v2 (`v2`) |
|---|---|---|
| Backbone | SpectrumFM encoder | Memory Transformer (RMT-style memory tokens) |
| Width | `d=256`, 4 heads, `d_ff=512` | `d=512`, 8 heads, `d_ff=2048` |
| Depth | 16 layers | 16 layers |
| Sequence | 128 symbols | 128 symbols (ManyTx input is 256; see long sequences) |
| Normalization | often `iq2ap` in the v1 AMC path | **sample-level RMS only** (I and Q jointly) |
| Mask | 15% | 30%, I and Q masked independently |
| Pretrain | masked reconstruction + next-slot prediction | **masked reconstruction only** (DeepSpeed ZeRO-2, bf16) |
| Fine-tune | full / task heads | LoRA (`r=16`, `α=32`) on `qkv`, `out_proj`, `ff` |
| Tasks here | AMC (see `main`) | AMC (`rml2016a` / `rml2016b`), WTC, ManyTx |

---

## 2. Normalization (important)

Every waveform that enters the network is stored as raw float32 IQ `(L, 2)` and **normalized on the fly**, never written back to disk.

**Sample-level RMS** (`dataset.rms_normalize` / `rms_normalize_torch`):

```
rms = sqrt( mean(I^2 + Q^2 over the L time steps) + 1e-8 )
I' = I / rms
Q' = Q / rms
```

Properties:

- **Per sample, not per batch.** Batch statistics are never used (no BatchNorm-style running mean).
- **I and Q share one scale.** The complex envelope energy of that one burst is 1. Relative I/Q phase and the I/Q ratio are unchanged.
- **Not `iq2ap`.** We do **not** convert to instantaneous amplitude/phase. That was a v1 AMC option and is not used here.
- **Not dataset-wide z-score.** Different SNRs / radios keep their own relative energy after per-sample scaling.
- **`prepare_data.py` does not normalize.** Shards stay raw. The GPU batch is RMS-normalized in `train.to_engine_tensor` (pretrain) and in `IQClassifier.features` (fine-tune / baselines).

For sequences longer than the pretrain length (ManyTx `L=256`, model `seq_len=128`, `long_mode=split`):

1. Cut the raw burst into non-overlapping 128-sample windows.
2. RMS-normalize **each window by itself** (same as a pretrain chunk).
3. Encode each window, pool, then **average the window features**.

So a 256-sample fingerprint is two independently normalized 128-chunks, not one RMS over 256.

**Input contract for inference:** pass **raw** `(B, L, 2)` float32. `eval_finetune.py` and `IQClassifier` apply RMS internally. Do not pre-normalize, or you will double-scale.

### 2.1 Normalization ablation (11 schemes)

A dedicated study measures how the **input representation and the scope of the statistics** change downstream transfer. This is part of the v2 contribution: the same Memory Transformer and the same LoRA recipe, only the normalizer changes.

**Protocol (keep this separate from the full-run table in §10):**

| Item | Ablation setting |
|---|---|
| Pretrain data | RML2018.01A only |
| Pretrain length | **1 epoch** |
| Fine-tune | LoRA, same architecture and training config for every row |
| AMC | RML2016.10a |
| WTC | stratified **10k-sample** subset (not the 220.8k `per_cell=9200` protocol) |
| RFFI | WiSig **ManySig** |
| Seeds | three independent run seeds; report mean ± std |

**Scope.** *None* = no learned/computed scale. *Sample-level* = statistics from that one burst only (same idea as the released RMS). *Dataset-level* = one set of statistics from the training corpus, applied to every burst.

**Input.** *IQ* keeps in-phase / quadrature. *AP* converts each sample to instantaneous amplitude and phase (`A = |I+jQ|`, `P = atan2(Q, I)`).

Let a burst be `x ∈ R^{L×2}` (IQ) or the AP pair of the same length. Sample-level maps use only that burst; dataset-level maps use train-set min / mean / std / RMS.

| Operator | Formula (per coordinate, shared stats over I and Q unless noted) |
|---|---|
| Raw | identity |
| min–max | `(x − min) / (max − min + ε)` |
| *z*-score | `(x − mean) / (std + ε)` |
| RMS-power | `x / sqrt(mean(I²+Q²) + ε)` |
| max-abs | `x / (max|x| + ε)` |

**Table. Normalization ablation after one-epoch pretraining.** Accuracy (%), mean ± std over three seeds.

| Scope | Input / normalization | AMC (%) | WTC (%) | RFFI (%) |
|---|---|---:|---:|---:|
| None | Raw IQ | 9.09 ± 0.00 | 67.72 ± 2.36 | 82.34 ± 10.75 |
| None | Raw AP | 52.80 ± 2.78 | 72.87 ± 5.34 | 96.46 ± 0.74 |
| Sample-level | IQ min–max | 56.06 ± 0.12 | 51.73 ± 4.33 | 93.13 ± 1.20 |
| Sample-level | IQ *z*-score | 62.53 ± 0.07 | 64.03 ± 1.18 | 98.53 ± 0.11 |
| Sample-level | **IQ RMS-power** | **63.65 ± 0.20** | 65.93 ± 0.67 | 98.40 ± 0.11 |
| Sample-level | IQ max-abs | 63.14 ± 0.42 | 65.25 ± 0.23 | 98.31 ± 0.15 |
| Sample-level | AP min–max | 63.32 ± 0.07 | 66.05 ± 0.44 | 98.34 ± 0.41 |
| Sample-level | AP *z*-score | 63.50 ± 0.17 | **66.37 ± 0.73** | **98.68 ± 0.10** |
| Dataset-level | IQ min–max | 47.79 ± 10.41 | 72.57 ± 1.85 | 86.63 ± 1.24 |
| Dataset-level | IQ *z*-score | 59.79 ± 0.81 | 74.28 ± 1.01 | 96.89 ± 0.28 |
| Dataset-level | IQ RMS-power | 52.49 ± 2.77 | 75.02 ± 3.68 | 83.39 ± 14.76 |

**What the table shows.**

- **AMC needs a per-burst scale.** Raw IQ is chance (`9.09%`, 1/11). Raw AP already recovers a lot (`52.80%`) because amplitude/phase removes some gain. Sample-level IQ/AP normalizers then sit in a tight band around 63%.
- **IQ RMS-power is the best AMC scheme** (`63.65 ± 0.20%`) and stays close to the top on WTC / RFFI. That is why the released pretrain + LoRA path uses it (`dataset.rms_normalize_torch`).
- **AP *z*-score** is strongest on the 1-epoch WTC subset and on ManySig RFFI, but only barely above RMS. We keep RMS as the default so pretrain and fine-tune share one real-valued IQ scale (no `atan2`, no dataset-wide stats).
- **Dataset-level maps are weaker and less stable** on AMC (min–max `47.79 ± 10.41%`; RMS-power `52.49 ± 2.77%`). A global gain mixes SNRs and radios; sample-level RMS does not.

The full-run numbers in §10 (10-epoch mixed pretrain, WTC 220.8k, ManyTx) are **not** this 1-epoch table. They use sample-level IQ RMS-power only.

---

## 3. Model

Burtsev / Recurrent Memory Transformer style (`model.py`).

**Input after RMS:** `iq` of shape `[B, L, 2]`.

**Embedding.** I and Q go through separate linear maps (`i_proj`, `q_proj`), then add. If a coordinate is masked, it is **replaced by a learnable mask token** (`mask_token_i` / `mask_token_q`) instead of being dropped, so the position still attends and is attended to. A learned `pos_embed` of length 128 is added.

**Memory tokens.** A learned bank of `n_memory=16` tokens is prepended **and** appended:

```
[ mem_read (16) | IQ tokens (L) | mem_write (16) ]
```

Full bidirectional attention over this sequence. Default size: `d_model=512`, 16 layers, 8 heads, `d_ff=2048`, dropout 0 in pretrain (~50M parameters).

**Pretrain head.** A linear map `d_model → 2` reconstructs I and Q at every time step. The reconstruction loss is computed **only on masked coordinates**.

**Fine-tune head.** No reconstruction. Pool the IQ tokens (default: mean) → LayerNorm → Dropout → Linear → logits.

---

## 4. Repository layout

```
model.py            Memory Transformer
dataset.py          IQ loaders, chunking, sample RMS
prepare_data.py     convert mixed radio files into raw training shards
train.py            DeepSpeed pretrain
config.yaml         pretrain hyperparameters
ds_config.json      DeepSpeed ZeRO-2 template
finetune.py         LoRA / full / linear-probe (DDP)
finetune_data.py    RML2016.10a/b, WTC, ManyTx
lora.py             LoRA inject / merge
eval_finetune.py    evaluate a saved run
baselines/          AWN, MCCFN, IQGCA, AvgNet, STF-GCN
ckpts/best.pt       released pretrained backbone
runs/finetune/      released LoRA adapters + results.json
run_pretrain.sh
run_finetune.sh
run_baselines.sh
```

Raw datasets and memmap caches are **not** in git. Released weights are (see §8).

---

## 5. Setup

```bash
pip install -r requirements.txt
```

Install a CUDA build of PyTorch for your GPU. Pretraining uses [DeepSpeed](https://www.deepspeed.ai/). Fine-tuning uses `torchrun`.

### Data paths

Put (or symlink) files under `./data`, or set environment variables.

| Variable | Default | Used for |
|---|---|---|
| `RADIO_ROOT` | `./data` | root of fine-tune datasets |
| `EMLM_CACHE` | `./cache` | cached `.npz` after the first load |
| `WTC_MAT` | `$RADIO_ROOT/TecRec/finetune.mat` | WTC |
| `MANYTX_PKL` | `$RADIO_ROOT/ManyTx.pkl` | ManyTx |

Fine-tune files:

```
data/
  RML2016.10a_dict.pkl          # or data/RML2016/RML2016.10a_dict.pkl
  RML2016.10b.dat
  TecRec/finetune.mat
  ManyTx.pkl
```

Pretrain files: any mix of radio IQ that `dataset.discover_raw_files` understands, under `data.root` in `config.yaml` (default `./data`). Then:

```bash
python prepare_data.py --config config.yaml
```

This extracts archives if needed, cuts waveforms into length-128 chunks, and writes shuffled memmap shards to `./cache/pretrain`. **RMS is not applied here.**

---

## 6. Pretraining

Self-supervised **masked I/Q reconstruction** only (no next-slot prediction in v2).

```bash
bash run_pretrain.sh                       # fresh
bash run_pretrain.sh --resume ./ckpts      # continue
bash run_pretrain.sh --dummy --steps 20    # smoke test
```

### Forward / loss

For each RMS-normalized batch `iq` of shape `[B, 128, 2]`:

1. **Independent I/Q mask** (`independent_iq_mask: true`). Each of the `128×2` scalars is dropped with probability `mask_ratio=0.3` (Bernoulli). I and Q at the same time index are **not** tied. If a sample would be fully visible, one coordinate is forced masked so the loss is defined.
2. Masked values are replaced by mask tokens; memory tokens are attached; 16 Transformer layers run.
3. Loss is **masked MSE**: mean of `(pred − iq)²` on masked coordinates only (`model.masked_mse`). Visible coordinates do not contribute.

### Default hyperparameters (`config.yaml`)

| Item | Value |
|---|---|
| Sequence length | 128 |
| Mask ratio | 0.3, independent I/Q |
| Precision | bf16 |
| DeepSpeed | ZeRO-2 (`ds_config.json`) |
| Optimizer | AdamW, `lr=4e-4` peak, cosine to `1e-6`, 2000 warmup steps, `wd=0.01` |
| Example batch | micro 512 × accum 2 × 4 GPUs → global 4096 (edit for your GPU) |
| Epochs | 10 |
| Dropout / drop-path | 0 / 0 (turned back on in fine-tune) |
| `torch.compile` | on |
| Checkpoints | `./ckpts/best.pt` (best val MSE), `./ckpts/last.pt` |

Val is a 1% hold-out (`val_ratio: 0.01`). `best.pt` is the lowest val MSE.

---

## 7. Fine-tuning

The backbone stays mostly frozen. LoRA adapters + a new classification head are trained with DDP (`finetune.py`).

```bash
# four tasks × seeds 42/123/456, split 6:2:2
CKPT=./ckpts/best.pt bash run_finetune.sh

TASKS=rml2016a SEEDS=42 CKPT=./ckpts/best.pt bash run_finetune.sh

# ManyTx on 2×24 GB (L=256; extras stay on CPU)
TASKS=manytx NGPU=2 BS=32 EVAL_BS=64 CKPT=./ckpts/best.pt bash run_finetune.sh
```

```bash
torchrun --standalone --nproc_per_node=2 finetune.py \
  --task rml2016a --ckpt ./ckpts/best.pt --mode lora \
  --lora-r 16 --lora-alpha 32 --split 6:2:2 --epochs 20 \
  --lr 1e-3 --lr-decay-epoch 10 --lr-decay 0.1 --bs 256
```

### LoRA

`y = Wx + (α/r) B A x` on selected `nn.Linear`s (`lora.py`). Base `W` is frozen.

| Item | Released protocol |
|---|---|
| Mode | `--mode lora` |
| Rank / alpha / dropout | `r=16`, `α=32`, 0.05 |
| Targets | `qkv`, `out_proj`, `ff` (attention + FFN) |
| Frozen | embeddings, memory tokens, LayerNorm (unless `--train-extra`) |
| Trained | LoRA A/B + classification head (~2.1M / 52.6M ≈ 4%) |
| Pooling | `--pool mean` over IQ tokens |
| Head | LayerNorm → Dropout 0.1 → Linear |
| Backbone dropout | 0.1 during fine-tune |

Other modes: `--mode full` (train everything), `--mode linear` (head only).

### Split protocol (must match to reproduce numbers)

`--split 6:2:2` and `--split-seed 0` (the seed that shuffles data is **not** the run seed):

1. Stratify on class, and on SNR when SNR exists (RML / WTC).
2. First take 80% as a train+val pool and 20% as test (`finetune_data.stratified_split`).
3. Then split that pool 75/25 → **60% train / 20% val / 20% test**.
4. `--seed` (42 / 123 / 456) only changes init and shuffle, **not** who is in train/val/test.

`best.pt` is selected on **validation** accuracy. Reported numbers are **test accuracy at that epoch**.

### Tasks

**RML2016.10a (`rml2016a`).** 11 modulations, DeepSig pickle. 220,000 bursts, length 128. After 6:2:2: 132k / 44k / 44k. Per-SNR accuracy is logged.

**RML2016.10b (`rml2016b`).** 10 modulations, length 128. Same split rule.

**WTC (`wtc`).** `TecRec/finetune.mat` (MATLAB v7.3): `out` is `(256, N)` stacked `[I(128); Q(128)]`, `label` is `(2, N)` = `[class; snr]`. Classes: WiFi / LTE / ZigBee. The raw file has ~2.06M columns. We first subsample **`per_cell=9200` per (class, SNR)** with RNG `split_seed+7` → 3 × 8 × 9200 = **220,800** samples (same scale as RML2016.10a), then 6:2:2 → 132,480 / 44,160 / 44,160.

**ManyTx (`manytx`).** WiSig RF fingerprint pickle, **length 256**, 150 TX ids. Train only on day 1 (`2021_03_01`), `equalized=0` (raw, not equalized). Split is stratified **per TX** on day 1. Days 2 / 3 / 4 (`2021_03_08`, `2021_03_15`, `2021_03_23`) are extra test sets (domain shift). `long_mode=split` as in §2.

### Optimization

| Item | Value |
|---|---|
| Epochs | 20 |
| LR | `1e-3` for epochs 1–10, then `×0.1` → `1e-4` |
| Weight decay | 0.01 |
| Label smoothing | 0 |
| Grad clip | 1.0 |
| AMP | bf16 unless `--no-amp` |
| Augment | off by default (`--aug-phase` / `--aug-flip`; phase aug hurts fingerprinting) |

Outputs in `./runs/finetune/<task>/seed<seed>/`:

| File | Contents |
|---|---|
| `best.pt` | trainable tensors only (LoRA A/B + head) at best val epoch — **this is what we release** |
| `last.pt` | same, after the final epoch |
| `last_merged.pt` | full backbone+head with LoRA folded in (~193 MB, **not** on GitHub) |
| `results.json` | test / extra-day metrics |

```bash
python eval_finetune.py ./runs/finetune/rml2016a/seed42
```

`eval_finetune` rebuilds the same split, loads `ckpts/best.pt` + the adapter, RMS-normalizes raw IQ, and reprints the report.

---

## 8. Released weights

On this `v2` branch after `git checkout v2`:

| File | What | Size |
|---|---|---|
| [`ckpts/best.pt`](ckpts/best.pt) | pretrained Memory Transformer | ~97 MB |
| [`runs/finetune/<task>/seed<seed>/best.pt`](runs/finetune) | LoRA + head at best val | ~8 MB each |
| `runs/finetune/<task>/seed<seed>/results.json` | metrics for that seed | small |

Tasks: `rml2016a`, `rml2016b`, `wtc`, `manytx`. Seeds: `42`, `123`, `456`.

`runs/finetune/.../best.pt` is **not** a standalone model. Load it with `ckpts/best.pt`. Merged 193 MB files exceed GitHub’s 100 MB limit and are not uploaded.

---

## 9. Baselines

AWN, MCCFN, IQGCA, AvgNet, STF-GCN use the **same** 6:2:2 split seed and the **same sample-RMS** (no `iq2ap`).

```bash
TASK=rml2016a bash run_baselines.sh
TASK=wtc bash run_baselines.sh
```

`baselines/train.py` registers `rml2016a`, `rml2016b`, `wtc`. ManyTx baseline numbers below used the v1-style trainers on a **128-symbol crop** (those CNNs/GNNs are built for length 128). LoRA ManyTx stays at 256 with split windows.

---

## 10. Reported numbers (LoRA)

Protocol: 6:2:2, seeds `{42,123,456}`, test accuracy at the best validation epoch. Mean ± std.

| Task | SpectrumFM v2 LoRA | Strongest baseline, same protocol |
|---|---|---|
| RML2016.10a | **63.86 ± 0.19%** | STFGCN 62.62% |
| RML2016.10b | **65.45 ± 0.21%** | STFGCN 65.25% |
| WTC | **72.07 ± 0.06%** | STFGCN 70.55% |
| ManyTx day 1 | **89.14 ± 0.08%** | AWN 87.31% |
| ManyTx day 2 / 3 / 4 | 32.46 / 30.72 / 28.11 | MCCFN 34.9 / 32.5 / 29.0 |

ManyTx cross-day is hard for both LoRA and the CNN/GNN baselines (train day 1 only, `equalized=0`).

---

## 11. Cite

```bibtex
@ARTICLE{11301740,
  author={Zhou, Fuhui and Liu, Chunyu and Zhang, Hao and Wu, Wei and Wu, Qihui and Quek, Tony Q. S. and Chae, Chan-Byoung},
  journal={IEEE Journal on Selected Areas in Communications},
  title={SpectrumFM: A Foundation Model for Intelligent Spectrum Management},
  year={2025},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSAC.2025.3644783}
}
```

## 12. Contributors

GitHub lists extra authors from the `Co-authored-by` line on each commit. This `v2` branch uses:

```
Co-authored-by: Cursor <cursoragent@cursor.com>
```

so [cursoragent](https://github.com/cursoragent) appears on the [Contributors](https://github.com/ChunyuLiu188/SpectrumFM/graphs/contributors) page together with the repo owner.

| | Role |
|---|---|
| [Chunyu Liu](https://github.com/ChunyuLiu188) | Research, model, experiments, paper |
| [Cursor](https://cursor.com) ([cursoragent](https://github.com/cursoragent)) | v2 code release, docs, packaging |
