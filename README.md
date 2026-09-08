# SpectrumFM v2 (EMLM)

This branch is **SpectrumFM version 2**: a Memory Transformer pretrained with masked I/Q reconstruction, then adapted to downstream spectrum tasks with LoRA.

- Repository: [ChunyuLiu188/SpectrumFM](https://github.com/ChunyuLiu188/SpectrumFM)
- This code: branch [`v2`](https://github.com/ChunyuLiu188/SpectrumFM/tree/v2)
- Version 1 (JSAC 2025 paper code): branch [`main`](https://github.com/ChunyuLiu188/SpectrumFM/tree/main)

中文：这是 SpectrumFM 第二版（EMLM / Memory Transformer + LoRA）。第一版论文代码仍在 `main`，本目录是 `v2` 分支，仓库地址不变。

```bash
git clone https://github.com/ChunyuLiu188/SpectrumFM.git
cd SpectrumFM
git checkout v2
```

## What changed from v1

| | v1 (`main`) | v2 (`v2`) |
|---|---|---|
| Backbone | SpectrumFM encoder | Memory Transformer (RMT-style memory tokens) |
| Width | `d=256`, 4 heads, `d_ff=512` | `d=512`, 8 heads, `d_ff=2048` |
| Depth | 16 layers | 16 layers |
| Sequence | 128 symbols | 128 symbols (ManyTx uses 256 with split windows) |
| Mask | 15% | 30%, I and Q masked independently |
| Pretrain | masked reconstruction + next-slot prediction | masked reconstruction (DeepSpeed ZeRO-2, bf16) |
| Fine-tune | full / task heads | LoRA (`r=16`, `α=32`) on `qkv,out_proj,ff` |
| Tasks in this tree | AMC (see `main`) | AMC (`rml2016a` / `rml2016b`), WTC, ManyTx |

## Model

Burtsev / Recurrent Memory Transformer style: learnable memory tokens are prepended and appended to the I/Q sequence and attend bidirectionally. Masked I/Q samples are replaced by a learnable mask token (not dropped). Each sample is RMS-normalized before the network.

Default pretrain size is about **50M** parameters (`d_model=512`, 16 layers, 8 heads, 16 memory tokens).

## Layout

```
model.py            Memory Transformer
dataset.py          pretrain memmap IQ dataset + RMS
prepare_data.py     convert mixed radio files into training shards
train.py            DeepSpeed pretrain
config.yaml         pretrain hyperparameters
ds_config.json      DeepSpeed ZeRO-2 template
finetune.py         LoRA / full / linear-probe fine-tune (DDP)
finetune_data.py    RML2016.10a/b, WTC, ManyTx loaders
lora.py             LoRA inject / merge
eval_finetune.py    evaluate a saved run
baselines/          AWN, MCCFN, IQGCA, AvgNet, STF-GCN
run_pretrain.sh
run_finetune.sh
run_baselines.sh
```

Weights, raw datasets, and caches are **not** in this repo. Point the scripts at your own files (see below).

## Setup

```bash
pip install -r requirements.txt
```

PyTorch with CUDA should be installed for your GPU. Pretraining uses [DeepSpeed](https://www.deepspeed.ai/). Fine-tuning uses `torchrun`.

### Data paths

Put (or symlink) datasets under `./data`, or set environment variables:

| Variable | Default | Used for |
|---|---|---|
| `RADIO_ROOT` | `./data` | root of all fine-tune datasets |
| `EMLM_CACHE` | `./cache` | cached `.npz` splits |
| `WTC_MAT` | `$RADIO_ROOT/TecRec/finetune.mat` | WTC |
| `MANYTX_PKL` | `$RADIO_ROOT/ManyTx.pkl` | ManyTx |

Expected fine-tune files:

```
data/
  RML2016.10a_dict.pkl          # or data/RML2016/RML2016.10a_dict.pkl
  RML2016.10b.dat
  TecRec/finetune.mat
  ManyTx.pkl
```

Pretrain data: put mixed radio IQ files under `data.root` in `config.yaml` (default `./data`), then:

```bash
python prepare_data.py --config config.yaml
```

## Pretrain

```bash
bash run_pretrain.sh
# resume
bash run_pretrain.sh --resume ./ckpts
# smoke test
bash run_pretrain.sh --dummy --steps 20
```

| Hyperparameter | Value |
|---|---|
| Mask ratio | 0.3, independent I/Q |
| Sequence length | 128 |
| Precision | bf16 |
| DeepSpeed | ZeRO-2 |
| Optimizer | AdamW, peak lr `4e-4`, cosine to `1e-6`, 2000 warmup steps |
| Example batch | micro 512 × accum 2 × 4 GPUs = global 4096 |
| Epochs | 10 |

Edit `config.yaml` for GPU count / micro-batch. Checkpoints go to `./ckpts` (`best.pt`, `last.pt`).

## Fine-tune (LoRA)

```bash
# all four tasks, seeds 42/123/456, 6:2:2 split
CKPT=./ckpts/best.pt bash run_finetune.sh

# one task
TASKS=rml2016a SEEDS=42 CKPT=./ckpts/best.pt bash run_finetune.sh

# ManyTx on 2×24 GB GPUs (length 256; keep eval extras off-GPU)
TASKS=manytx NGPU=2 BS=32 EVAL_BS=64 CKPT=./ckpts/best.pt bash run_finetune.sh
```

Or directly:

```bash
torchrun --standalone --nproc_per_node=2 finetune.py \
  --task rml2016a --ckpt ./ckpts/best.pt --mode lora \
  --lora-r 16 --lora-alpha 32 --split 6:2:2 --epochs 20 \
  --lr 1e-3 --lr-decay-epoch 10 --lr-decay 0.1 --bs 256
```

| Setting | Default |
|---|---|
| Mode | LoRA on `qkv,out_proj,ff` |
| Rank / alpha | `r=16`, `α=32` |
| Split | `6:2:2`, stratified; `split_seed=0` shared across run seeds |
| Schedule | 20 epochs, lr `1e-3` then `×0.1` after epoch 10 |
| Pooling | mean over the sequence |
| ManyTx length | 256; `long_mode=split` (two 128-symbol windows, features averaged) |
| ManyTx days | train/val/test on day 1 (`2021_03_01`); extra eval on days 2/3/4 |
| WTC | `per_cell=9200` → 220,800 samples (3 classes × 8 SNRs), then 6:2:2 |

Outputs under `./runs/finetune/<task>/seed<seed>/`: `best.pt`, `last.pt`, `last_merged.pt`, `results.json`.

```bash
python eval_finetune.py ./runs/finetune/rml2016a/seed42
```

## Baselines

AWN, MCCFN, IQGCA, AvgNet, and STF-GCN use the **same** 6:2:2 splits and sample-RMS normalization as LoRA.

```bash
TASK=rml2016a bash run_baselines.sh
TASK=wtc bash run_baselines.sh
```

`baselines/train.py` currently registers `rml2016a`, `rml2016b`, and `wtc`. ManyTx baseline numbers in the paper experiments used the v1-style trainers on a 128-symbol crop (architectures are built for length 128).

## Reported fine-tune numbers (LoRA)

Same protocol: 6:2:2, three seeds `{42,123,456}`, test accuracy at the best validation epoch. Mean ± std.

| Task | SpectrumFM v2 LoRA | Strongest baseline in the same protocol |
|---|---|---|
| RML2016.10a | **63.86 ± 0.19%** | STFGCN 62.62% |
| RML2016.10b | **65.45 ± 0.21%** | STFGCN 65.25% |
| WTC | **72.07 ± 0.06%** | STFGCN 70.55% |
| ManyTx day 1 | **89.14 ± 0.08%** | AWN 87.31% |
| ManyTx day 2 / 3 / 4 | 32.46 / 30.72 / 28.11 | MCCFN 34.9 / 32.5 / 29.0 |

ManyTx cross-day remains hard for both LoRA and the CNN/GNN baselines (train on day 1 only, `equalized=0`).

## Cite

Please cite the SpectrumFM paper if you use this repository:

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
