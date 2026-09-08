# SpectrumFM v2（EMLM）

[English](README.md) | **中文**

本分支是 **SpectrumFM 第二版**：用掩码 I/Q 重构预训练 Memory Transformer，再用 LoRA 适配下游频谱任务。

- 仓库：[ChunyuLiu188/SpectrumFM](https://github.com/ChunyuLiu188/SpectrumFM)
- 本代码：分支 [`v2`](https://github.com/ChunyuLiu188/SpectrumFM/tree/v2)
- 第一版（JSAC 2025 论文代码）：分支 [`main`](https://github.com/ChunyuLiu188/SpectrumFM/tree/main)

```bash
git clone https://github.com/ChunyuLiu188/SpectrumFM.git
cd SpectrumFM
git checkout v2
```

管线：原始 IQ → **按样本 RMS 归一化** → Memory Transformer **掩码重构预训练** → **LoRA 微调**（AMC / WTC / ManyTx）。第 2 节是归一化公式；第 2.1 节是 **11 种归一化消融**（本文贡献之一）。发布默认方案是 **样本级 IQ RMS-power**。

---

## 1. 相对 v1 的变化

| | v1（`main`） | v2（`v2`） |
|---|---|---|
| 骨干 | SpectrumFM 编码器 | Memory Transformer（RMT 风格 memory token） |
| 宽度 | `d=256`，4 头，`d_ff=512` | `d=512`，8 头，`d_ff=2048` |
| 深度 | 16 层 | 16 层 |
| 序列长度 | 128 符号 | 128 符号（ManyTx 输入为 256，见长序列处理） |
| 归一化 | v1 AMC 路径常用 `iq2ap` | **仅样本级 RMS**（I、Q 共用一个尺度） |
| 掩码 | 15% | 30%，I 与 Q **独立**掩码 |
| 预训练 | 掩码重构 + 下一时隙预测 | **仅掩码重构**（DeepSpeed ZeRO-2，bf16） |
| 微调 | 全量 / 任务头 | LoRA（`r=16`，`α=32`）作用在 `qkv`、`out_proj`、`ff` |
| 本仓库任务 | AMC（见 `main`） | AMC（`rml2016a` / `rml2016b`）、WTC、ManyTx |

---

## 2. 归一化（重要）

进入网络的波形在磁盘上始终是原始 float32 IQ，形状 `(L, 2)`，**只在训练/推理时在线归一化**，不写回磁盘。

**样本级 RMS**（`dataset.rms_normalize` / `rms_normalize_torch`）：

```
rms = sqrt( mean(I^2 + Q^2，对 L 个时刻求平均) + 1e-8 )
I' = I / rms
Q' = Q / rms
```

性质：

- **按样本，不按 batch。** 不用 BatchNorm 那种运行均值。
- **I、Q 共用一个尺度。** 该突发的复包络能量归一到 1，相对相位和 I/Q 比值不变。
- **不是 `iq2ap`。** 不会转成瞬时幅度/相位。那是 v1 AMC 的选项，本版主实验不用。
- **不是全数据集 z-score。** 不同 SNR / 电台在各自样本内归一后，相对能量关系仍保留。
- **`prepare_data.py` 不做归一化。** 分片保持原始值。预训练在 `train.to_engine_tensor`、微调/基线在 `IQClassifier.features` 里对 GPU batch 做 RMS。

序列长于预训练长度时（ManyTx `L=256`，模型 `seq_len=128`，`long_mode=split`）：

1. 把原始突发切成互不重叠的 128 点窗口；
2. **每个窗口各自做 RMS**（与预训练 chunk 一致）；
3. 分别编码、池化，再对窗口特征取平均。

因此 256 点指纹是两段独立归一化的 128 点，而不是对 256 点求一次 RMS。

**推理输入约定：** 传入 **原始** `(B, L, 2)` float32。`eval_finetune.py` 和 `IQClassifier` 内部会做 RMS。不要事先归一化，否则会做两次。

### 2.1 归一化消融（11 种方案）

专门实验考察 **输入表示** 和 **统计量作用范围** 对下游迁移的影响。这是 v2 的贡献之一：骨干和 LoRA 配方不变，只换归一化。

**协议（请与第 10 节全量结果表分开看）：**

| 项目 | 消融设定 |
|---|---|
| 预训练数据 | 仅 RML2018.01A |
| 预训练时长 | **1 个 epoch** |
| 微调 | LoRA，每一行架构与训练配置相同 |
| AMC | RML2016.10a |
| WTC | 分层抽样的 **1 万样本子集**（不是 22.08 万的 `per_cell=9200` 协议） |
| RFFI | WiSig **ManySig** |
| 随机种子 | 三个独立 run seed，报告 mean ± std |

**作用范围。** *None*：不做尺度变换。*样本级*：统计量只来自当前这一段突发（与发布的 RMS 同一思路）。*数据集级*：在训练集上算一套统计量，套到每一段突发。

**输入。** *IQ* 保留同相/正交。*AP* 先变成瞬时幅度与相位（`A = |I+jQ|`，`P = atan2(Q, I)`）。

一段突发记为 `x ∈ R^{L×2}`（IQ）或同长度的 AP。样本级只用该突发；数据集级用训练集的 min / mean / std / RMS。

| 算子 | 公式（除非另说，统计量在 I、Q 上共用） |
|---|---|
| Raw | 恒等 |
| min–max | `(x − min) / (max − min + ε)` |
| *z*-score | `(x − mean) / (std + ε)` |
| RMS-power | `x / sqrt(mean(I²+Q²) + ε)` |
| max-abs | `x / (max|x| + ε)` |

**表：预训练 1 个 epoch 后的归一化消融。** 准确率（%），三个种子的 mean ± std。

| 范围 | 输入 / 归一化 | AMC (%) | WTC (%) | RFFI (%) |
|---|---|---:|---:|---:|
| None | Raw IQ | 9.09 ± 0.00 | 67.72 ± 2.36 | 82.34 ± 10.75 |
| None | Raw AP | 52.80 ± 2.78 | 72.87 ± 5.34 | 96.46 ± 0.74 |
| 样本级 | IQ min–max | 56.06 ± 0.12 | 51.73 ± 4.33 | 93.13 ± 1.20 |
| 样本级 | IQ *z*-score | 62.53 ± 0.07 | 64.03 ± 1.18 | 98.53 ± 0.11 |
| 样本级 | **IQ RMS-power** | **63.65 ± 0.20** | 65.93 ± 0.67 | 98.40 ± 0.11 |
| 样本级 | IQ max-abs | 63.14 ± 0.42 | 65.25 ± 0.23 | 98.31 ± 0.15 |
| 样本级 | AP min–max | 63.32 ± 0.07 | 66.05 ± 0.44 | 98.34 ± 0.41 |
| 样本级 | AP *z*-score | 63.50 ± 0.17 | **66.37 ± 0.73** | **98.68 ± 0.10** |
| 数据集级 | IQ min–max | 47.79 ± 10.41 | 72.57 ± 1.85 | 86.63 ± 1.24 |
| 数据集级 | IQ *z*-score | 59.79 ± 0.81 | 74.28 ± 1.01 | 96.89 ± 0.28 |
| 数据集级 | IQ RMS-power | 52.49 ± 2.77 | 75.02 ± 3.68 | 83.39 ± 14.76 |

**读表：**

- **AMC 必须有按突发的尺度。** Raw IQ 是随机水平（`9.09%`，即 1/11）。Raw AP 就能到 `52.80%`，因为幅度/相位消掉一部分增益。样本级 IQ/AP 归一化随后都落在 63% 附近。
- **IQ RMS-power 是 AMC 上最好的方案**（`63.65 ± 0.20%`），在 WTC / RFFI 上也接近最好。因此发布的预训练 + LoRA 路径用它（`dataset.rms_normalize_torch`）。
- **AP *z*-score** 在 1-epoch 的 WTC 子集和 ManySig RFFI 上略高，但只比 RMS 高一点。默认仍用 RMS，让预训练和微调共用一套实值 IQ 尺度（不做 `atan2`，也不用全数据统计量）。
- **数据集级方案在 AMC 上更弱、更不稳**（min–max `47.79 ± 10.41%`；RMS-power `52.49 ± 2.77%`）。全局增益会把不同 SNR / 电台混在一起；样本级 RMS 不会。

第 10 节的全量数字（混合数据预训练 10 epoch、WTC 22.08 万、ManyTx）**不是**这张 1-epoch 表。全量实验只用样本级 IQ RMS-power。

---

## 3. 模型

Burtsev / Recurrent Memory Transformer 风格（`model.py`）。

**RMS 之后的输入：** `iq`，形状 `[B, L, 2]`。

**嵌入。** I、Q 分别过线性层（`i_proj`、`q_proj`）再相加。被掩码的坐标 **换成可学习 mask token**（`mask_token_i` / `mask_token_q`），而不是丢掉该位置，因此仍参与双向注意力。再加长度 128 的可学习 `pos_embed`。

**Memory token。** 一组可学习的 `n_memory=16` 个 token **前置并后置**：

```
[ mem_read (16) | IQ tokens (L) | mem_write (16) ]
```

整段做双向注意力。默认规模：`d_model=512`，16 层，8 头，`d_ff=2048`，预训练 dropout 为 0（约 5000 万参数）。

**预训练头。** 线性层 `d_model → 2`，在每个时刻重建 I、Q。损失 **只在被掩码的坐标上** 计算。

**微调头。** 不再重建。对 IQ token 池化（默认 mean）→ LayerNorm → Dropout → Linear → logits。

---

## 4. 目录结构

```
model.py            Memory Transformer
dataset.py          IQ 读取、切块、样本 RMS
prepare_data.py     把多种电台文件转成原始训练分片
train.py            DeepSpeed 预训练
config.yaml         预训练超参
ds_config.json      DeepSpeed ZeRO-2 模板
finetune.py         LoRA / 全量 / 线性探测（DDP）
finetune_data.py    RML2016.10a/b、WTC、ManyTx
lora.py             LoRA 注入 / 合并
eval_finetune.py    评估已保存的 run
baselines/          AWN、MCCFN、IQGCA、AvgNet、STF-GCN
ckpts/best.pt       发布的预训练骨干
runs/finetune/      发布的 LoRA 适配器 + results.json
run_pretrain.sh
run_finetune.sh
run_baselines.sh
```

原始数据和 memmap 缓存 **不进 git**。发布的权重在仓库里（见第 8 节）。

---

## 5. 环境

```bash
pip install -r requirements.txt
```

请安装对应 GPU 的 CUDA 版 PyTorch。预训练用 [DeepSpeed](https://www.deepspeed.ai/)，微调用 `torchrun`。

### 数据路径

把数据放在（或软链到）`./data`，或设置环境变量。

| 变量 | 默认 | 用途 |
|---|---|---|
| `RADIO_ROOT` | `./data` | 微调数据根目录 |
| `EMLM_CACHE` | `./cache` | 首次读取后缓存的 `.npz` |
| `WTC_MAT` | `$RADIO_ROOT/TecRec/finetune.mat` | WTC |
| `MANYTX_PKL` | `$RADIO_ROOT/ManyTx.pkl` | ManyTx |

微调文件：

```
data/
  RML2016.10a_dict.pkl          # 或 data/RML2016/RML2016.10a_dict.pkl
  RML2016.10b.dat
  TecRec/finetune.mat
  ManyTx.pkl
```

预训练文件：`config.yaml` 里 `data.root`（默认 `./data`）下、`dataset.discover_raw_files` 能识别的电台 IQ。然后：

```bash
python prepare_data.py --config config.yaml
```

必要时解压，切成长度 128 的块，打乱后写成 memmap 分片到 `./cache/pretrain`。**这里不做 RMS。**

---

## 6. 预训练

自监督 **掩码 I/Q 重构**（v2 不再做下一时隙预测）。

```bash
bash run_pretrain.sh                       # 全新开始
bash run_pretrain.sh --resume ./ckpts      # 续训
bash run_pretrain.sh --dummy --steps 20    # 冒烟
```

### 前向 / 损失

每个 RMS 后的 batch `iq`，形状 `[B, 128, 2]`：

1. **I、Q 独立掩码**（`independent_iq_mask: true`）。`128×2` 个标量各自以 `mask_ratio=0.3` 做伯努利丢弃。同一时刻的 I 与 Q **不绑定**。若某样本全部可见，则强制掩一个坐标，保证损失有定义。
2. 被掩位置换成 mask token，接上 memory token，过 16 层 Transformer。
3. 损失是 **masked MSE**：只对掩码坐标求 `(pred − iq)²` 的均值（`model.masked_mse`）。可见坐标不进损失。

### 默认超参（`config.yaml`）

| 项目 | 取值 |
|---|---|
| 序列长度 | 128 |
| 掩码比例 | 0.3，I/Q 独立 |
| 精度 | bf16 |
| DeepSpeed | ZeRO-2（`ds_config.json`） |
| 优化器 | AdamW，峰值 `lr=4e-4`，余弦降到 `1e-6`，2000 step warmup，`wd=0.01` |
| 示例 batch | micro 512 × accum 2 × 4 卡 → 全局 4096（按显存改） |
| Epoch | 10 |
| Dropout / drop-path | 0 / 0（微调时再打开） |
| `torch.compile` | 开 |
| 权重 | `./ckpts/best.pt`（最好 val MSE），`./ckpts/last.pt` |

验证集是 1% hold-out（`val_ratio: 0.01`）。`best.pt` 取最低 val MSE。

---

## 7. 微调

骨干基本冻结。用 DDP 训练 LoRA 适配器 + 新分类头（`finetune.py`）。

```bash
# 四个任务 × 种子 42/123/456，划分 6:2:2
CKPT=./ckpts/best.pt bash run_finetune.sh

TASKS=rml2016a SEEDS=42 CKPT=./ckpts/best.pt bash run_finetune.sh

# ManyTx，2×24 GB（L=256；跨天测试集留在 CPU）
TASKS=manytx NGPU=2 BS=32 EVAL_BS=64 CKPT=./ckpts/best.pt bash run_finetune.sh
```

```bash
torchrun --standalone --nproc_per_node=2 finetune.py \
  --task rml2016a --ckpt ./ckpts/best.pt --mode lora \
  --lora-r 16 --lora-alpha 32 --split 6:2:2 --epochs 20 \
  --lr 1e-3 --lr-decay-epoch 10 --lr-decay 0.1 --bs 256
```

### LoRA

在选定的 `nn.Linear` 上：`y = Wx + (α/r) B A x`（`lora.py`）。底权重 `W` 冻结。

| 项目 | 发布协议 |
|---|---|
| 模式 | `--mode lora` |
| 秩 / alpha / dropout | `r=16`，`α=32`，0.05 |
| 作用位置 | `qkv`、`out_proj`、`ff`（注意力 + FFN） |
| 冻结 | 嵌入、memory token、LayerNorm（除非 `--train-extra`） |
| 训练 | LoRA A/B + 分类头（约 210 万 / 5260 万 ≈ 4%） |
| 池化 | `--pool mean`，对 IQ token 取平均 |
| 分类头 | LayerNorm → Dropout 0.1 → Linear |
| 骨干 dropout | 微调时 0.1 |

其它模式：`--mode full`（全量）、`--mode linear`（只训头）。

### 划分协议（要对齐数字必须一致）

`--split 6:2:2` 且 `--split-seed 0`（划分种子 **不是** run 种子）：

1. 按类别分层；有 SNR 时再按 SNR 分层（RML / WTC）。
2. 先切出 80% 作为 train+val 池、20% 作为 test（`finetune_data.stratified_split`）。
3. 再把该池按 75/25 切开 → **60% 训练 / 20% 验证 / 20% 测试**。
4. `--seed`（42 / 123 / 456）只改初始化和打乱，**不改谁进 train/val/test**。

`best.pt` 按 **验证集** 准确率挑选。报告的是 **该 epoch 的测试准确率**。

### 任务

**RML2016.10a（`rml2016a`）。** 11 类调制，DeepSig pickle。220,000 段，长度 128。6:2:2 后为 13.2 万 / 4.4 万 / 4.4 万。会记 per-SNR 准确率。

**RML2016.10b（`rml2016b`）。** 10 类调制，长度 128。同一划分规则。

**WTC（`wtc`）。** `TecRec/finetune.mat`（MATLAB v7.3）：`out` 为 `(256, N)`，按列堆叠 `[I(128); Q(128)]`，`label` 为 `(2, N)` = `[类别; snr]`。三类：WiFi / LTE / ZigBee。原始约 206 万列。先按 **每个 (类别, SNR) 格 `per_cell=9200`**、RNG 为 `split_seed+7` 子采样 → 3 × 8 × 9200 = **220,800**（与 RML2016.10a 同量级），再 6:2:2 → 132,480 / 44,160 / 44,160。

**ManyTx（`manytx`）。** WiSig 射频指纹 pickle，**长度 256**，150 个 TX。只在第 1 天（`2021_03_01`）上训练，`equalized=0`（原始、未均衡）。第 1 天按 **TX** 分层划分。第 2 / 3 / 4 天（`2021_03_08`、`2021_03_15`、`2021_03_23`）作为额外测试（域偏移）。`long_mode=split`，见第 2 节。

### 优化

| 项目 | 取值 |
|---|---|
| Epoch | 20 |
| 学习率 | 第 1–10 epoch 为 `1e-3`，之后 `×0.1` → `1e-4` |
| 权重衰减 | 0.01 |
| 标签平滑 | 0 |
| 梯度裁剪 | 1.0 |
| AMP | 默认 bf16，`--no-amp` 可关 |
| 增强 | 默认关（`--aug-phase` / `--aug-flip`；相位增强会伤害指纹任务） |

输出在 `./runs/finetune/<task>/seed<seed>/`：

| 文件 | 内容 |
|---|---|
| `best.pt` | 仅可训练张量（LoRA A/B + 头），验证最好的 epoch — **发布的就是这个** |
| `last.pt` | 同上，最后一个 epoch |
| `last_merged.pt` | 把 LoRA 折进骨干后的完整权重（约 193 MB，**不上 GitHub**） |
| `results.json` | 测试 / 跨天指标 |

```bash
python eval_finetune.py ./runs/finetune/rml2016a/seed42
```

`eval_finetune` 会重建同一划分，加载 `ckpts/best.pt` + 适配器，对原始 IQ 做 RMS，并重打报告。

---

## 8. 发布的权重

`git checkout v2` 之后就在本分支里：

| 文件 | 含义 | 大小 |
|---|---|---|
| [`ckpts/best.pt`](ckpts/best.pt) | 预训练 Memory Transformer | 约 97 MB |
| [`runs/finetune/<task>/seed<seed>/best.pt`](runs/finetune) | 验证最好时的 LoRA + 头 | 每个约 8 MB |
| `runs/finetune/<task>/seed<seed>/results.json` | 该种子的指标 | 很小 |

任务：`rml2016a`、`rml2016b`、`wtc`、`manytx`。种子：`42`、`123`、`456`。

`runs/finetune/.../best.pt` **不能单独用**，必须搭配 `ckpts/best.pt`。合并后的 193 MB 文件超过 GitHub 单文件 100 MB 限制，未上传。

---

## 9. 基线

AWN、MCCFN、IQGCA、AvgNet、STF-GCN 使用 **同一套** 6:2:2 划分种子和 **同一种本级 RMS**（不用 `iq2ap`）。

```bash
TASK=rml2016a bash run_baselines.sh
TASK=wtc bash run_baselines.sh
```

`baselines/train.py` 目前注册 `rml2016a`、`rml2016b`、`wtc`。下面 ManyTx 基线数字来自 v1 风格训练器，输入 **裁成 128 点**（那些 CNN/GNN 按 128 设计）。LoRA 的 ManyTx 仍是 256 + 分窗。

---

## 10. 全量微调数字（LoRA）

协议：6:2:2，种子 `{42,123,456}`，取验证最好 epoch 的测试准确率。mean ± std。

| 任务 | SpectrumFM v2 LoRA | 同协议下最强基线 |
|---|---|---|
| RML2016.10a | **63.86 ± 0.19%** | STFGCN 62.62% |
| RML2016.10b | **65.45 ± 0.21%** | STFGCN 65.25% |
| WTC | **72.07 ± 0.06%** | STFGCN 70.55% |
| ManyTx 第 1 天 | **89.14 ± 0.08%** | AWN 87.31% |
| ManyTx 第 2 / 3 / 4 天 | 32.46 / 30.72 / 28.11 | MCCFN 34.9 / 32.5 / 29.0 |

ManyTx 跨天对 LoRA 和 CNN/GNN 都很难（只在第 1 天训练，`equalized=0`）。

---

## 11. 引用

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

## 12. 贡献者

GitHub 会把提交说明里的 `Co-authored-by` 算进仓库的 [Contributors](https://github.com/ChunyuLiu188/SpectrumFM/graphs/contributors)。本 `v2` 分支的提交带：

```
Co-authored-by: Cursor <cursoragent@cursor.com>
```

因此 [cursoragent](https://github.com/cursoragent) 会和仓库主人一起出现在贡献者页。

| | 角色 |
|---|---|
| [Chunyu Liu](https://github.com/ChunyuLiu188) | 研究、模型、实验、论文 |
| [Cursor](https://cursor.com)（[cursoragent](https://github.com/cursoragent)） | v2 代码整理、文档、发布打包 |
