# SpectrumFM v2

[English](README.md) | **中文**

SpectrumFM 第二版代码（Memory Transformer + LoRA）。第一版论文代码在 [`main`](https://github.com/ChunyuLiu188/SpectrumFM/tree/main)。

```bibtex
@ARTICLE{11301740,
  author={Zhou, Fuhui and Liu, Chunyu and Zhang, Hao and Wu, Wei and Wu, Qihui and Quek, Tony Q. S. and Chae, Chan-Byoung},
  journal={IEEE Journal on Selected Areas in Communications},
  title={SpectrumFM: A Foundation Model for Intelligent Spectrum Management},
  year={2025},
  doi={10.1109/JSAC.2025.3644783}}
```

```bash
git clone -b v2 https://github.com/ChunyuLiu188/SpectrumFM.git
cd SpectrumFM
pip install -r requirements.txt
```

预训练权重：[`ckpts/best.pt`](ckpts/best.pt)。LoRA：`runs/finetune/<task>/seed{42,123,456}/best.pt`（需与 `ckpts/best.pt` 一起加载）。

---

## 结构

Memory Transformer（`model.py`）：I/Q 分别投影，两端 RMT memory token，双向注意力。

| | v1 | v2 |
|---|---|---|
| $d$，头数，$d_\mathrm{ff}$，层数 | 256，4，512，16 | 512，8，2048，16 |
| Memory token | — | 16 |
| 序列 | 128 | 128（ManyTx 为 256，拆成两个 128 窗） |
| 预训练 | 掩码 + 下一时隙 | 仅掩码 I/Q 重构 |
| 微调 | 全量 | LoRA $r{=}16$，$\alpha{=}32$，作用于 `qkv,out_proj,ff` |
| 参数量 | — | $\approx$ 50M；LoRA 训练 $\approx$ 2.1M |

序列布局：`[mem (16) | IQ (128) | mem (16)]`。被掩 I/Q 替换为可学习 mask token。微调池化：对 IQ token 取均值。

---

## 归一化

预训练、LoRA 与基线默认均为 **样本级 IQ RMS-power**，在 GPU batch 上计算，不写回磁盘。

$$
\mathrm{rms}=\sqrt{\mathrm{mean}(I^2+Q^2)+\varepsilon},\quad I'=I/\mathrm{rms},\; Q'=Q/\mathrm{rms}.
$$

统计量按单条突发（I、Q 共用尺度），不用 batch / 数据集统计。网络输入为原始 `(B, L, 2)` float32。ManyTx（$L{=}256$）对每个 128 窗独立 RMS，再对窗特征平均（`long_mode=split`）。

### 消融（11 种）

RML2018.01A 预训练 1 epoch；LoRA 迁到 RML2016.10a、WTC 分层 1 万子集、WiSig ManySig（RFFI）。架构相同，三种子，mean$\pm$std（%）。**不是**下文全量协议。

| 范围 | 输入 | AMC | WTC | RFFI |
|---|---|---:|---:|---:|
| 无 | Raw IQ | $9.09\pm0.00$ | $67.72\pm2.36$ | $82.34\pm10.75$ |
| 无 | Raw AP | $52.80\pm2.78$ | $72.87\pm5.34$ | $96.46\pm0.74$ |
| 样本 | IQ min–max | $56.06\pm0.12$ | $51.73\pm4.33$ | $93.13\pm1.20$ |
| 样本 | IQ $z$-score | $62.53\pm0.07$ | $64.03\pm1.18$ | $98.53\pm0.11$ |
| 样本 | **IQ RMS-power** | $\mathbf{63.65\pm0.20}$ | $65.93\pm0.67$ | $98.40\pm0.11$ |
| 样本 | IQ max-abs | $63.14\pm0.42$ | $65.25\pm0.23$ | $98.31\pm0.15$ |
| 样本 | AP min–max | $63.32\pm0.07$ | $66.05\pm0.44$ | $98.34\pm0.41$ |
| 样本 | AP $z$-score | $63.50\pm0.17$ | $\mathbf{66.37\pm0.73}$ | $\mathbf{98.68\pm0.10}$ |
| 数据集 | IQ min–max | $47.79\pm10.41$ | $72.57\pm1.85$ | $86.63\pm1.24$ |
| 数据集 | IQ $z$-score | $59.79\pm0.81$ | $74.28\pm1.01$ | $96.89\pm0.28$ |
| 数据集 | IQ RMS-power | $52.49\pm2.77$ | $75.02\pm3.68$ | $83.39\pm14.76$ |

发布实验采用样本级 IQ RMS-power。

---

## 预训练

掩码重构。$128\times 2$ 个标量以 $r{=}0.3$ 独立掩码；损失仅为掩码坐标上的 MSE（`model.masked_mse`）。

```bash
python prepare_data.py --config config.yaml   # 分片写入 ./cache/pretrain，此处不做 RMS
bash run_pretrain.sh
```

| | |
|---|---|
| 数据 | `config.yaml` 中 `data.root` 下的电台 IQ |
| 精度 / ZeRO | bf16，DeepSpeed ZeRO-2 |
| 优化器 | AdamW，峰值 $4\times10^{-4}$，余弦至 $10^{-6}$，2000 warmup，$wd{=}0.01$ |
| Batch（示例） | micro 512 $\times$ accum 2 $\times$ 4 GPU $=$ 4096 |
| Epoch / 验证 | 10 / 1% hold-out；`ckpts/best.pt` 取最低 val MSE |
| Dropout | 0（微调改为 0.1） |

---

## 微调

```bash
CKPT=./ckpts/best.pt bash run_finetune.sh
python eval_finetune.py ./runs/finetune/rml2016a/seed42
```

LoRA $r{=}16$，$\alpha{=}32$，dropout 0.05，作用于 `qkv,out_proj,ff`。分类头：LayerNorm $\to$ Dropout 0.1 $\to$ Linear。20 epoch，lr $10^{-3}$，第 10 epoch 后 $\times0.1$，$wd{=}0.01$，梯度裁剪 1.0，bf16。

**划分。** `--split 6:2:2`，`--split-seed 0`（与 `--seed` 独立）。按类别分层，有 SNR 时再按 SNR。先 80/20，再对训练池 75/25 $\Rightarrow$ 60/20/20。`--seed` $\in\{42,123,456\}$ 只改初始化。按验证集选 checkpoint，报告该 epoch 测试准确率。

| 任务 | 数据 | 说明 |
|---|---|---|
| `rml2016a` | `data/RML2016.10a_dict.pkl` | 11 类，$L{=}128$，220k $\to$ 132k/44k/44k |
| `rml2016b` | `data/RML2016.10b.dat` | 10 类，$L{=}128$ |
| `wtc` | `data/TecRec/finetune.mat` | WiFi/LTE/ZigBee；`per_cell=9200` $\to$ 220800，再 132480/44160/44160 |
| `manytx` | `data/ManyTx.pkl` | 150 TX，$L{=}256$；训练日 `2021_03_01`，`equalized=0`；第 2–4 天为额外测试 |

路径变量：`RADIO_ROOT`、`EMLM_CACHE`、`WTC_MAT`、`MANYTX_PKL`（默认 `./data`、`./cache`）。

基线（AWN、MCCFN、IQGCA、AvgNet、STF-GCN）使用同一划分与样本 RMS。

```bash
TASK=rml2016a bash run_baselines.sh
```

CNN/GNN 的 ManyTx 为 128 点裁剪；LoRA ManyTx 为 256 + 分窗。

---

## 结果（全量协议）

6:2:2，种子 $\{42,123,456\}$，验证最好 epoch 的测试准确率，mean$\pm$std（%）。样本级 IQ RMS。

| 任务 | SpectrumFM v2 LoRA | 最强基线 |
|---|---|---|
| RML2016.10a | $\mathbf{63.86\pm0.19}$ | STFGCN 62.62 |
| RML2016.10b | $\mathbf{65.45\pm0.21}$ | STFGCN 65.25 |
| WTC | $\mathbf{72.07\pm0.06}$ | STFGCN 70.55 |
| ManyTx 第 1 天 | $\mathbf{89.14\pm0.08}$ | AWN 87.31 |
| ManyTx 第 2/3/4 天 | $32.46$ / $30.72$ / $28.11$ | MCCFN 34.9 / 32.5 / 29.0 |
