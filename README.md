# SpectrumFM v2

**English** | [中文](README_zh.md)

Code for SpectrumFM v2 (Memory Transformer + LoRA). Paper code for v1 is on [`main`](https://github.com/ChunyuLiu188/SpectrumFM/tree/main).

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

Pretrained checkpoint: [`ckpts/best.pt`](ckpts/best.pt). LoRA adapters: `runs/finetune/<task>/seed{42,123,456}/best.pt` (load together with `ckpts/best.pt`).

---

## Architecture

Memory Transformer (`model.py`): I/Q projected separately, RMT memory tokens at both ends, bidirectional attention.

| | v1 | v2 |
|---|---|---|
| $d$, heads, $d_\mathrm{ff}$, $L$ | 256, 4, 512, 16 | 512, 8, 2048, 16 |
| Memory tokens | — | 16 |
| Sequence | 128 | 128 (ManyTx: 256, split into two 128 windows) |
| Pretrain | mask + next-slot | masked I/Q reconstruction only |
| Fine-tune | full | LoRA $r{=}16$, $\alpha{=}32$ on `qkv,out_proj,ff` |
| Parameters | — | $\approx$ 50M; LoRA trains $\approx$ 2.1M |

Layout: `[mem (16) | IQ (128) | mem (16)]`. Masked I/Q are replaced by learnable mask tokens (not dropped). Fine-tune pooling: mean over IQ tokens.

---

## Normalization

Default (pretrain, LoRA, and baselines): **sample-level IQ RMS-power**, applied on the GPU batch, not written to disk.

$$
\mathrm{rms}=\sqrt{\mathrm{mean}(I^2+Q^2)+\varepsilon},\quad I'=I/\mathrm{rms},\; Q'=Q/\mathrm{rms}.
$$

Statistics are per burst (I and Q share one scale), not batch- or dataset-level. Input to the model is raw `(B, L, 2)` float32. For ManyTx ($L{=}256$), each 128-window is RMS-normalized independently, then window features are averaged (`long_mode=split`).

### Ablation (11 schemes)

Pretrain 1 epoch on RML2018.01A; LoRA on RML2016.10a, a stratified 10k WTC subset, and WiSig ManySig (RFFI). Same architecture; three seeds; mean$\pm$std (%). **Not** the full-run protocol below.

| Scope | Input | AMC | WTC | RFFI |
|---|---|---:|---:|---:|
| None | Raw IQ | $9.09\pm0.00$ | $67.72\pm2.36$ | $82.34\pm10.75$ |
| None | Raw AP | $52.80\pm2.78$ | $72.87\pm5.34$ | $96.46\pm0.74$ |
| Sample | IQ min–max | $56.06\pm0.12$ | $51.73\pm4.33$ | $93.13\pm1.20$ |
| Sample | IQ $z$-score | $62.53\pm0.07$ | $64.03\pm1.18$ | $98.53\pm0.11$ |
| Sample | **IQ RMS-power** | $\mathbf{63.65\pm0.20}$ | $65.93\pm0.67$ | $98.40\pm0.11$ |
| Sample | IQ max-abs | $63.14\pm0.42$ | $65.25\pm0.23$ | $98.31\pm0.15$ |
| Sample | AP min–max | $63.32\pm0.07$ | $66.05\pm0.44$ | $98.34\pm0.41$ |
| Sample | AP $z$-score | $63.50\pm0.17$ | $\mathbf{66.37\pm0.73}$ | $\mathbf{98.68\pm0.10}$ |
| Dataset | IQ min–max | $47.79\pm10.41$ | $72.57\pm1.85$ | $86.63\pm1.24$ |
| Dataset | IQ $z$-score | $59.79\pm0.81$ | $74.28\pm1.01$ | $96.89\pm0.28$ |
| Dataset | IQ RMS-power | $52.49\pm2.77$ | $75.02\pm3.68$ | $83.39\pm14.76$ |

Released runs use sample-level IQ RMS-power.

---

## Pretraining

Masked reconstruction. Each of $128\times 2$ scalars is masked independently with $r{=}0.3$; loss is MSE on masked coordinates only (`model.masked_mse`).

```bash
python prepare_data.py --config config.yaml   # shards under ./cache/pretrain; no RMS
bash run_pretrain.sh
```

| | |
|---|---|
| Data | mixed radio IQ under `data.root` (`config.yaml`) |
| Precision / ZeRO | bf16, DeepSpeed ZeRO-2 |
| Optimizer | AdamW, peak $4\times10^{-4}$, cosine to $10^{-6}$, 2000 warmup, $wd{=}0.01$ |
| Batch (example) | micro 512 $\times$ accum 2 $\times$ 4 GPU $=$ 4096 |
| Epochs / val | 10 / 1% hold-out; `ckpts/best.pt` = lowest val MSE |
| Dropout | 0 (set to 0.1 in fine-tune) |

---

## Fine-tuning

```bash
CKPT=./ckpts/best.pt bash run_finetune.sh
python eval_finetune.py ./runs/finetune/rml2016a/seed42
```

LoRA $r{=}16$, $\alpha{=}32$, dropout 0.05 on `qkv,out_proj,ff`. Head: LayerNorm $\to$ Dropout 0.1 $\to$ Linear. 20 epochs, lr $10^{-3}$ then $\times0.1$ after epoch 10, $wd{=}0.01$, grad clip 1.0, bf16.

**Split.** `--split 6:2:2`, `--split-seed 0` (independent of `--seed`). Stratify on class and SNR when available. 80/20 then 75/25 of the train pool $\Rightarrow$ 60/20/20. `--seed` $\in\{42,123,456\}$ changes init only. Checkpoint selected on val; report test at that epoch.

| Task | Data | Notes |
|---|---|---|
| `rml2016a` | `data/RML2016.10a_dict.pkl` | 11-way, $L{=}128$, 220k $\to$ 132k/44k/44k |
| `rml2016b` | `data/RML2016.10b.dat` | 10-way, $L{=}128$ |
| `wtc` | `data/TecRec/finetune.mat` | WiFi/LTE/ZigBee; `per_cell=9200` $\to$ 220800, then 132480/44160/44160 |
| `manytx` | `data/ManyTx.pkl` | 150 TX, $L{=}256$; train day `2021_03_01`, `equalized=0`; extra test days 2–4 |

Paths: `RADIO_ROOT`, `EMLM_CACHE`, `WTC_MAT`, `MANYTX_PKL` (defaults `./data`, `./cache`).

Baselines (AWN, MCCFN, IQGCA, AvgNet, STF-GCN): same split and sample RMS.

```bash
TASK=rml2016a bash run_baselines.sh
```

CNN/GNN ManyTx numbers use a 128-crop; LoRA ManyTx uses 256 + split.

---

## Results (full protocol)

6:2:2, seeds $\{42,123,456\}$, test accuracy at best val epoch, mean$\pm$std (%). Sample-level IQ RMS.

| Task | SpectrumFM v2 LoRA | Best baseline |
|---|---|---|
| RML2016.10a | $\mathbf{63.86\pm0.19}$ | STFGCN 62.62 |
| RML2016.10b | $\mathbf{65.45\pm0.21}$ | STFGCN 65.25 |
| WTC | $\mathbf{72.07\pm0.06}$ | STFGCN 70.55 |
| ManyTx day 1 | $\mathbf{89.14\pm0.08}$ | AWN 87.31 |
| ManyTx day 2/3/4 | $32.46$ / $30.72$ / $28.11$ | MCCFN 34.9 / 32.5 / 29.0 |
