#!/usr/bin/env bash
# LoRA fine-tuning: one job at a time, DDP, per-GPU bs=256.
#   bash run_finetune.sh
#   TASKS="rml2016a" SEEDS="42" bash run_finetune.sh
#   TASKS="manytx" NGPU=2 BS=32 EVAL_BS=64 bash run_finetune.sh
set -uo pipefail
cd "$(dirname "$0")"

CKPT=${CKPT:-./ckpts/best.pt}
TASKS=${TASKS:-"rml2016a rml2016b wtc manytx"}
SEEDS=${SEEDS:-"42 123 456"}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
OUT=${OUT:-./runs/finetune}
EPOCHS=${EPOCHS:-20}
SPLIT=${SPLIT:-6:2:2}
BS=${BS:-256}
EVAL_BS=${EVAL_BS:-1024}
EXTRA=${EXTRA:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$OUT"

for task in $TASKS; do
  for seed in $SEEDS; do
    if [ -f "$OUT/$task/seed$seed/results.json" ]; then
      echo "[skip] $task seed$seed already done"; continue
    fi
    echo "===== $task seed=$seed ($NGPU GPUs) $(date) ====="
    torchrun --standalone --nproc_per_node="$NGPU" finetune.py \
      --task "$task" --seed "$seed" --ckpt "$CKPT" --out-dir "$OUT" \
      --lora-r 16 --lora-alpha 32 --split "$SPLIT" --epochs "$EPOCHS" \
      --lr 1e-3 --lr-decay-epoch 10 --lr-decay 0.1 --bs "$BS" --eval-bs "$EVAL_BS" \
      $EXTRA 2>&1 | tee "$OUT/${task}_seed${seed}.log" | grep -vE "^\s*$|Warning|warn"
  done
done

echo "===== summary ====="
python3 - "$OUT" <<'PY'
import json, statistics as st, sys
from pathlib import Path
out = Path(sys.argv[1])
for task in sorted({p.parent.parent.name for p in out.glob("*/seed*/results.json")}):
    runs = sorted(out.glob(f"{task}/seed*/results.json"))
    rows = {}
    for r in runs:
        d = json.loads(r.read_text())
        for k, v in d.items():
            if isinstance(v, dict) and "acc" in v:
                rows.setdefault(k, []).append(v["acc"] * 100)
    print(task, f"({len(runs)} seeds)")
    for k, accs in rows.items():
        sd = st.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {k:24s} {st.mean(accs):6.2f} ± {sd:4.2f}   " + " ".join(f"{a:.2f}" for a in accs))
PY
