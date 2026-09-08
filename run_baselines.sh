#!/usr/bin/env bash
# Same 6:2:2 + sample RMS as LoRA. Default: 2 jobs x 2 GPUs.
#   bash run_baselines.sh
#   TASK=rml2016b PARALLEL=2 NGPU_JOB=2 bash run_baselines.sh
set -uo pipefail
cd "$(dirname "$0")"
TASK=${TASK:-rml2016a}
MODELS=${MODELS:-"awn mccfn iqgca avgnet stfgcn"}
SEEDS=${SEEDS:-"42 123 456"}
NGPU_JOB=${NGPU_JOB:-2}
PARALLEL=${PARALLEL:-2}
OUT=${OUT:-./runs/baselines}
EPOCHS=${EPOCHS:-40}
LR_DECAY_EPOCH=${LR_DECAY_EPOCH:-20}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$OUT"

# Override with GPU_SLOTS="0,1 2,3" or GPU_SLOTS="0 1" for single-GPU jobs.
if [ -n "${GPU_SLOTS:-}" ]; then
  read -r -a GPU_SLOTS <<< "$GPU_SLOTS"
else
  GPU_SLOTS=("0,1" "2,3")
fi
PORTS=(${PORTS:-29511 29512})

queue=()
for model in $MODELS; do
  for seed in $SEEDS; do
    if [ -f "$OUT/$TASK/$model/seed$seed/results.json" ]; then
      echo "[skip] $TASK $model seed$seed"
      continue
    fi
    queue+=("$model $seed")
  done
done

launch_one() {
  local slot=$1 model=$2 seed=$3
  local gpus=${GPU_SLOTS[$slot]}
  local port=${PORTS[$slot]}
  local bs log
  case "$model" in
    avgnet|stfgcn) bs=$((256 / NGPU_JOB)) ;;
    *)             bs=$((1024 / NGPU_JOB)) ;;
  esac
  log="$OUT/${TASK}_${model}_seed${seed}.log"
  echo "===== baseline $model $TASK seed=$seed gpus=$gpus bs=$bs $(date) ====="
  CUDA_VISIBLE_DEVICES="$gpus" MASTER_PORT="$port" \
    torchrun --standalone --nproc_per_node="$NGPU_JOB" baselines/train.py \
      --model "$model" --task "$TASK" --seed "$seed" --out-dir "$OUT" \
      --epochs "$EPOCHS" --lr-decay-epoch "$LR_DECAY_EPOCH" --bs "$bs" \
      >"$log" 2>&1
  local ec=$?
  grep -E '^(model=|test:|saved |\{"epoch")' "$log" | tail -n 8 || true
  if [ "$ec" -ne 0 ]; then
    echo "[fail] $TASK $model seed$seed (exit $ec)  see $log"
    tail -n 30 "$log" || true
  fi
  return "$ec"
}

slot_pid=()
slot_job=()
qi=0
n=${#queue[@]}
fails=0

while (( qi < n )); do
  launched=0
  for s in $(seq 0 $((PARALLEL - 1))); do
    if (( qi >= n )); then
      break
    fi
    if [[ -n "${slot_pid[$s]:-}" ]] && kill -0 "${slot_pid[$s]}" 2>/dev/null; then
      continue
    fi
    if [[ -n "${slot_pid[$s]:-}" ]]; then
      if ! wait "${slot_pid[$s]}"; then
        echo "[fail] ${slot_job[$s]}"
        fails=$((fails + 1))
      fi
      slot_pid[$s]=
      slot_job[$s]=
    fi
    read -r model seed <<<"${queue[$qi]}"
    qi=$((qi + 1))
    echo "[slot $s] start $TASK $model seed$seed  gpus=${GPU_SLOTS[$s]}"
    launch_one "$s" "$model" "$seed" &
    slot_pid[$s]=$!
    slot_job[$s]="$model seed$seed"
    launched=1
  done
  if (( launched == 0 )); then
    sleep 5
  fi
done

for s in $(seq 0 $((PARALLEL - 1))); do
  if [[ -n "${slot_pid[$s]:-}" ]]; then
    if ! wait "${slot_pid[$s]}"; then
      echo "[fail] ${slot_job[$s]}"
      fails=$((fails + 1))
    fi
  fi
done

echo "===== baseline summary $TASK ====="
python3 - "$OUT" "$TASK" <<'PY'
import json, statistics as st, sys
from pathlib import Path
out, task = Path(sys.argv[1]), sys.argv[2]
for model in sorted({p.parent.name for p in out.glob(f"{task}/*/seed*/results.json")}):
    accs = []
    for r in sorted(out.glob(f"{task}/{model}/seed*/results.json")):
        d = json.loads(r.read_text())
        for k, v in d.items():
            if isinstance(v, dict) and "acc" in v:
                accs.append(v["acc"] * 100)
                break
    if accs:
        sd = st.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {model:8s} {st.mean(accs):6.2f} ± {sd:4.2f}   " + " ".join(f"{a:.2f}" for a in accs))
PY
exit "$fails"
