#!/usr/bin/env bash
# train_segments.sh — Option A: split long CPU training into resumable segments
# Resumes automatically from models/tetris_ppo_model.zip (train.py PPO.load)
# Each segment saves every --checkpoint-freq sessions and on CTRL+C.

set -euo pipefail

# Defaults — 5M ≈ 600×4×2048, 1.6M = 200×4×2048, 10M = 1200×4×2048
TOTAL_SESSIONS=600
PER_SEGMENT=60
RUNS=4
STEPS=2048
POLICY="CnnPolicy"
SHAPED_ALPHA=0.1
BATCH_SIZE=32
GAMMA=0.95
LR=1e-4
CLIP=0.1
ENT=0.01
MODEL="models/tetris_ppo_model"
DEVICE="cpu"
N_ENVS=4
WINDOW="null"
CHECKPOINT_FREQ=10
TENSORBOARD=""

usage() {
  cat <<EOF
Usage: $0 [options]

Option A — segmented training (resume-safe). Runs TOTAL_SESSIONS as
PER_SEGMENT-sized invocations, each resumable via train.py PPO.load.

Options:
  --total N          total sessions (default $TOTAL_SESSIONS → ~5M steps with $RUNS×$STEPS)
  --per-segment N    sessions per invocation (default $PER_SEGMENT)
  --runs N           runs per session (default $RUNS)
  --steps N          n_steps per rollout (default $STEPS)
  --policy NAME      CnnPolicy|MlpPolicy (default $POLICY)
  --shaped-alpha F   PBRS alpha 0.0 legacy, 0.1 hybrid (default $SHAPED_ALPHA)
  --model PATH       model prefix (default $MODEL)
  --device NAME      auto|cpu|cuda (default $DEVICE)
  --n-envs N         VecEnv count 1=Dummy >1 Subproc (default $N_ENVS; 4→280 fps cpu, 292 fps cuda)
  --window NAME      null|SDL2|headless (default $WINDOW)
  --checkpoint-freq N save every N sessions (default $CHECKPOINT_FREQ, 0=only end)
  --tensorboard DIR  tensorboard log dir (default none)
  --batch-size N --gamma F --lr F --clip F --ent F  (PPO hyperparams)
  -h, --help         show this help

Examples:
  # 5M split 10×60 sessions (~500k each), resumable, checkpoints every 10
  $0 --total 600 --per-segment 60

  # resume after interruption — just re-run same command (loads $MODEL.zip)
  $0 --total 600 --per-segment 60

  # single 1.6M run with checkpoints
  uv run train.py --sessions 200 --runs 4 --steps 2048 --checkpoint-freq 10 --policy CnnPolicy --shaped-alpha 0.1

Requirements:
  roms/tetris.gb, states/init.state, uv sync --extra dev, libsdl2-2.0-0
EOF
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --total) TOTAL_SESSIONS="$2"; shift 2;;
    --per-segment) PER_SEGMENT="$2"; shift 2;;
    --runs) RUNS="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --policy) POLICY="$2"; shift 2;;
    --shaped-alpha) SHAPED_ALPHA="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --n-envs) N_ENVS="$2"; shift 2;;
    --window) WINDOW="$2"; shift 2;;
    --checkpoint-freq) CHECKPOINT_FREQ="$2"; shift 2;;
    --tensorboard) TENSORBOARD="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --gamma) GAMMA="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --clip) CLIP="$2"; shift 2;;
    --ent) ENT="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1" >&2; usage;;
  esac
done

if [[ ! -f "roms/tetris.gb" ]]; then
  echo "Missing roms/tetris.gb — provide legal dump (see README.md)" >&2
  exit 1
fi
mkdir -p "$(dirname "$MODEL")" logs 2>/dev/null || true

TOTAL_STEPS=$((TOTAL_SESSIONS * RUNS * STEPS))
PER_STEPS=$((PER_SEGMENT * RUNS * STEPS))
echo "=== Train segments: total $TOTAL_SESSIONS sessions ($TOTAL_STEPS steps), $PER_SEGMENT per segment ($PER_STEPS steps) ==="
echo "Policy $POLICY α=$SHAPED_ALPHA device $DEVICE n_envs $N_ENVS window $WINDOW checkpoint every $CHECKPOINT_FREQ sessions"
echo "Model $MODEL.zip — resumes automatically if exists"
echo

SEGMENTS=$(( (TOTAL_SESSIONS + PER_SEGMENT - 1) / PER_SEGMENT ))
for (( seg=1; seg<=SEGMENTS; seg++ )); do
  remaining=$((TOTAL_SESSIONS - (seg-1)*PER_SEGMENT))
  this_seg=$PER_SEGMENT
  if [[ $remaining -lt $PER_SEGMENT ]]; then this_seg=$remaining; fi

  echo "--- Segment $seg/$SEGMENTS: $this_seg sessions ($((this_seg*RUNS*STEPS)) steps) ---"
  TB_ARGS=()
  if [[ -n "$TENSORBOARD" ]]; then TB_ARGS=(--tensorboard "$TENSORBOARD"); fi

  # shellcheck disable=SC2086
  uv run train.py \
    --sessions "$this_seg" --runs "$RUNS" --steps "$STEPS" \
    --policy "$POLICY" --shaped-alpha "$SHAPED_ALPHA" \
    --batch-size "$BATCH_SIZE" --gamma "$GAMMA" --learning-rate "$LR" --clip-range "$CLIP" --ent-coef "$ENT" \
    --model-name "$MODEL" --device "$DEVICE" --n-envs "$N_ENVS" --window "$WINDOW" \
    --checkpoint-freq "$CHECKPOINT_FREQ" "${TB_ARGS[@]}"

  echo "--- Segment $seg done — $MODEL.zip updated ---"
  # quick eval every segment
  if [[ -f "${MODEL}.zip" ]]; then
    echo "Eval 2 runs (headless):"
    uv run play.py --window null --runs 2 --device "$DEVICE" --model "$MODEL" --shaped-alpha "$SHAPED_ALPHA" 2>&1 | tail -5 || true
  fi
done

echo
echo "All segments done. Final model: $MODEL.zip"
ls -lh "${MODEL}.zip" "${MODEL}"_ckpt_*.zip 2>/dev/null | tail -20 || ls -lh "${MODEL}.zip"
echo "Resume: re-run same command — train.py loads $MODEL.zip and continues"
echo "Play: uv run play.py --window SDL2 --model $MODEL --shaped-alpha $SHAPED_ALPHA"
