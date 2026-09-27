# tetris-rl

➖➖➖➖➖➖➖🟦🟦➖  
➖➖➖➖➖➖➖🟦➖➖  
➖➖➖➖➖➖➖🟦➖➖  
➖➖➖➖➖➖➖➖➖➖  
➖➖🟩🟩➖🟧➖➖➖➖  
➖🟩🟩🟧🟧🟧➖➖➖➖  
➖🟪🟪🟪🟦🟦🟦➖➖➖  
🟨🟨🟪🟪🟥🟥🟦➖🟨🟨  
🟨🟨🟪🟪🟪🟥🟥➖🟨🟨  
⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️  
⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️⬜️  

Playing around with Tetris and Reinforcement Learning

## Inspirations

The initial code is mainly inspired by this project:
* [blog post](https://rotational.io/blog/reinforcement-learning-automation-and-tetris)  
* [source code](https://github.com/pdeziel/ai-tetris)

However some of the metrics I used are rather taken from this project:
* [blog post](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player)
* [source code](https://github.com/LeeYiyuan/tetrisai)

## Requirements

- Python 3.12 (`pyproject.toml:8`, `.python-version`)
- `uv` — see [uv docs](https://docs.astral.sh/uv/)
- ROM: `roms/tetris.gb` (not tracked; provide your own legal dump)

## Setup

```bash
# System deps
# Ubuntu 24.04 (Noble) — runtime only needs SDL2:
sudo apt update && sudo apt install -y libsdl2-2.0-0
# only needed to build PyBoy from source:
# sudo apt install -y libsdl2-dev

# macOS (brew):
# brew install sdl2

# Install Python deps
uv sync --extra dev          # or: uv sync (runtime only)
# If network is slow or GPU has <4GB VRAM, use CPU-only torch:
# uv sync --no-install-package torch && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
# Maxwell GTX 960 sm_52 2GB (driver 580) pinned to torch==2.6.0+cu124 (last with sm_50; 2.7+ → sm_75+):
# uv sync already installs 2.6 → `uv run train.py --device cuda --window null --n-envs 4` just works
# (auto still picks cpu on <4GB; use --device cuda to force; ~400MiB VRAM, n_envs 4 → 297 fps vs 95)

# Provide ROM and create output dirs
mkdir -p roms models
# copy your dump → roms/tetris.gb
ls -lh roms/tetris.gb states/init.state  # init.state is PyBoy 2.7.0 save (already in repo)
```

`states/init.state` was generated with PyBoy 2.7.0 by pressing START after ROM load (`tetris_env.py:139`); regenerate if PyBoy version changes.

## Usage

```bash
uv run pytest tests/ -v          # headless, uses window="null"
uv run ruff check . && uv run ruff format --check .

uv run train.py --help           # trains PPO; saves to models_robust/tetris_ppo_next.zip
# defaults: MultiInputPolicy TetrisMultiInput (enriched) α=0.1 200×4×2048≈1.6M (600≈5M 1200≈10M) lr 1e-4 linear clip 0.1 ent 0.03 gamma 0.95 gae 0.95 vf 0.5 clip-vf 0.2 kl 0.02 batch 32 freq 8
uv run train.py --device cpu --window null --policy MultiInputPolicy --obs-mode enriched --shaped-alpha 0.1 --sessions 200 --steps 2048
uv run train.py --device cpu --obs-mode board --policy CnnPolicy --shaped-alpha 0.0 --sessions 40  # ablation: board-only + legacy pure score (sparse)
# tensorboard: uv run train.py --tensorboard logs_robust/  (graceful no-op if tensorboard pkg missing)
uv run train.py --device cpu     # auto→cpu on <4GB VRAM (e.g. GTX 960 2GB) or sm_52 incompat

uv run play.py --help            # loads model and plays with SDL2 window
uv run play.py --window null --shaped-alpha 0.1 --freq 8 --obs-mode enriched --device cpu --model models_robust/tetris_ppo_next
# headless / CI (no DISPLAY):
xvfb-run -a uv run play.py --window null
# or: SDL_VIDEODRIVER=dummy uv run train.py --window null
```

### Deterministic vs stochastic — why rotation can appear missing (`play.py:44`)

`--deterministic` (default `true`) always picks the highest-probability action (`argmax`); `--no-deterministic` samples from the learned distribution. Early checkpoints often learn `DOWN/LEFT/RIGHT` as `argmax` while `A/B` (rotate) stays at `10–15%` probability — e.g. `8k`-step `det {DOWN 203 LEFT124 RIGHT173 rot0/500}` vs `stoch {A/B 97/500 19%}`. This looks like “no rotation” in `deterministic` view but rotation *is* explored stochastically. Early `PASS` also let pieces fall centrally without moving (now removed).

* To **see** rotation on early/in-progress models: `uv run play.py --no-deterministic --freq 8 --shaped-alpha 0.1` (also Colab cell 6 shows both histograms)
* For final evaluation after `500k–5M` steps where `A/B` becomes `argmax`: `deterministic` will then show spins; keep `stochastic` only to diagnose collapse (if `stoch` `A/B <5%` you need longer training or higher `--ent-coef 0.03` `freq8`)

Early collapse was also caused by `7→6→5` `UP/PASS` removal + stronger bump (`-0.35→-0.5`) + hole (`-0.35→-0.8` `tetris_env.py`); old `7`-action / `6`-action / board-only / `ent 0.01` checkpoints need `rm models_robust/tetris_ppo_next*.zip` to train fresh (`PPO.load` keeps old hyper-params on resume; obs `board→enriched` also starts fresh).
```

### Long training — segmented & resumable (Option A)

CPU `1.6M ≈ 3-4h`, `5M ≈ 10h` at `~130 fps` `window="null"`. Split into segments — each resumes from `models_robust/tetris_ppo_next.zip` (`train.py` `PPO.load` + `TetrisMultiInput` `custom_objects`):

```bash
# 5M = 600×4×2048 — 10 segments ×60 sessions (~500k each), checkpoint every 10
./scripts/train_segments.sh --total 600 --per-segment 60
# 1.6M single run with checkpoints
uv run train.py --sessions 200 --runs 4 --steps 2048 --checkpoint-freq 10 --policy MultiInputPolicy --obs-mode enriched --shaped-alpha 0.1
# resume after CTRL+C / power loss — re-run same command (loads .zip)
./scripts/train_segments.sh --total 600 --per-segment 60
```

Options `train.py --help` / `scripts/train_segments.sh --help`:
- `--sessions N`, `--runs N`, `--steps N` → total steps `sessions×runs×steps` (`×n_envs` if `>1`)
- `--policy MultiInputPolicy|CnnPolicy|MlpPolicy` — `MultiInputPolicy` `TetrisMultiInput` for enriched Dict (default); `CnnPolicy` `TetrisCNN` `3×3` for `18×10` board-only ablation, `Mlp` fallback
- `--obs-mode enriched|board` — enriched Dict (`board` + `next`/`current` one-hot 7 + `features` 14 heights/holes/bump/wells) vs legacy Box; must match train & play
- `--shaped-alpha 0.1` hybrid `score+α*shaped` (`0.0` legacy sparse, `α=0.1-0.3`), `shaped = -0.5 height +0.75 lines -0.8 holes -0.5 bump` (bump ↑ spreads first pieces)
- `--freq 8` ticks per action (finer lateral before lock than legacy 24/12) — must match train & play
- `--learning-rate 1e-4 --lr-schedule linear --clip-range 0.1 --ent-coef 0.03 --gamma 0.95 --gae-lambda 0.95 --vf-coef 0.5 --clip-range-vf 0.2 --target-kl 0.02 --batch-size 32` (ent 0.03 spreads first pieces; linear LR + KL early-stop stabilize late training)
- `--max-steps 10000` truncation (kills stall loops, `0`=disabled) — `--noop-max 30` random no-ops on reset (RNG desync) — `--eval-freq 20000` best-model tracking
- `--checkpoint-freq 10` saves `models_robust/tetris_ppo_next.zip` every 10 sessions + `models_robust/tetris_ppo_next_ckpt_0010.zip` …; `0` only at end; `CTRL+C` also saves
 - `--n-envs 1` (`DummyVecEnv`) or `>1` `SubprocVecEnv` for throughput — `n_envs 4` `280 fps` `n_envs 8` `371 fps` on this 8-core CPU; `cuda n_envs 8` `423 fps` only ~14% faster (emulator-bound, gpu util 4%)
 - `--tensorboard logs_robust/` → `tensorboard --logdir logs_robust` (needs `pip install tensorboard`; missing pkg disables logging with warning)
 - `--device auto|cpu|cuda` — `auto`→`cpu` on `<4GB VRAM`; for `GTX 960` `cuda` works (`torch 2.6.0+cu124` `sm_50`, `~400MiB`) but CPU is recommended on this box (see GPU section)
- Actions `5` (`tetris_env.py:130`): `LEFT/RIGHT/DOWN/A/B` — `UP/PASS` removed (UP no-op wasted 14%, PASS let first pieces stack centrally; old 7/6-action zips auto-start fresh)

Checkpoints: `ls -lh models_robust/tetris_ppo_next*.zip` — resume is automatic; mismatched `Mlp→Cnn→MultiInput`, `board→enriched` obs, or `7→6→5` actions starts fresh with warning (legacy zips in `models/archive7/`).

### GPU — Maxwell GTX 960 sm_52 (2GB, driver 580 CUDA 13) enabled but CPU is faster

Pinned `torch==2.6.0+cu124` (`arch sm_50,sm_60,...,sm_90`) is last with `sm_50` (2.7+ `cu128` `sm_75+` → `no kernel image`). GPU *works* (`~400MiB` VRAM, `uv.lock` `2.6.0`/`sb3 2.8.0`) but is **not faster** on this workload — PyBoy emulator is CPU-bound (`500 steps` `145 fps` single `tetris_env.py:161`), PPO `TetrisCNN` is tiny (128-dim, 18×10).

Measured this box (`--runs 2 --steps 2048`, 8 cores, 15Gi RAM):
`cpu n_envs 1` `82 fps` `n_envs 4` `280 fps` `n_envs 8` `371 fps`
`cuda n_envs 1` `96 fps` `n_envs 4` `292 fps` `n_envs 8` `423 fps` — GPU only `~4–14%` faster, `gpu util 4–9%` vs `cpu util 20%` idle, so bottleneck is `pyboy.tick()` not `torch` matmul. Recommendation: use **CPU + `n_envs 4–8`** (saves 2GB CUDA download, no VRAM contention with Xorg).

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv   # → 5.2, 2048MiB
./scripts/setup_legacy_gpu.sh --check          # reports torch arch + cap + alloc test
uv run train.py --device cpu --window null --n-envs 4 --sessions 200 --runs 4 --steps 2048 --checkpoint-freq 10  # recommended
uv run train.py --device cuda --window null --n-envs 4 --sessions 200 --runs 4 --steps 2048 --checkpoint-freq 10  # works, ~5–15% faster at n_envs 8
./scripts/train_segments.sh --device cpu --total 600 --per-segment 60  # segmented; add --device cuda if you want GPU
./scripts/setup_legacy_gpu.sh --revert         # modern GPU: uv pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu130
```

`auto` still `→cpu` on `<4GB` (`train.py:75`); force `--device cuda` to test. For modern `sm_75+` you can `uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130` (needs `uv run --no-sync` until lock bumped).

### Window backends (`tetris_env.py:137`)

- `window="null"` → `set_emulation_speed(0)` unlimited (training, CI)
- `window="SDL2"` → visible window, respects `--speedup` (play); requires `DISPLAY` (`:1` on Ubuntu desktop / X11)

## References

- https://github.com/Baekalfen/PyBoy/wiki/Installation  
- https://github.com/Baekalfen/PyBoy/wiki/Example-Tetris
