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

uv run train.py --help           # trains PPO; saves to models/tetris_ppo_model.zip
# defaults: CnnPolicy TetrisCNN α=0.1 200×4×2048≈1.6M (600≈5M 1200≈10M) lr 1e-4 clip 0.1 ent 0.02 gamma 0.95 batch 32 freq 12
uv run train.py --device cpu --window null --policy CnnPolicy --shaped-alpha 0.1 --sessions 200 --steps 2048
uv run train.py --device cpu --policy MlpPolicy --shaped-alpha 0.0 --sessions 40  # legacy pure score (sparse)
# tensorboard: uv run train.py --tensorboard logs/
uv run train.py --device cpu     # auto→cpu on <4GB VRAM (e.g. GTX 960 2GB) or sm_52 incompat

uv run play.py --help            # loads model and plays with SDL2 window
uv run play.py --window null --shaped-alpha 0.1 --freq 12 --device cpu --model models/tetris_ppo_model
# headless / CI (no DISPLAY):
xvfb-run -a uv run play.py --window null
# or: SDL_VIDEODRIVER=dummy uv run train.py --window null
```

### Long training — segmented & resumable (Option A)

CPU `1.6M ≈ 3-4h`, `5M ≈ 10h` at `~130 fps` `window="null"`. Split into segments — each resumes from `models/tetris_ppo_model.zip` (`train.py:215` `PPO.load` + `TetrisCNN` `custom_objects`):

```bash
# 5M = 600×4×2048 — 10 segments ×60 sessions (~500k each), checkpoint every 10
./scripts/train_segments.sh --total 600 --per-segment 60
# 1.6M single run with checkpoints
uv run train.py --sessions 200 --runs 4 --steps 2048 --checkpoint-freq 10 --policy CnnPolicy --shaped-alpha 0.1
# resume after CTRL+C / power loss — re-run same command (loads .zip)
./scripts/train_segments.sh --total 600 --per-segment 60
```

Options `train.py --help` / `scripts/train_segments.sh --help`:
- `--sessions N`, `--runs N`, `--steps N` → total steps `sessions×runs×steps` (`×n_envs` if `>1`)
- `--policy CnnPolicy|MlpPolicy` — `CnnPolicy` `TetrisCNN` `3×3` for `18×10` (NatureCNN too large), `Mlp` fallback
- `--shaped-alpha 0.1` hybrid `score+α*shaped` (`0.0` legacy sparse, `α=0.1-0.3`), `shaped = -0.5 height +0.75 lines -0.8 holes -0.35 bump` (holes/bump ↑ to reward rotation)
- `--freq 12` ticks per action (finer rotation than legacy 24) — must match train & play
- `--learning-rate 1e-4 --clip-range 0.1 --ent-coef 0.02 --gamma 0.95 --batch-size 32` (ent 0.02 prevents LEFT/RIGHT/DOWN collapse without spin)
- `--checkpoint-freq 10` saves `models/tetris_ppo_model.zip` every 10 sessions + `models/tetris_ppo_model_ckpt_0010.zip` …; `0` only at end; `CTRL+C` also saves
 - `--n-envs 1` (`DummyVecEnv`) or `>1` `SubprocVecEnv` for throughput — `n_envs 4` `280 fps` `n_envs 8` `371 fps` on this 8-core CPU; `cuda n_envs 8` `423 fps` only ~14% faster (emulator-bound, gpu util 4%)
 - `--tensorboard logs/` → `tensorboard --logdir logs`
 - `--device auto|cpu|cuda` — `auto`→`cpu` on `<4GB VRAM`; for `GTX 960` `cuda` works (`torch 2.6.0+cu124` `sm_50`, `~400MiB`) but CPU is recommended on this box (see GPU section)
- Actions `6` (`tetris_env.py:130`): `LEFT/RIGHT/DOWN/A/B/PASS` — `UP` removed (no-op in GB Tetris, wasted 14% exploration; old 7-action zips auto-start fresh)

Checkpoints: `ls -lh models/tetris_ppo_model*.zip` — resume is automatic; mismatched `Mlp→Cnn` or `7→6` actions starts fresh with warning (`models/archive7/` has legacy 7-action zips).

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
