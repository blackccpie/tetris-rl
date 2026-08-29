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
# defaults: CnnPolicy TetrisCNN α=0.1 200×4×2048≈1.6M (600≈5M 1200≈10M) lr 1e-4 clip 0.1 ent 0.01 gamma 0.95 batch 32
uv run train.py --device cpu --window null --policy CnnPolicy --shaped-alpha 0.1 --sessions 200 --steps 2048
uv run train.py --device cpu --policy MlpPolicy --shaped-alpha 0.0 --sessions 40  # legacy pure score (sparse)
# tensorboard: uv run train.py --tensorboard logs/
uv run train.py --device cpu     # auto→cpu on <4GB VRAM (e.g. GTX 960 2GB) or sm_52 incompat

uv run play.py --help            # loads model and plays with SDL2 window
uv run play.py --window null --shaped-alpha 0.1 --device cpu --model models/tetris_ppo_model
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
- `--shaped-alpha 0.1` hybrid `score+α*shaped` (`0.0` legacy sparse, `α=0.1-0.3`), `shaped = -0.5 height +0.75 lines -0.35 holes -0.2 bump`
- `--learning-rate 1e-4 --clip-range 0.1 --ent-coef 0.01 --gamma 0.95 --batch-size 32` (tuned for dense)
- `--checkpoint-freq 10` saves `models/tetris_ppo_model.zip` every 10 sessions + `models/tetris_ppo_model_ckpt_0010.zip` …; `0` only at end; `CTRL+C` also saves
- `--n-envs 1` (`DummyVecEnv`) or `>1` `SubprocVecEnv` for CPU throughput
- `--tensorboard logs/` → `tensorboard --logdir logs`
- `--device auto|cpu|cuda` — `auto`→`cpu` on `<4GB VRAM` or `sm_52` (`torch 2.13+cu130` ≥`sm_75`)

Checkpoints: `ls -lh models/tetris_ppo_model*.zip` — resume is automatic; mismatched `Mlp→Cnn` starts fresh with warning.

### Window backends (`tetris_env.py:137`)

- `window="null"` → `set_emulation_speed(0)` unlimited (training, CI)
- `window="SDL2"` → visible window, respects `--speedup` (play); requires `DISPLAY` (`:1` on Ubuntu desktop / X11)

## References

- https://github.com/Baekalfen/PyBoy/wiki/Installation  
- https://github.com/Baekalfen/PyBoy/wiki/Example-Tetris
