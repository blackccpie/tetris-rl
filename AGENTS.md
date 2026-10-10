# AGENTS.md — Tetris-RL Project Conventions

## Project Overview
Toy/experimental Tetris RL using PyBoy (Game Boy emulator), Gymnasium, Stable-Baselines3, numpy, tqdm. Trains a PPO agent to play Tetris via the emulator screen observation.

## Key Design Decisions
- **Observation**: Enriched Dict default — `board` 18×10 compressed tiles (0–8) via `game_area()` + `next`/`current` one-hot 7 (`next_tetromino()`; current inferred as previous next on change, no direct PyBoy API) + `features` 14 (10 heights + holes/bump/max-well/n-wells). `--obs-mode board` legacy Box for ablation. `MultiInputPolicy` `TetrisMultiInput` (board CNN 128 + pieces 32 + feats 32 → 128).
- **Loss**: PPO `gae 0.95 vf 0.5 clip-vf 0.2 max-grad 0.5 target-kl 0.02 normalize-adv` + linear LR decay; `EvalCallback` best-model; `max_steps 10000` truncation kills stall loops; `noop_max 30` desyncs piece RNG across envs.
- **Reward**: Hybrid PBRS `tetris_env.py:206` `score_delta + α*shaped_delta` where `shaped = get_total_score()` (`-0.5 height +0.75 lines -0.8 holes -0.5 bump` `tetris_env.py:245`, legacy `-0.35/-0.2` + `PASS` let first pieces stack centrally → no rotation/spread). `α=0.1` default (`--shaped-alpha 0.0` legacy pure score, sparse → plateau `Score 5-9`). Preserves optimal policy (Ng 1999).
- **Game over**: `pyboy.game_wrapper.game_over()`, not observation-based detection. Returns `-100 + α*shaped_delta`.
- **PyBoy `no_input=False`** — required for SDL2 window display on macOS (event queue must be pumped).
- **PyBoy speed**: `set_emulation_speed(0)` when `window="null"` (unlimited), `speedup` arg when `window="SDL2"`.

## Package Management
- `uv` — no `requirements.txt`; dependencies in `[project.dependencies]` in `pyproject.toml`.
- `uv sync` to install, `uv run` to execute.
- PyTorch: `torch>=2.2,<2.7` (`uv.lock` `2.6.0+cu124` pinned for GTX 960 sm_52; `2.7+` drops `sm_50` → `no kernel image` `sm_75+`). Maxwell GTX 960 sm_52 (2GB, driver 580) now GPU-enabled: `uv run train.py --device cuda --n-envs 4` → `297 fps` vs `95 fps` `n_envs 1` within `~400MiB`; modern GPU: `uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130` (use `uv run --no-sync` until lock bump). Former `<2.3.0` macOS 12 pin removed.
- Ubuntu 24.04: requires `libsdl2-2.0-0` at runtime (`sudo apt install libsdl2-2.0-0`); `libsdl2-dev` only to build PyBoy from sdist. With NVIDIA GPU <4GB VRAM use `--device cpu` unless legacy GPU path (see `train.py`). Large CUDA wheels (~2GB) — on slow link use CPU wheel: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` after `uv sync --no-install-package torch`.

## Code Style (enforced by Ruff)
- Line length: 120
- Quotes: double
- Target: Python 3.12
- Ruff selects: `E, F, I, N, W, UP`
- Run: `ruff check . && ruff format --check .`

## Testing
- `uv run pytest tests/ -v` — 40 pass with `roms/tetris.gb`, 24 pass/16 skipped without (ROM-less CI via `helper_env` + `pytest.skip`).
- Fixtures: `env` (integration, `window="null"`, skips without ROM) and `helper_env` (`__new__` without PyBoy for pure helpers, e.g. `get_holes_count`, `get_column_height`).
- Guard coverage: `TestTetrisEnvGuards` checks ROM/init_state/window guards, `headless`→`null` alias, `close` idempotency, and compressed-tile `!=0` fix.
- Test files: `tests/test_helpers.py` (pure-function + guards, now ROM-less), `tests/test_env.py` (Gymnasium API integration, skips without ROM).
- Pillow warnings are expected without `pillow` (`pyboy.api.screen`).
- Run: `uv run ruff check . && ruff format --check .` must pass.

## Entry Points
- `uv run train.py --device auto|cpu|cuda --window null|SDL2|headless --policy MultiInputPolicy --obs-mode enriched --shaped-alpha 0.1 --sessions 200 --steps 2048 --freq 8` — robust worktree defaults: enriched Dict obs, `models_robust/tetris_ppo_next`, `lr 1e-4 constant clip 0.1 gae 0.95 vf 0.5 clip-vf 0.2 kl 0.02 ent 0.05 gamma 0.95 batch 32`; `--obs-mode board --policy CnnPolicy` for ablation; old Box/Cnn/Mlp + `7→6→5` zips auto-start fresh; resume re-applies CLI hyperparams (ent/lr/clip/kl/…) and the step counter accumulates (`reset_num_timesteps=False`); `--eval-freq 20000` best-model + `--tensorboard` optional (graceful if missing).
- `scripts/train_segments.sh --total 600 --per-segment 60 --n-envs 4` — Option A segmented resumable (10×500k for 5M), each `uv run train.py --sessions 60 --n-envs 4` resumes from `models_robust/tetris_ppo_next.zip`; checkpoints `models_robust/tetris_ppo_next_ckpt_*.zip` every `--checkpoint-freq`.
- `uv run play.py --rom/--init/--model/--window/--device/--deterministic --shaped-alpha 0.1 --freq 8` — must match train `α` and `freq`; SDL2 warns if no `DISPLAY`/`WAYLAND_DISPLAY` (use `xvfb-run -a`); `null`/`headless` for headless. `--deterministic` `true` = `argmax` (hides `A/B` early), `--no-deterministic` samples — use latter to diagnose rotation on `8k`-step `det 0%` vs `stoch 19%` checkpoints (see `README.md` Deterministic vs stochastic).

## Key Files
- `tetris_env.py` — Gymnasium Env subclass; 5 actions (`LEFT/RIGHT/DOWN/A/B`, UP/PASS removed); `obs_mode` `enriched` Dict (`board` 18×10 uint8 + `next`/`current` one-hot 7 + `features` 14) or `board` Box; reward hybrid `score + α*shaped` (`shaped_alpha`, `freq8`); `max_steps` truncation + `noop_max` randomization.
- `train.py` — PPO `TetrisMultiInput` (`MultiInputPolicy`, enriched default) + `TetrisCNN` (`CnnPolicy`) + `MlpPolicy` fallback, PBRS `α`, `SubprocVecEnv` optional; loss knobs `gae-lambda/vf-coef/clip-vf/max-grad-norm/target-kl/normalize-advantage` + linear `lr/clip` schedules + `EvalCallback` best-model.
- `agent_trainer.py` — simple training loop with tqdm.
- `states/init.state` — save state (PyBoy 2.7.0); regenerated by pressing START after ROM load.
- `roms/tetris.gb` — Tetris ROM (not tracked in git).
