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

PPO agent playing Game Boy Tetris via PyBoy (SB3, Gymnasium). This worktree trains an enriched observation (board + next/current piece + shape features) with `MultiInputPolicy`.

## Inspirations

* [RL automation + Tetris](https://rotational.io/blog/reinforcement-learning-automation-and-tetris) ([code](https://github.com/pdeziel/ai-tetris))
* [Near-perfect Tetris AI](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player) ([code](https://github.com/LeeYiyuan/tetrisai)) — height/holes/bump shaping weights

## Requirements

- Python 3.12, `uv` ([docs](https://docs.astral.sh/uv/))
- ROM `roms/tetris.gb` (not tracked; your own legal dump)
- Ubuntu: `sudo apt install -y libsdl2-2.0-0` · macOS: `brew install sdl2`

## Setup

```bash
uv sync --extra dev
mkdir -p roms models_robust
# copy your dump → roms/tetris.gb
ls -lh roms/tetris.gb states/init.state  # init.state: PyBoy 2.7.0 save, already in repo
```

Slow link / <4GB VRAM: `uv sync --no-install-package torch && uv pip install torch --index-url https://download.pytorch.org/whl/cpu`. GTX 960 sm_52: pinned `torch==2.6.0+cu124` already works (see GPU note).

## Quickstart

```bash
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format --check .

# Train (defaults → models_robust/tetris_ppo_next.zip; step totals print at launch, scale with --n-envs)
uv run train.py --device cpu --window null --sessions 200 --steps 2048
# Play
uv run play.py --window SDL2 --model models_robust/tetris_ppo_next
# Headless / CI
uv run play.py --window null --runs 2 --model models_robust/tetris_ppo_next
xvfb-run -a uv run play.py --window SDL2  # no DISPLAY
```

Defaults: enriched obs + `α=0.1`, `freq 8`, `lr 1e-4` constant, `clip 0.1`, `ent 0.05`, `gamma 0.95`, `gae 0.95`, `vf 0.5`, `clip-vf 0.2`, `kl 0.02`, `batch 32`.

## How it works

| | |
|---|---|
| Obs (enriched) | Dict: `board` 18×10 tiles (0–8) + `next`/`current` one-hot 7 + `features` 14 (heights, holes, bump, wells). `--obs-mode board` = legacy Box ablation. |
| Actions (5) | `LEFT/RIGHT/DOWN/A/B` (`UP` was a no-op, `PASS` let pieces stack centrally; old 7/6-act zips restart fresh). |
| Reward | Hybrid PBRS `score_delta + α·shaped_delta`, shaped `= -0.5·height +0.75·lines -0.8·holes -0.5·bump`; game over `-100`. `α=0` = legacy sparse. |
| Robustness | `--max-steps 10000` truncation, `--noop-max 30` reset desync, `--eval-freq` best-model tracking. |

`--freq`, `--shaped-alpha` and `--obs-mode` must match between train and play.

## Rotation looks missing? Read this

`--deterministic` (default) picks argmax; early models rank `DOWN/LEFT/RIGHT` first while `A/B` sits at 10–15% — e.g. `det rot 0/500` vs `stoch rot 97/500`. Use `uv run play.py --no-deterministic …` to see explored rotation; keep stochastic only for diagnosis (`stoch A/B <5%` → train longer or raise `--ent-coef`).

## Long training (segmented, resumable)

```bash
./scripts/train_segments.sh --total 600 --per-segment 60  # 10 resumable segments, ckpt every 10
# resume after CTRL+C: re-run the same command (loads .zip; CTRL+C also saves)
```

Checkpoints: `models_robust/tetris_ppo_next.zip` + `_ckpt_NNNN.zip`. Resume is automatic; policy/obs/action mismatch restarts fresh with a warning.

Throughput, end-to-end training fps on this 8-core box (enriched obs, `steps 512`): `n_envs 4 → 570`, `8 → 876`, `12 → 997`, `16 → 1086` — knee at 8–12, default `n_envs 8`. Profile: ~84% of step time is PyBoy ticks, so parallelism (not GPU) is the lever.

## GPU note (GTX 960 sm_52)

Pinned `torch 2.6.0+cu124` is the last with `sm_50`; `auto` → `cpu` on <4GB VRAM, force `--device cuda` to test (`~400MiB`). Modern GPUs: `uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130` (then `uv run --no-sync`).

## Window backends

- `null` — unlimited speed, training/CI · `SDL2` — visible window, `--speedup` applies, needs `DISPLAY` · `headless` — alias for `null`

## References

- https://github.com/Baekalfen/PyBoy/wiki/Installation
- https://github.com/Baekalfen/PyBoy/wiki/Example-Tetris
